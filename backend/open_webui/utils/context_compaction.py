from __future__ import annotations

import asyncio
import codecs
import copy
import hashlib
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import replace
from functools import cache
from typing import Any

import tiktoken
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from open_webui.config import TIKTOKEN_ENCODING_NAME
from open_webui.models.chats import Chats
from open_webui.models.config import Config
from open_webui.utils.chat_id import is_saved_chat_id
from open_webui.utils.externalized_refs import RefEntry, make_ref_entry
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.misc import convert_output_to_messages, get_content_from_message, get_message_list
from open_webui.utils.payload import apply_params_to_form_data
from open_webui.utils.task import (
    prompt_template,
    prompt_variables_template,
    replace_messages_variable,
    replace_prompt_variable,
)

log = logging.getLogger(__name__)

_BODY_TOKEN_EXTRA_KEYS = (
    'tools',
    'tool_choice',
    'functions',
    'function_call',
    'response_format',
    'parallel_tool_calls',
)
_TOKEN_MESSAGE_KEYS = (
    'role',
    'content',
    'name',
    'tool_call_id',
    'tool_calls',
    'function_call',
    'output',
    'files',
    'sources',
    'reasoning_content',
    'reasoning_details',
    'thinking',
)
_MEDIA_PART_TYPES = {'file', 'image', 'image_url', 'input_audio', 'input_file', 'input_image'}
_BOUNDARY_KEY = '_open_webui_context_compaction_boundary'
CONTEXT_COMPACTION_USAGE_ANCHOR_KEY = '_open_webui_context_compaction_usage_anchor'
CONTEXT_COMPACTION_TRANSIENT_MARKER_KEY = '_open_webui_context_compaction_transient'
_DEFAULT_EXCERPT_BYTES = 512
_DEFAULT_EXCERPT_COUNT = 32
_HISTORY_REF_XML_SUFFIX_RE = re.compile(
    r'<history_ref>history:[0-9a-f]{64}</history_ref>(?=</auto_compaction_context>\Z)'
)
_HISTORY_DB_KEYS = {
    'id',
    'parentId',
    'files',
    'output',
    'contextSummary',
    'context_summary',
    'usage',
    'model',
    'meta',
}


class CanonicalHistoryError(ValueError):
    pass


DEFAULT_CONTEXT_COMPACTION_PROMPT = """### Task:
Summarize the conversation history that will be compacted out of the active chat context.

### Instructions:
- Preserve key decisions, user preferences, and constraints.
- Preserve files, artifacts, tool results, and code changes that matter going forward.
- Preserve the current task state, unresolved questions, and next steps.
- Be factual and specific. Do not invent details.
- Keep the summary concise, but complete enough for the assistant to continue without the removed messages.

### Previous Summary:
{{PREVIOUS_SUMMARY}}

### Messages Being Compacted:
{{COMPACTED_MESSAGES}}

### Recent Messages Kept In Context:
{{RECENT_MESSAGES}}"""


def _xml_cdata(value: str) -> str:
    return value.replace(']]>', ']]]]><![CDATA[>')


def _middle_truncate_utf8(value: str, limit: int) -> str:
    if len(value) <= limit:
        encoded = value.encode('utf-8')
        if len(encoded) <= limit:
            return value
    marker = '…'
    budget = max(0, limit - len(marker.encode('utf-8')))
    head_budget = budget // 2
    tail_budget = budget - head_budget
    head = value[:head_budget].encode('utf-8')[:head_budget].decode('utf-8', 'ignore')
    tail = value[-tail_budget:].encode('utf-8')[-tail_budget:].decode('utf-8', 'ignore') if tail_budget else ''
    return f'{head}{marker}{tail}'


def _first_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for part in value:
            if isinstance(part, dict) and part.get('type') == 'text' and isinstance(part.get('text'), str):
                return part['text']
    return None


def _is_transient_message(message: Any, patterns: tuple[re.Pattern[str], ...] = ()) -> bool:
    if not isinstance(message, dict) or message.get('role') != 'user':
        return False
    if message.get(CONTEXT_COMPACTION_TRANSIENT_MARKER_KEY) is True:
        return True
    meta = message.get('meta')
    if isinstance(meta, dict) and meta.get('internal') is True:
        return True
    if not patterns:
        return False
    text = _first_text(message.get('content'))
    return text is not None and any(pattern.match(text.lstrip()) for pattern in patterns)


def get_last_persistent_user_message(
    messages: list[dict],
    patterns: tuple[re.Pattern[str], ...] = (),
) -> str | None:
    for message in reversed(messages):
        if message.get('role') == 'user' and not _is_transient_message(message, patterns):
            return get_content_from_message(message)
    return None


def _historical_user_excerpts(
    messages: list[dict],
    byte_limit: int,
    count: int,
    patterns: tuple[re.Pattern[str], ...] = (),
) -> list[str]:
    if count <= 0:
        return []
    excerpts = []
    for message in reversed(messages):
        if (
            message.get('role') == 'user'
            and not _is_transient_message(message, patterns)
            and (content := get_content_from_message(message))
        ):
            excerpts.append(_middle_truncate_utf8(content, byte_limit))
            if len(excerpts) == count:
                break
    excerpts.reverse()
    return excerpts


def _canonical_history_content(value: Any) -> Any:
    return _sanitize_token_value(value)[0]


def _canonical_direct_history_message(message: Any) -> dict[str, Any] | None:
    if not isinstance(message, dict):
        return None
    role = message.get('role')
    if not isinstance(role, str):
        return None
    return {
        key: _canonical_history_content(value)
        for key, value in message.items()
        if key not in _HISTORY_DB_KEYS
    }


def _canonical_output_messages(output: Any) -> list[dict[str, Any]]:
    if not isinstance(output, list):
        return []
    return [
        _canonical_direct_history_message(message)
        for message in convert_output_to_messages(output, flatten_tool_images=False)
    ]


def _canonical_messages(
    message: dict[str, Any],
) -> list[dict[str, Any]]:
    if message.get('role') == 'system' or _is_transient_message(message):
        return []
    if message.get('role') == 'assistant' and message.get('output'):
        converted = _canonical_output_messages(message['output'])
        if converted:
            return converted
    direct = _canonical_direct_history_message(message)
    return [direct] if direct is not None else []


def _canonical_history_entry(messages: list[dict[str, Any]]) -> RefEntry:
    records = [
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        for message in messages
        for item in _canonical_messages(message)
    ]
    text = '\n'.join(records)
    entry = make_ref_entry(text, kind='history')
    if entry is None:
        raise CanonicalHistoryError('history source is not valid UTF-8')
    return entry


def _bind_history_loader(
    entry: RefEntry,
    messages: list[dict[str, Any]],
    selected_index: int,
) -> RefEntry:
    @cache
    def checkpoint_refs() -> tuple[tuple[str, int], ...]:
        digest = hashlib.sha256()
        offset = 0
        first = True
        checkpoints = []
        for message in messages[:selected_index]:
            if message.get('contextSummary') or message.get('context_summary'):
                checkpoints.append((f'history:{digest.hexdigest()}', offset))
            for item in _canonical_messages(message):
                record = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                piece = record if first else f'\n{record}'
                if not entry.text.startswith(piece, offset):
                    return ()
                digest.update(piece.encode('utf-8'))
                offset += len(piece)
                first = False
        return tuple(checkpoints) if offset == len(entry.text) else ()

    resolved: dict[str, RefEntry] = {}

    def materialize(ref: str, end: int) -> RefEntry:
        ancestor = resolved.get(ref)
        if ancestor is None:
            ancestor = make_ref_entry(entry.text[:end], kind='history', load_history=load_ancestors)
            if ancestor is None or ancestor.ref != ref:
                raise CanonicalHistoryError('history source is not valid UTF-8')
            resolved[ref] = ancestor
        return ancestor

    async def load_ancestors(requested: str | None) -> tuple[RefEntry, ...]:
        def resolve() -> tuple[RefEntry, ...]:
            entries = []
            for ref, end in reversed(checkpoint_refs()):
                if requested is not None and ref != requested:
                    continue
                try:
                    ancestor = materialize(ref, end)
                except CanonicalHistoryError:
                    break
                if requested is not None:
                    return (ancestor,)
                entries.append(ancestor)
            return tuple(entries)

        return await asyncio.to_thread(resolve)

    return replace(entry, load_history=load_ancestors)


def _history_entry_at(messages: list[dict[str, Any]], carrier_index: int) -> RefEntry | None:
    if not 0 < carrier_index < len(messages):
        return None
    return _bind_history_loader(
        _canonical_history_entry(messages[:carrier_index]),
        messages,
        carrier_index,
    )


def _history_source(source: Any) -> tuple[list[dict[str, Any]], int] | None:
    if (
        isinstance(source, tuple)
        and len(source) == 2
        and isinstance(source[0], list)
        and isinstance(source[1], int)
        and 0 <= source[1] < len(source[0])
    ):
        return source
    return None


async def resolve_request_history(source: Any) -> RefEntry | None:
    if isinstance(source, RefEntry):
        return source
    checkpoint = _history_source(source)
    if checkpoint is None:
        return None
    return await asyncio.to_thread(_history_entry_at, *checkpoint)


def render_summary_message(summary: str, summary_meta: dict | None = None) -> dict:
    excerpts = (summary_meta or {}).get('historical_user_messages') or []
    excerpt_xml = ''.join(
        f'<historical_user_message ordinal="{index}"><![CDATA[{_xml_cdata(str(value))}]]></historical_user_message>'
        for index, value in enumerate(excerpts, 1)
    )
    return {
        'role': 'user',
        'content': (
            '<auto_compaction_context>'
            '<instruction>This is background from earlier turns. Use it for continuity; '
            'do not answer it as a new request.</instruction>'
            f'<checkpoint_summary><![CDATA[{_xml_cdata(summary.strip())}]]></checkpoint_summary>'
            + (f'<historical_user_messages>{excerpt_xml}</historical_user_messages>' if excerpt_xml else '')
            + '</auto_compaction_context>'
        ),
    }


async def replay_cached_compaction_messages(messages: list[dict], state: dict) -> list[dict]:
    """Replay the request-local checkpoint without resolving or hashing history again."""
    summary = state.get('summary')
    summary_meta = state.get('summary_meta')
    checkpoint_history = _history_source(state.get('checkpoint_history'))
    carrier_id = checkpoint_history[0][checkpoint_history[1]].get('id') if checkpoint_history else None
    prefetch = state.get('prefetch_task')
    if not summary and isinstance(prefetch, asyncio.Task):
        try:
            summary, summary_meta, history_entry = await prefetch
        except Exception:
            return messages
        state.update(
            {
                'summary': summary,
                'summary_meta': summary_meta,
                'selected_history': history_entry,
            }
        )
    if not summary:
        summary = state.get('previous_summary')
        summary_meta = state.get('previous_summary_meta')
        carrier_id = state.get('selected_checkpoint_message_id')
    if not isinstance(summary, str) or not summary.strip() or not isinstance(carrier_id, str):
        return messages

    system_messages, raw_messages = _split_leading_system_messages(messages)
    carrier_index = next(
        (index for index, message in enumerate(raw_messages) if message.get('id') == carrier_id),
        None,
    )
    if carrier_index is None:
        return messages
    return [
        *system_messages,
        render_summary_message(summary, summary_meta if isinstance(summary_meta, dict) else {}),
        *raw_messages[carrier_index:],
    ]


def set_summary_history_ref(body: dict, ref: str | None) -> dict:
    messages = body.get('messages')
    if not isinstance(messages, list):
        return body
    replacement = f'<history_ref>{ref}</history_ref>' if ref and re.fullmatch(r'history:[0-9a-f]{64}', ref) else ''
    updated = None
    for index, message in enumerate(messages):
        content = message.get('content') if isinstance(message, dict) else None
        closing = '</auto_compaction_context>'
        if (
            not isinstance(content, str)
            or not content.startswith('<auto_compaction_context>')
            or not content.endswith(closing)
        ):
            continue
        clean = _HISTORY_REF_XML_SUFFIX_RE.sub('', content)
        if replacement:
            clean = f'{clean[: -len(closing)]}{replacement}{closing}'
        if clean != content:
            updated = list(messages)
            updated[index] = {**message, 'content': clean}
        break
    return {**body, 'messages': updated} if updated is not None else body


def _checkpoint_summary(messages: list[dict]) -> tuple[int | None, str | None]:
    selected: tuple[int, str] | None = None
    for index, message in enumerate(messages):
        summary = message.get('contextSummary') or message.get('context_summary')
        if isinstance(summary, str) and summary.strip():
            selected = index, summary.strip()
    return selected or (None, None)


async def prepare_compaction_messages(messages: list[dict], metadata: dict) -> tuple[list[dict], dict]:
    """Apply one stored checkpoint and mark one safe future cut without DB state."""
    config = await _load_config()
    state: dict[str, Any] = {'config': config}
    if not config['enable'] or metadata.get('task') == 'context_compaction':
        return messages, state

    system_messages, checkpoint_messages = _split_leading_system_messages(messages)
    summary_index, previous_summary = _checkpoint_summary(checkpoint_messages)
    active_offset = summary_index if summary_index is not None else 0
    active_messages = await asyncio.to_thread(copy.deepcopy, checkpoint_messages[active_offset:])
    summary_meta: dict[str, Any] = {}
    if summary_index is not None:
        state['selected_checkpoint_message_id'] = checkpoint_messages[summary_index].get('id')
        state['selected_history'] = (checkpoint_messages, summary_index)
        summary_meta['historical_user_messages'] = await asyncio.to_thread(
            _historical_user_excerpts,
            checkpoint_messages[:summary_index],
            _DEFAULT_EXCERPT_BYTES,
            _DEFAULT_EXCERPT_COUNT,
            config['transient_patterns'],
        )
    for index in range(len(active_messages) - 1, -1, -1):
        message = active_messages[index]
        if message.get('role') != 'assistant':
            continue
        info = message.get('info')
        usage = message.get('usage') or (info.get('usage') if isinstance(info, dict) else None)
        input_tokens = _strict_usage_input_tokens(usage)
        if input_tokens is not None:
            message[CONTEXT_COMPACTION_USAGE_ANCHOR_KEY] = input_tokens
            break
    boundary = await asyncio.to_thread(
        find_safe_compaction_boundary,
        active_messages,
        config['retention_percentage'],
        config['transient_patterns'],
    )

    if boundary:
        active_messages[boundary][_BOUNDARY_KEY] = True
        raw_boundary = active_offset + boundary
        state['checkpoint_history'] = (checkpoint_messages, raw_boundary)

    if previous_summary:
        active_messages = [render_summary_message(previous_summary, summary_meta), *active_messages]
        state['previous_summary'] = previous_summary
        state['previous_summary_meta'] = summary_meta
    return [*system_messages, *active_messages], state


def _without_boundary_marker(
    messages: list[dict],
    *,
    keep_transient: bool = False,
) -> list[dict]:
    removed_keys = {_BOUNDARY_KEY, CONTEXT_COMPACTION_USAGE_ANCHOR_KEY}
    if not keep_transient:
        removed_keys.add(CONTEXT_COMPACTION_TRANSIENT_MARKER_KEY)
    return [
        {key: value for key, value in message.items() if key not in removed_keys}
        if any(key in message for key in removed_keys)
        else message
        for message in messages
    ]


async def _create_checkpoint(
    request,
    user,
    model_id: str,
    models: dict,
    metadata: dict,
    state: dict,
    config: dict,
    compacted_messages: list[dict],
    recent_messages: list[dict],
) -> tuple[str, dict[str, Any], Any]:
    checkpoint_history = _history_source(state.get('checkpoint_history'))
    if checkpoint_history and checkpoint_history[1] > 0:
        source_messages = checkpoint_history[0][: checkpoint_history[1]]
        checkpoint_message_id = checkpoint_history[0][checkpoint_history[1]].get('id')
    else:
        source_messages = []
        checkpoint_message_id = None
    chat_id = metadata.get('chat_id')
    if not checkpoint_message_id or not is_saved_chat_id(chat_id):
        raise RuntimeError('Context compaction checkpoint is not durable; provider request was not sent')
    summary_meta = {
        'historical_user_messages': await asyncio.to_thread(
            _historical_user_excerpts,
            source_messages,
            _DEFAULT_EXCERPT_BYTES,
            _DEFAULT_EXCERPT_COUNT,
            config['transient_patterns'],
        )
    }
    summary = await _generate_summary(
        request,
        user,
        model_id,
        models,
        compacted_messages,
        recent_messages,
        None,
        config['prompt_template'],
        config['transient_patterns'],
    )
    saved = await Chats.upsert_message_to_chat_by_id_and_message_id(
        chat_id,
        checkpoint_message_id,
        {'contextSummary': summary},
        touch=False,
    )
    if saved is None:
        raise RuntimeError('Context compaction checkpoint could not be saved; provider request was not sent')
    return summary, summary_meta, checkpoint_history


def _prefetch_done(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    try:
        task.result()
    except Exception:
        log.exception('Context compaction prefetch failed')


async def _emit_compaction_status(event_emitter, description: str, done: bool, *, error: bool = False) -> None:
    if not event_emitter:
        return
    data = {
        'action': 'context_compaction',
        'description': description,
        'done': done,
    }
    if error:
        data['error'] = True
    await event_emitter({'type': 'context_compaction', 'data': data})


async def compact_provider_payload(
    request,
    user,
    body: dict,
    metadata: dict,
    model_id: str,
    models: dict,
    state: dict,
    *,
    force: bool = False,
    finalize_candidate: Callable[[dict], Awaitable[dict]] | None = None,
) -> dict:
    """Compact the final provider candidate or fail before provider dispatch."""
    messages = body.get('messages')
    if not isinstance(messages, list):
        return body
    config = state.get('config') or await _load_config()
    if not config['enable']:
        return body
    if metadata.get('task') == 'context_compaction' or body.get('previous_response_id'):
        return {**body, 'messages': _without_boundary_marker(messages)}

    system_messages, working = _split_leading_system_messages(messages)
    source_messages = state.get('projection_source_messages')
    if isinstance(source_messages, list):
        _, source_working = _split_leading_system_messages(source_messages)
        if len(source_working) != len(working):
            source_working = working
    else:
        source_working = working
    boundary = next(
        (index for index, message in enumerate(working) if message.get(_BOUNDARY_KEY) is True),
        None,
    )
    if boundary is None and not state.get('compacted'):
        saved_boundary = state.get('provider_boundary')
        if isinstance(saved_boundary, int) and 0 < saved_boundary < len(working):
            working = list(working)
            working[saved_boundary] = {**working[saved_boundary], _BOUNDARY_KEY: True}
            body = {**body, 'messages': [*system_messages, *working]}
            boundary = saved_boundary
    if boundary is not None:
        state['provider_boundary'] = boundary

    usage_anchor = next(
        (
            (index, message[CONTEXT_COMPACTION_USAGE_ANCHOR_KEY])
            for index, message in enumerate(working)
            if isinstance(message.get(CONTEXT_COMPACTION_USAGE_ANCHOR_KEY), int)
        ),
        None,
    )
    if usage_anchor is None and not state.get('compacted'):
        saved_anchor = state.get('provider_usage_anchor')
        if (
            isinstance(saved_anchor, tuple)
            and len(saved_anchor) == 2
            and isinstance(saved_anchor[0], int)
            and 0 <= saved_anchor[0] < len(working)
        ):
            working = list(working)
            working[saved_anchor[0]] = {
                **working[saved_anchor[0]],
                CONTEXT_COMPACTION_USAGE_ANCHOR_KEY: saved_anchor[1],
            }
            body = {**body, 'messages': [*system_messages, *working]}
            usage_anchor = saved_anchor
    if usage_anchor is not None:
        state['provider_usage_anchor'] = usage_anchor

    before = await asyncio.to_thread(estimate_provider_tokens, body)
    threshold = _resolve_token_threshold(config['token_threshold'], config['token_cap'], metadata)
    if before <= threshold and not force:
        soft_ratio = config['soft_trigger_ratio']
        if (
            boundary
            and soft_ratio > 0
            and before >= int(threshold * soft_ratio)
            and not isinstance(state.get('prefetch_task'), asyncio.Task)
        ):
            # ponytail: duplicate cross-worker prefetches are harmless; add Redis
            # dedup only if measured summary cost warrants the coordination.
            task = asyncio.create_task(
                _create_checkpoint(
                    request,
                    user,
                    model_id,
                    models,
                    metadata,
                    state,
                    config,
                    _without_boundary_marker(source_working[:boundary], keep_transient=True),
                    _without_boundary_marker(source_working[boundary:], keep_transient=True),
                )
            )
            task.add_done_callback(_prefetch_done)
            state['prefetch_task'] = task
        return {**body, 'messages': _without_boundary_marker(messages)}

    if not boundary:
        raise RuntimeError(
            'Context limit reached, but no complete earlier user turn can be compacted; provider request was not sent'
        )
    compacted_messages = _without_boundary_marker(source_working[:boundary], keep_transient=True)
    summary_recent_messages = _without_boundary_marker(source_working[boundary:], keep_transient=True)
    recent_messages = _without_boundary_marker(working[boundary:], keep_transient=True)
    event_emitter = None
    if metadata.get('chat_id') and metadata.get('message_id'):
        from open_webui.socket.main import get_event_emitter

        event_emitter = await get_event_emitter(metadata)
    await _emit_compaction_status(event_emitter, 'Compacting context', False)
    try:
        prefetch_task = state.get('prefetch_task')
        if isinstance(prefetch_task, asyncio.Task):
            summary, summary_meta, history_entry = await prefetch_task
        else:
            summary, summary_meta, history_entry = await _create_checkpoint(
                request,
                user,
                model_id,
                models,
                metadata,
                state,
                config,
                compacted_messages,
                summary_recent_messages,
            )
        compacted_body = {
            **body,
            'messages': [
                *system_messages,
                render_summary_message(summary, summary_meta),
                *_without_boundary_marker(recent_messages),
            ],
        }
        state.update(
            {
                'summary': summary,
                'summary_meta': summary_meta,
                'selected_history': history_entry,
            }
        )
        if finalize_candidate is not None:
            compacted_body = await finalize_candidate(compacted_body)
        after = await asyncio.to_thread(estimate_provider_tokens, compacted_body)
        if after > threshold or (force and after >= before):
            raise RuntimeError(
                'Context limit remains exceeded after the largest safe compaction; reduce the active input and retry'
            )

        state['compacted'] = True
    except Exception:
        await _emit_compaction_status(event_emitter, 'Context compaction failed', True, error=True)
        raise
    await _emit_compaction_status(event_emitter, 'Context compacted', True)
    return compacted_body


async def compact_chat_branch(request, user, chat: Any, model_id: str, models: dict) -> dict:
    config = await _load_config()
    if not config['enable']:
        return {'ok': True, 'compacted': False, 'reason': 'disabled'}

    chat_data = chat.chat or {}
    history = chat_data.get('history') or {}
    current_id = getattr(chat, 'current_message_id', None) or history.get('currentId')
    if not current_id:
        current_id = chat_data.get('currentId') or chat_data.get('branchPointMessageId')
    if not current_id and isinstance(chat_data.get('messages'), list) and chat_data['messages']:
        current_id = chat_data['messages'][-1].get('id')
    if not current_id:
        return {'ok': True, 'compacted': False, 'reason': 'empty'}

    messages_map = await Chats.get_messages_map_by_chat_id(chat.id)
    if not messages_map:
        messages_map = history.get('messages') or {}

    messages, previous_summary = _apply_latest_summary_checkpoint(get_message_list(messages_map, current_id))
    compacted_messages = messages[:-1]
    recent_messages = messages[-1:]
    if not compacted_messages or not recent_messages:
        return {'ok': True, 'compacted': False, 'reason': 'too_short'}

    summary = await _generate_summary(
        request,
        user,
        model_id,
        models,
        compacted_messages,
        recent_messages,
        previous_summary,
        config['prompt_template'],
        config['transient_patterns'],
    )
    saved = await Chats.upsert_message_to_chat_by_id_and_message_id(
        chat.id,
        current_id,
        {'contextSummary': summary},
        touch=False,
    )
    if saved is None:
        raise RuntimeError('Context compaction checkpoint could not be saved')

    return {
        'ok': True,
        'compacted': True,
        'dropped_messages': len(compacted_messages),
        'kept_messages': len(recent_messages),
        'summary_chars': len(summary),
    }


async def _load_config() -> dict:
    values = await Config.get_many(
        'chat.context_compaction.enable',
        'chat.context_compaction.token_threshold',
        'chat.context_compaction.token_cap',
        'chat.context_compaction.retention_percentage',
        'chat.context_compaction.prompt_template',
        'chat.context_compaction.soft_trigger_ratio',
        'chat.context_compaction.transient_message_patterns',
        'chat.externalized_refs.enable',
        'chat.externalized_refs.token_threshold',
    )
    token_threshold = _parse_positive_int(values.get('chat.context_compaction.token_threshold')) or 80000
    enabled = bool(values.get('chat.context_compaction.enable', False))
    return {
        'enable': enabled,
        'token_threshold': token_threshold,
        'token_cap': _parse_positive_int(values.get('chat.context_compaction.token_cap')) or token_threshold,
        'retention_percentage': _clamp_retention_percentage(values.get('chat.context_compaction.retention_percentage')),
        'prompt_template': values.get('chat.context_compaction.prompt_template', '') or '',
        'soft_trigger_ratio': _soft_trigger_ratio(values.get('chat.context_compaction.soft_trigger_ratio')),
        'externalized_refs_enable': values.get('chat.externalized_refs.enable') is True,
        'externalized_refs_token_threshold': (
            _parse_positive_int(values.get('chat.externalized_refs.token_threshold')) or 10000
        ),
        'transient_patterns': (
            tuple(
                re.compile(line.strip())
                for line in str(values.get('chat.context_compaction.transient_message_patterns') or '').splitlines()
                if line.strip()
            )
            if enabled
            else ()
        ),
    }


def _parse_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _soft_trigger_ratio(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.5
    return parsed if math.isfinite(parsed) and 0 <= parsed < 1 else 0.5


def _clamp_retention_percentage(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 40
    return min(50, max(10, parsed))


def _resolve_token_threshold(global_threshold: int, global_cap: int, metadata: dict) -> int:
    configured_threshold = _parse_positive_int((metadata.get('params') or {}).get('compact_token_threshold'))
    return min(configured_threshold or global_threshold, global_cap)


async def get_chat_context_usage(chat: Any, model_id: str | None = None) -> dict | None:
    chat_data = chat.chat or {}
    history = chat_data.get('history') or {}
    current_id = getattr(chat, 'current_message_id', None) or history.get('currentId')
    if not current_id:
        current_id = chat_data.get('currentId') or chat_data.get('branchPointMessageId')
    if not current_id and isinstance(chat_data.get('messages'), list) and chat_data['messages']:
        current_id = chat_data['messages'][-1].get('id')
    if not current_id:
        return None

    messages_map = await Chats.get_messages_map_by_chat_id(chat.id)
    messages = get_message_list(messages_map or history.get('messages') or {}, current_id)
    if not messages:
        return None

    try:
        config = await _load_config()
    except re.error:
        log.exception('Context compaction configuration is invalid; context usage is unavailable')
        return None
    if not config['enable']:
        return None

    params = ((chat.chat or {}).get('params') or {}).copy()
    if model_id:
        params['model'] = model_id
    threshold = _resolve_token_threshold(config['token_threshold'], config['token_cap'], {'params': params})
    messages, previous_summary = _apply_latest_summary_checkpoint(messages)

    tokens = await asyncio.to_thread(_candidate_input_tokens, messages, summary=previous_summary)
    return _build_context_usage(tokens, threshold)


def _build_context_usage(tokens: int, threshold: int) -> dict:
    return {
        'tokens': tokens,
        'estimated_tokens': tokens,
        'threshold': threshold,
        'percent': round((tokens / threshold) * 100) if threshold > 0 else 0,
        'source': 'estimated',
    }


def _apply_latest_summary_checkpoint(messages: list[dict]) -> tuple[list[dict], str | None]:
    summary = None
    summary_idx = None

    for idx, message in enumerate(messages):
        value = message.get('contextSummary') or message.get('context_summary')
        if isinstance(value, str) and value.strip():
            summary = value
            summary_idx = idx

    if summary_idx is None:
        return messages, None
    return messages[summary_idx:], summary


def _split_leading_system_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    boundary = 0
    while boundary < len(messages) and messages[boundary].get('role') == 'system':
        boundary += 1
    return messages[:boundary], messages[boundary:]


def _strict_usage_input_tokens(usage: dict | None) -> int | None:
    if not isinstance(usage, dict) or not usage:
        return None

    def token(key: str) -> int | None:
        value = usage.get(key)
        if key not in usage or isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return int(value) if math.isfinite(value) and int(value) == value and value >= 0 else None

    for key in ('prompt_tokens', 'prompt_eval_count'):
        if key in usage:
            value = token(key)
            return value if value and value > 0 else None
    keys = (
        ('prompt_n', 'cache_n')
        if 'prompt_n' in usage
        else ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')
    )
    values = [token(key) for key in keys if key in usage or key in {'prompt_n', 'cache_n', 'input_tokens'}]
    return sum(values) if values and None not in values and sum(values) > 0 else None


def _sanitize_token_value(value: Any) -> tuple[Any, int]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return '<media omitted>', 0
    if isinstance(value, str):
        if value.startswith('data:') and ',' in value[:256]:
            return '<media omitted>', 0
        return value, 0
    if isinstance(value, dict):
        part_type = value.get('type')
        mime_type = value.get('mime_type') or value.get('content_type')
        if part_type in _MEDIA_PART_TYPES or (
            isinstance(mime_type, str) and mime_type.startswith(('image/', 'audio/', 'video/'))
        ):
            return {'type': part_type or mime_type, 'payload': '<media omitted>'}, 1
        items = {key: _sanitize_token_value(item) for key, item in value.items()}
        return {key: item for key, (item, _) in items.items()}, sum(count for _, count in items.values())
    if isinstance(value, (list, tuple)):
        items = [_sanitize_token_value(item) for item in value]
        return [item for item, _ in items], sum(count for _, count in items)
    if value is None or isinstance(value, (bool, int, float)):
        return value, 0
    return str(value), 0


def _token_count(encoder: Any, text: str) -> int | None:
    if not text:
        return 0

    def encode(sample: str) -> int:
        return len(encoder.encode(sample, disallowed_special=()))

    try:
        total_bytes = len(text.encode('utf-8'))
        if total_bytes <= 64 * 1024:
            return encode(text)
        width = max(1, int(16 * 1024 * len(text) / total_bytes))
        middle = len(text) // 2
        samples = [text[:width], text[middle - width // 2 : middle + width // 2], text[-width:]]
        counts = [encode(sample) for sample in samples]
        sample_bytes = sum(len(sample.encode('utf-8')) for sample in samples)
    except Exception:
        return None
    return math.ceil(sum(counts) * total_bytes / sample_bytes)


@cache
def _token_encoder() -> Any | None:
    try:
        encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING_NAME)
        encoder.encode('', disallowed_special=())
        return encoder
    except Exception:
        log.exception('tiktoken is unavailable; falling back to approximate token estimation')
        return None


def _serialized_token_count(encoder: Any | None, value: Any) -> int:
    try:
        text = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    except (TypeError, ValueError):
        return _estimate_tokens(value)
    count = _token_count(encoder, text) if encoder is not None else None
    return count if count is not None else _estimate_tokens(text)


def estimate_body_tokens(body: dict) -> int:
    """Estimate tokens in provider-visible messages and request extras."""
    if not isinstance(body, dict) or not isinstance(body.get('messages'), list):
        return 0
    encoder = _token_encoder()

    total = 3
    extras = {key: body[key] for key in _BODY_TOKEN_EXTRA_KEYS if key in body and body[key] not in (None, {}, [])}
    for message in body['messages']:
        if not isinstance(message, dict):
            continue
        payload = {key: message[key] for key in _TOKEN_MESSAGE_KEYS if key in message}
        sanitized, media_count = _sanitize_token_value(payload)
        total += 4 + _serialized_token_count(encoder, sanitized) + media_count * 1000
    if extras:
        total += 4 + _serialized_token_count(encoder, extras)
    return total


def estimate_text_tokens(text: str) -> int:
    encoder = _token_encoder()
    count = _token_count(encoder, text) if encoder is not None else None
    return count if count is not None else _estimate_tokens(text)


def estimate_provider_tokens(body: dict) -> int:
    messages = body.get('messages')
    if not isinstance(messages, list):
        return 0
    for index in range(len(messages) - 1, -1, -1):
        input_tokens = messages[index].get(CONTEXT_COMPACTION_USAGE_ANCHOR_KEY)
        if isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens <= 0:
            continue
        suffix = estimate_body_tokens({'messages': messages[index:]})
        return input_tokens + max(0, suffix - 3)
    return estimate_body_tokens(body)


def _candidate_input_tokens(messages: list[dict], system_prompt: str = '', summary: str | None = None) -> int:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get('role') != 'assistant':
            continue
        info = message.get('info')
        usage = message.get('usage') or (info.get('usage') if isinstance(info, dict) else None)
        input_tokens = _strict_usage_input_tokens(usage)
        if input_tokens is None:
            continue
        suffix_tokens = estimate_body_tokens({'messages': messages[index:]})
        return input_tokens + max(0, suffix_tokens - 3)

    fallback_messages = list(messages)
    if summary:
        fallback_messages.insert(0, {'role': 'system', 'content': f'[CONVERSATION SUMMARY]\n{summary}'})
    if system_prompt:
        fallback_messages.insert(0, {'role': 'system', 'content': system_prompt})
    return estimate_body_tokens({'messages': fallback_messages})


def find_safe_compaction_boundary(
    messages: list[dict],
    retention_percentage: int = 40,
    transient_patterns: tuple[re.Pattern[str], ...] = (),
) -> int:
    """Return a user-turn boundary that keeps system instructions and tool pairs intact."""
    retention_percentage = _clamp_retention_percentage(retention_percentage)
    keep_count = max(2, len(messages) * retention_percentage // 100)
    target = max(1, len(messages) - keep_count)
    events = []
    right_requested: Counter[str] = Counter()
    right_completed: Counter[str] = Counter()
    for message in messages:
        requested = Counter(
            call['id']
            for call in message.get('tool_calls') or []
            if isinstance(call, dict) and isinstance(call.get('id'), str)
        )
        completed = Counter(
            [message['tool_call_id']]
            if message.get('role') == 'tool' and isinstance(message.get('tool_call_id'), str)
            else []
        )
        persistent_user = message.get('role') == 'user' and not _is_transient_message(message, transient_patterns)
        events.append((requested, completed, persistent_user))
        right_requested.update(requested)
        right_completed.update(completed)

    left_requested: Counter[str] = Counter()
    left_completed: Counter[str] = Counter()

    def invalid(tool_id: str) -> bool:
        left_request = left_requested[tool_id] > 0
        left_result = left_completed[tool_id] > 0
        right_request = right_requested[tool_id] > 0
        right_result = right_completed[tool_id] > 0
        return (
            left_request != left_result
            or (right_result and not right_request)
            or (left_request and right_result)
            or (right_request and left_result)
        )

    invalid_ids = {
        tool_id
        for tool_id in right_requested.keys() | right_completed.keys()
        if invalid(tool_id)
    }
    persistent_users = 0
    system_messages = 0
    selected = 0
    for boundary, message in enumerate(messages):
        if boundary > target:
            break
        if (
            persistent_users > 0
            and events[boundary][2]
            and system_messages == 0
            and not invalid_ids
        ):
            selected = boundary

        requested, completed, persistent_user = events[boundary]
        for tool_id in requested.keys() | completed.keys():
            left_requested[tool_id] += requested[tool_id]
            left_completed[tool_id] += completed[tool_id]
            right_requested[tool_id] -= requested[tool_id]
            right_completed[tool_id] -= completed[tool_id]
            if invalid(tool_id):
                invalid_ids.add(tool_id)
            else:
                invalid_ids.discard(tool_id)
        if message.get('role') == 'system':
            system_messages += 1
        if persistent_user:
            persistent_users += 1
    return selected


async def _generate_summary(
    request,
    user,
    model_id: str,
    models: dict,
    compacted_messages: list[dict],
    recent_messages: list[dict],
    previous_summary: str | None,
    summary_prompt_template: str,
    transient_patterns: tuple[re.Pattern[str], ...] = (),
) -> str:
    from open_webui.utils.chat import generate_chat_completion

    task_config = await Config.get_many(
        'task.model.params',
        'chat.context_compaction.model',
    )
    context_compaction_model = task_config.get('chat.context_compaction.model')
    task_model_id = context_compaction_model if context_compaction_model in models else model_id
    if task_model_id not in models:
        raise ValueError('No available model for context compaction')

    summary_prompt_template = summary_prompt_template.strip() or DEFAULT_CONTEXT_COMPACTION_PROMPT
    all_messages = [*compacted_messages, *recent_messages]
    prompt = replace_prompt_variable(
        summary_prompt_template,
        get_last_persistent_user_message(all_messages, transient_patterns) or '',
    )
    prompt = replace_messages_variable(prompt, all_messages)
    prompt = replace_messages_variable(prompt, compacted_messages, 'COMPACTED_MESSAGES')
    prompt = replace_messages_variable(prompt, recent_messages, 'RECENT_MESSAGES')
    prompt = prompt_variables_template(prompt, {'{{PREVIOUS_SUMMARY}}': previous_summary or ''})
    prompt = await prompt_template(prompt, user)

    task_model_params = task_config.get('task.model.params') or {}
    if not isinstance(task_model_params, dict):
        task_model_params = {}
    task_model_params = {key: value for key, value in task_model_params.items() if value is not None and value != ''}
    task_model_params = task_model_params or {
        'max_tokens': models[task_model_id].get('info', {}).get('params', {}).get('max_tokens', 1000)
    }

    payload = {
        'model': task_model_id,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'metadata': {
            **(request.state.metadata if hasattr(request.state, 'metadata') else {}),
            'task': 'context_compaction',
        },
    }

    payload = apply_params_to_form_data(payload, models[task_model_id], task_model_params)
    response = await generate_chat_completion(
        request,
        form_data=payload,
        user=user,
        bypass_filter=True,
        bypass_system_prompt=True,
    )
    return _response_text(response)


def _response_text(response: Any) -> str:
    if isinstance(response, list) and len(response) == 1:
        response = response[0]

    if isinstance(response, JSONResponse):
        if response.status_code >= 400:
            raise RuntimeError(f'Context compaction model returned HTTP {response.status_code}')
        try:
            response = JSONCodec.loads(response.body.decode('utf-8', 'replace'))
        except Exception as exc:
            raise RuntimeError('Context compaction model returned invalid JSON') from exc

    if not isinstance(response, dict):
        raise RuntimeError('Context compaction model returned an unsupported response')
    if response.get('error'):
        raise RuntimeError(f'Context compaction model failed: {response["error"]}')

    choices = response.get('choices') or []
    if choices:
        if len(choices) != 1:
            raise RuntimeError('Context compaction model returned multiple choices')
        choice = choices[0]
        message = choice.get('message') or {}
        if message.get('tool_calls') or message.get('function_call'):
            raise RuntimeError('Context compaction model requested a tool instead of returning a summary')
        if choice.get('finish_reason') not in (None, 'stop'):
            raise RuntimeError('Context compaction model stopped before completing the summary')
        content = message.get('content')
        if isinstance(content, str) and content.strip():
            return content.strip()
        raise RuntimeError('Context compaction model returned an empty summary')

    if response.get('status') not in (None, 'completed') or response.get('incomplete_details'):
        raise RuntimeError('Context compaction model stopped before completing the summary')
    parts = []
    for item in response.get('output') or []:
        if item.get('type') in {'function_call', 'tool_call'}:
            raise RuntimeError('Context compaction model requested a tool instead of returning a summary')
        if item.get('status') not in (None, 'completed'):
            raise RuntimeError('Context compaction model stopped before completing the summary')
        for content in item.get('content') or []:
            if isinstance(content, dict):
                parts.append(content.get('text') or content.get('content') or '')
    summary = '\n'.join(part for part in parts if isinstance(part, str) and part.strip()).strip()
    if not summary:
        raise RuntimeError('Context compaction model returned an empty summary')
    return summary


def _estimate_tokens(value: Any) -> int:
    if value is None:
        return 0

    if not isinstance(value, str):
        try:
            value = JSONCodec.dumps(value, ensure_ascii=False)
        except Exception:
            value = str(value)

    if not value:
        return 0

    return max(1, len(value) // 4)


_CONTEXT_ERROR_CODES = (
    'context_length_exceeded',
    'context_window_exceeded',
    'context_limit',
    'max_context_length',
    'input_length_exceeded',
    'input_too_long',
    'prompt_too_long',
    'token_limit_exceeded',
)
_CONTEXT_ERROR_MESSAGES = (
    'maximum context length',
    'max context length',
    'context length exceeded',
    'context window',
    'context limit',
    'exceeds context',
    'exceeded context',
    'prompt is too long',
    'prompt too long',
    'input is too long',
    'input too long',
    'input token count exceeds',
    'input tokens exceed',
    'prompt tokens exceed',
    'too many input tokens',
)
_NON_CONTEXT_ERROR_MARKERS = (
    'rate limit',
    'rate_limit',
    'too many requests',
    'quota',
    'tokens per minute',
    'token-per-minute',
    'insufficient_quota',
    'authentication',
    'permission',
)
_OUTPUT_TOKEN_MARKERS = (
    'max_tokens',
    'max_completion_tokens',
    'max output tokens',
    'max_output_tokens',
)
_SSE_FIELDS = ('data', 'event', 'id', 'retry')
_SSE_PREFLIGHT_MAX_BYTES = 64 * 1024
_SSE_PREFLIGHT_MAX_CHUNKS = 256


def _structured_error_strings(value: Any) -> tuple[list[str], list[str]]:
    codes: list[str] = []
    messages: list[str] = []

    def visit(item: Any, key: str = '') -> None:
        if isinstance(item, str):
            (codes if key in {'code', 'type', 'error_code'} else messages).append(item)
        elif isinstance(item, list):
            for child in item:
                visit(child, key)
        elif isinstance(item, dict):
            for raw_key, child in item.items():
                child_key = str(raw_key).lower()
                if child_key in {
                    'code',
                    'type',
                    'error_code',
                    'message',
                    'detail',
                    'msg',
                    'error_description',
                    'error',
                    'errors',
                    'response',
                    'cause',
                }:
                    visit(child, child_key)

    visit(value)
    return codes, messages


def _is_context_overflow_payload(value: Any) -> bool:
    codes, messages = _structured_error_strings(value)
    joined_codes = ' '.join(codes).lower()
    joined_messages = ' '.join(messages).lower()
    combined = f'{joined_codes} {joined_messages}'
    if any(marker in combined for marker in _NON_CONTEXT_ERROR_MARKERS):
        return False
    if any(marker in joined_messages for marker in _OUTPUT_TOKEN_MARKERS) and not any(
        anchor in joined_messages for anchor in ('context', 'input', 'prompt')
    ):
        return False
    return any(code in joined_codes for code in _CONTEXT_ERROR_CODES) or any(
        marker in joined_messages for marker in _CONTEXT_ERROR_MESSAGES
    )


def is_context_overflow_error(value: Any) -> bool:
    """Return whether an HTTP provider result is a structured context overflow."""
    if isinstance(value, JSONResponse):
        status_code = value.status_code
        try:
            payload = json.loads(value.body.decode('utf-8', 'replace'))
        except Exception:
            return False
        if not isinstance(payload, (dict, list)):
            return False
    elif isinstance(value, HTTPException):
        status_code = value.status_code
        payload = value.detail
    else:
        return False
    return status_code in {400, 413, 422} and _is_context_overflow_payload(payload)


class _SSEProbeParser:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self._buffer = ''
        self._data: list[str] = []
        self._event = ''

    def feed(self, chunk: bytes | str) -> list[tuple[str, Any]]:
        text = self._decoder.decode(chunk, final=False) if isinstance(chunk, bytes) else str(chunk)
        return self._process(text, final=False)

    def flush(self) -> list[tuple[str, Any]]:
        return self._process(self._decoder.decode(b'', final=True), final=True)

    def has_unsafe_pending_line(self) -> bool:
        line = self._buffer
        if not line:
            return False
        if line.startswith(':'):
            return False
        return not any(field.startswith(line) or line == field or line.startswith(f'{field}:') for field in _SSE_FIELDS)

    def _process(self, text: str, *, final: bool) -> list[tuple[str, Any]]:
        self._buffer += text
        tokens: list[tuple[str, Any]] = []
        while self._buffer:
            cr = self._buffer.find('\r')
            lf = self._buffer.find('\n')
            positions = [position for position in (cr, lf) if position >= 0]
            if not positions:
                break
            index = min(positions)
            if self._buffer[index] == '\r' and index == len(self._buffer) - 1 and not final:
                break
            width = 2 if self._buffer[index : index + 2] == '\r\n' else 1
            line = self._buffer[:index]
            self._buffer = self._buffer[index + width :]
            tokens.extend(self._process_line(line))

        if final and self._buffer:
            line, self._buffer = self._buffer, ''
            tokens.extend(self._process_line(line))
        if final:
            event = self._consume_event()
            if event is not None:
                tokens.append(('event', event))
        return tokens

    def _process_line(self, line: str) -> list[tuple[str, Any]]:
        if line == '':
            event = self._consume_event()
            return [('event', event)] if event is not None else []
        if line.startswith(':'):
            return []
        field, separator, value = line.partition(':')
        if field not in _SSE_FIELDS:
            return [('unsafe', None)]
        if separator and value.startswith(' '):
            value = value[1:]
        if field == 'data':
            self._data.append(value if separator else '')
        elif field == 'event':
            self._event = value if separator else ''
        return []

    def _consume_event(self) -> tuple[str, str] | None:
        if not self._data:
            self._event = ''
            return None
        event = self._event, '\n'.join(self._data)
        self._data = []
        self._event = ''
        return event


def _stream_value_present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, list):
        return any(_stream_value_present(item) for item in value)
    if not isinstance(value, dict):
        return value not in (None, False)
    if value.get('type') in {
        'function_call',
        'tool_call',
        'computer_call',
        'file_search_call',
        'web_search_call',
    }:
        return True
    return any(
        _stream_value_present(value.get(key))
        for key in ('content', 'output', 'summary', 'text', 'refusal', 'arguments', 'tool_calls', 'function_call')
        if key in value
    )


def _responses_output_item_present(value: Any) -> bool:
    if not isinstance(value, dict):
        return _stream_value_present(value)
    return value.get('type') != 'message' or _stream_value_present(value)


def _stream_error_source(payload: dict, event_name: str) -> Any | None:
    if payload.get('error'):
        return payload['error']
    if payload.get('type') == 'response.failed':
        response = payload.get('response')
        return response.get('error') if isinstance(response, dict) else None
    if payload.get('type') == 'error' or event_name == 'error':
        return payload
    return None


def _responses_stream_event_state(payload: dict, event_type: str) -> str:
    if event_type in {'response.created', 'response.in_progress'}:
        response = payload.get('response')
        output = response.get('output') if isinstance(response, dict) else None
        return (
            'output'
            if output is not None
            and (
                not isinstance(output, list)
                or any(_responses_output_item_present(item) for item in output)
            )
            else 'control'
        )
    if event_type in {
        'response.output_item.added',
        'response.content_part.added',
        'response.reasoning_summary_part.added',
    }:
        value = payload.get('item') if event_type == 'response.output_item.added' else payload.get('part')
        present = (
            _responses_output_item_present(value)
            if event_type == 'response.output_item.added'
            else _stream_value_present(value)
        )
        return 'output' if present else 'control'
    if event_type.startswith('response.'):
        if event_type.endswith('.delta'):
            return 'output' if _stream_value_present(payload.get('delta')) else 'control'
        return 'output'


def _chat_stream_event_state(payload: dict) -> str:
    choices = payload.get('choices')
    if isinstance(choices, list) and choices:
        for choice in choices:
            if not isinstance(choice, dict) or choice.get('finish_reason') not in (None, ''):
                return 'output'
            delta = choice.get('delta')
            if not isinstance(delta, dict):
                return 'output'
            if delta.get('tool_calls') or delta.get('function_call'):
                return 'output'
            if any(
                _stream_value_present(delta.get(key))
                for key in ('content', 'reasoning', 'reasoning_content', 'reasoning_details', 'thinking', 'refusal')
            ):
                return 'output'
        return 'control'
    if (
        payload.get('usage') is not None
        and not choices
        and not any(
            _stream_value_present(payload.get(key))
            for key in ('content', 'reasoning', 'reasoning_content', 'refusal', 'tool_calls', 'function_call')
        )
    ):
        return 'control'
    return 'output'


def _stream_event_state(event: tuple[str, str]) -> str:
    event_name, data = event
    if not data:
        return 'control'
    if data.strip() == '[DONE]':
        return 'output'
    try:
        payload = json.loads(data)
    except Exception:
        return 'output'
    if not isinstance(payload, dict):
        return 'output'

    error = _stream_error_source(payload, event_name)
    if error is not None:
        return 'retry' if _is_context_overflow_payload(error) else 'output'
    event_type = payload.get('type', '')
    if event_type.startswith('response.'):
        return _responses_stream_event_state(payload, event_type)
    return _chat_stream_event_state(payload)


def _stream_tokens_state(tokens: list[tuple[str, Any]]) -> str | None:
    for kind, value in tokens:
        if kind == 'unsafe':
            return 'output'
        state = _stream_event_state(value)
        if state != 'control':
            return state
    return None


async def _replay_stream(chunks: list[Any], iterator=None, response: StreamingResponse | None = None):
    try:
        for chunk in chunks:
            yield chunk
        if iterator is not None:
            async for chunk in iterator:
                yield chunk
    finally:
        if response is not None:
            await _close_stream_response(response, iterator)


async def _close_stream_response(response: StreamingResponse, iterator) -> None:
    close = getattr(iterator, 'aclose', None)
    if callable(close):
        try:
            await close()
        except Exception:
            log.debug('Failed to close context-overflow stream iterator', exc_info=True)
    background = response.background
    response.background = None
    if background is not None:
        try:
            await background()
        except Exception:
            log.debug('Failed to close context-overflow stream background task', exc_info=True)


async def _snapshot_retry_body(body: dict) -> dict:
    def snapshot() -> dict:
        copied = copy.deepcopy({key: value for key, value in body.items() if key != 'metadata'})
        if 'metadata' in body:
            metadata = body['metadata']
            copied['metadata'] = dict(metadata) if isinstance(metadata, dict) else metadata
        return copied

    return await asyncio.to_thread(snapshot)


async def _preflight_stream_context_overflow(response: StreamingResponse) -> bool:
    if 'text/event-stream' not in response.headers.get('content-type', '').lower():
        return False

    iterator = response.body_iterator.__aiter__()
    buffered: list[Any] = []
    buffered_bytes = 0
    parser = _SSEProbeParser()
    try:
        while True:
            chunk = await iterator.__anext__()
            buffered.append(chunk)
            if isinstance(chunk, bytes):
                buffered_bytes += len(chunk)
            elif isinstance(chunk, str):
                buffered_bytes += len(chunk.encode('utf-8'))
            else:
                response.body_iterator = _replay_stream(buffered, iterator, response)
                return False
            if buffered_bytes > _SSE_PREFLIGHT_MAX_BYTES or len(buffered) > _SSE_PREFLIGHT_MAX_CHUNKS:
                response.body_iterator = _replay_stream(buffered, iterator, response)
                return False
            state = _stream_tokens_state(parser.feed(chunk))
            if state == 'output':
                response.body_iterator = _replay_stream(buffered, iterator, response)
                return False
            if state == 'retry':
                await _close_stream_response(response, iterator)
                response.body_iterator = _replay_stream(buffered)
                return True
            if parser.has_unsafe_pending_line():
                response.body_iterator = _replay_stream(buffered, iterator, response)
                return False
    except StopAsyncIteration:
        state = _stream_tokens_state(parser.flush())
        if state == 'output':
            response.body_iterator = _replay_stream(buffered, response=response)
            return False
        if state == 'retry':
            await _close_stream_response(response, iterator)
            response.body_iterator = _replay_stream(buffered)
            return True
        response.body_iterator = _replay_stream(buffered, response=response)
        return False
    except Exception:
        await _close_stream_response(response, iterator)
        raise


async def _retry_candidate(retry, body: dict) -> dict | None:
    if not callable(retry):
        return None
    try:
        candidate = await retry(body)
        if not isinstance(candidate, dict) or candidate == body:
            return None
        return candidate
    except Exception:
        log.debug('Context-overflow retry preparation failed', exc_info=True)
        return None


async def forward_with_context_retry(send, body: dict, retry=None):
    """Send once and return the response with the exact candidate used."""
    if not callable(retry):
        return await send(body), body
    initial_body = await _snapshot_retry_body(body)
    try:
        response = await send(body)
    except Exception as error:
        if not is_context_overflow_error(error):
            raise
        candidate = await _retry_candidate(retry, initial_body)
        if candidate is None:
            raise
        actual_body = await _snapshot_retry_body(candidate)
        return await send(candidate), actual_body

    retryable = is_context_overflow_error(response)
    if isinstance(response, StreamingResponse):
        retryable = await _preflight_stream_context_overflow(response)
    if not retryable:
        return response, initial_body

    candidate = await _retry_candidate(retry, initial_body)
    if candidate is None:
        return response, initial_body
    actual_body = await _snapshot_retry_body(candidate)
    return await send(candidate), actual_body
