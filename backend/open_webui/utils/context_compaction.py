from __future__ import annotations

import asyncio
import codecs
import copy
import json
import logging
import math
import re
from collections import Counter
from dataclasses import replace
from functools import cache
from typing import Any

import tiktoken
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from open_webui.config import TIKTOKEN_ENCODING_NAME
from open_webui.models.chat_messages import ChatMessages
from open_webui.models.chats import Chats
from open_webui.models.config import Config
from open_webui.utils.chat_id import is_saved_chat_id
from open_webui.utils.externalized_refs import RefEntry, can_externalize_refs, make_ref_entry, project_tool_refs
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
_SUMMARY_TRANSCRIPT_VARIABLE_RE = re.compile(
    r'\{\{(?:MESSAGES|COMPACTED_MESSAGES|RECENT_MESSAGES)'
    r'(?::(?:START|END|MIDDLETRUNCATE):\d+)?(?:\|\w+:\d+)?\}\}'
)
_SUMMARY_PROMPT_VARIABLE_RE = re.compile(
    r'\{\{prompt(?::(?:start|end|middletruncate):\d+)?\}\}',
    re.IGNORECASE,
)
_SUMMARY_MESSAGE_KEYS = (
    'role',
    'content',
    'name',
    'tool_call_id',
    'tool_calls',
    'function_call',
    'reasoning_content',
    'reasoning_details',
    'thinking',
    'refusal',
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

The preceding messages are the exact checkpoint source. If they include an
existing <auto_compaction_context>, merge it with the newer messages. Output
only the reusable continuity summary. Do not continue the conversation or call tools."""


def _non_image_files(messages: list[dict]) -> list[dict]:
    files: dict[str, dict] = {}
    for message in messages:
        for item in message.get('files') or []:
            if not isinstance(item, dict):
                continue
            if item.get('type') == 'image' or str(item.get('content_type') or '').startswith('image/'):
                continue
            file_id = item.get('id')
            if isinstance(file_id, str) and file_id:
                files.setdefault(file_id, item)
    return list(files.values())


def _absorbed_files(prefix: list[dict], recent: list[dict]) -> list[dict]:
    retained_ids = {item['id'] for item in _non_image_files(recent)}
    return [item for item in _non_image_files(prefix) if item['id'] not in retained_ids]


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


def _canonical_history_text(messages: list[dict[str, Any]]) -> str:
    records = [
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        for message in messages
        for item in _canonical_messages(message)
    ]
    return '\n'.join(records)


def _canonical_history_entry(messages: list[dict[str, Any]]) -> RefEntry:
    text = _canonical_history_text(messages)
    entry = make_ref_entry(text, kind='history')
    if entry is None:
        raise CanonicalHistoryError('history source is not valid UTF-8')
    return entry


def _history_position_key(position: tuple[int, int | None]) -> tuple[int, int]:
    message_index, output_index = position
    return message_index, -1 if output_index is None else output_index


def _history_prefix_messages(
    messages: list[dict[str, Any]],
    carrier_index: int,
    output_index: int | None = None,
) -> list[dict[str, Any]]:
    prefix = list(messages[:carrier_index])
    if output_index is None:
        return prefix
    carrier = messages[carrier_index]
    output = carrier.get('output')
    if output_index <= 0 or not isinstance(output, list):
        return prefix
    partial = dict(carrier)
    partial['content'] = ''
    partial['output'] = output[:output_index]
    prefix.append(partial)
    return prefix


def _checkpoint_positions(
    messages: list[dict[str, Any]],
) -> list[tuple[int, int | None, str]]:
    positions = []
    for message_index, message in enumerate(messages):
        summary = message.get('contextSummary') or message.get('context_summary')
        if isinstance(summary, str) and summary.strip():
            positions.append((message_index, None, summary.strip()))
        output = message.get('output')
        if not isinstance(output, list):
            continue
        for output_index, item in enumerate(output):
            if not isinstance(item, dict):
                continue
            summary = item.get('contextSummary') or item.get('context_summary')
            if isinstance(summary, str) and summary.strip():
                positions.append((message_index, output_index, summary.strip()))
    return positions


def _bind_history_loader(
    entry: RefEntry,
    messages: list[dict[str, Any]],
    selected_index: int,
    selected_output_index: int | None = None,
) -> RefEntry:
    selected = _history_position_key((selected_index, selected_output_index))
    checkpoint_positions = tuple(
        (message_index, output_index)
        for message_index, output_index, _ in _checkpoint_positions(messages)
        if _history_position_key((message_index, output_index)) < selected
    )
    resolved: dict[tuple[int, int | None], RefEntry | None] = {}
    canonical_tool_sources: dict[int, tuple[str, ...]] = {}
    measured_tools: dict[tuple[int, int], RefEntry | None] = {}
    source_messages = _history_prefix_messages(
        messages,
        selected_index,
        selected_output_index,
    )

    def ancestor_at(position: tuple[int, int | None]) -> RefEntry | None:
        if position not in resolved:
            ancestor = _canonical_history_entry(
                _history_prefix_messages(messages, position[0], position[1])
            )
            resolved[position] = (
                replace(ancestor, load_history=load_ancestors)
                if ancestor.text and entry.text.startswith(ancestor.text)
                else None
            )
        return resolved[position]

    def tool_entries(requested: str | None) -> tuple[RefEntry, ...]:
        entries = []
        for message_index, message in enumerate(source_messages):
            if message_index not in canonical_tool_sources:
                if message.get('role') == 'tool':
                    candidates = (_canonical_direct_history_message(message),)
                elif message.get('role') == 'assistant' and message.get('output'):
                    candidates = _canonical_output_messages(message['output'])
                else:
                    candidates = ()
                canonical_tool_sources[message_index] = tuple(
                    content
                    for item in candidates
                    if item is not None
                    and item.get('role') == 'tool'
                    and isinstance((content := item.get('content')), str)
                )
            sources = canonical_tool_sources[message_index]
            for source_index in range(len(sources)):
                key = message_index, source_index
                if key not in measured_tools:
                    measured_tools[key] = make_ref_entry(sources[source_index], kind='tool')
                candidate = measured_tools[key]
                if candidate is None:
                    continue
                if requested is not None and candidate.ref != requested:
                    continue
                entries.append(candidate)
                if requested is not None:
                    return tuple(entries)
        return tuple(entries)

    async def load_ancestors(requested: str | None) -> tuple[RefEntry, ...]:
        def resolve() -> tuple[RefEntry, ...]:
            if isinstance(requested, str) and requested.startswith('tool:'):
                return tool_entries(None if requested == 'tool:' else requested)
            entries = []
            for position in reversed(checkpoint_positions):
                ancestor = ancestor_at(position)
                if ancestor is None:
                    break
                if requested is not None and ancestor.ref != requested:
                    continue
                if requested is not None:
                    return (ancestor,)
                entries.append(ancestor)
            return tuple(entries)

        return await asyncio.to_thread(resolve)

    return replace(entry, load_history=load_ancestors)


def _history_entry_at(messages: list[dict[str, Any]], carrier_index: int) -> RefEntry | None:
    return _history_entry_at_position(messages, carrier_index, None)


def _history_entry_at_position(
    messages: list[dict[str, Any]],
    carrier_index: int,
    output_index: int | None,
) -> RefEntry | None:
    if not 0 <= carrier_index < len(messages):
        return None
    if output_index is None and carrier_index == 0:
        return None
    if output_index is not None:
        output = messages[carrier_index].get('output')
        if not isinstance(output, list) or not 0 <= output_index < len(output):
            return None
    source_messages = _history_prefix_messages(messages, carrier_index, output_index)
    if not source_messages:
        return None
    return _bind_history_loader(
        _canonical_history_entry(source_messages),
        messages,
        carrier_index,
        output_index,
    )


def _history_source(
    source: Any,
) -> tuple[list[dict[str, Any]], int] | tuple[list[dict[str, Any]], int, int] | None:
    if (
        isinstance(source, tuple)
        and len(source) in {2, 3}
        and isinstance(source[0], list)
        and isinstance(source[1], int)
        and 0 <= source[1] < len(source[0])
    ):
        if len(source) == 3:
            output = source[0][source[1]].get('output')
            if not isinstance(source[2], int) or not isinstance(output, list) or not 0 <= source[2] < len(output):
                return None
        return source
    return None


async def resolve_request_history(source: Any) -> RefEntry | None:
    if isinstance(source, RefEntry):
        return source
    checkpoint = _history_source(source)
    if checkpoint is None:
        return None
    if len(checkpoint) == 2:
        return await asyncio.to_thread(_history_entry_at, checkpoint[0], checkpoint[1])
    return await asyncio.to_thread(
        _history_entry_at_position,
        checkpoint[0],
        checkpoint[1],
        checkpoint[2],
    )


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


def _drop_rendered_summary_once(
    messages: list[dict],
    summary: str | None,
    summary_meta: dict | None,
) -> list[dict]:
    if not summary:
        return messages
    expected = render_summary_message(summary, summary_meta).get('content')
    for index, message in enumerate(messages):
        content = message.get('content')
        if isinstance(content, str) and _HISTORY_REF_XML_SUFFIX_RE.sub('', content) == expected:
            return [*messages[:index], *messages[index + 1 :]]
    return messages


def _summary_provider_messages(messages: list[dict]) -> list[dict]:
    return [{key: message[key] for key in _SUMMARY_MESSAGE_KEYS if key in message} for message in messages]


async def replay_cached_compaction_messages(messages: list[dict], state: dict) -> list[dict]:
    """Replay the request-local checkpoint without resolving or hashing history again."""
    summary = state.get('summary')
    summary_meta = state.get('summary_meta')
    checkpoint_history = _history_source(state.get('checkpoint_history'))
    carrier_id = checkpoint_history[0][checkpoint_history[1]].get('id') if checkpoint_history else None
    carrier_output_index = checkpoint_history[2] if checkpoint_history and len(checkpoint_history) == 3 else None
    prefetch = state.get('prefetch_task')
    if not summary and isinstance(prefetch, asyncio.Task):
        try:
            result = await prefetch
        except Exception:
            return messages
        if result is None:
            return messages
        summary, summary_meta, history_entry = result[:3]
        checkpoint_history = _history_source(history_entry)
        carrier_id = checkpoint_history[0][checkpoint_history[1]].get('id') if checkpoint_history else None
        carrier_output_index = checkpoint_history[2] if checkpoint_history and len(checkpoint_history) == 3 else None
        state.update(
            {
                'summary': summary,
                'summary_meta': summary_meta,
                'selected_history': history_entry,
                'checkpoint_history': history_entry,
            }
        )
    if not summary:
        summary = state.get('previous_summary')
        summary_meta = state.get('previous_summary_meta')
        carrier_id = state.get('selected_checkpoint_message_id')
        carrier_output_index = state.get('selected_checkpoint_output_index')
    if not isinstance(summary, str) or not summary.strip() or not isinstance(carrier_id, str):
        return messages

    system_messages, raw_messages = _split_leading_system_messages(messages)
    carrier_index = next(
        (index for index, message in enumerate(raw_messages) if message.get('id') == carrier_id),
        None,
    )
    if carrier_index is None:
        return messages
    if carrier_output_index is not None:
        output = raw_messages[carrier_index].get('output')
        if (
            isinstance(carrier_output_index, bool)
            or not isinstance(carrier_output_index, int)
            or not isinstance(output, list)
            or not 0 <= carrier_output_index < len(output)
        ):
            return messages
        carrier = output[carrier_output_index]
        carrier_summary = (
            carrier.get('contextSummary') or carrier.get('context_summary')
            if isinstance(carrier, dict)
            else None
        )
        if not isinstance(carrier_summary, str) or carrier_summary.strip() != summary.strip():
            return messages
    active_messages = _messages_from_checkpoint(raw_messages, carrier_index, carrier_output_index)
    summary_message = render_summary_message(
        summary, summary_meta if isinstance(summary_meta, dict) else {}
    )
    state['summary_message_content'] = summary_message['content']
    return [
        *system_messages,
        summary_message,
        *active_messages,
    ]

def set_summary_history_ref(body: dict, ref: str | None, state: dict | None = None) -> dict:
    if not isinstance(ref, str) or re.fullmatch(r'history:[0-9a-f]{64}', ref) is None:
        return body
    messages = body.get('messages')
    if not isinstance(messages, list):
        return body
    held = state.get('summary_message_content') if isinstance(state, dict) else None
    if not isinstance(held, str):
        return body
    replacement = f'<history_ref>{ref}</history_ref>'
    updated = None
    for index, message in enumerate(messages):
        content = message.get('content') if isinstance(message, dict) else None
        if content is not held:
            continue
        closing = '</auto_compaction_context>'
        clean = _HISTORY_REF_XML_SUFFIX_RE.sub('', content)
        clean = f'{clean[: -len(closing)]}{replacement}{closing}'
        if clean != content:
            updated = list(messages)
            updated[index] = {**message, 'content': clean}
            # The rewrite mints a new str; keep the held reference tracking the
            # content actually sent to the provider. Externally altered copies
            # never match `is` above, so they can never update this reference.
            state['summary_message_content'] = clean
        break
    return {**body, 'messages': updated} if updated is not None else body


def _checkpoint_summary(messages: list[dict]) -> tuple[int | None, int | None, str | None]:
    selected: tuple[int, int | None, str] | None = None
    for message_index, output_index, summary in _checkpoint_positions(messages):
        selected = message_index, output_index, summary
    return selected or (None, None, None)


def _messages_from_checkpoint(
    messages: list[dict],
    message_index: int,
    output_index: int | None,
) -> list[dict]:
    if not 0 <= message_index < len(messages):
        return list(messages)
    active = list(messages[message_index:])
    if not active:
        return active
    carrier = dict(active[0])
    carrier.pop('usage', None)
    info = carrier.get('info')
    if isinstance(info, dict) and 'usage' in info:
        carrier['info'] = {key: value for key, value in info.items() if key != 'usage'}
    active[0] = carrier
    if output_index is None:
        return active
    output = carrier.get('output')
    if (
        isinstance(output_index, bool)
        or not isinstance(output_index, int)
        or not isinstance(output, list)
        or not 0 <= output_index < len(output)
    ):
        return list(messages)
    carrier['content'] = ''
    carrier['output'] = output[output_index:]
    return active


def _stored_checkpoint_view(
    messages: list[dict],
) -> tuple[list[dict], list[dict], list[dict], int | None, int | None, str | None]:
    system_messages, checkpoint_messages = _split_leading_system_messages(messages)
    message_index, output_index, summary = _checkpoint_summary(checkpoint_messages)
    active_messages = (
        _messages_from_checkpoint(checkpoint_messages, message_index, output_index)
        if message_index is not None
        else list(checkpoint_messages)
    )
    return system_messages, checkpoint_messages, active_messages, message_index, output_index, summary


def replay_stored_compaction_checkpoint(messages: list[dict]) -> tuple[list[dict], dict]:
    """Replay a persisted checkpoint without consulting mutable admin configuration."""
    system, history, active, message_index, output_index, summary = _stored_checkpoint_view(messages)
    if message_index is None or not isinstance(summary, str):
        return messages, {}
    position = (
        (history, message_index, output_index)
        if output_index is not None
        else (history, message_index)
    )
    summary_message = render_summary_message(summary)
    state = {
        'active_offset': message_index,
        'checkpoint_messages': history,
        'selected_checkpoint_message_id': history[message_index].get('id'),
        'selected_checkpoint_output_index': output_index,
        'selected_history': position,
        'previous_summary': summary,
        'previous_summary_meta': {},
        'summary_message_content': summary_message['content'],
    }
    return [*system, summary_message, *active], state


async def prepare_compaction_messages(messages: list[dict], metadata: dict) -> tuple[list[dict], dict]:
    """Apply one stored checkpoint and mark one safe future cut without DB state."""
    system_messages, checkpoint_messages, active_messages, summary_index, summary_output_index, previous_summary = (
        _stored_checkpoint_view(messages)
    )
    config = await _load_config()
    state: dict[str, Any] = {'config': config}
    active_offset = summary_index if summary_index is not None else 0
    state['active_offset'] = active_offset
    state['checkpoint_messages'] = checkpoint_messages
    if summary_index is None and (not config['enable'] or metadata.get('task') == 'context_compaction'):
        return messages, state
    active_messages = await asyncio.to_thread(copy.deepcopy, active_messages)
    summary_meta: dict[str, Any] = {}
    if summary_index is not None:
        state['selected_checkpoint_message_id'] = checkpoint_messages[summary_index].get('id')
        state['selected_checkpoint_output_index'] = summary_output_index
        state['selected_history'] = (
            (checkpoint_messages, summary_index, summary_output_index)
            if summary_output_index is not None
            else (checkpoint_messages, summary_index)
        )
        summary_meta['historical_user_messages'] = await asyncio.to_thread(
            _historical_user_excerpts,
            _history_prefix_messages(checkpoint_messages, summary_index, summary_output_index),
            _DEFAULT_EXCERPT_BYTES,
            _DEFAULT_EXCERPT_COUNT,
            config['transient_patterns'],
        )
    if not config['enable'] or metadata.get('task') == 'context_compaction':
        if previous_summary:
            state['previous_summary'] = previous_summary
            state['previous_summary_meta'] = summary_meta
            summary_message = render_summary_message(previous_summary, summary_meta)
            state['summary_message_content'] = summary_message['content']
            active_messages = [summary_message, *active_messages]
        return [*system_messages, *active_messages], state
    for index in range(len(active_messages) - 1, -1, -1):
        message = active_messages[index]
        if message.get('role') != 'assistant':
            continue
        info = message.get('info')
        usage = message.get('usage') or (info.get('usage') if isinstance(info, dict) else None)
        if _is_merged_cache_usage(usage):
            break
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
        state['absorbed_files'] = _absorbed_files(
            active_messages[:boundary],
            active_messages[boundary:],
        )

    if previous_summary:
        summary_message = render_summary_message(previous_summary, summary_meta)
        state['summary_message_content'] = summary_message['content']
        active_messages = [summary_message, *active_messages]
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


async def _generate_checkpoint(
    request,
    user,
    model_id: str,
    models: dict,
    metadata: dict,
    state: dict,
    config: dict,
    compacted_messages: list[dict],
    recent_messages: list[dict],
    *,
    checkpoint_history: tuple[list[dict[str, Any]], int] | None = None,
    absorbed_files: list[dict] | None = None,
) -> tuple[str, dict[str, Any], tuple[list[dict[str, Any]], int], str]:
    checkpoint_history = _history_source(checkpoint_history or state.get('checkpoint_history'))
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
        state.get('previous_summary'),
        config['prompt_template'],
        config['transient_patterns'],
        state.get('absorbed_files') if absorbed_files is None else absorbed_files,
        previous_summary_meta=state.get('previous_summary_meta'),
        externalized_refs_enable=config.get('externalized_refs_enable', False),
        externalized_refs_token_threshold=config.get('externalized_refs_token_threshold', 10000),
    )
    return summary, summary_meta, checkpoint_history, checkpoint_message_id


async def _save_checkpoint(chat_id: str, checkpoint_message_id: str, summary: str) -> None:
    if not await ChatMessages.update_context_summary(chat_id, checkpoint_message_id, summary):
        raise RuntimeError('Context compaction checkpoint could not be saved; provider request was not sent')


async def _finalize_prefetch_checkpoint(
    generation: asyncio.Task,
    chat_id: str,
) -> tuple[str, dict[str, Any], tuple[list[dict[str, Any]], int], str]:
    summary, summary_meta, checkpoint_history, checkpoint_message_id = await generation
    await _save_checkpoint(chat_id, checkpoint_message_id, summary)
    return summary, summary_meta, checkpoint_history, checkpoint_message_id


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


def _checkpoint_for_boundary(
    state: dict,
    working: list[dict],
    boundary: int,
) -> tuple[tuple[list[dict[str, Any]], int], list[dict]] | None:
    default = _history_source(state.get('checkpoint_history'))
    history = default[0] if default is not None else state.get('checkpoint_messages')
    if not isinstance(history, list):
        return None
    carrier_id = working[boundary].get('id') if boundary < len(working) else None
    if not isinstance(carrier_id, str):
        return None
    raw_boundary = next(
        (index for index, message in enumerate(history) if message.get('id') == carrier_id),
        None,
    )
    if raw_boundary is None or raw_boundary <= 0:
        return None
    active_offset = state.get('active_offset')
    if not isinstance(active_offset, int) or not 0 <= active_offset < raw_boundary:
        active_offset = 0
    return (
        (history, raw_boundary),
        _absorbed_files(history[active_offset:raw_boundary], history[raw_boundary:]),
    )


async def compact_provider_payload(
    request,
    user,
    body: dict,
    metadata: dict,
    model_id: str,
    models: dict,
    state: dict,
    *,
    projected_messages: list[dict] | None = None,
) -> dict:
    """Create a durable checkpoint before request filters can alter the DB branch."""
    messages = body.get('messages')
    if not isinstance(messages, list):
        return body
    config = state.get('config') or await _load_config()
    if not config['enable']:
        return body
    if metadata.get('task') == 'context_compaction' or body.get('previous_response_id'):
        return {**body, 'messages': _without_boundary_marker(messages)}

    system_messages, working = _split_leading_system_messages(messages)
    projected_system, projected_working = _split_leading_system_messages(
        projected_messages if isinstance(projected_messages, list) else messages
    )
    if len(projected_working) != len(working):
        projected_system, projected_working = system_messages, working
    boundary = next(
        (index for index, message in enumerate(working) if message.get(_BOUNDARY_KEY) is True),
        None,
    )
    projected_body = {
        **body,
        'messages': [*projected_system, *projected_working],
    }
    before = await asyncio.to_thread(estimate_provider_tokens, projected_body)
    threshold = _resolve_token_threshold(config['token_threshold'], config['token_cap'], metadata)
    if before <= threshold:
        soft_ratio = config['soft_trigger_ratio']
        if (
            boundary
            and soft_ratio > 0
            and before >= int(threshold * soft_ratio)
            and not isinstance(state.get('prefetch_task'), asyncio.Task)
        ):
            # ponytail: duplicate cross-worker prefetches are harmless; add Redis
            # dedup only if measured summary cost warrants the coordination.
            generation = asyncio.create_task(
                _generate_checkpoint(
                    request,
                    user,
                    model_id,
                    models,
                    metadata,
                    state,
                    config,
                    _without_boundary_marker(projected_working[:boundary], keep_transient=True),
                    _without_boundary_marker(projected_working[boundary:], keep_transient=True),
                )
            )
            task = asyncio.create_task(
                _finalize_prefetch_checkpoint(generation, metadata['chat_id'])
            )
            task.add_done_callback(_prefetch_done)
            state['prefetch_task'] = task
        return {**body, 'messages': _without_boundary_marker(messages)}

    largest = find_safe_compaction_boundary(
        projected_working,
        config['retention_percentage'],
        config['transient_patterns'],
        maximize=True,
    )
    if not boundary:
        if not largest or _checkpoint_for_boundary(state, working, largest) is None:
            return {**body, 'messages': _without_boundary_marker(messages)}
        boundary = largest
    event_emitter = None
    if metadata.get('chat_id') and metadata.get('message_id'):
        from open_webui.socket.main import get_event_emitter

        event_emitter = await get_event_emitter(metadata)
    await _emit_compaction_status(event_emitter, 'Compacting context', False)
    try:
        boundaries = [boundary]
        if largest > boundary and _checkpoint_for_boundary(state, working, largest) is not None:
            boundaries.append(largest)

        compacted_body = None
        summary = None
        summary_meta = None
        history_entry = None
        selected_checkpoint = None
        prefetch_task = state.pop('prefetch_task', None)
        checkpoint_already_saved = False
        for candidate_boundary in boundaries:
            compacted_messages = _without_boundary_marker(
                projected_working[:candidate_boundary],
                keep_transient=True,
            )
            projected_recent = _without_boundary_marker(
                projected_working[candidate_boundary:],
                keep_transient=True,
            )
            recent_messages = _without_boundary_marker(working[candidate_boundary:], keep_transient=True)
            checkpoint = _checkpoint_for_boundary(state, working, candidate_boundary)
            if checkpoint is None:
                continue
            if candidate_boundary == boundary and isinstance(prefetch_task, asyncio.Task):
                summary, summary_meta, history_entry, checkpoint_message_id = await prefetch_task
                checkpoint_already_saved = True
            else:
                checkpoint_already_saved = False
                summary, summary_meta, history_entry, checkpoint_message_id = await _generate_checkpoint(
                    request,
                    user,
                    model_id,
                    models,
                    metadata,
                    state,
                    config,
                    compacted_messages,
                    projected_recent,
                    checkpoint_history=checkpoint[0],
                    absorbed_files=checkpoint[1],
                )
            # The held identity must track the candidate actually returned to
            # the provider; the projected copy below is estimation-only and its
            # separate render must never become the held reference.
            summary_message = render_summary_message(summary, summary_meta)
            compacted_body = {
                **body,
                'messages': [
                    *system_messages,
                    summary_message,
                    *recent_messages,
                ],
            }
            projected_candidate = {
                **compacted_body,
                'messages': [
                    *projected_system,
                    render_summary_message(summary, summary_meta),
                    *projected_recent,
                ],
            }
            after = await asyncio.to_thread(estimate_provider_tokens, projected_candidate)
            if after <= threshold:
                selected_checkpoint = checkpoint
                break
            compacted_body = None

        if compacted_body is None or summary is None or summary_meta is None:
            raise RuntimeError(
                'Context limit remains exceeded after the largest safe compaction; reduce the active input and retry'
            )
        if not checkpoint_already_saved:
            await _save_checkpoint(metadata['chat_id'], checkpoint_message_id, summary)
        state.update(
            {
                'summary': summary,
                'summary_meta': summary_meta,
                'previous_summary': summary,
                'previous_summary_meta': summary_meta,
                'selected_history': history_entry,
                'checkpoint_history': history_entry,
                'absorbed_files': selected_checkpoint[1],
                'durable_compacted': True,
                'summary_message_content': summary_message['content'],
            }
        )
    except Exception:
        await _emit_compaction_status(event_emitter, 'Context compaction failed', True, error=True)
        raise
    await _emit_compaction_status(event_emitter, 'Context compacted', True)
    return compacted_body


def _nested_checkpoint(
    metadata: dict,
    messages: list[dict],
    output: list[dict] | None,
    carrier: dict | None,
    message_start: int | None,
) -> tuple[list[dict], list[dict], list[dict], int] | None:
    if (
        not is_saved_chat_id(metadata.get('chat_id'))
        or not isinstance(metadata.get('message_id'), str)
        or not isinstance(output, list)
        or not isinstance(carrier, dict)
        or not isinstance(message_start, int)
        or not 0 < message_start < len(messages)
    ):
        return None
    carrier_index = next((index for index, item in enumerate(output) if item is carrier), None)
    if carrier_index is None:
        return None
    system_messages, working = _split_leading_system_messages(messages)
    working_start = message_start - len(system_messages)
    if not 0 < working_start < len(working):
        return None
    return system_messages, working[:working_start], working[working_start:], carrier_index


async def _save_nested_checkpoint(
    chat_id: str,
    message_id: str,
    output: list[dict],
    carrier: dict,
    carrier_index: int,
    summary: str,
) -> None:
    stored_output = list(output)
    stored_carrier = dict(carrier)
    stored_carrier['contextSummary'] = summary
    stored_output[carrier_index] = stored_carrier
    saved = await Chats.upsert_message_to_chat_by_id_and_message_id(
        chat_id,
        message_id,
        {'output': stored_output},
        touch=False,
    )
    if saved is None:
        raise RuntimeError('Context compaction checkpoint could not be saved; provider request was not sent')
    carrier['contextSummary'] = summary


async def _nested_checkpoint_history(
    state: dict,
    message_id: str,
    output: list[dict],
    carrier_index: int,
    transient_patterns: tuple[re.Pattern[str], ...],
) -> tuple[list[dict], tuple[list[dict], int, int], dict[str, Any]]:
    history = list(state.get('checkpoint_messages') or [])
    message_index = next(
        (index for index in range(len(history) - 1, -1, -1) if history[index].get('id') == message_id),
        None,
    )
    if message_index is None:
        message_index = len(history)
        history.append({'id': message_id, 'role': 'assistant', 'content': '', 'output': output})
    else:
        history[message_index] = {**history[message_index], 'output': output}
    position = (history, message_index, carrier_index)
    summary_meta = {
        'historical_user_messages': await asyncio.to_thread(
            _historical_user_excerpts,
            _history_prefix_messages(history, message_index, carrier_index),
            _DEFAULT_EXCERPT_BYTES,
            _DEFAULT_EXCERPT_COUNT,
            transient_patterns,
        )
    }
    return history, position, summary_meta


async def compact_transient_provider_payload(
    request,
    user,
    body: dict,
    metadata: dict,
    model_id: str,
    models: dict,
    state: dict,
    *,
    checkpoint_output: list[dict] | None = None,
    checkpoint_carrier: dict | None = None,
    checkpoint_message_start: int | None = None,
) -> dict:
    """Compact the provider payload and persist an in-turn cut when one is available."""
    messages = body.get('messages')
    if not isinstance(messages, list):
        return body
    config = state.get('config') or await _load_config()
    if not config['enable'] or metadata.get('task') == 'context_compaction' or body.get('previous_response_id'):
        return body

    before = await asyncio.to_thread(estimate_provider_tokens, body)
    threshold = _resolve_token_threshold(config['token_threshold'], config['token_cap'], metadata)
    if before <= threshold:
        return body

    nested_checkpoint = _nested_checkpoint(
        metadata,
        messages,
        checkpoint_output,
        checkpoint_carrier,
        checkpoint_message_start,
    )
    if nested_checkpoint is not None:
        system_messages, compacted_messages, recent_messages, carrier_index = nested_checkpoint
        event_emitter = None
        if metadata.get('chat_id') and metadata.get('message_id'):
            from open_webui.socket.main import get_event_emitter

            event_emitter = await get_event_emitter(metadata)
        await _emit_compaction_status(event_emitter, 'Compacting context', False)
        try:
            summary = await _generate_summary(
                request,
                user,
                model_id,
                models,
                _without_boundary_marker(compacted_messages, keep_transient=True),
                _without_boundary_marker(recent_messages, keep_transient=True),
                state.get('previous_summary'),
                config['prompt_template'],
                config['transient_patterns'],
                previous_summary_meta=state.get('previous_summary_meta'),
                externalized_refs_enable=config.get('externalized_refs_enable', False),
                externalized_refs_token_threshold=config.get('externalized_refs_token_threshold', 10000),
            )
            history, position, summary_meta = await _nested_checkpoint_history(
                state,
                metadata['message_id'],
                checkpoint_output,
                carrier_index,
                config['transient_patterns'],
            )
            summary_message = render_summary_message(summary, summary_meta)
            candidate = {
                **body,
                'messages': [
                    *system_messages,
                    summary_message,
                    *_without_boundary_marker(recent_messages, keep_transient=True),
                ],
            }
            after = await asyncio.to_thread(estimate_provider_tokens, candidate)
            if after > threshold:
                raise RuntimeError(
                    'Context limit remains exceeded after preserving the latest completed tool round; '
                    'reduce the active input and retry'
                )
            await _save_nested_checkpoint(
                metadata['chat_id'],
                metadata['message_id'],
                checkpoint_output,
                checkpoint_carrier,
                carrier_index,
                summary,
            )
            state.update(
                {
                    'summary': summary,
                    'summary_meta': summary_meta,
                    'previous_summary': summary,
                    'previous_summary_meta': summary_meta,
                    'selected_checkpoint_message_id': metadata['message_id'],
                    'selected_checkpoint_output_index': carrier_index,
                    'selected_history': position,
                    'checkpoint_history': position,
                    'checkpoint_messages': history,
                    'active_offset': len(history) - 1,
                    'compacted': True,
                    'durable_compacted': True,
                    'summary_message_content': summary_message['content'],
                }
            )
        except Exception:
            await _emit_compaction_status(event_emitter, 'Context compaction failed', True, error=True)
            raise
        await _emit_compaction_status(event_emitter, 'Context compacted', True)
        return candidate

    system_messages, working = _split_leading_system_messages(messages)
    boundary = find_safe_compaction_boundary(
        working,
        config['retention_percentage'],
        config['transient_patterns'],
        allow_tool_rounds=True,
    )
    largest = find_safe_compaction_boundary(
        working,
        config['retention_percentage'],
        config['transient_patterns'],
        maximize=True,
        allow_tool_rounds=True,
    )
    if not boundary:
        boundary = largest
    if not boundary:
        raise RuntimeError(
            'Context limit reached, but no complete earlier user turn can be compacted; provider request was not sent'
        )
    boundaries = [boundary, *([largest] if largest > boundary else [])]

    event_emitter = None
    if metadata.get('chat_id') and metadata.get('message_id'):
        from open_webui.socket.main import get_event_emitter

        event_emitter = await get_event_emitter(metadata)
    await _emit_compaction_status(event_emitter, 'Compacting context', False)
    try:
        compacted_body = None
        for candidate_boundary in boundaries:
            compacted_messages = _without_boundary_marker(
                working[:candidate_boundary],
                keep_transient=True,
            )
            recent_messages = _without_boundary_marker(working[candidate_boundary:], keep_transient=True)
            summary = await _generate_summary(
                request,
                user,
                model_id,
                models,
                compacted_messages,
                recent_messages,
                state.get('previous_summary'),
                config['prompt_template'],
                config['transient_patterns'],
                previous_summary_meta=state.get('previous_summary_meta'),
                externalized_refs_enable=config.get('externalized_refs_enable', False),
                externalized_refs_token_threshold=config.get('externalized_refs_token_threshold', 10000),
            )
            summary_message = render_summary_message(summary)
            candidate = {
                **body,
                'messages': [
                    *system_messages,
                    summary_message,
                    *recent_messages,
                ],
            }
            after = await asyncio.to_thread(estimate_provider_tokens, candidate)
            if after <= threshold:
                compacted_body = candidate
                break
        if compacted_body is None:
            raise RuntimeError(
                'Context limit remains exceeded after the largest safe compaction; reduce the active input and retry'
            )
        ref_config = state.get('externalized_refs') or {}
        registry = ref_config.get('registry')
        history_entry = (
            await asyncio.to_thread(_history_entry_at, working, candidate_boundary)
            if ref_config.get('enable') is True
            and ref_config.get('native') is True
            and isinstance(registry, dict)
            and can_externalize_refs(body, native=True, registry=registry)
            else None
        )
        summary_meta: dict[str, Any] = {}
        updates = {
            'summary': summary,
            'summary_meta': summary_meta,
            'previous_summary': summary,
            'previous_summary_meta': summary_meta,
            'compacted': True,
            'summary_message_content': summary_message['content'],
        }
        if history_entry is not None:
            updates['selected_history'] = history_entry
            updates['checkpoint_history'] = history_entry
        state.update(updates)
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

    branch_messages = get_message_list(messages_map, current_id)
    _system, _checkpoint_messages, active_messages, _summary_index, _summary_output, previous_summary = (
        _stored_checkpoint_view(branch_messages)
    )
    # Split the checkpoint-view active suffix in the saved-message space so the
    # tip message never leaks its own tool rounds into the summary input after
    # provider expansion, and leading system messages stay out of the summary.
    compacted_source = active_messages[:-1]
    recent_source = active_messages[-1:]
    if not compacted_source or not recent_source:
        return {'ok': True, 'compacted': False, 'reason': 'too_short'}
    absorbed_files = _absorbed_files(compacted_source, recent_source)
    model = models.get(model_id, {})
    compacted_messages, recent_messages = await asyncio.gather(
        _completed_turn_provider_messages(compacted_source, user, model),
        _completed_turn_provider_messages(recent_source, user, model),
    )
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
        absorbed_files,
        externalized_refs_enable=config.get('externalized_refs_enable', False),
        externalized_refs_token_threshold=config.get('externalized_refs_token_threshold', 10000),
    )
    if not await ChatMessages.update_context_summary(chat.id, current_id, summary):
        raise RuntimeError('Context compaction checkpoint could not be saved')

    return {
        'ok': True,
        'compacted': True,
        'dropped_messages': len(compacted_source),
        'kept_messages': len(recent_source),
        'summary_chars': len(summary),
    }


async def _completed_turn_provider_messages(
    messages: list[dict],
    user: Any,
    model: dict,
) -> list[dict]:
    from open_webui.utils.middleware import (
        convert_url_images_to_base64,
        get_reasoning_format,
        inject_message_file_images,
        process_messages_with_output,
        sanitize_tool_pairs,
    )

    def prepare() -> list[dict]:
        prepared = inject_message_file_images(copy.deepcopy(messages))
        prepared = process_messages_with_output(
            prepared,
            reasoning_format=get_reasoning_format(model),
            mark_transient=True,
        )
        return sanitize_tool_pairs(prepared)

    prepared = await asyncio.to_thread(prepare)
    return (await convert_url_images_to_base64({'messages': prepared}, user=user))['messages']


async def _completed_turn_checkpoint(
    request: Request,
    user: Any,
    messages: list[dict],
    metadata: dict,
    model_id: str,
    models: dict,
    config: dict,
) -> str | None:
    _, _, active, _, _, previous_summary = _stored_checkpoint_view(messages)
    if not active or active[-1].get('id') != metadata['message_id']:
        return None
    compacted_source = active[:-1]
    if not compacted_source:
        return None
    recent_source = active[-1:]
    model = models.get(model_id, {})
    compacted_messages, recent_messages = await asyncio.gather(
        _completed_turn_provider_messages(compacted_source, user, model),
        _completed_turn_provider_messages(recent_source, user, model),
    )
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
        _absorbed_files(compacted_source, recent_source),
        previous_summary_meta={},
        externalized_refs_enable=config.get('externalized_refs_enable', False),
        externalized_refs_token_threshold=config.get('externalized_refs_token_threshold', 10000),
    )
    await _save_checkpoint(metadata['chat_id'], metadata['message_id'], summary)
    return summary


def start_completed_turn_compaction_prefetch(
    request: Request,
    user: Any,
    messages: list[dict],
    metadata: dict,
    model_id: str,
    models: dict,
    state: dict,
    usage: dict | None,
) -> asyncio.Task | None:
    config = state.get('config') or {}
    if (
        not config.get('enable')
        or not is_saved_chat_id(metadata.get('chat_id'))
        or not isinstance(metadata.get('message_id'), str)
        or metadata.get('assistant_message_id')
        or metadata.get('task') == 'context_compaction'
        or isinstance(state.get('completed_prefetch_task'), asyncio.Task)
    ):
        return None

    durability_repair = state.get('compacted') is True and state.get('durable_compacted') is not True
    if not durability_repair:
        total = usage.get('total_tokens') if isinstance(usage, dict) else None
        if isinstance(usage, dict) and any(
            key in usage for key in ('cache_creation_input_tokens', 'cache_read_input_tokens')
        ):
            parts = [
                usage.get(key, 0)
                for key in (
                    'input_tokens',
                    'cache_creation_input_tokens',
                    'cache_read_input_tokens',
                    'output_tokens',
                )
            ]
            if all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                and value >= 0
                for value in parts
            ):
                if (
                    isinstance(total, bool)
                    or not isinstance(total, (int, float))
                    or not math.isfinite(total)
                ):
                    total = 0
                total = max(total, sum(parts))
        ratio = config.get('soft_trigger_ratio', 0)
        if (
            isinstance(total, bool)
            or not isinstance(total, (int, float))
            or not math.isfinite(total)
            or ratio <= 0
        ):
            return None
        threshold = _resolve_token_threshold(config['token_threshold'], config['token_cap'], metadata)
        if total < int(threshold * ratio) or total >= threshold:
            return None

    task = asyncio.create_task(
        _completed_turn_checkpoint(
            request,
            user,
            messages,
            metadata,
            model_id,
            models,
            config,
        )
    )
    task.add_done_callback(_prefetch_done)
    state['completed_prefetch_task'] = task
    return task


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
        # The threshold doubles as the reader response token budget; keep the
        # 1000 floor so reader pages stay non-empty.
        'externalized_refs_token_threshold': max(
            1000, _parse_positive_int(values.get('chat.externalized_refs.token_threshold')) or 10000
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


async def get_chat_context_usage(
    chat: Any,
    model_id: str | None = None,
    *,
    messages_map: dict | None = None,
) -> dict | None:
    chat_data = chat.chat or {}
    history = chat_data.get('history') or {}
    current_id = getattr(chat, 'current_message_id', None) or history.get('currentId')
    if not current_id:
        current_id = chat_data.get('currentId') or chat_data.get('branchPointMessageId')
    if not current_id and isinstance(chat_data.get('messages'), list) and chat_data['messages']:
        current_id = chat_data['messages'][-1].get('id')
    if not current_id:
        return None

    if messages_map is None:
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
    system, _, active, message_index, _, summary = _stored_checkpoint_view(messages)
    if message_index is None:
        return messages, None
    return [*system, *active], summary


def _split_leading_system_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    boundary = 0
    while boundary < len(messages) and messages[boundary].get('role') == 'system':
        boundary += 1
    return messages[:boundary], messages[boundary:]


def _is_merged_cache_usage(usage: Any) -> bool:
    return isinstance(usage, dict) and 'prompt_tokens' in usage and any(
        key in usage for key in ('cache_creation_input_tokens', 'cache_read_input_tokens')
    )


def _strict_usage_input_tokens(usage: dict | None) -> int | None:
    if not isinstance(usage, dict) or not usage:
        return None
    # merge_usage synthesizes prompt_tokens from only the latest non-cache leg.
    if _is_merged_cache_usage(usage):
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
        if _is_merged_cache_usage(usage):
            break
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
    *,
    maximize: bool = False,
    allow_tool_rounds: bool = False,
) -> int:
    """Return a user-turn boundary that keeps system instructions and tool pairs intact."""
    retention_percentage = _clamp_retention_percentage(retention_percentage)
    keep_count = max(2, len(messages) * retention_percentage // 100)
    target = len(messages) - 1 if maximize else max(1, len(messages) - keep_count)
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
    completed_round = 0

    def follows_complete_tool_round(boundary: int) -> bool:
        cursor = boundary - 1
        completed: Counter[str] = Counter()
        while cursor >= 0 and messages[cursor].get('role') == 'tool':
            tool_id = messages[cursor].get('tool_call_id')
            if isinstance(tool_id, str):
                completed[tool_id] += 1
            cursor -= 1
        if not completed or cursor < 0:
            return False
        requested = Counter(
            call['id']
            for call in messages[cursor].get('tool_calls') or []
            if isinstance(call, dict) and isinstance(call.get('id'), str)
        )
        return messages[cursor].get('role') == 'assistant' and requested == completed

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
        if (
            allow_tool_rounds
            and persistent_users > 0
            and boundary > 0
            and message.get('role') == 'assistant'
            and follows_complete_tool_round(boundary)
            and system_messages == 0
            and not invalid_ids
        ):
            completed_round = boundary

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
    return max(selected, completed_round)


async def _summary_file_sources(request, user, model_id: str, messages: list[dict], files: list[dict]) -> list[dict]:
    from open_webui.utils.middleware import chat_completion_files_handler

    async def discard_event(_event: Any) -> None:
        pass

    items = await asyncio.to_thread(copy.deepcopy, files)
    for item in items:
        item['context'] = 'full'
    _, flags = await chat_completion_files_handler(
        request,
        {
            'model': model_id,
            'messages': messages,
            'metadata': {'files': items},
        },
        {'__event_emitter__': discard_event},
        user,
    )
    sources = flags.get('sources') or []
    if not sources:
        raise RuntimeError('Context compaction could not read files that would be removed from active context')
    return sources


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
    absorbed_files: list[dict] | None = None,
    *,
    previous_summary_meta: dict | None = None,
    externalized_refs_enable: bool = False,
    externalized_refs_token_threshold: int = 10000,
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

    compacted_messages = _drop_rendered_summary_once(
        compacted_messages,
        previous_summary,
        previous_summary_meta,
    )
    recent_messages = _drop_rendered_summary_once(
        recent_messages,
        previous_summary,
        previous_summary_meta,
    )
    all_messages = [*compacted_messages, *recent_messages]
    if externalized_refs_enable:
        all_messages = await project_tool_refs(
            all_messages,
            threshold_tokens=externalized_refs_token_threshold,
            count_tokens=estimate_text_tokens,
        )
        compacted_messages = all_messages[: len(compacted_messages)]
        recent_messages = all_messages[len(compacted_messages) :]

    custom_prompt = summary_prompt_template.strip()
    legacy_transcript_prompt = bool(
        custom_prompt
        and (
            _SUMMARY_TRANSCRIPT_VARIABLE_RE.search(custom_prompt)
            or '{{PREVIOUS_SUMMARY}}' in custom_prompt
            or _SUMMARY_PROMPT_VARIABLE_RE.search(custom_prompt)
        )
    )
    summary_prompt_template = custom_prompt or DEFAULT_CONTEXT_COMPACTION_PROMPT
    prompt_compacted_messages = compacted_messages
    prompt_all_messages = [*prompt_compacted_messages, *recent_messages]
    prompt = replace_prompt_variable(
        summary_prompt_template,
        get_last_persistent_user_message(prompt_all_messages, transient_patterns) or '',
    )
    if custom_prompt:
        prompt = replace_messages_variable(prompt, prompt_all_messages)
        prompt = replace_messages_variable(prompt, prompt_compacted_messages, 'COMPACTED_MESSAGES')
        prompt = replace_messages_variable(prompt, recent_messages, 'RECENT_MESSAGES')
        prompt = prompt_variables_template(prompt, {'{{PREVIOUS_SUMMARY}}': previous_summary or ''})
    if legacy_transcript_prompt and previous_summary and '{{PREVIOUS_SUMMARY}}' not in custom_prompt:
        prompt = f'{render_summary_message(previous_summary, previous_summary_meta)["content"]}\n\n{prompt}'
    prompt = await prompt_template(prompt, user)

    if legacy_transcript_prompt:
        summary_messages = [{'role': 'user', 'content': prompt}]
    else:
        summary_messages = _without_boundary_marker(
            await asyncio.to_thread(copy.deepcopy, compacted_messages),
        )
        if previous_summary:
            summary_messages.insert(0, render_summary_message(previous_summary, previous_summary_meta))
        summary_messages.append({'role': 'user', 'content': prompt})
    if absorbed_files:
        from open_webui.utils.middleware import apply_source_context_to_messages

        sources = await _summary_file_sources(request, user, task_model_id, summary_messages, absorbed_files)
        summary_messages = await apply_source_context_to_messages(
            request,
            summary_messages,
            sources,
            prompt,
        )

    task_model_params = task_config.get('task.model.params') or {}
    if not isinstance(task_model_params, dict):
        task_model_params = {}
    task_model_params = {key: value for key, value in task_model_params.items() if value is not None and value != ''}
    task_model_params = task_model_params or {
        'max_tokens': models[task_model_id].get('info', {}).get('params', {}).get('max_tokens', 1000)
    }

    summary_metadata = dict(request.state.metadata) if hasattr(request.state, 'metadata') else {}
    summary_metadata.pop('tools', None)
    summary_metadata.pop('files', None)
    summary_metadata['task'] = 'context_compaction'
    summary_scope = {
        **request.scope,
        'state': {**(request.scope.get('state') or {}), 'metadata': summary_metadata},
    }
    summary_request = Request(summary_scope, receive=request.receive)
    payload = {
        'model': task_model_id,
        'messages': _summary_provider_messages(summary_messages),
        'stream': False,
        'metadata': summary_metadata,
    }

    payload = apply_params_to_form_data(payload, models[task_model_id], task_model_params)
    for key in ('tools', 'tool_choice', 'functions', 'function_call', 'parallel_tool_calls'):
        payload.pop(key, None)
    response = await generate_chat_completion(
        summary_request,
        form_data=payload,
        user=user,
        bypass_filter=True,
        bypass_system_prompt=True,
    )
    return await _response_text(response)


async def _stream_summary_payload(response: StreamingResponse) -> dict:
    from open_webui.utils.middleware import handle_responses_streaming_event

    iterator = response.body_iterator.__aiter__()
    parser = _SSEParser()
    mode = None
    output = []
    chat_parts: list[str] = []
    completed = False

    def consume(tokens: list[tuple[str, Any]]) -> None:
        nonlocal mode, output, completed
        for kind, value in tokens:
            if kind == 'unsafe':
                raise RuntimeError('Context compaction model returned an invalid event stream')
            event_name, data = value
            if data.strip() == '[DONE]':
                continue
            try:
                payload = JSONCodec.loads(data)
            except Exception as exc:
                raise RuntimeError('Context compaction model returned an invalid event stream') from exc
            if not isinstance(payload, dict):
                raise RuntimeError('Context compaction model returned an invalid event stream')
            if event_name == 'error' or payload.get('error') or payload.get('type') == 'error':
                raise RuntimeError(f'Context compaction model failed: {payload.get("error") or payload}')

            event_type = payload.get('type') or event_name
            if isinstance(event_type, str) and event_type.startswith('response.'):
                if mode == 'chat':
                    raise RuntimeError('Context compaction model returned an invalid event stream')
                mode = 'responses'
                response_data = payload.get('response')
                if event_type in {'response.failed', 'response.incomplete'} or (
                    isinstance(response_data, dict)
                    and (
                        response_data.get('status') not in (None, 'completed', 'in_progress')
                        or response_data.get('incomplete_details')
                    )
                ):
                    raise RuntimeError('Context compaction model stopped before completing the summary')
                if payload.get('type') != event_type:
                    payload = {**payload, 'type': event_type}
                output, metadata = handle_responses_streaming_event(payload, output)
                if metadata and metadata.get('error'):
                    raise RuntimeError(f'Context compaction model failed: {metadata["error"]}')
                if metadata and metadata.get('done'):
                    completed = True
                continue

            if mode == 'responses':
                raise RuntimeError('Context compaction model returned an invalid event stream')
            mode = 'chat'
            choices = payload.get('choices')
            if not choices and payload.get('usage') is not None:
                continue
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise RuntimeError('Context compaction model returned multiple choices')
            choice = choices[0]
            if choice.get('index', 0) != 0:
                raise RuntimeError('Context compaction model returned multiple choices')
            delta = choice.get('delta') or choice.get('message') or {}
            if not isinstance(delta, dict):
                raise RuntimeError('Context compaction model returned an invalid event stream')
            if delta.get('tool_calls') or delta.get('function_call'):
                raise RuntimeError('Context compaction model requested a tool instead of returning a summary')
            content = delta.get('content')
            if isinstance(content, str):
                chat_parts.append(content)
            finish_reason = choice.get('finish_reason')
            if finish_reason in (None, ''):
                continue
            if finish_reason in {'tool_calls', 'function_call'}:
                raise RuntimeError('Context compaction model requested a tool instead of returning a summary')
            if finish_reason != 'stop':
                raise RuntimeError('Context compaction model stopped before completing the summary')
            completed = True

    try:
        async for chunk in iterator:
            if not isinstance(chunk, (bytes, str)):
                raise RuntimeError('Context compaction model returned an invalid event stream')
            consume(parser.feed(chunk))
        consume(parser.flush())
    finally:
        await _close_stream_response(response, iterator)

    if not completed:
        raise RuntimeError('Context compaction model stopped before completing the summary')
    if mode == 'responses':
        return {'status': 'completed', 'output': output}
    return {
        'choices': [
            {
                'message': {'content': ''.join(chat_parts)},
                'finish_reason': 'stop',
            }
        ]
    }


async def _response_text(response: Any) -> str:
    if isinstance(response, list) and len(response) == 1:
        response = response[0]

    if isinstance(response, StreamingResponse):
        if response.status_code >= 400:
            await _close_stream_response(response, response.body_iterator.__aiter__())
            raise RuntimeError(f'Context compaction model returned HTTP {response.status_code}')
        response = await _stream_summary_payload(response)

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
        if not isinstance(item, dict):
            continue
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


_SSE_FIELDS = ('data', 'event', 'id', 'retry')


class _SSEParser:
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


async def _close_stream_response(response: StreamingResponse, iterator) -> None:
    close = getattr(iterator, 'aclose', None)
    if callable(close):
        try:
            await close()
        except Exception:
            log.debug('Failed to close context compaction stream iterator', exc_info=True)
    background = response.background
    response.background = None
    if background is not None:
        try:
            await background()
        except Exception:
            log.debug('Failed to close context compaction stream background task', exc_info=True)
