from __future__ import annotations

import asyncio
import codecs
import copy
import hashlib
import json
import logging
import math
import re
from dataclasses import replace
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
from open_webui.utils.misc import get_content_from_message, get_message_list
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
_SUMMARY_META_KEY = 'context_compaction'
_SUMMARY_META_VERSION = 1
_DEFAULT_EXCERPT_BYTES = 512
_DEFAULT_EXCERPT_COUNT = 32
_HISTORY_FORMAT = 'canonical-history-jsonl-v1'
_HISTORY_REF_XML_RE = re.compile(r'<history_ref>history:[0-9a-f]{64}</history_ref>')
_HISTORY_IGNORED_KEYS = frozenset(
    {
        'id',
        'parentId',
        'childrenIds',
        'timestamp',
        'created_at',
        'updated_at',
        'models',
        'model',
        'done',
        'usage',
        'info',
        'sources',
        'files',
        'embeds',
        'annotation',
        'annotations',
        'reasoning',
        'reasoning_content',
        'reasoning_details',
        'status',
        'statusHistory',
        'status_history',
        'contextSummary',
        'context_summary',
        'error',
        'feedback',
        'metadata',
        'meta',
    }
)
_FILE_IDENTITY_KEYS = frozenset(
    {
        'checksum',
        'collection_name',
        'collection_names',
        'content_type',
        'file',
        'file_hash',
        'file_id',
        'filename',
        'hash',
        'id',
        'legacy',
        'meta',
        'metadata',
        'mime_type',
        'name',
        'revision',
        'sha256',
        'type',
        'version',
    }
)


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
    encoded = value.encode('utf-8')
    if len(encoded) <= limit:
        return value
    marker = '…'
    budget = max(0, limit - len(marker.encode('utf-8')))
    head = encoded[: budget // 2].decode('utf-8', 'ignore')
    tail = encoded[-(budget - budget // 2) :].decode('utf-8', 'ignore') if budget else ''
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


def is_transient_message(message: Any, patterns: tuple[re.Pattern[str], ...] = ()) -> bool:
    return _is_transient_message(message, patterns)


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
    excerpts = [
        _middle_truncate_utf8(content, byte_limit)
        for message in messages
        if message.get('role') == 'user'
        and not _is_transient_message(message, patterns)
        and (content := get_content_from_message(message))
    ]
    return excerpts[-count:]


def _canonical_history_content(value: Any, *, required: bool) -> str | list[dict[str, str]]:
    if value is None and not required:
        return ''
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise CanonicalHistoryError('unknown content shape')
    canonical = []
    for part in value:
        if not isinstance(part, dict):
            raise CanonicalHistoryError('unknown content shape')
        part_type = part.get('type')
        if part_type in {'text', 'input_text', 'output_text'}:
            if set(part) != {'type', 'text'} or not isinstance(part.get('text'), str):
                raise CanonicalHistoryError('unknown content shape')
            canonical.append({'type': 'text', 'text': part['text']})
        elif part_type == 'input_image':
            if set(part) != {'type', 'image_url'} or not isinstance(part.get('image_url'), str):
                raise CanonicalHistoryError('unknown content shape')
            canonical.append({'type': 'omitted_media', 'media': 'image'})
        elif part_type == 'image_url':
            image_url = part.get('image_url')
            if set(part) != {'type', 'image_url'} or not (
                isinstance(image_url, str)
                or (isinstance(image_url, dict) and set(image_url) == {'url'} and isinstance(image_url.get('url'), str))
            ):
                raise CanonicalHistoryError('unknown content shape')
            canonical.append({'type': 'omitted_media', 'media': 'image'})
        else:
            raise CanonicalHistoryError('unknown content shape')
    return canonical


def _canonical_history_tool_calls(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise CanonicalHistoryError('unknown tool call shape')
    calls = []
    for call in value:
        function = call.get('function') if isinstance(call, dict) else None
        if (
            not isinstance(call, dict)
            or set(call) != {'id', 'type', 'function'}
            or call.get('type') != 'function'
            or not isinstance(call.get('id'), str)
            or not isinstance(function, dict)
            or set(function) != {'name', 'arguments'}
            or not isinstance(function.get('name'), str)
            or not isinstance(function.get('arguments'), str)
        ):
            raise CanonicalHistoryError('unknown tool call shape')
        calls.append({'id': call['id'], 'name': function['name'], 'arguments': function['arguments']})
    return calls


def _canonical_direct_history_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get('role')
    semantic_keys = set(message) - _HISTORY_IGNORED_KEYS
    if role == 'user':
        if semantic_keys != {'role', 'content'}:
            raise CanonicalHistoryError('unknown message shape')
        return {'role': role, 'content': _canonical_history_content(message.get('content'), required=True)}
    if role == 'assistant':
        if not semantic_keys <= {'role', 'content', 'tool_calls'} or 'role' not in semantic_keys:
            raise CanonicalHistoryError('unknown message shape')
        canonical = {
            'role': role,
            'content': _canonical_history_content(message.get('content'), required=False),
        }
        if 'tool_calls' in message:
            calls = _canonical_history_tool_calls(message['tool_calls'])
            if calls:
                canonical['tool_calls'] = calls
        return canonical
    if role == 'tool':
        if semantic_keys != {'role', 'content', 'tool_call_id'} or not isinstance(message.get('tool_call_id'), str):
            raise CanonicalHistoryError('unknown message shape')
        return {
            'role': role,
            'tool_call_id': message['tool_call_id'],
            'content': _canonical_history_content(message.get('content'), required=True),
        }
    raise CanonicalHistoryError('unknown message shape')


def _canonical_output_messages(output: Any) -> list[dict[str, Any]]:
    if not isinstance(output, list):
        raise CanonicalHistoryError('unknown output shape')
    requested = {
        item.get('call_id')
        for item in output
        if isinstance(item, dict) and item.get('type') == 'function_call' and isinstance(item.get('call_id'), str)
    }
    completed = {
        item.get('call_id')
        for item in output
        if isinstance(item, dict)
        and item.get('type') == 'function_call_output'
        and isinstance(item.get('call_id'), str)
    }
    messages: list[dict[str, Any]] = []
    content: list[str] = []
    calls: list[dict[str, str]] = []

    def flush() -> None:
        if content or calls:
            message: dict[str, Any] = {'role': 'assistant', 'content': '\n'.join(content) if content else ''}
            if calls:
                message['tool_calls'] = list(calls)
            messages.append(message)
            content.clear()
            calls.clear()

    for item in output:
        if not isinstance(item, dict) or not isinstance(item.get('type'), str):
            raise CanonicalHistoryError('unknown output shape')
        item_type = item['type']
        if item_type == 'message':
            parts = item.get('content')
            if not isinstance(parts, list):
                raise CanonicalHistoryError('unknown output shape')
            text = ''
            for part in parts:
                if (
                    not isinstance(part, dict)
                    or not set(part) <= {'type', 'text', 'annotations'}
                    or part.get('type') != 'output_text'
                    or not isinstance(part.get('text'), str)
                    or ('annotations' in part and not isinstance(part['annotations'], list))
                ):
                    raise CanonicalHistoryError('unknown content shape')
                text += part['text']
            if text:
                content.append(text)
        elif item_type == 'function_call':
            call_id = item.get('call_id')
            if not all(isinstance(item.get(key), str) for key in ('call_id', 'name', 'arguments')):
                raise CanonicalHistoryError('unknown output shape')
            if call_id in completed:
                calls.append({'id': call_id, 'name': item['name'], 'arguments': item['arguments']})
        elif item_type == 'function_call_output':
            call_id = item.get('call_id')
            parts = item.get('output')
            if not isinstance(call_id, str) or not isinstance(parts, list):
                raise CanonicalHistoryError('unknown output shape')
            flush()
            text = ''
            images = []
            for part in parts:
                if not isinstance(part, dict):
                    raise CanonicalHistoryError('unknown content shape')
                if part.get('type') == 'input_text':
                    if set(part) != {'type', 'text'} or not isinstance(part.get('text'), str):
                        raise CanonicalHistoryError('unknown content shape')
                    text += part['text']
                elif part.get('type') == 'input_image':
                    if set(part) != {'type', 'image_url'} or not isinstance(part.get('image_url'), str):
                        raise CanonicalHistoryError('unknown content shape')
                    images.append({'type': 'omitted_media', 'media': 'image'})
                else:
                    raise CanonicalHistoryError('unknown content shape')
            if call_id in requested:
                tool_content: str | list[dict[str, str]] = text
                if images:
                    tool_content = [{'type': 'text', 'text': text}, *images]
                messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': tool_content})
        elif item_type == 'open_webui:code_interpreter':
            code = item.get('code', '')
            code_output = item.get('output', '')
            if not isinstance(code, str):
                raise CanonicalHistoryError('unknown output shape')
            if code:
                content.append(f'<code_interpreter>\n{code}\n</code_interpreter>')
            if isinstance(code_output, dict):
                if not set(code_output) <= {'stdout', 'result', 'stderr'}:
                    raise CanonicalHistoryError('unknown output shape')
                output_text = code_output.get('stdout') or code_output.get('result') or code_output.get('stderr') or ''
                if not isinstance(output_text, str):
                    raise CanonicalHistoryError('unknown output shape')
            elif isinstance(code_output, str):
                output_text = code_output
            else:
                raise CanonicalHistoryError('unknown output shape')
            if output_text:
                content.append(f'<code_interpreter_output>\n{output_text}\n</code_interpreter_output>')
        elif item_type in {
            'reasoning',
            'web_search_call',
            'file_search_call',
            'computer_call',
        } or item_type.startswith('open_webui:'):
            continue
        else:
            raise CanonicalHistoryError('unknown output shape')
    flush()
    return messages


def _canonical_messages(
    message: dict[str, Any],
    patterns: tuple[re.Pattern[str], ...] = (),
) -> list[dict[str, Any]]:
    if message.get('role') == 'system' or _is_transient_message(message, patterns):
        return []
    if message.get('role') == 'assistant' and message.get('output'):
        semantic_keys = set(message) - _HISTORY_IGNORED_KEYS
        if not semantic_keys <= {'role', 'content', 'tool_calls', 'output'}:
            raise CanonicalHistoryError('unknown message shape')
        converted = _canonical_output_messages(message['output'])
        if converted:
            return converted
    direct = {key: value for key, value in message.items() if key != 'output'}
    return [_canonical_direct_history_message(direct)]


def _canonical_history_source(
    messages: list[dict[str, Any]],
    patterns: tuple[re.Pattern[str], ...] = (),
) -> tuple[str, str]:
    entry = _canonical_history_entry(messages, patterns)
    return entry.text, entry.ref.split(':', 1)[1]


def _canonical_history_entry(
    messages: list[dict[str, Any]],
    patterns: tuple[re.Pattern[str], ...] = (),
    expanded: list[tuple[dict[str, Any], list[dict[str, Any]]]] | None = None,
) -> RefEntry:
    expanded = expanded if expanded is not None else _expanded_history(messages, patterns)
    records = [
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        for _, canonical in expanded
        for item in canonical
    ]
    text = '\n'.join(records)
    entry = make_ref_entry(text, kind='history')
    if entry is None:
        raise CanonicalHistoryError('history source is not valid UTF-8')
    return entry


def _expanded_history(
    messages: list[dict[str, Any]],
    patterns: tuple[re.Pattern[str], ...],
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    return [(message, _canonical_messages(message, patterns)) for message in messages]


def _stable_file_identity(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: item
            for key in sorted(value)
            if key in _FILE_IDENTITY_KEYS and (item := _stable_file_identity(value[key])) not in (None, {}, [])
        }
    if isinstance(value, list):
        return [item for raw in value if (item := _stable_file_identity(raw)) not in (None, {}, [])]
    return value


def _stable_general_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: item
            for key in sorted(value)
            if key != 'distances' and (item := _stable_general_value(value[key])) not in (None, {}, [])
        }
    if isinstance(value, list):
        return [item for raw in value if (item := _stable_general_value(raw)) not in (None, {}, [])]
    return value


def _source_hash(
    messages: list[dict[str, Any]],
    patterns: tuple[re.Pattern[str], ...] = (),
    expanded_history: list[tuple[dict[str, Any], list[dict[str, Any]]]] | None = None,
) -> str:
    expanded = []
    history = expanded_history if expanded_history is not None else _expanded_history(messages, patterns)
    for raw, canonical in history:
        for index, message in enumerate(canonical):
            stable = dict(message)
            if index == 0:
                for key in ('name', 'function_call', 'sources', 'reasoning_content'):
                    if key in raw:
                        stable[key] = _stable_general_value(raw[key])
                files = _stable_file_identity(raw.get('files'))
                if files not in (None, {}, []):
                    stable['files'] = files
            expanded.append(stable)
    encoded = json.dumps(
        {'family': 'canonical-json-v1', 'messages': expanded},
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return f'sha256:{hashlib.sha256(encoded).hexdigest()}'


def _build_history_checkpoint(
    messages: list[dict[str, Any]],
    branch_anchor: str,
) -> tuple[dict[str, Any], RefEntry]:
    if not messages or not isinstance(branch_anchor, str) or messages[-1].get('id') != branch_anchor:
        raise CanonicalHistoryError('checkpoint branch anchor is invalid')
    expanded = _expanded_history(messages, ())
    entry = _canonical_history_entry(messages, (), expanded)
    raw_source_hash = entry.ref.split(':', 1)[1]
    return (
        {
            'version': _SUMMARY_META_VERSION,
            'branch_anchor': branch_anchor,
            'source_hash': _source_hash(messages, (), expanded),
            'history_ref': {
                'format': _HISTORY_FORMAT,
                'raw_source_hash': raw_source_hash,
            },
        },
        entry,
    )


def _resolve_history_checkpoint(
    messages: list[dict[str, Any]],
    carrier_index: int,
) -> RefEntry:
    carrier = messages[carrier_index]
    meta = carrier.get('meta') if isinstance(carrier.get('meta'), dict) else {}
    descriptor = meta.get(_SUMMARY_META_KEY) if isinstance(meta.get(_SUMMARY_META_KEY), dict) else None
    history_ref = descriptor.get('history_ref') if isinstance(descriptor, dict) else None
    if (
        not isinstance(descriptor, dict)
        or descriptor.get('version') != _SUMMARY_META_VERSION
        or not isinstance(descriptor.get('branch_anchor'), str)
        or re.fullmatch(r'sha256:[0-9a-f]{64}', str(descriptor.get('source_hash'))) is None
        or not isinstance(history_ref, dict)
        or set(history_ref) != {'format', 'raw_source_hash'}
        or history_ref.get('format') != _HISTORY_FORMAT
        or re.fullmatch(r'[0-9a-f]{64}', str(history_ref.get('raw_source_hash'))) is None
    ):
        raise CanonicalHistoryError('checkpoint metadata is invalid')
    branch_anchor = descriptor['branch_anchor']
    prefix = messages[:carrier_index]
    if carrier.get('parentId') != branch_anchor or not prefix or prefix[-1].get('id') != branch_anchor:
        raise CanonicalHistoryError('checkpoint branch anchor does not match the current branch')
    rebuilt, entry = _build_history_checkpoint(prefix, branch_anchor)
    if rebuilt['source_hash'] != descriptor['source_hash']:
        raise CanonicalHistoryError('checkpoint source hash does not match the current branch')
    if rebuilt['history_ref'] != history_ref:
        raise CanonicalHistoryError('checkpoint raw source hash does not match the current branch')
    return entry


def _bind_history_loader(
    entry: RefEntry,
    messages: list[dict[str, Any]],
    selected_index: int,
) -> RefEntry:
    async def load_ancestors(_requested: str | None) -> tuple[RefEntry, ...]:
        def resolve() -> tuple[RefEntry, ...]:
            entries = []
            for index in range(selected_index - 1, -1, -1):
                if not (messages[index].get('contextSummary') or messages[index].get('context_summary')):
                    continue
                try:
                    ancestor = _resolve_history_checkpoint(messages, index)
                except CanonicalHistoryError:
                    break
                entries.append(replace(ancestor, load_history=load_ancestors))
            return tuple(entries)

        return await asyncio.to_thread(resolve)

    return replace(entry, load_history=load_ancestors)


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
    carrier_id = state.get('checkpoint_message_id')
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
        if not isinstance(content, str) or not content.startswith('<auto_compaction_context>'):
            continue
        clean = _HISTORY_REF_XML_RE.sub('', content)
        if replacement:
            clean = clean.replace('</auto_compaction_context>', f'{replacement}</auto_compaction_context>', 1)
        if clean != content:
            updated = list(messages)
            updated[index] = {**message, 'content': clean}
        break
    return {**body, 'messages': updated} if updated is not None else body


def _checkpoint_summary(messages: list[dict]) -> tuple[int | None, str | None, dict]:
    selected: tuple[int, str, dict] | None = None
    for index, message in enumerate(messages):
        summary = message.get('contextSummary') or message.get('context_summary')
        if isinstance(summary, str) and summary.strip():
            meta = message.get('meta') if isinstance(message.get('meta'), dict) else {}
            summary_meta = meta.get(_SUMMARY_META_KEY) if isinstance(meta.get(_SUMMARY_META_KEY), dict) else {}
            selected = index, summary.strip(), summary_meta
    return selected or (None, None, {})


async def prepare_compaction_messages(messages: list[dict], metadata: dict) -> tuple[list[dict], dict]:
    """Apply one stored checkpoint and mark one safe future cut without DB state."""
    config = await _load_config()
    state: dict[str, Any] = {'config': config}
    if not config['enable'] or metadata.get('task') == 'context_compaction':
        return messages, state

    system_messages, raw_messages = _split_leading_system_messages(messages)
    raw_messages = await asyncio.to_thread(copy.deepcopy, raw_messages)
    summary_index, previous_summary, summary_meta = _checkpoint_summary(raw_messages)
    if summary_index is not None:
        state['selected_checkpoint_message_id'] = raw_messages[summary_index].get('id')
        carrier_meta = raw_messages[summary_index].get('meta')
        # Official v0.11 checkpoints predate history-ref metadata. Keep their
        # summary behavior, but expose no unverifiable history ref.
        if isinstance(carrier_meta, dict) and _SUMMARY_META_KEY in carrier_meta:
            state['selected_history'] = _bind_history_loader(
                await asyncio.to_thread(
                    _resolve_history_checkpoint,
                    raw_messages,
                    summary_index,
                ),
                raw_messages,
                summary_index,
            )
    active_offset = summary_index if summary_index is not None else 0
    active_messages = raw_messages[active_offset:]
    for index in range(len(active_messages) - 1, -1, -1):
        message = active_messages[index]
        if message.get('role') != 'assistant':
            continue
        info = message.get('info')
        usage = message.get('usage') or (info.get('usage') if isinstance(info, dict) else None)
        input_tokens = _strict_usage_input_tokens(usage)
        if input_tokens is not None:
            active_messages[index] = {**message, CONTEXT_COMPACTION_USAGE_ANCHOR_KEY: input_tokens}
            break
    boundary = find_safe_compaction_boundary(
        active_messages,
        config['retention_percentage'],
        config['transient_patterns'],
    )

    if boundary:
        marked = dict(active_messages[boundary])
        marked[_BOUNDARY_KEY] = True
        active_messages = [*active_messages[:boundary], marked, *active_messages[boundary + 1 :]]
        raw_boundary = active_offset + boundary
        checkpoint_message = raw_messages[raw_boundary]
        state.update(
            {
                'checkpoint_message_id': checkpoint_message.get('id'),
                'checkpoint_message_meta': checkpoint_message.get('meta') or {},
                'source_messages': raw_messages[:raw_boundary],
            }
        )

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
) -> tuple[str, dict[str, Any], RefEntry]:
    source_messages = state.get('source_messages') or []
    branch_anchor = source_messages[-1].get('id') if source_messages else None
    checkpoint_message_id = state.get('checkpoint_message_id')
    chat_id = metadata.get('chat_id')
    if not checkpoint_message_id or not is_saved_chat_id(chat_id):
        raise RuntimeError('Context compaction checkpoint is not durable; provider request was not sent')
    try:
        summary_meta, history_entry = await asyncio.to_thread(
            _build_history_checkpoint,
            source_messages,
            branch_anchor,
        )
        previous_history = state.get('selected_history')
        if isinstance(previous_history, RefEntry) and previous_history.load_history:
            history_entry = replace(history_entry, load_history=previous_history.load_history)
    except CanonicalHistoryError as exc:
        raise RuntimeError(f'Context compaction checkpoint is invalid: {exc}; provider request was not sent') from exc
    summary_meta['historical_user_messages'] = await asyncio.to_thread(
        _historical_user_excerpts,
        source_messages,
        _DEFAULT_EXCERPT_BYTES,
        _DEFAULT_EXCERPT_COUNT,
        config['transient_patterns'],
    )
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
    existing_meta = state.get('checkpoint_message_meta')
    saved = await Chats.upsert_message_to_chat_by_id_and_message_id(
        chat_id,
        checkpoint_message_id,
        {
            'contextSummary': summary,
            'meta': {
                **(existing_meta if isinstance(existing_meta, dict) else {}),
                _SUMMARY_META_KEY: summary_meta,
            },
        },
        touch=False,
    )
    if saved is None:
        raise RuntimeError('Context compaction checkpoint could not be saved; provider request was not sent')
    return summary, summary_meta, history_entry


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
) -> dict:
    """Compact the final provider candidate or fail before provider dispatch."""
    messages = body.get('messages')
    if not isinstance(messages, list):
        return body
    config = state.get('config') or await _load_config()
    if not config['enable'] or metadata.get('task') == 'context_compaction' or body.get('previous_response_id'):
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
    if before is None:
        raise RuntimeError('Context compaction could not estimate the provider input; request was not sent')
    state['tokens_before'] = before
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
        after = await asyncio.to_thread(estimate_provider_tokens, compacted_body)
        if after is None or after > threshold or (force and after >= before):
            raise RuntimeError(
                'Context limit remains exceeded after the largest safe compaction; reduce the active input and retry'
            )

        state.update(
            {
                'compacted': True,
                'tokens_after': after,
                'summary': summary,
                'summary_meta': summary_meta,
                'selected_history': history_entry,
            }
        )
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

    branch = get_message_list(messages_map, current_id)
    summary_index, previous_summary, _ = _checkpoint_summary(branch)
    if summary_index is not None:
        await asyncio.to_thread(_resolve_history_checkpoint, branch, summary_index)
    messages = branch[summary_index if summary_index is not None else 0 :]
    compacted_messages = messages[:-1]
    recent_messages = messages[-1:]
    if not compacted_messages or not recent_messages:
        return {'ok': True, 'compacted': False, 'reason': 'too_short'}

    source_messages = branch[:-1]
    branch_anchor = source_messages[-1].get('id') if source_messages else None
    summary_meta, _ = await asyncio.to_thread(
        _build_history_checkpoint,
        source_messages,
        branch_anchor,
    )
    summary_meta['historical_user_messages'] = await asyncio.to_thread(
        _historical_user_excerpts,
        source_messages,
        _DEFAULT_EXCERPT_BYTES,
        _DEFAULT_EXCERPT_COUNT,
        config['transient_patterns'],
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
    )
    carrier_meta = recent_messages[0].get('meta')
    saved = await Chats.upsert_message_to_chat_by_id_and_message_id(
        chat.id,
        current_id,
        {
            'contextSummary': summary,
            'meta': {
                **(carrier_meta if isinstance(carrier_meta, dict) else {}),
                _SUMMARY_META_KEY: summary_meta,
            },
        },
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
    )
    token_threshold = _parse_positive_int(values.get('chat.context_compaction.token_threshold')) or 80000
    return {
        'enable': bool(values.get('chat.context_compaction.enable', False)),
        'token_threshold': token_threshold,
        'token_cap': _parse_positive_int(values.get('chat.context_compaction.token_cap')) or token_threshold,
        'retention_percentage': _clamp_retention_percentage(values.get('chat.context_compaction.retention_percentage')),
        'prompt_template': values.get('chat.context_compaction.prompt_template', '') or '',
        'soft_trigger_ratio': _soft_trigger_ratio(values.get('chat.context_compaction.soft_trigger_ratio')),
        'transient_patterns': tuple(
            re.compile(line.strip())
            for line in str(values.get('chat.context_compaction.transient_message_patterns') or '').splitlines()
            if line.strip()
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

    config = await _load_config()
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
        try:
            return len(encoder.encode(sample, disallowed_special=()))
        except TypeError:
            return len(encoder.encode(sample))

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


def estimate_body_tokens(body: dict, *, encoding_name: str | None = None) -> int | None:
    """Estimate tokens in provider-visible messages and request extras."""
    if not isinstance(body, dict) or not isinstance(body.get('messages'), list):
        return None
    try:
        encoder = tiktoken.get_encoding(encoding_name or TIKTOKEN_ENCODING_NAME)
    except Exception:
        return None

    total = 3
    payloads = [
        {key: message[key] for key in _TOKEN_MESSAGE_KEYS if key in message}
        for message in body['messages']
        if isinstance(message, dict)
    ]
    extras = {key: body[key] for key in _BODY_TOKEN_EXTRA_KEYS if key in body and body[key] not in (None, {}, [])}
    for payload in payloads:
        sanitized, media_count = _sanitize_token_value(payload)
        text = json.dumps(sanitized, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        count = _token_count(encoder, text)
        if count is None:
            return None
        total += 4 + count + media_count * 1000
    if extras:
        try:
            text = json.dumps(extras, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        except (TypeError, ValueError):
            return None
        count = _token_count(encoder, text)
        if count is None:
            return None
        total += 4 + count
    return total


def estimate_text_tokens(text: str, *, encoding_name: str | None = None) -> int | None:
    try:
        encoder = tiktoken.get_encoding(encoding_name or TIKTOKEN_ENCODING_NAME)
    except Exception:
        return None
    return _token_count(encoder, text)


def estimate_provider_tokens(body: dict) -> int | None:
    estimated = estimate_body_tokens(body)
    if estimated is None:
        return None
    messages = body.get('messages')
    if not isinstance(messages, list):
        return estimated
    for index in range(len(messages) - 1, -1, -1):
        input_tokens = messages[index].get(CONTEXT_COMPACTION_USAGE_ANCHOR_KEY)
        if isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens <= 0:
            continue
        suffix = estimate_body_tokens({'messages': messages[index:]})
        if suffix is not None:
            return max(estimated, input_tokens + max(0, suffix - 3))
    return estimated


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
        if suffix_tokens is not None:
            return input_tokens + max(0, suffix_tokens - 3)

    fallback_messages = list(messages)
    if summary:
        fallback_messages.insert(0, {'role': 'system', 'content': f'[CONVERSATION SUMMARY]\n{summary}'})
    if system_prompt:
        fallback_messages.insert(0, {'role': 'system', 'content': system_prompt})
    return estimate_body_tokens({'messages': fallback_messages}) or (
        _estimate_tokens(system_prompt) + _estimate_tokens(summary or '') + _estimate_messages_tokens(messages)
    )


def _exceeds_token_threshold(messages: list[dict], system_prompt: str, summary: str | None, threshold: int) -> bool:
    if threshold <= 0:
        return False
    return _candidate_input_tokens(messages, system_prompt, summary) > threshold


def _tool_call_ids(messages: list[dict]) -> tuple[set[str], set[str]]:
    requested = {
        call['id']
        for message in messages
        for call in message.get('tool_calls') or []
        if isinstance(call, dict) and isinstance(call.get('id'), str)
    }
    completed = {
        message['tool_call_id']
        for message in messages
        if message.get('role') == 'tool' and isinstance(message.get('tool_call_id'), str)
    }
    return requested, completed


def find_safe_compaction_boundary(
    messages: list[dict],
    retention_percentage: int = 40,
    transient_patterns: tuple[re.Pattern[str], ...] = (),
) -> int:
    """Return a user-turn boundary that keeps system instructions and tool pairs intact."""
    retention_percentage = _clamp_retention_percentage(retention_percentage)
    keep_count = max(2, len(messages) * retention_percentage // 100)
    target = max(1, len(messages) - keep_count)
    boundaries = [
        idx
        for idx, message in enumerate(messages)
        if message.get('role') == 'user' and not _is_transient_message(message, transient_patterns)
    ][1:]
    for boundary in reversed(boundaries):
        if boundary > target or any(message.get('role') == 'system' for message in messages[:boundary]):
            continue
        left_requested, left_completed = _tool_call_ids(messages[:boundary])
        right_requested, right_completed = _tool_call_ids(messages[boundary:])
        if (
            left_requested != left_completed
            or not right_completed <= right_requested
            or left_requested & right_completed
            or right_requested & left_completed
        ):
            continue
        return boundary
    return 0


def _find_compaction_boundary(messages: list[dict], retention_percentage: int = 40) -> int:
    return find_safe_compaction_boundary(messages, retention_percentage)


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


def _estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        total += 4
        content = message.get('content')
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    total += _estimate_tokens(item)
                elif item.get('type') in {'image', 'image_url'}:
                    total += 1000
                else:
                    total += _estimate_tokens(item.get('text') or item.get('content') or item)
        else:
            total += _estimate_tokens(content)

        total += _estimate_tokens(message.get('output'))
        total += _estimate_tokens(message.get('tool_calls'))
        total += _estimate_tokens(message.get('files'))
    return total


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
        return 'output' if isinstance(response, dict) and _stream_value_present(response.get('output')) else 'control'
    if event_type in {
        'response.output_item.added',
        'response.content_part.added',
        'response.reasoning_summary_part.added',
    }:
        value = payload.get('item') if event_type == 'response.output_item.added' else payload.get('part')
        return 'output' if _stream_value_present(value) else 'control'
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
    metadata = body.get('metadata')
    tools = metadata.get('tools') if isinstance(metadata, dict) else None
    return await asyncio.to_thread(copy.deepcopy, body, {id(tools): tools} if isinstance(tools, dict) else None)


async def _preflight_stream_context_overflow(response: StreamingResponse) -> bool:
    if 'text/event-stream' not in response.headers.get('content-type', '').lower():
        return False

    iterator = response.body_iterator.__aiter__()
    buffered: list[Any] = []
    parser = _SSEProbeParser()
    try:
        while True:
            chunk = await iterator.__anext__()
            buffered.append(chunk)
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
        candidate = await retry(await _snapshot_retry_body(body))
        if not isinstance(candidate, dict) or candidate == body:
            return None
        return candidate
    except Exception:
        log.debug('Context-overflow retry preparation failed', exc_info=True)
        return None


async def forward_with_context_retry(send, body: dict, retry):
    """Send once and return the response with the exact candidate used."""
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
