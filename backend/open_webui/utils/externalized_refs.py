from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import re
import shlex
import time
from bisect import bisect_right
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Any

import regex
from open_webui.tools.knowledge_fs import (
    MATCH_BUDGET_SECONDS,
    MatchBudget,
    MatchBudgetExceeded,
    is_regex_pattern,
    normalize_regex,
)

_monotonic = time.monotonic


REF_EXEC_TOOL_NAME = 'auto_compact_ref_exec'
REF_EXEC_COMMAND_MAX_BYTES = 1_024
REF_EXEC_RESPONSE_MAX_BYTES = 65_536
REF_EXEC_TAIL_MAX_BYTES = 8 * 1024 * 1024
REF_TEXT_INDEX_CHARS = 16 * 1024
REF_EXEC_USAGE_ERROR = 'Error: usage: auto_compact_ref_exec(command). Expected REF: tool:<64 hex> or history:<64 hex>'

REF_EXEC_FUNCTION_SPEC: dict[str, Any] = {
    'name': REF_EXEC_TOOL_NAME,
    'description': (
        'Read externalized content in this chat. Oversized tool results and compacted '
        'history use tool:<64 hex> and history:<64 hex> references. '
        'Commands: ls [tool|history]; stat REF; '
        'wc -l|-w|-c REF; cat REF; head [-n N|-N|-c N] REF; '
        'tail [-n N|-N|-c N|-c +N] REF; sed -n Np|M,Np|M,$p REF; '
        'grep [-E] [-i] [-n] [-c] [-o] [--] PATTERN REF. '
        'grep/head/tail/sed/wc can be piped.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'command': {
                'type': 'string',
                'description': ("One command line, max 1,024 UTF-8 bytes, e.g. sed -n '1,120p' tool:<64 hex>"),
            }
        },
        'required': ['command'],
        'additionalProperties': False,
    },
}

REF_EXEC_TOOL_SPEC: dict[str, Any] = {
    'type': 'function',
    'function': REF_EXEC_FUNCTION_SPEC,
}


@dataclass(frozen=True, slots=True)
class RefEntry:
    """One immutable source reachable through a request-local reader."""

    ref: str
    text: str
    utf8_bytes: int
    line_count: int
    # Sparse character-to-UTF-8-byte checkpoints. This makes repeated paging seek
    # near the requested byte instead of rescanning the whole source each time.
    byte_index: tuple[tuple[int, int], ...]
    load_history: Callable[[str | None], Any] | None = None


@dataclass(frozen=True, slots=True)
class _Stage:
    command: str
    ref: str | None = None
    count: int | None = None
    byte_count: int | None = None
    byte_start: int | None = None
    flags: frozenset[str] = frozenset()
    pattern: str | None = None
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True, slots=True)
class _Line:
    text: str
    has_newline: bool
    source_byte_start: int | None = None
    display_prefix: str = ''
    match_start: int | None = None
    match_end: int | None = None

    @property
    def presented(self) -> str:
        return self.display_prefix + self.text


class RefExecError(RuntimeError):
    pass


_READER_MARKER = object()


def _valid_ref(value: str) -> bool:
    return re.fullmatch(r'(?:tool|history):[0-9a-f]{64}', value) is not None


def _measure_text(
    text: str,
) -> tuple[int, int, str, tuple[tuple[int, int], ...]] | None:
    digest = hashlib.sha256()
    utf8_bytes = 0
    line_count = 0
    checkpoints: list[tuple[int, int]] = []
    for offset in range(0, len(text), REF_TEXT_INDEX_CHARS):
        checkpoints.append((offset, utf8_bytes))
        chunk = text[offset : offset + REF_TEXT_INDEX_CHARS]
        try:
            encoded = chunk.encode('utf-8')
        except UnicodeEncodeError:
            return None
        utf8_bytes += len(encoded)
        line_count += chunk.count('\n')
        digest.update(encoded)
    if not checkpoints:
        checkpoints.append((0, 0))
    if checkpoints[-1] != (len(text), utf8_bytes):
        checkpoints.append((len(text), utf8_bytes))
    if text and not text.endswith('\n'):
        line_count += 1
    return (
        utf8_bytes,
        line_count,
        digest.hexdigest(),
        tuple(checkpoints),
    )


def _is_externalization_eligible(
    text: str,
    *,
    threshold_tokens: int,
    count_tokens: Callable[[str], int | None] | None,
) -> bool:
    # A source with at most this many characters can be classified for the token
    # threshold before paying for a digest. Longer valid strings are necessarily
    # over the byte cap and are externalized without tokenizer work.
    if len(text) <= REF_EXEC_RESPONSE_MAX_BYTES:
        try:
            encoded_size = len(text.encode('utf-8'))
        except UnicodeEncodeError:
            return False
        if encoded_size <= REF_EXEC_RESPONSE_MAX_BYTES:
            if count_tokens is None:
                return False
            try:
                tokens = count_tokens(text)
            except Exception:
                return False
            if tokens is None or tokens < threshold_tokens:
                return False
    return True


def _eligible_entry(
    text: str,
    *,
    threshold_tokens: int,
    count_tokens: Callable[[str], int | None] | None,
) -> RefEntry | None:
    if not _is_externalization_eligible(
        text,
        threshold_tokens=threshold_tokens,
        count_tokens=count_tokens,
    ):
        return None
    return make_ref_entry(text, kind='tool')


def make_ref_entry(
    text: str,
    *,
    kind: str,
    load_history: Callable[[str | None], Any] | None = None,
) -> RefEntry | None:
    if kind not in {'tool', 'history'}:
        return None
    measurement = _measure_text(text)
    if measurement is None:
        return None
    utf8_bytes, line_count, text_hash, byte_index = measurement
    return RefEntry(
        ref=f'{kind}:{text_hash}',
        text=text,
        utf8_bytes=utf8_bytes,
        line_count=line_count,
        byte_index=byte_index,
        load_history=load_history,
    )


def _tool_schema_name(tool: dict[str, Any]) -> str | None:
    function = tool.get('function')
    if tool.get('type') != 'function' or not isinstance(function, dict):
        return None
    name = function.get('name')
    return name if isinstance(name, str) else None


def _reader_is_selectable(body: dict[str, Any]) -> bool:
    choice = body.get('tool_choice')
    if isinstance(choice, str):
        return choice.lower() != 'none'
    if not isinstance(choice, dict):
        return True
    choice_type = choice.get('type')
    if isinstance(choice_type, str) and choice_type.lower() == 'none':
        return False
    function = choice.get('function')
    forced_name = function.get('name') if isinstance(function, dict) else choice.get('name')
    if isinstance(forced_name, str) and forced_name:
        return forced_name == REF_EXEC_TOOL_NAME
    return True


def _owned_reader(registry: dict[str, Any]) -> Callable[..., Any] | None:
    existing = registry.get(REF_EXEC_TOOL_NAME)
    if not isinstance(existing, dict):
        return None
    reader = existing.get('callable')
    if getattr(reader, '__externalized_ref_reader__', None) is not _READER_MARKER:
        return None
    if existing.get('spec') != REF_EXEC_FUNCTION_SPEC:
        return None
    return reader


def _has_collision(body: dict[str, Any], registry: dict[str, Any]) -> bool:
    owned = _owned_reader(registry)
    if REF_EXEC_TOOL_NAME in registry and owned is None:
        return True
    tools = body.get('tools')
    if tools is None:
        tools = []
    if not isinstance(tools, list):
        return True
    matching = [tool for tool in tools if isinstance(tool, dict) and _tool_schema_name(tool) == REF_EXEC_TOOL_NAME]
    if not matching:
        return False
    return owned is None or any(tool != REF_EXEC_TOOL_SPEC for tool in matching)


async def _capture_projections(
    messages: list[Any],
    *,
    threshold_tokens: int,
    count_tokens: Callable[[str], int | None] | None,
    existing_refs: Iterable[str] = (),
) -> tuple[tuple[int, dict[str, Any], RefEntry], ...]:
    existing = set(existing_refs)
    candidates = [
        (index, message, source)
        for index, message in enumerate(messages)
        if isinstance(message, dict)
        and message.get('role') == 'tool'
        and isinstance((source := message.get('content')), str)
        and (len(source) not in (69, 72) or source not in existing)
    ]

    def classify() -> tuple[tuple[int, dict[str, Any], RefEntry], ...]:
        projected: list[tuple[int, dict[str, Any], RefEntry]] = []
        for index, message, source in candidates:
            entry = _eligible_entry(
                source,
                threshold_tokens=threshold_tokens,
                count_tokens=count_tokens,
            )
            if entry is not None:
                projected.append((index, message, entry))
        return tuple(projected)

    return await asyncio.to_thread(classify)


async def externalize_refs(
    body: dict[str, Any],
    registry: dict[str, Any],
    *,
    native: bool,
    threshold_tokens: int,
    count_tokens: Callable[[str], int | None] | None = None,
    history_loader: Callable[[], Awaitable[RefEntry | None]] | None = None,
) -> bool:
    """Install or extend one Core-owned reader and project eligible tool messages.

    The function mutates ``body`` and ``registry`` only after all fallible checks.
    The original messages list and message dictionaries remain unchanged.
    """

    if (
        not native
        or body.get('stream') is not True
        or not _reader_is_selectable(body)
        or _has_collision(body, registry)
    ):
        return False
    messages = body.get('messages')
    if not isinstance(messages, list):
        return False

    reader = _owned_reader(registry)
    projections = await _capture_projections(
        messages,
        threshold_tokens=threshold_tokens,
        count_tokens=count_tokens,
        existing_refs=(reader.__externalized_ref_catalog__ if reader is not None else ()),
    )
    history_entry = await history_loader() if history_loader is not None else None
    if history_entry is not None and (
        not history_entry.ref.startswith('history:') or not _valid_ref(history_entry.ref)
    ):
        return False
    if not projections and history_entry is None:
        return False
    if not all(
        index < len(messages) and messages[index] is message and message.get('content') is entry.text
        for index, message, entry in projections
    ):
        return False

    if reader is None:
        catalog: dict[str, RefEntry] = {}
        reader = _new_reader(
            catalog,
            threshold_tokens=threshold_tokens,
            count_tokens=count_tokens,
        )
    else:
        catalog = reader.__externalized_ref_catalog__

    for _, _, entry in projections:
        catalog.setdefault(entry.ref, entry)
    if history_entry is not None:
        catalog.setdefault(history_entry.ref, history_entry)

    if REF_EXEC_TOOL_NAME not in registry:
        registry[REF_EXEC_TOOL_NAME] = {
            'spec': copy.deepcopy(REF_EXEC_FUNCTION_SPEC),
            'callable': reader,
        }

    tools = body.get('tools')
    if tools is None:
        body['tools'] = [copy.deepcopy(REF_EXEC_TOOL_SPEC)]
    elif not any(isinstance(tool, dict) and _tool_schema_name(tool) == REF_EXEC_TOOL_NAME for tool in tools):
        body['tools'] = [*tools, copy.deepcopy(REF_EXEC_TOOL_SPEC)]

    # Copy only the list and dictionaries whose content changes. Strings remain shared.
    projected_messages = list(messages)
    for index, original, entry in projections:
        message = dict(original)
        message['content'] = entry.ref
        projected_messages[index] = message
    body['messages'] = projected_messages
    return True


def _split_pipeline(command: str) -> tuple[str, ...]:
    stages: list[str] = []
    buffer: list[str] = []
    single_quoted = False
    double_quoted = False
    escaped = False
    for character in command:
        if escaped:
            buffer.append(character)
            escaped = False
            continue
        if character == '\\' and not single_quoted:
            buffer.append(character)
            escaped = True
            continue
        if character == "'" and not double_quoted:
            single_quoted = not single_quoted
            buffer.append(character)
            continue
        if character == '"' and not single_quoted:
            double_quoted = not double_quoted
            buffer.append(character)
            continue
        if character == '|' and not single_quoted and not double_quoted:
            stage = ''.join(buffer).strip()
            if not stage:
                raise RefExecError('Error: empty pipeline stage')
            stages.append(stage)
            buffer = []
            continue
        buffer.append(character)
    if escaped or single_quoted or double_quoted:
        raise RefExecError('Error: malformed quote or escape in command')
    stage = ''.join(buffer).strip()
    if not stage:
        raise RefExecError('Error: empty command or pipeline stage')
    stages.append(stage)
    return tuple(stages)


def _parse_count(tokens: list[str], *, command: str) -> tuple[int | None, int | None, int | None, list[str]]:
    remaining = list(tokens)
    line_count: int | None = 10
    byte_count: int | None = None
    byte_start: int | None = None
    if remaining and remaining[0] == '-n':
        if len(remaining) < 2 or re.fullmatch(r'[0-9]+', remaining[1]) is None:
            raise RefExecError(f'Error: invalid {command} count')
        line_count = int(remaining[1])
        remaining = remaining[2:]
    elif remaining and remaining[0] == '-c':
        if len(remaining) < 2:
            raise RefExecError(f'Error: invalid {command} byte count')
        value = remaining[1]
        if command == 'tail' and re.fullmatch(r'\+[1-9][0-9]*', value):
            byte_start = int(value[1:])
        elif re.fullmatch(r'[0-9]+', value):
            byte_count = int(value)
        else:
            raise RefExecError(f'Error: invalid {command} byte count')
        line_count = None
        remaining = remaining[2:]
    elif remaining and re.fullmatch(r'-[0-9]+', remaining[0]):
        line_count = int(remaining[0][1:])
        remaining = remaining[1:]
    return line_count, byte_count, byte_start, remaining


def _parse_grep(tokens: list[str], *, source: bool) -> _Stage:
    flags: set[str] = set()
    remaining = list(tokens)
    while remaining and remaining[0].startswith('-') and remaining[0] != '-':
        token = remaining.pop(0)
        if token == '--':
            break
        combined = token[1:]
        if not combined or any(flag not in 'Einco' for flag in combined):
            raise RefExecError('Error: invalid grep flags')
        flags.update(combined)
    expected = 2 if source else 1
    if len(remaining) != expected:
        raise RefExecError('Error: usage: grep [-Einco] [--] PATTERN [REF]')
    ref = remaining[1] if source else None
    if ref is not None and not _valid_ref(ref):
        raise RefExecError('Error: invalid externalized ref')
    return _Stage(command='grep', ref=ref, flags=frozenset(flags), pattern=remaining[0])


def _parse_sed(tokens: list[str], *, source: bool) -> _Stage:
    expected = 3 if source else 2
    if len(tokens) != expected or tokens[0] != '-n':
        raise RefExecError("Error: usage: sed -n 'M,Np' [REF]")
    match = re.fullmatch(r'([1-9][0-9]*)(?:,([1-9][0-9]*|\$))?p', tokens[1])
    if match is None:
        raise RefExecError("Error: usage: sed -n 'M,Np' [REF]")
    start = int(match.group(1))
    raw_end = match.group(2)
    end = start if raw_end is None else (None if raw_end == '$' else int(raw_end))
    if end is not None and start > end:
        raise RefExecError('Error: sed range start exceeds end')
    ref = tokens[2] if source else None
    if ref is not None and not _valid_ref(ref):
        raise RefExecError('Error: invalid externalized ref')
    return _Stage(command='sed', ref=ref, start_line=start, end_line=end)


def _parse_head_tail(command: str, arguments: list[str], *, source: bool) -> _Stage:
    count, byte_count, byte_start, remaining = _parse_count(arguments, command=command)
    if len(remaining) != int(source):
        raise RefExecError(f'Error: invalid {command} arguments')
    ref = remaining[0] if source else None
    if ref is not None and not _valid_ref(ref):
        raise RefExecError('Error: invalid externalized ref')
    return _Stage(
        command=command,
        ref=ref,
        count=count,
        byte_count=byte_count,
        byte_start=byte_start,
    )


def _parse_wc(arguments: list[str], *, source: bool) -> _Stage:
    expected = 2 if source else 1
    if len(arguments) != expected or arguments[0] not in {'-l', '-w', '-c'}:
        raise RefExecError('Error: usage: wc -l|-w|-c [REF]')
    ref = arguments[1] if source else None
    if ref is not None and not _valid_ref(ref):
        raise RefExecError('Error: invalid externalized ref')
    return _Stage(command='wc', ref=ref, flags=frozenset({arguments[0][1:]}))


def _stage_tokens(value: str) -> list[str]:
    try:
        tokens = shlex.split(value, posix=True)
    except ValueError as exc:
        raise RefExecError('Error: malformed quote in command') from exc
    if not tokens:
        raise RefExecError('Error: empty pipeline stage')
    return tokens


def _parse_stage(value: str, *, source: bool) -> _Stage:
    tokens = _stage_tokens(value)
    command, arguments = tokens[0], tokens[1:]
    if command not in {'ls', 'stat', 'wc', 'cat', 'head', 'tail', 'sed', 'grep'}:
        raise RefExecError('Error: unknown command. Available: cat, grep, head, ls, sed, stat, tail, wc')
    if not source and command not in {'grep', 'head', 'tail', 'sed', 'wc'}:
        raise RefExecError('Error: command is not a valid piped consumer')
    if command == 'ls':
        if arguments not in ([], ['tool'], ['history']):
            raise RefExecError('Error: usage: ls [tool|history]')
        return _Stage(command='ls', flags=frozenset(arguments))
    if command == 'grep':
        return _parse_grep(arguments, source=source)
    if command == 'sed':
        return _parse_sed(arguments, source=source)
    if command in {'head', 'tail'}:
        return _parse_head_tail(command, arguments, source=source)
    if command == 'wc':
        return _parse_wc(arguments, source=source)
    if len(arguments) != 1 or not _valid_ref(arguments[0]):
        raise RefExecError(f'Error: usage: {command} REF')
    return _Stage(command=command, ref=arguments[0])


def _parse_command(command: str) -> tuple[_Stage, ...]:
    if not isinstance(command, str) or not command.strip():
        raise RefExecError(REF_EXEC_USAGE_ERROR)
    try:
        encoded = command.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise RefExecError(REF_EXEC_USAGE_ERROR) from exc
    if len(encoded) > REF_EXEC_COMMAND_MAX_BYTES:
        raise RefExecError('Error: command exceeds the 1,024 UTF-8 byte parser limit')
    return tuple(_parse_stage(stage, source=index == 0) for index, stage in enumerate(_split_pipeline(command.strip())))


def _iter_text_lines(text: str, *, source_offsets: bool = False) -> Iterator[_Line]:
    offset = 0
    byte_start = 0
    while offset < len(text):
        newline = text.find('\n', offset)
        end = len(text) if newline < 0 else newline
        line = text[offset:end]
        yield _Line(
            text=line,
            has_newline=newline >= 0,
            source_byte_start=byte_start if source_offsets else None,
        )
        byte_start += len(line.encode('utf-8')) + int(newline >= 0)
        offset = end + int(newline >= 0)


def _iter_source_lines(entry: RefEntry) -> Iterator[_Line]:
    yield from _iter_text_lines(entry.text, source_offsets=True)


def _head(lines: Iterable[_Line], count: int) -> Iterator[_Line]:
    for index, line in enumerate(lines):
        if index >= count:
            return
        yield line


def _tail(lines: Iterable[_Line], count: int) -> Iterator[_Line]:
    if count <= 0:
        return
    retained: deque[_Line] = deque(maxlen=count)
    retained.extend(lines)
    yield from retained


def _sed(lines: Iterable[_Line], start: int, end: int | None) -> Iterator[_Line]:
    for ordinal, line in enumerate(lines, 1):
        if ordinal < start:
            continue
        if end is not None and ordinal > end:
            return
        yield line


def _literal_spans(
    text: str,
    pattern: str,
    *,
    ignore_case: bool,
    all_matches: bool,
) -> tuple[tuple[int, int], ...]:
    if pattern == '':
        return () if all_matches else ((0, 0),)
    flags = re.IGNORECASE if ignore_case else 0
    matches = re.finditer(re.escape(pattern), text, flags)
    if all_matches:
        return tuple(match.span() for match in matches)
    first = next(matches, None)
    return () if first is None else (first.span(),)


def _regex_spans(
    compiled: Any,
    text: str,
    budget: MatchBudget,
    *,
    all_matches: bool,
) -> tuple[tuple[int, int], ...]:
    started = _monotonic()
    try:
        if budget.remaining <= 0:
            raise TimeoutError
        matches = compiled.finditer(text, timeout=budget.remaining)
        spans = (match.span() for match in matches if not all_matches or match.end() > match.start())
        if all_matches:
            return tuple(spans)
        first = next(spans, None)
        return () if first is None else (first,)
    except TimeoutError:
        raise MatchBudgetExceeded(f'Search exceeded {MATCH_BUDGET_SECONDS:g}s, narrow the pattern') from None
    finally:
        budget.remaining -= _monotonic() - started


def _compile_grep(stage: _Stage) -> Any | None:
    pattern = stage.pattern or ''
    if 'E' not in stage.flags and not is_regex_pattern(pattern):
        return None
    try:
        return regex.compile(
            normalize_regex(pattern),
            regex.IGNORECASE if 'i' in stage.flags else 0,
        )
    except regex.error as exc:
        raise RefExecError(f'Invalid regex: {exc}') from exc


def _grep_spans(
    value: str,
    stage: _Stage,
    compiled: Any | None,
    budget: MatchBudget,
) -> tuple[tuple[int, int], ...]:
    pattern = stage.pattern or ''
    if pattern == '':
        return () if 'o' in stage.flags else ((0, 0),)
    if compiled is not None:
        return _regex_spans(
            compiled,
            value,
            budget,
            all_matches='o' in stage.flags,
        )
    return _literal_spans(
        value,
        pattern,
        ignore_case='i' in stage.flags,
        all_matches='o' in stage.flags,
    )


def _grep_matches(line: _Line, spans: tuple[tuple[int, int], ...], ordinal: int, stage: _Stage) -> Iterator[_Line]:
    value = line.presented
    prefix = f'{ordinal}:' if 'n' in stage.flags else ''
    if 'o' not in stage.flags:
        start, end = spans[0]
        yield _Line(
            text=value,
            has_newline=True,
            source_byte_start=(line.source_byte_start if not line.display_prefix else None),
            display_prefix=prefix,
            match_start=start,
            match_end=end,
        )
        return
    for start, end in spans:
        source_byte_start = None
        if line.source_byte_start is not None and not line.display_prefix:
            source_byte_start = line.source_byte_start + len(value[:start].encode('utf-8'))
        yield _Line(
            text=value[start:end],
            has_newline=True,
            source_byte_start=source_byte_start,
            display_prefix=prefix,
            match_start=0,
            match_end=end - start,
        )


def _grep(
    lines: Iterable[_Line],
    stage: _Stage,
    budget: MatchBudget,
) -> Iterator[_Line]:
    compiled = _compile_grep(stage)
    count = 0
    for ordinal, line in enumerate(lines, 1):
        spans = _grep_spans(line.presented, stage, compiled, budget)
        if not spans:
            continue
        count += 1
        if 'c' not in stage.flags:
            yield from _grep_matches(line, spans, ordinal, stage)
    if 'c' in stage.flags:
        yield _Line(str(count), False)


def _word_count(text: str) -> int:
    return sum(1 for _ in re.finditer(r'\S+', text))


def _wc(lines: Iterable[_Line], flag: str) -> Iterator[_Line]:
    line_count = 0
    word_count = 0
    byte_count = 0
    for line in lines:
        line_count += 1
        presented = line.presented
        word_count += _word_count(presented)
        byte_count += len(presented.encode('utf-8')) + int(line.has_newline)
    yield _Line(str({'l': line_count, 'w': word_count, 'c': byte_count}[flag]), False)


def _seek_source_byte(entry: RefEntry, requested: int) -> tuple[int, int]:
    requested = min(max(0, requested), entry.utf8_bytes)
    checkpoint_index = max(
        0,
        bisect_right(entry.byte_index, requested, key=lambda item: item[1]) - 1,
    )
    char_offset, byte_offset = entry.byte_index[checkpoint_index]
    while char_offset < len(entry.text) and byte_offset < requested:
        width = len(entry.text[char_offset].encode('utf-8'))
        char_offset += 1
        byte_offset += width
    return char_offset, byte_offset


def _utf8_prefix_end(text: str, start: int, max_bytes: int) -> tuple[int, int]:
    if max_bytes <= 0 or start >= len(text):
        return start, 0
    low = start
    high = min(len(text), start + max_bytes)
    while low < high:
        midpoint = (low + high + 1) // 2
        size = len(text[start:midpoint].encode('utf-8'))
        if size <= max_bytes:
            low = midpoint
        else:
            high = midpoint - 1
    return low, len(text[start:low].encode('utf-8'))


def _truncated_marker(next_command: str) -> str:
    payload = json.dumps({'next': next_command}, ensure_ascii=True, sort_keys=True, separators=(',', ':'))
    return f'\n<auto_compact_ref_truncated>{payload}</auto_compact_ref_truncated>'


def _truncate_rendered(
    text: str,
    marker: str,
    response_fits: Callable[[str], bool],
) -> str:
    if response_fits(text):
        return text
    result = marker if response_fits(marker) else ''
    low = 0
    high = min(len(text), REF_EXEC_RESPONSE_MAX_BYTES)
    while low <= high:
        boundary = (low + high) // 2
        candidate = text[:boundary] + marker
        if response_fits(candidate):
            result = candidate
            low = boundary + 1
        else:
            high = boundary - 1
    return result


def _page_source(
    entry: RefEntry,
    response_fits: Callable[[str], bool],
    *,
    start: int = 1,
    end: int | None = None,
) -> str:
    requested = max(0, start - 1)
    char_start, actual_start = _seek_source_byte(entry, requested)
    effective_end = entry.utf8_bytes if end is None else min(entry.utf8_bytes, end)
    remaining = max(0, effective_end - actual_start)
    if remaining <= REF_EXEC_RESPONSE_MAX_BYTES:
        char_end, _ = _utf8_prefix_end(entry.text, char_start, remaining)
        complete = entry.text[char_start:char_end]
        if response_fits(complete):
            return complete

    low = 0
    high = min(remaining, REF_EXEC_RESPONSE_MAX_BYTES)
    result = ''
    while low <= high:
        budget = (low + high) // 2
        char_end, emitted = _utf8_prefix_end(entry.text, char_start, budget)
        next_byte = actual_start + emitted + 1
        next_command = f'tail -c +{next_byte} {entry.ref}'
        if effective_end < entry.utf8_bytes:
            next_command += f' | head -c {effective_end - actual_start - emitted}'
        marker = _truncated_marker(next_command)
        candidate = entry.text[char_start:char_end] + marker
        if response_fits(candidate):
            result = candidate
            low = budget + 1
        else:
            high = budget - 1
    return result


def _render_lines(
    lines: Iterable[_Line],
    response_fits: Callable[[str], bool],
    *,
    preserve_source_newlines: bool,
) -> str:
    pieces: list[str] = []
    used = 0
    for line in lines:
        separator = '' if preserve_source_newlines or not pieces else '\n'
        ending = '\n' if preserve_source_newlines and line.has_newline else ''
        piece = separator + line.presented + ending
        size = len(piece.encode('utf-8'))
        if (
            not pieces
            and line.match_start is not None
            and line.match_end is not None
            and (size > REF_EXEC_RESPONSE_MAX_BYTES or not response_fits(piece))
        ):
            return _grep_excerpt(line, response_fits)
        if used + size <= REF_EXEC_RESPONSE_MAX_BYTES:
            pieces.append(piece)
            used += size
            continue
        marker = _truncated_marker('wc|grep|head|tail|sed')
        prefix = ''.join(pieces) + piece
        return _truncate_rendered(prefix, marker, response_fits)
    result = ''.join(pieces)
    return _truncate_rendered(result, _truncated_marker('wc|grep|head|tail|sed'), response_fits)


def _grep_excerpt(line: _Line, response_fits: Callable[[str], bool]) -> str:
    match_start = line.match_start or 0
    match_end = line.match_end or match_start
    match = line.text[match_start:match_end]
    match_bytes = len(match.encode('utf-8'))
    prefix_bytes = len(line.display_prefix.encode('utf-8'))
    before_match_bytes = len(line.text[:match_start].encode('utf-8'))
    after_match_bytes = len(line.text[match_end:].encode('utf-8'))

    # At most four UTF-8 bytes per character on each side. Reserving 512 bytes
    # for provenance makes this bounded without repeatedly re-encoding the line.
    margin = (REF_EXEC_RESPONSE_MAX_BYTES - prefix_bytes - match_bytes - 512) // 8

    def render(context_chars: int) -> str:
        left = max(0, match_start - context_chars)
        right = min(len(line.text), match_end + context_chars)
        excerpt = line.text[left:right]
        if line.source_byte_start is None:
            marker = _truncated_marker('wc|grep|head|tail|sed')
        else:
            left_context_bytes = len(line.text[left:match_start].encode('utf-8'))
            right_context_bytes = len(line.text[match_end:right].encode('utf-8'))
            match_byte_start = line.source_byte_start + before_match_bytes
            metadata = {
                'match_byte_start': match_byte_start,
                'match_byte_end': match_byte_start + match_bytes,
                'omitted_prefix_bytes': before_match_bytes - left_context_bytes,
                'omitted_suffix_bytes': after_match_bytes - right_context_bytes,
            }
            payload = json.dumps(metadata, ensure_ascii=True, sort_keys=True, separators=(',', ':'))
            marker = f'\n<auto_compact_ref_excerpt>{payload}</auto_compact_ref_excerpt>'
        return line.display_prefix + excerpt + marker

    result = render(0)
    if not response_fits(result):
        return 'Error: complete grep match exceeds the response budget; page source bytes with tail -c +N REF'
    low = 1
    high = max(0, margin)
    while low <= high:
        context_chars = (low + high) // 2
        candidate = render(context_chars)
        if response_fits(candidate):
            result = candidate
            low = context_chars + 1
        else:
            high = context_chars - 1
    return result


def _head_bytes(lines: Iterable[_Line], count: int) -> Iterator[_Line]:
    remaining = count
    for line in lines:
        presented = line.presented
        line_bytes = len(presented.encode('utf-8'))
        if line_bytes > remaining:
            end, _ = _utf8_prefix_end(presented, 0, remaining)
            if end:
                yield _Line(presented[:end], False)
            return
        yield _Line(presented, line.has_newline and line_bytes < remaining)
        remaining -= line_bytes
        if line.has_newline:
            if remaining == 0:
                return
            remaining -= 1
        if remaining <= 0:
            return


def _tail_bytes(lines: Iterable[_Line], count: int) -> Iterator[_Line]:
    if count > REF_EXEC_TAIL_MAX_BYTES:
        raise RefExecError('Error: tail byte count exceeds the 8 MiB limit')
    if count <= 0:
        return
    retained = bytearray()
    for line in lines:
        retained.extend(line.presented.encode('utf-8'))
        if line.has_newline:
            retained.append(0x0A)
        if len(retained) > count:
            del retained[: len(retained) - count]
    while retained and retained[0] & 0b1100_0000 == 0b1000_0000:
        del retained[0]
    yield from _iter_text_lines(retained.decode('utf-8'))


def _tail_from_byte(lines: Iterable[_Line], start: int) -> Iterator[_Line]:
    lines = iter(lines)
    remaining = max(0, start - 1)
    for line in lines:
        piece = line.presented + ('\n' if line.has_newline else '')
        encoded = piece.encode('utf-8')
        if remaining >= len(encoded):
            remaining -= len(encoded)
            continue
        suffix = encoded[remaining:]
        while suffix and suffix[0] & 0b1100_0000 == 0b1000_0000:
            suffix = suffix[1:]
        yield from _iter_text_lines(suffix.decode('utf-8'))
        break
    for line in lines:
        yield _Line(line.presented, line.has_newline)


def _apply_stage(lines: Iterable[_Line], stage: _Stage, budget: MatchBudget) -> Iterable[_Line]:
    if stage.command == 'grep':
        return _grep(lines, stage, budget)
    if stage.command == 'head':
        return _head_bytes(lines, stage.byte_count) if stage.byte_count is not None else _head(lines, stage.count or 0)
    if stage.command == 'tail':
        if stage.byte_start is not None:
            return _tail_from_byte(lines, stage.byte_start)
        return _tail_bytes(lines, stage.byte_count) if stage.byte_count is not None else _tail(lines, stage.count or 0)
    if stage.command == 'sed':
        return _sed(lines, stage.start_line or 1, stage.end_line)
    if stage.command == 'wc':
        return _wc(lines, next(iter(stage.flags)))
    return lines


def _bounded_head_page(
    stages: tuple[_Stage, ...],
    entry: RefEntry,
    response_fits: Callable[[str], bool],
) -> str | None:
    if len(stages) != 2:
        return None
    source, consumer = stages
    if source.command != 'tail' or source.byte_start is None:
        return None
    if consumer.command != 'head' or consumer.byte_count is None:
        return None
    return _page_source(
        entry,
        response_fits,
        start=source.byte_start,
        end=source.byte_start - 1 + consumer.byte_count,
    )


def _direct_wc(entry: RefEntry, flag: str) -> str:
    if flag == 'c':
        return str(entry.utf8_bytes)
    if flag == 'l':
        return str(entry.line_count)
    return str(_word_count(entry.text))


def _direct_response(
    stages: tuple[_Stage, ...],
    entry: RefEntry,
    response_fits: Callable[[str], bool],
) -> str | None:
    bounded_page = _bounded_head_page(stages, entry, response_fits)
    if bounded_page is not None:
        return bounded_page
    if len(stages) != 1:
        return None
    stage = stages[0]
    if stage.command == 'cat':
        return _page_source(entry, response_fits)
    if stage.command == 'head' and stage.byte_count is not None:
        return _page_source(entry, response_fits, end=stage.byte_count)
    if stage.command == 'tail' and stage.byte_start is not None:
        return _page_source(entry, response_fits, start=stage.byte_start)
    if stage.command == 'tail' and stage.byte_count is not None:
        if stage.byte_count > REF_EXEC_TAIL_MAX_BYTES:
            raise RefExecError('Error: tail byte count exceeds the 8 MiB limit')
        return _page_source(
            entry,
            response_fits,
            start=max(1, entry.utf8_bytes - stage.byte_count + 1),
        )
    if stage.command == 'wc':
        return _direct_wc(entry, next(iter(stage.flags)))
    return None


def _initial_lines(stage: _Stage, catalog: dict[str, RefEntry]) -> Iterable[_Line]:
    if stage.command == 'ls':
        prefix = f'{next(iter(stage.flags))}:' if stage.flags else ''
        return (_Line(ref, False) for ref in catalog if ref.startswith(prefix))
    entry = catalog.get(stage.ref or '')
    if entry is None:
        raise RefExecError('Error: externalized ref is not available in this request')
    if stage.command == 'stat':
        digest = entry.ref.split(':', 1)[1]
        value = (
            f'ref={entry.ref} kind={entry.ref.split(":", 1)[0]} utf8_bytes={entry.utf8_bytes} '
            f'lines={entry.line_count} chars={len(entry.text)} sha256={digest}'
        )
        return (_Line(value, False),)
    return _iter_source_lines(entry)


def _execute_reader(
    stages: tuple[_Stage, ...],
    catalog: dict[str, RefEntry],
    threshold_tokens: int,
    count_tokens: Callable[[str], int | None] | None,
) -> str:
    def response_fits(value: str) -> bool:
        return not _is_externalization_eligible(
            value,
            threshold_tokens=threshold_tokens,
            count_tokens=count_tokens,
        )

    first = stages[0]
    entry = catalog.get(first.ref or '')
    if first.command != 'ls' and entry is None:
        raise RefExecError('Error: externalized ref is not available in this request')
    if entry is not None:
        direct = _direct_response(stages, entry, response_fits)
        if direct is not None:
            return _truncate_rendered(
                direct,
                _truncated_marker('wc|grep|head|tail|sed'),
                response_fits,
            )

    lines = _initial_lines(first, catalog)
    budget = MatchBudget()
    budget.remaining = MATCH_BUDGET_SECONDS
    start = 0 if first.command in {'grep', 'head', 'tail', 'sed', 'wc'} else 1
    for stage in stages[start:]:
        lines = _apply_stage(lines, stage, budget)

    final = stages[-1]
    preserve = (
        (len(stages) == 1 and first.command == 'head') or final.byte_count is not None or final.byte_start is not None
    )
    return _render_lines(
        lines,
        response_fits,
        preserve_source_newlines=preserve,
    )


async def _load_history_refs(
    stages: tuple[_Stage, ...],
    catalog: dict[str, RefEntry],
) -> None:
    first = stages[0]
    requested = first.ref if isinstance(first.ref, str) and first.ref.startswith('history:') else None
    wants_listing = first.command == 'ls' and first.flags == frozenset({'history'})
    if not wants_listing and (requested is None or requested in catalog):
        return
    loader = next(
        (entry.load_history for entry in catalog.values() if entry.ref.startswith('history:') and entry.load_history),
        None,
    )
    if loader is None:
        return
    loaded = loader(requested)
    if inspect.isawaitable(loaded):
        loaded = await loaded
    for entry in loaded or ():
        if isinstance(entry, RefEntry) and entry.ref.startswith('history:') and _valid_ref(entry.ref):
            catalog.setdefault(entry.ref, replace(entry, load_history=None))
    for ref, entry in tuple(catalog.items()):
        if entry.ref.startswith('history:') and entry.load_history is loader:
            catalog[ref] = replace(entry, load_history=None)


def _new_reader(
    catalog: dict[str, RefEntry],
    *,
    threshold_tokens: int,
    count_tokens: Callable[[str], int | None] | None,
) -> Callable[[str], Any]:
    async def reader(command: str = '') -> str:
        """Read request-local externalized content with bounded text commands.

        :param command: Use ls, stat, wc, cat, head, tail, sed, grep, or a bounded pipeline.
        """

        try:
            if not isinstance(command, str):
                return REF_EXEC_USAGE_ERROR
            stages = _parse_command(command)
            await _load_history_refs(stages, catalog)
            return await asyncio.to_thread(
                _execute_reader,
                stages,
                dict(catalog),
                threshold_tokens,
                count_tokens,
            )
        except (RefExecError, MatchBudgetExceeded) as exc:
            return f'Error: {exc}' if not str(exc).startswith('Error:') else str(exc)
        except Exception:
            return 'Error: externalized ref reader is unavailable'

    reader.__name__ = REF_EXEC_TOOL_NAME
    reader.__externalized_ref_reader__ = _READER_MARKER
    reader.__externalized_ref_catalog__ = catalog
    return reader
