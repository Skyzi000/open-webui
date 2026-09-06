from __future__ import annotations

import copy
import gc
import importlib
import json
import os
import re
import tracemalloc
from pathlib import Path

import pytest

for name in ('data', 'static'):
    path = Path(f'/tmp/open-webui-externalized-refs-{name}')
    path.mkdir(parents=True, exist_ok=True)
    os.environ[name.upper() + '_DIR'] = str(path)
os.environ.setdefault('WEBUI_SECRET_KEY', 'local-test-only')

refs = importlib.import_module('open_webui.utils.externalized_refs')
compaction = importlib.import_module('open_webui.utils.context_compaction')

TOKEN_THRESHOLD = 1000


def _body(content: str, **overrides):
    body = {
        'model': 'test-model',
        'stream': True,
        'messages': [
            {'role': 'user', 'content': 'inspect the result'},
            {'role': 'tool', 'tool_call_id': 'call-1', 'content': content},
        ],
    }
    body.update(overrides)
    return body


def _preview_ref(content: str) -> str:
    match = re.search(r'tool:[0-9a-f]{64}', content)
    assert match is not None, content
    return match.group(0)


def _preview_parts(content: str) -> tuple[str, str, str]:
    """Split a projected preview into (prefix, next command, suffix)."""
    open_tag = '\n<auto_compact_ref_truncated>'
    close_tag = '</auto_compact_ref_truncated>'
    marker_start = content.find(open_tag)
    marker_end = content.find(close_tag, marker_start)
    assert 0 <= marker_start < marker_end, content
    encoded = content[marker_start + len(open_tag) : marker_end]
    command = json.loads(encoded)['next']
    suffix = content[marker_end + len(close_tag) + 1 :]
    return content[:marker_start], command, suffix


async def _project(content: str, *, registry=None, body=None):
    registry = {} if registry is None else registry
    body = _body(content) if body is None else body
    active = await refs.externalize_refs(
        body,
        registry,
        native=True,
        threshold_tokens=TOKEN_THRESHOLD,
        count_tokens=compaction.estimate_text_tokens,
    )
    assert active is True
    reader = registry[refs.REF_EXEC_TOOL_NAME]['callable']
    ref = _preview_ref(body['messages'][1]['content'])
    return body, registry, reader, ref


async def _history_reader(content: str):
    entry = refs.make_ref_entry(content, kind='history')
    assert entry is not None

    async def load_history():
        return entry

    body = {'model': 'test-model', 'stream': True, 'messages': [{'role': 'user', 'content': 'inspect'}]}
    registry = {}
    assert await refs.externalize_refs(
        body,
        registry,
        native=True,
        threshold_tokens=TOKEN_THRESHOLD,
        count_tokens=compaction.estimate_text_tokens,
        history_loader=load_history,
    )
    return registry[refs.REF_EXEC_TOOL_NAME]['callable'], entry.ref


async def _read_pages(reader, command: str) -> tuple[str, list[str]]:
    chunks: list[str] = []
    pages: list[str] = []
    for _ in range(100):
        result = await reader(command)
        pages.append(result)
        marker_start = result.find('\n<auto_compact_ref_truncated>')
        if marker_start < 0:
            chunks.append(result)
            return ''.join(chunks), pages
        chunks.append(result[:marker_start])
        encoded_marker = result[marker_start:].removeprefix('\n<auto_compact_ref_truncated>')
        encoded_marker = encoded_marker.removesuffix('</auto_compact_ref_truncated>')
        command = json.loads(encoded_marker)['next']
    pytest.fail('paging did not terminate')


@pytest.mark.asyncio
async def test_exact_wc_and_all_reader_commands():
    source = 'zero\none two\nthree\n'
    reader, ref = await _history_reader(source)

    assert await reader(f'wc -c {ref}') == str(len(source.encode('utf-8')))
    assert await reader(f'wc -l {ref}') == '3'
    assert await reader(f'wc -w {ref}') == '4'
    assert await reader(f'cat {ref}') == source
    assert await reader(f'head -1 {ref}') == 'zero\n'
    assert await reader(f'tail -1 {ref}') == 'three'
    assert await reader(f"sed -n '2,3p' {ref}") == 'one two\nthree'
    assert await reader(f'grep -n one {ref}') == '2:one two'
    assert await reader(f"grep -E '^one' {ref}") == 'one two'
    assert await reader(f"grep -E 'zero|three' {ref} | wc -l") == '2'
    assert await reader(f'grep o {ref} | head -2 | wc -l') == '2'
    assert await reader(f"grep 'unterminated {ref}") == 'Error: malformed quote or escape in command'
    assert ref in await reader('ls history')
    stat = await reader(f'stat {ref}')
    assert f'ref={ref}' in stat
    assert f'utf8_bytes={len(source.encode("utf-8"))}' in stat
    assert f'sha256={ref.split(":", 1)[1]}' in stat


@pytest.mark.asyncio
async def test_tail_rejects_line_counts_above_response_budget(monkeypatch):
    reader, ref = await _history_reader('one\ntwo\nthree\n')

    def unreadable(_entry):
        raise AssertionError('oversized tail must fail before reading the source')
        yield

    monkeypatch.setattr(refs, '_iter_source_lines', unreadable)

    result = await reader(f'tail -n {refs.REF_EXEC_RESPONSE_MAX_BYTES + 1} {ref}')

    assert result == 'Error: tail line count exceeds the 65,536 line limit'
    assert (
        await reader(f'cat {ref} | tail -n {refs.REF_EXEC_RESPONSE_MAX_BYTES + 1}')
        == 'Error: tail line count exceeds the 65,536 line limit'
    )


@pytest.mark.asyncio
async def test_oversized_tail_does_not_pre_empt_a_later_stage_error():
    reader, ref = await _history_reader('one\ntwo\nthree\n')

    command = f'tail -n {refs.REF_EXEC_RESPONSE_MAX_BYTES + 1} {ref} | grep -E "["'

    assert await reader(command) == 'Error: Invalid regex: unterminated character set at position 1'


@pytest.mark.parametrize(
    ('command', 'expected'),
    [
        ("grep '|' {ref}", 'left | right\nplain'),
        (r'grep \| {ref}', 'left | right\nplain'),
        ('\u00a0grep left {ref}', 'left | right'),
    ],
)
@pytest.mark.asyncio
async def test_reader_parser_preserves_literal_pipe_and_trim(command, expected):
    reader, ref = await _history_reader('left | right\nplain\n')

    assert await reader(command.format(ref=ref)) == expected


@pytest.mark.asyncio
async def test_catalog_unions_on_reentry_without_replacing_reader():
    first_source = 'first payload ' * 1000
    second_source = 'second payload ' * 1000
    first_body, registry, reader, first_ref = await _project(first_source)
    second_body = _body(second_source, tools=copy.deepcopy(first_body['tools']))

    assert await refs.externalize_refs(
        second_body,
        registry,
        native=True,
        threshold_tokens=TOKEN_THRESHOLD,
        count_tokens=compaction.estimate_text_tokens,
    )
    assert registry[refs.REF_EXEC_TOOL_NAME]['callable'] is reader
    second_ref = _preview_ref(second_body['messages'][1]['content'])
    assert (await reader('ls tool')).splitlines() == [first_ref, second_ref]
    assert await reader(f'wc -c {first_ref}') == str(len(first_source.encode('utf-8')))
    assert await reader(f'wc -c {second_ref}') == str(len(second_source.encode('utf-8')))


@pytest.mark.asyncio
async def test_summary_projection_keeps_raw_messages_and_needs_no_reader():
    source = 'summary payload ' * 1000
    messages = _body(source)['messages']
    before = copy.deepcopy(messages)

    projected = await refs.project_tool_refs(
        messages,
        threshold_tokens=TOKEN_THRESHOLD,
        count_tokens=compaction.estimate_text_tokens,
    )

    assert messages == before
    assert projected[0] is messages[0]
    assert projected[1] is not messages[1]
    content = projected[1]['content']
    assert content != source
    assert '<auto_compact_ref_truncated>' in content
    assert compaction.estimate_text_tokens(content) < TOKEN_THRESHOLD


@pytest.mark.asyncio
async def test_captured_tool_entry_is_admitted_without_rehashing(monkeypatch):
    source = 'captured payload ' * 1000

    def count_tokens(text):
        return 1 if text is source else 0

    projected, entries = await refs.capture_tool_ref_projections(
        _body(source)['messages'],
        threshold_tokens=1,
        count_tokens=count_tokens,
    )
    assert len(entries) == 1
    assert _preview_ref(projected[1]['content']) == entries[0].ref
    assert projected[1]['content'] != source

    def unexpected_hash():
        raise AssertionError('captured source must reuse its immutable entry')

    monkeypatch.setattr(refs.hashlib, 'sha256', unexpected_hash)
    body = _body(source)
    registry = {}
    assert await refs.externalize_refs(
        body,
        registry,
        native=True,
        threshold_tokens=1,
        count_tokens=count_tokens,
        seed_entries=entries,
    )
    reader = registry[refs.REF_EXEC_TOOL_NAME]['callable']
    assert _preview_ref(body['messages'][1]['content']) == entries[0].ref
    assert await reader(f'wc -c {entries[0].ref}') == str(len(source.encode('utf-8')))


@pytest.mark.asyncio
async def test_reader_output_is_below_threshold_and_not_reexternalized():
    _, registry, reader, ref = await _project('reader page ' * 5000)
    page = await reader(f'cat {ref}')
    assert compaction.estimate_text_tokens(page) < TOKEN_THRESHOLD
    assert '<auto_compact_ref_truncated>' in page

    continuation = {
        'model': 'test',
        'stream': True,
        'messages': [
            {'role': 'tool', 'tool_call_id': 'reader-call', 'content': page},
            {
                'role': 'tool',
                'tool_call_id': 'ordinary-call',
                'content': 'ordinary result ' * 2000,
            },
        ],
    }

    assert await refs.externalize_refs(
        continuation,
        registry,
        native=True,
        threshold_tokens=TOKEN_THRESHOLD,
        count_tokens=compaction.estimate_text_tokens,
    )
    assert continuation['messages'][0]['content'] == page
    projected = continuation['messages'][1]['content']
    assert '<auto_compact_ref_truncated>' in projected
    assert compaction.estimate_text_tokens(projected) < TOKEN_THRESHOLD


@pytest.mark.asyncio
async def test_tiktoken_failure_preserves_tool_and_reader_token_caps(monkeypatch):
    def unavailable(_name):
        raise OSError('offline')

    compaction._token_encoder.cache_clear()
    try:
        monkeypatch.setattr(compaction.tiktoken, 'get_encoding', unavailable)
        _, _, reader, ref = await _project('x' * 5000)
        page = await reader(f'cat {ref}')

        assert '<auto_compact_ref_truncated>' in page
        assert compaction.estimate_text_tokens(page) < TOKEN_THRESHOLD
    finally:
        compaction._token_encoder.cache_clear()


@pytest.mark.asyncio
async def test_copy_on_write_and_sibling_catalog_isolation():
    shared_messages = _body('shared payload ' * 1000)['messages']
    original_messages = copy.deepcopy(shared_messages)
    first_body = {'model': 'a', 'stream': True, 'messages': shared_messages}
    second_body = {'model': 'b', 'stream': True, 'messages': shared_messages}
    first_registry: dict = {}
    second_registry: dict = {}

    assert await refs.externalize_refs(
        first_body,
        first_registry,
        native=True,
        threshold_tokens=TOKEN_THRESHOLD,
        count_tokens=compaction.estimate_text_tokens,
    )
    assert shared_messages == original_messages
    assert second_body['messages'] is shared_messages
    assert await refs.externalize_refs(
        second_body,
        second_registry,
        native=True,
        threshold_tokens=TOKEN_THRESHOLD,
        count_tokens=compaction.estimate_text_tokens,
    )
    first_reader = first_registry[refs.REF_EXEC_TOOL_NAME]['callable']
    second_reader = second_registry[refs.REF_EXEC_TOOL_NAME]['callable']
    assert first_reader is not second_reader

    extra = _body('first sibling only ' * 1000, tools=copy.deepcopy(first_body['tools']))
    assert await refs.externalize_refs(
        extra,
        first_registry,
        native=True,
        threshold_tokens=TOKEN_THRESHOLD,
        count_tokens=compaction.estimate_text_tokens,
    )
    extra_ref = _preview_ref(extra['messages'][1]['content'])
    assert extra_ref in await first_reader('ls tool')
    assert extra_ref not in await second_reader('ls tool')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('native', 'body_overrides'),
    [
        (False, {}),
        (True, {'stream': False}),
        (True, {'tool_choice': 'none'}),
        (
            True,
            {'tool_choice': {'type': 'function', 'function': {'name': 'some_other_tool'}}},
        ),
        (
            True,
            {
                'tools': [
                    {
                        'type': 'function',
                        'function': {
                            'name': refs.REF_EXEC_TOOL_NAME,
                            'description': 'foreign',
                            'parameters': {'type': 'object'},
                        },
                    }
                ]
            },
        ),
    ],
)
async def test_inactive_or_unselectable_contexts_remain_raw(native, body_overrides):
    body = _body('must remain raw', **body_overrides)
    before = copy.deepcopy(body)
    registry: dict = {}

    async def unexpected_history():
        raise AssertionError('history must remain lazy')

    assert not await refs.externalize_refs(
        body,
        registry,
        native=native,
        threshold_tokens=1,
        count_tokens=lambda _text: 1,
        history_loader=unexpected_history,
    )
    assert body == before
    assert registry == {}


@pytest.mark.asyncio
async def test_ineligible_text_is_not_hashed(monkeypatch):
    body = _body('small raw result')

    def unexpected_hash():
        raise AssertionError('ineligible text must not be hashed')

    monkeypatch.setattr(refs.hashlib, 'sha256', unexpected_hash)
    assert not await refs.externalize_refs(
        body,
        {},
        native=True,
        threshold_tokens=2,
        count_tokens=lambda _text: 1,
    )


@pytest.mark.asyncio
async def test_eligible_source_is_hashed_once_and_never_on_read(monkeypatch):
    real_sha256 = refs.hashlib.sha256
    calls = 0

    def tracked_hash():
        nonlocal calls
        calls += 1
        return real_sha256()

    monkeypatch.setattr(refs.hashlib, 'sha256', tracked_hash)
    source = 'hash me once ' * 1000
    _, _, reader, ref = await _project(source)
    assert await reader(f'wc -c {ref}') == str(len(source))
    assert calls == 1


@pytest.mark.asyncio
async def test_continuation_hashes_only_new_raw_tool_output(monkeypatch):
    real_sha256 = refs.hashlib.sha256
    calls = 0

    def tracked_hash():
        nonlocal calls
        calls += 1
        return real_sha256()

    def count_tokens(text):
        return len(text) // 4

    class RawText(str):
        def __hash__(self):
            raise AssertionError('raw tool output must not be hashed for catalog lookup')

    monkeypatch.setattr(refs.hashlib, 'sha256', tracked_hash)
    registry = {}
    first = _body('first raw output ' * 600)
    assert await refs.externalize_refs(
        first,
        registry,
        native=True,
        threshold_tokens=1000,
        count_tokens=count_tokens,
    )
    first_content = first['messages'][1]['content']
    first_ref = _preview_ref(first_content)
    continuation = {
        'model': 'test',
        'stream': True,
        'messages': [
            {'role': 'tool', 'tool_call_id': 'call-1', 'content': first_content},
            {'role': 'tool', 'tool_call_id': 'call-2', 'content': RawText('second raw output ' * 600)},
        ],
    }
    assert await refs.externalize_refs(
        continuation,
        registry,
        native=True,
        threshold_tokens=1000,
        count_tokens=count_tokens,
    )

    assert continuation['messages'][0]['content'] == first_content
    assert '<auto_compact_ref_truncated>' in continuation['messages'][1]['content']
    assert _preview_ref(continuation['messages'][1]['content']) != first_ref
    assert calls == 2


@pytest.mark.asyncio
async def test_200kb_single_line_grep_finds_far_match_with_bounded_output():
    source = 'a' * 150_000 + 'NEEDLE' + 'b' * 50_000
    _, _, reader, ref = await _project(source)

    result = await reader(f'grep NEEDLE {ref}')
    visible, encoded_marker = result.rsplit('\n<auto_compact_ref_excerpt>', 1)
    marker = json.loads(encoded_marker.removesuffix('</auto_compact_ref_excerpt>'))
    assert 'NEEDLE' in visible
    assert marker['match_byte_start'] == 150_000
    assert marker['match_byte_end'] == 150_006
    assert marker['omitted_prefix_bytes'] + len(visible.split('NEEDLE')[0]) == 150_000
    assert len(result.encode('utf-8')) <= refs.REF_EXEC_RESPONSE_MAX_BYTES
    assert compaction.estimate_text_tokens(result) < TOKEN_THRESHOLD


@pytest.mark.asyncio
async def test_token_capped_single_line_grep_keeps_the_match():
    source = '🙂' * 2000 + 'NEEDLE' + 'tail'
    _, _, reader, ref = await _project(source)

    result = await reader(f'grep NEEDLE {ref}')
    visible, encoded_marker = result.rsplit('\n<auto_compact_ref_excerpt>', 1)
    marker = json.loads(encoded_marker.removesuffix('</auto_compact_ref_excerpt>'))
    assert 'NEEDLE' in visible
    assert marker['match_byte_start'] == 8000
    assert compaction.estimate_text_tokens(result) < TOKEN_THRESHOLD


@pytest.mark.asyncio
async def test_grep_only_matching_stops_scanning_when_the_response_is_full(monkeypatch):
    source = 'a' * 1_000_000
    _, _, reader, ref = await _project(source)
    original_finditer = refs.re.finditer
    matches = 0

    def bounded_finditer(*args, **kwargs):
        nonlocal matches
        for match in original_finditer(*args, **kwargs):
            matches += 1
            if matches > 100_000:
                raise AssertionError('grep materialized matches past the response budget')
            yield match

    monkeypatch.setattr(refs.re, 'finditer', bounded_finditer)
    result = await reader(f'grep -o a {ref}')

    assert 0 < matches < 100_000
    assert len(result.encode('utf-8')) <= refs.REF_EXEC_RESPONSE_MAX_BYTES
    assert '<auto_compact_ref_truncated>' in result


@pytest.mark.asyncio
async def test_multibyte_cat_continuations_reconstruct_every_source_byte():
    source = ('alpha-αβγ🙂-日本語-' * 1200) + 'done'
    _, _, reader, ref = await _project(source)
    reconstructed, pages = await _read_pages(reader, f'cat {ref}')
    for result in pages:
        assert len(result.encode('utf-8')) <= refs.REF_EXEC_RESPONSE_MAX_BYTES
        assert compaction.estimate_text_tokens(result) < TOKEN_THRESHOLD

    assert reconstructed == source


@pytest.mark.asyncio
async def test_byte_page_commands_reconstruct_exact_requested_slice():
    source = 'x' * 30_000
    _, _, reader, ref = await _project(source)
    cases = (
        (f'cat {ref}', source),
        (f'head -c 20000 {ref}', source[:20_000]),
        (f'tail -c 20000 {ref}', source[-20_000:]),
        (f'tail -c +10001 {ref}', source[10_000:]),
    )
    for command, expected in cases:
        reconstructed, pages = await _read_pages(reader, command)
        assert reconstructed == expected
        assert all(compaction.estimate_text_tokens(page) < TOKEN_THRESHOLD for page in pages)


@pytest.mark.asyncio
async def test_pipeline_regex_stages_share_one_match_budget(monkeypatch):
    reader, ref = await _history_reader('x')
    observed_timeouts: list[float] = []

    class Compiled:
        def finditer(self, text, pos=0, *, timeout):
            observed_timeouts.append(timeout)
            match = re.search('x', text[pos:])
            return iter([match] if match is not None else [])

    class Budget:
        def __init__(self):
            self.remaining = 1.0

    monotonic_values = iter([0.0, 0.75, 0.75, 1.0])
    monkeypatch.setattr(refs, 'MATCH_BUDGET_SECONDS', 1.0)
    monkeypatch.setattr(refs, 'MatchBudget', Budget)
    monkeypatch.setattr(refs.regex, 'compile', lambda *_args, **_kwargs: Compiled())
    monkeypatch.setattr(refs, '_monotonic', lambda: next(monotonic_values))

    assert await reader(f'grep -Eo x {ref} | head -1 | grep -E x') == 'x'
    assert observed_timeouts == pytest.approx([1.0, 0.25])


@pytest.mark.asyncio
async def test_grep_uses_the_core_regex_quantifier_limit():
    _, _, reader, ref = await _project('a' * 20_000)

    result = await reader(f'grep -E "a{{999999999}}" {ref}')

    assert result.startswith('Error: Regex quantifier counts over ')


@pytest.mark.asyncio
async def test_history_ancestors_load_only_on_demand_and_union_with_selected():
    ancestor = refs.make_ref_entry('older history', kind='history')
    assert ancestor is not None
    loads = []

    async def load_history(requested):
        loads.append(requested)
        return (ancestor,)

    selected = refs.make_ref_entry('selected history', kind='history', load_history=load_history)
    assert selected is not None
    body = {'model': 'test', 'stream': True, 'messages': [{'role': 'user', 'content': 'continue'}]}
    registry = {}

    async def load_selected():
        return selected

    assert await refs.externalize_refs(
        body,
        registry,
        native=True,
        threshold_tokens=1000,
        count_tokens=lambda _text: 1,
        history_loader=load_selected,
    )
    reader = registry[refs.REF_EXEC_TOOL_NAME]['callable']
    assert await reader(f'wc -c {selected.ref}') == str(len(selected.text))
    assert loads == []

    assert await reader(f'cat {ancestor.ref}') == ancestor.text
    assert loads == [ancestor.ref]
    assert (await reader('ls history')).splitlines() == [selected.ref, ancestor.ref]
    assert loads == [ancestor.ref, None]


@pytest.mark.asyncio
async def test_threshold_floor_keeps_reader_output_non_empty():
    source = 'floor budget content ' * 400
    assert compaction.estimate_text_tokens(source) >= 1000

    _, _, reader, ref = await _project(source)

    page = await reader(f'cat {ref}')
    assert page
    assert compaction.estimate_text_tokens(page) < TOKEN_THRESHOLD


@pytest.mark.asyncio
async def test_current_core_tool_wrapper_returns_exact_wc_bytes():
    from open_webui.utils.tools import get_updated_tool_function

    source = 'core dispatch 日本語\n' * 1000
    _, _, reader, ref = await _project(source)
    function = await get_updated_tool_function(
        function=reader,
        extra_params={'__messages__': [], '__files__': []},
    )

    assert await function(command=f'wc -c {ref}') == str(len(source.encode('utf-8')))
    assert f'sha256={ref.split(":", 1)[1]}' in await function(command=f'stat {ref}')


@pytest.mark.asyncio
async def test_giant_single_line_commands_never_copy_the_source():
    size = 4 * 1024 * 1024
    # A trailing newline makes the line a strict sub-span of the source, so the
    # rendered head is a real slice rather than the whole-string identity slice.
    # grep is excluded there: it still materialises `presented` for the regex.
    for source, templates in (
        (
            'a' * (size - 6) + 'NEEDLE',
            ('head -n 1 {0}', 'tail -n 1 {0}', 'sed -n 1p {0}',
             'grep NEEDLE {0}', 'grep -n NEEDLE {0}', 'grep -o NEEDLE {0}'),
        ),
        (
            'a' * (size - 7) + 'NEEDLE\n',
            ('head -n 1 {0}', 'tail -n 1 {0}', 'sed -n 1p {0}'),
        ),
    ):
        await _assert_giant_line_is_not_copied(source, size, templates)


async def _assert_giant_line_is_not_copied(source, size, templates):
    _, _, reader, ref = await _project(source)
    for command in (template.format(ref) for template in templates):
        await reader(command)
        tracemalloc.start()
        tracemalloc.reset_peak()
        result = await reader(command)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert len(result.encode('utf-8')) <= refs.REF_EXEC_RESPONSE_MAX_BYTES
        assert peak < size // 8, (command, peak)


@pytest.mark.asyncio
async def test_giant_single_line_head_is_truncated_with_a_marker():
    source = 'a' * (4 * 1024 * 1024)
    marker = refs._truncated_marker('wc|grep|head|tail|sed')
    for text in (source, source + '\n'):
        entry = refs.make_ref_entry(text, kind='tool')
        catalog = {entry.ref: entry}
        for command in (
            f'head -n 1 {entry.ref}',
            f'tail -n 1 {entry.ref}',
            f'sed -n 1p {entry.ref}',
        ):
            stages = refs._parse_command(command)
            # count_tokens=None makes any <=64 KiB response "fit", so a page that
            # skipped the budget check would come back without its marker.
            for count_tokens in (None, compaction.estimate_text_tokens):
                result = refs._execute_reader(
                    stages, catalog, TOKEN_THRESHOLD, count_tokens
                )
                assert result.endswith('</auto_compact_ref_truncated>'), command
                assert result == text[: len(result) - len(marker)] + marker
                assert len(result.encode('utf-8')) <= refs.REF_EXEC_RESPONSE_MAX_BYTES


@pytest.mark.asyncio
async def test_line_exactly_at_the_response_budget_is_not_truncated():
    source = 'q' * refs.REF_EXEC_RESPONSE_MAX_BYTES
    entry = refs.make_ref_entry(source, kind='tool')
    catalog = {entry.ref: entry}
    for command in (f'head -n 1 {entry.ref}', f'sed -n 1p {entry.ref}'):
        stages = refs._parse_command(command)
        assert refs._execute_reader(stages, catalog, TOKEN_THRESHOLD, None) == source


@pytest.mark.asyncio
async def test_grep_excerpt_offsets_cross_many_byte_index_checkpoints():
    stride = refs.REF_TEXT_INDEX_CHARS
    prefix = ('a' + 'é' + '漢' + '🙂') * stride
    suffix = ('🙂' + 'b') * stride
    source = prefix + 'NEEDLE' + suffix
    assert len(prefix) == 4 * stride
    _, _, reader, ref = await _project(source)

    result = await reader(f'grep -n NEEDLE {ref}')
    visible, encoded = result.rsplit('\n<auto_compact_ref_excerpt>', 1)
    marker = json.loads(encoded.removesuffix('</auto_compact_ref_excerpt>'))
    prefix_bytes = len(prefix.encode('utf-8'))
    assert marker['match_byte_start'] == prefix_bytes
    assert marker['match_byte_end'] == prefix_bytes + 6
    assert marker['omitted_prefix_bytes'] == prefix_bytes - len(
        visible.split('NEEDLE')[0].removeprefix('1:').encode('utf-8')
    )
    assert marker['omitted_suffix_bytes'] == len(suffix.encode('utf-8')) - len(
        visible.split('NEEDLE')[1].encode('utf-8')
    )


@pytest.mark.asyncio
async def test_grep_only_matching_tracks_source_bytes_across_multibyte_gaps():
    gap = ('🙂' + 'é' + 'x') * refs.REF_TEXT_INDEX_CHARS
    # The leading line puts the matched line at a non-zero source offset.
    head = 'first line\n'
    source = head + gap.join(('', 'NEEDLE', 'NEEDLE', 'NEEDLE'))
    entry = refs.make_ref_entry(source, kind='tool')
    stages = refs._parse_command(f'grep -o NEEDLE {entry.ref}')
    lines = list(
        refs._apply_stage(
            refs._iter_source_lines(entry), stages[0], refs.MatchBudget()
        )
    )

    gap_bytes = len(gap.encode('utf-8'))
    head_bytes = len(head.encode('utf-8'))
    assert [line.source_byte_start for line in lines] == [
        head_bytes + gap_bytes,
        head_bytes + 2 * gap_bytes + 6,
        head_bytes + 3 * gap_bytes + 12,
    ]
    assert [line.text for line in lines] == ['NEEDLE'] * 3


@pytest.mark.asyncio
async def test_grep_excerpt_never_reaches_past_its_own_line():
    # The trailing 'b' run is shorter than the excerpt context margin, so the
    # right edge is clamped by the line, not by the context budget.
    source = 'a' * 100_000 + 'NEEDLE' + 'b' * 10 + '\nSECOND LINE\n'
    _, _, reader, ref = await _project(source)

    result = await reader(f'grep -n NEEDLE {ref}')
    visible, encoded = result.rsplit('\n<auto_compact_ref_excerpt>', 1)
    marker = json.loads(encoded.removesuffix('</auto_compact_ref_excerpt>'))
    assert 'SECOND' not in result
    assert '\n' not in visible
    assert visible.endswith('NEEDLE' + 'b' * 10)
    assert marker['match_byte_start'] == 100_000
    assert marker['omitted_prefix_bytes'] > 0
    assert marker['omitted_suffix_bytes'] == 0


@pytest.mark.asyncio
async def test_history_ancestor_body_survives_being_listed_first():
    messages = [
        {'role': 'user', 'content': 'first question'},
        {'role': 'assistant', 'content': 'first answer', 'contextSummary': 'one'},
        {'role': 'user', 'content': 'second question'},
        {'role': 'assistant', 'content': 'second answer', 'contextSummary': 'two'},
        {'role': 'user', 'content': 'third question'},
        {'role': 'assistant', 'content': 'third answer', 'contextSummary': 'three'},
    ]
    selected = await compaction.resolve_request_history((messages, 5))
    assert selected is not None
    body = {'model': 'test', 'stream': True, 'messages': [{'role': 'user', 'content': 'continue'}]}
    registry = {}

    async def load_selected():
        return selected

    assert await refs.externalize_refs(
        body,
        registry,
        native=True,
        threshold_tokens=10 ** 9,
        count_tokens=lambda _text: 0,
        history_loader=load_selected,
    )
    reader = registry[refs.REF_EXEC_TOOL_NAME]['callable']
    listed = (await reader('ls history')).splitlines()
    assert listed[0] == selected.ref and len(listed) == 3

    for ref, prefix_end in zip(listed[1:], (3, 1)):
        want = compaction._canonical_history_entry(messages[:prefix_end])
        assert ref == want.ref
        assert await reader(f'cat {ref}') == want.text
        assert await reader(f'wc -c {ref}') == str(want.utf8_bytes)
        assert await reader(f'wc -l {ref}') == str(want.line_count)
        assert await reader(f'wc -w {ref}') == str(len(want.text.split()))
        assert (await reader(f'stat {ref}')).endswith(
            f'utf8_bytes={want.utf8_bytes} lines={want.line_count} '
            f'chars={len(want.text)} sha256={want.ref.split(":", 1)[1]}'
        )


@pytest.mark.asyncio
async def test_listed_history_ancestor_body_is_fetched_once_on_first_read():
    listed = refs.make_ref_entry('older history', kind='history')
    assert listed is not None
    stub = compaction.replace(listed, text='')
    loads = []

    async def load_history(requested):
        loads.append(requested)
        return (stub,) if requested is None else (listed,)

    selected = refs.make_ref_entry('selected history', kind='history', load_history=load_history)
    assert selected is not None
    body = {'model': 'test', 'stream': True, 'messages': [{'role': 'user', 'content': 'continue'}]}
    registry = {}

    async def load_selected():
        return selected

    assert await refs.externalize_refs(
        body,
        registry,
        native=True,
        threshold_tokens=1000,
        count_tokens=lambda _text: 1,
        history_loader=load_selected,
    )
    reader = registry[refs.REF_EXEC_TOOL_NAME]['callable']
    assert (await reader('ls history')).splitlines() == [selected.ref, listed.ref]
    # `ls` parked metadata only, so the first read of any kind fetches the body.
    assert await reader(f'wc -c {listed.ref}') == str(listed.utf8_bytes)
    assert loads == [None, listed.ref]
    # Every later read, `cat` included, is served from the cached body.
    assert await reader(f'cat {listed.ref}') == listed.text
    assert await reader(f'cat {listed.ref}') == listed.text
    assert loads == [None, listed.ref]


def test_short_line_spans_never_reach_for_the_byte_index(monkeypatch):
    entry = refs.make_ref_entry('line\n' * 5_000, kind='tool')
    assert entry is not None and entry.byte_index
    lookups = []
    bisect_right = refs.bisect_right

    def counted(*args, **kwargs):
        lookups.append(args[1])
        return bisect_right(*args, **kwargs)

    monkeypatch.setattr(refs, 'bisect_right', counted)
    lines = list(refs._iter_source_lines(entry))

    assert [line.text for line in lines] == ['line'] * 5_000
    assert [line.source_byte_start for line in lines] == [5 * index for index in range(5_000)]
    # Spans this short cost less to encode outright than two checkpoint lookups.
    assert lookups == []


@pytest.mark.asyncio
async def test_parked_history_stubs_reload_through_the_loader_that_listed_them():
    ancestors = tuple(
        refs.make_ref_entry(f'ancestor {index} body', kind='history') for index in range(2)
    )

    async def load_a(requested):
        if requested is None:
            # `ls` parks metadata-only stubs that carry their own loader back.
            return tuple(
                compaction.replace(item, text='', load_history=load_a) for item in ancestors
            )
        return tuple(item for item in ancestors if item.ref == requested)

    async def load_b(_requested):
        # A later selected history binds a different lineage and resolves nothing.
        return ()

    body = {'model': 'test-model', 'stream': True, 'messages': [{'role': 'user', 'content': 'go'}]}
    registry = {}

    async def install(text, loader):
        selected = refs.make_ref_entry(text, kind='history', load_history=loader)

        async def load_selected():
            return selected

        assert await refs.externalize_refs(
            body,
            registry,
            native=True,
            threshold_tokens=TOKEN_THRESHOLD,
            count_tokens=compaction.estimate_text_tokens,
            history_loader=load_selected,
        )
        return registry[refs.REF_EXEC_TOOL_NAME]['callable']

    reader = await install('round one selected history', load_a)
    assert (await reader('ls history')).splitlines()[1:] == [item.ref for item in ancestors]

    assert await install('round two selected history', load_b) is reader
    for ancestor in ancestors:
        assert await reader(f'cat {ancestor.ref}') == ancestor.text
        stat = await reader(f'stat {ancestor.ref}')
        assert f'utf8_bytes={ancestor.utf8_bytes} ' in stat
        assert f'chars={ancestor.utf8_bytes} ' in stat


def test_tail_line_windows_match_the_source_tail():
    for text in ('l1\nl2\nl3\nl4\n', 'l1\nl2\nl3\nl4'):
        entry = refs.make_ref_entry(text, kind='tool')
        assert entry is not None and entry.line_count == 4
        catalog = {entry.ref: entry}
        source_lines = text.rstrip('\n').split('\n')
        for count in (0, 1, 3, 4, 7):
            stages = refs._parse_command(f'tail -n {count} {entry.ref}')
            expected = '\n'.join(source_lines[-count:]) if count else ''
            assert refs._execute_reader(stages, catalog, TOKEN_THRESHOLD, None) == expected


def test_tail_of_the_whole_source_does_not_retain_a_line_per_source_line():
    entry = refs.make_ref_entry(('t' * 511 + '\n') * 20_000, kind='tool')
    assert entry is not None and entry.line_count == 20_000
    catalog = {entry.ref: entry}
    stages = refs._parse_command(f'tail -n 20000 {entry.ref}')
    warm = refs._execute_reader(stages, catalog, TOKEN_THRESHOLD, None)

    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    result = refs._execute_reader(stages, catalog, TOKEN_THRESHOLD, None)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert result == warm
    assert len(result.encode('utf-8')) <= refs.REF_EXEC_RESPONSE_MAX_BYTES
    # Retaining one _Line per source line costs >4 MiB here, a whole-line copy
    # each (HEAD) costs >13 MiB.
    assert peak < 2 * 1024 * 1024, peak


def test_tail_window_keeps_absolute_source_offsets_for_a_giant_line():
    giant = '漢' * 30_000
    source = 'NEEDLE decoy\n' + giant + 'NEEDLE' + giant + '\nlast line\n'
    entry = refs.make_ref_entry(source, kind='tool')
    assert entry is not None and entry.line_count == 3
    stages = refs._parse_command(f'tail -n 2 {entry.ref} | grep NEEDLE')

    result = refs._execute_reader(stages, {entry.ref: entry}, TOKEN_THRESHOLD, None)

    # The decoy sits outside the tail window; only the giant line survives.
    assert 'decoy' not in result
    _, encoded = result.rsplit('\n<auto_compact_ref_excerpt>', 1)
    marker = json.loads(encoded.removesuffix('</auto_compact_ref_excerpt>'))
    assert marker == {
        'match_byte_start': 90_013,
        'match_byte_end': 90_019,
        'omitted_prefix_bytes': 65_619,
        'omitted_suffix_bytes': 65_619,
    }


def test_tail_of_nothing_never_pulls_the_source(monkeypatch):
    entry = refs.make_ref_entry('l1\nl2\nl3\n', kind='tool')
    assert entry is not None
    catalog = {entry.ref: entry}

    def untouchable(_entry):
        raise AssertionError('tail -n 0 pulled the source')
        yield  # a generator like the real one, so only a pull can fail

    monkeypatch.setattr(refs, '_iter_source_lines', untouchable)
    stages = refs._parse_command(f'tail -n 0 {entry.ref}')
    assert refs._execute_reader(stages, catalog, TOKEN_THRESHOLD, None) == ''


def _rebuilt_preview(source: str, ref: str, prefix: str, suffix: str) -> str:
    prefix_bytes = len(prefix.encode('utf-8'))
    omitted_bytes = (
        len(source.encode('utf-8')) - prefix_bytes - len(suffix.encode('utf-8'))
    )
    command = f'tail -c +{prefix_bytes + 1} {ref} | head -c {omitted_bytes}'
    return prefix + refs._truncated_marker(command) + '\n' + suffix


@pytest.mark.asyncio
async def test_preview_keeps_head_and_tail_around_the_marker():
    source = 'head of the result\n' + 'middle line\n' * 900 + 'tail of the result\n'
    body, registry, reader, ref = await _project(source)

    content = body['messages'][1]['content']
    prefix, command, suffix = _preview_parts(content)
    assert command.startswith(f'tail -c +{len(prefix.encode("utf-8")) + 1} {ref} | head -c ')
    assert prefix == source[: len(prefix)]
    assert suffix == source[len(source) - len(suffix) :]
    assert prefix.startswith('head of the result\n')
    assert suffix.endswith('tail of the result\n')
    omitted = int(command.rsplit(' ', 1)[1])
    assert omitted == len(source.encode('utf-8')) - len(prefix.encode('utf-8')) - len(suffix.encode('utf-8'))
    assert abs(len(prefix.encode('utf-8')) - len(suffix.encode('utf-8'))) <= 4
    assert compaction.estimate_text_tokens(content) < TOKEN_THRESHOLD
    assert len(content.encode('utf-8')) <= refs.REF_EXEC_RESPONSE_MAX_BYTES


@pytest.mark.asyncio
async def test_preview_boundary_below_at_and_above_threshold():
    def count_tokens(text):
        return len(text) // 4

    for size, externalized in ((3996, False), (4000, True), (4004, True)):
        body = _body('x' * size)
        registry = {}
        active = await refs.externalize_refs(
            body,
            registry,
            native=True,
            threshold_tokens=1000,
            count_tokens=count_tokens,
        )
        assert active is externalized, size
        if externalized:
            content = body['messages'][1]['content']
            assert count_tokens(content) < 1000
            assert '<auto_compact_ref_truncated>' in content
        else:
            assert body['messages'][1]['content'] == 'x' * size


@pytest.mark.asyncio
async def test_preview_is_per_side_maximal_for_a_monotone_counter():
    source = 'word ' * 8000
    counter = lambda text: len(text.encode('utf-8'))  # noqa: E731

    body = _body(source)
    registry = {}
    assert await refs.externalize_refs(
        body,
        registry,
        native=True,
        threshold_tokens=10_000,
        count_tokens=counter,
    )
    content = body['messages'][1]['content']
    ref = _preview_ref(content)
    # counter == bytes, so tokens < 10_000 means exactly byte cap 9_999.
    assert len(content.encode('utf-8')) == 9_999
    prefix, _command, suffix = _preview_parts(content)
    for extended in (
        _rebuilt_preview(source, ref, prefix + source[len(prefix)], suffix),
        _rebuilt_preview(source, ref, prefix, source[len(source) - len(suffix) - 1] + suffix),
    ):
        assert len(extended.encode('utf-8')) > 9_999


@pytest.mark.asyncio
async def test_preview_byte_cap_binds_before_tokens():
    source = 'a' * 200_000
    body = _body(source)
    registry = {}
    assert await refs.externalize_refs(
        body,
        registry,
        native=True,
        threshold_tokens=1_000_000,
        count_tokens=lambda _text: 1,
    )
    content = body['messages'][1]['content']
    assert len(content.encode('utf-8')) == refs.REF_EXEC_RESPONSE_MAX_BYTES
    ref = _preview_ref(content)
    prefix, _command, suffix = _preview_parts(content)
    for extended in (
        _rebuilt_preview(source, ref, prefix + source[len(prefix)], suffix),
        _rebuilt_preview(source, ref, prefix, source[len(source) - len(suffix) - 1] + suffix),
    ):
        assert len(extended.encode('utf-8')) > refs.REF_EXEC_RESPONSE_MAX_BYTES


@pytest.mark.asyncio
async def test_preview_uses_95_percent_of_the_real_token_budget():
    source = 'Realistic english sentences with several words each.\n' * 700
    body, registry, reader, ref = await _project(source)

    content = body['messages'][1]['content']
    tokens = compaction.estimate_text_tokens(content)
    assert len(content.encode('utf-8')) < refs.REF_EXEC_RESPONSE_MAX_BYTES
    assert TOKEN_THRESHOLD * 0.95 <= tokens < TOKEN_THRESHOLD


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'source',
    [
        '日本語の長いテキスト行を含む結果。\n' * 1200,
        '🙂🚀漢字αβγ' * 3000,
        'single giant line ' + 'z' * 100_000,
    ],
)
async def test_multibyte_and_single_line_previews_stay_capped(source):
    body, registry, reader, ref = await _project(source)
    content = body['messages'][1]['content']

    assert compaction.estimate_text_tokens(content) < TOKEN_THRESHOLD
    assert len(content.encode('utf-8')) <= refs.REF_EXEC_RESPONSE_MAX_BYTES
    assert await reader(f'wc -c {ref}') == str(len(source.encode('utf-8')))


@pytest.mark.asyncio
async def test_preview_next_command_restores_the_exact_middle_across_pages():
    source = ('x' * 70 + '\n') * 1200
    body, registry, reader, ref = await _project(source)
    content = body['messages'][1]['content']
    prefix, command, suffix = _preview_parts(content)

    middle, pages = await _read_pages(reader, command)
    assert len(pages) >= 2

    rebuilt = (prefix + middle + suffix).encode('utf-8')
    assert rebuilt == source.encode('utf-8')
    assert '\n<auto_compact_ref_truncated>' not in middle
    assert await reader(f'cat {ref}') != source


@pytest.mark.asyncio
async def test_preview_ref_is_the_hash_of_the_full_source():
    import hashlib

    source = 'hash check payload ' * 800
    body, registry, reader, ref = await _project(source)

    assert ref == 'tool:' + hashlib.sha256(source.encode('utf-8')).hexdigest()
    assert await reader(f'wc -c {ref}') == str(len(source.encode('utf-8')))
    assert (await reader('ls tool')).splitlines() == [ref]


@pytest.mark.asyncio
async def test_preview_render_is_reused_and_byte_stable():
    source = 'reuse payload ' * 1000
    render_calls = []
    original = refs._render_ref_preview

    def tracked(entry, **kwargs):
        render_calls.append(entry.ref)
        return original(entry, **kwargs)

    refs._render_ref_preview = tracked
    try:
        cache = {}
        messages = _body(source)['messages']
        first, entries = await refs.capture_tool_ref_projections(
            messages,
            threshold_tokens=TOKEN_THRESHOLD,
            count_tokens=compaction.estimate_text_tokens,
            render_cache=cache,
        )
        second, second_entries = await refs.capture_tool_ref_projections(
            messages,
            threshold_tokens=TOKEN_THRESHOLD,
            count_tokens=compaction.estimate_text_tokens,
            seed_entries=entries,
            render_cache=cache,
        )
    finally:
        refs._render_ref_preview = original

    assert render_calls == [entries[0].ref]
    assert second[1]['content'] == first[1]['content']
    assert [entry.ref for entry in second_entries] == [entry.ref for entry in entries]

    threshold_changed, _ = await refs.capture_tool_ref_projections(
        messages,
        threshold_tokens=500,
        count_tokens=compaction.estimate_text_tokens,
        seed_entries=entries,
        render_cache=cache,
    )
    changed = threshold_changed[1]['content']
    assert changed != first[1]['content']
    assert compaction.estimate_text_tokens(changed) < 500


@pytest.mark.asyncio
async def test_uncountable_preview_is_not_adopted_or_cached():
    source = 'y' * 200_000
    cache = {}

    def failing(_text):
        raise OSError('counter down')

    body = _body(source)
    registry = {}
    assert not await refs.externalize_refs(
        body,
        registry,
        native=True,
        threshold_tokens=TOKEN_THRESHOLD,
        count_tokens=failing,
        render_cache=cache,
    )
    assert body['messages'][1]['content'] == source
    assert registry == {}

    assert await refs.externalize_refs(
        body,
        registry,
        native=True,
        threshold_tokens=TOKEN_THRESHOLD,
        count_tokens=compaction.estimate_text_tokens,
        render_cache=cache,
    )
    assert body['messages'][1]['content'] != source
    assert len(cache) == 1
