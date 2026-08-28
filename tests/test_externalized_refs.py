from __future__ import annotations

import copy
import importlib
import json
import os
import re
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
    ref = body['messages'][1]['content']
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
        threshold_tokens=1,
        count_tokens=lambda _text: 1,
    )
    assert registry[refs.REF_EXEC_TOOL_NAME]['callable'] is reader
    second_ref = second_body['messages'][1]['content']
    assert (await reader('ls tool')).splitlines() == [first_ref, second_ref]
    assert await reader(f'wc -c {first_ref}') == str(len(first_source))
    assert await reader(f'wc -c {second_ref}') == str(len(second_source))


@pytest.mark.asyncio
async def test_summary_projection_keeps_raw_messages_and_needs_no_reader():
    source = 'summary payload ' * 1000
    messages = _body(source)['messages']
    before = copy.deepcopy(messages)

    projected = await refs.project_tool_refs(
        messages,
        threshold_tokens=1,
        count_tokens=lambda _text: 1,
    )

    assert messages == before
    assert projected[0] is messages[0]
    assert projected[1] is not messages[1]
    assert re.fullmatch(r'tool:[0-9a-f]{64}', projected[1]['content'])


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
    assert projected[1]['content'] == entries[0].ref

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
    assert body['messages'][1]['content'] == entries[0].ref
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
    assert continuation['messages'][1]['content'].startswith('tool:')


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
    extra_ref = extra['messages'][1]['content']
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
        return 1 if re.fullmatch(r'tool:[0-9a-f]{64}', text) else 2

    class RawText(str):
        def __hash__(self):
            raise AssertionError('raw tool output must not be hashed for catalog lookup')

    monkeypatch.setattr(refs.hashlib, 'sha256', tracked_hash)
    registry = {}
    first = _body('first raw output')
    assert await refs.externalize_refs(
        first,
        registry,
        native=True,
        threshold_tokens=2,
        count_tokens=count_tokens,
    )
    first_ref = first['messages'][1]['content']
    continuation = {
        'model': 'test',
        'stream': True,
        'messages': [
            {'role': 'tool', 'tool_call_id': 'call-1', 'content': first_ref},
            {'role': 'tool', 'tool_call_id': 'call-2', 'content': RawText('second raw output')},
        ],
    }
    assert await refs.externalize_refs(
        continuation,
        registry,
        native=True,
        threshold_tokens=2,
        count_tokens=count_tokens,
    )

    assert continuation['messages'][0]['content'] == first_ref
    assert continuation['messages'][1]['content'].startswith('tool:')
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
