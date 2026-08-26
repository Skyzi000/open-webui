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
        threshold_tokens=1,
        count_tokens=lambda _text: 1,
    )
    assert active is True
    reader = registry[refs.REF_EXEC_TOOL_NAME]['callable']
    ref = body['messages'][1]['content']
    return body, registry, reader, ref


@pytest.mark.asyncio
async def test_exact_wc_and_all_reader_commands():
    source = 'zero\none two\nthree\n'
    body, _, reader, ref = await _project(source)

    assert await reader(f'wc -c {ref}') == str(len(source.encode('utf-8')))
    assert await reader(f'wc -l {ref}') == '3'
    assert await reader(f'wc -w {ref}') == '4'
    assert await reader(f'cat {ref}') == source
    assert await reader(f'head -1 {ref}') == 'zero\n'
    assert await reader(f'tail -1 {ref}') == 'three'
    assert await reader(f"sed -n '2,3p' {ref}") == 'one two\nthree'
    assert await reader(f'grep -n one {ref}') == '2:one two'
    assert await reader(f"grep -E '^one' {ref}") == 'one two'
    assert await reader(f'grep o {ref} | head -2 | wc -l') == '2'
    assert ref in await reader('ls tool')
    stat = await reader(f'stat {ref}')
    assert f'ref={ref}' in stat
    assert f'utf8_bytes={len(source.encode("utf-8"))}' in stat
    assert body['tools'][0]['function']['name'] == refs.REF_EXEC_TOOL_NAME


@pytest.mark.asyncio
async def test_catalog_unions_on_reentry_without_replacing_reader():
    first_body, registry, reader, first_ref = await _project('first payload')
    second_body = _body('second payload', tools=copy.deepcopy(first_body['tools']))

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
    assert await reader(f'wc -c {first_ref}') == str(len('first payload'))
    assert await reader(f'wc -c {second_ref}') == str(len('second payload'))


@pytest.mark.asyncio
async def test_reader_output_is_never_reexternalized():
    _, registry, reader, ref = await _project('reader page')
    page = await reader(f'cat {ref}')
    inferred = {
        'model': 'test',
        'stream': True,
        'messages': [
            {
                'role': 'assistant',
                'tool_calls': [
                    {
                        'id': 'reader-call',
                        'type': 'function',
                        'function': {'name': refs.REF_EXEC_TOOL_NAME, 'arguments': '{}'},
                    }
                ],
            },
            {'role': 'tool', 'tool_call_id': 'reader-call', 'content': page},
        ],
    }

    assert not await refs.externalize_refs(
        inferred,
        registry,
        native=True,
        threshold_tokens=1,
        count_tokens=lambda _text: 1,
    )
    assert inferred['messages'][1]['content'] == page

    current_only = {
        'model': 'test',
        'stream': True,
        'messages': [
            {'role': 'tool', 'tool_call_id': 'reader-call', 'content': page},
            {'role': 'tool', 'tool_call_id': 'ordinary-call', 'content': 'ordinary result'},
        ],
    }
    assert await refs.externalize_refs(
        current_only,
        registry,
        native=True,
        threshold_tokens=1,
        count_tokens=lambda _text: 1,
        excluded_tool_call_ids={'reader-call'},
    )
    assert current_only['messages'][0]['content'] == page
    assert current_only['messages'][1]['content'].startswith('tool:')


@pytest.mark.asyncio
async def test_copy_on_write_and_sibling_catalog_isolation():
    shared_messages = _body('shared payload')['messages']
    original_messages = copy.deepcopy(shared_messages)
    first_body = {'model': 'a', 'stream': True, 'messages': shared_messages}
    second_body = {'model': 'b', 'stream': True, 'messages': shared_messages}
    first_registry: dict = {}
    second_registry: dict = {}

    assert await refs.externalize_refs(
        first_body,
        first_registry,
        native=True,
        threshold_tokens=1,
        count_tokens=lambda _text: 1,
    )
    assert shared_messages == original_messages
    assert second_body['messages'] is shared_messages
    assert await refs.externalize_refs(
        second_body,
        second_registry,
        native=True,
        threshold_tokens=1,
        count_tokens=lambda _text: 1,
    )
    first_reader = first_registry[refs.REF_EXEC_TOOL_NAME]['callable']
    second_reader = second_registry[refs.REF_EXEC_TOOL_NAME]['callable']
    assert first_reader is not second_reader

    extra = _body('first sibling only', tools=copy.deepcopy(first_body['tools']))
    assert await refs.externalize_refs(
        extra,
        first_registry,
        native=True,
        threshold_tokens=1,
        count_tokens=lambda _text: 1,
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

    assert not await refs.externalize_refs(
        body,
        registry,
        native=native,
        threshold_tokens=1,
        count_tokens=lambda _text: 1,
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
    _, _, reader, ref = await _project('hash me once')
    assert await reader(f'cat {ref}') == 'hash me once'
    assert await reader(f'wc -c {ref}') == str(len('hash me once'))
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
            {'role': 'tool', 'tool_call_id': 'call-2', 'content': 'second raw output'},
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


@pytest.mark.asyncio
async def test_multibyte_cat_continuations_reconstruct_every_source_byte():
    source = ('alpha-αβγ🙂-日本語-' * 12_000) + 'done'
    _, _, reader, ref = await _project(source)
    command = f'cat {ref}'
    chunks: list[str] = []

    for _ in range(20):
        result = await reader(command)
        marker_start = result.find('\n<auto_compact_ref_truncated>')
        if marker_start < 0:
            chunks.append(result)
            break
        chunks.append(result[:marker_start])
        encoded_marker = result[marker_start:].removeprefix('\n<auto_compact_ref_truncated>')
        encoded_marker = encoded_marker.removesuffix('</auto_compact_ref_truncated>')
        command = json.loads(encoded_marker)['next']
        assert command.startswith('tail -c +')
        assert len(result.encode('utf-8')) <= refs.REF_EXEC_RESPONSE_MAX_BYTES
    else:
        pytest.fail('paging did not terminate')

    assert ''.join(chunks) == source


@pytest.mark.asyncio
async def test_pipeline_regex_stages_share_one_match_budget(monkeypatch):
    _, _, reader, ref = await _project('x')
    observed_timeouts: list[float] = []

    class Compiled:
        def finditer(self, text, *, timeout):
            observed_timeouts.append(timeout)
            match = re.search('x', text)
            return iter([match] if match is not None else [])

    monotonic_values = iter([0.0, 0.75, 0.75, 1.0])
    monkeypatch.setattr(refs, 'MATCH_BUDGET_SECONDS', 1.0)
    monkeypatch.setattr(refs.regex, 'compile', lambda *_args, **_kwargs: Compiled())
    monkeypatch.setattr(refs, '_monotonic', lambda: next(monotonic_values))

    assert await reader(f'grep -E x {ref} | grep -E x') == 'x'
    assert observed_timeouts == pytest.approx([1.0, 0.25])


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

    assert await refs.externalize_refs(
        body,
        registry,
        native=True,
        threshold_tokens=1000,
        count_tokens=lambda _text: 1,
        history_entry=selected,
    )
    reader = registry[refs.REF_EXEC_TOOL_NAME]['callable']
    assert await reader(f'wc -c {selected.ref}') == str(len(selected.text))
    assert loads == []

    assert await reader(f'cat {ancestor.ref}') == ancestor.text
    assert loads == [ancestor.ref]
    assert (await reader('ls history')).splitlines() == [selected.ref, ancestor.ref]
    assert loads == [ancestor.ref]


@pytest.mark.asyncio
async def test_current_core_tool_wrapper_returns_exact_wc_bytes():
    from open_webui.utils.tools import get_updated_tool_function

    source = 'core dispatch 日本語\n'
    _, _, reader, ref = await _project(source)
    function = await get_updated_tool_function(
        function=reader,
        extra_params={'__messages__': [], '__files__': []},
    )

    assert await function(command=f'wc -c {ref}') == str(len(source.encode('utf-8')))
