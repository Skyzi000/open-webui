import asyncio
import copy
import gc
import importlib
import json
import logging
import os
import re
import threading
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.responses import StreamingResponse

DATA_DIR = Path('/tmp/open-webui-context-compaction-tests')
STATIC_DIR = Path('/tmp/open-webui-context-compaction-static')
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
os.environ['DATA_DIR'] = str(DATA_DIR)
os.environ['STATIC_DIR'] = str(STATIC_DIR)

compaction = importlib.import_module('open_webui.utils.context_compaction')


def test_strict_usage_uses_only_complete_input_measurements():
    accepted = [
        ({'prompt_tokens': 100}, 100),
        ({'prompt_eval_count': 100}, 100),
        ({'prompt_n': 25, 'cache_n': 75}, 100),
        (
            {
                'input_tokens': 60,
                'cache_creation_input_tokens': 15,
                'cache_read_input_tokens': 25,
            },
            100,
        ),
    ]
    for usage, expected in accepted:
        assert compaction._strict_usage_input_tokens(usage) == expected

    rejected = [
        {'prompt_n': 100},
        {'prompt_tokens': 0, 'input_tokens': 100},
        {'total_tokens': 100},
        {'input_tokens': True},
        {'input_tokens': 100, 'cache_read_input_tokens': '10'},
    ]
    for usage in rejected:
        assert compaction._strict_usage_input_tokens(usage) is None


def test_usage_total_tokens_shares_prefetch_trigger_accounting():
    assert compaction._usage_total_tokens(None) is None
    assert compaction._usage_total_tokens('x') is None
    assert compaction._usage_total_tokens({'total_tokens': True}) is None
    assert compaction._usage_total_tokens({'total_tokens': 'big'}) is None
    assert (
        compaction._usage_total_tokens(
            {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}
        )
        == 150
    )
    assert (
        compaction._usage_total_tokens(
            {
                'input_tokens': 60,
                'cache_creation_input_tokens': 10,
                'cache_read_input_tokens': 5,
                'output_tokens': 10,
                'total_tokens': 70,
            }
        )
        == 85
    )
    assert (
        compaction._usage_total_tokens(
            {
                'input_tokens': 60,
                'cache_creation_input_tokens': 'bad',
                'cache_read_input_tokens': 5,
                'output_tokens': 10,
                'total_tokens': 70,
            }
        )
        == 70
    )


def test_compact_functions_record_pending_candidate_only(monkeypatch):
    def estimate(body, **_kwargs):
        return 60

    monkeypatch.setattr(compaction, 'estimate_provider_tokens', estimate)
    provider_state = {'config': _compaction_config(soft_trigger_ratio=0.5)}
    transient_state = {'config': _compaction_config(soft_trigger_ratio=0.5)}

    result = asyncio.run(
        compaction.compact_provider_payload(
            None,
            None,
            {'messages': [{'role': 'user', 'content': 'hello'}]},
            {'chat_id': 'local:unit'},
            'model',
            {},
            provider_state,
        )
    )
    transient_result = asyncio.run(
        compaction.compact_transient_provider_payload(
            None,
            None,
            {'messages': [{'role': 'user', 'content': 'hello'}]},
            {'chat_id': 'local:unit'},
            'model',
            {},
            transient_state,
        )
    )

    assert result['messages'] == [{'role': 'user', 'content': 'hello'}]
    assert transient_result['messages'] == [{'role': 'user', 'content': 'hello'}]
    pending = {
        'tokens': 60,
        'threshold': 100,
        'soft_threshold': 50,
        'source': 'estimated',
    }
    assert provider_state['pending_context_usage'] == pending
    assert transient_state['pending_context_usage'] == pending
    assert 'context_usage' not in provider_state
    assert 'context_usage' not in transient_state


def test_compact_provider_updates_pending_to_selected_candidate_after(monkeypatch):
    history = [
        {'id': 'u1', 'role': 'user', 'content': 'old'},
        {'id': 'a1', 'role': 'assistant', 'content': 'answer'},
        {'id': 'u2', 'role': 'user', 'content': 'new', compaction._BOUNDARY_KEY: True},
    ]

    async def generate_checkpoint(*_args, **_kwargs):
        return 'summary', {'historical_user_messages': []}, (history, 2), 'u2'

    async def save_checkpoint(*_args, **_kwargs):
        return True

    monkeypatch.setattr(compaction, '_generate_checkpoint', generate_checkpoint)
    monkeypatch.setattr(compaction, 'estimate_provider_tokens', _estimate_compacted_when_summarized)
    monkeypatch.setattr(compaction.ChatMessages, 'update_context_summary', save_checkpoint)
    confirmed = {
        'tokens': 90,
        'threshold': 100,
        'soft_threshold': 50,
        'source': 'usage',
    }
    state = {
        'active_offset': 0,
        'checkpoint_messages': history,
        'checkpoint_history': (history, 2),
        'config': _compaction_config(soft_trigger_ratio=0.5),
        'context_usage': dict(confirmed),
    }

    compacted = asyncio.run(
        compaction.compact_provider_payload(
            None,
            None,
            {'messages': history},
            {'chat_id': 'local:unit'},
            'model',
            {},
            state,
        )
    )

    assert compacted['messages'][0]['content'].startswith('<auto_compaction_context>')
    assert state['pending_context_usage'] == {
        'tokens': 20,
        'threshold': 100,
        'soft_threshold': 50,
        'source': 'estimated',
    }
    assert state['context_usage'] == confirmed


def test_transient_compact_updates_pending_to_selected_candidate_after(monkeypatch):
    async def generate_summary(*_args, **_kwargs):
        return 'BOUNDARY'

    monkeypatch.setattr(compaction, '_generate_summary', generate_summary)
    monkeypatch.setattr(compaction, 'estimate_provider_tokens', _estimate_compacted_when_summarized)
    confirmed = {
        'tokens': 90,
        'threshold': 100,
        'soft_threshold': 50,
        'source': 'usage',
    }
    state = {
        'config': _compaction_config(soft_trigger_ratio=0.5),
        'context_usage': dict(confirmed),
    }
    messages = [
        {'role': 'user', 'content': 'old question'},
        {'role': 'assistant', 'content': 'old answer'},
        {'role': 'user', 'content': 'new question'},
    ]

    compacted = asyncio.run(
        compaction.compact_transient_provider_payload(
            None,
            None,
            {'messages': messages},
            {'chat_id': 'local:unit'},
            'model',
            {},
            state,
        )
    )

    assert _summary_content_of(compacted['messages']) is state['summary_message_content']
    assert state['pending_context_usage'] == {
        'tokens': 20,
        'threshold': 100,
        'soft_threshold': 50,
        'source': 'estimated',
    }
    assert state['context_usage'] == confirmed


def test_compact_functions_emit_no_numeric_notifications(monkeypatch):
    socket_main = importlib.import_module('open_webui.socket.main')
    events = []

    async def emitter(event):
        events.append(event['data'])

    async def get_event_emitter(_metadata):
        return emitter

    def estimate(body, **_kwargs):
        return 40

    monkeypatch.setattr(socket_main, 'get_event_emitter', get_event_emitter)
    monkeypatch.setattr(compaction, 'estimate_provider_tokens', estimate)
    state = {'config': _compaction_config()}
    messages = [{'role': 'user', 'content': 'hello'}]

    asyncio.run(
        compaction.compact_transient_provider_payload(
            None,
            None,
            {'messages': messages},
            {'chat_id': 'chat', 'message_id': 'assistant'},
            'model',
            {},
            state,
        )
    )

    assert events == []
    assert state['pending_context_usage']['tokens'] == 40


def test_transient_blocking_compaction_events_carry_no_snapshot(monkeypatch):
    socket_main = importlib.import_module('open_webui.socket.main')
    events = []

    async def emitter(event):
        events.append(event['data'])

    async def get_event_emitter(_metadata):
        return emitter

    async def generate_summary(*_args, **_kwargs):
        return 'BOUNDARY'

    monkeypatch.setattr(socket_main, 'get_event_emitter', get_event_emitter)
    monkeypatch.setattr(compaction, '_generate_summary', generate_summary)
    monkeypatch.setattr(compaction, 'estimate_provider_tokens', _estimate_compacted_when_summarized)
    confirmed = {
        'tokens': 90,
        'threshold': 100,
        'soft_threshold': None,
        'source': 'usage',
    }
    state = {'config': _compaction_config(), 'context_usage': dict(confirmed)}
    messages = [
        {'role': 'user', 'content': 'old question'},
        {'role': 'assistant', 'content': 'old answer'},
        {'role': 'user', 'content': 'new question'},
    ]

    compacted = asyncio.run(
        compaction.compact_transient_provider_payload(
            None,
            None,
            {'messages': messages},
            {'chat_id': 'chat', 'message_id': 'assistant'},
            'model',
            {},
            state,
        )
    )

    assert _summary_content_of(compacted['messages']) is state['summary_message_content']
    assert [event['description'] for event in events] == ['Compacting context', 'Context compacted']
    assert all('context_usage' not in event for event in events)
    assert state['pending_context_usage'] == {
        'tokens': 20,
        'threshold': 100,
        'soft_threshold': None,
        'source': 'estimated',
    }
    assert state['context_usage'] == confirmed


def test_provider_replay_never_carries_context_usage():
    middleware = importlib.import_module('open_webui.utils.middleware')
    assert 'context_usage' not in middleware.MESSAGE_REPLAY_KEYS
    assert 'pending_context_usage' not in middleware.MESSAGE_REPLAY_KEYS


def test_response_usage_replaces_tokens_and_keeps_thresholds():
    state = {
        'context_usage': {
            'tokens': 60,
            'threshold': 100,
            'soft_threshold': 50,
            'source': 'estimated',
        }
    }
    updated = compaction.apply_response_usage_to_context_usage(
        state, {'prompt_tokens': 70, 'completion_tokens': 10, 'total_tokens': 80}
    )
    assert updated == {
        'tokens': 80,
        'threshold': 100,
        'soft_threshold': 50,
        'source': 'usage',
    }
    assert state['context_usage'] == updated
    assert state['context_usage'] is updated

    assert compaction.apply_response_usage_to_context_usage(state, None) is None
    assert compaction.apply_response_usage_to_context_usage(state, {}) is None
    assert compaction.apply_response_usage_to_context_usage(state, {'total_tokens': 'n/a'}) is None
    assert state['context_usage']['tokens'] == 80

    empty_state: dict = {}
    created = compaction.apply_response_usage_to_context_usage(
        empty_state,
        {
            'input_tokens': 60,
            'cache_creation_input_tokens': 10,
            'cache_read_input_tokens': 5,
            'output_tokens': 10,
            'total_tokens': 70,
        },
    )
    assert created == {
        'tokens': 85,
        'threshold': None,
        'soft_threshold': None,
        'source': 'usage',
    }
    assert empty_state['context_usage'] == created


def test_provider_estimate_uses_exact_input_anchor_plus_new_suffix():
    marker = compaction.CONTEXT_COMPACTION_USAGE_ANCHOR_KEY
    body = {
        'messages': [
            {'role': 'user', 'content': 'already measured ' * 10_000},
            {'role': 'assistant', 'content': 'previous', marker: 1_000},
            {'role': 'user', 'content': 'new input'},
        ],
        'tools': [{'type': 'function', 'function': {'name': 'lookup'}}],
    }
    suffix = compaction.estimate_body_tokens(
        {
            'messages': body['messages'][1:],
        }
    )

    assert compaction.estimate_provider_tokens(body) == 1_000 + suffix - 3


def test_body_estimate_includes_extras_and_bounds_large_payload_encoding(monkeypatch):
    base = {'messages': [{'role': 'user', 'content': 'hello'}]}
    base_tokens = compaction.estimate_body_tokens(base)
    assert base_tokens is not None

    extras = {
        'tools': [{'type': 'function', 'function': {'name': 'lookup', 'description': 'Look something up'}}],
        'tool_choice': 'required',
        'functions': [{'name': 'lookup', 'description': 'Look something up'}],
        'function_call': {'name': 'lookup'},
        'response_format': {'type': 'json_object'},
        'parallel_tool_calls': False,
    }
    for key, value in extras.items():
        assert compaction.estimate_body_tokens({**base, key: value}) > base_tokens

    def media_tokens(size: int) -> int | None:
        return compaction.estimate_body_tokens(
            {
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': 'inspect'},
                            {
                                'type': 'image_url',
                                'image_url': {'url': 'data:image/png;base64,' + 'A' * size},
                            },
                        ],
                        'files': [{'type': 'file', 'data': b'B' * size}],
                    }
                ]
            }
        )

    assert media_tokens(100) == media_tokens(1_000_000)

    encoded_lengths = []

    class RecordingEncoder:
        def encode(self, text, **_kwargs):
            encoded_lengths.append(len(text))
            return range(max(1, len(text) // 4))

    compaction._token_encoder.cache_clear()
    try:
        monkeypatch.setattr(compaction.tiktoken, 'get_encoding', lambda _name=None: RecordingEncoder())
        large_text_tokens = compaction.estimate_body_tokens(
            {'messages': [{'role': 'user', 'content': 'x' * 1_000_000}]}
        )
        assert large_text_tokens > 100_000
        assert max(encoded_lengths) <= 16 * 1024
    finally:
        compaction._token_encoder.cache_clear()


def test_tiktoken_failure_logs_once_and_uses_core_approximation(monkeypatch, caplog):
    attempts = 0

    def unavailable(_name):
        nonlocal attempts
        attempts += 1
        raise OSError('offline')

    compaction._token_encoder.cache_clear()
    try:
        monkeypatch.setattr(compaction.tiktoken, 'get_encoding', unavailable)
        body = {'messages': [{'role': 'user', 'content': 'hello'}]}

        with caplog.at_level(logging.ERROR, logger=compaction.__name__):
            body_tokens = compaction.estimate_body_tokens(body)
            assert compaction.estimate_body_tokens(body) == body_tokens
            assert compaction.estimate_text_tokens('hello') == compaction._estimate_tokens('hello')

            result = asyncio.run(
                compaction.compact_provider_payload(
                    None,
                    None,
                    body,
                    {},
                    'model',
                    {},
                    {
                        'config': {
                            'enable': True,
                            'token_threshold': 100_000,
                            'token_cap': 100_000,
                            'soft_trigger_ratio': 0,
                        }
                    },
                )
            )

        assert result == body
        assert body_tokens > 0
        assert attempts == 1
        assert caplog.messages.count('tiktoken is unavailable; falling back to approximate token estimation') == 1
    finally:
        compaction._token_encoder.cache_clear()


def test_tiktoken_encode_failure_uses_core_approximation(monkeypatch):
    class BrokenEncoder:
        def encode(self, _text, **_kwargs):
            raise RuntimeError('broken encoder')

    compaction._token_encoder.cache_clear()
    try:
        monkeypatch.setattr(compaction.tiktoken, 'get_encoding', lambda _name: BrokenEncoder())
        body = {'messages': [{'role': 'user', 'content': 'hello'}]}
        assert compaction.estimate_body_tokens(body) > 0
        assert compaction.estimate_text_tokens('hello') == compaction._estimate_tokens('hello')
    finally:
        compaction._token_encoder.cache_clear()


def test_safe_boundary_preserves_system_messages_and_complete_tool_rounds():
    class NoSlices(list):
        def __getitem__(self, key):
            if isinstance(key, slice):
                raise AssertionError('boundary search must not rescan message slices')
            return super().__getitem__(key)

    systems = [
        {'role': 'system', 'content': 'policy'},
        {'role': 'system', 'content': 'summary'},
    ]
    working = [
        {'role': 'user', 'content': 'old one'},
        {'role': 'assistant', 'tool_calls': [{'id': 'a'}]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': 'a result'},
        {'role': 'assistant', 'content': 'a answer'},
        {'role': 'user', 'content': 'old two'},
        {'role': 'assistant', 'tool_calls': [{'id': 'b'}]},
        {'role': 'tool', 'tool_call_id': 'b', 'content': 'b result'},
        {'role': 'assistant', 'content': 'b answer'},
        {'role': 'user', 'content': 'current'},
        {'role': 'assistant', 'content': 'current answer'},
    ]

    preserved, without_system = compaction._split_leading_system_messages([*systems, *working])
    boundary = compaction.find_safe_compaction_boundary(without_system)
    assert preserved == systems
    assert boundary == 4
    assert without_system[boundary]['content'] == 'old two'

    crossing_round = [
        {'role': 'user', 'content': 'first'},
        {'role': 'assistant', 'tool_calls': [{'id': 'crossing'}]},
        {'role': 'user', 'content': 'second'},
        {'role': 'tool', 'tool_call_id': 'crossing', 'content': 'late result'},
        {'role': 'assistant', 'content': 'answer'},
        {'role': 'user', 'content': 'current'},
        {'role': 'assistant', 'content': 'current answer'},
    ]
    assert compaction.find_safe_compaction_boundary(NoSlices(crossing_round), 50) == 0
    assert compaction.find_safe_compaction_boundary(working[:2] + working[4:], 50) == 0


def test_current_branch_uses_its_nearest_checkpoint(monkeypatch):
    history_hashes = 0
    canonical_history_entry = compaction._canonical_history_entry

    def count_history_hashes(*args, **kwargs):
        nonlocal history_hashes
        history_hashes += 1
        return canonical_history_entry(*args, **kwargs)

    root = {'id': 'u0', 'parentId': None, 'role': 'user', 'content': 'root'}
    earlier = {
        'id': 'u1',
        'parentId': 'u0',
        'role': 'user',
        'content': 'earlier branch point',
        'contextSummary': 'earlier summary',
    }
    messages = [
        root,
        earlier,
        {'id': 'a1', 'parentId': 'u1', 'role': 'assistant', 'content': 'main answer'},
        {
            'id': 'u2',
            'parentId': 'a1',
            'role': 'user',
            'content': 'latest branch point',
            'contextSummary': 'latest summary',
        },
        {'id': 'a2', 'parentId': 'u2', 'role': 'assistant', 'content': 'current'},
        {'id': 'b1', 'parentId': 'u1', 'role': 'assistant', 'content': 'fork answer'},
    ]
    messages_map = {message['id']: message for message in messages}
    main_branch = compaction.get_message_list(messages_map, 'a2')
    fork = compaction.get_message_list(messages_map, 'b1')

    async def load_config():
        return {'enable': True, 'retention_percentage': 40, 'transient_patterns': ()}

    async def prepare(messages):
        prepared, state = await compaction.prepare_compaction_messages(messages, {'chat_id': 'chat'})
        return prepared, state

    monkeypatch.setattr(compaction, '_load_config', load_config)
    monkeypatch.setattr(compaction, '_canonical_history_entry', count_history_hashes)
    main_prepared, main_state = asyncio.run(prepare(main_branch))
    fork_prepared, fork_state = asyncio.run(prepare(fork))
    assert history_hashes == 0

    main_history = asyncio.run(compaction.resolve_request_history(main_state['selected_history']))
    fork_history = asyncio.run(compaction.resolve_request_history(fork_state['selected_history']))

    assert history_hashes == 2
    assert main_history is not None and fork_history is not None
    assert main_state['selected_checkpoint_message_id'] == 'u2'
    assert fork_state['selected_checkpoint_message_id'] == 'u1'
    assert 'latest summary' in main_prepared[0]['content']
    assert 'earlier summary' in fork_prepared[0]['content']
    assert 'main answer' in main_history.text
    assert fork_history.text == canonical_history_entry([root]).text


def test_top_level_checkpoint_drops_stale_carrier_usage():
    carrier = {
        'role': 'assistant',
        'content': 'answer',
        'usage': {'prompt_tokens': 51_300},
        'info': {'usage': {'prompt_tokens': 51_300}, 'provider': 'test'},
    }

    active = compaction._messages_from_checkpoint([carrier], 0, None)

    assert 'usage' not in active[0]
    assert active[0]['info'] == {'provider': 'test'}
    assert carrier['usage'] == {'prompt_tokens': 51_300}
    assert carrier['info']['usage'] == {'prompt_tokens': 51_300}


def test_checkpoint_source_is_isolated_from_image_payload_mutation(monkeypatch):
    messages = [
        {
            'id': 'u1',
            'parentId': None,
            'role': 'user',
            'content': 'inspect',
            'files': [{'type': 'image', 'url': 'https://example.test/image.png'}],
        },
        {'id': 'a1', 'parentId': 'u1', 'role': 'assistant', 'content': 'answer'},
        {'id': 'u2', 'parentId': 'a1', 'role': 'user', 'content': 'continue'},
    ]
    expected = compaction._canonical_history_entry(messages[:2]).text

    async def load_config():
        return {
            'enable': True,
            'retention_percentage': 40,
            'transient_patterns': (),
        }

    monkeypatch.setattr(compaction, '_load_config', load_config)
    monkeypatch.setattr(compaction, 'find_safe_compaction_boundary', lambda *_args: 2)
    prepared, state = asyncio.run(compaction.prepare_compaction_messages(messages, {'chat_id': 'chat'}))
    prepared[0]['content'] = [
        {'type': 'text', 'text': 'inspect'},
        {'type': 'image_url', 'image_url': {'url': 'https://example.test/image.png'}},
    ]

    history = asyncio.run(compaction.resolve_request_history(state['checkpoint_history']))
    assert history.text == expected


def _summary_content_of(messages):
    return next(
        message['content']
        for message in messages
        if isinstance(message.get('content'), str)
        and message['content'].startswith('<auto_compaction_context>')
    )


def test_background_replays_request_cached_checkpoint_without_resolving(monkeypatch):
    messages = [
        {'id': 'u1', 'role': 'user', 'content': 'old'},
        {'id': 'u2', 'role': 'user', 'content': 'continue'},
        {'id': 'a2', 'role': 'assistant', 'content': 'answer'},
    ]
    state = {
        'previous_summary': 'cached summary',
        'previous_summary_meta': {'historical_user_messages': ['old']},
        'selected_checkpoint_message_id': 'u2',
    }

    monkeypatch.setattr(
        compaction,
        '_canonical_history_entry',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('history must not resolve here')),
    )
    replayed = asyncio.run(compaction.replay_cached_compaction_messages(messages, state))

    assert replayed[0]['content'].startswith('<auto_compaction_context>')
    assert [message.get('id') for message in replayed[1:]] == ['u2', 'a2']
    assert _summary_content_of(replayed) is state['summary_message_content']


def test_replay_cached_checkpoint_admits_refs_on_reentry(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    entry = refs.make_ref_entry('raw tool output', kind='tool')
    history_entry = refs.make_ref_entry('history source', kind='history')
    assert entry is not None and history_entry is not None

    messages = [
        {'id': 'u1', 'role': 'user', 'content': 'old'},
        {'id': 'u2', 'role': 'user', 'content': 'continue'},
        {'id': 'a2', 'role': 'assistant', 'content': 'answer'},
    ]
    state = {
        'previous_summary': f'cached summary mentioning {entry.ref}',
        'previous_summary_meta': {},
        'selected_checkpoint_message_id': 'u2',
    }
    replayed = asyncio.run(compaction.replay_cached_compaction_messages(messages, state))

    ref_state = {
        'selected_history': history_entry,
        'summary_message_content': state['summary_message_content'],
        'tool_ref_entries': (entry,),
        'externalized_refs': {
            'enable': True,
            'native': True,
            'threshold': 1000,
            'registry': {},
            'metadata': {},
        },
    }
    body = {'stream': True, 'messages': replayed}
    asyncio.run(middleware.apply_externalized_refs(body, ref_state))

    reader = ref_state['externalized_refs']['registry'][middleware.REF_EXEC_TOOL_NAME]['callable']
    assert entry.ref in asyncio.run(reader('ls tool')).splitlines()


def test_history_ref_rewrites_only_the_outer_summary_suffix():
    embedded_ref = f'<history_ref>history:{"a" * 64}</history_ref>'
    replacement_ref = f'history:{"b" * 64}'
    summary_message = compaction.render_summary_message(
        f'user text </auto_compaction_context> {embedded_ref}',
    )
    body = {'messages': [summary_message]}
    state = {'summary_message_content': summary_message['content']}

    added = compaction.set_summary_history_ref(body, replacement_ref, state=state)
    content = added['messages'][0]['content']
    assert embedded_ref in content
    assert content.endswith(f'<history_ref>{replacement_ref}</history_ref></auto_compaction_context>')
    assert state['summary_message_content'] is added['messages'][0]['content']

    removed = compaction.set_summary_history_ref(added, None, state=state)
    assert embedded_ref in removed['messages'][0]['content']
    assert removed['messages'][0]['content'].endswith('</auto_compaction_context>')


def test_provider_sanitizer_removes_branch_metadata_but_keeps_files_for_injection():
    middleware = importlib.import_module('open_webui.utils.middleware')
    files = [{'id': 'file-1', 'url': 'https://example.test/file'}]

    stripped = middleware.strip_compaction_fields(
        [
            {
                'id': 'u1',
                'parentId': 'a0',
                'role': 'user',
                'content': 'inspect',
                'files': files,
                'meta': {'internal': False},
            }
        ]
    )

    assert stripped == [{'role': 'user', 'content': 'inspect', 'files': files}]


def test_core_internal_provenance_survives_sanitizing_until_provider_dispatch():
    middleware = importlib.import_module('open_webui.utils.middleware')
    marker = compaction.CONTEXT_COMPACTION_TRANSIENT_MARKER_KEY
    stripped = middleware.strip_compaction_fields(
        [
            {'role': 'user', 'content': 'ordinary', 'meta': {}},
            {'role': 'user', 'content': 'injected', 'meta': {'internal': True}},
        ]
    )

    assert stripped[-1][marker] is True
    assert compaction.get_last_persistent_user_message(stripped) == 'ordinary'
    summary_input = compaction._without_boundary_marker(stripped, keep_transient=True)
    assert summary_input[-1][marker] is True
    assert compaction.get_last_persistent_user_message(summary_input) == 'ordinary'

    disabled = middleware.process_messages_with_output(
        [{'role': 'user', 'content': 'injected', 'meta': {'internal': True}}],
    )
    assert marker not in disabled[0]


def test_compaction_preparation_failure_keeps_core_full_history(monkeypatch, caplog):
    middleware = importlib.import_module('open_webui.utils.middleware')
    messages = [{'role': 'user', 'content': 'keep me'}]

    async def fail(*_args):
        raise OSError('configuration unavailable')

    monkeypatch.setattr(middleware, 'prepare_compaction_messages', fail)
    with caplog.at_level(logging.ERROR):
        prepared, state = asyncio.run(middleware._prepare_compaction_or_default(messages, {}))

    assert prepared is messages
    assert state == {'config': {'enable': False}}
    assert 'continuing with the latest stored checkpoint' in caplog.text


def test_model_params_are_not_consumed_across_sibling_requests():
    middleware = importlib.import_module('open_webui.utils.middleware')
    shared = {
        'system': 'target policy',
        'temperature': 0.5,
        'custom_params': {'top_p': '0.8'},
    }
    original = copy.deepcopy(shared)

    first = middleware.apply_params_to_form_data({'params': shared}, {'owned_by': 'openai'})
    second = middleware.apply_params_to_form_data({'params': shared}, {'owned_by': 'openai'})

    assert shared == original
    assert first == second == {'temperature': 0.5, 'top_p': 0.8}


def test_file_context_restores_by_message_id_after_filter_reordering():
    middleware = importlib.import_module('open_webui.utils.middleware')
    first = [{'id': 'file-1', 'url': 'https://example.test/one'}]
    second = [{'id': 'file-2', 'url': 'https://example.test/two'}]
    captured = {'u1': first, 'u2': second}
    filtered = [
        {'id': 'u2', 'role': 'user', 'content': 'two'},
        {'role': 'user', 'content': 'inserted'},
        {'role': 'assistant', 'content': 'answer'},
        {'id': 'u1', 'role': 'user', 'content': 'one'},
    ]

    restored = middleware.restore_message_files(filtered, captured)
    assert restored[0]['files'] is second
    assert 'files' not in restored[1]
    assert restored[3]['files'] is first
    assert all('id' not in message for message in restored)


def test_file_context_identity_survives_message_normalization():
    middleware = importlib.import_module('open_webui.utils.middleware')
    files = [{'id': 'file-1', 'url': 'https://example.test/file'}]
    messages = [
        {'id': 'u1', 'role': 'user', 'content': 'inspect', 'files': files},
        {
            'id': 'a1',
            'role': 'assistant',
            'content': '',
            'output': [
                {
                    'type': 'function_call',
                    'call_id': 'call-1',
                    'name': 'tool',
                    'arguments': '{}',
                    'status': 'completed',
                },
                {
                    'type': 'function_call_output',
                    'call_id': 'call-1',
                    'output': [{'type': 'input_image', 'image_url': 'data:image/png;base64,AA=='}],
                },
            ],
        },
    ]

    processed = middleware.process_messages_with_output(messages)
    restored = middleware.restore_message_files(processed, {'u1': files})

    user_messages = [message for message in restored if message.get('role') == 'user']
    assert user_messages[0]['files'] is files
    assert 'files' not in user_messages[1]


def test_approved_tool_pair_appends_without_replacing_prepared_messages(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    prepared = [
        {'role': 'system', 'content': 'prepared policy'},
        {'role': 'user', 'content': 'prepared user'},
    ]
    form_data = {'messages': copy.deepcopy(prepared)}
    stored = {
        'output': [
            {
                'type': 'function_call',
                'call_id': 'call-1',
                'name': 'lookup',
                'arguments': '{}',
                'status': 'queued',
                'approved': True,
            }
        ]
    }

    async def get_message(_chat_id, _message_id):
        return stored

    async def execute(*_args, **_kwargs):
        return {'tool_call_id': 'call-1', 'content': 'result'}

    async def get_events(_metadata):
        return None, None

    async def upsert(*_args, **_kwargs):
        return stored

    monkeypatch.setattr(middleware.Chats, 'get_message_by_id_and_message_id', get_message)
    monkeypatch.setattr(middleware, 'execute_tool_call_for_output', execute)
    monkeypatch.setattr(middleware, 'get_event_emitter_and_caller', get_events)
    monkeypatch.setattr(middleware.Chats, 'upsert_message_to_chat_by_id_and_message_id', upsert)

    paused, appended = asyncio.run(
        middleware.drain_approved_tool_calls(
            None,
            form_data,
            None,
            {'id': 'model'},
            {'chat_id': 'chat', 'message_id': 'assistant', 'assistant_message_id': 'assistant', 'params': {}},
        )
    )

    assert paused is False
    assert form_data['messages'][:2] == prepared
    assert form_data['messages'][2:] == appended
    assert [message['role'] for message in appended] == ['assistant', 'tool']
    assert appended[1]['content'] == 'result'


def test_internal_ref_reader_executes_before_other_tools_pause(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    reader_call = {
        'id': 'reader-call',
        'function': {'name': refs.REF_EXEC_TOOL_NAME, 'arguments': '{"command":"ls"}'},
    }
    write_call = {'id': 'write-call', 'function': {'name': 'write', 'arguments': '{}'}}
    output = [
        {
            'type': 'function_call',
            'call_id': call['id'],
            'name': call['function']['name'],
            'arguments': call['function']['arguments'],
            'status': 'in_progress',
        }
        for call in (reader_call, write_call)
    ]

    executed = []

    async def execute(*args):
        executed.append(args[-1]['id'])
        return {'tool_call_id': 'reader-call', 'content': 'catalog'}

    monkeypatch.setattr(middleware, 'execute_tool_call_for_output', execute)
    async def run():
        registry = {}
        await refs.externalize_refs(
            {
                'stream': True,
                'messages': [
                    {
                        'role': 'tool',
                        'tool_call_id': 'call',
                        'content': 'large result ' * 100,
                    }
                ],
            },
            registry,
            native=True,
            threshold_tokens=500,
            count_tokens=lambda value: len(value),
        )
        remaining = await middleware._execute_ref_calls_before_approval(
            None,
            {'messages': []},
            None,
            {'tools': registry},
            None,
            None,
            [reader_call, write_call],
            output,
        )
        custom = {
            refs.REF_EXEC_TOOL_NAME: {
                'spec': refs.REF_EXEC_FUNCTION_SPEC,
                'callable': lambda: None,
            }
        }
        custom_remaining = await middleware._execute_ref_calls_before_approval(
            None,
            {'messages': []},
            None,
            {'tools': custom},
            None,
            None,
            [reader_call],
            [],
        )
        return remaining, custom_remaining

    remaining, custom_remaining = asyncio.run(run())

    assert remaining == [write_call]
    assert custom_remaining == [reader_call]
    assert executed == ['reader-call']
    assert output[0]['status'] == 'completed'
    assert output[1]['status'] == 'in_progress'
    assert output[-1]['call_id'] == 'reader-call'
    assert output[-1]['output'][0]['text'] == 'catalog'


def test_approved_reader_rebuilds_catalog_and_keeps_its_result_literal(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    source = 'approved reader source ' * 600
    stored = {}

    async def get_message(_chat_id, _message_id):
        return stored

    async def get_events(_metadata):
        return None, None

    async def upsert(*_args, **_kwargs):
        return stored

    async def process_result(_request, _name, result, *_args):
        return result, [], []

    async def noop(*_args, **_kwargs):
        return None

    async def run():
        registry = {}
        metadata = {
            'chat_id': 'chat',
            'message_id': 'assistant',
            'assistant_message_id': 'assistant',
            'params': {},
        }
        state = {
            'externalized_refs': {
                'enable': True,
                'threshold': 1000,
                'native': True,
                'registry': registry,
                'metadata': metadata,
            }
        }
        body = {
            'stream': True,
            'messages': [
                {'role': 'user', 'content': 'prepared user'},
                {'role': 'tool', 'tool_call_id': 'old-call', 'content': source},
            ],
        }
        body = await middleware.apply_externalized_refs(body, state)
        projected = body['messages'][1]['content']
        ref = re.search(r'tool:[0-9a-f]{64}', projected).group(0)
        stored['output'] = [
            {
                'type': 'function_call',
                'call_id': 'reader-call',
                'name': refs.REF_EXEC_TOOL_NAME,
                'arguments': middleware.JSONCodec.dumps({'command': f'wc -c {ref}'}),
                'status': 'queued',
                'approved': True,
            }
        ]

        paused, appended = await middleware.drain_approved_tool_calls(
            None,
            body,
            None,
            {'id': 'model'},
            metadata,
        )
        body = await middleware.apply_externalized_refs(body, state)
        return paused, appended, body

    monkeypatch.setattr(middleware.Chats, 'get_message_by_id_and_message_id', get_message)
    monkeypatch.setattr(middleware, 'get_event_emitter_and_caller', get_events)
    monkeypatch.setattr(middleware.Chats, 'upsert_message_to_chat_by_id_and_message_id', upsert)
    monkeypatch.setattr(middleware, 'process_tool_result', process_result)
    monkeypatch.setattr(middleware, 'terminal_event_handler', noop)

    paused, appended, body = asyncio.run(run())
    expected = str(len(source.encode('utf-8')))
    assert paused is False
    assert appended[-1]['content'] == expected
    assert body['messages'][-1]['content'] == expected


def test_direct_compaction_can_use_configured_server_summary_model():
    middleware = importlib.import_module('open_webui.utils.middleware')
    request = SimpleNamespace(
        state=SimpleNamespace(direct=True),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={'summary': {'id': 'summary'}})),
    )
    direct = {'direct': {'id': 'direct'}}

    assert middleware.compaction_models_for_request(request, direct) == {
        'summary': {'id': 'summary'},
        'direct': {'id': 'direct'},
    }


def test_reader_installation_requires_the_core_native_dispatch_context():
    middleware = importlib.import_module('open_webui.utils.middleware')
    native = {'chat_id': 'chat', 'message_id': 'message', 'params': {}}

    assert middleware._can_install_externalized_ref_reader(native, None) is True
    assert middleware._can_install_externalized_ref_reader({**native, 'chat_id': ''}, None) is False
    assert middleware._can_install_externalized_ref_reader({**native, 'message_id': ''}, None) is False
    assert middleware._can_install_externalized_ref_reader(native, []) is False
    assert (
        middleware._can_install_externalized_ref_reader(
            {**native, 'params': {'function_calling': 'legacy'}},
            None,
        )
        is False
    )


def test_middleware_compaction_passes_through_without_applying_refs(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    state = {}

    async def externalize(candidate, actual_state):
        raise AssertionError('compaction must not apply externalized refs itself')

    async def compact(*args, **_kwargs):
        assert args[-1] is state
        state['checkpoint_history'] = object()
        return args[2]

    monkeypatch.setattr(middleware, 'apply_externalized_refs', externalize)
    monkeypatch.setattr(middleware, 'compact_transient_provider_payload', compact)
    body = {'messages': []}
    result = asyncio.run(
        middleware._compact_final_provider_payload(
            None,
            None,
            body,
            {},
            'model',
            {},
            state,
        )
    )

    assert result is body
    assert state['checkpoint_history'] is not None


def test_arena_uses_selected_target_model_params_before_system_bypass(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    target = {'id': 'target', 'owned_by': 'openai'}
    arena = {
        'owned_by': 'arena',
        'info': {'meta': {'model_ids': ['target']}},
    }
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(MODELS={'target': target})),
        state=SimpleNamespace(direct=False),
    )
    captured = None

    class Params:
        def model_dump(self):
            return {
                'system': 'target policy',
                'temperature': 0.2,
                'custom_params': {'model_only': 2, 'shared': 'model'},
            }

    async def get_model(_model_id):
        return SimpleNamespace(params=Params())

    class StopAfterParams(Exception):
        pass

    def capture(form_data, _model):
        nonlocal captured
        captured = copy.deepcopy(form_data)
        raise StopAfterParams

    monkeypatch.setattr(middleware.Models, 'get_model_by_id', staticmethod(get_model))
    monkeypatch.setattr(middleware, 'apply_params_to_form_data', capture)

    with pytest.raises(StopAfterParams):
        asyncio.run(
            middleware.process_chat_payload(
                request,
                {'model': 'arena', 'params': {'system': 'arena policy'}, 'messages': []},
                None,
                {'chat_id': ''},
                arena,
                default_model_params={
                    'top_p': 0.9,
                    'custom_params': {'default_only': 1, 'shared': 'default'},
                },
                request_params={
                    'temperature': 0.7,
                    'custom_params': {'request_only': 3, 'shared': 'request'},
                },
            )
        )

    assert captured['model'] == 'target'
    assert captured['params'] == {
        'top_p': 0.9,
        'system': 'target policy',
        'temperature': 0.7,
        'custom_params': {
            'default_only': 1,
            'model_only': 2,
            'request_only': 3,
            'shared': 'request',
        },
    }


def test_sync_missing_base_fallback_binds_custom_params_to_fallback_routing_id(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(MODELS={})),
        state=SimpleNamespace(direct=False),
    )
    captured = {}

    class FallbackParams:
        def model_dump(self):
            return {'system': 'fallback policy', 'temperature': 0.9}

    async def get_model(model_id):
        assert model_id == 'fallback-model'
        return SimpleNamespace(params=FallbackParams())

    class StopAfterParams(Exception):
        pass

    def capture(form_data, _model):
        captured.update(copy.deepcopy(form_data))
        raise StopAfterParams

    monkeypatch.setattr(middleware.Models, 'get_model_by_id', staticmethod(get_model))
    monkeypatch.setattr(middleware, 'apply_params_to_form_data', capture)

    with pytest.raises(StopAfterParams):
        asyncio.run(
            middleware.process_chat_payload(
                request,
                {'model': 'fallback-model', 'messages': []},
                None,
                {'chat_id': ''},
                {'id': 'fallback-model', 'owned_by': 'openai'},
                default_model_params={},
                request_params={'temperature': 0.7},
                resolved_model_id='fallback-model',
                resolved_model_params={
                    'system': 'custom policy',
                    'temperature': 0.1,
                    'custom_params': {'top_p': '0.8'},
                },
            )
        )

    assert captured['params'] == {
        'system': 'custom policy',
        'temperature': 0.7,
        'custom_params': {'top_p': '0.8'},
    }


def test_fallback_bound_custom_params_follow_arena_selection(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(MODELS={'target-model': {'id': 'target-model', 'owned_by': 'openai'}})
        ),
        state=SimpleNamespace(direct=False),
    )
    arena = {'owned_by': 'arena', 'info': {'meta': {'model_ids': ['target-model']}}}
    captured = {}

    async def get_model(_model_id):
        raise AssertionError('bound params must win without DB re-resolution')

    class StopAfterParams(Exception):
        pass

    def capture(form_data, _model):
        captured.update(copy.deepcopy(form_data))
        raise StopAfterParams

    monkeypatch.setattr(middleware.Models, 'get_model_by_id', staticmethod(get_model))
    monkeypatch.setattr(middleware, 'apply_params_to_form_data', capture)

    with pytest.raises(StopAfterParams):
        asyncio.run(
            middleware.process_chat_payload(
                request,
                {'model': 'arena-fallback', 'messages': []},
                None,
                {'chat_id': ''},
                arena,
                default_model_params={},
                request_params={'temperature': 0.7},
                resolved_model_id='arena-fallback',
                resolved_model_params={'system': 'custom policy', 'temperature': 0.1},
            )
        )

    assert captured['model'] == 'target-model'
    assert captured['params'] == {'system': 'custom policy', 'temperature': 0.7}


class _EntryParams:
    def __init__(self, values):
        self.values = dict(values)

    def model_dump(self):
        return dict(self.values)


class _EntryModelInfo:
    def __init__(self, values, base_model_id=None):
        self.params = _EntryParams(values)
        self.base_model_id = base_model_id

    def model_copy(self, update=None):
        return self


def test_chat_completion_binds_resolved_params_per_call(monkeypatch):
    main = importlib.import_module('open_webui.main')
    captured = {}

    async def payload(
        _request,
        form_data,
        _user,
        metadata,
        _model,
        *,
        default_model_params=None,
        request_params=None,
        resolved_model_id=None,
        resolved_model_params=None,
        **_kwargs,
    ):
        # Correlate per column: fan-out columns may share one routing id.
        captured[metadata['message_id']] = (form_data['model'], resolved_model_id, resolved_model_params)
        return form_data, metadata, None, {}

    async def handler(*_args, **_kwargs):
        return SimpleNamespace(status_code=200)

    async def build_ctx(*_args, **_kwargs):
        return {}

    async def respond(*_args, **_kwargs):
        return {'status': True}

    async def fake_create_task(_redis, process, id=None, task_id=None):
        await process
        return task_id, None

    async def fake_event_emitter(*_args, **_kwargs):
        return None

    async def fake_config_get(key, default=None):
        return {
            'models.default_params': {},
            'chat.tool_permissions.enable': False,
            'ui.default_models': 'fallback-model',
        }.get(key, default)

    async def fake_access(*_args, **_kwargs):
        return None

    model_infos: dict = {}

    async def get_model(model_id):
        return model_infos.get(model_id)

    monkeypatch.setattr(main, 'process_chat_payload', payload)
    monkeypatch.setattr(main, 'chat_completion_handler', handler)
    monkeypatch.setattr(main, 'build_chat_response_context', build_ctx)
    monkeypatch.setattr(main, 'process_chat_response', respond)
    monkeypatch.setattr(main, 'create_task', fake_create_task)
    monkeypatch.setattr(main, 'get_event_emitter', fake_event_emitter)
    monkeypatch.setattr(main, 'cleanup_task', fake_create_task)
    monkeypatch.setattr(main, 'has_active_tasks', fake_event_emitter)
    monkeypatch.setattr(main.Config, 'get', staticmethod(fake_config_get))
    monkeypatch.setattr(main.Models, 'get_model_by_id', staticmethod(get_model))
    monkeypatch.setattr(main, 'check_model_access', fake_access)
    monkeypatch.setattr(main, 'BYPASS_MODEL_ACCESS_CONTROL', False)
    monkeypatch.setattr(main, 'BYPASS_ADMIN_ACCESS_CONTROL', True)
    monkeypatch.setattr(main, 'ENABLE_CUSTOM_MODEL_FALLBACK', True)

    request = SimpleNamespace(
        headers={},
        state=SimpleNamespace(internal=False),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={}, redis=None)),
    )
    user = SimpleNamespace(id='user-1', role='admin')

    def run(model, message_ids):
        captured.clear()
        request.app.state.MODELS = models_map

        form_data = {
            'model': model,
            'messages': [{'role': 'user', 'content': 'hi'}],
            'session_id': 'sess-1',
            'chat_id': 'local:unit',
            'message_ids': message_ids,
        }
        result = asyncio.run(main.chat_completion(request, form_data, user))
        assert result['status'] is True
        return dict(captured)

    models_map = {
        'primary-model': {'id': 'primary-model', 'owned_by': 'openai', 'info': {}},
        'sibling-model': {'id': 'sibling-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos.update(
        {
            'primary-model': _EntryModelInfo({'system': 'primary policy'}),
            'sibling-model': _EntryModelInfo({'system': 'sibling policy'}),
        }
    )
    bound = run(
        'primary-model',
        [
            {'model_id': 'primary-model', 'message_id': 'm1'},
            {'model_id': 'sibling-model', 'message_id': 'm2'},
        ],
    )
    assert bound['m1'] == ('primary-model', 'primary-model', {'system': 'primary policy'})
    assert bound['m2'] == ('sibling-model', None, None)

    reordered = run(
        'primary-model',
        [
            {'model_id': 'sibling-model', 'message_id': 'm1'},
            {'model_id': 'primary-model', 'message_id': 'm2'},
        ],
    )
    assert reordered['m2'] == ('primary-model', 'primary-model', {'system': 'primary policy'})
    assert reordered['m1'] == ('sibling-model', None, None)

    models_map = {
        'custom-model': {'id': 'custom-model', 'owned_by': 'openai', 'info': {}},
        'fallback-model': {'id': 'fallback-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos.clear()
    model_infos.update(
        {
            'custom-model': _EntryModelInfo({'system': 'custom policy'}, base_model_id='missing-base'),
            'fallback-model': _EntryModelInfo({'system': 'fallback policy'}),
        }
    )
    fallback = run(
        'custom-model',
        [
            {'model_id': 'custom-model', 'message_id': 'm1'},
            {'model_id': 'fallback-model', 'message_id': 'm2'},
        ],
    )
    # Custom column routes through the fallback while keeping custom params;
    # the explicit fallback column re-resolves the fallback's own DB params.
    assert fallback['m1'] == ('fallback-model', 'fallback-model', {'system': 'custom policy'})
    assert fallback['m2'] == ('fallback-model', None, None)

    models_map = {
        'arena-model': {
            'id': 'arena-model',
            'owned_by': 'arena',
            'info': {'meta': {'model_ids': ['target-model']}},
        },
        'target-model': {'id': 'target-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos.clear()
    model_infos.update(
        {
            'arena-model': _EntryModelInfo({'system': 'arena policy'}),
            'target-model': _EntryModelInfo({'system': 'target policy'}),
        }
    )
    arena = run(
        'arena-model',
        [{'model_id': 'arena-model', 'message_id': 'm1'}],
    )
    assert arena['m1'] == ('arena-model', None, None)


def test_chat_completion_sync_leg_binds_fallback_params(monkeypatch):
    main = importlib.import_module('open_webui.main')
    captured = {}

    async def payload(
        _request,
        form_data,
        _user,
        metadata,
        _model,
        *,
        default_model_params=None,
        request_params=None,
        resolved_model_id=None,
        resolved_model_params=None,
        **_kwargs,
    ):
        captured[form_data['model']] = (resolved_model_id, resolved_model_params)
        return form_data, metadata, None, {}

    async def handler(*_args, **_kwargs):
        return SimpleNamespace(status_code=200)

    async def build_ctx(*_args, **_kwargs):
        return {}

    async def respond(*_args, **_kwargs):
        return {'status': True}

    async def fake_config_get(key, default=None):
        return {
            'models.default_params': {},
            'chat.tool_permissions.enable': False,
            'ui.default_models': 'fallback-model',
        }.get(key, default)

    async def fake_access(*_args, **_kwargs):
        return None

    model_infos = {
        'custom-model': _EntryModelInfo({'system': 'custom policy'}, base_model_id='missing-base'),
        'fallback-model': _EntryModelInfo({'system': 'fallback policy'}),
    }

    async def get_model(model_id):
        return model_infos.get(model_id)

    monkeypatch.setattr(main, 'process_chat_payload', payload)
    monkeypatch.setattr(main, 'chat_completion_handler', handler)
    monkeypatch.setattr(main, 'build_chat_response_context', build_ctx)
    monkeypatch.setattr(main, 'process_chat_response', respond)
    monkeypatch.setattr(main.Config, 'get', staticmethod(fake_config_get))
    monkeypatch.setattr(main.Models, 'get_model_by_id', staticmethod(get_model))
    monkeypatch.setattr(main, 'check_model_access', fake_access)
    monkeypatch.setattr(main, 'BYPASS_MODEL_ACCESS_CONTROL', False)
    monkeypatch.setattr(main, 'BYPASS_ADMIN_ACCESS_CONTROL', True)
    monkeypatch.setattr(main, 'ENABLE_CUSTOM_MODEL_FALLBACK', True)

    request = SimpleNamespace(
        headers={},
        state=SimpleNamespace(internal=False),
        app=SimpleNamespace(
            state=SimpleNamespace(
                MODELS={
                    'custom-model': {'id': 'custom-model', 'owned_by': 'openai', 'info': {}},
                    'fallback-model': {'id': 'fallback-model', 'owned_by': 'openai', 'info': {}},
                },
                redis=None,
            )
        ),
    )
    user = SimpleNamespace(id='user-1', role='admin')
    form_data = {
        'model': 'custom-model',
        'messages': [{'role': 'user', 'content': 'hi'}],
        'message_ids': [{'model_id': 'custom-model', 'message_id': 'm1'}],
    }

    result = asyncio.run(main.chat_completion(request, form_data, user))

    assert result == {'status': True}
    assert list(captured) == ['fallback-model']
    assert captured['fallback-model'] == ('fallback-model', {'system': 'custom policy'})


def test_fanout_sibling_re_resolves_own_db_params_without_primary_leak(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(MODELS={})),
        state=SimpleNamespace(direct=False),
    )
    captured = {}

    class SiblingParams:
        def model_dump(self):
            return {'system': 'sibling policy', 'temperature': 0.4}

    async def get_model(model_id):
        assert model_id == 'sibling-model'
        return SimpleNamespace(params=SiblingParams())

    class StopAfterParams(Exception):
        pass

    def capture(form_data, _model):
        captured.update(copy.deepcopy(form_data))
        raise StopAfterParams

    monkeypatch.setattr(middleware.Models, 'get_model_by_id', staticmethod(get_model))
    monkeypatch.setattr(middleware, 'apply_params_to_form_data', capture)

    with pytest.raises(StopAfterParams):
        asyncio.run(
            middleware.process_chat_payload(
                request,
                {'model': 'sibling-model', 'messages': []},
                None,
                {'chat_id': ''},
                {'id': 'sibling-model', 'owned_by': 'openai'},
                default_model_params={},
                request_params={},
                resolved_model_id=None,
                resolved_model_params=None,
            )
        )

    assert captured['params'] == {'system': 'sibling policy', 'temperature': 0.4}


def test_canonical_history_projects_core_fields_and_ignores_metadata():
    empty_string = [{'role': 'user', 'content': ''}]
    empty_parts = [{'role': 'user', 'content': []}]
    assert compaction._canonical_history_entry(empty_string).ref != compaction._canonical_history_entry(empty_parts).ref

    plain = compaction._canonical_history_entry([{'role': 'user', 'content': 'hello'}])
    with_metadata = compaction._canonical_history_entry(
        [
            {
                'id': 'message',
                'parentId': 'parent',
                'role': 'user',
                'content': 'hello',
                'files': [{'id': 'file'}],
                'usage': {'input_tokens': 10},
                'model': 'model',
                'meta': {'internal': False},
            }
        ]
    )
    assert with_metadata.ref == plain.ref

    provider_extension = compaction._canonical_history_entry(
        [{'role': 'user', 'content': 'hello', 'provider_extension': {'value': 42}}]
    )
    assert provider_extension.ref != plain.ref
    assert 'provider_extension' in provider_extension.text

    plain_part = compaction._canonical_history_entry(
        [{'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}]}]
    )
    part_with_metadata = compaction._canonical_history_entry(
        [{'role': 'user', 'content': [{'type': 'text', 'text': 'hello', 'cache_control': {}}]}]
    )
    assert part_with_metadata.ref != plain_part.ref
    assert 'cache_control' in part_with_metadata.text

    image_history = compaction._canonical_history_entry(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {'url': 'data:image/png;base64,AA==', 'detail': 'high'},
                    }
                ],
            }
        ]
    )
    assert '<media omitted>' in image_history.text

    extension_part = compaction._canonical_history_entry(
        [{'role': 'user', 'content': [{'type': 'provider_extension', 'value': {'answer': 42}}]}]
    )
    assert 'provider_extension' in extension_part.text
    assert '42' in extension_part.text

    tool_only = compaction._canonical_history_entry(
        [
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [{'id': 'call', 'function': {'name': 'lookup', 'arguments': '{}'}}],
            }
        ]
    )
    assert 'lookup' in tool_only.text


def test_responses_history_uses_core_projection_for_unknown_shapes():
    output = [
        {'type': 'web_search_call', 'id': 'search-1', 'status': 'completed'},
        {'type': 'file_search_call', 'id': 'search-2', 'status': 'completed'},
        {'type': 'computer_call', 'id': 'computer-1', 'status': 'completed'},
        {'type': 'mcp_call', 'id': 'mcp-1', 'status': 'completed'},
        {'type': 'image_generation_call', 'id': 'image-1', 'status': 'completed'},
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'text', 'text': ''}]},
        {
            'type': 'message',
            'role': 'assistant',
            'content': [
                {
                    'type': 'output_text',
                    'text': 'answer',
                    'annotations': [{'type': 'url_citation', 'url': 'https://example.test'}],
                    'logprobs': [],
                    'parsed': {'answer': True},
                },
                {'type': 'output_text', 'text': ' continued', 'annotations': [], 'logprobs': None},
                {'type': 'refusal', 'refusal': 'cannot comply'},
            ],
        },
    ]

    assert compaction._canonical_output_messages(output) == [
        {'role': 'assistant', 'content': 'answer continued'}
    ]


def test_history_ancestor_resolution_lists_metadata_and_serves_bodies_on_demand():
    messages = [
        {'role': 'user', 'content': 'root'},
        {'role': 'user', 'content': 'checkpoint 壱', 'contextSummary': 'one'},
        {'role': 'assistant', 'content': 'middle'},
        {'role': 'user', 'content': 'checkpoint two', 'contextSummary': 'two'},
        {'role': 'assistant', 'content': 'recent'},
        {'role': 'user', 'content': 'checkpoint three', 'contextSummary': 'three'},
    ]
    selected = compaction._history_entry_at(messages, 5)
    near = compaction._canonical_history_entry(messages[:3])
    far = compaction._canonical_history_entry(messages[:1])
    assert selected is not None and selected.load_history is not None

    assert asyncio.run(selected.load_history(f'history:{"f" * 64}')) == ()

    assert asyncio.run(selected.load_history(near.ref)) == (
        compaction.replace(near, load_history=selected.load_history),
    )

    loaded_far = asyncio.run(selected.load_history(far.ref))
    assert loaded_far == (compaction.replace(far, load_history=selected.load_history),)

    listed = asyncio.run(selected.load_history(None))
    assert [entry.ref for entry in listed] == [near.ref, far.ref]
    # Listing reports measurements only; bodies are cut from the selected text.
    assert [entry.text for entry in listed] == ['', '']
    assert [entry.utf8_bytes for entry in listed] == [near.utf8_bytes, far.utf8_bytes]
    assert [entry.line_count for entry in listed] == [near.line_count, far.line_count]


def _nested_checkpoint_chat():
    body = 'x' * 600
    messages = []
    for index in range(10):
        if index % 2 == 0:
            messages.append({'role': 'user', 'content': f'u{index} {body}'})
            continue
        message = {
            'role': 'assistant',
            'content': f'a{index} {body}',
            'contextSummary': f's{index}',
        }
        if index in (3, 7):
            message['output'] = [
                {'type': 'message', 'content': [{'type': 'output_text', 'text': f'o{index} {body}'}]},
                {
                    'type': 'function_call',
                    'call_id': f'k{index}',
                    'name': 'lookup',
                    'arguments': '{}',
                    'status': 'completed',
                    'contextSummary': f'n{index}-1',
                },
                {
                    'type': 'function_call_output',
                    'call_id': f'k{index}',
                    'output': [{'type': 'input_text', 'text': f'r{index} {body}'}],
                    'contextSummary': f'n{index}-2',
                },
                {
                    'type': 'message',
                    'content': [{'type': 'output_text', 'text': f'p{index} {body}'}],
                    'contextSummary': f'n{index}-3',
                },
            ]
        messages.append(message)
    return messages


def _canonical_ancestors(messages, selected_index, selected_output_index):
    """Ancestor walk rebuilt from full canonical prefix texts."""
    selected = compaction._canonical_history_entry(
        compaction._history_prefix_messages(messages, selected_index, selected_output_index)
    )
    limit = compaction._history_position_key((selected_index, selected_output_index))
    positions = [
        (message_index, output_index)
        for message_index, output_index, _ in compaction._checkpoint_positions(messages)
        if compaction._history_position_key((message_index, output_index)) < limit
    ]
    expected = []
    for position in reversed(positions):
        ancestor = compaction._canonical_history_entry(
            compaction._history_prefix_messages(messages, *position)
        )
        if not (ancestor.text and selected.text.startswith(ancestor.text)):
            break
        expected.append(ancestor)
    return len(positions), tuple(expected)


def test_history_listing_matches_canonical_prefixes_with_nested_checkpoints():
    messages = _nested_checkpoint_chat()
    selected_index, selected_output_index, _ = compaction._checkpoint_positions(messages)[-1]
    entry = compaction._history_entry_at_position(messages, selected_index, selected_output_index)
    assert entry is not None and entry.load_history is not None
    candidates, expected = _canonical_ancestors(messages, selected_index, selected_output_index)
    # A nested position that is not a prefix truncates the walk before the older ones.
    assert 0 < len(expected) < candidates

    listed = asyncio.run(entry.load_history(None))
    assert [item.ref for item in listed] == [item.ref for item in expected]
    for item, want in zip(listed, expected):
        assert (item.utf8_bytes, item.line_count) == (want.utf8_bytes, want.line_count)
        assert asyncio.run(entry.load_history(item.ref)) == (
            compaction.replace(want, load_history=entry.load_history),
        )


def test_history_listing_peak_memory_stays_below_one_prefix():
    body = 'x' * 4096
    messages = []
    for index in range(96):
        if index % 2 == 0:
            messages.append({'role': 'user', 'content': f'u{index} {body}'})
        else:
            messages.append(
                {'role': 'assistant', 'content': f'a{index} {body}', 'contextSummary': f's{index}'}
            )
    entry = compaction._history_entry_at(messages, len(messages) - 1)
    assert entry is not None and entry.load_history is not None

    gc.collect()
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        listed = asyncio.run(entry.load_history(None))
        peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()

    assert len(listed) == 47
    # Rebuilding every prefix would cost roughly len(listed) / 2 whole prefixes.
    assert peak < entry.utf8_bytes


def test_concurrent_history_listings_never_read_a_partial_prefix_cache(monkeypatch):
    messages = _nested_checkpoint_chat()
    fresh = compaction._history_entry_at(messages, len(messages) - 1)
    assert fresh is not None and fresh.load_history is not None
    expected = [item.ref for item in asyncio.run(fresh.load_history(None))]
    assert expected

    entry = compaction._history_entry_at(messages, len(messages) - 1)
    assert entry is not None and entry.load_history is not None
    records = compaction._canonical_history_records
    parked = threading.Event()
    release = threading.Event()

    def stalled(items):
        # The first resolver parks inside its prefix walk while the second runs to
        # completion; the second must build its own list, not read a partial one.
        if not parked.is_set():
            parked.set()
            assert release.wait(5)
        return records(items)

    monkeypatch.setattr(compaction, '_canonical_history_records', stalled)

    async def race():
        first = asyncio.ensure_future(entry.load_history(None))
        assert await asyncio.to_thread(parked.wait, 5)
        second = await entry.load_history(None)
        release.set()
        return await first, second

    first, second = asyncio.run(race())
    assert [item.ref for item in first] == [item.ref for item in second] == expected


def test_non_string_code_interpreter_result_round_trips_checkpoint():
    prefix = [
        {'id': 'u1', 'parentId': None, 'role': 'user', 'content': 'calculate'},
        {
            'id': 'a1',
            'parentId': 'u1',
            'role': 'assistant',
            'content': '',
            'output': [
                {
                    'type': 'open_webui:code_interpreter',
                    'code': '2 + 2',
                    'output': {'result': {'result': 4}},
                }
            ],
        },
    ]
    carrier = {
        'id': 'u2',
        'parentId': 'a1',
        'role': 'user',
        'content': 'continue',
        'contextSummary': 'summary',
    }
    source = compaction._history_entry_at([*prefix, carrier], 2)

    assert source is not None
    assert "{'result': 4}" in source.text
    assert compaction._canonical_output_messages(
        [
            {'type': 'open_webui:code_interpreter', 'code': '', 'output': None},
            {'type': 'open_webui:code_interpreter', 'code': '', 'output': [4]},
        ]
    ) == [{'role': 'assistant', 'content': '<code_interpreter_output>\n[4]\n</code_interpreter_output>'}]


def test_disabled_compaction_does_not_compile_transient_patterns(monkeypatch):
    values = {
        'chat.context_compaction.enable': False,
        'chat.context_compaction.transient_message_patterns': '[',
    }

    async def get_many(*_keys):
        return values

    monkeypatch.setattr(compaction.Config, 'get_many', get_many)
    assert asyncio.run(compaction._load_config())['transient_patterns'] == ()

    values['chat.context_compaction.enable'] = True
    with pytest.raises(re.error):
        asyncio.run(compaction._load_config())


def test_manual_compaction_persists_only_core_checkpoint(monkeypatch):
    messages = {
        'u1': {
            'id': 'u1',
            'parentId': None,
            'role': 'user',
            'content': 'find evidence',
            'sources': [{'source': {'id': 'search-result'}}],
        },
        'a1': {
            'id': 'a1',
            'parentId': 'u1',
            'role': 'assistant',
            'content': 'answer',
            'sources': [{'source': {'id': 'citation'}}],
        },
        'u2': {'id': 'u2', 'parentId': 'a1', 'role': 'user', 'content': 'continue'},
    }

    async def load_config():
        return {'enable': True, 'prompt_template': '', 'transient_patterns': ()}

    async def get_messages(_chat_id):
        return messages

    async def generate_summary(*_args, **kwargs):
        assert 'message_id' in kwargs
        return 'summary'

    saved_update = None

    async def save(chat_id, message_id, summary, **_kwargs):
        nonlocal saved_update
        saved_update = (chat_id, message_id, summary)
        messages[message_id]['contextSummary'] = summary
        return True

    monkeypatch.setattr(compaction, '_load_config', load_config)
    monkeypatch.setattr(compaction.Chats, 'get_messages_map_by_chat_id', get_messages)
    monkeypatch.setattr(compaction.ChatMessages, 'update_context_summary', save)
    monkeypatch.setattr(compaction, '_generate_summary', generate_summary)
    chat = SimpleNamespace(
        id='chat',
        current_message_id='u2',
        chat={'history': {'currentId': 'u2', 'messages': messages}},
    )
    result = asyncio.run(compaction.compact_chat_branch(None, None, chat, 'model', {}))

    assert result['compacted'] is True
    assert saved_update == ('chat', 'u2', 'summary')


def _tool_round_tip_messages():
    return {
        'u1': {'id': 'u1', 'parentId': None, 'role': 'user', 'content': 'find evidence'},
        'a1': {'id': 'a1', 'parentId': 'u1', 'role': 'assistant', 'content': 'answer'},
        'u2': {'id': 'u2', 'parentId': 'a1', 'role': 'user', 'content': 'continue'},
        'a2': {
            'id': 'a2',
            'parentId': 'u2',
            'role': 'assistant',
            'content': 'final',
            'output': [
                {
                    'type': 'function_call',
                    'call_id': 'call-1',
                    'name': 'lookup',
                    'arguments': '{}',
                    'status': 'completed',
                },
                {
                    'type': 'function_call_output',
                    'call_id': 'call-1',
                    'output': [{'type': 'input_text', 'text': 'tool result'}],
                    'status': 'completed',
                },
            ],
        },
    }


def test_manual_compact_excludes_tip_tool_round_from_summary_input(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    messages = _tool_round_tip_messages()
    captured = {}

    async def load_config():
        return {'enable': True, 'prompt_template': '', 'transient_patterns': ()}

    async def get_messages(_chat_id):
        return messages

    async def generate_summary(*args, **_kwargs):
        captured['compacted'] = args[4]
        captured['recent'] = args[5]
        return 'summary'

    async def save(chat_id, message_id, summary, **_kwargs):
        messages[message_id]['contextSummary'] = summary
        return True

    monkeypatch.setattr(compaction, '_load_config', load_config)
    monkeypatch.setattr(compaction.Chats, 'get_messages_map_by_chat_id', get_messages)
    monkeypatch.setattr(compaction.ChatMessages, 'update_context_summary', save)
    monkeypatch.setattr(compaction, '_generate_summary', generate_summary)
    chat = SimpleNamespace(
        id='chat',
        current_message_id='a2',
        chat={'history': {'currentId': 'a2', 'messages': messages}},
    )

    result = asyncio.run(compaction.compact_chat_branch(None, None, chat, 'model', {}))

    assert result['compacted'] is True
    assert result['dropped_messages'] == 3
    assert result['kept_messages'] == 1
    compacted_text = json.dumps(captured['compacted'])
    assert 'lookup' not in compacted_text
    assert 'tool result' not in compacted_text
    recent_text = json.dumps(captured['recent'])
    assert 'lookup' in recent_text
    assert 'tool result' in recent_text

    replayed, replay_state = compaction.replay_stored_compaction_checkpoint(
        compaction.get_message_list(messages, 'a2')
    )
    assert _summary_content_of(replayed) is replay_state['summary_message_content']
    expanded = middleware.process_messages_with_output(replayed)
    tool_calls = [call for message in expanded for call in (message.get('tool_calls') or [])]
    assert [call.get('id') for call in tool_calls] == ['call-1']
    assert [message.get('tool_call_id') for message in expanded if message.get('role') == 'tool'] == ['call-1']


def test_manual_compact_nested_tip_only_is_too_short(monkeypatch):
    messages = {'a1': _tool_round_tip_messages()['a2']}
    saves = []

    async def load_config():
        return {'enable': True, 'prompt_template': '', 'transient_patterns': ()}

    async def get_messages(_chat_id):
        return messages

    async def generate_summary(*_args, **_kwargs):
        raise AssertionError('too-short branches must not summarize')

    async def save(_chat_id, _message_id, _summary, **_kwargs):
        saves.append(True)
        return True

    monkeypatch.setattr(compaction, '_load_config', load_config)
    monkeypatch.setattr(compaction.Chats, 'get_messages_map_by_chat_id', get_messages)
    monkeypatch.setattr(compaction.ChatMessages, 'update_context_summary', save)
    monkeypatch.setattr(compaction, '_generate_summary', generate_summary)
    chat = SimpleNamespace(
        id='chat',
        current_message_id='a1',
        chat={'history': {'currentId': 'a1', 'messages': messages}},
    )

    result = asyncio.run(compaction.compact_chat_branch(None, None, chat, 'model', {}))

    assert result == {'ok': True, 'compacted': False, 'reason': 'too_short'}
    assert saves == []


def _run_manual_compact(monkeypatch, messages, current_id):
    captured = {}

    async def load_config():
        return {'enable': True, 'prompt_template': '', 'transient_patterns': ()}

    async def get_messages(_chat_id):
        return messages

    async def generate_summary(*args, **kwargs):
        assert 'message_id' in kwargs
        captured['compacted'] = args[4]
        captured['recent'] = args[5]
        return 'summary'

    async def save(chat_id, message_id, summary, **_kwargs):
        messages[message_id]['contextSummary'] = summary
        return True

    monkeypatch.setattr(compaction, '_load_config', load_config)
    monkeypatch.setattr(compaction.Chats, 'get_messages_map_by_chat_id', get_messages)
    monkeypatch.setattr(compaction.ChatMessages, 'update_context_summary', save)
    monkeypatch.setattr(compaction, '_generate_summary', generate_summary)
    chat = SimpleNamespace(
        id='chat',
        current_message_id=current_id,
        chat={'history': {'currentId': current_id, 'messages': messages}},
    )
    return asyncio.run(compaction.compact_chat_branch(None, None, chat, 'model', {})), captured


def test_manual_compact_summary_input_excludes_system_messages(monkeypatch):
    messages = {
        's1': {'id': 's1', 'parentId': None, 'role': 'system', 'content': 'system policy'},
        'u1': {'id': 'u1', 'parentId': 's1', 'role': 'user', 'content': 'find evidence'},
        'a1': {'id': 'a1', 'parentId': 'u1', 'role': 'assistant', 'content': 'answer'},
        'u2': {'id': 'u2', 'parentId': 'a1', 'role': 'user', 'content': 'continue'},
    }

    result, captured = _run_manual_compact(monkeypatch, messages, 'u2')

    assert result['compacted'] is True
    assert result['dropped_messages'] == 2
    assert result['kept_messages'] == 1
    assert all(message.get('role') != 'system' for message in captured['compacted'])
    assert all(message.get('role') != 'system' for message in captured['recent'])
    assert [message.get('content') for message in captured['compacted']] == ['find evidence', 'answer']
    assert [message.get('content') for message in captured['recent']] == ['continue']


def test_manual_compact_system_with_nested_checkpoint_tip_is_too_short(monkeypatch):
    tip = _tool_round_tip_messages()['a2']
    tip['parentId'] = 's1'
    tip['output'][0]['contextSummary'] = 'earlier nested checkpoint'
    messages = {
        's1': {'id': 's1', 'parentId': None, 'role': 'system', 'content': 'system policy'},
        'a2': tip,
    }
    saves = []

    async def load_config():
        return {'enable': True, 'prompt_template': '', 'transient_patterns': ()}

    async def get_messages(_chat_id):
        return messages

    async def generate_summary(*_args, **_kwargs):
        raise AssertionError('too-short branches must not summarize')

    async def save(_chat_id, _message_id, _summary, **_kwargs):
        saves.append(True)
        return True

    monkeypatch.setattr(compaction, '_load_config', load_config)
    monkeypatch.setattr(compaction.Chats, 'get_messages_map_by_chat_id', get_messages)
    monkeypatch.setattr(compaction.ChatMessages, 'update_context_summary', save)
    monkeypatch.setattr(compaction, '_generate_summary', generate_summary)
    chat = SimpleNamespace(
        id='chat',
        current_message_id='a2',
        chat={'history': {'currentId': 'a2', 'messages': messages}},
    )

    result = asyncio.run(compaction.compact_chat_branch(None, None, chat, 'model', {}))

    assert result == {'ok': True, 'compacted': False, 'reason': 'too_short'}
    assert saves == []


def test_normalized_checkpoint_survives_interleaved_chat_reconciliation(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    chat_messages = importlib.import_module('open_webui.models.chat_messages')

    @asynccontextmanager
    async def use_session(db=None):
        assert isinstance(db, AsyncSession)
        yield db

    monkeypatch.setattr(chat_messages, 'get_async_db_context', use_session)

    async def run():
        paused = asyncio.Event()
        resume = asyncio.Event()

        class PausingSession(AsyncSession):
            async def commit(self):
                paused.set()
                await resume.wait()
                await super().commit()

        engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "chat.db"}')
        normal_session = async_sessionmaker(engine, expire_on_commit=False)
        pausing_session = async_sessionmaker(engine, class_=PausingSession, expire_on_commit=False)
        try:
            async with engine.begin() as connection:
                await connection.run_sync(chat_messages.ChatMessage.__table__.create)

            async with normal_session() as db:
                await chat_messages.ChatMessages.upsert_message(
                    'message',
                    'chat',
                    'user',
                    {'role': 'assistant', 'content': 'before'},
                    db=db,
                )

            async with pausing_session() as db:
                chat_update = asyncio.create_task(
                    chat_messages.ChatMessages.upsert_message(
                        'message',
                        'chat',
                        'user',
                        {'role': 'assistant', 'content': 'after'},
                        db=db,
                    )
                )
                await paused.wait()
                async with normal_session() as checkpoint_db:
                    assert await chat_messages.ChatMessages.update_context_summary(
                        'chat',
                        'message',
                        'checkpoint',
                        db=checkpoint_db,
                    )
                resume.set()
                await chat_update

            async with normal_session() as db:
                await chat_messages.ChatMessages.upsert_messages(
                    'chat',
                    'user',
                    {'message': {'role': 'assistant', 'content': 'reconciled'}},
                    db=db,
                )
                message = await db.get(chat_messages.ChatMessage, 'chat-message')
                assert message.content == 'reconciled'
                assert message.context_summary == 'checkpoint'
        finally:
            resume.set()
            await engine.dispose()

    asyncio.run(run())


def test_compaction_cpu_work_is_dispatched_off_the_event_loop(monkeypatch):
    calls = []
    prefix = [
        {'id': 'u1', 'parentId': None, 'role': 'user', 'content': 'old'},
        {'id': 'a1', 'parentId': 'u1', 'role': 'assistant', 'content': 'answer'},
    ]
    messages = [
        *prefix,
        {
            'id': 'u2',
            'parentId': 'a1',
            'role': 'user',
            'content': 'continue',
            'contextSummary': 'summary',
        },
    ]

    async def to_thread(function, *args, **kwargs):
        calls.append(function)
        return function(*args, **kwargs)

    async def load_config():
        return {
            'enable': True,
            'retention_percentage': 40,
            'transient_patterns': (),
        }

    async def run():
        _, state = await compaction.prepare_compaction_messages(messages, {'chat_id': 'chat'})
        await compaction.resolve_request_history(state['selected_history'])
        await compaction.compact_provider_payload(
            None,
            None,
            {'messages': [{'role': 'user', 'content': 'small'}]},
            {},
            'model',
            {},
            {
                'config': {
                    'enable': True,
                    'token_threshold': 1000,
                    'token_cap': 1000,
                    'soft_trigger_ratio': 0,
                }
            },
        )
    monkeypatch.setattr(compaction.asyncio, 'to_thread', to_thread)
    monkeypatch.setattr(compaction, '_load_config', load_config)
    asyncio.run(run())

    assert copy.deepcopy in calls
    assert compaction.find_safe_compaction_boundary in calls
    assert compaction._history_entry_at in calls
    assert compaction.estimate_provider_tokens in calls


def test_checkpoint_preparation_copies_only_the_active_suffix(monkeypatch):
    messages = [
        {'id': 'u1', 'role': 'user', 'content': 'old'},
        {'id': 'a1', 'role': 'assistant', 'content': 'old answer'},
        {'id': 'u2', 'role': 'user', 'content': 'checkpoint', 'contextSummary': 'summary'},
        {'id': 'a2', 'role': 'assistant', 'content': 'recent answer'},
        {'id': 'u3', 'role': 'user', 'content': 'current'},
    ]
    copied = []
    real_deepcopy = copy.deepcopy

    async def load_config():
        return {'enable': True, 'retention_percentage': 40, 'transient_patterns': ()}

    def deepcopy(value, memo=None):
        copied.append(value)
        return real_deepcopy(value, memo)

    monkeypatch.setattr(compaction, '_load_config', load_config)
    monkeypatch.setattr(compaction.copy, 'deepcopy', deepcopy)
    prepared, state = asyncio.run(compaction.prepare_compaction_messages(messages, {'chat_id': 'chat'}))

    assert copied == [messages[2:]]
    assert _summary_content_of(prepared) is state['summary_message_content']


def test_prepare_disabled_replay_pins_summary_identity(monkeypatch):
    messages = [
        {'id': 'u1', 'role': 'user', 'content': 'old'},
        {'id': 'u2', 'role': 'user', 'content': 'current', 'contextSummary': 'stored summary'},
    ]

    async def load_config():
        return {'enable': False, 'transient_patterns': ()}

    monkeypatch.setattr(compaction, '_load_config', load_config)
    prepared, state = asyncio.run(compaction.prepare_compaction_messages(messages, {'chat_id': 'chat'}))

    assert _summary_content_of(prepared) is state['summary_message_content']
    assert state['previous_summary'] == 'stored summary'


def test_historical_excerpts_stop_after_the_requested_recent_messages():
    class Poison(dict):
        def get(self, *_args, **_kwargs):
            raise AssertionError('older messages must not be scanned')

    messages = [Poison(), *({'role': 'user', 'content': str(index)} for index in range(32))]
    assert compaction._historical_user_excerpts(messages, 512, 32) == [str(index) for index in range(32)]


def test_excerpt_truncation_does_not_encode_the_full_source():
    class HugeText(str):
        def encode(self, *_args, **_kwargs):
            raise AssertionError('the full source must not be encoded')

    excerpt = compaction._middle_truncate_utf8(HugeText('界' * 1_000_000), 512)
    assert len(excerpt.encode('utf-8')) <= 512


def test_history_is_not_hashed_when_core_cannot_install_the_reader(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    state = {
        'selected_history': (
            [
                {'role': 'user', 'content': 'old'},
                {'role': 'user', 'content': 'current', 'contextSummary': 'summary'},
            ],
            1,
        ),
        'externalized_refs': {
            'enable': True,
            'native': True,
            'threshold': 1000,
            'registry': {},
            'metadata': {},
        },
    }
    monkeypatch.setattr(
        compaction,
        '_canonical_history_entry',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('history must remain lazy')),
    )

    body = asyncio.run(
        middleware.apply_externalized_refs(
            {'stream': False, 'messages': [{'role': 'user', 'content': 'current'}]},
            state,
        )
    )

    assert body == {'stream': False, 'messages': [{'role': 'user', 'content': 'current'}]}
    assert state['externalized_refs']['registry'] == {}


def test_disabled_refs_never_rewrite_user_content():
    middleware = importlib.import_module('open_webui.utils.middleware')
    content = '<auto_compaction_context><history_ref>history:' + 'a' * 64 + '</history_ref></auto_compaction_context>'
    body = {'messages': [{'role': 'user', 'content': content}]}
    state = {'externalized_refs': {'enable': False}}

    result = asyncio.run(middleware.apply_externalized_refs(body, state))

    assert result is body
    assert result['messages'][0]['content'] == content
    assert 'projection_source_messages' not in state


def test_tool_only_refs_do_not_rewrite_user_content():
    middleware = importlib.import_module('open_webui.utils.middleware')
    content = '<auto_compaction_context><history_ref>history:' + 'a' * 64 + '</history_ref></auto_compaction_context>'
    body = {
        'stream': True,
        'messages': [
            {'role': 'user', 'content': content},
            {'role': 'tool', 'tool_call_id': 'call', 'content': 'large result ' * 1000},
        ],
    }
    state = {
        'externalized_refs': {
            'enable': True,
            'native': True,
            'threshold': 1000,
            'registry': {},
            'metadata': {},
        }
    }

    result = asyncio.run(middleware.apply_externalized_refs(body, state))

    assert result['messages'][0]['content'] == content
    projected = result['messages'][1]['content']
    assert projected != 'large result ' * 1000
    assert '<auto_compact_ref_truncated>' in projected
    assert re.search(r'tool:[0-9a-f]{64}', projected)


def test_post_filter_payload_controls_ref_catalog():
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    hidden = refs.make_ref_entry('filtered secret', kind='tool')
    visible = refs.make_ref_entry('allowed result', kind='tool')
    history = refs.make_ref_entry('filtered history', kind='history')
    assert hidden is not None and visible is not None and history is not None

    filtered_state = {
        'selected_history': history,
        'tool_ref_entries': (hidden,),
        'externalized_refs': {
            'enable': True,
            'native': True,
            'threshold': 1000,
            'registry': {},
            'metadata': {},
        },
    }
    filtered_body = {
        'stream': True,
        'messages': [
            {'role': 'user', 'content': 'filtered'},
            {'role': 'assistant', 'output': []},
        ],
    }

    assert asyncio.run(middleware.apply_externalized_refs(filtered_body, filtered_state)) is filtered_body
    assert filtered_state['externalized_refs']['registry'] == {}

    summary_text = f'preserve {visible.ref}'
    summary = compaction.render_summary_message(summary_text)
    allowed_state = {
        'selected_history': history,
        'previous_summary': summary_text,
        'previous_summary_meta': {},
        'summary_message_content': summary['content'],
        'tool_ref_entries': (hidden, visible),
        'externalized_refs': {
            'enable': True,
            'native': True,
            'threshold': 1000,
            'registry': {},
            'metadata': {},
        },
    }
    allowed = asyncio.run(
        middleware.apply_externalized_refs(
            {'stream': True, 'messages': [summary]},
            allowed_state,
        )
    )
    reader = allowed_state['externalized_refs']['registry'][middleware.REF_EXEC_TOOL_NAME]['callable']

    assert visible.ref in asyncio.run(reader('ls tool')).splitlines()
    assert hidden.ref not in asyncio.run(reader('ls tool')).splitlines()
    assert asyncio.run(reader(f'cat {hidden.ref}')).startswith('Error:')
    assert history.ref in asyncio.run(reader('ls history')).splitlines()
    assert f'<history_ref>{history.ref}</history_ref>' in allowed['messages'][0]['content']

    masked_state = {
        **allowed_state,
        'externalized_refs': {
            **allowed_state['externalized_refs'],
            'registry': {},
            'metadata': {},
        },
    }
    masked = {
        'stream': True,
        'messages': [compaction.render_summary_message(f'filtered {visible.ref}')],
    }

    masked = asyncio.run(middleware.apply_externalized_refs(masked, masked_state))
    assert masked_state['externalized_refs']['registry'] == {}
    assert '<history_ref>' not in masked['messages'][0]['content']


def _ref_admission_state(refs, seed_entries):
    return {
        'tool_ref_entries': tuple(seed_entries),
        'externalized_refs': {
            'enable': True,
            'native': True,
            'threshold': 2,
            'registry': {},
            'metadata': {},
        },
    }


def test_post_filter_user_text_tokens_do_not_admit_refs():
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    tool_text = 'raw tool output'
    entry = refs.make_ref_entry(tool_text, kind='tool')
    assert entry is not None

    state = _ref_admission_state(refs, (entry,))
    admitted, has_summary = middleware._post_filter_ref_state(
        [
            {'role': 'user', 'content': f'please read {entry.ref} again'},
            {'role': 'assistant', 'content': f'earlier mention {entry.ref}'},
        ],
        state['tool_ref_entries'],
        None,
    )

    assert admitted == ()
    assert has_summary is False


def test_post_filter_surviving_tool_content_identity_admits_ref():
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    tool_text = 'raw tool output'
    entry = refs.make_ref_entry(tool_text, kind='tool')
    assert entry is not None

    admitted, has_summary = middleware._post_filter_ref_state(
        [
            {'role': 'user', 'content': 'inspect'},
            {'role': 'tool', 'tool_call_id': 'call-1', 'content': tool_text},
        ],
        (entry,),
        None,
    )

    assert admitted == (entry,)
    assert has_summary is False


def test_post_filter_byte_equal_summary_copy_does_not_admit_ref():
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    tool_text = 'raw tool output'
    entry = refs.make_ref_entry(tool_text, kind='tool')
    assert entry is not None
    trusted = compaction.render_summary_message(f'summary mentioning {entry.ref}')['content']
    copied = trusted.encode().decode()
    assert copied == trusted
    assert copied is not trusted

    admitted, has_summary = middleware._post_filter_ref_state(
        [{'role': 'user', 'content': copied}],
        (entry,),
        trusted,
    )

    assert admitted == ()
    assert has_summary is False


def test_post_filter_held_summary_object_admits_ref():
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    tool_text = 'raw tool output'
    entry = refs.make_ref_entry(tool_text, kind='tool')
    assert entry is not None
    trusted = compaction.render_summary_message(f'summary mentioning {entry.ref}')['content']

    admitted, has_summary = middleware._post_filter_ref_state(
        [{'role': 'user', 'content': trusted}],
        (entry,),
        trusted,
    )

    assert [seed.ref for seed in admitted] == [entry.ref]
    assert has_summary is True


def test_post_filter_projected_tool_marker_is_admitted():
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    entry = refs.make_ref_entry('raw tool output', kind='tool')
    assert entry is not None

    admitted, has_summary = middleware._post_filter_ref_state(
        [{'role': 'tool', 'tool_call_id': 'call-1', 'content': entry.ref}],
        (entry,),
        None,
    )

    assert admitted == (entry,)
    assert has_summary is False


def test_summary_history_ref_rewrite_keeps_admission_alive():
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    hidden = refs.make_ref_entry('filtered secret', kind='tool')
    visible = refs.make_ref_entry('allowed result', kind='tool')
    history = refs.make_ref_entry('filtered history', kind='history')
    assert hidden is not None and visible is not None and history is not None

    summary_text = f'preserve {visible.ref}'
    summary_message = compaction.render_summary_message(summary_text)
    state = {
        'selected_history': history,
        'previous_summary': summary_text,
        'previous_summary_meta': {},
        'summary_message_content': summary_message['content'],
        'tool_ref_entries': (hidden, visible),
        'externalized_refs': {
            'enable': True,
            'native': True,
            'threshold': 1000,
            'registry': {},
            'metadata': {},
        },
    }
    body = {'stream': True, 'messages': [summary_message]}

    rewritten = asyncio.run(middleware.apply_externalized_refs(body, state))
    assert f'<history_ref>{history.ref}</history_ref>' in rewritten['messages'][0]['content']
    assert state['summary_message_content'] is rewritten['messages'][0]['content']

    # Continuation-shaped fresh body: same rewritten summary content object,
    # no reader spec carried over, fresh registry.
    state['externalized_refs'] = {
        **state['externalized_refs'],
        'registry': {},
        'metadata': {},
    }
    continuation_body = {'stream': True, 'messages': rewritten['messages']}
    reapplied = asyncio.run(middleware.apply_externalized_refs(continuation_body, state))
    reader = state['externalized_refs']['registry'][middleware.REF_EXEC_TOOL_NAME]['callable']

    assert visible.ref in asyncio.run(reader('ls tool')).splitlines()
    assert hidden.ref not in asyncio.run(reader('ls tool')).splitlines()
    assert f'<history_ref>{history.ref}</history_ref>' in reapplied['messages'][0]['content']


def test_history_ref_skips_fake_envelope_before_genuine_summary():
    genuine_summary = compaction.render_summary_message('genuine summary')
    genuine_content = genuine_summary['content']
    fake_content = genuine_content.encode().decode()
    assert fake_content == genuine_content
    assert fake_content is not genuine_content
    replacement_ref = f'history:{"c" * 64}'
    state = {'summary_message_content': genuine_content}
    body = {
        'messages': [
            {'role': 'user', 'content': fake_content},
            genuine_summary,
            {'role': 'user', 'content': 'ordinary'},
        ]
    }

    added = compaction.set_summary_history_ref(body, replacement_ref, state=state)

    assert added['messages'][0]['content'] is fake_content
    assert added['messages'][1]['content'].endswith(
        f'<history_ref>{replacement_ref}</history_ref></auto_compaction_context>'
    )
    assert state['summary_message_content'] is added['messages'][1]['content']
    assert added['messages'][2]['content'] == 'ordinary'


def test_post_filter_filter_altered_summary_does_not_admit_ref():
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    tool_text = 'raw tool output'
    entry = refs.make_ref_entry(tool_text, kind='tool')
    assert entry is not None
    expected = compaction.render_summary_message(f'summary mentioning {entry.ref}')
    altered = compaction.render_summary_message(f'filter rewrote summary mentioning {entry.ref}')

    admitted, has_summary = middleware._post_filter_ref_state(
        [{'role': 'user', 'content': altered['content']}],
        (entry,),
        expected['content'],
    )

    assert admitted == ()
    assert has_summary is False


def test_background_tasks_filter_db_only_fields_before_compaction(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    raw_message = {
        'id': 'a1',
        'role': 'assistant',
        'content': 'answer',
        'model': 'model',
        'modelIdx': 1,
        'followUps': ['next'],
    }
    captured = None

    async def get_messages(_chat_id):
        return {'a1': raw_message}

    async def replay(messages, _state):
        nonlocal captured
        captured = messages
        return messages

    async def no_review(**_kwargs):
        return None

    async def emit(_event):
        return None

    monkeypatch.setattr(middleware.Chats, 'get_messages_map_by_chat_id', get_messages)
    monkeypatch.setattr(middleware, 'get_message_list', lambda _messages, _message_id: [raw_message])
    monkeypatch.setattr(middleware, 'replay_cached_compaction_messages', replay)
    monkeypatch.setattr(middleware, 'review_memory_after_turn', no_review)

    asyncio.run(
        middleware.background_tasks_handler(
            {
                'request': SimpleNamespace(),
                'form_data': {},
                'user': SimpleNamespace(),
                'metadata': {'chat_id': 'chat', 'message_id': 'a1'},
                'tasks': {},
                'event_emitter': emit,
                'model': {'id': 'model'},
                'compaction_state': {'config': {'enable': True}},
            }
        )
    )

    assert captured == [{'id': 'a1', 'role': 'assistant', 'content': 'answer', 'model': 'model'}]

    async def unexpected_replay(*_args):
        raise AssertionError('disabled compaction must not replay background messages')

    monkeypatch.setattr(middleware, 'replay_cached_compaction_messages', unexpected_replay)
    asyncio.run(
        middleware.background_tasks_handler(
            {
                'request': SimpleNamespace(),
                'form_data': {},
                'user': SimpleNamespace(),
                'metadata': {'chat_id': 'chat', 'message_id': 'a1'},
                'tasks': {},
                'event_emitter': emit,
                'model': {'id': 'model'},
                'compaction_state': {'config': {'enable': False}},
            }
        )
    )


def test_soft_prefetch_is_reused_at_the_hard_threshold(monkeypatch):
    calls = 0
    before = 60
    history = [
        {'id': 'u1', 'role': 'user', 'content': 'old'},
        {'id': 'a1', 'role': 'assistant', 'content': 'answer'},
        {'id': 'u2', 'role': 'user', 'content': 'new', compaction._BOUNDARY_KEY: True},
    ]

    async def generate_checkpoint(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return (
            'summary',
            {'historical_user_messages': []},
            (history, 2),
            'u2',
        )

    async def save_checkpoint(*_args, **_kwargs):
        return True

    def estimate(body, **_kwargs):
        if any(
            isinstance(message.get('content'), str) and message['content'].startswith('<auto_compaction_context>')
            for message in body['messages']
        ):
            return 20
        return before

    monkeypatch.setattr(compaction, '_generate_checkpoint', generate_checkpoint)
    monkeypatch.setattr(compaction, 'estimate_provider_tokens', estimate)
    monkeypatch.setattr(compaction.ChatMessages, 'update_context_summary', save_checkpoint)
    body = {'messages': history}
    state = {
        'active_offset': 0,
        'checkpoint_messages': history,
        'checkpoint_history': (history, 2),
        'config': {
            'enable': True,
            'token_threshold': 100,
            'token_cap': 100,
            'retention_percentage': 40,
            'prompt_template': '',
            'soft_trigger_ratio': 0.5,
            'transient_patterns': (),
        }
    }

    async def run():
        nonlocal before
        metadata = {'chat_id': 'chat'}
        first = await compaction.compact_provider_payload(None, None, body, metadata, 'model', {}, state)
        assert compaction._BOUNDARY_KEY not in first['messages'][2]
        await state['prefetch_task']
        before = 120
        second = await compaction.compact_provider_payload(None, None, first, metadata, 'model', {}, state)
        assert second['messages'][0]['content'].startswith('<auto_compaction_context>')
        assert second['messages'][0]['content'] is state['summary_message_content']

    asyncio.run(run())
    assert calls == 1


def _compaction_config(**overrides):
    config = {
        'enable': True,
        'token_threshold': 100,
        'token_cap': 100,
        'retention_percentage': 40,
        'prompt_template': '',
        'soft_trigger_ratio': 0,
        'transient_patterns': (),
    }
    config.update(overrides)
    return config


def _estimate_compacted_when_summarized(body):
    return (
        20
        if any(
            isinstance(message.get('content'), str)
            and message['content'].startswith('<auto_compaction_context>')
            for message in body['messages']
        )
        else 120
    )


def test_nested_compact_pins_summary_identity(monkeypatch):
    socket_main = importlib.import_module('open_webui.socket.main')

    async def generate_summary(*_args, **_kwargs):
        return 'FOLDED'

    async def save(_chat_id, _message_id, update, **_kwargs):
        return update

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(compaction, '_generate_summary', generate_summary)
    monkeypatch.setattr(compaction, 'estimate_provider_tokens', _estimate_compacted_when_summarized)
    monkeypatch.setattr(compaction.Chats, 'upsert_message_to_chat_by_id_and_message_id', save)
    monkeypatch.setattr(socket_main, 'get_event_emitter', noop)

    carrier = {'type': 'function_call', 'call_id': 'call-1', 'name': 'lookup', 'arguments': '{}'}
    output = [carrier]
    messages = [
        {'id': 'user', 'role': 'user', 'content': 'start'},
        {'id': 'assistant', 'role': 'assistant', 'content': '', 'output': output},
        {'id': 'user2', 'role': 'user', 'content': 'follow-up'},
    ]
    state = {'config': _compaction_config()}

    candidate = asyncio.run(
        compaction.compact_transient_provider_payload(
            None,
            None,
            {'messages': messages},
            {'chat_id': 'chat', 'message_id': 'assistant'},
            'model',
            {},
            state,
            checkpoint_output=output,
            checkpoint_carrier=carrier,
            checkpoint_message_start=1,
        )
    )

    assert _summary_content_of(candidate['messages']) is state['summary_message_content']


def test_boundary_compact_pins_summary_identity(monkeypatch):
    async def generate_summary(*_args, **kwargs):
        assert 'message_id' in kwargs
        return 'BOUNDARY'

    monkeypatch.setattr(compaction, '_generate_summary', generate_summary)
    monkeypatch.setattr(compaction, 'estimate_provider_tokens', _estimate_compacted_when_summarized)

    messages = [
        {'role': 'user', 'content': 'old question'},
        {'role': 'assistant', 'content': 'old answer'},
        {'role': 'user', 'content': 'new question'},
    ]
    state = {'config': _compaction_config()}

    compacted = asyncio.run(
        compaction.compact_transient_provider_payload(
            None,
            None,
            {'messages': messages},
            {'chat_id': 'local:unit'},
            'model',
            {},
            state,
        )
    )

    assert _summary_content_of(compacted['messages']) is state['summary_message_content']


def test_stateful_continuation_sends_only_current_output():
    middleware = importlib.import_module('open_webui.utils.middleware')
    body = {
        'model': 'model',
        'messages': [
            {'role': 'system', 'content': 'policy'},
            {'role': 'user', 'content': 'start'},
            {
                'role': 'assistant',
                'tool_calls': [{'id': 'call-a', 'function': {'name': 'a', 'arguments': '{}'}}],
            },
            {'role': 'tool', 'tool_call_id': 'call-a', 'content': 'result a'},
            {
                'role': 'assistant',
                'tool_calls': [{'id': 'call-b', 'function': {'name': 'b', 'arguments': '{}'}}],
            },
            {'role': 'tool', 'tool_call_id': 'call-b', 'content': 'result b'},
        ],
    }
    stateful = middleware._stateful_continuation_body(body, 'response-b', {'call-b'})
    assert stateful['previous_response_id'] == 'response-b'
    assert stateful['messages'] == [body['messages'][0], body['messages'][-1]]


def test_tool_stream_preserves_nested_checkpoint_and_citation_shape(monkeypatch):  # noqa: C901
    middleware = importlib.import_module('open_webui.utils.middleware')
    current = {}

    def response(delta):
        async def chunks():
            yield f'data: {middleware.JSONCodec.dumps(delta)}\n\n'.encode()
            yield b'data: [DONE]\n\n'

        return StreamingResponse(chunks(), media_type='text/event-stream')

    def tool_response(call_id, name):
        return response(
            {
                'choices': [
                    {
                        'delta': {
                            'tool_calls': [
                                {
                                    'index': 0,
                                    'id': call_id,
                                    'type': 'function',
                                    'function': {'name': name, 'arguments': '{}'},
                                }
                            ]
                        },
                        'finish_reason': 'tool_calls',
                    }
                ]
            }
        )

    async def generate(_request, candidate, _user, **_kwargs):
        current['sent'].append(copy.deepcopy(candidate))
        return current['replies'].pop(0)

    async def process_result(_request, _name, result, *_args):
        return result, [], []

    async def noop(*_args, **_kwargs):
        return None

    async def config_get(_key, default=None):
        return default

    async def rag(_template, context, prompt):
        return f'RAG[{context}]PROMPT[{prompt}]'

    async def title(_chat_id):
        return ''

    async def generate_summary(*_args, **kwargs):
        assert 'message_id' in kwargs
        return 'FOLDED'

    async def save(_chat_id, _message_id, update, **_kwargs):
        current['saved'].append(copy.deepcopy(update))
        return update

    async def view_file():
        return 'result-a'

    async def fetch_url():
        return 'result-b'

    def source_for(name):
        return {
            'source': {'id': name, 'name': name, 'type': 'tool'},
            'document': [f'document-{name}'],
            'metadata': [{'source': name}],
        }

    def estimate(body):
        return (
            20
            if any(
                isinstance(message.get('content'), str)
                and message['content'].startswith('<auto_compaction_context>')
                for message in body['messages']
            )
            else 120
        )

    def make_ctx(enabled):
        tools = {
            name: {
                'spec': {'name': name, 'parameters': {'type': 'object', 'properties': {}}},
                'callable': function,
            }
            for name, function in (('view_file', view_file), ('fetch_url', fetch_url))
        }
        metadata = {
            'chat_id': 'chat',
            'message_id': 'assistant',
            'user_prompt': 'start',
            'params': {'tool_approval_mode': 'full'},
            'tools': tools,
        }
        ctx = {
            'request': SimpleNamespace(
                state=SimpleNamespace(max_tool_call_iterations=5),
                app=SimpleNamespace(state=SimpleNamespace(redis=None, MODELS={})),
            ),
            'form_data': {
                'model': 'model',
                'stream': True,
                'messages': [{'role': 'user', 'content': 'start'}],
                'metadata': metadata,
            },
            'user': SimpleNamespace(),
            'model': {'id': 'model', 'info': {'meta': {'capabilities': {'citations': True}}}},
            'metadata': metadata,
            'events': [],
            'tasks': {},
            'event_emitter': noop,
            'event_caller': None,
            'compaction_state': {
                'config': {
                    'enable': enabled,
                    'token_threshold': 100,
                    'token_cap': 100,
                    'retention_percentage': 40,
                    'prompt_template': '',
                    'soft_trigger_ratio': 0,
                    'transient_patterns': (),
                    'externalized_refs_enable': False,
                    'externalized_refs_token_threshold': 10_000,
                },
                'checkpoint_messages': [
                    {'id': 'user', 'role': 'user', 'content': 'start'},
                    {'id': 'assistant', 'role': 'assistant', 'content': ''},
                ],
                'externalized_refs': {'enable': False},
            },
        }
        ctx['compaction_state']['canonical_body'] = ctx['form_data']
        ctx['compaction_state']['last_send_body'] = ctx['form_data']
        return ctx

    real_compact = middleware._compact_final_provider_payload

    async def compact(*args, **kwargs):
        output = kwargs['checkpoint_output']
        carrier = kwargs['checkpoint_carrier']
        carrier_index = next(index for index, item in enumerate(output) if item is carrier)
        assert carrier['type'] == 'function_call'
        current['carrier_indices'].append(carrier_index)
        return await real_compact(*args, **kwargs)

    monkeypatch.setattr(middleware, 'ENABLE_RESPONSES_API_STATEFUL', False)
    monkeypatch.setattr(middleware, 'RAG_SYSTEM_CONTEXT', False)
    monkeypatch.setattr(middleware, 'generate_chat_completion', generate)
    monkeypatch.setattr(middleware, 'process_tool_result', process_result)
    monkeypatch.setattr(
        middleware,
        'get_citation_source_from_tool_result',
        lambda tool_name, **_kwargs: [source_for(tool_name)],
    )
    monkeypatch.setattr(middleware, 'rag_template', rag)
    monkeypatch.setattr(middleware.Config, 'get', config_get)
    monkeypatch.setattr(middleware, 'terminal_event_handler', noop)
    monkeypatch.setattr(middleware, 'get_system_oauth_token', noop)
    monkeypatch.setattr(middleware, 'outlet_filter_handler', noop)
    monkeypatch.setattr(middleware, 'background_tasks_handler', noop)
    monkeypatch.setattr(middleware, 'clear_response_stream', noop)
    monkeypatch.setattr(middleware, 'save_response_stream', noop)
    monkeypatch.setattr(middleware, 'publish_chat_finished_event', noop)
    monkeypatch.setattr(middleware.Chats, 'upsert_message_to_chat_by_id_and_message_id', save)
    monkeypatch.setattr(middleware.Chats, 'get_chat_title_by_id', title)
    monkeypatch.setattr(middleware, '_compact_final_provider_payload', compact)
    monkeypatch.setattr(compaction, '_generate_summary', generate_summary)
    monkeypatch.setattr(compaction, 'estimate_provider_tokens', estimate)
    socket_main = importlib.import_module('open_webui.socket.main')
    monkeypatch.setattr(socket_main, 'get_event_emitter', noop)

    async def run_case(enabled, replies):
        current.update(sent=[], saved=[], carrier_indices=[], replies=replies)
        await middleware.streaming_chat_response_handler(
            tool_response('call-a', 'view_file'),
            make_ctx(enabled),
        )
        return copy.deepcopy(current)

    folded = asyncio.run(
        run_case(
            True,
            [
                tool_response('call-b', 'fetch_url'),
                response({'choices': [{'delta': {'content': 'done'}, 'finish_reason': 'stop'}]}),
            ],
        )
    )
    assert folded['saved']
    assert any(
        item.get('contextSummary') == 'FOLDED'
        for update in folded['saved']
        for item in update.get('output', [])
        if isinstance(item, dict)
    )
    assert folded['carrier_indices'][0] == 0
    assert folded['carrier_indices'][1] > 0
    summaries = [
        [
            message
            for message in body['messages']
            if isinstance(message.get('content'), str)
            and message['content'].startswith('<auto_compaction_context>')
        ]
        for body in folded['sent']
    ]
    assert len(summaries) == 2 and len(summaries[0]) == 1
    assert summaries[1] == summaries[0]
    rag_users = [
        message
        for message in folded['sent'][1]['messages']
        if message.get('role') == 'user'
        and isinstance(message.get('content'), str)
        and message['content'].startswith('RAG[')
    ]
    assert len(rag_users) == 1
    assert rag_users[0][compaction.CONTEXT_COMPACTION_TRANSIENT_MARKER_KEY] is True
    assert 'resource-id="view_file"' in rag_users[0]['content']
    assert 'resource-id="fetch_url"' in rag_users[0]['content']

    plain = asyncio.run(
        run_case(
            False,
            [response({'choices': [{'delta': {'content': 'done'}, 'finish_reason': 'stop'}]})],
        )
    )
    expected = (
        'RAG[<source id="1" name="view_file" resource-type="tool" resource-id="view_file"></source>]'
        'PROMPT[start]\nstart'
    )
    assert plain['sent'][0]['messages'][0] == {'role': 'user', 'content': expected}
    assert [message['role'] for message in plain['sent'][0]['messages']] == ['user', 'assistant', 'tool']


def test_transient_messages_use_core_provenance_before_regex_fallback():
    patterns = (re.compile(r'^injected:'),)
    assert compaction._is_transient_message(
        {'role': 'user', 'content': 'ordinary', 'meta': {'internal': True, 'type': 'subagent'}},
        patterns,
    )
    assert compaction._is_transient_message({'role': 'user', 'content': '  injected:value'}, patterns)
    assert not compaction._is_transient_message({'role': 'user', 'content': 'ordinary'}, patterns)


# ---- v0.11.3 merge: canonical/send split, send copies, and ref apply contracts ----


def _payload_leg_patches(monkeypatch, middleware, *, request_filter=None, refs_runtime=None):
    async def noop(*_args, **_kwargs):
        return None

    async def config_get(_key, default=None):
        return default

    async def pipeline_inlet(_request, form_data, _user, _models):
        return form_data

    async def url_images(form_data, user=None):
        return form_data

    async def system_prompt(*_args, **_kwargs):
        return None

    def apply_params(form_data, _model):
        return form_data

    monkeypatch.setattr(middleware.Config, 'get', config_get)
    monkeypatch.setattr(middleware, 'process_pipeline_inlet_filter', pipeline_inlet)
    monkeypatch.setattr(middleware, 'convert_url_images_to_base64', url_images)
    monkeypatch.setattr(middleware, 'resolve_system_prompt', system_prompt)
    monkeypatch.setattr(middleware, 'apply_params_to_form_data', apply_params)
    monkeypatch.setattr(middleware, 'get_event_emitter', noop)
    monkeypatch.setattr(middleware, 'get_event_call', noop)
    monkeypatch.setattr(middleware, 'get_system_oauth_token', noop)
    monkeypatch.setattr(middleware, 'get_task_model_id', lambda model_id, *args, **kwargs: model_id)
    if refs_runtime is not None:
        async def runtime():
            return refs_runtime

        monkeypatch.setattr(middleware, '_runtime_externalized_refs_config', runtime)
    if request_filter is not None:
        async def process(**kwargs):
            if kwargs.get('filter_type') == 'request' and kwargs.get('filter_functions'):
                return await kwargs['filter_functions'][0](
                    body=kwargs['form_data'],
                    __metadata__=kwargs.get('extra_params', {}).get('__metadata__'),
                ), {}
            return kwargs['form_data'], {}

        async def get_filters(*_args, **_kwargs):
            return [request_filter]

        monkeypatch.setattr(middleware, 'process_filter_functions', process)
        monkeypatch.setattr(middleware, 'get_filter_functions', get_filters)
        monkeypatch.setattr(middleware, 'ENABLE_PLUGINS', True)


def _payload_leg_drain(monkeypatch, middleware, stored, tool_result=None, stub_execute=True):
    async def get_message(_chat_id, _message_id):
        return stored

    async def get_events(_metadata):
        return None, None

    async def upsert(*_args, **_kwargs):
        return stored

    async def execute(*_args, **_kwargs):
        return tool_result or {'tool_call_id': 'call-1', 'content': 'result'}

    monkeypatch.setattr(middleware.Chats, 'get_message_by_id_and_message_id', get_message)
    monkeypatch.setattr(middleware, 'get_event_emitter_and_caller', get_events)
    monkeypatch.setattr(middleware.Chats, 'upsert_message_to_chat_by_id_and_message_id', upsert)
    if stub_execute:
        monkeypatch.setattr(middleware, 'execute_tool_call_for_output', execute)


def _seed_state_on_capture(monkeypatch, middleware, seed):
    real = middleware._capture_pre_filter_tool_refs

    async def wrapper(body, metadata, state, payload_tools, native=None):
        # Seed once: pass 2 must observe the held that pass 1 left behind
        # (rolled back or not), not a fresh reseed.
        if 'summary_message_content' not in state:
            for key, value in seed.items():
                state[key] = value
        return await real(body, metadata, state, payload_tools, native=native)

    monkeypatch.setattr(middleware, '_capture_pre_filter_tool_refs', wrapper)


def _run_payload_leg(middleware, messages, *, params=None, seeded_metadata=None, message_id='message', assistant_message_id='assistant', form_tool_ids=None):
    request = SimpleNamespace(
        state=SimpleNamespace(direct=False),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={'model': {'id': 'model'}}, redis=None)),
    )
    metadata = {
        'chat_id': 'chat',
        'params': params or {},
        'features': {},
    }
    if message_id is not None:
        metadata['message_id'] = message_id
    if assistant_message_id is not None:
        metadata['assistant_message_id'] = assistant_message_id
    if seeded_metadata:
        metadata.update(seeded_metadata)
    form_data = {
        'model': 'model',
        'stream': True,
        'messages': messages,
        'metadata': metadata,
    }
    if form_tool_ids is not None:
        form_data['tool_ids'] = form_tool_ids
    model = {'id': 'model', 'info': {'meta': {'capabilities': {'file_context': False}}}}
    return middleware.process_chat_payload(request, form_data, None, metadata, model)


def _approved_stored():
    return {
        'output': [
            {
                'type': 'function_call',
                'call_id': 'call-1',
                'name': 'lookup',
                'arguments': '{}',
                'status': 'queued',
                'approved': True,
            }
        ]
    }


def test_payload_send_carries_filter_marker_and_canonical_stays_clean(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    calls = []

    async def marker_filter(body, __metadata__=None):
        calls.append(copy.deepcopy(body['messages']))
        messages = body['messages']
        messages[-1]['content'] = messages[-1].get('content', '') + '|MARK|'
        return body

    _payload_leg_patches(monkeypatch, middleware, request_filter=marker_filter, refs_runtime=(False, 1000))
    _payload_leg_drain(monkeypatch, middleware, {'output': []})
    messages = [{'role': 'user', 'content': 'start'}]

    send, _metadata, _events, state = asyncio.run(_run_payload_leg(middleware, copy.deepcopy(messages)))

    assert send['messages'][-1]['content'] == 'start|MARK|'
    assert state['canonical_body']['messages'] == messages
    assert state['last_send_body'] is send
    assert state['canonical_body'] is not send
    assert len(calls) == 1


def test_paused_rollback_restores_ref_apply_state(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    history = refs.make_ref_entry('history summary', kind='history')
    assert history is not None
    held = compaction.render_summary_message('summary text')['content']
    sentinel_tools = {'pre': {'spec': {'name': 'pre'}}}
    stored = {
        'output': [
            {
                'type': 'function_call',
                'call_id': 'call-1',
                'name': 'lookup',
                'arguments': '{}',
                'status': 'queued',
            }
        ]
    }

    async def marker_filter(body, __metadata__=None):
        # Touch only the opening user message: the held summary message must
        # keep its content identity so apply can mint and rollback can undo.
        first = body['messages'][0]
        first['content'] = first.get('content', '') + '|MARK|'
        return body

    _payload_leg_patches(monkeypatch, middleware, request_filter=marker_filter, refs_runtime=(True, 2))
    _payload_leg_drain(monkeypatch, middleware, stored)
    _seed_state_on_capture(
        monkeypatch,
        middleware,
        {'selected_history': history, 'summary_message_content': held},
    )
    messages = [
        {'role': 'user', 'content': 'start'},
        {'role': 'user', 'content': held},
    ]

    returned, metadata, _events, state = asyncio.run(
        _run_payload_leg(
            middleware,
            messages,
            params={'tool_approval_mode': 'ask'},
            seeded_metadata={'tools': sentinel_tools},
        )
    )

    assert state['paused'] is True
    assert all('<history_ref>' not in message.get('content', '') for message in returned['messages'])
    assert all('|MARK|' not in message.get('content', '') for message in returned['messages'])
    assert state['summary_message_content'] is held
    assert state['externalized_refs']['registry'] == {}
    assert metadata['tools'] is sentinel_tools
    assert 'canonical_body' not in state
    assert 'last_send_body' not in state


def test_resume_transforms_each_send_copy_once(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    seen = []

    async def needle_filter(body, __metadata__=None):
        first = body['messages'][0]
        seen.append(first['content'])
        first['content'] = first['content'].replace('NEEDLE', 'DONE', 1)
        return body

    _payload_leg_patches(monkeypatch, middleware, request_filter=needle_filter, refs_runtime=(False, 1000))
    _payload_leg_drain(monkeypatch, middleware, _approved_stored())

    send, _metadata, _events, state = asyncio.run(
        _run_payload_leg(middleware, [{'role': 'user', 'content': 'NEEDLE NEEDLE'}])
    )

    assert seen == ['NEEDLE NEEDLE', 'NEEDLE NEEDLE']
    assert send['messages'][0]['content'] == 'DONE NEEDLE'
    assert state['canonical_body']['messages'][0]['content'] == 'NEEDLE NEEDLE'


def test_approved_pass2_mint_reflects_into_canonical(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    history = refs.make_ref_entry('history summary', kind='history')
    assert history is not None
    held = compaction.render_summary_message('summary text')['content']

    _payload_leg_patches(monkeypatch, middleware, refs_runtime=(True, 1000))
    _payload_leg_drain(monkeypatch, middleware, _approved_stored())
    _seed_state_on_capture(
        monkeypatch,
        middleware,
        {'selected_history': history, 'summary_message_content': held},
    )
    messages = [
        {'role': 'user', 'content': 'question'},
        {'role': 'user', 'content': held},
    ]

    send, _metadata, _events, state = asyncio.run(_run_payload_leg(middleware, messages))

    canonical_held = next(
        message for message in state['canonical_body']['messages'] if '<auto_compaction_context>' in message['content']
    )
    assert '<history_ref>' in canonical_held['content']
    assert canonical_held['content'] is state['summary_message_content']
    assert '<history_ref>' in send['messages'][1]['content']
    registry = state['externalized_refs']['registry']
    assert refs.REF_EXEC_TOOL_NAME in registry


def test_pass2_filtered_ref_absent_from_final_catalog(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    big = 'filtered tool output ' * 600
    calls = []

    async def deleting_filter(body, __metadata__=None):
        calls.append(len(body['messages']))
        if len(calls) == 1:
            return body
        kept = [message for message in body['messages'] if message.get('role') != 'tool']
        return {**body, 'messages': kept}

    _payload_leg_patches(monkeypatch, middleware, request_filter=deleting_filter, refs_runtime=(True, 1000))
    _payload_leg_drain(monkeypatch, middleware, _approved_stored())
    messages = [
        {'role': 'user', 'content': 'question'},
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {
                    'id': 'old-call',
                    'type': 'function',
                    'function': {'name': 'lookup', 'arguments': '{}'},
                }
            ],
        },
        {'role': 'tool', 'tool_call_id': 'old-call', 'content': big},
    ]

    send, metadata, _events, state = asyncio.run(_run_payload_leg(middleware, messages))

    entry = refs.make_ref_entry(big, kind='tool')
    assert entry is not None
    registry = state['externalized_refs']['registry']
    assert refs.REF_EXEC_TOOL_NAME not in registry
    assert 'tools' not in metadata
    assert all(message.get('role') != 'tool' for message in send['messages'])
    assert state['canonical_body']['messages'][2]['content'] is big


def test_approved_rollback_keeps_filter_metadata_changes(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    history = refs.make_ref_entry('history summary', kind='history')
    assert history is not None
    held = compaction.render_summary_message('summary text')['content']
    sentinel = {'sentinel': {'spec': {'name': 'sentinel'}}}
    calls = []

    async def sentinel_filter(body, __metadata__=None):
        calls.append(1)
        if len(calls) == 1:
            __metadata__['tools'] = sentinel
            return body
        kept = [
            message
            for message in body['messages']
            if message.get('role') != 'tool' and message.get('content') is not held
        ]
        return {**body, 'messages': kept}

    _payload_leg_patches(monkeypatch, middleware, request_filter=sentinel_filter, refs_runtime=(True, 1000))
    _payload_leg_drain(monkeypatch, middleware, _approved_stored())
    _seed_state_on_capture(
        monkeypatch,
        middleware,
        {'selected_history': history, 'summary_message_content': held},
    )
    messages = [
        {'role': 'user', 'content': 'question'},
        {'role': 'user', 'content': held},
    ]

    _send, metadata, _events, state = asyncio.run(_run_payload_leg(middleware, messages))

    assert metadata['tools'] == sentinel
    assert refs.REF_EXEC_TOOL_NAME not in state['externalized_refs']['registry']
    assert state['summary_message_content'] is held


def test_projected_view_prevents_false_context_limit(monkeypatch):
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    big = 'tool output ' + 'x' * 500
    messages = [
        {'role': 'user', 'content': 'question'},
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {'id': 't1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}
            ],
        },
        {'role': 'tool', 'tool_call_id': 't1', 'content': big},
    ]
    entry = refs.make_ref_entry(big, kind='tool')
    assert entry is not None
    projected_messages = [message for message in messages]
    projected_messages[2] = {**projected_messages[2], 'content': entry.ref}

    def estimate(body):
        return sum(
            len(message.get('content')) if isinstance(message.get('content'), str) else 0
            for message in body['messages']
        )

    monkeypatch.setattr(compaction, 'estimate_provider_tokens', estimate)
    state = {
        'config': {
            'enable': True,
            'token_threshold': 100,
            'token_cap': 10_000,
            'retention_percentage': 40,
            'prompt_template': '',
            'transient_patterns': (),
            'soft_trigger_ratio': 0,
        }
    }
    body = {'messages': messages}

    result = asyncio.run(
        compaction.compact_transient_provider_payload(
            None,
            None,
            body,
            {},
            'model',
            {},
            state,
            projected_messages=projected_messages,
        )
    )

    assert result is body

    with pytest.raises(RuntimeError):
        asyncio.run(
            compaction.compact_transient_provider_payload(
                None,
                None,
                {'messages': copy.deepcopy(messages)},
                {},
                'model',
                {},
                {'config': dict(state['config'])},
                projected_messages=None,
            )
        )


def test_round0_non_native_disables_continuation_capture():
    middleware = importlib.import_module('open_webui.utils.middleware')
    big = 'capture payload line\n' * 400
    state = {
        'config': {'externalized_refs_enable': True, 'externalized_refs_token_threshold': 1000},
        'externalized_refs': {'native': False},
    }
    body = {
        'stream': True,
        'messages': [
            {'role': 'user', 'content': 'question'},
            {'role': 'tool', 'tool_call_id': 't1', 'content': big},
        ],
    }
    metadata = {'chat_id': 'chat', 'message_id': 'message', 'params': {}}

    gated = asyncio.run(
        middleware._capture_pre_filter_tool_refs(body, metadata, state, payload_tools=None, native=False)
    )

    assert gated is None
    assert 'tool_ref_entries' not in state

    unrestricted = asyncio.run(
        middleware._capture_pre_filter_tool_refs(body, metadata, state, payload_tools=None)
    )

    assert unrestricted is not None and unrestricted is not body['messages']


def test_gate_skipped_apply_with_lingering_history_is_safe():
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    history = refs.make_ref_entry('history summary', kind='history')
    assert history is not None
    held = compaction.render_summary_message('summary text')['content']
    registry = {}
    state = {
        'selected_history': history,
        'summary_message_content': held,
        'externalized_refs': {
            'enable': True,
            'native': True,
            'threshold': 2,
            'registry': registry,
            'metadata': {},
        },
    }
    body = {'stream': False, 'messages': [{'role': 'user', 'content': held}]}

    result = asyncio.run(middleware.apply_externalized_refs(body, state))

    assert result is body
    assert '<history_ref>' not in result['messages'][0]['content']
    assert registry == {}


def test_round1_install_failure_preserves_marker_and_held():
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    history = refs.make_ref_entry('history summary', kind='history')
    assert history is not None
    held = compaction.render_summary_message('summary text')['content']
    registry = {}
    state = {
        'selected_history': history,
        'summary_message_content': held,
        'externalized_refs': {
            'enable': True,
            'native': True,
            'threshold': 2,
            'registry': registry,
            'metadata': {},
        },
    }
    body = {'stream': True, 'messages': [{'role': 'user', 'content': held}]}

    minted = asyncio.run(middleware.apply_externalized_refs(body, state))

    assert minted is not body or minted['messages'][0]['content'] is not held
    assert '<history_ref>' in minted['messages'][0]['content']
    assert state['summary_message_content'] is minted['messages'][0]['content']
    held_after_round0 = state['summary_message_content']

    # Round 1: a same-name foreign tool occupies the registry, killing install.
    registry[refs.REF_EXEC_TOOL_NAME] = {
        'spec': {'name': refs.REF_EXEC_TOOL_NAME, 'parameters': {'type': 'object', 'properties': {}}},
        'callable': lambda: None,
    }
    body_round1 = {'stream': True, 'messages': [{'role': 'user', 'content': held_after_round0}]}

    result = asyncio.run(middleware.apply_externalized_refs(body_round1, state))

    assert result is body_round1
    assert result['messages'][0]['content'] is held_after_round0
    assert '<history_ref>' in result['messages'][0]['content']
    assert state['summary_message_content'] is held_after_round0


def test_send_copy_isolates_top_level_containers():
    middleware = importlib.import_module('open_webui.utils.middleware')
    body = {
        'model': 'model',
        'metadata': {'shared': True},
        'tools': [{'function': {'spec': {'kept': [1]}}}],
        'files': [{'data': {'blob': [2]}}],
        'stream_options': {'include_usage': True, 'nested': {'items': [3]}},
        'custom_params': {'params': {'deep': [4]}},
        'messages': [{'role': 'user', 'content': 'original'}],
    }

    send = middleware._send_copy(body)

    send['tools'][0]['function']['spec']['kept'].append(9)
    send['files'][0]['data']['blob'].append(9)
    send['stream_options']['nested']['items'].append(9)
    send['custom_params']['params']['deep'].append(9)
    send['messages'][0]['content'] = 'changed'

    assert body['tools'][0]['function']['spec']['kept'] == [1]
    assert body['files'][0]['data']['blob'] == [2]
    assert body['stream_options']['nested']['items'] == [3]
    assert body['custom_params']['params']['deep'] == [4]
    assert body['messages'][0]['content'] == 'original'
    assert send['metadata'] is body['metadata']


def test_send_copy_keeps_metadata_shared():
    middleware = importlib.import_module('open_webui.utils.middleware')
    client = object()
    metadata = {'mcp_clients': [client], 'tools': {}}

    send = middleware._send_copy({'metadata': metadata, 'messages': []})

    assert send['metadata'] is metadata
    assert send['metadata']['mcp_clients'][0] is client


def test_commit_walk_reflects_mint_by_content_identity():
    middleware = importlib.import_module('open_webui.utils.middleware')
    prev_held = 'held-before'
    state = {'summary_message_content': 'held-after'}
    canonical = {
        'messages': [
            {'role': 'assistant', 'content': 'unrelated'},
            {'role': 'user', 'content': prev_held},
        ]
    }

    middleware._commit_walk_held(canonical, prev_held, state)

    assert canonical['messages'][1]['content'] is state['summary_message_content']
    assert canonical['messages'][0]['content'] == 'unrelated'


def test_commit_walk_restores_held_when_identity_not_unique(caplog):
    middleware = importlib.import_module('open_webui.utils.middleware')
    prev_held = 'held-before'
    state = {'summary_message_content': 'held-after'}
    canonical = {
        'messages': [
            {'role': 'user', 'content': prev_held},
            {'role': 'user', 'content': prev_held},
        ]
    }

    with caplog.at_level(logging.ERROR):
        middleware._commit_walk_held(canonical, prev_held, state)

    assert canonical['messages'][0]['content'] is prev_held
    assert canonical['messages'][1]['content'] is prev_held
    assert state['summary_message_content'] is prev_held
    assert 'restored previous held reference' in caplog.text


def test_commit_walk_restores_held_without_match(caplog):
    middleware = importlib.import_module('open_webui.utils.middleware')
    state = {'summary_message_content': 'held-after'}
    canonical = {'messages': [{'role': 'user', 'content': 'unrelated'}]}

    with caplog.at_level(logging.ERROR):
        middleware._commit_walk_held(canonical, 'held-before', state)

    assert canonical['messages'][0]['content'] == 'unrelated'
    assert state['summary_message_content'] == 'held-before'
    assert 'restored previous held reference' in caplog.text


def test_rollback_spares_same_name_user_tool():
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    user_tool = {
        'spec': {'name': refs.REF_EXEC_TOOL_NAME, 'parameters': {'type': 'object', 'properties': {}}},
        'callable': lambda: None,
    }
    registry = {refs.REF_EXEC_TOOL_NAME: user_tool}
    metadata = {'tools': {'reader': True}}
    state = {
        'summary_message_content': 'held-after',
        'externalized_refs': {'enable': True, 'registry': registry, 'metadata': metadata},
    }
    snapshot = {'held': 'held-before', 'reader': middleware._ABSENT, 'tools': metadata['tools']}

    middleware._rollback_ref_apply_state(state, snapshot)

    assert registry[refs.REF_EXEC_TOOL_NAME] is user_tool
    assert metadata['tools'] == {'reader': True}
    assert state['summary_message_content'] == 'held-before'


def test_rollback_restores_previous_owned_reader():
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    previous = {'spec': refs.REF_EXEC_FUNCTION_SPEC, 'callable': object()}
    registry = {refs.REF_EXEC_TOOL_NAME: {'spec': refs.REF_EXEC_FUNCTION_SPEC, 'callable': object()}}
    state = {
        'summary_message_content': 'held-after',
        'externalized_refs': {'enable': True, 'registry': registry, 'metadata': {}},
    }
    snapshot = {'held': 'held-before', 'reader': previous, 'tools': {'kept': True}}

    middleware._rollback_ref_apply_state(state, snapshot)

    assert registry[refs.REF_EXEC_TOOL_NAME] is previous
    assert state['externalized_refs']['metadata']['tools'] == {'kept': True}
    assert state['summary_message_content'] == 'held-before'


def _stream_leg_patches(monkeypatch, middleware, *, replies, tools, request_filter=None):
    current = {'replies': list(replies), 'sent': [], 'filter_inputs': []}

    def response(delta):
        async def chunks():
            yield f'data: {middleware.JSONCodec.dumps(delta)}\n\n'.encode()
            yield b'data: [DONE]\n\n'

        return StreamingResponse(chunks(), media_type='text/event-stream')

    def tool_response(call_id, name):
        return response(
            {
                'choices': [
                    {
                        'delta': {
                            'tool_calls': [
                                {
                                    'index': 0,
                                    'id': call_id,
                                    'type': 'function',
                                    'function': {'name': name, 'arguments': '{}'},
                                }
                            ]
                        },
                        'finish_reason': 'tool_calls',
                    }
                ]
            }
        )

    async def noop(*_args, **_kwargs):
        return None

    async def config_get(_key, default=None):
        return default

    async def rag(_template, context, prompt):
        return f'RAG[{context}]PROMPT[{prompt}]'

    async def generate_summary(*_args, **_kwargs):
        return 'FOLDED'

    async def generate(_request, candidate, _user, **_kwargs):
        current['sent'].append(copy.deepcopy(candidate))
        return current['replies'].pop(0)

    async def process_result(_request, _name, result, *_args):
        return result, [], []

    monkeypatch.setattr(middleware, 'generate_chat_completion', generate)
    monkeypatch.setattr(middleware, 'process_tool_result', process_result)
    monkeypatch.setattr(middleware, 'get_citation_source_from_tool_result', lambda tool_name, **_kwargs: [])
    monkeypatch.setattr(middleware, 'rag_template', rag)
    monkeypatch.setattr(middleware.Config, 'get', config_get)
    monkeypatch.setattr(middleware, 'terminal_event_handler', noop)
    monkeypatch.setattr(middleware, 'get_system_oauth_token', noop)
    monkeypatch.setattr(middleware, 'outlet_filter_handler', noop)
    monkeypatch.setattr(middleware, 'background_tasks_handler', noop)
    monkeypatch.setattr(middleware, 'clear_response_stream', noop)
    monkeypatch.setattr(middleware, 'save_response_stream', noop)
    monkeypatch.setattr(middleware, 'publish_chat_finished_event', noop)
    async def upsert(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(middleware.Chats, 'upsert_message_to_chat_by_id_and_message_id', upsert)
    async def title(_chat_id):
        return ''

    monkeypatch.setattr(middleware.Chats, 'get_chat_title_by_id', title)
    monkeypatch.setattr(middleware, 'ENABLE_RESPONSES_API_STATEFUL', False)
    monkeypatch.setattr(middleware, 'RAG_SYSTEM_CONTEXT', False)
    monkeypatch.setattr(compaction, '_generate_summary', generate_summary)
    socket_main = importlib.import_module('open_webui.socket.main')
    monkeypatch.setattr(socket_main, 'get_event_emitter', noop)
    if request_filter is not None:
        async def process(**kwargs):
            if kwargs.get('filter_type') == 'request' and kwargs.get('filter_functions'):
                current['filter_inputs'].append(copy.deepcopy(kwargs['form_data']['messages']))
                return await kwargs['filter_functions'][0](body=kwargs['form_data']), {}
            return kwargs['form_data'], {}

        async def get_filters(*_args, **_kwargs):
            return [request_filter]

        monkeypatch.setattr(middleware, 'process_filter_functions', process)
        monkeypatch.setattr(middleware, 'get_filter_functions', get_filters)
        monkeypatch.setattr(middleware, 'ENABLE_PLUGINS', True)

    current['tool_response'] = tool_response
    current['text_response'] = lambda text: response(
        {'choices': [{'delta': {'content': text}, 'finish_reason': 'stop'}]}
    )
    return current


def _stream_leg_ctx(metadata, *, compaction_enabled, refs=None, estimate=None):
    compaction_state = {
        'config': {
            'enable': compaction_enabled,
            'token_threshold': 100,
            'token_cap': 100,
            'retention_percentage': 40,
            'prompt_template': '',
            'soft_trigger_ratio': 0,
            'transient_patterns': (),
            'externalized_refs_enable': refs is not None,
            'externalized_refs_token_threshold': 1000,
        },
        'checkpoint_messages': [
            {'id': 'user', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': ''},
        ],
        'externalized_refs': refs or {'enable': False},
    }
    form_data = {
        'model': 'model',
        'stream': True,
        'messages': [{'role': 'user', 'content': 'start'}],
        'metadata': metadata,
    }
    ctx = {
        'request': SimpleNamespace(
            state=SimpleNamespace(max_tool_call_iterations=5),
            app=SimpleNamespace(state=SimpleNamespace(redis=None, MODELS={})),
        ),
        'form_data': form_data,
        'user': SimpleNamespace(),
        'model': {
            'id': 'model',
            'info': {'meta': {'capabilities': {'citations': False, 'file_context': False}}},
        },
        'metadata': metadata,
        'events': [],
        'tasks': {},
        'event_emitter': lambda *a, **k: _noop_async(),
        'event_caller': None,
        'compaction_state': compaction_state,
    }
    ctx['compaction_state']['canonical_body'] = form_data
    ctx['compaction_state']['last_send_body'] = form_data
    return ctx


async def _noop_async():
    return None


def _plain_tools(view_result='result-a', fetch_result='result-b'):
    tools_module = importlib.import_module('open_webui.utils.tools')

    def spec(name):
        return {'name': name, 'parameters': {'type': 'object', 'properties': {}}}

    async def view_file():
        return view_result

    async def fetch_url():
        return fetch_result

    return {
        name: {'spec': spec(name), 'callable': callable_}
        for name, callable_ in (
            (
                'view_file',
                asyncio.run(
                    tools_module.get_async_tool_function_and_apply_extra_params(view_file, {})
                ),
            ),
            (
                'fetch_url',
                asyncio.run(
                    tools_module.get_async_tool_function_and_apply_extra_params(fetch_url, {})
                ),
            ),
        )
    }


def _marker_count(messages):
    return sum(
        message.get('content', '').count('|MARK|')
        for message in messages
        if isinstance(message.get('content'), str)
    )


def test_continuation_rounds_keep_canonical_clean_and_marker_once(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')

    async def marker_filter(body):
        messages = body['messages']
        messages[-1]['content'] = messages[-1].get('content', '') + '|MARK|'
        return body

    tools = _plain_tools()
    current = _stream_leg_patches(
        monkeypatch,
        middleware,
        replies=[],
        tools=tools,
        request_filter=marker_filter,
    )
    current['replies'] = [
        current['tool_response']('call-b', 'fetch_url'),
        current['text_response']('done'),
    ]
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    ctx = _stream_leg_ctx(metadata, compaction_enabled=False)

    asyncio.run(
        middleware.streaming_chat_response_handler(
            current['tool_response']('call-a', 'view_file'),
            ctx,
        )
    )

    assert len(current['sent']) == 2
    for sent in current['sent']:
        assert _marker_count(sent['messages']) == 1
    assert len(current['filter_inputs']) == 2
    for messages in current['filter_inputs']:
        assert _marker_count(messages) == 0
    assert _marker_count(ctx['compaction_state']['canonical_body']['messages']) == 0


@pytest.mark.parametrize('advanced', [True, False])
def test_filtered_tool_removal_controls_catalog_and_filter_sees_raw(monkeypatch, advanced):
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    big = 'raw output with several words per line\n' * 300
    entry = refs.make_ref_entry(big, kind='tool')
    assert entry is not None
    seen = []

    async def deleting_filter(body):
        contents = [
            message.get('content') for message in body['messages'] if isinstance(message.get('content'), str)
        ]
        seen.append(list(contents))
        kept = [message for message in body['messages'] if message.get('role') != 'tool']
        return {**body, 'messages': kept}

    tools = _plain_tools(view_result=big, fetch_result='result-b')
    current = _stream_leg_patches(
        monkeypatch,
        middleware,
        replies=[],
        tools=tools,
        request_filter=deleting_filter,
    )
    current['replies'] = [current['text_response']('done')]

    def estimate(body):
        return (
            20
            if any(
                isinstance(message.get('content'), str)
                and message['content'].startswith('<auto_compaction_context>')
                for message in body['messages']
            )
            else 120
        )

    monkeypatch.setattr(compaction, 'estimate_provider_tokens', estimate)
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    registry = {}
    state_refs = {
        'enable': True,
        'native': True,
        'threshold': 1000,
        'registry': registry,
        'metadata': metadata,
    }
    ctx = _stream_leg_ctx(metadata, compaction_enabled=advanced, refs=state_refs)

    asyncio.run(
        middleware.streaming_chat_response_handler(
            current['tool_response']('call-a', 'view_file'),
            ctx,
        )
    )

    assert seen and any(big in contents for contents in seen)
    assert not any(entry.ref in contents for contents in seen)
    if advanced:
        assert ctx['compaction_state'].get('compacted') is True
        assert refs.REF_EXEC_TOOL_NAME in registry
        catalog = registry[refs.REF_EXEC_TOOL_NAME]['callable'].__externalized_ref_catalog__
        assert entry.ref not in catalog
    else:
        assert refs.REF_EXEC_TOOL_NAME not in registry
    for sent in current['sent']:
        assert all(message.get('role') != 'tool' for message in sent['messages'])


def test_zero_filter_continuation_bodies_match_fixed_shape(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    tools = _plain_tools()
    current = _stream_leg_patches(monkeypatch, middleware, replies=[], tools=tools)
    current['replies'] = [
        current['tool_response']('call-b', 'fetch_url'),
        current['text_response']('done'),
    ]
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    ctx = _stream_leg_ctx(metadata, compaction_enabled=False)

    asyncio.run(
        middleware.streaming_chat_response_handler(
            current['tool_response']('call-a', 'view_file'),
            ctx,
        )
    )

    first_round = [
        {'role': 'user', 'content': 'start'},
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {
                    'id': 'call-a',
                    'type': 'function',
                    'function': {'name': 'view_file', 'arguments': '{}'},
                }
            ],
        },
        {'role': 'tool', 'tool_call_id': 'call-a', 'content': 'result-a'},
    ]
    second_round = [
        *first_round,
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {
                    'id': 'call-b',
                    'type': 'function',
                    'function': {'name': 'fetch_url', 'arguments': '{}'},
                }
            ],
        },
        {'role': 'tool', 'tool_call_id': 'call-b', 'content': 'result-b'},
    ]
    assert len(current['sent']) == 2
    assert current['sent'][0]['messages'] == first_round
    assert current['sent'][1]['messages'] == second_round
    for sent in current['sent']:
        assert sent['model'] == 'model'
        assert sent['stream'] is True
        assert 'previous_response_id' not in sent
        assert sent['metadata'] == metadata


def test_tool_messages_param_sees_sent_view(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    seen_views = []

    async def marker_filter(body):
        messages = body['messages']
        messages[-1]['content'] = messages[-1].get('content', '') + '|MARK|'
        return body

    tools_module = importlib.import_module('open_webui.utils.tools')

    def spec(name):
        return {'name': name, 'parameters': {'type': 'object', 'properties': {}}}

    async def view_file():
        return 'result-a'

    async def fetch_url(__messages__=None):
        if __messages__ is not None:
            seen_views.append(__messages__)
        return 'result-b'

    tools = {
        'view_file': {
            'spec': spec('view_file'),
            'callable': asyncio.run(
                tools_module.get_async_tool_function_and_apply_extra_params(view_file, {})
            ),
        },
        'fetch_url': {
            'spec': spec('fetch_url'),
            'callable': asyncio.run(
                tools_module.get_async_tool_function_and_apply_extra_params(fetch_url, {})
            ),
        },
    }
    current = _stream_leg_patches(
        monkeypatch,
        middleware,
        replies=[],
        tools=tools,
        request_filter=marker_filter,
    )
    current['replies'] = [
        current['tool_response']('call-b', 'fetch_url'),
        current['text_response']('done'),
    ]
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    ctx = _stream_leg_ctx(metadata, compaction_enabled=False)

    asyncio.run(
        middleware.streaming_chat_response_handler(
            current['tool_response']('call-a', 'view_file'),
            ctx,
        )
    )

    assert seen_views
    last_tool_contents = [
        message.get('content')
        for message in seen_views[-1]
        if message.get('role') == 'tool' and isinstance(message.get('content'), str)
    ]
    assert 'result-a|MARK|' in last_tool_contents


def _owned_reader_registry(refs, text):
    registry = {}
    body = {
        'stream': True,
        'messages': [{'role': 'tool', 'tool_call_id': 't0', 'content': text}],
    }
    installed = asyncio.run(
        refs.externalize_refs(
            body,
            registry,
            native=True,
            threshold_tokens=100,
            count_tokens=lambda value: max(1, len(value) // 20),
        )
    )
    assert installed
    return registry


def test_stateful_continuation_reattaches_reader_schema():
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    registry = _owned_reader_registry(refs, 'r' * 2400)
    reader = registry[refs.REF_EXEC_TOOL_NAME]['callable']
    catalog = reader.__externalized_ref_catalog__
    entries_before = dict(catalog)
    body = {
        'stream': True,
        'previous_response_id': 'resp_1',
        'messages': [{'role': 'tool', 'tool_call_id': 't1', 'content': 'tiny'}],
    }

    result = asyncio.run(
        refs.externalize_refs(
            body,
            registry,
            native=True,
            threshold_tokens=5,
            count_tokens=lambda value: max(1, len(value) // 10),
        )
    )

    assert result is True
    assert registry[refs.REF_EXEC_TOOL_NAME]['callable'] is reader
    assert reader.__externalized_ref_catalog__ is catalog
    assert catalog == entries_before
    assert body['messages'][0]['content'] == 'tiny'
    assert any(
        isinstance(tool, dict) and (tool.get('function') or {}).get('name') == refs.REF_EXEC_TOOL_NAME
        for tool in body['tools']
    )


def test_stateful_reattach_requires_native():
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    registry = _owned_reader_registry(refs, 'r' * 2400)
    body = {
        'stream': True,
        'previous_response_id': 'resp_1',
        'messages': [{'role': 'tool', 'tool_call_id': 't1', 'content': 'tiny'}],
    }

    result = asyncio.run(
        refs.externalize_refs(
            body,
            registry,
            native=False,
            threshold_tokens=5,
            count_tokens=lambda value: max(1, len(value) // 10),
        )
    )

    assert result is False
    assert 'tools' not in body


def test_stateful_reattach_requires_selectable_reader():
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    registry = _owned_reader_registry(refs, 'r' * 2400)
    body = {
        'stream': True,
        'tool_choice': 'none',
        'previous_response_id': 'resp_1',
        'messages': [{'role': 'tool', 'tool_call_id': 't1', 'content': 'tiny'}],
    }

    result = asyncio.run(
        refs.externalize_refs(
            body,
            registry,
            native=True,
            threshold_tokens=5,
            count_tokens=lambda value: max(1, len(value) // 10),
        )
    )

    assert result is False
    assert 'tools' not in body


def test_stateful_reattach_rejects_foreign_reader_slot():
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    registry = {
        refs.REF_EXEC_TOOL_NAME: {
            'spec': {'name': refs.REF_EXEC_TOOL_NAME, 'parameters': {'type': 'object', 'properties': {}}},
            'callable': lambda: None,
        }
    }
    body = {
        'stream': True,
        'previous_response_id': 'resp_1',
        'messages': [{'role': 'tool', 'tool_call_id': 't1', 'content': 'tiny'}],
    }

    result = asyncio.run(
        refs.externalize_refs(
            body,
            registry,
            native=True,
            threshold_tokens=5,
            count_tokens=lambda value: max(1, len(value) // 10),
        )
    )

    assert result is False
    assert 'tools' not in body


def test_stateful_two_round_continuation_keeps_reader(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    big = 'stateful big output ' + 's' * 600
    entry = refs.make_ref_entry(big, kind='tool')
    assert entry is not None
    tools = _plain_tools(view_result=big, fetch_result='small result')

    def responses_round(call_id, name, response_id):
        payload = {
            'type': 'response.completed',
            'response': {
                'id': response_id,
                'output': [
                    {
                        'type': 'function_call',
                        'id': f'fc-{call_id}',
                        'call_id': call_id,
                        'name': name,
                        'arguments': '{}',
                        'status': 'completed',
                    }
                ],
            },
        }

        async def chunks():
            yield f'data: {middleware.JSONCodec.dumps(payload)}\n\n'.encode()
            yield b'data: [DONE]\n\n'

        return StreamingResponse(chunks(), media_type='text/event-stream')

    def final_completed():
        payload = {'type': 'response.completed', 'response': {'id': 'resp-final', 'output': []}}

        async def chunks():
            yield f'data: {middleware.JSONCodec.dumps(payload)}\n\n'.encode()
            yield b'data: [DONE]\n\n'

        return StreamingResponse(chunks(), media_type='text/event-stream')

    current = _stream_leg_patches(monkeypatch, middleware, replies=[], tools=tools)
    monkeypatch.setattr(middleware, 'ENABLE_RESPONSES_API_STATEFUL', True)
    current['replies'] = [
        responses_round('call-b', 'fetch_url', 'resp-2'),
        final_completed(),
    ]
    registry = dict(tools)
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': registry,
    }
    state_refs = {
        'enable': True,
        'native': True,
        'threshold': 100,
        'registry': registry,
        'metadata': metadata,
    }
    ctx = _stream_leg_ctx(metadata, compaction_enabled=False, refs=state_refs)
    # Only the large round-1 result may be classified; tiny results must stay raw.
    ctx['compaction_state']['config']['externalized_refs_token_threshold'] = 10_000

    asyncio.run(
        middleware.streaming_chat_response_handler(
            responses_round('call-a', 'view_file', 'resp-1'),
            ctx,
        )
    )

    assert len(current['sent']) == 2
    round1, round2 = current['sent']
    assert round1.get('previous_response_id') == 'resp-1'
    assert any(
        message.get('role') == 'tool'
        and isinstance(message.get('content'), str)
        and '<auto_compact_ref_truncated>' in message['content']
        and entry.ref in message['content']
        for message in round1['messages']
    )
    assert (
        len(
            [
                tool
                for tool in round1.get('tools') or []
                if (tool.get('function') or {}).get('name') == refs.REF_EXEC_TOOL_NAME
            ]
        )
        == 1
    )
    reader = registry[refs.REF_EXEC_TOOL_NAME]['callable']
    assert entry.ref in asyncio.run(reader('ls tool')).splitlines()
    catalog = reader.__externalized_ref_catalog__
    entries_after_round1 = dict(catalog)

    assert round2.get('previous_response_id') == 'resp-2'
    assert any(
        message.get('role') == 'tool' and message.get('content') == 'small result'
        for message in round2['messages']
    )
    assert (
        len(
            [
                tool
                for tool in round2.get('tools') or []
                if (tool.get('function') or {}).get('name') == refs.REF_EXEC_TOOL_NAME
            ]
        )
        == 1
    )
    assert registry[refs.REF_EXEC_TOOL_NAME]['callable'] is reader
    assert reader.__externalized_ref_catalog__ is catalog
    assert catalog == entries_after_round1


def test_payload_apply_before_drain_executes_reader(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    big = 'approved reader source ' * 600
    entry = refs.make_ref_entry(big, kind='tool')
    assert entry is not None
    stored = {
        'output': [
            {
                'type': 'function_call',
                'call_id': 'call-1',
                'name': refs.REF_EXEC_TOOL_NAME,
                'arguments': middleware.JSONCodec.dumps({'command': f'wc -c {entry.ref}'}),
                'status': 'queued',
                'approved': True,
            }
        ]
    }

    async def process_result(_request, _name, result, *_args):
        return result, [], []

    async def noop(*_args, **_kwargs):
        return None

    _payload_leg_patches(monkeypatch, middleware, refs_runtime=(True, 1000))
    _payload_leg_drain(monkeypatch, middleware, stored, stub_execute=False)
    monkeypatch.setattr(middleware, 'process_tool_result', process_result)
    monkeypatch.setattr(middleware, 'terminal_event_handler', noop)
    messages = [
        {'role': 'user', 'content': 'question'},
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {'id': 'old-call', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}
            ],
        },
        {'role': 'tool', 'tool_call_id': 'old-call', 'content': big},
    ]

    send, metadata, _events, state = asyncio.run(_run_payload_leg(middleware, messages))

    expected = str(len(big.encode('utf-8')))
    assert state['canonical_body']['messages'][-1]['content'] == expected
    assert send['messages'][-1]['content'] == expected
    reader = metadata['tools'][refs.REF_EXEC_TOOL_NAME]['callable']
    assert reader.__externalized_ref_catalog__[entry.ref].text == big


def test_round0_non_native_continuation_sends_no_refs(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    big = 'non native payload line\n' * 400
    tools = _plain_tools(view_result=big)
    current = _stream_leg_patches(monkeypatch, middleware, replies=[], tools=tools)
    current['replies'] = [current['text_response']('done')]
    registry = dict(tools)
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': registry,
    }
    state_refs = {
        'enable': True,
        'native': False,
        'threshold': 1000,
        'registry': registry,
        'metadata': metadata,
    }
    ctx = _stream_leg_ctx(metadata, compaction_enabled=False, refs=state_refs)

    asyncio.run(
        middleware.streaming_chat_response_handler(
            current['tool_response']('call-a', 'view_file'),
            ctx,
        )
    )

    sent = current['sent'][0]
    assert any(
        message.get('role') == 'tool' and message.get('content') is big for message in sent['messages']
    )
    assert not any(
        isinstance(message.get('content'), str) and message.get('content').startswith('tool:')
        for message in sent['messages']
    )
    assert not any(
        isinstance(tool, dict) and (tool.get('function') or {}).get('name') == refs.REF_EXEC_TOOL_NAME
        for tool in sent.get('tools') or []
    )
    seeded = ctx['compaction_state'].get('tool_ref_entries') or ()
    assert all(getattr(item, 'text', None) is not big for item in seeded)


def test_projected_snapshot_prevents_continuation_compaction(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    big = 'projected payload line\n' * 350
    tools = _plain_tools(view_result=big)
    current = _stream_leg_patches(monkeypatch, middleware, replies=[], tools=tools)
    current['replies'] = [current['text_response']('done')]

    def content_chars(messages):
        return sum(
            len(message.get('content')) if isinstance(message.get('content'), str) else 0
            for message in messages
        )

    projected, _entries = asyncio.run(
        refs.capture_tool_ref_projections(
            [{'role': 'tool', 'tool_call_id': 'call-a', 'content': big}],
            threshold_tokens=1000,
            count_tokens=compaction.estimate_text_tokens,
        )
    )
    overhead = len('start')  # the leading user message carried by every round
    # The compaction threshold must sit strictly between the preview the model
    # actually receives and the raw result, so dropping the projected snapshot
    # (or an oversized preview) trips compaction again.
    compaction_limit = (overhead + content_chars(projected) + overhead + content_chars([{'content': big}])) // 2
    assert overhead + content_chars(projected) < compaction_limit < overhead + len(big)

    def estimate(body):
        return content_chars(body['messages'])

    monkeypatch.setattr(compaction, 'estimate_provider_tokens', estimate)
    registry = dict(tools)
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': registry,
    }
    state_refs = {
        'enable': True,
        'native': True,
        'threshold': 1000,
        'registry': registry,
        'metadata': metadata,
    }
    ctx = _stream_leg_ctx(metadata, compaction_enabled=True, refs=state_refs)
    ctx['compaction_state']['config']['token_threshold'] = compaction_limit
    ctx['compaction_state']['config']['token_cap'] = compaction_limit

    asyncio.run(
        middleware.streaming_chat_response_handler(
            current['tool_response']('call-a', 'view_file'),
            ctx,
        )
    )

    sent = current['sent'][0]
    assert len(sent['messages']) == 3
    assert not any(
        '<auto_compaction_context>' in message.get('content', '') for message in sent['messages']
    )
    tool_messages = [message for message in sent['messages'] if message.get('role') == 'tool']
    assert len(tool_messages) == 1
    assert tool_messages[0]['content'] == projected[0]['content']
    assert refs.REF_EXEC_TOOL_NAME in registry


def test_drain_gate_uses_assistant_message_id_with_message_fallback(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    _payload_leg_patches(monkeypatch, middleware, refs_runtime=(False, 1000))

    async def fail_get_message(*_args, **_kwargs):
        raise AssertionError('drain must not read the DB without assistant_message_id')

    _payload_leg_drain(monkeypatch, middleware, _approved_stored())
    monkeypatch.setattr(middleware.Chats, 'get_message_by_id_and_message_id', fail_get_message)

    _send, _metadata, _events, state = asyncio.run(
        _run_payload_leg(middleware, [{'role': 'user', 'content': 'q'}], assistant_message_id=None)
    )

    assert state.get('last_send_body') is not None

    seen_ids = []

    async def get_message(_chat_id, message_id):
        seen_ids.append(message_id)
        return _approved_stored()

    monkeypatch.setattr(middleware.Chats, 'get_message_by_id_and_message_id', get_message)

    _send2, _metadata2, _events2, state2 = asyncio.run(
        _run_payload_leg(middleware, [{'role': 'user', 'content': 'q'}], message_id=None)
    )

    assert seen_ids == ['assistant']
    assert state2['canonical_body']['messages'][-1]['role'] == 'tool'


def test_same_name_user_tool_survives_approved_round_trip(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    refs = importlib.import_module('open_webui.utils.externalized_refs')
    user_tool = {
        'spec': {'name': refs.REF_EXEC_TOOL_NAME, 'parameters': {'type': 'object', 'properties': {}}},
        'callable': lambda: None,
    }

    async def get_tools(*_args, **_kwargs):
        return {refs.REF_EXEC_TOOL_NAME: user_tool}

    monkeypatch.setattr(middleware, 'get_tools', get_tools)
    monkeypatch.setattr(middleware, 'ENABLE_PLUGINS', True)

    async def get_filters(*_args, **_kwargs):
        return []

    monkeypatch.setattr(middleware, 'get_filter_functions', get_filters)
    _payload_leg_patches(monkeypatch, middleware, refs_runtime=(True, 2))
    _payload_leg_drain(monkeypatch, middleware, _approved_stored())

    _send, metadata, _events, state = asyncio.run(
        _run_payload_leg(middleware, [{'role': 'user', 'content': 'q'}], form_tool_ids=['my-tool'])
    )

    registry = state['externalized_refs']['registry']
    assert registry[refs.REF_EXEC_TOOL_NAME] is user_tool
    assert metadata['tools'] is registry


def _run_summary_dispatch(monkeypatch, *, request_direct, direct_model_id, target_model_id, state_metadata, message_id):
    chat_module = importlib.import_module('open_webui.utils.chat')
    captured = {}

    async def fake_generate(request, form_data=None, user=None, **kwargs):
        captured['metadata'] = copy.deepcopy(form_data['metadata'])
        captured['metadata_ref'] = form_data['metadata']
        captured['kwargs'] = dict(kwargs)
        return {'choices': [{'message': {'content': 'summary'}}]}

    async def get_many(*_keys):
        return {'chat.context_compaction.model': target_model_id}

    async def pt(prompt, _user):
        return prompt

    monkeypatch.setattr(chat_module, 'generate_chat_completion', fake_generate)
    monkeypatch.setattr(compaction.Config, 'get_many', staticmethod(get_many))
    monkeypatch.setattr(compaction, 'prompt_template', pt)

    request = SimpleNamespace(
        state=SimpleNamespace(
            direct=request_direct,
            model={'id': direct_model_id} if direct_model_id else None,
            metadata=dict(state_metadata),
        ),
        scope={'type': 'http', 'state': {}},
        receive=None,
    )
    models = {
        'direct-model': {'id': 'direct-model', 'info': {'params': {'max_tokens': 500}}},
        'server-model': {'id': 'server-model', 'info': {}},
    }
    summary = asyncio.run(
        compaction._generate_summary(
            request,
            None,
            'direct-model',
            models,
            [],
            [],
            None,
            '',
            (),
            message_id=message_id,
        )
    )
    assert summary == 'summary'
    return captured, request


def test_direct_summary_dispatch_carries_message_id(monkeypatch):
    captured, request = _run_summary_dispatch(
        monkeypatch,
        request_direct=True,
        direct_model_id='direct-model',
        target_model_id='direct-model',
        state_metadata={'chat_id': 'chat', 'session_id': 'sess-1', 'user_id': 'user-1'},
        message_id='assistant-message',
    )
    socket_main = importlib.import_module('open_webui.socket.main')

    event_call = asyncio.run(socket_main.get_event_call(captured['metadata']))

    assert event_call is not None
    assert captured['metadata']['message_id'] == 'assistant-message'
    assert 'message_id' not in request.state.metadata
    assert captured['metadata_ref'] is not request.state.metadata


def test_direct_summary_dispatch_to_server_model_stays_emitter_inert(monkeypatch):
    captured, request = _run_summary_dispatch(
        monkeypatch,
        request_direct=True,
        direct_model_id='direct-model',
        target_model_id='server-model',
        state_metadata={
            'chat_id': 'chat',
            'session_id': 'sess-1',
            'user_id': 'user-1',
            'tools': {'legacy': {'spec': {'name': 'legacy'}}},
            'files': [{'id': 'file-1'}],
        },
        message_id='assistant-message',
    )

    assert 'message_id' not in captured['metadata']
    assert (
        set(captured['metadata'])
        - {'chat_id', 'session_id', 'user_id', 'tools', 'files'}
        == {'task'}
    )
    assert captured['metadata']['task'] == 'context_compaction'
    assert 'tools' not in captured['metadata']
    assert 'files' not in captured['metadata']
    assert 'message_id' not in request.state.metadata
    assert captured['metadata_ref'] is not request.state.metadata


def test_manual_compact_summary_strips_inherited_message_id(monkeypatch):
    captured, request = _run_summary_dispatch(
        monkeypatch,
        request_direct=True,
        direct_model_id='direct-model',
        target_model_id='server-model',
        state_metadata={
            'chat_id': 'chat',
            'session_id': 'sess-1',
            'user_id': 'user-1',
            'message_id': 'inherited-message',
        },
        message_id='assistant-message',
    )

    assert 'message_id' not in captured['metadata']
    assert set(captured['metadata']) - {
        'chat_id',
        'session_id',
        'user_id',
        'message_id',
    } == {'task'}
    assert request.state.metadata['message_id'] == 'inherited-message'
    assert captured['metadata_ref'] is not request.state.metadata


def test_direct_summary_dispatch_metadata_key_delta(monkeypatch):
    captured, _request = _run_summary_dispatch(
        monkeypatch,
        request_direct=True,
        direct_model_id='direct-model',
        target_model_id='direct-model',
        state_metadata={'chat_id': 'chat', 'session_id': 'sess-1', 'user_id': 'user-1'},
        message_id='assistant-message',
    )

    assert set(captured['metadata']) - {'chat_id', 'session_id', 'user_id'} == {'task', 'message_id'}
    assert captured['metadata']['task'] == 'context_compaction'


def test_durable_checkpoint_generation_receives_message_id(monkeypatch):
    history = [
        {'id': 'u1', 'role': 'user', 'content': 'old'},
        {'id': 'a1', 'role': 'assistant', 'content': 'answer'},
    ]
    seen = {}

    async def generate_summary(*_args, **kwargs):
        seen.update(kwargs)
        return 'summary'

    monkeypatch.setattr(compaction, '_generate_summary', generate_summary)

    summary, _meta, checkpoint_history, checkpoint_message_id = asyncio.run(
        compaction._generate_checkpoint(
            None,
            None,
            'model',
            {'model': {'id': 'model', 'info': {}}},
            {'chat_id': 'chat', 'message_id': 'assistant'},
            {'previous_summary': None},
            {'prompt_template': '', 'transient_patterns': ()},
            [],
            [],
            checkpoint_history=(history, 1),
        )
    )

    assert summary == 'summary'
    assert checkpoint_message_id == 'a1'
    assert checkpoint_history == (history, 1)
    assert seen['message_id'] == 'assistant'


def test_completed_turn_checkpoint_skips_nested_tip_guard():
    messages = _tool_round_tip_messages()
    tip = messages['a2']
    tip['output'][1]['contextSummary'] = 'SUM'
    branch = [messages['u1'], messages['a1'], messages['u2'], tip]
    config = {'prompt_template': '', 'transient_patterns': ()}
    writes = []
    socket_main = importlib.import_module('open_webui.socket.main')

    async def fail_summary(*_args, **_kwargs):
        raise AssertionError('nested-tip guard must skip summary generation')

    async def record_update(chat_id, message_id, summary, **_kwargs):
        writes.append((chat_id, message_id, summary))
        return True

    async def noop(*_args, **_kwargs):
        return None

    original_summary = compaction._generate_summary
    original_update = compaction.ChatMessages.update_context_summary
    original_emitter = socket_main.get_event_emitter
    try:
        compaction._generate_summary = fail_summary
        compaction.ChatMessages.update_context_summary = record_update
        socket_main.get_event_emitter = noop

        result = asyncio.run(
            compaction._completed_turn_checkpoint(
                None,
                None,
                branch,
                {'chat_id': 'chat', 'message_id': 'a2'},
                'model',
                {},
                config,
            )
        )
    finally:
        compaction._generate_summary = original_summary
        compaction.ChatMessages.update_context_summary = original_update
        socket_main.get_event_emitter = original_emitter

    assert result is None
    assert writes == []


def test_completed_turn_checkpoint_generates_without_nested_tip():
    messages = _tool_round_tip_messages()
    branch = [messages['u1'], messages['a1'], messages['u2'], messages['a2']]
    config = {'prompt_template': '', 'transient_patterns': ()}
    writes = []
    seen = {}
    socket_main = importlib.import_module('open_webui.socket.main')

    async def generate_summary(*_args, **kwargs):
        seen.update(kwargs)
        return 'summary'

    async def record_update(chat_id, message_id, summary, **_kwargs):
        writes.append((chat_id, message_id, summary))
        return True

    async def noop(*_args, **_kwargs):
        return None

    original_summary = compaction._generate_summary
    original_update = compaction.ChatMessages.update_context_summary
    original_emitter = socket_main.get_event_emitter
    try:
        compaction._generate_summary = generate_summary
        compaction.ChatMessages.update_context_summary = record_update
        socket_main.get_event_emitter = noop

        result = asyncio.run(
            compaction._completed_turn_checkpoint(
                None,
                None,
                branch,
                {'chat_id': 'chat', 'message_id': 'a2'},
                'model',
                {},
                config,
            )
        )
    finally:
        compaction._generate_summary = original_summary
        compaction.ChatMessages.update_context_summary = original_update
        socket_main.get_event_emitter = original_emitter

    assert result == 'summary'
    assert writes == [('chat', 'a2', 'summary')]
    assert seen['message_id'] == 'a2'


def _run_outlet(monkeypatch, middleware, messages_map, outlet_filter):
    emitted = []
    saved = []

    async def get_filters(*_args, **_kwargs):
        return [outlet_filter]

    async def process(**kwargs):
        if kwargs.get('filter_type') == 'outlet' and kwargs.get('filter_functions'):
            return await kwargs['filter_functions'][0](body=kwargs['form_data']), {}
        return kwargs['form_data'], {}

    async def pipeline(_request, outlet_data, _user, _models):
        return outlet_data

    async def get_messages(_chat_id):
        return messages_map

    async def upsert(_chat_id, _message_id, update, **_kwargs):
        saved.append(copy.deepcopy(update))
        return {}

    async def emit(event):
        emitted.append(copy.deepcopy(event))

    monkeypatch.setattr(middleware, 'ENABLE_PLUGINS', True)
    monkeypatch.setattr(middleware, 'get_filter_functions', get_filters)
    monkeypatch.setattr(middleware, 'process_filter_functions', process)
    monkeypatch.setattr(middleware, 'process_pipeline_outlet_filter', pipeline)
    monkeypatch.setattr(middleware, 'get_sorted_filters', lambda *_args, **_kwargs: [])
    monkeypatch.setattr(middleware.Chats, 'get_messages_map_by_chat_id', get_messages)
    monkeypatch.setattr(middleware.Chats, 'upsert_message_to_chat_by_id_and_message_id', upsert)

    request = SimpleNamespace(
        state=SimpleNamespace(),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={}, redis=None)),
    )
    ctx = {
        'request': request,
        'user': SimpleNamespace(),
        'model': {'id': 'model'},
        'metadata': {'chat_id': 'chat', 'message_id': 'a2', 'filter_ids': []},
        'event_emitter': emit,
        'event_caller': None,
    }
    asyncio.run(middleware.outlet_filter_handler(ctx))
    return emitted, saved


def _marked_tip_map():
    messages = _tool_round_tip_messages()
    messages['a2']['output'][1]['contextSummary'] = 'SUM'
    return messages


def _outlet_a2(emitted):
    for event in emitted:
        for message in event.get('data', {}).get('messages', []):
            if message.get('id') == 'a2':
                return message
    return None


def test_outlet_marker_restore_skips_untouched_output_persistence(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    messages_map = _marked_tip_map()

    async def outlet_filter(body):
        for message in body['messages']:
            if message.get('id') == 'a2':
                for item in message.get('output') or []:
                    item.pop('contextSummary', None)
                    item.pop('context_summary', None)
                message['content'] = 'edited'
        return body

    emitted, saved = _run_outlet(monkeypatch, middleware, messages_map, outlet_filter)

    assert len(saved) == 1
    assert 'output' not in saved[0]
    assert saved[0]['content'] == 'edited'
    emitted_a2 = _outlet_a2(emitted)
    assert any(item.get('contextSummary') == 'SUM' for item in emitted_a2['output'])


def test_outlet_marker_only_change_never_persists(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    messages_map = _marked_tip_map()

    async def outlet_filter(body):
        for message in body['messages']:
            if message.get('id') == 'a2':
                for item in message.get('output') or []:
                    item.pop('contextSummary', None)
                    item.pop('context_summary', None)
        return body

    emitted, saved = _run_outlet(monkeypatch, middleware, messages_map, outlet_filter)

    assert saved == []
    emitted_a2 = _outlet_a2(emitted)
    assert any(item.get('contextSummary') == 'SUM' for item in emitted_a2['output'])


def test_outlet_inserted_output_item_blocks_restore(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    messages_map = _marked_tip_map()

    async def outlet_filter(body):
        for message in body['messages']:
            if message.get('id') == 'a2':
                for item in message.get('output') or []:
                    item.pop('contextSummary', None)
                    item.pop('context_summary', None)
                message['output'].append({'type': 'message', 'role': 'assistant', 'content': []})
        return body

    emitted, saved = _run_outlet(monkeypatch, middleware, messages_map, outlet_filter)

    assert saved and 'output' in saved[0]
    assert all('contextSummary' not in item for item in saved[0]['output'])
    emitted_a2 = _outlet_a2(emitted)
    assert all('contextSummary' not in item for item in emitted_a2['output'])


def test_outlet_swapped_message_items_block_restore(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    messages_map = {
        'u1': {'id': 'u1', 'parentId': None, 'role': 'user', 'content': 'find evidence'},
        'a1': {'id': 'a1', 'parentId': 'u1', 'role': 'assistant', 'content': 'answer'},
        'u2': {'id': 'u2', 'parentId': 'a1', 'role': 'user', 'content': 'continue'},
        'a2': {
            'id': 'a2',
            'parentId': 'u2',
            'role': 'assistant',
            'content': 'final',
            'output': [
                {
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': 'answer', 'annotations': []}],
                    'contextSummary': 'SUM',
                },
                {
                    'type': 'message',
                    'role': 'assistant',
                    'content': [
                        {'type': 'output_text', 'text': ' continued', 'annotations': [], 'logprobs': None}
                    ],
                },
            ],
        },
    }

    async def outlet_filter(body):
        for message in body['messages']:
            if message.get('id') == 'a2':
                output = message['output']
                message['output'] = [output[1], output[0]]
        return body

    emitted, saved = _run_outlet(monkeypatch, middleware, messages_map, outlet_filter)

    emitted_a2 = _outlet_a2(emitted)
    assert 'contextSummary' not in emitted_a2['output'][0]
    assert emitted_a2['output'][1].get('contextSummary') == 'SUM'
    assert saved
    assert 'contextSummary' not in saved[0]['output'][0]
    assert saved[0]['output'][1].get('contextSummary') == 'SUM'


def test_summary_generation_does_not_bypass_model_access(monkeypatch):
    captured, _request = _run_summary_dispatch(
        monkeypatch,
        request_direct=False,
        direct_model_id=None,
        target_model_id='server-model',
        state_metadata={'chat_id': 'chat', 'session_id': 'sess-1', 'user_id': 'user-1'},
        message_id=None,
    )

    assert set(captured['kwargs']) - {'form_data', 'user'} == {'bypass_system_prompt'}
    assert captured['kwargs']['bypass_system_prompt'] is True
    assert 'bypass_filter' not in captured['kwargs']


def _adoption_item(summary, item_id='cc_1'):
    return {
        'type': compaction.CONTEXT_COMPACTION_OUTPUT_TYPE,
        'id': item_id,
        'compaction_summary': summary,
    }


def test_adoption_output_item_is_invisible_to_provider_conversion():
    misc = importlib.import_module('open_webui.utils.misc')

    def tool_pair(call_id, name, result, image=None):
        call = {
            'type': 'function_call',
            'id': call_id,
            'call_id': call_id,
            'name': name,
            'arguments': '{}',
            'status': 'completed',
        }
        parts = [{'type': 'input_text', 'text': result}]
        if image is not None:
            parts.append({'type': 'input_image', 'image_url': image})
        return [
            call,
            {'type': 'function_call_output', 'id': f'{call_id}-out', 'call_id': call_id, 'output': parts},
        ]

    # The record must sit directly between the two batches: any other item
    # (e.g. code_interpreter) would flush batch A itself and mask a regression
    # where the record introduces an extra flush.
    for image in (None, 'data:image/png;base64,AAA'):
        batch_a = tool_pair('call-a', 'first', 'result-a', image)
        batch_b = tool_pair('call-b', 'second', 'result-b')
        base = [*batch_a, *batch_b]
        with_record = [*batch_a, _adoption_item('SUMMARY'), *batch_b]
        for flatten in (True, False):
            assert misc.convert_output_to_messages(base, flatten_tool_images=flatten) == (
                misc.convert_output_to_messages(with_record, flatten_tool_images=flatten)
            )


def test_code_interpreter_survives_adoption_output_item():
    misc = importlib.import_module('open_webui.utils.misc')

    converted = misc.convert_output_to_messages(
        [
            _adoption_item('SUMMARY'),
            {
                'type': 'open_webui:code_interpreter',
                'id': 'ci_1',
                'code': 'print(1)',
                'output': {'stdout': '1'},
            },
        ],
        raw=True,
    )
    assert any('<code_interpreter>' in message.get('content', '') for message in converted)


def test_adoption_record_decision_uses_compaction_state_only():
    held_state = {
        'summary': 'S1',
        'summary_message_content': '<auto_compaction_context>...</auto_compaction_context>',
    }

    assert compaction.context_compaction_adoption_record({'summary': 'S1'}, []) is None
    unheld = dict(held_state)
    unheld.pop('summary_message_content')
    assert compaction.context_compaction_adoption_record(unheld, []) is None
    empty_held = dict(held_state)
    empty_held['summary_message_content'] = ''
    assert compaction.context_compaction_adoption_record(empty_held, []) is None

    record = compaction.context_compaction_adoption_record(held_state, [])
    assert record['type'] == compaction.CONTEXT_COMPACTION_OUTPUT_TYPE
    assert record['compaction_summary'] == 'S1'
    assert isinstance(record['id'], str) and record['id']

    assert compaction.context_compaction_adoption_record(held_state, [_adoption_item('S1')]) is None
    switched = compaction.context_compaction_adoption_record(held_state, [_adoption_item('S0')])
    assert switched['compaction_summary'] == 'S1'

    branch = [
        {'id': 'a1', 'role': 'assistant', 'content': '', 'output': [_adoption_item('S0', 'cc_0')]},
        {'id': 'u1', 'role': 'user', 'content': 'next'},
    ]
    assert compaction.context_compaction_adoption_record(held_state, [], branch) is not None
    same_branch = copy.deepcopy(branch)
    same_branch[0]['output'][0]['compaction_summary'] = 'S1'
    assert compaction.context_compaction_adoption_record(held_state, [], same_branch) is None

    fallback_state = {
        'previous_summary': 'S2',
        'summary_message_content': 'held',
    }
    fallback = compaction.context_compaction_adoption_record(fallback_state, [], [])
    assert fallback['compaction_summary'] == 'S2'


def test_prefetch_finalize_notifies_phase_and_boundary(monkeypatch):
    events = []

    async def emitter(event):
        events.append(event['data'])

    async def save_checkpoint(*_args):
        return True

    monkeypatch.setattr(compaction, '_save_checkpoint', save_checkpoint)

    async def generation_ok():
        return 'SUM', {}, ([{'id': 'u2'}], 2), 'u2'

    async def run_ok():
        return await compaction._finalize_prefetch_checkpoint(asyncio.ensure_future(generation_ok()), 'chat', emitter)

    result = asyncio.run(run_ok())
    assert result[0] == 'SUM'
    assert events == [
        {
            'action': 'context_compaction',
            'description': 'Summary ready',
            'done': True,
            'phase': 'prefetch',
            'message_id': 'u2',
            'summary': 'SUM',
        }
    ]

    events.clear()

    async def save_fails(*_args):
        raise RuntimeError('nope')

    monkeypatch.setattr(compaction, '_save_checkpoint', save_fails)

    async def generation_sum():
        return 'SUM', {}, ([{'id': 'u2'}], 2), 'u2'

    async def run_fail():
        await compaction._finalize_prefetch_checkpoint(asyncio.ensure_future(generation_sum()), 'chat', emitter)

    with pytest.raises(RuntimeError):
        asyncio.run(run_fail())
    assert events == [
        {
            'action': 'context_compaction',
            'description': 'Context compaction failed',
            'done': True,
            'error': True,
            'phase': 'prefetch',
        }
    ]


def test_completed_turn_prefetch_notifies_only_when_generating(monkeypatch):
    socket_main = importlib.import_module('open_webui.socket.main')
    events = []

    async def emitter(event):
        events.append(event['data'])

    async def noop(*_args, **_kwargs):
        return None

    async def get_event_emitter(_metadata):
        return emitter

    async def generate_summary(*_args, **_kwargs):
        return 'TURN-SUMMARY'

    async def save_checkpoint(*_args):
        return True

    monkeypatch.setattr(compaction, '_generate_summary', generate_summary)
    monkeypatch.setattr(compaction, '_save_checkpoint', save_checkpoint)
    monkeypatch.setattr(socket_main, 'get_event_emitter', get_event_emitter)

    metadata = {'chat_id': 'chat', 'message_id': 'assistant'}
    config = _compaction_config()

    async def run_guard_empty():
        return await compaction._completed_turn_checkpoint(None, None, [], metadata, 'model', {}, config)

    assert asyncio.run(run_guard_empty()) is None
    assert events == []

    messages = [
        {'id': 'user', 'role': 'user', 'content': 'start'},
        {'id': 'assistant', 'role': 'assistant', 'content': ''},
    ]

    async def run_success():
        return await compaction._completed_turn_checkpoint(
            None, None, copy.deepcopy(messages), metadata, 'model', {}, config
        )

    assert asyncio.run(run_success()) == 'TURN-SUMMARY'
    assert events[0]['done'] is False and events[0]['phase'] == 'prefetch'
    assert events[1] == {
        'action': 'context_compaction',
        'description': 'Summary ready',
        'done': True,
        'phase': 'prefetch',
        'message_id': 'assistant',
        'summary': 'TURN-SUMMARY',
    }

    events.clear()

    async def generate_fails(*_args, **_kwargs):
        raise RuntimeError('model down')

    monkeypatch.setattr(compaction, '_generate_summary', generate_fails)

    async def run_failure():
        await compaction._completed_turn_checkpoint(None, None, copy.deepcopy(messages), metadata, 'model', {}, config)

    with pytest.raises(RuntimeError):
        asyncio.run(run_failure())
    assert events[-1]['done'] is True and events[-1]['error'] is True and events[-1]['phase'] == 'prefetch'


def test_blocking_compaction_success_carries_boundary(monkeypatch):
    socket_main = importlib.import_module('open_webui.socket.main')
    events = []

    async def emitter(event):
        events.append(event['data'])

    async def get_event_emitter(_metadata):
        return emitter

    history = [
        {'id': 'u1', 'role': 'user', 'content': 'old'},
        {'id': 'a1', 'role': 'assistant', 'content': 'answer'},
        {'id': 'u2', 'role': 'user', 'content': 'new', compaction._BOUNDARY_KEY: True},
    ]

    async def generate_checkpoint(*_args, **_kwargs):
        return 'BLOCK-SUM', {}, (history, 2), 'u2'

    async def save_checkpoint(*_args):
        return True

    monkeypatch.setattr(compaction, '_generate_checkpoint', generate_checkpoint)
    monkeypatch.setattr(compaction, '_save_checkpoint', save_checkpoint)
    monkeypatch.setattr(compaction, 'estimate_provider_tokens', _estimate_compacted_when_summarized)
    monkeypatch.setattr(socket_main, 'get_event_emitter', get_event_emitter)

    state = {
        'active_offset': 0,
        'checkpoint_messages': history,
        'checkpoint_history': (history, 2),
        'config': _compaction_config(),
    }
    metadata = {'chat_id': 'chat', 'message_id': 'assistant'}

    compacted = asyncio.run(
        compaction.compact_provider_payload(
            None, None, {'messages': copy.deepcopy(history)}, metadata, 'model', {}, state
        )
    )
    assert compacted['messages'][0]['content'].startswith('<auto_compaction_context>')
    assert events == [
        {
            'action': 'context_compaction',
            'description': 'Compacting context',
            'done': False,
        },
        {
            'action': 'context_compaction',
            'description': 'Context compacted',
            'done': True,
            'message_id': 'u2',
            'summary': 'BLOCK-SUM',
        },
    ]


def _adoption_stream_harness(monkeypatch, middleware, *, summaries, existing_message=None):  # noqa: C901
    current = {'sent': [], 'emitted': [], 'saved': [], 'summaries': list(summaries)}
    estimate_calls = {'n': 0}

    def response(delta):
        async def chunks():
            yield f'data: {middleware.JSONCodec.dumps(delta)}\n\n'.encode()
            yield b'data: [DONE]\n\n'

        return StreamingResponse(chunks(), media_type='text/event-stream')

    def tool_response(call_id, name):
        return response(
            {
                'choices': [
                    {
                        'delta': {
                            'tool_calls': [
                                {
                                    'index': 0,
                                    'id': call_id,
                                    'type': 'function',
                                    'function': {'name': name, 'arguments': '{}'},
                                }
                            ]
                        },
                        'finish_reason': 'tool_calls',
                    }
                ]
            }
        )

    async def generate_summary(*_args, **_kwargs):
        assert current['summaries'], 'summary generator exhausted'
        return current['summaries'].pop(0)

    async def generate(_request, candidate, _user, **_kwargs):
        current['sent'].append(copy.deepcopy(candidate))
        return current['replies'].pop(0)

    async def emitter(event):
        current['emitted'].append(copy.deepcopy(event))

    async def process_result(_request, _name, result, *_args):
        return result, [], []

    async def noop(*_args, **_kwargs):
        return None

    async def config_get(_key, default=None):
        return default

    async def save(_chat_id, _message_id, update, **_kwargs):
        current['saved'].append(copy.deepcopy(update))
        return update

    async def get_message(_chat_id, _message_id):
        return copy.deepcopy(existing_message) if existing_message is not None else None

    def estimate(_body, **_kwargs):
        # Odd calls estimate the un-compacted body (over threshold), even calls
        # the nested candidate (under threshold), so every continuation compacts.
        estimate_calls['n'] += 1
        return 120 if estimate_calls['n'] % 2 == 1 else 20

    async def view_file():
        return 'result-a'

    async def fetch_url():
        return 'result-b'

    tools = {
        name: {
            'spec': {'name': name, 'parameters': {'type': 'object', 'properties': {}}},
            'callable': function,
        }
        for name, function in (('view_file', view_file), ('fetch_url', fetch_url))
    }

    monkeypatch.setattr(middleware, 'generate_chat_completion', generate)
    monkeypatch.setattr(middleware, 'process_tool_result', process_result)
    monkeypatch.setattr(middleware, 'get_citation_source_from_tool_result', lambda tool_name, **_kwargs: [])
    monkeypatch.setattr(middleware, 'rag_template', lambda _t, _c, _p: 'RAG')
    monkeypatch.setattr(middleware.Config, 'get', config_get)
    monkeypatch.setattr(middleware, 'terminal_event_handler', noop)
    monkeypatch.setattr(middleware, 'get_system_oauth_token', noop)
    monkeypatch.setattr(middleware, 'outlet_filter_handler', noop)
    monkeypatch.setattr(middleware, 'background_tasks_handler', noop)
    monkeypatch.setattr(middleware, 'clear_response_stream', noop)
    monkeypatch.setattr(middleware, 'save_response_stream', noop)
    monkeypatch.setattr(middleware, 'publish_chat_finished_event', noop)
    monkeypatch.setattr(middleware, 'review_memory_after_turn', noop)
    monkeypatch.setattr(middleware.Chats, 'upsert_message_to_chat_by_id_and_message_id', save)
    monkeypatch.setattr(middleware.Chats, 'get_message_by_id_and_message_id', get_message)
    monkeypatch.setattr(middleware.Chats, 'get_chat_title_by_id', noop)
    monkeypatch.setattr(middleware, 'ENABLE_RESPONSES_API_STATEFUL', False)
    monkeypatch.setattr(middleware, 'RAG_SYSTEM_CONTEXT', False)
    monkeypatch.setattr(compaction, '_generate_summary', generate_summary)
    monkeypatch.setattr(compaction, 'estimate_provider_tokens', estimate)
    socket_main = importlib.import_module('open_webui.socket.main')
    monkeypatch.setattr(socket_main, 'get_event_emitter', noop)

    current['tool_response'] = tool_response
    current['text_response'] = lambda text: response(
        {'choices': [{'delta': {'content': text}, 'finish_reason': 'stop'}]}
    )
    current['emitter'] = emitter
    return current, tools


def _adoption_stream_ctx(tools, metadata, state=None):
    compaction_state = state or {
        'config': {
            'enable': True,
            'token_threshold': 100,
            'token_cap': 100,
            'retention_percentage': 40,
            'prompt_template': '',
            'soft_trigger_ratio': 0,
            'transient_patterns': (),
            'externalized_refs_enable': False,
            'externalized_refs_token_threshold': 10_000,
        },
        'checkpoint_messages': [
            {'id': 'user', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': ''},
        ],
        'externalized_refs': {'enable': False},
    }
    form_data = {
        'model': 'model',
        'stream': True,
        'messages': [{'role': 'user', 'content': 'start'}],
        'metadata': metadata,
    }
    ctx = {
        'request': SimpleNamespace(
            state=SimpleNamespace(max_tool_call_iterations=5),
            app=SimpleNamespace(state=SimpleNamespace(redis=None, MODELS={})),
        ),
        'form_data': form_data,
        'user': SimpleNamespace(role='admin'),
        'model': {
            'id': 'model',
            'info': {'meta': {'capabilities': {'citations': False, 'file_context': False}}},
        },
        'metadata': metadata,
        'events': [],
        'tasks': {},
        'event_emitter': None,
        'event_caller': None,
        'compaction_state': compaction_state,
    }
    ctx['compaction_state']['canonical_body'] = form_data
    ctx['compaction_state']['last_send_body'] = form_data
    return ctx


def _adoption_shape(output):
    return [
        (
            item.get('type'),
            item.get('call_id') or item.get('name'),
            item.get('compaction_summary'),
            item.get('contextSummary'),
        )
        for item in output
    ]


def test_tool_loop_records_each_adopted_summary_position(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    current, tools = _adoption_stream_harness(
        monkeypatch,
        middleware,
        summaries=['S1', 'S2', 'S2'],
    )
    current['replies'] = [
        current['tool_response']('call-b', 'fetch_url'),
        current['tool_response']('call-c', 'fetch_url'),
        current['text_response']('done'),
    ]
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    ctx = _adoption_stream_ctx(tools, metadata)
    ctx['event_emitter'] = current['emitter']

    asyncio.run(
        middleware.streaming_chat_response_handler(
            current['tool_response']('call-a', 'view_file'),
            ctx,
        )
    )

    final = current['saved'][-1]
    assert _adoption_shape(final['output']) == [
        ('function_call', 'call-a', None, 'S1'),
        ('function_call_output', 'call-a', None, None),
        (compaction.CONTEXT_COMPACTION_OUTPUT_TYPE, None, 'S1', None),
        ('function_call', 'call-b', None, 'S2'),
        ('function_call_output', 'call-b', None, None),
        (compaction.CONTEXT_COMPACTION_OUTPUT_TYPE, None, 'S2', None),
        ('function_call', 'call-c', None, 'S2'),
        ('function_call_output', 'call-c', None, None),
        ('message', None, None, None),
    ]

    # The first adoption record must ride an existing full-output notification
    # out to the UI before the next leg streams anything.
    marker_emits = [
        event['data']['output']
        for event in current['emitted']
        if event.get('type') == 'chat:completion'
        and isinstance(event.get('data', {}).get('output'), list)
        and _adoption_shape(event['data']['output'])
        == [
            ('function_call', 'call-a', None, 'S1'),
            ('function_call_output', 'call-a', None, None),
            (compaction.CONTEXT_COMPACTION_OUTPUT_TYPE, None, 'S1', None),
        ]
    ]
    assert marker_emits


def test_completion_save_carries_snapshot_without_provider_leak(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    current, tools = _adoption_stream_harness(monkeypatch, middleware, summaries=[])
    current['replies'] = [current['text_response']('done')]
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    snapshot = {
        'tokens': 60,
        'threshold': 100,
        'soft_threshold': 50,
        'source': 'estimated',
    }
    state = {
        'config': {'enable': False},
        'checkpoint_messages': [
            {'id': 'u1', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': ''},
        ],
        'externalized_refs': {'enable': False},
        'context_usage': dict(snapshot),
    }
    ctx = _adoption_stream_ctx(tools, metadata, state=state)
    ctx['event_emitter'] = current['emitter']

    asyncio.run(middleware.streaming_chat_response_handler(current['tool_response']('call-a', 'view_file'), ctx))

    final_saves = [update for update in current['saved'] if update.get('done') is True and 'output' in update]
    assert final_saves
    for update in final_saves:
        assert update['context_usage'] == snapshot
        assert 'pending_context_usage' not in update

    done_events = [
        event['data']
        for event in current['emitted']
        if event.get('type') == 'chat:completion' and event.get('data', {}).get('done')
    ]
    assert done_events and done_events[-1]['context_usage'] == snapshot

    assert current['sent'], 'provider bodies must be captured for the leak check'
    for sent in current['sent']:
        assert 'context_usage' not in sent
        assert 'pending_context_usage' not in sent
        for message in sent.get('messages', []):
            assert 'context_usage' not in message


def test_streamed_usage_replaces_snapshot_tokens_per_response(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    current, tools = _adoption_stream_harness(monkeypatch, middleware, summaries=[])

    def usage_response(text, usage):
        async def chunks():
            payload = {
                'choices': [{'delta': {'content': text}, 'finish_reason': 'stop'}],
                'usage': usage,
            }
            yield f'data: {middleware.JSONCodec.dumps(payload)}\n\n'.encode()
            yield b'data: [DONE]\n\n'

        return StreamingResponse(chunks(), media_type='text/event-stream')

    current['replies'] = [
        usage_response(
            'leg one',
            {
                'prompt_tokens': 70,
                'completion_tokens': 10,
                'total_tokens': 80,
                'cache_creation_input_tokens': 5,
                'cache_read_input_tokens': 5,
            },
        )
    ]
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    state = {
        'config': {
            'enable': True,
            'token_threshold': 100,
            'token_cap': 100,
            'retention_percentage': 40,
            'prompt_template': '',
            'soft_trigger_ratio': 0.5,
            'transient_patterns': (),
        },
        'checkpoint_messages': [
            {'id': 'u1', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': ''},
        ],
        'externalized_refs': {'enable': False},
        'context_usage': {
            'tokens': 60,
            'threshold': 100,
            'soft_threshold': 50,
            'source': 'estimated',
        },
    }
    ctx = _adoption_stream_ctx(tools, metadata, state=state)
    ctx['event_emitter'] = current['emitter']

    asyncio.run(middleware.streaming_chat_response_handler(current['replies'][0], ctx))

    # 70 + 5 + 5 + 10 = 90 cache-inclusive, above the naive total of 80.
    assert state['context_usage'] == {
        'tokens': 90,
        'threshold': 100,
        'soft_threshold': 50,
        'source': 'usage',
    }
    numeric_events = [
        event['data']['context_usage']
        for event in current['emitted']
        if event.get('type') == 'context_compaction' and 'context_usage' in event.get('data', {})
    ]
    assert numeric_events == [
        {
            'tokens': 90,
            'threshold': 100,
            'soft_threshold': 50,
            'source': 'usage',
        }
    ]


def _numeric_snapshots(emitted):
    return [
        event['data']['context_usage']
        for event in emitted
        if event.get('type') == 'context_compaction' and 'context_usage' in event.get('data', {})
    ]


def test_tool_loop_send_commit_adopts_pending_and_notifies(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    current, tools = _adoption_stream_harness(monkeypatch, middleware, summaries=['S1'])
    current['replies'] = [current['text_response']('done')]
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    state = {
        'config': _compaction_config(),
        'checkpoint_messages': [
            {'id': 'u1', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': ''},
        ],
        'externalized_refs': {'enable': False},
        'context_usage': {
            'tokens': 90,
            'threshold': 100,
            'soft_threshold': None,
            'source': 'usage',
        },
    }
    ctx = _adoption_stream_ctx(tools, metadata, state=state)
    ctx['event_emitter'] = current['emitter']

    asyncio.run(middleware.streaming_chat_response_handler(current['tool_response']('call-a', 'view_file'), ctx))

    assert len(current['sent']) == 1
    assert 'pending_context_usage' not in state
    # Estimate call 1 measures 120 (blocking), the adopted candidate lands at 20.
    assert state['context_usage'] == {
        'tokens': 20,
        'threshold': 100,
        'soft_threshold': None,
        'source': 'estimated',
    }
    assert _numeric_snapshots(current['emitted']) == [
        {
            'tokens': 20,
            'threshold': 100,
            'soft_threshold': None,
            'source': 'estimated',
        }
    ]
    for sent in current['sent']:
        assert 'context_usage' not in sent
        assert 'pending_context_usage' not in sent
        for message in sent.get('messages', []):
            assert 'context_usage' not in message


def test_tool_execution_observes_usage_receipt_before_next_send(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    current, tools = _adoption_stream_harness(monkeypatch, middleware, summaries=[])
    current['replies'] = [current['text_response']('done')]
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    state = {
        'config': _compaction_config(),
        'checkpoint_messages': [
            {'id': 'u1', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': ''},
        ],
        'externalized_refs': {'enable': False},
        'context_usage': {
            'tokens': 60,
            'threshold': 100,
            'soft_threshold': None,
            'source': 'estimated',
        },
    }
    ctx = _adoption_stream_ctx(tools, metadata, state=state)
    ctx['event_emitter'] = current['emitter']
    observed = {}

    async def inspect_tool():
        observed['state'] = copy.deepcopy(ctx['compaction_state']['context_usage'])
        observed['notifications'] = copy.deepcopy(_numeric_snapshots(current['emitted']))
        return 'tool result'

    tools['view_file']['callable'] = inspect_tool

    def usage_tool_response(call_id, name, usage):
        async def chunks():
            payload = {
                'choices': [
                    {
                        'delta': {
                            'tool_calls': [
                                {
                                    'index': 0,
                                    'id': call_id,
                                    'type': 'function',
                                    'function': {'name': name, 'arguments': '{}'},
                                }
                            ]
                        },
                        'finish_reason': 'tool_calls',
                    }
                ],
                'usage': usage,
            }
            yield f'data: {middleware.JSONCodec.dumps(payload)}\n\n'.encode()
            yield b'data: [DONE]\n\n'

        return StreamingResponse(chunks(), media_type='text/event-stream')

    first = usage_tool_response(
        'call-a',
        'view_file',
        {'prompt_tokens': 80, 'completion_tokens': 10, 'total_tokens': 90},
    )
    asyncio.run(middleware.streaming_chat_response_handler(first, ctx))

    assert observed['state'] == {
        'tokens': 90,
        'threshold': 100,
        'soft_threshold': None,
        'source': 'usage',
    }
    assert observed['notifications'] == [
        {
            'tokens': 90,
            'threshold': 100,
            'soft_threshold': None,
            'source': 'usage',
        }
    ]


def test_usage_receipt_is_saved_when_cancelled_before_stream_closes(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    current, tools = _adoption_stream_harness(monkeypatch, middleware, summaries=[])
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    state = {
        'config': _compaction_config(),
        'checkpoint_messages': [
            {'id': 'u1', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': ''},
        ],
        'externalized_refs': {'enable': False},
        'context_usage': {
            'tokens': 60,
            'threshold': 100,
            'soft_threshold': None,
            'source': 'estimated',
        },
    }
    ctx = _adoption_stream_ctx(tools, metadata, state=state)
    ctx['event_emitter'] = current['emitter']

    async def exercise():
        received = asyncio.Event()
        release = asyncio.Event()

        async def chunks():
            payload = {
                'choices': [{'delta': {'content': 'answer'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 80, 'completion_tokens': 10, 'total_tokens': 90},
            }
            yield f'data: {middleware.JSONCodec.dumps(payload)}\n\n'.encode()
            yield b'data: [DONE]\n\n'
            received.set()
            await release.wait()

        task = asyncio.create_task(
            middleware.streaming_chat_response_handler(
                StreamingResponse(chunks(), media_type='text/event-stream'), ctx
            )
        )
        try:
            await asyncio.wait_for(received.wait(), 2)
            return (
                copy.deepcopy(ctx['compaction_state']['context_usage']),
                copy.deepcopy(_numeric_snapshots(current['emitted'])),
            )
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            release.set()

    observed, notifications = asyncio.run(exercise())
    cancelled_saves = [update for update in current['saved'] if update.get('done') is True and 'output' in update]
    assert cancelled_saves
    measured = {
        'tokens': 90,
        'threshold': 100,
        'soft_threshold': None,
        'source': 'usage',
    }
    assert observed == measured
    assert measured in notifications
    assert cancelled_saves[-1]['context_usage'] == measured


def test_cancelled_unsent_compaction_keeps_last_response_measurement(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    current, tools = _adoption_stream_harness(monkeypatch, middleware, summaries=[])
    current['replies'] = []
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    state = {
        'config': _compaction_config(),
        'checkpoint_messages': [
            {'id': 'u1', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': ''},
        ],
        'externalized_refs': {'enable': False},
        'context_usage': {
            'tokens': 60,
            'threshold': 100,
            'soft_threshold': None,
            'source': 'estimated',
        },
    }
    ctx = _adoption_stream_ctx(tools, metadata, state=state)
    ctx['event_emitter'] = current['emitter']

    async def exercise():
        pending = asyncio.Event()

        async def hanging_summary(*_args, **_kwargs):
            pending.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(compaction, '_generate_summary', hanging_summary)

        def usage_tool_response(call_id, name, usage):
            async def chunks():
                payload = {
                    'choices': [
                        {
                            'delta': {
                                'tool_calls': [
                                    {
                                        'index': 0,
                                        'id': call_id,
                                        'type': 'function',
                                        'function': {'name': name, 'arguments': '{}'},
                                    }
                                ]
                            },
                            'finish_reason': 'tool_calls',
                        }
                    ],
                    'usage': usage,
                }
                yield f'data: {middleware.JSONCodec.dumps(payload)}\n\n'.encode()
                yield b'data: [DONE]\n\n'

            return StreamingResponse(chunks(), media_type='text/event-stream')

        first = usage_tool_response(
            'call-a',
            'view_file',
            {'prompt_tokens': 80, 'completion_tokens': 10, 'total_tokens': 90},
        )
        task = asyncio.create_task(middleware.streaming_chat_response_handler(first, ctx))
        try:
            await asyncio.wait_for(pending.wait(), 2)
            return copy.deepcopy(ctx['compaction_state']['context_usage'])
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    measured = asyncio.run(exercise())
    cancelled_saves = [update for update in current['saved'] if update.get('done') is True and 'output' in update]
    assert cancelled_saves
    assert current['sent'] == []
    assert _numeric_snapshots(current['emitted']) == [
        {
            'tokens': 90,
            'threshold': 100,
            'soft_threshold': None,
            'source': 'usage',
        }
    ]
    assert cancelled_saves[-1]['context_usage'] == {
        'tokens': 90,
        'threshold': 100,
        'soft_threshold': None,
        'source': 'usage',
    }
    # The unsent 120-token candidate stays pending and never becomes visible.
    assert ctx['compaction_state']['pending_context_usage']['tokens'] == 120
    assert measured == {
        'tokens': 90,
        'threshold': 100,
        'soft_threshold': None,
        'source': 'usage',
    }


def test_initial_send_commit_adopts_pending_and_notifies(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    events = []

    async def emitter(event):
        events.append(copy.deepcopy(event))

    async def get_event_emitter(_metadata):
        return emitter

    _payload_leg_patches(monkeypatch, middleware, refs_runtime=(False, 1000))
    _payload_leg_drain(monkeypatch, middleware, {'output': []})
    _seed_state_on_capture(monkeypatch, middleware, {'summary_message_content': '', 'config': _compaction_config()})
    monkeypatch.setattr(middleware, 'get_event_emitter', get_event_emitter)
    monkeypatch.setattr(compaction, 'estimate_provider_tokens', lambda *_args, **_kwargs: 40)

    send, _metadata, _events, state = asyncio.run(
        _run_payload_leg(middleware, [{'role': 'user', 'content': 'start'}])
    )

    assert state['last_send_body'] is send
    assert 'pending_context_usage' not in state
    assert state['context_usage'] == {
        'tokens': 40,
        'threshold': 100,
        'soft_threshold': None,
        'source': 'estimated',
    }
    assert _numeric_snapshots(events) == [
        {
            'tokens': 40,
            'threshold': 100,
            'soft_threshold': None,
            'source': 'estimated',
        }
    ]


def test_repause_publishes_no_measurement_and_keeps_stored_snapshot(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    events = []
    saves = []

    async def emitter(event):
        events.append(copy.deepcopy(event))

    async def get_event_emitter(_metadata):
        return emitter

    async def save(_chat_id, _message_id, update, **_kwargs):
        saves.append(copy.deepcopy(update))
        return update

    stored = {
        'context_usage': {
            'tokens': 90,
            'threshold': 100,
            'soft_threshold': None,
            'source': 'usage',
        },
        'output': [
            {
                'type': 'function_call',
                'call_id': 'call-1',
                'name': 'lookup',
                'arguments': '{}',
                'status': 'queued',
            }
        ],
    }

    _payload_leg_patches(monkeypatch, middleware, refs_runtime=(False, 1000))
    _payload_leg_drain(monkeypatch, middleware, stored)
    _seed_state_on_capture(monkeypatch, middleware, {'summary_message_content': '', 'config': _compaction_config()})
    monkeypatch.setattr(middleware, 'get_event_emitter', get_event_emitter)
    monkeypatch.setattr(middleware.Chats, 'upsert_message_to_chat_by_id_and_message_id', save)
    monkeypatch.setattr(compaction, 'estimate_provider_tokens', lambda *_args, **_kwargs: 40)

    _send, _metadata, _events, state = asyncio.run(
        _run_payload_leg(
            middleware,
            [{'role': 'user', 'content': 'start'}],
            params={'tool_approval_mode': 'ask'},
        )
    )

    assert state.get('paused') is True
    assert 'last_send_body' not in state
    assert _numeric_snapshots(events) == []
    pause_saves = [update for update in saves if update.get('done') is False]
    assert pause_saves
    for update in pause_saves:
        assert 'context_usage' not in update
        assert 'pending_context_usage' not in update
    merged = {**stored, **pause_saves[-1]}
    assert merged['context_usage'] == stored['context_usage']


def test_first_send_dedupes_against_branch_and_continues_existing_output(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    current, tools = _adoption_stream_harness(monkeypatch, middleware, summaries=[])
    current['replies'] = [current['text_response']('done')]
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    branch_messages = [
        {'id': 'u0', 'role': 'user', 'content': 'start'},
        {
            'id': 'a0',
            'role': 'assistant',
            'content': '',
            'output': [_adoption_item('S1', 'cc_0')],
        },
    ]
    state = {
        'config': {
            'enable': True,
            'token_threshold': 100,
            'token_cap': 100,
            'retention_percentage': 40,
            'prompt_template': '',
            'soft_trigger_ratio': 0,
            'transient_patterns': (),
        },
        'checkpoint_messages': branch_messages,
        'summary': 'S1',
        'summary_message_content': '<auto_compaction_context>S1</auto_compaction_context>',
        'externalized_refs': {'enable': False},
    }
    ctx = _adoption_stream_ctx(tools, metadata, state=state)
    ctx['event_emitter'] = current['emitter']

    asyncio.run(middleware.streaming_chat_response_handler(current['text_response']('done'), ctx))

    assert _adoption_shape(current['saved'][-1]['output']) == [('message', None, None, None)]

    current2, tools2 = _adoption_stream_harness(
        monkeypatch,
        middleware,
        summaries=[],
        existing_message={
            'id': 'assistant',
            'role': 'assistant',
            'content': 'existing',
            'output': [
                {
                    'type': 'message',
                    'id': 'msg_0',
                    'status': 'completed',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': 'existing'}],
                }
            ],
        },
    )
    current2['replies'] = [current2['text_response']('more')]
    metadata2 = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'assistant_message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools2,
    }
    state2 = {
        'config': dict(state['config']),
        'checkpoint_messages': [
            {'id': 'u1', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': 'existing'},
        ],
        'summary': 'S2',
        'summary_message_content': '<auto_compaction_context>S2</auto_compaction_context>',
        'externalized_refs': {'enable': False},
    }
    ctx2 = _adoption_stream_ctx(tools2, metadata2, state=state2)
    ctx2['event_emitter'] = current2['emitter']
    ctx2['assistant_message'] = asyncio.run(middleware.Chats.get_message_by_id_and_message_id('chat', 'assistant'))

    asyncio.run(middleware.streaming_chat_response_handler(current2['text_response']('more'), ctx2))

    assert _adoption_shape(current2['saved'][-1]['output']) == [
        ('message', None, None, None),
        (compaction.CONTEXT_COMPACTION_OUTPUT_TYPE, None, 'S2', None),
        ('message', None, None, None),
    ]
    continuing_emits = [
        event['data']['output']
        for event in current2['emitted']
        if event.get('type') == 'chat:completion'
        and isinstance(event.get('data', {}).get('output'), list)
        and _adoption_shape(event['data']['output'])
        == [
            ('message', None, None, None),
            (compaction.CONTEXT_COMPACTION_OUTPUT_TYPE, None, 'S2', None),
        ]
    ]
    assert continuing_emits


def test_continuation_dedupes_against_branch_adoption_record(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    current, tools = _adoption_stream_harness(monkeypatch, middleware, summaries=[])
    current['replies'] = [current['text_response']('done')]
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    state = {
        'config': {
            'enable': True,
            'token_threshold': 1000,
            'token_cap': 1000,
            'retention_percentage': 40,
            'prompt_template': '',
            'soft_trigger_ratio': 0,
            'transient_patterns': (),
        },
        'checkpoint_messages': [
            {'id': 'u0', 'role': 'user', 'content': 'start'},
            {
                'id': 'a0',
                'role': 'assistant',
                'content': '',
                'output': [_adoption_item('S', 'cc_0')],
            },
            {'id': 'u1', 'role': 'user', 'content': 'next'},
        ],
        'summary': 'S',
        'summary_message_content': '<auto_compaction_context>S</auto_compaction_context>',
        'externalized_refs': {'enable': False},
    }
    ctx = _adoption_stream_ctx(tools, metadata, state=state)
    ctx['event_emitter'] = current['emitter']

    asyncio.run(
        middleware.streaming_chat_response_handler(
            current['tool_response']('call-a', 'view_file'),
            ctx,
        )
    )

    assert current['summaries'] == []
    saved_records = [
        item
        for update in current['saved']
        for item in update.get('output', [])
        if item.get('type') == compaction.CONTEXT_COMPACTION_OUTPUT_TYPE
    ]
    assert saved_records == []
    emitted_records = [
        item
        for event in current['emitted']
        if event.get('type') == 'chat:completion'
        for item in event.get('data', {}).get('output') or []
        if isinstance(item, dict) and item.get('type') == compaction.CONTEXT_COMPACTION_OUTPUT_TYPE
    ]
    assert emitted_records == []


def test_content_only_continuation_places_adoption_after_previous_text(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    current, tools = _adoption_stream_harness(
        monkeypatch,
        middleware,
        summaries=[],
        existing_message={'id': 'assistant', 'role': 'assistant', 'content': 'OLD'},
    )
    current['replies'] = [current['text_response']('MORE')]
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'assistant_message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    state = {
        'config': {
            'enable': True,
            'token_threshold': 1000,
            'token_cap': 1000,
            'retention_percentage': 40,
            'prompt_template': '',
            'soft_trigger_ratio': 0,
            'transient_patterns': (),
        },
        'checkpoint_messages': [
            {'id': 'u1', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': 'OLD'},
        ],
        'summary': 'S',
        'summary_message_content': '<auto_compaction_context>S</auto_compaction_context>',
        'externalized_refs': {'enable': False},
    }
    ctx = _adoption_stream_ctx(tools, metadata, state=state)
    ctx['event_emitter'] = current['emitter']
    ctx['assistant_message'] = asyncio.run(middleware.Chats.get_message_by_id_and_message_id('chat', 'assistant'))

    asyncio.run(middleware.streaming_chat_response_handler(current['text_response']('MORE'), ctx))

    final = current['saved'][-1]
    assert _adoption_shape(final['output']) == [
        ('message', None, None, None),
        (compaction.CONTEXT_COMPACTION_OUTPUT_TYPE, None, 'S', None),
        ('message', None, None, None),
    ]
    assert final['output'][0]['content'][0]['text'] == 'OLD'
    assert final['output'][2]['content'][0]['text'] == 'MORE'


def test_approval_pause_records_no_continuation_adoption(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    current, tools = _adoption_stream_harness(monkeypatch, middleware, summaries=[])
    current['replies'] = []
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'ask'},
        'tools': tools,
    }
    state = {
        'config': {
            'enable': True,
            'token_threshold': 100,
            'token_cap': 100,
            'retention_percentage': 40,
            'prompt_template': '',
            'soft_trigger_ratio': 0,
            'transient_patterns': (),
        },
        'checkpoint_messages': [
            {'id': 'u1', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': ''},
        ],
        'summary': 'S1',
        'summary_message_content': '<auto_compaction_context>S1</auto_compaction_context>',
        'externalized_refs': {'enable': False},
    }
    ctx = _adoption_stream_ctx(tools, metadata, state=state)
    ctx['event_emitter'] = current['emitter']

    paused_outputs = []

    async def pause(_chat_id, _message_id, output, _form_data, _metadata, _context_usage=None):
        paused_outputs.append(copy.deepcopy(output))

    monkeypatch.setattr(middleware, 'pause_for_tool_approval', pause)

    asyncio.run(
        middleware.streaming_chat_response_handler(
            current['tool_response']('call-a', 'view_file'),
            ctx,
        )
    )

    assert paused_outputs
    shape = _adoption_shape(paused_outputs[0])
    assert shape[0] == (compaction.CONTEXT_COMPACTION_OUTPUT_TYPE, None, 'S1', None)
    assert [item for item in shape if item[0] == compaction.CONTEXT_COMPACTION_OUTPUT_TYPE] == [
        (compaction.CONTEXT_COMPACTION_OUTPUT_TYPE, None, 'S1', None)
    ]


def test_cancelled_stream_keeps_adoption_record(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    current, tools = _adoption_stream_harness(monkeypatch, middleware, summaries=[])
    current['replies'] = []
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    state = {
        'config': {
            'enable': True,
            'token_threshold': 100,
            'token_cap': 100,
            'retention_percentage': 40,
            'prompt_template': '',
            'soft_trigger_ratio': 0,
            'transient_patterns': (),
        },
        'checkpoint_messages': [
            {'id': 'u1', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': ''},
        ],
        'summary': 'S1',
        'summary_message_content': '<auto_compaction_context>S1</auto_compaction_context>',
        'externalized_refs': {'enable': False},
    }
    ctx = _adoption_stream_ctx(tools, metadata, state=state)

    async def cancelling_emitter(event):
        if event.get('type') == 'response:completion':
            raise asyncio.CancelledError()

    ctx['event_emitter'] = cancelling_emitter

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            middleware.streaming_chat_response_handler(
                current['text_response']('done'),
                ctx,
            )
        )

    cancelled_saves = [update for update in current['saved'] if update.get('done') is True and 'output' in update]
    assert cancelled_saves
    for update in cancelled_saves:
        assert _adoption_shape(update['output'])[0] == (
            compaction.CONTEXT_COMPACTION_OUTPUT_TYPE,
            None,
            'S1',
            None,
        )


def test_non_stream_response_prepends_adoption_record(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    emitted = []
    saved = []

    async def emitter(event):
        emitted.append(copy.deepcopy(event))

    async def save(_chat_id, _message_id, update, **_kwargs):
        saved.append(copy.deepcopy(update))
        return update

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(middleware, 'outlet_filter_handler', noop)
    monkeypatch.setattr(middleware, 'background_tasks_handler', noop)
    monkeypatch.setattr(middleware, 'publish_chat_finished_event', noop)
    monkeypatch.setattr(middleware.Chats, 'upsert_message_to_chat_by_id_and_message_id', save)
    monkeypatch.setattr(middleware.Chats, 'get_chat_title_by_id', noop)

    metadata = {'chat_id': 'chat', 'message_id': 'assistant'}
    state = {
        'config': {'enable': False},
        'summary': 'S1',
        'summary_message_content': '<auto_compaction_context>S1</auto_compaction_context>',
        'checkpoint_messages': [
            {'id': 'u1', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': ''},
        ],
    }
    ctx = {
        'request': SimpleNamespace(),
        'form_data': {},
        'user': SimpleNamespace(role='admin'),
        'metadata': metadata,
        'events': [],
        'event_emitter': emitter,
        'model': {'id': 'model'},
        'compaction_state': state,
    }

    response = {'choices': [{'message': {'content': 'done'}}], 'usage': {}}
    result = asyncio.run(middleware.non_streaming_chat_response_handler(response, ctx))

    assert _adoption_shape(saved[-1]['output']) == [
        (compaction.CONTEXT_COMPACTION_OUTPUT_TYPE, None, 'S1', None),
        ('message', None, None, None),
    ]
    done_emits = [
        event['data']['output']
        for event in emitted
        if event.get('type') == 'chat:completion' and event.get('data', {}).get('done')
    ]
    assert done_emits and _adoption_shape(done_emits[-1]) == [
        (compaction.CONTEXT_COMPACTION_OUTPUT_TYPE, None, 'S1', None),
        ('message', None, None, None),
    ]
    message_item = saved[-1]['output'][-1]
    assert message_item['content'][0]['text'] == 'done'
    assert result['choices'][0]['message']['content'] == 'done'

    state['summary'] = 'S1'
    state['checkpoint_messages'][1]['output'] = [_adoption_item('S1', 'cc_0')]
    saved.clear()
    asyncio.run(middleware.non_streaming_chat_response_handler(copy.deepcopy(response), ctx))
    assert _adoption_shape(saved[-1]['output']) == [('message', None, None, None)]


def test_fork_carries_embedded_context_usage_snapshots(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    chats_model = importlib.import_module('open_webui.models.chats')
    chat_messages = importlib.import_module('open_webui.models.chat_messages')
    chats_router = importlib.import_module('open_webui.routers.chats')

    makers = {}

    @asynccontextmanager
    async def use_test_db(db=None):
        if isinstance(db, AsyncSession):
            yield db
        else:
            async with makers['session']() as session:
                yield session

    monkeypatch.setattr(chats_model, 'get_async_db_context', use_test_db)
    monkeypatch.setattr(chat_messages, 'get_async_db_context', use_test_db)

    async def permission_granted(*_args, **_kwargs):
        return None

    async def no_active_tasks(*_args, **_kwargs):
        return False

    async def no_event(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chats_router, 'require_chat_import_permission', permission_granted)
    monkeypatch.setattr(chats_router, 'has_active_tasks', no_active_tasks)
    monkeypatch.setattr(chats_router, 'publish_event', no_event)

    snapshot = {
        'tokens': 60,
        'threshold': 100,
        'soft_threshold': 50,
        'source': 'estimated',
    }
    source_messages = {
        'u1': {'id': 'u1', 'parentId': None, 'role': 'user', 'content': 'start'},
        'a1': {
            'id': 'a1',
            'parentId': 'u1',
            'role': 'assistant',
            'content': '',
            'done': True,
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'answer'}]}],
        },
    }
    source_chat_payload = {'history': {'currentId': 'a1', 'messages': copy.deepcopy(source_messages)}}

    async def run():
        engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "fork.db"}')
        makers['session'] = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with engine.begin() as connection:
                await connection.run_sync(chats_model.Chat.__table__.create)
                await connection.run_sync(chat_messages.ChatMessage.__table__.create)

            await chats_model.Chats.insert_new_chat(
                'source-chat',
                'user-1',
                chats_model.ChatForm(chat=source_chat_payload),
            )

            saved = await chats_model.Chats.upsert_message_to_chat_by_id_and_message_id(
                'source-chat',
                'a1',
                {
                    'done': True,
                    'output': source_messages['a1']['output'],
                    'context_usage': dict(snapshot),
                    'usage': {'prompt_tokens': 50, 'completion_tokens': 10, 'total_tokens': 60},
                },
            )
            assert saved is not None

            embedded = saved.chat['history']['messages']['a1']
            assert embedded['context_usage'] == snapshot

            normalized_map = await chats_model.Chats.get_messages_map_by_chat_id('source-chat')
            assert normalized_map is not None
            assert 'context_usage' not in normalized_map['a1']

            async with makers['session']() as db:
                request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=None)))
                fork = await chats_router.fork_chat_by_id(
                    request,
                    'source-chat',
                    None,
                    SimpleNamespace(id='user-1', role='admin'),
                    db=db,
                )
            assert fork is not None

            assert fork.chat['history']['messages']['a1']['context_usage'] == snapshot
            assert 'context_usage' not in fork.chat['history']['messages']['u1']

            fork_row = await chats_model.Chats.get_chat_by_id(fork.id)
            assert fork_row is not None
            assert fork_row.chat['history']['messages']['a1']['context_usage'] == snapshot
            assert 'context_usage' not in fork_row.chat['history']['messages']['u1']

            source_row = await chats_model.Chats.get_chat_by_id('source-chat')
            assert source_row.chat['history']['messages']['a1']['context_usage'] == snapshot
            assert 'context_usage' not in source_row.chat['history']['messages']['u1']
        finally:
            await engine.dispose()

    asyncio.run(run())


def _usage_chunk(total):
    return {
        'choices': [{'delta': {'content': 'PART'}, 'finish_reason': None}],
        'usage': {'prompt_tokens': 80, 'completion_tokens': total - 80, 'total_tokens': total},
    }


def _usage_only_chunk(total):
    return {
        'choices': [],
        'usage': {'prompt_tokens': 80, 'completion_tokens': total - 80, 'total_tokens': total},
    }


def _completed_chunk(total):
    return {
        'type': 'response.completed',
        'response': {
            'id': 'resp',
            'output': [
                {
                    'type': 'message',
                    'role': 'assistant',
                    'status': 'completed',
                    'content': [{'type': 'output_text', 'text': 'PART'}],
                }
            ],
            'usage': {
                'input_tokens': 80,
                'output_tokens': total - 80,
                'total_tokens': total,
            },
        },
    }


def test_same_response_usage_stays_latest_single_response(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    current, tools = _adoption_stream_harness(monkeypatch, middleware, summaries=[])
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    state = {
        'config': {
            'enable': True,
            'token_threshold': 100,
            'token_cap': 100,
            'retention_percentage': 40,
            'prompt_template': '',
            'soft_trigger_ratio': 0.5,
            'transient_patterns': (),
            'externalized_refs_enable': False,
        },
        'checkpoint_messages': [
            {'id': 'u1', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': ''},
        ],
        'externalized_refs': {'enable': False},
        'context_usage': {
            'tokens': 60,
            'threshold': 100,
            'soft_threshold': 50,
            'source': 'estimated',
        },
    }
    ctx = _adoption_stream_ctx(tools, metadata, state=state)
    ctx['event_emitter'] = current['emitter']

    async def run():
        response = StreamingResponse(
            _chunk_iter(middleware, [_usage_chunk(85), _usage_chunk(90)]), media_type='text/event-stream'
        )
        await middleware.streaming_chat_response_handler(response, ctx)
        task = compaction.start_completed_turn_compaction_prefetch(
            ctx['request'], ctx['user'], ctx['form_data']['messages'], metadata, 'model', {}, state,
            ctx['completed_compaction']['usage'],
        )
        assert task is not None, 'the single-response usage must arm the prefetch'
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return (
            copy.deepcopy(_numeric_snapshots(current['emitted'])),
            copy.deepcopy(state['context_usage']),
            copy.deepcopy(ctx['completed_compaction']),
        )

    notifications, snapshot, completed = asyncio.run(run())

    assert [item['tokens'] for item in notifications] == [85, 90]
    assert snapshot['tokens'] == 90
    assert completed['usage']['total_tokens'] == 90


def _chunk_iter(middleware, specs, done=True):
    async def chunks():
        for spec in specs:
            yield f'data: {middleware.JSONCodec.dumps(spec)}\n\n'.encode()
        if done:
            yield b'data: [DONE]\n\n'

    return chunks()


def test_responses_api_usage_stays_latest_single_response(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    current, tools = _adoption_stream_harness(monkeypatch, middleware, summaries=[])
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    state = {
        'config': {
            'enable': True,
            'token_threshold': 100,
            'token_cap': 100,
            'retention_percentage': 40,
            'prompt_template': '',
            'soft_trigger_ratio': 0,
            'transient_patterns': (),
            'externalized_refs_enable': False,
        },
        'checkpoint_messages': [
            {'id': 'u1', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': ''},
        ],
        'externalized_refs': {'enable': False},
    }
    ctx = _adoption_stream_ctx(tools, metadata, state=state)
    ctx['event_emitter'] = current['emitter']

    asyncio.run(
        middleware.streaming_chat_response_handler(
            StreamingResponse(
                _chunk_iter(middleware, [_completed_chunk(85), _completed_chunk(90)]),
                media_type='text/event-stream',
            ),
            ctx,
        )
    )

    assert [item['tokens'] for item in _numeric_snapshots(current['emitted'])] == [85, 90]
    assert state['context_usage']['tokens'] == 90


def test_next_round_usage_does_not_accumulate(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    current, tools = _adoption_stream_harness(monkeypatch, middleware, summaries=[])
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    state = {
        'config': {
            'enable': True,
            'token_threshold': 100,
            'token_cap': 100,
            'retention_percentage': 40,
            'prompt_template': '',
            'soft_trigger_ratio': 0,
            'transient_patterns': (),
            'externalized_refs_enable': False,
        },
        'checkpoint_messages': [
            {'id': 'u1', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': ''},
        ],
        'externalized_refs': {'enable': False},
    }
    ctx = _adoption_stream_ctx(tools, metadata, state=state)
    ctx['event_emitter'] = current['emitter']

    async def run():
        first = StreamingResponse(_chunk_iter(middleware, [_usage_chunk(85)]), media_type='text/event-stream')
        await middleware.streaming_chat_response_handler(first, ctx)
        current['emitted'].clear()
        second = StreamingResponse(_chunk_iter(middleware, [_usage_chunk(90)]), media_type='text/event-stream')
        await middleware.streaming_chat_response_handler(second, ctx)
        return copy.deepcopy(_numeric_snapshots(current['emitted'])), copy.deepcopy(state['context_usage'])

    notifications, snapshot = asyncio.run(run())

    assert [item['tokens'] for item in notifications] == [90]
    assert snapshot['tokens'] == 90


def _stream_store_harness(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    tasks = importlib.import_module('open_webui.tasks')
    chats_router = importlib.import_module('open_webui.routers.chats')
    current, tools = _adoption_stream_harness(monkeypatch, middleware, summaries=[])

    save_calls = []
    real_save = tasks.save_response_stream

    async def counting_save(redis, task_id, chat_id, message_id, content, output, **kwargs):
        save_calls.append(
            {
                'content': content,
                'context_usage': copy.deepcopy(kwargs.get('context_usage')),
            }
        )
        await real_save(redis, task_id, chat_id, message_id, content, output, **kwargs)

    monkeypatch.setattr(middleware, 'save_response_stream', counting_save)
    monkeypatch.setattr(middleware, 'clear_response_stream', tasks.clear_response_stream)
    monkeypatch.setattr(tasks, 'response_streams', {})
    monkeypatch.setattr(tasks, 'item_tasks', {'chat': ['stream-task']})

    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'task_id': 'stream-task',
        'user_prompt': 'start',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
    }
    confirmed = {
        'tokens': 60,
        'threshold': 100,
        'soft_threshold': 50,
        'source': 'estimated',
    }
    state = {
        'config': {
            'enable': True,
            'token_threshold': 100,
            'token_cap': 100,
            'retention_percentage': 40,
            'prompt_template': '',
            'soft_trigger_ratio': 0,
            'transient_patterns': (),
            'externalized_refs_enable': False,
        },
        'checkpoint_messages': [
            {'id': 'u1', 'role': 'user', 'content': 'start'},
            {'id': 'assistant', 'role': 'assistant', 'content': ''},
        ],
        'externalized_refs': {'enable': False},
        'context_usage': dict(confirmed),
        'pending_context_usage': {
            'tokens': 120,
            'threshold': 100,
            'soft_threshold': None,
            'source': 'estimated',
        },
    }
    ctx = _adoption_stream_ctx(tools, metadata, state=state)
    ctx['event_emitter'] = current['emitter']

    def blob():
        return {
            'chat': {
                'history': {
                    'currentId': 'assistant',
                    'messages': {
                        'assistant': {
                            'id': 'assistant',
                            'role': 'assistant',
                            'content': '',
                            'done': True,
                            'context_usage': dict(confirmed),
                        }
                    },
                },
                'messages': [
                    {'id': 'assistant', 'role': 'assistant', 'content': '', 'done': True},
                ],
            }
        }

    return {
        'middleware': middleware,
        'tasks': tasks,
        'chats_router': chats_router,
        'current': current,
        'ctx': ctx,
        'state': state,
        'metadata': metadata,
        'confirmed': confirmed,
        'save_calls': save_calls,
        'blob': blob,
    }


def _parked_stream(middleware, specs, park_before_done=True):
    parked = asyncio.Event()
    release = asyncio.Event()

    async def chunks():
        for spec in specs:
            yield f'data: {middleware.JSONCodec.dumps(spec)}\n\n'.encode()
        parked.set()
        if park_before_done:
            await release.wait()
        yield b'data: [DONE]\n\n'

    return StreamingResponse(chunks(), media_type='text/event-stream'), parked, release


def test_in_progress_reload_restores_notified_snapshot(monkeypatch):
    harness = _stream_store_harness(monkeypatch)
    middleware = harness['middleware']
    ctx = harness['ctx']

    async def run():
        response, parked, release = _parked_stream(
            middleware,
            [{'choices': [{'delta': {'content': 'PART'}, 'finish_reason': None}]}, _usage_only_chunk(90)],
        )
        running = asyncio.create_task(middleware.streaming_chat_response_handler(response, ctx))
        try:
            await asyncio.wait_for(parked.wait(), 5)
            streams = copy.deepcopy(await harness['tasks'].get_response_streams_by_chat_id(None, 'chat'))
            restored = harness['chats_router'].overlay_response_streams(harness['blob'](), streams)
            return streams, restored, copy.deepcopy(harness['current']['saved'])
        finally:
            release.set()
            await asyncio.wait_for(running, 5)

    streams, restored, saved_before_completion = asyncio.run(run())

    measured = {
        'tokens': 90,
        'threshold': 100,
        'soft_threshold': 50,
        'source': 'usage',
    }
    assert streams and streams[0]['content'] == 'PART'
    assert streams[0]['context_usage'] == measured
    assert saved_before_completion == []

    message = restored['chat']['history']['messages']['assistant']
    assert message['content'] == 'PART'
    assert message['context_usage'] == measured
    assert restored['chat']['messages'][0]['context_usage'] == measured

    # The unsent candidate must never leak into the stream store.
    for call in harness['save_calls']:
        if call['context_usage'] is not None:
            assert call['context_usage'] != harness['state']['pending_context_usage']


def test_usage_only_chunk_updates_stream_store_before_wait(monkeypatch):
    harness = _stream_store_harness(monkeypatch)
    middleware = harness['middleware']
    ctx = harness['ctx']

    async def run():
        response, parked, release = _parked_stream(
            middleware, [{'choices': [{'delta': {'content': 'PART'}, 'finish_reason': None}]}, _usage_only_chunk(90)]
        )
        running = asyncio.create_task(middleware.streaming_chat_response_handler(response, ctx))
        try:
            await asyncio.wait_for(parked.wait(), 5)
            streams = copy.deepcopy(await harness['tasks'].get_response_streams_by_chat_id(None, 'chat'))
            restored = harness['chats_router'].overlay_response_streams(harness['blob'](), streams)
            return streams, restored, list(harness['save_calls'])
        finally:
            release.set()
            await asyncio.wait_for(running, 5)

    streams, restored, save_calls = asyncio.run(run())

    measured = {
        'tokens': 90,
        'threshold': 100,
        'soft_threshold': 50,
        'source': 'usage',
    }
    assert streams[0]['content'] == 'PART'
    assert streams[0]['context_usage'] == measured
    message = restored['chat']['history']['messages']['assistant']
    assert message['content'] == 'PART'
    assert message['context_usage'] == measured

    usage_saves = [call for call in save_calls if call['context_usage'] == measured]
    assert usage_saves, 'usage receipt must reach the stream store before waiting'
    assert len(usage_saves) == 1, 'the same content must not be saved twice for one chunk'


def test_stream_start_persists_confirmed_snapshot(monkeypatch):
    harness = _stream_store_harness(monkeypatch)
    middleware = harness['middleware']
    ctx = harness['ctx']

    async def run():
        started = asyncio.Event()
        release = asyncio.Event()

        async def chunks():
            started.set()
            await release.wait()
            yield b'data: [DONE]\n\n'

        running = asyncio.create_task(
            middleware.streaming_chat_response_handler(
                StreamingResponse(chunks(), media_type='text/event-stream'), ctx
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 5)
            for _ in range(200):
                streams = await harness['tasks'].get_response_streams_by_chat_id(None, 'chat')
                if streams:
                    return copy.deepcopy(streams)
                await asyncio.sleep(0.01)
            return []
        finally:
            release.set()
            await asyncio.wait_for(running, 5)

    streams = asyncio.run(run())

    assert streams and streams[0]['context_usage'] == harness['confirmed']
    assert streams[0]['content'] == ''
    assert streams[0]['context_usage'] != harness['state']['pending_context_usage']


def test_overlay_keeps_existing_snapshot_without_stream_value():
    chats_router = importlib.import_module('open_webui.routers.chats')
    existing = {
        'tokens': 60,
        'threshold': 100,
        'soft_threshold': 50,
        'source': 'estimated',
    }
    chat_data = {
        'chat': {
            'history': {
                'currentId': 'a1',
                'messages': {
                    'a1': {'id': 'a1', 'role': 'assistant', 'content': 'old', 'context_usage': dict(existing)},
                    'sib': {'id': 'sib', 'role': 'assistant', 'content': 'sibling'},
                },
            },
            'messages': [{'id': 'a1', 'role': 'assistant', 'content': 'old'}],
        }
    }
    streams = [
        {
            'chat_id': 'chat',
            'message_id': 'a1',
            'content': 'restored',
            'output': [],
        }
    ]

    chats_router.overlay_response_streams(chat_data, streams)

    message = chat_data['chat']['history']['messages']['a1']
    assert message['content'] == 'restored'
    assert message['context_usage'] == existing
    assert 'context_usage' not in chat_data['chat']['history']['messages']['sib']

    streams.append(
        {
            'chat_id': 'chat',
            'message_id': 'a1',
            'content': 'restored',
            'output': [],
            'context_usage': None,
        }
    )
    chats_router.overlay_response_streams(chat_data, streams)
    assert chat_data['chat']['history']['messages']['a1']['context_usage'] == existing


def test_overlay_copies_stream_snapshot_to_both_message_paths():
    chats_router = importlib.import_module('open_webui.routers.chats')
    snapshot = {
        'tokens': 90,
        'threshold': 100,
        'soft_threshold': 50,
        'source': 'usage',
    }
    chat_data = {
        'chat': {
            'history': {
                'currentId': 'a1',
                'messages': {'a1': {'id': 'a1', 'role': 'assistant', 'content': ''}},
            },
            'messages': [{'id': 'a1', 'role': 'assistant', 'content': ''}],
        }
    }
    streams = [
        {
            'chat_id': 'chat',
            'message_id': 'a1',
            'content': 'PART',
            'output': [],
            'context_usage': dict(snapshot),
        }
    ]

    chats_router.overlay_response_streams(chat_data, streams)

    assert chat_data['chat']['history']['messages']['a1']['context_usage'] == snapshot
    assert chat_data['chat']['messages'][0]['context_usage'] == snapshot


class _RecordingRedis:
    def __init__(self):
        self.entries = {}
        self.writes = []
        self.exists_count = 0

    def pipeline(self, transaction=False):
        self.exists_count = 0
        return self

    def exists(self, key):
        self.exists_count += 1
        return self

    async def execute(self):
        return [True] * self.exists_count

    async def hset(self, key, field, value):
        self.entries[(key, field)] = value
        self.writes.append(value)

    async def hmget(self, key, fields):
        return [self.entries.get((key, field)) for field in fields]

    async def smembers(self, key):
        return {'stream-task'}

    async def hexpire(self, *args):
        return None

    async def hdel(self, key, field):
        self.entries.pop((key, field), None)


def _observe_redis_stream(monkeypatch, specs, batch_size=1):
    harness = _stream_store_harness(monkeypatch)
    middleware = harness['middleware']
    redis = _RecordingRedis()
    harness['ctx']['request'].app.state.redis = redis
    harness['metadata']['params']['stream_delta_chunk_size'] = batch_size
    monkeypatch.setattr(middleware, 'CHAT_RESPONSE_STREAM_DELTA_CHUNK_SIZE', batch_size)

    async def run():
        response, parked, release = _parked_stream(middleware, specs)
        running = asyncio.create_task(
            middleware.streaming_chat_response_handler(response, harness['ctx'])
        )
        try:
            await asyncio.wait_for(parked.wait(), 5)
            streams = copy.deepcopy(await harness['tasks'].get_response_streams_by_chat_id(redis, 'chat'))
            return {
                'streams': streams,
                'writes': [middleware.JSONCodec.loads(value) for value in redis.writes],
                'events': copy.deepcopy(harness['current']['emitted']),
                'state': copy.deepcopy(harness['state']['context_usage']),
            }
        finally:
            running.cancel()
            try:
                await running
            except asyncio.CancelledError:
                pass
            release.set()

    return asyncio.run(run())


def test_structured_updates_reach_stream_store_serialized(monkeypatch):
    base = {
        'type': 'function_call',
        'id': 'fc',
        'call_id': 'call',
        'name': 'view_file',
        'arguments': '',
        'status': 'in_progress',
    }
    specs = [
        {'type': 'response.output_item.added', 'output_index': 0, 'item': dict(base)},
        {
            'type': 'response.function_call_arguments.delta',
            'output_index': 0,
            'item_id': 'fc',
            'delta': '{"x":1}',
        },
        {
            'type': 'response.output_item.done',
            'output_index': 0,
            'item': {**base, 'arguments': '{"x":1}', 'status': 'completed'},
        },
    ]

    result = _observe_redis_stream(monkeypatch, specs)

    assert any(e.get('data', {}).get('type') == 'response.output_item.done' for e in result['events'])
    stored = result['streams'][0]['output'][0]
    assert stored['arguments'] == '{"x":1}'
    assert stored['status'] == 'completed'


def test_batched_body_usage_does_not_add_full_saves(monkeypatch):
    specs = [_usage_chunk(total) for total in range(81, 85)]
    no_usage_specs = [{k: v for k, v in spec.items() if k != 'usage'} for spec in specs]

    with pytest.MonkeyPatch.context() as inner:
        control = _observe_redis_stream(inner, no_usage_specs, batch_size=20)
    with pytest.MonkeyPatch.context() as inner:
        measured = _observe_redis_stream(inner, specs, batch_size=20)

    assert len(measured['writes']) == len(control['writes'])


@pytest.mark.parametrize(
    'invalid_usage',
    [
        {'prompt_tokens': None, 'completion_tokens': None, 'total_tokens': None},
        {'cost': 0.01},
    ],
)
def test_missing_token_counters_keep_latest(monkeypatch, invalid_usage):
    result = _observe_redis_stream(
        monkeypatch, [_usage_chunk(90), {'choices': [], 'usage': dict(invalid_usage)}]
    )

    assert result['state']['tokens'] == 90
    assert [item['tokens'] for item in _numeric_snapshots(result['events'])] == [90]


def _completed_stream_observe(monkeypatch, specs, batch_size=1):
    harness = _stream_store_harness(monkeypatch)
    middleware = harness['middleware']
    redis = _RecordingRedis()
    harness['ctx']['request'].app.state.redis = redis
    harness['metadata']['params']['stream_delta_chunk_size'] = batch_size
    monkeypatch.setattr(middleware, 'CHAT_RESPONSE_STREAM_DELTA_CHUNK_SIZE', batch_size)

    async def run():
        response = StreamingResponse(_chunk_iter(middleware, specs), media_type='text/event-stream')
        await middleware.streaming_chat_response_handler(response, harness['ctx'])
        return {
            'snapshot': copy.deepcopy(harness['state']['context_usage']),
            'completed': copy.deepcopy(harness['ctx']['completed_compaction']),
            'writes': [middleware.JSONCodec.loads(raw) for raw in redis.writes],
            'snapshots': copy.deepcopy(_numeric_snapshots(harness['current']['emitted'])),
        }

    return asyncio.run(run())


def test_invalid_total_preserves_single_response_usage(monkeypatch):
    result = _completed_stream_observe(
        monkeypatch,
        [
            _usage_chunk(90),
            {'choices': [], 'usage': {'prompt_tokens': 80, 'completion_tokens': 10, 'total_tokens': -1}},
        ],
    )

    assert result['snapshot']['tokens'] == 90
    assert result['completed']['usage']['total_tokens'] == 90


def test_finished_body_usage_has_no_redundant_save(monkeypatch):
    specs = [_usage_chunk(81), _usage_chunk(82)]
    plain = [{k: v for k, v in spec.items() if k != 'usage'} for spec in specs]

    with pytest.MonkeyPatch.context() as inner:
        control = _completed_stream_observe(inner, plain)
    with pytest.MonkeyPatch.context() as inner:
        measured = _completed_stream_observe(inner, specs)

    assert len(measured['writes']) == len(control['writes'])


@pytest.mark.parametrize(
    ('raw', 'tokens'),
    [
        ({'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, 0),
        ({'prompt_eval_count': 80, 'eval_count': 10}, 90),
        ({'prompt_n': 50, 'cache_n': 30, 'predicted_n': 10}, 90),
        (
            {
                'input_tokens': 20,
                'cache_creation_input_tokens': 10,
                'cache_read_input_tokens': 50,
                'output_tokens': 10,
            },
            90,
        ),
    ],
)
def test_valid_measurements_still_work(monkeypatch, raw, tokens):
    result = _completed_stream_observe(monkeypatch, [{'choices': [], 'usage': dict(raw)}])

    assert result['snapshot']['tokens'] == tokens


def _non_stream_observe(monkeypatch, usage):
    harness = _stream_store_harness(monkeypatch)
    middleware = harness['middleware']
    response = {'choices': [{'message': {'content': 'OK'}}], 'usage': copy.deepcopy(usage)}

    asyncio.run(middleware.non_streaming_chat_response_handler(response, harness['ctx']))

    saved = harness['current']['saved'][-1]
    notified = [
        event['data']['context_usage']
        for event in harness['current']['emitted']
        if isinstance(event.get('data'), dict) and 'context_usage' in event['data']
    ]
    return {
        'state': copy.deepcopy(harness['state']['context_usage']),
        'completed': copy.deepcopy(harness['ctx'].get('completed_compaction')),
        'saved': saved,
        'notified': notified,
    }


@pytest.mark.parametrize(
    'raw',
    [
        {'cost': 0.01},
        {'prompt_tokens': None, 'completion_tokens': None, 'total_tokens': None},
    ],
)
def test_non_stream_keeps_estimate_without_measurement(monkeypatch, raw):
    result = _non_stream_observe(monkeypatch, raw)

    assert result['state'] == _stream_confirmed_snapshot()
    assert result['saved']['context_usage'] == _stream_confirmed_snapshot()
    assert result['notified'][-1] == _stream_confirmed_snapshot()
    assert result['completed']['usage'] is None


def test_non_stream_invalid_total_is_not_adopted(monkeypatch):
    result = _non_stream_observe(
        monkeypatch, {'prompt_tokens': 80, 'completion_tokens': 10, 'total_tokens': -1}
    )

    assert result['state'] == _stream_confirmed_snapshot()
    assert result['completed']['usage'] is None


def test_non_stream_explicit_zero_is_adopted(monkeypatch):
    result = _non_stream_observe(monkeypatch, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})

    assert result['state']['tokens'] == 0
    assert result['state']['source'] == 'usage'
    assert result['completed']['usage']['total_tokens'] == 0


def test_non_stream_cache_inclusive_usage_is_adopted(monkeypatch):
    result = _non_stream_observe(
        monkeypatch,
        {
            'input_tokens': 20,
            'cache_creation_input_tokens': 10,
            'cache_read_input_tokens': 50,
            'output_tokens': 10,
        },
    )

    assert result['state']['tokens'] == 90
    assert result['state']['source'] == 'usage'
    completed = result['completed']['usage']
    assert completed is not None
    assert completed['cache_creation_input_tokens'] == 10
    assert completed['cache_read_input_tokens'] == 50
    assert (
        compaction._usage_total_tokens(completed) == 90
    ), 'the prefetch gate must see the cache-inclusive total'


def _sqlite_session_context(makers):
    from contextlib import asynccontextmanager

    from sqlalchemy.ext.asyncio import AsyncSession

    @asynccontextmanager
    async def use_test_db(db=None):
        if isinstance(db, AsyncSession):
            yield db
        else:
            async with makers['session']() as session:
                yield session

    return use_test_db


def _stream_confirmed_snapshot():
    return {
        'tokens': 60,
        'threshold': 100,
        'soft_threshold': 50,
        'source': 'estimated',
    }


def _entry_wait_harness(monkeypatch, tmp_path):
    routing = importlib.import_module('test_chat_routing')
    main = importlib.import_module('open_webui.main')
    middleware = importlib.import_module('open_webui.utils.middleware')
    tasks = importlib.import_module('open_webui.tasks')
    chats_model = importlib.import_module('open_webui.models.chats')
    chat_messages = importlib.import_module('open_webui.models.chat_messages')

    _, request, user, _ = routing._install_provider_sink_harness(monkeypatch, {'model': routing._EntryModelInfo({})})

    makers = {}
    monkeypatch.setattr(chats_model, 'get_async_db_context', _sqlite_session_context(makers))
    monkeypatch.setattr(chat_messages, 'get_async_db_context', _sqlite_session_context(makers))

    events = []
    upsert_calls = []

    async def emit(event):
        events.append(copy.deepcopy(event))

    async def get_emitter(*_args, **_kwargs):
        return emit

    async def noop(*_args, **_kwargs):
        return None

    async def config_many(*keys):
        values = {
            'chat.context_compaction.enable': True,
            'chat.context_compaction.token_threshold': 100,
            'chat.context_compaction.token_cap': 100,
            'chat.context_compaction.soft_trigger_ratio': 0,
            'chat.externalized_refs.enable': False,
        }
        return {key: values.get(key) for key in keys}

    real_upsert = chats_model.Chats.upsert_message_to_chat_by_id_and_message_id
    real_snapshot_update = chats_model.Chats.update_message_context_usage
    upsert_calls = []
    snapshot_calls = []

    async def capturing_upsert(chat_id, message_id, update, **kwargs):
        upsert_calls.append({'update': copy.deepcopy(update)})
        return await real_upsert(chat_id, message_id, update, **kwargs)

    async def capturing_snapshot_update(chat_id, message_id, snapshot):
        snapshot_calls.append({'message_id': message_id, 'snapshot': copy.deepcopy(snapshot)})
        return await real_snapshot_update(chat_id, message_id, snapshot)

    monkeypatch.setattr(middleware.Config, 'get_many', config_many)
    monkeypatch.setattr(compaction, 'estimate_provider_tokens', lambda *_args, **_kwargs: 60)
    for module in (main, middleware):
        monkeypatch.setattr(module, 'get_event_emitter', get_emitter)
    for name in ('update_chat_by_id', 'update_chat_variables_by_id', 'get_chat_folder_id'):
        monkeypatch.setattr(main.Chats, name, noop)
    monkeypatch.setattr(main.Chats, 'is_chat_owner', noop)
    monkeypatch.setattr(main.Chats, 'upsert_message_to_chat_by_id_and_message_id', capturing_upsert)
    monkeypatch.setattr(main.Chats, 'update_message_context_usage', capturing_snapshot_update)
    monkeypatch.setattr(main, 'publish_event', noop)
    monkeypatch.setattr(main, 'emit_chat_list_event', noop)
    monkeypatch.setattr(main, 'create_task', tasks.create_task)
    monkeypatch.setattr(main, 'cleanup_task', tasks.cleanup_task)
    monkeypatch.setattr(tasks, 'tasks', {})
    monkeypatch.setattr(tasks, 'item_tasks', {})
    monkeypatch.setattr(tasks, 'response_streams', {})
    monkeypatch.setattr(
        importlib.import_module('open_webui.utils.subagents'), 'process_pending_internal_messages', noop
    )
    monkeypatch.setattr(
        importlib.import_module('open_webui.utils.timers'), 'cancel_timers_for_chat', noop
    )

    return {
        'routing': routing,
        'main': main,
        'middleware': middleware,
        'tasks': tasks,
        'chats_model': chats_model,
        'request': request,
        'user': user,
        'events': events,
        'upsert_calls': upsert_calls,
        'snapshot_calls': snapshot_calls,
        'makers': makers,
        'tmp_path': tmp_path,
    }


def test_initial_wait_and_cancel_keep_confirmed_snapshot(monkeypatch, tmp_path):
    harness = _entry_wait_harness(monkeypatch, tmp_path)
    main = harness['main']
    tasks = harness['tasks']
    chats_model = harness['chats_model']
    routing = harness['routing']

    async def run():
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        engine = create_async_engine(f"sqlite+aiosqlite:///{harness['tmp_path'] / 'entry.db'}")
        harness['makers']['session'] = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with engine.begin() as connection:
                await connection.run_sync(chats_model.Chat.__table__.create)
                await connection.run_sync(
                    importlib.import_module('open_webui.models.chat_messages').ChatMessage.__table__.create
                )

            stored_output = [
                {
                    'type': 'message',
                    'id': 'msg_1',
                    'status': 'in_progress',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': 'old'}],
                }
            ]
            await chats_model.Chats.insert_new_chat(
                'entry-review-chat',
                'user-1',
                chats_model.ChatForm(
                    chat={
                        'history': {
                            'currentId': 'assistant',
                            'messages': {
                                'u1': {
                                    'id': 'u1',
                                    'parentId': None,
                                    'role': 'user',
                                    'content': 'start',
                                    'childrenIds': ['assistant'],
                                },
                                'assistant': {
                                    'id': 'assistant',
                                    'parentId': 'u1',
                                    'role': 'assistant',
                                    'content': 'old',
                                    'output': stored_output,
                                    'done': True,
                                },
                            },
                        }
                    }
                ),
            )

            request = harness['request']
            request.app.state.MODELS = {'model': {'id': 'model', 'owned_by': 'openai', 'info': {}}}

            async def run_completion():
                form_data = routing._base_form_data(
                    'model',
                    [{'model_id': 'model', 'message_id': 'assistant'}],
                    chat_id='entry-review-chat',
                    chat_variables={},
                    user_message={
                        'id': 'u1',
                        'role': 'user',
                        'content': 'start',
                        'childrenIds': ['assistant'],
                    },
                )
                return await main.chat_completion(request, dict(form_data), harness['user'])

            entered = asyncio.Event()
            release = asyncio.Event()

            async def provider(request=None, form_data=None, user=None):
                entered.set()
                await release.wait()
                return {'choices': [{'message': {'content': 'OK'}}]}

            monkeypatch.setattr(
                importlib.import_module('open_webui.utils.chat'),
                'generate_openai_chat_completion',
                provider,
            )

            result = await run_completion()
            task_id = result['task_ids'][0]
            task = tasks.tasks[task_id]
            try:
                await asyncio.wait_for(entered.wait(), 5)

                async def read_row():
                    row = await chats_model.Chats.get_chat_by_id('entry-review-chat')
                    message = row.chat['history']['messages']['assistant']
                    return copy.deepcopy(message)

                before = {
                    'row': await read_row(),
                    'streams': copy.deepcopy(
                        await tasks.get_response_streams_by_chat_id(None, 'entry-review-chat')
                    ),
                }
                await tasks.stop_task(None, task_id)
                await asyncio.sleep(0)
                after = {
                    'row': await read_row(),
                    'streams': copy.deepcopy(
                        await tasks.get_response_streams_by_chat_id(None, 'entry-review-chat')
                    ),
                }
                return {'before': before, 'after': after}
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                release.set()
        finally:
            await engine.dispose()

    observed = asyncio.run(run())

    notified = _numeric_snapshots(harness['events'])
    snapshot = {
        'tokens': 60,
        'threshold': 100,
        'soft_threshold': None,
        'source': 'estimated',
    }
    assert notified and notified[-1] == snapshot
    assert harness['snapshot_calls'] == [
        {'message_id': 'assistant', 'snapshot': snapshot}
    ], 'the send commit must persist through the dedicated snapshot update'
    assert not any(
        call['update'].keys() == {'context_usage'} for call in harness['upsert_calls']
    ), 'the send commit must not reuse the dual-writing upsert'

    for phase in ('before', 'after'):
        message = observed[phase]['row']
        assert message['context_usage'] == snapshot
    assert observed['after']['streams'] == []


def test_continuation_commit_saves_snapshot_before_provider_wait(monkeypatch):
    harness = _stream_store_harness(monkeypatch)
    middleware = harness['middleware']
    ctx = harness['ctx']
    monkeypatch.setattr(compaction, 'estimate_provider_tokens', lambda *_args, **_kwargs: 40)

    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()
        real_generate = middleware.generate_chat_completion

        async def parking_generate(request, candidate, user, **kwargs):
            entered.set()
            await release.wait()
            return await real_generate(request, candidate, user, **kwargs)

        monkeypatch.setattr(middleware, 'generate_chat_completion', parking_generate)
        harness['current']['replies'] = [harness['current']['text_response']('done')]
        db_saves_before = len(harness['current']['saved'])
        running = asyncio.create_task(
            middleware.streaming_chat_response_handler(
                harness['current']['tool_response']('call-a', 'view_file'), ctx
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            streams = copy.deepcopy(
                await harness['tasks'].get_response_streams_by_chat_id(None, 'chat')
            )
            restored = harness['chats_router'].overlay_response_streams(harness['blob'](), streams)
            return {
                'streams': streams,
                'restored': restored,
                'db_saves': len(harness['current']['saved']) - db_saves_before,
            }
        finally:
            release.set()
            await asyncio.wait_for(running, 5)

    result = asyncio.run(run())

    snapshot = {
        'tokens': 40,
        'threshold': 100,
        'soft_threshold': None,
        'source': 'estimated',
    }
    assert result['streams'] and result['streams'][0]['context_usage'] == snapshot
    message = result['restored']['chat']['history']['messages']['assistant']
    assert message['context_usage'] == snapshot
    assert message['output'], 'prior tool output must not be overwritten with an empty list'
    assert any(item.get('type') == 'function_call' for item in message['output'])
    assert result['db_saves'] == 0, 'continuation commits must not add per-round DB writes'


async def _run_dualwrite_send_commit_leg(monkeypatch, middleware, chats_model, real_upsert, real_snapshot_update):
    upsert_calls = []
    snapshot_calls = []
    events = []

    async def emitter(event):
        events.append(copy.deepcopy(event))

    async def get_event_emitter(_metadata):
        return emitter

    async def counting_upsert(chat_id, message_id, update, **kwargs):
        upsert_calls.append({'update': copy.deepcopy(update)})
        return await real_upsert(chat_id, message_id, update, **kwargs)

    async def counting_snapshot_update(chat_id, message_id, snapshot):
        snapshot_calls.append(copy.deepcopy(snapshot))
        return await real_snapshot_update(chat_id, message_id, snapshot)

    _payload_leg_patches(monkeypatch, middleware, refs_runtime=(False, 1000))
    _payload_leg_drain(monkeypatch, middleware, {'output': []})
    _seed_state_on_capture(
        monkeypatch, middleware, {'summary_message_content': '', 'config': _compaction_config()}
    )
    monkeypatch.setattr(
        middleware.Chats, 'upsert_message_to_chat_by_id_and_message_id', counting_upsert
    )
    if real_snapshot_update is not None:
        monkeypatch.setattr(
            middleware.Chats, 'update_message_context_usage', counting_snapshot_update
        )
    monkeypatch.setattr(compaction, 'estimate_provider_tokens', lambda *_args, **_kwargs: 60)
    monkeypatch.setattr(middleware, 'get_event_emitter', get_event_emitter)

    _send, _metadata, _events, state = await _run_payload_leg(
        middleware, [{'role': 'user', 'content': 'start'}], message_id='assistant'
    )
    return state, events, upsert_calls, snapshot_calls


def test_send_commit_snapshot_does_not_replay_saved_usage(monkeypatch, tmp_path):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    chats_model = importlib.import_module('open_webui.models.chats')
    chat_messages = importlib.import_module('open_webui.models.chat_messages')
    middleware = importlib.import_module('open_webui.utils.middleware')

    makers = {}
    monkeypatch.setattr(chats_model, 'get_async_db_context', _sqlite_session_context(makers))
    monkeypatch.setattr(chat_messages, 'get_async_db_context', _sqlite_session_context(makers))

    async def read_rows():
        async with makers['session']() as session:
            chat_item = await session.get(chats_model.Chat, 'chat')
            message_item = await session.get(chat_messages.ChatMessage, 'chat-assistant')
            return (
                copy.deepcopy(chat_item.chat),
                chat_item.updated_at,
                chat_item.current_message_id,
                chat_messages.ChatMessageModel.model_validate(message_item).model_dump(),
            )

    async def run():
        engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "dualwrite.db"}')
        makers['session'] = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with engine.begin() as connection:
                await connection.run_sync(chats_model.Chat.__table__.create)
                await connection.run_sync(chat_messages.ChatMessage.__table__.create)

            output = [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'OLD'}]}]
            await chats_model.Chats.insert_new_chat(
                'chat',
                'user-1',
                chats_model.ChatForm(
                    chat={
                        'history': {
                            'currentId': 'assistant',
                            'messages': {
                                'user': {
                                    'id': 'user',
                                    'parentId': None,
                                    'role': 'user',
                                    'content': 'question',
                                    'childrenIds': ['assistant'],
                                },
                                'assistant': {
                                    'id': 'assistant',
                                    'parentId': 'user',
                                    'role': 'assistant',
                                    'content': 'OLD',
                                    'output': output,
                                    'done': True,
                                },
                            },
                        }
                    }
                ),
            )
            for tokens in (60, 90):
                await chats_model.Chats.upsert_message_to_chat_by_id_and_message_id(
                    'chat',
                    'assistant',
                    {
                        'usage': {
                            'prompt_tokens': tokens - 10,
                            'completion_tokens': 10,
                            'total_tokens': tokens,
                        }
                    },
                )
            before = await read_rows()
            assert before[3]['usage']['total_tokens'] == 150
            assert before[0]['history']['messages']['assistant']['usage']['total_tokens'] == 90

            real_upsert = chats_model.Chats.upsert_message_to_chat_by_id_and_message_id
            real_snapshot_update = getattr(chats_model.Chats, 'update_message_context_usage', None)

            state, events, upsert_calls, snapshot_calls = await _run_dualwrite_send_commit_leg(
                monkeypatch, middleware, chats_model, real_upsert, real_snapshot_update
            )

            snapshot = {
                'tokens': 60,
                'threshold': 100,
                'soft_threshold': None,
                'source': 'estimated',
            }
            assert state['context_usage'] == snapshot
            after = await read_rows()
            assert after[3] == before[3], 'the send commit must not replay saved usage'
            expected_blob = copy.deepcopy(before[0])
            expected_blob['history']['messages']['assistant']['context_usage'] = snapshot
            assert after[0] == expected_blob
            assert after[1:3] == before[1:3]
            assert _numeric_snapshots(events) == [snapshot]
            assert not any(call['update'].keys() == {'context_usage'} for call in upsert_calls)
            assert snapshot_calls == [snapshot]

            if real_snapshot_update is not None:
                refreshed = {**snapshot, 'tokens': 45}
                await real_snapshot_update('chat', 'assistant', refreshed)
                assert (await read_rows())[3]['usage']['total_tokens'] == 150

            await chats_model.Chats.upsert_message_to_chat_by_id_and_message_id(
                'chat',
                'assistant',
                {
                    'usage': {
                        'prompt_tokens': 20,
                        'completion_tokens': 10,
                        'total_tokens': 30,
                    }
                },
            )
            assert (await read_rows())[3]['usage']['total_tokens'] == 180
        finally:
            await engine.dispose()

    asyncio.run(run())


@pytest.mark.parametrize('code_interpreter', [False, True])
@pytest.mark.parametrize('old_text', ['OLD', ''])
def test_continue_compaction_keeps_joined_text_outside_checkpoint(monkeypatch, code_interpreter, old_text):
    middleware = importlib.import_module('open_webui.utils.middleware')
    previous = {
        'id': 'assistant',
        'role': 'assistant',
        'content': old_text,
        'output': [
            {
                'id': 'old',
                'type': 'message',
                'role': 'assistant',
                'status': 'completed',
                'content': [{'type': 'output_text', 'text': old_text}],
            }
        ],
    }
    current, tools = _adoption_stream_harness(monkeypatch, middleware, summaries=['S'], existing_message=previous)
    metadata = {
        'chat_id': 'chat',
        'message_id': 'assistant',
        'assistant_message_id': 'assistant',
        'params': {'tool_approval_mode': 'full', 'function_calling': 'legacy' if code_interpreter else 'native'},
        'tools': tools,
        'features': {'code_interpreter': code_interpreter},
    }
    ctx = _adoption_stream_ctx(tools, metadata)
    ctx['assistant_message'] = copy.deepcopy(previous)
    ctx['event_emitter'] = current['emitter']
    if old_text:
        ctx['form_data']['messages'].append({'role': 'assistant', 'content': old_text})
    current['replies'] = [current['text_response']('DONE')]
    summarized = []

    async def config_get(key, default=None):
        return True if key == 'code_interpreter.enable' else default

    monkeypatch.setattr(middleware.Config, 'get', config_get)

    async def summarize(_request, _user, _model_id, _models, compacted, recent, *args, **kwargs):
        summarized.append((copy.deepcopy(compacted), copy.deepcopy(recent)))
        return 'S'

    monkeypatch.setattr(compaction, '_generate_summary', summarize)

    async def chunks():
        # The provider's stop sequence excludes the closing code tag.
        text = 'NEW<code_interpreter type="code">print(1)' if code_interpreter else 'NEW'
        yield f'data: {json.dumps({"choices": [{"delta": {"content": text}}]})}\n\n'.encode()
        if not code_interpreter:
            async for chunk in current['tool_response']('call-a', 'view_file').body_iterator:
                yield chunk
        else:
            yield b'data: [DONE]\n\n'

    asyncio.run(
        middleware.streaming_chat_response_handler(StreamingResponse(chunks(), media_type='text/event-stream'), ctx)
    )
    assert len(summarized) == 1
    assert [item.get('content') for item in summarized[0][0]] == ['start']
    assert summarized[0][1][0]['content'].startswith(old_text or 'NEW')
    assert len(current['sent']) == 1
    assert not any(event.get('type') == 'chat:message:error' for event in current['emitted'])
    final_output = current['saved'][-1]['output']
    assert final_output[0]['id'] == 'old'
    assert final_output[0]['contextSummary'] == 'S'
    assert ''.join(part['text'] for part in final_output[0]['content']) == old_text + 'NEW'
    replay = compaction._messages_from_checkpoint([{'role': 'assistant', 'output': final_output}], 0, 0)
    assert replay[0]['output'][0] == final_output[0]


def test_terminal_instructions_survive_compaction_and_count_towards_budget(monkeypatch):
    terminals = importlib.import_module('open_webui.utils.terminals')
    measured, summarized = [], []
    instructions = '# AGENTS.md\n\nKeep these instructions verbatim.'
    messages = terminals.add_terminal_agents_md(
        [
            {'role': 'system', 'content': 'SYSTEM'},
            {'role': 'user', 'content': 'old request'},
            {'role': 'assistant', 'content': 'old answer'},
            {'role': 'user', 'content': 'new request'},
        ],
        instructions,
    )

    def estimate(body, **kwargs):
        measured.append(copy.deepcopy(body['messages']))
        return (
            20 if any(str(m.get('content')).startswith('<auto_compaction_context>') for m in body['messages']) else 120
        )

    async def summarize(_request, _user, _model_id, _models, compacted, recent, *args, **kwargs):
        summarized.append(copy.deepcopy(compacted))
        return 'S'

    monkeypatch.setattr(compaction, 'estimate_provider_tokens', estimate)
    monkeypatch.setattr(compaction, '_generate_summary', summarize)
    state = {'config': _compaction_config(soft_trigger_ratio=0)}
    body = asyncio.run(
        compaction.compact_transient_provider_payload(
            None,
            None,
            {'messages': messages},
            {},
            'model',
            {},
            state,
        )
    )
    assert body['messages'][:2] == messages[:2]
    assert all(any(m.get('content') == instructions for m in candidate) for candidate in measured)
    assert all(m.get('content') != instructions for prefix in summarized for m in prefix)
    # Early no-compaction passes must not discard the marker needed by later passes.
    body = asyncio.run(compaction.compact_provider_payload(None, None, body, {}, 'model', {}, state))
    assert body['messages'][1][compaction.CONTEXT_COMPACTION_PREFIX_MARKER_KEY] is True
    stripped = compaction.strip_compaction_marker_keys(body['messages'])
    assert stripped[1] == {'role': 'user', 'content': instructions}
    assert compaction.estimate_body_tokens({'messages': [stripped[1]]}) > 0
    transient = {'role': 'user', 'content': 'RAG', compaction.CONTEXT_COMPACTION_TRANSIENT_MARKER_KEY: True}
    assert compaction._split_leading_system_messages([transient]) == ([], [transient])


def test_svg_attachments_remain_text_files_for_compaction():
    middleware = importlib.import_module('open_webui.utils.middleware')
    svg = {'id': 'svg', 'content_type': 'image/svg+xml', 'url': '/svg'}
    png = {'id': 'png', 'content_type': 'image/png', 'url': '/png'}
    message = {'role': 'user', 'content': 'files', 'files': [svg, png]}
    assert compaction._non_image_files([message]) == [svg]
    projected = middleware.inject_message_file_images([message])[0]
    assert projected['content'] == [
        {'type': 'text', 'text': 'files'},
        {'type': 'image_url', 'image_url': {'url': '/png'}},
    ]
