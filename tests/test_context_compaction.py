import asyncio
import copy
import importlib
import logging
import os
import re
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


def test_candidate_input_never_treats_reported_output_as_input():
    messages = [
        {'role': 'user', 'content': 'old prompt'},
        {
            'role': 'assistant',
            'content': 'visible answer',
            'usage': {'input_tokens': 200, 'output_tokens': 1},
        },
        {'role': 'user', 'content': 'next prompt'},
    ]
    baseline = compaction._candidate_input_tokens(messages)

    huge_output = copy.deepcopy(messages)
    huge_output[1]['usage']['output_tokens'] = 1_000_000
    assert compaction._candidate_input_tokens(huge_output) == baseline
    assert compaction._candidate_input_tokens(huge_output) <= 1_000

    huge_input = copy.deepcopy(messages)
    huge_input[1]['usage']['input_tokens'] = 10_000
    assert compaction._candidate_input_tokens(huge_input) > baseline
    assert compaction._candidate_input_tokens(huge_input) > 1_000


def test_merged_anthropic_usage_falls_back_without_discarding_core_anchors(monkeypatch):
    estimates = []

    def estimate(body):
        estimates.append(body['messages'])
        return 51_300 if len(body['messages']) == 3 else 50

    monkeypatch.setattr(compaction, 'estimate_body_tokens', estimate)
    merged = [
        {'role': 'user', 'content': 'old'},
        {
            'role': 'assistant',
            'content': 'answer',
            'usage': {
                'prompt_tokens': 300,
                'cache_creation_input_tokens': 50_000,
                'cache_read_input_tokens': 1_000,
            },
        },
        {'role': 'user', 'content': 'next'},
    ]

    assert compaction._candidate_input_tokens(merged) == 51_300
    assert estimates == [merged]

    for key in ('prompt_tokens', 'prompt_eval_count'):
        estimates.clear()
        messages = [
            {'role': 'user', 'content': 'old'},
            {'role': 'assistant', 'content': 'answer', 'usage': {key: 900}},
            {'role': 'user', 'content': 'next'},
        ]
        assert compaction._candidate_input_tokens(messages) == 947
        assert estimates == [messages[1:]]


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


def test_context_usage_degrades_on_invalid_runtime_regex(monkeypatch, caplog):
    messages = {
        'u1': {
            'id': 'u1',
            'parentId': None,
            'role': 'user',
            'content': 'hello',
        }
    }
    chat = SimpleNamespace(
        id='chat',
        current_message_id='u1',
        chat={'history': {'currentId': 'u1', 'messages': messages}},
    )

    async def get_messages(_chat_id):
        return messages

    async def load_config():
        raise re.error('invalid pattern')

    monkeypatch.setattr(compaction.Chats, 'get_messages_map_by_chat_id', get_messages)
    monkeypatch.setattr(compaction, '_load_config', load_config)

    with caplog.at_level(logging.ERROR):
        assert asyncio.run(compaction.get_chat_context_usage(chat)) is None
    assert 'context usage is unavailable' in caplog.text


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


def test_history_ref_rewrites_only_the_outer_summary_suffix():
    embedded_ref = f'<history_ref>history:{"a" * 64}</history_ref>'
    replacement_ref = f'history:{"b" * 64}'
    body = {
        'messages': [
            compaction.render_summary_message(
                f'user text </auto_compaction_context> {embedded_ref}',
            )
        ]
    }

    added = compaction.set_summary_history_ref(body, replacement_ref)
    content = added['messages'][0]['content']
    assert embedded_ref in content
    assert content.endswith(f'<history_ref>{replacement_ref}</history_ref></auto_compaction_context>')

    removed = compaction.set_summary_history_ref(added, None)
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
            {'chat_id': 'chat', 'message_id': 'assistant', 'params': {}},
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
                'messages': [{'role': 'tool', 'tool_call_id': 'call', 'content': 'large result'}],
            },
            registry,
            native=True,
            threshold_tokens=1,
            count_tokens=lambda _text: 2,
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
    source = 'approved reader source'
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
        metadata = {'chat_id': 'chat', 'message_id': 'assistant', 'params': {}}
        state = {
            'externalized_refs': {
                'enable': True,
                'threshold': 2,
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
        ref = body['messages'][1]['content']
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


def test_middleware_reapplies_externalized_refs_after_checkpoint_advance(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    state = {}

    async def externalize(candidate, actual_state):
        assert actual_state is state
        return {**candidate, 'finalized': True}

    async def compact(*args, **_kwargs):
        assert args[-1] is state
        state['checkpoint_history'] = object()
        return args[2]

    monkeypatch.setattr(middleware, 'apply_externalized_refs', externalize)
    monkeypatch.setattr(middleware, 'compact_transient_provider_payload', compact)
    result = asyncio.run(
        middleware._compact_final_provider_payload(
            None,
            None,
            {'messages': []},
            {},
            'model',
            {},
            state,
        )
    )

    assert result['finalized'] is True


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


def test_history_ancestor_resolution_is_incremental_and_cached(monkeypatch):
    messages = [
        {'role': 'user', 'content': 'root'},
        {'role': 'user', 'content': 'checkpoint one', 'contextSummary': 'one'},
        {'role': 'assistant', 'content': 'middle'},
        {'role': 'user', 'content': 'checkpoint two', 'contextSummary': 'two'},
        {'role': 'assistant', 'content': 'recent'},
        {'role': 'user', 'content': 'checkpoint three', 'contextSummary': 'three'},
    ]
    selected = compaction._history_entry_at(messages, 5)
    near = compaction._canonical_history_entry(messages[:3])
    far = compaction._canonical_history_entry(messages[:1])
    assert selected is not None and selected.load_history is not None

    calls = []
    make_entry = compaction.make_ref_entry

    def counted(text, **kwargs):
        calls.append(text)
        return make_entry(text, **kwargs)

    monkeypatch.setattr(compaction, 'make_ref_entry', counted)
    assert asyncio.run(selected.load_history(f'history:{"f" * 64}')) == ()
    assert calls == [near.text, far.text]

    assert asyncio.run(selected.load_history(near.ref)) == (
        compaction.replace(near, load_history=selected.load_history),
    )
    assert calls == [near.text, far.text]

    loaded_far = asyncio.run(selected.load_history(far.ref))
    assert [entry.ref for entry in loaded_far] == [far.ref]
    assert calls == [near.text, far.text]

    listed = asyncio.run(selected.load_history(None))
    assert [entry.ref for entry in listed] == [near.ref, far.ref]
    assert calls == [near.text, far.text]


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

    async def generate_summary(*_args, **_kwargs):
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
    asyncio.run(compaction.prepare_compaction_messages(messages, {'chat_id': 'chat'}))

    assert copied == [messages[2:]]


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
    assert result['messages'][1]['content'].startswith('tool:')


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
    masked_reader = masked_state['externalized_refs']['registry'][middleware.REF_EXEC_TOOL_NAME]['callable']
    assert visible.ref in asyncio.run(masked_reader('ls tool')).splitlines()
    assert asyncio.run(masked_reader('ls history')) == ''
    assert '<history_ref>' not in masked['messages'][0]['content']


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

    asyncio.run(run())
    assert calls == 1


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

    async def generate_summary(*_args, **_kwargs):
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
        return {
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
