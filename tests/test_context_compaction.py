import asyncio
import copy
import importlib
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
    assert not compaction._exceeds_token_threshold(huge_output, '', None, 1_000)

    huge_input = copy.deepcopy(messages)
    huge_input[1]['usage']['input_tokens'] = 10_000
    assert compaction._candidate_input_tokens(huge_input) > baseline
    assert compaction._exceeds_token_threshold(huge_input, '', None, 1_000)


def test_provider_estimate_uses_exact_input_anchor_plus_new_suffix():
    marker = compaction.CONTEXT_COMPACTION_USAGE_ANCHOR_KEY
    body = {
        'messages': [
            {'role': 'assistant', 'content': 'previous', marker: 1_000},
            {'role': 'user', 'content': 'new input'},
        ]
    }
    local = compaction.estimate_body_tokens(body)
    suffix = compaction.estimate_body_tokens({'messages': body['messages']})

    assert local is not None and suffix is not None
    assert compaction.estimate_provider_tokens(body) == max(local, 1_000 + suffix - 3)

    async def strip_internal_marker():
        stripped = await compaction.compact_provider_payload(
            None,
            None,
            body,
            {},
            'model',
            {},
            {'config': {'enable': False}},
        )
        assert marker not in stripped['messages'][0]

    asyncio.run(strip_internal_marker())


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

    monkeypatch.setattr(compaction.tiktoken, 'get_encoding', lambda _name=None: RecordingEncoder())
    large_text_tokens = compaction.estimate_body_tokens({'messages': [{'role': 'user', 'content': 'x' * 1_000_000}]})
    assert large_text_tokens > 100_000
    assert max(encoded_lengths) <= 16 * 1024


def test_safe_boundary_preserves_system_messages_and_complete_tool_rounds():
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
    assert compaction.find_safe_compaction_boundary(crossing_round, 50) == 0
    assert compaction.find_safe_compaction_boundary(working[:2] + working[4:], 50) == 0


def test_history_checkpoint_is_strict_and_branch_bound():
    prefix = [
        {
            'id': 'u1',
            'parentId': None,
            'role': 'user',
            'content': '',
            'meta': {'ui': True},
            'files': [{'id': 'file-1', 'name': 'notes.txt', 'signed_url': 'old'}],
        },
        {'id': 'a1', 'parentId': 'u1', 'role': 'assistant', 'content': 'answer', 'usage': {'input_tokens': 9}},
    ]
    descriptor, source = compaction._build_history_checkpoint(prefix, 'a1')
    carrier = {
        'id': 'u2',
        'parentId': 'a1',
        'role': 'user',
        'content': 'next',
        'contextSummary': 'summary',
        'meta': {'context_compaction': descriptor},
    }

    resolved = compaction._resolve_history_checkpoint([*prefix, carrier], 2)
    assert resolved.text == source.text
    assert resolved.ref == f'history:{descriptor["history_ref"]["raw_source_hash"]}'

    transient_file_change = copy.deepcopy(prefix)
    transient_file_change[0]['files'][0]['signed_url'] = 'new'
    assert (
        compaction._build_history_checkpoint(transient_file_change, 'a1')[0]['source_hash'] == descriptor['source_hash']
    )

    stable_file_change = copy.deepcopy(prefix)
    stable_file_change[0]['files'][0]['id'] = 'file-2'
    assert compaction._build_history_checkpoint(stable_file_change, 'a1')[0]['source_hash'] != descriptor['source_hash']

    bad_anchor = copy.deepcopy(carrier)
    bad_anchor['parentId'] = 'other'
    with pytest.raises(compaction.CanonicalHistoryError, match='branch anchor'):
        compaction._resolve_history_checkpoint([*prefix, bad_anchor], 2)


def test_checkpoint_identity_ignores_mutable_transient_patterns(monkeypatch):
    prefix = [
        {'id': 'u1', 'parentId': None, 'role': 'user', 'content': 'configured transient'},
        {'id': 'a1', 'parentId': 'u1', 'role': 'assistant', 'content': 'answer'},
    ]
    descriptor, source = compaction._build_history_checkpoint(prefix, 'a1')
    messages = [
        *prefix,
        {
            'id': 'u2',
            'parentId': 'a1',
            'role': 'user',
            'content': 'continue',
            'contextSummary': 'summary',
            'meta': {'context_compaction': descriptor},
        },
    ]

    async def prepare(patterns):
        async def load_config():
            return {
                'enable': True,
                'retention_percentage': 40,
                'transient_patterns': patterns,
            }

        monkeypatch.setattr(compaction, '_load_config', load_config)
        _, state = await compaction.prepare_compaction_messages(messages, {'chat_id': 'chat'})
        return state['selected_history']

    matching = asyncio.run(prepare((re.compile(r'^configured transient$'),)))
    changed = asyncio.run(prepare((re.compile(r'^something else$'),)))
    assert matching.ref == changed.ref == source.ref
    assert 'configured transient' in matching.text


def test_invalid_history_checkpoint_fails_before_provider_and_preserves_input(monkeypatch):
    prefix = [
        {'id': 'u1', 'parentId': None, 'role': 'user', 'content': 'original'},
        {'id': 'a1', 'parentId': 'u1', 'role': 'assistant', 'content': 'answer'},
    ]
    descriptor, _ = compaction._build_history_checkpoint(prefix, 'a1')
    messages = [
        {**prefix[0], 'content': 'tampered'},
        prefix[1],
        {
            'id': 'u2',
            'parentId': 'a1',
            'role': 'user',
            'content': 'continue',
            'contextSummary': 'summary',
            'meta': {'context_compaction': descriptor},
        },
    ]
    original = copy.deepcopy(messages)
    provider_calls = 0

    async def load_config():
        return {
            'enable': True,
            'retention_percentage': 40,
            'transient_patterns': (),
        }

    async def prepare_then_send():
        nonlocal provider_calls
        prepared, _ = await compaction.prepare_compaction_messages(messages, {'chat_id': 'chat'})
        provider_calls += 1
        return prepared

    monkeypatch.setattr(compaction, '_load_config', load_config)
    with pytest.raises(compaction.CanonicalHistoryError, match='source hash'):
        asyncio.run(prepare_then_send())

    assert provider_calls == 0
    assert messages == original


def test_official_v011_summary_without_ref_metadata_remains_usable(monkeypatch):
    messages = [
        {'id': 'u1', 'parentId': None, 'role': 'user', 'content': 'old'},
        {
            'id': 'u2',
            'parentId': 'u1',
            'role': 'user',
            'content': 'continue',
            'contextSummary': 'official summary',
        },
    ]

    async def load_config():
        return {
            'enable': True,
            'retention_percentage': 40,
            'transient_patterns': (),
        }

    monkeypatch.setattr(compaction, '_load_config', load_config)
    prepared, state = asyncio.run(compaction.prepare_compaction_messages(messages, {'chat_id': 'chat'}))

    assert prepared[0]['content'].startswith('<auto_compaction_context>')
    assert prepared[1]['id'] == 'u2'
    assert 'selected_history' not in state


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
        '_resolve_history_checkpoint',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('history must not resolve twice')),
    )
    replayed = asyncio.run(compaction.replay_cached_compaction_messages(messages, state))

    assert replayed[0]['content'].startswith('<auto_compaction_context>')
    assert [message.get('id') for message in replayed[1:]] == ['u2', 'a2']


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

    async def dispatch():
        return await compaction.compact_provider_payload(
            None,
            None,
            {'messages': stripped},
            {},
            'model',
            {},
            {'config': {'enable': False}},
        )

    body = asyncio.run(dispatch())
    assert all(marker not in message for message in body['messages'])


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


def test_file_context_sidecar_restores_exact_user_after_filter_normalization():
    middleware = importlib.import_module('open_webui.utils.middleware')
    marker = compaction.CONTEXT_COMPACTION_TRANSIENT_MARKER_KEY
    captured = ([], [{'id': 'file-2', 'url': 'https://example.test/two'}])
    filtered = [
        {'role': 'user', 'content': 'one'},
        {'role': 'user', 'content': 'internal', marker: True},
        {'role': 'assistant', 'content': 'answer'},
        {'role': 'user', 'content': 'two'},
    ]

    restored = middleware.restore_message_files(filtered, captured)
    assert 'files' not in restored[0]
    assert 'files' not in restored[1]
    assert restored[3]['files'] == captured[1]

    with pytest.raises(RuntimeError, match='filter changed user history'):
        middleware.restore_message_files([{'role': 'user', 'content': 'one'}], captured)


def test_file_sidecar_counts_synthetic_tool_image_users():
    middleware = importlib.import_module('open_webui.utils.middleware')
    files = [{'id': 'file-1', 'url': 'https://example.test/file'}]
    messages = [
        {'role': 'user', 'content': 'inspect', 'files': files},
        {
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

    processed = middleware.process_messages_with_output(messages, preserve_user_files=True)
    captured = tuple(
        copy.deepcopy(message.get('files') or []) for message in processed if message.get('role') == 'user'
    )
    for message in processed:
        message.pop('files', None)
    restored = middleware.restore_message_files(processed, captured)

    user_messages = [message for message in restored if message.get('role') == 'user']
    assert captured == (files, [])
    assert user_messages[0]['files'] == files
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


def test_approved_reader_rebuilds_catalog_and_keeps_its_result_literal(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
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
                'threshold': 1,
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
                'name': middleware.REF_EXEC_TOOL_NAME,
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


def test_canonical_history_preserves_empty_shape_and_rejects_unknown_semantics():
    empty_string = [{'role': 'user', 'content': ''}]
    empty_parts = [{'role': 'user', 'content': []}]
    assert compaction._canonical_history_source(empty_string)[1] != compaction._canonical_history_source(empty_parts)[1]

    with pytest.raises(compaction.CanonicalHistoryError, match='unknown message shape'):
        compaction._canonical_history_source([{'role': 'user', 'content': 'hello', 'untracked': True}])

    with pytest.raises(compaction.CanonicalHistoryError, match='unknown content shape'):
        compaction._canonical_history_source(
            [{'role': 'user', 'content': [{'type': 'text', 'text': 'hello', 'cache_control': {}}]}]
        )


def test_responses_output_accepts_builtin_calls_and_text_annotations():
    output = [
        {'type': 'web_search_call', 'id': 'search-1', 'status': 'completed'},
        {'type': 'file_search_call', 'id': 'search-2', 'status': 'completed'},
        {'type': 'computer_call', 'id': 'computer-1', 'status': 'completed'},
        {
            'type': 'message',
            'role': 'assistant',
            'content': [
                {
                    'type': 'output_text',
                    'text': 'answer',
                    'annotations': [{'type': 'url_citation', 'url': 'https://example.test'}],
                }
            ],
        },
    ]

    assert compaction._canonical_output_messages(output) == [{'role': 'assistant', 'content': 'answer'}]


def test_compaction_cpu_work_is_dispatched_off_the_event_loop(monkeypatch):
    calls = []
    prefix = [
        {'id': 'u1', 'parentId': None, 'role': 'user', 'content': 'old'},
        {'id': 'a1', 'parentId': 'u1', 'role': 'assistant', 'content': 'answer'},
    ]
    descriptor, _ = compaction._build_history_checkpoint(prefix, 'a1')
    messages = [
        *prefix,
        {
            'id': 'u2',
            'parentId': 'a1',
            'role': 'user',
            'content': 'continue',
            'contextSummary': 'summary',
            'meta': {'context_compaction': descriptor},
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
        await compaction.prepare_compaction_messages(messages, {'chat_id': 'chat'})
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
        await compaction._snapshot_retry_body({'messages': []})

    monkeypatch.setattr(compaction.asyncio, 'to_thread', to_thread)
    monkeypatch.setattr(compaction, '_load_config', load_config)
    asyncio.run(run())

    assert copy.deepcopy in calls
    assert compaction._resolve_history_checkpoint in calls
    assert compaction.estimate_provider_tokens in calls


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
            }
        )
    )

    assert captured == [{'id': 'a1', 'role': 'assistant', 'content': 'answer', 'model': 'model'}]


def test_soft_prefetch_is_reused_at_the_hard_threshold(monkeypatch):
    calls = 0
    before = 60
    history_entry = compaction.make_ref_entry('history', kind='history')
    assert history_entry is not None

    async def create_checkpoint(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return (
            'summary',
            {'historical_user_messages': [], 'history_ref': {'raw_source_hash': history_entry.ref.split(':')[1]}},
            history_entry,
        )

    def estimate(body, **_kwargs):
        if any(
            isinstance(message.get('content'), str) and message['content'].startswith('<auto_compaction_context>')
            for message in body['messages']
        ):
            return 20
        return before

    monkeypatch.setattr(compaction, '_create_checkpoint', create_checkpoint)
    monkeypatch.setattr(compaction, 'estimate_body_tokens', estimate)
    body = {
        'messages': [
            {'role': 'user', 'content': 'old'},
            {'role': 'assistant', 'content': 'answer'},
            {'role': 'user', 'content': 'new', compaction._BOUNDARY_KEY: True},
        ]
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
        }
    }

    async def run():
        nonlocal before
        first = await compaction.compact_provider_payload(None, None, body, {}, 'model', {}, state)
        assert compaction._BOUNDARY_KEY not in first['messages'][2]
        await state['prefetch_task']
        before = 120
        second = await compaction.compact_provider_payload(None, None, first, {}, 'model', {}, state)
        assert second['messages'][0]['content'].startswith('<auto_compaction_context>')

    asyncio.run(run())
    assert calls == 1


def test_provider_overflow_can_force_one_smaller_compaction_below_threshold(monkeypatch):
    history_entry = compaction.make_ref_entry('history', kind='history')
    assert history_entry is not None
    events = []

    async def get_event_emitter(_metadata):
        async def emit(event):
            events.append(event)

        return emit

    async def create_checkpoint(*_args, **_kwargs):
        return 'summary', {'historical_user_messages': []}, history_entry

    def estimate(body, **_kwargs):
        return (
            20
            if any(
                isinstance(message.get('content'), str) and message['content'].startswith('<auto_compaction_context>')
                for message in body['messages']
            )
            else 60
        )

    monkeypatch.setattr(compaction, '_create_checkpoint', create_checkpoint)
    monkeypatch.setattr(compaction, 'estimate_body_tokens', estimate)
    socket_main = importlib.import_module('open_webui.socket.main')
    monkeypatch.setattr(socket_main, 'get_event_emitter', get_event_emitter)
    body = {
        'messages': [
            {'role': 'user', 'content': 'old'},
            {'role': 'assistant', 'content': 'answer'},
            {'role': 'user', 'content': 'new', compaction._BOUNDARY_KEY: True},
        ]
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
        }
    }

    compacted = asyncio.run(
        compaction.compact_provider_payload(
            None,
            None,
            body,
            {'chat_id': 'chat', 'message_id': 'message'},
            'model',
            {},
            state,
            force=True,
        )
    )

    assert state['compacted'] is True
    assert compacted['messages'][0]['content'].startswith('<auto_compaction_context>')
    assert [event['data']['description'] for event in events] == [
        'Compacting context',
        'Context compacted',
    ]


def test_stateful_continuation_sends_current_output_and_retries_full_history(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    captured = None
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

    async def compact(_request, _user, candidate, *_args, **_kwargs):
        nonlocal captured
        captured = candidate
        state['compacted'] = True
        return candidate

    monkeypatch.setattr(middleware, 'compact_provider_payload', compact)
    state = {'config': {'enable': True}}
    result = asyncio.run(
        middleware.prepare_context_overflow_retry(
            None,
            None,
            body,
            {},
            'model',
            state,
        )
    )

    assert result is captured
    assert 'previous_response_id' not in captured
    assert captured['messages'] == body['messages']


def test_compaction_summarizes_raw_tool_text_while_thresholding_projected_payload(monkeypatch):
    history_entry = compaction.make_ref_entry('history', kind='history')
    assert history_entry is not None
    captured = None

    async def create_checkpoint(
        _request,
        _user,
        _model_id,
        _models,
        _metadata,
        _state,
        _config,
        compacted_messages,
        _recent_messages,
    ):
        nonlocal captured
        captured = compacted_messages
        return 'summary', {'historical_user_messages': []}, history_entry

    def estimate(body, **_kwargs):
        return (
            20
            if any(
                isinstance(message.get('content'), str) and message['content'].startswith('<auto_compaction_context>')
                for message in body['messages']
            )
            else 120
        )

    monkeypatch.setattr(compaction, '_create_checkpoint', create_checkpoint)
    monkeypatch.setattr(compaction, 'estimate_body_tokens', estimate)
    projected = [
        {'role': 'user', 'content': 'old'},
        {'role': 'tool', 'tool_call_id': 'call', 'content': 'tool:' + 'a' * 64},
        {'role': 'user', 'content': 'new', compaction._BOUNDARY_KEY: True},
    ]
    raw = copy.deepcopy(projected)
    raw[1]['content'] = 'full tool output needed by the summary'
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
        'projection_source_messages': raw,
    }

    asyncio.run(
        compaction.compact_provider_payload(
            None,
            None,
            {'messages': projected},
            {},
            'model',
            {},
            state,
        )
    )

    assert captured[1]['content'] == 'full tool output needed by the summary'


def test_native_tool_continuations_accumulate_each_round_once(monkeypatch):
    middleware = importlib.import_module('open_webui.utils.middleware')
    sent = []

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

    replies = [
        tool_response('call-b', 'tool_b'),
        response({'choices': [{'delta': {'content': 'done'}, 'finish_reason': 'stop'}]}),
    ]

    async def generate(_request, candidate, _user, **_kwargs):
        sent.append(copy.deepcopy(candidate))
        return replies.pop(0)

    async def process_result(_request, _name, result, *_args):
        return result, [], []

    async def noop(*_args, **_kwargs):
        return None

    async def tool_a():
        return 'result a'

    async def tool_b():
        return 'result b'

    tools = {
        name: {
            'spec': {'name': name, 'parameters': {'type': 'object', 'properties': {}}},
            'callable': function,
        }
        for name, function in (('tool_a', tool_a), ('tool_b', tool_b))
    }
    metadata = {
        'chat_id': '',
        'message_id': 'message',
        'params': {'tool_approval_mode': 'full'},
        'tools': tools,
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
        'model': {'id': 'model', 'info': {'meta': {'capabilities': {'citations': False}}}},
        'metadata': metadata,
        'events': [],
        'tasks': {},
        'event_emitter': noop,
        'event_caller': None,
        'compaction_state': {
            'config': {'enable': False},
            'externalized_refs': {
                'enable': True,
                'threshold': 1,
                'registry': tools,
                'native': True,
                'metadata': metadata,
            },
        },
    }

    monkeypatch.setattr(middleware, 'ENABLE_RESPONSES_API_STATEFUL', False)
    monkeypatch.setattr(middleware, 'generate_chat_completion', generate)
    monkeypatch.setattr(middleware, 'process_tool_result', process_result)
    monkeypatch.setattr(middleware, 'terminal_event_handler', noop)
    monkeypatch.setattr(middleware, 'get_system_oauth_token', noop)
    monkeypatch.setattr(middleware, 'outlet_filter_handler', noop)
    monkeypatch.setattr(middleware, 'background_tasks_handler', noop)
    monkeypatch.setattr(middleware, 'clear_response_stream', noop)
    monkeypatch.setattr(middleware, 'publish_chat_finished_event', noop)

    asyncio.run(middleware.streaming_chat_response_handler(tool_response('call-a', 'tool_a'), ctx))

    first_roles = [message.get('role') for message in sent[0]['messages']]
    second_ids = [message.get('tool_call_id') for message in sent[1]['messages'] if message.get('role') == 'tool']
    refs = {
        message['tool_call_id']: message['content'] for message in sent[1]['messages'] if message.get('role') == 'tool'
    }
    reader = tools[middleware.REF_EXEC_TOOL_NAME]['callable']
    assert first_roles == ['user', 'assistant', 'tool']
    assert second_ids == ['call-a', 'call-b']
    assert asyncio.run(reader(f'wc -c {refs["call-a"]}')) == str(len('result a'))
    assert asyncio.run(reader(f'wc -c {refs["call-b"]}')) == str(len('result b'))


def test_transient_messages_use_core_provenance_before_regex_fallback():
    patterns = (re.compile(r'^injected:'),)
    assert compaction._is_transient_message(
        {'role': 'user', 'content': 'ordinary', 'meta': {'internal': True, 'type': 'subagent'}},
        patterns,
    )
    assert compaction._is_transient_message({'role': 'user', 'content': '  injected:value'}, patterns)
    assert not compaction._is_transient_message({'role': 'user', 'content': 'ordinary'}, patterns)
