import asyncio
import copy
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

from fastapi.responses import StreamingResponse

for name in ('data', 'static'):
    path = Path(f'/tmp/open-webui-chat-routing-{name}')
    path.mkdir(parents=True, exist_ok=True)
    os.environ[name.upper() + '_DIR'] = str(path)
os.environ.setdefault('WEBUI_SECRET_KEY', 'local-test-only')

main = importlib.import_module('open_webui.main')
middleware = importlib.import_module('open_webui.utils.middleware')
chat = importlib.import_module('open_webui.utils.chat')
compaction = importlib.import_module('open_webui.utils.context_compaction')


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


def _install_provider_sink_harness(monkeypatch, model_infos, config_values=None):
    """Keep process_chat_payload and chat.py REAL; stub only the provider sink.

    The sink sits behind chat.py's request.state.metadata merge, so captures
    prove per-call values actually reach the provider.
    """
    captured = {}
    seen_keys = []
    config_values = {
        'models.default_params': {},
        'chat.tool_permissions.enable': False,
        'ui.default_models': 'fallback-model',
        **(config_values or {}),
    }

    async def openai_sink(request=None, form_data=None, user=None):
        metadata = form_data['metadata']
        captured[metadata['message_id']] = {
            'model': form_data['model'],
            'stream': form_data.get('stream', '<absent>'),
            'stream_options': form_data.get('stream_options', '<absent>'),
            'params': copy.deepcopy(metadata.get('params')),
            'params_ref': metadata.get('params'),
            'options_ref': form_data.get('stream_options'),
            'system_prompt': metadata.get('system_prompt'),
            'messages': copy.deepcopy(form_data.get('messages')),
        }
        if form_data.get('stream'):
            # chat.py's arena streaming wrapper reads body_iterator.
            async def chunks():
                yield b'data: [DONE]\n\n'

            return StreamingResponse(chunks(), media_type='text/event-stream')
        return {'choices': [{'message': {'content': 'ok'}}]}

    async def build_ctx(_request, form_data, _user, model, metadata, *_args, **_kwargs):
        entry = captured.setdefault(
            metadata['message_id'],
            {'model': None, 'stream': None, 'stream_options': None, 'params': None},
        )
        entry['ctx_model_id'] = (model or {}).get('id')
        return {}

    async def respond(*_args, **_kwargs):
        return {'status': True}

    async def fake_create_task(_redis, process, id=None, task_id=None):
        await process
        return task_id, None

    async def fake_noop(*_args, **_kwargs):
        return None

    async def config_get(key, default=None):
        seen_keys.append(key)
        return config_values.get(key, default)

    async def get_model(model_id):
        return model_infos.get(model_id)

    async def get_many_empty(*_keys):
        return {}

    monkeypatch.setattr(chat, 'generate_openai_chat_completion', openai_sink)
    monkeypatch.setattr(main, 'build_chat_response_context', build_ctx)
    monkeypatch.setattr(main, 'process_chat_response', respond)
    monkeypatch.setattr(main, 'create_task', fake_create_task)
    monkeypatch.setattr(main, 'get_event_emitter', fake_noop)
    monkeypatch.setattr(main, 'cleanup_task', fake_create_task)
    monkeypatch.setattr(main, 'has_active_tasks', fake_noop)
    monkeypatch.setattr(main.Config, 'get', staticmethod(config_get))
    monkeypatch.setattr(main.Models, 'get_model_by_id', staticmethod(get_model))
    monkeypatch.setattr(main, 'check_model_access', fake_noop)
    monkeypatch.setattr(main, 'BYPASS_MODEL_ACCESS_CONTROL', False)
    monkeypatch.setattr(main, 'BYPASS_ADMIN_ACCESS_CONTROL', True)
    monkeypatch.setattr(main, 'ENABLE_CUSTOM_MODEL_FALLBACK', True)
    monkeypatch.setattr(middleware.Config, 'get_many', staticmethod(get_many_empty))

    request = SimpleNamespace(
        headers={},
        state=SimpleNamespace(internal=False),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={}, redis=None)),
    )
    user = SimpleNamespace(id='user-1', role='admin', email='u@test', name='u')
    return captured, request, user, seen_keys


def _run_chat_completion(request, user, form_data):
    result = asyncio.run(main.chat_completion(request, dict(form_data), user))
    assert result['status'] is True


def _base_form_data(model, message_ids, **overrides):
    form_data = {
        'model': model,
        'stream': True,
        'messages': [{'role': 'user', 'content': 'hi'}],
        'session_id': 'sess-1',
        'chat_id': 'local:unit',
        'message_ids': message_ids,
    }
    form_data.update(overrides)
    return form_data


def test_fanout_sibling_controls_reach_provider(monkeypatch):
    models_map = {
        'primary-model': {'id': 'primary-model', 'owned_by': 'openai', 'info': {}},
        'sibling-model': {'id': 'sibling-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos = {
        'primary-model': _EntryModelInfo(
            {'function_calling': 'native', 'compact_token_threshold': 80000, 'stream_response': True}
        ),
        'sibling-model': _EntryModelInfo(
            {'function_calling': 'legacy', 'compact_token_threshold': 32000, 'stream_response': False}
        ),
    }
    captured, request, user, seen_keys = _install_provider_sink_harness(monkeypatch, model_infos)
    request.app.state.MODELS = models_map

    _run_chat_completion(
        request,
        user,
        _base_form_data(
            'primary-model',
            [
                {'model_id': 'primary-model', 'message_id': 'm1'},
                {'model_id': 'sibling-model', 'message_id': 'm2'},
            ],
        ),
    )

    primary, sibling = captured['m1'], captured['m2']
    assert sibling['model'] == 'sibling-model'
    assert sibling['stream'] is False
    assert sibling['stream_options'] == '<absent>'
    assert sibling['params']['compact_token_threshold'] == 32000
    assert sibling['params']['function_calling'] == 'legacy'
    assert primary['stream'] is True
    assert primary['params']['compact_token_threshold'] == 80000
    assert primary['params']['function_calling'] == 'native'

    # The compaction threshold reader consumes exactly this per-call block.
    assert (
        compaction._resolve_token_threshold(80000, 80000, {'params': sibling['params']}) == 32000
    )

    assert sibling['params_ref'] is not primary['params_ref']
    assert sibling['params_ref'] is not request.state.metadata['params']


def test_fanout_target_without_values_falls_to_core_defaults(monkeypatch):
    models_map = {
        'primary-model': {'id': 'primary-model', 'owned_by': 'openai', 'info': {}},
        'bare-model': {'id': 'bare-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos = {
        'primary-model': _EntryModelInfo({'compact_token_threshold': 80000, 'reasoning_tags': ['p', 'q']}),
        'bare-model': _EntryModelInfo({}),
    }
    captured, request, user, seen_keys = _install_provider_sink_harness(monkeypatch, model_infos)
    request.app.state.MODELS = models_map

    _run_chat_completion(
        request,
        user,
        _base_form_data(
            'primary-model',
            [
                {'model_id': 'primary-model', 'message_id': 'm1'},
                {'model_id': 'bare-model', 'message_id': 'm2'},
            ],
        ),
    )

    bare = captured['m2']['params']
    assert bare['compact_token_threshold'] is None
    assert bare['reasoning_tags'] is None
    assert bare['stream_delta_chunk_size'] is None
    assert bare['function_calling'] == 'native'


def test_caller_stream_restored_per_column(monkeypatch):
    models_map = {
        'primary-model': {'id': 'primary-model', 'owned_by': 'openai', 'info': {}},
        'sibling-model': {'id': 'sibling-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos = {
        'primary-model': _EntryModelInfo({'stream_response': True}),
        'sibling-model': _EntryModelInfo({}),
    }
    captured, request, user, seen_keys = _install_provider_sink_harness(monkeypatch, model_infos)
    request.app.state.MODELS = models_map
    columns = [
        {'model_id': 'primary-model', 'message_id': 'm1'},
        {'model_id': 'sibling-model', 'message_id': 'm2'},
    ]

    _run_chat_completion(
        request,
        user,
        _base_form_data('primary-model', columns, stream=False),
    )
    assert captured['m1']['stream'] is True
    assert captured['m2']['stream'] is False

    captured.clear()
    form_data = _base_form_data('primary-model', columns)
    del form_data['stream']
    _run_chat_completion(request, user, form_data)
    assert captured['m1']['stream'] is True
    assert captured['m2']['stream'] == '<absent>'


def test_stream_options_rebuilt_per_column(monkeypatch):
    models_map = {
        'primary-model': {
            'id': 'primary-model',
            'owned_by': 'openai',
            'info': {'meta': {'capabilities': {'usage': True}}},
        },
        'sibling-model': {'id': 'sibling-model', 'owned_by': 'openai', 'info': {}},
        'capable-sibling': {
            'id': 'capable-sibling',
            'owned_by': 'openai',
            'info': {'meta': {'capabilities': {'usage': True}}},
        },
    }
    model_infos = {
        'primary-model': _EntryModelInfo({'stream_response': True}),
        'sibling-model': _EntryModelInfo({'stream_response': True}),
        'capable-sibling': _EntryModelInfo({'stream_response': True}),
    }
    captured, request, user, seen_keys = _install_provider_sink_harness(monkeypatch, model_infos)
    request.app.state.MODELS = models_map
    original_options = {'custom': 1}

    _run_chat_completion(
        request,
        user,
        _base_form_data(
            'primary-model',
            [
                {'model_id': 'primary-model', 'message_id': 'm1'},
                {'model_id': 'sibling-model', 'message_id': 'm2'},
                {'model_id': 'capable-sibling', 'message_id': 'm3'},
            ],
            stream_options=original_options,
        )
    )

    assert captured['m2']['stream_options'] == {'custom': 1}
    assert captured['m3']['stream_options'] == {'custom': 1, 'include_usage': True}
    assert captured['m1']['stream_options'] == {'custom': 1, 'include_usage': True}
    # Every column gets its own copy — including one distinct from the
    # caller's original dict object.
    assert captured['m2']['options_ref'] is not original_options
    assert captured['m1']['options_ref'] is not original_options
    assert captured['m3']['options_ref'] is not original_options
    assert captured['m1']['options_ref'] is not captured['m3']['options_ref']

    # A caller-explicit include_usage stays intact on a non-capability column.
    captured.clear()
    _run_chat_completion(
        request,
        user,
        _base_form_data(
            'primary-model',
            [
                {'model_id': 'primary-model', 'message_id': 'm1'},
                {'model_id': 'sibling-model', 'message_id': 'm2'},
            ],
            stream_options={'include_usage': True, 'custom': 1},
        ),
    )

    assert captured['m2']['stream_options'] == {'include_usage': True, 'custom': 1}
    assert captured['m1']['stream_options'] == {'include_usage': True, 'custom': 1}


def test_fallback_primary_uses_custom_capabilities_for_include_usage(monkeypatch):
    models_map = {
        'custom-model': {
            'id': 'custom-model',
            'owned_by': 'openai',
            'info': {'meta': {'capabilities': {'usage': True}}},
        },
        'fallback-model': {'id': 'fallback-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos = {
        'custom-model': _EntryModelInfo(
            {'stream_response': True, 'compact_token_threshold': 12345}, base_model_id='missing-base'
        ),
        'fallback-model': _EntryModelInfo({}),
    }
    captured, request, user, seen_keys = _install_provider_sink_harness(monkeypatch, model_infos)
    request.app.state.MODELS = models_map

    _run_chat_completion(
        request,
        user,
        _base_form_data('custom-model', [{'model_id': 'custom-model', 'message_id': 'm1'}]),
    )

    primary = captured['m1']
    assert primary['model'] == 'fallback-model'
    assert primary['stream'] is True
    assert primary['stream_options'] == {'include_usage': True}
    assert primary['params']['compact_token_threshold'] == 12345


def test_inlet_reroute_resolves_final_model_system(monkeypatch):
    models_map = {
        'primary-model': {'id': 'primary-model', 'owned_by': 'openai', 'info': {}},
        'rerouted-model': {'id': 'rerouted-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos = {
        'primary-model': _EntryModelInfo({'system': 'primary system'}),
        'rerouted-model': _EntryModelInfo({'system': 'rerouted system'}),
    }
    captured, request, user, seen_keys = _install_provider_sink_harness(monkeypatch, model_infos)

    async def rerouting_inlet(_request, form_data, _user, _models):
        form_data['model'] = 'rerouted-model'
        return form_data

    monkeypatch.setattr(middleware, 'process_pipeline_inlet_filter', rerouting_inlet)
    request.app.state.MODELS = models_map

    _run_chat_completion(
        request,
        user,
        _base_form_data('primary-model', [{'model_id': 'primary-model', 'message_id': 'm1'}]),
    )

    assert captured['m1']['model'] == 'rerouted-model'
    assert captured['m1']['system_prompt'] == 'rerouted system'


def test_zero_stream_delta_chunk_size_does_not_override_request(monkeypatch):
    models_map = {
        'primary-model': {'id': 'primary-model', 'owned_by': 'openai', 'info': {}},
        'sibling-model': {'id': 'sibling-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos = {
        'primary-model': _EntryModelInfo({'compact_token_threshold': 80000}),
        'sibling-model': _EntryModelInfo({'stream_delta_chunk_size': 0}),
    }
    captured, request, user, seen_keys = _install_provider_sink_harness(monkeypatch, model_infos)
    request.app.state.MODELS = models_map

    _run_chat_completion(
        request,
        user,
        _base_form_data(
            'primary-model',
            [
                {'model_id': 'primary-model', 'message_id': 'm1'},
                {'model_id': 'sibling-model', 'message_id': 'm2'},
            ],
            params={'stream_delta_chunk_size': 5},
        ),
    )

    assert captured['m2']['params']['stream_delta_chunk_size'] == 5
    assert captured['m1']['params']['stream_delta_chunk_size'] == 5


def test_per_call_priority_pins_at_provider(monkeypatch):
    models_map = {
        'primary-model': {'id': 'primary-model', 'owned_by': 'openai', 'info': {}},
        'sibling-model': {'id': 'sibling-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos = {
        'primary-model': _EntryModelInfo({'compact_token_threshold': 80000}),
        'sibling-model': _EntryModelInfo(
            {
                'compact_token_threshold': 32000,
                'reasoning_tags': ['x', 'y'],
                'stream_delta_chunk_size': 2,
                'function_calling': 'legacy',
                'tool_approval_mode': 'ask',
            }
        ),
    }
    captured, request, user, seen_keys = _install_provider_sink_harness(monkeypatch, model_infos)
    request.app.state.MODELS = models_map

    _run_chat_completion(
        request,
        user,
        _base_form_data(
            'primary-model',
            [
                {'model_id': 'primary-model', 'message_id': 'm1'},
                {'model_id': 'sibling-model', 'message_id': 'm2'},
            ],
            params={
                'compact_token_threshold': 999999,
                'reasoning_tags': ['a', 'b'],
                'stream_delta_chunk_size': 5,
                'function_calling': 'native',
            },
        ),
    )

    sibling = captured['m2']['params']
    assert sibling['compact_token_threshold'] == 32000
    assert sibling['reasoning_tags'] == ['x', 'y']
    assert sibling['stream_delta_chunk_size'] == 2
    assert sibling['function_calling'] == 'native'
    # Feature is disabled in config → forced 'full' wins over any target value.
    assert sibling['tool_approval_mode'] == 'full'


def test_nonforced_tool_approval_derives_from_target(monkeypatch):
    models_map = {
        'primary-model': {'id': 'primary-model', 'owned_by': 'openai', 'info': {}},
        'sibling-model': {'id': 'sibling-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos = {
        'primary-model': _EntryModelInfo({'tool_approval_mode': 'full'}),
        'sibling-model': _EntryModelInfo({'tool_approval_mode': 'ask'}),
    }
    captured, request, user, seen_keys = _install_provider_sink_harness(
        monkeypatch, model_infos, config_values={'chat.tool_permissions.enable': True}
    )
    request.app.state.MODELS = models_map

    _run_chat_completion(
        request,
        user,
        _base_form_data(
            'primary-model',
            [
                {'model_id': 'primary-model', 'message_id': 'm1'},
                {'model_id': 'sibling-model', 'message_id': 'm2'},
            ],
        ),
    )

    assert captured['m2']['params']['tool_approval_mode'] == 'ask'
    assert captured['m1']['params']['tool_approval_mode'] == 'full'


def test_tool_approval_request_wins_over_target_model_level(monkeypatch):
    models_map = {
        'primary-model': {'id': 'primary-model', 'owned_by': 'openai', 'info': {}},
        'sibling-model': {'id': 'sibling-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos = {
        'primary-model': _EntryModelInfo({}),
        'sibling-model': _EntryModelInfo({'tool_approval_mode': 'full'}),
    }
    captured, request, user, seen_keys = _install_provider_sink_harness(
        monkeypatch, model_infos, config_values={'chat.tool_permissions.enable': True}
    )
    request.app.state.MODELS = models_map

    _run_chat_completion(
        request,
        user,
        _base_form_data(
            'primary-model',
            [
                {'model_id': 'primary-model', 'message_id': 'm1'},
                {'model_id': 'sibling-model', 'message_id': 'm2'},
            ],
            params={'tool_approval_mode': 'ask'},
        ),
    )

    assert captured['m2']['params']['tool_approval_mode'] == 'ask'


def test_tool_approval_forced_by_automation_pins_full(monkeypatch):
    models_map = {
        'primary-model': {'id': 'primary-model', 'owned_by': 'openai', 'info': {}},
        'sibling-model': {'id': 'sibling-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos = {
        'primary-model': _EntryModelInfo({'tool_approval_mode': 'ask'}),
        'sibling-model': _EntryModelInfo({'tool_approval_mode': 'ask'}),
    }
    captured, request, user, seen_keys = _install_provider_sink_harness(
        monkeypatch, model_infos, config_values={'chat.tool_permissions.enable': True}
    )
    request.app.state.MODELS = models_map

    _run_chat_completion(
        request,
        user,
        _base_form_data(
            'primary-model',
            [
                {'model_id': 'primary-model', 'message_id': 'm1'},
                {'model_id': 'sibling-model', 'message_id': 'm2'},
            ],
            params={'tool_approval_mode': 'ask'},
            automation_id='auto-1',
        ),
    )

    assert captured['m1']['params']['tool_approval_mode'] == 'full'
    assert captured['m2']['params']['tool_approval_mode'] == 'full'
    assert 'chat.tool_permissions.enable' not in seen_keys


def test_tool_approval_forced_by_channel_chat_pins_full(monkeypatch):
    subagents = importlib.import_module('open_webui.utils.subagents')
    models_map = {
        'primary-model': {'id': 'primary-model', 'owned_by': 'openai', 'info': {}},
        'sibling-model': {'id': 'sibling-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos = {
        'primary-model': _EntryModelInfo({'tool_approval_mode': 'ask'}),
        'sibling-model': _EntryModelInfo({'tool_approval_mode': 'ask'}),
    }
    captured, request, user, seen_keys = _install_provider_sink_harness(
        monkeypatch, model_infos, config_values={'chat.tool_permissions.enable': True}
    )
    request.app.state.MODELS = models_map

    async def get_channel(channel_id):
        return SimpleNamespace(id=channel_id, type='public')

    async def get_message(_message_id):
        return None

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main.Channels, 'get_channel_by_id', get_channel)
    monkeypatch.setattr(main.Messages, 'get_message_by_id', get_message)
    monkeypatch.setattr(subagents, 'process_pending_internal_messages', noop)

    form_data = _base_form_data(
        'primary-model',
        [
            {'model_id': 'primary-model', 'message_id': 'm1'},
            {'model_id': 'sibling-model', 'message_id': 'm2'},
        ],
        chat_id='channel:ch1',
    )
    _run_chat_completion(request, user, form_data)

    assert captured['m1']['params']['tool_approval_mode'] == 'full'
    assert captured['m2']['params']['tool_approval_mode'] == 'full'
    assert 'chat.tool_permissions.enable' not in seen_keys


def test_single_model_fanout_fallback_does_not_rewrite_message_ids(monkeypatch):
    models_map = {
        'custom-model': {'id': 'custom-model', 'owned_by': 'openai', 'info': {}},
        'fallback-model': {'id': 'fallback-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos = {
        'custom-model': _EntryModelInfo(
            {'system': 'custom policy', 'compact_token_threshold': 4321}, base_model_id='missing-base'
        ),
        'fallback-model': _EntryModelInfo({'system': 'fallback policy'}),
    }
    captured, request, user, seen_keys = _install_provider_sink_harness(monkeypatch, model_infos)
    request.app.state.MODELS = models_map
    columns = [{'model_id': 'custom-model', 'message_id': 'm1'}]

    _run_chat_completion(
        request,
        user,
        _base_form_data('custom-model', columns),
    )

    primary = captured['m1']
    assert primary['model'] == 'fallback-model'
    assert primary['system_prompt'] == 'custom policy'
    assert primary['params']['compact_token_threshold'] == 4321
    assert columns == [{'model_id': 'custom-model', 'message_id': 'm1'}]


def test_fallback_arena_routes_selected_target_with_custom_params(monkeypatch):
    models_map = {
        'custom-model': {
            'id': 'custom-model',
            'owned_by': 'openai',
            'info': {'meta': {'capabilities': {'usage': True}}},
        },
        'fallback-model': {
            'id': 'fallback-model',
            'owned_by': 'arena',
            'info': {'meta': {'model_ids': ['target-model']}},
        },
        'target-model': {'id': 'target-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos = {
        'custom-model': _EntryModelInfo(
            {'stream_response': True, 'compact_token_threshold': 777}, base_model_id='missing-base'
        ),
        'fallback-model': _EntryModelInfo({}),
        'target-model': _EntryModelInfo({'compact_token_threshold': 888}),
    }
    captured, request, user, seen_keys = _install_provider_sink_harness(monkeypatch, model_infos)
    request.app.state.MODELS = models_map

    _run_chat_completion(
        request,
        user,
        _base_form_data('custom-model', [{'model_id': 'custom-model', 'message_id': 'm1'}]),
    )

    primary = captured['m1']
    assert primary['model'] == 'target-model'
    assert primary['params']['compact_token_threshold'] == 777
    # The arena wrapper (the fallback routing target) is what the response
    # context sees — not the pre-swap custom model object.
    assert primary['ctx_model_id'] == 'fallback-model'


def test_chat_merge_keeps_state_authority_except_per_call_params(monkeypatch):
    per_call = {'compact_token_threshold': 32000}
    state_params = {'compact_token_threshold': 80000}
    form_metadata = {'a': 1, 'b': 2, 'params': per_call}
    state_metadata = {'b': 9, 'c': 3, 'chat_id': 'chat-1', 'params': state_params}
    captured = {}

    async def sink(request=None, form_data=None, user=None):
        captured['metadata'] = form_data['metadata']
        return {}

    monkeypatch.setattr(chat, 'generate_openai_chat_completion', sink)
    request = SimpleNamespace(
        state=SimpleNamespace(metadata=state_metadata),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={'m': {'id': 'm', 'owned_by': 'openai', 'info': {}}})),
    )
    form_data = {
        'model': 'm',
        'messages': [],
        'metadata': form_metadata,
    }

    asyncio.run(chat.generate_chat_completion(request, form_data, SimpleNamespace(role='admin')))

    merged = captured['metadata']
    assert list(merged) == ['a', 'b', 'params', 'c', 'chat_id']
    assert merged['b'] == 9
    assert merged['c'] == 3
    assert merged['chat_id'] == 'chat-1'
    assert merged['params'] is per_call

    captured.clear()
    second_form = {
        'model': 'm',
        'messages': [],
        'metadata': {'a': 1, 'b': 2},
    }
    asyncio.run(chat.generate_chat_completion(request, second_form, SimpleNamespace(role='admin')))

    assert captured['metadata']['params'] is state_params


def test_generate_chat_completion_strips_markers_without_mutating_canonical_body(monkeypatch):
    models_map = {
        'primary-model': {'id': 'primary-model', 'owned_by': 'openai', 'info': {}},
    }
    model_infos = {
        'primary-model': _EntryModelInfo({'function_calling': 'native', 'stream_response': True}),
    }
    original_build = main.build_chat_response_context
    captured, request, user, _seen = _install_provider_sink_harness(monkeypatch, model_infos)
    request.app.state.MODELS = models_map

    async def build_ctx(request, form_data, user, model, metadata, *args, **kwargs):
        ctx = await original_build(request, form_data, user, model, metadata, *args, **kwargs)
        captured['ctx'] = ctx
        return ctx

    monkeypatch.setattr(main, 'build_chat_response_context', build_ctx)

    marker_key = compaction.CONTEXT_COMPACTION_TRANSIENT_MARKER_KEY
    marked = {'role': 'user', 'content': 'please summarize', marker_key: True}
    form_data = _base_form_data(
        'primary-model',
        [{'model_id': 'primary-model', 'message_id': 'm1'}],
        messages=[marked],
    )

    _run_chat_completion(request, user, form_data)

    assert 'm1' in captured
    sink_messages = captured['m1']['messages']
    assert sink_messages
    assert all(marker_key not in message for message in sink_messages)
    assert any(
        message.get('role') == 'user' and message.get('content') == 'please summarize'
        for message in sink_messages
    )

    canonical = captured['ctx']['compaction_state']['canonical_body']['messages']
    assert any(message.get(marker_key) is True for message in canonical)
