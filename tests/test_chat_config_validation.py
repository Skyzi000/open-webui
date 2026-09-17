import asyncio
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

for name in ('data', 'static'):
    path = Path(f'/tmp/open-webui-chat-config-{name}')
    path.mkdir(parents=True, exist_ok=True)
    os.environ[name.upper() + '_DIR'] = str(path)
os.environ.setdefault('WEBUI_SECRET_KEY', 'local-test-only')

chats_router = importlib.import_module('open_webui.routers.chats')
ChatConfigForm = chats_router.ChatConfigForm
CompactChatForm = chats_router.CompactChatForm

app = FastAPI()


@app.post('/config')
async def validate_chat_config(form_data: ChatConfigForm):
    return form_data


client = TestClient(app)

BASE_CONFIG = {
    'ENABLE_CONTEXT_COMPACTION': False,
    'CONTEXT_COMPACTION_TOKEN_THRESHOLD': 80000,
    'CONTEXT_COMPACTION_PROMPT_TEMPLATE': '',
}


def test_chat_config_additional_field_defaults():
    response = client.post('/config', json=BASE_CONFIG)

    assert response.status_code == 200
    data = response.json()
    assert data['CONTEXT_COMPACTION_MODEL'] == ''
    assert data['CONTEXT_COMPACTION_TOKEN_CAP'] is None
    assert data['CONTEXT_COMPACTION_RETENTION_PERCENTAGE'] == 40
    assert data['ENABLE_TOOL_PERMISSIONS'] is False
    assert data['CONTEXT_COMPACTION_SOFT_TRIGGER_RATIO'] == 0.5
    assert data['CONTEXT_COMPACTION_TRANSIENT_MESSAGE_PATTERNS'] == ''
    assert data['ENABLE_EXTERNALIZED_REFS'] is False
    assert data['EXTERNALIZED_REFS_TOKEN_THRESHOLD'] == 10000


def test_chat_config_accepts_valid_additional_fields():
    response = client.post(
        '/config',
        json={
            **BASE_CONFIG,
            'CONTEXT_COMPACTION_SOFT_TRIGGER_RATIO': 0,
            'CONTEXT_COMPACTION_TRANSIENT_MESSAGE_PATTERNS': '\n^foo\\s+bar$\n',
            'ENABLE_EXTERNALIZED_REFS': True,
            'EXTERNALIZED_REFS_TOKEN_THRESHOLD': 1000,
        },
    )

    assert response.status_code == 200
    assert response.json()['CONTEXT_COMPACTION_SOFT_TRIGGER_RATIO'] == 0
    assert response.json()['EXTERNALIZED_REFS_TOKEN_THRESHOLD'] == 1000


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('CONTEXT_COMPACTION_SOFT_TRIGGER_RATIO', -0.01),
        ('CONTEXT_COMPACTION_SOFT_TRIGGER_RATIO', 1),
        ('CONTEXT_COMPACTION_SOFT_TRIGGER_RATIO', 'nan'),
        ('CONTEXT_COMPACTION_SOFT_TRIGGER_RATIO', 'inf'),
        ('CONTEXT_COMPACTION_TRANSIENT_MESSAGE_PATTERNS', '('),
        ('EXTERNALIZED_REFS_TOKEN_THRESHOLD', 0),
        ('EXTERNALIZED_REFS_TOKEN_THRESHOLD', 999),
    ],
)
def test_chat_config_rejects_invalid_additional_fields(field, value):
    response = client.post('/config', json={**BASE_CONFIG, field: value})

    assert response.status_code == 422


def test_externalized_refs_threshold_floor_clamps_persisted_runtime_values(monkeypatch):
    compaction = importlib.import_module('open_webui.utils.context_compaction')
    middleware = importlib.import_module('open_webui.utils.middleware')

    async def get_many(*_keys):
        return {
            'chat.context_compaction.enable': True,
            'chat.externalized_refs.enable': True,
            'chat.externalized_refs.token_threshold': 1,
        }

    async def runtime_get_many(*_keys):
        return {
            'chat.externalized_refs.enable': True,
            'chat.externalized_refs.token_threshold': 1,
        }

    monkeypatch.setattr(compaction.Config, 'get_many', get_many)
    monkeypatch.setattr(middleware.Config, 'get_many', runtime_get_many)

    config = asyncio.run(compaction._load_config())
    enabled, threshold = asyncio.run(middleware._runtime_externalized_refs_config())

    assert config['externalized_refs_token_threshold'] == 1000
    assert (enabled, threshold) == (True, 1000)


def test_chat_config_reports_invalid_regex_line():
    response = client.post(
        '/config',
        json={
            **BASE_CONFIG,
            'CONTEXT_COMPACTION_TRANSIENT_MESSAGE_PATTERNS': '^valid$\n\n(',
        },
    )

    assert response.status_code == 422
    assert 'Invalid transient message regex on line 3' in str(response.json())


def test_direct_manual_compact_preserves_websocket_metadata(monkeypatch):
    chat = SimpleNamespace(
        id='chat-1',
        current_message_id='assistant-1',
        chat={},
    )
    captured = {}

    async def get_chat(*_args, **_kwargs):
        return chat

    async def no_active_tasks(*_args, **_kwargs):
        return False

    async def get_messages(_chat_id):
        return {}

    async def compact(request, *_args, **_kwargs):
        captured.update(request.state.metadata)
        return {'ok': True, 'compacted': False}

    monkeypatch.setattr(chats_router.Chats, 'get_chat_by_id_and_user_id', get_chat)
    monkeypatch.setattr(chats_router.Chats, 'get_messages_map_by_chat_id', get_messages)
    monkeypatch.setattr(chats_router, 'has_active_tasks', no_active_tasks)
    monkeypatch.setattr(chats_router, 'compact_chat_branch', compact)

    request = SimpleNamespace(
        state=SimpleNamespace(),
        app=SimpleNamespace(
            state=SimpleNamespace(MODELS={'server-model': {'id': 'server-model'}}, redis=None)
        ),
    )
    form = CompactChatForm(
        model='direct-model',
        model_item={'id': 'direct-model', 'direct': True},
        session_id='socket-1',
    )
    user = SimpleNamespace(id='user-1', role='user')

    asyncio.run(chats_router.compact_chat_by_id(request, 'chat-1', form, user, None))

    assert captured == {
        'user_id': 'user-1',
        'session_id': 'socket-1',
        'chat_id': 'chat-1',
        'message_id': 'assistant-1',
    }


def test_get_chat_config_normalizes_externalized_refs_threshold(monkeypatch):
    def config_with(persisted):
        async def get_many(*_keys):
            return {'chat.externalized_refs.token_threshold': persisted}

        monkeypatch.setattr(chats_router.Config, 'get_many', staticmethod(get_many))
        return asyncio.run(chats_router.get_chat_config_values())

    assert config_with(999)['EXTERNALIZED_REFS_TOKEN_THRESHOLD'] == 1000
    assert config_with(1)['EXTERNALIZED_REFS_TOKEN_THRESHOLD'] == 1000
    assert config_with('5000')['EXTERNALIZED_REFS_TOKEN_THRESHOLD'] == 5000
    assert config_with('not-a-number')['EXTERNALIZED_REFS_TOKEN_THRESHOLD'] == 10000


def test_get_chat_by_id_overlays_normalized_context_summary(monkeypatch):
    embedded_a1 = {
        'id': 'a1',
        'parentId': 'u1',
        'role': 'assistant',
        'content': 'answer',
        'modelIdx': 1,
        'followUps': ['next'],
        'output': [
            {'type': 'reasoning', 'content': 'think'},
            {'type': 'message', 'content': 'final'},
        ],
    }
    embedded_history = {
        'currentId': 'a1',
        'messages': {
            'u1': {'id': 'u1', 'parentId': None, 'role': 'user', 'content': 'hello'},
            'a1': embedded_a1,
        },
    }
    chat = SimpleNamespace(
        id='chat-1',
        user_id='user-1',
        title='chat',
        chat={'history': embedded_history},
        updated_at=1,
        created_at=1,
        share_id=None,
        archived=False,
        pinned=False,
        meta={},
        variables={},
        folder_id=None,
        tasks=None,
        summary=None,
        current_message_id='a1',
    )
    normalized_map = {
        'u1': {'id': 'u1', 'parentId': None, 'role': 'user', 'content': 'hello'},
        'a1': {
            'id': 'a1',
            'parentId': 'u1',
            'role': 'assistant',
            'content': 'different stored content',
            'output': [
                {'type': 'reasoning', 'content': 'think'},
                {'type': 'message', 'content': 'final', 'contextSummary': 'nested summary'},
            ],
            'contextSummary': 'normalized summary',
        },
    }

    async def get_chat(_chat_id, _user, db=None):
        return chat

    async def get_messages_map(_chat_id):
        return normalized_map

    async def get_streams(*_args):
        return {}

    monkeypatch.setattr(chats_router.Chats, 'get_chat_by_id_for_user', get_chat)
    monkeypatch.setattr(chats_router.Chats, 'get_messages_map_by_chat_id', get_messages_map)
    monkeypatch.setattr(chats_router, 'get_response_streams_by_chat_id', get_streams)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=None)))
    data = asyncio.run(chats_router.get_chat_by_id('chat-1', request, SimpleNamespace(id='user-1'), None))

    message = data['chat']['history']['messages']['a1']
    assert message['contextSummary'] == 'normalized summary'
    assert message['output'][1]['contextSummary'] == 'nested summary'
    assert message['modelIdx'] == 1
    assert message['followUps'] == ['next']
    assert message['content'] == 'answer'
    assert message['output'][0] == {'type': 'reasoning', 'content': 'think'}
    assert 'contextSummary' not in data['chat']['history']['messages']['u1']
    assert 'contextSummary' not in embedded_a1
