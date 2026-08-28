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

ChatConfigForm = importlib.import_module('open_webui.routers.chats').ChatConfigForm

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
            'EXTERNALIZED_REFS_TOKEN_THRESHOLD': 1,
        },
    )

    assert response.status_code == 200
    assert response.json()['CONTEXT_COMPACTION_SOFT_TRIGGER_RATIO'] == 0
    assert response.json()['EXTERNALIZED_REFS_TOKEN_THRESHOLD'] == 1


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('CONTEXT_COMPACTION_SOFT_TRIGGER_RATIO', -0.01),
        ('CONTEXT_COMPACTION_SOFT_TRIGGER_RATIO', 1),
        ('CONTEXT_COMPACTION_SOFT_TRIGGER_RATIO', 'nan'),
        ('CONTEXT_COMPACTION_SOFT_TRIGGER_RATIO', 'inf'),
        ('CONTEXT_COMPACTION_TRANSIENT_MESSAGE_PATTERNS', '('),
        ('EXTERNALIZED_REFS_TOKEN_THRESHOLD', 0),
    ],
)
def test_chat_config_rejects_invalid_additional_fields(field, value):
    response = client.post('/config', json={**BASE_CONFIG, field: value})

    assert response.status_code == 422


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


def test_context_usage_ignores_invalid_persisted_regex(monkeypatch, caplog):
    compaction = importlib.import_module('open_webui.utils.context_compaction')

    async def get_many(*_keys):
        return {
            'chat.context_compaction.enable': True,
            'chat.context_compaction.transient_message_patterns': '(',
        }

    async def get_messages_map(_chat_id):
        return {
            'm1': {
                'id': 'm1',
                'parentId': None,
                'role': 'user',
                'content': 'hello',
            }
        }

    monkeypatch.setattr(compaction.Config, 'get_many', get_many)
    monkeypatch.setattr(compaction.Chats, 'get_messages_map_by_chat_id', get_messages_map)
    chat = SimpleNamespace(id='chat-1', current_message_id='m1', chat={})

    with caplog.at_level('ERROR', logger=compaction.__name__):
        result = asyncio.run(compaction.get_chat_context_usage(chat))

    assert result is None
    assert 'Context compaction configuration is invalid' in caplog.text
