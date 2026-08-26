import importlib
import os
from pathlib import Path

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
        ('EXTERNALIZED_REFS_TOKEN_THRESHOLD', 999),
        ('CONTEXT_COMPACTION_TRANSIENT_MESSAGE_PATTERNS', '('),
    ],
)
def test_chat_config_rejects_invalid_additional_fields(field, value):
    response = client.post('/config', json={**BASE_CONFIG, field: value})

    assert response.status_code == 422
