import asyncio
import importlib
import os
from pathlib import Path

for name in ('data', 'static'):
    path = Path(f'/tmp/open-webui-bypass-{name}')
    path.mkdir(parents=True, exist_ok=True)
    os.environ[name.upper() + '_DIR'] = str(path)
os.environ.setdefault('WEBUI_SECRET_KEY', 'local-test-only')


def test_chat_completion_bypass_is_task_local(monkeypatch):
    chat = importlib.import_module('open_webui.utils.chat')
    payload = importlib.import_module('open_webui.utils.payload')

    async def observe(_request, _form_data, _user, _bypass_filter, _bypass_system_prompt):
        await asyncio.sleep(0)
        return payload.get_chat_completion_bypass()

    monkeypatch.setattr(chat, '_generate_chat_completion', observe)

    async def run():
        return await asyncio.gather(
            chat.generate_chat_completion(None, {}, None, True, False),
            chat.generate_chat_completion(None, {}, None, False, True),
        )

    assert asyncio.run(run()) == [(True, False), (False, True)]
    assert payload.get_chat_completion_bypass() == (False, False)
