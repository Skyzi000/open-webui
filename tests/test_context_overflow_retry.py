import importlib
import json
import os
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.background import BackgroundTask

DATA_DIR = Path('/tmp/open-webui-context-overflow-tests')
STATIC_DIR = Path('/tmp/open-webui-context-overflow-static')
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
os.environ['DATA_DIR'] = str(DATA_DIR)
os.environ['STATIC_DIR'] = str(STATIC_DIR)
os.environ.setdefault('WEBUI_SECRET_KEY', 'local-test-only')

compaction = importlib.import_module('open_webui.utils.context_compaction')

ERROR = {
    'error': {
        'code': 'context_length_exceeded',
        'message': 'maximum context length exceeded',
    }
}


def sse(payload) -> bytes:
    return f'data: {json.dumps(payload)}\n\n'.encode()


ERROR_SSE = sse(ERROR)
ROLE = sse({'choices': [{'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})
USAGE = sse({'choices': [], 'usage': {'prompt_tokens': 9, 'completion_tokens': 0}})


class TrackedStream:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.index = 0
        self.closed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk

    async def aclose(self):
        self.closed += 1
        self.index = len(self.chunks)


def streaming_response(chunks, background_calls=None, media_type='text/event-stream'):
    stream = TrackedStream(chunks)

    async def close_background():
        background_calls.append(True)

    background = BackgroundTask(close_background) if background_calls is not None else None
    return StreamingResponse(stream, media_type=media_type, background=background), stream


async def consume(response):
    return [chunk async for chunk in response.body_iterator]


@pytest.mark.asyncio
async def test_disabled_retry_uses_the_core_forwarding_path(monkeypatch):
    body = {'messages': ['unchanged']}
    response, stream = streaming_response([sse({'error': ERROR['error']})])

    async def unexpected(*_args, **_kwargs):
        raise AssertionError('disabled retry must not copy or inspect the response')

    async def send(candidate):
        assert candidate is body
        return response

    monkeypatch.setattr(compaction, '_snapshot_retry_body', unexpected)
    monkeypatch.setattr(compaction, '_preflight_stream_context_overflow', unexpected)
    actual_response, actual_body = await compaction.forward_with_context_retry(send, body)

    assert actual_response is response
    assert actual_body is body
    assert stream.index == 0


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (JSONResponse(ERROR, status_code=400), True),
        (JSONResponse({'error': {'message': 'input is too long for the context window'}}, status_code=413), True),
        (HTTPException(422, detail={'error': {'type': 'context_window_exceeded'}}), True),
        (HTTPException(400, detail='prompt is too long'), True),
        (JSONResponse(ERROR, status_code=200), False),
        (JSONResponse(ERROR, status_code=429), False),
        (JSONResponse(ERROR, status_code=500), False),
        (PlainTextResponse('maximum context length exceeded', status_code=413), False),
        (ERROR, False),
        (RuntimeError('maximum context length exceeded'), False),
        (JSONResponse({'metadata': {'message': 'maximum context length exceeded'}}, status_code=400), False),
        (
            JSONResponse(
                {'error': {'code': 'context_length_exceeded', 'message': 'rate limit quota exceeded'}},
                status_code=400,
            ),
            False,
        ),
        (JSONResponse({'error': {'message': 'max_completion_tokens must be at least 1'}}, status_code=400), False),
    ],
)
def test_http_classifier_is_strict(value, expected):
    assert compaction.is_context_overflow_error(value) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'chunks',
    [
        [
            b'da',
            b'ta: {"error":{"code":"context_length_',
            b'exceeded","message":"maximum context length exceeded"}}\r',
            b'\n\r\n',
        ],
        [
            b'data: {"error":\r\n'
            b'data: {"code":"context_length_exceeded","message":"maximum context length exceeded"}}\r\n\r\n'
        ],
        [
            ROLE,
            USAGE,
            sse({'type': 'response.created', 'response': {'id': 'r1'}}),
            sse({'type': 'response.in_progress', 'response': {'id': 'r1'}}),
            sse(
                {
                    'type': 'response.output_item.added',
                    'item': {'type': 'message', 'role': 'assistant', 'content': []},
                }
            ),
            ERROR_SSE,
        ],
        [
            sse(
                {
                    'type': 'response.failed',
                    'response': {'error': {'code': 'context_window_exceeded', 'message': 'context window exceeded'}},
                }
            )
        ],
        [b'event: error\r\ndata: {"message":"input is too long"}\r\n\r\n'],
        [ERROR_SSE + sse({'choices': [{'delta': {'content': 'too late'}, 'finish_reason': None}]})],
    ],
)
async def test_stream_controls_and_split_events_retry_once(chunks):
    background_calls = []
    first, first_stream = streaming_response(chunks, background_calls)
    second = {'ok': True}
    sends = []
    retries = []
    body = {'messages': ['large']}
    smaller = {'messages': ['small']}

    async def send(candidate):
        sends.append(candidate)
        return first if len(sends) == 1 else second

    async def retry(candidate):
        retries.append(candidate)
        return smaller

    result, actual_body = await compaction.forward_with_context_retry(send, body, retry)

    assert result is second
    assert actual_body == smaller
    assert sends == [body, smaller]
    assert retries == [body]
    assert first_stream.closed == 1
    assert background_calls == [True]
    assert first.background is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'prefix',
    [
        sse({'choices': [{'delta': {'content': 'visible'}, 'finish_reason': None}]}),
        sse({'choices': [{'delta': {'reasoning_content': 'thinking'}, 'finish_reason': None}]}),
        sse({'choices': [{'delta': {'refusal': 'no'}, 'finish_reason': None}]}),
        sse({'choices': [{'delta': {'tool_calls': [{'id': 'call-1'}]}, 'finish_reason': None}]}),
        sse({'choices': [{'delta': {'function_call': {'name': 'lookup'}}, 'finish_reason': None}]}),
        sse({'choices': [{'delta': {}, 'finish_reason': 'stop'}]}),
        sse(
            {
                'type': 'response.content_part.added',
                'part': {'type': 'output_text', 'text': 'visible'},
            }
        ),
        sse(
            {
                'type': 'response.created',
                'response': {'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'visible'}]}]},
            }
        ),
        sse({'type': 'response.output_item.added', 'item': {'type': 'mcp_call', 'id': 'mcp-1'}}),
        sse(
            {
                'type': 'response.output_item.added',
                'item': {'type': 'image_generation_call', 'id': 'image-1'},
            }
        ),
        sse(
            {
                'type': 'response.output_item.added',
                'item': {'type': 'code_interpreter_call', 'id': 'code-1'},
            }
        ),
        sse({'error': {'code': 'context_length_exceeded', 'message': 'quota exceeded'}}),
        sse({'error': {'message': 'max_tokens must be at least 1'}}),
        sse({'unknown': 'event'}),
        b'data: Answer: visible\n\n',
        b'Answer: visible\n\n',
    ],
)
async def test_irreversible_stream_event_prevents_retry_and_replays_exactly(prefix):
    original = prefix + ERROR_SSE
    first, _ = streaming_response([original])
    sends = []
    retries = []
    body = {'messages': ['large']}

    async def send(candidate):
        sends.append(candidate)
        return first

    async def retry(candidate):
        retries.append(candidate)
        return {'messages': ['small']}

    result, actual_body = await compaction.forward_with_context_retry(send, body, retry)

    assert result is first
    assert actual_body == body
    assert sends == [body]
    assert retries == []
    assert await consume(result) == [original]


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['none', 'equal', 'raise'])
async def test_failed_or_unchanged_retry_replays_original_stream_error(mode):
    background_calls = []
    first, first_stream = streaming_response([ERROR_SSE], background_calls)
    sends = []
    body = {'messages': ['large']}

    async def send(candidate):
        sends.append(candidate)
        return first

    async def retry(candidate):
        if mode == 'none':
            return None
        if mode == 'equal':
            return dict(candidate)
        raise RuntimeError('cannot compact')

    result, actual_body = await compaction.forward_with_context_retry(send, body, retry)

    assert result is first
    assert actual_body == body
    assert sends == [body]
    assert await consume(result) == [ERROR_SSE]
    assert first_stream.closed == 1
    assert background_calls == [True]
    assert first.background is None


@pytest.mark.asyncio
async def test_http_exception_retry_and_original_error_fallback():
    class OpaqueToolClient:
        def __deepcopy__(self, _memo):
            raise TypeError('opaque client cannot be copied')

    tools = {'reader': {'client': OpaqueToolClient()}}
    mcp_clients = {'server': OpaqueToolClient()}
    body = {
        'messages': ['large'],
        'metadata': {'chat_id': 'chat-1', 'tools': tools, 'mcp_clients': mcp_clients},
    }
    smaller = {
        'messages': ['small'],
        'metadata': {'retry': True, 'tools': tools, 'mcp_clients': mcp_clients},
    }
    overflow = HTTPException(400, detail={'error': {'code': 'context_length_exceeded'}})
    sends = []
    retry_inputs = []

    async def send(candidate):
        sends.append(candidate)
        if len(sends) == 1:
            candidate.pop('metadata')
            raise overflow
        candidate.pop('metadata')
        return {'ok': True}

    async def retry(candidate):
        retry_inputs.append(candidate)
        return smaller

    response, actual_body = await compaction.forward_with_context_retry(send, body, retry)
    assert response == {'ok': True}
    assert actual_body == {
        'messages': ['small'],
        'metadata': {'retry': True, 'tools': tools, 'mcp_clients': mcp_clients},
    }
    assert actual_body['metadata']['tools'] is tools
    assert actual_body['metadata']['mcp_clients'] is mcp_clients
    assert len(sends) == 2
    assert sends[1] == {'messages': ['small']}
    assert retry_inputs == [
        {
            'messages': ['large'],
            'metadata': {'chat_id': 'chat-1', 'tools': tools, 'mcp_clients': mcp_clients},
        }
    ]
    assert retry_inputs[0]['metadata']['tools'] is tools
    assert retry_inputs[0]['metadata']['mcp_clients'] is mcp_clients

    async def fail_send(_candidate):
        raise overflow

    async def fail_retry(_candidate):
        raise RuntimeError('cannot compact')

    with pytest.raises(HTTPException) as caught:
        await compaction.forward_with_context_retry(fail_send, body, fail_retry)
    assert caught.value is overflow

    json_error = JSONResponse(ERROR, status_code=400)
    json_body = {'messages': ['large']}

    async def json_send(_candidate):
        return json_error

    json_result, actual_body = await compaction.forward_with_context_retry(json_send, json_body, fail_retry)
    assert json_result is json_error
    assert actual_body == json_body


@pytest.mark.asyncio
async def test_second_stream_is_not_probed_or_retried():
    first, first_stream = streaming_response([ERROR_SSE])
    second, second_stream = streaming_response([ERROR_SSE])
    sends = []
    retries = []
    body = {'messages': ['large']}
    smaller = {'messages': ['small']}

    async def send(candidate):
        sends.append(candidate)
        return first if len(sends) == 1 else second

    async def retry(candidate):
        retries.append(candidate)
        return smaller

    result, actual_body = await compaction.forward_with_context_retry(send, body, retry)

    assert result is second
    assert actual_body == smaller
    assert sends == [body, smaller]
    assert retries == [body]
    assert first_stream.closed == 1
    assert second_stream.index == 0
    assert await consume(result) == [ERROR_SSE]
    assert sends == [body, smaller]


@pytest.mark.asyncio
async def test_nonretry_streams_are_replayed_without_rechunking():
    content = sse({'choices': [{'delta': {'content': 'visible'}, 'finish_reason': None}]})
    chunks = [ROLE, USAGE, content, b'data: [DONE]\n\n']
    background_calls = []
    response, _ = streaming_response(chunks, background_calls)
    sends = []
    retries = []

    async def send(candidate):
        sends.append(candidate)
        return response

    async def retry(candidate):
        retries.append(candidate)
        return {'messages': ['small']}

    result, actual_body = await compaction.forward_with_context_retry(send, {'messages': ['large']}, retry)
    assert await consume(result) == chunks
    assert actual_body == {'messages': ['large']}
    assert sends == [{'messages': ['large']}]
    assert retries == []
    assert result.background is None
    assert background_calls == [True]

    raw, raw_stream = streaming_response([ERROR_SSE], media_type='text/plain')

    async def send_raw(_candidate):
        return raw

    raw_result, actual_body = await compaction.forward_with_context_retry(send_raw, {'messages': ['large']}, retry)
    assert raw_result is raw
    assert actual_body == {'messages': ['large']}
    assert raw_stream.index == 0
    assert await consume(raw) == [ERROR_SSE]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'chunks',
    [
        [ROLE] * (compaction._SSE_PREFLIGHT_MAX_CHUNKS + 1) + [ERROR_SSE],
        [b'data: ' + b'x' * (compaction._SSE_PREFLIGHT_MAX_BYTES + 1), ERROR_SSE],
    ],
)
async def test_stream_preflight_bounds_control_buffering(chunks):
    response, _ = streaming_response(chunks)
    retries = []

    async def send(_candidate):
        return response

    async def retry(candidate):
        retries.append(candidate)
        return {'messages': ['small']}

    result, actual_body = await compaction.forward_with_context_retry(send, {'messages': ['large']}, retry)

    assert result is response
    assert actual_body == {'messages': ['large']}
    assert retries == []
    assert await consume(result) == chunks


@pytest.mark.asyncio
async def test_replay_early_close_closes_original_iterator_and_background():
    content = sse({'choices': [{'delta': {'content': 'visible'}, 'finish_reason': None}]})
    background_calls = []
    response, original = streaming_response([content, b'data: [DONE]\n\n'], background_calls)

    async def send(_candidate):
        return response

    async def retry(_candidate):
        return {'messages': ['small']}

    result, actual_body = await compaction.forward_with_context_retry(send, {'messages': ['large']}, retry)
    assert actual_body == {'messages': ['large']}
    iterator = result.body_iterator.__aiter__()
    assert await iterator.__anext__() == content
    await iterator.aclose()

    assert original.closed == 1
    assert background_calls == [True]
    assert result.background is None
