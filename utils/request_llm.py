import base64
import json
import os
import urllib.error
import urllib.request

__all__ = ['request_llm']

OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions'


def _openrouter_content(content):
    if not isinstance(content, list):
        return str(content)

    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append({'type': 'text', 'text': part})
        elif isinstance(part, dict) and 'data' in part:
            data = part['data']
            if isinstance(data, bytes):
                data = base64.b64encode(data).decode('ascii')
            mime_type = part.get('mime_type', 'application/octet-stream')
            parts.append({
                'type': 'image_url',
                'image_url': {'url': f'data:{mime_type};base64,{data}'},
            })
        elif isinstance(part, dict) and 'type' in part:
            parts.append(part)
        else:
            parts.append({'type': 'text', 'text': str(part)})
    return parts


def _openrouter_messages(messages: list[dict]) -> list[dict]:
    result = []
    for msg in messages:
        role = msg['role']
        if role == 'model':
            role = 'assistant'
        if role not in {'system', 'user', 'assistant'}:
            raise ValueError(f'Unsupported message role for OpenRouter: {role}')
        result.append({
            'role': role,
            'content': _openrouter_content(msg['content']),
        })
    return result


def _content_to_text(content) -> str | None:
    if isinstance(content, list):
        return ''.join(
            part.get('text', '') if isinstance(part, dict) else str(part)
            for part in content
        )
    if content is None:
        return None
    return str(content)


def _normalize_usage(usage: dict | None) -> dict:
    usage = usage or {}
    return {
        'prompt_tokens': usage.get('prompt_tokens', 0),
        'completion_tokens': usage.get('completion_tokens', 0),
        'total_tokens': usage.get('total_tokens', 0),
    }


def _parse_stream_response(body: str) -> tuple[str, dict] | None:
    content_parts = []
    usage = {}
    saw_stream_event = False

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line.startswith('data:'):
            continue

        data = line[5:].strip()
        if not data or data == '[DONE]':
            continue

        saw_stream_event = True
        chunk = json.loads(data)
        if chunk.get('error'):
            raise RuntimeError(f"OpenRouter stream error: {chunk['error']}")

        for choice in chunk.get('choices') or []:
            delta = choice.get('delta') or choice.get('message') or {}
            text = _content_to_text(delta.get('content'))
            if text:
                content_parts.append(text)

        if chunk.get('usage'):
            usage = chunk['usage']

    if not saw_stream_event:
        return None

    content = ''.join(content_parts)
    if not content:
        raise RuntimeError(f'OpenRouter stream response had no content: {body[:2000]}')
    return content, _normalize_usage(usage)


def request_llm(messages: list[dict], model: str) -> tuple[str, dict]:
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise RuntimeError('OPENROUTER_API_KEY is not set')

    payload = {
        'model': model,
        'messages': _openrouter_messages(messages),
        'stream': False,
    }

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }

    site_url = os.environ.get('OPENROUTER_SITE_URL')
    app_name = os.environ.get('OPENROUTER_APP_NAME')
    if site_url:
        headers['HTTP-Referer'] = site_url
    if app_name:
        headers['X-Title'] = app_name

    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(
        OPENROUTER_API_URL,
        data=data,
        headers=headers,
        method='POST',
    )

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read().decode('utf-8')
            content_type = response.headers.get('Content-Type', '')
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'OpenRouter request failed: HTTP {e.code}: {error_body}') from e
    except urllib.error.URLError as e:
        raise RuntimeError(f'OpenRouter request failed: {e.reason}') from e

    try:
        result = json.loads(body)
    except json.JSONDecodeError as e:
        stream_result = _parse_stream_response(body)
        if stream_result is not None:
            return stream_result
        preview = body[:2000].replace('\r', '\\r')
        raise RuntimeError(
            'OpenRouter returned a non-JSON response '
            f'(content-type: {content_type or "unknown"}, error at line {e.lineno}, '
            f'column {e.colno}). First 2000 chars:\n{preview}'
        ) from e

    choices = result.get('choices') or []
    if not choices:
        raise RuntimeError(f'OpenRouter returned no choices: {body}')

    message = choices[0].get('message') or {}
    content = _content_to_text(message.get('content'))
    if content is None:
        raise RuntimeError(f'OpenRouter returned an empty message: {body}')

    return content, _normalize_usage(result.get('usage'))
