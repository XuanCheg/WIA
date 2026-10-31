"""Run one CPU-only Gateway vision request.

Run: python scripts/test_vercel_multimodal.py --key-file /absolute/private/key
Alternatively set AI_GATEWAY_API_KEY. No SDK, GPU, simulator, or retries are used.
"""

import argparse
import base64
import json
import os
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request
import zlib


def make_image():
    """Encode a red square left of a blue circle as a small RGB PNG."""
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    rows = bytearray()
    for y in range(200):
        rows.append(0)
        for x in range(320):
            color = (255, 255, 255)
            if 35 <= x < 115 and 60 <= y < 140:
                color = (230, 20, 20)
            if (x - 240) ** 2 + (y - 100) ** 2 <= 40 ** 2:
                color = (20, 50, 230)
            rows.extend(color)
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('!2I5B', 320, 200, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(bytes(rows))) + chunk(b'IEND', b''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    official_providers = {'deepseek/deepseek-v4.1-flash': 'deepseek',
                          'alibaba/qwen3.5-plus': 'alibaba'}
    parser.add_argument('--model', choices=official_providers,
                        default='deepseek/deepseek-v4.1-flash')
    args = parser.parse_args()
    provider = official_providers[args.model]
    out = Path(__file__).resolve().parents[1] / 'outputs' / 'vercel_multimodal' / time.strftime('%Y%m%dT%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    png = make_image()
    (out / 'input.png').write_bytes(png)
    payload = {
        'model': args.model,
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Identify the two colored shapes in this image and describe their left-to-right order. Answer briefly in English.'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}}
        ]}],
        'max_tokens': 2048,
        'providerOptions': {'gateway': {'only': [provider]}},
    }
    (out / 'request.json').write_text(json.dumps(payload, indent=2) + '\n')
    print('Artifacts:', out, flush=True)
    if args.prepare_only:
        return
    key = args.key_file.read_text().strip() if args.key_file else os.environ.get('AI_GATEWAY_API_KEY', '')
    if not key:
        raise SystemExit('Missing Gateway API key; no request sent.')
    request = urllib.request.Request('https://ai-gateway.vercel.sh/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            status, body = response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read().decode()
    elapsed = time.monotonic() - start
    (out / 'response.json').write_text(body.replace(key, '[REDACTED]') + '\n')
    metadata = {'http_status': status, 'elapsed_seconds': elapsed, 'model': payload['model'],
                'allowed_providers': [provider], 'gpu_used': False, 'requests': 1}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata))
    if status != 200:
        raise SystemExit('Request failed; inspect the saved response.')
    result = json.loads(body)
    print(json.dumps({'choices': result.get('choices'), 'usage': result.get('usage')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
