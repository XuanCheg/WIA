"""Estimate the pinned DeepSeek input budget; run pytest tests/harness_integration/test_vercel_history.py."""
import base64
from copy import deepcopy
from io import BytesIO
import json
import math

from PIL import Image
from core.agent.decision import ModelError


def estimate_multimodal_input(payload):
    """Use UTF-8 bytes and padded 16px image patches, not a provider tokenizer count."""
    content = deepcopy({'messages': payload['messages'], 'tools': payload.get('tools', [])})
    vision = 0
    for message in content['messages']:
        if not isinstance(message.get('content'), list):
            continue
        for part in message['content']:
            if part.get('type') != 'image_url':
                continue
            url = part['image_url']['url']
            if not url.startswith('data:image/') or ';base64,' not in url:
                raise ModelError('Input budget requires inspectable inline images.')
            with Image.open(BytesIO(base64.b64decode(url.split(',', 1)[1]))) as im:
                width, height = im.size
            vision += math.ceil(width/16)*math.ceil(height/16)*4 + 1024
            part['image_url']['url'] = '[image accounted separately]'
    text = len(json.dumps(content, ensure_ascii=False).encode('utf-8'))
    return {'method': 'utf8_bytes_plus_4_tokens_per_16px_patch_plus_1024_per_image',
        'exact_token_count': False, 'text_estimate': text, 'image_estimate': vision,
        'input_estimate': text+vision}


def deepseek_budget(payload):
    """Retain the pinned model's existing reservation and safety margin."""
    result = estimate_multimodal_input(payload)
    reserve = payload.get('max_tokens', payload.get('max_completion_tokens', 0))
    result.update(output_reservation=reserve, safety_margin=8192, context_limit=1000000)
    result['within_budget'] = result['input_estimate']+reserve+8192 <= 1000000
    return result
