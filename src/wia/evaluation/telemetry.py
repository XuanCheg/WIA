"""Request billing receipts and terminal summaries. Run python -m pytest tests/test_bench.py."""
import math
from .usage import normalize_usage


def receipt(raw, *, model, decision, seconds):
    raw = raw or {}
    usage = normalize_usage(raw) or {}
    cached = raw.get('cache_read_input_tokens')
    for key in ('prompt_tokens_details', 'input_tokens_details'):
        value = (raw.get(key) or {}).get('cached_tokens')
        if type(value) is int:
            cached = value
    cost = raw.get('cost')
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not math.isfinite(cost):
        cost = None
    return {'model': model, 'decision': decision, 'seconds': seconds,
            **usage, 'cached_tokens': cached if type(cached) is int else None,
            'cost_usd': cost, 'cost_source': 'provider_receipt' if cost is not None else None,
            'raw_usage': raw}


def totals(records):
    fields = ('input_tokens', 'output_tokens', 'reasoning_tokens', 'cached_tokens', 'cost_usd')
    return {'requests': len(records), **{key: {
        'known_sum': sum(r[key] for r in records if r.get(key) is not None),
        'missing_requests': sum(r.get(key) is None for r in records)} for key in fields}}


def format_totals(value):
    def field(name, money=False):
        data = value[name]
        number = f"${data['known_sum']:.6f}" if money else f"{data['known_sum']:,}"
        if data['missing_requests']:
            number += f" (+{data['missing_requests']} unknown)"
        return number
    return (f"requests={value['requests']} in={field('input_tokens')} cached={field('cached_tokens')} "
            f"out={field('output_tokens')} reasoning={field('reasoning_tokens')} "
            f"cost={field('cost_usd', True)}")
