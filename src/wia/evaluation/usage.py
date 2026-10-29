"""Normalize provider usage without inventing missing cost. Run: pytest tests/test_usage.py."""


def normalize_usage(raw):
    if not isinstance(raw, dict) or not raw:
        return None
    aliases = {'input_tokens': ('input_tokens', 'prompt_tokens', 'inputTokens'),
               'output_tokens': ('output_tokens', 'completion_tokens', 'outputTokens'),
               'total_tokens': ('total_tokens', 'totalTokens'),
               'reasoning_tokens': ('reasoning_tokens', 'reasoningOutputTokens')}
    result = {name: next((raw[key] for key in keys if type(raw.get(key)) is int), None)
              for name, keys in aliases.items()}
    for key in ('output_tokens_details', 'completion_tokens_details'):
        details = raw.get(key) or {}
        if type(details.get('reasoning_tokens')) is int:
            result['reasoning_tokens'] = details['reasoning_tokens']
    if result['total_tokens'] is None and all(result[k] is not None for k in ('input_tokens', 'output_tokens')):
        result['total_tokens'] = result['input_tokens'] + result['output_tokens']
    return result


def usage_report(summary):
    session = summary.get('native_session') or {}
    raw = summary.get('usage', summary.get('model_usage', session.get('usage')))
    transport = session.get('transport_usage')
    native = (session.get('codex_transport') == 'app_server_stdio'
              or session.get('schema') == 'wia_history_session_v1')
    return {'reported_usage': normalize_usage(raw),
            'reported_usage_scope': 'native_session' if native else 'completed_decisions',
            'observed_request_usage': normalize_usage(transport),
            'coverage': session.get('usage_coverage'),
            'reasoning_tokens_included_in_output': True,
            'billing_complete': None}


def aggregate_usage(trials):
    """Publish known subtotals and explicit missing-trial denominators."""
    result = {}
    for scope in ('reported_usage', 'observed_request_usage'):
        records = [(trial.get('token_accounting') or {}).get(scope) for trial in trials]
        fields = {}
        for field in ('input_tokens', 'output_tokens', 'total_tokens', 'reasoning_tokens'):
            known = [record[field] for record in records if record and record.get(field) is not None]
            fields[field] = {'known_sum': sum(known) if known else None,
                             'reported_trials': len(known), 'missing_trials': len(trials) - len(known)}
        result[scope] = fields
    result['billing_complete'] = None
    return result
