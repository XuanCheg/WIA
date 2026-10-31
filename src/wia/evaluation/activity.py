"""Aggregate private activity without modifying quality. Run pytest tests/test_activity.py."""
from collections import Counter


def aggregate_activity(trials):
    accounts = [t.get('request_accounting') for t in trials]
    fields = ('model_calls', 'api_attempts', 'api_successes', 'api_failures', 'api_timeouts', 'api_retries')
    result = {key: {'known_sum': sum(a[key] for a in accounts if a and type(a.get(key)) is int),
                    'missing_trials': sum(not a or type(a.get(key)) is not int for a in accounts)} for key in fields}
    result['decisions'] = sum(t.get('agent_decisions') or 0 for t in trials)
    result['decisions_missing_trials'] = sum(type(t.get('agent_decisions')) is not int for t in trials)
    result['executed_actions'] = sum(len(t['executed_actions']) for t in trials if isinstance(t.get('executed_actions'), list))
    result['actions_missing_trials'] = sum(not isinstance(t.get('executed_actions'), list) for t in trials)
    result['tools'] = {}
    for status in ('requested', 'executed', 'failed', 'rejected'):
        counts = Counter()
        for a in accounts:
            counts.update(((a or {}).get('tool_calls', {}).get(status) or {}).get('by_name', {}))
        result['tools'][status] = {'total': sum(counts.values()), 'by_name': dict(counts),
            'missing_trials': sum(not a or 'tool_calls' not in a for a in accounts)}
    safety = Counter()
    for trial in trials:
        safety.update((trial.get('safety_statistics') or {}).get('event_counts', {}))
    result['safety'] = {'mode': 'statistics_only', 'event_counts': dict(safety),
        'missing_trials': sum(t.get('safety_statistics') is None for t in trials),
        'affects_scoring': False, 'passed_to_model': False}
    for key in ('nonfloor_contact_intervals', 'furniture_contact_intervals', 'structure_contact_intervals',
                'incidental_contact_candidates', 'controller_interlock_events', 'unsafe_confirmed_events'):
        result['safety'][key] = {'known_sum': sum((t.get('safety_statistics') or {}).get(key, 0) for t in trials),
            'missing_trials': sum(key not in (t.get('safety_statistics') or {}) for t in trials)}
    return result
