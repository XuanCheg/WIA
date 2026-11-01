"""Check activity aggregation. Run pytest -q tests/test_activity.py."""
from wia.evaluation.activity import aggregate_activity


def test_missing_activity_is_unknown_and_safety_never_becomes_score():
    result = aggregate_activity([{'agent_decisions': 1, 'request_accounting': {
        'model_calls': 3, 'api_failures': 1, 'tool_calls': {
            'executed': {'by_name': {'episode_note': 1}}}},
        'safety_statistics': {'event_counts': {'contact': 4}}}, {}])
    assert result['model_calls'] == {'known_sum': 3, 'missing_trials': 1}
    assert result['tools']['executed']['total'] == 1
    assert 'safety' not in result and 'safety_statistics' not in result
