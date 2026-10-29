"""Verify token accounting scopes. Run: pytest tests/test_usage.py."""
from wia.evaluation.usage import normalize_usage, usage_report


def test_usage_scopes_and_reasoning_are_not_double_counted():
    result = usage_report({'native_session': {'usage': {'prompt_tokens': 171744, 'completion_tokens': 11848,
        'total_tokens': 183592, 'completion_tokens_details': {'reasoning_tokens': 11110}},
        'transport_usage': {'prompt_tokens': 252968, 'completion_tokens': 12880, 'total_tokens': 265848},
        'usage_coverage': {'unknown_usage_requests': 1, 'complete': False}}})
    assert result['reported_usage']['total_tokens'] == 183592
    assert result['observed_request_usage']['total_tokens'] == 265848
    assert result['coverage']['complete'] is False
    assert result['billing_complete'] is None
    assert normalize_usage({'inputTokens': 20, 'outputTokens': 10, 'reasoningOutputTokens': 8})['total_tokens'] == 30
    assert normalize_usage({}) is None


def test_aggregate_retains_missing_trial_denominator():
    from wia.evaluation.usage import aggregate_usage
    result = aggregate_usage([{'token_accounting': usage_report({'usage': {'input_tokens': 4, 'output_tokens': 2}})}, {}])
    assert result['reported_usage']['total_tokens'] == {'known_sum': 6, 'reported_trials': 1, 'missing_trials': 1}
    assert result['observed_request_usage']['total_tokens']['known_sum'] is None
