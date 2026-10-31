"""Validate context options; used by `wia eval --config FILE` and scene launchers."""

HISTORY_POLICIES = ('full', 'current_sensors_full_text', 'current_sensors_recent2_notes', 'current_sensors_last_result_notes', 'recent_text', 'sliding_window', 'showharness_state')


def validate_history_policy(policy):
    """Validate explicit selections while preserving legacy parameter-only configs."""
    mode = policy.get('history_policy')
    if mode == 'current_sensors_last_result_notes':
        mode = 'current_sensors_recent2_notes'
    recent = policy.get('history_recent_moves')
    rounds = policy.get('max_perception_rounds')
    if rounds is not None and (mode != 'current_sensors_recent2_notes' or type(rounds) is not int or rounds < 1):
        raise ValueError('max_perception_rounds requires current_sensors_last_result_notes (or its legacy alias) and a positive integer.')
    if mode is not None and mode not in HISTORY_POLICIES:
        raise ValueError('Unknown history_policy: ' + str(mode))
    if recent is not None and (mode != 'showharness_state' or type(recent) is not int or recent < 0):
        raise ValueError('history_recent_moves requires showharness_state and a nonnegative integer.')
    if mode is None:
        return
    if policy.get('backend') != 'openai_chat_completions':
        raise ValueError('history_policy requires openai_chat_completions.')
    if mode in ('current_sensors_full_text', 'current_sensors_recent2_notes', 'showharness_state') and str(policy.get('base_url', '')).rstrip('/') != 'https://ai-gateway.vercel.sh/v1':
        raise ValueError('This history policy currently requires the Vercel Gateway transport.')
    for name, owner in (('history_text_turns', 'recent_text'), ('sliding_window_tokens', 'sliding_window')):
        value = policy.get(name)
        if mode == owner and value is None:
            raise ValueError(f'{mode} requires {name}.')
        if mode != owner and value is not None:
            raise ValueError(f'{mode} conflicts with {name}.')
