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
    backend = policy.get('backend')
    if backend:
        # Built-in configuration validation must work without optional Harness imports.
        owners = {'codex': 'native', 'codex_cli': 'native', 'local_http': 'harness',
                  'openai_chat_completions': 'harness', 'openai_responses': 'harness'}
        owner = owners.get(backend)
        if backend not in owners:
            from core.agent.backend_registry import available_backends
            capability = available_backends().get(backend)
            owner = capability.history_owner if capability else None
        controlled = mode not in (None, 'full') or any(policy.get(k) is not None for k in
            ('history_text_turns', 'sliding_window_tokens', 'image_max_edge')) or policy.get('json_schema_output', False)
        if owner is not None and owner != 'harness' and controlled:
            raise ValueError('Selected context policy requires client-owned history; this backend owns history natively.')
    if mode is None:
        return
    for name, owner in (('history_text_turns', 'recent_text'), ('sliding_window_tokens', 'sliding_window')):
        value = policy.get(name)
        if mode == owner and value is None:
            raise ValueError(f'{mode} requires {name}.')
        if mode != owner and value is not None:
            raise ValueError(f'{mode} conflicts with {name}.')
