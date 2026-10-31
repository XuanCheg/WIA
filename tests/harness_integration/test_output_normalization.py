"""Verify extra-field tolerance without relaxing action validation; run python -m pytest -q this file."""
import asyncio
from dataclasses import replace, asdict
import json
import pytest

from core.agent.decision import DecisionInput, ModelReply
from core.action_compiler import ParameterizedAction, PrimitiveResolution, MacroLimits, compile_action
from wia.runtime.agent.output_normalization import normalize_extra_fields
from wia.runtime.agent.zero_shot import run_episode
from harness_integration.test_vercel_history import fixture
from harness_integration.test_api_sessions import mock_http, chat_reply


@pytest.mark.parametrize('history_policy', ['current_sensors_full_text', 'current_sensors_last_result_notes'])
def test_extras_execute_and_feedback_next_step_with_raw_preserved(tmp_path, monkeypatch, history_policy):
    from harness_integration.test_api_sessions import session_fixture
    from wia.runtime.agent.vercel_session import VercelChatSession
    env, session = session_fixture(tmp_path, session_type=VercelChatSession,
        provider='compatible', history_policy=history_policy)
    session.model = 'alibaba/qwen3.5-plus' 
    raw = json.dumps({'command': {'action': 'BASE_FWD', 'comment': 'move'},
        'world_observation': 'Visible workspace.', 'command_note': 'extra', 'world_observation_ready': True})
    seen = mock_http(monkeypatch, [chat_reply(raw), chat_reply('{"action":"DONE"}')])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=2))
    assert result['termination_reason'] == 'model_done' and env.calls == ['BASE_FWD']
    assert not result['invalid_decisions']
    fields = result['output_normalizations'][0]['ignored_fields']
    assert set(fields) == {'command_note', 'world_observation_ready', 'command.comment'}
    current = seen[1]['payload']['messages'][-1]['content'][0]['text']
    assert 'Previous decision accepted after ignoring extra output fields' in current
    assert all(name in current for name in fields)
    record = json.loads((tmp_path/'steps.jsonl').read_text().splitlines()[0])['attempts'][0]
    assert record['output_normalization']['original_raw_output'] == raw
    assert record['model_metadata']['structured_output']['raw_output'] == raw
    assert record['model_metadata']['world_observation'] == 'Visible workspace.'
    if history_policy == 'current_sensors_last_result_notes':
        previous = json.loads(seen[1]['payload']['messages'][2]['content'])
        assert previous == {'command': {'action': 'BASE_FWD'}, 'world_observation': 'Visible workspace.'}


def parameterized_request(env):
    return DecisionInput(env.source.agent_observation(), ('BASE_FWD', 'DONE'), action_mode='parameterized',
        world_observation_log=True, action_context={'parameterized_actions': {
            'limits': asdict(MacroLimits()), 'resolution': asdict(PrimitiveResolution(base_translation_m=.05))}})


@pytest.mark.parametrize('command', [
    {'action': 'BASE_FWD'}, {'action': 'BASE_FWD', 'distance_m': '0.1'},
    {'action': 'BASE_FWD', 'distance_m': -1}, {'action': 'UNKNOWN', 'distance_m': .1}])
def test_required_action_values_stay_invalid(tmp_path, command):
    env, _ = fixture(tmp_path)
    request = parameterized_request(env)
    reply = normalize_extra_fields(request, ModelReply(json.dumps({
        'command': dict(command, extra=True), 'world_observation': 'Visible.', 'extra': 123})))
    with pytest.raises(ValueError):
        ParameterizedAction.parse(reply.raw_text, request.available_actions)


def test_valid_parameterized_action_unchanged_and_limits_still_checked(tmp_path):
    env, _ = fixture(tmp_path)
    request = parameterized_request(env)
    for distance in [.1, 1000]:
        reply = normalize_extra_fields(request, ModelReply(json.dumps({'command': {
            'action': 'BASE_FWD', 'distance_m': distance, 'extra': 123}, 'world_observation': 'Visible.'})))
        assert json.loads(reply.raw_text) == {'action': 'BASE_FWD', 'distance_m': distance}
        command = ParameterizedAction.parse(reply.raw_text, request.available_actions)
        if distance == 1000:
            with pytest.raises(ValueError):
                compile_action(command, env.capabilities, PrimitiveResolution(base_translation_m=.05), MacroLimits())


def test_duplicate_keys_and_missing_required_envelope_not_repaired(tmp_path):
    env, _ = fixture(tmp_path)
    request = parameterized_request(env)
    raw = '{"command":{"action":"DONE","action":"BASE_FWD"},"world_observation":"Visible.","extra":1}'
    original = ModelReply(raw)
    assert normalize_extra_fields(request, original) is original
    missing = normalize_extra_fields(request, ModelReply('{"command":{"action":"DONE"},"extra":1}'))
    with pytest.raises(ValueError):
        ParameterizedAction.parse(missing.raw_text, request.available_actions)
