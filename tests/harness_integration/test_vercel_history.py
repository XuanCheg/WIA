"""Verify Gateway sensor-history projection; run python -m pytest -q this file."""
import asyncio
from copy import deepcopy
import json
import pytest

from core.agent.decision import DecisionInput
from wia.runtime.agent.vercel_session import VercelChatSession
from wia.runtime.agent.zero_shot import run_episode
from harness_integration.test_api_sessions import session_fixture, mock_http, chat_reply, depth_request


def fixture(tmp_path, enabled=True):
    env, session = session_fixture(tmp_path, session_type=VercelChatSession,
                                  provider='compatible', current_sensors_only=enabled)
    session.model = 'alibaba/qwen3.5-plus'
    return env, session


def images(messages):
    return sum(p.get('type') == 'image_url' for m in messages
               if isinstance(m.get('content'), list) for p in m['content'])


def test_two_decisions_keep_verbatim_text_and_reasoning_only(tmp_path, monkeypatch):
    env, session = fixture(tmp_path)
    fields = {'reasoning': 'Synthetic reasoning fixture.',
              'reasoning_details': [{'type': 'reasoning.text', 'text': 'Synthetic reasoning fixture.'}]}
    seen = mock_http(monkeypatch, [chat_reply('{"action":"BASE_FWD"}', **fields),
                                  chat_reply('{"action":"DONE"}')])
    asyncio.run(run_episode(env, session, tmp_path, max_steps=2))
    first, second = [x['payload']['messages'] for x in seen]
    assert images(first) == 9 and images(second) == 4
    assert 'Historical RGB recall tool: read_historical_rgb' in first[0]['content'][0]['text']
    assert 'Historical RGB recall tool: read_historical_rgb' not in second[-1]['content'][0]['text']
    assert 'Current decision index: 2' in second[-1]['content'][0]['text']
    assert session.provenance()['historical_rgb_prompt_sha256']
    assert second[0]['content'] == [p for p in first[0]['content'] if p['type'] == 'text']
    assert second[1]['content'] == '{"action":"BASE_FWD"}'
    assert all(second[1][k] == v for k, v in fields.items())
    assert seen[1]['payload']['providerOptions']['gateway'] == {'only': ['alibaba'], 'caching': 'auto'}
    archived = json.loads((tmp_path/'gateway_assistant_messages.jsonl').read_text().splitlines()[0])
    assert archived['message']['reasoning_details'] == fields['reasoning_details']
    metrics = [json.loads(x) for x in (tmp_path/'gateway_requests.jsonl').read_text().splitlines()]
    assert metrics[1]['image_count'] == 4 and metrics[1]['reasoning_messages'] == 1


def test_current_depth_and_reasoning_survive_tool_round_but_old_depth_is_omitted(tmp_path, monkeypatch):
    env, session = fixture(tmp_path)
    call = {'id': 'depth1', 'type': 'function', 'function': {'name': 'read_depth_pixels',
            'arguments': json.dumps({'sensor': 'head_depth_m', 'pixels_uv': [[4, 0]]})}}
    seen = mock_http(monkeypatch, [chat_reply(None, tool_calls=[call], reasoning='Tool fixture.'),
                                  chat_reply('{"action":"DONE"}', reasoning='Final fixture.')])
    asyncio.run(session.decide(depth_request(env, session)))
    current = seen[1]['payload']['messages']
    assert images(current) > 0 and current[-2]['reasoning'] == 'Tool fixture.'
    raw_depth = current[-1]['content']
    assert '0.75' in raw_depth
    original = deepcopy(session.history)
    projected = session.request_history()
    assert session.history == original and images(projected) == 0
    assert projected[-2]['content'].startswith('[Historical depth measurement omitted')
    assert projected[-3]['tool_calls'] == [call] and projected[-3]['reasoning'] == 'Tool fixture.'
    assert projected[-1]['reasoning'] == 'Final fixture.'


def test_existing_full_history_mode_stays_unchanged(tmp_path):
    _, session = fixture(tmp_path, enabled=False)
    session.history = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:test'}}]}]
    assert session.request_history() == session.history
    assert session.history_policy == 'full_no_truncation'


def illegal_tool():
    return chat_reply(None, tool_calls=[{'id': 'bad-command', 'type': 'function',
        'function': {'name': 'command', 'arguments': '{}'}}], reasoning='Invalid call fixture.')


def test_illegal_tool_consumes_step_and_next_decision_gets_feedback(tmp_path, monkeypatch):
    env, session = fixture(tmp_path)
    seen = mock_http(monkeypatch, [illegal_tool(), chat_reply('{"action":"DONE"}')])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=2))
    assert result['agent_decisions'] == 2 and result['termination_reason'] == 'model_done'
    assert len(result['invalid_decisions']) == 1 and env.calls == []
    assert len(seen) == 2
    history = seen[1]['payload']['messages']
    assert history[1]['reasoning'] == 'Invalid call fixture.'
    assert 'consumed one step' in history[2]['content']
    assert 'no robot action' in history[-1]['content'][0]['text'].lower()
    assert images(history) == 4
    records = [json.loads(x) for x in (tmp_path/'steps.jsonl').read_text().splitlines()]
    assert records[0]['action_result'] is None and records[0]['termination_reason'] is None
    assert records[0]['attempts'][0]['invalid_tool_decision'] is True


def test_repeated_invalid_tools_exhaust_budget_without_extra_retries(tmp_path, monkeypatch):
    env, session = fixture(tmp_path)
    seen = mock_http(monkeypatch, [illegal_tool(), illegal_tool()])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=2, max_invalid_retries=9))
    assert result['agent_decisions'] == 2 and result['termination_reason'] == 'max_steps'
    assert len(result['invalid_decisions']) == len(seen) == 2 and env.calls == []


def test_invalid_final_json_consumes_step_before_valid_action(tmp_path, monkeypatch):
    env, session = fixture(tmp_path)
    seen = mock_http(monkeypatch, [chat_reply('not valid action JSON'),
                                  chat_reply('{"action":"BASE_FWD"}')])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=2))
    assert result['agent_decisions'] == 2 and result['termination_reason'] == 'max_steps'
    assert len(result['invalid_decisions']) == 1 and env.calls == ['BASE_FWD']
    assert len(seen) == 2


def test_transport_failure_is_not_counted_as_invalid_model_decision(tmp_path, monkeypatch):
    import requests
    env, session = fixture(tmp_path)
    def failed(*args, **kwargs):
        response = requests.Response()
        response.status_code = 413
        response._content = b'Payload too large'
        raise requests.HTTPError(response=response)
    monkeypatch.setattr('core.agent.api_sessions.requests.post', failed)
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=2))
    assert result['termination_reason'] == 'model_error' and result['agent_decisions'] == 0
    assert result['invalid_decisions'] == [] and env.calls == []


def test_recall_old_image_only_for_requested_turn_and_preserve_text(tmp_path, monkeypatch):
    env, session = fixture(tmp_path)
    call = {'id': 'old-image', 'type': 'function', 'function': {'name': 'read_historical_rgb',
        'arguments': json.dumps({'decision_step': 1, 'camera': 'agentview_rgb'})}}
    seen = mock_http(monkeypatch, [chat_reply('{"action":"BASE_FWD"}'),
        chat_reply(None, tool_calls=[call], reasoning='Recall fixture.'),
        chat_reply('{"action":"BASE_FWD"}'), chat_reply('{"action":"DONE"}')])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=3))
    assert result['termination_reason'] == 'model_done' and result['agent_decisions'] == 3
    assert env.calls == ['BASE_FWD', 'BASE_FWD']
    assert [images(x['payload']['messages']) for x in seen] == [9, 4, 5, 4]
    old_url = seen[0]['payload']['messages'][0]['content'][2]['image_url']['url']
    recalled = seen[2]['payload']['messages'][-1]['content']
    assert recalled[-1]['image_url']['url'] == old_url
    assert 'past decision' in recalled[0]['text']
    assert seen[2]['payload']['messages'][-3]['reasoning'] == 'Recall fixture.'
    assert session.usage['completion_tokens'] == 12
    assert len((tmp_path/'historical_image_reads.jsonl').read_text().splitlines()) == 1
    assert 'Requested historical RGB:' in json.dumps(seen[3]['payload'])


def test_recall_rejects_future_unknown_and_cross_episode_paths(tmp_path):
    from wia.runtime.agent.historical_images import read_historical_image
    _, session = fixture(tmp_path)
    session._current_decision_step, session._history_images_used = 3, 0
    for args in [{'decision_step': 3, 'camera': 'agentview_rgb'},
                 {'decision_step': 1, 'camera': '../private'}]:
        result, url = read_historical_image(session, args)
        assert not result['ok'] and url is None
    session._history_images = {1: {'agentview_rgb': {'path': '../not-this-episode.png'}}}
    result, url = read_historical_image(session, {'decision_step': 1, 'camera': 'agentview_rgb'})
    assert not result['ok'] and url is None


def test_showharness_state_rebuilds_each_decision(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path, session_type=VercelChatSession,
        provider='compatible', history_policy='showharness_state', history_recent_moves=1)
    session.model = 'alibaba/qwen3.5-plus'
    prepared = []
    original_prepare = session.prepare_request
    def capture(request):
        result = original_prepare(request)
        prepared.append(result)
        return result
    session.prepare_request = capture
    seen = mock_http(monkeypatch, [chat_reply('{"action":"BASE_FWD"}', reasoning='Old reasoning sentinel'),
        chat_reply('{"action":"BASE_FWD"}'), chat_reply('{"action":"DONE"}')])
    asyncio.run(run_episode(env, session, tmp_path, max_steps=3))
    assert all(len(r.observation.action_history) <= 1 for r in prepared)
    assert len(prepared[-1].observation.action_history) == 1
    assert [images(x['payload']['messages']) for x in seen] == [9, 4, 4]
    for request in seen:
        assert [m['role'] for m in request['payload']['messages']] == ['system', 'user']
        assert 'Old reasoning sentinel' not in json.dumps(request['payload'])
        assert not any(t['function']['name'] == 'read_historical_rgb' for t in request['payload']['tools'])
    last = seen[-1]['payload']['messages'][-1]['content'][0]['text']
    assert session.public_task['objective'] in seen[-1]['payload']['messages'][0]['content']
    assert 'at most 1 recent executed actions' in last
    assert session.provenance()['history_recent_moves'] == 1
    assert 'Old reasoning sentinel' in (tmp_path / 'gateway_assistant_messages.jsonl').read_text()


def test_history_policy_validation_and_forwarding():
    import argparse
    import pytest
    from wia.runtime.agent.history_policy import validate_history_policy
    from wia.runtime.commands.run_esi_agent import add_backend_arguments, backend_cli_arguments, validate_backend_arguments
    parser = argparse.ArgumentParser()
    add_backend_arguments(parser)
    args = parser.parse_args(['--backend', 'openai_chat_completions', '--provider', 'compatible',
        '--base-url', 'https://ai-gateway.vercel.sh/v1', '--history-policy', 'showharness_state', '--history-recent-moves', '5'])
    validate_backend_arguments(args)
    forwarded = backend_cli_arguments(args)
    assert forwarded[forwarded.index('--history-policy') + 1] == 'showharness_state'
    assert forwarded[forwarded.index('--history-recent-moves') + 1] == '5'
    for mode in ('full', 'current_sensors_full_text', 'showharness_state'):
        with pytest.raises(ValueError):
            validate_history_policy(dict(vars(args), history_policy=mode, history_text_turns=2))
    with pytest.raises(ValueError):
        validate_history_policy(dict(vars(args), history_policy='recent_text', history_recent_moves=None))


def test_previous_result_notes_eviction_budget_and_global_prompt(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path, session_type=VercelChatSession,
        provider='compatible', history_policy='current_sensors_last_result_notes')
    session.model = 'alibaba/qwen3.5-plus'
    def note(identifier, operation, text=None):
        args = {'operation': operation}
        if text is not None:
            args['text'] = text
        return chat_reply(None, tool_calls=[{'id': identifier, 'type': 'function', 'function': {
            'name': 'episode_note', 'arguments': json.dumps(args)}}])
    move = '{"action":"BASE_FWD"}'
    seen = mock_http(monkeypatch, [
        note('write1', 'append', '界' * 128), note('overflow', 'append', 'x'),
        note('read1', 'read'), chat_reply(move, reasoning='reason-step-one'),
        note('write2', 'append', 'next'), chat_reply(move, reasoning='reason-step-two'),
        chat_reply(move, reasoning='reason-step-three'),
        note('read4', 'read'), chat_reply(move, reasoning='reason-step-four'),
        chat_reply('{"action":"DONE"}')])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=5))
    assert result['agent_decisions'] == 5 and result['termination_reason'] == 'model_done'
    assert len(env.calls) == 4
    assert len(seen) == 10
    prefixes = [x['payload']['messages'][0] for x in seen]
    assert all(prefix == prefixes[0] for prefix in prefixes)
    assert 'Episode note tool: episode_note' in prefixes[0]['content']
    assert 'Historical RGB recall tool: read_historical_rgb' in prefixes[0]['content']
    fourth = seen[7]['payload']['messages']
    assert 'reason-step-one' not in json.dumps(fourth)
    assert 'reason-step-two' not in json.dumps(fourth) and 'reason-step-three' not in json.dumps(fourth)
    assert fourth[2] == {'role': 'assistant', 'content': move}
    assert sum(m['role'] == 'assistant' for m in fourth) == 1
    assert images(fourth) == 4
    receipts = [json.loads(m['content']) for m in seen[8]['payload']['messages']
                if m.get('tool_call_id') == 'read4']
    assert receipts[0]['notes'] == [{'decision_step': 1, 'text': '界' * 128},
                                    {'decision_step': 2, 'text': 'next'}]
    events = [json.loads(x) for x in (tmp_path/'note_tool_calls.jsonl').read_text().splitlines()]
    assert events[0]['result']['remaining_write_characters'] == 0
    assert not events[1]['result']['ok']
    assert len((tmp_path/'episode_notes.jsonl').read_text().splitlines()) == 2
    assert 'reason-step-one' in (tmp_path/'gateway_assistant_messages.jsonl').read_text()
    assert len(session._recent_turns) == 1
    assert session.provenance()['history_projection'] == 'previous_public_result_only'


def test_recent_notes_invalid_turn_and_recall_outside_window(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path, session_type=VercelChatSession,
        provider='compatible', history_policy='current_sensors_recent2_notes')
    session.model = 'alibaba/qwen3.5-plus'
    illegal = {'id': 'bad', 'type': 'function', 'function': {'name': 'command', 'arguments': '{}'}}
    recall = {'id': 'old', 'type': 'function', 'function': {'name': 'read_historical_rgb',
        'arguments': json.dumps({'decision_step': 1, 'camera': 'agentview_rgb'})}}
    seen = mock_http(monkeypatch, [chat_reply(None, tool_calls=[illegal], reasoning='invalid-one'),
        chat_reply('{"action":"BASE_FWD"}', reasoning='valid-two'),
        chat_reply('{"action":"BASE_FWD"}', reasoning='valid-three'),
        chat_reply(None, tool_calls=[recall]), chat_reply('{"action":"DONE"}')])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=4))
    assert result['agent_decisions'] == 4 and len(result['invalid_decisions']) == 1
    assert 'invalid-one' not in json.dumps(seen[1]['payload'])
    assert 'rejected' in json.dumps(seen[1]['payload'])
    assert 'invalid-one' not in json.dumps(seen[3]['payload'])
    assert 'valid-two' not in json.dumps(seen[3]['payload'])
    assert 'valid-three' not in json.dumps(seen[3]['payload'])
    assert [images(x['payload']['messages']) for x in seen] == [9, 4, 4, 4, 5]
    first_url = seen[0]['payload']['messages'][-1]['content'][2]['image_url']['url']
    assert seen[-1]['payload']['messages'][-1]['content'][-1]['image_url']['url'] == first_url


def test_deepseek_budget_preserves_output_reservation():
    from wia.runtime.agent.input_budget import deepseek_budget
    payload = {'messages': [{'role': 'user', 'content': 'small'}], 'max_tokens': 384000}
    budget = deepseek_budget(payload)
    assert budget['within_budget'] and budget['output_reservation'] == 384000
    payload['messages'][0]['content'] = 'x' * 616000
    budget = deepseek_budget(payload)
    assert not budget['within_budget'] and not budget['exact_token_count']


def test_deepseek_rejects_wrong_model_receipt(tmp_path, monkeypatch):
    env, session = fixture(tmp_path)
    session.model = 'deepseek/deepseek-v4.1-flash-1017'
    response = chat_reply('{"action":"DONE"}', provider_metadata={'gateway': {'routing': {
        'finalProvider': 'deepseek', 'canonicalSlug': 'deepseek/other'}}})
    mock_http(monkeypatch, [response])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=1))
    assert result['termination_reason'] == 'model_error' and env.calls == []


def test_perception_round_cap_batches_resets_and_keeps_note_reads(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path, session_type=VercelChatSession,
        provider='compatible', history_policy='current_sensors_recent2_notes', max_perception_rounds=1)
    session.model = 'alibaba/qwen3.5-plus'
    def call(identifier, name, args):
        return {'id': identifier, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}
    depth = call('d1', 'read_rgb_camera', {'camera': 'chassis_left_rgb'})
    depth2 = call('d2', 'read_rgb_camera', {'camera': 'chassis_right_rgb'})
    note = call('n1', 'episode_note', {'operation': 'read'})
    seen = mock_http(monkeypatch, [chat_reply(None, tool_calls=[depth, depth2]),
        chat_reply(None, tool_calls=[note]), chat_reply(None, tool_calls=[dict(note, id='n2')]),
        chat_reply('{"action":"BASE_FWD"}'), chat_reply(None, tool_calls=[depth]),
        chat_reply('{"action":"DONE"}')])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=2))
    assert result['agent_decisions'] == 2 and result['termination_reason'] == 'model_done'
    names = lambda index: {t['function']['name'] for t in seen[index]['payload']['tools']}
    assert 'read_rgb_camera' in names(0) and 'read_historical_rgb' in names(0)
    assert all(names(i) == names(0) for i in range(6))
    assert 'episode_note' in names(0)
    assert 'read_rgb_camera' in names(4)
    assert 'Perception round budget exhausted' in json.dumps(seen[1]['payload']['messages'])
    assert 'Use at most 1 perception-tool rounds' in seen[0]['payload']['messages'][0]['content']
    assert session.provenance()['max_perception_rounds'] == 1
    events = [json.loads(x) for x in (tmp_path/'gateway_requests.jsonl').read_text().splitlines()]
    assert [x['perception_rounds_used'] for x in events] == [0, 1, 1, 1, 0, 1]


def test_perception_after_cap_is_invalid_and_next_decision_resets(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path, session_type=VercelChatSession,
        provider='compatible', history_policy='current_sensors_recent2_notes', max_perception_rounds=1)
    session.model = 'alibaba/qwen3.5-plus'
    call = {'id': 'd1', 'type': 'function', 'function': {'name': 'read_rgb_camera',
        'arguments': json.dumps({'camera': 'chassis_left_rgb'})}}
    seen = mock_http(monkeypatch, [chat_reply(None, tool_calls=[call]),
        chat_reply(None, tool_calls=[dict(call, id='forbidden')]), chat_reply('{"action":"BASE_FWD"}')])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=2))
    assert len(result['invalid_decisions']) == 1 and result['agent_decisions'] == 2
    assert env.calls == ['BASE_FWD']
    assert seen[0]['payload']['tools'] == seen[1]['payload']['tools']
    assert 'read_rgb_camera' in {t['function']['name'] for t in seen[2]['payload']['tools']}


@pytest.mark.parametrize('provider', ['openai', 'anthropic', 'deepseek', 'moonshotai', 'alibaba'])
def test_cache_prefix_is_static_and_provider_neutral(tmp_path, monkeypatch, provider):
    env, session = session_fixture(tmp_path, session_type=VercelChatSession,
        provider='compatible', history_policy='current_sensors_last_result_notes')
    session.model = provider + '/transport-test-fixture'
    seen = mock_http(monkeypatch, [chat_reply('{"action":"BASE_FWD"}'),
                                   chat_reply('{"action":"DONE"}')])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=2))
    assert result['termination_reason'] == 'model_done'
    first, second = [x['payload'] for x in seen]
    assert first['messages'][0] == second['messages'][0]
    prefix = first['messages'][0]['content']
    assert 'Exact structured output contract' in prefix
    assert 'Current measured robot state' not in prefix
    assert 'Actual action capabilities' not in prefix
    assert 'retry_feedback' not in prefix
    for payload in (first, second):
        current = payload['messages'][-1]['content'][0]['text']
        assert 'Exact structured output contract' not in current
        assert 'Current action capabilities' in prefix
        assert 'camera_geometry' in current
        assert 'Last execution feedback' in current
        assert payload['providerOptions']['gateway'] == {'only': [provider], 'caching': 'auto'}
    assert first['tools'] == second['tools']
    assert [images(p['messages']) for p in (first, second)] == [9, 4]
    events = [json.loads(x) for x in (tmp_path/'gateway_requests.jsonl').read_text().splitlines()]
    assert events[0]['system_prefix_sha256'] == events[1]['system_prefix_sha256']
    assert events[0]['tool_catalog_sha256'] == events[1]['tool_catalog_sha256']


def test_cache_layout_preserves_measurements_and_refreshes_contract(tmp_path):
    from copy import copy
    from dataclasses import replace
    env, session = fixture(tmp_path)
    request = depth_request(env, session)
    original = session.assembler.build(request, session.public_task,
        session.episode_configuration, first_turn=True)
    assembler = copy(session.assembler)
    assembler.cache_prefix_layout = True
    optimized = assembler.build(request, session.public_task,
        session.episode_configuration, first_turn=True)
    def section(prompt, label):
        return json.JSONDecoder().raw_decode(prompt.split(label + ':\n', 1)[1])[0]
    assert section(original.assembled_prompt, 'Initial authorized sensor observation') == section(
        optimized.assembled_prompt, 'Authorized sensor observation')
    assert section(original.resident_prompt, 'Actual action capabilities') == section(
        optimized.resident_prompt, 'Current action capabilities')
    changed = assembler.build(replace(request, feedback='Retry feedback sentinel'),
        session.public_task, session.episode_configuration, first_turn=False)
    assert changed.resident_prompt == optimized.resident_prompt
    assert 'Retry feedback sentinel' in changed.assembled_prompt
    recovering = assembler.build(replace(request, action_context={
        **(request.action_context or {}), 'recovery': {'reason': 'Blocked motion sentinel'}}),
        session.public_task, session.episode_configuration, first_turn=False)
    assert recovering.resident_prompt == optimized.resident_prompt
    assert 'Blocked motion sentinel' in recovering.assembled_prompt
    changed = assembler.build(replace(request, available_actions=('DONE',)),
        session.public_task, session.episode_configuration, first_turn=False)
    assert section(changed.resident_prompt, 'Current action capabilities')['available_actions'] == ['DONE']
    assert not session.assembler.cache_prefix_layout
