"""Cross-transport context checks. Run python -m pytest -q tests/harness_integration/test_context_session.py."""
import asyncio
import json

import pytest

from wia.runtime.agent.context_session import ContextChatSession
from wia.runtime.agent.responses_context import ResponsesContextSession
from wia.runtime.agent.zero_shot import run_episode
from harness_integration.test_api_sessions import session_fixture, mock_http, chat_reply, responses_reply


@pytest.mark.parametrize('session_type', [ContextChatSession, ResponsesContextSession])
@pytest.mark.parametrize('mode', ['full', 'current_sensors_full_text', 'current_sensors_last_result_notes',
                                 'showharness_state', 'recent_text', 'sliding_window'])
def test_history_policies_on_direct_transports(tmp_path, monkeypatch, session_type, mode):
    options = {'history_policy': mode}
    if mode == 'recent_text':
        options['history_text_turns'] = 1
    if mode == 'sliding_window':
        options['sliding_window_tokens'] = 2000000
    env, session = session_fixture(tmp_path, session_type=session_type, **options)
    reply = responses_reply if session_type is ResponsesContextSession else chat_reply
    seen = mock_http(monkeypatch, [reply('{"action":"BASE_FWD"}'), reply('{"action":"DONE"}')])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=2))
    assert result['termination_reason'] == 'model_done'
    assert len(seen) == 2
    key = 'input' if session_type is ResponsesContextSession else 'messages'
    for request in seen:
        assert 'providerOptions' not in request['payload']
        assert request['url'].endswith('/responses' if key == 'input' else '/chat/completions')
    if mode in ('showharness_state', 'current_sensors_last_result_notes'):
        assert seen[0]['payload'][key][0] == seen[1]['payload'][key][0]
    if mode == 'showharness_state':
        assert [m['role'] for m in seen[1]['payload'][key]] == ['system', 'user']
    if mode == 'sliding_window':
        audit = json.loads((tmp_path/'context_budget.jsonl').read_text().splitlines()[0])
        assert audit['exact_token_count'] is False


def test_responses_notes_preserve_live_reasoning_and_drop_old_items(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path, session_type=ResponsesContextSession,
        history_policy='current_sensors_last_result_notes', reasoning_effort='medium')
    reasoning = {'type': 'reasoning', 'id': 'r1', 'summary': [], 'encrypted_content': 'opaque-fixture'}
    output = [reasoning, {'type': 'function_call', 'id': 'f1', 'call_id': 'c1',
        'name': 'episode_note', 'arguments': json.dumps({'operation': 'append', 'text': 'Remember fixture.'})}]
    seen = mock_http(monkeypatch, [{'status': 'completed', 'output': output, 'usage': {}},
        responses_reply('{"action":"BASE_FWD"}'), responses_reply('{"action":"DONE"}')])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=2))
    assert result['termination_reason'] == 'model_done'
    assert reasoning in seen[1]['payload']['input']
    receipt = next(x for x in seen[1]['payload']['input'] if x.get('type') == 'function_call_output')
    assert receipt['call_id'] == 'c1' and json.loads(receipt['output'])['ok']
    assert 'opaque-fixture' not in json.dumps(seen[2]['payload'])
    assert all(x['payload']['reasoning'] == {'effort': 'medium'} for x in seen)
    assert seen[0]['payload']['tools'] == seen[1]['payload']['tools'] == seen[2]['payload']['tools']


def test_responses_perception_cap_rejects_before_execution(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path, session_type=ResponsesContextSession,
        history_policy='current_sensors_last_result_notes', max_perception_rounds=1)
    def response(identifier):
        return {'status': 'completed', 'output': [{'type': 'function_call', 'call_id': identifier,
            'name': 'read_rgb_camera', 'arguments': json.dumps({'camera': 'chassis_left_rgb'})}]}
    seen = mock_http(monkeypatch, [response('c1'), response('c2'), responses_reply('{"action":"DONE"}')])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=2))
    assert len(result['invalid_decisions']) == 1 and env.calls == []
    assert result['agent_decisions'] == 2
    assert seen[0]['payload']['tools'] == seen[1]['payload']['tools']


@pytest.mark.parametrize('backend', ['local_http', 'openai_chat_completions', 'openai_responses'])
def test_policy_factory_uses_shared_engine(tmp_path, monkeypatch, backend):
    from harness_integration.test_backend_cli_integration import options
    from wia.runtime.commands.run_esi_agent import create_declarative_session, validate_backend_arguments
    _, original = session_fixture(tmp_path)
    args = options(backend, '--base-url', 'http://localhost:8000/v1',
                   '--history-policy', 'current_sensors_last_result_notes')
    validate_backend_arguments(args)
    session = create_declarative_session(args, public_task=original.public_task,
        policy=original.episode_configuration, output=tmp_path, assembler=original.assembler)
    assert isinstance(session, ContextChatSession) and session.recent_notes
    assert session.backend == backend
    assert (isinstance(session, ResponsesContextSession)) == (backend == 'openai_responses')


def test_gateway_responses_composes_routing_and_context(tmp_path, monkeypatch):
    from wia.runtime.agent.vercel_session import VercelResponsesSession
    env, session = session_fixture(tmp_path, session_type=VercelResponsesSession,
        history_policy='current_sensors_last_result_notes')
    session.model = 'openai/transport-test-fixture'
    seen = mock_http(monkeypatch, [responses_reply('{"action":"DONE"}', providerMetadata={
        'gateway': {'routing': {'finalProvider': 'openai'}}})])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=1))
    assert result['termination_reason'] == 'model_done'
    assert seen[0]['url'].endswith('/responses')
    assert seen[0]['payload']['providerOptions']['gateway'] == {'only': ['openai'], 'caching': 'auto'}
    assert (tmp_path/'gateway_routing.jsonl').exists()


def test_failed_transport_is_counted_as_unknown_billing(tmp_path, monkeypatch):
    from core.agent.decision import ModelError
    from wia.evaluation.telemetry import totals
    env, session = session_fixture(tmp_path, session_type=ContextChatSession,
        history_policy='current_sensors_last_result_notes')

    async def fail(*args):
        raise ModelError('API model request failed or timed out.')

    monkeypatch.setattr(session, '_send_payload', fail)
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=1))
    assert result['termination_reason'] == 'model_error'
    records = [json.loads(line) for line in (tmp_path/'api_usage.jsonl').read_text().splitlines()]
    assert len(records) == 1 and records[0]['status'] == 'error'
    accounting = totals(records)
    assert accounting['requests'] == 1
    assert accounting['cost_usd'] == {'known_sum': 0, 'missing_requests': 1}
    assert accounting['input_tokens']['missing_requests'] == 1
    assert env.calls == []
