"""Validate request retries and private statistics. Run pytest -q tests/harness_integration/test_api_accounting.py."""
import asyncio
import json

import pytest
import requests

from core.agent.api_sessions import ChatCompletionsEpisodeSession, ResponsesEpisodeSession
from wia.runtime.agent.context_session import ContextChatSession
from wia.runtime.agent.responses_context import ResponsesContextSession
from wia.runtime.agent.zero_shot import run_episode
from harness_integration.test_api_sessions import session_fixture, chat_reply, responses_reply, mock_http


@pytest.mark.parametrize('session_type', [ChatCompletionsEpisodeSession, ResponsesEpisodeSession,
                                         ContextChatSession, ResponsesContextSession])
@pytest.mark.parametrize('failure,limit', [(requests.ConnectionError, 3), (requests.Timeout, 10)])
def test_limits_count_each_failed_attempt_without_robot_motion(tmp_path, monkeypatch, session_type, failure, limit):
    env, session = session_fixture(tmp_path, session_type=session_type)
    payloads = []
    def fail(url, **kwargs):
        payloads.append(kwargs['json'])
        raise failure('private transport fixture')
    monkeypatch.setattr('core.agent.api_sessions.requests.post', fail)
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=8))
    assert result['termination_reason'] == 'model_error'
    assert result['decision_calls'] == 1 and result['model_calls'] == limit
    assert len(payloads) == limit and all(p == payloads[0] for p in payloads)
    assert env.calls == [] and result['num_executed_actions'] == 0
    a = result['request_accounting']
    assert a['api_failures'] == limit and a['api_retries'] == limit-1
    assert a['api_timeouts'] == (limit if failure is requests.Timeout else 0)
    records = [json.loads(x) for x in (tmp_path/'api_usage.jsonl').read_text().splitlines()]
    assert len(records) == limit and all(x['cost_usd'] is None for x in records)


def test_transport_retry_then_tool_round_then_final_is_one_decision(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path, session_type=ContextChatSession,
        history_policy='current_sensors_last_result_notes')
    replies = iter([requests.ConnectionError('fixture'), chat_reply(None, tool_calls=[{
        'id': 'note1', 'type': 'function', 'function': {'name': 'episode_note',
        'arguments': '{"operation":"read"}'}}]), chat_reply('{"action":"DONE"}')])
    def post(*args, **kwargs):
        value = next(replies)
        if isinstance(value, Exception):
            raise value
        class Response:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return value
        return Response()
    monkeypatch.setattr('core.agent.api_sessions.requests.post', post)
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=1))
    assert result['termination_reason'] == 'model_done'
    assert result['agent_decisions'] == result['decision_calls'] == 1
    assert result['model_calls'] == 3
    a = result['request_accounting']
    assert (a['api_successes'], a['api_failures'], a['api_retries']) == (2, 1, 1)
    assert a['tool_calls']['requested']['by_name'] == {'episode_note': 1}
    assert a['tool_calls']['executed']['by_name'] == {'episode_note': 1}
    assert env.calls == []


def test_retries_reset_after_successful_request(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path)
    calls = 0
    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls % 3:
            raise requests.ConnectionError('fixture')
        class Response:
            def raise_for_status(self):
                pass
            def json(self):
                return chat_reply('{"action":"BASE_FWD"}' if calls == 3 else '{"action":"DONE"}')
        return Response()
    monkeypatch.setattr('core.agent.api_sessions.requests.post', post)
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=2))
    assert result['termination_reason'] == 'model_done' and result['model_calls'] == 6
    assert result['agent_decisions'] == 2 and env.calls == ['BASE_FWD']


@pytest.mark.parametrize('status,limit', [(429, 3), (503, 3), (408, 10), (504, 10)])
def test_http_status_classification(tmp_path, monkeypatch, status, limit):
    env, session = session_fixture(tmp_path, session_type=ContextChatSession)
    calls = 0
    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        response = requests.Response()
        response.status_code = status
        response._content = b'fixture error'
        raise requests.HTTPError('fixture', response=response)
    monkeypatch.setattr('core.agent.api_sessions.requests.post', post)
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=2))
    assert result['model_calls'] == calls == limit
    assert result['request_accounting']['api_timeouts'] == (limit if status in (408, 504) else 0)
    assert env.calls == []


def test_legacy_local_http_uses_same_attempt_ledger(tmp_path, monkeypatch):
    from core.agent.local_http_session import LocalHTTPEpisodeSession
    env, session = session_fixture(tmp_path, session_type=LocalHTTPEpisodeSession)
    mock_http(monkeypatch, [chat_reply('{"action":"DONE"}')])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=1))
    assert result['model_calls'] == 1 and result['decision_calls'] == 1
    assert result['request_accounting']['api_successes'] == 1


def test_cancelled_http_attempt_is_logged_once_without_retry(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path)
    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError()
    monkeypatch.setattr('core.agent.api_sessions.asyncio.to_thread', cancelled)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_episode(env, session, tmp_path, max_steps=2))
    rows = [json.loads(x) for x in (tmp_path/'api_usage.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['status'] == 'cancelled'
    assert session.usage_requests == 1 and session.api_retries == 0
    assert env.calls == []
