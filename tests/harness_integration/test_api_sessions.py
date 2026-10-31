"""Verify API dialects and complete histories. Run: python -m pytest tests/test_api_sessions.py."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from core.agent.api_sessions import ChatCompletionsEpisodeSession, ResponsesEpisodeSession
from core.agent.decision import DecisionInput, ModelError
from wia.runtime.agent.zero_shot import run_episode
from core.sensor_observations import SensorReadings
from harness_integration.test_local_http_episode import local_episode


def session_fixture(tmp_path, session_type=ChatCompletionsEpisodeSession, **kwargs):
    env, original = local_episode(tmp_path)
    session = session_type(assembler=original.assembler, public_task=original.public_task,
        episode_configuration=original.episode_configuration, output=tmp_path,
        model="same-model", base_url="https://provider.example/v1", api_key="SECRET_KEY", **kwargs)
    return env, session


def mock_http(monkeypatch, replies):
    seen, outputs = [], iter(replies)

    def post(url, *, json, headers, timeout):
        seen.append({"url": url, "payload": deepcopy(json), "timeout": timeout})
        data = next(outputs)

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return data

        return Response()

    monkeypatch.setattr("core.agent.api_sessions.requests.post", post)
    return seen


def chat_reply(text, **message):
    return {"choices": [{"message": {"content": text, **message}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3}}


def test_sliding_window_preserves_initial_and_current_tool_chains(tmp_path):
    _, session = session_fixture(tmp_path, provider='vllm', sliding_window_tokens=10000,
                                 max_output_tokens=1000)
    def turn(label):
        return [{'role': 'user', 'content': label}, {'role': 'assistant', 'content': None,
            'tool_calls': [{'id': label, 'type': 'function', 'function': {'name': 'depth', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': label, 'content': label}]
    initial, oldest, newest, current = [turn(x) for x in ('initial', 'oldest', 'newest', 'current')]
    messages = deepcopy(initial + oldest + newest + current)
    async def count(payload, deadline):
        return sum(x['role'] == 'user' for x in payload['messages']) * 2000, 10000
    session._count_window_tokens = count
    asyncio.run(session._fit_sliding_window({'messages': messages}, 1e30))
    assert messages == initial + newest + current
    assert session.sliding_window_dropped_turns == 1
    assert session.history_policy == 'sliding_complete_turns_with_pinned_initial_task'


def test_sliding_window_refuses_to_drop_current_observation_or_initial_task(tmp_path):
    _, session = session_fixture(tmp_path, provider='vllm', sliding_window_tokens=10000,
                                 max_output_tokens=1000)
    messages = [{'role': 'user', 'content': 'initial'}, {'role': 'assistant', 'content': 'decision'},
                {'role': 'user', 'content': 'current'}, {'role': 'assistant', 'tool_calls': [{'id': 'live'}]},
                {'role': 'tool', 'tool_call_id': 'live', 'content': 'current depth'}]
    original = deepcopy(messages)
    async def count(payload, deadline):
        return 9000, 10000
    session._count_window_tokens = count
    with pytest.raises(ModelError, match='Pinned initial task'):
        asyncio.run(session._fit_sliding_window({'messages': messages}, 1e30))
    assert messages == original


def test_sliding_server_context_rejection_retries_without_replaying_an_action(tmp_path):
    env, session = session_fixture(tmp_path, provider='vllm', sliding_window_tokens=10000,
                                 max_output_tokens=1000)
    session.history = [{'role': 'user', 'content': 'initial'}, {'role': 'assistant', 'content': 'first'},
                       {'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'second'}]
    async def count(payload, deadline):
        return 100, 10000
    seen = []
    async def post(endpoint, payload, deadline):
        seen.append(deepcopy(payload['messages']))
        if len(seen) == 1:
            raise ModelError('API model request failed (HTTP 400, context_length_exceeded).')
        return chat_reply('{"action":"DONE"}')
    session._count_window_tokens, session._post = count, post
    request = session.prepare_request(DecisionInput(env.source.agent_observation(), ('DONE',)))
    asyncio.run(session.decide(request))
    assert [sum(x['role'] == 'user' for x in m) for m in seen] == [3, 2]
    assert seen[0][-1] == seen[1][-1] and seen[0][:2] == seen[1][:2]
    assert env.calls == []


def responses_reply(text, *, phase="final_answer", **fields):
    return {"status": "completed", "output": [{"type": "message", "id": "msg-1",
        "role": "assistant", "status": "completed", "phase": phase,
        "content": [{"type": "output_text", "text": text, "annotations": []}]}],
        "usage": {"input_tokens": 10, "output_tokens": 3}, **fields}


@pytest.mark.parametrize("session_type", [ChatCompletionsEpisodeSession, ResponsesEpisodeSession])
def test_two_turn_multimodal_history_is_complete(tmp_path, monkeypatch, session_type):
    env, session = session_fixture(tmp_path, session_type)
    texts = ['{"command":{"action":"BASE_FWD"},"world_observation":"Visible workspace."}',
             '{"command":{"action":"DONE"},"world_observation":"Still visible."}']
    adapter = chat_reply if session_type is ChatCompletionsEpisodeSession else responses_reply
    seen = mock_http(monkeypatch, [adapter(text) for text in texts])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=2))
    assert result["termination_reason"] == "model_done"
    assert env.calls == ["BASE_FWD"]
    key = "messages" if session_type is ChatCompletionsEpisodeSession else "input"
    history = [record["payload"][key] for record in seen]
    assert [len(items) for items in history] == [1, 3]
    assert history[1][0] == history[0][0]
    image_type = "image_url" if key == "messages" else "input_image"
    assert [sum(item["type"] == image_type for item in items[-1]["content"]) for items in history] == [9, 4]
    assert session.provenance()["history_owner"] == "harness"
    assert session.description()["model"] == "same-model"
    assert "SECRET_KEY" not in json.dumps(session.provenance())


@pytest.mark.parametrize("session_type", [ChatCompletionsEpisodeSession, ResponsesEpisodeSession])
def test_api_trace_keeps_transport_and_backend_history_identity(tmp_path, monkeypatch, session_type):
    env, session = session_fixture(tmp_path, session_type)
    text = '{"command":{"action":"DONE"},"world_observation":"Visible workspace."}'
    adapter = chat_reply if session_type is ChatCompletionsEpisodeSession else responses_reply
    seen = mock_http(monkeypatch, [adapter(text)])
    result = asyncio.run(run_episode(env, session, tmp_path, max_steps=1))
    assert result["termination_reason"] == "model_done"
    inputs = [json.loads(line) for line in (tmp_path / "model_inputs.jsonl").read_text().splitlines()]
    step = next(row for row in inputs if row["event"] == "step_input")
    assert step["transport"] == "structured" and step["response_schema"] is not None
    key = "messages" if session_type is ChatCompletionsEpisodeSession else "input"
    assert step["prompt"] == seen[0]["payload"][key][0]["content"][0]["text"]
    outputs = [json.loads(line) for line in (tmp_path / "model_outputs.jsonl").read_text().splitlines()]
    response = next(row for row in outputs if row["event"] == "model_output")
    assert response["backend"] == session.backend and response["history_owner"] == "harness"
    steps = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    metadata = steps[0]["attempts"][0]["model_metadata"]
    assert metadata["backend"] == session.backend and metadata["history_owner"] == "harness"
    assert "SECRET_KEY" not in json.dumps([inputs, outputs, steps])


def test_structured_trace_uses_explicit_api_flag_and_preserves_legacy_defaults():
    from core.agent.trace_log import uses_structured_transport
    assert uses_structured_transport({"backend": "codex", "structured_output": False})
    assert uses_structured_transport({"backend": "local_http"})
    assert not uses_structured_transport({"backend": "local_http", "structured_output": False})
    for backend in ("openai_chat_completions", "openai_responses"):
        assert uses_structured_transport({"backend": backend, "structured_output": True})
        assert not uses_structured_transport({"backend": backend, "structured_output": False})
        assert not uses_structured_transport({"backend": backend})


@pytest.mark.parametrize("provider,field", [("openai", "max_completion_tokens"),
    ("vllm", "max_tokens"), ("compatible", "max_tokens")])
def test_chat_dialect_uses_only_explicit_options(tmp_path, monkeypatch, provider, field):
    env, session = session_fixture(tmp_path, provider=provider, max_output_tokens=777, reasoning_effort="low")
    seen = mock_http(monkeypatch, [chat_reply('{"action":"DONE"}')])
    request = session.prepare_request(DecisionInput(env.source.agent_observation(), ("DONE",)))
    reply = asyncio.run(session.decide(request))
    payload = seen[0]["payload"]
    assert payload[field] == 777 and payload["reasoning_effort"] == "low"
    assert "temperature" not in payload and "chat_template_kwargs" not in payload
    assert "response_format" not in payload
    assert reply.metadata["backend"] == "openai_chat_completions"
    assert reply.metadata["usage"] == {"completion_tokens": 3}


def test_vllm_template_options_are_opt_in(tmp_path):
    _, session = session_fixture(tmp_path, provider="vllm", temperature=0,
                                chat_template_kwargs={"enable_thinking": False})
    options = session.generation_options(None)
    assert options["temperature"] == 0
    assert options["chat_template_kwargs"] == {"enable_thinking": False}
    for provider in ("openai", "compatible"):
        with pytest.raises(ValueError, match="only supported"):
            session_fixture(tmp_path, provider=provider, chat_template_kwargs={})


def depth_request(env, session):
    observation = env.source.agent_observation()
    observation = replace(observation, sensors=SensorReadings(
        head_depth_m=np.full((12, 16), .75, dtype=np.float32)))
    return session.prepare_request(DecisionInput(observation, ("DONE",)))


def test_diagnostic_history_keeps_only_two_verbatim_public_outputs(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path, provider="vllm", history_text_turns=2)
    replies = [chat_reply(json.dumps({'command': {'action': 'DONE'},
                                    'world_observation': f'Public snapshot {index}.'}))
               for index in range(4)]
    seen = mock_http(monkeypatch, replies)
    for _ in range(4):
        request = session.prepare_request(DecisionInput(env.source.agent_observation(), ('DONE',)))
        assert 'Diagnostic conversation compression' in request.structured_prompt()
        asyncio.run(session.decide(request))
    assert [len(row['payload']['messages']) for row in seen] == [1, 3, 5, 5]
    last = seen[-1]['payload']['messages']
    assert 'Public snapshot 0.' not in json.dumps(last)
    assert 'Public snapshot 1.' in last[1]['content']
    assert 'Public snapshot 2.' in last[3]['content']
    assert all(isinstance(item['content'], str) for item in last[:-1])
    assert sum(item['type'] == 'image_url' for item in last[-1]['content']) == 4
    assert session.provenance()['history_policy'] == 'diagnostic_last_2_public_final_outputs'


def test_truncated_response_retains_usage_without_reasoning(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path)
    mock_http(monkeypatch, [{'choices': [{'finish_reason': 'length', 'message': {'content': None,
        'reasoning_content': 'PRIVATE_CANARY'}}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 32}}])
    request = session.prepare_request(DecisionInput(env.source.agent_observation(), ('DONE',)))
    with pytest.raises(ModelError, match="finish_reason='length'"):
        asyncio.run(session.decide(request))
    assert session.provenance()['transport_usage'] == {'prompt_tokens': 10, 'completion_tokens': 32}
    assert 'PRIVATE_CANARY' not in json.dumps(session.provenance())


def test_json_schema_decoding_is_explicit_and_keeps_tool_registration(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path, json_schema_output=True)
    call = {'id': 'decision-1', 'type': 'function', 'function': {'name': 'submit_decision',
            'arguments': '{"command":{"action":"DONE"},"world_observation":"Visible."}'}}
    seen = mock_http(monkeypatch, [chat_reply(None, tool_calls=[call])])
    request = depth_request(env, session)
    asyncio.run(session.decide(request))
    payload = seen[0]['payload']
    assert payload['tools'][-1]['function']['parameters'] == request.structured_schema
    assert payload['tool_choice'] == 'required' and payload['parallel_tool_calls'] is False
    assert [item['function']['name'] for item in payload['tools']] == ['read_depth_pixels', 'read_depth_region', 'read_rgb_camera', 'submit_decision']
    assert session.provenance()['guided_decoding'] is True


def test_rgb_resize_trace_matches_wire_and_preserves_native_depth(tmp_path, monkeypatch):
    import base64
    import io
    from PIL import Image
    env, session = session_fixture(tmp_path, image_max_edge=8)
    original = env.source.agent_observation()
    seen = mock_http(monkeypatch, [chat_reply('{"command":{"action":"DONE"},"world_observation":"Visible."}')])
    summary = asyncio.run(run_episode(env, session, tmp_path, max_steps=1))
    assert summary['termination_reason'] == 'model_done'
    row = next(json.loads(line) for line in (tmp_path / 'model_inputs.jsonl').read_text().splitlines()
               if json.loads(line).get('event') == 'step_input')
    assert 'u_native=(u_model+0.5)' in row['prompt']
    wire = seen[0]['payload']['messages'][0]['content'][1:]
    assert len(wire) == len(row['images']) == 9
    for item, logged in zip(wire, row['images']):
        png = base64.b64decode(item['image_url']['url'].split(',', 1)[1])
        assert png == (tmp_path / logged['path']).read_bytes()
        with Image.open(io.BytesIO(png)) as image:
            assert max(image.size) <= 8
            assert list(reversed(image.size)) == logged['shape'][:2]
    assert original.agentview_rgb.shape == (12, 16, 3)


def test_schema_decision_function_drives_existing_loop_without_output_repair(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path, json_schema_output=True)
    def submit(identifier, action):
        return chat_reply(None, tool_calls=[{'id': identifier, 'type': 'function', 'function': {
            'name': 'submit_decision', 'arguments': json.dumps({'command': {'action': action},
                                                               'world_observation': 'Visible workspace.'})}}])
    seen = mock_http(monkeypatch, [submit('first', 'BASE_FWD'), submit('second', 'DONE')])
    summary = asyncio.run(run_episode(env, session, tmp_path, max_steps=2))
    assert env.calls == ['BASE_FWD'] and summary['termination_reason'] == 'model_done'
    assert seen[1]['payload']['messages'][1]['tool_calls'][0]['id'] == 'first'
    assert seen[1]['payload']['messages'][2]['tool_call_id'] == 'first'
    assert session.provenance()['history_policy'] == 'full_no_truncation'


def test_chat_tools_are_bound_to_current_snapshot_and_preserved(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path)
    call = {"id": "tool-A", "type": "function", "function": {"name": "read_depth_pixels",
        "arguments": json.dumps({"sensor": "head_depth_m", "pixels_uv": [[4, 0]]})}}
    seen = mock_http(monkeypatch, [chat_reply(None, tool_calls=[call], reasoning_content="PRIVATE_CANARY"),
                                  chat_reply('{"action":"DONE"}')])
    request = depth_request(env, session)
    reply = asyncio.run(session.decide(request))
    assert len(seen) == 2 and seen[0]["url"].endswith("/chat/completions")
    history = seen[1]["payload"]["messages"]
    assert history[1]["tool_calls"] == [call]
    assert history[2]["tool_call_id"] == "tool-A"
    result = json.loads(history[2]["content"])
    assert result["snapshot_id"] == request.sensor_inputs.snapshot_id
    assert result["optical_z_m"] == [.75]
    assert reply.metadata["depth_query_count"] == 1
    assert "PRIVATE_CANARY" not in json.dumps(session.history)


def test_responses_replays_opaque_reasoning_phase_and_tool_call_ids(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path, ResponsesEpisodeSession, reasoning_effort="high")
    reasoning = {"type": "reasoning", "id": "rs-1", "summary": [],
                 "encrypted_content": "OPAQUE_ENCRYPTED_CANARY"}
    commentary = responses_reply("Checking supplied depth.", phase="commentary")["output"][0]
    call = {"type": "function_call", "id": "fc-item", "call_id": "call-actual",
            "name": "read_depth_pixels", "arguments": '{"sensor":"head_depth_m","pixels_uv":[[4,0]]}',
            "status": "completed"}
    final = responses_reply('{"action":"DONE"}')
    seen = mock_http(monkeypatch, [{"status": "completed", "output": [reasoning, commentary, call]}, final,
                                  responses_reply('{"action":"DONE"}')])
    events = []
    session.attach_input_logger(events.append)
    session.attach_output_logger(events.append)
    request = depth_request(env, session)
    reply = asyncio.run(session.decide(request))
    payload = seen[1]["payload"]
    assert payload["store"] is False and "previous_response_id" not in payload
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["input"][1:4] == [reasoning, commentary, call]
    assert payload["input"][4]["call_id"] == "call-actual"
    assert payload["input"][4]["type"] == "function_call_output"
    assert "strict" in payload["tools"][0] and "function" not in payload["tools"][0]
    assert reply.raw_text == '{"action":"DONE"}'
    assert session.history[1] == reasoning
    assert "OPAQUE_ENCRYPTED_CANARY" not in json.dumps(events)
    assert "OPAQUE_ENCRYPTED_CANARY" not in (tmp_path / "openai_responses_session.json").read_text()
    asyncio.run(session.decide(session.prepare_request(DecisionInput(request.observation, ("DONE",)))))
    assert seen[2]["payload"]["input"][:6] == session.history[:6]


@pytest.mark.parametrize("output", [
    {"status": "failed", "output": []},
    {"status": "incomplete", "output": []},
    {"status": "completed", "output": []},
    {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "No."}]}]},
    {"status": "completed", "output": [{"type": "web_search_call", "id": "forbidden"}]},
])
def test_responses_failures_do_not_produce_actions(tmp_path, monkeypatch, output):
    env, session = session_fixture(tmp_path, ResponsesEpisodeSession)
    mock_http(monkeypatch, [output])
    request = session.prepare_request(DecisionInput(env.source.agent_observation(), ("DONE",)))
    with pytest.raises(ModelError):
        asyncio.run(session.decide(request))
    assert env.calls == [] and session.turn_count == 0


@pytest.mark.parametrize("session_type", [ChatCompletionsEpisodeSession, ResponsesEpisodeSession])
def test_raw_invalid_final_text_is_not_repaired(tmp_path, monkeypatch, session_type):
    env, session = session_fixture(tmp_path, session_type)
    factory = chat_reply if session_type is ChatCompletionsEpisodeSession else responses_reply
    mock_http(monkeypatch, [factory("MV_FWD")])
    request = session.prepare_request(DecisionInput(env.source.agent_observation(), ("DONE",)))
    reply = asyncio.run(session.decide(request))
    assert reply.raw_text == "MV_FWD" and session.history[-1]["role"] == "assistant"


def test_overall_timeout_does_not_retry_action(tmp_path, monkeypatch):
    env, session = session_fixture(tmp_path, timeout_s=.01)

    async def stalled(*args, **kwargs):
        await asyncio.sleep(.1)

    monkeypatch.setattr("core.agent.api_sessions.asyncio.to_thread", stalled)
    request = session.prepare_request(DecisionInput(env.source.agent_observation(), ("DONE",)))
    with pytest.raises(ModelError, match="timeout|timed out"):
        asyncio.run(session.decide(request))
    assert env.calls == []


@pytest.mark.parametrize("failure", [
    {"choices": [{"finish_reason": "length", "message": {"content": '{"action":"DONE"}'}}]},
    {"choices": [{"finish_reason": "error", "message": {"content": '{"action":"DONE"}'}}]},
    {"choices": [{"finish_reason": "tool_calls", "message": {"content": '{"action":"DONE"}'}}]},
    chat_reply('{"action":"DONE"}', role="user"),
    chat_reply(None, refusal="Refused."),
    chat_reply("<think>PRIVATE_REASONING</think>"),
    chat_reply(None, tool_calls=[{"id": "bad", "type": "function", "function": {
        "name": "move_robot", "arguments": "{}"}}]),
])
def test_chat_failures_and_unauthorized_tools_do_not_produce_actions(tmp_path, monkeypatch, failure):
    env, session = session_fixture(tmp_path)
    mock_http(monkeypatch, [failure])
    request = depth_request(env, session)
    with pytest.raises(ModelError):
        asyncio.run(session.decide(request))
    assert env.calls == [] and session.history == []


def test_request_usage_includes_failed_decision_and_marks_unknown(tmp_path, monkeypatch):
    import time
    _, session = session_fixture(tmp_path)
    mock_http(monkeypatch, [chat_reply(''), {'choices': []}])
    asyncio.run(session._post('/chat/completions', {}, time.monotonic() + 10))
    asyncio.run(session._post('/chat/completions', {}, time.monotonic() + 10))
    assert session.usage == {}
    assert session.transport_usage == {'completion_tokens': 3}
    assert session.provenance()['usage_coverage'] == {'attempted_requests': 2, 'reported_requests': 1,
        'unknown_usage_requests': 1, 'complete': False}


def test_failed_http_request_has_unknown_usage(tmp_path, monkeypatch):
    import time
    import requests
    _, session = session_fixture(tmp_path)
    def fail(*args, **kwargs):
        raise requests.Timeout('test timeout')
    monkeypatch.setattr('core.agent.api_sessions.requests.post', fail)
    with pytest.raises(ModelError):
        asyncio.run(session._post('/chat/completions', {}, time.monotonic() + 10))
    assert session.provenance()['usage_coverage']['unknown_usage_requests'] == 10
