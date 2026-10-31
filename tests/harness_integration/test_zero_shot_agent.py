"""Decision boundaries, fresh observations, termination and local deployment seam."""
import asyncio
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from core.action_interface import BackendCapabilities, action_result
from core.action_units import DONE_ATOM
from core.agent.backends import CodexAgent, LocalHTTPAgent
from core.agent.decision import DecisionInput, ModelError, ModelReply, parse_action
from wia.runtime.agent.zero_shot import run_episode
from harness_integration.test_agent_observation import source_and_rig


class Environment:
    capabilities = BackendCapabilities(base_translation=True, base_rotation=True)
    initialization = {"scene_reset": False}

    def __init__(self, *, failed=False, stale=False):
        self.source, self.rig = source_and_rig()
        self.source.capture(action_boundary=True)
        self.calls, self.failed, self.stale = [], failed, stale

    async def observe(self):
        return self.source.agent_observation()

    async def execute(self, action):
        self.calls.append(action)
        if not self.stale:
            self.source.after_action(action_result(action, completed=not self.failed, timeout=False,
                                                  requested={}, actual={}, message="completed",
                                                  reason="arm_motion_blocked" if self.failed else "completed"))
        return {"ok": not self.failed, "results": [self.source.last_action_feedback.to_dict()]
                if self.source.last_action_feedback else []}


class Agent:
    def __init__(self, *outputs):
        self.outputs, self.inputs = iter(outputs), []

    def description(self):
        return {"backend": "test"}

    async def decide(self, request):
        self.inputs.append(request)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return ModelReply(output)


@pytest.mark.parametrize("raw", [
    '{"action":"BASE_FWD","distance":1}', '{"actions":["BASE_FWD"]}',
    '{"action":"BASE_FWD"}\n{"action":"BASE_BACK"}', '[{"action":"DONE"}]',
    '{"action":"DONE","action":"BASE_FWD"}', '{"action":"BASE_FWD(0.5)"}',
    '{"action":"MV_FWD"}', '{"action":null}', '```json\n{"action":"DONE"}\n```',
])
def test_invalid_outputs_do_not_become_a_default_robot_action(raw):
    with pytest.raises(ValueError):
        parse_action(raw, Environment.capabilities.available_actions + (DONE_ATOM,))


def test_loop_retries_with_same_observation_then_executes_once_and_reobserves(tmp_path):
    env = Environment()
    agent = Agent('{"action":"BASE_FWD","distance":1}', '{"action":"BASE_FWD"}', '{"action":"DONE"}')
    summary = asyncio.run(run_episode(env, agent, tmp_path, success_provider=lambda: False))
    assert env.calls == ["BASE_FWD"] and summary["executed_actions"] == env.calls
    assert summary["model_declared_done"] and summary["task_success"] is False
    assert summary["termination_reason"] == "model_done"
    assert agent.inputs[0].observation is agent.inputs[1].observation and agent.inputs[1].feedback
    assert agent.inputs[2].observation.action_step == 1
    assert agent.inputs[2].observation.snapshot_id > agent.inputs[0].observation.snapshot_id
    assert agent.inputs[2].observation.action_history[-1].action == "BASE_FWD"
    records = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert records[0]["attempts"][0]["parse_error"] and records[0]["next_observation"]
    assert records[1]["action_result"] is None and records[1]["termination_reason"] == "model_done"
    assert (tmp_path / records[0]["observation"]).exists()
    assert (tmp_path / records[0]["next_observation"]).exists()
    assert "BASE_" in agent.inputs[0].prompt() and "MV_FWD" not in agent.inputs[0].schema["properties"]["action"]["enum"]


@pytest.mark.parametrize("environment, outputs, expected, calls", [
    ({}, ['{"action":"UNKNOWN"}'] * 3, "invalid_output", []),
    ({}, ['{"action":"BASE_FWD"}'], "max_steps", ["BASE_FWD"]),
    ({"failed": True}, ['{"action":"BASE_FWD"}'], "execution_error", ["BASE_FWD"]),
    ({"stale": True}, ['{"action":"BASE_FWD"}'], "observation_error", ["BASE_FWD"]),
    ({}, [ModelError("unavailable")], "model_error", []),
])
def test_bounded_failure_and_step_termination(environment, outputs, expected, calls, tmp_path):
    env = Environment(**environment)
    summary = asyncio.run(run_episode(env, Agent(*outputs), tmp_path, max_steps=1))
    assert summary["termination_reason"] == expected and env.calls == calls
    assert summary["task_success"] is None
    assert json.loads((tmp_path / "summary.json").read_text())["termination_reason"] == expected


def test_evaluator_success_stays_outside_model_input(tmp_path):
    env, agent = Environment(), Agent()
    summary = asyncio.run(run_episode(env, agent, tmp_path, success_provider=lambda: True))
    assert summary["task_success"] and summary["termination_reason"] == "environment_success"
    assert not agent.inputs and not env.calls and not summary["model_declared_done"]


def test_cancellation_writes_summary(tmp_path):
    class Cancel(Agent):
        async def decide(self, request):
            raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_episode(Environment(), Cancel(), tmp_path))
    assert json.loads((tmp_path / "summary.json").read_text())["termination_reason"] == "cancelled"


@pytest.mark.parametrize("tool_attempt", [False, True])
def test_codex_explicit_model_images_and_discarded_reasoning_events(tmp_path, tool_attempt):
    binary = tmp_path / "fake-codex"
    binary.write_text("#!/usr/bin/env python3\n" + '''import json, sys
from pathlib import Path
args=sys.argv[1:]
assert args[args.index('--model')+1]=='gpt-6-luna'
assert 'model_reasoning_effort="medium"' in args
assert '--ignore-user-config' in args and '--ephemeral' in args
assert 'features.code_mode_host=false' in args and '--no-daemon' in args
assert args.count('--image')==2
assert 'shell_tool' in args and 'plugins' in args
assert 'Show-Harness' not in str(Path.cwd())
schema=json.loads(Path(args[args.index('--output-schema')+1]).read_text())
assert 'BASE_FWD' in schema['properties']['action']['enum']
print(json.dumps({'type':'item.completed','item':{'type':'reasoning','text':'PRIVATE_THOUGHT'}}))
''' + ("print(json.dumps({'type':'item.completed','item':{'type':'command_execution','command':'forbidden'}}))\n" if tool_attempt else "") + '''Path(args[args.index('--output-last-message')+1]).write_text('{"action":"BASE_FWD"}')
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10}}))
''')
    binary.chmod(0o755)
    env = Environment()
    agent = CodexAgent(executable=str(binary))
    request = DecisionInput(asyncio.run(env.observe()), env.capabilities.available_actions + (DONE_ATOM,))
    if tool_attempt:
        with pytest.raises(ModelError, match="tool operation"):
            asyncio.run(agent.decide(request))
    else:
        reply = asyncio.run(agent.decide(request))
        assert reply.raw_text == '{"action":"BASE_FWD"}'
        assert "PRIVATE_THOUGHT" not in json.dumps(reply.metadata)


def test_local_http_vision_endpoint_receives_same_observation_without_reasoning_logs(tmp_path):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"choices": [{"message": {"content": '{"action":"DONE"}',
                                                        "reasoning_content": "PRIVATE_THOUGHT"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = Environment()
        agent = LocalHTTPAgent(base_url=f"http://127.0.0.1:{server.server_port}/v1", model="local-vision-model")
        summary = asyncio.run(run_episode(env, agent, tmp_path))
        assert summary["termination_reason"] == "model_done"
        content = received[0]["messages"][0]["content"]
        assert len(content) == 3 and received[0]["response_format"]["json_schema"]["strict"]
        rgb = np.array(Image.open(io.BytesIO(base64.b64decode(content[1]["image_url"]["url"].split(",")[1]))))
        assert np.array_equal(rgb, env.source.agent_observation().agentview_rgb)
        assert "PRIVATE_THOUGHT" not in (tmp_path / "steps.jsonl").read_text()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_accidentally_exposed_local_thought_text_is_discarded_before_logging(tmp_path):
    summary = asyncio.run(run_episode(Environment(), Agent('<think>PRIVATE_THOUGHT</think>{"action":"DONE"}'), tmp_path))
    assert summary["termination_reason"] == "model_error"
    assert "PRIVATE_THOUGHT" not in (tmp_path / "steps.jsonl").read_text()


def test_physical_runner_continues_past_thirty_decisions(tmp_path):
    env = Environment()
    agent = Agent(*(['{"action":"BASE_FWD"}'] * 31), '{"action":"DONE"}')
    summary = asyncio.run(run_episode(env, agent, tmp_path, max_steps=300, success_provider=lambda: False))
    assert len(env.calls) == 31
    assert summary['decision_calls'] == 32 and summary['termination_reason'] == 'model_done'
    assert agent.inputs[29].steps_remaining == 271
    assert agent.inputs[30].steps_remaining == 270
