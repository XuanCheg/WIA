"""Metric protocol, atomic macro boundaries, partial outcomes and Agent budgets."""
import asyncio
import json
import threading
from dataclasses import replace

import pytest
import requests

from core.action_compiler import MacroLimits, ParameterizedAction, PrimitiveResolution, compile_action
from core.action_interface import ActionExecutor, BackendCapabilities, action_result
from core.agent.decision import DecisionInput, parse_action
from wia.runtime.agent.zero_shot import run_episode
from core.observations import AgentObservation
from harness_integration.test_agent_observation import source_and_rig
from harness_integration.test_agent_control import control_http as shared_control_http
from harness_integration.test_zero_shot_agent import Agent


CAPABILITIES = BackendCapabilities(base_translation=True, base_rotation=True, ee_translation=True,
                                   ee_rotation=True, gripper=True, trunk_height=True)
RESOLUTION = PrimitiveResolution(0.05, 5, 0.01, 5, 0.01)
LIMITS = MacroLimits()
AVAILABLE = CAPABILITIES.available_actions + ("DONE",)
control_http = shared_control_http


@pytest.mark.parametrize("command,count,effective,error", [
    ({"action": "BASE_RIGHT", "distance_m": 0.10}, 2, 0.10, 0),
    ({"action": "BASE_FWD", "distance_m": 0.40}, 8, 0.40, 0),
    ({"action": "BASE_TURN_LEFT", "angle_deg": 30}, 6, 30, 0),
    ({"action": "BASE_FWD", "distance_m": 0.12}, 2, 0.10, -0.02),
    ({"action": "BASE_FWD", "distance_m": 0.125}, 3, 0.15, 0.025),
    ({"action": "MV_UP", "distance_m": 0.06}, 6, 0.06, 0),
    ({"action": "TRUNK_DOWN", "distance_m": 0.02}, 2, 0.02, 0),
    ({"action": "GRASP"}, 1, None, None),
    ({"action": "DONE"}, 0, None, None),
])
def test_compiler_examples_and_explicit_half_up(command, count, effective, error):
    plan = compile_action(command, CAPABILITIES, RESOLUTION, LIMITS)
    assert plan.repeat_count == count and plan.effective_magnitude == effective
    assert plan.quantization_error == error


@pytest.mark.parametrize("raw", [
    '{"action":"BASE_FWD"}', '{"action":"BASE_FWD","angle_deg":5}',
    '{"action":"BASE_FWD","distance_m":-1}', '{"action":"BASE_FWD","distance_m":0}',
    '{"action":"BASE_FWD","distance_m":true}', '{"action":"BASE_FWD","distance_m":"0.1"}',
    '{"action":"BASE_FWD","distance_m":NaN}', '{"action":"BASE_FWD","distance_m":Infinity}',
    '{"action":"BASE_FWD","distance_m":0.1,"angle_deg":5}',
    '{"action":"BASE_FWD","distance_m":0.1,"other":1}',
    '{"action":"BASE_FWD","distance_m":0.1,"distance_m":0.2}',
    '{"action":"DONE","distance_m":null}', '{"action":"RELEASE","angle_deg":5}',
    '[{"action":"BASE_FWD","distance_m":0.1}]', '{"actions":["GRASP","RELEASE"]}',
])
def test_strict_metric_command_protocol(raw):
    with pytest.raises(ValueError):
        ParameterizedAction.parse(raw, AVAILABLE)


def test_zero_bounds_count_capabilities_and_actual_backend_resolution():
    for command, resolution, limits in (
        ({"action": "BASE_FWD", "distance_m": 0.01}, RESOLUTION, LIMITS),
        ({"action": "BASE_FWD", "distance_m": 1.21}, RESOLUTION, LIMITS),
        ({"action": "BASE_TURN_LEFT", "angle_deg": 121}, RESOLUTION, LIMITS),
        ({"action": "MV_UP", "distance_m": 0.25}, RESOLUTION, LIMITS),
        ({"action": "BASE_FWD", "distance_m": 0.10}, replace(RESOLUTION, base_translation_m=0.25), LIMITS),
        ({"action": "BASE_FWD", "distance_m": 0.10}, replace(RESOLUTION, base_translation_m=None), LIMITS),
    ):
        with pytest.raises(ValueError):
            compile_action(command, CAPABILITIES, resolution, limits)
    with pytest.raises(ValueError):
        compile_action({"action": "ROTATE_CW", "angle_deg": 5}, BackendCapabilities(), RESOLUTION, LIMITS)
    with pytest.raises(ValueError):
        parse_action('{"action":"BASE_FWD","distance_m":0.1}', AVAILABLE)


def make_executor(*, fail_on=None, cancel_on=None, raise_on=None, clock=None):
    source, rig = source_and_rig()
    source.capture(action_boundary=True)
    initial = source.agent_observation()
    calls, macro_results = [], []
    cancel = threading.Event()

    def execute(token, *, should_cancel):
        assert threading.current_thread() is threading.main_thread()
        calls.append(token)
        source.capture()  # Human render continues without an Agent boundary.
        assert source.agent_observation() is initial
        if raise_on == len(calls):
            raise RuntimeError("PRIVATE_CONTROLLER_ERROR")
        if cancel_on == len(calls):
            cancel.set()
        if clock is not None:
            clock[0] += 1
        cancelled = should_cancel()
        failed = fail_on == len(calls)
        return action_result(token, completed=not (failed or cancelled), timeout=False,
            requested={"distance_m": 0.05}, actual={"distance_m": 0.047}, message="completed",
            reason="arm_joint_limit" if failed else "cancelled" if cancelled else "completed",
            cancelled=cancelled, stop_steps=3, control_steps=10,
            pose_after="PRIVATE_WORLD_POSE")

    executor = ActionExecutor(execute, capabilities=CAPABILITIES, on_result=source.after_action)
    executor.on_macro_result = macro_results.append
    return executor, source, rig, calls, macro_results, cancel


@pytest.mark.parametrize("options,executed,status", [
    ({}, 5, "completed"), ({"fail_on": 3}, 3, "arm_joint_limit"),
    ({"cancel_on": 2}, 2, "cancelled"), ({"raise_on": 3}, 3, "controller_error"),
])
def test_single_macro_boundary_and_partial_failure(options, executed, status):
    executor, source, rig, calls, results, cancel = make_executor(**options)
    outcome = executor.execute_macro({"action": "BASE_RIGHT", "distance_m": 0.25},
        resolution=RESOLUTION, limits=LIMITS, should_cancel=cancel.is_set)
    assert len(calls) == executed and len(results) == 1 and results[0] is outcome
    assert outcome["execution"]["executed_count"] == executed
    assert outcome["execution"]["remaining_count"] == 5-executed
    assert outcome["reason"] == status
    observation = source.agent_observation()
    assert observation.action_step == 1 and len(observation.action_history) == 1
    assert observation.action_history[0].distance_m == 0.25
    assert rig.cycle == executed+2
    assert "PRIVATE_" not in json.dumps(observation.metadata())
    assert observation.last_action_feedback.estimated_motion == {}  # No base odometry adapter.
    if status == "completed":
        assert outcome["actual"]["distance_m"] == pytest.approx(0.235)
    assert AgentObservation.from_payload(observation.to_payload()).metadata() == observation.metadata()


def test_macro_deadline_cancels_current_primitive_then_stops(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("core.action_interface.time.monotonic", lambda: clock[0])
    executor, source, _, calls, _, cancel = make_executor(clock=clock)
    result = executor.execute_macro({"action": "BASE_RIGHT", "distance_m": 0.25},
        resolution=RESOLUTION, limits=replace(LIMITS, max_duration_s=2), should_cancel=cancel.is_set)
    assert result["timeout"] and not result["completed"] and len(calls) == 2
    assert source.agent_observation().last_action_feedback.execution_status == "timeout"


def test_fine_profile_keeps_arm_8cm_and_trunk_30cm_with_other_caps_unchanged():
    from pathlib import Path
    from core.action_compiler import action_bounds
    root = __import__('wia.resources', fromlist=['ROOT']).ROOT
    profile = json.loads((root / "configs/esi_primitives_fine.json").read_text())
    limits = MacroLimits(**json.loads((root / "configs/esi_parameterized_fine.json").read_text()))
    resolution = replace(RESOLUTION, arm_translation_m=profile["arm"]["translation_m"],
                         trunk_translation_m=profile["trunk"]["translation_m"])
    bounds = action_bounds(AVAILABLE, resolution, limits)
    for action in ("MV_FWD", "MV_DOWN"):
        assert bounds[action]["primitive_step"] == .005 and bounds[action]["maximum"] == .08
        assert bounds[action]["minimum"] == .0025 and bounds[action]["max_primitives"] == 16
        assert bounds[action]["max_duration_s"] == 180
        assert compile_action({"action": action, "distance_m": .08}, CAPABILITIES, resolution, limits).repeat_count == 16
        assert compile_action({"action": action, "distance_m": .0025}, CAPABILITIES, resolution, limits).effective_magnitude == .005
        with pytest.raises(ValueError, match="maximum"):
            compile_action({"action": action, "distance_m": .085}, CAPABILITIES, resolution, limits)
    for action in ("TRUNK_UP", "TRUNK_DOWN"):
        assert bounds[action]["maximum"] == .30
        assert compile_action({"action": action, "distance_m": .30}, CAPABILITIES, resolution, limits).repeat_count == 60
        with pytest.raises(ValueError, match="maximum"):
            compile_action({"action": action, "distance_m": .305}, CAPABILITIES, resolution, limits)
    assert bounds["BASE_FWD"]["maximum"] == .4 and bounds["BASE_FWD"]["max_primitives"] == 8
    assert bounds["ROTATE_CW"]["maximum"] == 40 and bounds["ROTATE_CW"]["max_duration_s"] == 90
    assert limits.request_timeout_s == 210


def test_macro_uses_per_action_duration_and_reports_remaining_count(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("core.action_interface.time.monotonic", lambda: clock[0])
    limits = replace(LIMITS, max_arm_primitives=16, max_arm_duration_s=3, max_duration_s=2)
    resolution = replace(RESOLUTION, arm_translation_m=.005)
    executor, _, _, calls, _, cancel = make_executor(clock=clock)
    result = executor.execute_macro({"action": "MV_UP", "distance_m": .08},
        resolution=resolution, limits=limits, should_cancel=cancel.is_set)
    assert result["timeout"] and len(calls) == 3 and result["execution"]["remaining_count"] == 13


def test_fixed_action_still_has_one_original_execution_and_history_shape():
    executor, source, _, calls, _, _ = make_executor()
    executor.execute("BASE_RIGHT", should_cancel=lambda: False)
    assert calls == ["BASE_RIGHT"] and source.action_step == 1
    assert source.agent_observation().metadata()["action_history"] == [
        {"action": "BASE_RIGHT", "result_summary": "completed"}]


def test_parameterized_prompt_schema_and_fixed_prompt_remain_separate():
    executor, source, _, _, _, _ = make_executor()
    context = {"parameterized_actions": {"resolution": RESOLUTION.to_dict(), "limits": LIMITS.to_dict()}}
    fixed = DecisionInput(source.agent_observation(), AVAILABLE, action_context=context)
    metric = replace(fixed, action_mode="parameterized")
    assert "No parameters" in fixed.prompt() and "distance_m" not in fixed.schema["properties"]
    assert "FULL command" in metric.prompt() and "half up" in metric.prompt()
    import jsonschema
    for command in ({"action": "BASE_RIGHT", "distance_m": 0.1}, {"action": "DONE"}):
        jsonschema.validate(command, metric.schema)
        jsonschema.validate({"command": command}, metric.structured_schema)
    assert set(metric.structured_schema["properties"]) == {"command"}
    assert "anyOf" not in metric.structured_schema
    for command in ({"action": "BASE_RIGHT"}, {"action": "DONE", "distance_m": 0.1}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(command, metric.schema)


def test_codex_parameterized_structured_envelope_returns_one_flat_command(tmp_path):
    from core.agent.backends import CodexAgent
    binary = tmp_path / "fake-codex"
    binary.write_text("#!/usr/bin/env python3\n" + '''import json, sys
from pathlib import Path
args = sys.argv[1:]
schema = json.loads(Path(args[args.index('--output-schema')+1]).read_text())
assert 'anyOf' not in schema and schema['required'] == ['command']
assert set(schema['properties']) == {'command'}
branches = schema['properties']['command']['anyOf']
assert any('distance_m' in b['required'] for b in branches)
assert 'wrap the single command' in sys.stdin.read()
Path(args[args.index('--output-last-message')+1]).write_text('{"command":{"action":"BASE_RIGHT","distance_m":0.10}}')
print(json.dumps({'type':'item.completed','item':{'type':'reasoning','text':'PRIVATE_THOUGHT'}}))
''')
    binary.chmod(0o755)
    _, source, _, _, _, _ = make_executor()
    request = DecisionInput(source.agent_observation(), AVAILABLE, action_mode="parameterized",
        action_context={"parameterized_actions": {"resolution": RESOLUTION.to_dict(), "limits": LIMITS.to_dict()}})
    reply = asyncio.run(CodexAgent(executable=str(binary)).decide(request))
    assert ParameterizedAction.parse(reply.raw_text, AVAILABLE).to_dict() == {"action": "BASE_RIGHT", "distance_m": 0.1}
    assert json.loads(reply.metadata["structured_output"]["raw_output"]) == {
        "command": {"action": "BASE_RIGHT", "distance_m": 0.1}}
    assert "PRIVATE_THOUGHT" not in json.dumps(reply.metadata)


@pytest.mark.parametrize("structured", [True, False])
def test_local_http_parameterized_schema_and_reply_protocol(structured):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from core.agent.backends import LocalHTTPAgent
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            reply = {"command": {"action": "DONE"}} if structured else {"action": "DONE"}
            body = json.dumps({"choices": [{"message": {"content": json.dumps(reply)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _, source, _, _, _, _ = make_executor()
        request = DecisionInput(source.agent_observation(), AVAILABLE, action_mode="parameterized",
            action_context={"parameterized_actions": {"resolution": RESOLUTION.to_dict(), "limits": LIMITS.to_dict()}})
        agent = LocalHTTPAgent(base_url=f"http://127.0.0.1:{server.server_port}/v1", model="local-test", structured_output=structured)
        reply = asyncio.run(agent.decide(request))
        assert ParameterizedAction.parse(reply.raw_text, AVAILABLE).action == "DONE"
        assert ("response_format" in received[0]) == structured
        if structured:
            assert received[0]["response_format"]["json_schema"]["schema"] == request.structured_schema
            assert reply.metadata["structured_output"]["raw_output"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_mcp_macro_owner_transaction_and_three_decision_budget(control_http, monkeypatch, tmp_path):
    from mcp import Client
    from wia.runtime.agent.mcp_environment import MCPRobotEnvironment
    from wia.runtime.web.mcp_server import create_server
    from harness_integration.test_agent_control import pump

    backend, _, url, calls, _ = control_http
    backend.executor.capabilities = CAPABILITIES
    backend.executor.primitive_resolution, backend.executor.macro_limits = RESOLUTION, LIMITS
    boundaries = []
    original_callback = backend.executor.on_result
    backend.executor.on_result = lambda result: (boundaries.append(result), original_callback(result))
    monkeypatch.setattr("wia.runtime.agent.mcp_environment.Client", lambda *args, **kwargs: Client(create_server(url)))
    finished = asyncio.Event()
    agent = Agent('{"action":"BASE_RIGHT"}', '{"action":"BASE_RIGHT","distance_m":0.10}',
                  '{"action":"BASE_TURN_LEFT","angle_deg":30}', '{"action":"DONE"}')

    async def run():
        task = asyncio.create_task(pump(backend, finished))
        try:
            async with MCPRobotEnvironment(url, run_id="metric-test", task_text="Approach the table.",
                                           action_mode="parameterized") as env:
                assert requests.post(url+"/api/move", json={"token": "BASE_FWD"}).status_code == 409
                return await run_episode(env, agent, tmp_path / "episode", max_steps=3, action_mode="parameterized")
        finally:
            finished.set()
            await task

    summary = asyncio.run(run())
    assert summary["termination_reason"] == "model_done"
    assert (summary["agent_decisions"], summary["model_calls"], summary["macro_actions"], summary["primitive_actions"]) == (3, 4, 2, 8)
    assert calls == ["BASE_RIGHT"]*2 + ["BASE_TURN_LEFT"]*6
    assert len(boundaries) == backend.steps == backend.observation_source.action_step == 2
    assert len(backend.observation_source.history) == 2
    assert backend.primitive_actions == 8 and backend.control_owner == "human"
    assert agent.inputs[0].observation is agent.inputs[1].observation
    assert agent.inputs[-1].observation.action_history[0].distance_m == 0.10
    inputs = [json.loads(line) for line in (tmp_path / "episode/model_inputs.jsonl").read_text().splitlines()]
    receipts = [row for row in inputs if row["event"] == "mcp_response"
                and row["tool"] == "robot_execute_parameterized"]
    assert len(receipts) == 2
    assert receipts[0]["arguments"]["command"] == {"action": "BASE_RIGHT", "distance_m": 0.10}
    assert receipts[0]["arguments"]["control_token"] == "[redacted]"
    outputs = [json.loads(line) for line in (tmp_path / "episode/model_outputs.jsonl").read_text().splitlines()]
    assert len([row for row in outputs if row["event"] == "model_output"]) == 4


def test_server_rejects_untrusted_metric_payloads_before_motion(control_http):
    from harness_integration.test_agent_control import claim
    backend, _, url, calls, _ = control_http
    backend.executor.primitive_resolution, backend.executor.macro_limits = RESOLUTION, LIMITS
    token = claim(backend, url)  # Fixed mode cannot use the metric endpoint.
    assert requests.post(url+"/api/agent/command", json={"command": {"action": "BASE_FWD", "distance_m": 0.1},
        "control_token": token}).status_code == 400
    backend.action_mode = "parameterized"
    for command in ({"action": "BASE_FWD"}, {"action": "BASE_FWD", "distance_m": True},
                    {"action": "BASE_FWD", "distance_m": 2}, {"action": "DONE"},
                    {"action": "ROTATE_CW", "angle_deg": 5}):
        assert requests.post(url+"/api/agent/command", json={"command": command, "control_token": token}).status_code == 400
    duplicate = '{"command":{"action":"BASE_FWD","distance_m":0.1,"distance_m":0.2},"control_token":' + json.dumps(token) + '}'
    assert requests.post(url+"/api/agent/command", data=duplicate).status_code == 400
    assert not calls and backend.commands.empty()
    backend.cancel_active_action()


def test_human_cancel_during_macro_stops_children_and_retains_lock_until_stop(control_http):
    from harness_integration.test_agent_control import claim
    backend, _, url, calls, _ = control_http
    backend.executor.primitive_resolution, backend.executor.macro_limits = RESOLUTION, LIMITS
    token = claim(backend, url)
    backend.action_mode = "parameterized"

    def execute(action, *, should_cancel):
        calls.append(action)
        assert backend.busy and backend.control_owner == "agent"
        if len(calls) == 2:
            assert requests.post(url+"/api/cancel", json={}).json()["ok"]
            assert backend.control_owner == "agent" and backend.busy
            assert requests.post(url+"/api/move", json={"token": "BASE_FWD"}).status_code == 409
        return action_result(action, completed=not should_cancel(), timeout=False,
            requested={}, actual={}, message="stopped", reason="cancelled" if should_cancel() else "completed",
            cancelled=should_cancel(), stop_steps=3, control_steps=10)

    backend.executor._execute_action = execute
    responses = []
    thread = threading.Thread(target=lambda: responses.append(requests.post(url+"/api/agent/command",
        json={"command": {"action": "BASE_RIGHT", "distance_m": 0.25}, "control_token": token}, timeout=3)))
    thread.start()
    with backend.frame_cond:
        assert backend.frame_cond.wait_for(lambda: backend.busy, timeout=1)
    backend.process_next()
    thread.join(timeout=3)
    result = responses[0].json()
    assert calls == ["BASE_RIGHT"]*2 and len(result["results"]) == 1
    assert result["macro"]["execution"]["executed_count"] == 2
    assert result["results"][0]["execution_status"] == "cancelled"
    assert backend.control_owner == "human" and not backend.busy
    assert backend.steps == backend.observation_source.action_step == 1
    assert requests.post(url+"/api/agent/command", json={"command": {"action": "BASE_FWD", "distance_m": 0.1},
        "control_token": token}).status_code == 403


def test_timed_out_pending_macro_never_moves_later(control_http):
    from harness_integration.test_agent_control import claim
    backend, _, url, calls, _ = control_http
    backend.executor.primitive_resolution, backend.executor.macro_limits = RESOLUTION, LIMITS
    token = claim(backend, url)
    backend.action_mode = "parameterized"
    responses = []
    thread = threading.Thread(target=lambda: responses.append(backend.submit_parameterized(
        {"action": "BASE_RIGHT", "distance_m": 0.25}, token, timeout=0.01)))
    thread.start()
    thread.join(timeout=1)
    assert responses[0]["status"] == 504 and backend.busy and backend.active_cancel.is_set()
    assert backend.process_next() and not calls
    assert backend.observation_source.action_step == 1
    assert backend.observation_source.last_action_feedback.execution_status == "cancelled"
    backend.cancel_active_action()
