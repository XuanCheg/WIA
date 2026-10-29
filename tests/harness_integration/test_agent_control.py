"""Server-enforced ownership through human HTTP, Agent MCP and the owner queue."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
import requests

pytest.importorskip("mcp")
from mcp import Client

from core.action_interface import ActionExecutor, BackendCapabilities, action_result
from wia.runtime.agent.mcp_environment import MCPRobotEnvironment
from wia.runtime.agent.zero_shot import run_episode
from core.agent.decision import ModelError
from core.viewer_config import ViewerConfig
from wia.runtime.web.mcp_server import create_server
from wia.runtime.web.web_teleop.esi_control import ESIControlBackend
from wia.runtime.web.web_teleop.esi_viewer import ESIViewerBackend
from gumi.web_teleop.server import serve
from harness_integration.test_agent_observation import source_and_rig
from harness_integration.test_esi_preview import METADATA
from harness_integration.test_zero_shot_agent import Agent
from harness_integration.test_task_edit2_initial_overview import install_overview


@pytest.fixture
def control_http(tmp_path):
    source, rig = source_and_rig()
    install_overview(rig)
    capture_overview = rig.initial_overview_capture

    def episode_overview(**boundary):
        rig.initial_overview_identity['episode_id'] = rig.episode_id
        return capture_overview(**boundary)

    rig.initial_overview_capture = episode_overview
    calls = []

    def execute(action, *, should_cancel=None):
        calls.append(action)
        return action_result(action, completed=True, timeout=False, requested={}, actual={},
                             message="completed", reason="completed")

    executor = ActionExecutor(execute, capabilities=BackendCapabilities(base_translation=True), on_result=source.after_action)
    backend = ESIControlBackend(dict(METADATA), executor, observation_source=source)
    viewer = ESIViewerBackend(backend, tmp_path / "recordings", ViewerConfig())

    def publish(snapshot):
        backend.publish(snapshot)
        viewer.publish(snapshot)

    source.on_capture = publish
    source.capture(action_boundary=True)
    server = serve(viewer, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield backend, viewer, f"http://127.0.0.1:{server.server_port}", calls, rig
    finally:
        backend.shutdown()
        viewer.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


async def pump(backend, completed):
    # Tests retain exactly the production owner rule: main thread does execution/capture.
    while not completed.is_set():
        backend.process_next()
        await asyncio.sleep(0.005)


def claim(backend, url, *, action_mode="fixed"):
    responses = []
    thread = threading.Thread(target=lambda: responses.append(requests.post(url + "/api/control/claim",
        json={"run_id": "test-run", "task_text": "Approach the table.", "action_mode": action_mode}, timeout=3)))
    thread.start()
    with backend.frame_cond:
        assert backend.frame_cond.wait_for(lambda: backend.busy, timeout=1)
    assert backend.process_next()
    thread.join(timeout=3)
    assert responses[0].status_code == 200
    return responses[0].json()["control_token"]


def test_human_movement_is_locked_while_agent_thinks_and_cancel_invalidates_token(control_http):
    backend, _, url, calls, _ = control_http
    token = claim(backend, url)
    state = requests.get(url + "/api/state").json()
    assert state["control_owner"] == "agent" and not state["busy"]
    assert state["task_text"] == "Approach the table." and "control_token" not in json.dumps(state)
    assert requests.get(url + "/api/agent-observation").json()["task_text"] == state["task_text"]
    for path, body in (("move", {"token": "BASE_FWD"}), ("step", {"tokens": ["BASE_FWD"]}),
                       ("control/claim", {"run_id": "other", "task_text": "Other task"})):
        assert requests.post(url + "/api/" + path, json=body).status_code == 409
    for body in ({"action": "BASE_FWD", "control_token": None},
                 {"action": "BASE_FWD", "control_token": "wrong"},
                 {"action": ["BASE_FWD"], "control_token": token},
                 {"action": "BASE_FWD", "distance": 1, "control_token": token}):
        assert requests.post(url + "/api/agent/step", json=body).status_code in (400, 403, 405)
    assert not calls and backend.commands.empty()
    assert requests.post(url + "/api/cancel", json={}).json()["ok"]
    assert backend.control_owner == "human"
    assert requests.post(url + "/api/agent/step", json={"action": "BASE_FWD", "control_token": token}).status_code == 403
    assert not calls


@pytest.mark.parametrize("outputs, expected", [
    (['{"action":"BASE_FWD"}', '{"action":"DONE"}'], "model_done"),
    (['{"action":"INVALID"}'] * 3, "invalid_output"),
    ([RuntimeError("test unavailable")], "runtime_error"),
])
def test_mcp_runner_reuses_owner_executor_and_releases_on_all_termination_paths(control_http, monkeypatch, tmp_path, outputs, expected):
    backend, _, url, calls, _ = control_http
    # Use the actual SDK MCP protocol with its HTTP RobotBridge, without another listener.
    monkeypatch.setattr("wia.runtime.agent.mcp_environment.Client", lambda *args, **kwargs: Client(create_server(url)))
    completed = asyncio.Event()

    async def run():
        task = asyncio.create_task(pump(backend, completed))
        try:
            async with MCPRobotEnvironment(url, run_id="mcp-test", task_text="Approach the table.") as environment:
                summary = await run_episode(environment, Agent(*outputs), tmp_path / "episode")
                assert backend.control_owner == "agent" and not backend.busy
            return summary
        finally:
            completed.set()
            await task

    summary = asyncio.run(run())
    assert summary["termination_reason"] == expected
    assert calls == (["BASE_FWD"] if expected == "model_done" else [])
    assert backend.control_owner == "human" and not backend.busy
    inputs = [json.loads(line) for line in (tmp_path / "episode/model_inputs.jsonl").read_text().splitlines()]
    assert [row["event"] for row in inputs[:2]] == ["global_prompt", "task_prompt"]
    mcp_rows = [row for row in inputs if row["event"] == "mcp_response"]
    assert mcp_rows[0]["tool"] == "robot_get_context"
    assert mcp_rows[1]["tool"] == "robot_get_observation"
    assert all(isinstance(row["result"], dict) and row["passed_to_model"] is False for row in mcp_rows)
    if calls:
        before = json.loads((tmp_path / "episode/step_0001/before/observation.json").read_text())
        after = json.loads((tmp_path / "episode/step_0001/after/observation.json").read_text())
        assert after["action_step"] == before["action_step"] + 1
        assert after["action_history"][-1]["action"] == calls[0]
        receipt = next(row for row in mcp_rows if row["tool"] == "robot_execute_actions")
        assert receipt["arguments"]["control_token"] == "[redacted]"
        assert receipt["step"] == receipt["attempt"] == 1
        runtime = receipt["result"]["structuredContent"]["timing"]
        assert runtime["actions"][0]["boundary_capture_wall_time_s"] >= 0
        steps = [json.loads(line) for line in (tmp_path / "episode/steps.jsonl").read_text().splitlines()]
        assert steps[0]["action_result"]["timing"] == runtime


def test_human_cancel_stops_queued_agent_action_without_moving_or_resetting_scene(control_http):
    backend, _, url, calls, rig = control_http
    token = claim(backend, url)
    cycle = rig.cycle
    responses = []
    thread = threading.Thread(target=lambda: responses.append(requests.post(url + "/api/agent/step",
        json={"action": "BASE_FWD", "control_token": token}, timeout=3)))
    thread.start()
    with backend.frame_cond:
        assert backend.frame_cond.wait_for(lambda: backend.busy, timeout=1)
    assert requests.post(url + "/api/control/release", json={"control_token": token}).status_code == 409
    assert requests.post(url + "/api/control/cancel", json={"control_token": token}).json()["ok"]
    backend.process_next()
    thread.join(timeout=3)
    assert not calls and rig.cycle == cycle and backend.control_owner == "human"
    assert responses[0].status_code == 504


def test_declarative_resume_retains_action_boundary_and_does_not_replay(control_http, monkeypatch, tmp_path):
    backend, _, url, calls, _ = control_http
    text = backend.observation_source.task_text
    contract = {"episode_id": "retained-task", "action_mode": "fixed",
                "world_advances_during_inference": True}
    backend.metadata["episode_contract"] = contract
    monkeypatch.setattr("wia.runtime.agent.mcp_environment.Client", lambda *args, **kwargs: Client(create_server(url)))
    completed = asyncio.Event()

    async def run():
        worker = asyncio.create_task(pump(backend, completed))
        try:
            async with MCPRobotEnvironment(url, run_id="retained-task", task_text=text,
                                           episode_contract=contract) as environment:
                first = await run_episode(environment, Agent('{"action":"BASE_FWD"}', ModelError("serverOverloaded")),
                                          tmp_path / "first", max_steps=2)
            assert backend.steps == 1 and calls == ["BASE_FWD"]
            history = list(backend.history)
            progress = SimpleNamespace(episode=1, grasp_seen=True, data=None)

            def record(observation, *, action):
                progress.data = {"schema": "private_grasp_lift_v1", "task_success": False,
                    "task_text": text, "snapshot_id": observation.snapshot_id,
                    "action_step": observation.action_step, "episode": progress.episode}

            progress.record = record
            monkeypatch.setattr(backend.observation_source.session, "task_evaluator", progress, raising=False)
            def forbidden_reset(*args, **kwargs):
                pytest.fail("Resume must not initialize another episode")
            monkeypatch.setattr(backend.observation_source, "begin_episode", forbidden_reset)
            async with MCPRobotEnvironment(url, run_id="retained-task", task_text=text,
                                           episode_contract=contract, resume_action_step=1) as environment:
                assert backend.steps == 1 and backend.history == history
                assert environment.initialization["interruption_resume"]["new_lease_action_step"] == 1
                def success():
                    from wia.native.esi_category_evaluation import validate_episode_evaluation
                    loaded = SimpleNamespace(definition=SimpleNamespace(kind="rigid_robot_task"))
                    return validate_episode_evaluation(progress.data, environment.last_observation, text, loaded)

                result = await run_episode(environment, Agent('{"action":"BASE_FWD"}', '{"action":"DONE"}'),
                    tmp_path / "resumed", max_steps=4, resume_summary=first,
                    expected_action_step=1, budget_extension=True, success_provider=success)
                assert progress.episode == 1 and progress.grasp_seen
            assert backend.steps == 2 and calls == ["BASE_FWD", "BASE_FWD"]
            assert result["agent_decisions"] == 3 and result["termination_reason"] == "model_done"
            assert result["simulator_action_step_offset"] == 0
        finally:
            completed.set()
            await worker

    asyncio.run(run())
    assert backend.control_owner == "human"


def test_declarative_resume_rejects_wrong_boundary_and_identity(control_http):
    backend, _, url, calls, _ = control_http
    text = backend.observation_source.task_text
    backend.metadata["episode_contract"] = {"episode_id": "retained-task", "action_mode": "fixed",
                                            "world_advances_during_inference": True}
    body = {"run_id": "retained-task", "task_text": text, "resume_action_step": 0}
    assert requests.post(url + "/api/control/claim", json=body).status_code == 409
    backend._declarative_episode_claimed = True
    backend.steps = 58
    for changes, status in [({}, 409), ({"resume_action_step": True}, 400),
                            ({"resume_action_step": 58, "run_id": "other"}, 409),
                            ({"resume_action_step": 58, "task_text": "Different task"}, 409)]:
        assert requests.post(url + "/api/control/claim", json=dict(body, **changes)).status_code == status
    assert backend.control_owner == "human" and backend.steps == 58 and not calls


def test_declarative_resume_rejects_old_backend_before_claim(control_http, monkeypatch):
    backend, _, url, calls, _ = control_http
    contract = {"episode_id": "retained-task", "action_mode": "fixed",
                "world_advances_during_inference": True}
    backend.metadata["episode_contract"] = contract
    original_state = backend.state
    def old_state():
        state = original_state()
        state.pop("declarative_resume_available")
        return state
    monkeypatch.setattr(backend, "state", old_state)
    async def run():
        with pytest.raises(RuntimeError, match="does not support declarative resume"):
            async with MCPRobotEnvironment(url, run_id="retained-task",
                    task_text=backend.observation_source.task_text, episode_contract=contract,
                    resume_action_step=0):
                pytest.fail("An old backend must not acquire a new lease")
    asyncio.run(run())
    assert backend.control_owner == "human" and backend.steps == 0
    assert backend.commands.empty() and not calls


@pytest.mark.parametrize("parameterized", [False, True])
def test_http_command_timing_includes_final_capture_and_excludes_private_fields(control_http, parameterized):
    from core.action_compiler import MacroLimits, ParameterizedAction, PrimitiveResolution
    from core.observations import validate_macro_receipt

    backend, _, url, calls, _ = control_http
    executor = backend.executor
    if parameterized:
        executor.primitive_resolution = PrimitiveResolution(base_translation_m=.05)
        executor.macro_limits = MacroLimits()
    token = claim(backend, url, action_mode="parameterized" if parameterized else "fixed")
    capture = executor.on_result
    def captured_result(result):
        capture(result)
        # This field becomes available only after final capture returns.
        result.setdefault("timing", {}).update(boundary_capture_wall_time_s=.125,
            boundary_render_count=2, private_object_pose="PRIVATE", unsupported_elapsed_s=5)
    executor.on_result = captured_result
    command = {"action": "BASE_FWD", "distance_m": .05}
    route = "/api/agent/command" if parameterized else "/api/agent/step"
    body = {"command": command} if parameterized else {"action": "BASE_FWD"}
    responses = []
    thread = threading.Thread(target=lambda: responses.append(requests.post(url + route,
        json=dict(body, control_token=token), timeout=3)))
    thread.start()
    with backend.frame_cond:
        assert backend.frame_cond.wait_for(lambda: backend.busy, timeout=1)
    backend.process_next()
    thread.join(timeout=3)
    assert responses[0].status_code == 200 and calls == ["BASE_FWD"]
    response = responses[0].json()
    timing = response["timing"]
    assert timing["command_wall_time_s"] >= 0
    assert timing["actions"][0]["boundary_capture_wall_time_s"] == .125
    assert timing["actions"][0]["boundary_render_count"] == 2
    assert "private_object_pose" not in json.dumps(response) and "unsupported_elapsed_s" not in json.dumps(response)
    assert backend.agent_snapshot.action_step == 1
    if parameterized:
        # Timing is response metadata; the established strict macro receipt is unchanged.
        validate_macro_receipt(response["macro"], command=ParameterizedAction(**command),
            capabilities=executor.capabilities, resolution=executor.primitive_resolution,
            limits=executor.macro_limits, feedback=backend.observation_source.last_action_feedback)


def test_debug_lease_release_pauses_world_and_preserves_action_boundary(control_http):
    backend, viewer, url, calls, rig = control_http
    backend.metadata['debug_retain'] = True
    token = claim(backend, url)
    boundary = backend.steps
    released = requests.post(url + '/api/control/release', json={'control_token':token}, timeout=3)
    assert released.json()['ok']
    assert backend.control_owner == 'human'
    assert not backend.should_advance_world()
    assert backend.steps == boundary
    backend.owner_phase('human_idle')
    assert not backend.should_advance_world()
    assert backend.state()['physics_paused'] is True


def test_human_metric_http_right_arm_and_ownership(control_http):
    from core.action_compiler import MacroLimits, PrimitiveResolution
    backend, _, url, calls, _ = control_http
    backend.executor.capabilities = BackendCapabilities(base_translation=True, right_ee_translation=True)
    backend.executor.primitive_resolution = PrimitiveResolution(base_translation_m=.05, arm_translation_m=.01)
    backend.executor.macro_limits = MacroLimits()
    command = {"action": "RIGHT_MV_UP", "distance_m": .02}
    responses = []
    thread = threading.Thread(target=lambda: responses.append(requests.post(url + '/api/command', json={"command": command}, timeout=3)))
    thread.start()
    with backend.frame_cond:
        assert backend.frame_cond.wait_for(lambda: backend.busy, timeout=1)
    backend.process_next()
    thread.join(timeout=3)
    assert responses[0].status_code == 200, responses[0].text
    assert calls == ['RIGHT_MV_UP', 'RIGHT_MV_UP']
    assert responses[0].json()['state']['human_parameterized_commands']
    before = list(calls)
    assert requests.post(url + '/api/command', json={"command": {"action": "RIGHT_MV_UP", "distance_m": 99}}, timeout=3).status_code == 400
    claim(backend, url, action_mode='parameterized')
    assert requests.post(url + '/api/command', json={"command": command}, timeout=3).status_code == 409
    assert calls == before
