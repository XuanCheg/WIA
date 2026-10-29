"""Model-visible feedback across the real MCP protocol, HTTP and owner-thread queue."""
import asyncio
import base64
import json
import threading
from dataclasses import replace

import numpy as np
import pytest
import requests

pytest.importorskip("mcp")
from mcp import Client

from core.observations import AgentActionFeedback
from core.action_interface import ActionExecutor
from wia.runtime.sim.esi_actions import create_r1_executor, r1_action_result
from wia.runtime.web.mcp_server import RobotBridge, create_server
from wia.runtime.web.web_teleop.esi_control import ESIControlBackend
from gumi.web_teleop.server import serve
from harness_integration.test_esi_arm import make_arm_controller
from harness_integration.test_esi_preview import METADATA


def observed_result(token, *, reason="completed"):
    completed = reason == "completed"
    return r1_action_result({
        "action": token, "motion": "arm_translation", "frame": "robot_base",
        "completed": completed, "reason": reason, "timeout": False, "cancelled": False,
        "requested_m": 0.02, "actual_m": 0.02 if completed else 0.00001,
        "actual_delta_m": [0, 0, -0.02 if completed else -0.00001], "stop_steps": 3,
        "recovery_steps": 0 if completed else 2, "recovery_completed": None if completed else True,
        "stop_stability_verified": True,
        "ik_diagnostics": {"joint_hold": not completed, "min_jacobian_singular_value": 0.001},
        "ee_pose_after": {"base_position": [0.5, 0.2, 0.9], "world_position": [999, 888, 777]},
        "scene_object_poses": "must not enter the model",
    })


@pytest.fixture
def robot_http():
    metadata = dict(METADATA, image_shape=[8, 8, 3],
                    arm_control={"working_arm": "left", "config": {"translation_m": 0.02}},
                    base_control={"config": {"translation_m": 0.25, "rotation_deg": 15}})
    controller = make_arm_controller()[0]
    backend = ESIControlBackend(metadata, create_r1_executor(controller))
    state = replace(controller.controller_state(),
                    ee_pose={"base_position": [0.5, 0.2, 0.9], "base_quaternion_xyzw": [0, 0, 0, 1]},
                    proprioception={"frame": "robot_base", "source": "live",
                                    "eef_pos": [0.5, 0.2, 0.9], "gripper_width_m": 0.06,
                                    "world_position": [999, 888, 777], "table_height_m": None})
    backend.publish({"head_rgb": np.zeros((8, 8, 3), np.uint8),
                     "wrist_rgb": np.full((8, 8, 3), 50, np.uint8),
                     "scene_rgb": np.full((8, 8, 3), 100, np.uint8)}, controller_state=state)
    http = serve(backend, host="127.0.0.1", port=0)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    try:
        yield backend, f"http://127.0.0.1:{http.server_port}"
    finally:
        backend.shutdown()
        http.shutdown()
        http.server_close()
        worker.join(timeout=2)


async def queued_call(client, backend, tokens, execute):
    reader = getattr(backend.executor, "controller_state_provider", None)
    backend.executor = ActionExecutor(execute, capabilities=backend.capabilities)
    if reader is not None:
        backend.executor.controller_state_provider = reader
    pending = asyncio.create_task(client.call_tool("robot_execute_actions", {"tokens": tokens}))
    for _ in range(300):
        if backend.busy:
            break
        if pending.done():
            return await pending
        await asyncio.sleep(0.01)
    assert backend.busy
    assert backend.process_next()
    return await asyncio.wait_for(pending, timeout=5)


@pytest.mark.parametrize("reason", ["arm_joint_limit", "ik_joint_jump"])
def test_limit_reaches_model_as_tool_error_and_skips_remaining_batch(robot_http, reason):
    backend, url = robot_http
    calls, owners = [], []

    def execute(token, **kwargs):
        calls.append(token)
        owners.append(threading.get_ident())
        return observed_result(token, reason=reason if token == "MV_DOWN" else "completed")

    async def run():
        async with Client(create_server(url)) as client:
            result = await queued_call(client, backend, ["MV_FWD", "MV_DOWN", "MV_UP"], execute)
            assert result.is_error
            data = result.structured_content
            assert not data["ok"] and data["reason"] == reason and data["http_status"] == 409
            assert data["unexecuted_actions"] == ["MV_UP"] and data["motion_stopped"] is True
            assert data["results"][-1]["motion_stopped"] is True
            assert data["results"][-1]["estimated_motion"]["delta_m"] < 0.02 * 0.1
            assert data == json.loads(result.content[0].text)
            assert "world_position" not in result.content[0].text
            assert "scene_object_poses" not in result.content[0].text
            assert "min_jacobian" not in result.content[0].text
            state = (await client.call_tool("robot_get_state", {})).structured_content["state"]
            assert state["history"] == calls and state["last_execution"]["observable_error"] == reason
            recovered = await queued_call(client, backend, ["MV_UP"],
                                          lambda token, **kw: observed_result(token))
            assert not recovered.is_error and recovered.structured_content["completed"]

    asyncio.run(run())
    assert calls == ["MV_FWD", "MV_DOWN"] and set(owners) == {threading.get_ident()}


def test_original_context_proprioception_and_rgb_are_model_visible_without_ground_truth(robot_http):
    backend, url = robot_http

    async def run():
        async with Client(create_server(url)) as client:
            tools = await client.list_tools()
            assert {t.name for t in tools.tools} == {
                "robot_get_context", "robot_get_state", "robot_get_camera", "robot_execute_actions", "robot_get_observation",
                "robot_execute_parameterized"}
            context = (await client.call_tool("robot_get_context", {})).structured_content["context"]
            assert context["steps"] == {"arm_translation_m": 0.02, "base_translation_m": 0.25,
                                         "base_rotation_deg": 15}
            assert context["task_text"] and context["camera_shape"] == [8, 8, 3]
            assert not context["task_evaluation_available"]
            state = (await client.call_tool("robot_get_state", {})).structured_content["state"]
            assert state["robot_state"]["ee_pose"]["frame"] == "robot_base"
            assert state["robot_state"]["gripper_width_m"] is None  # Mock has no verified finger layout.
            assert "world_position" not in state["robot_state"]
            assert state["task_progress"] is None and "hardware_errors" not in state
            camera = await client.call_tool("robot_get_camera", {"view": "head"})
            assert not camera.is_error and camera.content[1].type == "image"
            assert camera.structured_content["frame_seq"] == backend.frame_seq
            assert base64.b64decode(camera.content[1].data).startswith(b"\xff\xd8")
            assert backend.steps == 0

    asyncio.run(run())


@pytest.mark.parametrize("tokens", [["MV_UP", "GRASP"], ["RT_YAW_CW"], ["MV_DOWN(0.5)"], [], ["MV_UP"] * 65])
def test_unsupported_or_parameterized_actions_never_reach_robot_queue(robot_http, tokens):
    backend, url = robot_http

    async def run():
        async with Client(create_server(url)) as client:
            result = await client.call_tool("robot_execute_actions", {"tokens": tokens})
            assert result.is_error and backend.commands.empty() and not backend.busy

    asyncio.run(run())


def test_read_only_capability_is_enforced_before_motion(robot_http):
    backend, url = robot_http
    backend.read_only = True

    async def run():
        async with Client(create_server(url)) as client:
            result = await client.call_tool("robot_execute_actions", {"tokens": ["MV_UP"]})
            assert result.is_error and result.structured_content["reason"] == "action_disabled"
            assert backend.commands.empty()

    asyncio.run(run())


def test_transport_timeout_reports_unknown_outcome_and_never_retries(monkeypatch):
    requests_made = []

    async def request(self, method, path, **kwargs):
        requests_made.append(method)
        if method == "POST":
            raise requests.ReadTimeout("test timeout")
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"read_only": False, "action_groups": {"arm": True},
                                       "step_tokens": ["MV_DOWN"]}).encode()
        return response

    monkeypatch.setattr(RobotBridge, "request", request)

    async def run():
        async with Client(create_server()) as client:
            result = await client.call_tool("robot_execute_actions", {"tokens": ["MV_DOWN"]})
            data = result.structured_content
            assert result.is_error and not data["outcome_known"] and data["motion_stopped"] is None
            assert data["reason"] == "execution_feedback_unavailable"

    asyncio.run(run())
    assert requests_made == ["GET", "POST"]


def test_semantic_failure_is_not_mistaken_for_http_success():
    result = RobotBridge("http://localhost:8615")._execution_result(
        ["MV_DOWN", "MV_UP"], 200, {"ok": True, "results": [AgentActionFeedback.from_result(observed_result("MV_DOWN", reason="ik_joint_jump")).to_dict()]})
    assert result.is_error and result.structured_content["reason"] == "ik_joint_jump"
    assert result.structured_content["unexecuted_actions"] == ["MV_UP"]


def test_unconfirmed_recovery_does_not_claim_motion_has_stopped():
    raw = observed_result("MV_DOWN", reason="ik_joint_jump")
    raw.pop("recovery_completed")
    raw.pop("stop_stability_verified")
    result = RobotBridge("http://localhost:8615")._execution_result(
        ["MV_DOWN"], 409, {"ok": False, "results": [AgentActionFeedback.from_result(raw).to_dict()], "state": {"busy": False}})
    assert result.structured_content["motion_stopped"] is None
    assert result.structured_content["results"][0]["motion_stopped"] is None


def test_mcp_preserves_capture_timing_metadata_without_private_fields():
    payload = {"ok": True, "results": [AgentActionFeedback.from_result(observed_result("MV_UP")).to_dict()],
        "timing": {"command_wall_time_s": .9, "private_score": 1,
            "actions": [{"boundary_capture_wall_time_s": .3, "boundary_render_count": 2,
                         "unrendered_sim_step_wall_time": .5, "private_object_pose": [1, 2, 3]}]}}
    result = RobotBridge("http://localhost:8615")._execution_result(["MV_UP"], 200, payload)
    assert result.structured_content["timing"] == {"command_wall_time_s": .9,
        "actions": [{"boundary_capture_wall_time_s": .3, "boundary_render_count": 2,
                     "unrendered_sim_step_wall_time": .5}]}
    assert "private_" not in result.content[0].text
