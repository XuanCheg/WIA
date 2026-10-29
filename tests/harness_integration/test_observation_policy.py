"""Real-robot policy: truth changes cannot alter otherwise identical Agent input."""
import asyncio
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from core.observation_config import ObservationConfig
from core.observations import AgentActionFeedback, AgentObservation, AgentRobotState, RobotPose, RobotSensorFeedback
from core.sensor_observations import SensorReadings
from wia.runtime.sim.esi_actions import create_r1_executor
from wia.runtime.web.web_teleop.esi_control import ESIControlBackend
from gumi.web_teleop.server import serve
from harness_integration.test_agent_observation import source_and_rig
from harness_integration.test_esi_gripper import make_manipulation_controller
from harness_integration.test_esi_preview import METADATA


def test_attachment_changes_do_not_change_robot_state_feedback_or_history():
    controller, robot, _ = make_manipulation_controller()
    robot.joints[controller.gripper_dofs] = 0.02
    unheld = controller.controller_state()
    raw = {"action": "GRASP", "requested": {"state": "closed"}, "actual": {"holding": False},
           "completed": False, "success": False, "reason": "empty_grasp", "timeout": False, "stop_steps": 3,
           "debug_gripper_after": {"object_name": None}}
    before = AgentActionFeedback.from_result(raw, state=unheld).to_dict()
    robot._ag_obj_in_hand["left"] = SimpleNamespace(name="PRIVATE_HIDDEN_TARGET")
    robot._ag_obj_constraints["left"] = object()
    held = controller.controller_state()
    raw.update(actual={"holding": True}, completed=True, success=True, reason="already_holding",
               debug_gripper_after={"object_name": "PRIVATE_HIDDEN_TARGET"})
    after = AgentActionFeedback.from_result(raw, state=held).to_dict()
    assert AgentRobotState.from_controller_state(unheld).to_dict() == AgentRobotState.from_controller_state(held).to_dict()
    assert before == after
    assert before["execution_status"] == "completed" and before["target_reached"] is False
    assert "holding" not in before["estimated_motion"]
    assert "PRIVATE" not in json.dumps(after)


def test_simulator_world_pose_and_velocity_cannot_fill_missing_odometry():
    state = make_manipulation_controller()[0].controller_state()
    changed = replace(state, base_pose={"position": [999, -888, 77], "quaternion_xyzw": [0, 0, 1, 0]},
                      base_velocity={"frame": "world", "linear_m_s": [123, 4, 5], "angular_rad_s": [0, 0, 6]})
    assert AgentRobotState.from_controller_state(state).to_dict() == AgentRobotState.from_controller_state(changed).to_dict()
    assert AgentRobotState.from_controller_state(changed).base_pose is None


def test_declared_hardware_adapter_is_connected_and_visibility_switches_apply():
    controller = make_manipulation_controller()[0]
    reading = RobotSensorFeedback(
        base_odometry=RobotPose([1, 2, 0], [0, 0, 0, 1], "odometry_local", "odometry"),
        base_linear_velocity_m_s=[0.1, 0, 0], holding_estimate=True, holding_source="gripper_contact_estimator",
        gripper_contact=[True, False], contact_source="gripper_contact_sensor",
        sensors=SensorReadings(lidar_points_m=np.array([[1, 0, 0]]), force_torque=np.zeros(6), tactile=np.ones(2)),
    )
    controller.session.robot_sensor_feedback_provider = lambda: reading
    source, _ = source_and_rig()
    controller.session.observation_source = source
    create_r1_executor(controller)
    observation = source.capture(action_boundary=True).observation
    assert observation.robot_state.base_pose == reading.base_odometry
    assert observation.robot_state.holding_estimate is True
    assert observation.robot_state.gripper_contact == (True, False)
    assert observation.robot_state.sources["holding_estimate"] == "gripper_contact_estimator"
    assert observation.sensors.lidar_points_m.shape == (1, 3)
    assert observation.sensors.force_torque.shape == (6,)
    source.config = replace(source.config, base_odometry=False, holding_estimate=False, gripper_contact=False,
                            lidar=False, force_torque=False, tactile=False)
    observation = source.capture(action_boundary=True).observation
    assert observation.robot_state.base_pose is None and observation.robot_state.base_linear_velocity_m_s is None
    assert observation.robot_state.holding_estimate is None and observation.robot_state.gripper_contact is None
    assert observation.sensors.lidar_points_m is None and observation.sensors.force_torque is None


@pytest.mark.parametrize("kwargs", [
    {"base_odometry": RobotPose([0, 0, 0], [0, 0, 0, 1], "world")},
    {"holding_estimate": True, "holding_source": "assisted_attachment"},
    {"gripper_contact": [True], "contact_source": "scene_contact_truth"},
])
def test_hardware_adapter_rejects_ground_truth_sources(kwargs):
    with pytest.raises(ValueError):
        RobotSensorFeedback(**kwargs)


def test_observation_switches_disable_all_paths_and_export_unknown_without_fallback(tmp_path):
    controller = make_manipulation_controller()[0]
    source, rig = source_and_rig()
    source.config = ObservationConfig(**{name: False for name in ObservationConfig.__dataclass_fields__})
    controller.session.observation_source = source
    executor = create_r1_executor(controller)
    backend = ESIControlBackend(dict(METADATA), executor, observation_source=source)
    source.on_capture = lambda snapshot: backend.publish(snapshot, controller_state=snapshot.controller_state)
    source.capture(action_boundary=True)
    executor.execute("RELEASE")
    observation = source.agent_observation()
    assert observation.agentview_rgb is None and observation.wrist_rgb is None
    assert observation.robot_state.ee_pose is None and observation.robot_state.gripper_width_m is None
    assert observation.last_action_feedback is None and not observation.action_history
    assert not backend.frames
    assert AgentObservation.from_payload(observation.to_payload()).metadata() == observation.metadata()
    path = observation.save(tmp_path)
    assert not list(tmp_path.glob("*.png"))
    assert json.loads(path.read_text())["observation_config"]["action_feedback"] is False
    assert rig.cycle == 2


def test_empty_close_is_execution_completion_and_not_grasp_or_task_success():
    controller, _, env = make_manipulation_controller()
    env.available_object = False
    source, _ = source_and_rig()
    controller.session.observation_source = source
    raw = create_r1_executor(controller).execute("GRASP")
    assert raw["reason"] == "completed" and raw["completed"]
    assert raw["debug_gripper_after"]["holding"] is False
    observation = source.agent_observation()
    feedback = observation.last_action_feedback
    assert feedback.execution_status == "completed" and feedback.target_reached is True
    assert observation.robot_state.holding_estimate is None
    assert observation.action_history[-1].result_summary == "completed"
    assert not {"success", "completed", "holding", "task_success", "reason"}.intersection(feedback.to_dict())


def test_http_and_atomic_mcp_tool_share_filtered_boundary_and_reject_debug_routes():
    pytest.importorskip("mcp")
    from mcp import Client
    from wia.runtime.web.mcp_server import create_server

    controller = make_manipulation_controller()[0]
    source, _ = source_and_rig()
    source.include_scene = True
    controller.session.observation_source = source
    executor = create_r1_executor(controller)
    backend = ESIControlBackend(dict(METADATA), executor, observation_source=source)
    source.on_capture = lambda snapshot: backend.publish(snapshot, controller_state=snapshot.controller_state)
    source.capture(action_boundary=True)
    server = serve(backend, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert requests.get(url + "/snapshot/scene", timeout=2).status_code == 404
        state = requests.get(url + "/api/state", timeout=2).json()
        assert state["robot_state"]["base_pose"] is None and state["robot_state"]["holding_estimate"] is None
        assert "world_position" not in json.dumps(state)

        async def read():
            async with Client(create_server(url)) as client:
                result = await client.call_tool("robot_get_observation", {})
                data = result.structured_content["observation"]
                assert data["snapshot_id"] == source.agent_observation().snapshot_id
                assert len([item for item in result.content if item.type == "image"]) == 2
                assert data["robot_state"] == state["robot_state"]
                assert "debug_gripper" not in result.content[0].text
        asyncio.run(read())
    finally:
        backend.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_mcp_rejects_raw_diagnostic_results_even_if_execution_succeeded():
    pytest.importorskip("mcp")
    from wia.runtime.web.mcp_server import RobotBridge
    from harness_integration.test_robot_mcp import observed_result
    with pytest.raises(ValueError, match="Unreviewed"):
        RobotBridge("http://localhost:8615")._execution_result(
            ["MV_UP"], 200, {"ok": True, "results": [observed_result("MV_UP")]})
