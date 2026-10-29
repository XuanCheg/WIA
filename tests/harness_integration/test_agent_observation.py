"""Privilege exclusion, coherent camera cycles and stable action-boundary inputs."""
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import requests
from PIL import Image

from core.action_interface import ActionExecutor, BackendCapabilities, action_result
from core.observations import (ActionHistoryEntry, AgentActionFeedback, AgentObservation,
                               AgentRobotState, ControllerState, EvaluatorState, RobotPose)
from core.sensor_observations import EstimatedObjectPosition, PerceptionEstimate, SensorReadings
from wia.runtime.sim.esi_actions import create_r1_executor
from wia.runtime.sim.esi_observations import ESIObservationSource
from wia.runtime.sim.esi_omnigibson import ESISession
from wia.runtime.sim.esi_rotation import RotationControlConfig
from wia.runtime.web.web_teleop.esi_control import ESIControlBackend
from gumi.web_teleop.server import serve
from wia.runtime.commands.capture_esi_observation import capture
from harness_integration.test_esi_base import make_controller
from harness_integration.test_esi_gripper import make_manipulation_controller
from harness_integration.test_esi_preview import METADATA


class CameraRig:
    def __init__(self):
        self.cycle = 0
        self.reads = []
        self.owner = threading.get_ident()
        self.head = np.zeros((12, 16, 3), np.uint8)
        self.wrist = np.zeros_like(self.head)
        self.pipeline = SimpleNamespace(og=SimpleNamespace(sim=SimpleNamespace(render=self.render)))

    def render(self):
        assert threading.get_ident() == self.owner
        self.cycle += 1
        self.head.fill(self.cycle % 200)
        self.wrist.fill(self.cycle % 200 + 40)

    def observe(self, *, include_wrist, include_scene=False, include_sensors=False):
        assert include_wrist and threading.get_ident() == self.owner
        self.reads.extend([("head", self.cycle), ("wrist", self.cycle)])
        raw = {"head_rgb": self.head, "wrist_rgb": self.wrist,
               "object_pose": "PRIVATE_OBJECT_POSE", "proprioception": "PRIVATE_EE",
               "segmentation_ids": "PRIVATE_SEGMENTATION", "task_state": "PRIVATE_TASK_STATE"}
        if include_scene:
            raw["scene_rgb"] = np.full_like(self.head, 250)
        return raw


def source_and_rig():
    rig = CameraRig()
    source = ESIObservationSource(rig, task_text="Approach the table.")
    return source, rig


def result(token="BASE_FWD", **extras):
    return action_result(token, completed=extras.pop("completed", True),
                         timeout=extras.pop("timeout", False), requested={"target": "PRIVATE_TARGET"},
                         actual={"object_pose": "PRIVATE_ACTUAL"}, message="PRIVATE_ERROR_WITH_ANSWER",
                         debug_object={"name": "PRIVATE_OBJECT_ID"}, **extras)


def test_snapshot_freezes_one_raw_cycle_and_allowlists_metadata(tmp_path):
    source, rig = source_and_rig()
    source.include_scene = True
    snapshot = source.capture(action_boundary=True)
    observation = snapshot.observation
    assert rig.reads == [("head", 1), ("wrist", 1)]
    assert np.all(observation.agentview_rgb == 1) and np.all(observation.wrist_rgb == 41)
    assert np.all(snapshot.scene_rgb == 250)
    rig.head.fill(200)
    rig.wrist.fill(0)
    assert np.all(observation.agentview_rgb == 1) and np.all(observation.wrist_rgb == 41)
    with pytest.raises(ValueError):
        observation.agentview_rgb[0, 0, 0] = 3
    with pytest.raises(ValueError):
        observation.wrist_rgb.setflags(write=True)
    metadata_path = observation.save(tmp_path)
    text = metadata_path.read_text()
    assert "PRIVATE_" not in text and "scene_rgb" not in text and "proprioception" not in text
    metadata = json.loads(text)
    assert set(metadata) == {"task_text", "snapshot_id", "captured_at", "action_step",
                             "action_boundary", "action_history", "images", "robot_state", "last_action_feedback", "sensors", "perception", "observation_config", "camera_geometry"}
    for name in ("agentview_rgb", "wrist_rgb"):
        assert np.array_equal(np.array(Image.open(tmp_path / metadata["images"][name]["path"])),
                              getattr(observation, name))
    restored = AgentObservation.from_payload(observation.to_payload())
    assert restored.metadata() == observation.metadata()


@pytest.mark.parametrize("kwargs,summary", [({}, "completed"), ({"completed": False}, "failed"),
    ({"completed": False, "timeout": True}, "timeout"), ({"completed": False, "cancelled": True}, "cancelled")])
def test_history_retains_only_fixed_action_and_public_outcome(kwargs, summary):
    source, _ = source_and_rig()
    source.after_action(result(**kwargs))
    observation = source.agent_observation()
    assert observation.action_history == (ActionHistoryEntry("BASE_FWD", summary),)
    assert observation.action_step == 1 and observation.action_boundary
    assert observation.last_action_feedback.execution_status == summary
    assert observation.last_action_feedback.requested == {} and observation.last_action_feedback.estimated_motion == {}
    assert "PRIVATE_" not in json.dumps(observation.metadata())


def test_history_limit_does_not_rewrite_old_snapshots():
    source, _ = source_and_rig()
    source.after_action(result())
    first = source.agent_observation()
    for _ in range(14):
        source.after_action(result("BASE_BACK"))
    assert len(source.agent_observation().action_history) == 12
    assert source.agent_observation().action_step == 15
    assert first.action_step == 1 and len(first.action_history) == 1


@pytest.mark.parametrize("action", ["ROTATE_CW", "ROTATE_CCW"])
def test_existing_optional_wrist_rotation_is_preserved_in_history(action):
    source, _ = source_and_rig()
    source.after_action(result(action))
    assert source.agent_observation().action_history == (ActionHistoryEntry(action, "completed"),)


@pytest.mark.parametrize("action", ["ROTATE_CW", "ROTATE_CCW"])
def test_native_rotation_numeric_scalars_are_normalized_without_changing_feedback(action):
    controller = make_manipulation_controller()[0]
    controller.rotation_config = RotationControlConfig.load()
    source, _ = source_and_rig()
    controller.session.observation_source = source
    outcome = create_r1_executor(controller).execute(action)
    feedback = source.agent_observation().last_action_feedback
    assert isinstance(outcome["actual"]["angle_deg"], np.floating)
    assert type(feedback.estimated_motion["angle_deg"]) is float
    assert feedback.estimated_motion["angle_deg"] == outcome["actual"]["angle_deg"]
    assert feedback.execution_status == "completed" and feedback.action == action
    assert AgentObservation.from_payload(source.agent_observation().to_payload()).metadata() == source.agent_observation().metadata()


def test_executor_captures_after_final_result_and_keeps_prior_boundary_during_motion():
    source, rig = source_and_rig()
    source.capture(action_boundary=True)
    initial = source.agent_observation()
    events = []

    def execute(token, **kwargs):
        source.capture()
        assert source.agent_observation() is initial
        events.append("motion_stopped")
        return result(token)

    source.on_capture = lambda snapshot: events.append(("capture", snapshot.observation.action_step))
    executor = ActionExecutor(execute, capabilities=BackendCapabilities(base_translation=True),
                              on_result=source.after_action)
    executor.execute("BASE_FWD")
    final = source.agent_observation()
    assert final.snapshot_id == 3 and final.action_step == 1
    assert events[-2:] == ["motion_stopped", ("capture", 1)]
    assert rig.reads[-2:] == [("head", 3), ("wrist", 3)]
    with pytest.raises(ValueError):
        executor.execute("BASE_FWD(0.5)")
    assert source.agent_observation() is final
    executor._execute_action = lambda *args, **kwargs: {"action": "BASE_FWD"}
    with pytest.raises(RuntimeError, match="invalid ActionResult"):
        executor.execute("BASE_FWD")
    assert source.agent_observation() is final


def test_r1_adapter_uses_same_source_and_internal_state_does_not_enter_observation():
    controller, robot, env = make_manipulation_controller()
    source, rig = source_and_rig()
    controller.session.observation_source = source
    source.capture(action_boundary=True)
    internal = controller.controller_state()
    assert isinstance(internal, ControllerState)
    assert internal.base_pose == dict(controller.pose(), source="simulator_ground_truth") and internal.ee_pose == controller.ee_pose()
    assert internal.joint_state["positions"] == robot.get_joint_positions().tolist()
    assert internal.controller_status["hold_action"] == controller.hold_action.tolist()
    assert "holding" not in internal.gripper_state and "holding" not in internal.proprioception
    assert internal.proprioception["eef_pos"] == controller.ee_pose()["base_position"]
    assert internal.joint_state["velocities"] is None  # No invented readback.
    executor = create_r1_executor(controller)
    executor.execute("RELEASE")
    assert source.agent_observation().action_history[-1].action == "RELEASE"
    assert "hold_action" not in source.agent_observation().metadata()
    assert source.agent_observation().action_step == 1 and rig.cycle == 2
    assert source.agent_observation().robot_state.ee_position_m == tuple(controller.ee_pose()["base_position"])
    assert source.agent_observation().last_action_feedback.action == "RELEASE"
    assert source.agent_observation().last_action_feedback.target_reached is True
    assert "holding" not in source.agent_observation().last_action_feedback.estimated_motion


def test_compact_robot_state_excludes_simulator_velocity_world_joint_target_and_identity():
    controller = make_controller(yaw=90)[0]
    state = replace(controller.controller_state(),
        base_velocity={"frame": "world", "linear_m_s": [0, 1, 0], "angular_rad_s": [0, 0, 0.5]},
        ee_pose={"base_position": [0.4, 0.2, 0.8], "base_quaternion_xyzw": [0, 0, 0, 1],
                 "world_position": "PRIVATE_WORLD_EE"},
        gripper_state={"gripper_width_m": 0.04, "gripper_closed": True, "holding": True,
                       "object_name": "PRIVATE_HELD_OBJECT"})
    compact = AgentRobotState.from_controller_state(state)
    assert compact.base_linear_velocity_m_s is None
    assert compact.base_angular_velocity_rad_s is None
    assert compact.ee_position_m == (0.4, 0.2, 0.8) and compact.holding_estimate is None
    source, _ = source_and_rig()
    source.controller_state_provider = lambda: state
    metadata = source.capture(action_boundary=True).observation.metadata()
    assert "PRIVATE_" not in json.dumps(metadata)
    assert metadata["robot_state"]["base_pose"] is None
    assert not {"joint_state", "hold_action", "controller_status"}.intersection(metadata["robot_state"])
    state.ee_pose["base_position"][0] = 99
    assert source.agent_observation().robot_state.ee_position_m == (0.4, 0.2, 0.8)


def test_previous_feedback_preserves_actual_movement_and_public_failure_but_filters_backend_details():
    raw = result("MV_DOWN", completed=False, reason="arm_motion_blocked")
    raw["requested"] = {"delta_m": 0.02, "delta_base_m": [0, 0, -0.02], "target": "PRIVATE_TARGET"}
    raw["actual"] = {"delta_m": 0.015, "delta_base_m": [0, 0, -0.015], "object_pose": "PRIVATE_OBJECT"}
    feedback = AgentActionFeedback.from_result(raw)
    assert feedback.observable_error == "arm_motion_blocked" and feedback.execution_status == "failed"
    assert feedback.requested["delta_m"] == 0.02 and feedback.estimated_motion["delta_m"] == 0.015
    assert "PRIVATE_" not in json.dumps(feedback.to_dict())
    raw["actual"]["delta_base_m"][2] = -99
    assert feedback.estimated_motion["delta_base_m"] == (0, 0, -0.015)
    with pytest.raises(TypeError):
        feedback.estimated_motion["target"] = [1, 2, 3]
    source, _ = source_and_rig()
    source.after_action(raw)
    observation = source.agent_observation()
    assert observation.last_action_feedback.estimated_motion["delta_m"] == 0.015
    assert AgentObservation.from_payload(observation.to_payload()).metadata() == observation.metadata()


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(object_pose=[1, 2, 3]),
    lambda p: p["images"].update(scene_rgb={}),
    lambda p: p["images"]["agentview_rgb"].update(transform="crop"),
    lambda p: p["images"]["wrist_rgb"].update(sha256="wrong"),
    lambda p: p["action_history"].append({"action": "BASE_FWD", "result_summary": "PRIVATE_OBJECT"}),
    lambda p: p["action_history"].append({"action": "BASE_FWD", "result_summary": "completed", "target": [1, 2]}),
    lambda p: p.update(robot_state={"world_position": [1, 2, 3]}),
    lambda p: p.update(last_action_feedback={"action": "BASE_FWD", "object_pose": [1, 2, 3]}),
    lambda p: p["sensors"].update(segmentation_ids={}),
    lambda p: p.update(perception={"object_poses": [1, 2, 3], "source": "ground_truth"}),
])
def test_unreviewed_fields_transforms_and_history_cannot_cross_serialization(mutate):
    source, _ = source_and_rig()
    payload = source.capture(action_boundary=True).observation.to_payload()
    mutate(payload)
    with pytest.raises((ValueError, TypeError)):
        AgentObservation.from_payload(payload)


def test_missing_wrist_and_wrong_thread_cannot_produce_agent_input():
    source, rig = source_and_rig()
    errors = []

    def worker():
        try:
            source.capture(action_boundary=True)
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=2)
    assert errors and rig.cycle == 0
    rig.observe = lambda **kwargs: {"head_rgb": rig.head}
    with pytest.raises(RuntimeError, match="both native R1"):
        source.capture(action_boundary=True)
    with pytest.raises(RuntimeError, match="No action-boundary"):
        source.agent_observation()


def test_session_keeps_evaluator_truth_separate_and_rejects_scene_or_wrist_as_head():
    session = ESISession()
    session.robot = SimpleNamespace(sensors={"r1:eyes:Camera:0": object(), "r1:left_eef_link:Camera:0": object()})
    session.camera_name, session.wrist_camera_name = tuple(session.robot.sensors)
    session.payload = {"_question": "Approach the table.", "answer": "PRIVATE_ANSWER"}
    session.task_state = {"success": "PRIVATE_TASK_STATE"}
    assert session.observation_source.task_text == "Approach the table."
    truth = session.evaluator_state()
    assert isinstance(truth, EvaluatorState) and truth.task_metadata["answer"] == "PRIVATE_ANSWER"
    truth.task_metadata["answer"] = "modified"
    assert session.payload["answer"] == "PRIVATE_ANSWER"
    for selected in (session.wrist_camera_name, "viewer_camera"):
        session._observation_source = None
        session.camera_name = selected
        with pytest.raises(RuntimeError, match="native R1 head"):
            _ = session.observation_source


def test_http_bundle_and_debug_export_share_gumi_raw_pixels_without_recapture(tmp_path):
    source, rig = source_and_rig()
    source.include_scene = True
    controller = make_controller()[0]
    internal = replace(controller.controller_state(), proprioception={"eef_pos": "PRIVATE_EE"})
    backend = ESIControlBackend(dict(METADATA), create_r1_executor(controller), observation_source=source)
    source.on_capture = lambda snapshot: backend.publish(snapshot, controller_state=internal)
    server = serve(backend, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert requests.get(url + "/api/agent-observation", timeout=2).status_code == 503
        source.after_action(result())
        boundary = source.agent_observation()
        assert "PRIVATE_EE" not in json.dumps(backend.state())
        # Both public GUMI and Agent views use mounted cameras; local scene stays private.
        assert backend.frames["agentview"] == backend.frames["head"] and "scene" not in backend.frames
        for view, rgb in (("head", boundary.agentview_rgb), ("wrist", boundary.wrist_rgb)):
            decoded = cv2.cvtColor(cv2.imdecode(np.frombuffer(backend.frames[view], np.uint8), cv2.IMREAD_COLOR),
                                   cv2.COLOR_BGR2RGB)
            assert np.array_equal(decoded, rgb)
        path = capture(url, tmp_path)
        assert rig.cycle == 1 and "PRIVATE_" not in path.read_text()
        source.capture()  # Monitor refreshes, but cannot replace the stable input.
        returned = AgentObservation.from_payload(requests.get(url + "/api/agent-observation", timeout=2).json())
        assert returned.snapshot_id == boundary.snapshot_id and rig.cycle == 2
        assert backend.frame_seq == 2
    finally:
        backend.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_simulator_base_pose_stays_unavailable_and_gripper_command_is_explicit():
    controller, robot, env = make_manipulation_controller()
    robot.yaw = np.pi / 2
    source, _ = source_and_rig()
    controller.session.observation_source = source
    executor = create_r1_executor(controller)
    source.capture(action_boundary=True)
    assert source.agent_observation().robot_state.base_pose is None
    executor.execute("RELEASE")
    executor.execute("BASE_FWD")
    state = source.agent_observation().robot_state
    assert state.base_pose is None and state.base_linear_velocity_m_s is None
    assert state.gripper_command_name == "RELEASE"
    assert state.ee_pose.frame == "robot_base"
    feedback = source.agent_observation().last_action_feedback
    assert not feedback.estimated_motion and feedback.target_reached is None
    assert feedback.execution_status == "completed"
    assert AgentObservation.from_payload(source.agent_observation().to_payload()).metadata() == source.agent_observation().metadata()


def test_robot_pose_relative_transform_handles_rotated_translated_origin():
    from scipy.spatial.transform import Rotation
    origin = RobotPose([10, 20, 0], Rotation.from_euler("z", 90, degrees=True).as_quat(), "world")
    current = RobotPose([10, 21, 0], Rotation.from_euler("z", 105, degrees=True).as_quat(), "world")
    relative = current.relative_to(origin)
    assert relative.position_m == pytest.approx([1, 0, 0])
    assert Rotation.from_quat(relative.quaternion_xyzw).as_euler("xyz", degrees=True) == pytest.approx([0, 0, 15])


def test_native_rgb_and_depth_read_each_camera_once_and_exclude_segmentation(tmp_path):
    class Sensor:
        all_modalities = {"rgb", "depth_linear", "seg_instance"}

        def __init__(self, name):
            self.name, self.reads = name, 0
            self.modalities = {"rgb"}

        def add_modality(self, modality):
            self.modalities.add(modality)

        def get_obs(self):
            self.reads += 1
            data = {"rgb": np.ones((8, 10, 4), np.uint8), "seg_instance": "PRIVATE_SEGMENTATION"}
            if "depth_linear" in self.modalities:
                data["depth_linear"] = np.full((8, 10), 1.5, np.float32)
                data["depth_linear"][0, 0] = np.inf  # Invalid native background range.
            return data, {"labels": "PRIVATE_LABELS"}

    session = ESISession()
    head, wrist = Sensor("r1:eyes:Camera:0"), Sensor("r1:left_eef_link:Camera:0")
    session.robot = SimpleNamespace(sensors={head.name: head, wrist.name: wrist})
    session.camera_name, session.wrist_camera_name = head.name, wrist.name
    session.pipeline = SimpleNamespace(og=SimpleNamespace(sim=SimpleNamespace(render=lambda: None)))
    session.payload = {"_question": "Approach the table.", "answer": "PRIVATE_ANSWER"}
    source = session.observation_source
    source.controller_state_provider = None
    snapshot = source.capture(action_boundary=True)
    assert head.reads == wrist.reads == 1
    assert head.modalities == wrist.modalities == {"rgb", "depth_linear"}
    obs = snapshot.observation
    assert obs.sensors.head_depth_m[1, 1] == 1.5 and np.isinf(obs.sensors.head_depth_m[0, 0])
    assert obs.sensors.lidar_points_m is obs.sensors.force_torque is obs.sensors.tactile is None
    assert "PRIVATE_" not in json.dumps(obs.metadata())
    payload = obs.to_payload()
    restored = AgentObservation.from_payload(payload)
    assert np.array_equal(restored.sensors.head_depth_m, obs.sensors.head_depth_m)
    path = restored.save(tmp_path)
    metadata = json.loads(path.read_text())
    assert np.array_equal(np.load(tmp_path / metadata["sensors"]["head_depth_m"]["path"], allow_pickle=False),
                          obs.sensors.head_depth_m)
    payload["sensors"]["head_depth_m"]["source"] = "simulator_ground_truth"
    with pytest.raises(ValueError, match="provenance"):
        AgentObservation.from_payload(payload)


def test_future_perception_extension_receives_only_allowed_snapshot_and_exports_estimates(tmp_path):
    source, rig = source_and_rig()
    source.include_scene = True
    received = []

    def estimate(observation):
        received.append(observation)
        assert not hasattr(observation, "controller_state") and not hasattr(observation, "scene_rgb")
        assert observation.perception is None
        return PerceptionEstimate(snapshot_id=observation.snapshot_id, input_sensors=("agentview_rgb",),
                                  frame="robot_base", objects=(EstimatedObjectPosition("red object", [0.4, 0.1, 0.8]),),
                                  point_cloud_m=np.array([[0.4, 0.1, 0.8]], np.float32))

    source.perception_provider = estimate
    observation = source.capture(action_boundary=True).observation
    assert len(received) == 1 and rig.cycle == 1
    restored = AgentObservation.from_payload(observation.to_payload())
    assert restored.metadata() == observation.metadata()
    path = restored.save(tmp_path)
    metadata = json.loads(path.read_text())
    assert metadata["perception"]["source"] == "estimated_from_allowed_sensors"
    points = np.load(tmp_path / metadata["perception"]["point_cloud_m"]["path"], allow_pickle=False)
    assert points.shape == (1, 3)
    with pytest.raises(ValueError):
        observation.perception.point_cloud_m.setflags(write=True)
    for invalid in (replace(observation.perception, snapshot_id=99),
                    replace(observation.perception, input_sensors=("lidar_points_m",))):
        source.perception_provider = lambda obs, value=invalid: value
        with pytest.raises(ValueError, match="this snapshot"):
            source.capture(action_boundary=True)
    with pytest.raises(ValueError, match="provenance"):
        replace(observation.perception, source="ground_truth")


def test_sensor_contract_rejects_unaligned_depth_and_nonphysical_force_values():
    source, _ = source_and_rig()
    observation = source.capture(action_boundary=True).observation
    with pytest.raises(ValueError, match="aligned"):
        replace(observation, sensors=SensorReadings(head_depth_m=np.ones((2, 3), np.float32)))
    with pytest.raises(ValueError):
        SensorReadings(force_torque=np.array([0, 0, np.nan, 0, 0, 0]))
    with pytest.raises(ValueError):
        SensorReadings(head_depth_m=np.full((12, 16), -1, np.float32))
