"""Joint-limited continuation, measurement-only gripper feedback and calibrated geometry."""
import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest

from core.action_compiler import action_bounds, compile_action
from core.action_interface import action_result
from core.agent.decision import DecisionInput
from core.agent.image_inputs import image_inputs, reference_depth_samples
from wia.runtime.agent.zero_shot import run_episode
from core.camera_geometry import CameraGeometry, RobotKinematics
from core.feedback_semantics import FEEDBACK_SEMANTICS
from core.observations import AgentActionFeedback, AgentObservation, AgentRobotState, ControllerState
from core.sensor_observations import SensorReadings
from harness_integration.test_parameterized_actions import AVAILABLE, CAPABILITIES, LIMITS, RESOLUTION
from harness_integration.test_zero_shot_agent import Agent, Environment


def test_each_action_bound_is_shared_by_schema_prompt_and_compiler():
    bounds = action_bounds(AVAILABLE, RESOLUTION, LIMITS)
    assert [bounds[a]["maximum"] for a in ("BASE_FWD", "MV_DOWN", "TRUNK_UP", "BASE_TURN_LEFT", "ROTATE_CW")] == [.4, .08, .08, 40, 40]
    obs = asyncio.run(Environment().observe())
    request = DecisionInput(obs, AVAILABLE, action_mode="parameterized", action_context={
        "parameterized_actions": {"resolution": RESOLUTION.to_dict(), "limits": LIMITS.to_dict()}})
    for action in ("BASE_FWD", "MV_DOWN", "TRUNK_UP", "BASE_TURN_LEFT", "ROTATE_CW"):
        limit = bounds[action]
        legal = {"action": action, limit["magnitude_field"]: limit["maximum"]}
        jsonschema.validate(legal, request.schema)
        assert compile_action(legal, CAPABILITIES, RESOLUTION, LIMITS).repeat_count == 8
        invalid = dict(legal, **{limit["magnitude_field"]: limit["maximum"] + limit["primitive_step"]})
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, request.schema)
        with pytest.raises(ValueError, match='maximum'):
            compile_action(invalid, CAPABILITIES, RESOLUTION, LIMITS)
    changed = action_bounds(AVAILABLE, replace(RESOLUTION, arm_translation_m=.02), replace(LIMITS, max_primitives=5))
    assert changed['MV_DOWN']['maximum'] == .1
    assert 'Per-action request bounds' in request.prompt()


def test_mcp_rejects_oversize_before_queueing_and_explains_the_actual_bound():
    from wia.runtime.web.mcp_server import RobotBridge
    bridge = RobotBridge('http://localhost:8615')
    calls = []

    async def request(method, path, **kwargs):
        calls.append((method, path))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'capabilities': CAPABILITIES.to_dict(), 'parameterized_actions': {
                'resolution': RESOLUTION.to_dict(), 'limits': LIMITS.to_dict()}})

    bridge.request = request
    reply = asyncio.run(bridge.execute_parameterized({'action': 'MV_FWD', 'distance_m': .1}, control_token='test'))
    assert reply.structured_content['outcome_known'] and '0.08' in reply.structured_content['message']
    assert calls == [('GET', '/api/agent/state')]


def test_mcp_confirms_stopped_limit_continuation_without_banning_smaller_same_direction():
    from wia.runtime.web.mcp_server import RobotBridge
    bridge = RobotBridge('http://localhost:8615')
    feedback = AgentActionFeedback.from_result(action_result('MV_FWD', completed=False, timeout=False,
        requested={}, actual={}, message='stopped', reason='arm_joint_limit', stop_steps=3,
        stop_stability_verified=True)).to_dict()
    reply = bridge._execution_result(['MV_FWD'], 200, {'ok': False, 'results': [feedback]}).structured_content
    assert reply['can_continue'] and reply['smaller_same_action_allowed']
    assert reply['automatic_replay'] is False and 'retry_same_action' not in reply
    assert 'new observation' in reply['message'] and 'arm_joint_limit' in reply['message']
    feedback['motion_stopped'] = None
    reply = bridge._execution_result(['MV_FWD'], 200, {'ok': False, 'results': [feedback]}).structured_content
    assert 'can_continue' not in reply


class RepeatedLimits(Environment):
    async def execute(self, action):
        self.calls.append(action)
        result = action_result(action, completed=False, timeout=False, requested={}, actual={},
            message='joint limit', reason='arm_joint_limit', stop_steps=1, recovery_completed=True,
            stop_stability_verified=True)
        self.source.after_action(result)
        return {'ok': False, 'outcome_known': True, 'motion_stopped': True, 'reason': 'arm_joint_limit',
                'results': [self.source.last_action_feedback.to_dict()]}


def test_more_than_three_confirmed_limits_continue_and_keep_same_action_available(tmp_path):
    environment = RepeatedLimits()
    agent = Agent(*(['{"action":"BASE_FWD"}']*5 + ['{"action":"DONE"}']))
    summary = asyncio.run(run_episode(environment, agent, tmp_path, max_steps=6))
    assert summary['termination_reason'] == 'model_done'
    assert len(environment.calls) == len(summary['recovered_execution_errors']) == 5
    assert summary['agent_decisions'] == summary['model_calls'] == 6
    for request in agent.inputs[1:]:
        recovery = request.action_context['recovery']
        assert recovery['reason'] == 'arm_joint_limit' and recovery['episode_continues']
        assert 'BASE_FWD' in request.available_actions
        assert request.observation.last_action_feedback.observable_error == 'arm_joint_limit'
    assert agent.inputs[-1].observation.action_step == 5


@pytest.mark.parametrize('width,fraction', [(0., 1.), (.038, .62), (.1, 0.)])
def test_gripper_feedback_is_closure_geometry_without_grasp_judgement(width, fraction):
    state = ControllerState(base_pose={}, base_velocity=None, ee_pose=None, joint_state={}, controller_status={},
        gripper_state={'gripper_width_m': width, 'gripper_joint_positions_m': [width/2, width/2],
                       'gripper_closed': width == 0, 'gripper_open': width == .1, 'holding': True, 'object_name': 'PRIVATE_TARGET'},
        gripper_control={'close_joint_positions_m': [0., 0.], 'open_joint_positions_m': [.05, .05]})
    public = AgentRobotState.from_controller_state(state)
    assert public.gripper_closure_fraction == pytest.approx(fraction)
    assert public.gripper_open_width_m == .1 and public.gripper_closed_width_m == 0
    assert public.sources['gripper_closure_fraction'] == 'finger_joint_encoders'
    feedback = AgentActionFeedback.from_result(action_result('GRASP', completed=True, timeout=False,
        requested={}, actual={}, message='completed', reason='completed'), state=state)
    assert feedback.execution_status == 'completed'
    assert feedback.target_reached == (width == 0)
    assert feedback.estimated_motion['gripper_closure_fraction'] == pytest.approx(fraction)
    data = json.dumps({'robot_state': public.to_dict(), 'feedback': feedback.to_dict(), 'semantics': FEEDBACK_SEMANTICS})
    assert 'PRIVATE_TARGET' not in data and 'empty_grasp' not in data and 'likely_empty' not in data
    assert public.holding_estimate is None


def camera(name='wrist_rgb', size=(16, 12)):
    return CameraGeometry(name=name, image_size=size, intrinsics=((10, 0, 8), (0, 10, 6), (0, 0, 1)),
        camera_to_base=np.eye(4), control_point_base_m=(0, 0, 1), grasp_center_base_m=(.1, .2, 1))


def test_projection_uses_offset_reference_and_metric_optical_depth_roundtrip():
    geometry = camera()
    projection = geometry.project(geometry.grasp_center_base_m)
    assert projection['pixel_uv'] == [9., 8.] and projection['inside_image']
    assert geometry.project(geometry.control_point_base_m)['pixel_uv'] == [8., 6.]
    assert np.allclose(geometry.unproject(projection['pixel_uv'], projection['optical_depth_m']), [.1, .2, 1])
    assert geometry.project([0, 0, -1]) is None
    with pytest.raises(ValueError):
        CameraGeometry.from_payload(dict(geometry.to_dict(), target_position=[1, 2, 3]))


def test_geometry_roundtrip_preserves_native_rgb_and_omits_depth_visualizations():
    original = asyncio.run(Environment().observe())
    depth = np.ones((12, 16), np.float32)
    observation = replace(original, camera_geometry={'wrist_rgb': camera()}, sensors=SensorReadings(wrist_depth_m=depth))
    loaded = AgentObservation.from_payload(observation.to_payload())
    assert loaded.camera_geometry['wrist_rgb'].to_dict() == observation.camera_geometry['wrist_rgb'].to_dict()
    assert np.array_equal(loaded.wrist_rgb, original.wrist_rgb)
    inputs = image_inputs(loaded)
    assert inputs[1][3] == 'identity' and np.array_equal(inputs[1][1], original.wrist_rgb)
    assert len(inputs) == 2 and all(item[3] == 'identity' for item in inputs)
    assert np.array_equal(observation.sensors.wrist_depth_m, depth)
    assert 'Calibrated camera geometry:' in DecisionInput(loaded, ('DONE',)).prompt()


def test_reference_depth_samples_are_measurements_and_preserve_invalid_centre(tmp_path):
    original = asyncio.run(Environment().observe())
    depth = np.ones((12, 16), np.float32)
    depth[8, 9] = np.nan
    observation = replace(original, camera_geometry={'wrist_rgb': camera()}, sensors=SensorReadings(wrist_depth_m=depth))
    sample = reference_depth_samples(observation)['wrist_rgb']
    assert sample['pixel_uv'] == [9, 8] and sample['sensor_optical_z_m'] is None
    assert sample['patch_valid_pixels'] == 24 and sample['patch_median_optical_z_m'] == 1
    assert sample['robot_reference_optical_z_m'] == 1
    assert reference_depth_samples(original)['wrist_rgb'] is None
    assert np.isnan(observation.sensors.wrist_depth_m[8, 9])
    prompt = DecisionInput(observation, ('DONE',))
    assert 'Raw metric depth access:' in prompt.prompt()
    assert 'Robot-reference depth samples:' not in prompt.prompt()
    assert 'Raw metric depth access:' not in prompt.global_prompt()


def test_camera_fk_moves_with_encoder_positions_and_rejects_missing_encoder(tmp_path):
    path = tmp_path/'robot.urdf'
    path.write_text('<robot><joint name="j1" type="revolute"><parent link="base_link"/><child link="arm"/>'
                    '<axis xyz="0 0 1"/><origin xyz="1 0 0"/></joint><joint name="camera" type="fixed">'
                    '<parent link="arm"/><child link="camera"/><origin xyz="1 0 0"/></joint></robot>')
    fk = RobotKinematics(path)
    assert np.allclose(fk.frames({'j1': 0})['camera'][:3, 3], [2, 0, 0])
    assert np.allclose(fk.frames({'j1': np.pi/2})['camera'][:3, 3], [1, 1, 0])
    with pytest.raises(ValueError, match='encoder'):
        fk.frames({})


def test_camera_fk_prunes_unrelated_encoder_chains_without_guessing_positions(tmp_path):
    path = tmp_path/'robot.urdf'
    path.write_text('<robot><joint name="arm" type="prismatic"><parent link="base_link"/>'
        '<child link="eef"/><axis xyz="0 0 1"/></joint><joint name="steering" type="revolute">'
        '<parent link="base_link"/><child link="wheel"/><axis xyz="0 0 1"/></joint></robot>')
    fk = RobotKinematics(path)
    assert np.allclose(fk.frames({'arm': .2}, links={'eef'})['eef'][:3, 3], [0, 0, .2])
    with pytest.raises(ValueError, match='steering'):
        fk.frames({'arm': .2}, links={'wheel'})
    with pytest.raises(ValueError, match='No URDF chain'):
        fk.frames({'arm': .2}, links={'eyes'})


def test_calibrated_joint_uses_both_mechanical_anchors_and_measured_angle(tmp_path):
    path = tmp_path/'robot.urdf'
    path.write_text('<robot><joint name="j" type="revolute"><parent link="base_link"/>'
                    '<child link="arm"/><axis xyz="0 0 1"/></joint></robot>')
    fk = RobotKinematics(path)
    a, b = np.eye(4), np.eye(4)
    a[:3, 3], b[:3, 3] = [0, 0, .04165], [.1, 0, 0]
    fk.calibrate_joint('j', parent='base_link', child='arm', parent_joint=a, child_joint=b, axis=[0, 0, 1])
    assert np.allclose(fk.frames({'j': 0})['arm'][:3, 3], [-.1, 0, .04165])
    assert np.allclose(fk.frames({'j': np.pi/2})['arm'][:3, 3], [0, -.1, .04165])
    with pytest.raises(ValueError, match='topology'):
        fk.calibrate_joint('j', parent='wrong', child='arm', parent_joint=a, child_joint=b, axis=[0, 0, 1])


def test_camera_position_and_wrist_offset_are_explicit_and_follow_robot_pose():
    from core.camera_geometry import pose_matrix
    pose = ([.5, .2, 1.], [1., 0., 0., 0.])
    camera = pose_matrix([.55, .2, 1.05], [1., 0., 0., 0.])
    geometry = CameraGeometry('wrist_rgb', (16, 12), [[10, 0, 8], [0, 10, 6], [0, 0, 1]], camera,
                              control_point_base_m=pose[0], grasp_center_base_m=pose[0])
    values = geometry.mounting_measurements(pose)
    assert np.allclose(values['camera_origin_base_m'], [.55, .2, 1.05])
    assert np.allclose(values['camera_offset_from_grasp_reference_base_m'], [.05, 0, .05])
    assert np.allclose(values['camera_offset_from_end_effector_m'], [.05, 0, -.05])
    assert np.allclose(values['optical_forward_axis_base'], [0, 0, -1])
    obs = asyncio.run(Environment().observe())
    obs = replace(obs, camera_geometry={'wrist_rgb': geometry})
    request = DecisionInput(obs, ('DONE',))
    assert 'ahead of and above' in request.prompt()
    assert 'Camera mounting measurements:' in request.prompt()
    assert 'camera_origin_base_m' in request.prompt()
    assert 'Camera mounting measurements:' not in request.global_prompt()


def test_native_intrinsics_wait_for_render_readiness_without_physics():
    from wia.runtime.sim.esi_camera_geometry import native_intrinsics_when_ready
    valid = np.array([[829.1497500210653, 0, 512], [0, 829.1497500210653, 512], [0, 0, 1]])
    readings = iter((np.zeros((3, 3)), np.zeros((3, 3)), valid))
    class Sensor:
        name = 'test_robot:eyes'
        @property
        def intrinsic_matrix(self):
            return next(readings)
    calls = []
    session = SimpleNamespace(invalidate_camera_cache=lambda: calls.append('invalidate'),
        pipeline=SimpleNamespace(og=SimpleNamespace(sim=SimpleNamespace(render=lambda: calls.append('render')))))
    actual = native_intrinsics_when_ready(session, Sensor())
    assert np.array_equal(actual, valid)
    assert calls == ['invalidate', 'render', 'invalidate', 'render']


def test_persistently_invalid_native_intrinsics_are_never_substituted():
    from wia.runtime.sim.esi_camera_geometry import native_intrinsics_when_ready
    from wia.runtime.sim.esi_omnigibson import NativeCameraNotReady
    calls = []
    sensor = SimpleNamespace(name='test_robot:eyes', intrinsic_matrix=np.zeros((3, 3)))
    session = SimpleNamespace(pipeline=SimpleNamespace(og=SimpleNamespace(
        sim=SimpleNamespace(render=lambda: calls.append('render')))))
    with pytest.raises(NativeCameraNotReady, match='eight reads'):
        native_intrinsics_when_ready(session, sensor)
    assert len(calls) == 7
