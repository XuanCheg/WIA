"""Eight native views, metric depth, height kinematics and safe episode recovery."""
import asyncio
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.action_interface import BackendCapabilities, action_result
from core.agent.decision import DecisionInput
from core.agent.image_inputs import image_inputs
from wia.runtime.agent.zero_shot import run_episode
from core.observations import AgentObservation
from core.observations import AgentRobotState, ControllerState
from core.robot_cameras import ADDITIONAL_RGB_NAMES, ROBOT_CAMERA_LABELS
from core.sensor_observations import SensorReadings
from wia.runtime.sim.esi_trunk import TrunkControlConfig, TrunkKinematics, bounded_joint_step
from harness_integration.test_agent_observation import source_and_rig
from harness_integration.test_esi_ik import guarded_solver
from harness_integration.test_zero_shot_agent import Agent, Environment


def test_eight_camera_roundtrip_preserves_native_pixels_and_rejects_debug_views(tmp_path):
    source, _ = source_and_rig()
    observation = source.capture(action_boundary=True).observation
    extra = {name: np.full((12, 16, 3), index+1, np.uint8) for index, name in enumerate(ADDITIONAL_RGB_NAMES)}
    observation = replace(observation, additional_rgb=extra)
    payload = observation.to_payload()
    loaded = AgentObservation.from_payload(payload)
    assert list(loaded.rgb_images()) == list(ROBOT_CAMERA_LABELS)
    for name, rgb in observation.rgb_images().items():
        assert np.array_equal(rgb, loaded.rgb_images()[name])
    observation.save(tmp_path)
    assert len(list(tmp_path.glob('*.png'))) == 8
    with pytest.raises(ValueError, match='robot-mounted'):
        replace(observation, additional_rgb={'global_overhead': extra[ADDITIONAL_RGB_NAMES[0]]})


def test_all_rgb_is_labelled_and_raw_depth_is_separate_and_not_changed():
    source, _ = source_and_rig()
    original = source.capture(action_boundary=True).observation
    depth = np.ones((12, 16), np.float32)
    observation = replace(original, additional_rgb={name: original.wrist_rgb for name in ADDITIONAL_RGB_NAMES},
                          sensors=SensorReadings(head_depth_m=depth, wrist_depth_m=depth*2))
    inputs = image_inputs(observation)
    assert len(inputs) == 8 and list(observation.rgb_images()) == [item[0] for item in inputs]
    assert all(item[3] == 'identity' for item in inputs[:8])
    assert np.array_equal(observation.sensors.head_depth_m, depth)
    prompt = DecisionInput(observation, ('DONE',)).prompt()
    assert all(name in prompt for name in ROBOT_CAMERA_LABELS)
    assert 'Raw metric depth access:' in prompt and 'optical Z in metres' in prompt
    assert 'warm=near' not in prompt and 'black=invalid' not in prompt


def test_trunk_hold_suspends_cartesian_solver_without_changing_fault_semantics():
    guard, ctrl, solver, state = guarded_solver(proposal=[1, 0, 0, 0, 0, 0])
    held = state['joint_position'].copy()
    guard.explicit_joint_hold = held
    assert np.array_equal(ctrl.compute_control({}, state), held)
    assert guard.fault is None
    guard.reset()
    ctrl.compute_control({}, state)
    assert guard.fault == 'ik_joint_jump'


def test_trunk_encoder_arrays_are_projected_with_explicit_provenance_and_visibility():
    state = ControllerState(base_pose={}, base_velocity=None, ee_pose=None, joint_state={},
                            gripper_state=None, controller_status={}, proprioception={
                                'trunk_joint_positions_rad': np.array([.7, -1.4, -.7, 0]), 'trunk_height_m': .96})
    measured = AgentRobotState.from_controller_state(state)
    assert measured.trunk_joint_positions_rad == (.7, -1.4, -.7, 0)
    assert measured.sources['trunk_joint_positions_rad'] == 'joint_encoders'
    from core.observation_config import ObservationConfig
    hidden = AgentRobotState.from_controller_state(state, config=ObservationConfig(trunk_state=False))
    assert hidden.trunk_height_m is None and hidden.trunk_joint_positions_rad is None


def test_camera_resolution_override_applies_to_all_native_and_chassis_views(monkeypatch):
    from wia.runtime.sim.esi_robot_cameras import configure_robot_cameras
    class Sensor:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.modalities = kwargs.get('modalities', {'rgb'})
            self.prim_path = '/World/scene_0'+kwargs.get('relative_prim_path', '/robot/native')
        def get_obs(self):
            return {'rgb': np.zeros((2, 2, 3), dtype=np.uint8)}, {}
        def load(self, scene):
            pass
        def initialize(self):
            pass
        def set_position_orientation(self, **kwargs):
            self.pose = kwargs
        def get_position_orientation(self, **kwargs):
            return np.zeros(3), np.array([0, 0, 0, 1])
    native = ['r:eyes:Camera:0', 'r:left_eef_link:Camera:0', 'r:right_eef_link:Camera:0']
    sensors = {name: Sensor(name=name, image_width=512, image_height=512) for name in native}
    robot = SimpleNamespace(name='r', sensors=sensors, base_footprint_link_name='base_link',
                            links={'base_link': SimpleNamespace(prim_path='/World/scene_0/robot/base_link')})
    session = SimpleNamespace(robot=robot, rgb_cameras=native, camera_name=native[0], wrist_camera_name=native[1],
                              env=SimpleNamespace(scene=SimpleNamespace(prim_path='/World/scene_0')),
                              pipeline=SimpleNamespace(og=SimpleNamespace(sim=SimpleNamespace(render=lambda: None))))
    monkeypatch.setitem(sys.modules, 'omnigibson.sensors', SimpleNamespace(VisionSensor=Sensor))
    settings = SimpleNamespace(set_int=lambda key, value: None)
    monkeypatch.setitem(sys.modules, 'carb', SimpleNamespace(settings=SimpleNamespace(get_settings=lambda: settings)))
    configure_robot_cameras(session, 'configs/esi_robot_cameras.json', resolution=(1280, 720))
    assert len(session.robot_camera_registry) == 8
    assert all((sensor.image_width, sensor.image_height) == (1280, 720) for sensor in sensors.values())
    assert all(mount['render_size'] == [1280, 720] for mount in session.robot_camera_mounts.values())


def kinematics(tmp_path):
    # Native R1 URDF geometry. The known folded pose gives a 0.961971 m endpoint.
    origins = [('-0.032845 0 0.301', '-1.5708 0 0'), ('0 -0.4 0', '0 0 0'),
               ('0 -0.3 0', '0 0 0'), ('0 -0.12474 0', '1.5708 0 0')]
    limits = [(-1.1345, 1.8326), (-2.7925, 2.5307), (-2.0944, 1.8326), (-3.0543, 3.0543)]
    joints = []
    for i, ((xyz, rpy), (lower, upper)) in enumerate(zip(origins, limits), 1):
        parent = 'base_link' if i == 1 else 'torso_link'+str(i-1)
        axis = '0 0 1' if i <= 2 else '0 0 -1'
        joints.append(f'<joint name="torso_joint{i}" type="revolute"><origin xyz="{xyz}" rpy="{rpy}"/>'
                      f'<parent link="{parent}"/><child link="torso_link{i}"/><axis xyz="{axis}"/>'
                      f'<limit lower="{lower}" upper="{upper}"/></joint>')
    path = tmp_path/'r1.urdf'
    path.write_text('<robot>'+''.join(joints)+'</robot>')
    return TrunkKinematics(path)


def test_trunk_lift_and_lower_preserve_horizontal_position_orientation_and_return(tmp_path):
    model = kinematics(tmp_path)
    joints = np.deg2rad([40, -80, -40, 0])
    initial_position, initial_orientation = model.pose(joints)
    assert initial_position[2] == pytest.approx(.96197111018, abs=1e-7)
    for delta in [.02]*4 + [-.02]*4:
        joints, target, _ = model.solve(joints, delta, model.lower+.04, model.upper-.04)
        position, orientation = model.pose(joints)
        assert np.linalg.norm(position-target) < 1e-6
        assert np.linalg.norm(position[:2]-initial_position[:2]) < 1e-6
        assert (initial_orientation.inv()*orientation).magnitude() < 1e-6
    assert np.linalg.norm(position-initial_position) < 1e-6
    assert BackendCapabilities(trunk_height=True).available_actions == ('TRUNK_UP', 'TRUNK_DOWN')
    with pytest.raises(ValueError):
        TrunkControlConfig(minimum_height_m=1.1, maximum_height_m=1.0)


def test_coupled_trunk_trajectory_keeps_body_upright_between_endpoints(tmp_path):
    model = kinematics(tmp_path)
    current = np.deg2rad([40, -80, -40, 0])
    before, orientation = model.pose(current)
    target, _, _ = model.solve(current, .02, model.lower+.04, model.upper-.04)
    for _ in range(30):
        step = bounded_joint_step(current, target, .2/30)
        assert np.max(np.abs(step)) <= .2/30+1e-12
        current += step
        position, rotation = model.pose(current)
        assert np.linalg.norm(position[:2]-before[:2]) < .005
        assert np.rad2deg((orientation.inv()*rotation).magnitude()) < 1.0
    assert position[2]-before[2] == pytest.approx(.02, abs=1e-6)


def test_installed_trunk_height_corrects_mount_offset_without_changing_control_workspace(tmp_path):
    from scipy.spatial.transform import Rotation
    from wia.runtime.sim.esi_camera_geometry import calibrated_robot_kinematics
    from wia.runtime.sim.esi_trunk import R1TrunkController
    from wia.runtime.sim.esi_arm import R1ArmController

    nominal = kinematics(tmp_path)
    names = ['base_link']+[f'torso_link{i}' for i in range(1, 5)]
    links = {name: SimpleNamespace(prim_path='/robot/'+name) for name in names}
    mounts = {}
    for i, (origin, axis) in enumerate(zip(nominal.origins, nominal.axes), 1):
        transform = origin.copy()
        if i == 1:
            transform[2, 3] += .04165
        flip = np.eye(4)
        if axis[2] < 0:
            flip[:3, :3] = Rotation.from_euler('x', np.pi).as_matrix()
        parent = transform @ flip
        mounts[f'torso_joint{i}'] = SimpleNamespace(body0=links[names[i-1]].prim_path,
            body1=links[names[i]].prim_path, local_position_0=parent[:3, 3],
            local_orientation_0=Rotation.from_matrix(parent[:3, :3]).as_quat(),
            local_position_1=flip[:3, 3], local_orientation_1=Rotation.from_matrix(flip[:3, :3]).as_quat(), axis='Z')
    q = np.deg2rad([40, -80, -40, 0])
    robot = SimpleNamespace(urdf_path=tmp_path/'r1.urdf', base_footprint_link_name='base_link', links=links,
        joints=mounts, controllers={'trunk': SimpleNamespace(dof_idx=np.arange(4)),
            'arm_left': SimpleNamespace(dof_idx=np.array([6])), 'gripper_left': SimpleNamespace(dof_idx=np.array([4, 5]))},
        get_joint_positions=lambda: np.r_[q, .03, .03, 0.])
    measured, _ = calibrated_robot_kinematics(robot, joint_names=set(mounts))
    trunk = R1TrunkController.__new__(R1TrunkController)
    trunk.height_kinematics, trunk.kinematics = measured, nominal
    arm = R1ArmController.__new__(R1ArmController)
    arm.robot, arm.working_arm, arm.trunk_controller = robot, 'left', trunk
    arm.arm_group = 'arm_left'
    arm.ee_pose = lambda: {'base_position': [.5, .18, 1.08], 'base_quaternion_xyzw': [1., 0., 0., 0.]}
    state = arm.proprioception()
    assert state['trunk_height_m'] == pytest.approx(1.00362111018, abs=1e-7)
    assert state['trunk_control_height_m'] == pytest.approx(.96197111018, abs=1e-7)
    before = state['trunk_height_m']
    q, _, _ = nominal.solve(q, .01, nominal.lower+.04, nominal.upper-.04)
    after = arm.proprioception()
    assert after['trunk_height_m']-before == pytest.approx(.01, abs=1e-6)
    robot.scale = np.full(3, .5)
    scaled, _ = calibrated_robot_kinematics(robot, joint_names=set(mounts))
    encoders = dict(zip(mounts, q))
    full_pose = measured.frames(encoders)['torso_link4']
    scaled_pose = scaled.frames(encoders)['torso_link4']
    assert np.allclose(scaled_pose[:3, 3], full_pose[:3, 3] * .5)
    assert np.allclose(scaled_pose[:3, :3], full_pose[:3, :3])
    assert after['trunk_height_m']-after['trunk_control_height_m'] == pytest.approx(.04165, abs=1e-7)
    public = AgentRobotState.from_controller_state(ControllerState(base_pose={}, base_velocity=None, ee_pose=None,
        joint_state={}, gripper_state=None, controller_status={}, proprioception=after))
    assert public.trunk_control_height_m == after['trunk_control_height_m']
    assert public.sources['trunk_control_height_m'] == 'forward_kinematics'
    from core.observation_config import ObservationConfig
    hidden = AgentRobotState.from_controller_state(ControllerState(base_pose={}, base_velocity=None, ee_pose=None,
        joint_state={}, gripper_state=None, controller_status={}, proprioception=after), config=ObservationConfig(trunk_state=False))
    assert hidden.trunk_height_m is None and hidden.trunk_control_height_m is None


class RecoveringEnvironment(Environment):
    def __init__(self, *, confirmed=True):
        super().__init__()
        self.confirmed = confirmed

    async def execute(self, action):
        if self.calls:
            return await super().execute(action)
        self.calls.append(action)
        self.source.after_action(action_result(action, completed=False, timeout=False,
            requested={}, actual={}, message='IK target rejected', reason='ik_joint_jump',
            stop_steps=9, recovery_completed=None, ik_diagnostics={'joint_hold': True},
            stop_stability_verified=self.confirmed))
        feedback = self.source.last_action_feedback.to_dict()
        return {'ok': False, 'reason': 'ik_joint_jump', 'outcome_known': self.confirmed,
                'motion_stopped': feedback['motion_stopped'], 'results': [feedback]}


@pytest.mark.parametrize('confirmed', [True, False])
def test_recovery_requires_confirmed_stop_and_fresh_observation_and_uses_budget(tmp_path, confirmed):
    environment = RecoveringEnvironment(confirmed=confirmed)
    agent = Agent('{"action":"BASE_FWD"}', '{"action":"BASE_BACK"}', '{"action":"DONE"}')
    summary = asyncio.run(run_episode(environment, agent, tmp_path, max_steps=3))
    assert len(environment.calls) == (2 if confirmed else 1)
    assert summary['termination_reason'] == ('model_done' if confirmed else 'execution_error')
    if confirmed:
        assert 'BASE_FWD' in agent.inputs[1].available_actions
        assert agent.inputs[1].action_context['recovery']['reason'] == 'ik_joint_jump'
        assert agent.inputs[1].action_context['recovery']['episode_continues'] is True
        assert agent.inputs[1].observation.action_step == 1
        assert len(summary['recovered_execution_errors']) == 1
        assert len((tmp_path/'steps.jsonl').read_text().splitlines()) == 3
    else:
        assert not summary['recovered_execution_errors']
