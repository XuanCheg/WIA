"""Check gripper execution. Run .venv/bin/python -m pytest -q this file."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wia.runtime.sim.esi_arm import ArmControlConfig
from wia.runtime.sim.esi_gripper import R1ManipulationController
from harness_integration.test_esi_arm import ArmEnv, ArmRobot
from harness_integration.test_esi_base import JointController


class GripperEnv(ArmEnv):
    available_object = True
    finger_response = 1.0
    drop_on_step = None

    def step(self, actions):
        r = self.robot
        q_before = r.joints[r.gripper_control_idx['left']].copy()
        super().step(actions)
        target = actions[r.name][r.controller_action_idx['gripper_left']]
        r.joints[r.gripper_control_idx['left']] = q_before + self.finger_response * (target - q_before)
        if len(self.actions) == self.drop_on_step:
            r._ag_obj_in_hand['left'] = r._ag_obj_constraints['left'] = None
            self.available_object = False
        if np.all(target == 0.05):
            if r._ag_obj_in_hand['left'] is not None:
                r._ag_release_counter['left'] = (r._ag_release_counter['left'] or 0) + 1
                if r._ag_release_counter['left'] >= 2:
                    r._ag_obj_in_hand['left'] = r._ag_obj_constraints['left'] = None
                    r._ag_release_counter['left'] = None
        elif np.all(target == 0) and self.available_object:
            r._ag_obj_in_hand['left'] = SimpleNamespace(name='private_test_object')
            r._ag_obj_constraints['left'] = object()
        if r._ag_obj_in_hand['left'] is not None:
            r.joints[r.gripper_control_idx['left']] = 0.02


def make_manipulation_controller():
    robot = ArmRobot(0)
    robot.action_dim = 16
    robot.joints = np.r_[robot.joints, 0.03]
    robot.joints[11] = 0.03
    robot.controllers['gripper_left'] = JointController([11, 15])
    robot.controller_action_idx['gripper_left'] = np.array([4, 15])
    robot.gripper_control_idx = {'left': np.array([11, 15])}
    robot.control_limits['position'] = (np.full(16, -10.0), np.full(16, 10.0))
    robot.control_limits['velocity'] = (np.full(16, -5.0), np.full(16, 5.0))
    robot.control_limits['position'][0][[11, 15]] = 0
    robot.control_limits['position'][1][[11, 15]] = 0.05
    robot.finger_joint_names = {'left': ['finger_a', 'finger_b']}
    robot.joints_by_name = {name: SimpleNamespace(joint_type='JOINT_PRISMATIC') for name in robot.finger_joint_names['left']}
    # Existing fake uses numeric joints as readback, while native exposes named
    # JointPrim objects separately. Preserve both through a small array wrapper.
    class NamedJointArray(np.ndarray):
        def __getitem__(self, key):
            return robot.joints_by_name[key] if isinstance(key, str) else super().__getitem__(key)
    robot.joints = robot.joints.view(NamedJointArray)
    robot.grasping_mode = 'assisted'
    robot._ag_obj_in_hand = {'left': None}
    robot._ag_obj_constraints = {'left': None}
    robot._ag_release_counter = {'left': None}
    robot.is_grasping = lambda arm: (1 if robot._ag_obj_in_hand[arm] and robot._ag_release_counter[arm] is None else -1)
    env = GripperEnv(robot)
    controller = R1ManipulationController(SimpleNamespace(robot=robot, env=env),
        arm_config=replace(ArmControlConfig.load(), initial_trunk_joint_positions_rad=None))
    return controller, robot, env


def test_grasp_records_attachment_and_repeats_do_not_reverse_fingers():
    c, r, env = make_manipulation_controller()
    assert c.execute('RELEASE')['completed']
    before = c.ee_pose()
    grasp = c.execute('GRASP')
    assert grasp['completed'] and grasp['gripper_after']['holding'] is True
    assert grasp['control_steps'] > 1 and grasp['grasp_mode'] == 'assisted'
    repeated = c.execute('GRASP')
    assert repeated['noop'] and repeated['reason'] == 'already_holding'
    assert r._ag_obj_in_hand['left'] is not None
    assert c.execute('MV_UP')['completed'] and c.execute('BASE_RIGHT')['completed']
    release = c.execute('RELEASE')
    assert release['completed'] and release['gripper_after']['gripper_open']
    assert release['debug_gripper_after']['object_name'] is None
    assert r._ag_obj_in_hand['left'] is None and r._ag_obj_constraints['left'] is None
    assert c.execute('RELEASE')['reason'] == 'already_open'
    assert 'arm_hold_max_error_m' not in grasp
    assert np.allclose(grasp['ee_pose_after']['base_position'], before['base_position'])
    # The working arm/fingers are never assumed to occupy contiguous slices.
    assert all(a[r.controller_action_idx['gripper_right']] == 1.6 for a in env.actions)


def test_empty_close_finishes_without_certifying_attachment():
    c, r, env = make_manipulation_controller()
    env.available_object = False
    result = c.execute('GRASP')
    assert result['completed'] and result['reason'] == 'completed' and not result['timeout']
    assert result['gripper_after']['gripper_closed'] and result['gripper_after']['holding'] is False
    repeat = c.execute('GRASP')
    assert repeat['noop'] and repeat['completed'] and repeat['reason'] == 'completed'
    assert c.execute('RELEASE')['completed']


def test_blocked_close_returns_measurements_and_allows_the_next_motion():
    c, r, env = make_manipulation_controller()
    env.available_object = False
    env.finger_response = 0.0
    c.gripper_config = replace(c.gripper_config, timeout_wall_s=1e-9)
    result = c.execute('GRASP')
    assert result['completed'] and not result['timeout']
    assert result['gripper_after']['holding'] is False
    assert not result['gripper_after']['gripper_closed']
    assert result['gripper_after']['gripper_width_m'] == pytest.approx(.06)
    assert result['debug_gripper_after']['constraint_active'] is False
    assert result['sim_duration_s'] < .6
    assert np.all(c.hold_action[c.indices['gripper_left']] == c.closed_target)
    assert c.execute('MV_UP')['completed']


def test_native_unknown_and_false_grasp_values_are_not_cast_to_bool():
    c, r, _ = make_manipulation_controller()
    r.is_grasping = lambda arm: 0
    assert c.proprioception()['holding'] is None
    r.is_grasping = lambda arm: -1
    assert c.proprioception()['holding'] is False


def test_lost_grasp_stops_motion_and_reports_failure():
    c, r, env = make_manipulation_controller()
    assert c.execute('GRASP')['completed']
    env.drop_on_step = len(env.actions) + 1
    result = c.execute('BASE_FWD')
    assert not result['completed'] and result['reason'] == 'grasp_lost'
    assert result['gripper_after']['holding'] is False and result['actual_m'] < 0.25


def test_cancelled_gripper_freezes_fingers_and_reports_cancel():
    c, r, env = make_manipulation_controller()
    env.available_object = False
    env.finger_response = 0.1
    result = c.execute('GRASP', should_cancel=lambda: len(env.actions) >= 2)
    assert not result['completed'] and result['cancelled'] and result['reason'] == 'cancelled'
    assert np.allclose(c.hold_action[c.indices['gripper_left']], r.get_joint_positions()[c.gripper_dofs])


def test_gripper_owner_thread_is_enforced():
    import threading
    c, _, _ = make_manipulation_controller()
    errors = []
    def worker():
        try:
            c.gripper_state()
        except RuntimeError as error:
            errors.append(str(error))
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert errors and 'main thread' in errors[0]


def test_gripper_exception_freezes_unfinished_target():
    c, r, env = make_manipulation_controller()
    env.available_object = False
    env.finger_response = 0.1
    env.fail_step = 2
    with pytest.raises(RuntimeError, match='simulator failure'):
        c.execute('GRASP')
    assert np.allclose(c.hold_action[c.indices['gripper_left']], r.get_joint_positions()[c.gripper_dofs])
