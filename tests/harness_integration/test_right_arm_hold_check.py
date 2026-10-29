"""Verify all-component offset freedom; run: python -m pytest -q this file."""
from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.action_units import BASE_ATOMS
from wia.runtime.sim.esi_base import BaseControlConfig
from wia.runtime.sim.esi_rotation import RotationControlConfig
from wia.runtime.sim.esi_trunk import R1TrunkController, TrunkControlConfig
from harness_integration.test_esi_base import make_controller
from harness_integration.test_esi_arm import make_arm_controller
from harness_integration.test_esi_gripper import make_manipulation_controller


OFFSET = 0.0067154765
REMOVED_FIELDS = {'inactive_joint_max_drift', 'base_max_drift_m', 'base_max_yaw_drift_deg',
    'arm_hold_max_error_m', 'arm_hold_max_orientation_error_deg', 'arm_joint_max_drift_rad',
    'max_horizontal_drift_m', 'max_path_error_m', 'max_rotation_path_error_deg', 'right_arm_hold_check'}


def persistent_offset(robot, env, group="arm_right"):
    """Offset measured passive joints while leaving every motor hold intact."""
    dofs = robot.controllers[group].dof_idx
    robot.joints[dofs] += OFFSET
    original = env.step

    def step(actions):
        original(actions)
        robot.joints[dofs] += OFFSET

    env.step = step


@pytest.mark.parametrize("group", ['arm_right', 'arm_left', 'trunk', 'gripper_left', 'gripper_right'])
@pytest.mark.parametrize("action", BASE_ATOMS)
def test_base_accepts_passive_offsets_preserves_holds_and_omits_policy_telemetry(group, action):
    c, r, env = make_controller()
    hold = c.hold_action[c.indices[group]].copy()
    target = c.inactive_targets[group].copy()
    persistent_offset(r, env, group)
    result = c.execute(action)
    assert result['completed'] and result['reason'] == 'completed'
    assert not REMOVED_FIELDS.intersection(result)
    assert np.array_equal(c.inactive_targets[group], target)
    assert all(np.array_equal(a[c.indices[group]], hold) for a in env.actions)
    assert any(np.any(a[c.indices['base']]) for a in env.actions)


@pytest.mark.parametrize('group', ['arm_right', 'gripper_right', 'trunk'])
@pytest.mark.parametrize('action', ['MV_FWD', 'ROTATE_CW', 'GRASP', 'RELEASE'])
def test_manipulation_accepts_unrelated_joint_offsets(group, action):
    c, r, env = make_manipulation_controller()
    c.rotation_config = RotationControlConfig.load()
    persistent_offset(r, env, group)
    result = c.execute(action)
    assert result['completed'] and not REMOVED_FIELDS.intersection(result)


@pytest.mark.parametrize('group', ['arm_right', 'gripper_right', 'gripper_left', 'arm_left', 'base'])
def test_trunk_preserves_motor_hold_without_passive_offset_rejection(group):
    c, r, env = make_arm_controller()
    # A linear trunk fixture isolates policy from the native URDF solver.
    r.joints[4] = 0.9
    c.hold_action[c.indices['trunk']] = r.joints[[3, 4]]
    c.inactive_targets['trunk'] = r.joints[[3, 4]].copy()
    trunk = R1TrunkController.__new__(R1TrunkController)
    trunk.owner, trunk.config = c, TrunkControlConfig.load('configs/esi_trunk.json')
    trunk.lower, trunk.upper = np.full(2, -10.), np.full(2, 10.)
    trunk.joints = lambda: r.joints[[3, 4]].copy()
    trunk.kinematics = SimpleNamespace(
        pose=lambda q: (np.array([0., 0., q[1]]), Rotation.identity()),
        solve=lambda q, delta, lower, upper: (q + [0., delta], np.array([0., 0., q[1]+delta]), None))
    arm_joints = r.joints[r.arm_control_idx['left']].copy()
    original = env.step
    explicit_holds = []

    def joint_hold(actions):
        explicit_holds.append(c.ik_guard.explicit_joint_hold.copy())
        original(actions)
        r.joints[r.arm_control_idx['left']] = arm_joints
        if group == 'base':
            r.xy[0] += 0.03

    env.step = joint_hold
    if group != 'base':
        persistent_offset(r, env, group)
    result = trunk.execute('TRUNK_DOWN')
    assert result['completed'] and not REMOVED_FIELDS.intersection(result)
    expected = arm_joints + (OFFSET if group == 'arm_left' else 0.)
    assert explicit_holds and all(np.array_equal(value, expected) for value in explicit_holds)


@pytest.mark.parametrize('action', ['GRASP', 'RELEASE'])
def test_gripper_ignores_working_arm_and_base_offsets(action):
    c, r, env = make_manipulation_controller()
    original = env.step

    def displaced(actions):
        original(actions)
        r.ee[0] += .01
        r.ee_rot = r.ee_rot * Rotation.from_euler('x', 4, degrees=True)
        r.xy[0] += .03

    env.step = displaced
    result = c.execute(action)
    assert result['completed'] and not REMOVED_FIELDS.intersection(result)


def test_old_json_policy_keys_are_inert_and_absent_from_active_config(tmp_path):
    values = vars(BaseControlConfig.load()).copy()
    values.update(inactive_joint_tolerance=0., right_arm_hold_check='old-policy')
    path = tmp_path / 'old.json'
    path.write_text(json.dumps(values))
    config = BaseControlConfig.load(path)
    assert vars(config) == vars(BaseControlConfig.load())
    c, _, _ = make_controller()
    assert not hasattr(c, 'inactive_drift')
    assert not hasattr(c, 'max_guarded_inactive_drift')


def test_passive_offset_policy_does_not_disable_native_hard_limit_stop_verification():
    c, r, env = make_controller(progress_timeout_sim_s=.2)
    env.response = 0
    low, high = np.full(r.action_dim, -10.), np.full(r.action_dim, 10.)
    dofs = r.controllers['arm_right'].dof_idx
    high[dofs] = r.joints[dofs] + .003
    r.control_limits = {'position': (low, high)}
    persistent_offset(r, env)
    result = c.execute('BASE_BACK')
    assert result['reason'] == 'base_motion_blocked'
    c.verify_stopped_result(result)
    assert result['stop_stability']['hard_limits_ok'] is False
    assert result['stop_stability_verified'] is False


@pytest.mark.parametrize('rotation', [False, True])
def test_temporary_off_axis_pose_error_does_not_reject_reachable_command(rotation):
    c, r, env = make_arm_controller()
    c.arm_config = replace(c.arm_config, max_step_translation_m=.05)
    c.rotation_config = replace(RotationControlConfig.load(), max_step_rotation_deg=10.)
    original = env.step

    def transient(actions):
        original(actions)
        if len(env.actions) == 1:
            r.ee[1] += .01
            r.ee_rot = r.ee_rot * Rotation.from_euler('x', 3, degrees=True)

    env.step = transient
    result = c.execute('ROTATE_CW' if rotation else 'MV_FWD')
    assert result['completed'] and not REMOVED_FIELDS.intersection(result)
