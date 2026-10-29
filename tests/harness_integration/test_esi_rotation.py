"""Wrist-frame signs, bounded targets and failures using shuffled R1 slices."""
from dataclasses import replace
import threading

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.action_units import ROTATE_ATOMS
from wia.runtime.sim.esi_rotation import RotationControlConfig
from harness_integration.test_esi_arm import make_arm_controller


def rotation_controller(yaw=0, **kwargs):
    controller, robot, env = make_arm_controller(yaw, **kwargs)
    controller.rotation_config = RotationControlConfig.load()
    return controller, robot, env


@pytest.mark.parametrize("yaw", [0, 90])
@pytest.mark.parametrize("token,sign", [("ROTATE_CW", 1), ("ROTATE_CCW", -1)])
def test_rotation_is_local_z_about_fixed_tcp_with_one_bounded_target(yaw, token, sign):
    c, r, env = rotation_controller(yaw)
    # Include an arbitrarily tilted wrist and pi-crossing quaternion branch.
    r.ee_rot = Rotation.from_euler("xyz", [179, 27, 37], degrees=True)
    c._set_arm_target(r.ee, r.ee_rot.as_quat())
    before = c.ee_pose()
    initial = r.ee_rot
    result = c.execute(token)
    assert result["completed"] and result["frame"] == "eef_local"
    expected = initial * Rotation.from_euler("z", sign * 15, degrees=True)
    assert np.rad2deg((expected.inv() * r.ee_rot).magnitude()) <= c.rotation_config.rotation_tolerance_deg
    assert np.allclose(r.ee, before["base_position"])
    assert result["requested_deg"] == pytest.approx(15) and result["actual_deg"] == pytest.approx(15, abs=1)
    assert "requested_m" not in result and result["position_error_m"] < 1e-7
    assert "position_hold_error_m" not in result
    previous = initial
    for action in env.actions:
        commanded = Rotation.from_rotvec(action[r.controller_action_idx['arm_left']][3:])
        # The fake has a measured 50% response, so commanded-to-measured angles
        # are independently checked against the configured angular speed.
        remaining = (previous.inv() * commanded).as_rotvec()
        assert np.rad2deg(np.linalg.norm(remaining)) <= 15 * c.dt + 1e-4
        previous = previous * Rotation.from_rotvec(env.arm_response * remaining)
    assert set(env.threads) == {threading.get_ident()}
    assert 'inactive_joint_max_drift' not in result


def test_rotated_pose_is_held_by_translation_base_and_idle_and_reverses():
    c, r, _ = rotation_controller()
    initial = r.ee_rot
    for _ in range(3):
        assert c.execute("ROTATE_CW")["completed"]
    rotated = r.ee_rot
    assert np.rad2deg((initial.inv() * rotated).magnitude()) == pytest.approx(45, abs=3)
    for token in ("MV_UP", "BASE_BACK", "BASE_TURN_LEFT", "MV_DOWN", "BASE_FWD"):
        assert c.execute(token)["completed"]
        assert np.rad2deg((rotated.inv() * r.ee_rot).magnitude()) < 1e-6
    for _ in range(3):
        assert c.execute("ROTATE_CCW")["completed"]
    assert np.rad2deg((initial.inv() * r.ee_rot).magnitude()) < 0.1


@pytest.mark.parametrize("interrupt", ["timeout", "cancel", "exception"])
def test_interrupted_rotation_freezes_measured_pose_and_does_not_resume(interrupt):
    c, r, env = rotation_controller(timeout_sim_s=0.1)
    if interrupt == "timeout":
        env.arm_response = 0
        result = c.execute("ROTATE_CW")
        assert result['timeout'] and not result['completed']
    elif interrupt == "cancel":
        result = c.execute("ROTATE_CW", should_cancel=lambda: len(env.actions) >= 2)
        assert result['cancelled'] and not result['completed']
    else:
        env.fail_step = 2
        with pytest.raises(RuntimeError, match="simulator failure"):
            c.execute("ROTATE_CW")
    stopped = r.ee_rot
    env.arm_response = 0.5
    for _ in range(20):
        c.step_base_debug()
    assert np.rad2deg((stopped.inv() * r.ee_rot).magnitude()) < 1e-4


def test_blocked_rotation_reports_failure_without_waiting_for_whole_timeout():
    c, _, env = rotation_controller()
    env.arm_response = 0
    result = c.execute("ROTATE_CCW")
    assert not result['completed'] and result['reason'] == 'arm_motion_blocked'
    assert result['sim_duration_s'] < c.arm_config.timeout_sim_s


@pytest.mark.parametrize("kind", ["base", "inactive"])
def test_rotation_ignores_unrelated_component_offsets(kind):
    c, r, env = rotation_controller()
    original = env.step

    def offset(actions):
        original(actions)
        if kind == 'base':
            r.xy[0] += 0.03
        else:
            r.joints[12] += 0.03

    env.step = offset
    result = c.execute('ROTATE_CW')
    assert result['completed'] and result['reason'] == 'completed'


@pytest.mark.parametrize("kind", ["position", "orientation"])
def test_rotation_preserves_actual_step_motion_limits(kind):
    c, r, env = rotation_controller()
    original = env.step

    def excessive_step(actions):
        original(actions)
        if kind == 'position':
            r.ee[0] += 0.02
        else:
            r.ee_rot = r.ee_rot * Rotation.from_euler('x', 3, degrees=True)

    env.step = excessive_step
    result = c.execute('ROTATE_CW')
    assert not result['completed'] and result['reason'] == 'arm_step_limit'


def test_opt_in_and_invalid_actions_reject_before_physics_and_enforce_owner():
    c, _, env = make_arm_controller()
    for token in ROTATE_ATOMS:
        with pytest.raises(ValueError, match='rejects'):
            c.execute(token)
    c.rotation_config = RotationControlConfig.load()
    for token in ('RT_YAW_CW', 'ROTATE_CW(90)', 'ARM_ROTATE_DEBUG'):
        with pytest.raises(ValueError, match='rejects'):
            c.execute(token)
    for direction, angle in (('left', 15), ('cw', 0), ('ccw', 180), ('cw', np.nan)):
        with pytest.raises(ValueError):
            c.rotate_ee_debug(direction, angle)
    errors = []

    def worker():
        try:
            c.execute('ROTATE_CW')
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert errors and 'main thread' in errors[0] and not env.actions


@pytest.mark.parametrize("changes", [{'rotation_deg': 180}, {'rotation_deg': 1}, {'rotation_deg': True},
                                     {'max_rotation_speed_deg_s': float('inf')}, {'max_step_rotation_deg': 0}])
def test_rotation_config_requires_finite_bounded_motion(changes):
    with pytest.raises(ValueError):
        replace(RotationControlConfig.load(), **changes)


@pytest.mark.parametrize("axis,kind", [(0, "ROLL"), (1, "PITCH")])
@pytest.mark.parametrize("sign,suffix", [(1, "POS"), (-1, "NEG")])
def test_local_tilt_metric_holds_tcp_and_reports_correct_axis(axis, kind, sign, suffix):
    c, r, env = rotation_controller()
    r.ee_rot = Rotation.from_euler("xyz", [75, 27, -39], degrees=True)
    initial, position = r.ee_rot, r.ee.copy()
    c._set_arm_target(position, initial.as_quat())
    token = f"ROTATE_{kind}_{suffix}"
    assert token in c.capabilities.available_actions
    result = c.execute_metric(token, 12)
    expected = initial * Rotation.from_rotvec(np.eye(3)[axis] * np.deg2rad(sign * 12))
    assert result["completed"]
    assert np.rad2deg((expected.inv() * r.ee_rot).magnitude()) < 1.01
    assert np.allclose(r.ee, position)
    assert result["rotation_axis"] == "+" + "XYZ"[axis]
    assert result["signed_actual_deg"] == pytest.approx(sign * 12, abs=1)
    assert result["actual_deg"] == pytest.approx(12, abs=1)
    assert result["max_step_rotation_deg"] <= c.rotation_config.max_step_rotation_deg


def test_tilt_requires_explicit_backend_capability():
    from core.action_interface import BackendCapabilities
    from core.action_compiler import ParameterizedAction, PrimitiveResolution
    assert "ROTATE_PITCH_POS" not in BackendCapabilities(ee_rotation=True).available_actions
    caps = BackendCapabilities(ee_tilt=True, right_ee_tilt=True)
    for token in caps.available_actions:
        assert ParameterizedAction.from_payload({"action": token, "angle_deg": 10}, caps.available_actions).angle_deg == 10
        assert PrimitiveResolution(arm_rotation_deg=5).step_for(token) == 5
