"""Native callback regressions; run python -m pytest -q tests/test_ik_rejection_native.py."""
from dataclasses import replace

import numpy as np
import pytest

from wia.runtime.sim.esi_rotation import RotationControlConfig
from harness_integration.test_esi_arm import make_arm_controller


def native_arm_fixture(fault, *, reject_step=3, contact_blocks_return=False):
    """Consume guarded native joint targets in the existing deterministic arm fixture."""
    controller, robot, env = make_arm_controller(max_ik_joint_step_rad=.08)
    controller.rotation_config = RotationControlConfig.load()
    dofs = robot.arm_control_idx["left"]
    robot.joints[dofs] = np.r_[robot.ee, robot.ee_rot.as_rotvec()]
    env.arm_response = 1.0
    original_step = env.step
    history, proposals = [], []
    blocked_at = None

    def native_solve(goal, control):
        current = control["joint_position"][dofs]
        candidate = current + 0.5 * (goal["target"] - current)
        if len(proposals) + 1 == reject_step:
            candidate[0] = (current[0] + 1 if fault == "ik_joint_jump" else
                            11 if fault == "arm_joint_limit" else np.nan)
        proposals.append(candidate.copy())
        return candidate

    def native_step(actions):
        nonlocal blocked_at
        action = actions[robot.name].copy()
        control = robot.get_control_dict()
        control["eef_left_jacobian_relative"] = np.zeros((6, robot.action_dim))
        control["eef_left_jacobian_relative"][:, dofs] = np.eye(6)
        current = robot.joints[dofs].copy()
        deployed = robot.controllers["arm_left"].compute_control(
            {"target": action[controller.indices["arm_left"]]}, control)
        if controller.ik_guard.fault is not None and blocked_at is None:
            blocked_at = current[0]
        if contact_blocks_return and blocked_at is not None:
            deployed[0] = max(deployed[0], blocked_at)
        history.append({"before": current, "deployed": deployed.copy(),
                        "cartesian": action[controller.indices["arm_left"]].copy()})
        action[controller.indices["arm_left"]] = deployed
        original_step({robot.name: action})

    controller.ik_guard.original_compute = native_solve
    env.step = native_step
    return controller, robot, env, history, proposals


@pytest.mark.parametrize("fault", ["ik_joint_jump", "arm_joint_limit"])
@pytest.mark.parametrize("token,metric", [
    ("MV_FWD", False), ("MV_FWD", True), ("ROTATE_CW", False), ("ROTATE_CW", True),
])
def test_native_rejection_after_progress_holds_current_for_all_arm_entrypoints(fault, token, metric):
    controller, robot, env, history, proposals = native_arm_fixture(fault)
    start = robot.joints[robot.arm_control_idx["left"]].copy()
    inactive = {name: value.copy() for name, value in controller.inactive_targets.items()}
    result = (controller.execute_metric(token, 5.0 if "ROTATE" in token else 0.03)
              if metric else controller.execute(token))
    rejected = history[2]["before"]
    assert np.any(np.abs(rejected - start) > 1e-5)
    assert np.all(start != 0)
    assert result["reason"] == fault and not result["completed"] and not result["timeout"]
    assert len(proposals) == 3
    assert np.max(np.abs(proposals[-1] - rejected)) > 0.08
    for entry in history[2:]:
        np.testing.assert_allclose(entry["deployed"], rejected, atol=1e-6)
    np.testing.assert_allclose(controller.ik_guard.hold, rejected, atol=1e-6)
    assert controller.ik_guard.fault == fault
    assert result["recovery_steps"] == 0
    assert result["recovery_completed"] is None
    assert result["recovery_max_joint_error_rad"] is None
    assert result["control_steps"] == 3 + controller.config.stop_steps
    assert result["stop_steps"] == controller.config.stop_steps
    np.testing.assert_allclose(controller._arm_target_pose["base_position"], robot.ee, atol=1e-6)
    if "ROTATE" not in token:
        assert not np.allclose(controller._arm_target_pose["base_position"],
                               result["target_base_position"])
    for action in env.actions:
        assert np.all(action[controller.indices["base"]] == 0)
        for name, target in inactive.items():
            np.testing.assert_allclose(action[controller.indices[name]], target)


@pytest.mark.parametrize("fault", ["ik_joint_jump", "arm_joint_limit"])
def test_contact_blocking_old_posture_does_not_launch_macro_start_rollback(fault):
    controller, robot, env, history, _ = native_arm_fixture(
        fault, reject_step=9, contact_blocks_return=True)
    # A much smaller tolerance makes the distinct old posture require a return
    # under legacy recovery, while the fixture blocks that contact-like return.
    controller.arm_config = replace(controller.arm_config, ik_recovery_joint_tolerance_rad=1e-6)
    result = controller.execute_metric("MV_FWD", 0.04)
    rejected = history[8]["before"]
    assert rejected[0] - controller.ik_guard.recovery[0] > 0.005
    assert result["reason"] == fault
    assert result["recovery_steps"] == 0 and result["recovery_completed"] is None
    assert len(env.actions) == 9 + controller.config.stop_steps
    np.testing.assert_allclose(robot.joints[robot.arm_control_idx["left"]], rejected, atol=1e-6)


@pytest.mark.parametrize("fault", ["ik_joint_jump", "arm_joint_limit"])
def test_next_fresh_retreat_clears_rejection_and_completes(fault):
    controller, robot, _, history, _ = native_arm_fixture(fault)
    assert controller.execute("MV_FWD")["reason"] == fault
    stopped = robot.ee.copy()
    old_count = len(history)
    result = controller.execute("MV_BACK")
    assert result["completed"] and result["reason"] == "completed"
    assert controller.ik_guard.fault is None and controller.ik_guard.hold is None
    assert result["actual_m"] > 0.015
    assert robot.ee[0] < stopped[0]
    assert np.any(history[old_count]["deployed"] != history[old_count]["before"])


def test_nonfinite_native_fault_still_attempts_legacy_macro_start_return():
    controller, _, _, _, _ = native_arm_fixture("ik_nonfinite", reject_step=9)
    controller.ik_guard.max_step[:] = 0.001
    controller.arm_config = replace(controller.arm_config, ik_recovery_joint_tolerance_rad=1e-6)
    result = controller.execute_metric("MV_FWD", 0.04)
    assert result["reason"] == "ik_nonfinite"
    assert result["recovery_steps"] > 0
    assert result["recovery_completed"] is True
