"""Guard regressions; run with python -m pytest -q tests/test_esi_ik.py."""
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from wia.runtime.sim.esi_arm import ArmControlConfig
from wia.runtime.sim.esi_ik import NativeIKGuard
from harness_integration.test_esi_arm import make_arm_controller


def guarded_solver(q=None, proposal=None, limits=4.0, jump_limit=.08):
    q = np.zeros(6) if q is None else np.asarray(q, dtype=float)
    solver = SimpleNamespace(proposal=np.zeros(6) if proposal is None else np.asarray(proposal, dtype=float))
    ctrl = SimpleNamespace(dof_idx=np.arange(6), task_name="eef_left",
                           compute_control=lambda goal, state: solver.proposal.copy())
    motor_limits = {"position": (np.full(6, -limits), np.full(6, limits)),
                    "velocity": (np.full(6, -2.0), np.full(6, 2.0))}
    guard = NativeIKGuard(ctrl, motor_limits, 1 / 120, replace(ArmControlConfig.load(), max_ik_joint_step_rad=jump_limit))
    state = {"joint_position": q, "eef_left_jacobian_relative": np.eye(6)}
    return guard, ctrl, solver, state


def test_singular_solution_cannot_deploy_large_joint_jump():
    guard, ctrl, solver, state = guarded_solver(proposal=[1, 0, 0, 0, 0, 0])
    state["eef_left_jacobian_relative"][0, 0] = 0.001
    # Native pinv would turn a 1 mm error into a 1 rad correction.
    assert np.linalg.pinv(state["eef_left_jacobian_relative"])[0, 0] * 0.001 == pytest.approx(1)
    deployed = ctrl.compute_control({}, state)
    assert np.array_equal(deployed, state["joint_position"])
    assert guard.fault == "ik_joint_jump"
    assert guard.diagnostics()["min_jacobian_singular_value"] == 0.001
    assert guard.diagnostics()["max_native_joint_step_rad"] == 1
    solver.proposal[:] = 0.001
    state["joint_position"][:] = 0.02
    # A fault latches its original joint target even while physics settles.
    assert np.allclose(ctrl.compute_control({}, state), 0.015)
    guard.freeze(state["joint_position"])
    assert np.array_equal(ctrl.compute_control({}, state), state["joint_position"])


def test_joint_speed_is_bounded_at_physics_frequency_without_changing_direction():
    guard, ctrl, _, state = guarded_solver(proposal=[0.03, 0.015, 0, 0, 0, 0])
    output = ctrl.compute_control({}, state)
    assert guard.fault is None
    assert np.max(np.abs(output - state["joint_position"])) == pytest.approx(0.6 / 120)
    assert output[0] / output[1] == pytest.approx(2)


def test_near_limit_outward_motion_is_refused_but_retreat_is_allowed():
    guard, ctrl, solver, state = guarded_solver(q=[0.99, 0, 0, 0, 0, 0],
                                               proposal=[1, 0, 0, 0, 0, 0], limits=1)
    assert np.array_equal(ctrl.compute_control({}, state), state["joint_position"])
    assert guard.fault == "arm_joint_limit"
    guard.reset()
    solver.proposal[0] = 0.98
    output = ctrl.compute_control({}, state)
    assert guard.fault is None and output[0] < state["joint_position"][0]


def test_nonfinite_solver_result_holds_and_next_macro_can_retry():
    guard, ctrl, solver, state = guarded_solver(proposal=[float("nan"), 0, 0, 0, 0, 0])
    assert np.array_equal(ctrl.compute_control({}, state), state["joint_position"])
    assert guard.fault == "ik_nonfinite"
    guard.reset()
    solver.proposal[:] = 0.001
    assert np.allclose(ctrl.compute_control({}, state), 0.001)
    assert guard.fault is None


def test_fault_hold_preserves_native_backend_and_dtype_after_robot_readback():
    guard, ctrl, _, state = guarded_solver(proposal=[1, 0, 0, 0, 0, 0])
    state["joint_position"] = np.zeros(6, dtype=np.float32)
    ctrl.compute_control({}, state)
    guard.freeze(np.full(6, 0.02, dtype=np.float64))
    state["joint_position"][:] = 0.02
    output = ctrl.compute_control({}, state)
    assert isinstance(output, np.ndarray) and output.dtype == np.float32
    assert np.allclose(output, 0.02)


@pytest.mark.parametrize("fault", ["ik_joint_jump", "arm_joint_limit"])
def test_rejected_solution_holds_current_pose_instead_of_macro_start(fault):
    guard, ctrl, solver, state = guarded_solver(q=[0.04, 0, 0, 0, 0, 0],
                                               proposal=[1 if fault == "ik_joint_jump" else 5, 0, 0, 0, 0, 0])
    guard.reset(recovery_joint_position=np.full(6, -0.2))
    rejected_at = state["joint_position"].copy()
    for _ in range(12):
        output = ctrl.compute_control({}, state)
        assert np.array_equal(output, rejected_at)
        state["joint_position"] = output
    assert guard.fault == fault
    state["joint_position"] += 0.001
    guard.freeze(state["joint_position"], preserve_recovery=True)
    assert np.array_equal(ctrl.compute_control({}, state), state["joint_position"])
    guard.reset()
    solver.proposal[:] = state["joint_position"] - 0.001
    assert np.allclose(ctrl.compute_control({}, state), solver.proposal)
    assert guard.fault is None


@pytest.mark.parametrize("failure", ["ik_nonfinite", "ik_solver_failure"])
def test_other_native_faults_retain_bounded_macro_start_recovery(failure):
    guard, ctrl, _, state = guarded_solver(q=np.full(6, 0.04), proposal=np.full(6, np.nan))
    if failure == "ik_solver_failure":
        def fail(goal, control):
            raise RuntimeError("native solve failed")
        guard.original_compute = fail
    guard.reset(recovery_joint_position=np.full(6, -0.02))
    output = ctrl.compute_control({}, state)
    assert guard.fault == failure
    assert np.allclose(guard.hold, -0.02)
    assert np.allclose(output, 0.035)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("fault", ["ik_joint_jump", "arm_joint_limit"])
def test_rejection_freeze_keeps_native_dtype_with_distinct_readback_backend(backend, fault):
    guard, ctrl, _, state = guarded_solver(q=np.full(6, 0.04),
        proposal=[1 if fault == "ik_joint_jump" else 5, 0, 0, 0, 0, 0])
    state["joint_position"] = state["joint_position"].astype(np.float32)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state["joint_position"] = torch.tensor(state["joint_position"])
        readback = np.full(6, 0.05, dtype=np.float64)
    else:
        torch = pytest.importorskip("torch")
        readback = torch.full((6,), 0.05, dtype=torch.float64)
    guard.reset(recovery_joint_position=np.full(6, -0.2))
    output = ctrl.compute_control({}, state)
    assert guard.fault == fault and output.dtype == state["joint_position"].dtype
    guard.freeze(readback, preserve_recovery=True)
    assert type(guard.hold) is type(state["joint_position"])
    assert guard.hold.dtype == state["joint_position"].dtype
    np.testing.assert_allclose(guard.hold, 0.05)
    state["joint_position"][:] = 0.05
    output = ctrl.compute_control({}, state)
    assert type(output) is type(state["joint_position"])
    assert output.dtype == state["joint_position"].dtype
    np.testing.assert_allclose(output, 0.05)


def test_excessive_measured_step_stops_before_timeout():
    controller, robot, env = make_arm_controller()
    original_step = env.step

    def disturbed(action):
        original_step(action)
        if len(env.actions) == 1:
            robot.ee[1] += 0.02

    env.step = disturbed
    result = controller.execute("MV_FWD")
    assert not result["completed"] and not result["timeout"]
    assert result["reason"] == "arm_step_limit"
    assert result["max_step_motion_m"] > controller.arm_config.max_step_translation_m
    assert result["control_steps"] == 1 + controller.config.stop_steps
    stopped = robot.ee.copy()
    controller.stop()
    assert np.allclose(robot.ee, stopped)


def test_blocked_arm_stops_without_spending_full_ik_timeout():
    controller, _, env = make_arm_controller()
    env.arm_response = 0
    result = controller.execute("MV_FWD")
    assert result["reason"] == "arm_motion_blocked" and not result["timeout"]
    assert not result["completed"] and result["actual_m"] == 0
    assert result["sim_duration_s"] < controller.arm_config.timeout_sim_s


@pytest.mark.parametrize("proposal,accepted", [(1.24, True), (5.0, True), (5.01, False)])
def test_requested_five_radian_threshold_preserves_deployed_speed_limit(proposal, accepted):
    assert ArmControlConfig.load().max_ik_joint_step_rad == 5.0
    guard, ctrl, _, state = guarded_solver(proposal=[proposal, 0, 0, 0, 0, 0], limits=6., jump_limit=5.)
    deployed = ctrl.compute_control({}, state)
    if accepted:
        assert guard.fault is None
        assert deployed[0] == pytest.approx(.6 / 120)
    else:
        assert guard.fault == 'ik_joint_jump'
        assert np.array_equal(deployed, state['joint_position'])


def test_damped_native_pose_retreat_is_bounded_and_reduces_error():
    guard, ctrl, solver, state = guarded_solver(proposal=[2, 0, 0, 0, 0, 0])
    state.update(eef_left_pos_relative=np.zeros(3), eef_left_quat_relative=np.array([0., 0., 0., 1.]))
    state['eef_left_jacobian_relative'] = np.diag([.001, 1, 1, 1, 1, 1])
    goal = {'target_pos': np.array([-.0001, 0, 0]), 'target_ori_mat': np.eye(3)}
    result = ctrl.compute_control(goal, state)
    assert guard.fault is None and guard.damped_recovery_steps == 1
    assert result[0] < 0 and np.max(np.abs(result)) <= guard.max_step.max() + 1e-9
    assert abs(goal['target_pos'][0] - .001*result[0]) < abs(goal['target_pos'][0])


def test_damped_soft_limit_allows_inward_but_never_outward_step():
    q = np.array([3.99, 0, 0, 0, 0, 0])
    guard, ctrl, solver, state = guarded_solver(q=q, proposal=[4, 0, 0, 0, 0, 0])
    state.update(eef_left_pos_relative=np.zeros(3), eef_left_quat_relative=np.array([0., 0., 0., 1.]))
    goal = {'target_pos': np.array([-.001, 0, 0]), 'target_ori_mat': np.eye(3)}
    result = ctrl.compute_control(goal, state)
    assert guard.fault is None and result[0] < q[0]
    assert np.max(np.abs(result-q)) <= guard.max_step.max() + 1e-9
