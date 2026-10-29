"""Meaningful base contract checks without Isaac: shuffled slices, feedback and failures."""
from dataclasses import replace
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from core.action_units import BASE_ATOMS, MOVE_ATOMS, ATOMIC_ACTIONS
from wia.runtime.sim.esi_base import BaseControlConfig, R1BaseController


class JointController:
    motor_type = "position"
    use_delta_commands = False
    command_input_limits = None

    def __init__(self, dof_idx):
        self.dof_idx = np.asarray(dof_idx)
        self.command_dim = len(dof_idx)

    def compute_no_op_action(self, control_dict):
        return control_dict["joint_position"][self.dof_idx].copy()


class HolonomicBaseJointController(JointController):
    def compute_no_op_action(self, control_dict):
        return np.zeros(3)


class FeedbackRobot:
    """Position servo with finite response; action slices deliberately noncontiguous."""
    name = "r1_test"
    action_dim = 11

    def __init__(self, yaw=0.0):
        self.xy = np.array([3.0, -2.0])
        self.yaw = np.deg2rad(yaw)
        self.joints = np.arange(11, dtype=float) / 10 + 0.2
        self.controllers = {
            "base": HolonomicBaseJointController([0, 1, 2]),
            "trunk": JointController([3, 4]), "arm_left": JointController([5, 6]),
            "gripper_left": JointController([7]), "arm_right": JointController([8, 9]),
            "gripper_right": JointController([10]),
        }
        self.controller_action_idx = {
            "base": np.array([8, 3, 0]), "trunk": np.array([2, 6]),
            "arm_left": np.array([1, 7]), "gripper_left": np.array([4]),
            "arm_right": np.array([9, 5]), "gripper_right": np.array([10]),
        }

    def get_control_dict(self):
        return {"joint_position": self.joints}

    def get_joint_positions(self):
        return self.joints.copy()

    def get_position_orientation(self):
        return np.r_[self.xy, 0.0], np.array([0, 0, np.sin(self.yaw / 2), np.cos(self.yaw / 2)])


class FeedbackEnv:
    env_config = {"action_frequency": 30}

    def __init__(self, robot):
        self.robot = robot
        self.actions = []
        self.response = 0.5
        self.drift = 0.0
        self.fail_step = None
        self.threads = []

    def step(self, action):
        self.threads.append(threading.get_ident())
        action = action[self.robot.name]
        self.actions.append(action.copy())
        if self.fail_step == len(self.actions):
            raise RuntimeError("simulator failure")
        r = self.robot
        dx, dy, da = action[r.controller_action_idx["base"]]
        c, s = np.cos(r.yaw), np.sin(r.yaw)
        r.xy += self.response * np.array([c * dx - s * dy, s * dx + c * dy])
        r.yaw += self.response * da
        for name, ctrl in r.controllers.items():
            if name != "base":
                r.joints[ctrl.dof_idx] = action[r.controller_action_idx[name]]
        r.joints[5] += self.drift


def make_controller(yaw=0, **config_changes):
    robot = FeedbackRobot(yaw)
    env = FeedbackEnv(robot)
    cfg = replace(BaseControlConfig.load(), **config_changes)
    controller = R1BaseController(SimpleNamespace(robot=robot, env=env), cfg)
    return controller, robot, env


def test_hold_uses_current_joint_positions_and_runtime_slices():
    controller, robot, env = make_controller()
    initial = robot.joints.copy()
    controller.stop()
    assert np.allclose(robot.joints[3:], initial[3:])
    for action in env.actions:
        assert np.all(action[robot.controller_action_idx["base"]] == 0)
        for name, ctrl in robot.controllers.items():
            if name != "base":
                assert np.allclose(action[robot.controller_action_idx[name]], initial[ctrl.dof_idx])
    assert set(env.threads) == {threading.get_ident()}


@pytest.mark.parametrize("yaw", [0, 90])
@pytest.mark.parametrize("token,local", [("BASE_FWD", [1, 0]), ("BASE_BACK", [-1, 0]),
                                         ("BASE_LEFT", [0, 1]), ("BASE_RIGHT", [0, -1])])
def test_translation_is_robot_relative_and_feedback_reaches_one_fixed_target(yaw, token, local):
    controller, robot, env = make_controller(yaw)
    before = robot.xy.copy()
    result = controller.execute(token)
    c, s = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
    axis = np.array([[c, -s], [s, c]]) @ local
    assert result["completed"] and not result["timeout"]
    assert result["action"] == token
    assert result["actual_m"] == pytest.approx(0.25, abs=controller.config.translation_tolerance_m)
    assert np.allclose(robot.xy - before, axis * 0.25, atol=controller.config.translation_tolerance_m)
    assert result["control_steps"] > 1
    assert np.all(env.actions[-1][robot.controller_action_idx["base"]] == 0)


def test_turn_crosses_yaw_wrap_then_forward_uses_new_heading():
    controller, robot, _ = make_controller(yaw=179)
    left = controller.execute("BASE_TURN_LEFT")
    assert left["completed"] and left["actual_deg"] == pytest.approx(15, abs=0.1)
    before = robot.xy.copy()
    heading = robot.yaw
    forward = controller.execute("BASE_FWD")
    assert forward["completed"]
    assert np.allclose(robot.xy - before, 0.25 * np.array([np.cos(heading), np.sin(heading)]),
                       atol=controller.config.translation_tolerance_m)
    right = controller.execute("BASE_TURN_RIGHT")
    assert right["completed"] and right["actual_deg"] == pytest.approx(15, abs=0.1)


def test_blocked_motion_times_out_and_stops_with_honest_result():
    controller, robot, env = make_controller(timeout_sim_s=0.1)
    env.response = 0
    logged = []
    controller.on_result = logged.append
    result = controller.execute("BASE_FWD")
    assert not result["completed"] and result["timeout"]
    assert result["actual_m"] == 0 and result["requested_m"] == 0.25
    assert result["reason"] == "timeout" and logged == [result]
    assert np.all(env.actions[-1][robot.controller_action_idx["base"]] == 0)


def test_cancellation_stops_and_inactive_offset_does_not_reject_motion():
    controller, robot, env = make_controller()
    result = controller.execute("BASE_FWD", should_cancel=lambda: len(env.actions) >= 2)
    assert result["cancelled"] and not result["completed"] and not result["timeout"]
    assert np.all(env.actions[-1][robot.controller_action_idx["base"]] == 0)
    env.drift = 0.1
    result = controller.execute("BASE_FWD")
    assert result["completed"] and result["reason"] == "completed"
    assert "inactive_joint_max_drift" not in result


def test_simulator_failure_still_sends_stop_and_wrong_thread_fails_before_access():
    controller, robot, env = make_controller()
    env.fail_step = 2
    with pytest.raises(RuntimeError, match="simulator failure"):
        controller.execute("BASE_FWD")
    assert np.all(env.actions[-1][robot.controller_action_idx["base"]] == 0)
    errors = []

    def worker():
        try:
            controller.execute("BASE_FWD")
        except RuntimeError as exc:
            errors.append(str(exc))

    count = len(env.actions)
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=2)
    assert errors and "main thread" in errors[0] and len(env.actions) == count


def test_unknown_layout_and_manipulation_tokens_are_rejected():
    controller, robot, env = make_controller()
    for token in (*MOVE_ATOMS, "GRASP", "RELEASE", "RT_YAW_CW"):
        with pytest.raises(ValueError, match="rejects"):
            controller.execute(token)
    assert not env.actions
    robot.controllers["arm_left"].use_delta_commands = True
    with pytest.raises(RuntimeError, match="Unsupported arm_left"):
        R1BaseController(SimpleNamespace(robot=robot, env=env))
    assert set(BASE_ATOMS).isdisjoint(MOVE_ATOMS)
    assert set(BASE_ATOMS).isdisjoint(ATOMIC_ACTIONS)
