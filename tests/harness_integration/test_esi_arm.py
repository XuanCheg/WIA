"""Arm contracts with finite feedback, shuffled slices and interrupted targets."""
from dataclasses import replace
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.action_units import MOVE_ATOMS
from wia.runtime.sim.esi_arm import ArmControlConfig, R1ArmController
from wia.runtime.sim.esi_base import BaseControlConfig
from harness_integration.test_esi_base import HolonomicBaseJointController, JointController


class InverseKinematicsController(JointController):
    mode = "absolute_pose"
    command_output_limits = None
    task_name = "eef_left"

    def compute_control(self, goal_dict, control_dict):
        return control_dict["joint_position"][self.dof_idx].copy()

    def compute_no_op_action(self, control):
        return np.r_[control["eef_left_pos_relative"],
                     Rotation.from_quat(control["eef_left_quat_relative"]).as_rotvec()]


class ArmRobot:
    name = "r1_arm_test"
    action_dim = 15

    def __init__(self, yaw):
        self.xy = np.array([3.0, -2.0])
        self.yaw = np.deg2rad(yaw)
        self.ee = np.array([0.45, 0.18, 1.24])
        self.ee_rot = Rotation.from_euler("xyz", [175, 2, 1], degrees=True)
        self.joints = np.arange(self.action_dim, dtype=float) / 10 + 0.2
        self.controllers = {
            "base": HolonomicBaseJointController([0, 1, 2]),
            "trunk": JointController([3, 4]), "arm_left": JointController(list(range(5, 11))),
            "gripper_left": JointController([11]), "arm_right": JointController([12, 13]),
            "gripper_right": JointController([14]),
        }
        self.controller_action_idx = {
            "base": np.array([14, 3, 0]), "trunk": np.array([2, 6]),
            "arm_left": np.array([1, 7, 8, 9, 10, 11]), "gripper_left": np.array([4]),
            "arm_right": np.array([12, 5]), "gripper_right": np.array([13]),
        }
        self.arm_control_idx = {"left": self.controllers["arm_left"].dof_idx}
        self._controller_config = {name: {"name": type(ctrl).__name__}
                                   for name, ctrl in self.controllers.items()}
        self._default_arm_ik_controller_configs = {"left": {"name": "InverseKinematicsController"}}
        self.reloads = []
        self.control_limits = {"position": (np.full(15, -10.0), np.full(15, 10.0)),
                               "velocity": (np.full(15, -5.0), np.full(15, 5.0))}

    def reload_controllers(self, config):
        self.reloads.append(config)
        self.controllers["arm_left"] = InverseKinematicsController(self.arm_control_idx["left"])

    def get_control_dict(self):
        return {"joint_position": self.joints, "eef_left_pos_relative": self.ee.copy(),
                "eef_left_quat_relative": self.ee_rot.as_quat()}

    def get_joint_positions(self):
        return self.joints.copy()

    def get_position_orientation(self):
        return np.r_[self.xy, 0.0], Rotation.from_euler("z", self.yaw).as_quat()

    def get_eef_pose(self, arm):
        assert arm == "left"
        base = Rotation.from_euler("z", self.yaw)
        return np.r_[self.xy, 0.0] + base.apply(self.ee), (base * self.ee_rot).as_quat()


class ArmEnv:
    env_config = {"action_frequency": 30}

    def __init__(self, robot):
        self.robot = robot
        self.actions = []
        self.arm_response = 0.5
        self.other_drift = self.arm_base_drift = self.base_arm_drift = 0.0
        self.fail_step = None
        self.threads = []

    def step(self, actions):
        self.threads.append(threading.get_ident())
        r = self.robot
        action = actions[r.name]
        self.actions.append(action.copy())
        if len(self.actions) == self.fail_step:
            raise RuntimeError("simulator failure")
        base_cmd = action[r.controller_action_idx["base"]]
        r.xy += 0.5 * Rotation.from_euler("z", r.yaw).apply(np.r_[base_cmd[:2], 0])[:2]
        r.yaw += 0.5 * base_cmd[2]
        arm = action[r.controller_action_idx["arm_left"]]
        r.ee += self.arm_response * (arm[:3] - r.ee)
        remaining = r.ee_rot.inv() * Rotation.from_rotvec(arm[3:])
        r.ee_rot = r.ee_rot * Rotation.from_rotvec(self.arm_response * remaining.as_rotvec())
        r.joints[r.arm_control_idx["left"]] = np.r_[r.ee, r.ee_rot.as_rotvec()]
        for name in ("trunk", "gripper_left", "arm_right", "gripper_right"):
            r.joints[r.controllers[name].dof_idx] = action[r.controller_action_idx[name]]
        r.joints[12] += self.other_drift
        if np.any(base_cmd):
            r.ee[0] += self.arm_base_drift
        else:
            r.xy[0] += self.base_arm_drift


def make_arm_controller(yaw=0, **arm_config_changes):
    robot = ArmRobot(yaw)
    env = ArmEnv(robot)
    arm_config_changes.setdefault("initial_trunk_joint_positions_rad", None)
    arm_config = replace(ArmControlConfig.load(), **arm_config_changes)
    controller = R1ArmController(SimpleNamespace(robot=robot, env=env),
                                 BaseControlConfig.load(), arm_config=arm_config)
    return controller, robot, env


def test_initial_trunk_settles_despite_slow_wall_clock(monkeypatch):
    import wia.runtime.sim.esi_arm as esi_arm
    from itertools import count

    joints = np.zeros(4)
    robot = SimpleNamespace(
        controllers={'trunk': SimpleNamespace(command_dim=4, dof_idx=np.arange(4))},
        control_limits={'position': (np.full(4, -3.), np.full(4, 3.))},
        get_joint_positions=lambda: joints.copy(),
        get_control_dict=lambda: {'eef_left_pos_relative': np.zeros(3),
                                 'eef_left_quat_relative': np.array([0., 0., 0., 1.])})
    hold = SimpleNamespace(config=BaseControlConfig.load(), dt=1 / 30,
                           indices={'trunk': np.arange(4)}, hold_action=joints.copy(), stop=lambda: None)
    hold.step_base_debug = lambda: np.copyto(joints, hold.hold_action)
    monkeypatch.setattr(esi_arm, 'R1BaseController', lambda *args: hold)
    clock = count(step=60.)
    monkeypatch.setattr(esi_arm.time, 'monotonic', lambda: next(clock))
    session = SimpleNamespace(robot=robot)
    config = ArmControlConfig.load()
    result = esi_arm._initialize_trunk(session, None, config)
    assert config.timeout_wall_s is None
    assert result['wall_time_s'] > 30.
    assert result['wall_timeout_s'] is None
    assert session.initial_trunk_diagnostics['sim_time_s'] < config.timeout_sim_s
    np.testing.assert_allclose(joints, config.initial_trunk_joint_positions_rad)


@pytest.mark.parametrize("yaw", [0, 90])
@pytest.mark.parametrize("token,delta", list(zip(MOVE_ATOMS, (
    [0.02, 0, 0], [-0.02, 0, 0], [0, 0.02, 0], [0, -0.02, 0], [0, 0, 0.02], [0, 0, -0.02]))))
def test_six_directions_use_one_base_frame_target_and_hold_orientation(yaw, token, delta):
    controller, robot, env = make_arm_controller(yaw)
    before = controller.ee_pose()
    base_before = robot.xy.copy()
    joints_before = robot.joints.copy()
    result = controller.execute(token)
    tol = controller.arm_config.translation_tolerance_m
    assert result["completed"] and not result["timeout"]
    assert np.allclose(result["target_base_position"], np.array(before["base_position"]) + delta)
    assert np.allclose(robot.ee - before["base_position"], delta, atol=tol)
    expected_world = Rotation.from_euler("z", yaw, degrees=True).apply(delta)
    assert np.allclose(np.array(result["ee_pose_after"]["world_position"]) - before["world_position"],
                       expected_world, atol=tol)
    assert result["orientation_error_deg"] < 0.001
    assert np.array_equal(robot.xy, base_before)
    for name in controller.inactive_targets:
        idx = robot.controllers[name].dof_idx
        assert np.allclose(robot.joints[idx], joints_before[idx])
    assert result["control_steps"] > 1
    # Commands progress toward the target but never accumulate another 2 cm.
    cmds = np.array([a[controller.indices["arm_left"]][:3] for a in env.actions])
    assert np.max(np.linalg.norm(cmds - before["base_position"], axis=1)) <= 0.02 + 1e-6
    assert all(np.all(a[controller.indices["base"]] == 0) for a in env.actions)
    assert set(env.threads) == {threading.get_ident()}


def test_native_override_changes_only_working_arm_and_supports_direct_debug():
    controller, robot, _ = make_arm_controller()
    assert controller.original_arm_controller["type"] == "JointController"
    assert len(robot.reloads) == 1
    for name in robot.controllers:
        if name != "arm_left":
            assert robot.reloads[0][name] == robot._controller_config[name]
    config = robot.reloads[0]["arm_left"]
    assert config["mode"] == "absolute_pose"
    assert config["command_input_limits"] is None and config["command_output_limits"] is None
    assert not config["use_impedances"] and config["smoothing_filter_size"] is None
    result = controller.move_ee_debug(0, 0.02, 0)
    assert result["completed"] and result["action"] == "ARM_DEBUG"


def test_proprioception_is_robot_relative_and_requires_simulator_owner_thread():
    controller, robot, _ = make_arm_controller()
    state = controller.proprioception()
    assert state["eef_pos"] == robot.ee.tolist() and state["frame"] == "robot_base"
    assert "world_position" not in state and state["gripper_width_m"] is None
    errors = []

    def read_from_worker():
        try:
            controller.proprioception()
        except RuntimeError as error:
            errors.append(str(error))

    worker = threading.Thread(target=read_from_worker)
    worker.start()
    worker.join(timeout=2)
    assert errors and "main thread" in errors[0]


def test_moved_arm_holds_relative_to_body_during_base_translation_and_turn():
    controller, robot, _ = make_arm_controller()
    for _ in range(3):
        assert controller.execute("MV_FWD")["completed"]
    away = robot.ee.copy()
    logged = []
    controller.on_result = logged.append
    for token in ("BASE_FWD", "BASE_TURN_LEFT", "BASE_BACK"):
        result = controller.execute(token)
        assert result["completed"]
        assert "arm_hold_max_error_m" not in result
        assert np.allclose(robot.ee, away)
    assert len(logged) == 3
    # Direct base diagnostics preserve the same arm target.
    assert controller.execute_base_rotation("left", 90)["completed"]


@pytest.mark.parametrize("interrupt", ["timeout", "cancel"])
def test_interrupted_arm_target_is_frozen_before_idle_or_base_motion(interrupt):
    controller, robot, env = make_arm_controller(timeout_sim_s=0.1)
    if interrupt == "timeout":
        env.arm_response = 0
        result = controller.execute("MV_FWD")
        assert result["timeout"]
    else:
        result = controller.execute("MV_FWD", should_cancel=lambda: len(env.actions) >= 2)
        assert result["cancelled"] and not result["timeout"]
    assert not result["completed"]
    stopped = robot.ee.copy()
    env.arm_response = 0.5
    for _ in range(20):
        controller.step_base_debug()
    assert np.allclose(robot.ee, stopped, atol=1e-7)
    assert controller.execute("BASE_FWD")["completed"]
    assert np.allclose(robot.ee, stopped, atol=1e-7)


@pytest.mark.parametrize("offset,token", [
    ("other_drift", "MV_FWD"),
    ("base_arm_drift", "MV_FWD"),
    ("arm_base_drift", "BASE_FWD"),
])
def test_unrelated_component_offsets_do_not_reject_motion(offset, token):
    controller, _, env = make_arm_controller()
    setattr(env, offset, 0.03)
    logged = []
    controller.on_result = logged.append
    result = controller.execute(token)
    assert result["completed"] and result["reason"] == "completed"
    assert not result["cancelled"]
    assert logged == [result]


def test_failure_stops_arm_and_wrong_thread_never_steps():
    controller, robot, env = make_arm_controller()
    env.fail_step = 2
    with pytest.raises(RuntimeError, match="simulator failure"):
        controller.execute("MV_FWD")
    stopped = robot.ee.copy()
    controller.stop()
    assert np.allclose(robot.ee, stopped, atol=1e-7)
    errors = []

    def worker():
        try:
            controller.move_ee_debug(0.02, 0, 0)
        except RuntimeError as exc:
            errors.append(str(exc))

    count = len(env.actions)
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=2)
    assert errors and "main thread" in errors[0] and len(env.actions) == count


def test_invalid_parameters_and_uninspected_controller_fail_before_physics():
    controller, robot, env = make_arm_controller()
    for token in ("GRASP", "RELEASE", "RT_YAW_CW", "ROTATE_CCW", "STAY"):
        with pytest.raises(ValueError, match="rejects"):
            controller.execute(token)
    for delta in ((0, 0, 0), (float("nan"), 0, 0)):
        with pytest.raises(ValueError, match="finite and nonzero"):
            controller.move_ee_debug(*delta)
    robot.controllers["arm_left"].mode = "pose_delta_ori"
    with pytest.raises(RuntimeError, match="Unsupported working-arm"):
        R1ArmController(controller.session)
    assert not env.actions
    with pytest.raises(ValueError, match="only the left"):
        replace(ArmControlConfig.load(), working_arm="right")


def test_startup_posture_rejects_unverified_trunk_layout_before_motion():
    robot = ArmRobot(0)
    env = ArmEnv(robot)
    with pytest.raises(RuntimeError, match="four-joint trunk"):
        R1ArmController(SimpleNamespace(robot=robot, env=env))
    assert not env.actions and not robot.reloads
    with pytest.raises(ValueError, match="four finite"):
        replace(ArmControlConfig.load(), initial_trunk_joint_positions_rad=[0, 1])


def test_trunk_control_height_survives_controller_state_projection():
    from core.observations import AgentRobotState
    controller, _, _ = make_arm_controller()
    measured = controller.proprioception()
    controller.proprioception = lambda: dict(measured, trunk_height_m=1.00361,
        trunk_control_height_m=.96196, trunk_joint_positions_rad=[.7, -1.4, -.7, 0.], object_name='PRIVATE_OBJECT')
    state = controller.controller_state()
    assert state.proprioception['trunk_control_height_m'] == .96196
    assert 'object_name' not in state.proprioception
    public = AgentRobotState.from_controller_state(state)
    assert public.trunk_height_m == 1.00361 and public.trunk_control_height_m == .96196


def test_action_completion_ignores_slow_wall_clock_by_default(monkeypatch):
    from itertools import count
    import wia.runtime.sim.esi_arm as arm_module
    controller, robot, env = make_arm_controller()
    assert controller.config.timeout_wall_s is None
    assert controller.arm_config.timeout_sim_s == 120.
    clock = count(0., 300.)
    monkeypatch.setattr(arm_module.time, 'monotonic', lambda: next(clock))
    result = controller.execute('MV_FWD')
    assert result['completed'] and not result['timeout']
    assert result['duration_s'] > 120.
    assert result['sim_duration_s'] < 120.


def test_action_stops_at_two_minutes_of_simulation():
    # Prevent completion to exercise the deadline, while leaving motion feasible.
    controller, _, _ = make_arm_controller(settle_steps=10000)
    result = controller.execute('MV_FWD')
    assert result['timeout'] and not result['completed']
    assert (result['control_steps'] - result['stop_steps']) * controller.dt == pytest.approx(120.)
    assert result['sim_duration_s'] == pytest.approx(120. + result['stop_steps'] * controller.dt)


def test_all_action_configs_default_to_simulation_only_deadline():
    from wia.runtime.sim.esi_gripper import GripperControlConfig
    from wia.runtime.sim.esi_trunk import TrunkControlConfig
    from wia.runtime.sim.esi_omnigibson import SHOWHARNESS_ROOT
    configs = (BaseControlConfig.load(), ArmControlConfig.load(), GripperControlConfig.load(),
               TrunkControlConfig.load(SHOWHARNESS_ROOT / 'configs/esi_trunk.json'))
    assert all(config.timeout_sim_s == 120. and config.timeout_wall_s is None for config in configs)
