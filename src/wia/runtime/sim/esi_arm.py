"""Single left-arm Cartesian macros using native R1 IK and fixed pose targets.

Targets and orientation holds are expressed in the current robot base frame.
Only the working arm is changed to native absolute-pose IK; ESI's scene/task and
all other controllers retain their existing semantics. Wrist yaw is opt-in.
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from wia.runtime.sim.esi_rotation import rotation_axis_sign
from core.action_units import BASE_ATOMS, MOVE_ATOMS, WRIST_ROTATION_ATOMS, TRUNK_ATOMS
from core.feedback_semantics import IK_REJECTION_ERRORS

from wia.runtime.sim.esi_base import BaseControlConfig, R1BaseController, _main_thread
from wia.runtime.sim.esi_omnigibson import SHOWHARNESS_ROOT, _array
from wia.runtime.sim.esi_ik import NativeIKGuard
from wia.runtime.sim.esi_rotation import RotationControlConfig

DEFAULT_ARM_CONFIG = SHOWHARNESS_ROOT / "configs" / "esi_arm.json"
ARM_DIRECTIONS = dict(zip(MOVE_ATOMS, (
    (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
)))


@dataclass(frozen=True)
class ArmControlConfig:
    working_arm: str
    translation_m: float
    translation_tolerance_m: float
    orientation_tolerance_deg: float
    max_translation_speed_m_s: float
    timeout_sim_s: float
    timeout_wall_s: float | None
    settle_steps: int
    max_joint_speed_rad_s: float = 0.6
    max_ik_joint_step_rad: float = 5.0
    joint_limit_margin_rad: float = 0.02
    max_step_translation_m: float = 0.006
    progress_timeout_sim_s: float = 0.5
    ik_damping_singular_value: float = 0.02
    ik_damping_max: float = 0.01
    ik_recovery_progress_timeout_sim_s: float = 2.0
    initial_trunk_joint_tolerance_rad: float = 0.005
    ik_recovery_joint_tolerance_rad: float = 0.005
    initial_trunk_joint_positions_rad: list[float] | None = None

    def __post_init__(self):
        if self.working_arm != "left":
            raise ValueError("Milestone 3 supports only the left working arm.")
        for name, value in vars(self).items():
            if name == "timeout_wall_s" and value is None:
                continue
            if name not in ("working_arm", "initial_trunk_joint_positions_rad") and (
                    isinstance(value, bool) or not np.isfinite(value) or value <= 0):
                raise ValueError(f"Arm configuration {name} must be finite and positive.")
        if not isinstance(self.settle_steps, int):
            raise ValueError("settle_steps must be an integer.")
        if self.translation_tolerance_m >= self.translation_m:
            raise ValueError("Arm translation tolerance must be smaller than the step.")
        if self.initial_trunk_joint_positions_rad is not None:
            pose = np.asarray(self.initial_trunk_joint_positions_rad, dtype=float)
            if pose.shape != (4,) or not np.isfinite(pose).all():
                raise ValueError("Initial trunk posture must contain four finite joint positions in radians.")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ARM_CONFIG) -> ArmControlConfig:
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
            for key in ("path_tolerance_m", "base_hold_position_tolerance_m", "base_hold_orientation_tolerance_deg"):
                values.pop(key, None)
            return cls(**values)


def _orientation_error(before, after) -> float:
    return float(np.rad2deg((Rotation.from_quat(before).inv()
                            * Rotation.from_quat(after)).magnitude()))


def _preserve_aligned_reflection_trunk(session):
    """Retain only an explicitly aligned revised optical initialization."""
    profile = getattr(session, 'reflection_sensor_profile', None)
    if not isinstance(profile, dict) or profile.get('version') != 'reflection_sensor_profile_v3':
        return False
    from wia.runtime.sim.esi_reflection import reflection_profile_sha256
    receipt = getattr(session, 'reflection_alignment_initialization', {})
    return bool(getattr(session, 'native_reset_id', None) and receipt.get('completed') is True
        and receipt.get('reset_id') == getattr(session, 'native_reset_id', None)
        and receipt.get('profile_sha256') == reflection_profile_sha256(profile))


def _initialize_trunk(session, base_config, arm_config):
    """One startup joint posture; preserve base, arm and gripper joint targets."""
    hold = R1BaseController(session, base_config)
    robot, cfg = session.robot, arm_config
    ctrl = robot.controllers["trunk"]
    if ctrl.command_dim != 4:
        raise RuntimeError("Initial R1 working posture requires the verified four-joint trunk.")
    dofs = _array(ctrl.dof_idx).astype(int)
    target = np.asarray(cfg.initial_trunk_joint_positions_rad)
    lower, upper = [_array(x)[dofs] for x in robot.control_limits["position"]]
    if np.any(target <= lower + cfg.joint_limit_margin_rad) or np.any(target >= upper - cfg.joint_limit_margin_rad):
        raise ValueError("Initial trunk posture must stay inside its measured joint limits.")
    before = _array(robot.get_joint_positions())[dofs].copy()
    before_control = robot.get_control_dict()
    before_ee = {"position": _array(before_control["eef_left_pos_relative"]).tolist(),
                 "quaternion_xyzw": _array(before_control["eef_left_quat_relative"]).tolist()}
    started = time.monotonic()
    last_report = started
    stable = steps = 0
    try:
        while stable < cfg.settle_steps:
            q = _array(robot.get_joint_positions())[dofs]
            error = target - q
            stable = stable + 1 if np.max(np.abs(error)) <= cfg.initial_trunk_joint_tolerance_rad else 0
            now = time.monotonic()
            session.initial_trunk_diagnostics = {
                "control_steps": steps, "sim_time_s": steps * hold.dt,
                "wall_time_s": now - started, "wall_timeout_s": None,
                "sim_timeout_s": cfg.timeout_sim_s,
                "joint_positions_rad": q.tolist(), "target_rad": target.tolist(),
                "max_joint_error_rad": float(np.max(np.abs(error))),
                "stable_steps": stable, "required_stable_steps": cfg.settle_steps}
            if now - last_report >= 10:
                print('[arm-runtime] trunk initialization: ' +
                      json.dumps(session.initial_trunk_diagnostics), flush=True)
                last_report = now
            # Hardware/rendering latency is not evidence of physical failure.
            if steps * hold.dt >= cfg.timeout_sim_s and stable < cfg.settle_steps:
                raise RuntimeError('Initial trunk posture did not settle within its simulated-time bound: ' +
                                   json.dumps(session.initial_trunk_diagnostics))
            hold.hold_action[hold.indices["trunk"]] = q + np.clip(
                error, -cfg.max_joint_speed_rad_s * hold.dt, cfg.max_joint_speed_rad_s * hold.dt)
            hold.step_base_debug()
            steps += 1
    finally:
        hold.hold_action[hold.indices["trunk"]] = _array(robot.get_joint_positions())[dofs]
        hold.stop()
    after_control = robot.get_control_dict()
    after_ee = {"position": _array(after_control["eef_left_pos_relative"]).tolist(),
                "quaternion_xyzw": _array(after_control["eef_left_quat_relative"]).tolist()}
    result = {"requested_joint_positions_rad": target.tolist(), "joint_positions_before": before.tolist(),
              "joint_positions_after": _array(robot.get_joint_positions())[dofs].tolist(),
              "ee_before": before_ee, "ee_after": after_ee,
              "control_steps": steps,
              "wall_time_s": time.monotonic() - started, "wall_timeout_s": None}
    print(f"[arm-runtime] initial trunk posture: {result}", flush=True)
    return result


class R1ArmController(R1BaseController):
    @property
    def capabilities(self):
        return replace(super().capabilities, ee_translation=True,
                       ee_rotation=self.rotation_config is not None,
                       ee_tilt=self.rotation_config is not None,
                       trunk_height=getattr(self, "trunk_controller", None) is not None)

    def __init__(self, session, config: BaseControlConfig | None = None, *,
                 arm_config: ArmControlConfig | None = None,
                 rotation_config: RotationControlConfig | None = None,
                 on_step: Callable[[], None] | None = None,
                 on_result: Callable[[dict], None] | None = None):
        _main_thread()
        self.arm_config = arm_config or ArmControlConfig.load()
        self.rotation_config = rotation_config
        self.working_arm = self.arm_config.working_arm
        self.initial_trunk_state = None
        group = f"arm_{self.working_arm}"
        robot = session.robot
        original = robot.controllers[group]
        self.original_arm_controller = {
            "type": type(original).__name__, "motor_type": original.motor_type,
            "use_delta_commands": original.use_delta_commands,
            "command_dim": int(original.command_dim),
            "action_indices": _array(robot.controller_action_idx[group]).tolist(),
            "dof_indices": _array(original.dof_idx).tolist(),
            "mode": getattr(original, "mode", None),
        }
        print(f"[arm-runtime] original {group}: {self.original_arm_controller}", flush=True)
        if type(original).__name__ == "JointController":
            if original.motor_type != "position" or original.use_delta_commands:
                raise RuntimeError("Inspect the working-arm joint controller semantics before enabling IK.")
            if self.arm_config.initial_trunk_joint_positions_rad is not None:
                if _preserve_aligned_reflection_trunk(session):
                    self.initial_trunk_state = {'operation': 'retain_explicit_reflection_alignment',
                        'reset_id': session.native_reset_id,
                        'profile_sha256': session.reflection_alignment_initialization['profile_sha256'],
                        'joint_positions_rad': _array(robot.get_joint_positions())[
                            _array(robot.controllers['trunk'].dof_idx).astype(int)].tolist(),
                        'generic_trunk_preset_applied': False, 'physical_qualification_complete': False}
                else:
                    self.initial_trunk_state = _initialize_trunk(session, config, self.arm_config)
        elif not (type(original).__name__ == "InverseKinematicsController"
                  and original.mode == "absolute_pose"):
            raise RuntimeError("Unsupported working-arm controller; inspect before replacing its semantics.")
        # Finish any trunk preset with the original joint controllers first.
        # Configure all requested IK arms in one reload before installing guards.
        controller_config = deepcopy(robot._controller_config)
        reload_needed = False
        for arm in (self.working_arm,) + getattr(self, "_additional_ik_arms", ()):
            arm_group = f"arm_{arm}"
            existing = robot.controllers[arm_group]
            if type(existing).__name__ == "JointController":
                if existing.motor_type != "position" or existing.use_delta_commands:
                    raise RuntimeError("Arm IK conversion requires absolute joint position control.")
                controller_config[arm_group] = deepcopy(robot._default_arm_ik_controller_configs[arm])
                controller_config[arm_group].update(mode="absolute_pose", command_input_limits=None,
                    command_output_limits=None, smoothing_filter_size=None, use_impedances=False)
                reload_needed = True
            elif not (type(existing).__name__ == "InverseKinematicsController" and existing.mode == "absolute_pose"):
                raise RuntimeError("Unsupported arm controller semantics.")
        if reload_needed:
            robot.reload_controllers(controller_config)
        ctrl = robot.controllers[group]
        if (not np.array_equal(_array(ctrl.dof_idx), _array(robot.arm_control_idx[self.working_arm]))
                or ctrl.command_output_limits is not None):
            raise RuntimeError("Working-arm IK must control only its own joints with unscaled absolute commands.")
        super().__init__(session, config, on_step=on_step, on_result=on_result,
                         cartesian_arm=self.working_arm)
        self.arm_group = group
        physics_dt = (session.pipeline.og.sim.get_physics_dt()
                      if getattr(session, "pipeline", None) is not None else self.dt)
        self.ik_guard = NativeIKGuard(ctrl, robot.control_limits, physics_dt, self.arm_config)
        print(f"[arm-runtime] {group}: {type(ctrl).__name__}, mode={ctrl.mode}, "
              f"actions={self.indices[group].tolist()}, dofs={_array(ctrl.dof_idx).tolist()}", flush=True)
        self.ee_pose()

    def ee_pose(self) -> dict:
        """Debug pose: actual base-relative control frame and world EE pose."""
        _main_thread()
        control = self.robot.get_control_dict()
        pos = np.asarray(_array(control[f"eef_{self.working_arm}_pos_relative"]), dtype=float)
        quat = np.asarray(_array(control[f"eef_{self.working_arm}_quat_relative"]), dtype=float)
        world_pos, world_quat = (_array(v) for v in self.robot.get_eef_pose(self.working_arm))
        if (pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all()
                or not np.isfinite(quat).all() or np.linalg.norm(quat) == 0
                or not np.isfinite(world_pos).all() or not np.isfinite(world_quat).all()):
            raise RuntimeError("Working-arm EE pose must be finite with a valid quaternion.")
        return {"base_position": pos.tolist(), "base_quaternion_xyzw": quat.tolist(),
                "world_position": world_pos.tolist(), "world_quaternion_xyzw": world_quat.tolist()}

    def _set_arm_target(self, position, quaternion):
        self._arm_target_pose = {"base_position": np.asarray(position).tolist(),
                                 "base_quaternion_xyzw": np.asarray(quaternion).tolist(), "frame": "robot_base"}
        self.hold_action[self.indices[self.arm_group]] = np.r_[
            position, Rotation.from_quat(quaternion).as_rotvec()]

    def proprioception(self) -> dict:
        """Measured robot-relative state, captured on main with the camera frames."""
        _main_thread()
        pose = self.ee_pose()
        joints = _array(self.robot.get_joint_positions())
        gripper = self.robot.controllers[f"gripper_{self.working_arm}"]
        fingers = getattr(self.robot, "finger_joint_names", {}).get(self.working_arm, ())
        # R1's opposing prismatic finger coordinates add to its opening width.
        # Leave this unavailable on an unverified finger layout.
        width = None
        if (len(fingers) == 2 and len(_array(gripper.dof_idx)) == 2
                and all("prismatic" in str(self.robot.joints[name].joint_type).lower()
                        for name in fingers)):
            width = float(np.sum(joints[_array(gripper.dof_idx).astype(int)]))
        trunk = getattr(self, "trunk_controller", None)
        trunk_joints = joints[_array(self.robot.controllers["trunk"].dof_idx).astype(int)]
        trunk_height = trunk.measured_height(trunk_joints) if trunk is not None else None
        control_height = float(trunk.kinematics.pose(trunk_joints)[0][2]) if trunk is not None else None
        return {
            "frame": "robot_base", "source": "live", "working_arm": self.working_arm,
            "eef_pos": pose["base_position"],
            "eef_quaternion_xyzw": pose["base_quaternion_xyzw"],
            "eef_height_base_m": pose["base_position"][2],
            "arm_joint_positions_rad": joints[_array(
                self.robot.controllers[self.arm_group].dof_idx).astype(int)].tolist(),
            "trunk_joint_positions_rad": trunk_joints.tolist(),
            "trunk_height_m": trunk_height,
            "trunk_control_height_m": control_height,
            "gripper_width_m": width, "gripper_command_name": "HOLD",
            "gripper_closed": None, "holding": None,
            "table_height_m": None, "height_above_table_m": None,
        }

    def _freeze_arm_pose(self):
        current = self.ee_pose()
        self._set_arm_target(current["base_position"], current["base_quaternion_xyzw"])
        if self.ik_guard.fault in IK_REJECTION_ERRORS:
            self.hold_action[self.indices["base"]] = 0
        self.ik_guard.freeze(self.robot.get_joint_positions()[self.robot.controllers[self.arm_group].dof_idx],
                             preserve_recovery=True)

    def execute(self, token: str, *, should_cancel: Callable[[], bool] | None = None) -> dict:
        _main_thread()
        if token in TRUNK_ATOMS and getattr(self, "trunk_controller", None) is not None:
            return self.trunk_controller.execute(token, should_cancel=should_cancel)
        if token in MOVE_ATOMS:
            delta = np.asarray(ARM_DIRECTIONS[token]) * self.arm_config.translation_m
            return self._move_ee(delta, token, should_cancel)
        if token in WRIST_ROTATION_ATOMS and self.rotation_config is not None:
            axis, sign = rotation_axis_sign(token)
            return self._execute_ee_pose(np.zeros(3), token, should_cancel,
                                        rotation_rad=sign * np.deg2rad(self.rotation_config.rotation_deg), rotation_axis=axis)
        if token not in BASE_ATOMS:
            raise ValueError(f"Single-arm executor rejects action {token!r}.")
        return super().execute(token, should_cancel=should_cancel)

    def _execute(self, motion, direction, base_before, target_xy, target_yaw,
                 requested, axis, should_cancel):
        """Execute base motion while retaining the current arm target."""
        before = self.ee_pose()
        previous_result = self.on_result
        try:
            self.on_result = None
            result = super()._execute(motion, direction, base_before, target_xy,
                                      target_yaw, requested, axis, should_cancel)
        finally:
            self.on_result = previous_result
        result.update(working_arm=self.working_arm, ee_pose_before=before, ee_pose_after=self.ee_pose(),
                      ik_diagnostics=self.ik_guard.diagnostics())
        return self._finish_result(result)

    def execute_metric(self, token, magnitude, *, should_cancel=None):
        _main_thread()
        if not np.isfinite(magnitude) or magnitude <= 0:
            raise ValueError("Metric motion must be finite and positive.")
        if token in TRUNK_ATOMS and getattr(self, "trunk_controller", None) is not None:
            return self.trunk_controller.execute_metric(token, magnitude, should_cancel=should_cancel)
        if token in MOVE_ATOMS:
            return self._move_ee(np.asarray(ARM_DIRECTIONS[token])*magnitude, token, should_cancel)
        if token in WRIST_ROTATION_ATOMS and self.rotation_config is not None:
            axis, sign = rotation_axis_sign(token)
            return self._execute_ee_pose(np.zeros(3), token, should_cancel,
                                        rotation_rad=sign*np.deg2rad(magnitude), rotation_axis=axis)
        return super().execute_metric(token, magnitude, should_cancel=should_cancel)

    def move_ee_debug(self, dx: float, dy: float, dz: float, *,
                      should_cancel: Callable[[], bool] | None = None) -> dict:
        """One direct base-frame EE translation; the delta is applied once only."""
        _main_thread()
        delta = np.asarray([dx, dy, dz], dtype=float)
        if not np.isfinite(delta).all() or np.linalg.norm(delta) == 0:
            raise ValueError("EE delta must be finite and nonzero.")
        return self._move_ee(delta, "ARM_DEBUG", should_cancel)

    def _move_ee(self, delta, action, should_cancel):
        return self._execute_ee_pose(delta, action, should_cancel)

    def rotate_ee_debug(self, direction: str, angle_deg: float, *,
                        should_cancel: Callable[[], bool] | None = None) -> dict:
        """Direct local-Z yaw diagnostic; HTTP/MCP accept fixed tokens only."""
        _main_thread()
        if self.rotation_config is None:
            raise ValueError("End-effector rotation is disabled.")
        if direction not in ("cw", "ccw") or not np.isfinite(angle_deg) or not 0 < angle_deg < 180:
            raise ValueError("Rotation requires cw/ccw and a finite angle in (0, 180).")
        sign = 1 if direction == "cw" else -1
        return self._execute_ee_pose(np.zeros(3), "ARM_ROTATE_DEBUG", should_cancel,
                                    rotation_rad=sign * np.deg2rad(angle_deg))

    def _execute_ee_pose(self, delta, action, should_cancel, *, rotation_rad=None, rotation_axis=2):
        cfg = self.arm_config
        rotating = rotation_rad is not None
        rot_cfg = self.rotation_config
        self.ik_guard.reset(self.robot.get_joint_positions()[self.robot.controllers[self.arm_group].dof_idx])
        before, base_before = self.ee_pose(), self.pose()
        # A single target and reference orientation, independent of control steps.
        target = np.asarray(before["base_position"]) + delta
        start_rotation = Rotation.from_quat(before["base_quaternion_xyzw"])
        target_rotation = (start_rotation * Rotation.from_rotvec(np.eye(3)[rotation_axis] * rotation_rad)
                           if rotating else start_rotation)
        target_quat = target_rotation.as_quat().tolist()
        self._active_control = {"execution_status": "running", "action": action,
                                "base_initial_pose": base_before, "arm_initial_pose": before}
        orientation_tolerance = (rot_cfg.rotation_tolerance_deg if rotating else cfg.orientation_tolerance_deg)
        started = time.monotonic()
        control_steps = stable_steps = 0
        timeout = cancelled = False
        reason = "completed"
        previous_position = np.asarray(before["base_position"])
        previous_quat = before["base_quaternion_xyzw"]
        best_error = abs(float(np.rad2deg(rotation_rad))) if rotating else float(np.linalg.norm(delta))
        progress_step = 0
        max_step_motion = max_step_rotation = 0.0
        recovery_steps = 0
        recovery_completed = recovery_error = None

        def measure():
            current = self.ee_pose()
            error = target - current["base_position"]
            ori_error = _orientation_error(target_quat, current["base_quaternion_xyzw"])
            return current, error, ori_error

        try:
            while True:
                current, error, ori_error = measure()
                if should_cancel is not None and should_cancel():
                    cancelled, reason = True, "cancelled"
                    break
                if self.ik_guard.fault is not None:
                    reason = self.ik_guard.fault
                    break
                position = np.asarray(current["base_position"])
                current_rotation = Rotation.from_quat(current["base_quaternion_xyzw"])
                step_rotation = _orientation_error(previous_quat, current["base_quaternion_xyzw"])
                previous_quat = current["base_quaternion_xyzw"]
                max_step_rotation = max(max_step_rotation, step_rotation)
                step_motion = float(np.linalg.norm(position - previous_position))
                max_step_motion = max(max_step_motion, step_motion)
                previous_position = position.copy()
                if (step_motion > cfg.max_step_translation_m
                        or (rotating and step_rotation > rot_cfg.max_step_rotation_deg)):
                    reason = "arm_step_limit"
                    break
                reached = np.linalg.norm(error) <= cfg.translation_tolerance_m and ori_error <= orientation_tolerance
                stable_steps = stable_steps + 1 if reached else 0
                if stable_steps >= cfg.settle_steps:
                    break
                remaining = ori_error if rotating else float(np.linalg.norm(error))
                progress_tolerance = orientation_tolerance if rotating else cfg.translation_tolerance_m
                if remaining < best_error - progress_tolerance / 2:
                    best_error, progress_step = remaining, control_steps
                elif not reached and (control_steps - progress_step) * self.dt >= (
                        cfg.ik_recovery_progress_timeout_sim_s if self.ik_guard.damped_recovery_steps
                        else cfg.progress_timeout_sim_s):
                    reason = "arm_motion_blocked"
                    break
                if (control_steps * self.dt >= cfg.timeout_sim_s
                        or (cfg.timeout_wall_s is not None and time.monotonic() - started >= cfg.timeout_wall_s)):
                    timeout, reason = True, "timeout"
                    break
                # Feed an absolute intermediate pose toward the fixed target.
                # Repeating the requested 2 cm as a delta would accumulate.
                step = error.copy()
                norm, limit = np.linalg.norm(step), cfg.max_translation_speed_m_s * self.dt
                if norm > limit:
                    step *= limit / norm
                command_quat = target_quat
                if rotating:
                    remaining_rotation = (current_rotation.inv() * target_rotation).as_rotvec()
                    angle = np.linalg.norm(remaining_rotation)
                    angle_limit = np.deg2rad(rot_cfg.max_rotation_speed_deg_s) * self.dt
                    if angle > angle_limit:
                        remaining_rotation *= angle_limit / angle
                    command_quat = (current_rotation * Rotation.from_rotvec(remaining_rotation)).as_quat()
                self._set_arm_target(np.asarray(current["base_position"]) + step, command_quat)
                self.step_base_debug()
                control_steps += 1
        finally:
            diagnostic = getattr(self.session, "private_motion_diagnostics", None)
            if diagnostic is not None and reason != "completed":
                diagnostic.record(action=action, reason=reason, phase="before_stop", control_steps=control_steps)
            # Halt an unreachable/cancelled target too; continuing to send the
            # old target after reporting a failure would continue arm motion.
            self._freeze_arm_pose()
            self.stop()
            # Rejected native proposals hold the measured pose for a fresh
            # action. Other solver faults retain their bounded legacy recovery.
            if (self.ik_guard.hold is not None
                    and self.ik_guard.fault not in IK_REJECTION_ERRORS):
                dofs = self.robot.controllers[self.arm_group].dof_idx
                max_recovery_steps = int(cfg.timeout_sim_s / self.dt)
                while recovery_steps < max_recovery_steps:
                    joints = _array(self.robot.get_joint_positions()[dofs])
                    if np.max(np.abs(joints - _array(self.ik_guard.hold))) <= cfg.ik_recovery_joint_tolerance_rad:
                        break
                    self.step_base_debug()
                    recovery_steps += 1
                recovery_error = float(np.max(np.abs(
                    _array(self.robot.get_joint_positions()[dofs]) - _array(self.ik_guard.hold))))
                recovery_completed = recovery_error <= cfg.ik_recovery_joint_tolerance_rad
            self._freeze_arm_pose()
        # Holding the measured end pose may itself reject a native IK proposal.
        # Preserve that late rejection so the executor runs measured settling;
        # a stale arm_motion_blocked reason would skip that stop certificate.
        pre_stop_reason = reason
        if (not cancelled and not timeout and self.ik_guard.fault in IK_REJECTION_ERRORS
                and reason in {"completed", "arm_motion_blocked", "target_not_reached"}):
            reason = self.ik_guard.fault
        after, error, ori_error = measure()
        completed = (reason == "completed" and np.linalg.norm(error) <= cfg.translation_tolerance_m
                     and ori_error <= orientation_tolerance)
        if reason == "completed" and not completed:
            reason = "target_not_reached"
        actual = np.asarray(after["base_position"]) - before["base_position"]
        result = {
            "action": action, "motion": "arm_rotation" if rotating else "arm_translation",
            "working_arm": self.working_arm, "frame": "eef_local" if rotating else "robot_base",
            "pre_stop_reason": pre_stop_reason,
            "target_base_position": target.tolist(), "target_base_quaternion_xyzw": target_quat,
            "ee_pose_before": before, "ee_pose_after": after,
            "base_pose_before": base_before, "base_pose_after": self.pose(),
            "position_error_m": float(np.linalg.norm(error)), "orientation_error_deg": ori_error,
            "duration_s": time.monotonic() - started,
            "control_steps": control_steps + self.config.stop_steps + recovery_steps,
            "stop_steps": self.config.stop_steps + recovery_steps,
            "recovery_steps": recovery_steps,
            "recovery_completed": recovery_completed,
            "recovery_max_joint_error_rad": recovery_error,
            "sim_duration_s": (control_steps + self.config.stop_steps + recovery_steps) * self.dt,
            "completed": bool(completed), "timeout": timeout, "cancelled": cancelled, "reason": reason,
            "max_step_motion_m": max_step_motion,
            "ik_diagnostics": self.ik_guard.diagnostics(),
        }
        if rotating:
            signed_actual = float(np.rad2deg((start_rotation.inv() * Rotation.from_quat(
                after["base_quaternion_xyzw"])).as_rotvec()[rotation_axis]))
            result.update(requested_deg=abs(float(np.rad2deg(rotation_rad))),
                          actual_deg=signed_actual * np.sign(rotation_rad),
                          signed_requested_deg=float(np.rad2deg(rotation_rad)), signed_actual_deg=signed_actual,
                          rotation_axis="+" + "XYZ"[rotation_axis],
                          max_step_rotation_deg=max_step_rotation)
        else:
            requested = float(np.linalg.norm(delta))
            result.update(requested_delta_m=delta.tolist(), requested_m=requested,
                          actual_delta_m=actual.tolist(), actual_m=float(actual @ (delta / requested)))
        return self._finish_result(result)
