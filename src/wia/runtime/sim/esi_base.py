"""R1 base macros using the existing ESI position controllers and pose readback.

Runtime experiment (research_log.md): base +X is forward, +Y is left, and
positive yaw is counterclockwise viewed from above. Directions use the base
orientation at macro start; a fixed world target is then followed to completion.
OmniGibson imports remain in ESISession. Every simulator call belongs to main.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from core.action_units import BASE_ATOMS
from core.action_interface import BackendCapabilities
from core.observations import ControllerState
from core.feedback_semantics import SETTLED_MOTION_ERRORS

from wia.runtime.sim.esi_actions import r1_action_result
from wia.runtime.sim.esi_omnigibson import SHOWHARNESS_ROOT, _array

DEFAULT_BASE_CONFIG = SHOWHARNESS_ROOT / "configs" / "esi_base.json"
BASE_FRAME = {
    "forward": "+X", "left": "+Y", "up": "+Z",
    "positive_yaw": "counterclockwise viewed from above",
    "translation_reference": "current robot base frame at macro start",
}
TRANSLATION_DIRECTIONS = {"forward": (1.0, 0.0), "back": (-1.0, 0.0),
                          "left": (0.0, 1.0), "right": (0.0, -1.0)}
BASE_DIRECTIONS = dict(zip(BASE_ATOMS, ("forward", "back", "left", "right", "left", "right")))


@dataclass(frozen=True)
class BaseControlConfig:
    translation_m: float
    rotation_deg: float
    translation_tolerance_m: float
    rotation_tolerance_deg: float
    max_translation_speed_m_s: float
    max_rotation_speed_deg_s: float
    timeout_sim_s: float
    timeout_wall_s: float | None
    settle_steps: int
    stop_steps: int
    progress_timeout_sim_s: float = 1.5
    progress_min_translation_m: float = 0.001
    progress_min_rotation_deg: float = 0.1
    recovery_settle_steps: int = 6
    recovery_max_steps: int = 30
    recovery_max_joint_step_rad: float = 0.001
    recovery_max_finger_step_m: float = 0.0005
    recovery_max_base_step_m: float = 0.0005
    recovery_max_base_yaw_step_deg: float = 0.05

    def __post_init__(self):
        object.__setattr__(self, 'timeout_wall_s', None)
        for name, value in vars(self).items():
            if name == "timeout_wall_s" and value is None:
                continue
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Base configuration {name} must be finite and positive.")
        for name in ("settle_steps", "stop_steps", "recovery_settle_steps", "recovery_max_steps"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer.")
        if self.rotation_deg >= 180:
            raise ValueError("rotation_deg must be less than 180.")
        if (self.translation_tolerance_m >= self.translation_m
                or self.rotation_tolerance_deg >= self.rotation_deg):
            raise ValueError("Base tolerances must be smaller than the requested motion.")
        if self.recovery_max_steps < self.recovery_settle_steps:
            raise ValueError("Recovery must allow its required consecutive settling samples.")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_BASE_CONFIG) -> BaseControlConfig:
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
            # Historical launch files may still contain these inert keys.
            for key in ("inactive_joint_tolerance", "right_arm_hold_check"):
                values.pop(key, None)
            return cls(**values)


def _main_thread() -> None:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("R1 control and sensor updates must run on the main thread.")


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _rotation(yaw: float) -> np.ndarray:
    return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])


class R1BaseController:
    """Retain the upstream controllers and freeze inactive absolute joint targets.

    Supported layout is verified at construction rather than silently adapting
    an unknown controller. No reset, robot teleport or controller override occurs.
    """

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(base_translation=True, base_rotation=True)

    def _finish_result(self, result: dict) -> dict:
        self._last_control_result = result
        self._active_control = dict(getattr(self, "_active_control", {}),
                                    execution_status="stopped", action=result.get("action"))
        result = r1_action_result(result)
        if self.on_result is not None:
            self.on_result(result)
        return result

    def __init__(self, session, config: BaseControlConfig | None = None, *,
                 on_step: Callable[[], None] | None = None,
                 on_result: Callable[[dict], None] | None = None,
                 cartesian_arm: str | None = None):
        _main_thread()
        self.session = session
        self.robot = robot = session.robot
        self.config = config or BaseControlConfig.load()
        self.on_step = on_step
        self.on_result = on_result
        self._last_control_result = None
        self._active_control = {"execution_status": "idle", "action": None}
        self.cartesian_arm_group = f"arm_{cartesian_arm}" if cartesian_arm else None
        self.dt = 1.0 / float(session.env.env_config["action_frequency"])
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise RuntimeError("Invalid environment action frequency.")
        self.indices = {name: _array(idx).astype(int) for name, idx in
                        robot.controller_action_idx.items()}
        base = robot.controllers.get("base")
        if (base is None or type(base).__name__ != "HolonomicBaseJointController"
                or base.motor_type != "position" or base.command_dim != 3):
            raise RuntimeError("Base macros require the verified 3D HolonomicBaseJointController in position mode.")
        for name, ctrl in robot.controllers.items():
            cartesian = (name in ("arm_left", "arm_right")
                         and type(ctrl).__name__ == "InverseKinematicsController"
                         and ctrl.mode == "absolute_pose" and ctrl.command_dim == 6)
            if (ctrl.command_input_limits is not None or ctrl.motor_type != "position"
                    or ctrl.use_delta_commands
                    or (name == self.cartesian_arm_group and not cartesian)
                    or (name != "base" and type(ctrl).__name__ != "JointController" and not cartesian)):
                raise RuntimeError(f"Unsupported {name} controller semantics; inspect runtime before enabling base control.")
        covered = np.concatenate(list(self.indices.values()))
        if not np.array_equal(np.sort(covered), np.arange(robot.action_dim)):
            raise RuntimeError("Controller indices must cover the complete R1 action exactly once.")
        self.hold_action = np.empty(robot.action_dim, dtype=np.float32)
        control = robot.get_control_dict()
        for name, ctrl in robot.controllers.items():
            self.hold_action[self.indices[name]] = _array(ctrl.compute_no_op_action(control))
        if not np.isfinite(self.hold_action).all() or np.any(self.hold_action[self.indices["base"]]):
            raise RuntimeError("Controller-provided hold action is invalid.")
        joints = _array(robot.get_joint_positions())
        self.inactive_targets = {name: joints[_array(ctrl.dof_idx).astype(int)].copy()
                                 for name, ctrl in robot.controllers.items()
                                 if name not in ("base", self.cartesian_arm_group)}
        self.pose()  # Validate pose before the first physics step.

    def pose(self) -> dict:
        _main_thread()
        position, quaternion = (np.asarray(_array(v), dtype=float).copy()
                                for v in self.robot.get_position_orientation())
        if (position.shape != (3,) or quaternion.shape != (4,)
                or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
                or np.linalg.norm(quaternion) == 0):
            raise RuntimeError("R1 base pose must contain finite position and a valid xyzw quaternion.")
        x, y, z, w = quaternion / np.linalg.norm(quaternion)
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        return {"position": position.tolist(), "quaternion_xyzw": quaternion.tolist(), "yaw_rad": yaw}

    def controller_state(self) -> ControllerState:
        """Internal measured robot state and control targets, without task truth."""
        from wia.runtime.sim.esi_state import read_controller_state

        return read_controller_state(self.session, controller=self, working_arm=getattr(self, "working_arm", "left"))

    def step_base_debug(self, dx: float = 0.0, dy: float = 0.0, yaw_rad: float = 0.0) -> None:
        """One direct position-controller step; inputs are local offsets, not a macro.

        Use this for controller diagnostics only. Public tokens go through the
        closed-loop APIs below, which measure residual error after each env.step.
        """
        _main_thread()
        command = np.asarray([dx, dy, yaw_rad], dtype=np.float32)
        if not np.isfinite(command).all():
            raise ValueError("Base controller command must be finite.")
        action = self.hold_action.copy()
        action[self.indices["base"]] = command
        # Explicitly wake the articulation before driving it. The native
        # unrendered path can retain a sleeping robot after a blocked target.
        wake = getattr(self.robot, "wake", None)
        if callable(wake):
            wake()
        step = getattr(self.session, "step_control", self.session.env.step)
        step({self.robot.name: action})
        if self.on_step is not None:
            self.on_step()

    def stop(self) -> None:
        """Zero base offset plus the same frozen inactive targets, followed by settling."""
        _main_thread()
        for _ in range(self.config.stop_steps):
            self.step_base_debug()

    def base_joint_diagnostics(self) -> dict:
        """Private native encoder/limit evidence; no ideal odometry to the model."""
        limits = getattr(self.robot, "control_limits", {}).get("position")
        if limits is None:
            return {"available": False}
        dofs = _array(self.robot.controllers["base"].dof_idx).astype(int)
        q = _array(self.robot.get_joint_positions())[dofs]
        low, high = (_array(value)[dofs] for value in limits)
        finite = np.isfinite(low) & np.isfinite(high)
        margin = np.minimum(q-low, high-q)
        return {"available": True, "dof_indices": dofs.tolist(), "position": q.tolist(),
                "finite_limits": finite.tolist(),
                "margin": [float(value) if valid else None for value, valid in zip(margin, finite)],
                "near_limit": bool(np.any(finite & (margin <= 0.001)))}

    def verify_stopped_result(self, result: dict) -> None:
        """Measure a quiescent stop before one canonical action observation.

        Called by the executor after all outer arm/grasp guards have finalized.
        Hold existing targets; never replay a command, rebase joints or teleport.
        A transport failure or cancellation never enters this recovery path.
        """
        _main_thread()
        if (result.get("reason") not in SETTLED_MOTION_ERRORS or result.get("cancelled")
                or "stop_stability_verified" in result):
            return
        if result.get("recovery_completed") is False:
            result["stop_stability_verified"] = False
            return
        cfg = self.config
        started = time.monotonic()
        previous_pose = self.pose()
        previous_joints = _array(self.robot.get_joint_positions()).copy()
        angular_dofs = np.concatenate([_array(self.robot.controllers[name].dof_idx).astype(int)
                                      for name in ("trunk", "arm_left", "arm_right")])
        finger_dofs = np.concatenate([_array(self.robot.controllers[name].dof_idx).astype(int)
                                     for name in ("gripper_left", "gripper_right")])
        stable = steps = 0
        peak_joint = peak_finger = peak_base = peak_yaw = 0.0
        hard_limits_ok = True
        holding_before = self.gripper_state().get("holding") if hasattr(self, "gripper_state") else False
        lost_grasp = False
        while steps < cfg.recovery_max_steps and stable < cfg.recovery_settle_steps:
            self.step_base_debug()
            steps += 1
            current_pose = self.pose()
            current_joints = _array(self.robot.get_joint_positions()).copy()
            finite = np.isfinite(current_joints).all()
            joint_step = float(np.max(np.abs(current_joints[angular_dofs]-previous_joints[angular_dofs]))) if finite else float("inf")
            finger_step = float(np.max(np.abs(current_joints[finger_dofs]-previous_joints[finger_dofs]))) if finite else float("inf")
            base_step = float(np.linalg.norm(np.asarray(current_pose["position"])-previous_pose["position"]))
            yaw_step = abs(float(np.rad2deg(_wrap(current_pose["yaw_rad"]-previous_pose["yaw_rad"]))))
            peak_joint, peak_base, peak_yaw = max(peak_joint, joint_step), max(peak_base, base_step), max(peak_yaw, yaw_step)
            peak_finger = max(peak_finger, finger_step)
            limits = getattr(self.robot, "control_limits", {}).get("position")
            if limits is not None:
                low, high = (_array(value) for value in limits)
                hard_limits_ok = bool(finite and np.all(current_joints >= low-1e-4)
                                      and np.all(current_joints <= high+1e-4))
            lost_grasp |= bool(holding_before and not self.gripper_state().get("holding"))
            quiet = (finite and hard_limits_ok and not lost_grasp
                     and joint_step <= cfg.recovery_max_joint_step_rad
                     and finger_step <= cfg.recovery_max_finger_step_m
                     and base_step <= cfg.recovery_max_base_step_m
                     and yaw_step <= cfg.recovery_max_base_yaw_step_deg)
            stable = stable+1 if quiet else 0
            previous_pose, previous_joints = current_pose, current_joints
        result["stop_stability_verified"] = bool(stable >= cfg.recovery_settle_steps and not lost_grasp)
        result["stop_stability"] = {"schema": "measured_stop_v1", "steps": steps,
            "consecutive_quiet_samples": stable, "required_quiet_samples": cfg.recovery_settle_steps,
            "peak_joint_step_rad": peak_joint, "peak_finger_step_m": peak_finger,
            "peak_base_step_m": peak_base, "peak_yaw_step_deg": peak_yaw,
            "hard_limits_ok": hard_limits_ok, "grasp_lost": lost_grasp,
            "final_base_pose": previous_pose, "final_joint_positions": previous_joints.tolist()}
        result["control_steps"] = result.get("control_steps", 0)+steps
        result["stop_steps"] = result.get("stop_steps", 0)+steps
        result["sim_duration_s"] = result.get("sim_duration_s", 0.)+steps*self.dt
        result["duration_s"] = result.get("duration_s", 0.)+time.monotonic()-started
        # The boundary follows settling, so BASE feedback includes that final
        # measured pose instead of the earlier controller-stop pose.
        if result.get("motion") in ("translation", "rotation") and "pose_before" in result:
            before = result["pose_before"]
            delta = np.asarray(previous_pose["position"][:2])-before["position"][:2]
            yaw = float(np.rad2deg(_wrap(previous_pose["yaw_rad"]-before["yaw_rad"])))
            result.update(pose_after=previous_pose, actual_world_xy_m=delta.tolist(), actual_yaw_deg=yaw,
                position_error_m=float(np.linalg.norm(np.asarray(result["target_xy"])-previous_pose["position"][:2])),
                yaw_error_deg=float(np.rad2deg(_wrap(result["target_yaw_rad"]-previous_pose["yaw_rad"]))))
            if result["motion"] == "translation":
                axis = _rotation(before["yaw_rad"]) @ TRANSLATION_DIRECTIONS[result["direction"]]
                result["actual_m"] = float(delta @ axis)
            else:
                result["actual_deg"] = yaw*(1 if result["direction"] == "left" else -1)
        elif result.get("motion") in ("arm_translation", "arm_rotation") and "ee_pose_before" in result:
            from scipy.spatial.transform import Rotation
            before, after = result["ee_pose_before"], self.ee_pose()
            delta = np.asarray(after["base_position"])-before["base_position"]
            result.update(ee_pose_after=after, base_pose_after=previous_pose)
            target_rotation = Rotation.from_quat(result["target_base_quaternion_xyzw"])
            after_rotation = Rotation.from_quat(after["base_quaternion_xyzw"])
            result.update(position_error_m=float(np.linalg.norm(np.asarray(result["target_base_position"])-after["base_position"])),
                orientation_error_deg=float(np.rad2deg((target_rotation.inv()*after_rotation).magnitude())))
            if result["motion"] == "arm_translation":
                requested = np.asarray(result["requested_delta_m"])
                result.update(actual_delta_m=delta.tolist(), actual_m=float(delta @ (requested/np.linalg.norm(requested))))
            else:
                before_rotation = Rotation.from_quat(before["base_quaternion_xyzw"])
                signed = float(np.rad2deg((before_rotation.inv()*after_rotation).as_rotvec()[2]))
                result.update(signed_actual_deg=signed, actual_deg=float(signed*np.sign(result["signed_requested_deg"])))
        elif result.get("motion") == "trunk_translation" and hasattr(self, "trunk_controller"):
            trunk = self.trunk_controller
            before, _ = trunk.kinematics.pose(np.asarray(result["trunk_joints_before"]))
            after, _ = trunk.kinematics.pose(trunk.joints())
            delta = after-before
            signed_height_error = float(result["target_height_m"]-after[2])
            result.update(trunk_joints_after=trunk.joints().tolist(), trunk_height_after_m=float(after[2]),
                actual_delta_m=delta.tolist(), actual_m=float(delta[2]),
                signed_height_error_m=signed_height_error, absolute_height_error_m=abs(signed_height_error))
        if lost_grasp:
            result.update(completed=False, reason="grasp_lost")
        if hasattr(self, "gripper_state"):
            result["gripper_after"] = self.gripper_state()
        r1_action_result(result)

    def execute(self, token: str, *, should_cancel: Callable[[], bool] | None = None) -> dict:
        """The GUMI contract accepts fixed BASE tokens only, without parameters."""
        _main_thread()
        if token not in BASE_ATOMS:
            raise ValueError(f"Base executor rejects action {token!r}.")
        direction = BASE_DIRECTIONS[token]
        if token in BASE_ATOMS[:4]:
            return self.execute_base_translation(direction, self.config.translation_m,
                                                 should_cancel=should_cancel)
        return self.execute_base_rotation(direction, self.config.rotation_deg,
                                          should_cancel=should_cancel)

    def execute_base_translation(self, direction: str, distance_m: float, *,
                                 should_cancel: Callable[[], bool] | None = None) -> dict:
        _main_thread()
        if direction not in TRANSLATION_DIRECTIONS:
            raise ValueError(f"Unknown base translation direction: {direction!r}.")
        if not np.isfinite(distance_m) or distance_m <= 0:
            raise ValueError("distance_m must be finite and positive.")
        before = self.pose()
        axis = _rotation(before["yaw_rad"]) @ TRANSLATION_DIRECTIONS[direction]
        target_xy = np.asarray(before["position"][:2]) + axis * distance_m
        return self._execute("translation", direction, before, target_xy, before["yaw_rad"],
                             float(distance_m), axis, should_cancel)

    def execute_metric(self, token, magnitude, *, should_cancel=None):
        """Internal compiled-command interface; the same native guards and stop."""
        _main_thread()
        if token not in BASE_ATOMS:
            raise ValueError(f"Unsupported metric base action: {token!r}.")
        direction = BASE_DIRECTIONS[token]
        method = self.execute_base_translation if token in BASE_ATOMS[:4] else self.execute_base_rotation
        return method(direction, magnitude, should_cancel=should_cancel)

    def execute_base_rotation(self, direction: str, angle_deg: float, *,
                              should_cancel: Callable[[], bool] | None = None) -> dict:
        _main_thread()
        if direction not in ("left", "right"):
            raise ValueError(f"Unknown base rotation direction: {direction!r}.")
        if not np.isfinite(angle_deg) or not 0 < angle_deg < 180:
            raise ValueError("angle_deg must be finite and in (0, 180).")
        before = self.pose()
        sign = 1 if direction == "left" else -1
        target_yaw = _wrap(before["yaw_rad"] + sign * np.deg2rad(angle_deg))
        return self._execute("rotation", direction, before, np.asarray(before["position"][:2]),
                             target_yaw, float(angle_deg), sign, should_cancel)

    def _execute(self, motion, direction, before, target_xy, target_yaw, requested, axis,
                 should_cancel) -> dict:
        self._active_control = {"execution_status": "running", "action": direction,
                                "base_initial_pose": before,
                                "base_target_pose": {"position_xy": target_xy.tolist(), "yaw_rad": target_yaw}}
        cfg = self.config
        started = time.monotonic()
        control_steps = stable_steps = 0
        timeout = cancelled = False
        reason = "completed"
        progress_step = 0
        best_error = (float(np.linalg.norm(target_xy-before["position"][:2])) if motion == "translation"
                      else abs(float(np.rad2deg(_wrap(target_yaw-before["yaw_rad"])))))

        def measure():
            current = self.pose()
            error_xy = target_xy - current["position"][:2]
            error_yaw = _wrap(target_yaw - current["yaw_rad"])
            self._active_control["base_error"] = {"position_xy_m": error_xy.tolist(), "yaw_rad": error_yaw}
            return current, error_xy, error_yaw

        try:
            while True:
                current, error_xy, error_yaw = measure()
                if should_cancel is not None and should_cancel():
                    cancelled, reason = True, "cancelled"
                    break
                reached = (np.linalg.norm(error_xy) <= cfg.translation_tolerance_m
                           and abs(np.rad2deg(error_yaw)) <= cfg.rotation_tolerance_deg)
                stable_steps = stable_steps + 1 if reached else 0
                if stable_steps >= cfg.settle_steps:
                    break
                error = float(np.linalg.norm(error_xy)) if motion == "translation" else abs(float(np.rad2deg(error_yaw)))
                threshold = cfg.progress_min_translation_m if motion == "translation" else cfg.progress_min_rotation_deg
                if best_error-error >= threshold:
                    best_error, progress_step = error, control_steps
                elif not reached and (control_steps-progress_step)*self.dt >= cfg.progress_timeout_sim_s:
                    reason = "base_motion_blocked"
                    break
                if (control_steps * self.dt >= cfg.timeout_sim_s
                        or (cfg.timeout_wall_s is not None and time.monotonic() - started >= cfg.timeout_wall_s)):
                    timeout, reason = True, "timeout"
                    break
                # The target is fixed. Recompute its remaining local offset;
                # never accumulate the requested distance once per physics step.
                local_xy = _rotation(current["yaw_rad"]).T @ error_xy
                norm = np.linalg.norm(local_xy)
                limit = cfg.max_translation_speed_m_s * self.dt
                if norm > limit:
                    local_xy *= limit / norm
                yaw_limit = np.deg2rad(cfg.max_rotation_speed_deg_s) * self.dt
                self.step_base_debug(*local_xy, float(np.clip(error_yaw, -yaw_limit, yaw_limit)))
                control_steps += 1
        finally:
            # Includes cancellation, timeout, callback failure and interruption.
            self.stop()
        after, error_xy, error_yaw = measure()
        completed = (reason == "completed" and np.linalg.norm(error_xy) <= cfg.translation_tolerance_m
                     and abs(np.rad2deg(error_yaw)) <= cfg.rotation_tolerance_deg)
        if reason == "completed" and not completed:
            reason = "target_not_reached"
        delta_xy = np.asarray(after["position"][:2]) - before["position"][:2]
        actual_yaw = np.rad2deg(_wrap(after["yaw_rad"] - before["yaw_rad"]))
        result = {
            "action": next(token for token in BASE_ATOMS
                           if BASE_DIRECTIONS[token] == direction
                           and (token in BASE_ATOMS[:4]) == (motion == "translation")),
            "motion": motion, "direction": direction, "frame": "robot_base",
            "pose_before": before, "pose_after": after,
            "target_xy": target_xy.tolist(), "target_yaw_rad": target_yaw,
            "actual_world_xy_m": delta_xy.tolist(), "actual_yaw_deg": float(actual_yaw),
            "position_error_m": float(np.linalg.norm(error_xy)),
            "yaw_error_deg": float(np.rad2deg(error_yaw)),
            "duration_s": time.monotonic() - started,
            "control_steps": control_steps + cfg.stop_steps, "stop_steps": cfg.stop_steps,
            "sim_duration_s": (control_steps + cfg.stop_steps) * self.dt,
            "completed": bool(completed), "timeout": timeout, "cancelled": cancelled,
            "reason": reason,
            "base_joint_diagnostics": self.base_joint_diagnostics(),
        }
        if motion == "translation":
            result.update(requested_m=requested, actual_m=float(delta_xy @ axis))
        else:
            result.update(requested_deg=requested, actual_deg=float(actual_yaw * axis))
        return self._finish_result(result)
