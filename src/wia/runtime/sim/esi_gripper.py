"""Measured gripper commands using OmniGibson's assisted grasp.

Run through src/wia/runtime/commands/run_esi_robot.py --gripper-control --web.

Retain the verified native absolute finger JointController. Attach/detach is
performed by OmniGibson after contact; no object teleport or manual constraint.
Only debug results contain held object identity and constraint information.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from core.action_units import GRASP_ATOM, GRIPPER_ATOMS

from wia.runtime.sim.esi_arm import R1ArmController
from wia.runtime.sim.esi_base import _main_thread
from wia.runtime.sim.esi_omnigibson import SHOWHARNESS_ROOT, _array


@dataclass(frozen=True)
class GripperControlConfig:
    grasp_mode: str = "assisted"
    joint_tolerance_m: float = 0.001
    minimum_settle_sim_s: float = 0.4
    settle_steps: int = 6
    timeout_sim_s: float = 120.0
    timeout_wall_s: float | None = None

    def __post_init__(self):
        object.__setattr__(self, 'timeout_wall_s', None)
        if self.grasp_mode != "assisted":
            raise ValueError("Milestone 4 uses assisted grasp only.")
        for name, value in vars(self).items():
            if name == "timeout_wall_s" and value is None:
                continue
            if name != "grasp_mode" and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
                raise ValueError(f"Gripper configuration {name} must be finite and positive.")
        if not isinstance(self.settle_steps, int) or self.minimum_settle_sim_s >= self.timeout_sim_s:
            raise ValueError("Gripper settling must finish before its timeout.")

    @classmethod
    def load(cls, path=SHOWHARNESS_ROOT / "configs/esi_gripper.json"):
        return cls(**json.loads(Path(path).read_text()))


class R1ManipulationController(R1ArmController):
    """Fixed BASE/MV/gripper macros with opt-in wrist yaw; grasp holds the robot."""

    @property
    def capabilities(self):
        return replace(super().capabilities, gripper=True)

    def __init__(self, session, config=None, *, gripper_config=None, **kwargs):
        self.gripper_config = gripper_config or GripperControlConfig.load()
        if session.robot.grasping_mode != self.gripper_config.grasp_mode:
            raise RuntimeError("Enable assisted grasp in the robot config before constructing the environment.")
        super().__init__(session, config, **kwargs)
        self.gripper_group = f"gripper_{self.working_arm}"
        ctrl = self.robot.controllers[self.gripper_group]
        self.gripper_dofs = _array(ctrl.dof_idx).astype(int)
        fingers = self.robot.finger_joint_names[self.working_arm]
        if (type(ctrl).__name__ != "JointController" or len(self.gripper_dofs) != 2
                or len(fingers) != 2 or ctrl.command_dim != 2
                or any("prismatic" not in str(self.robot.joints[name].joint_type).lower() for name in fingers)):
            raise RuntimeError("Gripper macros require two verified prismatic absolute-position finger joints.")
        self.closed_target, self.open_target = [
            _array(limits)[self.gripper_dofs].copy() for limits in self.robot.control_limits["position"]]
        if (not np.isfinite(self.closed_target).all() or not np.isfinite(self.open_target).all()
                or np.any(self.open_target <= self.closed_target)):
            raise RuntimeError("Invalid measured finger joint limits.")
        self.gripper_command = "HOLD"

    def gripper_state(self):
        _main_thread()
        positions = _array(self.robot.get_joint_positions())[self.gripper_dofs]
        if not np.isfinite(positions).all():
            raise RuntimeError("Finger joint readback must be finite.")
        # Native IsGraspingState: TRUE=1, UNKNOWN=0, FALSE=-1.
        # Casting to bool would incorrectly report FALSE as holding.
        reading = int(self.robot.is_grasping(self.working_arm))
        holding = {-1: False, 1: True}.get(reading)
        closed = bool(holding or np.max(np.abs(positions - self.closed_target))
                      <= self.gripper_config.joint_tolerance_m)
        opened = bool(np.max(np.abs(positions - self.open_target))
                      <= self.gripper_config.joint_tolerance_m)
        return {"grasp_mode": self.gripper_config.grasp_mode,
                "gripper_width_m": float(np.sum(positions)),
                "gripper_command_name": self.gripper_command,
                "gripper_closed": closed, "gripper_open": opened, "holding": holding}

    def debug_gripper_state(self):
        """Privileged identity/attachment stays in execution logs, never observation."""
        state = self.gripper_state()
        obj = self.robot._ag_obj_in_hand[self.working_arm]
        state.update(object_name=obj.name if obj is not None else None,
                     constraint_active=self.robot._ag_obj_constraints[self.working_arm] is not None,
                     release_pending=self.robot._ag_release_counter[self.working_arm] is not None)
        return state

    def proprioception(self):
        return dict(super().proprioception(), **self.gripper_state())

    def execute(self, token, *, should_cancel=None):
        _main_thread()
        if token in GRIPPER_ATOMS:
            return self.set_gripper_debug(token == GRASP_ATOM, should_cancel=should_cancel)
        return self._guarded_motion(lambda cancel: super(R1ManipulationController, self).execute(
            token, should_cancel=cancel), should_cancel)

    def execute_metric(self, token, magnitude, *, should_cancel=None):
        _main_thread()
        return self._guarded_motion(lambda cancel: super(R1ManipulationController, self).execute_metric(
            token, magnitude, should_cancel=cancel), should_cancel)

    def _guarded_motion(self, execute_motion, should_cancel):
        """Fixed and metric motions share grasp monitoring, callbacks and stop."""
        before = self.gripper_state()
        previous_step, previous_result = self.on_step, self.on_result
        lost = False

        def monitor():
            nonlocal lost
            lost |= before["holding"] is True and self.gripper_state()["holding"] is False
            if previous_step is not None:
                previous_step()

        def cancelled():
            return lost or bool(should_cancel and should_cancel())

        try:
            self.on_step, self.on_result = monitor, None
            result = execute_motion(cancelled)
        finally:
            self.on_step, self.on_result = previous_step, previous_result
        result.update(grasp_mode="assisted", gripper_before=before, gripper_after=self.gripper_state())
        if lost:
            result.update(completed=False, reason="grasp_lost", cancelled=bool(should_cancel and should_cancel()))
        result["debug_gripper_after"] = self.debug_gripper_state()
        return self._finish_result(result)

    def set_gripper_debug(self, closed, *, should_cancel=None):
        """Direct macro first; the public vocabulary supplies only an explicit bool."""
        _main_thread()
        cfg = self.gripper_config
        before, base_before, ee_before = self.debug_gripper_state(), self.pose(), self.ee_pose()
        action = "GRASP" if closed else "RELEASE"
        self._active_control = {"execution_status": "running", "action": action,
                                "base_initial_pose": base_before, "arm_initial_pose": ee_before}
        noop = (before["holding"] is True or before["gripper_closed"]) if closed else (
            before["gripper_open"] and before["object_name"] is None)
        steps = stable = 0
        timeout = cancelled = False
        reason = "completed"
        started = time.monotonic()
        self.gripper_command = action
        if should_cancel and should_cancel():
            cancelled, reason = True, "cancelled"
        if not noop and not cancelled:
            # Native assisted release observes the controller's full-open
            # target. Ramping from measured fingers would stall while native
            # attachment freezes them. Native motor velocity limits still apply.
            self.hold_action[self.indices[self.gripper_group]] = self.closed_target if closed else self.open_target

        def measure():
            return self.debug_gripper_state()

        try:
            while not noop and not cancelled:
                current = measure()
                if should_cancel and should_cancel():
                    cancelled, reason = True, "cancelled"
                    break
                # Closing is a finite motor command, not an attachment check.
                # Return the measured aperture even when contact prevents full
                # closure, so the next decision can inspect the resulting pose.
                if closed and steps * self.dt >= cfg.minimum_settle_sim_s:
                    break
                reached = current["gripper_open"] and current["object_name"] is None
                stable = stable + 1 if reached else 0
                if not closed and stable >= cfg.settle_steps and steps * self.dt >= cfg.minimum_settle_sim_s:
                    break
                if not closed and (steps * self.dt >= cfg.timeout_sim_s or (cfg.timeout_wall_s is not None and time.monotonic() - started >= cfg.timeout_wall_s)):
                    timeout, reason = True, "gripper_timeout"
                    break
                self.step_base_debug()
                steps += 1
        except BaseException:
            reason = "execution_exception"
            raise
        finally:
            if reason != "completed":
                self.hold_action[self.indices[self.gripper_group]] = _array(
                    self.robot.get_joint_positions())[self.gripper_dofs]
                self._freeze_arm_pose()
            # Keep measured command bookkeeping aligned with the active gripper.
            self.inactive_targets[self.gripper_group] = _array(self.robot.get_joint_positions())[self.gripper_dofs].copy()
            self.stop()
        after = measure()
        if reason == "completed":
            if closed:
                reason = "already_holding" if noop and after["holding"] is True else "completed"
            elif not after["gripper_open"] or after["object_name"] is not None:
                reason = "release_incomplete"
            elif noop:
                reason = "already_open"
        result = {
            "action": action, "motion": "gripper", "frame": "robot_base", "working_arm": self.working_arm,
            "grasp_mode": "assisted", "noop": bool(noop),
            "completed": reason in ("completed", "already_holding", "already_open"),
            "reason": reason, "timeout": timeout, "cancelled": cancelled,
            "requested_state": "closed" if closed else "open",
            "gripper_before": {k: v for k, v in before.items() if k not in ("object_name", "constraint_active", "release_pending")},
            "gripper_after": self.gripper_state(),
            "debug_gripper_before": before, "debug_gripper_after": after,
            "base_pose_before": base_before, "base_pose_after": self.pose(),
            "ee_pose_before": ee_before, "ee_pose_after": self.ee_pose(),
            "control_steps": steps + self.config.stop_steps, "stop_steps": self.config.stop_steps,
            "duration_s": time.monotonic() - started, "sim_duration_s": (steps + self.config.stop_steps) * self.dt,
            "ik_diagnostics": self.ik_guard.diagnostics(),
        }
        return self._finish_result(result)
