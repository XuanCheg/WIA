"""Bound native IK output before position targets reach the physical joints."""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from core.feedback_semantics import IK_REJECTION_ERRORS

from wia.runtime.sim.esi_base import _main_thread
from wia.runtime.sim.esi_omnigibson import _array


class NativeIKGuard:
    """Keep the native solver; reject jumps/limits and cap joint target speed.

    The wrapper executes inside the controller callback on main, before PhysX
    receives its position targets. A rejected solution latches a joint hold.
    A later arm macro may clear that latch to move away from the constraint.
    """

    def __init__(self, controller, limits, physics_dt, config):
        _main_thread()
        self.controller, self.config = controller, config
        self.original_compute = controller.compute_control
        while (isinstance(getattr(self.original_compute, "__self__", None), NativeIKGuard)
               and getattr(self.original_compute, "__func__", None) is NativeIKGuard.compute_control):
            self.original_compute = self.original_compute.__self__.original_compute
        self.indices = _array(controller.dof_idx).astype(int)
        self.lower, self.upper = [np.asarray(_array(x)[self.indices], dtype=float)
                                 for x in limits["position"]]
        velocity = np.asarray(_array(limits["velocity"][1])[self.indices], dtype=float)
        self.max_step = np.minimum(velocity, config.max_joint_speed_rad_s) * physics_dt
        if not np.isfinite(self.max_step).all() or np.any(self.max_step <= 0):
            raise RuntimeError("IK guard requires positive finite joint speed limits and physics timestep.")
        self.reset()
        controller.compute_control = self.compute_control

    def reset(self, recovery_joint_position=None):
        _main_thread()
        self.fault = None
        self.hold = None
        self.native_reference = None
        self.recovery = (_array(recovery_joint_position).copy()
                         if recovery_joint_position is not None else None)
        self.explicit_joint_hold = None
        self.damped_recovery_steps = 0
        self.max_native_joint_step_rad = 0.0
        self.min_joint_margin_rad = None
        self.min_jacobian_singular_value = None

    def freeze(self, joint_position, *, preserve_recovery=False):
        if self.fault is not None:
            # Public robot readback can be Torch while the control backend is
            # NumPy. Return the native callback's dtype/backend for clipping.
            # A rejected proposal never requests a return to the macro start.
            values = (self.recovery if (preserve_recovery and self.recovery is not None
                                       and self.fault not in IK_REJECTION_ERRORS)
                      else _array(joint_position))
            if isinstance(self.native_reference, np.ndarray):
                self.hold = np.array(values, dtype=self.native_reference.dtype, copy=True)
            else:
                self.hold = self.native_reference.new_tensor(values)

    def diagnostics(self):
        return {"fault": self.fault, "damped_recovery_steps": self.damped_recovery_steps, "max_native_joint_step_rad": self.max_native_joint_step_rad,
                "min_joint_margin_rad": self.min_joint_margin_rad,
                "min_jacobian_singular_value": self.min_jacobian_singular_value,
                "joint_hold": self.hold is not None}

    def _bounded(self, current, target):
        delta = np.asarray(_array(target - current), dtype=float)
        scale = min(1.0, float(np.min(self.max_step / np.maximum(np.abs(delta), 1e-12))))
        return current + (target - current) * scale

    def _damped_candidate(self, goal, state, jac, current):
        """Solve a local pose correction inside joint-position and velocity bounds."""
        task = self.controller.task_name
        required = (f"{task}_pos_relative", f"{task}_quat_relative")
        try:
            current_position = state[required[0]]
            current_quaternion = state[required[1]]
            target_position = goal["target_pos"]
            target_orientation = goal["target_ori_mat"]
        except KeyError:
            return None
        q = np.asarray(_array(current), dtype=float)
        position_error = _array(target_position) - _array(current_position)
        current_rotation = Rotation.from_quat(_array(current_quaternion)).as_matrix()
        rotation_error = Rotation.from_matrix(_array(target_orientation) @ current_rotation.T).as_rotvec()
        weight = np.array([1., 1., 1., .2, .2, .2])
        error = np.r_[position_error, rotation_error] * weight
        matrix = np.asarray(jac, dtype=float) * weight[:, None]
        if not np.isfinite(error).all() or not np.isfinite(matrix).all():
            return None
        soft_lower = self.lower + self.config.joint_limit_margin_rad
        soft_upper = self.upper - self.config.joint_limit_margin_rad
        lower = np.maximum(-self.max_step, np.minimum(0., soft_lower-q))
        upper = np.minimum(self.max_step, np.maximum(0., soft_upper-q))
        active = upper-lower > 1e-12
        if not active.any():
            return None
        singular = float(np.linalg.svd(matrix, compute_uv=False)[-1])
        threshold = self.config.ik_damping_singular_value
        damping = max(.001, self.config.ik_damping_max * max(0., 1.-singular/threshold))
        reduced = matrix[:, active]
        solution = lsq_linear(np.vstack([reduced, damping*np.eye(active.sum())]),
                              np.r_[error, np.zeros(active.sum())],
                              bounds=(lower[active], upper[active]), tol=1e-10, max_iter=100)
        delta = np.zeros_like(q)
        delta[active] = solution.x
        if (not solution.success or not np.isfinite(delta).all()
                or np.linalg.norm(delta) < 1e-10
                or np.linalg.norm(error-matrix@delta) >= np.linalg.norm(error)-1e-12):
            return None
        self.damped_recovery_steps += 1
        return (np.asarray(q+delta, dtype=current.dtype) if isinstance(current, np.ndarray)
                else current.new_tensor(q+delta))

    def compute_control(self, goal_dict, control_dict):
        _main_thread()
        current = control_dict["joint_position"][self.controller.dof_idx]
        self.native_reference = current
        if self.explicit_joint_hold is not None:
            values = self.explicit_joint_hold
            target = (np.asarray(values, dtype=current.dtype) if isinstance(current, np.ndarray)
                      else current.new_tensor(values))
            return self._bounded(current, target)
        if self.hold is not None:
            return self._bounded(current, self.hold)
        q = np.asarray(_array(current), dtype=float)
        margin = float(np.min(np.minimum(q - self.lower, self.upper - q)))
        self.min_joint_margin_rad = (margin if self.min_joint_margin_rad is None
                                     else min(self.min_joint_margin_rad, margin))
        jac = _array(control_dict[f"{self.controller.task_name}_jacobian_relative"])[:, self.indices]
        if not np.isfinite(jac).all():
            self.fault = "ik_nonfinite"
            self.freeze(current, preserve_recovery=True)
            return self._bounded(current, self.hold)
        sigma = float(np.linalg.svd(jac, compute_uv=False)[-1])
        self.min_jacobian_singular_value = (sigma if self.min_jacobian_singular_value is None
                                           else min(self.min_jacobian_singular_value, sigma))
        try:
            candidate = self.original_compute(goal_dict, control_dict)
        except (RuntimeError, np.linalg.LinAlgError):
            self.fault = "ik_solver_failure"
            self.freeze(current, preserve_recovery=True)
            return self._bounded(current, self.hold)
        target = np.asarray(_array(candidate), dtype=float)
        if not np.isfinite(target).all():
            self.fault = "ik_nonfinite"
        else:
            delta = target - q
            largest = float(np.max(np.abs(delta)))
            self.max_native_joint_step_rad = max(self.max_native_joint_step_rad, largest)
            soft_lower = self.lower + self.config.joint_limit_margin_rad
            soft_upper = self.upper - self.config.joint_limit_margin_rad
            outward = (((target < soft_lower) & (delta < -1e-7))
                       | ((target > soft_upper) & (delta > 1e-7)))
            needs_recovery = (np.any(outward) or largest > self.config.max_ik_joint_step_rad
                              or sigma < self.config.ik_damping_singular_value)
            if needs_recovery:
                damped = self._damped_candidate(goal_dict, control_dict, jac, current)
                if damped is not None:
                    return self._bounded(current, damped)
            if np.any(outward):
                self.fault = "arm_joint_limit"
            elif largest > self.config.max_ik_joint_step_rad:
                self.fault = "ik_joint_jump"
            else:
                return self._bounded(current, candidate)
        self.freeze(current, preserve_recovery=True)
        return self._bounded(current, self.hold)
