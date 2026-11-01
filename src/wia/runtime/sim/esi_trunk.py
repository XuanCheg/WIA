"""Fixed upper-body height macros using measured joints and URDF kinematics."""
from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from wia.runtime.sim.esi_base import _main_thread
from wia.runtime.sim.esi_omnigibson import _array


@dataclass(frozen=True)
class TrunkControlConfig:
    translation_m: float = 0.02
    translation_tolerance_m: float = 0.003
    orientation_tolerance_deg: float = 1.0
    max_joint_speed_rad_s: float = 0.2
    joint_limit_margin_rad: float = 0.04
    # Nominal URDF workspace for upright, fixed-horizontal trunk motion.
    # Calibrated torso_link4 height is reported separately; this is not a global pose minimum.
    minimum_height_m: float = 0.58
    maximum_height_m: float = 1.08
    timeout_sim_s: float = 120.0
    timeout_wall_s: float | None = None
    settle_steps: int = 6

    def __post_init__(self):
        object.__setattr__(self, 'timeout_wall_s', None)
        if any(isinstance(v, bool) or not np.isfinite(v) or v <= 0 for name, v in vars(self).items() if not (name == "timeout_wall_s" and v is None)):
            raise ValueError("Trunk control settings must be finite and positive.")
        if (self.minimum_height_m >= self.maximum_height_m or self.translation_tolerance_m >= self.translation_m
                or type(self.settle_steps) is not int):
            raise ValueError("Invalid trunk height range, tolerance or settling count.")

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        for key in ("horizontal_tolerance_m", "arm_hold_joint_tolerance_rad"):
            values.pop(key, None)
        return cls(**values)


class TrunkKinematics:
    """Four native torso revolute joints, base_link → torso_link4; no scene data."""
    def __init__(self, urdf):
        root = ET.parse(urdf).getroot()
        self.origins, self.axes, lower, upper = [], [], [], []
        parent = "base_link"
        for index in range(1, 5):
            joint = root.find(f"joint[@name='torso_joint{index}']")
            if (joint is None or joint.get("type") != "revolute"
                    or joint.find("parent").get("link") != parent):
                raise ValueError("Height control requires the verified four-joint R1 trunk chain.")
            parent = joint.find("child").get("link")
            origin = joint.find("origin")
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler("xyz", list(map(float, origin.get("rpy", "0 0 0").split()))).as_matrix()
            transform[:3, 3] = list(map(float, origin.get("xyz", "0 0 0").split()))
            axis = np.array(list(map(float, joint.find("axis").get("xyz").split())))
            self.origins.append(transform)
            self.axes.append(axis / np.linalg.norm(axis))
            lower.append(float(joint.find("limit").get("lower")))
            upper.append(float(joint.find("limit").get("upper")))
        if parent != "torso_link4":
            raise ValueError("Unexpected R1 trunk endpoint.")
        self.lower, self.upper = np.array(lower), np.array(upper)

    def pose(self, joints):
        transform = np.eye(4)
        for origin, axis, angle in zip(self.origins, self.axes, joints, strict=True):
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            transform = transform @ origin @ motion
        return transform[:3, 3], Rotation.from_matrix(transform[:3, :3])

    def solve(self, joints, delta_m, lower, upper):
        position, orientation = self.pose(joints)
        target = position + np.array([0., 0., delta_m])

        def residual(q):
            p, r = self.pose(q)
            return np.r_[p-target, 0.2*(orientation.inv()*r).as_rotvec()]

        solution = least_squares(residual, joints, bounds=(lower, upper),
                                 max_nfev=100, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        return solution.x, target, orientation


def bounded_joint_step(current, target, maximum):
    """Synchronize the coupled joints; independent clipping tilts the torso."""
    error = np.asarray(target)-current
    scale = min(1., maximum / max(float(np.max(np.abs(error))), 1e-12))
    return error * scale


class R1TrunkController:
    """Shares the established controller, executor and main-thread physics loop."""
    def __init__(self, controller, config):
        _main_thread()
        from wia.tasks.native_retry import install_sampling_keepalive

        install_sampling_keepalive(controller)
        self.owner, self.config = controller, config
        robot = controller.robot
        trunk = robot.controllers["trunk"]
        if (trunk.command_dim != 4 or trunk.motor_type != "position" or trunk.use_delta_commands
                or type(trunk).__name__ != "JointController"):
            raise RuntimeError("Trunk height control needs native absolute-position joint control.")
        self.dofs = _array(trunk.dof_idx).astype(int)
        self.kinematics = TrunkKinematics(robot.urdf_path)
        from wia.runtime.sim.esi_camera_geometry import calibrated_robot_kinematics
        self.height_kinematics, _ = calibrated_robot_kinematics(robot,
            joint_names={f"torso_joint{i}" for i in range(1, 5)})
        native_lower, native_upper = (_array(x)[self.dofs] for x in robot.control_limits["position"])
        self.lower = np.maximum(native_lower, self.kinematics.lower) + config.joint_limit_margin_rad
        self.upper = np.minimum(native_upper, self.kinematics.upper) - config.joint_limit_margin_rad
        if np.any(self.lower >= self.upper):
            raise RuntimeError("No valid trunk joint workspace.")
        self.joints()

    def joints(self):
        q = _array(self.owner.robot.get_joint_positions())[self.dofs].astype(float)
        if q.shape != (4,) or not np.isfinite(q).all():
            raise RuntimeError("Invalid native trunk readback.")
        return q

    def measured_height(self, joints=None):
        """Installed torso_link4 height in robot_base, from the same encoders."""
        q = self.joints() if joints is None else joints
        frame = self.height_kinematics.frames(
            dict(zip((f"torso_joint{i}" for i in range(1, 5)), q, strict=True)),
            links={"torso_link4"})["torso_link4"]
        return float(frame[2, 3])

    def execute(self, token, *, should_cancel=None):
        return self.execute_metric(token, self.config.translation_m, should_cancel=should_cancel)

    def execute_metric(self, token, distance_m, *, should_cancel=None):
        _main_thread()
        if token not in ("TRUNK_UP", "TRUNK_DOWN"):
            raise ValueError("Trunk controller accepts only fixed height tokens.")
        if not np.isfinite(distance_m) or distance_m <= 0:
            raise ValueError("Trunk distance must be finite and positive.")
        owner, cfg = self.owner, self.config
        started = time.monotonic()
        delta = distance_m * (1 if token == "TRUNK_UP" else -1)
        initial_joints = self.joints()
        before, _ = self.kinematics.pose(initial_joints)
        requested_height = float(before[2] + delta)
        bounded_height = float(np.clip(requested_height, cfg.minimum_height_m, cfg.maximum_height_m))
        # Saturation must not reverse a request if readback is just outside a bound.
        target_height = min(float(before[2]), bounded_height) if delta < 0 else max(float(before[2]), bounded_height)
        target_clamped = not np.isclose(target_height, requested_height, rtol=0., atol=1e-12)
        effective_delta = target_height - float(before[2])
        target_position = before + [0, 0, effective_delta]
        reason, timeout, cancelled = "completed", False, False
        steps = stable = 0
        base_before = owner.pose()
        arm_dofs = _array(owner.robot.controllers[owner.arm_group].dof_idx).astype(int)
        arm_joints = _array(owner.robot.get_joint_positions())[arm_dofs].copy()
        owner._active_control = {"execution_status": "running", "action": token, "base_initial_pose": base_before}
        if np.any(initial_joints < self.lower) or np.any(initial_joints > self.upper):
            reason = "trunk_height_limit"
            solution = initial_joints
        else:
            solution, target_position, _ = self.kinematics.solve(initial_joints, effective_delta, self.lower, self.upper)
            solved_position, _ = self.kinematics.pose(solution)
            if abs(float(solved_position[2]-target_position[2])) > cfg.translation_tolerance_m / 3:
                reason = "trunk_height_limit"
        owner.ik_guard.explicit_joint_hold = arm_joints
        try:
            while reason == "completed":
                joints = self.joints()
                position, orientation = self.kinematics.pose(joints)
                if should_cancel is not None and should_cancel():
                    cancelled, reason = True, "cancelled"
                    break
                reached = abs(float(target_position[2]-position[2])) <= cfg.translation_tolerance_m
                stable = stable + 1 if reached else 0
                if stable >= cfg.settle_steps:
                    break
                if steps * owner.dt >= cfg.timeout_sim_s or (cfg.timeout_wall_s is not None and time.monotonic()-started >= cfg.timeout_wall_s):
                    timeout, reason = True, "timeout"
                    break
                joint_step = bounded_joint_step(joints, solution, cfg.max_joint_speed_rad_s*owner.dt)
                owner.hold_action[owner.indices["trunk"]] = joints + joint_step
                owner.step_base_debug()
                steps += 1
        finally:
            current = self.joints()
            owner.hold_action[owner.indices["trunk"]] = current
            owner.inactive_targets["trunk"] = current.copy()
            owner.stop()
            pose = owner.ee_pose()
            owner._set_arm_target(pose["base_position"], pose["base_quaternion_xyzw"])
            owner.ik_guard.reset()
        after, rotation = self.kinematics.pose(self.joints())
        movement = after-before
        signed_height_error = float(target_position[2]-after[2])
        absolute_height_error = abs(signed_height_error)
        if reason == "completed" and absolute_height_error > cfg.translation_tolerance_m:
            reason = "target_not_reached"
        return owner._finish_result({"action": token, "motion": "trunk_translation", "frame": "robot_base",
            "completed": reason == "completed", "reason": reason, "timeout": timeout, "cancelled": cancelled,
            "requested_m": delta, "requested_delta_m": [0., 0., delta], "actual_m": float(movement[2]),
            "actual_delta_m": movement.tolist(),
            "signed_height_error_m": signed_height_error, "absolute_height_error_m": absolute_height_error,
            "trunk_height_before_m": float(before[2]), "trunk_height_after_m": float(after[2]),
            "requested_height_m": requested_height, "target_clamped": bool(target_clamped),
            "target_height_m": float(target_position[2]), "trunk_joints_before": initial_joints.tolist(),
            "trunk_joints_after": self.joints().tolist(),
            "control_steps": steps+owner.config.stop_steps, "stop_steps": owner.config.stop_steps, "recovery_completed": None,
            "duration_s": time.monotonic()-started, "sim_duration_s": (steps+owner.config.stop_steps)*owner.dt})
