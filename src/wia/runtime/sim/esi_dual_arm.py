"""Sequential dual-arm R1 control with shared physics and independent guard state.

Run through ``python -m wia.runtime.commands.run_esi_robot`` with arm and gripper
control enabled. Existing unprefixed commands address the left arm; RIGHT_*
commands address the right arm. No action resets or teleports the robot.
"""
from contextlib import contextmanager
from dataclasses import replace

import numpy as np

from core.action_units import RIGHT_ARM_ATOMS, arm_primitive
from wia.runtime.sim.esi_gripper import R1ManipulationController
from wia.runtime.sim.esi_ik import NativeIKGuard
from wia.runtime.sim.esi_omnigibson import _array
from wia.runtime.sim.esi_state import read_controller_state


class R1DualManipulationController(R1ManipulationController):
    """Reuse the guarded single-arm loops without moving the inactive arm."""

    _additional_ik_arms = ("right",)

    _arm_fields = ("working_arm", "arm_group", "cartesian_arm_group", "ik_guard",
                   "gripper_group", "gripper_dofs", "closed_target", "open_target",
                   "gripper_command", "_arm_target_pose")

    @property
    def capabilities(self):
        return replace(super().capabilities, right_ee_translation=True,
                       right_ee_rotation=self.rotation_config is not None,
                       right_ee_tilt=self.rotation_config is not None, right_gripper=True)

    def __init__(self, session, config=None, **kwargs):
        robot = session.robot
        super().__init__(session, config, **kwargs)
        if self.working_arm != "left":
            raise ValueError("Unprefixed commands must remain bound to the left arm.")
        right = robot.controllers["arm_right"]
        if (not np.array_equal(_array(right.dof_idx), _array(robot.arm_control_idx["right"]))
                or right.command_output_limits is not None):
            raise RuntimeError("Right-arm IK must control only its own unscaled joints.")
        physics_dt = (session.pipeline.og.sim.get_physics_dt()
                      if getattr(session, "pipeline", None) is not None else self.dt)
        guard = NativeIKGuard(right, robot.control_limits, physics_dt, self.arm_config)
        gripper = robot.controllers["gripper_right"]
        dofs = _array(gripper.dof_idx).astype(int)
        fingers = robot.finger_joint_names["right"]
        if (type(gripper).__name__ != "JointController" or len(dofs) != 2 or len(fingers) != 2
                or gripper.command_dim != 2
                or any("prismatic" not in str(robot.joints[name].joint_type).lower() for name in fingers)):
            raise RuntimeError("Right gripper requires two absolute-position prismatic finger joints.")
        closed, opened = [_array(x)[dofs].copy() for x in robot.control_limits["position"]]
        if not np.isfinite(closed).all() or not np.isfinite(opened).all() or np.any(opened <= closed):
            raise RuntimeError("Invalid right gripper joint limits.")
        self._arm_target_pose = getattr(self, "_arm_target_pose", None)
        self._arm_contexts = {"left": self._save_arm(), "right": dict(
            working_arm="right", arm_group="arm_right", cartesian_arm_group="arm_right",
            ik_guard=guard, gripper_group="gripper_right", gripper_dofs=dofs,
            closed_target=closed, open_target=opened, gripper_command="HOLD", _arm_target_pose=None)}
        guard.explicit_joint_hold = _array(robot.get_joint_positions())[guard.indices].copy()
        self.inactive_targets.pop("arm_right", None)

    def _save_arm(self):
        return {key: getattr(self, key, None) for key in self._arm_fields}

    @contextmanager
    def _select_arm(self, arm):
        previous = self.working_arm
        if arm == previous:
            yield
            return
        self._arm_contexts[previous] = self._save_arm()
        for key, value in self._arm_contexts[arm].items():
            setattr(self, key, value)
        try:
            yield
        finally:
            self._arm_contexts[arm] = self._save_arm()
            for key, value in self._arm_contexts[previous].items():
                setattr(self, key, value)

    def _right_action(self, token, magnitude=None, *, should_cancel=None):
        # The other hand retains measured joint targets, including through
        # trunk motion and recovery. Both controllers use one shared hold action.
        left_guard = self.ik_guard
        prior_hold = left_guard.explicit_joint_hold
        left_guard.explicit_joint_hold = _array(self.robot.get_joint_positions())[left_guard.indices].copy()
        sink = self.on_result
        self.on_result = None
        try:
            with self._select_arm("right"):
                self.ik_guard.explicit_joint_hold = None
                try:
                    primitive = arm_primitive(token)
                    result = (super().execute(primitive, should_cancel=should_cancel) if magnitude is None
                              else super().execute_metric(primitive, magnitude, should_cancel=should_cancel))
                finally:
                    self.ik_guard.explicit_joint_hold = _array(
                        self.robot.get_joint_positions())[self.ik_guard.indices].copy()
                result["action"] = token
                result["working_arm"] = "right"
                self._last_control_result = result
                self._active_control = dict(self._active_control, action=token, execution_status="stopped")
        finally:
            left_guard.explicit_joint_hold = prior_hold
            self.on_result = sink
        if sink is not None:
            sink(result)
        return result

    def execute(self, token, *, should_cancel=None):
        if token in RIGHT_ARM_ATOMS:
            return self._right_action(token, should_cancel=should_cancel)
        return super().execute(token, should_cancel=should_cancel)

    def execute_metric(self, token, magnitude, *, should_cancel=None):
        if token in RIGHT_ARM_ATOMS:
            return self._right_action(token, magnitude, should_cancel=should_cancel)
        return super().execute_metric(token, magnitude, should_cancel=should_cancel)

    def _guarded_motion(self, execute_motion, should_cancel):
        other = "right" if self.working_arm == "left" else "left"
        with self._select_arm(other):
            before = self.gripper_state()
        lost = False

        def cancelled():
            nonlocal lost
            with self._select_arm(other):
                lost |= before["holding"] is True and self.gripper_state()["holding"] is False
            return lost or bool(should_cancel and should_cancel())

        sink = self.on_result
        self.on_result = None
        try:
            result = super()._guarded_motion(execute_motion, cancelled)
        finally:
            self.on_result = sink
        if lost:
            result.update(completed=False, reason="grasp_lost", cancelled=bool(should_cancel and should_cancel()))
        return self._finish_result(result)

    def verify_stopped_result(self, result):
        with self._select_arm("right" if result.get("action") in RIGHT_ARM_ATOMS else "left"):
            return super().verify_stopped_result(result)

    def controller_state(self):
        # Public legacy fields always refer to the left arm; the extra readback
        # is projected through the same allowlist, never assisted grasp truth.
        with self._select_arm("left"):
            left = read_controller_state(self.session, controller=self, working_arm="left")
        with self._select_arm("right"):
            right = read_controller_state(self.session, controller=self, working_arm="right")
        return replace(left, right_arm=right)
