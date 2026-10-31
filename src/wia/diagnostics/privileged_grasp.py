"""Measure privileged assisted grasp evidence without issuing simulator actions.

Run via ``python scripts/run_privileged_grasp_validation.py --help``. The runner
must create a probe before acting and call ``sample_step()`` exactly once after
each completed controller.dt of simulation. ``evidence()`` and ``summary()`` do
not advance simulation or hold counters. All native reads require the main
thread. Native assisted attachments are not force-closure/friction certificates.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
import json
from pathlib import Path

import numpy as np

from wia.native.utils import _array, _main_thread


def jsonable(value):
    """Convert dataclasses, numpy values and detached torch tensors to JSON data."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: jsonable(getattr(value, field.name)) for field in fields(value)}
    if hasattr(value, "detach"):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported diagnostic JSON value: {type(value).__name__}")


def _object_state(obj):
    position, quaternion = obj.get_position_orientation()
    lower, upper = obj.aabb
    return jsonable({"name": str(obj.name), "category": str(obj.category),
                     "position": position, "quaternion_xyzw": quaternion,
                     "aabb_min": lower, "aabb_max": upper,
                     "fixed_base": bool(obj.fixed_base), "visual_only": bool(obj.visual_only)})


def _contacts(target):
    return [{"body0": body0, "body1": body1} for body0, body1 in sorted({
        (str(contact.body0), str(contact.body1)) for contact in target.contact_list()})]


def _arms(controller, *, include_eef=False):
    result = {}
    for arm in ("left", "right"):
        with controller._select_arm(arm):
            row = {"gripper": jsonable(controller.debug_gripper_state())}
            if include_eef:
                row["eef"] = jsonable(controller.ee_pose())
            result[arm] = row
    return result


def _target(session, target_name):
    target = session.env.scene.object_registry("name", target_name)
    if target is None:
        raise ValueError(f"Unknown native grasp target: {target_name}")
    return target


def _joint_state(robot):
    positions = _array(robot.get_joint_positions()).reshape(-1)
    names = getattr(robot, "dof_names_ordered", None)
    if names is None:
        names = [None] * len(positions)
        for name, joint in robot.joints.items():
            if joint.n_dof == 0:
                continue
            indices = list(joint.dof_indices)
            if joint.n_dof != 1 or len(indices) != 1:
                raise ValueError("Native R1 diagnostic requires single-DOF joint indices.")
            index = int(indices[0])
            if not 0 <= index < len(names) or names[index] is not None:
                raise ValueError("Native robot joint indices are inconsistent.")
            names[index] = str(name)
    names = list(names)
    if len(names) != len(positions) or None in names or len(set(names)) != len(names):
        raise ValueError("Native robot joint names do not match measured encoder positions.")
    limits = {str(name): [float(joint.lower_limit), float(joint.upper_limit)]
              for name, joint in robot.joints.items() if joint.n_dof == 1}
    return {"names": names, "values": positions.tolist(), "limits": limits}


def _link_states(robot):
    """Read every native robot link, retaining absent fields as unavailable."""
    missing = object()
    links = getattr(robot, "links", missing)
    if links is missing:
        return {"unavailable": "Native robot.links property is absent."}
    rows = {}
    for name, link in sorted(links.items()):
        row = {"name": str(name), "position": None, "quaternion_xyzw": None,
               "aabb_min": None, "aabb_max": None}
        unavailable = {}
        get_pose = getattr(link, "get_position_orientation", missing)
        if get_pose is missing:
            unavailable["world_pose"] = "Native get_position_orientation method is absent."
        else:
            row["position"], row["quaternion_xyzw"] = get_pose()
        aabb = getattr(link, "aabb", missing)
        if aabb is missing:
            unavailable["world_aabb"] = "Native aabb property is absent."
        else:
            row["aabb_min"], row["aabb_max"] = aabb
        if unavailable:
            row["unavailable"] = unavailable
        rows[str(name)] = jsonable(row)
    return rows


def world_snapshot(session, controller, target_name):
    """Read all scene geometry, robot encoders, both hands, and target contacts."""
    _main_thread()
    target = _target(session, target_name)
    objects = session.env.scene.objects
    if isinstance(objects, Mapping):
        objects = objects.values()
    return jsonable({"schema": "privileged_grasp_world_v1", "target_name": target_name,
                     "objects": [_object_state(obj) for obj in sorted(objects, key=lambda obj: obj.name)],
                     "robot": {"base": controller.pose(), "joints": _joint_state(session.robot),
                               "links": _link_states(session.robot),
                               "grasping_mode": str(session.robot.grasping_mode),
                               "arms": _arms(controller, include_eef=True)},
                     "target_contacts": _contacts(target)})


class GraspProbe:
    """Latch success only after a measured, consecutive lifted assisted hold."""

    def __init__(self, session, controller, target_name, output,
                 minimum_lift_m=.05, required_hold_sim_s=.5):
        _main_thread()
        for name, value in (("minimum_lift_m", minimum_lift_m),
                            ("required_hold_sim_s", required_hold_sim_s), ("controller.dt", controller.dt)):
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        self.session, self.controller = session, controller
        self.target = _target(session, target_name)
        if (self.target.visual_only or self.target.fixed_base
                or getattr(self.target, "collision_enabled", True) is False):
            raise ValueError("Grasp target must be an existing movable collidable object.")
        arms = _arms(controller)
        if any(row["gripper"].get("object_name") == self.target.name for row in arms.values()):
            raise ValueError("Target must initially be unheld by both arms; existing target identity rejects the baseline.")
        self.minimum_lift_m = float(minimum_lift_m)
        self.required_hold_sim_s = float(required_hold_sim_s)
        self.dt = float(controller.dt)
        self.initial_target = _object_state(self.target)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.sample_count = 0
        self.consecutive_hold_steps = 0
        self.maximum_hold_sim_s = 0.
        self.maximum_held_lift_m = 0.
        self.grasped_ever = False
        self.success = False
        self.success_evidence = None

    def evidence(self):
        """Read current native state without changing counters or retained evidence."""
        _main_thread()
        target = _object_state(self.target)
        lift = float(target["aabb_min"][2] - self.initial_target["aabb_min"][2])
        if not np.isfinite(lift):
            raise ValueError("Target AABB lift must be finite.")
        arms = _arms(self.controller, include_eef=True)
        held_arms = [arm for arm, row in arms.items()
                     if row["gripper"].get("holding") is True
                     and row["gripper"].get("object_name") == self.target.name
                     and row["gripper"].get("constraint_active") is True]
        return {"schema": "privileged_grasp_step_v1", "sample_step": self.sample_count,
                "sampled_sim_s": self.sample_count * self.dt,
                "target": target, "lift_m": lift, "held_arms": held_arms,
                "target_held": bool(held_arms), "lift_threshold_reached": lift >= self.minimum_lift_m,
                "qualifying_hold_now": bool(held_arms) and lift >= self.minimum_lift_m,
                "consecutive_hold_sim_s": self.consecutive_hold_steps * self.dt,
                "success": self.success, "arms": arms,
                "base": jsonable(self.controller.pose()), "contacts": _contacts(self.target)}

    def sample_step(self):
        """Record one completed controller.dt; the caller alone steps simulation."""
        _main_thread()
        state = self.evidence()
        self.sample_count += 1
        if state["target_held"]:
            self.grasped_ever = True
            self.maximum_held_lift_m = max(self.maximum_held_lift_m, state["lift_m"])
        self.consecutive_hold_steps = self.consecutive_hold_steps + 1 if state["qualifying_hold_now"] else 0
        duration = self.consecutive_hold_steps * self.dt
        self.maximum_hold_sim_s = max(self.maximum_hold_sim_s, duration)
        reached = duration >= self.required_hold_sim_s - 1e-12
        self.success |= reached
        state.update(sample_step=self.sample_count, sampled_sim_s=self.sample_count * self.dt,
                     consecutive_hold_sim_s=duration, success=self.success)
        if reached and self.success_evidence is None:
            self.success_evidence = deepcopy(state)
        with (self.output / "evidence.jsonl").open("a") as stream:
            stream.write(json.dumps(state, separators=(",", ":"), allow_nan=False) + "\n")
        return state

    def summary(self):
        """Persist current measurements and retained success without adding hold time."""
        _main_thread()
        result = {"schema": "privileged_grasp_summary_v1", "target_name": str(self.target.name),
                  "criterion": {"initially_unheld": True, "minimum_lift_m": self.minimum_lift_m,
                                "required_hold_sim_s": self.required_hold_sim_s,
                                "held_requires": "Either arm: target identity, holding is True, constraint_active is True",
                                "lift_reference": "Target world AABB bottom relative to original baseline",
                                "hold_clock": "One controller.dt per sample_step; resets on any failed sample"},
                  "grasping_mode": str(self.session.robot.grasping_mode),
                  "limitation": "Existing native assisted grasp is not a force-closure or friction certificate.",
                  "initial_target": deepcopy(self.initial_target), "sample_count": self.sample_count,
                  "sampled_sim_s": self.sample_count * self.dt, "controller_dt_s": self.dt,
                  "grasped_ever": self.grasped_ever, "maximum_held_lift_m": self.maximum_held_lift_m,
                  "maximum_hold_sim_s": self.maximum_hold_sim_s,
                  "consecutive_hold_sim_s": self.consecutive_hold_steps * self.dt,
                  "success": self.success, "success_evidence": deepcopy(self.success_evidence),
                  "current": self.evidence()}
        temporary = self.output / "summary.json.tmp"
        temporary.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
        temporary.replace(self.output / "summary.json")
        return result
