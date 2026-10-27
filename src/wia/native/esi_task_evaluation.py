"""Private native grasp/lift evidence. Never part of AgentObservation or MCP. Run: python -m pytest tests/test_extraction.py. """
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .utils import _main_thread
from .utils import _array


class GraspLiftEvaluator:
    def __init__(self, session, controller, config, output):
        allowed = {"type", "scene", "room", "target", "minimum_lift_m"}
        if set(config) - allowed or config.get("type") != "grasp_lift":
            raise ValueError("Use a grasp_lift evaluation with explicit target and threshold.")
        if (config.get("scene") != session.scene_name or config.get("room") != session.room_name):
            raise ValueError("Evaluation belongs to a different scene/room.")
        minimum = config.get("minimum_lift_m", 0.05)
        if isinstance(minimum, bool) or not np.isfinite(minimum) or minimum <= 0:
            raise ValueError("Minimum lift must be a positive finite distance.")
        target = session.env.scene.object_registry("name", config.get("target"))
        if target is None or target.visual_only or target.fixed_base:
            raise ValueError("Evaluation target must be an existing movable collidable object.")
        self.session, self.controller, self.target = session, controller, target
        self.config = dict(config, minimum_lift_m=float(minimum))
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode = 0
        self.initial = None

    def _target_state(self):
        position, quaternion = self.target.get_position_orientation()
        lower, upper = self.target.aabb
        return {"name": self.target.name, "category": self.target.category,
                "position": _array(position).tolist(), "quaternion_xyzw": _array(quaternion).tolist(),
                "aabb_min": _array(lower).tolist(), "aabb_max": _array(upper).tolist()}

    def begin_episode(self, observation):
        _main_thread()
        self.episode += 1
        self.initial = self._target_state()
        self.initially_held = self.controller.debug_gripper_state().get("object_name") == self.target.name
        self.record(observation, action=None)

    def additional_evidence(self, target):
        """Optional physical criteria; returned evidence remains private."""
        return {}, {}

    def record(self, observation, *, action):
        _main_thread()
        if self.initial is None:
            # Manual control remains usable before a runner starts a task.
            return None
        target = self._target_state()
        gripper = self.controller.debug_gripper_state()
        lifted = float(target["aabb_min"][2] - self.initial["aabb_min"][2])
        held = (gripper.get("holding") is True and gripper.get("object_name") == self.target.name
                and gripper.get("constraint_active") is True)
        extra_criteria, evidence = self.additional_evidence(target)
        success = (not self.initially_held and held and lifted >= self.config["minimum_lift_m"]
                   and all(extra_criteria.values()))
        record = {"schema": "private_grasp_lift_v1", "episode": self.episode,
                  "task_text": observation.task_text, "task_success": bool(success),
                  "snapshot_id": observation.snapshot_id, "action_step": observation.action_step,
                  "action": action, "evaluated_at": datetime.now(timezone.utc).isoformat(),
                  "criteria": {"initially_unheld": not self.initially_held, "target_held": held,
                               "lift_threshold_reached": lifted >= self.config["minimum_lift_m"], **extra_criteria},
                  "lift_m": lifted, "configuration": self.config, "initial_target": self.initial,
                  "target": target, "gripper": gripper, "base": self.controller.pose(),
                  "eef": self.controller.ee_pose(), **evidence}
        safety = getattr(self.session, 'safety_observer', None)
        if safety is not None:
            record['safety'] = safety.result()
            record['safety_score'] = record['safety']['safety_score']
        temporary = self.output / "state.json.tmp"
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.output / "state.json")
        with (self.output / "trajectory.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        return record
