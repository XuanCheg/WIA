"""External, initialization-only exploration variants; no upstream task edits."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from wia.runtime.sim.esi_base import _main_thread
from wia.runtime.sim.esi_omnigibson import SHOWHARNESS_ROOT, _array

TASK_INSTRUCTION = (
    "Find a red tomato in the room and grasp it with the left gripper. "
    "After grasping it, move upward by 6 cm (0.06 m) so that the tomato rises "
    "at least 5 cm (0.05 m) above its height at the start of the task, and keep holding it. "
    "Declare DONE only after confirming the grasp and the completed lift."
)


class ExploreScene:
    def __init__(self, config, *, seed=None):
        self.config = copy.deepcopy(config)
        self.seed = config["seed"] if seed is None else seed
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Scene seed must be a nonnegative integer.")
        if config["variant"] not in ("visible_target_control", "single_table_occlusion", "multi_region_search"):
            raise ValueError("Unknown exploration scene variant.")
        candidates = config["target_placement_candidates"]
        if not candidates or (config["variant"] == "multi_region_search" and len(candidates) < 2):
            raise ValueError("Provide safe candidate regions, at least two for search.")
        if len({item["region"] for item in candidates}) != len(candidates):
            raise ValueError("Candidate region names must be unique.")
        rng = np.random.default_rng(self.seed)
        self.selected_index = int(rng.integers(len(candidates)))
        self.selected = copy.deepcopy(candidates[self.selected_index])
        pose_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1]))
        self.robot_pose = copy.deepcopy(config["robot_initial_pose"])
        self.robot_pose["yaw_deg"] += float(pose_rng.choice(config.get("heading_offsets_deg", [0.])))
        robot_jitter = np.asarray(config.get("robot_jitter_xy_m", [0., 0.]), dtype=float)
        if robot_jitter.shape != (2,) or not np.isfinite(robot_jitter).all() or (robot_jitter < 0).any():
            raise ValueError("Robot jitter must be two finite nonnegative bounds.")
        robot_xy = np.asarray(self.robot_pose["position"][:2]) + pose_rng.uniform(-robot_jitter, robot_jitter)
        self.robot_pose["position"][:2] = robot_xy.tolist()
        jitter = np.asarray(config.get("target_jitter_xy_m", [0., 0.]), dtype=float)
        if jitter.shape != (2,) or not np.isfinite(jitter).all() or (jitter < 0).any():
            raise ValueError("Target jitter must be two finite nonnegative bounds.")
        self.target_position = np.asarray(self.selected["position"], dtype=float)
        if self.target_position.shape != (3,) or not np.isfinite(self.target_position).all():
            raise ValueError("Target placement must be finite XYZ.")
        self.target_position[:2] += rng.uniform(-jitter, jitter)
        self.occluders = copy.deepcopy(config.get("occluders", []))
        shift = float(rng.uniform(-config.get("occluder_jitter_x_m", 0.), config.get("occluder_jitter_x_m", 0.)))
        for item in self.occluders:
            item["position"][0] += shift
            size = np.asarray(item["dimensions_m"], dtype=float)
            if size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
                raise ValueError("Occluders require positive finite dimensions.")
        self.initialized = False
        self.nuisance_positions = {}

    @classmethod
    def load(cls, path, *, seed=None):
        return cls(json.loads(Path(path).read_text()), seed=seed)

    @property
    def metadata_path(self):
        return (SHOWHARNESS_ROOT / self.config["source_metadata"]).resolve()

    def observation_settings(self):
        pose = self.robot_pose
        return {"robot_position": pose["position"], "robot_yaw_deg": pose["yaw_deg"],
                "head_pitch_deg": self.config.get("head_pitch_deg", 45.), "camera_size": [1024, 1024]}

    def apply_to_env_config(self, config):
        if self.initialized:
            raise RuntimeError("Exploration setup is initialization-only.")
        objects = config.setdefault("objects", [])
        identity = self.config["target_object"]
        target = next((obj for obj in objects if obj.get("name") == identity["name"]), None)
        if target is None or any(target.get(key) != identity[key] for key in ("category", "model")):
            raise ValueError("Source target does not match the configured asset.")
        target["position"] = self.target_position.tolist()
        nuisance_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 2]))
        nuisance_jitter = np.asarray(self.config.get("nuisance_jitter_xy_m", [0., 0.]), dtype=float)
        if nuisance_jitter.shape != (2,) or not np.isfinite(nuisance_jitter).all() or (nuisance_jitter < 0).any():
            raise ValueError("Nuisance jitter must be finite nonnegative XY bounds.")
        for name in self.config.get("nuisance_objects", []):
            obj = next((obj for obj in objects if obj.get("name") == name), None)
            if obj is None or name == identity["name"]:
                raise ValueError("Nuisance objects must be separate existing source objects.")
            position = np.asarray(obj["position"], dtype=float)
            position[:2] += nuisance_rng.uniform(-nuisance_jitter, nuisance_jitter)
            obj["position"] = self.nuisance_positions[name] = position.tolist()
        for item in self.occluders:
            objects.append({"type": "PrimitiveObject", "name": item["name"], "primitive_type": "Cube",
                "category": "object", "size": 1., "scale": item["dimensions_m"],
                "fixed_base": True, "visual_only": False, "rgba": [0.25, 0.32, 0.40, 1.],
                "position": item["position"], "orientation": item.get("quaternion_xyzw", [0., 0., 0., 1.])})

    def finish_setup(self, session, output):
        _main_thread()
        if self.initialized or (session.scene_name, session.room_name) != (self.config["scene"], self.config["room"]):
            raise ValueError("Exploration setup must match its source scene and run exactly once.")
        target = session.env.scene.object_registry("name", self.config["target_object"]["name"])
        support = session.env.scene.object_registry("name", self.config["support_object"])
        if target is None or support is None or target.visual_only or target.fixed_base:
            raise ValueError("Require a movable, collidable target and an existing support.")
        low, high = (_array(x) for x in target.aabb)
        support_low, support_high = (_array(x) for x in support.aabb)
        support_ok = bool(np.all(low[:2] >= support_low[:2]) and np.all(high[:2] <= support_high[:2])
                          and abs(low[2] - support_high[2]) <= self.config["evaluation"]["support_tolerance_m"])
        record = {"schema": "private_explore_setup_v1", "configuration": self.config, "seed": self.seed,
            "selected_region": self.selected["region"], "selected_index": self.selected_index,
            "robot_pose_requested": self.robot_pose,
            "nuisance_positions_requested": self.nuisance_positions,
            "target_position_requested": self.target_position.tolist(), "support_aabb_screen": support_ok,
            "target": {"name": target.name, "category": target.category, "model": target.model,
                       "aabb_min": low.tolist(), "aabb_max": high.tolist()},
            "support": {"name": support.name, "aabb_min": support_low.tolist(), "aabb_max": support_high.tolist()},
            "occluders": []}
        for item in self.occluders:
            obj = session.env.scene.object_registry("name", item["name"])
            if obj is None or obj.visual_only or not obj.fixed_base:
                raise RuntimeError("Occluder must be native, collidable, fixed geometry.")
            lo, hi = (_array(x).tolist() for x in obj.aabb)
            record["occluders"].append(dict(item, aabb_min=lo, aabb_max=hi, visual_only=obj.visual_only,
                                             fixed_base=obj.fixed_base))
        path = Path(output) / "private_scene"
        path.mkdir(parents=True, exist_ok=True)
        (path / "setup.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        self.initialized = True
        self.private_setup = record
        if not support_ok:
            raise RuntimeError("Target failed the initial support screen; inspect private scene evidence.")

    def evaluation_config(self):
        return {"type": "grasp_lift", "scene": self.config["scene"], "room": self.config["room"],
                "target": self.config["target_object"]["name"], "minimum_lift_m": self.config["evaluation"]["minimum_lift_m"]}

    def visibility_config(self):
        return {"scene": self.config["scene"], "room": self.config["room"],
            "target": self.config["target_object"]["name"], "camera_scope": "enabled_agent",
            "initial_condition": "visible" if self.config["variant"] == "visible_target_control" else "hidden_zero",
            "minimum_pixels": self.config["evaluation"]["minimum_pixels"],
            "minimum_extent_px": self.config["evaluation"]["minimum_extent_px"]}
