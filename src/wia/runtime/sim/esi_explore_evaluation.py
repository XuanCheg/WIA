"""Private exploration stages and physical trajectory; never a model input."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from wia.runtime.sim.esi_base import _main_thread
from wia.runtime.sim.esi_omnigibson import _array


class ExploreEvaluator:
    def __init__(self, session, controller, scene, output):
        self.session, self.controller, self.scene = session, controller, scene
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.active = False
        self.last_record = None

    def begin_episode(self, observation):
        _main_thread()
        self.active = True
        self.last_record = None
        self.start_wall, self.start_sim = time.perf_counter(), float(self.session.pipeline.og.sim.current_time)
        self.previous_position = np.asarray(self.controller.pose()["position"][:2])
        self.previous_boundary = self.previous_position.copy()
        self.last_regions, self.visits = set(), {}
        self.metrics = {"schema": "private_explore_metrics_v1", "seed": self.scene.seed,
            "variant": self.scene.config["variant"], "selected_region": self.scene.selected["region"],
            "first_objectively_visible_step": None, "first_recognizably_visible_step": None,
            "first_target_approach_step": None, "first_grasp_attempt_step": None,
            "first_successful_attachment_step": None, "first_valid_lift_step": None,
            "task_success": False, "base_actions": 0, "arm_trunk_actions": 0, "grasp_attempts": 0,
            "macro_executions": 0, "controller_env_steps": 0, "blocked_motions": 0,
            "physical_trajectory_length_m": 0., "boundary_trajectory_length_m": 0.,
            "region_revisits": 0, "region_visits": [],
            "recognizability_criterion": self.scene.visibility_config(),
            "region_visit_definition": "Entry of any enabled robot camera into the configured inspection camera AABB; does not prove semantic inspection.",
            "visibility_does_not_prove_model_recognition": True}
        self.record(observation, result=None)

    def sample_step(self):
        _main_thread()
        if not self.active:
            return
        position = np.asarray(self.controller.pose()["position"][:2])
        self.metrics["physical_trajectory_length_m"] += float(np.linalg.norm(position-self.previous_position))
        self.previous_position = position
        self.metrics["controller_env_steps"] += 1
        with (self.output / "physical_trajectory.jsonl").open("a") as stream:
            stream.write(json.dumps({"env_step": self.metrics["controller_env_steps"],
                "simulation_time_s": float(self.session.pipeline.og.sim.current_time),
                "base_xy_m": position.tolist()}) + "\n")

    def record(self, observation, *, result):
        _main_thread()
        if not self.active:
            return
        visibility = self.session.private_visibility_checker.record(phase="decision_boundary", observation=observation)
        grasp = json.loads((self.session.task_evaluator.output / "state.json").read_text())
        step = observation.action_step
        target = _array(self.session.task_evaluator.target.get_position_orientation()[0])
        ee = np.asarray(self.controller.ee_pose()["world_position"])
        approached = float(np.linalg.norm(ee-target)) <= self.scene.config["evaluation"]["approach_distance_m"]
        tests = {"first_objectively_visible_step": visibility["any_objectively_visible"],
                 "first_recognizably_visible_step": visibility["any_recognizably_visible"],
                 "first_target_approach_step": approached,
                 "first_grasp_attempt_step": result is not None and result["action"] == "GRASP",
                 "first_successful_attachment_step": grasp["criteria"]["target_held"],
                 "first_valid_lift_step": grasp["task_success"]}
        for key, reached in tests.items():
            if reached and self.metrics[key] is None:
                self.metrics[key] = step
        if result is not None:
            action = result["action"]
            self.metrics["macro_executions"] += 1
            self.metrics["base_actions"] += int(action.startswith("BASE_"))
            self.metrics["arm_trunk_actions"] += int(action.startswith(("MV_", "TRUNK_", "ROTATE_")))
            self.metrics["grasp_attempts"] += int(action == "GRASP")
            self.metrics["blocked_motions"] += int(result.get("reason") not in (None, "completed"))
        position = np.asarray(self.controller.pose()["position"][:2])
        self.metrics["boundary_trajectory_length_m"] += float(np.linalg.norm(position-self.previous_boundary))
        self.previous_boundary = position
        views = []
        from core.robot_cameras import RGB_TO_VIEW
        for name, sensor_name in self.session.robot_camera_registry.items():
            if RGB_TO_VIEW[name] in visibility["views"]:
                views.append(_array(self.session.robot.sensors[sensor_name].get_position_orientation()[0]))
        regions = set()
        for candidate in self.scene.config["target_placement_candidates"]:
            box = candidate.get("inspection_camera_aabb")
            if box and any(np.all(p >= box[0]) and np.all(p <= box[1]) for p in views):
                regions.add(candidate["region"])
        for name in sorted(regions-self.last_regions):
            self.metrics["region_revisits"] += int(self.visits.get(name, 0) > 0)
            self.visits[name] = self.visits.get(name, 0)+1
            self.metrics["region_visits"].append({"region": name, "action_step": step})
        self.last_regions = regions
        self.metrics.update(task_success=grasp["task_success"], wall_duration_s=time.perf_counter()-self.start_wall,
                            simulated_elapsed_time_s=float(self.session.pipeline.og.sim.current_time)-self.start_sim)
        row = dict(self.metrics, action_step=step, snapshot_id=observation.snapshot_id,
                   visibility=visibility["views"], base=self.controller.pose(), eef=self.controller.ee_pose())
        self.last_record = row
        temporary = self.output / "state.json.tmp"
        temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.output / "state.json")
        with (self.output / "trajectory.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def finish_episode(self):
        """Owner-thread timing finalization; does not capture sensors or advance physics."""
        _main_thread()
        if not self.active:
            return
        self.active = False
        self.metrics.update(wall_duration_s=time.perf_counter()-self.start_wall,
            simulated_elapsed_time_s=float(self.session.pipeline.og.sim.current_time)-self.start_sim,
            episode_finished=True)
        if self.last_record is not None:
            row = dict(self.last_record, **self.metrics)
            (self.output / "state.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")

    def record_error(self, observation, error):
        with (self.output / "errors.jsonl").open("a") as stream:
            stream.write(json.dumps({"action_step": observation.action_step, "snapshot_id": observation.snapshot_id,
                                     "error": f"{type(error).__name__}: {error}"})+"\n")
