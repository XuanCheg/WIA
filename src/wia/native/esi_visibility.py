"""Private setup acceptance and grasp-view diagnostics, never a model input. Run: python -m pytest tests/test_extraction.py. """
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .utils import _main_thread
from .utils import _array


def mask_metrics(mask, rgb):
    """Report visible pixels, extent and local contrast; do not infer grasp state."""
    mask, rgb = np.asarray(mask, dtype=bool), np.asarray(rgb)
    if mask.ndim != 2 or rgb.shape != (*mask.shape, 3):
        raise ValueError("Visibility masks must match their native RGB image.")
    y, x = np.nonzero(mask)
    if not len(x):
        return {"visible_pixels": 0, "bbox_xyxy_px": None, "minimum_extent_px": 0,
                "touches_image_border": False, "local_rgb_contrast": None}
    x0, x1, y0, y1 = int(x.min()), int(x.max()), int(y.min()), int(y.max())
    ring = np.zeros(mask.shape, dtype=bool)
    ring[max(0, y0-4):y1+5, max(0, x0-4):x1+5] = True
    ring &= ~mask
    contrast = (float(np.linalg.norm(np.median(rgb[mask], axis=0)-np.median(rgb[ring], axis=0)))
                if ring.any() else None)
    return {"visible_pixels": int(len(x)), "bbox_xyxy_px": [x0, y0, x1, y1],
            "minimum_extent_px": min(x1-x0+1, y1-y0+1),
            "touches_image_border": bool(x0 == 0 or y0 == 0 or x1 == mask.shape[1]-1 or y1 == mask.shape[0]-1),
            "local_rgb_contrast": contrast}


def passes_visibility(metrics, *, minimum_pixels, minimum_extent_px):
    return (metrics["visible_pixels"] >= minimum_pixels
            and metrics["minimum_extent_px"] >= minimum_extent_px
            and not metrics["touches_image_border"])


class PrivateVisibilityCheck:
    """Reject a bad initial view; later grasp diagnostics never gate execution.

    Instance segmentation is enabled temporarily for local evidence only. It is
    removed before normal observations resume. Pixel acceptance is supplemented
    by human review of saved RGBs when qualifying an experiment's start pose.
    """

    def __init__(self, session, config, output, *, working_arm="left"):
        allowed = {"scene", "room", "target", "minimum_pixels", "minimum_extent_px", "camera_scope", "initial_condition"}
        if set(config)-allowed or config.get("scene") != session.scene_name or config.get("room") != session.room_name:
            raise ValueError("Visibility configuration must belong to this scene/room.")
        for name, default in (("minimum_pixels", 64), ("minimum_extent_px", 8)):
            if type(config.get(name, default)) is not int or config.get(name, default) <= 0:
                raise ValueError("Visibility thresholds must be positive integers.")
        target = session.env.scene.object_registry("name", config.get("target"))
        if target is None:
            raise ValueError("Visibility target must be an existing scene object.")
        evaluator = getattr(session, "task_evaluator", None)
        if evaluator is not None and evaluator.target is not target:
            raise ValueError("Visibility and task evaluation must use the same target.")
        self.session, self.target, self.working_arm = session, target, working_arm
        self.config = dict(config, minimum_pixels=config.get("minimum_pixels", 64),
                           minimum_extent_px=config.get("minimum_extent_px", 8))
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if config.get("camera_scope", "head_wrist") not in ("head_wrist", "enabled_agent"):
            raise ValueError("Unknown private camera scope.")
        if config.get("initial_condition", "visible") not in ("visible", "hidden_zero", "hidden_unrecognizable"):
            raise ValueError("Unknown initial visibility condition.")

    def record(self, *, phase, observation=None):
        _main_thread()
        registry = getattr(self.session, "robot_camera_registry", None) or {
            "agentview_rgb": self.session.camera_name, "wrist_rgb": self.session.wrist_camera_name}
        if self.config.get("camera_scope") == "enabled_agent":
            from core.robot_cameras import RGB_TO_VIEW
            config = self.session.agent_observation_config
            sensors = {RGB_TO_VIEW[name]: self.session.robot.sensors[sensor_name]
                       for name, sensor_name in registry.items() if
                       (name == "agentview_rgb" and config.head_rgb or name == "wrist_rgb" and config.wrist_rgb
                        or name not in ("agentview_rgb", "wrist_rgb") and config.additional_robot_rgb)}
            if not sensors:
                raise ValueError("Qualification requires at least one enabled Agent RGB camera.")
        else:
            sensors = {view: self.session.robot.sensors[registry[name]] for view, name in
                       (("head", "agentview_rgb"), ("wrist", "wrist_rgb"))}
        added = []
        record = {"schema": "private_visibility_v1", "phase": phase,
                  "captured_at": datetime.now(timezone.utc).isoformat(), "configuration": self.config,
                  "action_step": observation.action_step if observation is not None else None,
                  "snapshot_id": observation.snapshot_id if observation is not None else None,
                  "views": {}}
        try:
            for sensor in sensors.values():
                for modality in ("seg_semantic", "seg_instance", "seg_instance_id"):
                    if modality not in sensor.modalities:
                        sensor.add_modality(modality)
                        added.append((sensor, modality))
            for _ in range(5):
                self.session.pipeline.og.sim.render()
            from core.record.images import save_png
            from PIL import Image, ImageDraw

            stamp = f"{phase}-{record['action_step'] or 0:03d}-{record['snapshot_id'] or 0:05d}"
            for view, sensor in sensors.items():
                obs, info = sensor.get_obs()
                rgb = _array(obs["rgb"])[..., :3].astype(np.uint8)
                labels = [int(key) for key, value in info["seg_instance"].items() if value == self.target.name]
                mask = np.isin(_array(obs["seg_instance"]), labels)
                metrics = mask_metrics(mask, rgb)
                metrics["passes_pixel_gate"] = passes_visibility(metrics,
                    minimum_pixels=self.config["minimum_pixels"], minimum_extent_px=self.config["minimum_extent_px"])
                finger_paths = [link.prim_path for link in self.session.robot.finger_links[self.working_arm]]
                ids = [int(key) for key, value in info["seg_instance_id"].items()
                       if any(value == path or value.startswith(path+"/") for path in finger_paths)]
                finger_mask = np.isin(_array(obs["seg_instance_id"]), ids)
                metrics["working_finger_visible_pixels"] = int(finger_mask.sum())
                record["views"][view] = metrics
                save_png(self.output / f"{stamp}-{view}.png", rgb)
                save_png(self.output / f"{stamp}-{view}-target-mask.png", np.repeat(mask[..., None], 3, axis=2).astype(np.uint8)*255)
                annotated = Image.fromarray(rgb)
                if metrics["bbox_xyxy_px"] is not None:
                    ImageDraw.Draw(annotated).rectangle(metrics["bbox_xyxy_px"], outline=(255, 0, 255), width=2)
                annotated.save(self.output / f"{stamp}-{view}-private-target-box.png")
        finally:
            for sensor, modality in reversed(added):
                sensor.remove_modality(modality)
        record["initial_head_accepted"] = record["views"].get("head", {}).get("passes_pixel_gate", False)
        record["wrist_target_and_fingers_visible"] = bool(record["views"].get("wrist", {}).get("passes_pixel_gate")
            and record["views"].get("wrist", {}).get("working_finger_visible_pixels", 0) > 0)
        record["all_views_zero_pixels"] = all(v["visible_pixels"] == 0 for v in record["views"].values())
        record["any_objectively_visible"] = any(v["visible_pixels"] > 0 for v in record["views"].values())
        record["any_recognizably_visible"] = any(v["passes_pixel_gate"] for v in record["views"].values())
        with (self.output / "trajectory.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
        (self.output / f"{stamp}.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        return record

    def check_initial(self, observation=None):
        record = self.record(phase="initial", observation=observation)
        condition = self.config.get("initial_condition", "visible")
        visible = record["any_recognizably_visible"] if self.config.get("camera_scope") == "enabled_agent" else record["initial_head_accepted"]
        accepted = (visible if condition == "visible" else
                    record["all_views_zero_pixels"] if condition == "hidden_zero" else
                    not record["any_recognizably_visible"])
        if not accepted:
            raise RuntimeError("Initial target visibility did not pass the configured private camera gate; "
                               "inspect private_visibility evidence and adjust the setup before running a model.")
        return record

    def record_grasp(self, observation):
        # Diagnostic failures must not change an already completed robot action
        # or turn an otherwise known, stopped receipt into an unknown outcome.
        try:
            return self.record(phase="after_grasp", observation=observation)
        except Exception as error:
            record = {"schema": "private_visibility_error_v1", "phase": "after_grasp",
                      "action_step": observation.action_step, "snapshot_id": observation.snapshot_id,
                      "captured_at": datetime.now(timezone.utc).isoformat(),
                      "error": f"{type(error).__name__}: {error}"}
            try:
                with (self.output / "errors.jsonl").open("a") as stream:
                    stream.write(json.dumps(record)+"\n")
            except OSError:
                return None
            return record
