#!/usr/bin/env python3
"""Validate a scene's R1 head view and capture all review views in one world.

Run inside the retained interactive allocation:
python src/wia/runtime/commands/verify_esi_head_camera.py --scene-file CASE.yaml --output-dir OUT
--data-root explore_to_achieve/scenes. Failed camera checks retain the simulator;
edit the referenced camera configuration, then write camera_retry.json in OUT.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("PBS_JOBID"):
        parser.error("Use the retained interactive GPU allocation.")
    from wia.runtime.scene_review import write_json
    from wia.tasks.loader import load_scene
    import wia.runtime.sim.esi_web as web
    from wia.runtime.commands.verify_esi_rigid_grasp import main as probe_main

    selected = load_scene(args.scene_file, data_root=args.data_root, variant="tabletop")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def check(session, metadata, controller, executor, **kwargs):
        attempt = 0
        while True:
            attempt += 1
            try:
                mounts = importlib.reload(importlib.import_module("wia.runtime.sim.esi_robot_cameras"))
                diagnostics = importlib.reload(importlib.import_module("wia.runtime.sim.esi_head_diagnostics"))
                camera_config = ROOT / selected.definition.runtime.robot_camera_config
                mounts.apply_head_camera_mount(session, json.loads(camera_config.read_text()))
                path = diagnostics.diagnose_head_camera(session, args.output_dir / f"head_check_{attempt:03d}")
                evidence = json.loads(Path(path).read_text())
                visible = evidence["trials"][0]["visible_instances"]
                robot_pixels = sum(row["pixels"] for row in visible
                    if row["label"] == session.robot.name or row["label"].startswith(session.robot.prim_path))
                total = session.robot.sensors[session.camera_name].image_width * session.robot.sensors[session.camera_name].image_height
                if robot_pixels / total > .5:
                    raise ValueError("More than half of the head view is occluded by the robot.")
                cameras = importlib.reload(importlib.import_module("wia.runtime.commands.capture_esi_scene_reviews"))
                receipt = cameras.capture_native_review_views(session, selected, args.output_dir / f"review_views_{attempt:03d}")
                write_json(args.output_dir / "native_review_complete.json", receipt)
                write_json(args.output_dir / "validation_complete.json", {"scene_file": str(selected.path),
                    "source_json": selected.definition.source.json_path, "variant": selected.variant,
                    "head_robot_pixel_fraction": robot_pixels / total,
                    "head_diagnostic": path, "physical_grasp_verified": False, "model_calls": 0})
                write_json(args.output_dir / "camera_state.json", {"status": "review_ready", "pid": os.getpid(),
                    "attempts": attempt, "head_robot_pixel_fraction": robot_pixels / total,
                    "head_diagnostic": path, "head_image": evidence["trials"][0]["rgb"]})
                request = args.output_dir / "camera_retry.json"
                while not request.is_file():
                    time.sleep(.5)
                command = json.loads(request.read_text())
                request.replace(args.output_dir / f"camera_review_request_{attempt:03d}.json")
                if command.get("action") == "finish":
                    write_json(args.output_dir / "camera_state.json", {"status": "complete", "attempts": attempt,
                        "head_robot_pixel_fraction": robot_pixels / total, "manual_head_inspection_passed": True})
                    return
            except Exception as error:
                write_json(args.output_dir / "camera_state.json", {"status": "waiting", "pid": os.getpid(),
                    "attempts": attempt, "error": f"{type(error).__name__}: {error}"})
                print("[head-check] Keeping the world loaded:", error, flush=True)
                request = args.output_dir / "camera_retry.json"
                while not request.is_file():
                    time.sleep(.5)
                request.replace(args.output_dir / f"camera_request_{attempt:03d}.json")

    web.run_esi_control = check
    return probe_main(["--viewer-only", "--scene-file", str(selected.path), "--data-root", str(args.data_root),
        "--variant", "tabletop", "--web", "--gripper-control", "--trunk-control", "--parameterized-actions",
        "--primitive-profile", "fine", "--performance-profile", "benchmark", "--macro-execution", "metric_target",
        "--preview-fps", "0", "--output-dir", str(args.output_dir)])


if __name__ == "__main__":
    raise SystemExit(main())
