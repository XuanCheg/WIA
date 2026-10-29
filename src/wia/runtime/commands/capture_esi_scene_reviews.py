#!/usr/bin/env python3
"""Capture final-layout review cameras sequentially in one interactive GPU job.

Run after the simulation environment is activated:
python src/wia/runtime/commands/capture_esi_scene_reviews.py --conversion-root RUN --output-dir REVIEW
--job-id PBS_JOB_ID. Completed exact-layout captures are reused on rerun.
The worker reuses the scene startup checks and executes no robot or Agent actions.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.runtime.scene_review import SceneReview, fingerprint, read_json, sha256, write_json  # noqa: E402


def load_pending_preview(row, data_root):
    """Load a pending definition for diagnostic rendering without changing its status."""
    from wia.tasks.loader import LoadedScene, UniqueLoader
    from wia.tasks.schema import SceneDefinition
    import yaml

    path = Path(row["scene_file"]).resolve()
    definition = SceneDefinition.model_validate(yaml.load(path.read_text(), Loader=UniqueLoader))
    if definition.status != "pending":
        raise ValueError("Diagnostic pending preview requires a pending scene.")
    source = (Path(data_root) / definition.source.json_path).resolve()
    if not source.is_relative_to(Path(data_root).resolve()):
        raise ValueError("Preview source must be inside the copied catalogue.")
    return LoadedScene(definition, path, source, sha256(path), sha256(source),
                       row["variant"], row["seed"], row["scene"], row["room"])


def pending_worker(row, store, out):
    """Render before controller startup; retain every pending rejection and guard."""
    from wia.runtime.sim.esi_omnigibson import ESISession
    from wia.runtime.sim.esi_robot_cameras import configure_robot_cameras
    from wia.tasks.scene_only import SceneOnly
    from core.viewer_config import ViewerConfig
    from core.record.images import save_png

    selected = load_pending_preview(row, store.data_root)
    adapter = SceneOnly(selected)
    session = ESISession(selected.source_path, **adapter.observation_settings())
    session.scene_adapter = adapter
    try:
        session.reset()
        runtime = selected.definition.runtime
        configure_robot_cameras(session, ROOT / runtime.robot_camera_config, resolution=runtime.camera_size)
        session.configure_scene_camera()
        session.configure_human_cameras(ViewerConfig.load(ROOT / runtime.viewer_config))
        raw = session.observe(include_wrist=True, include_scene=True)
        views = {"head": raw["head_rgb"], "wrist": raw["wrist_rgb"]}
        views.update({name: session._sensor_rgb(sensor) for name, sensor in session.global_cameras.items()})
        if raw.get("scene_rgb") is not None:
            views["scene"] = raw["scene_rgb"]
        selected.verify_unchanged()
        files = {}
        for name, pixels in views.items():
            path = out / f"{name}.png"
            save_png(path, pixels)
            files[name] = str(path)
        write_json(out / "preview_complete.json", {"schema": "scene_review_preview_v1", "id": row["id"],
            "fingerprint": row["fingerprint"], "variant": selected.variant, "seed": selected.seed,
            "images": files, "provenance": "Diagnostic pending layout before controller startup; native qualification was not run.",
            "diagnostic_pending": True, "controller_initialization_run": False, "native_qualification_run": False,
            "job_id": os.environ["PBS_JOBID"], "model_calls": 0, "robot_actions": 0,
            "physical_grasp_verified": False})
        print(f"[scene-review] Rendered pending layout: {row['source']}", flush=True)
        return 0
    finally:
        session.close()


def capture_target_view(session, selected):
    """Locate the bound pickup object with an independent human-only close camera."""
    import numpy as np
    from omnigibson.sensors import VisionSensor
    from omnigibson.utils.sampling_utils import raytest
    from omnigibson.utils.usd_utils import PoseAPI
    from wia.runtime.sim.esi_debug_cameras import camera_orientation
    from wia.runtime.sim.esi_omnigibson import _array
    from wia.tasks.support_sampling import same_body

    binding = selected.definition.bindings[selected.definition.target]
    target = session.env.scene.object_registry("name", binding.name)
    low, high = (_array(v) for v in target.aabb)
    centre = (low + high) / 2
    base = _array(session.robot.get_position_orientation()[0])
    direction = base[:2] - centre[:2]
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    if np.linalg.norm(direction) < .1:
        direction = np.array([1., 0.])
    distance = max(.40, float(np.max(high - low)) * 2)
    positions = [centre + np.r_[d * distance, distance * .55] for d in (
        direction, np.array([-direction[1], direction[0]]), -direction,
        np.array([direction[1], -direction[0]]))]
    scores = []
    for position in positions:
        hit = raytest(position.tolist(), centre.tolist())
        reverse = raytest(centre.tolist(), position.tolist(), only_closest=False)
        # A forward ray can miss geometry containing its origin. The reverse
        # ray rejects viewpoints inside the robot or another nearby object.
        clear = (not hit["hit"] or same_body(target.prim_path, hit["rigidBody"])) and all(
            same_body(target.prim_path, result["rigidBody"]) for result in reverse if result["hit"])
        scores.append(int(clear))
    position = positions[int(np.argmax(scores))]
    sensor = getattr(session, "_review_target_sensor", None)
    if sensor is None:
        sensor = VisionSensor(relative_prim_path="/human_cameras/global_target", name="global_target",
            modalities=("rgb", "seg_instance_id"), image_width=512, image_height=512, focal_length=12.)
        sensor.load(session.env.scene)
        sensor.initialize()
        session._review_target_sensor = sensor
    sensor.set_position_orientation(position=position.tolist(),
        orientation=camera_orientation(position, centre).tolist(), frame="world")
    # Late-created sensors can render their initialization viewpoint while
    # reporting the correct USD pose. Synchronize Fabric before exporting RGB.
    PoseAPI.invalidate()
    actual_position, actual_quaternion = sensor.get_position_orientation()
    # New render products can return their initial RGB on the first read even
    # after the world pose is correct. Drain it and warm subsequent frames.
    for attempt in range(8):
        for _ in range(3):
            session.pipeline.og.sim.render()
        data, info = sensor.get_obs()
        visible_ids = [int(key) for key, label in info["seg_instance_id"].items()
            if str(label) == binding.name or same_body(target.prim_path, str(label))]
        visible_pixels = int(np.isin(_array(data["seg_instance_id"]), visible_ids).sum())
        if attempt >= 2 and visible_pixels >= 64:
            pixels = session._rgb_from_data(data, sensor.name)
            break
    else:
        raise RuntimeError(f"Object camera could not verify visible target pixels for {binding.name}.")
    support_key = selected.definition.qualification.object_supports.get(
        selected.definition.target, selected.definition.qualification.support)
    location = {"name": binding.name, "category": binding.category, "model": binding.model,
        "position_m": _array(target.get_position_orientation()[0]).tolist(),
        "aabb_min_m": low.tolist(), "aabb_max_m": high.tolist(),
        "support": selected.definition.bindings[support_key].name,
        "camera_position_m": position.tolist(), "centre_ray_clear": bool(max(scores)),
        "camera_actual_position_m": _array(actual_position).tolist(),
        "camera_quaternion_xyzw": _array(actual_quaternion).tolist(), "camera_fabric_synchronized": True,
        "camera_capture_protocol": "target_instance_verified_v3", "visible_target_pixels": visible_pixels,
        "initial_robot_distance_m": float(np.linalg.norm(base[:2] - centre[:2]))}
    return pixels, location


def capture_native_review_views(session, selected, output):
    """Export human views after native validation, without another scene startup."""
    from core.record.images import save_png
    from core.viewer_config import ViewerConfig

    if session.scene_camera is None:
        session.configure_scene_camera()
    if not session.global_cameras:
        session.configure_human_cameras(ViewerConfig.load(ROOT / selected.definition.runtime.viewer_config))
    source = session.observation_source
    source.include_scene = True
    # Drain newly created global render products as well as the close camera.
    for _ in range(3):
        session.invalidate_camera_cache()
        frame = source.capture(action_boundary=True)
        views = {"head": frame.rgb_views()["head_rgb"], "wrist": frame.rgb_views()["wrist_rgb"]}
        views.update({name: value for name, value in frame.human_views().items() if name.startswith("global_")})
        if frame.scene_rgb is not None:
            views["scene"] = frame.scene_rgb
    views["global_target"], location = capture_target_view(session, selected)
    selected.verify_unchanged()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, pixels in views.items():
        path = output / f"{name}.png"
        save_png(path, pixels)
        files[name] = str(path)
    return {"scene_file": str(selected.path), "scene_sha256": sha256(selected.path),
        "source_json": selected.definition.source.json_path, "source_sha256": sha256(selected.source_path),
        "variant": selected.variant, "seed": selected.seed, "images": files, "target_location": location,
        "job_id": os.environ["PBS_JOBID"], "model_calls": 0, "robot_actions": 0,
        "physical_grasp_verified": False}


def worker(args, store):
    from wia.tasks.loader import load_scene
    import wia.runtime.sim.esi_web as web
    from wia.runtime.commands.verify_esi_rigid_grasp import main as probe_main

    row = next(r for r in store.catalogue()["scenes"] if r["id"] == args.scene_id)
    out = store.output / "captures" / row["id"]
    out.mkdir(parents=True, exist_ok=True)
    if args.pending_only:
        return pending_worker(row, store, out)
    selected = load_scene(row["scene_file"], data_root=store.data_root,
                          variant=row["variant"], seed=row["seed"])
    out = store.output / "captures" / row["id"]
    out.mkdir(parents=True, exist_ok=True)

    def capture(session, metadata, controller, executor, **kwargs):
        from core.record.images import save_png
        source = session.observation_source
        source.include_scene = True
        frame = source.capture(action_boundary=True)
        views = {"head": frame.rgb_views()["head_rgb"], "wrist": frame.rgb_views()["wrist_rgb"]}
        views.update({name: value for name, value in frame.human_views().items()
                      if name.startswith("global_")})
        if frame.scene_rgb is not None:
            views["scene"] = frame.scene_rgb
        views["global_target"], target_location = capture_target_view(session, selected)
        if not any(name.startswith("global_") for name in views):
            raise RuntimeError("Global cameras were not captured.")
        selected.verify_unchanged()
        if fingerprint(selected.path, selected.source_path, selected.variant, selected.seed) != row["fingerprint"]:
            raise RuntimeError("The published scene changed during capture.")
        files = {}
        for name, pixels in views.items():
            path = out / f"{name}.png"
            save_png(path, pixels)
            files[name] = str(path)
        write_json(out / "preview_complete.json", {"schema": "scene_review_preview_v1",
            "id": row["id"], "fingerprint": row["fingerprint"], "variant": selected.variant,
            "seed": selected.seed, "images": files,
            "target_location": target_location,
            "provenance": "Fresh native initialization of the exact published final layout.",
            "job_id": os.environ["PBS_JOBID"], "model_calls": 0, "robot_actions": 0,
            "physical_grasp_verified": False})
        print(f"[scene-review-capture] Saved {len(files)} views for {row['source']}", flush=True)

    web.run_esi_control = capture
    return probe_main(["--viewer-only", "--scene-file", str(selected.path), "--data-root", str(store.data_root),
        "--variant", selected.variant, "--scene-seed", str(selected.seed), "--web", "--debug-cameras",
        "--human-preview", "--preview-fps", "0", "--gripper-control", "--trunk-control",
        "--parameterized-actions", "--primitive-profile", "fine", "--macro-execution", "metric_target",
        "--performance-profile", "benchmark", "--output-dir", str(out)])


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--capture-manifest" in arguments:
        return capture_manifest_entrypoint(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "explore_to_achieve/scenes")
    parser.add_argument("--variant", default="tabletop")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--scene-id", help=argparse.SUPPRESS)
    parser.add_argument("--require-object-view", action="store_true",
                        help="Recapture ready layouts whose matching previews lack a synchronized Object camera.")
    parser.add_argument("--pending-only", action="store_true",
                        help="Render pending layouts before controller startup; never qualify or publish them.")
    parser.add_argument("--wall-budget-s", type=float, default=600)
    parser.add_argument("--slow-wall-budget-s", type=float, default=1800)
    args = parser.parse_args(argv)
    if os.environ.get("PBS_JOBID") != args.job_id:
        parser.error("Run captures in the specified retained interactive GPU allocation.")
    store = SceneReview(args.conversion_root, args.output_dir, data_root=args.data_root, variant=args.variant)
    if args.scene_id:
        return worker(args, store)
    if not args.pending_only:
        pending_status = read_json(store.output / "pending_capture_summary.json", {}).get("status")
        if pending_status == "complete_with_failures":
            print("[scene-review] Retrying the explicitly requested pending previews before ready captures.", flush=True)
            main(["--conversion-root", str(store.conversion_root), "--output-dir", str(store.output),
                  "--data-root", str(store.data_root), "--variant", store.variant,
                  "--job-id", args.job_id, "--pending-only"])
    store.import_previews()
    # Promote complete views captured in an accepted validation process so the
    # coordinator can reuse them through its ordinary exact-layout cache.
    for row in store.catalogue()["scenes"]:
        imported = read_json(store.output / "imported" / row["id"] / "preview_complete.json", {})
        existing = read_json(store.output / "captures" / row["id"] / "preview_complete.json", {})
        target = imported.get("target_location", {})
        if (imported.get("fingerprint") == row["fingerprint"]
                and target.get("camera_capture_protocol") == "target_instance_verified_v3"
                and target.get("visible_target_pixels", 0) >= 64
                and len(imported.get("images", {})) == 10
                and not (existing.get("fingerprint") == row["fingerprint"]
                    and existing.get("target_location", {}).get("camera_capture_protocol") == "target_instance_verified_v3")):
            write_json(store.output / "captures" / row["id"] / "preview_complete.json", imported)
    selected_status = "pending" if args.pending_only else "ready"
    rows = [r for r in store.catalogue()["scenes"] if r["conversion_status"] == selected_status]
    # Capture the three previously unseen random final placements first. Slow
    # conference scenes follow the ordinary cases so review can start promptly.
    rows.sort(key=lambda r: (r["scene"].startswith("hall_conference"), r["has_final_preview"], r["source"]))
    results, started = [], time.monotonic()

    def save(status, current=None):
        summary_name = "pending_capture_summary.json" if args.pending_only else "capture_summary.json"
        write_json(store.output / summary_name, {"status": status, "job_id": args.job_id,
            "results": results, "current": current, "wall_elapsed_s": time.monotonic() - started,
            "model_calls": 0, "physical_grasp_verified": False})

    for row in rows:
        out = store.output / "captures" / row["id"]
        marker = read_json(out / "preview_complete.json", {})
        if (marker.get("fingerprint") == row["fingerprint"]
                and marker.get("variant") == row["variant"] and marker.get("seed") == row["seed"]
                and all(Path(p).is_file() for p in marker.get("images", {}).values())
                and any(v.startswith("global_") for v in marker.get("images", {}))
                and (not args.require_object_view or args.pending_only or (
                    "global_target" in marker.get("images", {})
                    and marker.get("target_location", {}).get("camera_capture_protocol") == "target_instance_verified_v3"))):
            results.append({"id": row["id"], "source": row["source"], "status": "complete", "reused": True})
            save("running")
            continue
        out.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while (out / f"native_{attempt:02d}.log").exists():
            attempt += 1
        budget = args.slow_wall_budget_s if row["scene"].startswith("hall_conference") else args.wall_budget_s
        save("running", {"id": row["id"], "scene": row["scene"], "source": row["source"]})
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--conversion-root", str(store.conversion_root),
            "--output-dir", str(store.output), "--data-root", str(store.data_root), "--variant", store.variant,
            "--job-id", args.job_id, "--scene-id", row["id"]]
        if args.pending_only:
            command.append("--pending-only")
        began = time.monotonic()
        with (out / f"native_{attempt:02d}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=budget)
            except KeyboardInterrupt:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                save("interrupted", {"id": row["id"], "scene": row["scene"], "source": row["source"]})
                return 130
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = -999
        marker = read_json(out / "preview_complete.json", {})
        complete = code == 0 and marker.get("fingerprint") == row["fingerprint"]
        result = {"id": row["id"], "source": row["source"], "status": "complete" if complete else "failed",
            "exit_code": code, "wall_elapsed_s": time.monotonic() - began,
            "wall_budget_s": budget, "native_log": str(out / f"native_{attempt:02d}.log")}
        write_json(out / f"attempt_{attempt:02d}.json", result)
        results.append(result)
        save("running")
        print(f"[scene-review] {len(results)}/{len(rows)} {result['status']} {row['source']}", flush=True)
    failed = sum(r["status"] == "failed" for r in results)
    save("complete" if not failed else "complete_with_failures")
    return 0 if not failed else 1


def capture_manifest_entrypoint(argv):
    """Single immutable queue job; the legacy sequential CLI remains unchanged."""
    import traceback
    from wia.runtime.scene_capture_adapters import capture_snapshot
    from wia.runtime.scene_capture_queue import read_json as capture_read, write_json as capture_write

    parser = argparse.ArgumentParser(description="Capture one accepted immutable scene snapshot.")
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--capture-output", type=Path, required=True)
    parser.add_argument("--gpu-info", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--execution-mode", choices=("local", "pbs"), default="pbs")
    args = parser.parse_args(argv)
    gpu = capture_read(args.gpu_info)
    if args.execution_mode == "pbs" and os.environ.get("PBS_JOBID") != args.job_id:
        parser.error("Use the specified retained interactive PBS allocation.")
    if args.execution_mode == "local":
        if gpu.get("execution_mode") != "local" or gpu.get("pbs_job_id"):
            parser.error("Local capture requires the worker's direct GPU assignment.")
        if capture_read(args.capture_manifest).get("job_id") != args.job_id:
            parser.error("Local capture job identity differs from its immutable manifest.")
    try:
        capture_snapshot(args.capture_manifest, args.capture_output, gpu)
        return 0
    except Exception as error:
        capture_write(args.capture_output / "failure.json", {
            "error": str(error), "error_type": type(error).__name__,
            "failure_stage": capture_read(args.capture_output / "progress.json", {}).get("stage", "snapshot_validation"),
            "retryable": type(error).__name__ == "NativeCameraNotReady", "complete": False})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
