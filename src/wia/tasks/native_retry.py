"""Retain a failed native support screen and reload sampling code in place.

Converter workers enable this through warm_keepalive.json in their output.
Write {"action": "retry", "config": {...}} to warm/request.json for a new
screen in the same simulator, or {"action": "snapshot"} for geometry evidence.
Every request and attempt is retained; no robot action is performed here.
 Run: python -m pytest tests/test_extraction.py. """
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from wia.authoring.io import read_json, sha256, review_write_json as write_json


def promote_retry(output, attempt, bank):
    """Publish a retry index while retaining the original bank and image files."""
    output, attempt = Path(output), Path(attempt)
    history = output / "warm/history"
    history.mkdir(parents=True, exist_ok=True)
    original = output / "support_bank.json"
    saved = history / "original_support_bank.json"
    if original.is_file() and not saved.exists():
        shutil.copy2(original, saved)
    bank = json.loads(json.dumps(bank))
    for row in bank["accepted"]:
        candidate = attempt / row["scene_file"]
        row["scene_file"] = str(candidate.relative_to(output))
        marker = attempt / "review_views" / candidate.stem / "native_review_complete.json"
        if marker.is_file():
            destination = output / "review_views" / candidate.stem / marker.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(marker, destination)
    bank["warm_retry"] = {"attempt_dir": str(attempt), "original_bank": str(saved),
        "same_native_process": True, "worker_pid": os.getpid(), "model_calls": 0}
    write_json(original, bank)


def install_sampling_keepalive(controller):
    """Attach an opt-in main-thread retry hook before this worker can tear down."""
    argv = sys.argv
    if Path(argv[0]).name != "sample_esi_supports.py" or "--worker" not in argv:
        return
    output = Path(argv[argv.index("--output-dir") + 1])
    marker = read_json(output / "warm_keepalive.json", {})
    if not marker.get("enabled") or marker.get("job_id") != os.environ.get("PBS_JOBID"):
        return
    session = controller.session
    if getattr(session, "sampling_keepalive_installed", False):
        return
    original_close = session.close

    def close_after_repair():
        bank = read_json(output / "support_bank.json", {})
        if bank.get("status") != "screened" or bank.get("review_capture_failures"):
            keepalive(session, controller, output, argv)
        original_close()

    session.close = close_after_repair
    session.sampling_keepalive_installed = True
    print("[native-retry] Failed-screen keepalive installed for", output, flush=True)


def keepalive(session, controller, output, argv):
    """Handle file requests on the owning simulator thread until a screen passes."""
    from wia.tasks.loader import load_scene
    import wia.tasks.support_sampling as sampling

    output = Path(output)
    warm = output / "warm"
    warm.mkdir(exist_ok=True)
    selected = load_scene(Path(argv[argv.index("--base-scene") + 1]),
        data_root=Path(argv[argv.index("--data-root") + 1]),
        variant=argv[argv.index("--variant") + 1], seed=int(argv[argv.index("--seed") + 1]))
    initial_bank = read_json(output / "support_bank.json", {})
    config_values = initial_bank.get("config", {})
    state = {"status": "waiting", "pid": os.getpid(), "job_id": os.environ.get("PBS_JOBID"),
        "source_json": selected.definition.source.json_path, "world_startups": 1,
        "requests_completed": 0, "reason": initial_bank.get("reason"),
        "physical_grasp_verified": False, "model_calls": 0}

    def save():
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "warm_state.json", state)

    save()
    print("[native-retry] Keeping the simulator loaded; waiting for", warm / "request.json", flush=True)
    while True:
        request_file = warm / "request.json"
        if not request_file.is_file():
            time.sleep(.5)
            continue
        index = state["requests_completed"] + 1
        retained = warm / "requests" / f"{index:03d}.json"
        retained.parent.mkdir(exist_ok=True)
        request_file.replace(retained)
        state.pop("error", None)
        state.update(status="running", requests_completed=index)
        save()
        try:
            request = json.loads(retained.read_text())
            sampling = importlib.reload(sampling)
            values = {**config_values, **request.get("config", {})}
            if "preferred_support_names" in values:
                values["preferred_support_names"] = tuple(values["preferred_support_names"])
            config = sampling.SamplingConfig(**values)
            attempt = warm / f"attempt_{index:03d}"
            sampler = sampling.SupportSampler(session, controller, None, config, attempt, selected)
            if request["action"] == "snapshot":
                from wia.native.utils import _array
                import numpy as np

                supports = sampler.eligible_supports(selected.definition.bindings[
                    selected.definition.qualification.support].name)
                geometry = {obj.name: {"category": obj.category, "in_rooms": obj.in_rooms,
                    "aabb_min_m": _array(obj.aabb[0]).tolist(), "aabb_max_m": _array(obj.aabb[1]).tolist()}
                    for obj in supports}
                trav, free = sampler.navigation_grid()
                attempt.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(attempt / "navigation_grid.npz", free=free,
                    resolution=trav.map_resolution, map_size=trav.map_size)
                write_json(attempt / "snapshot.json", {"supports": geometry, "robot": controller.pose(),
                    "support_diagnostics": sampler.report.get("preferred_support_diagnostics"),
                    "chassis_radius_m": sampler.report["robot_footprint_radius_m"]})
                state.update(status="waiting", last_result=str(attempt / "snapshot.json"))
            elif request["action"] == "retry":
                qualification = read_json(output / "scene_status.json", {})
                if qualification.get("qualification_status") not in ("accepted_original", "accepted_minimal_patch"):
                    raise ValueError("Native initialization must pass before retrying pickup placement.")

                def capture(candidate):
                    from wia.tasks.schema import SceneDefinition
                    import yaml

                    cameras = importlib.reload(importlib.import_module("wia.runtime.commands.capture_esi_scene_reviews"))
                    definition = SceneDefinition.model_validate(yaml.safe_load(candidate.read_text()))
                    placement = replace(selected, definition=definition, path=candidate,
                        file_sha256=sha256(candidate))
                    views = attempt / "review_views" / candidate.stem
                    receipt = cameras.capture_native_review_views(session, placement, views)
                    write_json(views / "native_review_complete.json", receipt)

                sampler.on_accepted_placement = capture
                bank = sampler.run()
                state["last_result"] = str(attempt / "support_bank.json")
                if bank.get("status") == "screened" and not bank.get("review_capture_failures"):
                    promote_retry(output, attempt, bank)
                    state.update(status="complete", accepted=len(bank["accepted"]))
                    save()
                    print("[native-retry] Same-world retry passed; returning completed evidence.", flush=True)
                    return
                state.update(status="waiting", reason=bank.get("reason"))
            else:
                raise ValueError("Warm requests must use retry or snapshot.")
        except Exception as error:
            state.update(status="waiting", error=f"{type(error).__name__}: {error}")
            print("[native-retry]", state["error"], flush=True)
        save()
