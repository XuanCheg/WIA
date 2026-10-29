#!/usr/bin/env python3
"""Convert canonical Rigid Containment scenes and validate each before publication.

Preview: .venv/bin/python src/wia/runtime/commands/convert_esi_rigid_catalog.py --dry-run
Run in the retained GPU shell: python src/wia/runtime/commands/convert_esi_rigid_catalog.py
    --output-dir /project/scratch/rigid_conversion --attempts 12 --seed 0
Each source is sampled; restart mode independently reloads the chosen placement.
Use --validation-mode screen for one native screen and deferred manual review.
Use --resume to retain completed results after an interrupted run.
Only passing descriptions become ready. Tasks and raw JSONs are never changed.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

import yaml

from wia.runtime.scene_review import write_json
from wia.tasks.loader import SCENE_ROOT
from wia.tasks.schema import SceneDefinition
from wia.runtime.commands.sample_esi_supports import positive_int, selections


def native_worker(scene_file, output, args, *, validate=False):
    output.mkdir(parents=True, exist_ok=True)
    if not validate:
        write_json(output / "warm_keepalive.json", {"enabled": True,
            "job_id": os.environ.get("PBS_JOBID"),
            "policy": "Retain failed sampling worlds for same-process code reload and repair."})
    command = [sys.executable, str(ROOT / "src/wia/runtime/commands/sample_esi_supports.py"), "--worker",
        "--base-scene", str(scene_file), "--data-root", str(args.data_root.resolve()),
        "--output-dir", str(output), "--variant", "tabletop", "--seed", str(args.seed),
        "--attempts", str(args.attempts), "--placements", "1", "--wall-budget-s", str(args.wall_budget_s)]
    if validate:
        command.append("--validate-only")
    command.append("--capture-review")
    source_name = yaml.safe_load(scene_file.read_text())["source"]["json"]
    preferences = getattr(args, "support_preferences", None)
    if preferences is not None:
        for name in json.loads(preferences.read_text()).get(source_name, []):
            command.extend(("--preferred-support", name))
    if getattr(args, "resample_current_placement", False):
        command.append("--resample-current-placement")
    if not getattr(args, "random_support_sampling", False):
        command.append("--targeted-support-sampling")
    if getattr(args, "no_added_support", False):
        command.append("--no-added-support")
    if getattr(args, "require_planar_footprint", False):
        command.append("--require-planar-footprint")
    initialization_budget = (getattr(args, "slow_initialization_wall_budget_s", 1800)
        if "hall_conference" in source_name else args.initialization_wall_budget_s)
    budget = initialization_budget if validate else args.wall_budget_s + initialization_budget
    with (output / "native.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + budget
            while True:
                remaining = deadline - time.monotonic()
                warm = output / "warm_state.json"
                state = json.loads(warm.read_text()) if warm.is_file() else {}
                retained = (state.get("pid") == process.pid
                    and state.get("job_id") == os.environ.get("PBS_JOBID")
                    and state.get("status") in ("waiting", "running", "complete"))
                if remaining <= 0 and not retained:
                    raise subprocess.TimeoutExpired(command, budget)
                try:
                    return process.wait(timeout=30 if retained else min(30, remaining))
                except subprocess.TimeoutExpired:
                    continue
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if isinstance(error, KeyboardInterrupt):
                raise
            return -999


def failure_reason(output, exit_code):
    status = output / "scene_status.json"
    if status.is_file():
        value = json.loads(status.read_text())
        if value.get("qualification_status") == "skipped":
            return value.get("reason", "Scene initialization or physical preflight failed.")
    bank = output / "support_bank.json"
    if bank.is_file():
        value = json.loads(bank.read_text())
        return value.get("reason") or dict(Counter(row.get("reason", "unknown") for row in value["attempts"]))
    return "Worker wall timeout." if exit_code == -999 else f"Worker did not complete; exit code {exit_code}. Inspect native.log."


def restrict_sources(candidates, excluded, source_list, data_root):
    """Keep inspection-only sources unchanged without forging human approval."""
    if source_list is None:
        return candidates, excluded
    selected = set(json.loads(source_list.read_text()))
    available = {str(source.resolve().relative_to(data_root.resolve())) for source, _ in candidates}
    if not selected or not selected <= available:
        raise ValueError("Source list must contain eligible canonical relative source paths.")
    kept = []
    excluded = list(excluded)
    for source, definition in candidates:
        if str(source.resolve().relative_to(data_root.resolve())) in selected:
            kept.append((source, definition))
        else:
            excluded.append({"source": str(source), "reason": "inspection_only_preserved"})
    return kept, excluded


def publish_result(data_root, source, row, *, candidate=None, validation_dir=None, sampling_dir=None):
    """Require native evidence for the exact candidate under the selected protocol."""
    root = Path(data_root).resolve()
    source = Path(source).resolve()
    relative = source.relative_to(root)
    descriptor = source.with_suffix(".yaml")
    index_path = root / "catalog.json"
    index = json.loads(index_path.read_text())
    matches = [item for item in index["scenes"]
        if item["source_json"] == str(relative) and item["scene_file"] == str(relative.with_suffix(".yaml"))]
    if len(matches) != 1:
        raise ValueError("Exactly one canonical catalogue descriptor is required.")
    entry = matches[0]
    if row["status"] == "ready":
        value = yaml.safe_load(candidate.read_text())
        if value["source"]["json"] != str(relative):
            raise ValueError("Converted scene refers to a different source question.")
        if row.get("verification_level") == "approximate_native_initialization":
            status = json.loads((sampling_dir / "scene_status.json").read_text())
            manifest = json.loads((sampling_dir / "scene_manifest.json").read_text())
            if status.get("qualification_status") not in ("accepted_minimal_patch", "accepted_original"):
                raise ValueError("Cannot publish a fallback from skipped native initialization.")
            if Path(manifest["source_path"]).resolve() != source or Path(manifest["scene_file"]).resolve() != candidate.resolve():
                raise ValueError("Native initialization belongs to a different source or candidate.")
            if hashlib.sha256(candidate.read_bytes()).hexdigest() != manifest["scene_sha256"]:
                raise ValueError("Initialized fallback differs from the natively accepted scene.")
        elif row.get("verification_level") == "approximate_native_screen":
            status = json.loads((sampling_dir / "scene_status.json").read_text())
            bank = json.loads((sampling_dir / "support_bank.json").read_text())
            if status.get("qualification_status") not in ("accepted_minimal_patch", "accepted_original") or bank.get("status") != "screened":
                raise ValueError("Cannot publish without a completed native support screen.")
            accepted = [item for item in bank["accepted"]
                if (sampling_dir / item["scene_file"]).resolve() == candidate.resolve() and item.get("accepted")]
            if bank["source_json"] != str(relative) or len(accepted) != 1:
                raise ValueError("Native screen does not certify this source and candidate.")
            certified = accepted[0]
            if certified.get("scene_sha256") and certified["scene_sha256"] != hashlib.sha256(candidate.read_bytes()).hexdigest():
                raise ValueError("Converted geometry differs from the completed native screen.")
            poses = [patch for variant in value["scene"]["variants"].values() for patch in variant["patches"]
                if patch["operation"] == "object_pose" and patch["binding"] == value["target"]]
            if len(poses) != 1 or poses[0]["position"] != certified["position"] or poses[0]["quaternion_xyzw"] != certified["quaternion_xyzw"]:
                raise ValueError("Converted pickup pose differs from the completed native screen.")
            if value["bindings"]["pickup_support"]["name"] != certified["support"]:
                raise ValueError("Converted pickup support differs from the completed native screen.")
        else:
            status = json.loads((validation_dir / "scene_status.json").read_text())
            marker = validation_dir / "validation_complete.json"
            if status.get("qualification_status") not in ("accepted_minimal_patch", "accepted_original") or not marker.is_file():
                raise ValueError("Cannot publish without independent native initialization acceptance.")
            marker_value = json.loads(marker.read_text())
            if marker_value["source_json"] != str(relative):
                raise ValueError("Native validation belongs to a different source question.")
            tested_path = Path(marker_value["scene_file"])
            manifest = json.loads((validation_dir / "scene_manifest.json").read_text())
            if (Path(manifest["source_path"]).resolve() != source
                    or Path(manifest["scene_file"]).resolve() != tested_path.resolve()
                    or hashlib.sha256(tested_path.read_bytes()).hexdigest() != manifest["scene_sha256"]):
                raise ValueError("Restarted descriptor differs from its native validation receipt.")
            tested = yaml.safe_load(tested_path.read_text())
            tested["status"] = value.get("status", "ready")
            if tested != value:
                raise ValueError("Converted scene differs from the independently restarted placement.")
        value["status"] = "ready"
        definition = SceneDefinition.model_validate(value)
        descriptor.write_text(yaml.safe_dump(definition.model_dump(mode="json", by_alias=True), sort_keys=False))
        entry.update(status="ready", conversion="converted_and_native_validated",
            converted_from=str(candidate), native_qualification=status["qualification_status"],
            conversion_result=row)
    else:
        value = yaml.safe_load(descriptor.read_text())
        value["status"] = "pending"
        descriptor.write_text(yaml.safe_dump(value, sort_keys=False))
        entry.update(status="pending", conversion="conversion_failed", native_qualification="failed",
            conversion_result=row)
    index["status_counts"] = dict(Counter(item["status"] for item in index["scenes"]))
    index["status"] = "partially_ready" if index["status_counts"].get("ready", 0) else "pending"
    index["native_qualification"] = "partial_native_screening" if index["status_counts"].get("ready", 0) else "no_ready_scenes"
    index_path.write_text(json.dumps(index, indent=2) + "\n")


def initialized_tabletop_fallback(data_root, row, *, approved=False):
    """An initial-table fallback requires explicit human layout approval."""
    if not approved:
        return None
    if row["status"] != "pending" or row.get("failure_stage") != "sampling":
        return None
    folder = Path(row["output"])
    sampling = folder / "sampling"
    required = [sampling / "scene_status.json", sampling / "scene_manifest.json",
        sampling / "private_scene/qualification.json", sampling / "trunk_initialization.json"]
    if not all(path.is_file() for path in required):
        return None
    status = json.loads(required[0].read_text())
    if status.get("qualification_status") not in ("accepted_minimal_patch", "accepted_original"):
        return None
    candidate = folder / "initialization.yaml"
    definition = SceneDefinition.model_validate(yaml.safe_load(candidate.read_text()))
    checks = json.loads(required[2].read_text())["checks"]
    target = checks.get("supported_objects", {}).get(definition.target)
    if not target or not target.get("passed") or checks.get("robot_nonfloor_contacts"):
        return None
    base = json.loads(required[3].read_text())["base_position"]
    centre = [(lo + hi) / 2 for lo, hi in zip(target["aabb_min"], target["aabb_max"])]
    distance = math.dist(base[:2], centre[:2])
    if distance > .90 or not .35 <= target["support_surface_z_m"] <= 1.10:
        return None
    result = dict(row)
    result.update(status="ready", verification_level="approximate_native_initialization",
        manual_review_required=True, independent_restart_verified=False, random_placement=False,
        fallback="previously_accepted_initial_tabletop", random_sampling_failure=row.get("reason"),
        support=target["support_name"], approximate_initial_stance_distance_m=distance,
        route_connectivity_screen_passed=False,
        aperture_warning=min(hi-lo for lo, hi in zip(target["aabb_min"], target["aabb_max"])) > .1,
        candidate_file=str(candidate), sampling_dir=str(sampling), physical_grasp_verified=False)
    result.pop("failure_stage", None)
    result.pop("reason", None)
    publish_result(data_root, Path(data_root) / row["source"], result, candidate=candidate, sampling_dir=sampling)
    return result


def restart_added_support(source, row, bank, sampling, folder, args):
    """Treat added furniture as an unaccepted draft until its exact native restart passes."""
    if bank.get("source_json") != row["source"]:
        raise ValueError("Added-support drafts belong to a different source.")
    attempts = []
    for index, draft in enumerate(bank.get("fallback_candidates", [])):
        candidate = sampling / draft["scene_file"]
        value = yaml.safe_load(candidate.read_text())
        if value["bindings"]["pickup_support"]["name"] == value["bindings"][value["qualification"]["support"]]["name"]:
            raise ValueError("An added pickup support cannot reuse the initial work table.")
        value["status"] = "ready"
        restart = folder / f"separate_restart_{index:03d}.yaml"
        restart.write_text(yaml.safe_dump(value, sort_keys=False))
        validation = folder / f"separate_validation_{index:03d}"
        code = native_worker(restart, validation, args, validate=True)
        attempt = {"candidate_file": str(candidate), "validation_dir": str(validation), "exit_code": code}
        attempts.append(attempt)
        if code == 0 and (validation / "validation_complete.json").is_file():
            checks = json.loads((validation / "validation_complete.json").read_text()).get("added_pickup_support_checks", {})
            if not checks.get("support_contact_passed"):
                attempt["reason"] = "Added-support contact/stance screen did not complete."
                continue
            result = {**row, "status": "ready", "verification_level": "approximate_reachability_then_native_restart",
                "manual_review_required": True, "independent_restart_verified": True,
                "support": draft["support"], "support_kind": "added_separate_pickup_surface",
                "initial_robot_distance_m": draft["initial_robot_distance_m"],
                "route_connectivity_screen_passed": checks["route_connectivity_screen_passed"],
                "aperture_warning": checks["aperture_warning"], "physical_grasp_verified": False,
                "candidate_file": str(candidate), "validation_dir": str(validation),
                "added_support_attempts": attempts, "validation_exit_code": code}
            result.pop("failure_stage", None)
            result.pop("reason", None)
            publish_result(args.data_root, source, result, candidate=candidate, validation_dir=validation)
            return result
        attempt["reason"] = failure_reason(validation, code)
    row["added_support_attempts"] = attempts
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=SCENE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--review-file", type=Path,
        help="Preserve matching Approved layouts and omit manually excluded scenes.")
    parser.add_argument("--source-list", type=Path,
        help="JSON list of canonical relative sources to repair; preserve other eligible scenes for inspection.")
    parser.add_argument("--support-preferences", type=Path,
        help="JSON mapping from canonical relative sources to ordered named same-room support preferences.")
    parser.add_argument("--resample-current-placement", action="store_true",
        help="Do not accept the current off-table pickup pose unchanged during a requested relocation.")
    parser.add_argument("--random-support-sampling", action="store_true",
        help="Use the previous random furniture/pose proposals instead of reachable-edge proposals.")
    parser.add_argument("--no-added-support", action="store_true",
        help="Require existing furniture; keep unsuccessful sources pending without added stands.")
    parser.add_argument("--require-planar-footprint", action="store_true",
        help="Use the legacy planar footprint test instead of stable native support alone.")
    parser.add_argument("--grasp-width-m", type=float, default=.08,
        help="Uniformly shrink oversized targets toward this width, within the measured 0.10 m opening.")
    parser.add_argument("--keep-target-scale", action="store_true", help="Retain existing target scales.")
    parser.add_argument("--attempts", type=positive_int, default=12)
    parser.add_argument("--wall-budget-s", type=float, default=180.)
    parser.add_argument("--initialization-wall-budget-s", type=positive_int, default=300,
        help="Wall allowance for native loading/initialization, separate from support sampling.")
    parser.add_argument("--slow-initialization-wall-budget-s", type=positive_int, default=1800,
        help="Separate startup budget for the previously measured slow conference scenes.")
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validation-mode", choices=("restart", "screen"), default="restart")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.wall_budget_s <= 0 or not 0 < args.grasp_width_m <= .10:
        parser.error("Seed must be nonnegative, wall budget positive, and grasp width within (0, 0.10] m.")
    args.source = None
    args.source_dir = args.data_root / "Physical Structure/Rigid Containment"
    candidates, excluded = selections(args)
    candidates, excluded = restrict_sources(candidates, excluded, args.source_list, args.data_root)
    candidates.sort(key=lambda item: ("hall_conference" in item[0].name, str(item[0])))
    if args.dry_run:
        print(json.dumps({"candidate_count": len(candidates),
            "sources": [str(path) for path, _ in candidates],
            "excluded_counts": dict(Counter(row["reason"] for row in excluded)),
            "target_sizing": {str(path.resolve().relative_to(args.data_root.resolve())):
                args.selection_details[str(path.resolve())] for path, _ in candidates},
            "pickup_policy": "Native same-room supports first; added separate support requires native restart.",
            "initial_work_table_fallback": "Disabled for unapproved layouts.",
            "verification_level": "approximate_native_screen" if args.validation_mode == "screen" else "approximate_reachability_then_native_restart"}, indent=2))
        return 0
    if not os.environ.get("PBS_JOBID") or args.output_dir is None:
        parser.error("Native conversion requires --output-dir and the retained interactive GPU shell.")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "conversion_summary.json").exists() and not args.resume:
        parser.error("Preserve previous conversion evidence; use a fresh output directory.")
    config = {"seed": args.seed, "attempts": args.attempts,
        "sampling_wall_budget_s": args.wall_budget_s, "placements_per_source": 1,
        "initialization_wall_budget_s": args.initialization_wall_budget_s,
        "slow_initialization_wall_budget_s": args.slow_initialization_wall_budget_s,
        "grasp_width_m": args.grasp_width_m, "keep_target_scale": args.keep_target_scale,
        "review_sha256": hashlib.sha256(args.review_file.read_bytes()).hexdigest() if args.review_file else None,
        "source_list_sha256": hashlib.sha256(args.source_list.read_bytes()).hexdigest() if args.source_list else None,
        "pickup_policy": "native_support_then_separate_added_support_v2"}
    if args.support_preferences is not None:
        config["support_preferences_sha256"] = hashlib.sha256(args.support_preferences.read_bytes()).hexdigest()
    if args.resample_current_placement:
        config["resample_current_placement"] = True
    config['support_proposal_mode'] = 'random' if args.random_support_sampling else 'reachable_edge'
    config['allow_added_support'] = not args.no_added_support
    config['support_footprint_policy'] = 'planar_local_raycast' if args.require_planar_footprint else 'native_contact_and_settling'
    if args.resume:
        summary = json.loads((root / "conversion_summary.json").read_text())
        prior_config = dict(summary["config"])
        prior_config.setdefault("initialization_wall_budget_s", 300)
        prior_config.setdefault('support_proposal_mode', 'random')
        prior_config.setdefault('allow_added_support', True)
        prior_config.setdefault('support_footprint_policy', 'planar_local_raycast')
        if prior_config != config or summary["job_id"] != os.environ["PBS_JOBID"]:
            parser.error("Resume requires the same sampling settings and allocation.")
        summary["config"] = prior_config
        expected = {str(source.resolve().relative_to(args.data_root.resolve())) for source, _ in candidates}
        if summary["candidate_count"] != len(candidates) or any(row["source"] not in expected for row in summary["results"]):
            parser.error("Resume source selection differs from the original run.")
        summary.setdefault("protocol_changes", []).append({"at_utc": datetime.now(timezone.utc).isoformat(),
            "validation_mode": args.validation_mode, "completed_results_retained": len(summary["results"])})
        summary["status"] = "running"
    else:
        backup = ROOT / "explore_to_achieve/temp" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") / "rigid_before_conversion"
        backup.mkdir(parents=True)
        shutil.copy2(args.data_root / "catalog.json", backup / "catalog.json")
        for source, _ in candidates:
            destination = backup / source.resolve().relative_to(args.data_root.resolve()).with_suffix(".yaml")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.with_suffix(".yaml"), destination)
        summary = {"status": "running", "candidate_count": len(candidates), "excluded": excluded,
            "job_id": os.environ["PBS_JOBID"], "config": config,
            "backup": str(backup), "physical_grasp_verified": False, "model_calls": 0,
            "results": [], "current": None}
        catalogue = json.loads((args.data_root / "catalog.json").read_text())
        preserved = {str(Path(r["source"]).resolve().relative_to(args.data_root.resolve())): r["reason"]
            for r in excluded if r["reason"] in ("approved_layout_preserved", "inspection_only_preserved")}
        summary["preserved_scenes"] = [{**entry.get("conversion_result", {}),
            "source": entry["source_json"], "status": entry["status"],
            "preserved_human_approval": preserved[entry["source_json"]] == "approved_layout_preserved",
            "preserved_for_inspection": preserved[entry["source_json"]] == "inspection_only_preserved"}
            for entry in catalogue["scenes"] if entry["source_json"] in preserved
            and entry["scene_file"] == str(Path(entry["source_json"]).with_suffix(".yaml"))]
        if len(summary["preserved_scenes"]) != len(preserved):
            raise ValueError("Every approved source must have exactly one canonical catalogue descriptor.")
        if args.review_file:
            shutil.copy2(args.review_file, root / "human_review_snapshot.json")
    summary["validation_mode"] = args.validation_mode
    previous_elapsed = summary.get("wall_elapsed_s", 0.)
    started = time.monotonic()

    def save():
        summary["wall_elapsed_s"] = previous_elapsed + time.monotonic() - started
        summary["status_counts"] = dict(Counter(row["status"] for row in summary["results"]))
        summary["initialized_tabletop_fallback_count"] = sum(row.get("fallback") is not None for row in summary["results"])
        (root / "conversion_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for number, (source, definition) in enumerate(candidates, 1):
        relative = source.resolve().relative_to(args.data_root.resolve())
        if any(row["source"] == str(relative) for row in summary["results"]):
            continue
        folder = root / "cases" / relative.with_suffix("")
        folder.mkdir(parents=True, exist_ok=True)
        initialization = folder / "initialization.yaml"
        initialization.write_text(yaml.safe_dump(definition.model_dump(mode="json", by_alias=True), sort_keys=False))
        summary["current"] = {"index": number, "source": str(relative), "phase": "sampling"}
        save()
        print(f"[rigid-convert] {number}/{len(candidates)} sampling {relative}", flush=True)
        sampling = folder / "sampling"
        existing_bank = sampling / "support_bank.json"
        reuse = args.resume and args.validation_mode == "screen" and existing_bank.is_file() and json.loads(existing_bank.read_text()).get("status") == "screened"
        if reuse:
            code = 0
        else:
            if sampling.exists():
                sampling.rename(folder / ("sampling_interrupted_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
            code = native_worker(initialization, sampling, args)
        bank_path = sampling / "support_bank.json"
        sizing = args.selection_details[str(source.resolve())]
        row = {"source": str(relative), "output": str(folder), "status": "pending", "sampling_exit_code": code,
            "target_sizing": sizing, "physical_grasp_verified": False,
            "source_answer_revalidated": False, "initial_work_table_fallback_allowed": False}
        if code == 0 and bank_path.is_file() and json.loads(bank_path.read_text())["accepted"]:
            bank = json.loads(bank_path.read_text())
            chosen = bank["accepted"][0]
            candidate = sampling / chosen["scene_file"]
            if args.validation_mode == "screen":
                row.update(status="ready", verification_level="approximate_native_screen",
                    manual_review_required=True, independent_restart_verified=False,
                    support=chosen["support"], route_connectivity_screen_passed=chosen["route_connectivity_screen_passed"],
                    aperture_warning=chosen["aperture_warning"], sampling_wall_s=bank["wall_elapsed_s"],
                    candidate_file=str(candidate), sampling_dir=str(sampling), physical_grasp_verified=False,
                    reused_complete_screen=reuse)
                publish_result(args.data_root, source, row, candidate=candidate, sampling_dir=sampling)
                summary["results"].append(row)
                summary["current"] = None
                save()
                print(f"[rigid-convert] {number}/{len(candidates)} ready {row['support']} (manual review required)", flush=True)
                continue
            value = yaml.safe_load(candidate.read_text())
            value["status"] = "ready"
            restart = folder / "restart_scene.yaml"
            restart.write_text(yaml.safe_dump(value, sort_keys=False))
            validation = folder / "validation"
            summary["current"]["phase"] = "independent_native_restart"
            save()
            code = native_worker(restart, validation, args, validate=True)
            row["validation_exit_code"] = code
            if code == 0 and (validation / "validation_complete.json").is_file():
                row.update(status="ready", verification_level="approximate_reachability_then_native_restart",
                    support=chosen["support"], route_connectivity_screen_passed=chosen["route_connectivity_screen_passed"],
                    aperture_warning=chosen["aperture_warning"], sampling_wall_s=bank["wall_elapsed_s"],
                    candidate_file=str(candidate), validation_dir=str(validation), physical_grasp_verified=False)
                publish_result(args.data_root, source, row, candidate=candidate, validation_dir=validation)
            else:
                row.update(failure_stage="independent_native_restart", reason=failure_reason(validation, code))
                publish_result(args.data_root, source, row)
        else:
            row.update(failure_stage="sampling", reason=failure_reason(sampling, code))
            bank = json.loads(bank_path.read_text()) if code == 0 and bank_path.is_file() else {}
            if bank.get("fallback_candidates"):
                summary["current"]["phase"] = "added_support_native_restart"
                save()
            fallback = restart_added_support(source, row, bank, sampling, folder, args) if bank.get("fallback_candidates") else None
            if fallback is not None:
                row = fallback
            else:
                publish_result(args.data_root, source, row)
        summary["results"].append(row)
        summary["current"] = None
        save()
        print(f"[rigid-convert] {number}/{len(candidates)} {row['status']} {row.get('reason', row.get('support'))}", flush=True)
    summary["status"] = "complete"
    save()
    print(json.dumps({"summary": str(root / "conversion_summary.json"), "status_counts": summary["status_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
