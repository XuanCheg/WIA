#!/usr/bin/env python3
"""Batch-screen seeded pickup placements on existing same-room furniture.

Preview on CPU: .venv/bin/python src/wia/runtime/commands/sample_esi_supports.py --dry-run
Run in the retained GPU shell: python src/wia/runtime/commands/sample_esi_supports.py
    --output-dir /project/scratch/supports --limit 1 --attempts 8 --placements 2
Use --base-scene FILE --variant NAME with --source SAMPLE.json to reuse a tested
physical layout. Outputs stay pending and never create task files. Reachability
is approximate; no physical navigation or grasp command is executed.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

import yaml

from wia.tasks.loader import SCENE_ROOT, UniqueLoader, load_scene
from wia.tasks.support_sampling import SamplingConfig
from wia.tasks.pickup_policy import matching_review, review_entries, size_target
from wia.tasks.tabletop import generate_tabletop


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive.")
    return number


def selections(args):
    paths = [args.source] if args.source is not None else sorted(args.source_dir.rglob("*.json"))
    candidates, excluded = [], []
    reviews = review_entries(getattr(args, "review_file", None))
    args.selection_details = {}
    for path in paths:
        descriptor = path.with_suffix(".yaml")
        status = yaml.load(descriptor.read_text(), Loader=UniqueLoader).get("status") if descriptor.is_file() else None
        if status == "discarded":
            excluded.append({"source": str(path), "reason": "discarded"})
        elif "__" in path.stem:
            excluded.append({"source": str(path), "reason": "canonical_questions_only"})
        else:
            try:
                review = matching_review(path, args.data_root, reviews)
                if review and review["decision"] in ("approved", "skipped"):
                    excluded.append({"source": str(path), "reason": "approved_layout_preserved"
                        if review["decision"] == "approved" else "human_review_excluded"})
                    continue
                if status == "ready":
                    definition = load_scene(descriptor, data_root=args.data_root, variant="tabletop").definition
                else:
                    definition, _ = generate_tabletop(path, args.data_root, seed=args.seed)
                definition, details = size_target(definition, path, args.data_root,
                    grasp_width_m=getattr(args, "grasp_width_m", .08),
                    keep_scale=getattr(args, "keep_target_scale", False))
                definition = definition.model_copy(update={"seed": args.seed})
                args.selection_details[str(path.resolve())] = {**details, "review": review}
                candidates.append((path, definition))
            except (ValueError, KeyError) as error:
                excluded.append({"source": str(path), "reason": str(error)})
    if args.limit is not None:
        candidates = candidates[:args.limit]
    return candidates, excluded


def worker(args):
    from wia.tasks.support_sampling import SupportSampler
    import wia.runtime.sim.esi_web as web
    from wia.runtime.commands.verify_esi_rigid_grasp import main as probe_main

    selected = load_scene(args.base_scene, data_root=args.data_root, variant=args.variant, seed=args.seed)
    config = SamplingConfig(seed=args.seed, attempts=args.attempts, placements=args.placements,
        wall_budget_s=args.wall_budget_s, minimum_height_m=args.minimum_height_m,
        maximum_height_m=args.maximum_height_m, require_connected_route=args.require_connected_route,
        preferred_support_names=tuple(args.preferred_support),
        reuse_current_placement=not args.resample_current_placement,
        allow_added_support=not bool(args.preferred_support) and not args.no_added_support,
        targeted_support_sampling=args.targeted_support_sampling,
        require_planar_footprint=args.require_planar_footprint)

    def screen(session, metadata, controller, executor, **kwargs):
        if args.validate_only:
            pickup = selected.definition.bindings.get("pickup_support")
            pickup_checks = {}
            if pickup and pickup.name.startswith("pickup_surface_"):
                from wia.tasks.support_sampling import same_body
                from wia.runtime.sim.esi_omnigibson import _array
                import numpy as np

                sampler = SupportSampler(session, controller, executor, config, args.output_dir, selected)
                support = session.env.scene.object_registry("name", pickup.name)
                contacts = sampler.target.contact_list()
                if not contacts or not all(any(same_body(support.prim_path, getattr(c, side))
                    for side in ("body0", "body1")) for c in contacts):
                    raise ValueError("Added-support target contacts unexpected geometry or lacks support contact.")
                route, stance, connected = sampler.route_to_target()
                extent = _array(sampler.target.aabb[1]) - _array(sampler.target.aabb[0])
                pickup_checks = {"support_contact_passed": True, "approximate_grasp_stance_xy_m": stance,
                    "route_xy_m": route, "route_connectivity_screen_passed": connected,
                    "aperture_warning": bool(np.min(extent) > .10), "minimum_target_extent_m": float(np.min(extent))}
            (args.output_dir / "validation_complete.json").write_text(json.dumps({
                "scene_file": str(selected.path), "source_json": selected.definition.source.json_path,
                "variant": selected.variant, "source_question_unchanged": True,
                "physical_grasp_verified": False, "model_calls": 0,
                "added_pickup_support_checks": pickup_checks}, indent=2) + "\n")
            request = args.output_dir.parent / "capture_review_after_validation.json"
            enabled = getattr(args, "capture_review", False)
            if request.is_file():
                capture_config = json.loads(request.read_text())
                enabled = enabled or (capture_config.get("source_json") == selected.definition.source.json_path
                    and capture_config.get("job_id") == os.environ.get("PBS_JOBID"))
            if enabled:
                try:
                    from wia.runtime.commands.capture_esi_scene_reviews import capture_native_review_views
                    from wia.runtime.scene_review import write_json

                    receipt = capture_native_review_views(session, selected, args.output_dir / "review_views")
                    write_json(args.output_dir / "native_review_complete.json", receipt)
                except Exception as error:
                    # A photo failure leaves the native validation intact; the
                    # review coordinator will perform its normal fallback capture.
                    (args.output_dir / "review_capture_failed.json").write_text(json.dumps({
                        "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
                    print(f"[support-sampler] Review capture failed: {error}", flush=True)
        else:
            def capture_placement(scene_file):
                from dataclasses import replace
                from wia.runtime.scene_review import sha256, write_json
                from wia.tasks.schema import SceneDefinition
                from wia.runtime.commands.capture_esi_scene_reviews import capture_native_review_views

                definition = SceneDefinition.model_validate(yaml.safe_load(scene_file.read_text()))
                placement = replace(selected, definition=definition, path=scene_file, file_sha256=sha256(scene_file))
                directory = args.output_dir / "review_views" / scene_file.stem
                receipt = capture_native_review_views(session, placement, directory)
                write_json(directory / "native_review_complete.json", receipt)

            SupportSampler(session, controller, executor, config, args.output_dir, selected,
                on_accepted_placement=capture_placement if getattr(args, "capture_review", False) else None).run()

    web.run_esi_control = screen
    return probe_main(["--viewer-only", "--scene-file", str(args.base_scene), "--data-root", str(args.data_root),
        "--variant", args.variant, "--scene-seed", str(args.seed), "--gripper-control", "--trunk-control",
        "--parameterized-actions", "--primitive-profile", "fine", "--performance-profile", "benchmark",
        "--macro-execution", "metric_target", "--web", "--no-human-preview", "--no-debug-cameras",
        "--preview-fps", "0", "--output-dir", str(args.output_dir)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source", type=Path)
    source.add_argument("--source-dir", type=Path, default=SCENE_ROOT / "Physical Structure/Rigid Containment")
    parser.add_argument("--data-root", type=Path, default=SCENE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-scene", type=Path)
    parser.add_argument("--variant", default="tabletop")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--review-file", type=Path,
        help="Preserve fingerprint-matched Approved layouts and skip manually excluded scenes.")
    parser.add_argument("--grasp-width-m", type=float, default=.08,
        help="Shrink oversized targets uniformly toward this approximate width (maximum 0.10 m).")
    parser.add_argument("--keep-target-scale", action="store_true", help="Disable automatic target shrinking.")
    parser.add_argument("--attempts", type=positive_int, default=12)
    parser.add_argument("--placements", type=positive_int, default=3)
    parser.add_argument("--wall-budget-s", type=float, default=600.)
    parser.add_argument("--minimum-height-m", type=float, default=.35)
    parser.add_argument("--maximum-height-m", type=float, default=1.10)
    parser.add_argument("--require-connected-route", action="store_true",
        help="Reject placements when the conservative grid cannot verify route connectivity.")
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--capture-review", action="store_true",
        help="Capture Global and Object views during accepted support screens or restart validation.")
    parser.add_argument("--preferred-support", action="append", default=[],
        help="Try only these named same-room supports in order; repeat for an explicit fallback.")
    parser.add_argument("--resample-current-placement", action="store_true",
        help="Sample a new pose instead of accepting the current off-table placement unchanged.")
    parser.add_argument("--targeted-support-sampling", action="store_true",
        help="Rank existing furniture and propose footprint-safe edges from nearby free stance cells.")
    parser.add_argument("--no-added-support", action="store_true",
        help="Require an existing support instead of drafting separate pickup furniture.")
    parser.add_argument("--require-planar-footprint", action="store_true",
        help="Retain the legacy nine-point planar footprint screen for controlled comparisons.")
    args = parser.parse_args(argv)
    if (args.seed < 0 or args.wall_budget_s <= 0 or not 0 < args.minimum_height_m < args.maximum_height_m
            or not 0 < args.grasp_width_m <= .10):
        parser.error("Seed must be nonnegative, wall budget and support height window positive, and grasp width within (0, 0.10] m.")
    if args.base_scene is not None and args.source is None and not args.worker:
        parser.error("--base-scene requires an explicit --source.")
    if args.worker:
        if not os.environ.get("PBS_JOBID"):
            parser.error("Run native screening inside the retained interactive GPU allocation.")
        return worker(args)
    candidates, excluded = selections(args)
    if args.base_scene is not None:
        selected = load_scene(args.base_scene, data_root=args.data_root, variant=args.variant)
        if selected.source_path != args.source.resolve():
            parser.error("--base-scene must refer to the exact --source question.")
    preview = {"candidate_count": len(candidates), "excluded_counts": dict(Counter(row["reason"] for row in excluded)),
        "verification_level": "approximate_reachability", "physical_grasp_verified": False, "seed": args.seed,
        "sources": [str(path.resolve()) for path, _ in candidates], "excluded": excluded}
    preview["target_sizing"] = {str(path.resolve()): args.selection_details[str(path.resolve())]
        for path, _ in candidates}
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return 0 if candidates else 1
    if args.output_dir is None:
        parser.error("--output-dir is required for native screening.")
    if not os.environ.get("PBS_JOBID"):
        parser.error("Run native screening inside the retained interactive GPU allocation.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {**preview, "pbs_job_id": os.environ["PBS_JOBID"], "runs": [], "status": "running"}
    summary_path = args.output_dir / "batch_summary.json"
    for path, definition in candidates:
        relative = path.resolve().relative_to(args.data_root.resolve())
        folder = args.output_dir / relative.with_suffix("") / f"seed_{args.seed}"
        result_path = folder / "support_bank.json"
        row = {"source": str(relative), "output": str(folder.resolve())}
        if result_path.exists():
            row.update(status="existing_output_preserved", accepted=len(json.loads(result_path.read_text()).get("accepted", [])))
        elif folder.exists() and any(folder.iterdir()):
            row.update(status="existing_incomplete_output_preserved", reason="Use a fresh output directory to retry.")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            base = args.base_scene
            if base is None:
                base = folder / "initialization.yaml"
                base.write_text(yaml.safe_dump(definition.model_dump(mode="json", by_alias=True), sort_keys=False))
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--base-scene", str(base.resolve()),
                "--data-root", str(args.data_root.resolve()), "--variant", args.variant, "--output-dir", str(folder.resolve()),
                "--seed", str(args.seed), "--attempts", str(args.attempts), "--placements", str(args.placements),
                "--wall-budget-s", str(args.wall_budget_s), "--minimum-height-m", str(args.minimum_height_m),
                "--maximum-height-m", str(args.maximum_height_m)]
            if args.require_connected_route:
                command.append("--require-connected-route")
            if args.capture_review:
                command.append("--capture-review")
            if args.resample_current_placement:
                command.append("--resample-current-placement")
            if args.targeted_support_sampling:
                command.append("--targeted-support-sampling")
            if args.no_added_support:
                command.append("--no-added-support")
            for name in args.preferred_support:
                command.extend(("--preferred-support", name))
            try:
                with (folder / "native.log").open("w") as log:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                        timeout=args.wall_budget_s + 300, check=False)
                row["exit_code"] = process.returncode
                if result_path.is_file():
                    result = json.loads(result_path.read_text())
                    row.update(status=result["status"], accepted=len(result["accepted"]), wall_elapsed_s=result["wall_elapsed_s"])
                else:
                    status_path = folder / "scene_status.json"
                    row.update(status="initialization_failed", reason=json.loads(status_path.read_text()).get("reason")
                        if status_path.is_file() else "Inspect native.log.")
            except subprocess.TimeoutExpired:
                row.update(status="worker_timeout", reason="Native initialization/sampling exceeded its wall budget.")
        summary["runs"].append(row)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print("[support-batch]", relative, row["status"], row.get("accepted", 0), flush=True)
    summary.update(status="complete", status_counts=dict(Counter(row["status"] for row in summary["runs"])))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": str(summary_path.resolve()), "status_counts": summary["status_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
