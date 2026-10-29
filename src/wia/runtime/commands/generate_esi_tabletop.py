#!/usr/bin/env python3
"""Generate scene-only tabletop YAMLs without running models or simulation.

Run .venv/bin/python src/wia/runtime/commands/generate_esi_tabletop.py --source SAMPLE.json
    --output-dir OUTPUT [--target-scale 0.35 --robot-position X Y Z --robot-yaw-deg 135].
Use --source-dir DIRECTORY to select canonical questions only, or add --dry-run
to inspect conversion without writing files. Discarded catalogue records are
excluded. Explicit single-source variants remain pending separate task support.
Native qualification is still required.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

import yaml
from wia.tasks.loader import DATA_ROOT, UniqueLoader
from wia.tasks.tabletop import generate_tabletop


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path)
    source.add_argument("--source-dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Inspect conversion and skip reasons without writing files.")
    parser.add_argument("--table-height", type=float, default=.45)
    parser.add_argument("--floor-z", type=float, default=0.)
    parser.add_argument("--target-scale", type=float, default=1.,
                        help="Explicit target-only multiplier; never infer graspability from AABB alone.")
    parser.add_argument("--robot-position", nargs=3, type=float)
    parser.add_argument("--robot-yaw-deg", type=float, default=0.)
    parser.add_argument("--head-pitch-deg", type=float, default=0.,
                        help="Additional local downward head pitch; zero retains the native exploration view.")
    parser.add_argument("--front-clearance", type=float, default=.45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trunk-config", default="configs/esi_trunk_low_grasp.json")
    args = parser.parse_args(argv)
    if not args.dry_run and args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is selected.")
    if args.source_dir is not None and (args.target_scale != 1. or args.robot_position is not None):
        parser.error("Batch generation retains source scales and derives per-source spawns; use explicit single-source overrides.")
    sources = [args.source] if args.source is not None else sorted(args.source_dir.rglob("*.json"))
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"generated": [], "skipped": [], "discarded": [], "ignored_task_variants": [],
               "input_samples": len(sources), "dry_run": args.dry_run,
               "native_qualification": "not_run", "model_calls": 0}
    for path in sources:
        descriptor = path.with_suffix(".yaml")
        if descriptor.is_file():
            value = yaml.load(descriptor.read_text(), Loader=UniqueLoader)
            if value.get("status") == "discarded":
                summary["discarded"].append({"source": str(path.resolve()),
                    "reason": value.get("discard_reason", "Discarded by catalogue selection.")})
                continue
        if args.source_dir is not None and "__" in path.stem:
            summary["ignored_task_variants"].append({"source": str(path.resolve()),
                "reason": "Canonical-question-only batch selection."})
            continue
        try:
            definition, details = generate_tabletop(path, args.data_root,
                table_height=args.table_height, floor_z=args.floor_z, target_scale=args.target_scale,
                robot_position=args.robot_position, robot_yaw_deg=args.robot_yaw_deg,
                head_pitch_deg=args.head_pitch_deg, front_clearance=args.front_clearance,
                seed=args.seed, trunk_config=args.trunk_config)
            relative = path.resolve().relative_to(args.data_root.resolve())
            destination = args.output_dir / relative.with_suffix(".yaml") if args.output_dir is not None else None
            if not args.dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise ValueError("Generated YAML already exists; preserve it or use a fresh output directory.")
                destination.write_text(yaml.safe_dump(definition.model_dump(mode="json", by_alias=True), sort_keys=False))
                destination.with_suffix(".generation.json").write_text(json.dumps(details, indent=2) + "\n")
            summary["generated"].append({"source": str(path.resolve()),
                "scene_file": str(destination.resolve()) if destination is not None else None,
                "status": definition.status, "initialization_source_json": details["initialization_source_json"],
                "source_question_json": details["source_question_json"], "task_variant": details["task_variant"],
                "task_execution_status": details["task_execution_status"],
                "table_dimensions_m": details["table_dimensions_m"]})
        except (ValueError, KeyError) as error:
            summary["skipped"].append({"source": str(path.resolve()), "reason": str(error)})
    if not args.dry_run:
        (args.output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"generated": 0 if args.dry_run else len(summary["generated"]),
        "convertible": len(summary["generated"]), "skipped": len(summary["skipped"]),
        "input_samples": summary["input_samples"], "discarded": len(summary["discarded"]),
        "ignored_task_variants": len(summary["ignored_task_variants"]),
        "dry_run": args.dry_run,
        "candidate_status_counts": dict(Counter(row["status"] for row in summary["generated"])),
        "skip_reasons": dict(Counter(row["reason"] for row in summary["skipped"])),
        "summary": str((args.output_dir / "generation_summary.json").resolve()) if not args.dry_run else None}))
    return 0 if summary["generated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
