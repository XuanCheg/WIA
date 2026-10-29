#!/usr/bin/env python3
"""Run unattended private diagnostics: python src/wia/runtime/commands/preflight_esi_tasks.py --all --mode static."""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.runtime.preflight.models import GROUPS, Options


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Value must be positive and finite.")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Seed must be a nonnegative integer.")
    return number


def directions(value):
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Directions must be comma-separated angles in degrees.") from error
    if not result or not all(math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("Directions must be finite angles in degrees.")
    return result


def check_groups(value):
    result = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not result or set(result) - set(GROUPS):
        raise argparse.ArgumentTypeError("Checks must be comma-separated groups: " + ",".join(GROUPS))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Private deterministic task preflight; no Agent, model, or automatic repairs.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task-file", action="append", default=[], help="Exact task or standalone scene YAML/JSON path; repeat to check several inputs.")
    selection.add_argument("--all", action="store_true", help="Check registered ready tasks and discovered ready task definitions.")
    parser.add_argument("--mode", choices=("static", "full"), default="static")
    parser.add_argument("--seed", type=nonnegative_int)
    parser.add_argument("--variant")
    parser.add_argument("--data-root")
    parser.add_argument("--wall-budget-s", type=positive_float, default=1800.)
    parser.add_argument("--output-dir", default="outputs/preflight")
    parser.add_argument("--candidate-count", type=positive_int, default=6)
    parser.add_argument("--stance-count", type=positive_int, default=12)
    parser.add_argument("--directions-deg", type=directions, default=(0., 90., 180., 270.))
    parser.add_argument("--search-budget-s", type=positive_float, default=240.)
    parser.add_argument("--settle-steps", type=positive_int, default=30)
    parser.add_argument("--hold-steps", type=positive_int, default=30)
    parser.add_argument("--lift-m", type=positive_float, default=.06)
    parser.add_argument("--checks", type=check_groups, default=GROUPS,
                        help="Enabled validator groups; static safety validation always runs.")
    parser.add_argument("--observation-config", help="Use the measured episode's observation contract configuration.")
    parser.add_argument("--worker-request", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from wia.runtime.preflight.runner import run_batch, run_worker
    if args.worker_request:
        return run_worker(args.worker_request)
    if not args.task_file and not args.all:
        parser.error("Provide --task-file or --all.")
    options = Options(mode=args.mode, seed=args.seed, variant=args.variant, data_root=args.data_root,
                      wall_budget_s=args.wall_budget_s, candidate_count=args.candidate_count,
                      stance_count=args.stance_count, directions_deg=args.directions_deg,
                      search_budget_s=args.search_budget_s, settle_steps=args.settle_steps,
                      hold_steps=args.hold_steps, lift_m=args.lift_m,
                      enabled=args.checks, observation_config=args.observation_config)
    summary = run_batch(args.task_file, options, args.output_dir, all_tasks=args.all)
    # Keep the full skipped-entry list in summary.json rather than flooding terminals.
    print(json.dumps({key: summary[key] for key in ("summary_path", "task_count", "completed_task_count", "counts", "skipped_count", "wall_time_s")}, indent=2))
    return 1 if summary["counts"]["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
