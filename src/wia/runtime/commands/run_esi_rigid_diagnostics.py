#!/usr/bin/env python3
"""Run an explicit native diagnostic plan sequentially in the retained GPU shell.

Run: python src/wia/runtime/commands/run_esi_rigid_diagnostics.py --plan /absolute/plan.json
    --summary /absolute/summary.json --job-id PBS_JOB_ID
The plan is a JSON list of named base_scene/output_dir/wall_budget_s records.
Completed receipts are retained. This runner never publishes or releases jobs.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = WIA_RUNTIME_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args(argv)
    if os.environ.get("PBS_JOBID") != args.job_id:
        parser.error("Run diagnostics in the specified retained interactive allocation.")
    results = []
    started = time.monotonic()
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    def save(status, current=None):
        args.summary.write_text(json.dumps({"status": status, "job_id": args.job_id,
            "results": results, "current": current, "wall_elapsed_s": time.monotonic()-started}, indent=2)+"\n")

    for case in json.loads(args.plan.read_text()):
        out = Path(case["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        receipt = out / "worker_exit.json"
        if receipt.exists():
            result = json.loads(receipt.read_text())
            results.append({**case, **result, "reused_completed_receipt": True})
            save("running")
            continue
        if (out / "native.log").exists():
            raise RuntimeError(f"Incomplete output must be archived before retrying: {out}")
        save("running", case)
        command = [sys.executable, "-u", str(ROOT / "src/wia/runtime/commands/diagnose_esi_rigid_scene.py"),
            "--base-scene", case["base_scene"], "--output-dir", str(out)]
        if case.get("cache_assets"):
            command.append("--cache-assets")
        wall_start = time.monotonic()
        with (out / "native.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            try:
                code = process.wait(timeout=case["wall_budget_s"])
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = -999
        result = {"exit_code": code, "wall_elapsed_s": time.monotonic()-wall_start,
            "wall_budget_s": case["wall_budget_s"],
            "validation_complete": (out / "validation_complete.json").exists()}
        receipt.write_text(json.dumps(result, indent=2)+"\n")
        results.append({**case, **result})
        save("running")
        print(json.dumps({"case": case["name"], **result}), flush=True)
    save("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
