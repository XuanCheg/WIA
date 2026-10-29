#!/usr/bin/env python3
"""Run representative ESI probes sequentially in one interactive allocation.

Run after activation: python src/wia/runtime/commands/explore_esi_categories.py --run-root RUN
Read samples.json produced by audit_esi_categories.py. This never publishes tasks
or releases the allocation; qdel follows human inspection of saved outcomes.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = WIA_RUNTIME_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--extras", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("Use the retained interactive PBS shell.")
    samples = json.loads((args.run_root / "samples.json").read_text())
    if args.extras:
        by_task = {s["module"].split(".")[-1]: s for s in samples}
        samples = [dict(by_task[s["task"]], **{k: v for k, v in s.items() if k != "task"})
                   for s in json.loads((args.run_root / "extra_samples.json").read_text())]
    results = []
    for sample in samples:
        task = sample["module"].split(".")[-1]
        if args.only and task not in args.only:
            continue
        modes = [sample.get("mode", "original")]
        if not args.extras and task in ("liquid_volume", "deformable"):
            modes.append("add_robot")
        for mode in modes:
            output = args.run_root / "native" / task / mode
            output.mkdir(parents=True, exist_ok=True)
            if (output / "finished.json").exists():
                results.append(json.loads((output / "finished.json").read_text()))
                continue
            command = [sys.executable, "-u", str(ROOT / "src/wia/runtime/commands/probe_esi_category.py"),
                       "--sample", sample["source"], "--output-dir", str(output)]
            if mode == "add_robot":
                command.append("--add-robot")
            command += sample.get("flags", [])
            if task in ("inclined_plane", "liquid_volume"):
                command.append("--upstream-effect")
            started = time.monotonic()
            progress = {"status": "running", "task": task, "mode": mode, "command": command,
                        "started_at_utc": datetime.now(timezone.utc).isoformat(), "completed": len(results)}
            (args.run_root / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(f"[exploration] start {task}/{mode}", flush=True)
            with (output / "simulator.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=900)
                    timeout = False
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        code = process.wait(timeout=25)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        code = process.wait()
                    timeout = True
            probe = json.loads((output / "probe.json").read_text()) if (output / "probe.json").exists() else {}
            result = {"task": task, "big_task": sample["big_task"], "small_task": sample["small_task"],
                      "mode": mode, "exit_code": code, "timeout": timeout,
                      "status": probe.get("status", "failed"), "wall_s": time.monotonic()-started,
                      "output_dir": str(output), "source": sample["source"]}
            (output / "finished.json").write_text(json.dumps(result, indent=2)+"\n")
            results.append(result)
            (args.run_root / ("extra_results.json" if args.extras else "batch_results.json")).write_text(json.dumps(results, indent=2)+"\n")
            print(f"[exploration] finish {task}/{mode}: {result['status']}, exit={code}, {result['wall_s']:.1f}s", flush=True)
    (args.run_root / "progress.json").write_text(json.dumps({"status": "complete", "completed": len(results)}, indent=2)+"\n")


if __name__ == "__main__":
    main()
