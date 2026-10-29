#!/usr/bin/env python3
"""Publish completed multi-object captures while preserving approved base scenes.

Run on the CPU host: .venv/bin/python src/wia/runtime/commands/publish_esi_rigid_multi.py
--generation-root RUN --review-dir REVIEW. This does not launch a simulator
or change human review decisions. Serve RUN/merged_review with review_esi_scenes.py.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import yaml

SCENE_ROOT = WIA_RUNTIME_ROOT / "explore_to_achieve/scenes"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def review_is_current(approval, scene_path, question_path):
    return (approval.get("decision") == "approved"
        and approval.get("scene_sha256") == hashlib.sha256(scene_path.read_bytes()).hexdigest()
        and approval.get("question_sha256") == hashlib.sha256(question_path.read_bytes()).hexdigest())


def register_tasks(root, summary, scene_root):
    path = scene_root / "catalog.json"
    value = json.loads(path.read_text())
    by_scene = {row["scene_file"]: row for row in value["scenes"]}
    for result in summary["results"]:
        scene_path = scene_root / result["scene_file"]
        definition = yaml.safe_load(scene_path.read_text())
        existing = by_scene.get(result["scene_file"])
        approval = (existing or {}).get("manual_review", {})
        approved = review_is_current(approval, scene_path, scene_root / result["source"])
        task_path = Path(result["task_file"]).resolve()
        task_root = (scene_root.parent / "tasks").resolve()
        task_file = str(task_path.relative_to(task_root)) if task_path.is_relative_to(task_root) else str(task_path)
        row = {"scene_file": result["scene_file"], "source_json": definition["source"]["json"],
            "question_source_json": result["source"], "task_file": task_file,
            "status": result["status"], "conversion": "multi_object_manual_review",
            "verification_level": result["verification_level"], "manual_review_required": not approved,
            "manual_review_stale": bool(approval) and not approved,
            "physical_grasp_verified": False, "qualification_performed": False,
            "source_answer_revalidated": False, "conversion_result": result}
        if existing is None:
            value["scenes"].append(row)
        else:
            existing.update(row)
    value["scene_description_count"] = len(value["scenes"])
    value["status_counts"] = dict(Counter(row["status"] for row in value["scenes"]))
    value["task_generation"] = "rigid_containment_merged_derivatives"
    value["latest_rigid_task_expansion"] = {"root": str(root), "status": summary["status"],
        "new_tasks": len(summary["results"]),
        "ready_tasks": sum(row["status"] == "ready" for row in summary["results"]),
        "qualification_performed": False, "physical_grasp_verified": False}
    atomic_json(path, value)


def publish(root, review, scene_root=SCENE_ROOT):
    summary = json.loads((root / "run/conversion_summary.json").read_text())
    ready = [row for row in summary["results"] if row["status"] == "ready"]
    destination = root / "merged_review/run/conversion_summary.json"
    value = {**summary, "results": ready}
    atomic_json(destination, value)
    register_tasks(root, summary, scene_root)
    progress = review / "preparation_status.json"
    atomic_json(progress, {"status": summary["status"], "stage": "multi_object_generation",
        "tasks_completed": len(ready), "tasks_total": len(summary["results"]),
        "qualification_performed": False})
    return summary["status"] == "complete" and not summary.get("simulator_retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    previous = None
    while True:
        stamp = (args.generation_root / "run/conversion_summary.json").stat().st_mtime_ns
        if stamp != previous:
            complete = publish(args.generation_root, args.review_dir)
            previous = stamp
            print("Published completed task captures.", flush=True)
            if complete or args.once:
                return
        time.sleep(3)


if __name__ == "__main__":
    main()
