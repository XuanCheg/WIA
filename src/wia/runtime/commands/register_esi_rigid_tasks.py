#!/usr/bin/env python3
"""Register approved single- and multi-object containment tasks without simulation.

Run: .venv/bin/python src/wia/runtime/commands/register_esi_rigid_tasks.py
--conversion-root RUN --review-dir REVIEW. Task directories mirror the scene
hierarchy. Existing valid tasks and every scene/source byte are preserved.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

import yaml

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
from wia.runtime.scene_review import SceneReview, read_json, write_json
from wia.tasks.loader import SCENE_ROOT, TASK_ROOT, load_scene, load_task, sha256


def register_approved(conversion_root, review_dir, *, scene_root=SCENE_ROOT, task_root=TASK_ROOT):
    scene_root, task_root = Path(scene_root).resolve(), Path(task_root).resolve()
    store = SceneReview(conversion_root, review_dir, data_root=scene_root)
    rows = [row for row in store.catalogue()["scenes"] if Path(row["source"]).parts[-4] == "Rigid Containment"]
    if not rows or any(row["decision"] != "approved" or row["review_stale"] for row in rows):
        raise ValueError("Register only layouts with current human approval.")
    frozen = {Path(row["scene_file"]): sha256(row["scene_file"]) for row in rows}
    tasks = []
    scene_catalog_path = scene_root / "catalog.json"
    scene_catalog = read_json(scene_catalog_path)
    scene_entries = {entry["scene_file"]: entry for entry in scene_catalog["scenes"]}
    for row in rows:
        scene = load_scene(row["scene_file"], data_root=scene_root)
        relative_scene = scene.path.relative_to(scene_root)
        task_path = (Path(row["task_file"]) if row.get("task_file") else task_root / relative_scene).resolve()
        if not task_path.is_relative_to(task_root) or task_path.parent.relative_to(task_root) != relative_scene.parent:
            raise ValueError("Task directory must mirror its approved scene directory.")
        targets = scene.definition.target_bindings or (scene.definition.target,)
        if not task_path.is_file():
            value = {"kind": "rigid_containment_task", "id": scene.definition.id + "_answer_and_place",
                "version": 1, "status": "ready", "source": {"provider": "esi", "json": row["source"]},
                "scene_file": str(relative_scene), "targets": list(targets), "goal": "answer_and_place",
                "allow_remove_after_verification": True,
                "public": {"objective": row["question"],
                    "success_criteria": "Answer the original containment question and complete a real pickup and stable partial insertion for every named target.",
                    "environment": "Retrieve the target objects using the real R1 robot.",
                    "constraints": ["Complete every target subtask before declaring DONE. An object resting entirely on top does not count as inside."]},
                "defaults": {"seed": scene.seed, "max_steps": 60}}
            task_path.parent.mkdir(parents=True, exist_ok=True)
            task_path.write_text(yaml.safe_dump(value, sort_keys=False))
        task = load_task(task_path, data_root=scene_root)
        if task.scene.path != scene.path or task.question_path != scene_root / row["source"] or task.definition.targets != targets:
            raise ValueError("Existing task does not match this approved scene and question.")
        approval = {"decision": "approved", "scene_id": row["id"], "fingerprint": row["fingerprint"],
            "scene_sha256": sha256(scene.path), "question_sha256": sha256(task.question_path),
            "revision": row["review"]["revision"], "approved_at_utc": row["review"]["updated_at"]}
        entry = {"id": task.definition.id, "task_file": str(task_path.relative_to(task_root)),
            "task_sha256": sha256(task_path), "scene_file": str(relative_scene),
            "question_source_json": row["source"], "status": "ready", "targets": list(targets),
            "target_count": len(targets), "goal": task.definition.goal,
            "max_steps": task.definition.defaults.max_steps, "seed": task.seed, "manual_review": approval}
        tasks.append(entry)
        scene_entries[str(relative_scene)].update(task_file=entry["task_file"], manual_review=approval,
            manual_review_required=False, manual_review_stale=False)
    assert all(sha256(path) == digest for path, digest in frozen.items())
    assert len({task["id"] for task in tasks}) == len(tasks)
    catalogue = {"schema": "esi_task_catalog_v1", "status": "ready", "scene_root": "../scenes",
        "task_count": len(tasks), "target_count_counts": dict(Counter(str(task["target_count"]) for task in tasks)),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": sorted(tasks, key=lambda task: task["task_file"])}
    write_json(task_root / "catalog.json", catalogue)
    scene_catalog["task_catalog"] = "../tasks/catalog.json"
    scene_catalog["latest_rigid_task_release"] = {"status": "ready", "task_count": len(tasks),
        "human_approved_layouts": len(tasks), "target_count_counts": catalogue["target_count_counts"]}
    write_json(scene_catalog_path, scene_catalog)
    summary_path = Path(conversion_root) / "run/conversion_summary.json"
    summary = read_json(summary_path)
    paths = {task["scene_file"]: str(task_root / task["task_file"]) for task in tasks}
    for result in summary["results"] + summary.get("preserved_scenes", []):
        scene_file = result.get("scene_file", str(Path(result["source"]).with_suffix(".yaml")))
        if scene_file in paths:
            result.update(scene_file=scene_file, task_file=paths[scene_file])
    write_json(summary_path, summary)
    return catalogue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args()
    catalogue = register_approved(args.conversion_root, args.review_dir)
    print(f"Registered {catalogue['task_count']} approved containment tasks.")


if __name__ == "__main__":
    main()
