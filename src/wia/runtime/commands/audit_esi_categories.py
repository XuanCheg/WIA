#!/usr/bin/env python3
"""Inventory ESI task contracts without simulation.

Run: .venv/bin/python src/wia/runtime/commands/audit_esi_categories.py --output-dir RUN
The selected sample list is private research input, never Agent context.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
import ast
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = WIA_RUNTIME_ROOT
ESI = WIA_ESI_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = ESI / "src/active_explore/tasks/_registry.py"
    spec = importlib.util.spec_from_file_location("esi_registry_audit", registry_path)
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)
    rows = defaultdict(list)
    for path in sorted((ESI / "dataset/json_clean").rglob("*.json")):
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or "metadata_json" not in raw:
            continue
        key = (registry._key(raw["big_task"]), registry._key(raw["small_task"]))
        rows[key].append((path, raw))
    tasks = []
    preferred = ["Beechwood_0_garden", "Pomaria_1_int", "Ihlen_0_int", "Benevolence_1_int", "Rs_int"]
    for key, module_name in registry.SMALL_TASK_MODULES.items():
        candidates = rows[key]
        def rank(item):
            path, row = item
            meta = json.loads(row["metadata_json"])
            return (bool(meta.get("skip_reason")), "__no" in path.stem,
                    preferred.index(row["scene"]) if row["scene"] in preferred else len(preferred),
                    len(str(path)), str(path))
        path, row = min(candidates, key=rank)
        if module_name.endswith("action_order_inference"):
            path, row = next((p, r) for p, r in candidates if r["scene"] == "Beechwood_0_garden" and r["room"] == "dining_room_0")
        source = ESI / "src/active_explore" / (module_name.replace(".", "/") + ".py")
        files = [source]
        tree = ast.parse(source.read_text())
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("common"):
                files.append(source.with_name("common.py") if node.level else ESI / "src/active_explore" / (node.module.replace(".", "/") + ".py"))
        contracts = []
        selected_functions = {"build_env_objects", "build_env_config", "preprocess", "postprocess_env", "initial_camera", "score", "score_for_task", "execute_action", "execute_physical_action", "execute_task_action", "reference_image_paths"}
        for file in files:
            text = file.read_text()
            nodes = ast.parse(text).body
            contracts.append({"path": str(file), "functions": {n.name: {"line": n.lineno, "source": ast.get_source_segment(text, n)} for n in nodes if isinstance(n, ast.FunctionDef) and n.name in selected_functions}, "constants": {n.targets[0].id: ast.get_source_segment(text, n) for n in nodes if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {"FULL_SCENE", "VALID_ACTIONS", "SKIP_INITIAL_SETTLE", "DISABLE_VIEWER_CAMERA_MODALITIES"}}})
        tasks.append({"big_task": row["big_task"], "small_task": row["small_task"], "module": module_name,
                      "sample_count": len(candidates), "scene_counts": dict(Counter(r["scene"] for _, r in candidates)),
                      "source": str(path), "scene": row["scene"], "room": row["room"],
                      "question": row["question"], "answer": row["answer"], "answer_type": row["answer_type"],
                      "metadata_keys": sorted(json.loads(row["metadata_json"])), "contracts": contracts})
    result = {"sample_count": sum(len(v) for v in rows.values()), "big_task_count": len({t["big_task"] for t in tasks}), "small_task_count": len(tasks), "tasks": tasks}
    (args.output_dir / "contracts.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = [{k: v for k, v in t.items() if k not in {"contracts", "scene_counts", "metadata_keys"}} for t in tasks]
    (args.output_dir / "samples.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("sample_count", "big_task_count", "small_task_count")}))
    for t in tasks:
        print(f"{t['big_task']} / {t['small_task']}: {t['sample_count']}, {t['scene']}/{t['room']} | {t['question'][:155]}")


if __name__ == "__main__":
    main()
