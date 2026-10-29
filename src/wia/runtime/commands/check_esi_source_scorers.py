#!/usr/bin/env python3
"""Check selected original ESI QA packets with pure source scoring functions.

Run: .venv/bin/python src/wia/runtime/commands/check_esi_source_scorers.py --run-root RUN
AST loading omits simulator imports and physics classes; scoring/helper function
bodies remain unchanged. This is a protocol check, not a model accuracy metric.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
import ast
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = WIA_RUNTIME_ROOT
ACTIVE = WIA_ESI_ROOT / 'src/active_explore'
sys.path.insert(0, str(ACTIVE))


def namespace(path):
    tree = ast.parse(path.read_text())
    common = path.with_name("common.py")
    shared = namespace(common) if any(isinstance(n, ast.ImportFrom) and n.module == "common" for n in tree.body) else {}
    ns = dict(shared)
    if shared:
        ns["common"] = SimpleNamespace(**shared)
    protocol = path.with_name("rigid_containment_protocol.py")
    if path.stem == "rigid_containment" and protocol.is_file():
        ns.update(namespace(protocol))
    ns["__file__"] = str(path)
    safe = {"__future__", "typing", "pathlib", "collections", "dataclasses", "json", "re", "math", "os", "random", "time", "traceback", "shutil", "numpy", "scipy", "utils"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            if not all(alias.name.split(".")[0] in safe for alias in node.names):
                continue
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module or node.module.split(".")[0] not in safe:
                continue
        elif isinstance(node, ast.Assign):
            if not all(isinstance(t, ast.Name) for t in node.targets):
                continue
        elif not isinstance(node, ast.FunctionDef):
            continue
        piece = ast.Module(body=[node], type_ignores=[])
        try:
            exec(compile(piece, str(path), "exec", flags=__import__("__future__").annotations.compiler_flag), ns)
        except NameError:
            # Simulator-valued module constants are irrelevant to pure scoring.
            if not isinstance(node, ast.Assign):
                raise
    return ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for sample in json.loads((args.run_root / "samples.json").read_text()):
        raw = json.loads(Path(sample["source"]).read_text())
        payload = json.loads(raw["metadata_json"])
        payload.update(scene=raw["scene"], room=raw["room"], _hf_id=raw["id"],
                       _hf_big_task=raw["big_task"], _hf_small_task=raw["small_task"],
                       _hf_options=json.loads(raw["options_json"]), _hf_image_paths=json.loads(raw["image_paths_json"]))
        payload.setdefault("_question", raw["question"])
        try:
            answer = json.loads(raw["answer"])
        except (TypeError, json.JSONDecodeError):
            answer = raw["answer"]
        payload.setdefault("_ground_truth", answer)
        payload.setdefault("answer", answer)
        if isinstance(answer, (list, dict)):
            answer = json.dumps(answer)
        result = {"task": sample["module"].split(".")[-1], "source": sample["source"]}
        try:
            ns = namespace(ACTIVE / (sample["module"].replace(".", "/") + ".py"))
            camera = ns["initial_camera"](payload)[2]
            # Source directional answer is tied to view 0, whereas the active
            # runner can select another best camera. Preserve both results.
            def score(info):
                options = {"camera_info": info, "task_state": {}}
                return ns["score"](payload, {"answer": answer, "confidence": 1.}, **{k: v for k, v in options.items() if k in inspect.signature(ns["score"]).parameters})
            result["score"] = score(camera)
            if result["task"] == "rigid_containment":
                result["raw_packet_note"] = "HF keyed-list assignments are not accepted as a single container answer by the original scorer."
                size = next((label for label in ("medium", "small", "large") if label in answer.lower()), None)
                if size:
                    normalized = {"answer": size, "confidence": 1.}
                    result["normalized_source_packet_score"] = ns["score"](payload, normalized, camera_info=camera)
            if result["task"] == "action_order_inference":
                qa = ns["keyed_list_to_map"](payload["qa_per_view"])
                result["source_reference_score"] = score({"view_idx": 0, "qa": qa.get("0") or qa.get("view_0") or camera["qa"]})
            result["status"] = "checked"
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        results.append(result)
    (args.run_root / "source_scorer_checks.json").write_text(json.dumps(results, indent=2, default=lambda v: v.tolist()) + "\n")
    for result in results:
        print(result["task"], result["status"], result.get("score", {}).get("correct"), result.get("error", ""))


if __name__ == "__main__":
    main()
