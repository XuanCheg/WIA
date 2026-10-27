"""Validate this diagnostic candidate on CPU without importing a simulator.

From the repository root, run:
/scratch/xinyangzhou/models/Show-Harness/.venv/bin/python explore_to_achieve/candidates/pilot_action_order/validate.py
Use --scene-only before the first commit, then run without it after handoff creation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml

CANDIDATE_DIR = Path(__file__).resolve().parent
ROOT = CANDIDATE_DIR.parents[2]
sys.path.insert(0, str(ROOT))

from core.tasks.loader import UniqueLoader
from core.tasks.schema import SceneRuntime

CANDIDATE_ID = "pilot_action_order_original_v1"
SCENE_ID = "pilot_action_order_beechwood_0_garden_dining_room_0_original_v1"
SOURCE = (
    "Action Sequencing/Action Order Inference/Beechwood_0_garden/dining_room_0/"
    "Beechwood_0_garden__dining_room_0.json"
)
SCENE_FILE = "explore_to_achieve/candidates/pilot_action_order/scene.yaml"
VIEWS = [
    "global_diagonal", "global_overhead", "global_front", "global_back",
    "global_left", "global_right", "head", "wrist", "scene",
]
MODIFIED_FILES = [
    "explore_to_achieve/candidates/pilot_action_order/plan.md",
    SCENE_FILE,
    "explore_to_achieve/candidates/pilot_action_order/validate.py",
    "explore_to_achieve/candidates/pilot_action_order/handoff.json",
    "research_log.md",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(scene_only=False):
    source_path = ROOT / "ESI-Bench/dataset/json_clean" / SOURCE
    source = json.loads(source_path.read_text())
    require(
        tuple(source[key] for key in ("big_task", "small_task", "scene", "room"))
        == ("Action Sequencing", "Action Order Inference", "Beechwood_0_garden", "dining_room_0"),
        "Original source identity does not match the assigned instance.",
    )
    require(isinstance(source.get("question"), str) and source["question"].strip(),
            "The original source question is unavailable.")
    scene_path = ROOT / SCENE_FILE
    scene = yaml.load(scene_path.read_text(), Loader=UniqueLoader)
    runtime = SceneRuntime.model_validate(scene["runtime"])
    require(runtime == SceneRuntime(), "Rendering must use the existing runtime defaults.")
    expected_scene = {
        "kind": "scene_patch", "id": SCENE_ID, "version": 1, "status": "pending",
        "source": {"provider": "esi", "json": SOURCE},
        "target": None, "bindings": {}, "qualification": None,
        "scene": {"policy": "native_then_minimal_patch_else_skip",
                  "variants": {"original": {"patches": []}}},
        "runtime": runtime.model_dump(mode="json"), "seed": 0,
    }
    require(scene == expected_scene,
            "The scene must remain an unpatched original diagnostic overlay with no physical task.")
    for config in (runtime.trunk_config, runtime.robot_camera_config, runtime.viewer_config):
        json.loads((ROOT / config).read_text())

    public_files = [scene_path, CANDIDATE_DIR / "plan.md"]
    if not scene_only:
        manifest_path = CANDIDATE_DIR / "handoff.json"
        manifest = json.loads(manifest_path.read_text())
        expected_manifest = {
            "candidate_id": CANDIDATE_ID,
            "source_json": "ESI-Bench/dataset/json_clean/" + SOURCE,
            "scene_file": SCENE_FILE, "task_file": None,
            "category": "Action Sequencing", "subcategory": "Action Order Inference",
            "scene_model": "Beechwood_0_garden", "room": "dining_room_0",
            "variant": "original", "seed": 0, "reuse_level": "original",
            "modified_files": MODIFIED_FILES, "expected_views": VIEWS,
            "known_limitations": manifest.get("known_limitations"),
            "code_commit": manifest.get("code_commit"), "owner": "scene_builder_a",
            "parent_candidate_id": None, "physical_task_id": None, "render_config": {},
        }
        require(manifest == expected_manifest, "Handoff identity or configuration is inconsistent.")
        require(manifest["candidate_id"] != scene["id"],
                "Candidate and scene identities must remain distinct.")
        require(isinstance(manifest["known_limitations"], list) and manifest["known_limitations"],
                "The diagnostic handoff must declare unverified conditions.")
        require(isinstance(manifest["code_commit"], str)
                and re.fullmatch(r"[0-9a-f]{40}", manifest["code_commit"]),
                "The handoff must pin an actual committed revision.")
        committed_scene = subprocess.check_output(
            ["git", "show", manifest["code_commit"] + ":" + SCENE_FILE], cwd=ROOT,
        )
        require(committed_scene == scene_path.read_bytes(),
                "The pinned scene revision differs from the current overlay.")
        for path in manifest["modified_files"]:
            require((ROOT / path).is_file(), "An owned handoff file is missing.")
        public_files.append(manifest_path)

    metadata = json.loads(source["metadata_json"])
    private_markers = {"answer_A", "answer_C", "ground_truth", "_gt_auxiliary", "gt_outcome"}
    private_markers.update(
        item["_key"] for item in metadata.get("objects", [])
        if isinstance(item, dict) and ("_correct" in item.get("_key", "")
                                     or "_wrong" in item.get("_key", ""))
    )
    for path in public_files:
        content = path.read_text()
        require(all(marker not in content for marker in private_markers),
                "Private source labels appear in a public candidate artifact.")
        require(source["answer"] not in content and source["metadata_json"] not in content,
                "Private source data appear in a public candidate artifact.")
    require(not any(name == "omnigibson" or name.startswith("omnigibson.") for name in sys.modules),
            "CPU validation must not import the simulator.")
    mode = "scene only" if scene_only else "scene and committed handoff"
    print(f"PASS: {mode}; source identity, unpatched original, runtime defaults, private exclusion.")
    print("No simulator was imported or run; native initialization and physical execution remain unverified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-only", action="store_true")
    validate(parser.parse_args().scene_only)
