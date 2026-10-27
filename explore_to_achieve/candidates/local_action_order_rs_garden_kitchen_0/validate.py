"""CPU validation for one original-layout diagnostic candidate.

Run this file from any directory with the project's Python 3.10 environment.
Use --scene-only before the implementation commit; run without it after handoff creation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import yaml

CANDIDATE_DIR = Path(__file__).resolve().parent
ROOT = CANDIDATE_DIR.parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from core.scene_capture_candidates import validate_handoff
from core.tasks.loader import UniqueLoader
from core.tasks.schema import SceneDefinition, SceneRuntime

CANDIDATE_ID = "local_action_order_rs_garden_kitchen_0_original_v1"
SCENE_ID = "local_action_order_rs_garden_kitchen_0_original_scene_v1"
SOURCE = "Action Sequencing/Action Order Inference/Rs_garden/kitchen_0/Rs_garden__kitchen_0.json"
SCENE_MODEL = "Rs_garden"
ROOM = "kitchen_0"
DIRECTORY = "explore_to_achieve/candidates/local_action_order_rs_garden_kitchen_0"
SCENE_FILE = DIRECTORY + "/scene.yaml"
CONSTRUCTOR = "src/active_explore/tasks/action_sequencing/action_order_inference.py"
VIEWS = [
    "global_diagonal", "global_overhead", "global_front", "global_back",
    "global_left", "global_right", "head", "wrist", "scene",
]
MODIFIED_FILES = [
    DIRECTORY + "/plan.md", SCENE_FILE, DIRECTORY + "/validate.py",
    DIRECTORY + "/handoff.json", "research_log.md",
]
LIMITATIONS = [
    "Native initialization, licensed asset loading and camera framing remain unverified.",
    "Physical reach, grasping and stacking remain unverified.",
    "Pending diagnostic capture only; no physical task, evaluator, robot episode or success claim is supplied.",
    "Human camera views are review evidence and do not become Agent observations.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(*arguments, repo=ROOT):
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], stderr=subprocess.PIPE,
    )


def original_constructor(esi_commit):
    path = ROOT / "ESI-Bench" / CONSTRUCTOR
    require(
        path.read_bytes() == git("show", esi_commit + ":" + CONSTRUCTOR, repo=ROOT / "ESI-Bench"),
        "The native ESI constructor differs from the committed original.",
    )
    spec = importlib.util.spec_from_file_location("original_action_order_constructor", path)
    require(spec is not None and spec.loader is not None, "The native constructor cannot be read.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.TASK_NAME == "action_order_inference", "The native constructor has the wrong category.")
    return module


def validate(scene_only=False):
    revision = git("rev-parse", "HEAD").decode().strip()
    esi_commit = git("ls-tree", revision, "ESI-Bench").decode().split()[2]
    source_path = ROOT / "ESI-Bench/dataset/json_clean" / SOURCE
    source_bytes = source_path.read_bytes()
    require(
        source_bytes == git(
            "show", esi_commit + ":dataset/json_clean/" + SOURCE, repo=ROOT / "ESI-Bench",
        ),
        "The original source JSON has changed.",
    )
    source = json.loads(source_bytes)
    require(
        tuple(source[key] for key in ("big_task", "small_task", "scene", "room"))
        == ("Action Sequencing", "Action Order Inference", SCENE_MODEL, ROOM),
        "Original source identity does not match the candidate.",
    )
    require(
        isinstance(source.get("question"), str) and source["question"].strip(),
        "The original source question is unavailable.",
    )

    scene_path = ROOT / SCENE_FILE
    scene = yaml.load(scene_path.read_text(), Loader=UniqueLoader)
    definition = SceneDefinition.model_validate(scene)
    runtime = SceneRuntime()
    require(definition.runtime == runtime, "Rendering must use the existing runtime defaults.")
    require(scene == {
        "kind": "scene_patch", "id": SCENE_ID, "version": 1, "status": "pending",
        "source": {"provider": "esi", "json": SOURCE},
        "target": None, "bindings": {}, "qualification": None,
        "scene": {"policy": "native_then_minimal_patch_else_skip",
                  "variants": {"original": {"patches": []}}},
        "runtime": runtime.model_dump(mode="json"), "seed": 0,
    }, "The diagnostic overlay must preserve the empty original variant and have no physical task.")
    require(
        definition.target is None and not definition.target_bindings
        and definition.qualification is None,
        "The diagnostic must not acquire a physical target or qualification.",
    )
    for config in (runtime.trunk_config, runtime.robot_camera_config, runtime.viewer_config):
        json.loads((ROOT / config).read_text())

    payload = json.loads(source["metadata_json"])
    # The native HF-row loader adds scene/room before calling the task constructor.
    payload.update(scene=source["scene"], room=source["room"])
    native = original_constructor(esi_commit)
    require(native.scene_room(payload) == (SCENE_MODEL, ROOM), "Native scene/room identity differs.")
    objects = native.build_env_objects(payload)
    infos = {item["_key"]: item for item in payload["objects"]}
    poses = {item["_key"]: item for item in payload["initial_poses"]}
    require(payload["n_objects"] == 3 and len(objects) == len(infos) == 5,
            "The selected source must contain its original five task instances.")
    require(len({obj["name"] for obj in objects}) == 5, "Native object names must remain distinct.")
    for obj in objects:
        info, pose = infos[obj["name"]], poses[obj["name"]]
        require(obj == {
            "type": "DatasetObject", "name": obj["name"],
            "category": info.get("cat") or info.get("category"), "model": info["model"],
            "scale": info.get("scale"), "position": pose["position"],
            "orientation": pose["quaternion_xyzw"],
        }, "Native asset, scale or pose differs from the original metadata.")
        require(obj["category"] and obj["model"], "A native asset reference is missing.")
    eye, quaternion, camera_info = native.initial_camera(payload)
    require(
        camera_info["view_idx"] == 0 and np.shape(eye) == (3,)
        and np.shape(quaternion) == (4,) and np.isfinite(eye).all()
        and np.isfinite(quaternion).all() and np.isclose(np.linalg.norm(quaternion), 1.0),
        "The native camera must retain the selected source orbital view 0.",
    )
    require(native.postprocess_env(None, payload, camera_info) == {},
            "This native diagnostic must not introduce postprocessing changes.")

    public_files = [scene_path, CANDIDATE_DIR / "plan.md"]
    if not scene_only:
        manifest_path = CANDIDATE_DIR / "handoff.json"
        manifest = json.loads(manifest_path.read_text())
        expected = {
            "candidate_id": CANDIDATE_ID,
            "source_json": "ESI-Bench/dataset/json_clean/" + SOURCE,
            "source_identity": SOURCE, "scene_identity": SCENE_ID,
            "scene_file": SCENE_FILE, "task_file": None,
            "category": "Action Sequencing", "subcategory": "Action Order Inference",
            "scene_model": SCENE_MODEL, "room": ROOM,
            "variant": "original", "seed": 0, "reuse_level": "original",
            "modified_files": MODIFIED_FILES, "expected_views": VIEWS,
            "known_limitations": LIMITATIONS, "code_commit": manifest.get("code_commit"),
            "owner": "scene_builder_a", "parent_candidate_id": None,
            "physical_task_id": None, "render_config": {},
        }
        require(manifest == expected, "Handoff identity or configuration is inconsistent.")
        require(len({CANDIDATE_ID, SCENE_ID, SOURCE}) == 3,
                "Source, candidate and scene identities must remain distinct.")
        require(
            isinstance(manifest["code_commit"], str)
            and re.fullmatch(r"[0-9a-f]{40}", manifest["code_commit"]),
            "The handoff must pin an actual committed implementation revision.",
        )
        for relative_path in (SCENE_FILE, DIRECTORY + "/plan.md", DIRECTORY + "/validate.py"):
            require(
                git("show", manifest["code_commit"] + ":" + relative_path)
                == (ROOT / relative_path).read_bytes(),
                "The pinned implementation differs from the current candidate.",
            )
        for relative_path in MODIFIED_FILES:
            require((ROOT / relative_path).is_file(), "An owned handoff file is missing.")
        accepted = validate_handoff(manifest, repo=ROOT)
        require(accepted["definition"] == definition and accepted["esi_commit"] == esi_commit,
                "Shared capture acceptance differs from the candidate contract.")
        public_files.append(manifest_path)

    private_markers = {
        "answer_A", "answer_C", "ground_truth", "_ground_truth", "_gt_auxiliary",
        "gt_outcome", "metadata_json", "qa_per_view",
    }
    private_markers.update(
        name for name in infos if "_correct" in name or "_wrong" in name
    )
    for path in public_files:
        content = path.read_text()
        require(all(marker not in content for marker in private_markers),
                "Private source labels appear in a public candidate artifact.")
        require(source["answer"] not in content and source["metadata_json"] not in content,
                "Private source data appear in a public candidate artifact.")
    require(
        not any(name == "omnigibson" or name.startswith("omnigibson.")
                or name == "isaacsim" or name.startswith("isaacsim.") for name in sys.modules),
        "CPU validation must not import the simulator.",
    )
    mode = "scene only" if scene_only else "scene and committed handoff"
    print("PASS: " + CANDIDATE_ID + "; " + mode)
    print("Full scene schema, unchanged source/constructor, five native objects, source camera view 0,")
    print("default runtime and private exclusion pass; source_sha256=" + hashlib.sha256(source_bytes).hexdigest())
    print("No simulator was imported or run; native initialization and physical execution remain unverified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-only", action="store_true")
    validate(parser.parse_args().scene_only)
