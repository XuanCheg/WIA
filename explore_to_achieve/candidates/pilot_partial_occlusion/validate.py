"""Validate the original Partial Occlusion candidate on CPU.

Run from any directory:
  /scratch/xinyangzhou/models/Show-Harness/.venv/bin/python /absolute/path/to/validate.py
Before the handoff is committed, append --scene-only to validate the scene files.
No simulator, robot, model inference, or GPU initialization is imported or run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import yaml


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]
sys.path.insert(0, str(ROOT))

from core.tasks.schema import Scene, SceneRuntime, Source


CANDIDATE_ID = "pilot_partial_occlusion_original_v1"
SOURCE_JSON = (
    "Perceptual Grounding/Partial Occlusion/Beechwood_0_garden/garden_0/"
    "Beechwood_0_garden__garden_0_0.json"
)
SOURCE_FILE = "ESI-Bench/dataset/json_clean/" + SOURCE_JSON
SCENE_FILE = "explore_to_achieve/candidates/pilot_partial_occlusion/scene.yaml"
OWNED_FILES = [
    "explore_to_achieve/candidates/pilot_partial_occlusion/plan.md",
    SCENE_FILE,
    "explore_to_achieve/candidates/pilot_partial_occlusion/validate.py",
    "explore_to_achieve/candidates/pilot_partial_occlusion/handoff.json",
    "research_log.md",
]
EXPECTED_VIEWS = [
    "global_diagonal", "global_overhead", "global_front", "global_back",
    "global_left", "global_right", "head", "wrist", "scene",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def committed_bytes(repository: Path, revision: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), "show", f"{revision}:{path}"],
        capture_output=True, check=False,
    )
    require(result.returncode == 0, "A required committed file is unavailable.")
    return result.stdout


def check_private_exclusion(raw: dict, metadata: dict, files: list[Path]) -> None:
    objects = metadata["objects"]
    keyed = objects if isinstance(objects, dict) else {
        item["_key"]: item for item in objects
    }
    target = keyed["target"]
    private_values = [raw.get("answer"), metadata.get("answer")]
    private_values.extend(target.get(key) for key in (
        "category", "requested_category", "model", "name",
    ))
    for path in files:
        authored = path.read_text().lower().replace("_", " ")
        require("ground truth" not in authored and "metadata json" not in authored,
                "Authored candidate content contains a private source field.")
        for value in private_values:
            if not isinstance(value, str) or len(value.strip()) < 3:
                continue
            normalized = value.strip().lower().replace("_", " ")
            pattern = r"(?<![a-z0-9])" + re.escape(normalized) + r"(?![a-z0-9])"
            require(re.search(pattern, authored) is None,
                    "Authored candidate content contains a private answer or identity.")


def validate(scene_only: bool = False) -> dict:
    scene_path = DIRECTORY / "scene.yaml"
    scene = yaml.safe_load(scene_path.read_text())
    require(set(scene) == {
        "kind", "id", "version", "status", "source", "target", "bindings",
        "qualification", "scene", "runtime", "seed",
    }, "Scene keys must match the assigned diagnostic contract.")
    require((scene["kind"], scene["id"], scene["version"], scene["status"], scene["seed"])
            == ("scene_patch", CANDIDATE_ID, 1, "pending", 0),
            "Scene identity, status, version, or seed differs from the assignment.")
    require(scene["target"] is None and scene["qualification"] is None
            and scene["bindings"] == {}, "A diagnostic scene cannot invent a physical task.")
    source = Source.model_validate(scene["source"])
    require(source.json_path == SOURCE_JSON, "The scene references a different source.")
    native_scene = Scene.model_validate(scene["scene"])
    require(scene["scene"] == {
        "policy": "native_then_minimal_patch_else_skip",
        "variants": {"original": {"patches": []}},
    } and not native_scene.variants["original"].patches,
            "The original scene must have no asset, layout, or camera patches.")
    runtime = SceneRuntime.model_validate(scene["runtime"])
    require(runtime.model_dump() == SceneRuntime().model_dump(),
            "Runtime settings must use the existing defaults.")
    for config in (runtime.trunk_config, runtime.robot_camera_config, runtime.viewer_config):
        require((ROOT / config).is_file(), "A runtime configuration is missing.")

    source_path = ROOT / SOURCE_FILE
    source_bytes = source_path.read_bytes()
    require(source_bytes == committed_bytes(ROOT / "ESI-Bench", "HEAD", "dataset/json_clean/" + SOURCE_JSON),
            "The read-only original source differs from its committed content.")
    raw = json.loads(source_bytes)
    require((raw["big_task"], raw["small_task"], raw["scene"], raw["room"])
            == tuple(Path(SOURCE_JSON).parts[:4]), "Source identity does not match its hierarchy.")
    require(raw["runner_task"] == "occlusion" and isinstance(raw["question"], str)
            and bool(raw["question"].strip()), "The original question contract is missing.")
    metadata = raw["metadata_json"]
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    native_path = ROOT / "ESI-Bench/src/active_explore/tasks/perceptual_grounding/partial_occlusion.py"
    spec = importlib.util.spec_from_file_location("pilot_partial_occlusion_native", native_path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    require(len(native.build_env_objects(metadata)) == 2,
            "The native object constructor cannot read the original source.")
    position, orientation, _ = native.initial_camera(metadata)
    require(position.shape == (3,) and orientation.shape == (4,),
            "The native initial camera cannot read the original source.")

    authored_files = [DIRECTORY / "plan.md", scene_path]
    if not scene_only:
        handoff_path = DIRECTORY / "handoff.json"
        handoff = json.loads(handoff_path.read_text())
        expected = {
            "candidate_id": CANDIDATE_ID, "source_json": SOURCE_FILE,
            "scene_file": SCENE_FILE, "task_file": None,
            "category": "Perceptual Grounding", "subcategory": "Partial Occlusion",
            "scene_model": "Beechwood_0_garden", "room": "garden_0",
            "variant": "original", "seed": 0, "reuse_level": "original",
            "modified_files": OWNED_FILES, "expected_views": EXPECTED_VIEWS,
            "owner": "scene_builder_b", "parent_candidate_id": None,
            "physical_task_id": None, "render_config": {},
        }
        require(set(handoff) == set(expected) | {"known_limitations", "code_commit"},
                "The handoff keys do not match the assigned contract.")
        require(all(handoff[key] == value for key, value in expected.items()),
                "The handoff does not match the original diagnostic scene.")
        require(handoff["known_limitations"] == [
            "Diagnostic only; candidate remains pending.",
            "Native framing and initialization are unverified in this iteration.",
            "Changing viewpoint for review never becomes an Agent observation.",
            "QA and physical success are unverified.",
        ], "The handoff must retain all assigned limitations.")
        revision = handoff["code_commit"]
        require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
                "The handoff must record an actual full commit revision.")
        require(committed_bytes(ROOT, revision, SCENE_FILE) == scene_path.read_bytes(),
                "The handoff revision does not contain this scene.")
        require(all((ROOT / path).is_file() for path in OWNED_FILES),
                "A handoff-owned file is missing.")
        authored_files.append(handoff_path)
    check_private_exclusion(raw, metadata, authored_files)
    return {
        "validation": "passed", "mode": "scene_only" if scene_only else "full_handoff",
        "candidate_id": CANDIDATE_ID, "variant": "original", "seed": 0,
        "reuse_level": "original", "source_unchanged": True,
        "private_answer_exclusion": "passed", "native_constructor_cpu": "passed",
        "runtime_defaults": "passed", "simulation_run": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.scene_only), indent=2))
