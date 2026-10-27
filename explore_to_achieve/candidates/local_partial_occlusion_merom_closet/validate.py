"""Validate this original-layout candidate using CPU source contracts only.

Run with a Python environment containing NumPy, PyYAML, and Pydantic 2:
  python explore_to_achieve/candidates/local_partial_occlusion_merom_closet/validate.py
Use --scene-only before writing the separately committed handoff.
No simulator, GPU initialization, model inference, or robot action is run.
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

sys.dont_write_bytecode = True

import numpy as np
import yaml

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]
sys.path.insert(0, str(ROOT))

from core.tasks.loader import UniqueLoader
from core.tasks.schema import SceneDefinition, SceneRuntime

NAME = "local_partial_occlusion_merom_closet"
CANDIDATE_ID = NAME + "_original_v1"
SCENE_ID = NAME + "_scene_v1"
SOURCE_JSON = (
    "Perceptual Grounding/Partial Occlusion/Merom_0_int/closet_0/"
    "Merom_0_int__closet_0_4.json"
)
SOURCE_RECORD_ID = "1167"
SOURCE_QUESTION_ID = "partial_occlusion/Merom_0_int/closet_0/4"
SOURCE_SEED = 1452418930
SOURCE_SHA256 = "4eddbae5b8a15a2b49e3dc52f685e6979780864e9d2a866a256711fdd088a428"
SOURCE_FIRST_VISIBILITY = {"target_obj": True, "occluder_obj": True}
FLOOR_XY = [1.765, 0.980]
ESI_COMMIT = "9f6f15ed43e81654a97de9e4db2a1fedadca2bc1"
SOURCE_FILE = "ESI-Bench/dataset/json_clean/" + SOURCE_JSON
PREFIX = "explore_to_achieve/candidates/" + NAME
SCENE_FILE = PREFIX + "/scene.yaml"
IMPLEMENTATION_FILES = [PREFIX + "/plan.md", SCENE_FILE, PREFIX + "/validate.py"]
OWNED_FILES = [*IMPLEMENTATION_FILES, PREFIX + "/handoff.json", "research_log.md"]
EXPECTED_VIEWS = [
    "global_diagonal", "global_overhead", "global_front", "global_back",
    "global_left", "global_right", "head", "wrist", "scene",
]
LIMITATIONS = [
    "Diagnostic only; candidate remains pending with no physical task or evaluator.",
    "Native initialization, robot clearance, framing, and mounted-camera visibility are unverified.",
    "The original low floor layout and compact room may challenge existing R1 initialization.",
    "Review camera views never become Agent observations.",
    "QA and physical success are unverified; no simulation or Agent episode was run.",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git_bytes(repository: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), *args], capture_output=True, check=False,
    )
    require(result.returncode == 0, "Required Git provenance is unavailable.")
    return result.stdout


def committed_bytes(repository: Path, revision: str, path: str) -> bytes:
    return git_bytes(repository, "show", f"{revision}:{path}")


def check_private_exclusion(raw: dict, metadata: dict, files: list[Path]) -> None:
    native_objects = metadata["objects"]
    keyed = native_objects if isinstance(native_objects, dict) else {
        item["_key"]: item for item in native_objects
    }
    private_values = [raw.get("answer"), metadata.get("answer")]
    private_values.extend(keyed["target"].get(key) for key in (
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
                    "Authored candidate content contains a private answer or object identity.")


def validate(scene_only: bool = False) -> dict:
    scene_path = DIRECTORY / "scene.yaml"
    scene = yaml.load(scene_path.read_text(), Loader=UniqueLoader)
    definition = SceneDefinition.model_validate(scene)
    require(set(scene) == {
        "kind", "id", "version", "status", "source", "target", "bindings",
        "qualification", "scene", "runtime", "seed",
    }, "Scene fields differ from the bounded diagnostic contract.")
    require((definition.kind, definition.id, definition.version, definition.status, definition.seed)
            == ("scene_patch", SCENE_ID, 1, "pending", 0),
            "Scene identity, version, status, or capture seed changed.")
    require(definition.target is None and not definition.target_bindings
            and definition.qualification is None and definition.bindings == {},
            "A diagnostic candidate cannot invent a physical target or task.")
    require(definition.source.json_path == SOURCE_JSON, "Scene source identity changed.")
    require(scene["scene"] == {
        "policy": "native_then_minimal_patch_else_skip",
        "variants": {"original": {"patches": []}},
    }, "Original reuse must preserve all assets, layouts, scales, and camera poses.")
    require(definition.runtime.model_dump() == SceneRuntime().model_dump(),
            "Use only the existing runtime defaults.")
    for config in (definition.runtime.trunk_config, definition.runtime.robot_camera_config,
                   definition.runtime.viewer_config):
        json.loads((ROOT / config).read_text())

    source_path = ROOT / SOURCE_FILE
    source_bytes = source_path.read_bytes()
    require(hashlib.sha256(source_bytes).hexdigest() == SOURCE_SHA256,
            "Original source bytes differ from the inspected native instance.")
    require(git_bytes(ROOT / "ESI-Bench", "rev-parse", "HEAD").decode().strip() == ESI_COMMIT,
            "Original ESI implementation revision changed.")
    require(source_bytes == committed_bytes(ROOT / "ESI-Bench", ESI_COMMIT,
                                            "dataset/json_clean/" + SOURCE_JSON),
            "Original source differs from its committed official version.")
    raw = json.loads(source_bytes)
    require((raw["big_task"], raw["small_task"], raw["scene"], raw["room"])
            == tuple(Path(SOURCE_JSON).parts[:4]), "Source hierarchy is inconsistent.")
    require(raw["id"] == SOURCE_RECORD_ID and raw["runner_task"] == "occlusion"
            and isinstance(raw["question"], str) and bool(raw["question"].strip()),
            "Original source record or question contract changed.")
    metadata = raw["metadata_json"]
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    require(metadata["question_id"] == SOURCE_QUESTION_ID and metadata["seed"] == SOURCE_SEED,
            "Native question identity or source seed changed.")
    require(metadata["placement_mode"] == "floor_center" and metadata["support_type"] == "floor",
            "Original placement mechanism changed.")

    native_path = ROOT / "ESI-Bench/src/active_explore/tasks/perceptual_grounding/partial_occlusion.py"
    spec = importlib.util.spec_from_file_location(NAME + "_native_cpu", native_path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    keyed = native.objects_map(metadata)
    expected_objects = []
    for key, fallback in (("target", "target_obj"), ("occluder", "occluder_obj")):
        obj = keyed[key]
        expected = {
            "type": "DatasetObject", "name": str(obj.get("name") or fallback).strip(),
            "category": obj.get("requested_category") or obj["category"],
            "model": obj["model"], "position": obj["position"],
            "orientation": obj["quaternion_xyzw"],
        }
        scale = obj.get("applied_scale") or obj.get("scale")
        if scale is not None:
            expected["scale"] = scale
        expected_objects.append(expected)
    require(native.build_env_objects(metadata) == expected_objects,
            "Native constructor does not preserve the original two object specifications.")
    views = native.view_records(metadata)
    require(len(views) == 7 and views[0].get("visibility") == SOURCE_FIRST_VISIBILITY,
            "Original reference-view metadata changed.")
    position, orientation, info = native.initial_camera(metadata)
    require(np.array_equal(position, np.asarray(views[0]["position"], dtype=float))
            and np.array_equal(orientation, np.asarray(views[0]["quaternion_xyzw"], dtype=float))
            and info["selection"] == "first_view" and info["view_index"] == 0,
            "Native initial reference camera changed.")
    require(position.shape == (3,) and orientation.shape == (4,)
            and np.isfinite(position).all() and np.isfinite(orientation).all(),
            "Original reference camera is not finite and well shaped.")
    support = keyed["support"]
    floor_xy = np.asarray(support["aabb_max"][:2]) - np.asarray(support["aabb_min"][:2])
    require(np.allclose(floor_xy, FLOOR_XY, atol=0.001) and np.prod(floor_xy) < 3.5,
            "Selected original floor-support geometry changed.")

    authored_files = [DIRECTORY / "plan.md", scene_path]
    if not scene_only:
        handoff_path = DIRECTORY / "handoff.json"
        handoff = json.loads(handoff_path.read_text())
        expected = {
            "candidate_id": CANDIDATE_ID, "scene_identity": SCENE_ID,
            "source_identity": SOURCE_JSON, "source_record_id": SOURCE_RECORD_ID,
            "source_question_id": SOURCE_QUESTION_ID, "source_sha256": SOURCE_SHA256,
            "esi_commit": ESI_COMMIT, "source_json": SOURCE_FILE,
            "scene_file": SCENE_FILE, "task_file": None,
            "category": "Perceptual Grounding", "subcategory": "Partial Occlusion",
            "scene_model": raw["scene"], "room": raw["room"],
            "variant": "original", "seed": 0, "reuse_level": "original",
            "modified_files": OWNED_FILES, "expected_views": EXPECTED_VIEWS,
            "known_limitations": LIMITATIONS, "owner": "scene_builder_b",
            "parent_candidate_id": None, "physical_task_id": None, "render_config": {},
        }
        require(set(handoff) == set(expected) | {"code_commit"}
                and all(handoff[key] == value for key, value in expected.items()),
                "Handoff differs from the source, scene, or capture contract.")
        revision = handoff["code_commit"]
        require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
                "Handoff must pin an actual full implementation commit.")
        for path in IMPLEMENTATION_FILES:
            require(committed_bytes(ROOT, revision, path) == (ROOT / path).read_bytes(),
                    "Handoff revision does not contain the current candidate implementation.")
        require(all((ROOT / path).is_file() for path in OWNED_FILES),
                "A declared candidate file is missing.")
        from core.scene_capture_candidates import validate_handoff
        validate_handoff(handoff, repo=ROOT, code_commit=revision)
        authored_files.append(handoff_path)
    check_private_exclusion(raw, metadata, authored_files)
    require(not any(name.split(".")[0] in {"omnigibson", "isaacsim", "torch"}
                    for name in sys.modules), "CPU validation imported a native GPU runtime.")
    return {
        "validation": "passed", "mode": "scene_only" if scene_only else "full_handoff",
        "candidate_id": CANDIDATE_ID, "scene_identity": SCENE_ID,
        "source_identity": SOURCE_JSON, "source_record_id": SOURCE_RECORD_ID,
        "source_question_id": SOURCE_QUESTION_ID, "source_sha256": SOURCE_SHA256,
        "source_unchanged": True, "scene_schema": "passed",
        "native_constructor_cpu": "passed", "native_initial_camera_cpu": "passed",
        "source_floor_support_xy_m": np.round(floor_xy, 3).tolist(),
        "source_first_view_visibility": SOURCE_FIRST_VISIBILITY,
        "runtime_defaults": "passed", "private_answer_exclusion": "passed",
        "simulation_run": False, "model_calls": 0, "robot_actions": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-only", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(validate(arguments.scene_only), indent=2))
