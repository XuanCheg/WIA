"""Regression comparisons detect changed actions, paths and final placement."""
import json
from copy import deepcopy

import pytest

from wia.runtime.commands.compare_esi_manipulation import compare_runs


def _sample(phase, step, distance):
    return {"phase": phase, "control_step": step,
            "base": {"position": [distance, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]},
            "eef": {"world_position": [distance, 0, 1], "base_position": [0, 0, 1],
                    "base_quaternion_xyzw": [0, 0, 0, 1]},
            "gripper": {"holding": False, "gripper_open": True, "constraint_active": False},
            "debug_object": {"position": [distance, 0, 0.7], "quaternion_xyzw": [0, 0, 0, 1]}}


def _artifacts(directory, report, samples):
    directory.mkdir()
    (directory / "manipulation_validation.json").write_text(json.dumps(report))
    (directory / "debug_frames.jsonl").write_text("\n".join(json.dumps(s) for s in samples) + "\n")
    (directory / "esi_base.json").write_text(json.dumps({"translation_tolerance_m": 0.02, "rotation_tolerance_deg": 2}))
    (directory / "esi_arm.json").write_text(json.dumps({"translation_tolerance_m": 0.003, "orientation_tolerance_deg": 2}))


@pytest.fixture
def preserved_runs(tmp_path):
    report = {"completed": True, "checks": {"placed_on_support": True},
              "demo": {"stages": {"carry": ["BASE_FWD"], "release": ["RELEASE"]}},
              "results": [{"action": t, "completed": True, "reason": "completed"} for t in ("BASE_FWD", "RELEASE")],
              "initial": _sample("initial", 0, 0),
              "checkpoints": {"carry": _sample("carry", 10, 0.25), "release": _sample("release", 20, 0.25)}}
    old_samples = [_sample("carry", 5, 0.125), _sample("release", 15, 0.25)]
    new_samples = [_sample("carry", 2, 0.05), _sample("carry", 8, 0.2), _sample("release", 12, 0.25)]
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    _artifacts(baseline, report, old_samples)
    _artifacts(candidate, deepcopy(report), new_samples)
    return baseline, candidate


def test_semantically_equal_runs_allow_different_camera_sampling(preserved_runs):
    result = compare_runs(*preserved_runs)
    assert result["passed"] and all(result["checks"].values())
    assert result["metrics"]["max_trajectory_base_position_error_m"] < 1e-10


@pytest.mark.parametrize("mutation", ["actions", "placement", "path", "acceptance"])
def test_changed_behavior_fails_regression(preserved_runs, mutation):
    baseline, candidate = preserved_runs
    path = candidate / "manipulation_validation.json"
    report = json.loads(path.read_text())
    if mutation == "actions":
        report["results"].reverse()
    elif mutation == "placement":
        report["checkpoints"]["release"]["debug_object"]["position"][0] += 0.1
    elif mutation == "path":
        samples = candidate / "debug_frames.jsonl"
        sample = _sample("carry", 5, 0.5)
        samples.write_text(samples.read_text() + json.dumps(sample) + "\n")
    else:
        report["checks"]["placed_on_support"] = False
    path.write_text(json.dumps(report))
    assert not compare_runs(baseline, candidate)["passed"]
