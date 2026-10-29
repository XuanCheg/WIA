#!/usr/bin/env python3
"""Compare preserved M4 debug artifacts; never read or control a live robot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _read(directory, name):
    return json.loads((Path(directory) / name).read_text())


def _position_error(left, right):
    return float(np.linalg.norm(np.asarray(left) - right))


def _orientation_error(left, right):
    return float(np.rad2deg((Rotation.from_quat(left).inv() * Rotation.from_quat(right)).magnitude()))


def _stage_tracks(directory, report):
    """Compare each stage at equal control progress despite different sampling."""
    frames = [json.loads(line) for line in (Path(directory) / "debug_frames.jsonl").read_text().splitlines()]
    previous, tracks = report["initial"], {}
    for phase, final in report["checkpoints"].items():
        samples = {sample["control_step"]: sample for sample in
                   [previous] + [f for f in frames if f["phase"] == phase] + [final]}
        samples = [samples[step] for step in sorted(samples)]
        steps = np.asarray([sample["control_step"] for sample in samples], dtype=float)
        progress = ((steps - steps[0]) / (steps[-1] - steps[0])
                    if steps[-1] > steps[0] else np.zeros(len(steps)))
        values = {
            "base_position": [s["base"]["position"] for s in samples],
            "ee_world_position": [s["eef"]["world_position"] for s in samples],
            "ee_base_position": [s["eef"]["base_position"] for s in samples],
        }
        tracks[phase] = {key: np.stack([np.interp(np.linspace(0, 1, 101), progress, axis)
                                       for axis in np.asarray(value).T], axis=1)
                         for key, value in values.items()}
        previous = final
    return tracks


def compare_runs(baseline, candidate):
    old, new = (_read(directory, "manipulation_validation.json") for directory in (baseline, candidate))
    base_config, arm_config = _read(baseline, "esi_base.json"), _read(baseline, "esi_arm.json")
    checks = {
        "baseline_passed": bool(old["completed"] and old["checks"] and all(old["checks"].values())),
        "candidate_passed": bool(new["completed"] and new["checks"] and all(new["checks"].values())),
        "same_demo": old["demo"] == new["demo"],
        "same_action_sequence": [r["action"] for r in old["results"]] == [r["action"] for r in new["results"]],
        "same_action_outcomes": [(r["completed"], r["reason"]) for r in old["results"]]
                                == [(r["completed"], r["reason"]) for r in new["results"]],
        "same_acceptance_checks": old["checks"] == new["checks"],
        "same_stages": tuple(old["checkpoints"]) == tuple(new["checkpoints"]),
    }
    metrics = {}
    if checks["same_stages"]:
        pairs = [(old["initial"], new["initial"])] + list(zip(old["checkpoints"].values(), new["checkpoints"].values()))
        metrics["max_checkpoint_base_position_error_m"] = max(_position_error(a["base"]["position"], b["base"]["position"]) for a, b in pairs)
        metrics["max_checkpoint_base_orientation_error_deg"] = max(_orientation_error(a["base"]["quaternion_xyzw"], b["base"]["quaternion_xyzw"]) for a, b in pairs)
        metrics["max_checkpoint_ee_orientation_error_deg"] = max(_orientation_error(a["eef"]["base_quaternion_xyzw"], b["eef"]["base_quaternion_xyzw"]) for a, b in pairs)
        metrics["max_checkpoint_object_position_error_m"] = max(_position_error(a["debug_object"]["position"], b["debug_object"]["position"]) for a, b in pairs)
        old_tracks, new_tracks = _stage_tracks(baseline, old), _stage_tracks(candidate, new)
        for key in ("base_position", "ee_world_position", "ee_base_position"):
            metrics[f"max_trajectory_{key}_error_m"] = max(float(np.max(np.linalg.norm(
                old_tracks[phase][key] - new_tracks[phase][key], axis=1))) for phase in old_tracks)
        limits = {
            "max_checkpoint_base_position_error_m": base_config["translation_tolerance_m"],
            "max_checkpoint_base_orientation_error_deg": base_config["rotation_tolerance_deg"],
            "max_checkpoint_ee_orientation_error_deg": arm_config["orientation_tolerance_deg"],
            "max_checkpoint_object_position_error_m": 0.02,  # Existing M4 support/retreat acceptance tolerance.
            "max_trajectory_base_position_error_m": base_config["translation_tolerance_m"],
            "max_trajectory_ee_world_position_error_m": base_config["translation_tolerance_m"] + arm_config["translation_tolerance_m"],
            "max_trajectory_ee_base_position_error_m": arm_config["translation_tolerance_m"],
        }
        checks.update({key: metrics[key] <= limit for key, limit in limits.items()})
        final_old, final_new = pairs[-1]
        checks["same_final_gripper_state"] = all(final_old["gripper"][key] == final_new["gripper"][key]
                                                 for key in ("holding", "gripper_open", "constraint_active"))
        metrics["final_object_position_error_m"] = _position_error(final_old["debug_object"]["position"], final_new["debug_object"]["position"])
        metrics["final_object_orientation_error_deg"] = _orientation_error(final_old["debug_object"]["quaternion_xyzw"], final_new["debug_object"]["quaternion_xyzw"])
    return {"passed": all(checks.values()), "baseline": str(baseline), "candidate": str(candidate),
            "action_count": len(new["results"]), "checks": checks, "metrics": metrics}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_runs(args.baseline, args.candidate)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
