"""Refine older inventory base geometry from an explicit native vertex reference.

Run CPU-only, choosing a new immutable output directory:
  python scripts/refine_wia_geometry_inputs.py --reference REFERENCE/inventory.json \
    --inputs INPUT_A/inventory.json INPUT_B/inventory.json --output NEW_DIRECTORY

Source files are never changed. Raw input/reference bytes and independent output
inventories are retained with SHA256 manifests. No simulator measurement of a
target, policy/robot call, worker modification, or GPU launch is performed.
This transfers one unscaled native base-link shape through measured root poses;
it never fits, scales, resizes, or recenters geometry to make an AABB match.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA = "wia_base_geometry_refinement_v1"
BASE_SCHEMA = "wia_robot_base_geometry_v1"
RECORD_KEY = "base_geometry_refinement"
AABB_TOLERANCE_M = .002
CALIBRATION_ATOL = 1e-8
ASSUMPTION = "same unscaled native base-link geometry corroborated by measured target AABB"


class RefinementRejected(ValueError):
    """The declared transfer cannot be corroborated from these measured inputs."""


def _require(condition, reason):
    if not condition:
        raise RefinementRejected(reason)


def _finite_array(value, shape, label):
    array = np.asarray(value, dtype=float)
    _require(array.shape == shape and np.isfinite(array).all(), f"{label}: finite array of shape {shape} required")
    return array


def _box(value, label):
    box = _finite_array(value, (2, 3), label)
    _require(np.all(box[0] <= box[1]), label + ": ordered bounds required")
    return box


def _calibration_equal(left, right, pointer="/robot/joint_calibration"):
    """Compare the entire same-schema calibration, with no relative tolerance."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        _require(set(left) == set(right), pointer + ": calibration key schema differs")
        for key in left:
            _calibration_equal(left[key], right[key], pointer + "/" + str(key))
    elif isinstance(left, list) and isinstance(right, list):
        _require(len(left) == len(right), pointer + ": calibration list schema differs")
        for index, (a, b) in enumerate(zip(left, right)):
            _calibration_equal(a, b, pointer + "/" + str(index))
    elif isinstance(left, Real) and not isinstance(left, bool) and isinstance(right, Real) and not isinstance(right, bool):
        values = np.asarray([left, right], dtype=float)
        _require(np.isfinite(values).all(), pointer + ": nonfinite calibration value")
        _require(abs(float(left) - float(right)) <= CALIBRATION_ATOL, pointer + ": calibration numeric value differs")
    else:
        _require(type(left) is type(right) and left == right, pointer + ": calibration value or schema differs")


def _pose(robot, label):
    position = _finite_array(robot["position"], (3,), label + " position")
    quaternion = _finite_array(robot["orientation"], (4,), label + " XYZW quaternion")
    rotation = Rotation.from_quat(quaternion)
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation.as_matrix(), position
    return position, quaternion, rotation, matrix


def refine_inventory(target, reference, *, input_sha256, reference_sha256, input_file=None, reference_file=None):
    """Return an independent inventory and an accepted/rejected/already-present record.

    Existing available base geometry is returned exactly, without adding a new
    top-level record. Rejected transfers keep all old geometry; when the record
    key is free they add only a rejection record. CLI hashes describe the exact
    read-once source bytes, not a normalized reserialization.
    """
    record = {"schema": SCHEMA, "status": "rejected", "input_file": input_file,
        "input_sha256": input_sha256, "reference_file": reference_file, "reference_sha256": reference_sha256,
        "aabb_tolerance_m": AABB_TOLERANCE_M, "calibration_atol": CALIBRATION_ATOL,
        "assumption": ASSUMPTION, "new_target_simulator_measurement": False,
        "policy_calls": 0, "robot_calls": 0, "checks": {}}
    original = deepcopy(target)
    try:
        _require(isinstance(target, Mapping), "target inventory mapping required")
        robot = target["robot"]
        _require(isinstance(robot, Mapping), "target robot metadata unavailable")
        if isinstance(robot.get("base_geometry"), Mapping) and robot["base_geometry"].get("available") is True:
            record.update(status="already_present", reason="available target base_geometry preserved exactly", refinement_applied=False)
            return original, record
        _require(RECORD_KEY not in target, "existing refinement record is preserved; select its original raw input for another attempt")
        _require(isinstance(reference, Mapping), "reference inventory mapping required")
        ref_robot = reference["robot"]
        _require(isinstance(ref_robot, Mapping), "reference robot metadata unavailable")
        _require(target.get("schema") == reference.get("schema") == "wia_candidate_native_inventory_v1", "inventory schema differs or is unsupported")
        geometry = ref_robot.get("base_geometry")
        _require(isinstance(geometry, Mapping) and geometry.get("available") is True
                 and geometry.get("status") == "available" and geometry.get("schema") == BASE_SCHEMA,
                 "reference base_geometry is not available in the declared schema")
        _require(geometry.get("provenance") == "enabled_native_base_link_collision_mesh_vertices",
                 "reference must contain fresh native measured base vertices, not a prior transferred proxy")
        _require(isinstance(geometry.get("base_link_name"), str) and bool(geometry["base_link_name"]), "reference base link name unavailable")
        for asset in ("urdf_path", "usd_path"):
            _require(isinstance(robot.get(asset), str) and bool(robot[asset]) and robot[asset] == ref_robot.get(asset), asset + ": exact asset path strings differ")
        record["checks"]["asset_paths_identical"] = True
        old_geometry = robot.get("base_geometry")
        if isinstance(old_geometry, Mapping) and old_geometry.get("base_link_name") is not None:
            _require(old_geometry["base_link_name"] == geometry["base_link_name"], "target/reference base link names differ")
        for owner in (robot, ref_robot):
            _require(isinstance(owner.get("joint_calibration"), Mapping) and bool(owner["joint_calibration"]), "nonempty native joint_calibration required")
            _require(isinstance(owner.get("joints"), Mapping) and bool(owner["joints"]), "native robot joint names/values unavailable")
            values = np.asarray(list(owner["joints"].values()), dtype=float)
            _require(values.ndim == 1 and np.isfinite(values).all(), "finite scalar robot joint values required")
        _require(set(robot["joints"]) == set(ref_robot["joints"]), "robot joint-name sets differ")
        record["checks"]["joint_names_identical"] = True
        _calibration_equal(robot["joint_calibration"], ref_robot["joint_calibration"])
        record["checks"]["entire_joint_calibration_identical_within_atol"] = True
        world = np.asarray(geometry.get("world_vertices"), dtype=float)
        _require(world.ndim == 2 and world.shape[1] == 3 and len(world) > 0 and np.isfinite(world).all(), "reference world vertices must be finite nonempty Nx3")
        _require(geometry.get("vertex_count") == len(world), "reference vertex count disagrees with measured vertices")
        ref_box, target_box = _box(ref_robot["base_aabb"], "reference base_aabb"), _box(robot["base_aabb"], "target base_aabb")
        ref_residual = float(np.max(np.abs(np.stack((world.min(axis=0), world.max(axis=0))) - ref_box)))
        record["reference_world_aabb_residual_m"] = ref_residual
        _require(ref_residual <= AABB_TOLERANCE_M, "reference vertices do not match measured reference base_aabb")
        record["checks"]["reference_vertices_match_reference_aabb"] = True
        ref_position, ref_quaternion, ref_rotation, _ = _pose(ref_robot, "reference root")
        position, quaternion, rotation, matrix = _pose(robot, "target root")
        local = ref_rotation.inv().apply(world - ref_position)
        predicted = rotation.apply(local) + position
        _require(np.isfinite(local).all() and np.isfinite(predicted).all(), "rigid root-pose transfer produced nonfinite vertices")
        predicted_box = np.stack((predicted.min(axis=0), predicted.max(axis=0)))
        residual = float(np.max(np.abs(predicted_box - target_box)))
        inverse_reference = np.eye(4)
        inverse_reference[:3, :3] = ref_rotation.as_matrix().T
        inverse_reference[:3, 3] = -ref_rotation.as_matrix().T @ ref_position
        record.update(reference_root_pose={"position": ref_position.tolist(), "orientation_xyzw": ref_quaternion.tolist()},
            target_root_pose={"position": position.tolist(), "orientation_xyzw": quaternion.tolist()},
            frame_transform={"reference_world_to_reference_root": inverse_reference.tolist(),
                "reference_root_to_target_world": matrix.tolist(),
                "reference_world_to_target_world": (matrix @ inverse_reference).tolist(),
                "scale_applied": False, "fit_applied": False, "recenter_applied": False},
            predicted_world_aabb=predicted_box.tolist(), measured_target_base_aabb=deepcopy(robot["base_aabb"]),
            target_world_aabb_residual_m=residual)
        _require(residual <= AABB_TOLERANCE_M, "predicted world vertices do not match measured target base_aabb")
        record["checks"]["predicted_vertices_match_target_aabb"] = True
        record.update(status="accepted", reason="rigid reference geometry corroborated by measured target AABB", refinement_applied=True)
        original["robot"]["base_geometry"] = {
            "schema": BASE_SCHEMA, "available": True, "status": "available",
            "provenance": "rigid_pose_transfer_of_native_reference_base_vertices",
            "base_link_name": geometry["base_link_name"], "vertex_count": len(predicted),
            "world_vertices": predicted.tolist(),
            "base_local_xy_bounds": [local[:, :2].min(axis=0).tolist(), local[:, :2].max(axis=0).tolist()],
            "predicted_world_aabb": predicted_box.tolist(), "measured_base_aabb": deepcopy(robot["base_aabb"]),
            "max_world_aabb_residual_m": residual, "aabb_tolerance_m": AABB_TOLERANCE_M,
            "reference_inventory_file": reference_file, "reference_inventory_sha256": reference_sha256,
            "reference_mesh_metadata": {key: deepcopy(geometry[key]) for key in
                ("enabled_mesh_names", "enabled_mesh_count", "measured_mesh_count", "meshes") if key in geometry},
            "target_vertices_newly_measured": False, "assumption": ASSUMPTION,
        }
    except (RefinementRejected, KeyError, TypeError, ValueError, OverflowError) as error:
        original = deepcopy(target)
        record.update(status="rejected", reason=str(error), refinement_applied=False)
    if isinstance(original, dict) and RECORD_KEY not in original:
        original[RECORD_KEY] = deepcopy(record)
    return original, record


def _read(path):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    return path, raw, hashlib.sha256(raw).hexdigest(), json.loads(raw)


def _write_new(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)


def _json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("Choose a new immutable output directory: " + str(output))
    reference_path, reference_raw, reference_sha, reference = _read(args.reference)
    inputs = [_read(path) for path in args.inputs]
    if len({str(path) for path, *_ in inputs}) != len(inputs):
        raise ValueError("--inputs must contain distinct source files.")
    if output == reference_path or any(output == path for path, *_ in inputs):
        raise ValueError("Output must not overwrite any source.")
    output.mkdir(parents=True, exist_ok=False)
    _write_new(output / "reference/inventory.json", reference_raw)
    records = []
    for path, raw, input_sha, inventory in inputs:
        refined, record = refine_inventory(inventory, reference, input_sha256=input_sha, reference_sha256=reference_sha,
                                           input_file=str(path), reference_file=str(reference_path))
        key = input_sha[:16] + "_" + hashlib.sha256(str(path).encode()).hexdigest()[:8]
        directory = output / "items" / key
        _write_new(directory / "original_inventory.json", raw)
        result_raw = raw if record["status"] == "already_present" else _json_bytes(refined)
        _write_new(directory / "inventory.json", result_raw)
        record.update(original_copy=str(directory / "original_inventory.json"), output_file=str(directory / "inventory.json"),
                      output_sha256=hashlib.sha256(result_raw).hexdigest())
        records.append(record)
    manifest = {"schema": SCHEMA, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_file": str(reference_path), "reference_sha256": reference_sha,
        "reference_copy": str(output / "reference/inventory.json"),
        "counts": dict(Counter(record["status"] for record in records)), "records": records,
        "new_target_simulator_measurement": False, "policy_calls": 0, "robot_calls": 0}
    _write_new(output / "manifest.json", _json_bytes(manifest))
    for status in ("accepted", "rejected", "already_present"):
        _write_new(output / (status + ".json"), _json_bytes({"schema": SCHEMA, "status": status,
            "reference_sha256": reference_sha, "records": [record for record in records if record["status"] == status]}))
    print(json.dumps({"output": str(output), "counts": manifest["counts"], "new_target_simulator_measurement": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
