"""Validate strict CPU base-geometry transfer. Run: pytest -q tests/test_refine_wia_geometry_inputs.py."""
from copy import deepcopy
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


_SPEC = importlib.util.spec_from_file_location("refine_wia_geometry_inputs", Path(__file__).resolve().parents[1] / "scripts/refine_wia_geometry_inputs.py")
refine = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(refine)


def inventory_pair(*, angles=(0., 0., 0.), position=(0., 0., 0.)):
    local = np.array(list(product((-.31, .29), (-.25, .27), (.03, .42))))
    reference_rotation = Rotation.from_euler("xyz", [.03, -.02, -.4])
    reference_position = np.array([1.4, -.9, .02])
    world = reference_rotation.apply(local) + reference_position
    robot = {"urdf_path": "/installed/r1.urdf", "usd_path": "/installed/r1.usda",
             "position": reference_position.tolist(), "orientation": reference_rotation.as_quat().tolist(),
             "base_aabb": [world.min(axis=0).tolist(), world.max(axis=0).tolist()],
             "joints": {"base_x_joint": .1, "arm_joint": .2},
             "joint_calibration": {"base_x_joint": {"parent": "root", "child": "base_link",
                 "parent_joint": np.eye(4).tolist(), "child_joint": np.eye(4).tolist(), "axis": [1., 0., 0.]},
                 "arm_joint": {"parent": "base_link", "child": "arm", "axis": [0., 0., 1.]}},
             "base_geometry": {"schema": refine.BASE_SCHEMA, "available": True, "status": "available",
                 "provenance": "enabled_native_base_link_collision_mesh_vertices", "base_link_name": "base_link",
                 "world_vertices": world.tolist(), "vertex_count": len(world),
                 "enabled_mesh_names": ["base_link::body"], "enabled_mesh_count": 1}}
    reference = {"schema": "wia_candidate_native_inventory_v1", "id": "reference", "robot": robot}
    target = deepcopy(reference)
    target["id"] = "target"
    rotation = Rotation.from_euler("xyz", angles)
    vertices = rotation.apply(local) + np.asarray(position)
    target["robot"].update(position=list(position), orientation=rotation.as_quat().tolist(),
        base_aabb=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        base_geometry={"available": False, "reason": "old inventory has only measured world AABB"})
    target["robot"]["joints"] = {"arm_joint": -.7, "base_x_joint": 2.3}
    target.update(objects=[{"name": "protected_actor", "protected": True, "aabb": [[1, 2, 3], [4, 5, 6]]}],
                  structural_targets=[{"name": "unit_a", "protected": True}],
                  regions={"delivery": {"support": "native_table"}}, read_only=True,
                  custom_metadata={"retain": ["every", "original", "nonrobot", "field"]})
    return target, reference, local, vertices


def transfer(target, reference):
    return refine.refine_inventory(target, reference, input_sha256="a" * 64, reference_sha256="b" * 64,
                                    input_file="/inputs/target.json", reference_file="/inputs/reference.json")


@pytest.mark.parametrize("angles,position", [
    ((0., 0., 0.), (0., 0., 0.)),
    ((0., 0., np.pi / 4), (2.1, -1.4, .04)),
    ((.025, -.03, np.pi / 4), (-.8, 1.2, .02)),
])
def test_only_rigid_measured_pose_transform_is_accepted(angles, position):
    target, reference, local, expected_world = inventory_pair(angles=angles, position=position)
    before, reference_before = deepcopy(target), deepcopy(reference)
    result, record = transfer(target, reference)
    assert record["status"] == "accepted" and record["refinement_applied"]
    geometry = result["robot"]["base_geometry"]
    assert geometry["available"] and geometry["provenance"] == "rigid_pose_transfer_of_native_reference_base_vertices"
    assert np.allclose(geometry["world_vertices"], expected_world, atol=1e-12)
    assert np.allclose(geometry["base_local_xy_bounds"], [local[:, :2].min(axis=0), local[:, :2].max(axis=0)])
    assert geometry["max_world_aabb_residual_m"] < 1e-12
    assert not geometry["target_vertices_newly_measured"]
    transform = np.asarray(record["frame_transform"]["reference_world_to_target_world"])
    reference_vertices = np.asarray(reference["robot"]["base_geometry"]["world_vertices"])
    assert np.allclose((np.column_stack((reference_vertices, np.ones(len(reference_vertices)))) @ transform.T)[:, :3], expected_world)
    assert not any(record["frame_transform"][key] for key in ("scale_applied", "fit_applied", "recenter_applied"))
    for key in target:
        if key != "robot":
            assert result[key] == target[key]
    for key in target["robot"]:
        if key != "base_geometry":
            assert result["robot"][key] == target["robot"][key]
    assert target == before and reference == reference_before
    assert result[refine.RECORD_KEY]["input_sha256"] == "a" * 64
    assert result[refine.RECORD_KEY]["reference_sha256"] == "b" * 64
    assert record["assumption"] == refine.ASSUMPTION


def test_target_aabb_mismatch_is_rejected_without_fitting_or_overwriting_old_geometry():
    target, reference, *_ = inventory_pair()
    target["robot"]["base_aabb"][1][0] += .003
    before = deepcopy(target)
    result, record = transfer(target, reference)
    assert record["status"] == "rejected" and "target base_aabb" in record["reason"]
    assert record["target_world_aabb_residual_m"] == pytest.approx(.003)
    assert result["robot"] == before["robot"] and target == before
    assert result["objects"] == before["objects"] and result["structural_targets"] == before["structural_targets"]
    assert not record["refinement_applied"]


@pytest.mark.parametrize("change", ["calibration_value", "calibration_schema", "calibration_nonfinite", "arm_calibration", "urdf", "usd", "joint_names", "empty_calibration"])
def test_differing_assets_calibration_or_joint_schema_rejects_transfer(change):
    target, reference, *_ = inventory_pair()
    if change == "calibration_value":
        target["robot"]["joint_calibration"]["base_x_joint"]["parent_joint"][0][3] = 2e-8
    elif change == "calibration_schema":
        del target["robot"]["joint_calibration"]["base_x_joint"]["child"]
    elif change == "calibration_nonfinite":
        target["robot"]["joint_calibration"]["base_x_joint"]["axis"][0] = float("nan")
    elif change == "arm_calibration":
        target["robot"]["joint_calibration"]["arm_joint"]["axis"][0] = .01
    elif change in {"urdf", "usd"}:
        target["robot"][change + "_path"] = "/another/model"
    elif change == "joint_names":
        target["robot"]["joints"]["extra_joint"] = 0.
    else:
        target["robot"]["joint_calibration"] = {}
    result, record = transfer(target, reference)
    assert record["status"] == "rejected"
    assert not result["robot"]["base_geometry"]["available"]
    assert not record["refinement_applied"]


def test_same_schema_numerical_calibration_within_absolute_tolerance_and_changed_encoders_passes():
    target, reference, *_ = inventory_pair()
    target["robot"]["joint_calibration"]["base_x_joint"]["parent_joint"][0][3] = .5e-8
    result, record = transfer(target, reference)
    assert record["status"] == "accepted"
    assert result["robot"]["joints"] != reference["robot"]["joints"]


@pytest.mark.parametrize("change", ["nonfinite_vertex", "reference_aabb", "not_native", "vertex_count", "zero_quaternion"])
def test_reference_must_remain_a_corroborated_native_measurement(change):
    target, reference, *_ = inventory_pair()
    geometry = reference["robot"]["base_geometry"]
    if change == "nonfinite_vertex":
        geometry["world_vertices"][0][0] = float("inf")
    elif change == "reference_aabb":
        reference["robot"]["base_aabb"][0][0] -= .003
    elif change == "not_native":
        geometry["provenance"] = "rigid_pose_transfer_of_native_reference_base_vertices"
    elif change == "vertex_count":
        geometry["vertex_count"] += 1
    else:
        reference["robot"]["orientation"] = [0., 0., 0., 0.]
    result, record = transfer(target, reference)
    assert record["status"] == "rejected" and result["robot"] == target["robot"]


def test_available_target_geometry_and_preexisting_record_are_never_overwritten():
    target, reference, *_ = inventory_pair()
    target["robot"]["base_geometry"] = deepcopy(reference["robot"]["base_geometry"])
    target[refine.RECORD_KEY] = {"original_record": True}
    for unused_reference in ({}, None):
        result, record = transfer(target, unused_reference)
        assert record["status"] == "already_present" and result == target
    target["robot"]["base_geometry"]["available"] = False
    result, record = transfer(target, reference)
    assert record["status"] == "rejected" and result == target


def test_cli_retains_raw_bytes_records_exact_hashes_and_uses_an_immutable_output(tmp_path):
    target, reference, *_ = inventory_pair(angles=(0., 0., np.pi / 4), position=(.4, .3, .02))
    bad = deepcopy(target)
    bad["robot"]["base_aabb"][0][0] -= .01
    present = deepcopy(target)
    present["robot"]["base_geometry"] = {"available": True, "leave_exactly": [1, 2, 3]}
    sources = {"reference": reference, "good": target, "bad": bad, "present": present}
    paths, raw = {}, {}
    for name, value in sources.items():
        paths[name] = tmp_path / (name + ".json")
        raw[name] = (json.dumps(value, indent=4) + "\n\n").encode()
        paths[name].write_bytes(raw[name])
    output = tmp_path / "immutable_results"
    argv = ["--reference", str(paths["reference"]), "--inputs", *[str(paths[key]) for key in ("good", "bad", "present")], "--output", str(output)]
    assert refine.main(argv) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["counts"] == {"accepted": 1, "rejected": 1, "already_present": 1}
    assert (output / "reference/inventory.json").read_bytes() == raw["reference"]
    for record in manifest["records"]:
        name = Path(record["input_file"]).stem
        assert Path(record["original_copy"]).read_bytes() == raw[name]
        assert record["input_sha256"] == hashlib.sha256(raw[name]).hexdigest()
        assert record["reference_sha256"] == hashlib.sha256(raw["reference"]).hexdigest()
        result_raw = Path(record["output_file"]).read_bytes()
        assert record["output_sha256"] == hashlib.sha256(result_raw).hexdigest()
        if record["status"] == "already_present":
            assert result_raw == raw[name]
        assert paths[name].read_bytes() == raw[name]
    assert paths["reference"].read_bytes() == raw["reference"]
    assert all((output / (status + ".json")).is_file() for status in ("accepted", "rejected", "already_present"))
    with pytest.raises(FileExistsError):
        refine.main(argv)
