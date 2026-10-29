"""Check that publication requires native acceptance for the exact placement.

Run: .venv/bin/python -m pytest -q tests/test_rigid_catalog_conversion.py
"""
import json
import hashlib

import pytest
import yaml

from wia.tasks.loader import DATA_ROOT, load_scene
from wia.tasks.tabletop import generate_tabletop
from wia.runtime.commands.convert_esi_rigid_catalog import initialized_tabletop_fallback, publish_result, restrict_sources
from harness_integration.test_tabletop_scenes import SOURCE


def test_explicit_repair_scope_preserves_inspection_only_sources_without_approval(tmp_path):
    root = tmp_path / "scenes"
    repair, inspect = root / "repair.json", root / "inspect.json"
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps(["repair.json"]))
    definition = object()
    prior = [{"source": str(root / "skipped.json"), "reason": "human_review_excluded"}]
    kept, excluded = restrict_sources([(repair, definition), (inspect, definition)], prior, selected, root)
    assert kept == [(repair, definition)]
    assert excluded == prior + [{"source": str(inspect), "reason": "inspection_only_preserved"}]
    assert prior == [{"source": str(root / "skipped.json"), "reason": "human_review_excluded"}]
    selected.write_text(json.dumps(["missing.json"]))
    with pytest.raises(ValueError, match="eligible canonical"):
        restrict_sources([(repair, definition)], prior, selected, root)


def publication(tmp_path):
    root = tmp_path / "scenes"
    relative = SOURCE.relative_to(DATA_ROOT)
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(SOURCE.read_bytes())
    descriptor = source.with_suffix(".yaml")
    descriptor.write_text(yaml.safe_dump({"kind": "scene_import", "status": "pending",
        "source": {"provider": "esi", "json": str(relative)}}))
    (root / "catalog.json").write_text(json.dumps({"scenes": [{"source_json": str(relative),
        "scene_file": str(relative.with_suffix(".yaml")), "status": "pending"}]}))
    definition, _ = generate_tabletop(source, root)
    value = definition.model_dump(mode="json", by_alias=True)
    candidate = tmp_path / "candidate.yaml"
    value["status"] = "pending"
    candidate.write_text(yaml.safe_dump(value))
    restarted = tmp_path / "restart.yaml"
    value["status"] = "ready"
    restarted.write_text(yaml.safe_dump(value))
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "scene_status.json").write_text(json.dumps({"qualification_status": "accepted_minimal_patch"}))
    (validation / "validation_complete.json").write_text(json.dumps({"source_json": str(relative),
        "scene_file": str(restarted)}))
    (validation / "scene_manifest.json").write_text(json.dumps({"source_path": str(source),
        "scene_file": str(restarted), "scene_sha256": hashlib.sha256(restarted.read_bytes()).hexdigest()}))
    return root, source, descriptor, candidate, validation


def test_only_exact_restarted_placement_is_published_and_source_remains_unchanged(tmp_path):
    root, source, descriptor, candidate, validation = publication(tmp_path)
    before = source.read_bytes()
    publish_result(root, source, {"status": "ready"}, candidate=candidate, validation_dir=validation)
    assert load_scene(descriptor, data_root=root).definition.status == "ready"
    assert source.read_bytes() == before
    assert json.loads((root / "catalog.json").read_text())["status_counts"] == {"ready": 1}


def test_different_pose_or_failed_native_preflight_cannot_publish_ready(tmp_path):
    root, source, descriptor, candidate, validation = publication(tmp_path)
    original = descriptor.read_bytes()
    value = yaml.safe_load(candidate.read_text())
    value["scene"]["variants"]["tabletop"]["patches"][0]["position"][0] += .1
    candidate.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="differs from the independently restarted"):
        publish_result(root, source, {"status": "ready"}, candidate=candidate, validation_dir=validation)
    (validation / "scene_status.json").write_text(json.dumps({"qualification_status": "skipped"}))
    with pytest.raises(ValueError, match="Cannot publish without independent native"):
        publish_result(root, source, {"status": "ready"}, candidate=candidate, validation_dir=validation)
    assert descriptor.read_bytes() == original


def native_screen(tmp_path):
    root, source, descriptor, candidate, sampling = publication(tmp_path)
    value = yaml.safe_load(candidate.read_text())
    pose = next(patch for patch in value["scene"]["variants"]["tabletop"]["patches"]
        if patch["operation"] == "object_pose" and patch["binding"] == value["target"])
    value["bindings"]["pickup_support"] = {"name": "screened_table", "category": "table"}
    value["qualification"]["object_supports"][value["target"]] = "pickup_support"
    candidate.write_text(yaml.safe_dump(value))
    bank = {"status": "screened", "source_json": value["source"]["json"], "accepted": [{
        "accepted": True, "scene_file": str(candidate), "support": "screened_table",
        "position": pose["position"], "quaternion_xyzw": pose["quaternion_xyzw"]}]}
    (sampling / "support_bank.json").write_text(json.dumps(bank))
    return root, source, descriptor, candidate, sampling


def test_completed_native_screen_publishes_with_manual_review_flag(tmp_path):
    root, source, descriptor, candidate, sampling = native_screen(tmp_path)
    (sampling / "validation_complete.json").unlink()
    row = {"status": "ready", "verification_level": "approximate_native_screen", "manual_review_required": True}
    publish_result(root, source, row, candidate=candidate, sampling_dir=sampling)
    assert load_scene(descriptor, data_root=root).definition.status == "ready"
    index = json.loads((root / "catalog.json").read_text())
    assert index["scenes"][0]["conversion_result"]["manual_review_required"] is True


@pytest.mark.parametrize("invalid", ["unfinished", "changed_pose", "wrong_source"])
def test_incomplete_or_mismatched_native_screen_cannot_publish(tmp_path, invalid):
    root, source, descriptor, candidate, sampling = native_screen(tmp_path)
    bank_path = sampling / "support_bank.json"
    bank = json.loads(bank_path.read_text())
    if invalid == "unfinished":
        bank["status"] = "running"
    elif invalid == "changed_pose":
        bank["accepted"][0]["position"][0] += .1
    else:
        bank["source_json"] = "different_source.json"
    bank_path.write_text(json.dumps(bank))
    original = descriptor.read_bytes()
    with pytest.raises(ValueError):
        publish_result(root, source, {"status": "ready", "verification_level": "approximate_native_screen"},
            candidate=candidate, sampling_dir=sampling)
    assert descriptor.read_bytes() == original


@pytest.mark.parametrize("invalid", [None, "skipped", "too_far", "changed_scene"])
def test_initialized_fallback_requires_accepted_exact_scene_and_coarse_reach(tmp_path, invalid):
    root, source, descriptor, candidate, sampling = publication(tmp_path)
    folder = tmp_path / "case"
    folder.mkdir()
    initialized = folder / "initialization.yaml"
    initialized.write_bytes(candidate.read_bytes())
    (sampling / "private_scene").mkdir()
    value = yaml.safe_load(candidate.read_text())
    (sampling / "private_scene/qualification.json").write_text(json.dumps({"checks": {
        "supported_objects": {value["target"]: {"passed": True, "aabb_min": [.5,-.1,.45],
            "aabb_max": [.7,.1,.65], "support_surface_z_m": .45, "support_name": "support_table_top"}},
        "robot_nonfloor_contacts": []}}))
    (sampling / "trunk_initialization.json").write_text(json.dumps({"base_position": [0.,0.,.1]}))
    (sampling / "scene_manifest.json").write_text(json.dumps({"source_path": str(source),
        "scene_file": str(initialized), "scene_sha256": hashlib.sha256(initialized.read_bytes()).hexdigest()}))
    sampling.rename(folder / "sampling")
    sampling = folder / "sampling"
    if invalid == "skipped":
        (sampling / "scene_status.json").write_text(json.dumps({"qualification_status": "skipped"}))
    elif invalid == "too_far":
        (sampling / "trunk_initialization.json").write_text(json.dumps({"base_position": [-1.,0.,.1]}))
    elif invalid == "changed_scene":
        initialized.write_text(initialized.read_text() + "\n# Changed after native validation.\n")
    row = {"source": str(source.relative_to(root)), "output": str(folder), "status": "pending",
        "failure_stage": "sampling", "reason": "No accepted alternate furniture placement."}
    if invalid == "changed_scene":
        with pytest.raises(ValueError, match="differs from the natively accepted"):
            initialized_tabletop_fallback(root, row, approved=True)
    else:
        result = initialized_tabletop_fallback(root, row, approved=True)
        if invalid is None:
            assert result["status"] == "ready"
            assert result["random_placement"] is False
            assert result["manual_review_required"] is True
        else:
            assert result is None


def test_unapproved_initial_table_never_becomes_a_fallback():
    assert initialized_tabletop_fallback(None, {"status": "pending", "failure_stage": "sampling"}) is None


def test_native_screen_receipt_rejects_changed_target_scale(tmp_path):
    root, source, descriptor, candidate, sampling = native_screen(tmp_path)
    bank_path = sampling / "support_bank.json"
    bank = json.loads(bank_path.read_text())
    bank["accepted"][0]["scene_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    bank_path.write_text(json.dumps(bank))
    value = yaml.safe_load(candidate.read_text())
    value["scene"]["variants"]["tabletop"]["patches"].append({"operation": "object_scale", "binding": "target", "multiplier": .5})
    candidate.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="Converted geometry differs"):
        publish_result(root, source, {"status": "ready", "verification_level": "approximate_native_screen"},
            candidate=candidate, sampling_dir=sampling)
