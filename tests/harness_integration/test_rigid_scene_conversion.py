"""Check physical-source reuse and non-writing converter previews.

Run: .venv/bin/python -m pytest -q tests/test_rigid_scene_conversion.py
"""

import json

import pytest

from wia.tasks.loader import DATA_ROOT, load_scene
from wia.tasks.tabletop import generate_tabletop
from wia.runtime.commands.generate_esi_tabletop import main
from harness_integration.test_tabletop_scenes import SOURCE


def sources(tmp_path):
    root = tmp_path / "source"
    relative = SOURCE.relative_to(DATA_ROOT)
    canonical = root / relative
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(SOURCE.read_bytes())
    variant = canonical.with_name(canonical.stem + "__criterion_2.json")
    question = json.loads(canonical.read_text())
    question["question"] = "Which container does each object best fit inside?"
    question["answer"] = "PRIVATE_VARIANT_ANSWER"
    variant.write_text(json.dumps(question))
    return root, canonical, variant


def test_variant_reuses_verified_construction_but_keeps_its_question_and_pending_status(tmp_path):
    root, canonical, variant = sources(tmp_path)
    before = variant.read_bytes()
    definition, details = generate_tabletop(variant, root)
    assert definition.source.json_path == str(canonical.relative_to(root))
    assert definition.status == "pending"
    assert details["source_question_json"] == str(variant.relative_to(root))
    assert details["source_question"] == "Which container does each object best fit inside?"
    assert details["requires_separate_task_definition"]
    assert details["canonical_construction_verified"]
    assert definition.runtime.trunk_config == "configs/esi_trunk_low_grasp.json"
    assert definition.scene.variants["tabletop"].patches[0].head_pitch_deg == 0.0
    assert variant.read_bytes() == before
    import yaml
    descriptor = tmp_path / "variant.yaml"
    descriptor.write_text(yaml.safe_dump(definition.model_dump(mode="json", by_alias=True)))
    with pytest.raises(ValueError, match="pending and must be skipped"):
        load_scene(descriptor, data_root=root)


def test_variant_with_changed_geometry_cannot_reuse_canonical_initialization(tmp_path):
    root, _, variant = sources(tmp_path)
    record = json.loads(variant.read_text())
    metadata = json.loads(record["metadata_json"])
    poses = metadata["initial_poses"]
    target = next(pose for pose in poses if pose["_key"] == "obj_containee") if isinstance(poses, list) else poses["obj_containee"]
    for key in ["position", "aabb_min", "aabb_max"]:
        target[key][0] += .1
    record["metadata_json"] = json.dumps(metadata)
    variant.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="changes construction inputs"):
        generate_tabletop(variant, root)


def test_corrupt_support_heights_remain_rejected(tmp_path):
    root, canonical, _ = sources(tmp_path)
    record = json.loads(canonical.read_text())
    metadata = json.loads(record["metadata_json"])
    poses = metadata["initial_poses"]
    target = next(pose for pose in poses if pose["_key"] == "obj_containee") if isinstance(poses, list) else poses["obj_containee"]
    for key in ["position", "aabb_min", "aabb_max"]:
        target[key][2] -= 132.
    record["metadata_json"] = json.dumps(metadata)
    canonical.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="support heights differ"):
        generate_tabletop(canonical, root)


def test_dry_run_reports_candidates_without_creating_output(tmp_path, capsys):
    root, _, variant = sources(tmp_path)
    output = tmp_path / "not_created"
    assert main(["--source", str(variant), "--data-root", str(root), "--output-dir", str(output), "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] and result["generated"] == 0 and result["convertible"] == 1
    assert result["candidate_status_counts"] == {"pending": 1}
    assert result["summary"] is None
    assert not output.exists()


def test_batch_keeps_main_question_and_omits_discarded_family_before_conversion(tmp_path, capsys):
    import yaml

    root, canonical, _ = sources(tmp_path)
    for stem in ("corrupt", "corrupt__none"):
        record = canonical.with_name(stem + ".json")
        record.write_text("{}")
        record.with_suffix(".yaml").write_text(yaml.safe_dump({"status": "discarded",
            "discard_reason": "Corrupted source pose."}))
    assert main(["--source-dir", str(root), "--data-root", str(root), "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["input_samples"] == 4
    assert result["convertible"] == 1 and result["generated"] == 0
    assert result["candidate_status_counts"] == {"ready": 1}
    assert result["discarded"] == 2 and result["ignored_task_variants"] == 1
    assert result["skipped"] == 0
