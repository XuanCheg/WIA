"""Check refresh planning without real queue mutation. Run: PYTHONPATH=src pytest -q tests/test_refresh_wia_captures.py."""
from copy import deepcopy
import importlib.util
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location("refresh_wia_captures", Path(__file__).resolve().parents[1] / "scripts/refresh_wia_captures.py")
refresh = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(refresh)


def png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fixture image bytes")
    return path


def reference(path, alias="T01"):
    return {"path": str(path), "alias": alias, "origin": "sealed_receipt.json", "mapping": "exact_sealed_job_receipt"}


def test_plan_selects_only_explicit_regular_campaign_images_and_is_read_only(tmp_path):
    campaign = tmp_path / "campaign"
    selected = png(campaign / "jobs/selected/head.png")
    unrelated = png(campaign / "jobs/unrelated/head.png")
    outside = png(tmp_path / "outside.png")
    log = campaign / "jobs/selected/native.log"
    log.write_text("retained log")
    renamed_json = campaign / "jobs/selected/receipt.png"
    renamed_json.write_text('{"this": "is a receipt, not an image"}')
    refs = [reference(path) for path in [selected, outside, log, renamed_json]]
    before = {path: path.read_bytes() for path in [selected, unrelated, outside, log, renamed_json]}
    deleted, retained = refresh.plan_image_deletions(refs, campaign)
    assert [entry["path"] for entry in deleted] == [str(selected)]
    assert {entry["reason"] for entry in retained} == {"outside_campaign", "not_an_image_file", "image_header_unrecognized"}
    assert {path: path.read_bytes() for path in before} == before
    assert len(deleted[0]["sha256"]) == 64


def test_shared_images_and_symlinked_assets_are_protected(tmp_path):
    campaign = tmp_path / "campaign"
    shared = png(campaign / "assets/shared.png")
    target = png(campaign / "assets/other.png")
    alias = campaign / "assets/alias.png"
    alias.symlink_to(target)
    directory_alias = campaign / "linked"
    directory_alias.symlink_to(campaign / "assets", target_is_directory=True)
    refs = [reference(shared), reference(alias), reference(directory_alias / "other.png")]
    deleted, retained = refresh.plan_image_deletions(refs, campaign, protected_paths=[shared])
    assert not deleted
    assert {entry["reason"] for entry in retained} == {"also_referenced_by_unselected_alias", "not_a_regular_image_file"}
    assert alias.is_symlink() and shared.is_file() and target.is_file()


def test_index_update_preserves_unrelated_attempt02_and_all_other_fields():
    index = {"schema": "review", "extra": {"retain": True}, "items": [
        {"document_alias": "T01", "job_id": "old-a", "custom": [1]},
        {"document_alias": "T45", "job_id": "unchanged", "receipt_file": "attempts/02/preview_complete.json", "custom": [2]},
    ]}
    original = deepcopy(index)
    updates = {"T01": {"job_id": "new-a", "receipt_file": "new/attempts/01/preview_complete.json"}}
    changed = refresh.replacement_index(index, updates)
    assert index == original and changed["extra"] == original["extra"]
    assert changed["items"][1] == original["items"][1]
    assert changed["items"][0]["custom"] == [1]
    assert refresh.replacement_index(changed, updates) == changed


def test_separate_source_roles_are_retained_in_new_record_identity(tmp_path):
    task_identity = "Physical Structure/Rigid Containment/scene/room/criterion_4.json"
    scene_identity = "Physical Structure/Rigid Containment/scene/room/revised.json"
    members = [{"source_identity": task_identity, "source_file": "qa.json", "source_sha256": "qa-hash"},
               {"source_identity": scene_identity, "source_file": "native.json", "source_sha256": "native-hash"}]
    definition = {"alias": "T23", "task_source_identity": task_identity, "scene_source_identity": scene_identity,
                  "capture_source_members": members}
    record = refresh._record(definition, {"code_commit": "frozen-code"}, tmp_path)
    assert record["source_identity"] == scene_identity and record["original_source"] == "native.json"
    assert record["capture_source_members"] == members
    assert record["task_source_identity"] == task_identity
    assert record == refresh._record(definition, {"code_commit": "frozen-code"}, tmp_path)


def test_published_assets_require_exact_current_alias_variant_and_job(tmp_path):
    import json

    image = png(tmp_path / "current.png")
    unrelated = png(tmp_path / "unrelated.png")
    item = {"row_id": "row-a", "document_alias": "T01", "variant_id": "current", "job_id": "job-a"}
    document = {"row_updates": {"row-a": {"variants": [
        {"id": "current", "capture_review": {"job_id": "job-a"}, "published_assets": [str(image)]},
        {"id": "old", "images": [str(unrelated)]},
        {"id": "current", "capture_review": {"job_id": "other-job"}, "images": [str(unrelated)]},
    ]}}}
    (tmp_path / "capture_evidence.json").write_text(json.dumps(document))
    assets = refresh._published_assets(tmp_path, item)
    assert [entry["path"] for entry in assets] == [str(image)]
