"""Verify persistent human reviews and exact-layout preview selection.

Run: .venv/bin/python -m pytest -q tests/test_scene_review.py
"""
import csv
import io
import json
import threading

import pytest
import requests
import yaml

from wia.runtime.scene_review import VIEW_LABELS, ReviewConflict, SceneReview, read_json, sha256, write_json
from wia.runtime.commands.convert_esi_rigid_catalog import publish_result
from wia.runtime.commands.review_esi_scenes import create_server
from harness_integration.test_rigid_catalog_conversion import publication


@pytest.fixture
def review(tmp_path):
    root, source, descriptor, candidate, validation = publication(tmp_path)
    publish_result(root, source, {"status": "ready"}, candidate=candidate, validation_dir=validation)
    marker = read_json(validation / "validation_complete.json")
    marker["variant"] = "tabletop"
    write_json(validation / "validation_complete.json", marker)
    for name in ("initial_head.png", "initial_wrist.png"):
        (validation / name).write_bytes(b"test preview")
    conversion = tmp_path / "conversion"
    write_json(conversion / "run/conversion_summary.json", {"config": {"seed": 0}, "results": [{
        "source": str(source.relative_to(root)), "status": "ready", "validation_dir": str(validation),
        "verification_level": "approximate_native_repaired_restart", "physical_grasp_verified": False}]})
    store = SceneReview(conversion, tmp_path / "reviews", data_root=root)
    return store, source, descriptor, validation


def payload(row, decision="approved", notes="Layout looks usable."):
    return {"decision": decision, "notes": notes, "revision": row["review"].get("revision", 0),
            "fingerprint": row["fingerprint"]}


def test_reuses_final_restart_without_exposing_source_answers(review):
    store, source, _, _ = review
    raw = read_json(source)
    raw["answer"] = "PRIVATE_ANSWER_CANARY"
    raw["metadata_json"] = json.loads(raw["metadata_json"])
    raw["metadata_json"]["containee"]["placement"]["reason"] = "PRIVATE_HISTORY_CANARY"
    write_json(source, raw)
    assert store.import_previews() == 1
    value = store.catalogue()
    assert value["scenes"][0]["has_final_preview"]
    assert not value["scenes"][0]["has_global_preview"]
    assert not value["scenes"][0]["physical_grasp_verified"]
    assert "PRIVATE_" not in json.dumps(value)


def test_old_initial_photos_and_changed_candidate_are_not_reused(review):
    store, _, _, validation = review
    marker = read_json(validation / "validation_complete.json")
    from pathlib import Path
    candidate = Path(marker["scene_file"])
    value = yaml.safe_load(candidate.read_text())
    value["scene"]["variants"]["tabletop"]["patches"][0]["position"][0] += .2
    candidate.write_text(yaml.safe_dump(value))
    assert store.import_previews() == 0
    assert not store.catalogue()["scenes"][0]["has_final_preview"]
    row = store.catalogue()["scenes"][0]
    saved = store.save_review(row["id"], payload(row))
    assert saved["approval_scope"] == "layout"
    assert not saved["approval_evidence"]["has_final_preview"]


def test_persistence_export_and_scene_files_are_independent(review):
    store, source, descriptor, _ = review
    before_source, before_scene = source.read_bytes(), descriptor.read_bytes()
    assert store.import_previews() == 1
    row = store.catalogue()["scenes"][0]
    saved = store.save_review(row["id"], payload(row, notes="Clearance checked.\nKeep this layout."))
    reopened = SceneReview(store.conversion_root, store.output, data_root=store.data_root)
    row = reopened.catalogue()["scenes"][0]
    assert row["decision"] == "approved" and row["review"]["revision"] == 1
    exported = next(csv.DictReader(io.StringIO(reopened.export_csv())))
    assert exported["notes"] == saved["notes"]
    assert exported["physical_grasp_verified"] == "False"
    assert source.read_bytes() == before_source and descriptor.read_bytes() == before_scene


def test_preserved_approved_scene_remains_visible_in_a_new_conversion_summary(review):
    store, _, descriptor, _ = review
    store.import_previews()
    row = store.catalogue()["scenes"][0]
    store.save_review(row["id"], payload(row))
    before = descriptor.read_bytes()
    summary = read_json(store.summary_path)
    summary["preserved_scenes"] = summary.pop("results")
    summary["results"] = []
    write_json(store.summary_path, summary)
    retained = store.catalogue()["scenes"][0]
    assert retained["id"] == row["id"] and retained["decision"] == "approved"
    assert retained["has_final_preview"] and descriptor.read_bytes() == before


@pytest.mark.parametrize("invalid", [None, "source_sha256", "scene_sha256", "visible_target_pixels"])
def test_reuses_complete_views_captured_during_exact_native_validation(review, invalid):
    from pathlib import Path

    store, source, _, validation = review
    tested = Path(read_json(validation / "validation_complete.json")["scene_file"])
    row = store.catalogue()["scenes"][0]
    inline = {"scene_file": str(tested), "scene_sha256": sha256(tested), "source_sha256": sha256(source),
        "source_json": row["source"], "variant": row["variant"], "seed": row["seed"],
        "images": {view: str(validation / "initial_head.png") for view in VIEW_LABELS},
        "target_location": {"camera_capture_protocol": "target_instance_verified_v3", "visible_target_pixels": 256}}
    if invalid == "visible_target_pixels":
        inline["target_location"][invalid] = 0
    elif invalid:
        inline[invalid] = "changed"
    write_json(validation / "native_review_complete.json", inline)
    assert store.import_previews() == 1
    result = store.catalogue()["scenes"][0]
    assert result["has_global_preview"] is (invalid is None)
    assert (len(result["images"]) == 10) is (invalid is None)
    if invalid is None:
        assert result["target_location"]["visible_target_pixels"] == 256


@pytest.mark.parametrize("changed_bank_hash", [False, True])
def test_native_screen_views_require_the_certified_candidate(review, changed_bank_hash):
    from pathlib import Path

    store, source, _, validation = review
    tested = Path(read_json(validation / "validation_complete.json")["scene_file"])
    folder = store.conversion_root / "case"
    sampling = folder / "sampling"
    sampling.mkdir(parents=True)
    candidate = sampling / "placement_000.yaml"
    candidate.write_bytes(tested.read_bytes())
    summary = read_json(store.summary_path)
    summary["results"][0].update(verification_level="approximate_native_screen", output=str(folder),
        candidate_file=str(candidate))
    write_json(store.summary_path, summary)
    row = store.catalogue()["scenes"][0]
    write_json(sampling / "scene_status.json", {"qualification_status": "accepted_minimal_patch"})
    write_json(sampling / "support_bank.json", {"status": "screened", "source_json": row["source"], "accepted": [{
        "scene_file": candidate.name, "accepted": True,
        "scene_sha256": "changed" if changed_bank_hash else sha256(candidate)}]})
    write_json(sampling / "review_views/placement_000/native_review_complete.json", {
        "scene_file": str(candidate), "scene_sha256": sha256(candidate), "source_sha256": sha256(source),
        "source_json": row["source"], "variant": row["variant"], "seed": row["seed"],
        "images": {view: str(validation / "initial_head.png") for view in VIEW_LABELS},
        "target_location": {"camera_capture_protocol": "target_instance_verified_v3", "visible_target_pixels": 256}})
    assert store.import_previews() == (0 if changed_bank_hash else 1)
    assert store.catalogue()["scenes"][0]["has_global_preview"] is (not changed_bank_hash)


def test_object_camera_requires_visible_target_evidence_without_hiding_global_views(review):
    store, _, _, _ = review
    row = store.catalogue()["scenes"][0]
    folder = store.output / "captures" / row["id"]
    folder.mkdir(parents=True)
    image = folder / "test.png"
    image.write_bytes(b"native image fixture")
    marker = {"fingerprint": row["fingerprint"], "variant": row["variant"], "seed": row["seed"],
        "images": {"global_diagonal": str(image), "global_target": str(image)},
        "target_location": {"camera_capture_protocol": "target_instance_verified_v3", "visible_target_pixels": 0}}
    receipt = folder / "preview_complete.json"
    write_json(receipt, marker)
    result = store.catalogue()["scenes"][0]
    assert result["has_global_preview"]
    assert [i["view"] for i in result["images"]] == ["global_diagonal"]
    marker["target_location"]["visible_target_pixels"] = 128
    write_json(receipt, marker)
    assert "global_target" in {i["view"] for i in store.catalogue()["scenes"][0]["images"]}


def test_conflicting_tabs_and_changed_layout_require_new_review(review):
    store, _, descriptor, _ = review
    store.import_previews()
    row = store.catalogue()["scenes"][0]
    store.save_review(row["id"], payload(row))
    with pytest.raises(ReviewConflict, match="another tab"):
        store.save_review(row["id"], payload(row, "needs_fix"))
    value = yaml.safe_load(descriptor.read_text())
    value["scene"]["variants"]["tabletop"]["patches"][0]["position"][0] += .2
    descriptor.write_text(yaml.safe_dump(value))
    current = store.catalogue()["scenes"][0]
    assert current["review_stale"] and current["decision"] == "unreviewed"
    assert not current["has_final_preview"]
    with pytest.raises(ReviewConflict, match="scene changed"):
        store.save_review(row["id"], payload(row))


def test_final_review_archives_old_layout_notes_verbatim(review):
    store, _, descriptor, _ = review
    store.import_previews()
    row = store.catalogue()["scenes"][0]
    store.save_review(row["id"], payload(row, "needs_fix", "Move the target to another surface."))
    original = store.review_file.read_bytes()
    value = yaml.safe_load(descriptor.read_text())
    value["scene"]["variants"]["tabletop"]["patches"][0]["position"][0] += .2
    descriptor.write_text(yaml.safe_dump(value))
    result = store.archive_superseded_reviews()
    from pathlib import Path

    assert result["removed_review_ids"] == [row["id"]]
    assert Path(result["archive"]).read_bytes() == original
    latest = store.catalogue()["scenes"][0]
    assert latest["decision"] == "unreviewed" and not latest["review_stale"]
    assert latest["review"]["notes"] == "" and latest["review"]["revision"] == 0
    with pytest.raises(ReviewConflict, match="scene changed"):
        store.save_review(row["id"], payload(row))


def test_final_review_preserves_current_decisions_and_removes_excluded_records(review):
    store, _, _, _ = review
    store.import_previews()
    row = store.catalogue()["scenes"][0]
    saved = store.save_review(row["id"], payload(row))
    records = read_json(store.review_file)
    records["entries"]["excluded_scene"] = {"decision": "skipped", "notes": "Historical exclusion."}
    write_json(store.review_file, records)
    result = store.archive_superseded_reviews()
    assert result["removed_review_ids"] == ["excluded_scene"]
    assert result["retained_review_count"] == 1
    assert store.catalogue()["scenes"][0]["review"] == saved
    assert store.archive_superseded_reviews()["archive"] is None


def test_manual_approval_does_not_make_pending_scene_runnable(review):
    store, _, descriptor, _ = review
    store.import_previews()
    value = yaml.safe_load(descriptor.read_text())
    value["status"] = "pending"
    descriptor.write_text(yaml.safe_dump(value))
    row = store.catalogue()["scenes"][0]
    saved = store.save_review(row["id"], payload(row))
    assert saved["approval_evidence"]["conversion_status"] == "pending"
    row = store.catalogue()["scenes"][0]
    store.save_review(row["id"], payload(row, "needs_fix", "Repair the startup wall contact."))
    assert yaml.safe_load(descriptor.read_text())["status"] == "pending"


def test_pending_diagnostic_preview_is_visible_without_relaxing_runtime_gate(review):
    from wia.runtime.commands.capture_esi_scene_reviews import load_pending_preview
    from wia.tasks.loader import load_scene

    store, _, descriptor, validation = review
    value = yaml.safe_load(descriptor.read_text())
    value["status"] = "pending"
    descriptor.write_text(yaml.safe_dump(value))
    row = store.catalogue()["scenes"][0]
    before = descriptor.read_bytes()
    preview = load_pending_preview(row, store.data_root)
    assert preview.definition.status == "pending"
    preview.verify_unchanged()
    write_json(store.output / "captures" / row["id"] / "preview_complete.json", {
        "fingerprint": row["fingerprint"], "variant": row["variant"], "seed": row["seed"],
        "diagnostic_pending": True, "images": {"global_diagonal": str(validation / "initial_head.png")}})
    visible = store.catalogue()["scenes"][0]
    assert visible["has_global_preview"] and visible["diagnostic_pending"]
    assert visible["conversion_status"] == "pending"
    with pytest.raises(ValueError, match="pending"):
        load_scene(descriptor, data_root=store.data_root)
    saved = store.save_review(visible["id"], payload(visible))
    assert saved["approval_scope"] == "layout"
    assert saved["approval_evidence"]["conversion_status"] == "pending"
    with pytest.raises(ValueError, match="pending"):
        load_scene(descriptor, data_root=store.data_root)
    assert descriptor.read_bytes() == before


@pytest.mark.parametrize("field,value", [("fingerprint", "different"), ("variant", "original"), ("seed", 99)])
def test_mismatched_capture_receipt_is_not_displayed(review, field, value):
    store, _, _, validation = review
    row = store.catalogue()["scenes"][0]
    receipt = {"fingerprint": row["fingerprint"], "variant": row["variant"], "seed": row["seed"],
               "images": {"head": str(validation / "initial_head.png")}}
    receipt[field] = value
    write_json(store.output / "captures" / row["id"] / "preview_complete.json", receipt)
    assert not store.catalogue()["scenes"][0]["has_final_preview"]


def test_source_hash_mismatch_blocks_preview_import(review):
    store, source, _, validation = review
    manifest = read_json(validation / "scene_manifest.json")
    manifest["source_sha256"] = sha256(source)
    write_json(validation / "scene_manifest.json", manifest)
    raw = read_json(source)
    raw["question"] += " Changed."
    write_json(source, raw)
    assert store.import_previews() == 0


def test_scene_seed_overrides_conversion_run_default(review):
    store, _, descriptor, _ = review
    value = yaml.safe_load(descriptor.read_text())
    value["seed"] = 7
    descriptor.write_text(yaml.safe_dump(value))
    row = store.catalogue()["scenes"][0]
    assert row["seed"] == 7
    assert "--scene-seed 7" in row["viewer_command"]


def test_http_review_save_images_and_error_responses(review):
    store, _, _, _ = review
    store.import_previews()
    server = create_server(store, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        response = requests.get(url + "/api/scenes", timeout=5)
        assert response.status_code == 200
        row = response.json()["scenes"][0]
        assert requests.get(url + "/" + row["images"][0]["url"], timeout=5).content == b"test preview"
        assert requests.get(url + f"/images/{row['id']}/unknown", timeout=5).status_code == 404
        response = requests.post(url + f"/api/reviews/{row['id']}", json=payload(row), timeout=5)
        assert response.status_code == 200
        assert requests.post(url + f"/api/reviews/{row['id']}", json=payload(row), timeout=5).status_code == 409
        assert requests.post(url + f"/api/reviews/{row['id']}", json=[], timeout=5).status_code == 400
        assert "approved" in requests.get(url + "/api/reviews.csv", timeout=5).text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_shared_gateway_mount_preserves_review_posts_and_csv(review):
    from starlette.testclient import TestClient
    from wia.runtime.web.http_gateway import create_gateway

    store, _, _, _ = review
    store.import_previews()
    backend = create_server(store, port=0)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    app = create_gateway("http://127.0.0.1:1", review_url=f"http://127.0.0.1:{backend.server_port}")
    try:
        with TestClient(app.streamable_http_app(json_response=True)) as client:
            page = client.get("/review")
            assert page.status_code == 200 and str(page.url).endswith("/review/")
            row = client.get("/review/api/scenes").json()["scenes"][0]
            saved = client.post(f"/review/api/reviews/{row['id']}", json=payload(row))
            assert saved.status_code == 200
            assert client.get("/review/app.js").headers["content-type"].startswith("text/javascript")
            exported = client.get("/review/api/reviews.csv")
            assert "approved" in exported.text and "attachment" in exported.headers["content-disposition"]
    finally:
        backend.shutdown()
        backend.server_close()
        thread.join(timeout=5)
