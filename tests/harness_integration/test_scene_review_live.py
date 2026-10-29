"""Check the metadata-only Review and lazy HTTP image delivery.

Run: python -m pytest -q tests/test_scene_review_live.py
The fixtures deliberately use non-PNG bytes: Review must trust completion
metadata and must not perform another image validation or directory scan.
"""
import json
from pathlib import Path
import threading

import pytest
import requests
import yaml

from wia.runtime.scene_capture_queue import write_json
from wia.runtime.scene_review import ReviewConflict
from wia.runtime.scene_review_live import LiveSceneReview
from wia.runtime.commands.review_esi_scenes import create_server
from wia.runtime.commands.update_esi_review_catalogue import CataloguePublisher


@pytest.fixture
def live(tmp_path):
    campaign, queue = tmp_path / "campaign", tmp_path / "queue"
    output = campaign / "review_catalogue"
    write_json(queue / "capture_runtime_policy.json", {"allowed_code_commits": ["current"]})
    write_json(campaign / "inventory.json", {"categories": [{
        "big_task": "Cognitive Mapping", "small_task": "Connectivity",
        "sources": ["a", "b", "c", "d"], "report_file": "report.md"}]})
    records = []
    for suffix, state, code in (("aa", "REVIEW_PENDING", "current"),
                                ("bb", "RUNNING", "current"),
                                ("cc", "FAILED", "current"),
                                ("dd", "REVIEW_PENDING", "old")):
        job, key = "capture_" + suffix, "category_" + suffix
        directory = queue / "jobs" / job
        image = directory / "attempts/01/global_overhead.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"Screenshot bytes are only read when requested.")
        task = directory / "input/task.yaml"
        task.parent.mkdir(parents=True)
        task.write_text(yaml.safe_dump({"public": {"objective": "Navigate to the kitchen."},
                                        "private": {"answer": "PRIVATE_ANSWER_CANARY"}}))
        record = {"candidate_id": key, "fingerprint": "stored-identity-" + suffix,
                  "source_identity": "Cognitive Mapping/Connectivity/house/room/" + suffix + ".json",
                  "source_sha256": "stored-source-identity", "scene_model": "house", "room": "room",
                  "big_task": "Cognitive Mapping", "small_task": "Connectivity", "seed": 0,
                  "status": "generated", "task_file": str(task),
                  "capture_input": str(task.parent / "manifest.json"), "job_id": job,
                  "warning_codes": [], "scene_file": "scene.yaml", "diagnostic_dir": str(directory),
                  "diagnostic_images": [{"path": "/obsolete/screenshot.png", "view": "head",
                                         "kind": "historical"}], "diagnostic_logs": []}
        write_json(queue / "category_records" / (key + ".json"), record)
        write_json(directory / "input/manifest.json", {"candidate_id": key, "code_commit": code})
        receipt = directory / "attempts/01/preview_complete.json"
        write_json(receipt, {"complete": True, "images": {"global_overhead": str(image)}})
        write_json(directory / "status.json", {"state": state, "receipt": str(receipt),
            "partial_images": {"head": str(image)} if state == "FAILED" else {},
            "error": "Native loading failure" if state == "FAILED" else None})
        records.append(record)
    log = campaign / "formal_resident_workers/worker0.log"
    log.parent.mkdir(parents=True)
    log.write_text("".join("[capture-worker] worker0 capture_" + suffix + " " + state + "\n"
                           for suffix, state in (("aa", "REVIEW_PENDING"), ("bb", "RUNNING"),
                                                 ("cc", "FAILED"), ("dd", "REVIEW_PENDING"))))
    output.mkdir(parents=True)
    publisher = CataloguePublisher(campaign, queue, output)
    store = LiveSceneReview(None, tmp_path / "reviews", queue_root=queue,
                            catalogue_file=output / "snapshot.json")
    return publisher, store, records, log


def forbid_scan(*args, **kwargs):
    raise AssertionError("Review must not traverse directories or validate image bytes.")


def test_completed_metadata_and_selected_detail_never_scan_or_validate(live, monkeypatch):
    publisher, store, _, _ = live
    import PIL.Image
    import wia.runtime.scene_review
    monkeypatch.setattr(Path, "glob", forbid_scan)
    monkeypatch.setattr(Path, "rglob", forbid_scan)
    monkeypatch.setattr(Path, "iterdir", forbid_scan)
    monkeypatch.setattr(PIL.Image, "open", forbid_scan)
    monkeypatch.setattr(wia.runtime.scene_review, "sha256", forbid_scan)
    original_open = Path.open
    def metadata_only(path, *args, **kwargs):
        if path.suffix == ".png":
            forbid_scan()
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", metadata_only)
    publisher.tick()
    listing = store.catalogue(page=0)
    assert [row["id"] for row in listing["scenes"]] == ["category_aa"]
    assert listing["scenes"][0]["images"] == []
    assert listing["scenes"][0]["objective"] == "Navigate to the kitchen."
    assert not store.images
    assert "PRIVATE_" not in json.dumps(listing)
    assert [row["id"] for row in store.catalogue(filters={"scope": "diagnostics"})["scenes"]] == ["category_cc"]
    detail = store.scene_detail("category_aa")
    assert len(detail["images"]) == 1 and detail["images"][0]["view"] == "global_overhead"
    assert "/obsolete/" not in json.dumps(detail)
    assert "PRIVATE_" not in json.dumps(detail)
    with pytest.raises(KeyError):
        store.scene_detail("category_bb")
    with pytest.raises(KeyError):
        store.scene_detail("../category_aa")


def test_capture_ready_filter_includes_rendered_pending_tasks(live):
    publisher, store, _, _ = live
    publisher.tick()
    ready = store.catalogue(filters={'scope': 'all', 'capture_status': 'ready'})
    assert [row['id'] for row in ready['scenes']] == ['category_aa']
    assert ready['scenes'][0]['conversion_status'] == 'pending'
    assert not ready['scenes'][0]['physical_grasp_verified']
    incomplete = store.catalogue(filters={'scope': 'all', 'capture_status': 'pending'})
    assert [row['id'] for row in incomplete['scenes']] == ['category_cc']
    assert store.catalogue(filters={'scope': 'all', 'conversion_status': 'ready'})['matched'] == 0


def test_incremental_completion_and_note_persistence(live):
    publisher, store, records, log = live
    publisher.tick()
    write_json(publisher.queue / "jobs/capture_bb/status.json", {"state": "REVIEW_PENDING",
        "receipt": str(publisher.queue / "jobs/capture_bb/attempts/01/preview_complete.json")})
    with log.open("a") as stream:
        stream.write("[capture-worker] worker0 capture_bb REVIEW_PENDING\n")
    publisher.tick()
    assert len(store.catalogue()["scenes"]) == 2
    row = store.scene_detail("category_aa")
    payload = {"decision": "needs_fix", "notes": "Move the object onto a support.",
               "fingerprint": row["fingerprint"], "revision": 0}
    assert store.save_review(row["id"], payload)["revision"] == 1
    assert store.scene_detail(row["id"])["review"]["notes"] == payload["notes"]
    assert store.catalogue()["categories"][0]["counts"]["needs_repair"] == 1
    with pytest.raises(ReviewConflict):
        store.save_review(row["id"], payload)
    approved = store.save_review(row["id"], {**payload, "revision": 1, "decision": "approved"})
    assert approved['approval_scope'] == 'layout' and approved['revision'] == 2
    detail = store.scene_detail(row['id'])
    assert detail['conversion_status'] == 'pending'
    assert not detail['physical_grasp_verified'] and detail['layout_approval_available']
    diagnostic = store.scene_detail('category_cc')
    saved = store.save_review(diagnostic['id'], {'decision': 'approved', 'notes': '', 'revision': 0,
        'fingerprint': diagnostic['fingerprint']})
    assert saved['approval_scope'] == 'layout'
    assert not saved['approval_evidence']['native_capture_complete']


def test_authorized_reset_removes_embedded_old_feedback_and_keeps_new_reviews(live):
    publisher, store, _, _ = live
    write_json(publisher.output / 'review_reset.json', {'clear_non_rigid_history': True})
    publisher.tick()
    row = store.scene_detail('category_aa')
    row['prior_feedback'] = [{'notes': 'Obsolete feedback.'}]
    row['composite'] = {'member_count': 2, 'public_goals': [
        {'candidate_id': 'old', 'previous_review': {'decision': 'approved', 'notes': 'Old approval.'}}]}
    annotated = store._annotate(row, {}, [])
    assert not annotated['review_history']
    assert annotated['source_review_summary']['reviewed_goals'] == 0
    assert 'previous_review' not in annotated['composite']['public_goals'][0]
    store.save_review(row['id'], {'decision': 'needs_fix', 'notes': 'Fresh review.',
                                'fingerprint': row['fingerprint'], 'revision': 0})
    assert store.scene_detail(row['id'])['review']['notes'] == 'Fresh review.'


def test_bulk_submission_failure_is_visible_without_native_images(live, monkeypatch):
    publisher, store, records, _ = live
    record = {**records[1], 'candidate_id': 'category_ee', 'job_id': None,
        'capture_submission_error': 'Immutable input could not be submitted.',
        'capture_submission_stage': 'capture_submission'}
    write_json(publisher.queue / 'category_records/category_ee.json', record)
    progress = publisher.campaign / 'bulk/submission_progress.json'
    write_json(progress, {'submission_failures': [{'candidate_id': 'category_ee'}]})
    write_json(publisher.queue / 'capture_sample_gate.json', {
        'state': 'RELEASED_AFTER_USER_CONFIRMATION', 'bulk_submission_progress': str(progress)})
    monkeypatch.setattr(Path, 'glob', forbid_scan)
    monkeypatch.setattr(Path, 'iterdir', forbid_scan)
    publisher.tick()
    diagnostic = store.scene_detail('category_ee')
    assert diagnostic['failure'] == record['capture_submission_error']
    assert diagnostic['instance_status'] == 'capture_submission_failed'
    assert not diagnostic['native_capture_complete'] and not diagnostic['images']


def test_goal_revision_reuses_capture_and_layout_review(live, monkeypatch):
    publisher, store, records, _ = live
    publisher.tick()
    original = store.scene_detail('category_aa')
    store.save_review('category_aa', {'decision': 'approved', 'notes': 'Layout accepted.',
        'fingerprint': original['fingerprint'], 'revision': 0})
    task = publisher.campaign / 'goal_revisions/task.yaml'
    task.parent.mkdir()
    task.write_text(yaml.safe_dump({'public': {
        'objective': 'Arrange the bottles in a straight line; stop if already aligned.',
        'success_criteria': 'Current released objects form a line.',
        'environment': 'Unchanged environment.', 'constraints': []},
        'private': {'answer': 'PRIVATE_ANSWER_CANARY'}}))
    write_json(publisher.output / 'goal_revisions.json', {'records': {
        'category_aa': {'task_file': str(task), 'goal_revision': 'goal_test'}}})
    # Reusing a capture must not inspect or replace its existing screenshot.
    monkeypatch.setattr(Path, 'glob', forbid_scan)
    monkeypatch.setattr(Path, 'rglob', forbid_scan)
    monkeypatch.setattr(Path, 'iterdir', forbid_scan)
    publisher.tick()
    row = store.catalogue()['scenes'][0]
    assert row['objective'].startswith('Arrange the bottles')
    assert row['images'] == [] and row['capture_reused']
    detail = store.scene_detail('category_aa')
    assert detail['task_file'] == str(task)
    assert detail['captured_task_file'] == records[0]['task_file']
    assert detail['capture']['job_id'] == original['capture']['job_id']
    assert detail['images'] == original['images']
    assert detail['fingerprint'] == original['fingerprint']
    assert detail['human_approved'] and not detail['review_stale']
    assert detail['conversion_status'] == 'pending'
    assert 'PRIVATE_' not in json.dumps(detail)


def test_retired_versions_leave_review_without_removing_current_failures_or_history(live):
    publisher, store, _, _ = live
    publisher.tick()
    original = store.scene_detail('category_aa')
    store.save_review('category_aa', {'decision': 'needs_fix', 'notes': 'Historical review.',
        'fingerprint': original['fingerprint'], 'revision': 0})
    write_json(publisher.output / 'retired_candidates.json', {'candidates': {
        'category_aa': {'reason': 'Superseded by a continuous two-phase task.'}}})
    publisher.tick()
    assert not store.catalogue()['scenes']
    assert [row['id'] for row in store.catalogue(filters={'scope': 'diagnostics'})['scenes']] == ['category_cc']
    assert not store.catalogue(filters={'id': 'category_aa'})['scenes']
    with pytest.raises(KeyError):
        store.scene_detail('category_aa')
    reviews = json.loads(store.review_file.read_text())
    assert reviews['entries']['category_aa']['notes'] == 'Historical review.'
    assert (publisher.queue / 'jobs/capture_aa/input/manifest.json').is_file()


def test_failed_attempt_metadata_remains_available_after_a_successful_retry(live):
    publisher, store, _, _ = live
    publisher.tick()
    path = publisher.queue / "jobs/capture_aa/status.json"
    status = json.loads(path.read_text())
    write_json(path, {**status, "attempts": 2})
    partial = publisher.queue / "jobs/capture_aa/attempts/01/head.png"
    write_json(publisher.queue / "jobs/capture_aa/attempts/01/capture_failure.json",
               {"partial_images": {"head": str(partial)}})
    images = store.scene_detail("category_aa")["images"]
    assert any(item["kind"] == "partial" and "attempt 1" in item["label"] for item in images)


def test_completed_screenshots_do_not_hide_rejected_native_placement(live):
    publisher, store, _, _ = live
    path = publisher.queue / "jobs/capture_aa/attempts/01/preview_complete.json"
    receipt = json.loads(path.read_text())
    write_json(path, {**receipt, "interaction_setup": {
        "accepted": False, "relocated": False, "reason": "No suitable existing support."}})
    publisher.tick()
    row = store.catalogue()["scenes"][0]
    assert row["native_capture_complete"] is True
    assert row["placement_state"] == "needs_repair"
    diagnostics = store.catalogue(filters={"scope": "diagnostics"})["scenes"]
    assert any(item["id"] == row["id"] for item in diagnostics)
    assert store.scene_detail(row["id"])["failure"] == "No suitable existing support."


def test_new_cpu_failure_remains_reviewable_before_held_capture_publication(live):
    publisher, store, records, _ = live
    publisher.tick()
    record = {**records[0], 'candidate_id': 'category_ff', 'fingerprint': 'new-current-failure',
        'status': 'conversion_failed', 'failure_stage': 'source_inspection',
        'failure_reason': 'Original geometry is unsupported.', 'task_file': None,
        'capture_input': None, 'job_id': None,
        'diagnostic_dir': str(publisher.campaign / 'new_failure')}
    write_json(Path(record['diagnostic_dir']) / 'record.json', record)
    batch = publisher.campaign / 'revised_batch.json'
    write_json(batch, {'records': [record]})
    write_json(publisher.output / 'category_revisions.json', {'categories': {
        'category_key': {'complete': True, 'new_batch_file': str(batch)}}})
    publisher.tick()
    assert not store.catalogue()['scenes']
    row = store.scene_detail('category_ff')
    assert row['failure'] == 'Original geometry is unsupported.'
    assert not row['images'] and row['layout_approval_available']


def test_http_list_does_not_deliver_images_before_selection(live):
    publisher, store, _, _ = live
    publisher.tick()
    server = create_server(store, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:" + str(server.server_port)
    try:
        listing = requests.get(base + "/api/catalogue?page=0", timeout=5).json()
        assert listing["scenes"][0]["images"] == [] and not store.images
        assert requests.get(base + "/images/category_aa/global_overhead", timeout=5).status_code == 404
        detail = requests.get(base + "/api/scene?id=category_aa", timeout=5).json()["scene"]
        response = requests.get(base + "/" + detail["images"][0]["url"], timeout=5)
        assert response.status_code == 200
        assert response.content == b"Screenshot bytes are only read when requested."
        assert response.headers["Cache-Control"] == "no-store"
        assert requests.get(base + "/api/scene?id=category_bb", timeout=5).status_code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
