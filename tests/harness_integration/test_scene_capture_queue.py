"""Check real file claiming, frozen inputs, cache completeness and review routing.

Run: .venv/bin/python -m pytest -q tests/test_scene_capture_queue.py
All images here are CPU fixtures, not native capture or physical evidence.
"""
import json
import multiprocessing as mp
from pathlib import Path
import subprocess

from PIL import Image
import pytest
import requests
import threading
from types import SimpleNamespace
import yaml

from wia.runtime.scene_capture_candidates import ROOT, export_feedback, submit_candidate
from wia.runtime.scene_capture_queue import CaptureQueue, PROTOCOL, read_json, verify_snapshot, write_json
from wia.runtime.scene_review import SceneReview
from wia.runtime.commands.review_esi_scenes import create_server


def run_git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL).decode().strip()


@pytest.fixture
def candidates(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    esi = repo / "ESI"
    esi.mkdir()
    for directory in (repo, esi):
        run_git(directory, "init", "-q")
        run_git(directory, "config", "user.email", "fixture@example.invalid")
        run_git(directory, "config", "user.name", "CPU fixture")
    for relative in ("configs/esi_trunk.json", "configs/esi_robot_cameras.json", "configs/esi_viewer.json"):
        path = repo / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    (repo / "src/wia/runtime/sim").mkdir(parents=True)
    (repo / "src/wia/runtime/sim/fixture.py").write_text('"""CPU signature fixture; never executed."""\n')
    handoffs = []
    for index in range(2):
        source = f"Action Sequencing/Action Order Inference/Beechwood_0_garden/dining_room_0/sample_{index}.json"
        path = esi / "dataset/json_clean" / source
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"big_task": "Action Sequencing", "small_task": "Action Order Inference",
            "scene": "Beechwood_0_garden", "room": "dining_room_0", "question": "Which object is visible?",
            "answer": "PRIVATE_ANSWER_CANARY", "metadata_json": '{"_ground_truth":"PRIVATE_METADATA_CANARY"}'})
        scene_file = f"candidates/scene_{index}.yaml"
        path = repo / scene_file
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump({"kind": "scene_patch", "id": f"fixture_scene_{index}", "status": "pending",
            "source": {"provider": "esi", "json": source}, "scene": {
                "policy": "native_then_minimal_patch_else_skip", "variants": {"original": {"patches": []}}}, "seed": 0}))
        handoffs.append({"candidate_id": f"fixture_{index}", "source_json": "ESI/dataset/json_clean/" + source,
            "scene_file": scene_file, "task_file": None, "physical_task_id": None, "category": "Action Sequencing",
            "subcategory": "Action Order Inference", "scene_model": "Beechwood_0_garden", "room": "dining_room_0",
            "variant": "original", "seed": 0, "reuse_level": "original", "modified_files": [scene_file],
            "expected_views": ["global_diagonal", "head", "wrist", "scene"], "known_limitations": ["CPU fixture only."],
            "owner": f"builder_{index}", "parent_candidate_id": None, "render_config": {}})
    run_git(esi, "add", ".")
    run_git(esi, "commit", "-qm", "Original fixture source")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-qm", "Capture fixture")
    commit = run_git(repo, "rev-parse", "HEAD")
    for handoff in handoffs:
        handoff["code_commit"] = commit
    return repo, handoffs, tmp_path / "queue"


def submit_all(candidates):
    repo, handoffs, queue_root = candidates
    jobs = [submit_candidate(h, queue_root, repo=repo) for h in handoffs]
    return CaptureQueue(queue_root), jobs


def test_category_sample_gate_holds_other_tasks_until_explicit_release(candidates):
    queue, jobs = submit_all(candidates)
    selected, held_job = (job['job_id'] for job in jobs)
    selected_candidate = queue.manifest(selected)['candidate_id']
    write_json(queue.root / 'capture_sample_gate.json', {
        'state': 'ONE_PER_CATEGORY_AWAITING_REVIEW', 'selected_candidate_ids': [selected_candidate]})
    assert queue.claim({'id': 'other'}, job_id=held_job) is None
    (queue.root / 'queued' / (held_job + '.json')).unlink()
    queue.enqueue(queue.manifest(held_job))
    held = queue.root / 'held_for_category_review' / (held_job + '.json')
    assert held.is_file() and not (queue.root / 'queued' / held.name).exists()
    claim = queue.claim({'id': 'sample'})
    assert claim['job_id'] == selected
    assert queue.claim({'id': 'waiting'}) is None
    assert queue.status(held_job)['state'] == 'QUEUED'
    write_json(queue.root / 'capture_sample_gate.json', {'state': 'RELEASED_AFTER_USER_CONFIRMATION'})
    held.rename(queue.root / 'queued' / held.name)
    assert queue.claim({'id': 'resumed'})['job_id'] == held_job


def claim_in_process(root, worker_id, results):
    queue = CaptureQueue(root)
    claim = queue.claim({"id": worker_id})
    results.put(claim["job_id"] if claim else None)


def test_paused_queue_limits_pilot_to_exact_worker_gpu_and_explicit_job(candidates):
    queue, jobs = submit_all(candidates)
    selected, other = [item['job_id'] for item in jobs]
    worker = {'id': 'worker0', 'uuid': 'GPU-CPU-fixture'}
    write_json(queue.root / 'capture_pause.json', {'state': 'PAUSED_BY_USER',
        'authorized_pilot': {'state': 'RUNNING', 'worker_id': worker['id'],
            'gpu_uuid': worker['uuid'], 'job_ids': [selected]}})
    assert queue.claim(worker) is None
    assert queue.claim(worker, job_id=other) is None
    assert queue.claim({**worker, 'id': 'worker1'}, job_id=selected) is None
    assert queue.claim({**worker, 'uuid': 'GPU-other'}, job_id=selected) is None
    assert len(list((queue.root / 'claimed').glob('*.json'))) == 0
    claim = queue.claim(worker, job_id=selected)
    assert claim['job_id'] == selected
    assert queue.status(other)['state'] == 'QUEUED'


def test_runtime_policy_keeps_historical_jobs_intact_and_claims_only_current_code(candidates):
    queue, jobs = submit_all(candidates)
    selected = jobs[0]['job_id']
    commit = queue.manifest(selected)['code_commit']
    write_json(queue.root / 'capture_runtime_policy.json', {'allowed_code_commits': ['different-runtime']})
    assert queue.claim({'id': 'worker0'}) is None
    assert all(queue.status(job['job_id'])['state'] == 'QUEUED' for job in jobs)
    write_json(queue.root / 'capture_runtime_policy.json', {'allowed_code_commits': [commit]})
    assert queue.claim({'id': 'worker0'}, job_id=selected)['job_id'] == selected


@pytest.mark.parametrize('claimed', [False, True])
def test_cancel_with_durable_job_symlink_never_renames_a_queue_token_across_filesystems(candidates, monkeypatch, claimed):
    queue, jobs = submit_all(candidates)
    job = jobs[0]['job_id']
    if claimed:
        queue.claim({'id': 'worker0'}, job_id=job)
    original = Path.rename
    operations = []

    def filesystem_fence(path, target):
        target = Path(target)
        operations.append((path, target))
        if path.parent == queue.root / 'queued' and target.is_relative_to(queue.job_dir(job)):
            raise OSError(18, 'Synthetic durable home/local queue boundary')
        return original(path, target)

    monkeypatch.setattr(Path, 'rename', filesystem_fence)
    result = queue.cancel(job)
    assert (queue.job_dir(job) / 'cancel.json').is_file()
    if claimed:
        assert result['state'] == 'CLAIMED' and not operations
    else:
        assert result['state'] == 'CANCELLED'
        assert read_json(queue.job_dir(job) / 'cancelled_token.json')['job_id'] == job
        assert not (queue.root / 'queued' / (job + '.json')).exists()


def receipt_for(queue, claim):
    manifest = queue.manifest(claim["job_id"])
    output = queue.job_dir(claim["job_id"]) / "attempts/01"
    output.mkdir(parents=True)
    images = {}
    for view in manifest["expected_views"]:
        path = output / f"{view}.png"
        Image.new("RGB", (64, 64), "blue").save(path)
        images[view] = str(path)
    path = output / "preview_complete.json"
    write_json(path, {"capture_signature": manifest["capture_signature"], "fingerprint": manifest["fingerprint"],
        "capture_protocol": PROTOCOL, "variant": manifest["variant"], "seed": manifest["seed"],
        "complete": True, "images": images, "diagnostic_pending": True, "eligible_final_layout": False,
        "native_qualification_run": False, "cpu_fixture_only": True, "code_commit": manifest["code_commit"],
        "esi_commit": manifest["esi_commit"], "provenance": "CPU fixture only; no simulator was run."})
    return path


def test_two_processes_claim_distinct_jobs_and_duplicate_submission(candidates):
    queue, jobs = submit_all(candidates)
    duplicate = submit_candidate(candidates[1][0], queue.root, repo=candidates[0])
    assert duplicate["job_id"] == jobs[0]["job_id"] and len(queue.jobs()) == 2
    context = mp.get_context("fork")
    results = context.Queue()
    processes = [context.Process(target=claim_in_process, args=(queue.root, f"worker{i}", results)) for i in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert {results.get(timeout=2) for _ in processes} == {job["job_id"] for job in jobs}
    assert queue.claim({"id": "third"}) is None


def test_explicit_gpu_job_claim_leaves_other_candidates_available(candidates):
    queue, jobs = submit_all(candidates)
    selected = queue.claim({"id": "assigned_gpu"}, job_id=jobs[1]["job_id"])
    assert selected["job_id"] == jobs[1]["job_id"]
    assert queue.claim({"id": "same_mapping"}, job_id=jobs[1]["job_id"]) is None
    other = queue.claim({"id": "shared_queue"})
    assert other["job_id"] == jobs[0]["job_id"]


def test_frozen_source_is_independent_and_tampering_is_stale(candidates):
    queue, jobs = submit_all(candidates)
    manifest = queue.manifest(jobs[0]["job_id"])
    (candidates[0] / candidates[1][0]["source_json"]).write_text("Mutable source changed after acceptance.")
    verify_snapshot(manifest)
    Path(manifest["snapshot_source_file"]).write_text("Snapshot accidentally changed.")
    with pytest.raises(ValueError, match="Immutable capture input changed"):
        verify_snapshot(manifest)
    path = queue.job_dir(jobs[1]["job_id"]) / "input/manifest.json"
    value = read_json(path)
    value["seed"] = 99
    write_json(path, value)
    with pytest.raises(ValueError, match="manifest changed"):
        queue.manifest(jobs[1]["job_id"])


def test_cache_requires_complete_matching_readable_images(candidates):
    queue, jobs = submit_all(candidates)
    claim = queue.claim({"id": "worker0"})
    queue.transition(claim, "RUNNING")
    receipt = receipt_for(queue, claim)
    queue.finish(claim, receipt)
    manifest = queue.manifest(claim["job_id"])
    assert queue.cached(manifest)
    handoff = next(h for h in candidates[1] if h["candidate_id"] == manifest["candidate_id"])
    repeated = submit_candidate(handoff, queue.root, repo=candidates[0])
    assert repeated["submission_state"] == "CACHED"
    assert queue.status(claim["job_id"])["attempts"] == 1
    assert len(queue.jobs()) == len(jobs)
    Path(read_json(receipt)["images"]["head"]).unlink()
    assert queue.cached(manifest) is None
    invalidated = submit_candidate(handoff, queue.root, repo=candidates[0])
    assert invalidated["state"] == "QUEUED"


def test_successful_retry_reports_current_completion_and_preserves_failed_attempt(candidates):
    queue, jobs = submit_all(candidates)
    claim = queue.claim({"id": "worker0"})
    queue.transition(claim, "RUNNING")
    queue.transition(claim, "FAILED", error="Earlier asset-loading failure",
                     failure_stage="cold_scene_loading", complete=False, retryable=True)
    queue.release(claim)
    queue.retry(claim["job_id"])
    retry = queue.claim({"id": "worker0"}, job_id=claim["job_id"])
    queue.transition(retry, "RUNNING")
    receipt = receipt_for(queue, retry)
    queue.finish(retry, receipt)
    current = queue.status(retry["job_id"])
    assert current["state"] == "REVIEW_PENDING" and current["complete"] is True
    assert current["error"] is None and current["failure_stage"] is None
    assert current["retryable"] is False
    assert current["native_log"] == str(receipt.parent / "native.log")
    history = [read_json(p) for p in (queue.job_dir(retry["job_id"]) / "events").glob("*.json")]
    assert any(row.get("error") == "Earlier asset-loading failure" for row in history)


def test_failure_and_bounded_recovery_do_not_disrupt_other_job(candidates):
    queue, jobs = submit_all(candidates)
    first = queue.claim({"id": "worker0"}, lease_s=-1)
    second = queue.claim({"id": "worker1"})
    queue.transition(first, "RUNNING")
    queue.transition(second, "RUNNING")
    assert queue.recover(lambda claim: False) == []
    assert queue.recover(lambda claim: claim["token"] == first["token"]) == [first["job_id"]]
    assert queue.status(first["job_id"])["state"] == "QUEUED"
    assert queue.status(second["job_id"])["state"] == "RUNNING"
    retry = queue.claim({"id": "replacement"})
    queue.transition(retry, "FAILED", retryable=True)
    queue.release(retry)
    with pytest.raises(ValueError, match="budget is exhausted"):
        queue.retry(retry["job_id"])
    queue.transition(second, "FAILED", retryable=False)
    queue.release(second)
    assert {row["state"] for row in queue.jobs()} == {"FAILED"}


def test_camera_or_renderer_changes_invalidate_capture_but_not_layout_fingerprint(candidates):
    queue, jobs = submit_all(candidates)
    before = queue.manifest(jobs[0]["job_id"])
    repo = candidates[0]
    path = repo / "configs/esi_viewer.json"
    value = json.loads(path.read_text())
    value["focal_length"] = 16.
    write_json(path, value)
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-qm", "Change human camera framing")
    after_job = submit_candidate(candidates[1][0], queue.root, repo=repo, code_commit="HEAD")
    after = queue.manifest(after_job["job_id"])
    assert before["fingerprint"] == after["fingerprint"]
    assert before["capture_signature"] != after["capture_signature"]


def test_gallery_discovers_completion_and_routes_repairs_without_restart(candidates):
    queue, jobs = submit_all(candidates)
    store = SceneReview(None, queue.root.parent / "reviews", queue_root=queue.root)
    server = create_server(store, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        before = requests.get(url + "/api/scenes", timeout=5).json()
        assert len(before["scenes"]) == 2 and not any(row["images"] for row in before["scenes"])
        assert "PRIVATE_" not in json.dumps(before)
        claim = queue.claim({"id": "worker0"})
        queue.transition(claim, "RUNNING")
        queue.finish(claim, receipt_for(queue, claim))
        rows = requests.get(url + "/api/scenes", timeout=5).json()["scenes"]
        row = next(row for row in rows if row["capture"]["job_id"] == claim["job_id"])
        assert len(row["images"]) == 4 and row["has_global_preview"] and not row["has_final_preview"]
        assert not row["physical_grasp_verified"] and row["conversion_status"] == "pending"
        payload = {"decision": "approved", "notes": "", "fingerprint": row["fingerprint"], "revision": 0}
        approved = requests.post(url + f"/api/reviews/{row['id']}", json=payload, timeout=5)
        assert approved.status_code == 200
        approved_row = next(r for r in store.catalogue()['scenes'] if r['id'] == row['id'])
        assert approved_row['review']['approval_scope'] == 'layout'
        assert approved_row['human_approved'] and not approved_row['physical_qualified']
        payload.update(decision="needs_fix", revision=1, notes="CPU review-flow fixture: inspect camera framing.")
        saved = requests.post(url + f"/api/reviews/{row['id']}", json=payload, timeout=5)
        assert saved.status_code == 200
        assert requests.post(url + f"/api/reviews/{row['id']}", json=payload, timeout=5).status_code == 409
        feedback = export_feedback(queue.root, store.review_file)
        assert feedback[0]["owner"] == row["owner"] and feedback[0]["notes"] == payload["notes"]
        payload.update(decision="skipped", revision=2)
        assert requests.post(url + f"/api/reviews/{row['id']}", json=payload, timeout=5).status_code == 200
        history = store.catalogue()["scenes"]
        assert next(r for r in history if r["id"] == row["id"])["review_history"][-1]["notes"] == payload["notes"]
        # A revised candidate is independent and carries historical feedback.
        original = next(h for h in candidates[1] if h["candidate_id"] == row["id"])
        revised = {**original, "candidate_id": row["id"] + "_revision", "parent_candidate_id": row["id"],
                   "scene_file": "candidates/revised_scene.yaml", "modified_files": ["candidates/revised_scene.yaml"]}
        definition = yaml.safe_load((candidates[0] / original["scene_file"]).read_text())
        definition.update(id="fixture_revision", version=2)
        (candidates[0] / revised["scene_file"]).write_text(yaml.safe_dump(definition))
        run_git(candidates[0], "add", ".")
        run_git(candidates[0], "commit", "-qm", "Bounded CPU repair-flow revision")
        revised["code_commit"] = run_git(candidates[0], "rev-parse", "HEAD")
        submitted = submit_candidate(revised, queue.root, repo=candidates[0])
        current = next(r for r in store.catalogue()["scenes"] if r["id"] == revised["candidate_id"])
        assert current["capture"]["state"] == "QUEUED" and current["decision"] == "unreviewed"
        assert current["review_history"][-1]["notes"] == payload["notes"]
        while (next_claim := queue.claim({"id": "repair_fixture"})) is not None:
            queue.transition(next_claim, "RUNNING")
            queue.finish(next_claim, receipt_for(queue, next_claim))
        current = next(r for r in store.catalogue()["scenes"] if r["id"] == revised["candidate_id"])
        assert current["has_global_preview"] and current["capture"]["job_id"] == submitted["job_id"]
        previous_row = next(r for r in store.catalogue()["scenes"] if r["id"] == row["id"])
        assert previous_row["decision"] == "skipped" and previous_row["review"]["revision"] == 3
        image = requests.get(url + "/" + row["images"][0]["url"], timeout=5)
        assert image.status_code == 200 and image.content.startswith(b"\x89PNG")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_native_robot_free_diagnostics_preserve_existing_loader_default(tmp_path):
    from wia.runtime.sim.esi_omnigibson import ESISession
    renders = []
    module = SimpleNamespace(scene_room=lambda payload: ("native_scene", "native_room"),
        initial_camera=lambda payload: ([0., 0., 1.], [0., 0., 0., 1.], {}),
        build_env_objects=lambda payload: [], __name__="fixture.robot_free")
    pipeline = SimpleNamespace(select_question=lambda *args: SimpleNamespace(source_path=tmp_path / "source.json", payload={}),
        load_task_module_for_payload=lambda *args: module, task_preprocess=lambda *args: {},
        task_build_env_config=lambda *args, **kwargs: {"robots": []},
        task_initial_settle_steps=lambda *args: 0, task_postprocess_env=lambda *args: {},
        og=SimpleNamespace(Environment=lambda configs: SimpleNamespace(robots=[], scene=SimpleNamespace(scene_model="native_scene")),
                           sim=SimpleNamespace(render=lambda: renders.append(True))))
    diagnostic = ESISession(tmp_path / "source.json", camera_size=(64, 64), require_robot=False)
    diagnostic.pipeline = pipeline
    diagnostic._initialize_scene(pipeline)
    assert diagnostic.env.robots == [] and diagnostic.robot is None and len(renders) == 3
    original_entrypoint = ESISession(tmp_path / "source.json")
    with pytest.raises(RuntimeError, match="environment has no robot"):
        original_entrypoint._initialize_scene(pipeline)


def test_corrupted_manifest_fails_one_job_without_losing_other_claim(candidates):
    from wia.runtime.commands.run_esi_capture_worker import capture_job
    queue, _ = submit_all(candidates)
    first, second = queue.claim({"id": "worker0"}), queue.claim({"id": "worker1"})
    write_json(queue.job_dir(first["job_id"]) / "input/manifest.json", {"tampered": True})
    capture_job(queue, first, {"uuid": "CPU-fixture-only"})
    assert queue.status(first["job_id"])["state"] == "STALE"
    assert queue.status(second["job_id"])["state"] == "CLAIMED"
    queue.transition(second, "FAILED", retryable=False)
    queue.release(second)
