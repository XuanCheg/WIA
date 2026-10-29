"""Validate builder handoffs and freeze committed capture inputs.

Run: python src/wia/runtime/commands/submit_esi_capture.py --queue-root QUEUE submit --handoff FILE
This module uses existing scene/task contracts and imports no simulator.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import io
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import time
import uuid

import yaml

from wia.runtime.scene_capture_queue import (CaptureQueue, PROTOCOL, digest, publish_json,
                                      read_json, sha256, verify_snapshot, write_json)
from wia.runtime.scene_review import ALL_VIEW_LABELS, fingerprint
from wia.tasks.loader import UniqueLoader, task_definition
from wia.tasks.schema import SceneDefinition

ROOT = WIA_RUNTIME_ROOT
FIELDS = {"candidate_id", "source_json", "scene_file", "task_file", "category", "subcategory",
          "scene_model", "room", "variant", "seed", "reuse_level", "modified_files",
          "expected_views", "known_limitations", "code_commit"}


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


def relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not value:
        raise ValueError("Candidate files must have repository-relative paths.")
    return str(path)


def committed_file(repo, commit, path):
    return git(repo, "show", f"{commit}:{relative(path)}")


def archive_revision(repo, commit, destination):
    """Materialize each committed code tree once; publish only complete trees."""
    destination = Path(destination)
    if (destination / ".capture_revision.json").is_file():
        return destination
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(git(repo, "archive", commit))) as archive:
            # Git paths are trusted repository entries, still reject escape links.
            for member in archive.getmembers():
                path = (temporary / member.name).resolve()
                if not path.is_relative_to(temporary.resolve()) or member.islnk():
                    raise ValueError("Unsafe path in committed code archive.")
                if member.issym() and (PurePosixPath(member.linkname).is_absolute()
                                      or ".." in PurePosixPath(member.linkname).parts):
                    raise ValueError("External symlink in capture implementation archive.")
            archive.extractall(temporary)
        write_json(temporary / ".capture_revision.json", {"commit": commit})
        try:
            temporary.rename(destination)
        except OSError:
            if not (destination / ".capture_revision.json").is_file():
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def rendering_hashes(code):
    """Invalidate renderer changes while preserving unrelated layout approvals."""
    paths = []
    for directory in ("src/wia", "ESI/src", "harness/core", "harness/showharness", "harness/gumi"):
        paths.extend((code / directory).rglob("*.py"))
    from wia.runtime.resources import runtime_source
    from showharness.resources import ROOT as harness_root
    paths.extend(runtime_source(name, code) for name in (
        "src/wia/runtime/scene_capture_adapters.py", "core/viewer_config.py", "core/observation_config.py",
        "core/sensor_observations.py", "core/observation.py", "src/wia/runtime/commands/capture_esi_scene_reviews.py",
        "src/wia/runtime/commands/verify_esi_rigid_grasp.py", "src/wia/runtime/commands/run_esi_robot.py"))
    omitted = {"scene_capture_queue.py", "scene_capture_candidates.py", "scene_review.py"}
    return {str(p.relative_to(code)) if p.is_relative_to(code) else "harness/" + str(p.relative_to(harness_root)): sha256(p) for p in sorted(set(paths))
            if p.is_file() and p.name not in omitted}


def validate_handoff(handoff, *, repo=ROOT, code_commit=None):
    repo = Path(repo).resolve()
    if FIELDS - handoff.keys():
        raise ValueError("Handoff is missing fields: " + ", ".join(sorted(FIELDS - handoff.keys())))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", handoff["candidate_id"]):
        raise ValueError("Candidate identity must be a stable filename-safe identifier.")
    if handoff["reuse_level"] not in {"original", "lightweight_adaptation", "new_asset"}:
        raise ValueError("Unknown reuse level.")
    if type(handoff["seed"]) is not int or handoff["seed"] < 0:
        raise ValueError("Candidate seed must be a nonnegative integer.")
    views = handoff["expected_views"]
    if (not isinstance(views, list) or not views or len(set(views)) != len(views)
            or set(views) - ALL_VIEW_LABELS.keys()):
        raise ValueError("Request distinct existing review camera views.")
    if not isinstance(handoff["known_limitations"], list) or not isinstance(handoff["modified_files"], list):
        raise ValueError("Limitations and modified files must be lists.")
    for value in handoff["modified_files"]:
        relative(value)
    commit = git(repo, "rev-parse", "--verify", (code_commit or handoff["code_commit"]) + "^{commit}").decode().strip()
    builder_commit = git(repo, "rev-parse", "--verify", handoff["code_commit"] + "^{commit}").decode().strip()
    scene_relative = relative(handoff["scene_file"])
    scene_bytes = committed_file(repo, commit, scene_relative)
    if scene_bytes != (repo / scene_relative).read_bytes():
        raise ValueError("Scene edits are not committed at the accepted capture revision.")
    if scene_bytes != committed_file(repo, builder_commit, scene_relative):
        raise ValueError("Builder handoff does not identify the submitted scene revision.")
    definition = SceneDefinition.model_validate(yaml.load(scene_bytes, Loader=UniqueLoader))
    if definition.qualification is None and definition.status != "pending":
        raise ValueError("A diagnostic scene without a native qualification contract must remain pending.")
    if definition.status == "discarded" or handoff["variant"] not in definition.scene.variants:
        raise ValueError("Candidate is discarded or requests an undeclared variant.")
    source_relative = relative(handoff["source_json"])
    source_path = repo / source_relative
    source_bytes = source_path.read_bytes()
    raw = json.loads(source_bytes)
    source_parts = PurePosixPath(definition.source.json_path).parts
    if not source_relative.endswith(definition.source.json_path):
        raise ValueError("Handoff and scene refer to different ESI source files.")
    if (raw.get("scene"), raw.get("room"), raw.get("big_task"), raw.get("small_task")) != (
            handoff["scene_model"], handoff["room"], handoff["category"], handoff["subcategory"]):
        raise ValueError("Candidate category, scene or room differs from the original source.")
    if tuple(source_parts[-3:-1]) != (handoff["scene_model"], handoff["room"]):
        raise ValueError("Scene source hierarchy is inconsistent.")
    esi_commit = git(repo, "ls-tree", commit, "ESI").decode().split()[2]
    if git(repo / "ESI", "status", "--porcelain").strip():
        raise ValueError("Original ESI submodule has uncommitted modifications.")
    original = committed_file(repo / "ESI", esi_commit, "dataset/json_clean/" + definition.source.json_path)
    if source_bytes != original:
        raise ValueError("Original ESI question differs from its committed source; preserve it verbatim.")
    patches = definition.scene.variants[handoff["variant"]].patches
    if handoff["reuse_level"] == "original" and any(p.operation != "robot_pose" for p in patches):
        raise ValueError("Original reuse cannot alter source objects or geometry.")
    if handoff["reuse_level"] != "new_asset" and any(p.operation == "add_usd_object" for p in patches):
        raise ValueError("New USD assets require an explicit new_asset declaration.")
    if any(view.startswith("global_target") for view in views) and definition.target is None:
        raise ValueError("Target closeups require an existing explicit physical binding.")
    task_bytes = None
    physical_id = None
    if handoff["task_file"]:
        task_relative = relative(handoff["task_file"])
        task_bytes = committed_file(repo, commit, task_relative)
        if task_bytes != (repo / task_relative).read_bytes():
            raise ValueError("Task edits must be committed before submission.")
        task = task_definition(yaml.load(task_bytes, Loader=UniqueLoader))
        if task.source.json_path != definition.source.json_path:
            raise ValueError("Task and scene source references disagree.")
        if hasattr(task, "scene_file"):
            task_scene = (repo / "explore_to_achieve/scenes" / task.scene_file).resolve()
            if task_scene != (repo / scene_relative).resolve():
                raise ValueError("Task references a different scene overlay.")
        public = task.public.model_dump(mode="json")
        private_keys = {"answer", "_ground_truth", "ground_truth", "metadata_json", "qa_per_view"}
        if private_keys & public.keys():
            raise ValueError("Private source truth cannot enter the public task contract.")
        physical_id = task.id
    if handoff.get("physical_task_id") != physical_id:
        raise ValueError("Physical task identity must match its optional task definition.")
    render = handoff.get("render_config", {})
    if render:
        raise ValueError("M1 uses scene.runtime camera configs; change and commit those files to alter rendering.")
    return {"commit": commit, "builder_commit": builder_commit, "esi_commit": esi_commit,
            "definition": definition, "source_bytes": source_bytes, "scene_bytes": scene_bytes,
            "task_bytes": task_bytes, "physical_task_id": physical_id}


def submit_candidate(handoff, queue_root, *, repo=ROOT, code_commit=None, timeout_s=1800,
                     max_attempts=2, acceptance_note="Accepted by Main after static review."):
    if timeout_s <= 0 or max_attempts not in (1, 2, 3):
        raise ValueError("Use a positive timeout and one to three bounded attempts.")
    accepted = validate_handoff(handoff, repo=repo, code_commit=code_commit)
    definition = accepted["definition"]
    queue = CaptureQueue(queue_root)
    revisions = queue.root / "revisions"
    from wia.runtime.resources import harness_runtime_sources
    harness_sources = harness_runtime_sources()
    harness_hashes = {name: sha256(path) for name, path in harness_sources.items()}
    revision_key = accepted['commit'] + '_' + digest(harness_hashes)[:16]
    code = archive_revision(repo, accepted['commit'], revisions / revision_key)
    if not (code / 'harness').is_dir():
        temporary_harness = code / ('.harness_' + uuid.uuid4().hex)
        temporary_harness.mkdir()
        try:
            for name, source in harness_sources.items():
                destination = temporary_harness / Path(name).relative_to('harness')
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
                if sha256(destination) != harness_hashes[name]:
                    raise RuntimeError('Harness source changed during capture freezing: ' + name)
            try:
                temporary_harness.rename(code / 'harness')
            except OSError:
                if not (code / 'harness').is_dir():
                    raise
        finally:
            if temporary_harness.exists():
                shutil.rmtree(temporary_harness)
    esi = archive_revision(Path(repo) / "ESI", accepted["esi_commit"],
                           revisions / ("esi_" + accepted["esi_commit"]))
    submodule = code / "ESI"
    if submodule.is_dir() and not submodule.is_symlink():
        submodule.rmdir()
    if not submodule.exists():
        submodule.symlink_to(esi, target_is_directory=True)
    implementation = rendering_hashes(code)
    runtime_paths = [definition.runtime.trunk_config, definition.runtime.robot_camera_config,
                     definition.runtime.viewer_config]
    for patch in definition.scene.variants[handoff["variant"]].patches:
        if patch.operation == "add_usd_object":
            runtime_paths.append(patch.asset)
    runtime_hashes = {}
    for value in runtime_paths:
        relative(value)
        path = code / value
        if not path.is_file() or path.read_bytes() != (Path(repo) / value).read_bytes():
            raise ValueError("Referenced asset/runtime config is missing or uncommitted: " + value)
        runtime_hashes[value] = sha256(path)
        if value.endswith(".json"):
            json.loads(path.read_text())
    implementation.update(runtime_hashes)
    layout_fingerprint = fingerprint(Path(repo) / handoff["scene_file"], Path(repo) / handoff["source_json"],
                                     handoff["variant"], handoff["seed"])
    signature = digest({"fingerprint": layout_fingerprint, "expected_views": sorted(handoff["expected_views"]),
        "runtime": definition.runtime.model_dump(mode="json"), "implementation_hashes": implementation,
        "task_sha256": hashlib.sha256(accepted["task_bytes"] or b"").hexdigest(),
        "protocol": PROTOCOL, "output_format": "PNG"})
    job_id = "capture_" + signature[:32]
    job = queue.job_dir(job_id)
    temporary = queue.root / "jobs" / (".building_" + uuid.uuid4().hex)
    snapshot = temporary / "input"
    snapshot.mkdir(parents=True)
    source_relative = "data/" + definition.source.json_path
    (snapshot / source_relative).parent.mkdir(parents=True)
    (snapshot / source_relative).write_bytes(accepted["source_bytes"])
    (snapshot / "scene.yaml").write_bytes(accepted["scene_bytes"])
    if accepted["task_bytes"] is not None:
        (snapshot / "task.yaml").write_bytes(accepted["task_bytes"])
    files = {str(p.relative_to(snapshot)): sha256(p) for p in snapshot.rglob("*") if p.is_file()}
    manifest = {**handoff, "schema": "esi_capture_candidate_v1", "job_id": job_id,
        "code_commit": accepted["commit"], "builder_commit": accepted["builder_commit"],
        "esi_commit": accepted["esi_commit"], "physical_task_id": accepted["physical_task_id"],
        "fingerprint": layout_fingerprint, "capture_signature": signature, "capture_protocol": PROTOCOL,
        "snapshot_dir": str(job / "input"), "code_root": str(code),
        "snapshot_scene_file": str(job / "input/scene.yaml"), "snapshot_data_root": str(job / "input/data"),
        "snapshot_source_file": str(job / "input" / source_relative),
        "input_hashes": files, "implementation_hashes": implementation,
        "timeout_s": float(timeout_s), "max_attempts": max_attempts,
        "conversion_status": definition.status, "diagnostic_pending": definition.status == "pending" or definition.qualification is None,
        "accepted_at": time.time(), "acceptance_note": acceptance_note,
        "question": json.loads(accepted["source_bytes"])["question"],
        "scene_identity": definition.id, "source_identity": definition.source.json_path}
    write_json(snapshot / "manifest.json", manifest)
    write_json(temporary / "accepted.json", {"manifest_sha256": sha256(snapshot / "manifest.json"),
        "code_commit": accepted["commit"], "esi_commit": accepted["esi_commit"],
        "candidate_id": handoff["candidate_id"], "capture_signature": signature})
    try:
        temporary.rename(job)
    except OSError:
        if not (job / "input/manifest.json").is_file():
            raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    manifest = queue.manifest(job_id)
    verify_snapshot(manifest)
    identity = {k: handoff[k] for k in ("source_json", "scene_file", "task_file", "variant", "seed")}
    identity["fingerprint"] = layout_fingerprint
    candidate = queue.root / "candidates" / handoff["candidate_id"]
    candidate.mkdir(exist_ok=True)
    publish_json(candidate / "identity.json", identity)
    if read_json(candidate / "identity.json") != identity:
        raise ValueError("Candidate identity already names a different layout; submit a new candidate_id.")
    write_json(candidate / "submissions" / f"{signature}.json", {**manifest,
        "candidate_id": handoff["candidate_id"], "owner": handoff.get("owner"),
        "parent_candidate_id": handoff.get("parent_candidate_id"), "modified_files": handoff["modified_files"],
        "known_limitations": handoff["known_limitations"], "requested_code_commit": accepted["commit"],
        "submitted_at": time.time()})
    return queue.enqueue(manifest)


def export_feedback(queue_root, review_file):
    """Persist bounded repair handoffs; actual AI delegation remains with Codex."""
    queue = CaptureQueue(queue_root)
    records = read_json(review_file, {"entries": {}})["entries"]
    requests = []
    for candidate in (queue.root / "candidates").iterdir():
        submissions = [read_json(p) for p in (candidate / "submissions").glob("*.json")]
        if not submissions:
            continue
        latest = max(submissions, key=lambda s: s["submitted_at"])
        review = records.get(candidate.name, {})
        if review.get("fingerprint") != latest["fingerprint"] or review.get("decision") != "needs_fix":
            continue
        request = {"schema": "esi_scene_repair_request_v1", "state": "REPAIR_REQUESTED",
            "candidate_id": candidate.name, "owner": latest.get("owner"),
            "parent_job_id": latest["job_id"], "code_commit": latest["code_commit"],
            "owned_files": latest["modified_files"], "source_json": latest["source_json"],
            "notes": review.get("notes", ""), "review_revision": review["revision"],
            "fingerprint": latest["fingerprint"], "requirements": [
                "Preserve original ESI semantics and source data.",
                "Edit only assigned owned files; return a new candidate_id and parent_candidate_id.",
                "Main reviews, integrates and submits; the builder does not enqueue or render."],
            "validation": "Run the candidate CPU validator and git diff --check."}
        path = queue.root / "feedback" / f"{candidate.name}_{review['revision']}.json"
        publish_json(path, request)
        requests.append({**request, "request_file": str(path)})
    return requests
