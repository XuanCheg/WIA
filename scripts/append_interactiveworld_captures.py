"""Append 19 pending candidate captures to an existing queue without starting workers.

Run from the WIA checkout in the capture environment, after review:
  PYTHONPATH=src:../Show-Harness-Explore python scripts/append_interactiveworld_captures.py \
    explore_to_achieve/candidates/interactiveworld_20261029/catalogue.json \
    --output outputs/edge_review_20261029/interactiveworld_candidate_receipts

``--queue-root`` defaults to outputs/edge_review_20261029/queue. Original jobs
keep their priorities and allowed runtimes; new jobs have priority 10, two
attempts, and a 2400-second timeout. Only immutable capture inputs are validated.
No worker, simulator, robot reference execution, release admission, or physical
certification is performed. Receipts are separate from the original review index.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

from wia.resources import ROOT
from wia.runtime.esi_category_capture import freeze_working_code, publish_record
from wia.runtime.scene_capture_adapters import bind_capture_inputs, load_capture_scene
from wia.runtime.scene_capture_queue import digest, read_json, sha256, write_json
from wia.tasks.loader import load_scene, load_task
from wia.tasks.sources import AUTHORED_ROOT, resolve_source


REVISION = "interactiveworld_capture_append_v1"
CANDIDATE_MODULE = "wia.authoring.converters.interactiveworld_candidates"
DEFAULT_QUEUE = ROOT / "outputs/edge_review_20261029/queue"
EXPECTED_COUNT = 19
PRIORITY = 10


def _path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _collect_identities(value, identities):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("source_identity", "source_json") and isinstance(item, str):
                path = Path(item)
                if len(path.parts) == 5 and path.suffix == ".json":
                    identities.add(item)
            _collect_identities(item, identities)
    elif isinstance(value, list):
        for item in value:
            _collect_identities(item, identities)


def _definitions(catalogue_path):
    catalogue = read_json(catalogue_path)
    if (not isinstance(catalogue, dict) or catalogue.get("schema") != "wia_interactiveworld_candidates_v1"
            or catalogue.get("release_admitted") is not False):
        raise ValueError("Expected the unadmitted InteractiveWorld candidate catalogue.")
    selections = catalogue.get("tasks")
    if not isinstance(selections, list) or len(selections) != EXPECTED_COUNT:
        raise ValueError(f"This bounded append requires exactly {EXPECTED_COUNT} catalogue variants.")
    ids = [selection["id"] for selection in selections]
    if len(set(ids)) != EXPECTED_COUNT:
        raise ValueError("Every candidate catalogue variant must have a unique identity.")
    prepared = []
    for selected in selections:
        task_path, scene_path = _path(selected["task_file"]), _path(selected["scene_file"])
        variant = selected["variant"]
        task = load_task(task_path, variant=variant, run_mode="diagnostic")
        scene = load_scene(scene_path, variant=variant, run_mode="diagnostic")
        evaluation = task.definition.private.evaluation
        if (task.definition.kind != "category_robot_task" or task.definition.status != "pending"
                or scene.definition.status != "pending" or task.release_v1 or scene.release_v1):
            raise ValueError("Only pending candidate task/scene definitions may be appended.")
        if (task.definition.id != selected["id"] or scene.definition.id != selected["id"]
                or evaluation.module != CANDIDATE_MODULE
                or evaluation.operation != selected["prototype"]):
            raise ValueError("Catalogue identity, prototype, and candidate evaluator must agree.")
        if ((task.seed, task.scene_name, task.room_name) != (scene.seed, scene.scene_name, scene.room_name)
                or task.definition.defaults.seed != scene.definition.seed
                or task.definition.scene.variants[variant] != scene.definition.scene.variants[variant]
                or task.definition.private.bindings != scene.definition.bindings):
            raise ValueError("The actual task and separate scene must share seed, initialization, and bindings.")
        parameters = evaluation.parameters
        if (parameters.get("source_task") != selected["source_task"]
                or parameters.get("condition") != selected["condition"]):
            raise ValueError("Catalogue provenance and condition must match the actual task.")
        identities = {task.definition.source.json_path, scene.definition.source.json_path}
        _collect_identities(task.definition.model_dump(mode="json", by_alias=True), identities)
        _collect_identities(scene.definition.model_dump(mode="json", by_alias=True), identities)
        members = []
        for identity in sorted(identities):
            source = resolve_source(identity).resolve()
            members.append({"source_identity": identity, "source_file": str(source), "source_sha256": sha256(source)})
        primary = next(member for member in members if member["source_identity"] == scene.definition.source.json_path)
        if not Path(primary["source_file"]).is_relative_to(AUTHORED_ROOT.resolve()):
            raise ValueError("The original_source capture transport must be project-authored JSON.")
        if (scene.source_sha256 != primary["source_sha256"]
                or scene.source_path.resolve() != Path(primary["source_file"])):
            raise ValueError("The loaded native scene must use the exact authored capture transport.")
        task.verify_unchanged()
        scene.verify_unchanged()
        prepared.append((selected, task, scene, primary, members))
    return prepared


def _record(prepared, frozen, queue_root, catalogue_path, catalogue_hash):
    selected, task, scene, primary, members = prepared
    identity = {"task_sha256": task.file_sha256, "scene_sha256": scene.file_sha256,
                "code_commit": frozen["code_commit"], "variant": task.variant, "seed": task.seed}
    key = digest(identity)[:32]
    parameters = task.definition.private.evaluation.parameters
    record = {
        "candidate_id": "category_iw_" + key, "document_alias": selected["id"],
        "status": "generated", "definition_only": True, "evidence_scope": "initial_layout_capture_pending",
        "source_identity": primary["source_identity"], "original_source": primary["source_file"],
        "source_sha256": primary["source_sha256"], "capture_source_members": members,
        "big_task": Path(primary["source_identity"]).parts[0],
        "small_task": Path(primary["source_identity"]).parts[1],
        "task_file": str(task.path), "scene_file": str(scene.path),
        "task_sha256": task.file_sha256, "scene_sha256": scene.file_sha256,
        "task_source_identity": task.definition.source.json_path, "task_source_sha256": task.source_sha256,
        "seed": task.seed, "variant": task.variant,
        "converter_module": CANDIDATE_MODULE, "converter_version": REVISION,
        "reuse_level": "interactiveworld_robot_candidate_authored_transport",
        "diagnostic_dir": str(queue_root / "interactiveworld_diagnostics" / key),
        "capture_initial_agent_bundle": True, "task_edit2_capture_campaign": True,
        "task_edit2_variant_key": key, "definition_identity": identity,
        "reference_only": False, "reference_execution_status": "not_run",
        "release_admitted": False, "reference_admitted": False, "selected_final_variant": False,
        "success_claimed": False, "native_validation_run": False, "physical_certification": False,
        "interactiveworld_provenance": {
            "catalogue_file": str(catalogue_path), "catalogue_sha256": catalogue_hash,
            "id": selected["id"], "prototype": selected["prototype"],
            "condition": deepcopy(selected["condition"]), "source_task": selected["source_task"],
            "source_file": parameters.get("source_file"), "source_sha256": parameters.get("source_sha256"),
            "authored_transport_file": primary["source_file"],
        },
        "limitations": [
            "Pending candidate initial native images only; capture does not establish task success or physical certification.",
            "The authored JSON is native scene-loading transport; InteractiveWorld task provenance is recorded separately.",
            "No legal robot reference execution, release admission, or reference admission is established.",
            "Existing release capture records and review selections retain their original identities and priorities.",
        ],
    }
    record["revision"] = digest(record)
    return record


def _receipt_document(path, schema, key, queue_root):
    document = read_json(path)
    if document is None:
        return {"schema": schema, "queue_root": str(queue_root), key: [], "physical_certification": False}
    if (document.get("schema") != schema or document.get("queue_root") != str(queue_root)
            or not isinstance(document.get(key), list)):
        raise ValueError("Receipt destination contains another document: " + str(path))
    return document


def _append_receipt(document, key, value):
    identity = value["candidate_id"]
    existing = next((item for item in document[key] if item["candidate_id"] == identity), None)
    # Workers may later enrich a published card with images; preserve the original
    # append receipt while checking that its immutable job/definition did not change.
    immutable = ("record_revision", "job_id", "task_sha256", "scene_sha256", "code_commit")
    if existing is not None and any(existing.get(field) != value.get(field) for field in immutable):
        raise ValueError("An existing candidate receipt identity changed: " + identity)
    if existing is None:
        document[key].append(value)


def _validate_published(result, record, queue_root):
    job_id = result.get("job_id")
    if not job_id:
        raise RuntimeError(result.get("capture_submission_error", "Capture publication did not produce a job."))
    manifest_path = queue_root / "jobs" / job_id / "input/manifest.json"
    manifest = read_json(manifest_path)
    if (manifest.get("candidate_id") != record["candidate_id"]
            or manifest.get("code_commit") != record["definition_identity"]["code_commit"]
            or manifest.get("task_sha256") != record["task_sha256"]
            or manifest.get("capture_initial_agent_bundle") is not True
            or manifest.get("task_edit2_capture_campaign") is not True
            or manifest.get("diagnostic_pending") is not True
            or manifest.get("timeout_s") != 2400 or manifest.get("max_attempts") != 2):
        raise ValueError("Published candidate manifest does not match its immutable capture contract.")
    probe = SimpleNamespace()
    bind_capture_inputs(probe, load_capture_scene(manifest), manifest)
    if (probe.loaded_task.definition.id != record["document_alias"]
            or probe.capture_initial_agent_bundle is not True or probe._capture_release_edge_policy is not False):
        raise ValueError("Frozen candidate input binding did not preserve the actual pending task.")
    return job_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogue", type=Path, help="Candidate catalogue containing exactly 19 variants.")
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path, required=True, help="Separate candidate receipt directory.")
    args = parser.parse_args(argv)
    catalogue_path, queue_root, output = _path(args.catalogue), _path(args.queue_root), _path(args.output)
    if not (queue_root / "category_storage.json").is_file() or not (queue_root / "jobs").is_dir():
        raise ValueError("Append requires the existing configured capture queue.")
    if (output.is_relative_to(queue_root) or queue_root.is_relative_to(output)
            or output.is_relative_to(ROOT / "explore_to_achieve/wia_v1")):
        raise ValueError("Use a separate candidate receipt directory, outside the queue and release definitions.")
    prepared = _definitions(catalogue_path)
    catalogue_hash = sha256(catalogue_path)
    # Serialize repeat invocations without changing the worker, pause, or storage policy.
    with (queue_root / ".interactiveworld_append.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipts = _receipt_document(output / "appended_records.json", REVISION, "records", queue_root)
        index = _receipt_document(output / "review_index.json", "wia_capture_review_index_v1", "items", queue_root)
        policy_path = queue_root / "capture_runtime_policy.json"
        policy = read_json(policy_path)
        if (not isinstance(policy, dict) or not isinstance(policy.get("allowed_code_commits"), list)
                or not policy["allowed_code_commits"] or any(not isinstance(item, str) for item in policy["allowed_code_commits"])):
            raise ValueError("The existing allowed runtime policy must remain explicit and intact.")
        original_allowed = list(policy["allowed_code_commits"])
        frozen = freeze_working_code(queue_root)
        # Re-read the policy at publication time and append; never replace old IDs or other fields.
        policy = read_json(policy_path)
        if (not isinstance(policy, dict) or not isinstance(policy.get("allowed_code_commits"), list)
                or not policy["allowed_code_commits"]
                or any(commit not in policy["allowed_code_commits"] for commit in original_allowed)):
            raise ValueError("Existing runtime policy changed during snapshot preparation.")
        if frozen["code_commit"] not in policy["allowed_code_commits"]:
            policy["allowed_code_commits"].append(frozen["code_commit"])
            write_json(policy_path, policy)
        for item in prepared:
            record = _record(item, frozen, queue_root, catalogue_path, catalogue_hash)
            task, scene = item[1], item[2]
            task.verify_unchanged()
            scene.verify_unchanged()
            record_path = queue_root / "category_records" / (record["candidate_id"] + ".json")
            result = read_json(record_path)
            if result is not None:
                if result.get("record_revision") != record["revision"]:
                    raise ValueError("An existing category record identity cannot be overwritten.")
            else:
                result = publish_record(record, queue_root, frozen=frozen, priority=PRIORITY,
                                        timeout_s=2400, max_attempts=2)
            job_id = _validate_published(result, record, queue_root)
            _append_receipt(receipts, "records", result)
            _append_receipt(index, "items", {
                "candidate_id": record["candidate_id"], "row_id": record["document_alias"],
                "document_alias": record["document_alias"], "variant_id": task.definition.id,
                "label": f"{item[0]['prototype']} candidate initial native capture",
                "task_file": record["task_file"], "task_sha256": record["task_sha256"],
                "scene_file": record["scene_file"], "scene_sha256": record["scene_sha256"],
                "job_id": job_id, "code_commit": frozen["code_commit"],
                "receipt_file": str(queue_root / "jobs" / job_id / "attempts/01/preview_complete.json"),
                "queue_status_file": str(queue_root / "jobs" / job_id / "status.json"),
                "priority": PRIORITY, "release_admitted": False, "physical_certification": False,
            })
            write_json(output / "appended_records.json", receipts)
            write_json(output / "review_index.json", index)
            print(record["document_alias"], job_id, flush=True)
    print(json.dumps({"validated_candidate_count": len(prepared), "output": str(output),
                      "priority": PRIORITY, "workers_started": False, "physical_certification": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
