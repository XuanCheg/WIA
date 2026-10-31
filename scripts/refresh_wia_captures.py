"""Plan or apply an explicit baseline capture refresh without starting workers.

Plan only (default; no filesystem or queue mutation):
  PYTHONPATH=src:../Show-Harness-Explore python scripts/refresh_wia_captures.py --aliases T01 T45 --dry-run
After review, replace --dry-run with --apply. Optional --campaign defaults to
outputs/edge_review_20261029; --queue-root defaults to its queue subdirectory.
Apply publishes and validates every replacement before deleting any explicitly
mapped old image. Logs, receipts, configs, registry, curation, and other aliases
are retained. New captures use priority 10, two attempts, and a 2400 s timeout.
This utility does not execute a robot, start GPU workers, or certify physics.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

from wia.resources import ROOT
from wia.runtime.scene_capture_adapters import bind_capture_inputs, load_capture_scene
from wia.runtime.scene_capture_queue import digest, read_json, sha256, write_json
from wia.tasks.loader import load_scene, load_task
from wia.tasks.sources import resolve_source


REVISION = "wia_selected_capture_refresh_v1"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff", ".bmp"}
LIMITATION = "Images from superseded jobs were intentionally deleted by an explicit alias refresh; old receipts are retained and intentionally stale."


def _path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _declared_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _image_header(path):
    with Path(path).open("rb") as handle:
        head = handle.read(16)
    return (head.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"II*\x00", b"MM\x00*", b"BM"))
            or head.startswith(b"RIFF") and head[8:12] == b"WEBP")


def _symlink_component(path, campaign):
    try:
        parts = Path(path).relative_to(campaign).parts
    except ValueError:
        return True
    current = campaign
    for part in parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _collect_sources(value, identities):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("source_identity", "source_json") and isinstance(child, str):
                if len(Path(child).parts) == 5 and child.endswith(".json"):
                    identities.add(child)
            _collect_sources(child, identities)
    elif isinstance(value, list):
        for child in value:
            _collect_sources(child, identities)


def _definition(selected):
    task = load_task(_path(selected["task_file"]), variant=selected["variant"], run_mode="diagnostic")
    scene = load_scene(_path(selected["scene_file"]), variant=selected["variant"], run_mode="diagnostic")
    if (not task.release_v1 or not scene.release_v1 or task.definition.id != selected["task_id"]
            or task.seed != scene.seed or task.definition.defaults.seed != scene.definition.seed
            or (task.scene_name, task.room_name) != (scene.scene_name, scene.room_name)
            or task.definition.scene.variants[task.variant] != scene.definition.scene.variants[scene.variant]
            or task.definition.private.bindings != scene.definition.bindings):
        raise ValueError("The selected baseline task/scene identity, seed, initialization, or bindings disagree.")
    # QA and native-scene source roles may differ, as declared for T23.
    identities = {task.definition.source.json_path, scene.definition.source.json_path}
    _collect_sources(task.definition.model_dump(mode="json", by_alias=True), identities)
    _collect_sources(scene.definition.model_dump(mode="json", by_alias=True), identities)
    members = [{"source_identity": identity, "source_file": str(resolve_source(identity)),
                "source_sha256": sha256(resolve_source(identity))} for identity in sorted(identities)]
    for loaded in (task, scene):
        member = next(value for value in members if value["source_identity"] == loaded.definition.source.json_path)
        if loaded.source_path.resolve() != Path(member["source_file"]).resolve() or loaded.source_sha256 != member["source_sha256"]:
            raise ValueError("Actual source bytes do not match their declared role.")
        loaded.verify_unchanged()
    return {"alias": selected["alias"], "task_id": task.definition.id, "task_file": str(task.path),
            "scene_file": str(scene.path), "task_sha256": task.file_sha256, "scene_sha256": scene.file_sha256,
            "task_source_identity": task.definition.source.json_path,
            "scene_source_identity": scene.definition.source.json_path, "capture_source_members": members,
            "variant": task.variant, "seed": task.seed,
            "converter_module": task.definition.private.evaluation.module}


def _old_capture(item, queue, *, validate=False):
    job = queue / "jobs" / item["job_id"]
    manifest_path = job / "input/manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("job_id") != item["job_id"]:
        raise ValueError("Current alias has no matching sealed job: " + item["document_alias"])
    if validate:
        accepted = read_json(job / "accepted.json", {})
        if accepted.get("manifest_sha256") != sha256(manifest_path):
            raise ValueError("Accepted old-job manifest changed.")
        for relative in ("task.yaml", "scene.yaml"):
            if sha256(job / "input" / relative) != manifest["input_hashes"][relative]:
                raise ValueError("Old sealed definition changed: " + relative)
        if (manifest.get("task_sha256") != item.get("task_sha256")
                or manifest["input_hashes"].get("scene.yaml") != item.get("scene_sha256")
                or _path(manifest["task_file"]) != _path(item["task_file"])
                or _path(manifest["scene_file"]) != _path(item["scene_file"])):
            raise ValueError("Old sealed job does not belong to this exact current alias selection.")
        probe = SimpleNamespace()
        bind_capture_inputs(probe, load_capture_scene(manifest), manifest)
        if not probe._capture_release_edge_policy:
            raise ValueError("Old selected baseline capture lacks its release edge policy.")
    receipts = []
    for path in sorted((job / "attempts").glob("*/preview_complete.json")):
        receipt = read_json(path, {})
        if (receipt.get("job_id") != item["job_id"]
                or receipt.get("capture_signature") != manifest["capture_signature"]):
            continue
        receipts.append({"file": str(path), "sha256": sha256(path), "complete": receipt.get("complete") is True,
                         "images": {name: str(_declared_path(value)) for name, value in receipt.get("images", {}).items()
                                    if isinstance(value, str)}})
    return {"job_id": item["job_id"], "manifest_file": str(manifest_path), "manifest_sha256": sha256(manifest_path),
            "task_sha256": manifest.get("task_sha256"), "scene_sha256": manifest.get("input_hashes", {}).get("scene.yaml"),
            "code_commit": manifest.get("code_commit"),
            "source_members": [{"source_identity": member["source_identity"], "source_sha256": member["source_sha256"]}
                               for member in manifest.get("capture_source_members", [])],
            "capture_signature": manifest["capture_signature"], "receipts": receipts,
            "indexed_receipt_file": item.get("receipt_file")}


def _image_strings(value):
    """Follow explicit asset metadata only, never enumerate image directories."""
    if isinstance(value, str) and Path(value).suffix.lower() in IMAGE_SUFFIXES:
        yield str(_declared_path(value))
    elif isinstance(value, dict):
        for child in value.values():
            yield from _image_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _image_strings(child)


def _published_assets(campaign, item):
    assets = []
    for filename in ("capture_evidence.json", "registry.json", "curation.json"):
        path = campaign / filename
        document = read_json(path, {})
        scopes = []
        row = document.get("row_updates", {}).get(item["row_id"])
        if isinstance(row, dict):
            scopes.append(row)
        scopes += [row for row in document.get("rows", []) if row.get("document_alias") == item["document_alias"]]
        for row in scopes:
            versions = list(row.get("variants", []))
            if row.get("capture_review", {}).get("job_id", row.get("job_id")) == item["job_id"]:
                versions.append({**row, "id": item.get("variant_id")})
            for variant in versions:
                if variant.get("id") != item.get("variant_id"):
                    continue
                declared_job = variant.get("capture_review", {}).get("job_id", variant.get("job_id"))
                if declared_job is not None and declared_job != item["job_id"]:
                    continue
                for key in ("images", "diagnostic_images", "review_assets", "published_assets"):
                    for image in _image_strings(variant.get(key, [])):
                        assets.append({"path": image, "origin": str(path), "mapping": "exact_alias_row_and_current_variant", "alias": item["document_alias"]})
    return assets


def plan_image_deletions(references, campaign, protected_paths=()):
    """Return exact authorized regular images and retained references without mutation."""
    campaign = Path(campaign).resolve()
    protected = {Path(path).resolve() for path in protected_paths}
    deletions, retained = {}, []
    for reference in references:
        path = Path(reference["path"])
        resolved = path.resolve()
        reason = None
        if not resolved.is_relative_to(campaign):
            reason = "outside_campaign"
        elif resolved in protected:
            reason = "also_referenced_by_unselected_alias"
        elif path.suffix.lower() not in IMAGE_SUFFIXES:
            reason = "not_an_image_file"
        elif _symlink_component(path, campaign) or (path.exists() and not path.is_file()):
            reason = "not_a_regular_image_file"
        elif not path.is_file():
            reason = "already_missing"
        elif not _image_header(path):
            reason = "image_header_unrecognized"
        if reason:
            retained.append({**reference, "reason": reason})
            continue
        key = str(resolved)
        entry = deletions.setdefault(key, {"path": key, "sha256": sha256(path), "bytes": path.stat().st_size, "references": []})
        entry["references"].append(reference)
    return sorted(deletions.values(), key=lambda value: value["path"]), retained


def build_plan(aliases, campaign, queue):
    campaign, queue = Path(campaign).resolve(), Path(queue).resolve()
    if not queue.is_relative_to(campaign) or not (queue / "category_storage.json").is_file():
        raise ValueError("Use an existing configured queue inside this campaign.")
    catalogue = read_json(ROOT / "explore_to_achieve/wia_v1/release/catalogue.json")
    selected = {item["alias"]: item for item in catalogue["tasks"]}
    if not aliases or len(set(aliases)) != len(aliases) or any(alias not in selected for alias in aliases):
        raise ValueError("--aliases requires distinct explicit baseline catalogue aliases; excluded rows are unavailable.")
    index_path = campaign / "review_index.json"
    index = read_json(index_path)
    items, references, protected, old = [], [], set(), {}
    for item in index["items"]:
        alias = item.get("document_alias")
        capture = _old_capture(item, queue, validate=alias in aliases)
        refs = [{"path": image, "origin": receipt["file"], "mapping": "exact_sealed_job_receipt", "alias": alias}
                for receipt in capture["receipts"] for image in receipt["images"].values()]
        refs += _published_assets(campaign, item)
        if alias in aliases:
            old[alias] = capture
            references += refs
            items.append(deepcopy(item))
        else:
            protected.update(reference["path"] for reference in refs)
    if sorted(item["document_alias"] for item in items) != sorted(aliases):
        raise ValueError("Every requested alias must have exactly one current review-index row.")
    definitions = [_definition(selected[alias]) for alias in aliases]
    deletions, retained = plan_image_deletions(references, campaign, protected)
    return {"schema": REVISION, "mode": "plan_only", "campaign": str(campaign), "queue_root": str(queue),
            "aliases": list(aliases), "review_index_file": str(index_path), "review_index_sha256": sha256(index_path),
            "definitions": definitions, "previous_index_items": items, "old_captures": old,
            "delete_images": deletions, "retained_image_references": retained,
            "new_jobs": "Computed only by --apply after one immutable code freeze.",
            "priority": 10, "max_attempts": 2, "timeout_s": 2400,
            "limitations": [LIMITATION, "Initial images do not certify physical task success."]}


def replacement_index(index, replacements):
    """Pure alias-local update, preserving every unrelated row and top-level field."""
    result = deepcopy(index)
    for item in result["items"]:
        alias = item.get("document_alias")
        if alias in replacements:
            item.update(deepcopy(replacements[alias]))
    return result


def _record(definition, frozen, campaign):
    identity = {**definition, "code_commit": frozen["code_commit"]}
    key = digest(identity)[:32]
    primary = next(value for value in definition["capture_source_members"] if value["source_identity"] == definition["scene_source_identity"])
    record = {**definition, "candidate_id": "category_" + key, "status": "generated",
              "source_identity": primary["source_identity"], "original_source": primary["source_file"],
              "source_sha256": primary["source_sha256"], "big_task": Path(primary["source_identity"]).parts[0],
              "small_task": Path(primary["source_identity"]).parts[1], "capture_initial_agent_bundle": True,
              "task_edit2_capture_campaign": True, "task_edit2_variant_key": key,
              "converter_version": REVISION, "reuse_level": "repaired_current_release_definition",
              "diagnostic_dir": str(campaign / "refreshes" / "diagnostics" / key),
              "limitations": [LIMITATION, "Initial native review only; no robot reference execution or task success is inferred."],
              "reference_only": False, "physical_success_inferred": False, "refresh_identity": identity}
    record["revision"] = digest(record)
    return record


def _validate_replacement(record, result, queue):
    if not result.get("job_id"):
        raise RuntimeError(result.get("capture_submission_error", "Replacement publication has no job."))
    job = queue / "jobs" / result["job_id"]
    manifest = read_json(job / "input/manifest.json")
    for relative, expected in (("task.yaml", record["task_sha256"]), ("scene.yaml", record["scene_sha256"])):
        if manifest["input_hashes"].get(relative) != expected or sha256(job / "input" / relative) != expected:
            raise ValueError("Replacement sealed input differs from its actual selected definition.")
    if (manifest.get("candidate_id") != record["candidate_id"]
            or manifest.get("code_commit") != record["refresh_identity"]["code_commit"]
            or manifest.get("timeout_s") != 2400 or manifest.get("max_attempts") != 2):
        raise ValueError("Replacement job identity or queue settings differ.")
    probe = SimpleNamespace()
    bind_capture_inputs(probe, load_capture_scene(manifest), manifest)
    if not probe._capture_release_edge_policy or not probe.capture_initial_agent_bundle or probe.loaded_task.definition.id != record["task_id"]:
        raise ValueError("Replacement must bind the actual baseline task and release edge policy.")
    for member in record["capture_source_members"]:
        if manifest["input_hashes"].get("data/" + member["source_identity"]) != member["source_sha256"]:
            raise ValueError("Replacement source membership changed.")
    receipts = _old_capture({"job_id": result["job_id"], "document_alias": record["alias"]}, queue)["receipts"]
    complete = [receipt for receipt in receipts if receipt["complete"] and receipt["images"]
                and all(Path(path).is_file() for path in receipt["images"].values())]
    return result["job_id"], complete[-1]["file"] if complete else str(job / "attempts/01/preview_complete.json")


def apply_plan(plan):
    """Publish all replacements first; any publication/binding error leaves old images intact."""
    from wia.runtime.esi_category_capture import freeze_working_code, publish_record

    campaign, queue = Path(plan["campaign"]), Path(plan["queue_root"])
    with (campaign / ".refresh_wia_captures.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current_plan = build_plan(plan["aliases"], campaign, queue)
        if digest(current_plan) != digest(plan):
            raise ValueError("Refresh plan changed before apply; regenerate the dry-run.")
        frozen = freeze_working_code(queue)
        policy_path = queue / "capture_runtime_policy.json"
        policy = read_json(policy_path)
        if not isinstance(policy, dict) or not isinstance(policy.get("allowed_code_commits"), list) or not policy["allowed_code_commits"]:
            raise ValueError("Preserve the existing explicit allowed runtime policy.")
        if frozen["code_commit"] not in policy["allowed_code_commits"]:
            policy["allowed_code_commits"].append(frozen["code_commit"])
            write_json(policy_path, policy)
        replacements, records = {}, []
        previous = {item["document_alias"]: item for item in plan["previous_index_items"]}
        for definition in plan["definitions"]:
            record = _record(definition, frozen, campaign)
            old = previous[definition["alias"]]
            path = queue / "category_records" / (record["candidate_id"] + ".json")
            result = read_json(path)
            if result is None:
                result = publish_record(record, queue, frozen=frozen, priority=10, timeout_s=2400, max_attempts=2)
            elif result.get("record_revision") != record["revision"]:
                raise ValueError("Existing replacement record identity cannot be overwritten.")
            job_id, receipt = _validate_replacement(record, result, queue)
            records.append(result)
            if job_id == old["job_id"]:
                continue
            limitations = list(old.get("limitations", []))
            if LIMITATION not in limitations:
                limitations.append(LIMITATION)
            replacements[definition["alias"]] = {
                "job_id": job_id, "receipt_file": receipt, "queue_status_file": str(queue / "jobs" / job_id / "status.json"),
                "task_file": definition["task_file"], "scene_file": definition["scene_file"],
                "task_sha256": definition["task_sha256"], "scene_sha256": definition["scene_sha256"],
                "capture_candidate_id": record["candidate_id"], "code_commit": frozen["code_commit"],
                "refresh_revision": REVISION, "limitations": limitations,
                "previous_capture_jobs": list(dict.fromkeys([*old.get("previous_capture_jobs", []), old["job_id"]])),
            }
        if not replacements:
            return {"state": "already_current", "aliases": plan["aliases"], "deleted_images": []}
        active = set(replacements)
        deletes = [entry for entry in plan["delete_images"] if all(ref["alias"] in active for ref in entry["references"])]
        # Recheck every planned image before any unlink; symlink escapes and changed bytes fail closed.
        for entry in deletes:
            path = Path(entry["path"])
            if _symlink_component(path, campaign) or not path.resolve().is_relative_to(campaign) or not path.is_file() or sha256(path) != entry["sha256"]:
                raise ValueError("A planned old image changed before deletion: " + str(path))
        index_path = Path(plan["review_index_file"])
        current_index = read_json(index_path)
        for item in current_index["items"]:
            alias = item.get("document_alias")
            if alias in active and item["job_id"] != previous[alias]["job_id"]:
                raise ValueError("Selected alias changed while replacement jobs were being published.")
        new_ids = {alias: value["job_id"] for alias, value in replacements.items()}
        refresh_key = digest({"aliases": sorted(active), "new_jobs": new_ids,
                              "old_jobs": {alias: plan["old_captures"][alias]["job_id"] for alias in active}})[:32]
        manifest_path = campaign / "refreshes" / refresh_key / "deletion_manifest.json"
        manifest = {**plan, "mode": "apply", "state": "replacements_validated", "new_jobs": replacements,
                    "published_records": records, "delete_images": deletes, "deleted_images": [], "physical_success_inferred": False}
        prior_manifest = read_json(manifest_path)
        if prior_manifest is not None:
            prior_ids = {alias: value["job_id"] for alias, value in prior_manifest.get("new_jobs", {}).items()}
            if prior_ids != new_ids or prior_manifest.get("old_captures") != plan["old_captures"]:
                raise ValueError("Existing deletion manifest belongs to another refresh identity.")
            # Resume a partial unlink phase without losing the original complete
            # path/hash inventory or the already recorded deletion receipts.
            manifest = prior_manifest
            manifest["new_jobs"] = replacements
        write_json(manifest_path, manifest)
        for entry in deletes:
            if _symlink_component(Path(entry["path"]), campaign) or sha256(entry["path"]) != entry["sha256"]:
                raise ValueError("Old image changed during deletion: " + entry["path"])
            Path(entry["path"]).unlink()
            if entry["path"] not in manifest["deleted_images"]:
                manifest["deleted_images"].append(entry["path"])
            write_json(manifest_path, manifest)
        # Keep the original index until deletion completes, so a repeat after an
        # interrupted unlink phase reuses the same old job and remaining paths.
        latest_index = read_json(index_path)
        for item in latest_index["items"]:
            alias = item.get("document_alias")
            if alias in active and item["job_id"] != previous[alias]["job_id"]:
                raise ValueError("Selected alias changed during image retirement.")
        write_json(index_path, replacement_index(latest_index, replacements))
        manifest["state"] = "applied"
        write_json(manifest_path, manifest)
        return {"state": "applied", "aliases": sorted(active), "manifest_file": str(manifest_path), "deleted_images": manifest["deleted_images"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aliases", nargs="+", required=True)
    parser.add_argument("--campaign", type=Path, default=ROOT / "outputs/edge_review_20261029")
    parser.add_argument("--queue-root", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Plan only; this is the default.")
    mode.add_argument("--apply", action="store_true", help="Publish validated replacements, then retire explicitly mapped old images.")
    args = parser.parse_args(argv)
    campaign = _path(args.campaign)
    queue = _path(args.queue_root) if args.queue_root else campaign / "queue"
    plan = build_plan(args.aliases, campaign, queue)
    result = apply_plan(plan) if args.apply else plan
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
