"""Read selected WIA definitions and capture evidence without changing the dataset.

Run CPU-only from the WIA checkout:
  PYTHONPATH=src python scripts/scan_wia_dataset.py \
    --native-captures outputs/interactiveworld_20261029/baseline_full/captures \
    --native-captures outputs/interactiveworld_20261029/baseline_r1 \
    --native-captures outputs/interactiveworld_20261029/baseline_r2 \
    --output outputs/interactiveworld_20261029/dataset_scan

Writes report.json and report.csv only under --output. Pending diagnostics and
frozen release selection do not establish failure, qualification, or success.
Known false initialization measurements retain their exact JSON pointers. No
repair, source edit, queue mutation, worker, simulator, or GPU launch occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from wia.resources import ROOT
from wia.runtime.scene_capture_adapters import bind_capture_inputs, load_capture_scene
from wia.tasks.loader import LoadedRigidTask, load_scene, load_task
from wia.tasks.sources import resolve_source


SCHEMA = "wia_dataset_read_only_scan_v1"
DEFAULT_SUPPLEMENTARY_INDEX = ROOT / "outputs/edge_review_20261029/household_geometry_receipts/review_index.json"
HISTORICAL_POINTERS = ("/attempts/", "/before/", "/original_robot_start_checks/",
                       "/deformable_first_failed_placement/", "/optical_selection/candidates/",
                       "/independent_group_clearance_before/")


def _utc(timestamp=None):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat() if timestamp is not None else datetime.now(timezone.utc).isoformat()


def _path(value):
    value = Path(value).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def _file(path):
    path = Path(path)
    result = {"file": str(path), "exists": path.is_file()}
    if result["exists"]:
        raw = path.read_bytes()
        stat = path.stat()
        result.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                      modified_at_utc=_utc(stat.st_mtime), modified_at_unix=stat.st_mtime)
    return result


def _json(path):
    path = Path(path)
    metadata = _file(path)
    if not metadata["exists"]:
        return None, metadata
    try:
        return json.loads(path.read_text()), metadata
    except (ValueError, OSError) as error:
        metadata["read_error"] = f"{type(error).__name__}: {error}"
        return None, metadata


def _pointer(parent, key):
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


def _issue(row, code, reason, *, kind="evidence_gap", file=None, pointer=None, observed=None):
    value = {"kind": kind, "code": code, "reason": reason}
    if file is not None:
        value["file"] = str(file)
    if pointer is not None:
        value["json_pointer"] = pointer
    if observed is not None:
        value["observed"] = observed
    row["issues"].append(value)


def _walk(value, pointer="", parts=()):
    yield pointer, parts, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, _pointer(pointer, key), (*parts, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, _pointer(pointer, index), (*parts, str(index)))


def _known_concerns(row, receipt, path):
    """Inspect only the requested measured booleans, without interpreting other false flags."""
    def flag(code, reason, pointer):
        historical = any(marker in pointer for marker in HISTORICAL_POINTERS)
        if historical:
            reason = "Historical or nonterminal measurement retained verbatim; excluded from current layout concerns. " + reason
        _issue(row, code, reason, kind="historical_measurement_information" if historical else "initial_support_concern",
               file=path, pointer=pointer, observed=False)

    for pointer, parts, value in _walk(receipt):
        if not isinstance(value, dict):
            continue
        for key in ("accepted", "within_footprint", "floor_supported", "collision_free"):
            if value.get(key) is False:
                flag("initial_" + key + "_false", f"The receipt reports {key}=false; no impossibility result is inferred.", _pointer(pointer, key))
        if ("supported_items" in parts and value.get("touching") is False
                and value.get("support_contract") != "fixed_mount"
                and value.get("physical_contact_required") is not False):
            object_name = parts[parts.index("supported_items") + 1] if parts.index("supported_items") + 1 < len(parts) else None
            spec = row.get("declared_object_specs", {}).get(object_name, {})
            if spec.get("fixed_base") is not True and spec.get("fixed") is not True:
                flag("mutable_supported_item_touching_false", "A supported-item receipt requiring physical contact reports touching=false; fixed-mount receipts are excluded.", _pointer(pointer, "touching"))
        if "pickup_edge_placements" in parts and value.get("same_room_stance_found") is False:
            flag("pickup_edge_same_room_stance_false", "The bounded pickup-edge placement receipt did not find a same-room stance; no general reachability conclusion is inferred.", _pointer(pointer, "same_room_stance_found"))


def _metadata(row, selected):
    task_path, scene_path = _path(selected["task_file"]), _path(selected["scene_file"])
    row["files"] = {"task": _file(task_path), "scene": _file(scene_path)}
    try:
        task = load_task(task_path, variant=selected["variant"], run_mode="diagnostic")
        scene = load_scene(scene_path, variant=selected["variant"], run_mode="diagnostic")
        td, sd = task.definition, scene.definition
        private = getattr(td, "private", None)
        evaluation = getattr(private, "evaluation", None)
        task_scene = task.scene.definition if isinstance(task, LoadedRigidTask) else td
        task_bindings = getattr(private, "bindings", task_scene.bindings if isinstance(task, LoadedRigidTask) else {})
        row.update(load_status="loaded", task_id=td.id, scene_id=sd.id,
                   task_status=getattr(td, "status", None), scene_status=sd.status,
                   seed=task.seed, scene_seed=scene.seed, variant=task.variant,
                   task_source_identity=td.source.json_path, scene_source_identity=sd.source.json_path,
                   task_source_scene=task.scene_name, task_source_room=task.room_name,
                   scene_model=scene.scene_name, room=scene.room_name,
                   module=getattr(evaluation, "module", None), operation=getattr(evaluation, "operation", None),
                   evaluation_type=getattr(evaluation, "type", None),
                   public_objective=td.public.objective,
                   public=td.public.model_dump(mode="json"),
                   declared_requirements=td.requirements.model_dump(mode="json") if hasattr(td, "requirements") else None,
                   bindings={key: value.model_dump(mode="json") for key, value in task_bindings.items()},
                   scene_bindings={key: value.model_dump(mode="json") for key, value in sd.bindings.items()},
                   frozen_release_selection=task.release_v1, physical_qualification_inferred=False)
        row["files"].update(task_source=_file(task.source_path), scene_source=_file(scene.source_path))
        row["layout_parameters"] = {
            "evaluation": getattr(evaluation, "parameters", {}),
            "selected_scene_patches": [patch.model_dump(mode="json") for patch in sd.scene.variants[scene.variant].patches],
            "scene_qualification_contract": sd.qualification.model_dump(mode="json") if sd.qualification else None,
            "task_qualification_contract": td.qualification.model_dump(mode="json") if hasattr(td, "qualification") else None,
            "runtime": sd.runtime.model_dump(mode="json"),
        }
        specs = {}
        for _, _, value in _walk(row["layout_parameters"]):
            if isinstance(value, dict) and isinstance(value.get("objects"), list):
                for spec in value["objects"]:
                    if isinstance(spec, dict) and isinstance(spec.get("name"), str):
                        specs[spec["name"]] = spec
        row["declared_object_specs"] = specs
        comparisons = [
            (td.id == selected.get("task_id", selected.get("id", td.id)), "catalogue_task_identity_disagreement", "/id", str(task_path)),
            (task.seed == scene.seed and td.defaults.seed == sd.seed, "task_scene_seed_disagreement", "/seed", str(scene_path)),
            ((task.scene_name, task.room_name) == (scene.scene_name, scene.room_name), "task_scene_location_disagreement", "/source/json", str(scene_path)),
            (td.source.json_path == sd.source.json_path, "task_scene_source_identity_disagreement", "/source/json", str(scene_path)),
            (row["bindings"] == row["scene_bindings"], "task_scene_bindings_disagreement", "/bindings", str(scene_path)),
            (task_scene.scene.variants[task.variant] == sd.scene.variants[scene.variant], "task_scene_initialization_disagreement", "/scene/variants/" + scene.variant, str(scene_path)),
            (task.source_path.resolve() == resolve_source(td.source.json_path).resolve(), "task_actual_source_disagreement", "/source/json", str(task_path)),
            (scene.source_path.resolve() == resolve_source(sd.source.json_path).resolve(), "scene_actual_source_disagreement", "/source/json", str(scene_path)),
        ]
        for passed, code, pointer, file in comparisons:
            if not passed:
                _issue(row, code, "The actual selected task/scene/source pairing disagrees on this declared field; no repair classification is assigned.", kind="definition_error", file=file, pointer=pointer)
        provenance = td.model_dump(mode="json").get("repair", {}).get("original_qa_provenance", {})
        if td.source.json_path != sd.source.json_path and all((
                provenance.get("question_source_identity") == td.source.json_path,
                provenance.get("scene_source_identity") == sd.source.json_path,
                provenance.get("question_source_sha256") == row["files"]["task_source"].get("sha256"),
                provenance.get("scene_source_sha256") == row["files"]["scene_source"].get("sha256"))):
            row["issues"] = [issue for issue in row["issues"] if issue["code"] != "task_scene_source_identity_disagreement"]
            _issue(row, "declared_separate_source_roles", "The task explicitly declares separate QA and scene sources; both identities and current bytes match its recorded provenance.",
                   kind="declared_source_role_information", file=task_path, pointer="/repair/original_qa_provenance")
        task.verify_unchanged()
        scene.verify_unchanged()
    except Exception as error:
        row["load_status"] = "error"
        _issue(row, "diagnostic_load_error", f"{type(error).__name__}: {error}", kind="definition_error")


def _mismatch(row, observed, expected, path, pointer, code):
    if observed is not None and expected is not None and observed != expected:
        _issue(row, code, "Captured provenance differs from the current selected definition.", kind="stale_evidence", file=path, pointer=pointer, observed=observed)


def _coverage(row, directory):
    inventory_path = directory / "inventory.json" if directory else None
    audit_path = directory / "reachability.json" if directory else None
    inventory, inventory_meta = _json(inventory_path) if inventory_path else (None, {"exists": False})
    audit, audit_meta = _json(audit_path) if audit_path else (None, {"exists": False})
    coverage = {
        "inventory": inventory_meta, "measured_inventory_available": isinstance(inventory, dict),
        "inventory_validation_scope": inventory.get("validation_scope") if isinstance(inventory, dict) else None,
        "full_measured_inventory_declared": isinstance(inventory, dict) and inventory.get("full_measured_inventory") is True,
        "kinematic_audit": audit_meta, "kinematic_audit_report_available": isinstance(audit, dict),
        "full_kinematic_audit_declared": isinstance(audit, dict) and audit.get("full_kinematic_audit") is True,
        "interpretation": "Available files are reported separately from explicitly declared full audit coverage; no completeness or physical success is inferred.",
    }
    if isinstance(inventory, dict):
        coverage.update(inventory_id=inventory.get("id"), inventory_schema=inventory.get("schema"),
                        measured_object_count=len(inventory.get("objects", [])),
                        robot_calibration_present=bool(inventory.get("robot", {}).get("joint_calibration")))
        _mismatch(row, inventory.get("id"), row["alias"], inventory_path, "/id", "inventory_identity_stale")
    if isinstance(audit, dict):
        coverage.update(kinematic_audit_status=audit.get("status"), kinematic_audit_method=audit.get("method"))
    row["audit_coverage"] = coverage


def _validate_current_sealed_inputs(row, manifest):
    """Use the existing binding API and exact current bytes, including separate source roles."""
    try:
        hashes = manifest["input_hashes"]
        expected = {
            "task.yaml": row["files"]["task"]["sha256"],
            "scene.yaml": row["files"]["scene"]["sha256"],
            "data/" + row["task_source_identity"]: row["files"]["task_source"]["sha256"],
            "data/" + row["scene_source_identity"]: row["files"]["scene_source"]["sha256"],
        }
        for relative, expected_hash in expected.items():
            if hashes.get(relative) != expected_hash:
                raise ValueError("Frozen input differs from current selection: " + relative)
            if _file(Path(manifest["snapshot_dir"]) / relative).get("sha256") != expected_hash:
                raise ValueError("Actual sealed bytes differ from current selection: " + relative)
        if (manifest.get("task_sha256") != expected["task.yaml"]
                or manifest.get("source_identity") != row["scene_source_identity"]
                or (manifest.get("seed"), manifest.get("variant")) != (row["seed"], row["variant"])):
            raise ValueError("Manifest task/source/seed/variant identity differs from the current selection.")
        probe = SimpleNamespace()
        bind_capture_inputs(probe, load_capture_scene(manifest), manifest)
        if probe.loaded_task.definition.id != row["task_id"]:
            raise ValueError("The sealed actual task identity differs from the current selection.")
        return {"valid": True, "capture_input_identity": probe.capture_input_identity,
                "method": "exact_current_input_hashes_and_bind_capture_inputs"}
    except Exception as error:
        return {"valid": False, "error": f"{type(error).__name__}: {error}"}


def _latest_complete_receipt(row, item, indexed_path, manifest, validation):
    """Resolve retries inside one exact sealed job; never borrow another alias's capture."""
    job = indexed_path.parents[2]
    candidates, matching = [], []
    for path in sorted((job / "attempts").glob("*/preview_complete.json")):
        receipt, metadata = _json(path)
        reasons = []
        if not isinstance(receipt, dict):
            reasons.append("receipt_unreadable")
        else:
            binding = receipt.get("initial_agent_bundle", {}).get("capture_input_identity", {})
            checks = {
                "same_sealed_job": item.get("job_id") == job.name == manifest.get("job_id") == receipt.get("job_id"),
                "signature": receipt.get("capture_signature") == manifest.get("capture_signature"),
                "runtime": receipt.get("code_commit") == manifest.get("code_commit") and receipt.get("esi_commit") == manifest.get("esi_commit"),
                "seed_variant": (receipt.get("seed"), receipt.get("variant")) == (manifest.get("seed"), manifest.get("variant")),
                "actual_task_binding": binding.get("task_sha256") == row["files"]["task"].get("sha256"),
                "current_scene_task_source": validation["valid"],
                "complete": receipt.get("complete") is True,
                "images": bool(receipt.get("images")) and isinstance(receipt.get("images"), dict)
                    and set(manifest.get("expected_views", [])) <= set(receipt["images"])
                    and all(isinstance(image, str) and Path(image).is_file() for image in receipt["images"].values()),
            }
            reasons.extend(name for name, passed in checks.items() if not passed)
        candidate = {"receipt": metadata, "matching_complete": not reasons, "rejection_reasons": reasons}
        candidates.append(candidate)
        if not reasons:
            try:
                attempt = int(path.parent.name)
            except ValueError:
                attempt = -1
            matching.append((attempt, metadata.get("modified_at_unix", 0), path))
    selected = max(matching, key=lambda value: value[:2])[2] if matching else indexed_path
    return selected, {"indexed_receipt_file": str(indexed_path), "selected_receipt_file": str(selected),
                      "selection": "latest_complete_matching_sealed_job" if matching else "indexed_receipt_fallback_no_current_matching_complete_attempt",
                      "attempts": candidates}


def _candidate_initialization(row, receipt, candidate_dir, inventory_path):
    """Link a shared native receipt only through a matching capture interval and measured inventory."""
    if isinstance(receipt.get("candidate_initialization"), dict):
        current = (receipt.get("id") == row["alias"]
                   and receipt.get("scene_sha256") == row["files"]["scene"].get("sha256")
                   and receipt.get("source_sha256") == row["files"]["scene_source"].get("sha256"))
        row["candidate_initialization"] = {"attribution": "embedded_in_exact_capture_receipt",
            "used_for_current_concerns": current, "raw_receipt": receipt["candidate_initialization"],
            "receipt": _file(candidate_dir / row["alias"] / "capture.json")}
        return
    path = candidate_dir.parent / "native" / row["alias"] / "initialization.json"
    initialization, metadata = _json(path)
    evidence = {"receipt": metadata, "attribution": "missing", "used_for_current_concerns": False}
    row["candidate_initialization"] = evidence
    if not isinstance(initialization, dict):
        return
    evidence["raw_receipt"] = initialization
    inventory, inventory_meta = _json(inventory_path)
    started, duration = receipt.get("started"), receipt.get("wall_time_s")
    interval_available = type(started) in (int, float) and type(duration) in (int, float) and duration >= 0
    checks = {
        "capture_identity": receipt.get("id") == row["alias"],
        "current_scene_hash": receipt.get("scene_sha256") == row["files"]["scene"].get("sha256"),
        "current_source_hash": receipt.get("source_sha256") == row["files"].get("scene_source", {}).get("sha256"),
        "capture_interval_available": interval_available,
        "initialization_time_in_capture": interval_available and started <= metadata["modified_at_unix"] <= started + duration,
        "inventory_time_in_capture": interval_available and inventory_meta.get("exists", False)
            and started <= inventory_meta["modified_at_unix"] <= started + duration,
        "inventory_identity": isinstance(inventory, dict) and inventory.get("id") == row["alias"],
    }
    measured = {obj["name"]: obj for obj in inventory.get("objects", [])} if isinstance(inventory, dict) else {}
    initial_objects = initialization.get("objects")
    checks["exact_initial_object_inventory_binding"] = isinstance(initial_objects, dict) and bool(initial_objects) and all(
        name in measured and value.get("aabb") == measured[name].get("aabb")
        and value.get("support") == measured[name].get("support")
        for name, value in initial_objects.items() if isinstance(value, dict)) and all(isinstance(value, dict) for value in initial_objects.values()) if isinstance(initial_objects, dict) else False
    evidence.update(matching_checks=checks, capture_interval_unix=[started, started + duration] if interval_available else None,
                    inventory_receipt=inventory_meta)
    matched = all(checks.values())
    evidence.update(attribution="matched_capture_interval_and_exact_inventory" if matched else "independent_unsealed",
                    used_for_current_concerns=matched)
    if matched:
        _known_concerns(row, initialization, path)
    else:
        _issue(row, "candidate_initialization_independent_unsealed", "Shared native initialization evidence cannot be attributed to the selected capture with matching current provenance, time, and measured object inventory; raw flags are retained independently.",
               kind="independent_evidence_information", file=path)


def _capture(row, item=None, candidate_dir=None):
    manifest, manifest_meta, validation, resolution = None, None, None, None
    if candidate_dir is not None:
        directory = candidate_dir / row["alias"]
        path = directory / "capture.json"
        manifest_path = None
    elif item is not None:
        path = _path(item["receipt_file"])
        manifest_path = path.parents[2] / "input/manifest.json"
        manifest, manifest_meta = _json(manifest_path)
        if isinstance(manifest, dict):
            validation = _validate_current_sealed_inputs(row, manifest)
            path, resolution = _latest_complete_receipt(row, item, path, manifest, validation)
            if validation["valid"] and row.get("task_source_identity") != row.get("scene_source_identity"):
                row["issues"] = [issue for issue in row["issues"] if issue["code"] != "task_scene_source_identity_disagreement"]
                _issue(row, "declared_separate_source_roles", "Separate QA-task and native-scene source roles are declared and validated by exact current sealed hashes and bind_capture_inputs; neither source is rewritten.",
                       kind="declared_source_role_information", file=manifest_path, pointer="/capture_source_members")
        directory = path.parent
    else:
        _issue(row, "capture_mapping_missing", "No exact-alias capture mapping is available.")
        row["capture"] = {"state": "missing", "observed_at_utc": _utc()}
        _coverage(row, None)
        return
    receipt, metadata = _json(path)
    row["capture"] = {"receipt": metadata, "state": "missing", "observed_at_utc": _utc(),
                      "capture_is_task_success": False, "runtime_currentness": "not_compared"}
    if item is not None:
        row["capture"].update(receipt_resolution=resolution, sealed_input_binding=validation)
    _coverage(row, directory)
    if candidate_dir is not None:
        _candidate_initialization(row, receipt if isinstance(receipt, dict) else {}, candidate_dir, directory / "inventory.json")
    if not isinstance(receipt, dict):
        _issue(row, "capture_receipt_unavailable", "The selected capture receipt is missing or unreadable; a capture may still be running.", file=path)
        return
    captured = row["capture"]
    captured.update(receipt_status=receipt.get("status"), complete=receipt.get("complete"),
                    code_commit=receipt.get("code_commit"), started=receipt.get("started"),
                    diagnostic_pending=receipt.get("diagnostic_pending"),
                    native_qualification_run=receipt.get("native_qualification_run"),
                    model_calls=receipt.get("model_calls"), robot_actions=receipt.get("robot_actions"))
    images = receipt.get("images")
    if isinstance(images, dict):
        paths = {name: Path(value) for name, value in images.items() if isinstance(value, str)}
    else:
        paths = {name: directory / (name + ".png") for name in receipt.get("views", []) if isinstance(name, str)}
    captured["images"] = {name: {"file": str(image), "exists": image.is_file()} for name, image in paths.items()}
    present = sum(image["exists"] for image in captured["images"].values())
    captured.update(image_count=len(paths), image_files_present=present)
    rendered = bool(paths) and present == len(paths) and (receipt.get("complete") is True or receipt.get("status") == "captured")
    captured["state"] = "rendered" if rendered else "incomplete"
    if not rendered:
        _issue(row, "capture_not_complete", "The mapped receipt and image-file presence do not establish a complete render set.", file=path)
    if receipt.get("status") == "error":
        _issue(row, "capture_reported_error", str(receipt.get("error", "Capture reported an error.")), kind="capture_error", file=path, pointer="/error")
    _mismatch(row, receipt.get("id") if candidate_dir is not None else None, row["alias"], path, "/id", "capture_identity_stale")
    for key, expected in (("scene_sha256", row["files"]["scene"].get("sha256")),
                          ("source_sha256", row["files"].get("scene_source", {}).get("sha256")),
                          ("seed", row.get("seed")), ("variant", row.get("variant"))):
        _mismatch(row, receipt.get(key), expected, path, "/" + key, "capture_" + key + "_stale")
    if item is not None:
        for key in ("task", "scene"):
            _mismatch(row, item.get(key + "_sha256"), row["files"][key].get("sha256"), None, None, "review_index_" + key + "_stale")
        captured["manifest"] = manifest_meta
        if isinstance(manifest, dict):
            hashes = manifest.get("input_hashes", {})
            for observed, expected, pointer, code in (
                (manifest.get("task_sha256"), row["files"]["task"].get("sha256"), "/task_sha256", "frozen_task_stale"),
                (hashes.get("scene.yaml"), row["files"]["scene"].get("sha256"), "/input_hashes/scene.yaml", "frozen_scene_stale"),
                (hashes.get("data/" + row.get("scene_source_identity", "")), row["files"].get("scene_source", {}).get("sha256"), "/input_hashes/" + ("data/" + row.get("scene_source_identity", "")).replace("/", "~1"), "frozen_source_stale"),
                (receipt.get("capture_signature"), manifest.get("capture_signature"), "/capture_signature", "receipt_signature_stale"),
                (receipt.get("code_commit"), manifest.get("code_commit"), "/code_commit", "receipt_code_stale"),
                (receipt.get("seed"), manifest.get("seed"), "/seed", "receipt_seed_stale"),
                (receipt.get("variant"), manifest.get("variant"), "/variant", "receipt_variant_stale"),
            ):
                _mismatch(row, observed, expected, manifest_path if code.startswith("frozen_") else path, pointer, code)
        else:
            _issue(row, "capture_manifest_unavailable", "The immutable job manifest is unavailable for provenance comparison.", file=manifest_path)
    else:
        captured["provenance_gaps"] = [key for key in ("task_sha256", "seed", "variant", "code_commit") if key not in receipt]
    _known_concerns(row, receipt, path)
    captured["initial_support_concern_count"] = sum(issue["kind"] == "initial_support_concern" for issue in row["issues"])
    if any(issue["kind"] == "stale_evidence" for issue in row["issues"]):
        captured["state"] = "stale"


def _selected_tasks(release, candidates):
    """Resolve the selected aliases before inspecting definitions or evidence."""
    selections = []
    aliases = set()
    unified = release.get("unified_selection") is True
    catalogues = (("release", release),) if unified else (("release", release), ("candidate", candidates))
    for catalogue_group, catalogue in catalogues:
        for selected in catalogue["tasks"]:
            alias = selected["id"] if catalogue_group == "candidate" else selected["alias"]
            if alias in aliases:
                raise ValueError(f"Duplicate selected alias: {alias!r}")
            aliases.add(alias)
            group = "candidate" if (catalogue_group == "candidate"
                or unified and selected.get("origin", "original") != "original") else "release"
            selections.append((group, alias, selected))
    return selections


def _supplementary_items(paths):
    """Map exact supplementary aliases once; duplicate mappings are an error."""
    items = {}
    for path in paths:
        index = json.loads(path.read_text())
        if (not isinstance(index, dict) or index.get("schema") != "wia_capture_review_index_v1"
                or not isinstance(index.get("items"), list)):
            raise ValueError("Supplementary capture input must be an explicit review index.")
        for item in index["items"]:
            if not isinstance(item, dict) or not isinstance(item.get("document_alias"), str):
                raise ValueError("Every supplementary capture item must name its exact document_alias.")
            alias = item["document_alias"]
            if not alias or item.get("row_id") != alias:
                raise ValueError("Supplementary row_id must equal its exact document_alias.")
            if alias in items:
                raise ValueError("Duplicate supplementary exact-alias capture mapping: " + alias)
            items[alias] = item
    return items


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-catalogue", type=Path, default=ROOT / "explore_to_achieve/wia_v1/release/catalogue.json")
    parser.add_argument("--candidate-catalogue", type=Path, default=ROOT / "explore_to_achieve/candidates/interactiveworld_20261029/catalogue.json")
    parser.add_argument("--review-index", type=Path, default=ROOT / "outputs/edge_review_20261029/review_index.json")
    parser.add_argument("--candidate-review-index", type=Path,
                        default=ROOT / "outputs/edge_review_20261029/interactiveworld_candidate_receipts/review_index.json")
    parser.add_argument("--supplementary-review-index", type=Path, action="append",
                        help="Exact supplementary review index; repeat for separate candidate pools. Defaults to the household index when present.")
    parser.add_argument("--candidate-captures", type=Path)
    parser.add_argument("--native-captures", type=Path, action="append", default=[],
                        help="Prefer current native captures for every alias; repeat for separately tracked pilots.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    began = _utc()
    release_path, candidate_path, review_path = map(_path, (args.release_catalogue, args.candidate_catalogue, args.review_index))
    release, candidates, review = [json.loads(path.read_text()) for path in (release_path, candidate_path, review_path)]
    candidate_review_path = _path(args.candidate_review_index)
    candidate_review = json.loads(candidate_review_path.read_text()) if candidate_review_path.is_file() else {}
    supplementary_paths = (list(map(_path, args.supplementary_review_index))
        if args.supplementary_review_index is not None else
        [DEFAULT_SUPPLEMENTARY_INDEX] if DEFAULT_SUPPLEMENTARY_INDEX.is_file() else [])
    supplementary_items = _supplementary_items(supplementary_paths)
    candidate_dir = _path(args.candidate_captures) if args.candidate_captures else None
    rows = []
    for group, alias, selected in _selected_tasks(release, candidates):
        row = {"record_type": "selected_task", "group": group, "alias": alias,
               "family_or_prototype": selected.get("family", selected.get("prototype")),
               "selection": selected, "issues": []}
        _metadata(row, selected)
        native_options = [] if selected.get("origin") == "household_geometry" else [(root, _json(root / alias / "capture.json")[0])
                          for root in map(_path, args.native_captures)
                          if (root / alias / "capture.json").is_file()]
        native_root = next((root for root, receipt in native_options if isinstance(receipt, dict)
            and receipt.get("status") == "captured"
            and receipt.get("scene_sha256") == row["files"]["scene"].get("sha256")
            and receipt.get("source_sha256") == row["files"].get("scene_source", {}).get("sha256")),
            native_options[0][0] if native_options else None)
        if selected.get("origin") == "household_geometry":
            # Newly authored layouts use their exact sealed jobs, never legacy alias filenames.
            _capture(row, item=supplementary_items.get(alias))
        elif native_root is not None:
            _capture(row, candidate_dir=native_root)
        elif group == "candidate" and candidate_dir is not None:
            _capture(row, candidate_dir=candidate_dir)
        else:
            index, index_path = (review, review_path) if group == "release" else (candidate_review, candidate_review_path)
            matches = [item for item in index.get("items", []) if item.get("document_alias") == alias]
            if len(matches) > 1:
                _issue(row, "ambiguous_exact_alias_capture_mapping", "Multiple review-index records have the exact alias; none is selected automatically.", kind="definition_error", file=index_path)
            _capture(row, item=matches[0] if len(matches) == 1 else None)
        rows.append(row)
    excluded = [{**item, "record_type": "excluded", "catalogue_file": str(release_path),
                 "catalogue_json_pointer": f"/excluded/{index}", "task_fabricated": False}
                for index, item in enumerate(release.get("excluded", []))]
    all_issues = [issue for row in rows for issue in row["issues"]]
    summary = {
        "selected_count": len(rows), "groups": dict(Counter(row["group"] for row in rows)),
        "loaded_count": sum(row.get("load_status") == "loaded" for row in rows),
        "definition_error_count": sum(issue["kind"] == "definition_error" for issue in all_issues),
        "aliases_with_definition_errors": [row["alias"] for row in rows if any(issue["kind"] == "definition_error" for issue in row["issues"])],
        "capture_states": dict(Counter(row["capture"]["state"] for row in rows)),
        "issue_counts": dict(Counter(issue["code"] for issue in all_issues)),
        "initial_support_concern_count": sum(issue["kind"] == "initial_support_concern" for issue in all_issues),
        "current_initial_support_issue_counts": dict(Counter(issue["code"] for issue in all_issues if issue["kind"] == "initial_support_concern")),
        "historical_nonterminal_measurement_count": sum(issue["kind"] == "historical_measurement_information" for issue in all_issues),
        "declared_separate_source_role_count": sum(issue["kind"] == "declared_source_role_information" for issue in all_issues),
        "matched_candidate_initialization_count": sum(row.get("candidate_initialization", {}).get("used_for_current_concerns", False) for row in rows),
        "independent_unsealed_candidate_initialization_count": sum(row.get("candidate_initialization", {}).get("attribution") == "independent_unsealed" for row in rows),
        "aliases_with_initial_support_concerns": [row["alias"] for row in rows if any(issue["kind"] == "initial_support_concern" for issue in row["issues"])],
        "measured_inventory_file_count": sum(row["audit_coverage"]["measured_inventory_available"] for row in rows),
        "aliases_lacking_measured_inventory": [row["alias"] for row in rows if not row["audit_coverage"]["measured_inventory_available"]],
        "aliases_lacking_explicit_full_measured_inventory_audit": [row["alias"] for row in rows if not row["audit_coverage"]["full_measured_inventory_declared"]],
        "aliases_lacking_kinematic_audit_report": [row["alias"] for row in rows if not row["audit_coverage"]["kinematic_audit_report_available"]],
        "aliases_lacking_explicit_full_kinematic_audit": [row["alias"] for row in rows if not row["audit_coverage"]["full_kinematic_audit_declared"]],
        "excluded_count": len(excluded), "physical_success_inferred": False,
    }
    report = {"schema": SCHEMA, "started_at_utc": began, "finished_at_utc": _utc(),
              "scope": "Read-only selected definitions and mapped capture evidence; no repair classification or physical qualification.",
              "inputs": {"release_catalogue": _file(release_path), "candidate_catalogue": _file(candidate_path),
                         "review_index": _file(review_path), "candidate_review_index": _file(candidate_review_path),
                         "candidate_captures": str(candidate_dir) if candidate_dir else None,
                         "native_captures": [str(_path(root)) for root in args.native_captures]},
              "summary": summary, "tasks": rows, "excluded": excluded}
    if supplementary_paths:
        report["inputs"]["supplementary_review_indices"] = [_file(path) for path in supplementary_paths]
    output = _path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    columns = ["record_type", "group", "alias", "family_or_prototype", "task_id", "scene_id", "task_status", "scene_status",
               "load_status", "seed", "variant", "scene_model", "room", "module", "operation", "public_objective",
               "task_source_identity", "scene_source_identity", "bindings", "declared_requirements", "files", "layout_parameters",
               "capture", "candidate_initialization", "audit_coverage", "issues", "reason"]
    with (output / "report.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in [*rows, *excluded]:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    print(json.dumps({"output": str(output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
