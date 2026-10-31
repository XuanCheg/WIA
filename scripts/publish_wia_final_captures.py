"""Plan or publish the local 61-baseline plus 19-candidate screenshot review.

Read-only validation: ``PYTHONPATH=src python scripts/publish_wia_final_captures.py --dry-run``.
After review: ``PYTHONPATH=src python scripts/publish_wia_final_captures.py --apply --watch``.
--campaign defaults to outputs/edge_review_20261029; --candidate-receipts defaults
to its interactiveworld_candidate_receipts directory. --watch requires --apply
and polls every 15 seconds until all active jobs are terminal, including exhausted
or nonretryable failures. No worker, queue operation, network request, registry
change, release admission, or feedback edit is performed. Only --apply writes
derived files/progress in final_refresh_review and the existing local snapshot.
Original curation may declare temporarily_excluded_aliases and candidate_row_metadata;
these control active visibility and candidate presentation without changing captures.
Optional supplementary_catalogues entries name a catalogue_file and review_index_file;
their pending household rows and audit evidence stay separate from the frozen pools.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import yaml

from wia.resources import ROOT
from wia.authoring.commands.build_wia_capture_review import build_supplement, publish_supplement
from wia.authoring.commands.build_wia_review import (PUBLIC_FIELDS, _PublicLoader, build_snapshot,
                                                    _safe_path, _unique_json_pairs, publish_snapshot,
                                                    validate_excluded_aliases)
from wia.runtime.scene_capture_queue import write_json


REVISION = "wia_final_capture_review_v1"
DEFAULT_CAMPAIGN = ROOT / "outputs/edge_review_20261029"
SNAPSHOT = ROOT.parent / "Show-Harness-Explore/outputs/wia_v1_20261023/review/snapshot.json"
INTERVAL_S = 15


def _path(value, project_root=ROOT):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compose_curation(curation, candidate_items, candidate_supplement, *, project_root=ROOT):
    """Copy curation and add exact candidate variants with optional authored taxonomy."""
    derived = deepcopy(curation)
    if (derived.get("schema") != "wia_review_evidence_v1" or not isinstance(derived.get("row_updates", {}), dict)
            or not isinstance(derived.get("extra_rows", []), list)):
        raise ValueError("The original curation must retain its declared evidence schema.")
    extras = derived.setdefault("extra_rows", [])
    existing_ids = {row.get("id", row.get("wia_id")) for row in extras}
    aliases = [item["document_alias"] for item in candidate_items]
    if len(aliases) != 19 or len(set(aliases)) != 19:
        raise ValueError("Provide exactly 19 distinct current candidate aliases.")
    metadata = derived.get("candidate_row_metadata", {})
    allowed_metadata = {"title", "primary_family", "primary_subclass", "summary", "capability_tags"}
    if not isinstance(metadata, dict) or set(metadata) - set(aliases):
        raise ValueError("candidate_row_metadata must map only archived candidate aliases to presentation fields.")
    for alias, fields in metadata.items():
        if not isinstance(fields, dict) or set(fields) - allowed_metadata:
            raise ValueError(f"Candidate metadata for {alias} contains unsupported fields.")
        for field, value in fields.items():
            if field == "capability_tags":
                if not isinstance(value, list) or any(not isinstance(tag, str) for tag in value):
                    raise ValueError("Candidate capability_tags must be a list of strings.")
            elif not isinstance(value, str):
                raise ValueError(f"Candidate metadata {field} must be a string.")
        if "primary_family" in fields and fields["primary_family"] not in {"AP", "NT", "GF", "SR", "GG", "TM"}:
            raise ValueError("Candidate primary_family must use an existing WIA family.")
    if existing_ids & set(aliases):
        raise ValueError("Candidate IDs already exist in original curation; original extras cannot be overwritten.")
    if set(candidate_supplement["row_updates"]) != set(aliases):
        raise ValueError("Candidate supplement must name the exact candidate aliases, never registry row IDs.")
    for item in candidate_items:
        alias = item["document_alias"]
        if item["row_id"] != alias:
            raise ValueError("Each candidate index row_id must equal its document_alias.")
        task_path = _path(item["task_file"], project_root)
        task = yaml.load(task_path.read_text(), Loader=_PublicLoader)
        if task.get("id") != alias or task.get("status") != "pending":
            raise ValueError("Each candidate must retain its exact pending task definition.")
        public = task.get("public")
        if not isinstance(public, dict) or not isinstance(public.get("objective"), str) or not public["objective"].strip():
            raise ValueError("The current candidate task must supply its public objective.")
        evaluation = task["private"]["evaluation"]
        parameters = evaluation["parameters"]
        prototype, condition = parameters["prototype"], parameters["condition"]
        if evaluation.get("module") != "wia.authoring.converters.interactiveworld_candidates" or evaluation.get("operation") != prototype:
            raise ValueError("Candidate prototype must match its current owned evaluator.")
        variants = candidate_supplement["row_updates"][alias]["variants"]
        if not isinstance(variants, list) or not variants:
            raise ValueError("Every candidate must retain its supplement variants, including pending captures.")
        extra = {
            "id": alias, "document_alias": alias,
            "title": str(prototype) + " " + json.dumps(condition, sort_keys=True, separators=(",", ":"), allow_nan=False),
            "primary_family": None, "primary_subclass": None,
            "proposal_membership": False, "release_admitted": False,
            "status": "pending", "implementation_state": "implemented_pending_review",
            "summary": "Pending candidate screenshot review only; no task success, physical qualification, or release admission is inferred.",
            "public": {field: deepcopy(public[field]) for field in PUBLIC_FIELDS if field in public},
            "variants": deepcopy(variants),
        }
        extra.update(deepcopy(metadata.get(alias, {})))
        extras.append(extra)
    return derived


def compose_household_curation(curation, catalogue, items, supplement, *, project_root=ROOT):
    """Add the four exact pending household tasks without changing either frozen pool."""
    if (not isinstance(catalogue, dict) or catalogue.get("schema") != "wia_household_geometry_candidates_v1"
            or catalogue.get("release_admitted") is not False):
        raise ValueError("Supplementary catalogue must declare unadmitted household geometry candidates.")
    selections = catalogue.get("tasks")
    expected = {"GF2-1", "GF2-2", "GF3-1", "GF3-2"}
    if (not isinstance(selections, list) or len(selections) != 4
            or any(not isinstance(item, dict) for item in selections)
            or {item.get("alias") for item in selections} != expected):
        raise ValueError("Supplementary catalogue must contain the exact four household aliases.")
    if (len(items) != 4 or {item.get("document_alias") for item in items} != expected
            or set(supplement["row_updates"]) != expected):
        raise ValueError("Supplementary index must contain one exact record per household alias.")
    identities = [selected.get("task_id") for selected in selections]
    if any(not isinstance(identity, str) or not identity for identity in identities) or len(set(identities)) != 4:
        raise ValueError("Household catalogue task identities must be nonempty and distinct.")
    derived = deepcopy(curation)
    extras = derived.setdefault("extra_rows", [])
    occupied = {row.get("id", row.get("wia_id")) for row in extras}
    occupied.update(row.get("document_alias") for row in extras)
    if occupied & expected:
        raise ValueError("Supplementary household rows cannot overwrite existing curation.")
    indexed = {item["document_alias"]: item for item in items}
    release_root = _path("explore_to_achieve/wia_v1/release", project_root)
    for selected in selections:
        alias, identity = selected["alias"], selected["task_id"]
        subclass = "GF-2" if alias.startswith("GF2-") else "GF-3"
        if (selected.get("id") != identity or selected.get("variant") != "adapted"
                or selected.get("origin") != "household_geometry" or selected.get("family") != "GF"
                or selected.get("subclass") != subclass or selected.get("physical_success_verified") is not False
                or not isinstance(selected.get("title"), str) or not selected["title"].strip()):
            raise ValueError("Household catalogue identity, taxonomy, and pending evidence must agree.")
        item = indexed[alias]
        task_path = _safe_path(selected["task_file"], Path(project_root).resolve())
        scene_path = _safe_path(selected["scene_file"], Path(project_root).resolve())
        if (task_path.is_relative_to(release_root) or scene_path.is_relative_to(release_root)
                or item.get("row_id") != alias or item.get("variant_id") != identity
                or _safe_path(item.get("task_file"), Path(project_root).resolve()) != task_path
                or _safe_path(item.get("scene_file"), Path(project_root).resolve()) != scene_path
                or item.get("task_sha256") != _sha(task_path) or item.get("scene_sha256") != _sha(scene_path)):
            raise ValueError("Supplementary index must identify the exact nonrelease household definitions and hashes.")
        task = yaml.load(task_path.read_text(), Loader=_PublicLoader)
        scene = yaml.load(scene_path.read_text(), Loader=_PublicLoader)
        if (not isinstance(task, dict) or not isinstance(scene, dict)
                or task.get("id") != identity or task.get("kind") != "category_robot_task"
                or task.get("status") != "pending" or scene.get("status") != "pending"
                or task.get("private", {}).get("evaluation", {}).get("module") != "wia.authoring.converters.wia_household_fit"):
            raise ValueError("Household task identity, pending definitions, and evaluator must agree.")
        variant = selected["variant"]
        task_initialization = task.get("scene", {}).get("variants", {}).get(variant)
        scene_initialization = scene.get("scene", {}).get("variants", {}).get(variant)
        task_seed = task.get("defaults", {}).get("seed")
        if (task_initialization is None or task_initialization != scene_initialization
                or task_seed is None or task_seed != scene.get("seed")
                or task.get("private", {}).get("bindings") != scene.get("bindings")):
            raise ValueError("Household task and scene must share exact seed, initialization, and bindings.")
        public = task.get("public")
        if not isinstance(public, dict) or not isinstance(public.get("objective"), str) or not public["objective"].strip():
            raise ValueError("Every household task must supply its current public objective.")
        variants = supplement["row_updates"][alias]["variants"]
        if not isinstance(variants, list) or len(variants) != 1 or variants[0]["id"] != identity:
            raise ValueError("Every household task must retain its exact supplementary capture variant.")
        extras.append({"id": alias, "document_alias": alias, "title": selected["title"],
            "primary_family": selected["family"], "primary_subclass": selected["subclass"],
            "proposal_membership": False, "release_admitted": False, "physical_success_verified": False,
            "status": "pending", "implementation_state": "implemented_pending_review",
            "summary": "Pending household layout review only; no task success, physical qualification, or release admission is inferred.",
            "public": {field: deepcopy(public[field]) for field in PUBLIC_FIELDS if field in public},
            "variants": deepcopy(variants)})
    return derived


def job_progress(items, *, project_root=ROOT):
    """Read exact indexed job state and immutable retry limits without modifying a queue."""
    rows = []
    for group, item in items:
        status_path = _path(item["queue_status_file"], project_root)
        status = _read(status_path) if status_path.is_file() else {}
        state = status.get("state", "MISSING")
        attempts, retryable = status.get("attempts", 0), status.get("retryable", False)
        manifest_path = status_path.parent / "input/manifest.json"
        manifest = _read(manifest_path) if manifest_path.is_file() else {}
        maximum = manifest.get("max_attempts")
        valid_limit = type(maximum) is int and maximum >= 1
        valid_status = (status.get("job_id") == item["job_id"] and status_path.parent.name == item["job_id"]
                        and type(attempts) is int and attempts >= 0 and type(retryable) is bool)
        if manifest and manifest.get("job_id") != item["job_id"]:
            valid_status = False
        terminal = valid_status and (state in {"REVIEW_PENDING", "CACHED", "STALE", "CANCELLED"}
            or state == "FAILED" and (not retryable or valid_limit and attempts >= maximum))
        rows.append({"group": group, "alias": item["document_alias"], "job_id": item["job_id"], "state": state,
            "attempts": attempts, "retryable": retryable, "max_attempts": maximum,
            "terminal": bool(terminal), "status_identity_valid": valid_status,
            "error": status.get("error"), "receipt": status.get("receipt"), "status_file": str(status_path)})
    return {"total": len(rows), "states": dict(Counter(row["state"] for row in rows)), "rows": rows,
            "all_terminal": bool(rows) and all(row["terminal"] for row in rows)}


def _active_review_summary(supplement, items):
    """Summarize only the exact variants selected by nonexcluded indexed jobs."""
    variants = [variant for item in items
                for variant in supplement["row_updates"][item["row_id"]]["variants"]
                if variant["id"] == item["variant_id"]]
    return {"indexed_variant_count": len(variants), "row_count": len({item["row_id"] for item in items}),
            "variant_status_counts": dict(sorted(Counter(variant["status"] for variant in variants).items()))}


def prepare(campaign, candidate_receipts, *, project_root=ROOT):
    campaign, candidate_receipts = Path(campaign), Path(candidate_receipts)
    registry_path, curation_path = campaign / "registry.json", campaign / "curation.json"
    baseline_index_path, candidate_index_path = campaign / "review_index.json", candidate_receipts / "review_index.json"
    input_paths = (("registry", registry_path), ("original_curation", curation_path),
                   ("baseline_index", baseline_index_path), ("candidate_index", candidate_index_path))
    inputs = {name: {"file": str(path), "sha256": _sha(path)} for name, path in input_paths}
    registry, curation = _read(registry_path), _read(curation_path)
    baseline_index, candidate_index = _read(baseline_index_path), _read(candidate_index_path)
    baseline_items, candidate_items = baseline_index["items"], candidate_index["items"]
    if len(baseline_items) != 61 or len({item["document_alias"] for item in baseline_items}) != 61:
        raise ValueError("The baseline index must retain exactly 61 distinct selections.")
    registry_aliases = {row["document_alias"]: row["wia_id"] for row in registry["rows"]}
    if set(registry_aliases) != {item["document_alias"] for item in baseline_items} or any(
            registry_aliases[item["document_alias"]] != item["row_id"] for item in baseline_items):
        raise ValueError("The baseline index must match the original registry aliases and row IDs exactly.")
    if len(candidate_items) != 19 or len({item["job_id"] for item in baseline_items + candidate_items}) != 80:
        raise ValueError("Track exactly 61 baseline plus 19 candidate jobs with distinct immutable job identities.")
    baseline = build_supplement(baseline_index_path, project_root=project_root)
    candidate = build_supplement(candidate_index_path, project_root=project_root)
    derived = compose_curation(curation, candidate_items, candidate, project_root=project_root)
    supplementary = []
    declarations = curation.get("supplementary_catalogues", [])
    if not isinstance(declarations, list):
        raise ValueError("supplementary_catalogues must be a list of explicit catalogue/index mappings.")
    used_paths, used_jobs = {path.resolve() for _, path in input_paths}, {item["job_id"] for item in baseline_items + candidate_items}
    for number, declaration in enumerate(declarations, 1):
        if not isinstance(declaration, dict) or set(declaration) != {"catalogue_file", "review_index_file"}:
            raise ValueError("Each supplementary_catalogues entry requires catalogue_file and review_index_file only.")
        catalogue_path = _safe_path(declaration["catalogue_file"], Path(project_root).resolve())
        index_path = _safe_path(declaration["review_index_file"], Path(project_root).resolve())
        for name, path in ((f"supplementary_catalogue_{number}", catalogue_path),
                           (f"supplementary_index_{number}", index_path)):
            if path in used_paths:
                raise ValueError("Supplementary catalogue/index inputs must be distinct from all frozen inputs.")
            used_paths.add(path)
            inputs[name] = {"file": str(path), "sha256": _sha(path)}
        catalogue = json.loads(catalogue_path.read_text(), object_pairs_hook=_unique_json_pairs)
        index = json.loads(index_path.read_text(), object_pairs_hook=_unique_json_pairs)
        if (not isinstance(index, dict) or not isinstance(index.get("items"), list)
                or any(not isinstance(item, dict) for item in index["items"])):
            raise ValueError("Supplementary review index must declare its exact items list.")
        items = index["items"]
        jobs = [item.get("job_id") for item in items]
        if (any(not isinstance(job, str) or not job for job in jobs)
                or len(set(jobs)) != len(jobs) or used_jobs & set(jobs)):
            raise ValueError("Supplementary jobs must have distinct identities outside both frozen pools.")
        used_jobs.update(jobs)
        supplement = build_supplement(index_path, project_root=project_root)
        if set(registry_aliases) & set(supplement["row_updates"]):
            raise ValueError("Supplementary aliases cannot overlap the frozen registry.")
        derived = compose_household_curation(derived, catalogue, items, supplement, project_root=project_root)
        supplementary.append({"catalogue_file": str(catalogue_path), "review_index_file": str(index_path),
                              "items": items, "supplement": supplement})
    additional_items = []
    for number, declared in enumerate(curation.get('additional_capture_indices', []), 1):
        path = _safe_path(declared, Path(project_root).resolve())
        if path in used_paths:
            raise ValueError('Additional capture indexes must be distinct inputs.')
        used_paths.add(path)
        inputs[f'additional_capture_index_{number}'] = {'file': str(path), 'sha256': _sha(path)}
        items = _read(path)['items']
        supplement = build_supplement(path, project_root=project_root)
        extras = {row.get('id'): row for row in derived['extra_rows']}
        for item in items:
            row = extras.get(item['row_id'])
            if (row is None or row.get('document_alias') != item['document_alias']
                    or item['job_id'] in used_jobs):
                raise ValueError('Additional captures must match distinct explicitly authored extra rows.')
            variants = supplement['row_updates'][item['row_id']]['variants']
            expected = {v['id']: v for v in row['variants']}
            for variant in variants:
                original = expected.get(variant['id'])
                if original is None or any(_safe_path(original[k], Path(project_root).resolve()) !=
                        _safe_path(variant[k], Path(project_root).resolve()) for k in ('task_file', 'scene_file')):
                    raise ValueError('Additional capture definitions differ from the authored extra row.')
            row['variants'] = deepcopy(variants)
            used_jobs.add(item['job_id'])
        additional_items.extend(items)
    known_aliases = [*registry_aliases, *(row.get("document_alias", row.get("id", row.get("wia_id")))
                                        for row in derived["extra_rows"])]
    excluded_aliases = validate_excluded_aliases(derived.get("temporarily_excluded_aliases", []), known_aliases)
    excluded = set(excluded_aliases)
    active_baseline = [item for item in baseline_items if item["document_alias"] not in excluded]
    active_candidates = [item for item in candidate_items if item["document_alias"] not in excluded]
    active_jobs = [*(('baseline', item) for item in active_baseline),
                   *(('candidate', item) for item in active_candidates)]
    active_supplementary = [[item for item in entry["items"] if item["document_alias"] not in excluded]
                            for entry in supplementary]
    active_jobs.extend(("supplementary", item) for items in active_supplementary for item in items)
    active_jobs.extend(('additional', item) for item in additional_items if item['document_alias'] not in excluded)
    progress = job_progress(active_jobs, project_root=project_root)
    progress.update(baseline_count=len(active_baseline), candidate_count=len(active_candidates),
        active_job_count=len(active_jobs), temporarily_excluded_aliases=excluded_aliases,
        baseline_review=_active_review_summary(baseline, active_baseline),
        candidate_review=_active_review_summary(candidate, active_candidates),
        review_summary_scope="active_nonexcluded_indexed_jobs",
        archived_baseline_count=len(baseline_items), archived_candidate_count=len(candidate_items),
        archived_baseline_review=baseline["summary"], archived_candidate_review=candidate["summary"],
        derived_extra_row_count=sum(row.get("document_alias", row.get("id", row.get("wia_id"))) not in excluded
                                    for row in derived["extra_rows"]), physical_success_inferred=False,
        release_admitted=False, local_review_only=True)
    if supplementary:
        progress.update(supplementary_count=sum(map(len, active_supplementary)),
            archived_supplementary_count=sum(len(entry["items"]) for entry in supplementary),
            supplementary_reviews=[{"catalogue_file": entry["catalogue_file"],
                "review_index_file": entry["review_index_file"], "active_count": len(items),
                "archived_count": len(entry["items"]),
                "review": _active_review_summary(entry["supplement"], items),
                "archived_review": entry["supplement"]["summary"]}
                for entry, items in zip(supplementary, active_supplementary)])
    if any(_sha(value["file"]) != value["sha256"] for value in inputs.values()):
        raise ValueError("Review inputs changed during composition; retry with the current exact job selections.")
    prepared = {"derived_curation": derived, "baseline_supplement": baseline, "candidate_supplement": candidate,
                "progress": progress, "inputs": inputs}
    if supplementary:
        prepared["supplementary_catalogues"] = supplementary
    return prepared


def publish_prepared(prepared, campaign, *, project_root=ROOT):
    """Write only derived review artifacts and the existing local screenshot snapshot."""
    campaign = Path(campaign)
    destination = campaign / "final_refresh_review"
    derived_path = destination / "curation.json"
    baseline_path = destination / "baseline_capture_evidence.json"
    candidate_path = destination / "candidate_capture_evidence.json"
    protected = [Path(value["file"]).resolve() for value in prepared["inputs"].values()]
    supplementary = prepared.get("supplementary_catalogues", [])
    audit_paths = [destination / f"supplementary_capture_evidence_{number:02d}.json"
                   for number in range(1, len(supplementary) + 1)]
    if any(path.resolve() in protected for path in (derived_path, baseline_path, candidate_path, SNAPSHOT, *audit_paths)):
        raise ValueError("Derived review outputs must never overwrite catalogue, index, or other review inputs.")
    if any(_sha(value["file"]) != value["sha256"] for value in prepared["inputs"].values()):
        raise ValueError("Review inputs changed after composition; prepare again before publishing.")
    write_json(derived_path, prepared["derived_curation"])
    publish_supplement(baseline_path, prepared["baseline_supplement"], project_root=project_root, protected_paths=protected)
    # Retain candidate evidence for audit only. It is never a registry supplement.
    publish_supplement(candidate_path, prepared["candidate_supplement"], project_root=project_root, protected_paths=protected)
    for entry, path in zip(supplementary, audit_paths):
        publish_supplement(path, entry["supplement"], project_root=project_root, protected_paths=protected)
    snapshot = build_snapshot(campaign / "registry.json", derived_path, supplements=[baseline_path], project_root=project_root)
    aliases = {row["document_alias"]: row for row in snapshot["rows"]}
    excluded = set(snapshot["provenance"]["temporarily_excluded_aliases"])
    candidate_aliases = set(prepared["candidate_supplement"]["row_updates"]) - excluded
    candidate_aliases.update(alias for entry in supplementary for alias in entry["supplement"]["row_updates"] if alias not in excluded)
    if excluded & set(aliases):
        raise ValueError("Temporarily excluded aliases must stay outside the active review snapshot.")
    if not candidate_aliases <= set(aliases) or any(aliases[alias].get("proposal_membership") is not False
            or aliases[alias].get("release_admitted") is not False for alias in candidate_aliases):
        raise ValueError("Candidate rows must remain outside proposal and release admission.")
    for resolution in prepared['derived_curation'].get('review_resolutions', []):
        if (resolution.get('new_physical_success_claim') is not False
                or resolution.get('additional_validation_required') is not False):
            raise ValueError('Human review closure must not imply a new physical certificate.')
        for alias in resolution['aliases']:
            if alias not in aliases:
                raise ValueError('Review closure names an inactive alias: ' + alias)
            row = aliases[alias]
            row['review_resolution'] = deepcopy(resolution)
            row['summary'] = resolution['reason'] + ' Additional validation is not pending. Historical evidence is preserved.'
            if row['variants']:
                row['variants'][0]['evidence_summary'] = (
                    resolution['reason'] + ' This is human review closure, not a new physical-success claim.')
    publish_snapshot(SNAPSHOT, snapshot, project_root=project_root)
    result = {"snapshot": str(SNAPSHOT), "snapshot_row_count": len(snapshot["rows"]),
            "derived_curation": str(derived_path), "baseline_supplement": str(baseline_path),
            "candidate_evidence_audit_file": str(candidate_path), "snapshot_updated": True}
    if supplementary:
        result["supplementary_evidence_audit_files"] = [str(path) for path in audit_paths]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--candidate-receipts", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Read-only composition and progress check; the default.")
    modes.add_argument("--apply", action="store_true", help="Update the local review snapshot and separate derived artifacts.")
    parser.add_argument("--watch", action="store_true", help="With --apply, update every 15 s until all active jobs are terminal.")
    args = parser.parse_args(argv)
    if args.watch and not args.apply:
        parser.error("--watch requires --apply; default dry-run never writes or watches.")
    campaign = _path(args.campaign)
    candidate_receipts = _path(args.candidate_receipts) if args.candidate_receipts else campaign / "interactiveworld_candidate_receipts"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    last_states = None
    while True:
        try:
            prepared = prepare(campaign, candidate_receipts)
            report = {"schema": REVISION, "run_id": run_id, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "apply" if args.apply else "dry_run", "inputs": prepared["inputs"], **prepared["progress"],
                "snapshot": str(SNAPSHOT), "snapshot_updated": False,
                "derived_directory": str(campaign / "final_refresh_review"),
                "candidate_supplement_passed_to_registry_merge": False,
                "original_registry_curation_feedback_modified": False}
            if args.apply:
                report.update(publish_prepared(prepared, campaign))
                write_json(campaign / "final_refresh_review/progress.json", report)
                if report["all_terminal"]:
                    write_json(campaign / "final_refresh_review/completion.json", report)
            if not args.watch or report["all_terminal"]:
                print(json.dumps(report, indent=2, allow_nan=False))
                return 0
            if report["states"] != last_states:
                print(json.dumps({"run_id": run_id, "states": report["states"], "baseline_review": report["baseline_review"],
                                  "candidate_review": report["candidate_review"], "snapshot_updated": True}), flush=True)
                last_states = report["states"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            report = {"schema": REVISION, "run_id": run_id, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "apply" if args.apply else "dry_run", "error": f"{type(error).__name__}: {error}",
                "snapshot": str(SNAPSHOT), "snapshot_updated": False, "all_terminal": False,
                "original_registry_curation_feedback_modified": False}
            if args.apply:
                write_json(campaign / "final_refresh_review/progress.json", report)
            print(json.dumps(report, indent=2), file=sys.stderr, flush=True)
            if not args.watch:
                return 1
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    raise SystemExit(main())
