"""Summarize one read-only geometry-audit state snapshot as JSON and CSV.

Run: python scripts/summarize_wia_geometry_audit.py --state PATH --output DIR
The script reads the state and its exact input/report references; it never runs an audit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


def _read_json(path):
    return json.loads(Path(path).read_text())


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _dimensions(box):
    try:
        if len(box) == 2 and all(len(row) == 3 for row in box):
            dimensions = [float(box[1][i]) - float(box[0][i]) for i in range(3)]
            if all(math.isfinite(v) and v >= 0 for v in dimensions):
                return dimensions
    except (TypeError, ValueError):
        pass
    return None


def _category(reason, *, status=None, ineligible=False):
    text = str(reason).lower()
    if ineligible or "rigid_grasp_proxy_eligible:false" in text or "different grasp representation" in text:
        return "rigid_proxy_unavailable"
    if "aperture" in text or "aabb_width_exceeds" in text:
        return "aperture_proxy_rejection"
    if any(token in text for token in ("no_connected", "initial_base", "lattice", "start_to", "navigation.failure")):
        return "navigation_sampling/no_connectivity"
    if ":ik_residual" in text:
        return "orientation_aware_ik_residual"
    if "ik_attempt_budget" in text or "aggregate ik budget" in text:
        return "bounded_ik_budget_exhausted"
    if "environment:" in text or ":self:" in text:
        return "collision_proxy_rejection"
    if "coarse_shoulder" in text:
        return "coarse_kinematic_screen"
    if "overlap" in text and ("payload" in text or "initial-state" in text):
        return "retained_object_overlap_prefilter"
    if "not evaluated" in text or "untested" in text or "not_evaluated" in text:
        return "bounded_sampling_not_evaluated"
    if "no geometric witness" in text or "unresolved" in text:
        return "bounded_search_unresolved"
    if status == "unavailable":
        return "data_or_representation_unavailable"
    return "other_reported_rejection"


def _stance(row, navigation=None):
    sampling = row.get("stance_sampling") or {}
    fields = ("method", "raw_proposal_counts", "raw_proposal_count", "deduplicated_proposal_count",
              "pose_clear_counts_by_source", "connected_feasible_counts_by_source", "connected_feasible_count",
              "selected_counts_by_source", "selected_count", "max_stances")
    result = {key: sampling[key] for key in fields if key in sampling}
    result["sampled_stance_count"] = row.get("sampled_stance_count")
    if navigation:
        result["navigation_failure"] = navigation.get("failure")
        result["navigation_mode"] = navigation.get("mode")
        result["reachable_state_count"] = navigation.get("reachable_state_count")
        result["reachable_cell_count"] = navigation.get("reachable_cell_count")
    return result


def _query(row, kind, pointer, inventory, navigation):
    samples = row.get("sample_records") or []
    evidence = []
    reasons = []
    category_counts = Counter()
    category_reason_counts = Counter()
    sample_records = []

    def collect(result, reference):
        counts = result.get("rejection_counts") or {}
        evidence.append({"json_pointer": reference, "rejection_counts": counts})
        for reason, count in counts.items():
            if _number(count):
                category_counts[_category(reason, status=result.get("status"))] += count
        if result.get("reason"):
            reason = {"json_pointer": reference + "/reason", "reason": result["reason"]}
            reason["category"] = _category(result["reason"], status=result.get("status"),
                                            ineligible=result.get("rigid_grasp_proxy_eligible") is False)
            reasons.append(reason)
            category_reason_counts[reason["category"]] += 1

    if samples:
        # Aggregate destination fields mirror one sample. Count each sample once.
        for index, sample in enumerate(samples):
            reference = f"{pointer}/sample_records/{index}"
            result = sample.get("result") or {}
            collect(result, reference + "/result")
            sample_records.append({
                "sample_index": sample.get("sample_index", index), "json_pointer": reference,
                "status": sample.get("status"), "ik_attempts": sample.get("ik_attempts"),
                "approach_evaluated": sample.get("approach_evaluated"),
                "payload_center_world_m": sample.get("payload_center_world_m"),
                "aabb": sample.get("aabb"), "aabb_dimensions_m": _dimensions(sample.get("aabb")),
                "retained_object_aabb_overlaps": sample.get("retained_object_aabb_overlaps", []),
                "permitted_payload_support_contacts": sample.get("permitted_payload_support_contacts", []),
                "stance_availability": _stance(result, sample.get("navigation")),
                "witness": result.get("witness") is not None,
                "kinematic_witness": result.get("kinematic_witness") is not None,
                "reason": result.get("reason"), "rejection_counts": result.get("rejection_counts", {}),
            })
        # Keep the exact aggregate reason without counting the mirrored selected sample twice.
        if row.get("reason"):
            reasons.append({"json_pointer": pointer + "/reason", "reason": row["reason"],
                            "category": _category(row["reason"], status=row.get("status")),
                            "aggregate_only": True})
    else:
        collect(row, pointer)
    categories = set(category_counts) | set(category_reason_counts)
    top = [{"category": category, "rejection_count": category_counts[category],
            "reason_record_count": category_reason_counts[category]} for category in categories]
    top.sort(key=lambda item: (-(item["rejection_count"] + item["reason_record_count"]), item["category"]))
    source = next((obj for obj in inventory.get("objects", []) + inventory.get("targets", [])
                   if obj.get("name") == row.get("name")), {})
    payload = row.get("hypothetical_payload") or {}
    dimensions = row.get("target_aabb_dimensions_m")
    dimension_source = "report.target_aabb_dimensions_m" if dimensions is not None else None
    if dimensions is None:
        dimensions = _dimensions((row.get("input_metadata") or {}).get("aabb") or payload.get("aabb") or source.get("aabb"))
        if dimensions is not None:
            dimension_source = "input_metadata_or_exact_payload_or_input_object_aabb"
    return {
        "name": row.get("name"), "kind": kind, "status": row.get("status"), "json_pointer": pointer,
        "ik_attempts": row.get("ik_attempts"), "ik_attempts_complete": row.get("ik_attempts_complete"),
        "elapsed_s": row.get("elapsed_s"), "aabb_dimensions_m": dimensions,
        "aabb_dimensions_source": dimension_source, "region_aabb_dimensions_m": _dimensions(row.get("region_aabb")),
        "stance_availability": _stance(row, row.get("navigation") or navigation),
        "witness": row.get("witness") is not None, "kinematic_witness": row.get("kinematic_witness") is not None,
        "rigid_grasp_proxy_eligible": row.get("rigid_grasp_proxy_eligible", source.get("rigid_grasp_proxy_eligible")),
        "policy_metadata": row.get("policy_metadata"), "policy_permission_inferred": row.get("policy_permission_inferred"),
        "reason": row.get("reason"), "rejection_counts": row.get("rejection_counts", {}),
        "rejection_counter_evidence": evidence, "reason_evidence": reasons,
        "rejection_category_counts": dict(sorted(category_counts.items())),
        "reason_category_record_counts": dict(sorted(category_reason_counts.items())),
        "top5_rejection_categories": top[:5], "sample_records": sample_records,
        "sample_status_counts": row.get("sample_status_counts"), "sample_count": row.get("sample_count"),
        "sampled_only_limitation": row.get("sampled_only_limitation"), "scope": row.get("scope"),
        "selected_sample_index": row.get("selected_sample_index"),
    }


def _actions(report):
    # Preserve explicit raw evidence only. False command execution does not measure model calls.
    return {key: value for key, value in report.items()
            if key in {"robot_commands_executed", "robot_action_count", "robot_actions_executed",
                       "robot_commands_executed_count", "model_action_count", "model_call_count",
                       "model_calls", "model_calls_executed"}}


def _runtime(values):
    values = [value for value in values if _number(value)]
    return {"count": len(values), "sum_s": sum(values),
            "median_s": statistics.median(values) if values else None,
            "max_s": max(values) if values else None}


def _select_job(jobs):
    if not jobs:
        return None
    # A completed exact-dispatch attempt takes priority; never inspect another dispatch.
    indexed = list(enumerate(jobs))
    return max(indexed, key=lambda pair: (pair[1].get("status") == "completed", pair[0]))[1]


def summarize(state_path):
    state_path = Path(state_path).resolve()
    raw_state = state_path.read_bytes()
    state = json.loads(raw_state)
    latest = state["latest_inputs"]
    expected = {f"{group}/{alias}/phase_1" for group, aliases in state["scope"].items() for alias in aliases}
    identities = expected | set(latest)
    jobs = defaultdict(list)
    for job in state["jobs"]:
        jobs[(job.get("identity"), job.get("dispatch_key"))].append(job)
    inventories = []
    for identity in sorted(identities):
        group, alias, phase = identity.split("/", 2)
        spec = latest.get(identity)
        matching = jobs.get((identity, spec.get("dispatch_key")), []) if spec else []
        job = _select_job(matching)
        row = {"identity": identity, "group": group, "alias": alias, "phase": phase,
               "expected_phase1": identity in expected, "input_available": spec is not None,
               "state_job_status": job.get("status") if job else "missing_job" if spec else "missing_input",
               "report_status": None, "queries": [], "action_evidence": {},
               "structural_target_count": None, "inactive_target_count": None, "unavailable_target_count": None,
               "matching_attempt_count": len(matching), "matching_attempts": [
                   {k: attempt.get(k) for k in ("status", "job_key_sha256", "output", "started_utc", "finished_utc")}
                   for attempt in matching],
               "phase_coverage": state.get("phase_coverage", {}).get(identity)}
        for key in ("source", "source_tier", "provisional", "provisional_reason", "current_definition_match",
                    "current_capture", "definition", "capture_receipt_sha256", "capture_receipt_copy",
                    "dispatch_key", "raw_input_sha256", "code_sha256", "reference_sha256"):
            row[key] = spec.get(key) if spec else None
        for key in ("raw_input_snapshot", "refined_input_sha256", "audit_input", "output", "job_key_sha256",
                    "returncode", "started_utc", "finished_utc", "wall_elapsed_s", "audit_elapsed_s", "refinement"):
            row[key] = job.get(key) if job else None
        row["code_bundle"] = job.get("code_bundle") if job else None
        row["current_nonprovisional"] = bool(spec and spec.get("current_capture") is True
                                                and spec.get("current_definition_match") is True
                                                and spec.get("provisional") is False)
        report = None
        if job and job.get("output"):
            try:
                report = _read_json(job["output"])
            except (OSError, ValueError) as error:
                row["report_read_error"] = f"{type(error).__name__}: {error}"
        if report is not None:
            row["report_status"] = report.get("status")
            row["report_reason"] = report.get("reason")
            row["audit_elapsed_s"] = report.get("elapsed_s")
            row["action_evidence"] = _actions(report)
            row["report_limitations"] = report.get("limitations")
            row["navigation"] = report.get("navigation")
            row["inventory_accounting_scope"] = report.get("inventory_accounting_scope")
            for key in ("structural", "inactive", "unavailable"):
                accounting = report.get(f"{key}_targets")
                row[f"{key}_targets"] = accounting
                row[f"{key}_target_count"] = len(accounting) if isinstance(accounting, list) else None
            inventory = {}
            if job.get("audit_input"):
                try:
                    inventory = _read_json(job["audit_input"])
                except (OSError, ValueError) as error:
                    row["input_read_error"] = f"{type(error).__name__}: {error}"
            for kind, key in (("initial", "targets"), ("destination", "destination_queries")):
                row["queries"].extend(_query(query, kind, f"/{key}/{index}", inventory, report.get("navigation"))
                                      for index, query in enumerate(report.get(key, [])))
        row["initial_status_counts"] = dict(Counter(q["status"] for q in row["queries"] if q["kind"] == "initial"))
        row["destination_status_counts"] = dict(Counter(q["status"] for q in row["queries"] if q["kind"] == "destination"))
        inventories.append(row)
    phase_totals = []
    for group, phase in sorted({(row["group"], row["phase"]) for row in inventories}):
        rows = [row for row in inventories if (row["group"], row["phase"]) == (group, phase)]
        queries = [query for row in rows for query in row["queries"]]
        phase_totals.append({"group": group, "phase": phase, "identities": len(rows),
            "input_available": sum(row["input_available"] for row in rows),
            "state_job_status_counts": dict(Counter(row["state_job_status"] for row in rows)),
            "report_status_counts": dict(Counter(row["report_status"] or "no_report" for row in rows)),
            "current_nonprovisional_input_count": sum(row["current_nonprovisional"] for row in rows),
            "current_nonprovisional_completed_report_count": sum(row["current_nonprovisional"] and
                row["state_job_status"] == "completed" and row["report_status"] is not None for row in rows),
            "initial_status_counts": dict(Counter(q["status"] for q in queries if q["kind"] == "initial")),
            "destination_status_counts": dict(Counter(q["status"] for q in queries if q["kind"] == "destination")),
            "accounting_counts": {key: sum(row[f"{key}_target_count"] or 0 for row in rows)
                                  for key in ("structural", "inactive", "unavailable")},
            "wall_runtime": _runtime(row["wall_elapsed_s"] for row in rows),
            "audit_runtime": _runtime(row["audit_elapsed_s"] for row in rows)})
    return {"schema": "wia_geometry_audit_summary_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "state_file": str(state_path), "state_sha256": hashlib.sha256(raw_state).hexdigest(),
        "state_last_discovery_utc": state.get("last_discovery_utc"), "state_controller_status": state.get("status"),
        "selection": "Exact latest_inputs identity and dispatch_key; all scope phase_1 identities retained when missing.",
        "interpretation": "Reported statuses describe bounded geometric screening only. No physical impossibility, task repair, carrying, placement or grasp-success conclusion is added.",
        "category_count_units": {"rejection_count": "Sum of exact numeric rejection counters, without destination aggregate/sample duplication.",
                                 "reason_record_count": "Occurrences of explicit per-query/sample reason records; separate from rejection counters."},
        "action_count_policy": "Only explicit raw report action/command/model fields are copied; absent counts remain unknown. robot_commands_executed:false does not measure model calls.",
        "expected_phase1_count": len(expected), "independent_additional_phase_count": len(identities - expected),
        "phase_totals": phase_totals, "wall_runtime": _runtime(row["wall_elapsed_s"] for row in inventories),
        "audit_runtime": _runtime(row["audit_elapsed_s"] for row in inventories),
        "blocked_inputs": state.get("blocked_inputs", {}), "inventories": inventories}


def _csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":"), sort_keys=True)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = summarize(args.state)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    inventories = summary["inventories"]
    inventory_fields = ["identity", "group", "alias", "phase", "input_available", "state_job_status", "report_status",
        "current_nonprovisional", "current_capture", "current_definition_match", "provisional", "provisional_reason",
        "source_tier", "source", "dispatch_key", "job_key_sha256", "raw_input_sha256", "refined_input_sha256",
        "reference_sha256", "code_sha256", "audit_input", "output", "wall_elapsed_s", "audit_elapsed_s",
        "initial_status_counts", "destination_status_counts", "structural_target_count", "inactive_target_count",
        "unavailable_target_count", "action_evidence", "report_read_error"]
    _csv(args.output / "inventories.csv", inventories, inventory_fields)
    query_rows = []
    for inventory in inventories:
        for query in inventory["queries"]:
            query_rows.append({**{key: inventory.get(key) for key in ("identity", "group", "alias", "phase", "output",
                "current_nonprovisional", "provisional", "dispatch_key", "raw_input_sha256", "refined_input_sha256", "code_sha256")}, **query})
    _csv(args.output / "queries.csv", query_rows, ["identity", "group", "alias", "phase", "name", "kind", "status",
        "ik_attempts", "ik_attempts_complete", "elapsed_s", "aabb_dimensions_m", "region_aabb_dimensions_m",
        "stance_availability", "witness", "kinematic_witness", "rigid_grasp_proxy_eligible", "reason",
        "top5_rejection_categories", "sample_count", "sample_status_counts", "current_nonprovisional", "provisional",
        "dispatch_key", "raw_input_sha256", "refined_input_sha256", "code_sha256", "output", "json_pointer"])
    _csv(args.output / "phases.csv", summary["phase_totals"], ["group", "phase", "identities", "input_available",
        "state_job_status_counts", "report_status_counts", "current_nonprovisional_input_count",
        "current_nonprovisional_completed_report_count", "initial_status_counts", "destination_status_counts",
        "accounting_counts", "wall_runtime", "audit_runtime"])
    print(json.dumps({"output": str(args.output.resolve()), "expected_phase1": summary["expected_phase1_count"],
        "additional_phases": summary["independent_additional_phase_count"], "state_job_status_counts": dict(Counter(
            row["state_job_status"] for row in inventories)), "queries": len(query_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
