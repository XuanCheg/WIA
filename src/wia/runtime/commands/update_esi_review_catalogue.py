#!/usr/bin/env python3
"""Publish Review metadata from existing completion messages, without image scans.

Run with behavior Python:
  python src/wia/runtime/commands/update_esi_review_catalogue.py --campaign-root ~/esi_category_conversion/CAMPAIGN
The eight fixed worker logs are consumed incrementally. Category publication
receipts supply declared tasks and CPU failures; screenshots are never opened.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import fcntl
from pathlib import Path
import re
import sys
import time

import yaml

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
from wia.runtime.scene_capture_queue import read_json, write_json


def task_summary(record, status, public=None):
    public = public or {}
    source = record["source_identity"]
    scene = record.get("scene_model") or Path(source).parts[-3]
    small = record["small_task"]
    complete = status.get("state") in ("REVIEW_PENDING", "CACHED")
    row = {
        "id": record["candidate_id"], "candidate_id": record["candidate_id"],
        "name": small + " · " + scene + " / " + Path(source).stem,
        "objective": public.get("objective") or record.get("failure_reason")
                     or "No operation generated; diagnostic guidance is requested.",
        "robot_task_public": public or None, "category": record["big_task"], "subcategory": small,
        "scene": scene, "room": record.get("room") or Path(source).parts[-2],
        "sample": Path(source).stem, "source": source, "question": record.get("question", ""),
        "fingerprint": record.get("fingerprint", record.get("revision", "")),
        "conversion_status": "pending", "conversion_result_status": record["status"],
        "instance_status": "capture_complete" if complete else record["status"] if record["status"] != "generated" else "capture_failed",
        "native_capture_complete": complete, "has_final_preview": False,
        "has_global_preview": complete, "no_native_image": not complete, "images": [],
        "capture": {"state": status.get("state", "REVIEW_ONLY"), "job_id": record.get("job_id")},
        "warning_codes": record.get("warning_codes", []), "failure_stage": status.get("failure_stage") or record.get("failure_stage"),
        "failure": status.get("error") or record.get("failure_reason", ""),
        "aperture_warning": False, "route_confirmed": False, "physical_grasp_verified": False,
        "verification_level": "pending_category_goal", "origin": "current_campaign",
        "target": {"category": small}, "targets": [], "containers": [], "support": None,
        "task_file": record.get("task_file"), "scene_file": record.get("scene_file"),
        "diagnostic_dir": record.get('diagnostic_dir'),
        "converter_version": record.get("converter_version"), "reuse_level": record.get("reuse_level"),
        "variant": record.get("variant", "adapted"), "seed": record.get("seed", 0),
        "owner": record.get("owner"), "parent_candidate_id": record.get("parent_candidate_id"),
        "prior_feedback": record.get('prior_feedback', []),
        "native_recheck_required": bool(record.get('native_recheck_required') and not complete)}
    for field in ('composite', 'category_tags', 'review_goals', 'prior_feedback', 'legacy_review_id'):
        if field in record:
            row[field] = record[field]
    row['progress'] = {'status': 'not_executed', 'completion_score': None,
        'completed_goals': None, 'total_goals': (record.get('composite') or {}).get('member_count', 1)}
    # A screenshot is initial-layout evidence, never an operation score.
    evaluation = record.get('evaluation_file')
    if evaluation:
        measured = read_json(evaluation, {})
        if (measured.get('schema') == 'private_category_task_v1'
                and measured.get('task_id') == record.get('physical_task_id')
                and measured.get('progress', {}).get('status') == 'measured'):
            from wia.tasks.loader import sha256
            if record.get('task_file') and measured.get('task_file_sha256') == sha256(record['task_file']):
                row['progress'] = measured['progress']
                row['progress']['run_mode'] = measured.get('run_mode', 'benchmark')
                row['background_movements'] = measured.get('background_movements', [])
                row['evaluation_failure'] = {key: measured.get('native_result', {}).get(key)
                    for key in ('failure_stage', 'failure_reason')}
    return row


class CataloguePublisher:
    def __init__(self, campaign, queue, output):
        self.campaign, self.queue, self.output = Path(campaign), Path(queue), Path(output)
        self.offsets, self.loaded_batches, self.latest = {}, set(), {}
        self.completed, self.failed, self.retry_jobs = {}, {}, set()
        self.seen_submission_failures = set()
        self.public = {}
        self.last_snapshot = None
        self.groups = read_json(self.campaign / "inventory.json", {}).get("categories", [])

    def _public(self, record):
        path = record.get("task_file")
        if path and record.get("capture_input") and not record.get('goal_revision'):
            path = str(Path(record["capture_input"]).parent / "task.yaml")
        if not path:
            return {}
        if path not in self.public:
            try:
                data = (yaml.safe_load(Path(path).read_text()) or {}).get("public", {})
                self.public[path] = {key: data[key] for key in
                    ("objective", "success_criteria", "environment", "constraints") if key in data}
            except FileNotFoundError:
                self.public[path] = {}
        return self.public[path]

    def consume_capture(self, job):
        directory = self.queue / "jobs" / job
        status = read_json(directory / "status.json", {})
        state = status.get("state")
        if state not in ("REVIEW_PENDING", "CACHED", "FAILED", "STALE", "CANCELLED"):
            self.retry_jobs.add(job)
            return
        manifest = read_json(directory / "input/manifest.json", {})
        policy = read_json(self.queue / 'capture_runtime_policy.json', {})
        allowed = policy.get('review_allowed_code_commits', policy.get('allowed_code_commits', []))
        if manifest.get("code_commit") not in allowed:
            self.retry_jobs.discard(job)
            return
        record = read_json(self.queue / "category_records" / (manifest["candidate_id"] + ".json"))
        if record is None:
            self.retry_jobs.add(job)
            return
        if record.get('reference_only') and not record.get('recapture_review'):
            self.retry_jobs.discard(job)
            return
        # Completion comes from the existing worker, not a second image check.
        row = task_summary(record, status, self._public(record))
        receipt = read_json(status["receipt"], {}) if status.get("receipt") else {}
        setup = receipt.get("interaction_setup") or {}
        row['warning_codes'] = sorted(set(row.get('warning_codes', [])) | set(setup.get('warning_codes', [])))
        rejected = setup.get("accepted") is False
        if rejected:
            row.update(placement_state="needs_repair",
                       placement_warning=setup.get("reason", "Native placement was rejected."),
                       failure=setup.get("reason", "Native placement was rejected."),
                       instance_status="interaction_setup_failed")
        key = row["id"]
        if state in ("REVIEW_PENDING", "CACHED"):
            self.completed[key] = row
            if rejected:
                self.failed[key] = row
            elif record["status"] == "generated":
                self.failed.pop(key, None)
        else:
            self.failed[key] = row
            self.completed.pop(key, None)
        self.retry_jobs.discard(job)

    def consume_publication(self):
        run = read_json(self.campaign / "runtime_capture_refresh/publication_run.json", {})
        revisions = read_json(self.output / 'category_revisions.json', {}).get('categories', {})
        batches = list(run.get('categories', {}).items()) + list(revisions.items())
        for key, result in batches:
            marker = (key, result.get('new_batch_file'))
            if marker in self.loaded_batches or not result.get("complete"):
                continue
            batch = read_json(result["new_batch_file"], {})
            for record in batch.get("records", []):
                self.latest[record["source_identity"]] = record["candidate_id"]
                if record["status"] != "generated" or record.get('native_recheck_required'):
                    published = read_json(self.queue / "category_records" / (record["candidate_id"] + ".json"), record) or record
                    row = task_summary(published, {}, self._public(published))
                    if record.get('native_recheck_required') and record['status'] == 'generated':
                        row.update(instance_status='native_recheck_pending',
                            failure_stage='previous_revision_recheck',
                            failure=record['native_recheck_evidence'].get('reason', 'Previous revision requires a native recheck.'))
                    self.failed[record["candidate_id"]] = row
            self.loaded_batches.add(marker)

    def tick(self):
        self.consume_publication()
        gate = read_json(self.queue / 'capture_sample_gate.json', {})
        progress_file = gate.get('bulk_submission_progress')
        if progress_file:
            progress = read_json(progress_file, {})
            for failure in progress.get('submission_failures', []):
                key = failure['candidate_id']
                if key in self.seen_submission_failures:
                    continue
                record = read_json(self.queue / 'category_records' / (key + '.json'), {})
                if not record or record.get('job_id'):
                    continue
                row = task_summary(record, {}, self._public(record))
                row.update(instance_status='capture_submission_failed',
                    failure_stage=record.get('capture_submission_stage', 'capture_submission'),
                    failure=record.get('capture_submission_error') or failure.get('error'))
                self.failed[key] = row
                self.seen_submission_failures.add(key)
        pattern = re.compile(r"\[capture-worker\] worker\d+ (capture_[0-9a-f]+) (\w+)")
        for number in range(8):
            worker = read_json(self.queue / "workers" / ("worker" + str(number) + ".json"), {})
            if worker.get("job_id"):
                self.retry_jobs.add(worker["job_id"])
            storage = read_json(self.queue / 'category_storage.json', {})
            log_root = Path(storage.get('worker_log_root', self.campaign / 'formal_resident_workers'))
            path = log_root / ("worker" + str(number) + ".log")
            try:
                with path.open() as stream:
                    stream.seek(self.offsets.get(number, 0))
                    while True:
                        start = stream.tell()
                        line = stream.readline()
                        if not line or not line.endswith("\n"):
                            self.offsets[number] = start
                            break
                        match = pattern.search(line)
                        if match:
                            self.consume_capture(match[1])
            except FileNotFoundError:
                pass
        for job in list(self.retry_jobs):
            self.consume_capture(job)
        retired = read_json(self.output / 'retired_candidates.json', {}).get('candidates', {})
        def current(row):
            return (self.latest.get(row["source"], row["id"]) == row["id"]
                    and row['id'] not in retired
                    and row.get('conversion_result_status') not in ('discarded', 'superseded', 'retired', 'merged', 'repeat'))
        goals = read_json(self.output / 'goal_revisions.json', {}).get('records', {})
        def current_goal(row):
            revision = goals.get(row['id'])
            if not revision:
                return dict(row)
            public = self._public(revision)
            return {**row, 'objective': public['objective'], 'robot_task_public': public,
                    'captured_task_file': row.get('task_file'), 'task_file': revision['task_file'],
                    'goal_revision': revision['goal_revision'],
                    'capture_reused': row.get('task_file') != revision['task_file'],
                    **({'fingerprint': revision['goal_revision'], 'task_repair': revision['task_repair'],
                        'preview_validates_current_goal': False, 'progress': row['progress'] if
                        row.get('task_file') == revision['task_file'] and row.get('progress', {}).get('status') == 'measured' else {
                            'status': 'not_executed', 'completion_score': None,
                            'evaluation_validity': 'unverified', 'scoring_version': 'task_completion_v2'},
                        **({'composite': revision['composite']} if revision.get('composite') else {})}
                       if revision.get('task_repair') else {})}
        completed = [current_goal(row) for row in self.completed.values() if current(row)]
        failed = [current_goal(row) for row in self.failed.values() if current(row)]
        conflict_reviews = read_json(self.output / 'task_repair_reviews.json', {}).get('records', {})
        represented = {row['id'] for row in completed + failed}
        for conflict in conflict_reviews.values():
            candidate = conflict.get('candidate_id')
            if candidate in represented:
                continue
            record = read_json(self.queue / 'category_records' / (str(candidate) + '.json'))
            if record is not None and current({'id': candidate, 'source': record['source_identity']}):
                row = current_goal(task_summary(record, {}, self._public(record)))
                row.update(instance_status='source_conflict_review', failure_stage='source_label_review',
                    failure='Saved distance evidence is inconsistent; native identity and metric require review.')
                failed.append(row)
                represented.add(candidate)
        for row in completed + failed:
            if row['source'] in conflict_reviews:
                row['source_conflict_review'] = conflict_reviews[row['source']]
        review_reset = read_json(self.output / 'review_reset.json', {})
        for row in completed + failed:
            if review_reset.get('clear_non_rigid_history') and row['subcategory'] != 'Rigid Containment':
                row['prior_feedback'] = []
                row['human_review_reset'] = True
        gate = read_json(self.queue / 'capture_sample_gate.json', {})
        selected = set(gate.get('selected_candidate_ids', []))
        sample_phase = gate.get('state') == 'ONE_PER_CATEGORY_AWAITING_REVIEW' and not gate.get('selection_pending')
        if sample_phase:
            completed = [row for row in completed if row['id'] in selected]
        for row in completed + failed:
            row['is_category_sample'] = row['id'] in selected
        categories = {(group["big_task"], group["small_task"]): {
            "big_task": group["big_task"], "small_task": group["small_task"],
            "source_count": len(group["sources"]), "report_file": group.get("report_file"),
            "job": {}, "counts": {field: 0 for field in
                ("total", "generated", "captured", "failed", "warned", "reviewed", "approved", "needs_repair", "skipped", "new", "existing")}}
            for group in self.groups}
        for row in {item["id"]: item for item in completed + failed}.values():
            tags = row.get('category_tags') or [{'big_task': row['category'], 'small_task': row['subcategory']}]
            for pair in {(tag['big_task'], tag['small_task']) for tag in tags}:
                group = categories.get(pair)
                if group is None:
                    continue
                counts = group["counts"]
                counts["total"] += 1
                counts["new"] += 1
                counts["generated"] += row["conversion_result_status"] == "generated"
                counts["captured"] += row["native_capture_complete"]
                counts["failed"] += row["id"] in self.failed
                counts["warned"] += bool(row["warning_codes"])
        snapshot = {"schema": "esi_completed_review_catalogue_v1", "updated_at": time.time(),
            "scenes": completed, "diagnostics": failed, "categories": list(categories.values()),
            "indexing": {"status": "ready"}, "capture_queue": {"counts": {
                "REVIEW_PENDING": len(completed), "DIAGNOSTICS": len(failed)}},
            "metadata_only": True, "image_validation": False, "directory_scanning": False}
        snapshot['category_samples'] = {'active': gate.get('state') == 'ONE_PER_CATEGORY_AWAITING_REVIEW',
            'requested': gate.get('requested_total_categories', len(selected)),
            'completed': sum(row['id'] in selected for row in completed),
            'awaiting_user_confirmation': bool(selected) and not bool(gate.get('recapture_campaign_id')),
            'automatic_bulk_release': bool(gate.get('recapture_campaign_id'))}
        snapshot['retired_candidate_count'] = len(retired)
        payload = {key: value for key, value in snapshot.items() if key != "updated_at"}
        if payload != self.last_snapshot:
            write_json(self.output / "snapshot.json", snapshot)
            self.last_snapshot = payload
        return snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, default=ROOT / "outputs/local_capture_20261014/queue_workers_initialized")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--interval-s", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir or args.campaign_root / "review_catalogue"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "publisher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publisher = CataloguePublisher(args.campaign_root, args.queue_root, output)
        while True:
            result = publisher.tick()
            print(f"Completed tasks: {len(result['scenes'])}; diagnostics: {len(result['diagnostics'])}", flush=True)
            if args.once:
                return 0
            time.sleep(max(1, args.interval_s))


if __name__ == "__main__":
    raise SystemExit(main())
