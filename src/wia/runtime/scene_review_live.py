"""Serve completed-task metadata and load one selected scene on demand.

Run src/wia/runtime/commands/review_esi_scenes.py --queue-root QUEUE --output-dir REVIEW
--catalogue-file ~/esi_category_conversion/CAMPAIGN/review_catalogue/snapshot.json.
Legacy views use metadata only. Explicit TaskEdit2 campaign captures verify
their sealed inputs and current image receipt before enabling layout review.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re

from wia.runtime.scene_review import ALL_VIEW_LABELS, DECISIONS, ReviewConflict, SceneReview, now, read_json, write_json
from wia.tasks.sources import resolve_source


def verified_task_edit2_capture(queue_root, record, status, receipt):
    """Bind the new campaign's screenshot completion to its exact sealed inputs."""
    from wia.runtime.scene_capture_queue import CaptureQueue, complete_images, sha256
    try:
        manifest = CaptureQueue(queue_root).manifest(record['job_id'])
        if (manifest.get('candidate_id') != record.get('candidate_id')
                or manifest.get('task_edit2_capture_campaign') is not True
                or manifest.get('source_identity') != record.get('source_identity')):
            raise ValueError('Capture candidate/source identity differs from this Review card.')
        snapshot = Path(manifest['snapshot_dir']).resolve()
        for relative, expected in manifest['input_hashes'].items():
            frozen = (snapshot / relative).resolve()
            if not frozen.is_relative_to(snapshot) or sha256(frozen) != expected:
                raise ValueError('Sealed capture input changed: ' + relative)
        for kind in ('task', 'scene'):
            path = Path(record[kind + '_file']).resolve()
            if (path != Path(manifest[kind + '_file']).resolve()
                    or sha256(path) != manifest['input_hashes'][kind + '.yaml']):
                raise ValueError('Current ' + kind + ' definition differs from the sealed capture.')
        source = Path(record['original_source']).resolve()
        if (source != Path(manifest['source_json']).resolve()
                or sha256(Path(source)) != manifest['input_hashes']['data/' + record['source_identity']]):
            raise ValueError('Current source differs from the sealed capture.')
        if (receipt.get('id') != record.get('candidate_id')
                or receipt.get('capture_job_id') != record.get('job_id')
                or status.get('state') not in ('REVIEW_PENDING', 'CACHED')
                or not complete_images(manifest, receipt)):
            raise ValueError('Complete exact-snapshot screenshot receipt is unavailable.')
        return True, None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return False, str(error)


def _category_tags(row):
    """Return represented categories once, preserving the primary category."""
    tags = [(row.get('category', ''), row.get('subcategory', ''))]
    tags += [(tag.get('big_task', ''), tag.get('small_task', ''))
             for tag in row.get('category_tags', []) if isinstance(tag, dict)]
    return list(dict.fromkeys(tags))


def _declared_public_metadata(row):
    """Allow only the publisher's public composite fields through to Review."""
    composite = row.get('composite')
    if isinstance(composite, dict):
        fields = ('id', 'big_task', 'small_task', 'source', 'objective', 'success_criteria',
                  'constraints', 'original_question', 'original_options')
        row['composite'] = {key: deepcopy(composite[key]) for key in ('group_id', 'member_count', 'sources') if key in composite}
        row['composite']['public_goals'] = []
        for goal in composite.get('public_goals', []):
            if not isinstance(goal, dict):
                continue
            public = goal.get('public') if isinstance(goal.get('public'), dict) else {}
            flattened = {**{k: public[k] for k in ('objective', 'success_criteria', 'constraints')
                            if k in public}, 'source': goal.get('source_identity'), **goal}
            row['composite']['public_goals'].append(
                {key: deepcopy(flattened[key]) for key in fields if key in flattened})
        original_goals = [goal for goal in composite.get('public_goals', []) if isinstance(goal, dict)]
        for public_goal, original_goal in zip(row['composite']['public_goals'], original_goals):
            previous = original_goal.get('previous_capture')
            if isinstance(previous, dict):
                public_goal['previous_capture'] = {key: deepcopy(previous[key]) for key in
                    ('state', 'failure_stage', 'error', 'job_id', 'candidate_id') if key in previous}
            previous_review = original_goal.get('previous_review')
            if isinstance(previous_review, dict):
                public_goal['previous_review'] = {key: deepcopy(previous_review[key]) for key in
                    ('candidate_id', 'decision', 'notes', 'revision', 'updated_at') if key in previous_review}
    if isinstance(row.get('progress'), dict):
        row['progress'] = {key: deepcopy(row['progress'][key]) for key in
            ('status', 'completion_score', 'score_percent', 'completed_goals', 'total_goals',
             'evaluation_validity', 'scoring_version', 'diagnostic_lower_bound', 'run_mode') if key in row['progress']}
    if 'category_tags' in row:
        row['category_tags'] = [{'big_task': big, 'small_task': small} for big, small in _category_tags(row)]
    return row


class LiveSceneReview(SceneReview):
    """Read a published catalogue snapshot instead of constructing a gallery."""

    def __init__(self, *args, catalogue_file, **kwargs):
        super().__init__(*args, **kwargs)
        self.catalogue_file = Path(catalogue_file).expanduser()

    def _snapshot(self):
        return read_json(self.catalogue_file, {
            "scenes": [], "diagnostics": [], "categories": [],
            "indexing": {"status": "refreshing"}, "capture_queue": {}})

    def _annotate(self, row, entries, history):
        row = _declared_public_metadata(deepcopy(row))
        if row.get('human_review_reset'):
            row['prior_feedback'] = []
            for goal in (row.get('composite') or {}).get('public_goals', []):
                goal.pop('previous_review', None)
                goal.pop('previous_capture', None)
        source_decisions = []
        for goal in (row.get('composite') or {}).get('public_goals', []):
            previous_review = goal.get('previous_review')
            if not previous_review:
                continue
            candidate = previous_review.get('candidate_id')
            if candidate in entries:
                previous_review = {'candidate_id': candidate, **{key: deepcopy(entries[candidate][key])
                    for key in ('decision', 'notes', 'revision', 'updated_at') if key in entries[candidate]}}
                goal['previous_review'] = previous_review
            if previous_review.get('decision') not in (None, 'unreviewed'):
                source_decisions.append(previous_review['decision'])
        row['source_review_summary'] = {'is_merged': (row.get('composite') or {}).get('member_count', 0) > 1,
            'reviewed_goals': len(source_decisions),
            'approved': source_decisions.count('approved'), 'needs_fix': source_decisions.count('needs_fix'),
            'skipped': source_decisions.count('skipped')}
        review = entries.get(row["id"], {"decision": "unreviewed", "notes": "", "revision": 0})
        stale = bool(review.get("fingerprint") and review["fingerprint"] != row["fingerprint"])
        previous = [item["review"] for item in history if item["id"] == row["id"]]
        if entries.get(row.get("parent_candidate_id")):
            previous.append(entries[row["parent_candidate_id"]])
        previous.extend(row.get('prior_feedback', []))
        row.update(review=review, review_stale=stale, review_history=previous,
                   decision="unreviewed" if stale else review["decision"],
                   human_approved=not stale and review["decision"] == "approved",
                   layout_approval_available=True)
        row['review_checks'] = {} if stale else deepcopy(review.get('review_checks', {}))
        return row

    def catalogue(self, *, filters=None, page=None, page_size=48, **unused):
        snapshot = self._snapshot()
        reviews = read_json(self.review_file, {"entries": {}})
        filters = filters or {}
        grouped = 'core_groups' in snapshot
        scope = filters.get("scope", "all" if grouped else "completed")
        allowed_scopes = ("completed", "diagnostics", "all", "additional") if grouped else ("completed", "diagnostics", "all")
        if scope not in allowed_scopes:
            raise ValueError("Unknown catalogue scope.")
        canonical = {row['id']: row for row in snapshot.get('scenes', []) + snapshot.get('diagnostics', [])}
        complete_ids = {row['id'] for row in snapshot.get('scenes', [])}
        failed_ids = {row['id'] for row in snapshot.get('diagnostics', [])}
        core_rows = []
        if grouped:
            for group in snapshot['core_groups']:
                default_id = group['default_variant_id']
                if default_id not in canonical:
                    raise KeyError(default_id)
                # Annotate before replacing the ID: reviews belong to actual versions.
                row = self._annotate(canonical[default_id], reviews['entries'], reviews.get('history', []))
                parts = group['source_identity'].split('/')
                if len(parts) != 5:
                    raise ValueError('Core group source identity must have five parts.')
                row.update(id=group['id'], category=parts[0], subcategory=parts[1],
                           scene=parts[2], room=parts[3], source=group['source_identity'],
                           sample=Path(parts[4]).stem,
                           name=f'{parts[1]} · {parts[2]} / {Path(parts[4]).stem}',
                           category_tags=[], core_group=deepcopy(group))
                core_rows.append(row)
        requested_id = filters.get('id')
        if requested_id in canonical:
            source = [canonical[requested_id]]
            rows = [self._annotate(row, reviews['entries'], reviews.get('history', [])) for row in source]
        elif grouped:
            if scope == 'additional':
                rows = [self._annotate(canonical[key], reviews['entries'], reviews.get('history', []))
                        for key in dict.fromkeys(snapshot.get('additional_variant_ids', [])) if key in canonical]
            else:
                rows = [deepcopy(row) for row in core_rows if scope == 'all' or
                        row['core_group']['default_variant_id'] in
                        (complete_ids if scope == 'completed' else failed_ids)]
        else:
            source = snapshot.get('diagnostics', []) if scope == 'diagnostics' else snapshot.get('scenes', [])
            if scope == 'all' or requested_id:
                source = list(canonical.values())
            rows = [self._annotate(row, reviews['entries'], reviews.get('history', []))
                    for row in {item['id']: item for item in source}.values()]
        for row in rows:
            row['images'] = []
        collection_rows = deepcopy(rows)
        total = len(rows)
        requested_big, requested_small = filters.get('big_task'), filters.get('small_task')
        rows = [row for row in rows if any(
            (requested_big in (None, '', 'all') or big == requested_big) and
            (requested_small in (None, '', 'all') or small == requested_small)
            for big, small in _category_tags(row))]
        if filters.get('capture_status') in ('ready', 'pending'):
            complete = filters['capture_status'] == 'ready'
            rows = [row for row in rows if bool(row.get('native_capture_complete')) == complete]
        fields = {"id": "id",
                  "converter_version": "converter_version", "reuse_level": "reuse_level",
                  "review_status": "decision", "failure_stage": "failure_stage",
                  "scene_model": "scene", "conversion_status": "conversion_status"}
        for name, field in fields.items():
            if name == 'review_status' and filters.get(name) == 'merged_reviewed':
                rows = [row for row in rows if row['source_review_summary']['is_merged']
                        and row['source_review_summary']['reviewed_goals']]
                continue
            if filters.get(name) not in (None, "", "all"):
                rows = [row for row in rows if str(row.get(field, "")) == filters[name]]
        options = {field: sorted({str(row[field]) for row in rows if row.get(field)})
                   for field in ("converter_version", "failure_stage", "scene")}
        options["warning_codes"] = sorted({w for row in rows for w in row.get("warning_codes", [])})
        warning = filters.get("warning")
        if warning not in (None, "", "all"):
            rows = [row for row in rows if (bool(row.get("warning_codes")) if warning == "any"
                    else warning in row.get("warning_codes", []))]
        if filters.get("search"):
            term = filters["search"].lower()
            rows = [row for row in rows if term in " ".join(str(row.get(field, "")) for field in
                    ("name", "objective", "scene", "room", "sample", "question", "category", "subcategory")).lower()]
        rows.sort(key=lambda row: (row["category"], row["subcategory"],
            row.get("decision") != "unreviewed" if filters.get("sort", "attention") == "attention" else False,
            row.get("source", "")))
        matched = len(rows)
        if page is not None:
            page, page_size = max(0, page), min(200, max(1, page_size))
            rows = rows[page * page_size:(page + 1) * page_size]
        if grouped:
            groups = []
            group_index = {}
            for item in core_rows:
                tag = (item['category'], item['subcategory'])
                if tag not in group_index:
                    entry = {'big_task': tag[0], 'small_task': tag[1], 'source_count': 0,
                             'counts': {field: 0 for field in
                                        ('captured', 'failed', 'reviewed', 'approved', 'needs_repair', 'skipped')}}
                    groups.append(entry)
                    group_index[tag] = entry
                entry = group_index[tag]
                entry['source_count'] += 1
                default_id = item['core_group']['default_variant_id']
                entry['counts']['captured'] += default_id in complete_ids
                entry['counts']['failed'] += default_id in failed_ids
                decision = item['decision']
                entry['counts']['reviewed'] += decision != 'unreviewed'
                for field, value in (('approved', 'approved'), ('needs_repair', 'needs_fix'), ('skipped', 'skipped')):
                    entry['counts'][field] += decision == value
        else:
            groups = deepcopy(snapshot.get("categories", []))
            group_index = {(group["big_task"], group["small_task"]): group for group in groups}
            all_rows = {item['id']: item for item in snapshot.get('scenes', []) + snapshot.get('diagnostics', [])}
            complete_ids = {item['id'] for item in snapshot.get('scenes', [])}
            failed_ids = {item['id'] for item in snapshot.get('diagnostics', [])}
            for item in all_rows.values():
                for big, small in _category_tags(item):
                    if (big, small) not in group_index:
                        group = {'big_task': big, 'small_task': small, 'source_count': 0, 'counts': {}}
                        groups.append(group)
                        group_index[(big, small)] = group
            for group in groups:
                for field in ("captured", "failed", "reviewed", "approved", "needs_repair", "skipped"):
                    group["counts"][field] = 0
            for item in all_rows.values():
                review = reviews["entries"].get(item["id"], {})
                for tag in _category_tags(item):
                    group = group_index[tag]
                    group['counts']['captured'] += item['id'] in complete_ids
                    group['counts']['failed'] += item['id'] in failed_ids
                    if review.get('fingerprint') != item['fingerprint']:
                        continue
                    decision = review.get("decision", "unreviewed")
                    group["counts"]["reviewed"] += decision != "unreviewed"
                    for field, value in (("approved", "approved"), ("needs_repair", "needs_fix"), ("skipped", "skipped")):
                        group["counts"][field] += decision == value
        capture_queue = snapshot.get("capture_queue", {})
        return {"scenes": rows, "categories": groups,
                "collection": scope,
                "core_grouped": grouped, "core_group_count": len(core_rows),
                "additional_variant_count": len(set(snapshot.get('additional_variant_ids', []))),
                "filter_options": options, "total": total, "matched": matched,
                "page": page, "page_size": page_size, "indexing": snapshot.get("indexing", {"status": "ready"}),
                "capture_queue": capture_queue, "capture": {},
                "category_samples": snapshot.get('category_samples', {}),
                "pending_capture": {}, "preparation": {}, "generated_at": now(),
                "review_file": str(self.review_file), "conversion_root": None}

    def scene_detail(self, key, *, variant_id=None):
        snapshot = self._snapshot()
        group = next((item for item in snapshot.get('core_groups', []) if item['id'] == key), None)
        if group is not None:
            active = variant_id if variant_id is not None else group['default_variant_id']
            if active not in {version['id'] for version in group['versions']}:
                raise KeyError(active)
            row = self.scene_detail(active)
            row['core_group'] = {**deepcopy(group), 'active_variant_id': active}
            return row
        if variant_id is not None:
            raise KeyError(variant_id)
        if not re.fullmatch(r"category_[0-9a-f]+", key):
            raise KeyError(key)
        snapshot = self._snapshot()
        summary = next((row for row in snapshot.get("scenes", []) + snapshot.get("diagnostics", [])
                        if row["id"] == key), None)
        if summary is None:
            raise KeyError(key)
        record_key = key
        record = read_json(self.queue_root / "category_records" / (record_key + ".json"))
        if record is None and summary.get('diagnostic_dir'):
            record = read_json(Path(summary['diagnostic_dir']) / 'record.json')
        if record is None:
            raise KeyError(key)
        status = read_json(self.queue_root / "jobs" / record["job_id"] / "status.json", {}) if record.get("job_id") else {}
        receipt = read_json(status["receipt"], {}) if status.get("receipt") else {}
        complete = status.get("state") in ("REVIEW_PENDING", "CACHED") and bool(receipt.get("complete"))
        if complete and record.get('task_edit2_capture_campaign') is True:
            complete, binding_error = verified_task_edit2_capture(self.queue_root, record, status, receipt)
            if not complete:
                status = {**status, 'failure_stage': 'capture_receipt_binding', 'error': binding_error}
        baseline = bool(summary.get('original_baseline') or record.get('original_baseline'))
        check_files = 'core_groups' in snapshot or baseline
        if baseline:
            complete = False
        if complete and check_files:
            paths = receipt.get('images', {})
            complete = bool(paths) and all(Path(path).is_file() for path in paths.values())
            if not complete:
                status = {**status, 'failure_stage': 'capture_images_missing',
                          'error': 'One or more explicitly declared capture images are missing.'}
        evidence = []
        if complete:
            evidence += [{"path": path, "view": view,
                          "kind": "human_floorplan" if view == "global_floorplan" else "native",
                          "label": ALL_VIEW_LABELS.get(view, view)}
                         for view, path in receipt.get("images", {}).items()]
        else:
            evidence += [{"path": path, "view": view, "kind": "partial",
                          "label": "Partial capture · " + view}
                         for view, path in status.get("partial_images", {}).items()]
        # Old screenshots are intentionally excluded. Segmentation diagrams remain
        # human-only metadata attachments and are not native capture evidence.
        evidence += [item for item in record.get("diagnostic_images", [])
                     if item.get("kind") == "human_floorplan"
                     or (record.get('protected_rigid') and item.get('kind') == 'protected_historical')
                     or (item.get("kind") == "partial" and item.get("capture_job_id") == record.get("job_id"))]
        if baseline:
            # Only publisher-declared context is eligible; never discover old folders.
            evidence += [deepcopy(item) for item in
                         record.get('diagnostic_images', []) + summary.get('diagnostic_images', [])
                         + summary.get('images', [])
                         if isinstance(item, dict) and item.get('path') and item.get('view')]
        # Read only this task's explicitly numbered attempt metadata. No directory
        # enumeration or screenshot inspection is needed to retain a failed try.
        if record.get("job_id"):
            for attempt in range(1, int(status.get("attempts", 0)) + 1):
                failure = read_json(self.queue_root / "jobs" / record["job_id"] / "attempts"
                                    / f"{attempt:02}" / "capture_failure.json", {})
                evidence += [{"path": path, "view": view, "kind": "partial",
                              "label": f"Partial capture from attempt {attempt} · {view}"}
                             for view, path in failure.get("partial_images", {}).items()
                             if path not in {item["path"] for item in evidence}]
        if baseline:
            evidence = [{**item, 'kind': 'historical',
                         'label': 'Original/historical context · ' + item.get('label', item['view'])}
                        for item in evidence]
        unique = {item["path"]: item for item in evidence
                  if not check_files or Path(item['path']).is_file()}
        images = []
        with self.lock:
            if check_files:
                # A previously selected then deleted path must not remain registered.
                for image_key in [image_key for image_key in self.images if image_key[0] == key]:
                    del self.images[image_key]
            for index, item in enumerate(unique.values()):
                view = item["view"] if item["kind"] == "native" else "diagnostic_" + str(index)
                self.images[(key, view)] = Path(item["path"])
                images.append({"view": view, "original_view": item["view"], "kind": item["kind"],
                    "label": item.get("label", item["view"]),
                    "url": f"images/{key}/{view}?v={receipt.get('capture_signature', summary['fingerprint'])[:16]}"})
        logs = list(record.get("diagnostic_logs", []))
        if status.get("native_log"):
            logs.append(status["native_log"])
        elif status.get("native_output"):
            logs.append(str(Path(status["native_output"]) / "native.log"))
        row = {**summary, "images": images, "diagnostic_logs": sorted(set(logs)),
               "has_global_preview": complete and any(v.startswith("global_") for v in receipt.get("images", {})),
               "native_capture_complete": complete, "no_native_image": not complete,
               "original_baseline": baseline,
               "capture": {**{k: status.get(k) for k in
                   ("state", "attempts", "duration_s", "failure_stage", "error", "gpu")},
                   "job_id": record.get("job_id"), "timings": receipt.get("timings")},
               "preview_provenance": receipt.get("provenance", "Recorded diagnostic evidence."),
               "target_location": receipt.get("target_location"),
               "target_locations": receipt.get("target_locations", []),
               'temporal_episode_preview': receipt.get('temporal_episode_preview'),
               'human_review_cameras': receipt.get('human_review_cameras'),
               "preview_is_current": complete and (not summary.get("task_repair") or
                    record.get('task_file') == summary.get('task_file')), "diagnostic_pending": receipt.get("diagnostic_pending", True),
               "viewer_command": "", "physical_qualified": False}
        for name in ('composite', 'category_tags', 'review_goals', 'progress'):
            if name not in row and name in record:
                row[name] = deepcopy(record[name])
        for name in ("assumptions", "change_summary", "attempted_repairs", "next_step",
                     "diagnostic_dir", "exception_summary", "prior_conversion_failures",
                     "code_commit", "owner", "parent_candidate_id", "task_file", "scene_file"):
            if name in record:
                if name == 'task_file' and summary.get('goal_revision'):
                    continue
                row[name] = record[name]
        row["known_limitations"] = record.get("limitations", [])
        if summary.get('goal_revision'):
            row['captured_change_summary'] = record.get('change_summary', [])
            row['change_summary'] = ['Goal revision: ' + summary['objective'],
                ('Fresh screenshots show the current initialization; they do not certify task execution.'
                 if row['preview_is_current'] else
                 'Existing screenshots describe the retained layout; they do not validate the new goal or scoring.')]
            if summary.get('task_repair') and not row['preview_is_current']:
                row['preview_provenance'] = 'Historical native layout evidence. New scoring and active survey are not certified by these images.'
        setup = receipt.get("interaction_setup") or {}
        setup_error = setup.get("reason", "Native placement was rejected.") if setup.get("accepted") is False else ""
        row["capture"]["interaction_setup"] = setup
        row["failure"] = (status.get("error") or setup_error or record.get("failure_reason")
            or record.get("capture_submission_error", "") or (record.get('native_recheck_evidence', {}).get('reason', '')
                if not complete else ''))
        row["failure_stage"] = status.get("failure_stage") or record.get("failure_stage")
        reviews = read_json(self.review_file, {"entries": {}})
        row = self._annotate(row, reviews["entries"], reviews.get("history", []))
        # Original answers are human Review detail only. Read the source JSON,
        # never the converted task's scorer or Agent-facing public payload.
        source_root = WIA_RUNTIME_ROOT / 'ESI/dataset/json_clean'
        original_source = record.get('original_source')
        source_identity = record.get('source_identity')
        answers = {}

        def source_answer(source):
            if source not in answers:
                path = (Path(original_source) if original_source and source == source_identity
                        else source_root / source) if source else None
                try:
                    if path and not path.is_file():
                        path = resolve_source(source)
                    answers[source] = read_json(path, {}).get('answer') if path else None
                except (OSError, ValueError):
                    answers[source] = None
            return answers[source]

        row['original_esi_answer'] = source_answer(source_identity)
        for goal in (row.get('composite') or {}).get('public_goals', []):
            goal['original_esi_answer'] = source_answer(goal.get('source'))
        return row

    def diagnostic_log(self, key, index):
        paths = self.scene_detail(key)["diagnostic_logs"]
        if not 0 <= index < len(paths):
            raise KeyError(index)
        with Path(paths[index]).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 65536))
            return stream.read().decode("utf-8", errors="replace")

    def save_review(self, key, payload):
        # The existing campaign coordinator consumes review changes separately;
        # saving a note does not reconstruct the source corpus or Gallery.
        if not re.fullmatch(r"category_[0-9a-f]+", key):
            raise KeyError(key)
        with self.lock:
            rows = self.catalogue(filters={"id": key})["scenes"]
            if not rows:
                raise KeyError(key)
            row = rows[0]
            decision, notes = payload.get("decision"), payload.get("notes", "")
            if decision not in DECISIONS or not isinstance(notes, str) or len(notes) > 10000:
                raise ValueError("Provide a valid decision and at most 10,000 characters.")
            if payload.get("fingerprint") != row["fingerprint"]:
                raise ReviewConflict("The task changed. Refresh before saving.")
            if payload.get("revision", 0) != row["review"].get("revision", 0):
                raise ReviewConflict("This review changed in another tab. Refresh before saving.")
            if decision == "approved":
                # Human review is available for every version, including
                # historical or incomplete captures. Preserve measured status.
                row = self.scene_detail(key)
            checks = payload.get('review_checks', row.get('review_checks', {}))
            goals = row.get('review_goals')
            if goals is None:
                record = read_json(self.queue_root / 'category_records' / (key + '.json'), {})
                goals = record.get('review_goals', [])
            allowed_indices = {str(index) for index in range(len(goals))}
            if (not isinstance(checks, dict) or any(index not in allowed_indices or type(value) is not bool
                                                   for index, value in checks.items())):
                raise ValueError('Review checks must use existing goal indices and boolean values.')
            entry = {name: row.get(name) for name in
                     ("fingerprint", "source", "scene_file", "variant", "seed", "candidate_id",
                      "owner", "parent_candidate_id", "converter_version")}
            if row.get('goal_revision'):
                entry['goal_revision'] = row['goal_revision']
            entry.update(decision=decision, notes=notes, revision=row["review"].get("revision", 0) + 1,
                         updated_at=now(), big_task=row["category"], small_task=row["subcategory"])
            if checks or 'review_checks' in payload or 'review_checks' in row['review']:
                entry['review_checks'] = deepcopy(checks)
            if decision == 'approved':
                entry['approval_scope'] = 'layout'
                entry['approval_evidence'] = {
                    'native_capture_complete': bool(row.get('native_capture_complete')),
                    'original_baseline': bool(row.get('original_baseline')),
                    'capture_job_id': row.get('capture', {}).get('job_id')}
            value = read_json(self.review_file, {"schema": "scene_reviews_v1", "entries": {}})
            if value["entries"].get(key):
                value.setdefault("history", []).append({"id": key, "review": value["entries"][key]})
            value["entries"][key] = entry
            write_json(self.review_file, value)
            return entry
