"""Scene review records and preview provenance.

Run the CPU gallery with src/wia/runtime/commands/review_esi_scenes.py --conversion-root RUN
--output-dir REVIEW. This module imports no simulation packages.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import csv
import hashlib
import io
import json
from pathlib import Path
import shlex
import threading
from datetime import datetime, timezone

import yaml

ROOT = WIA_RUNTIME_ROOT
DECISIONS = ("unreviewed", "approved", "needs_fix", "skipped")
VIEW_LABELS = {
    "global_diagonal": "Global · diagonal",
    "global_target": "Pickup · target",
    "global_overhead": "Global · overhead",
    "global_front": "Global · front",
    "global_back": "Global · back",
    "global_left": "Global · left",
    "global_right": "Global · right",
    "head": "Robot · head",
    "wrist": "Robot · working wrist",
    "scene": "ESI · scene",
}
EXTRA_VIEW_LABELS = {"global_target_extra1": "Pickup · extra 1", "global_target_extra2": "Pickup · extra 2",
                     'phase_1_head': 'Phase 1 · robot head (observation only)',
                     'phase_1_wrist': 'Phase 1 · working wrist (observation only)',
                     'phase_2_head': 'Phase 2 · robot head',
                     'phase_2_wrist': 'Phase 2 · working wrist',
                     'phase_2_global_diagonal': 'Phase 2 · human global view',
                     "global_floorplan": "Room map (segmentation; not a native render)"}
ALL_VIEW_LABELS = {**VIEW_LABELS, **EXTRA_VIEW_LABELS}
from core.robot_cameras import ADDITIONAL_RGB_NAMES, ROBOT_CAMERA_LABELS, RGB_TO_VIEW
ALL_VIEW_LABELS.update({RGB_TO_VIEW[name]: ROBOT_CAMERA_LABELS[name] for name in ADDITIONAL_RGB_NAMES})
ALL_VIEW_LABELS.update({'object_overview': 'Object · group context', 'object_top': 'Object · top',
    'object_side': 'Object · side', **{f'object_detail_{i:02d}': f'Object · detail {i + 1}' for i in range(8)},
    **{f'phase_{phase}_object_{view}': f'Phase {phase} · object {view}'
       for phase in (1, 2) for view in ('overview', 'top', 'side')}})


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def scene_id(source, variant, seed):
    return hashlib.sha256(f"{source}\n{variant}\n{seed}".encode()).hexdigest()[:24]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(scene_path, source_path, variant, seed):
    value = {"scene_sha256": sha256(scene_path), "source_sha256": sha256(source_path),
             "variant": variant, "seed": seed}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def same_layout(first, second):
    """Publication changes status; the physical description must otherwise match."""
    from wia.tasks.schema import SceneDefinition

    first, second = dict(first), dict(second)
    first["status"] = second["status"] = "ready"
    return (SceneDefinition.model_validate(first).model_dump()
            == SceneDefinition.model_validate(second).model_dump())


def matching_native_views(inline, candidate, row):
    """Require exact scene/source identity and verified complete human views."""
    target = inline.get("target_location") or {}
    return (inline.get("scene_sha256") == sha256(candidate)
        and inline.get("source_sha256") == sha256(row["_source_path"])
        and inline.get("source_json") == row["source"]
        and inline.get("variant") == row["variant"] and inline.get("seed") == row["seed"]
        and target.get("camera_capture_protocol") == "target_instance_verified_v3"
        and target.get("visible_target_pixels", 0) >= 64
        and set(VIEW_LABELS) <= set(inline.get("images", {})) <= set(ALL_VIEW_LABELS)
        and all(Path(path).is_file() for path in inline["images"].values()))


class ReviewConflict(ValueError):
    pass


class SceneReview:
    def __init__(self, conversion_root, output_dir, *, data_root=None, variant="tabletop", queue_root=None, campaign_root=None):
        self.conversion_root = Path(conversion_root).resolve() if conversion_root else None
        self.queue_root = Path(queue_root).resolve() if queue_root else None
        self.campaign_root = Path(campaign_root).expanduser().resolve() if campaign_root else None
        self.output = Path(output_dir).resolve()
        self.data_root = Path(data_root or ROOT / "explore_to_achieve/scenes").resolve()
        self.variant = variant
        self.lock = threading.RLock()
        self.output.mkdir(parents=True, exist_ok=True)
        self.review_file = self.output / "reviews.json"
        self.summary_path = self.conversion_root / "run/conversion_summary.json" if self.conversion_root else None
        if not self.queue_root and (self.summary_path is None or not self.summary_path.is_file()):
            raise ValueError("The conversion root must contain run/conversion_summary.json.")
        self.rows = []
        self.images = {}
        self._public_task_cache = {}
        self._category_row_cache = {}

    def _public_task(self, path, fingerprint):
        """Display declared operations without exposing private bindings or scorers."""
        if not path or not Path(path).is_file():
            return None
        key = (str(path), fingerprint)
        if key not in self._public_task_cache:
            public = (yaml.safe_load(Path(path).read_text()) or {}).get('public', {})
            self._public_task_cache[key] = {field: public[field] for field in
                ('objective', 'success_criteria', 'environment', 'constraints') if field in public}
        return self._public_task_cache[key] or None

    def _base_rows(self, *, category_detail_ids=None):
        summary = read_json(self.summary_path, {"results": []}) if self.summary_path else {"results": []}
        rows = []
        results = summary["results"] + summary.get("preserved_scenes", [])
        for result in sorted(results, key=lambda r: r["source"]):
            source = result["source"]
            source_path = (self.data_root / source).resolve()
            scene_path = (self.data_root / result["scene_file"]).resolve() if result.get("scene_file") else source_path.with_suffix(".yaml")
            if not scene_path.is_relative_to(self.data_root):
                raise ValueError("Review scene descriptions must remain inside the copied catalogue.")
            if not source_path.is_relative_to(self.data_root):
                raise ValueError("Review sources must remain inside the copied catalogue.")
            raw = read_json(source_path)
            definition = yaml.safe_load(scene_path.read_text())
            seed = definition.get("seed", summary.get("config", {}).get("seed", 0))
            key = scene_id(source, self.variant, seed)
            metadata = raw.get("metadata_json", {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            bindings = definition.get("bindings", {})
            target = bindings.get(definition.get("target", "target"), {})
            item = metadata.get("containee", {})
            containers = metadata.get("containers", [])
            if isinstance(containers, dict):
                containers = list(containers.values())
            # Review labels use public object identity; source answers and fit
            # experiment outcomes are never included in gallery payloads.
            objects = [{"name": b.get("name"), "category": b.get("category"),
                        "model": b.get("model")} for role, b in bindings.items()
                       if role.startswith("container_")]
            if not objects:
                objects = [{"name": f"container_{i}", "category": b.get("cat"),
                            "model": b.get("model")} for i, b in enumerate(containers)]
            parts = Path(source).parts
            command = shlex.join(["python", "src/wia/runtime/commands/verify_esi_rigid_grasp.py", "--viewer-only",
                "--scene-file", str(scene_path), "--data-root", str(self.data_root),
                "--variant", self.variant, "--scene-seed", str(seed), "--web", "--debug-cameras",
                "--human-preview", "--gripper-control", "--rotation-control", "--trunk-control",
                "--parameterized-actions", "--primitive-profile", "fine", "--macro-execution",
                "metric_target", "--host", "0.0.0.0", "--port", "8615",
                "--output-dir", str(self.output / "live" / key)])
            rows.append({
                "id": key, "source": source, "scene_file": str(scene_path), 'origin': 'existing_catalogue',
                "scene": parts[-3], "room": parts[-2], "sample": Path(source).stem,
                "variant": self.variant, "seed": seed,
                "category": parts[0], "subcategory": parts[1], "reuse_level": result.get("reuse_level", "legacy"),
                "fingerprint": fingerprint(scene_path, source_path, self.variant, seed),
                "conversion_status": definition.get("status", "pending"),
                "question": raw.get("question", ""),
                "target": {"name": target.get("name", "obj_containee"),
                    "category": target.get("category", item.get("cat")),
                    "model": target.get("model", item.get("model")), "scale": item.get("scale")},
                "containers": sorted(objects, key=lambda b: b["name"] or ""),
                "support": result.get("support", bindings.get("pickup_support", {}).get("name", "support_table_top")),
                "verification_level": result.get("verification_level", "not_qualified"),
                "aperture_warning": bool(result.get("aperture_warning")),
                "route_confirmed": bool(result.get("route_connectivity_screen_passed")),
                "physical_grasp_verified": bool(result.get("physical_grasp_verified", False)),
                "failure": result.get("detailed_failure") or result.get("reason", ""),
                "correction": result.get("correction", ""),
                "diagnostic_dir": result.get("latest_diagnostic_dir") or result.get("validation_dir"),
                "viewer_command": command,
                "targets": result.get("targets", []), "task_file": result.get("task_file"),
                "_result": result, "_definition": definition, "_source_path": source_path,
            })
        return rows + self._candidate_rows() + self._category_rows(detail_ids=category_detail_ids)

    def _category_rows(self, *, detail_ids=None, only_ids=None):
        if self.queue_root is None:
            return []
        from wia.runtime.esi_category_capture import collect_diagnostics
        from wia.runtime.scene_capture_queue import CaptureQueue
        queue = CaptureQueue(self.queue_root)
        rows = []
        for path in sorted((self.queue_root / 'category_records').glob('*.json')):
            if only_ids is not None and path.stem not in only_ids:
                continue
            record = read_json(path)
            state = queue.status(record['job_id']) if record.get('job_id') else None
            state = state or {'state': 'REVIEW_ONLY'}
            cache_key = (hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
                         state.get('state'), state.get('updated_at'))
            cached = self._category_row_cache.get(record['candidate_id'])
            cacheable = state['state'] not in ('RUNNING', 'CLAIMED')
            if cached and cached[0] == cache_key and cacheable:
                rows.append(dict(cached[1]))
                continue
            detailed = detail_ids is None or record['candidate_id'] in detail_ids
            if detailed:
                diagnostic = collect_diagnostics(record, self.queue_root)
            else:
                # Worker completion states provide category counts. Pending
                # category tasks remain ineligible for approval. Verify actual
                # image bytes and discover partial evidence only for the page
                # being viewed, rather than rereading the entire image corpus.
                complete = state['state'] in ('REVIEW_PENDING', 'CACHED')
                failed = state['state'] in ('FAILED', 'STALE', 'CANCELLED')
                instance = record['status']
                if failed:
                    initialized = Path(state.get('native_output', queue.root / 'missing')) / 'initialization_complete.json'
                    instance = 'capture_failed' if initialized.is_file() else 'native_load_failed'
                elif record.get('capture_submission_error') and instance == 'generated':
                    instance = 'capture_failed'
                elif complete and instance == 'generated':
                    instance = 'capture_complete'
                diagnostic = {'receipt': {}, 'diagnostic_images': [], 'diagnostic_logs': [],
                    'capture': {**state, 'job_id': record.get('job_id')}, 'instance_status': instance,
                    'native_complete': complete, 'no_native_image': not complete}
            receipt = diagnostic['receipt']
            evidence = diagnostic['diagnostic_images'] + record.get('diagnostic_images', [])
            unique = {i['path']: i for i in evidence if detailed and Path(i['path']).is_file()}
            images = {f'diagnostic_{index}': Path(item['path']) for index, item in enumerate(unique.values())}
            image_info = {f'diagnostic_{index}': {**item, 'label': item['label'] + ' · ' + item.get('view', '')}
                          for index, item in enumerate(unique.values())}
            task_path = record.get('task_file')
            if detailed and task_path and record.get('capture_input'):
                sealed_task = Path(record['capture_input']).parent / 'task.yaml'
                if sealed_task.is_file():
                    task_path = str(sealed_task)
            rows.append({'id': record['candidate_id'], 'candidate_id': record['candidate_id'], 'origin': 'current_campaign',
                'source': record['source_identity'], 'scene_file': record.get('scene_file'),
                'source_file': record['original_source'], 'task_file': record.get('task_file'),
                'scene': record.get('scene_model') or Path(record['source_identity']).parts[-3],
                'room': record.get('room') or Path(record['source_identity']).parts[-2],
                'sample': Path(record['source_identity']).stem, 'variant': record.get('variant', 'original'),
                'seed': record['seed'], 'fingerprint': record['fingerprint'], 'conversion_status': 'pending',
                'instance_status': diagnostic['instance_status'], 'review_submitted': True,
                'conversion_result_status': record['status'], 'native_capture_complete': diagnostic['native_complete'],
                'prior_conversion_failures': record.get('prior_conversion_failures', []),
                'category': record['big_task'], 'subcategory': record['small_task'],
                'reuse_level': record['reuse_level'], 'converter_version': record['converter_version'],
                'question': record.get('question', ''), 'target': {'name': None, 'category': record['small_task'], 'model': None},
                'robot_task_public': self._public_task(task_path, record['fingerprint']) if detailed else None,
                'targets': [], 'containers': [], 'support': None, 'aperture_warning': False,
                'route_confirmed': False, 'physical_grasp_verified': False, 'verification_level': 'pending_category_goal',
                'failure': diagnostic['capture'].get('error') or record.get('failure_reason') or record.get('capture_submission_error', ''),
                'failure_stage': diagnostic['capture'].get('failure_stage') or record.get('failure_stage'),
                'warning_codes': record.get('warning_codes', []), 'assumptions': record.get('assumptions', []),
                'change_summary': record.get('change_summary', []), 'known_limitations': record.get('limitations', []),
                'attempted_repairs': record.get('attempted_repairs', []), 'next_step': record.get('next_step'),
                'diagnostic_dir': record['diagnostic_dir'], 'diagnostic_logs': sorted(set(diagnostic['diagnostic_logs'] + record.get('diagnostic_logs', []))),
                'exception_summary': record.get('exception_summary') or record.get('capture_submission_error'),
                'no_native_image': diagnostic['no_native_image'], 'source_only': record.get('source_only', False),
                'report_file': record.get('report_file'), 'batch_file': record.get('batch_file'),
                'viewer_command': '', 'capture': diagnostic['capture'], 'code_commit': record.get('code_commit'),
                'owner': record.get('owner'), 'parent_candidate_id': record.get('parent_candidate_id'),
                '_result': {}, '_definition': {}, '_source_path': Path(record['original_source']),
                '_category_record': True, '_capture_receipt': receipt,
                '_evidence_images': images, '_image_info': image_info})
            if detailed and cacheable:
                self._category_row_cache[record['candidate_id']] = (cache_key, dict(rows[-1]))
        return rows

    def _candidate_rows(self):
        """Discover accepted snapshots on every request; no server restart."""
        if self.queue_root is None:
            return []
        from wia.runtime.scene_capture_queue import CaptureQueue, complete_images

        queue = CaptureQueue(self.queue_root)
        rows = []
        for directory in sorted((queue.root / "candidates").iterdir()):
            submissions = [read_json(path) for path in (directory / "submissions").glob("*.json")]
            if not submissions:
                continue
            candidate = max(submissions, key=lambda item: item["submitted_at"])
            state = queue.status(candidate["job_id"]) or {"state": "QUEUED"}
            receipt = read_json(state.get("receipt", queue.root / "missing"), {})
            valid = complete_images(candidate, receipt)
            if not valid:
                for previous in sorted(submissions, key=lambda item: item["submitted_at"], reverse=True):
                    old_state = queue.status(previous["job_id"]) or {}
                    old_receipt = read_json(old_state.get("receipt", queue.root / "missing"), {})
                    if previous["fingerprint"] == candidate["fingerprint"] and complete_images(previous, old_receipt):
                        receipt, valid = old_receipt, True
                        break
            worker = (state.get("claim") or {}).get("worker", {})
            gpu = receipt.get("gpu") or state.get("gpu") or {}
            rows.append({"id": candidate["candidate_id"], "candidate_id": candidate["candidate_id"], 'origin': 'existing_capture',
                "source": candidate["source_identity"], "source_file": candidate["source_json"],
                "scene_file": candidate["scene_file"], "scene": candidate["scene_model"], "room": candidate["room"],
                "sample": Path(candidate["source_json"]).stem, "variant": candidate["variant"], "seed": candidate["seed"],
                "fingerprint": candidate["fingerprint"], "conversion_status": candidate["conversion_status"],
                "category": candidate["category"], "subcategory": candidate["subcategory"],
                "reuse_level": candidate["reuse_level"], "question": candidate["question"],
                "target": {"name": None, "category": candidate["subcategory"], "model": None},
                "containers": [], "targets": [], "support": None,
                "verification_level": "diagnostic_preview" if candidate["diagnostic_pending"] else "existing_native_gate",
                "aperture_warning": False, "route_confirmed": False, "physical_grasp_verified": False,
                "failure": state.get("error", ""), "correction": "", "diagnostic_dir": state.get("native_output"),
                "viewer_command": "", "task_file": candidate["task_file"],
                "physical_task_id": candidate.get("physical_task_id"), "scene_identity": candidate["scene_identity"],
                "code_commit": candidate["code_commit"], "owner": candidate.get("owner"),
                "parent_candidate_id": candidate.get("parent_candidate_id"),
                "known_limitations": candidate["known_limitations"],
                "instance_status": ('capture_complete' if valid else 'native_load_failed' if state['state'] in ('FAILED', 'STALE') else 'generated'),
                "failure_stage": state.get('failure_stage'), "warning_codes": [], "assumptions": [],
                "no_native_image": not valid,
                "capture": {"state": state["state"], "job_id": candidate["job_id"], "worker": worker.get("id"),
                    "gpu": gpu, "duration_s": state.get("duration_s"), "attempts": state.get("attempts", 0),
                    "error": state.get("error"), "timings": receipt.get("timings"),
                    "progress": read_json(Path(state.get("native_output", queue.root / "missing")) / "progress.json", {})},
                "_result": {}, "_definition": {}, "_source_path": Path(candidate["snapshot_source_file"]),
                "_capture_receipt": receipt if valid else {}, "_queue_candidate": True,
                "_capture_current": valid and receipt.get("capture_signature") == candidate["capture_signature"]})
            if not valid and state['state'] in ('FAILED', 'STALE', 'CANCELLED'):
                from wia.runtime.esi_category_capture import collect_diagnostics
                diagnostic = collect_diagnostics({**candidate, 'status': 'generated', 'source_only': False,
                    'diagnostic_dir': str(Path(state.get('native_output', queue.root / 'missing')))}, queue.root)
                row = rows[-1]
                row.update(instance_status=diagnostic['instance_status'], diagnostic_logs=diagnostic['diagnostic_logs'],
                    failure_stage=state.get('failure_stage'), no_native_image=True)
                partial = diagnostic['diagnostic_images']
                row['_evidence_images'] = {f'diagnostic_{i}': Path(item['path']) for i, item in enumerate(partial)}
                row['_image_info'] = {f'diagnostic_{i}': item for i, item in enumerate(partial)}
        return rows

    def import_previews(self):
        """Reuse only accepted independent restarts of the exact published layout."""
        imported = 0
        for row in self._base_rows(category_detail_ids=set()):
            result = row["_result"]
            if row["conversion_status"] == "ready" and result.get("verification_level") == "approximate_native_screen":
                directory = Path(result["output"]) / "sampling"
                candidate = Path(result.get("candidate_file", ""))
                bank = read_json(directory / "support_bank.json", {})
                status = read_json(directory / "scene_status.json", {})
                certified = [item for item in bank.get("accepted", [])
                    if (directory / item["scene_file"]).resolve() == candidate.resolve() and item.get("accepted")]
                inline = read_json(directory / "review_views" / candidate.stem / "native_review_complete.json", {})
                if (candidate.is_file() and bank.get("status") == "screened" and bank.get("source_json") == row["source"]
                        and status.get("qualification_status") in ("accepted_original", "accepted_minimal_patch")
                        and len(certified) == 1 and certified[0].get("scene_sha256") == sha256(candidate)
                        and same_layout(yaml.safe_load(candidate.read_text()), row["_definition"])
                        and matching_native_views(inline, candidate, row)):
                    write_json(self.output / "imported" / row["id"] / "preview_complete.json", {
                        "schema": "scene_review_preview_v1", "id": row["id"], "fingerprint": row["fingerprint"],
                        "variant": row["variant"], "seed": row["seed"], "images": inline["images"],
                        "target_location": inline["target_location"],
                        "provenance": "Views captured during the accepted native support screen of this exact layout.",
                        "evidence_dir": str(directory), "model_calls": 0, "physical_grasp_verified": False})
                    imported += 1
                continue
            if row["conversion_status"] != "ready" or not result.get("validation_dir"):
                continue
            directory = Path(result["validation_dir"])
            marker = read_json(directory / "validation_complete.json", {})
            manifest = read_json(directory / "scene_manifest.json", {})
            status = read_json(directory / "scene_status.json", {})
            candidate = Path(marker.get("scene_file", ""))
            if (not candidate.is_file() or marker.get("source_json") != row["source"]
                    or marker.get("variant") != row["variant"]
                    or manifest.get("scene_sha256") != sha256(candidate)
                    or status.get("qualification_status") not in ("accepted_original", "accepted_minimal_patch")
                    or not same_layout(yaml.safe_load(candidate.read_text()), row["_definition"])):
                continue
            source_hash = manifest.get("source_sha256")
            if source_hash and source_hash != sha256(row["_source_path"]):
                continue
            files = {view: str(directory / name) for view, name in
                     (("head", "initial_head.png"), ("wrist", "initial_wrist.png"))
                     if (directory / name).is_file()}
            inline = read_json(directory / "native_review_complete.json", {})
            target_location = {}
            if matching_native_views(inline, candidate, row):
                files = inline["images"]
                target_location = inline["target_location"]
            if not files:
                continue
            receipt = self.output / "imported" / row["id"] / "preview_complete.json"
            write_json(receipt, {"schema": "scene_review_preview_v1", "id": row["id"],
                "fingerprint": row["fingerprint"], "variant": row["variant"], "seed": row["seed"],
                "images": files, "captured_at": manifest.get("created_at"),
                "target_location": target_location,
                "provenance": "Accepted independent restart of the published layout.",
                "evidence_dir": str(directory), "model_calls": 0, "physical_grasp_verified": False})
            imported += 1
        return imported

    def _preview(self, row):
        if row.get('_category_record'):
            return row['_capture_receipt'], row['_evidence_images']
        if row.get("_queue_candidate"):
            receipt = row["_capture_receipt"]
            return receipt, {**{view: Path(path) for view, path in receipt.get("images", {}).items()
                             if view in ALL_VIEW_LABELS}, **row.get('_evidence_images', {})}
        for directory in ("captures", "imported"):
            receipt = read_json(self.output / directory / row["id"] / "preview_complete.json", {})
            if (receipt.get("fingerprint") != row["fingerprint"]
                    or receipt.get("variant") != row["variant"] or receipt.get("seed") != row["seed"]):
                continue
            images = {}
            locations = receipt.get("target_locations") or [receipt.get("target_location") or {}]
            for view, path in receipt.get("images", {}).items():
                if view not in ALL_VIEW_LABELS or not Path(path).is_file():
                    continue
                if view.startswith("global_target"):
                    index = 0 if view == "global_target" else int(view[-1])
                    location = locations[index] if index < len(locations) else {}
                    if (location.get("camera_capture_protocol") != "target_instance_verified_v3"
                            or location.get("visible_target_pixels", 0) < 64):
                        continue
                images[view] = Path(path)
            if images:
                return receipt, images
        return {}, {}

    def catalogue(self, *, filters=None, page=None, page_size=48, _category_detail_ids=None,
                  _prepared_rows=None, _prepared_queue=None):
        with self.lock:
            reviews = read_json(self.review_file, {"schema": "scene_reviews_v1", "entries": {}})
            records = reviews["entries"]
            rows, image_table = [], {}
            detail_ids = set() if page is not None and _category_detail_ids is None else _category_detail_ids
            base_rows = self._base_rows(category_detail_ids=detail_ids) if _prepared_rows is None else _prepared_rows
            for row in base_rows:
                receipt, images = self._preview(row)
                review = records.get(row["id"], {"decision": "unreviewed", "notes": "", "revision": 0})
                stale = bool(review.get("fingerprint") and review["fingerprint"] != row["fingerprint"])
                diagnostic = bool(receipt.get("diagnostic_pending"))
                eligible = bool(images) and not diagnostic
                if row.get("_queue_candidate"):
                    eligible = eligible and bool(receipt.get("eligible_final_layout")) and row["_capture_current"]
                if row.get('_category_record'):
                    eligible = False  # Native diagnostics do not qualify a pending robot goal.
                history = [item["review"] for item in reviews.get("history", []) if item["id"] == row["id"]]
                parent_review = records.get(row.get("parent_candidate_id"))
                if parent_review:
                    history.append(parent_review)
                row.update(review=review, review_stale=stale, review_history=history,
                    decision="unreviewed" if stale else review["decision"],
                    human_approved=not stale and review['decision'] == 'approved',
                    physical_qualified=bool(row.get('physical_grasp_verified')),
                    warning_review_required='assumption_review_required' in row.get('warning_codes', []),
                    has_final_preview=eligible, has_global_preview=any(v.startswith("global_") for v in images),
                    preview_provenance=("Previous matching-layout preview; latest requested capture awaits completion. "
                        if row.get("_queue_candidate") and receipt and not row["_capture_current"] else "")
                        + receipt.get("provenance", "Final layout awaits capture."),
                    preview_is_current=row.get("_capture_current", True),
                    diagnostic_pending=diagnostic,
                    target_location=receipt.get("target_location"),
                    target_locations=receipt.get("target_locations", []),
                    images=[{"view": view, "label": ALL_VIEW_LABELS.get(view, row.get('_image_info', {}).get(view, {}).get('label', view)),
                        'kind': row.get('_image_info', {}).get(view, {}).get('kind', 'native'),
                        'original_view': row.get('_image_info', {}).get(view, {}).get('view', view),
                        "url": f"images/{row['id']}/{view}?v={row['fingerprint'][:12]}"} for view in images])
                for index, target in enumerate(row["targets"]):
                    view = "global_target" if index == 0 else f"global_target_extra{index}"
                    for image in row["images"]:
                        if image["view"] == view:
                            image["label"] = "Pickup · " + target["category"].replace("_", " ")
                image_table.update({(row["id"], view): path for view, path in images.items()})
                rows.append({k: v for k, v in row.items() if not k.startswith("_")})
            if page is None or _category_detail_ids is not None:
                self.rows = rows
                self.images.update(image_table)
            progress = read_json(self.output / "capture_summary.json", {"status": "not_started"})
            queue_status = _prepared_queue
            if self.queue_root and queue_status is None:
                from wia.runtime.scene_capture_queue import CaptureQueue
                queue_status = CaptureQueue(self.queue_root).summary()
                queue_status.pop('jobs', None)
            groups = self._category_groups(rows)
            total = len(rows)
            matched = rows
            filters = filters or {}
            if filters.get('capture_status') in ('ready', 'pending'):
                complete = filters['capture_status'] == 'ready'
                matched = [r for r in matched
                           if bool(r.get('native_capture_complete', r.get('has_final_preview'))) == complete]
            fields = {'id': 'id', 'big_task': 'category', 'small_task': 'subcategory', 'converter_version': 'converter_version',
                      'reuse_level': 'reuse_level', 'review_status': 'decision', 'failure_stage': 'failure_stage', 'scene_model': 'scene',
                      'conversion_status': 'conversion_status'}
            for name, field in fields.items():
                if filters.get(name) and filters[name] != 'all':
                    matched = [r for r in matched if str(r.get(field, '')) == filters[name]]
            if filters.get('warning') and filters['warning'] != 'all':
                matched = [r for r in matched if (bool(r.get('warning_codes')) if filters['warning'] == 'any'
                           else filters['warning'] in r.get('warning_codes', []))]
            if filters.get('search'):
                query = filters['search'].lower()
                matched = [r for r in matched if query in ' '.join(str(r.get(f, '')) for f in
                    ('scene', 'room', 'sample', 'question', 'category', 'subcategory', 'failure')).lower()]
            if filters.get('sort', 'attention') == 'attention':
                matched = sorted(matched, key=lambda r: (r['decision'] != 'unreviewed', not bool(r.get('failure')),
                    not bool(r.get('warning_codes')), r.get('source', '')))
            elif filters.get('sort') == 'source':
                matched = sorted(matched, key=lambda r: r['source'])
            elif filters.get('sort') == 'scene':
                matched = sorted(matched, key=lambda r: (r['scene'], r['room'], r['sample']))
            match_count = len(matched)
            if page is not None:
                page_size = min(200, max(1, page_size))
                page = max(0, page)
                matched = matched[page * page_size:(page + 1) * page_size]
                if _category_detail_ids is None:
                    selected_ids = {row['id'] for row in matched}
                    details = {row['id']: row for row in self._category_rows(
                        detail_ids=selected_ids, only_ids=selected_ids)}
                    return self.catalogue(filters=filters, page=page, page_size=page_size,
                        _category_detail_ids=selected_ids, _prepared_queue=queue_status,
                        _prepared_rows=[details.get(row['id'], row) for row in base_rows])
            option_rows = [r for r in rows if (not filters.get('big_task') or r.get('category') == filters['big_task'])
                           and (not filters.get('small_task') or r.get('subcategory') == filters['small_task'])]
            options = {field: sorted({str(r[field]) for r in option_rows if r.get(field)})
                       for field in ('converter_version', 'failure_stage', 'scene')}
            options['warning_codes'] = sorted({warning for r in option_rows for warning in r.get('warning_codes', [])})
            return {"scenes": matched, 'categories': groups, 'filter_options': options, 'total': total, 'matched': match_count,
                    'page': page, 'page_size': page_size, "capture": progress, "capture_queue": queue_status, "generated_at": now(),
                    "preparation": read_json(self.output / "preparation_status.json", {}),
                    "pending_capture": read_json(self.output / "pending_capture_summary.json", {"status": "not_started"}),
                    "review_file": str(self.review_file), "conversion_root": str(self.conversion_root) if self.conversion_root else None}

    def _category_groups(self, rows):
        groups = {}
        if self.campaign_root:
            inventory = read_json(self.campaign_root / 'inventory.json', {})
            jobs = read_json(self.campaign_root / 'jobs.json', {'jobs': {}})['jobs']
            for item in inventory.get('categories', []):
                groups[(item['big_task'], item['small_task'])] = {
                    'big_task': item['big_task'], 'small_task': item['small_task'],
                    'source_count': len(item['sources']), 'report_file': item['report_file'],
                    'existing_ready_scene_count': item.get('existing_ready_scene_count', 0),
                    'existing_ready_task_count': item.get('existing_ready_task_count', 0),
                    'job': {field: value for field, value in jobs.get(item['key'], {}).items()
                            if field in ('state', 'owner', 'slot', 'converter_version',
                                         'source_count', 'counts', 'review_count', 'reason')}}
        for row in rows:
            key = (row.get('category', 'Unclassified'), row.get('subcategory', 'Unclassified'))
            group = groups.setdefault(key, {'big_task': key[0], 'small_task': key[1], 'source_count': 0,
                'report_file': row.get('report_file'), 'job': {}})
            counts = group.setdefault('counts', {k: 0 for k in ('total', 'generated', 'captured', 'failed', 'warned',
                'reviewed', 'approved', 'needs_repair', 'skipped')})
            counts['total'] += 1
            counts['generated'] += row.get('conversion_result_status', 'generated') == 'generated'
            counts['captured'] += bool(row.get('native_capture_complete', row['has_final_preview'] or row.get('has_global_preview')))
            counts['failed'] += bool(row.get('failure')) or row.get('instance_status') in ('conversion_failed', 'native_load_failed', 'capture_failed')
            counts['warned'] += bool(row.get('warning_codes')) or bool(row.get('aperture_warning'))
            counts['reviewed'] += row['decision'] != 'unreviewed'
            counts['approved'] += row['decision'] == 'approved'
            counts['needs_repair'] += row['decision'] == 'needs_fix'
            counts['skipped'] += row['decision'] == 'skipped'
            counts['new'] = counts.get('new', 0) + (row.get('origin') == 'current_campaign')
            counts['existing'] = counts.get('existing', 0) + (row.get('origin') != 'current_campaign')
        for group in groups.values():
            group.setdefault('counts', {k: 0 for k in ('total', 'generated', 'captured', 'failed', 'warned',
                'reviewed', 'approved', 'needs_repair', 'skipped')})
            group['counts'].setdefault('new', 0)
            group['counts'].setdefault('existing', 0)
        return sorted(groups.values(), key=lambda g: (g['big_task'], g['small_task']))

    def category_report(self, big_task, small_task):
        from wia.runtime.esi_category_scheduler import report_path
        path = report_path(big_task, small_task)
        if not path.is_file():
            return '# ' + small_task + '\n\nCategory supervisor report has not been written yet.\n'
        return path.read_text()

    def diagnostic_log(self, key, index):
        row = next((r for r in self.catalogue(filters={'id': key}, page=0, page_size=1)['scenes'] if r['id'] == key), None)
        if row is None:
            raise KeyError(key)
        path = Path(row.get('diagnostic_logs', [])[index])
        with path.open('rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 65536))
            return handle.read().decode('utf-8', errors='replace')

    def save_review(self, key, payload):
        with self.lock:
            rows = self.catalogue(filters={'id': key}, page=0, page_size=1)["scenes"]
            row = next((r for r in rows if r["id"] == key), None)
            if row is None:
                raise KeyError(key)
            decision, notes = payload.get("decision"), payload.get("notes", "")
            if decision not in DECISIONS or not isinstance(notes, str) or len(notes) > 10000:
                raise ValueError("Provide a valid review decision and notes of at most 10,000 characters.")
            if payload.get("fingerprint") != row["fingerprint"]:
                raise ReviewConflict("The scene changed. Refresh and review the current layout.")
            if payload.get("revision", 0) != row["review"].get("revision", 0):
                raise ReviewConflict("This review changed in another tab. Refresh before saving.")
            record = {"decision": decision, "notes": notes, "revision": row["review"].get("revision", 0) + 1,
                "fingerprint": row["fingerprint"], "source": row["source"], "scene_file": row["scene_file"],
                "variant": row["variant"], "seed": row["seed"], "updated_at": now()}
            if decision == "approved":
                record.update(approval_scope="layout", approval_evidence={
                    "conversion_status": row["conversion_status"],
                    "has_final_preview": bool(row["has_final_preview"])})
            value = read_json(self.review_file, {"schema": "scene_reviews_v1", "entries": {}})
            previous = value["entries"].get(key)
            if previous:
                value.setdefault("history", []).append({"id": key, "review": previous})
            if row.get("candidate_id"):
                record.update(candidate_id=row["candidate_id"], owner=row.get("owner"),
                              parent_candidate_id=row.get("parent_candidate_id"))
            record.update(big_task=row.get('category'), small_task=row.get('subcategory'),
                          converter_version=row.get('converter_version'))
            value["entries"][key] = record
            write_json(self.review_file, value)
            if self.queue_root and decision == "needs_fix":
                from wia.runtime.scene_capture_candidates import export_feedback
                export_feedback(self.queue_root, self.review_file)
            if self.queue_root:
                from wia.runtime.esi_category_runner import aggregate_feedback
                scheduler = None
                if self.campaign_root:
                    from wia.runtime.esi_category_scheduler import CategoryScheduler
                    scheduler = CategoryScheduler(self.campaign_root)
                aggregate_feedback(self.queue_root, self.review_file, scheduler=scheduler)
            return record

    def archive_superseded_reviews(self):
        """Retain current-layout annotations and archive the exact prior file."""
        with self.lock:
            rows = self._base_rows(category_detail_ids=set())
            current = {row["id"]: row for row in rows}
            if len(current) != len(rows):
                raise ValueError("Merge conversion results before retiring duplicate review scenes.")
            value = read_json(self.review_file, {"schema": "scene_reviews_v1", "entries": {}})
            removed = {key: record for key, record in value["entries"].items()
                if key not in current or record.get("fingerprint") != current[key]["fingerprint"]}
            archive = None
            if removed:
                archive = self.output / "archive" / ("reviews_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
                archive.parent.mkdir(parents=True, exist_ok=True)
                archive.write_bytes(self.review_file.read_bytes())
                value["entries"] = {key: record for key, record in value["entries"].items() if key not in removed}
                write_json(self.review_file, value)
            return {"scene_count": len(rows), "retained_review_count": len(value["entries"]),
                "removed_review_ids": sorted(removed), "archive": str(archive) if archive else None}

    def export_csv(self):
        rows = self.catalogue()["scenes"]
        stream = io.StringIO()
        fields = ("source", "scene_file", "variant", "seed", "conversion_status", "decision",
                  "review_stale", "has_final_preview", "has_global_preview", "aperture_warning",
                  "route_confirmed", "physical_grasp_verified", "notes", "updated_at")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            value = {k: row.get(k, "") for k in fields}
            value.update(notes=row["review"].get("notes", ""), updated_at=row["review"].get("updated_at", ""))
            writer.writerow(value)
        return stream.getvalue()
