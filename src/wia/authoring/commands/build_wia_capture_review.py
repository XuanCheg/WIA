"""Build a WIA layout-image supplement from an explicit capture-review index.

Run: python -B scripts/build_wia_capture_review.py --index PATH [--output PATH]
  [--watch --interval-s 15]
Then pass its output to scripts/build_wia_review.py --supplement PATH.
Only exact declared files are read. No queues, simulators, GPUs, models, services,
image decoders, source resolvers or full reference executions are started.


Canonical invocation: python -m wia.authoring.commands.build_wia_capture_review --help
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

from wia.resources import ROOT
from wia.authoring.commands._inputs import ARTIFACT_ROOT

from wia.authoring.commands.build_wia_review import (ALLOWED_VIEWS, DEFAULT_EVIDENCE, EVIDENCE_SCHEMA,
                                     _portable, _safe_path, _token, _unique_json_pairs)

INDEX_SCHEMA = 'wia_capture_review_index_v1'
DEFAULT_OUTPUT = ARTIFACT_ROOT / 'wia_v1_20261023/review/capture_evidence.json'
BUILDER = 'scripts/build_wia_capture_review.py'
LAYOUT_ONLY = ('Native layout images only; full reference execution deferred until user review. '
               'No success or qualification is inferred.')
SHA256 = re.compile(r'^[0-9a-fA-F]{64}$')


def _definition_check(declared, expected, root, cache):
    record = {'declared_path': declared, 'path': '', 'expected_sha256': expected,
              'current_sha256': None, 'matches': False}
    try:
        path = _safe_path(declared, root)
        record['path'] = _portable(path, root)
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise ValueError('The expected definition hash is not a SHA-256 hex digest.')
        if path not in cache:
            cache[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        record['current_sha256'] = cache[path]
        record['matches'] = cache[path].lower() == expected.lower()
        if not record['matches']:
            record['reason'] = 'Current definition bytes do not match the indexed capture input.'
    except (OSError, ValueError) as error:
        record['reason'] = f'{type(error).__name__}: {error}'
    return record


def _receipt_images(receipt, receipt_path, root):
    """Validate declared PNGs, never search, relocate, decode or hash images."""
    declared = receipt.get('images')
    if not isinstance(declared, dict):
        raise ValueError('Receipt images must be an explicit view-to-PNG mapping.')
    views, ignored, missing = [], [], []
    for view, value in declared.items():
        if view not in ALLOWED_VIEWS:
            ignored.append(view)
            continue
        if not isinstance(value, str) or not value:
            raise ValueError('A known receipt view has no explicit PNG path: ' + view)
        raw = Path(value)
        # A basename is explicitly receipt-relative; multi-part relative paths
        # are repository-relative. Neither spelling triggers a relocation search.
        path = _safe_path(receipt_path.parent / raw if not raw.is_absolute() and len(raw.parts) == 1 else raw, root)
        if (path.parent != receipt_path.parent or path.suffix != '.png'
                or path.name != view + '.png'):
            raise ValueError('Receipt image must be its named PNG in the same receipt directory: ' + view)
        if not path.is_file():
            missing.append(view)
            continue
        views.append(view)
    return sorted(views), sorted(ignored), sorted(missing)


def _attempt_receipt(path, job_directory, root):
    """Accept only the exact named receipt in this declared job's attempt tree."""
    if (not path.is_relative_to(root) or path.name != 'preview_complete.json'
            or path.parent.parent != job_directory / 'attempts'):
        raise ValueError('Status receipt must remain inside the declared job attempts directory and project root.')
    attempt = path.parent.name
    if not attempt.isascii() or not attempt.isdecimal() or int(attempt) < 1 or f'{int(attempt):02d}' != attempt:
        raise ValueError('Status receipt requires a canonical positive attempt directory.')
    return int(attempt)


def _resolve_indexed_receipt(item, declared_path, root, review):
    """Read an optional exact queue status, without discovering or modifying jobs."""
    if 'queue_status_file' not in item:
        return declared_path
    provenance = {'declared_path': item['queue_status_file'], 'path': '',
                  'available': False, 'receipt_selected': False}
    review['queue_status'] = provenance
    job_directory = declared_path.parent.parent.parent
    if job_directory.name != item['job_id']:
        raise ValueError('Declared receipt directory does not match the indexed job identity.')
    _attempt_receipt(declared_path, job_directory, root)
    status_path = _safe_path(item['queue_status_file'], root)
    if not status_path.is_relative_to(root) or status_path != job_directory / 'status.json':
        raise ValueError('Queue status must be the exact status.json in the declared receipt job directory.')
    provenance['path'] = _portable(status_path, root)
    try:
        status = json.loads(status_path.read_text(encoding='utf-8'), object_pairs_hook=_unique_json_pairs)
    except FileNotFoundError:
        provenance['resolution'] = 'missing_status_use_declared_receipt'
        return declared_path
    except (OSError, UnicodeError, ValueError) as error:
        provenance['read_error_type'] = type(error).__name__
        raise ValueError('Explicit queue status is unreadable or invalid JSON.') from error
    provenance['available'] = True
    if not isinstance(status, dict) or status.get('job_id') != item['job_id']:
        raise ValueError('Queue status job identity does not match the indexed job.')
    state = status.get('state')
    if state not in {'QUEUED', 'CLAIMED', 'RUNNING', 'CAPTURED', 'REVIEW_PENDING', 'CACHED',
                     'FAILED', 'STALE', 'CANCELLED'}:
        raise ValueError('Queue status has a malformed capture state.')
    if 'complete' in status and type(status['complete']) is not bool:
        raise ValueError('Queue status complete must be a boolean when supplied.')
    attempts = status.get('attempts')
    if 'attempts' in status and (type(attempts) is not int or attempts < 0):
        raise ValueError('Queue status attempts must be a nonnegative integer when supplied.')
    provenance.update(job_id=status['job_id'], state=state, complete=status.get('complete'),
                      attempts=attempts, declared_receipt=status.get('receipt'),
                      error=status.get('error'), failure_stage=status.get('failure_stage'))
    resolved = None
    if 'receipt' in status:
        resolved = _safe_path(status['receipt'], root)
        selected_attempt = _attempt_receipt(resolved, job_directory, root)
        provenance['resolved_receipt'] = _portable(resolved, root)
    if state == 'REVIEW_PENDING' and status.get('complete') is True and resolved is not None:
        if attempts is not None and attempts != selected_attempt:
            raise ValueError('Queue status attempts disagrees with its exact receipt attempt directory.')
        provenance.update(receipt_selected=True, resolution='completed_status_receipt')
        return resolved
    provenance['resolution'] = 'pending_status_use_declared_receipt'
    return declared_path


def _capture_variant(item, root, cache):
    task = _definition_check(item.get('task_file'), item.get('task_sha256'), root, cache)
    scene = _definition_check(item.get('scene_file'), item.get('scene_sha256'), root, cache)
    review = {'job_id': item['job_id'], 'declared_receipt_file': item.get('receipt_file'),
              'definition_checks': {'task': task, 'scene': scene}, 'reason': '',
              'full_reference_execution_deferred': True, 'physical_success_inferred': False}
    variant = {'id': item['variant_id'], 'label': item['label'],
               'task_file': task['path'], 'scene_file': scene['path'],
               'status': 'capture_pending', 'evidence_summary': LAYOUT_ONLY,
               'evidence': [], 'captures': [], 'capture_review': review}

    def finish(status, reason):
        variant['status'], review['reason'] = status, reason
        variant['evidence_summary'] = (LAYOUT_ONLY if status == 'native_images_ready'
                                       else reason + ' ' + LAYOUT_ONLY)
        return variant

    try:
        receipt_path = _safe_path(item.get('receipt_file'), root)
        review['receipt_file'] = _portable(receipt_path, root)
        variant['evidence'] = [{'label': 'Exact native capture receipt', 'path': review['receipt_file']}]
    except ValueError as error:
        return finish('evidence_mismatch', 'Declared receipt path is disallowed: ' + str(error))
    if not task['matches'] or not scene['matches']:
        return finish('evidence_mismatch', 'Indexed task or scene bytes are missing, disallowed or changed.')
    try:
        receipt_path = _resolve_indexed_receipt(item, receipt_path, root, review)
    except (OSError, ValueError, TypeError) as error:
        return finish('evidence_mismatch', 'Explicit queue status or receipt is disallowed: ' + str(error))
    review['receipt_file'] = review['resolved_receipt_file'] = _portable(receipt_path, root)
    variant['evidence'] = [{'label': 'Exact native capture receipt', 'path': review['receipt_file']}]
    try:
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'), object_pairs_hook=_unique_json_pairs)
    except FileNotFoundError:
        if review.get('queue_status', {}).get('state') == 'FAILED':
            return finish('capture_failed',
                'The explicit queue status reports an infrastructure or initialization capture failure '
                'and the exact receipt is missing; this does not establish task infeasibility.')
        return finish('capture_pending', 'The exact indexed receipt has not been published.')
    except (OSError, UnicodeError, ValueError) as error:
        review['receipt_read_error_type'] = type(error).__name__
        return finish('capture_failed', 'The exact indexed receipt is unreadable or invalid JSON.')
    if not isinstance(receipt, dict):
        return finish('capture_failed', 'The exact indexed receipt is not a JSON mapping.')
    # capture_job_id is the queue identity when job_id is a scheduler/process id.
    actual_job_id = receipt.get('capture_job_id') or receipt.get('job_id')
    review['receipt_job_id'] = receipt.get('job_id')
    review['receipt_capture_job_id'] = receipt.get('capture_job_id')
    review['receipt_complete'] = receipt.get('complete') if type(receipt.get('complete')) is bool else None
    review['receipt_errors_present'] = bool(receipt.get('error') or receipt.get('errors'))
    for field in ('execution_qualified', 'native_qualification_run'):
        review['receipt_' + field] = receipt.get(field) if type(receipt.get(field)) is bool else None
    if actual_job_id != item['job_id']:
        return finish('evidence_mismatch', 'Receipt capture job identity does not match the indexed job.')
    if receipt.get('complete') is not True:
        return finish('capture_failed' if receipt.get('complete') is False or review['receipt_errors_present']
                      else 'capture_pending', 'The indexed receipt does not declare a complete capture.')
    bundle = receipt.get('initial_agent_bundle')
    identity = bundle.get('capture_input_identity') if isinstance(bundle, dict) else None
    actual_hash = identity.get('task_sha256') if isinstance(identity, dict) else None
    review['receipt_initial_task_sha256'] = actual_hash
    if actual_hash != item['task_sha256']:
        return finish('evidence_mismatch', 'Receipt frozen initial-agent task hash does not match the index.')
    if review['receipt_errors_present']:
        return finish('capture_failed', 'The indexed receipt records capture errors; no images are attached.')
    try:
        views, ignored, missing = _receipt_images(receipt, receipt_path, root)
        review.update(accepted_views=views, ignored_views=ignored, missing_views=missing)
    except (OSError, ValueError) as error:
        return finish('evidence_mismatch', 'Receipt image declaration is disallowed: ' + str(error))
    if missing or not views:
        return finish('capture_pending', 'The indexed receipt has missing or no existing allowed layout PNGs.')
    variant['captures'] = [{'directory': _portable(receipt_path.parent, root),
                            'label': item['label'], 'role': 'layout_review_only', 'views': views}]
    return finish('native_images_ready', 'Identity-matched layout PNGs are available for user review.')


def build_supplement(index, *, project_root=ROOT, generated_at=None):
    """Keep every indexed variant visible, with per-item failures kept separate."""
    root = Path(project_root).resolve()
    index_path = _safe_path(index, root)
    value = json.loads(index_path.read_text(encoding='utf-8'), object_pairs_hook=_unique_json_pairs)
    if not isinstance(value, dict) or value.get('schema') != INDEX_SCHEMA or not isinstance(value.get('items'), list):
        raise ValueError(f'Provide {INDEX_SCHEMA} with an items list.')
    updates, cache, used, statuses = {}, {}, set(), Counter()
    for item in value['items']:
        if not isinstance(item, dict):
            raise ValueError('Each capture index item must be a mapping.')
        row_id = _token(item.get('row_id'), 'Indexed row id')
        variant_id = _token(item.get('variant_id'), 'Indexed variant id')
        if not isinstance(item.get('label'), str) or not item['label'] or not isinstance(item.get('job_id'), str) or not item['job_id']:
            raise ValueError('Each capture item requires a label and exact job_id.')
        if (row_id, variant_id) in used:
            raise ValueError(f'Conflicting duplicate capture row/variant id: {row_id}/{variant_id}')
        used.add((row_id, variant_id))
        variant = _capture_variant(item, root, cache)
        updates.setdefault(row_id, {'variants': []})['variants'].append(variant)
        statuses[variant['status']] += 1
    return {'schema': EVIDENCE_SCHEMA,
            'generated_at': generated_at or datetime.now(timezone.utc).isoformat(),
            'row_updates': updates, 'extra_rows': [],
            'summary': {'indexed_variant_count': len(value['items']), 'row_count': len(updates),
                        'variant_status_counts': dict(sorted(statuses.items()))},
            'provenance': {'builder': BUILDER, 'index': _portable(index_path, root),
                           'scope': LAYOUT_ONLY, 'full_reference_execution_deferred': True,
                           'physical_success_inferred': False, 'native_experiments_run': False,
                           'queue_jobs_submitted': False, 'images_decoded_or_hashed': False}}


def publish_supplement(output, supplement, *, project_root=ROOT, protected_paths=()):
    """Replace only an owned supplement, preserving curation and user feedback."""
    root = Path(project_root).resolve()
    path = _safe_path(output, root)
    protected = {_safe_path(DEFAULT_EVIDENCE, root), *(_safe_path(value, root) for value in protected_paths)}
    if path.name == 'reviews.json' or path in protected:
        raise ValueError('The capture publisher must not overwrite main curation, its index or reviews.json.')
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if (not isinstance(previous, dict) or previous.get('schema') != EVIDENCE_SCHEMA
                or previous.get('provenance', {}).get('builder') != BUILDER):
            raise ValueError('Refusing to overwrite an unrelated capture-evidence output.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(supplement, ensure_ascii=True, indent=2, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval-s', type=float, default=15.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval_s) or args.interval_s <= 0:
        parser.error('--interval-s must be positive and finite.')
    try:
        while True:
            try:
                supplement = build_supplement(args.index)
                path = publish_supplement(args.output, supplement, protected_paths=(args.index,))
                print(f'Published {supplement["summary"]["indexed_variant_count"]} capture-review variants: {path}', flush=True)
            except (OSError, ValueError) as error:
                if not args.watch:
                    raise
                print(f'Capture review refresh failed; previous supplement retained: {error}', file=sys.stderr, flush=True)
            if not args.watch:
                break
            time.sleep(args.interval_s)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
