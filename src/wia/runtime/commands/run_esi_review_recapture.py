#!/usr/bin/env python3
"""Publish and resume a bounded representative-first capture campaign.

Run: python src/wia/runtime/commands/run_esi_review_recapture.py --records-file records.json
    --queue-root QUEUE --output-dir CAMPAIGN [--prepare-only]
The input contains records and representatives keyed by big_task/small_task.
Resume with the same arguments. Existing workers perform captures; this process
never launches workers, retries failed jobs, or approves native qualification.
Completed image validation is cached in memory until status/receipt metadata
changes; every coordinator start validates images afresh, including missing files.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.runtime.esi_category_capture import freeze_working_code, publish_record
from wia.runtime.scene_capture_queue import CaptureQueue, complete_images, digest, publish_json, read_json, write_json

PILOT_GATE = 'ONE_PER_CATEGORY_AWAITING_REVIEW'
BULK_GATE = 'RELEASED_AFTER_USER_CONFIRMATION'
CAPTURED_STATES = {'REVIEW_PENDING', 'SUCCEEDED', 'CACHED'}
FAILURE_STATES = {'FAILED', 'STALE', 'CANCELLED'}
PUBLICATION_FIELDS = ('candidate_id', 'job_id', 'record_revision', 'code_commit',
                      'capture_submission_error', 'capture_submission_stage')


def publication_reference(record):
    """Keep campaign state bounded; full evidence remains in category_records."""
    return {key: record[key] for key in PUBLICATION_FIELDS if key in record}


def initial_categories(payload):
    """Validate the fixed candidate pool and deterministic fallback order."""
    groups, seen = {}, set()
    for record in payload['records']:
        candidate = record['candidate_id']
        if candidate in seen:
            raise ValueError('Duplicate candidate_id: ' + candidate)
        seen.add(candidate)
        key = record['big_task'] + '/' + record['small_task']
        groups.setdefault(key, []).append(candidate)
    if not groups or set(groups) != set(payload['representatives']):
        raise ValueError('Exactly one representative is required for every nonempty category.')
    categories = {}
    for key, candidates in sorted(groups.items()):
        selected = payload['representatives'][key]
        if selected not in candidates:
            raise ValueError('Representative does not belong to category: ' + key)
        categories[key] = {'candidates': candidates, 'active': selected,
                           'tried': [selected], 'success': None, 'failed': False}
    return categories


def advance_stage(state, observations):
    """Pure bounded pilot/fallback transition; each candidate is tried at most once."""
    result = deepcopy(state)
    if result['stage'] in {'pilot', 'bulk'}:
        for category in result['categories'].values():
            if category['success'] or category['failed']:
                continue
            while True:
                outcome = observations[category['active']]['outcome']
                if outcome == 'captured':
                    category['success'] = category['active']
                    break
                if outcome != 'failed':
                    break
                remaining = [candidate for candidate in category['candidates']
                             if candidate not in category['tried']]
                if not remaining:
                    category['failed'] = True
                    break
                category['active'] = remaining[0]
                category['tried'].append(remaining[0])
        if result['stage'] == 'pilot' and all(c['success'] or c['failed'] for c in result['categories'].values()):
            result['stage'] = 'bulk'
    if result['stage'] == 'bulk' and all(
            item['outcome'] in {'captured', 'failed'} for item in observations.values()):
        result['stage'] = 'finished'
    return result


def receipt_revision(path):
    """Identify a receipt revision without reopening its image payload."""
    try:
        stat = Path(path).stat()
        return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except OSError:
        return (str(path), None)


def observe_record(queue, published, completion_cache=None):
    """Validate complete view evidence independently from scene qualification."""
    job_id = published.get('job_id')
    if not job_id:
        return {'outcome': 'failed', 'state': 'SUBMISSION_FAILED', 'job_id': None,
                'error': published.get('capture_submission_error', 'No capture job was published.')}
    status = queue.status(job_id) or {}
    result = {'job_id': job_id, 'state': status.get('state', 'MISSING'), 'outcome': 'pending',
              'attempts': status.get('attempts', 0)}
    if status.get('state') in CAPTURED_STATES:
        receipt_path = status.get('receipt') or queue.root / 'missing'
        revision = (digest(status), receipt_revision(receipt_path),
                    receipt_revision(queue.job_dir(job_id) / 'input/manifest.json'),
                    receipt_revision(queue.job_dir(job_id) / 'accepted.json'))
        cache_key = (str(queue.root), job_id)
        cached = completion_cache.get(cache_key) if completion_cache is not None else None
        if cached and cached['revision'] == revision:
            return dict(cached['result'])
        try:
            manifest = queue.manifest(job_id)
            receipt = read_json(receipt_path, {})
            valid = complete_images(manifest, receipt)
        except (OSError, ValueError, KeyError, TypeError) as error:
            valid = False
            result['error'] = str(error)
        result['outcome'] = 'captured' if valid else 'failed'
        if not valid:
            result.setdefault('error', 'Completed job has missing, invalid, or incomplete expected views.')
        result['receipt'] = status.get('receipt')
        if completion_cache is not None:
            completion_cache[cache_key] = {'revision': revision, 'result': dict(result)}
    elif status.get('state') in FAILURE_STATES:
        # Workers release the claim and then automatically retry. A retryable
        # failure with remaining attempts is still pending during that interval.
        manifest = queue.manifest(job_id)
        retry_pending = (status['state'] == 'FAILED' and status.get('retryable')
                         and status.get('attempts', 0) < manifest['max_attempts'])
        result.update(outcome='pending' if retry_pending else 'failed',
                      retry_pending=bool(retry_pending), error=status.get('error'),
                      failure_stage=status.get('failure_stage'))
    return result


def reconcile_gate(queue, state):
    """Apply the persisted stage, moving only this campaign's queued tokens."""
    current = set(state['candidate_ids'])
    selected = {c['active'] for c in state['categories'].values()}
    path = queue.root / 'capture_sample_gate.json'
    gate = read_json(path, {})
    previous_gate = dict(gate)
    # Preserve foreign metadata and selected identities; never edit annotations.
    foreign = set(gate.get('selected_candidate_ids', [])) - current
    gate.update(state=PILOT_GATE if state['stage'] == 'pilot' else BULK_GATE,
                selected_candidate_ids=sorted(foreign | selected),
                recapture_campaign_id=state['campaign_id'],
                authorized_candidate_ids=sorted(set(gate.get('authorized_candidate_ids', [])) | current),
                reason=('Automatic representative capture gate; native qualification remains independent.'
                        if state['stage'] == 'pilot' else
                        'Automatic bulk release after representative captures or explicit category exhaustion; '
                        'no native qualification approval is implied.'))
    if gate != previous_gate:
        write_json(path, gate)
    # Bulk release is monotonic. Move only held tokens, without rereading every
    # job status or rewriting priority metadata. Subsequent ticks do not call
    # reconciliation unless the stage or active representative changes.
    if state['stage'] != 'pilot':
        held_dir = queue.root / 'held_for_category_review'
        queued_dir = queue.root / 'queued'
        queued_dir.mkdir(exist_ok=True)
        for record in state['published'].values():
            job_id = record.get('job_id')
            if not job_id:
                continue
            old = held_dir / (job_id + '.json')
            if old.exists():
                os.rename(old, queued_dir / old.name)
        return
    held_dir = queue.root / 'held_for_category_review'
    held_dir.mkdir(exist_ok=True)
    ordered = sorted(state['published'].items(), key=lambda item: item[0] not in selected)
    for candidate, record in ordered:
        job_id = record.get('job_id')
        if not job_id or (queue.status(job_id) or {}).get('state') != 'QUEUED':
            continue
        if (queue.root / 'claimed' / (job_id + '.json')).exists():
            continue
        if state['stage'] == 'pilot' and candidate not in selected:
            source, target = queue.root / 'queued', held_dir
        else:
            source, target = held_dir, queue.root / 'queued'
            priority = 1000 if candidate in selected else 100
            existing_priority = read_json(queue.root / 'priorities' / (job_id + '.json'), {})
            if existing_priority.get('priority', 0) < priority:
                queue.prioritize(job_id, priority)
        old, new = source / (job_id + '.json'), target / (job_id + '.json')
        if new.exists():
            old.unlink(missing_ok=True)
            continue
        try:
            os.link(old, new)
        except FileExistsError:
            pass
        except FileNotFoundError:
            # Reconcile only missing tokens belonging to QUEUED campaign jobs.
            if ((queue.status(job_id) or {}).get('state') == 'QUEUED'
                    and not (queue.root / 'claimed' / (job_id + '.json')).exists()):
                publish_json(new, {'job_id': job_id})
            continue
        old.unlink(missing_ok=True)


def save_receipts(output, state, queue):
    """Write compact receipts linked to complete durable publication cards."""
    path = output / 'receipts.jsonl'
    temporary = output / '.receipts.jsonl.tmp'
    with temporary.open('w') as handle:
        for candidate in state['candidate_ids']:
            if candidate in state['published']:
                receipt = {**state['published'][candidate], 'category_record_file': str(
                    queue.root / 'category_records' / (candidate + '.json'))}
                handle.write(json.dumps(receipt, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def progress_document(state, observations):
    """Produce stable snapshots with explicit failures, counts, and pilot history."""
    categories = state['categories']
    return {'campaign_id': state['campaign_id'], 'stage': state['stage'],
            'code_commit': state['frozen']['code_commit'], 'total': len(state['candidate_ids']),
            'counts': dict(sorted(Counter(item['outcome'] for item in observations.values()).items())),
            'queue_states': dict(sorted(Counter(item['state'] for item in observations.values()).items())),
            'pilot_success_count': sum(bool(c['success']) for c in categories.values()),
            'pilot_failure_count': sum(c['failed'] for c in categories.values()),
            'categories': categories,
            'terminal_failures': {key: value for key, value in observations.items() if value['outcome'] == 'failed'},
            'records': observations}


def prepare_campaign(payload, queue, output, *, capture_timeout_s=1800):
    """Freeze once and publish representatives first; cards journal each record."""
    categories = initial_categories(payload)
    identity = digest({'payload': payload, 'queue_root': str(queue.root)})
    path = output / 'campaign.json'
    state = read_json(path)
    state_dirty = False
    if state:
        if state['campaign_id'] != identity:
            raise ValueError('Resume requires the original records and queue root.')
        if state.get('capture_timeout_s', 1800) != capture_timeout_s:
            raise ValueError('Resume requires the original capture timeout.')
        compact = {candidate: publication_reference(record)
                   for candidate, record in state['published'].items()}
        state_dirty = compact != state['published']
        state['published'] = compact
        # A user-authorized bulk release is durable queue state.  A coordinator
        # restart must not resurrect the representative-only pilot gate from
        # its in-memory checkpoint.
        if (read_json(queue.root / 'capture_sample_gate.json', {}).get('state') == BULK_GATE
                and state.get('stage') == 'pilot'):
            state['stage'] = 'bulk'
            state_dirty = True
    else:
        frozen = freeze_working_code(queue.root)
        state = {'campaign_id': identity, 'stage': 'pilot', 'categories': categories,
                 'candidate_ids': [r['candidate_id'] for r in payload['records']],
                 'frozen': frozen, 'published': {}, 'capture_timeout_s': capture_timeout_s}
        write_json(path, state)
    reconcile_gate(queue, state)
    selected = {category['active'] for category in state['categories'].values()}
    ordered = sorted(payload['records'], key=lambda record: record['candidate_id'] not in selected)
    checkpoint_count = 0
    for record in ordered:
        candidate = record['candidate_id']
        if candidate in state['published']:
            continue
        existing = read_json(queue.root / 'category_records' / (candidate + '.json'), {})
        if existing.get('job_id') and existing.get('code_commit') != state['frozen']['code_commit']:
            raise ValueError('Existing candidate uses a different frozen revision: ' + candidate)
        if existing and existing.get('record_revision') != record['revision']:
            raise ValueError('Existing candidate uses a different record revision: ' + candidate)
        if existing.get('job_id') or existing.get('capture_submission_error'):
            # A card written after the last campaign checkpoint is the durable
            # per-record journal. Never republish or reset its running job.
            published = existing
        else:
            clean = {k: v for k, v in record.items() if k not in {'_batch_file', '_report_file'}}
            published = publish_record(clean, queue.root, frozen=state['frozen'],
                                       priority=1000 if candidate in selected else 100,
                                       timeout_s=capture_timeout_s,
                                       batch_file=record.get('_batch_file'), report_file=record.get('_report_file'))
        state['published'][candidate] = publication_reference(published)
        state_dirty = True
        checkpoint_count += 1
        if checkpoint_count % 25 == 0:
            write_json(path, state)
            state_dirty = False
    if state_dirty:
        write_json(path, state)
    if state_dirty or checkpoint_count or not (output / 'receipts.jsonl').exists():
        save_receipts(output, state, queue)
    reconcile_gate(queue, state)
    return state


def step_campaign(queue, output, state, completion_cache=None, *, progress_output=None):
    observations = {candidate: observe_record(queue, record, completion_cache)
                    for candidate, record in state['published'].items()}
    gate = read_json(queue.root / 'capture_sample_gate.json', {})
    previous = state
    if gate.get('state') == BULK_GATE and state['stage'] == 'pilot':
        state = {**state, 'stage': 'bulk'}
    state = advance_stage(state, observations)
    # Persist transitions before token moves so interrupted release is resumable.
    if state != previous:
        write_json(output / 'campaign.json', state)
        if (state['stage'] != previous['stage'] or
                any(state['categories'][key]['active'] != category['active']
                    for key, category in previous['categories'].items())):
            reconcile_gate(queue, state)
    progress = progress_document(state, observations)
    progress_path = (progress_output or output) / 'progress.json'
    if completion_cache is None or completion_cache.get('_progress') != progress:
        write_json(progress_path, progress)
        if completion_cache is not None:
            completion_cache['_progress'] = progress
    if state['stage'] == 'finished' and progress_output:
        write_json(output / 'final_progress.json', progress)
    return state, progress


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records-file', type=Path, required=True)
    parser.add_argument('--queue-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--runtime-dir', type=Path,
                        help='Keep frequently updated progress on local temporary storage.')
    parser.add_argument('--release-bulk', action='store_true',
                        help='Run the full queue concurrently with remaining representatives.')
    parser.add_argument('--capture-timeout-s', type=float, default=1800,
                        help='Capture deadline in seconds; 0 disables the loading deadline.')
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    queue = CaptureQueue(args.queue_root)
    # One coordinator per queue; workers and existing terminal sessions are untouched.
    with (queue.root / 'recapture_coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = prepare_campaign(read_json(args.records_file), queue, output,
                                 capture_timeout_s=args.capture_timeout_s)
        if args.release_bulk and state['stage'] == 'pilot':
            state['stage'] = 'bulk'
            write_json(output / 'campaign.json', state)
            reconcile_gate(queue, state)
        runtime = args.runtime_dir.expanduser().resolve() if args.runtime_dir else output
        runtime.mkdir(parents=True, exist_ok=True)
        if runtime != output:
            # Replace the reference atomically; monitoring callers retain the
            # established path while all progress payload writes stay local.
            reference = output / '.progress.local.tmp'
            reference.unlink(missing_ok=True)
            reference.symlink_to(runtime / 'progress.json')
            reference.replace(output / 'progress.json')
        completion_cache = {}
        if args.prepare_only:
            observations = {c: observe_record(queue, r, completion_cache) for c, r in state['published'].items()}
            write_json(runtime / 'progress.json', progress_document(state, observations))
            print(json.dumps({'stage': state['stage'], 'published': len(state['published'])}), flush=True)
            return 0
        last_summary = None
        while True:
            state, progress = step_campaign(queue, output, state, completion_cache,
                                            progress_output=runtime if runtime != output else None)
            summary = {key: progress[key] for key in ('stage', 'counts', 'pilot_success_count', 'pilot_failure_count')}
            if summary != last_summary:
                print(json.dumps(summary), flush=True)
                last_summary = summary
            if state['stage'] == 'finished':
                return 1 if progress['terminal_failures'] else 0
            time.sleep(10)


if __name__ == '__main__':
    raise SystemExit(main())
