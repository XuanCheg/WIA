#!/usr/bin/env python3
"""Archive the six audited duplicate families without interrupting GPU engines.

Run with behavior Python:
  python src/wia/runtime/commands/archive_esi_repeats.py --campaign-root ~/esi_category_conversion/CAMPAIGN
Inspect the printed plan, then add --apply. Resume an existing archive with
--apply; --finish-pending only archives jobs whose original owners have exited.
Only the six audited families are eligible. Source JSON and sealed job inputs
remain unchanged. Existing images are moved, never decoded or validated.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import yaml

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
from wia.runtime.scene_capture_queue import CaptureQueue, TERMINAL, digest, read_json, write_json

AUDITED = {
    'Geometric Configuration': 210, 'Material Transparency': 93,
    'Traversable Passage': 11, 'Counting w Occlusion': 7,
    'Spatial Segmentation': 5, 'Liquid Volume': 12,
}


def camera_selector(family):
    """Read the original pure selector without importing the simulator package."""
    path = WIA_ESI_ROOT / 'src/active_explore/tasks' / family / 'common.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'extract_primary_camera_pose')
    namespace = {'Any': Any, 'Path': Path, 'normalize_text': lambda x: str(x or '').strip()}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['extract_primary_camera_pose']


def source_state(record, selectors):
    """Retain all inputs used by each audited native construction path."""
    raw = read_json(record['original_source'])
    payload = raw['metadata_json']
    payload = json.loads(payload) if isinstance(payload, str) else payload
    small = record['small_task']
    identity = [raw.get(k) for k in ('scene', 'room', 'runner_task')]
    if small in ('Geometric Configuration', 'Material Transparency'):
        return [identity, payload, raw.get('gt_camera_poses')]
    if small == 'Liquid Volume':
        return [identity, {key: payload.get(key) for key in
            ('left_category', 'left_model', 'right_category', 'right_model', 'fixed_env')}]
    family = 'cognitive_mapping' if small == 'Traversable Passage' else 'enumerative_perception'
    if family not in selectors:
        selectors[family] = camera_selector(family)
    pose = selectors[family](payload)
    camera = {key: pose.get(key) for key in ('position', 'quaternion_xyzw')}
    if small == 'Traversable Passage':
        return [identity, camera]
    question = payload['question_data']
    return [identity, camera, payload.get('floor_name'), question.get('case'),
            question.get('case_metadata'), question.get('render', {}).get('resolved_objects')]


def operation_key(record, goals):
    task = yaml.load(Path(goals.get(record['candidate_id'], {}).get('task_file')
                          or record['task_file']).read_text(), Loader=yaml.CSafeLoader)
    scene = yaml.load(Path(record['scene_file']).read_text(), Loader=yaml.CSafeLoader)
    return digest([
        record['big_task'], record['small_task'], record.get('scene_model'), record.get('room'),
        {key: value for key, value in task.items() if key not in ('id', 'source', 'public', 'scene')},
        {key: value for key, value in scene.items() if key not in ('id', 'source')},
        {key: value for key, value in task.get('public', {}).items()
         if key in ('success_criteria', 'constraints')},
    ])


def preference(record, queue, reviews):
    card = read_json(queue / 'category_records' / (record['candidate_id'] + '.json'), {})
    status = read_json(queue / 'jobs' / card.get('job_id', 'missing') / 'status.json', {})
    reviewed = reviews.get(record['candidate_id'], {})
    complete = status.get('state') in ('REVIEW_PENDING', 'CACHED')
    approved = complete and reviewed.get('decision') == 'approved'
    active = status.get('state') in ('CLAIMED', 'RUNNING', 'CAPTURED')
    return (not approved, not (complete and bool(reviewed)), not complete, not active,
            status.get('state') in ('FAILED', 'STALE', 'CANCELLED'), record['source_identity'])


def plan_archive(campaign, queue, review_file):
    goals = read_json(campaign / 'review_catalogue/goal_revisions.json', {}).get('records', {})
    reviews = read_json(review_file, {}).get('entries', {})
    revisions = read_json(campaign / 'review_catalogue/category_revisions.json')['categories']
    groups, selectors, batches = defaultdict(list), {}, {}
    for category, revision in revisions.items():
        path = Path(revision['new_batch_file'])
        batch = read_json(path)
        records = [row for row in batch['records'] if row['status'] == 'generated'
                   and row['small_task'] in AUDITED]
        if not records:
            continue
        batches[category] = str(path)
        for record in records:
            groups[operation_key(record, goals)].append((category, record))
    duplicates, counts = [], Counter()
    for signature, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        states = {digest(source_state(record, selectors)) for _, record in members}
        if len(states) != 1:
            raise ValueError('Audited native inputs no longer match: ' + signature)
        members.sort(key=lambda pair: preference(pair[1], queue, reviews))
        keeper = members[0][1]
        for category, record in members[1:]:
            counts[record['small_task']] += 1
            duplicates.append({'candidate_id': record['candidate_id'],
                'repeat_of': keeper['candidate_id'], 'retained_source': keeper['source_identity'],
                'category_key': category, 'operation_signature': signature, 'record': record})
    if dict(counts) != AUDITED:
        raise ValueError('The audited duplicate counts changed; no mutation was made: ' + str(dict(counts)))
    return {'schema': 'esi_repeat_archive_v1', 'created_at': time.time(),
            'counts': dict(counts), 'repeat_count': len(duplicates), 'batches': batches,
            'policy': 'Keep approved/reviewed complete captures, then complete, then active jobs; '
                      'retain all distinct layouts, starts, operation predicates and source provenance.',
            'entries': duplicates}


def withdraw_job(queue, job_id, entry, destination):
    """Atomically withdraw unclaimed tokens; a concurrent owner keeps its job."""
    for name in ('queued', 'held_for_category_review'):
        token = queue.root / name / (job_id + '.json')
        fence = queue.root / 'repeat_tokens' / token.name
        fence.parent.mkdir(exist_ok=True)
        try:
            token.rename(fence)
        except FileNotFoundError:
            continue
        write_json(destination / (name + '_token.json'), read_json(fence))
        fence.unlink()
    status = queue.status(job_id)
    if (status['state'] == 'QUEUED'
            and not (queue.root / 'claimed' / (job_id + '.json')).exists()):
        queue._save(job_id, {**status, 'state': 'CANCELLED', 'retryable': False,
            'archive_reason': 'Equivalent robot task archived at the user request.',
            'repeat_of': entry['repeat_of'], 'repeat_archive': str(destination)})


def relocate_directory(original, destination):
    """Move bytes within home; leave a compatibility link for sealed references."""
    original, destination = Path(original), Path(destination)
    if original.is_symlink() and original.resolve() == destination.resolve():
        return
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        original.rename(destination)
    if not original.exists() and not original.is_symlink():
        original.symlink_to(destination, target_is_directory=True)


def repeat_record(record, entry, destination):
    value = deepcopy(record)
    value.update(status='repeat', previous_status=record['status'],
        repeat_of=entry['repeat_of'], retained_source=entry['retained_source'],
        repeat_archive=str(destination), archive_reason='Equivalent robot task; original ESI QA retained.')
    return value


def drain_jobs(queue, root, plan):
    pending, archived = [], 0
    for entry in plan['entries']:
        job = entry.get('job_id')
        if not job:
            continue
        path = queue.job_dir(job)
        state = queue.status(job)['state']
        if (state not in TERMINAL or (queue.root / 'claimed' / (job + '.json')).exists()):
            pending.append(job)
            continue
        destination = root / 'tasks' / entry['candidate_id'] / 'capture' / job
        physical = path.resolve()
        if physical != destination.resolve():
            relocate_directory(physical, destination)
        archived += 1
    result = {'repeat_count': len(plan['entries']), 'archived_capture_jobs': archived,
              'pending_capture_jobs': pending, 'updated_at': time.time(), 'complete': not pending}
    write_json(root / 'result.json', result)
    return result


def apply_archive(campaign, queue_root, review_file):
    root, review = campaign / 'repeat', campaign / 'review_catalogue'
    root.mkdir(exist_ok=True)
    with (review / 'task_update.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = read_json(root / 'manifest.json')
        if plan is None:
            plan = plan_archive(campaign, queue_root, review_file)
            write_json(root / 'manifest.json', plan)
        queue = CaptureQueue(queue_root)
        retired = read_json(review / 'retired_candidates.json', {'candidates': {}})
        for entry in plan['entries']:
            retired['candidates'][entry['candidate_id']] = {
                'reason': 'Equivalent robot task archived in repeat at the user request.',
                'repeat_of': entry['repeat_of'], 'retired_at': plan['created_at'],
                'repeat_archive': str(root / 'tasks' / entry['candidate_id'])}
        write_json(review / 'retired_candidates.json', retired)
        write_json(queue_root / 'repeat_candidates.json', {'candidates': {
            entry['candidate_id']: {'repeat_of': entry['repeat_of'],
                'source_identity': entry['record']['source_identity']}
            for entry in plan['entries']}, 'manifest': str(root / 'manifest.json')})
        reviews = read_json(review_file, {})
        for entry in plan['entries']:
            destination = root / 'tasks' / entry['candidate_id']
            destination.mkdir(parents=True, exist_ok=True)
            card_path = queue_root / 'category_records' / (entry['candidate_id'] + '.json')
            card = read_json(card_path, {})
            if card.get('job_id'):
                entry['job_id'] = card['job_id']
                status_path = queue.job_dir(card['job_id']) / 'status.json'
                if not (destination / 'original_capture_status.json').exists():
                    shutil.copyfile(status_path, destination / 'original_capture_status.json')
                withdraw_job(queue, card['job_id'], entry, destination)
            if not (destination / 'original_record.json').exists():
                write_json(destination / 'original_record.json', entry['record'])
                write_json(destination / 'original_queue_record.json', card)
                write_json(destination / 'review_history.json', {
                    'entry': reviews.get('entries', {}).get(entry['candidate_id']),
                    'history': [row for row in reviews.get('history', []) if row['id'] == entry['candidate_id']]})
                shutil.copyfile(entry['record']['original_source'], destination / 'source.json')
            if card and card.get('status') != 'repeat':
                write_json(card_path, repeat_record(card, entry, destination))
        # Record the full job mapping before moving any mutable artifacts.
        write_json(root / 'manifest.json', plan)
        by_id = {entry['candidate_id']: entry for entry in plan['entries']}
        for key, batch_path in plan['batches'].items():
            batch = read_json(batch_path)
            backup = root / 'original_batches' / (key + '.json')
            if not backup.exists():
                backup.parent.mkdir(exist_ok=True)
                shutil.copyfile(batch_path, backup)
            for index, record in enumerate(batch['records']):
                entry = by_id.get(record['candidate_id'])
                if entry and record['status'] != 'repeat':
                    batch['records'][index] = repeat_record(record, entry, root / 'tasks' / record['candidate_id'])
            batch['counts'] = dict(Counter(record['status'] for record in batch['records']))
            batch['repeat_archive'] = str(root)
            write_json(batch_path, batch)
        revisions_path = review / 'category_revisions.json'
        if not (root / 'original_category_revisions.json').exists():
            shutil.copyfile(revisions_path, root / 'original_category_revisions.json')
        revisions = read_json(revisions_path)
        for key, batch_path in plan['batches'].items():
            revisions['categories'][key]['counts'] = read_json(batch_path)['counts']
            revisions['categories'][key]['repeat_archive'] = str(root)
        write_json(revisions_path, revisions)
        for entry in plan['entries']:
            record = entry['record']
            destination = root / 'tasks' / entry['candidate_id']
            directories = list(dict.fromkeys([str(Path(record[field]).parent)
                for field in ('task_file', 'scene_file') if record.get(field)]
                + ([record['diagnostic_dir']] if record.get('diagnostic_dir') else [])))
            for index, directory in enumerate(directories):
                path = Path(directory)
                if not path.resolve().is_relative_to(campaign.resolve()):
                    raise ValueError('Only campaign-owned candidate directories may be archived.')
                if not path.name.startswith('category_'):
                    raise ValueError('Refusing to archive a shared category directory: ' + directory)
                archived = destination / ('definition' if index == 0 else 'diagnostics_' + str(index))
                relocate_directory(path, archived)
                local = archived / 'record.json'
                original = read_json(local)
                if original and original.get('status') != 'repeat':
                    shutil.copyfile(local, destination / ('original_definition_record_' + str(index) + '.json'))
                    write_json(local, repeat_record(original, entry, destination))
        gate_path = queue_root / 'capture_sample_gate.json'
        gate = read_json(gate_path)
        if gate:
            for field in ('authorized_candidate_ids', 'selected_candidate_ids', 'representative_candidate_ids'):
                if field in gate:
                    gate[field] = list(dict.fromkeys(
                        by_id[key]['repeat_of'] if field != 'authorized_candidate_ids' and key in by_id else key
                        for key in gate[field] if field != 'authorized_candidate_ids' or key not in by_id))
            write_json(gate_path, gate)
        (root / 'README.md').write_text(
            '# Archived equivalent robot tasks\n\n'
            'Plan: retain one representative per audited operation/scene/start/scoring group; '
            'withdraw only unclaimed screenshot tokens, hide repeats through the existing Review retirement '
            'registry, and move definitions plus terminal capture jobs into this directory. '
            'Allow already owned GPU captures to finish without interrupting their engines.\n\n'
            'manifest.json maps every archived candidate to its retained task and original ESI source. '
            'All source questions, answers, task files, screenshots, diagnostics and review history are retained. '
            'Original paths are compatibility symlinks to the archived bytes; sealed capture inputs are unchanged. '
            'Original batches, records and statuses are retained for reversible restoration. '
            'No source failure or near-duplicate View Hallucination pair was removed.\n')
        result = drain_jobs(queue, root, plan)
        result['counts'] = plan['counts']
        write_json(root / 'result.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--queue-root', type=Path,
        default=ROOT / 'outputs/local_capture_20261014/queue_workers_initialized')
    parser.add_argument('--review-file', type=Path,
        default=ROOT / 'outputs/local_capture_20261014/gallery/reviews.json')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--finish-pending', action='store_true')
    args = parser.parse_args()
    campaign, queue = args.campaign_root.expanduser().resolve(), args.queue_root.expanduser().resolve()
    if args.finish_pending:
        result = drain_jobs(CaptureQueue(queue), campaign / 'repeat',
                            read_json(campaign / 'repeat/manifest.json'))
    elif args.apply:
        result = apply_archive(campaign, queue, args.review_file)
    else:
        plan = plan_archive(campaign, queue, args.review_file)
        result = {'dry_run': True, 'repeat_count': plan['repeat_count'], 'counts': plan['counts']}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
