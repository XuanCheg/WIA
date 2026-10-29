#!/usr/bin/env python3
"""Publish compatible composite tasks, withdraw predecessors and resume capture.

Run with behavior Python from the project root:
  python src/wia/runtime/commands/compose_esi_tasks.py --campaign-root ~/esi_category_conversion/CAMPAIGN
The default prints a CPU plan. Add --apply to publish while capture is paused;
then --resume after integration checks. Repeating --apply resumes its journal.
Use --refresh-metadata to update historical member diagnostics during capture.
Only declared superseded PNGs are cleaned. Sources, task definitions, review
history, logs, approved Rigid work and the prior repeat archive are preserved.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import sys
import time

import yaml

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
from wia.runtime.esi_category_capture import freeze_working_code, publish_record
from wia.runtime.esi_task_composition import build_group
from wia.runtime.scene_capture_queue import CaptureQueue, read_json, write_json
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition
from wia.runtime.commands.publish_esi_annotation_repairs import cleanup

CAMPAIGN_DIR = 'composite_tasks_20261016'


def policies():
    from wia.authoring.converters import composition_mapping, composition_perception, composition_manipulation
    return [composition_mapping, composition_perception, composition_manipulation]


def load_inputs(campaign, queue):
    revisions = read_json(campaign / 'review_catalogue/category_revisions.json')['categories']
    goals = read_json(campaign / 'review_catalogue/goal_revisions.json', {}).get('records', {})
    reviews = read_json(ROOT / 'outputs/local_capture_20261014/gallery/reviews.json', {'entries': {}})
    records, tasks, scenes, batches = {}, {}, {}, {}
    for key, revision in revisions.items():
        batch = read_json(revision['new_batch_file'])
        batches[key] = batch
        for record in batch['records']:
            if record['status'] != 'generated':
                continue
            record = deepcopy(record)
            identity = record['candidate_id']
            record['composition_source_raw'] = read_json(record['original_source'])
            card = read_json(queue / 'category_records' / (identity + '.json'), {})
            if card.get('job_id'):
                status = read_json(queue / 'jobs' / card['job_id'] / 'status.json', {})
                if status.get('receipt'):
                    receipt = read_json(status['receipt'], {})
                    record['interaction_setup'] = receipt.get('interaction_setup', {})
            tasks[identity] = yaml.load(Path(goals.get(identity, {}).get('task_file') or record['task_file']).read_text(), Loader=yaml.CSafeLoader)
            scenes[identity] = yaml.load(Path(record['scene_file']).read_text(), Loader=yaml.CSafeLoader)
            records[identity] = record
    return revisions, batches, records, tasks, scenes, reviews


def prepare(campaign, queue):
    revisions, batches, records, tasks, scenes, feedback = load_inputs(campaign, queue)
    repairs = {}
    for module in policies():
        repair = getattr(module, 'repair_definitions', None)
        if repair:
            for identity, record in records.items():
                result = repair(record, tasks[identity], scenes[identity], feedback=feedback)
                if result.get('changed'):
                    tasks[identity], scenes[identity] = result['task'], result['scene']
                    repairs[identity] = {key: value for key, value in result.items() if key not in ('task', 'scene')}
    groups = [group for module in policies()
              for group in module.plan_groups(list(records.values()), tasks, scenes, feedback=feedback)]
    grouped = {identity for group in groups for identity in group['member_ids']}
    for identity, record in records.items():
        if identity in grouped:
            continue
        if record['big_task'] not in ('Specular Reflection', 'Temporal Understanding'):
            raise ValueError('No owned composition policy for ' + record['source_identity'])
        checks = ['Verify the native supported collision-free robot start and mounted head/wrist views.',
                  'Retain the original source QA separately from measured physical task completion.']
        if record['big_task'] == 'Specular Reflection':
            checks += ['Preserve original mirror pose, optical material, prop identities and source reference frame.',
                       'Inspect each native approach, pickup or return checkpoint required by this original task.']
        elif record['small_task'] == 'Unobserved Change':
            checks += ['Phase 1 must expose the original relevant contents through legitimate mounted observations; no manipulation is permitted.',
                       'Keep one world through the real Phase 1 to Phase 2 transition; only Phase 2 permits operations.',
                       'Score the original answer and measured phase/navigation proof; later views cannot replace missing Phase 1 evidence.']
        else:
            checks += ['Inspect the original target population, actual visits, guarded pickup and required return.',
                       'Preserve unpicked-object geometry and the original observation-history distinction.']
        groups.append({'representative_id': identity, 'member_ids': [identity],
            'label': record['small_task'], 'reason': 'Preserve this distinct source-specific optical or temporal world.',
            'review_goals': checks, 'requires_recapture': False})
    covered = [identity for group in groups for identity in group['member_ids']]
    if Counter(covered) != Counter({identity: 1 for identity in records}):
        raise ValueError('Composition policies must partition every current generated source exactly once.')
    output = campaign / CAMPAIGN_DIR
    derivatives = []
    for group in groups:
        if (len(group['member_ids']) == 1 and not group.get('requires_recapture')
                and not any(key in repairs for key in group['member_ids'])
                and records[group['representative_id']]['small_task'] != 'Action Order Inference'):
            continue
        record, task, scene = build_group(group, records, tasks, scenes, output)
        SceneDefinition.model_validate(scene)
        task_definition(task)
        record['prior_feedback'] = [dict(feedback['entries'][key], previous_candidate_id=key,
            scope='Previous layout/goal feedback; new derivatives require fresh review.')
            for key in group['member_ids'] if key in feedback.get('entries', {})]
        record['feedback_repairs'] = [repairs[key] for key in group['member_ids'] if key in repairs]
        if record['small_task'] == 'Action Order Inference':
            record['change_summary'].append('Enforce fresh horizontal clearance between independent original candidates; preserve each native support/contact check.')
        derivatives.append({'group': group, 'record': record, 'task': task, 'scene': scene})
    summary = {'active_sources_before': len(records), 'active_groups_after': len(groups),
        'merged_sources': len(records) - len(groups), 'multi_goal_groups': sum(len(g['member_ids']) > 1 for g in groups),
        'new_capture_jobs': len(derivatives), 'feedback_repairs': len(repairs),
        'ordered_mapping_routes_retained': sum(bool(tasks[key]['private']['evaluation'].get('parameters', {}).get('required_order'))
            for key in records if records[key]['small_task'] == 'Long-Term Navigation')}
    return {'revisions': revisions, 'batches': batches, 'records': records, 'groups': groups,
            'derivatives': derivatives, 'summary': summary}


def save_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(yaml.safe_dump(value, sort_keys=False))


def withdraw(queue, job):
    status = queue.status(job)
    if status and status['state'] == 'QUEUED':
        queue.cancel(job)
    held = queue.root / 'held_for_category_review' / (job + '.json')
    if held.exists():
        destination = queue.root / 'superseded_tokens' / held.name
        destination.parent.mkdir(exist_ok=True)
        held.replace(destination)


def preserve_member_history(campaign, queue_root):
    """Refresh review-only provenance without changing any sealed capture input."""
    output, review = campaign / CAMPAIGN_DIR, campaign / 'review_catalogue'
    with (review / 'task_update.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        journal = read_json(output / 'publication.json', {})
        if not journal.get('complete'):
            raise ValueError('Complete publication before refreshing member history.')
        feedback = read_json(ROOT / 'outputs/local_capture_20261014/gallery/reviews.json', {'entries': {}})
        updated, failures = {}, 0
        for original in journal['pending_records']:
            record = deepcopy(original)
            by_source, source_reviews, logs = {}, {}, set(record.get('diagnostic_logs', []))
            historical = [item for item in record.get('prior_conversion_failures', [])
                          if item.get('origin') != 'composite_parent_capture']
            for parent in record['parent_candidate_ids']:
                card = read_json(queue_root / 'category_records' / (parent + '.json'), {})
                if parent in feedback.get('entries', {}):
                    entry = feedback['entries'][parent]
                    source_reviews[card.get('source_identity')] = {'candidate_id': parent,
                        **{key: entry[key] for key in ('decision', 'notes', 'revision', 'updated_at') if key in entry}}
                job = card.get('job_id')
                if not job:
                    continue
                status = read_json(queue_root / 'jobs' / job / 'status.json', {})
                receipt = read_json(status['receipt'], {}) if status.get('receipt') else {}
                setup = receipt.get('interaction_setup') or {}
                rejected = setup.get('accepted') is False
                if status.get('state') not in ('FAILED', 'STALE') and not rejected:
                    continue
                diagnostic = {'state': status.get('state', 'UNKNOWN'),
                    'failure_stage': status.get('failure_stage') or ('interaction_setup' if rejected else 'native_capture'),
                    'error': status.get('error') or setup.get('reason') or 'Previous native capture failed; retained logs provide details.',
                    'job_id': job, 'candidate_id': parent}
                by_source[card.get('source_identity')] = diagnostic
                historical.append({**diagnostic, 'origin': 'composite_parent_capture',
                    'source': card.get('source_identity')})
                native_log = status.get('native_log')
                if not native_log and status.get('native_output'):
                    native_log = str(Path(status['native_output']) / 'native.log')
                if native_log:
                    logs.add(native_log)
            for goal in record['composite']['public_goals']:
                if goal['source'] in by_source:
                    goal['previous_capture'] = by_source[goal['source']]
                if goal['source'] in source_reviews:
                    goal['previous_review'] = source_reviews[goal['source']]
            record['prior_conversion_failures'] = historical
            record['diagnostic_logs'] = sorted(logs)
            if by_source:
                record['warning_codes'] = sorted(set(record.get('warning_codes', [])) |
                    {'prior_member_capture_failure_requires_review'})
                failures += len(by_source)
            record['prior_feedback'] = [dict(feedback['entries'][key], previous_candidate_id=key,
                scope='Previous layout/goal feedback; new derivatives require fresh review.')
                for key in record['parent_candidate_ids'] if key in feedback.get('entries', {})]
            updated[record['candidate_id']] = record
            card_path = queue_root / 'category_records' / (record['candidate_id'] + '.json')
            card = read_json(card_path, {})
            for field in ('composite', 'prior_conversion_failures', 'diagnostic_logs', 'warning_codes', 'prior_feedback'):
                card[field] = record[field]
            write_json(card_path, card)
            diagnostic_path = Path(record['diagnostic_dir']) / 'record.json'
            diagnostic_record = read_json(diagnostic_path, record)
            for field in ('composite', 'prior_conversion_failures', 'diagnostic_logs', 'warning_codes', 'prior_feedback'):
                diagnostic_record[field] = record[field]
            write_json(diagnostic_path, diagnostic_record)
        for revision in read_json(review / 'category_revisions.json')['categories'].values():
            path = Path(revision['new_batch_file'])
            batch = read_json(path)
            if any(row['candidate_id'] in updated for row in batch['records']):
                batch['records'] = [updated.get(row['candidate_id'], row) for row in batch['records']]
                write_json(path, batch)
        journal['pending_records'] = list(updated.values())
        journal['member_history'] = {'updated_at': time.time(), 'records': len(updated),
            'previous_native_failures': failures, 'sealed_capture_inputs_changed': False}
        write_json(output / 'publication.json', journal)
        return journal['member_history']


def apply(campaign, queue_root):
    output, review = campaign / CAMPAIGN_DIR, campaign / 'review_catalogue'
    output.mkdir(exist_ok=True)
    if read_json(queue_root / 'capture_pause.json', {}).get('state') != 'PAUSED_BY_USER':
        raise ValueError('Pause screenshot claims before publishing composite replacements.')
    with (review / 'task_update.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        journal = read_json(output / 'publication.json')
        if journal and journal.get('complete'):
            journal['cleanup'] = cleanup(campaign, queue_root, journal)
            write_json(output / 'publication.json', journal)
            return journal
        queue = CaptureQueue(queue_root)
        if journal is None:
            plan = prepare(campaign, queue_root)
            write_json(output / 'plan.json', {**plan['summary'], 'groups': plan['groups']})
            write_json(output / 'revisions_before.json', plan['revisions'])
            write_json(output / 'reviews_before.json', read_json(ROOT / 'outputs/local_capture_20261014/gallery/reviews.json'))
            frozen = freeze_working_code(queue_root)
            replacements = {key: item['record'] for item in plan['derivatives'] for key in item['group']['member_ids']}
            chosen = {item['group']['representative_id'] for item in plan['derivatives']}
            retired, cleanup_jobs = read_json(review / 'retired_candidates.json', {'candidates': {}}), {}
            for old, new in replacements.items():
                retired['candidates'][old] = {'reason': 'Replaced by a compatible composite or feedback repair.',
                    'replacement_candidate_id': new['candidate_id'], 'retired_at': time.time()}
                card = read_json(queue_root / 'category_records' / (old + '.json'), {})
                if card.get('job_id'):
                    withdraw(queue, card['job_id'])
                    cleanup_jobs[card['job_id']] = {'replacement_candidate_id': new['candidate_id'],
                                                   'replacement_published': True}
            # Retirement is visible before obsolete screenshots are removed.
            write_json(review / 'retired_candidates.json', retired)
            by_member = {key: group for group in plan['groups'] for key in group['member_ids']}
            revisions = deepcopy(plan['revisions'])
            for category, batch in plan['batches'].items():
                old_file = Path(plan['revisions'][category]['new_batch_file'])
                write_json(output / 'original_batches' / (category + '.json'), batch)
                rows = []
                for old in batch['records']:
                    key = old['candidate_id']
                    if key in replacements:
                        replacement = replacements[key]
                        if key in chosen:
                            rows.append(deepcopy(replacement))
                        else:
                            rows.append({**old, 'status': 'merged', 'merged_into': replacement['candidate_id'],
                                'merged_group': replacement['composite']['group_id'],
                                'review_goals': by_member[key].get('review_goals', [])})
                    else:
                        row = deepcopy(old)
                        if key in by_member:
                            row['review_goals'] = by_member[key].get('review_goals', [])
                            card_path = queue_root / 'category_records' / (key + '.json')
                            card = read_json(card_path)
                            if card:
                                card['review_goals'] = row['review_goals']
                                write_json(card_path, card)
                        rows.append(row)
                batch = {**batch, 'records': rows, 'counts': dict(Counter(r['status'] for r in rows)),
                         'source_cpu_batch': batch.get('source_cpu_batch', str(old_file))}
                path = output / 'batches' / category / 'batch_manifest.json'
                write_json(path, batch)
                revisions[category] = {**revisions[category], 'new_batch_file': str(path),
                    'counts': batch['counts'], 'composition_publication': str(output / 'publication.json')}
            for item in plan['derivatives']:
                record = item['record']
                save_yaml(record['task_file'], item['task'])
                save_yaml(record['scene_file'], item['scene'])
                write_json(Path(record['diagnostic_dir']) / 'record.json', record)
                write_json(Path(record['diagnostic_dir']) / 'private_qa.json',
                           {'sources': [m['qa'] for m in item['task']['private']['evaluation']['parameters']['members']]})
            journal = {'schema': 'esi_composite_publication_v1', **plan['summary'], 'complete': False,
                'output_dir': str(output), 'code': frozen, 'cleanup_jobs': cleanup_jobs,
                'cleanup_on_submission': True, 'samples': [], 'retired_ids': sorted(replacements),
                'pending_records': [item['record'] for item in plan['derivatives']]}
            write_json(output / 'publication.json', journal)
            write_json(review / 'category_revisions.json', {'categories': revisions})
        frozen = journal['code']
        policy_path = queue_root / 'capture_runtime_policy.json'
        policy = read_json(policy_path, {})
        for field in ('allowed_code_commits', 'review_allowed_code_commits'):
            policy[field] = sorted(set(policy.get(field, policy.get('allowed_code_commits', []))) | {frozen['code_commit']})
        write_json(policy_path, policy)
        submitted = {row['candidate_id'] for row in journal['samples']}
        for record in journal['pending_records']:
            if record['candidate_id'] in submitted:
                continue
            published = publish_record(record, queue_root, frozen=frozen, priority=300,
                batch_file=output / 'publication.json')
            if not published.get('job_id'):
                raise RuntimeError('Composite capture submission failed: ' + str(published.get('capture_submission_error')))
            journal['samples'].append({'candidate_id': record['candidate_id'], 'job_id': published['job_id']})
            write_json(output / 'publication.json', journal)
            print('Published composite/repair ' + str(len(journal['samples'])) + '/' + str(len(journal['pending_records'])), flush=True)
        jobs = {row['candidate_id']: row['job_id'] for row in journal['samples']}
        represented, priority_samples = set(), []
        ordered = sorted(journal['pending_records'], key=lambda row: (
            not any(item.get('decision') == 'needs_fix' for item in row.get('prior_feedback', [])),
            row['candidate_id']))
        for record in ordered:
            tags = {(tag['big_task'], tag['small_task']) for tag in record['category_tags']}
            if tags - represented:
                queue.prioritize(jobs[record['candidate_id']], 1000)
                priority_samples.append(record['candidate_id'])
                represented.update(tags)
        journal['priority_sample_candidate_ids'] = priority_samples
        current_ids = set()
        for revision in read_json(review / 'category_revisions.json')['categories'].values():
            current_ids.update(row['candidate_id'] for row in read_json(revision['new_batch_file'])['records']
                               if row['status'] == 'generated')
        gate_path = queue_root / 'capture_sample_gate.json'
        gate = read_json(gate_path, {})
        gate.update(authorized_candidate_ids=sorted(current_ids), selected_candidate_ids=[],
                    state='RELEASED_AFTER_USER_CONFIRMATION', reason='User authorized merged-task publication and resumed rendering after integration checks.')
        write_json(gate_path, gate)
        journal['complete'], journal['completed_at'] = True, time.time()
        write_json(output / 'publication.json', journal)
        journal['cleanup'] = cleanup(campaign, queue_root, journal)
        write_json(output / 'publication.json', journal)
        return journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--queue-root', type=Path, default=ROOT / 'outputs/local_capture_20261014/queue_workers_initialized')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--refresh-metadata', action='store_true',
                        help='Refresh parent feedback and failure provenance; preserve sealed capture inputs.')
    args = parser.parse_args()
    campaign, queue = args.campaign_root.expanduser().resolve(), args.queue_root.expanduser().resolve()
    if args.refresh_metadata:
        print(json.dumps(preserve_member_history(campaign, queue), indent=2))
    elif args.resume:
        publication = read_json(campaign / CAMPAIGN_DIR / 'publication.json', {})
        if not publication.get('complete'):
            raise ValueError('Complete composite publication before resuming capture.')
        pause = read_json(queue / 'capture_pause.json')
        pause.update(state='RUNNING', resumed_at=time.time(), reason='Composite goals, Review metadata and obsolete-cache retirement are published; user authorized resumption.')
        write_json(queue / 'capture_pause.json', pause)
        print('Resident capture workers resumed.', flush=True)
    elif args.apply:
        result = apply(campaign, queue)
        preserve_member_history(campaign, queue)
        print(json.dumps({key: result.get(key) for key in ('active_sources_before', 'active_groups_after', 'merged_sources', 'new_capture_jobs', 'complete', 'cleanup')}, indent=2))
    else:
        print(json.dumps(prepare(campaign, queue)['summary'], indent=2))


if __name__ == '__main__':
    main()
