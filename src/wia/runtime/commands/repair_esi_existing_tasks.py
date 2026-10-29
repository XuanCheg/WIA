#!/usr/bin/env python3
"""Version existing goals and publish repair evidence through the current Review index.

Run: python src/wia/runtime/commands/repair_esi_existing_tasks.py --campaign-root CAMPAIGN
     --audit-dir outputs/task_design_audit_20261016 --output-dir outputs/task_repair_20261016
Add --publish after reviewing prepared definitions. No scenes, captures, source answers,
composite memberships, original definitions or human annotations are modified.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

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
from wia.runtime.scene_capture_queue import digest, read_json, write_json
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition
from wia.tasks.temporal_policy import survey_policy
from wia.authoring.converters.temporal_understanding_two_phase import _public_task


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def repair(args):
    campaign, audit, output = (Path(value).resolve() for value in
                              (args.campaign_root, args.audit_dir, args.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    review = campaign / 'review_catalogue'
    revisions = read_json(review / 'category_revisions.json')
    overrides = read_json(review / 'goal_revisions.json', {'schema': 'esi_goal_revisions_v1', 'records': {}})
    output_definitions = campaign / 'goal_revisions/task_repair_20261016'
    changes, audits = {}, []
    for owner in ('mapping', 'perception', 'manipulation'):
        audits.extend(jsonlines(output / (owner + '_goal_audit.jsonl')))
        for row in jsonlines(output / (owner + '_goal_changes.jsonl')):
            changes[(row['candidate_id'], row['source_identity'])] = row
    audit_lookup = {(row.get('candidate_id'), row.get('source_identity')): row for row in audits}
    conflicts = jsonlines(output / 'distance_review.jsonl')
    conflict_sources = {row.get('source_identity') for row in conflicts}
    current_goals, goal_changes, records_all, changed, batches = [], [], [], [], {}
    scene_cache = {}
    counts = Counter()
    for category_key, revision in revisions['categories'].items():
        parent_batch = Path(revision['new_batch_file'])
        batch = read_json(parent_batch)
        new_batch = deepcopy(batch)
        for record in new_batch['records']:
            records_all.append(record)
            counts[record['status']] += 1
            if record['status'] != 'generated':
                continue
            if (record['big_task'], record['small_task']) == ('Physical Structure', 'Rigid Containment'):
                raise ValueError('Protected Rigid tasks cannot enter this migration.')
            candidate = record['candidate_id']
            original_path = Path(overrides['records'].get(candidate, {}).get('task_file', record['task_file']))
            original = yaml.load(original_path.read_text(), Loader=yaml.CSafeLoader)
            if original.get('repair', {}).get('scoring_version') == 'task_completion_v2':
                raise ValueError('This repair has already been applied; do not compound revisions.')
            task_definition(original)
            scene_path = record['scene_file']
            if scene_path not in scene_cache:
                scene_cache[scene_path] = yaml.load(Path(scene_path).read_text(), Loader=yaml.CSafeLoader)
                SceneDefinition.model_validate(scene_cache[scene_path])
            task = deepcopy(original)
            evaluation = task['private']['evaluation']
            composite = evaluation.get('operation') == 'composite_goals'
            members = evaluation['parameters']['members'] if composite else [{
                'id': task['id'], 'source_identity': record['source_identity'],
                'public': task['public'], 'evaluation': evaluation, 'bindings': task['private']['bindings']}]
            task_changes = []
            for member in members:
                source = member['source_identity']
                subclass = source.split('/')[1]
                before = deepcopy(member['public'])
                change = changes.get((candidate, source))
                if change:
                    member['public'] = deepcopy(change.get('new_goal', change.get('new_target')))
                parameters = member['evaluation'].get('parameters', {})
                if member['evaluation'].get('operation') == 'observe_two_phase_change':
                    parameters.update(survey_policy(parameters))
                    raw = read_json(WIA_ESI_ROOT / 'dataset/json_clean' / source)
                    from types import SimpleNamespace
                    member['public'] = _public_task(SimpleNamespace(raw=raw), parameters['public_options'], parameters)
                    row = {'candidate_id': candidate, 'source_identity': source,
                        'source_category': source.split('/')[:2], 'status': 'optimized',
                        'actual_operation': 'observe_two_phase_change', 'parameter_branch': survey_policy(parameters),
                        'old_goal': before, 'new_goal': member['public'],
                        'embodied_skills': ['bounded_read_only_survey', 'phase_2_navigation', 'temporal_comparison'],
                        'scored_capabilities': ['native_change_answer', 'phase_2_departure_and_container_approach'],
                        'qa': {'preserved': True, 'executed': True, 'scored': True},
                        'validation': 'CPU protocol replay; native action execution pending'}
                    audits.append(row)
                    task_changes.append(row)
                else:
                    row = audit_lookup.get((candidate, source))
                    if row is None:
                        raise ValueError('Missing subclass/member audit: ' + source)
                    if change:
                        task_changes.append(change)
                current_goals.append({'candidate_id': candidate, 'source_identity': source,
                    'subclass': subclass, 'operation': member['evaluation'].get('operation', member['evaluation']['type']),
                    'status': row.get('status', row.get('decision')), 'member_id': member['id']})
                if not composite:
                    task['public'] = member['public']
            if composite:
                task['public']['objective'] = 'Complete the following goals in one shared scene.\n' + '\n'.join(
                    f"{index + 1}. [{m['source_identity'].split('/')[1]}] {m['public']['objective']}"
                    for index, m in enumerate(members))
                task['public']['success_criteria'] = (
                    'Complete each declared native obligation. Unordered goals receive the fraction of confirmed items; '
                    'ordered goals receive their confirmed legal prefix. Equivalent unordered physical requirements '
                    'share credit only when all native conditions agree. Historical checkpoints remain credited, '
                    'while protection and safety constraints remain effective. Unavailable items stay declared; '
                    'partial validity is reported separately from a complete comparable score. Original source QA remains separate.')
                task['public']['constraints'] = list(dict.fromkeys(c for m in members for c in m['public'].get('constraints', [])))
                if 'composite' in record:
                    for public_goal, member in zip(record['composite']['public_goals'], members):
                        public_goal.update(deepcopy(member['public']))
            task['version'] += 1
            task['requirements']['capabilities'] = list(dict.fromkeys(task['requirements']['capabilities'] +
                ['ee_translation', 'ee_rotation', 'trunk_height', 'gripper']))
            task['repair'] = {'scoring_version': 'task_completion_v2', 'previous_task_file': str(original_path),
                'previous_task_version': original['version'], 'source_category': [record['big_task'], record['small_task']],
                'run_mode_for_validation': 'diagnostic', 'native_validation': 'not_executed',
                'source_conflict': 'unresolved' if record['source_identity'] in conflict_sources else None}
            if task['scene'] != original['scene'] or task['defaults'] != original['defaults'] or task['private']['bindings'] != original['private']['bindings']:
                raise ValueError('Repair changed protected world/budget/bindings.')
            task_definition(task)
            path = output_definitions / candidate / 'task.yaml'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.dump(task, Dumper=yaml.CSafeDumper, sort_keys=False, allow_unicode=True))
            goal_revision = 'goal_' + digest({'task': task, 'scoring_version': 'task_completion_v2'})[:24]
            repair_metadata = {'scoring_version': 'task_completion_v2', 'run_mode': 'diagnostic',
                'old_public': original['public'], 'new_public': task['public'],
                'previous_version': original['version'], 'current_version': task['version'],
                'native_validation': 'not_executed', 'capture_evidence_scope': 'historical_layout_only',
                'subgoal_changes': task_changes}
            override = {'task_file': str(path), 'goal_revision': goal_revision,
                'captured_task_file': record.get('captured_task_file', str(original_path)),
                'capture_reused': True, 'capture_validates_new_score': False,
                'task_repair': repair_metadata, 'composite': record.get('composite'),
                'scope': 'existing_goal_and_scoring_repair'}
            overrides['records'][candidate] = override
            record.update(override, goal_predicate=task['private']['evaluation'])
            goal_changes.extend(task_changes)
            changed.append({'candidate_id': candidate, 'task_file': str(path), 'previous_task_file': str(original_path),
                            'goal_revision': goal_revision, 'version': task['version']})
        destination = output_definitions / 'batches' / category_key / 'batch_manifest.json'
        new_batch.update(goal_revision_parent_batch=str(parent_batch), goal_revision_scope='existing_task_repair',
                         source_cpu_batch=batch.get('source_cpu_batch', str(parent_batch)))
        write_json(destination, new_batch)
        batches[category_key] = str(destination)
    audited_sources = {row.get('source_identity') for row in audits}
    for record in records_all:
        if record['source_identity'] not in audited_sources:
            if record['status'] not in ('repeat', 'merged', 'conversion_failed', 'needs_human_decision'):
                raise ValueError('Missing executable goal audit: ' + record['source_identity'])
            audits.append({'candidate_id': record.get('candidate_id'), 'source_identity': record['source_identity'],
                'source_category': [record['big_task'], record['small_task']],
                'status': 'kept' if record['status'] in ('repeat', 'merged') else 'needs_review',
                'source_record_status': record['status'], 'executable_task_created': False,
                'source_mapping': {key: record.get(key) for key in ('repeat_of', 'merged_into', 'composition_representative')},
                'reason': 'Preserved source-only status and existing provenance; no new task or combination.',
                'validation': 'Current revision manifest read; no native execution.'})
    rigid = []
    for saved in read_json(audit / 'rigid_definitions.json'):
        path = Path(saved['path'])
        actual = yaml.safe_load(path.read_text())
        if actual != saved['definition']:
            raise ValueError('Protected Rigid baseline differs from the audit: ' + str(path))
        row = {'candidate_id': actual['id'], 'source_identity': actual['source']['json'],
               'task_file': str(path), 'source_category': ['Physical Structure', 'Rigid Containment'],
               'status': 'protected_baseline', 'reason': 'Read-only comparison; task/layout/visibility/budget/approval unchanged.'}
        rigid.append(row)
    audits.extend(rigid)
    active_sources = {row['source_identity'] for row in current_goals}
    source_status = {row['source_identity']: row['status'] for row in records_all}
    for row in audits:
        source = row['source_identity']
        row['audit_scope'] = ('protected_baseline' if row['status'] == 'protected_baseline' else
                              'active_goal' if source in active_sources else 'source_only')
        row['source_record_status'] = source_status.get(source, 'protected_baseline')
    for filename, rows in [('goal_audit.jsonl', audits), ('goal_changes.jsonl', goal_changes),
                           ('unresolved_review.jsonl', conflicts)]:
        (output / filename).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    result = {'schema': 'existing_task_repair_v1', 'published': False, 'current_nonrigid_tasks': len(changed),
        'current_nonrigid_goals': len(current_goals), 'protected_rigid_tasks': len(rigid),
        'subclasses': sorted({row['subclass'] for row in current_goals} | {'Rigid Containment'}),
        'source_statuses': dict(counts), 'scene_definitions_validated': len(scene_cache),
        'all_task_schemas_valid': True, 'source_answers_modified': 0, 'scene_files_modified': 0,
        'new_combinations': 0, 'capture_jobs_created': 0, 'changed': changed, 'batches': batches,
        'native_execution': 'not_executed_by_this_script'}
    write_json(output / 'prepared_revisions.json', result)
    if args.publish:
        with (campaign / '.scheduler.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = read_json(review / 'category_revisions.json')
            if current != revisions:
                raise ValueError('Current category revisions changed during preparation; rebase the migration.')
            write_json(output / 'category_revisions_before.json', revisions)
            write_json(output / 'goal_revisions_before.json', read_json(review / 'goal_revisions.json', {}))
            for key, batch_file in batches.items():
                current['categories'][key].update(new_batch_file=batch_file, goal_revision_scope='existing_task_repair',
                                                  complete=True, updated_at=time.time())
            write_json(review / 'task_repair_reviews.json', {'records': {row['source_identity']: row for row in conflicts}})
            write_json(review / 'goal_revisions.json', overrides)
            write_json(review / 'category_revisions.json', current)
        result['published'] = True
        write_json(output / 'prepared_revisions.json', result)
    return {key: value for key, value in result.items() if key not in ('changed', 'batches')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    print(json.dumps(repair(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
