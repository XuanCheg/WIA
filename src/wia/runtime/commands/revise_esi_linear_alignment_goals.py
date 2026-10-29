#!/usr/bin/env python3
"""Revise existing Linear Alignment goals while reusing scenes and captures.

Run with behavior Python:
  python src/wia/runtime/commands/revise_esi_linear_alignment_goals.py --campaign-root CAMPAIGN
Only separate task YAMLs and metadata pointers are written. Existing task/scene
files, private QA, capture inputs, screenshots and queue tokens remain unchanged.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import sys
import time

import yaml

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.authoring.converters.spatial_relations_linear_alignment import revise_goal
from wia.runtime.scene_capture_queue import digest, read_json, write_json
from wia.tasks.loader import task_definition


def revise(campaign, output=None):
    campaign = Path(campaign).expanduser().resolve()
    output = Path(output).expanduser().resolve() if output else campaign / 'goal_revisions/linear_alignment_v2'
    inventory = read_json(campaign / 'inventory.json')['categories']
    category = next(row for row in inventory
                    if (row['big_task'], row['small_task']) == ('Spatial Relations', 'Linear Alignment'))
    job = read_json(campaign / 'jobs.json')['jobs'][category['key']]
    parent_batch = Path(job['batch_file'])
    manifest = read_json(parent_batch)
    revised = deepcopy(manifest)
    overrides = read_json(campaign / 'review_catalogue/goal_revisions.json',
                          {'schema': 'esi_goal_revisions_v1', 'records': {}})
    changed, failures = [], 0
    for record in revised['records']:
        if not record.get('task_file'):
            failures += 1
            continue
        original_task = Path(record['task_file'])
        previous = yaml.safe_load(original_task.read_text())
        task = revise_goal(previous)
        # The physical scene and all source/identity data are inherited verbatim.
        for field in previous.keys() - {'version', 'public', 'requirements', 'private'}:
            if task[field] != previous[field]:
                raise ValueError('A goal-only revision changed task field: ' + field)
        if (task['private']['bindings'] != previous['private']['bindings']
                or task['public']['environment'] != previous['public']['environment']):
            raise ValueError('A goal-only revision changed bindings or the environment.')
        task_definition(task)
        revision = 'goal_' + digest({'candidate_id': record['candidate_id'],
                                    'public': task['public'],
                                    'evaluation': task['private']['evaluation']})[:24]
        task_path = output / record['candidate_id'] / 'task.yaml'
        task_path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(task, sort_keys=False, allow_unicode=True)
        if not task_path.is_file() or task_path.read_text() != text:
            task_path.write_text(text)
        captured_task = record.get('captured_task_file', str(original_task))
        record.update(task_file=str(task_path), captured_task_file=captured_task,
                      goal_revision=revision, goal_predicate=task['private']['evaluation'],
                      operation=task['private']['evaluation']['operation'], capture_reused=True)
        record['change_summary'] = [
            'Arrange the three original objects in a straight line; an initially valid line allows zero actions.'
            if text == 'Retain the guarded R1 navigation survey and stationary return around the initialized formation.'
            else text for text in record.get('change_summary', [])]
        overrides['records'][record['candidate_id']] = {
            'task_file': str(task_path), 'goal_revision': revision,
            'captured_task_file': captured_task,
            'scope': 'objective_and_success_only', 'capture_reused': True}
        changed.append(record['candidate_id'])
    batch_file = output / 'batch_manifest.json'
    if parent_batch != batch_file:
        revised['goal_revision_parent_batch'] = str(parent_batch)
    revised['goal_revision_scope'] = 'objective_and_success_only'
    revised['capture_reused'] = True
    write_json(batch_file, revised)
    write_json(campaign / 'review_catalogue/goal_revisions.json', overrides)
    revisions_path = campaign / 'review_catalogue/category_revisions.json'
    revisions = read_json(revisions_path, {'categories': {}})
    previous_revision = revisions['categories'].get(category['key'], {})
    revisions['categories'][category['key']] = {
        **previous_revision, 'complete': True, 'new_batch_file': str(batch_file),
        'goal_revision_scope': 'objective_and_success_only', 'capture_reused': True,
        'updated_at': time.time()}
    write_json(revisions_path, revisions)
    with (campaign / '.scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_json(campaign / 'jobs.json')
        state['jobs'][category['key']].update(batch_file=str(batch_file),
            goal_revision_batch=str(batch_file), capture_reused=True, updated_at=time.time())
        write_json(campaign / 'jobs.json', state)
    result = {'complete': True, 'category': category['small_task'],
              'revised_tasks': len(changed), 'retained_diagnostic_records': failures,
              'source_records': len(revised['records']), 'batch_file': str(batch_file),
              'scene_files_changed': 0, 'capture_jobs_created': 0,
              'screenshots_changed': 0, 'original_task_files_changed': 0,
              'capture_reused': True, 'updated_at': time.time()}
    write_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    import json
    print(json.dumps(revise(args.campaign_root, args.output_dir)), flush=True)


if __name__ == '__main__':
    main()
