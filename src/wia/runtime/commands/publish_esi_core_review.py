#!/usr/bin/env python3
"""Publish only the provisional Core shortlist to the existing Review UI.

Run: python src/wia/runtime/commands/publish_esi_core_review.py --root CAMPAIGN [--watch]
Reads metadata and existing capture receipts; never renders or changes reviews.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT
import argparse
from collections import Counter
from pathlib import Path
import sys
import time
import yaml

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
from wia.runtime.scene_capture_queue import read_json, write_json
from wia.runtime.commands.update_esi_review_catalogue import task_summary


def publish(root):
    selection = read_json(root / 'source_shortlist.json')
    generation = read_json(root / 'candidate_generation.json')
    core_sources = {s for row in selection['core_candidates'] for s in row['member_sources']}
    final = read_json(root / 'core_manifest.json', {})
    final_ids = {row['candidate_id'] for row in final.get('tasks', [])}
    queue = root / 'queue'
    completed, pending, groups, candidates = [], [], Counter(), []
    selection_by_source = {source: item for item in selection['core_candidates'] for source in item['member_sources']}
    for generated in generation['records']:
        if generated['source_identity'] not in core_sources:
            continue
        key = generated['candidate_id']
        path = queue / 'category_records' / (key + '.json')
        record = read_json(path, generated)
        record['core_selection_status'] = ('protected_rigid' if generated.get('protected_rigid') else
            'frozen_core' if key in final_ids else
            'core_candidate' if generated['source_identity'] in core_sources else 'backup_candidate')
        record['core_revision'] = generation['revision']
        status = read_json(queue / 'jobs' / record['job_id'] / 'status.json', {}) if record.get('job_id') else {}
        public = {}
        if record.get('task_file'):
            public = (yaml.safe_load(Path(record['task_file']).read_text()) or {}).get('public', {})
        row = task_summary(record, status, public)
        receipt = read_json(status['receipt'], {}) if status.get('receipt') else {}
        setup = receipt.get('interaction_setup') or {}
        if setup.get('accepted') is False:
            row.update(placement_state='needs_repair', instance_status='interaction_setup_failed',
                placement_warning=setup.get('reason', 'Native placement was rejected.'),
                failure=setup.get('reason', 'Native placement was rejected.'))
        row['warning_codes'] = sorted(set(row.get('warning_codes', [])) | set(setup.get('warning_codes', [])))
        row.update(core_selection_status=record['core_selection_status'], core_revision=generation['revision'],
            selection_is_final=key in final_ids, protected_baseline=bool(generated.get('protected_rigid')))
        if record.get('diagnostic_only') and record['status'] != 'generated':
            row['instance_status'] = 'definition_pending'
            row['warning_codes'] = sorted(set(row['warning_codes']) | {'definition_pending_diagnostic_capture'})
        if not row['native_capture_complete'] and record['status'] == 'generated' and not status.get('error'):
            row['instance_status'] = 'capture_pending'
        if generated.get('protected_rigid'):
            row['instance_status'] = 'protected_baseline_reference'
            row['capture']['state'] = 'PRESERVED_BASELINE'
        chosen = selection_by_source[generated['source_identity']]
        candidates.append({key: row.get(key) for key in ('candidate_id', 'source', 'category', 'subcategory',
            'scene', 'room', 'task_file', 'scene_file', 'instance_status', 'capture', 'native_capture_complete',
            'protected_baseline', 'warning_codes', 'failure')})
        candidates[-1].update(selection_reason=chosen.get('selection_reason'),
            space_screen=chosen.get('space_screen'), conversion_status=generated['status'],
            mechanism_contract=generated.get('mechanism_contract'),
            protected_historical_images=len(record.get('diagnostic_images', [])) if generated.get('protected_rigid') else 0)
        if row['native_capture_complete']:
            completed.append(row)
        else:
            pending.append(row)
        groups[(row['category'], row['subcategory'])] += 1
    snapshot = {'schema': 'core_review_catalogue_v1', 'scenes': completed, 'diagnostics': pending,
        'categories': [{'big_task': big, 'small_task': small, 'source_count': count, 'counts': {}}
            for (big, small), count in sorted(groups.items())],
        'indexing': {'status': 'ready', 'selection_status': 'frozen' if final.get('frozen') else 'provisional',
            'purpose': 'Provisional Core candidates only; final selection is owned by the user.'},
        'capture_queue': {'root': str(queue), 'candidate_count': len(completed) + len(pending),
            'completed': len(completed), 'pending_or_diagnostic': len(pending)},
        'historical_approval_policy': 'Prior layout feedback retained; revised tasks require new review.'}
    write_json(root / 'catalogue/snapshot.json', snapshot)
    write_json(root / 'core_candidates_for_review.json', {
        'schema': 'provisional_core_review_manifest_v1', 'frozen': False, 'final_selection_owner': 'user',
        'selection_does_not_use_model_performance': True, 'capture_scope': 'Core candidates only',
        'revision': generation['revision'], 'count': len(candidates), 'tasks': candidates,
        'counts': snapshot['capture_queue'],
        'qualification_note': 'Captures and layout feedback do not establish operation success or calibrated Safety.'})
    print(f'Core Review: {len(completed)} captured, {len(pending)} pending or diagnostic', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        publish(args.root.resolve())
        if not args.watch:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
