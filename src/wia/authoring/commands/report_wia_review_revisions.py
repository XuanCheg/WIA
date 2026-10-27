"""Report only main-selected exact versions and their current indexed receipts.

Run: python -B scripts/report_wia_review_revisions.py --manifest PATH
  --visual-reviews PATH --output-dir PATH [--project-root PATH]
Relative paths use the project root. This CPU-only report copies main-authored
visual decisions and never assesses designs, changes Review or records feedback.


Canonical invocation: python -m wia.authoring.commands.report_wia_review_revisions --help
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from wia.resources import ROOT
INDEX = 'outputs/wia_v1_20261023/capture_campaign/review_index.json'
REGISTRY = 'explore_to_achieve/wia_v1/registry.json'
COMPLETED_STATES = {'REVIEW_PENDING', 'CACHED'}
HOLD_DECISIONS = {'hold', 'held', 'hold_initial_design', 'on_hold'}
FALSE_FLAGS = {'full_task_success_verified': False, 'physical_success_verified': False,
               'scientific_controls_verified': False, 'legal_robot_reference_verified': False,
               'full_reference_execution_verified': False}
CSV_FIELDS = ('document_alias', 'variant_id', 'state', 'images_complete', 'initial_setup_accepted',
              'main_visual_review', 'native_optical_selection_accepted', 'receipt_file', 'task_file', 'scene_file')


def path_at(value, root):
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _definition_identity(entry, indexed, root, errors):
    paths, hashes = {}, {}
    for kind in ('task', 'scene'):
        field = kind + '_file'
        paths[field] = path_at(entry[field], root)
        if not paths[field].is_file():
            errors.append('Selected ' + kind + ' file is missing.')
            hashes[kind + '_sha256'] = None
            continue
        actual = file_hash(paths[field])
        hashes[kind + '_sha256'] = actual
        expected = entry.get(kind + '_sha256', actual)
        if expected != actual:
            errors.append('Selected manifest ' + kind + ' hash differs from the current file.')
        if indexed:
            if indexed.get(kind + '_sha256') != expected:
                errors.append('Indexed ' + kind + ' hash differs from the selected version.')
            indexed_path = indexed.get(field)
            if indexed_path is None or path_at(indexed_path, root) != paths[field]:
                errors.append('Indexed ' + kind + ' path differs from the selected version.')
    return paths, hashes


def _receipt_errors(receipt, status, entry, task_hash, job_id):
    errors = []
    if status.get('job_id') != job_id or receipt.get('job_id') != job_id:
        errors.append('Status or receipt job identity differs from the indexed job.')
    if 'capture_job_id' in receipt and receipt['capture_job_id'] != job_id:
        errors.append('Receipt capture_job_id differs from the indexed job.')
    identity = (receipt.get('initial_agent_bundle') or {}).get('capture_input_identity') or {}
    if identity.get('job_id') != job_id:
        errors.append('Frozen initial bundle job identity differs from the indexed job.')
    if task_hash is None or identity.get('task_sha256') != task_hash:
        errors.append('Frozen initial bundle task hash differs from the selected exact task.')
    for field in ('variant', 'seed'):
        if field in entry:
            if identity.get(field) != entry[field] or receipt.get(field) != entry[field]:
                errors.append('Receipt or frozen initial bundle ' + field + ' differs from the selected version.')
    return errors


def _failed_supports(setup, field):
    values = setup.get(field) or {}
    if not isinstance(values, dict):
        return deepcopy(values)
    return {name: deepcopy(value) for name, value in values.items()
            if not isinstance(value, dict) or value.get('accepted') is not True}


def build_report(manifest_path, visual_reviews_path, project_root=ROOT):
    """Read selected versions and authoritative status pointers, without writing."""
    root = Path(project_root).expanduser().resolve()
    manifest_path, visual_reviews_path = (path_at(path, root) for path in (manifest_path, visual_reviews_path))
    index_path, registry_path = path_at(INDEX, root), path_at(REGISTRY, root)
    manifest, reviews, index = read(manifest_path), read(visual_reviews_path), read(index_path)
    if not isinstance(manifest, dict) or manifest.get('schema') != 'wia_variant_manifest_v1' or not isinstance(manifest.get('entries'), list):
        raise ValueError('An explicit wia_variant_manifest_v1 entries manifest is required.')
    if not isinstance(reviews, dict):
        raise ValueError('Visual reviews must be a main-authored dictionary keyed by variant_id.')
    if not isinstance(index, dict) or not isinstance(index.get('items'), list):
        raise ValueError('The current capture campaign review index is required.')
    registry = read(registry_path, {'rows': []})
    aliases = {row['document_alias']: row['wia_id'] for row in registry['rows']
               if row.get('document_alias') and row.get('wia_id')}
    indexed = {}
    for item in index['items']:
        key = (item['row_id'], item['variant_id'])
        if key in indexed:
            raise ValueError('Duplicate row_id and variant_id in the current review index.')
        indexed[key] = item
        if item.get('document_alias'):
            alias = item['document_alias']
            if alias in aliases and aliases[alias] != item['row_id']:
                raise ValueError('Review index and registry disagree on proposal alias: ' + alias)
            aliases[alias] = item['row_id']
    protected = {manifest_path, visual_reviews_path, index_path, registry_path}
    rows, seen = [], set()
    for entry in manifest['entries']:
        alias, variant_id = entry['document_alias'], entry['variant_id']
        row_id = entry.get('row_id', aliases.get(alias))
        key = (row_id or alias, variant_id)
        if key in seen:
            raise ValueError('Selected manifest contains a duplicate proposal and variant.')
        seen.add(key)
        item = indexed.get((row_id, variant_id), {})
        errors = []
        definition_paths, hashes = _definition_identity(entry, item, root, errors)
        protected.update(definition_paths.values())
        job_id = item.get('job_id')
        status_path = None
        if job_id:
            status_path = path_at(item.get('queue_status_file') or
                ('outputs/wia_v1_20261023/capture_campaign/queue/jobs/' + job_id + '/status.json'), root)
            protected.add(status_path)
        status = read(status_path, {}) if status_path else {}
        state = status.get('state', 'STATUS_MISSING' if item else 'NOT_SUBMITTED')
        selected_receipt_path = None
        receipt = {}
        # The index's receipt_file is historical metadata. Only status.receipt
        # for the current indexed completed job may select completed evidence.
        if state in COMPLETED_STATES:
            if not status.get('receipt'):
                errors.append('Completed indexed status has no receipt pointer.')
            else:
                selected_receipt_path = path_at(status['receipt'], root)
                protected.add(selected_receipt_path)
                candidate = read(selected_receipt_path)
                if not isinstance(candidate, dict):
                    errors.append('Current status receipt is missing or is not an object.')
                else:
                    errors.extend(_receipt_errors(candidate, status, entry, hashes['task_sha256'], job_id))
                    if not errors:
                        receipt = candidate
        bundle = receipt.get('initial_agent_bundle') or {}
        setup = receipt.get('interaction_setup') or {}
        geometry = ((bundle.get('review_diagnostics') or {}).get('geometry') or {})
        selection = geometry.get('optical_selection') or {}
        review = reviews.get(variant_id, {})
        if not isinstance(review, dict):
            raise ValueError('Each visual review must be an object keyed by variant_id.')
        review_mismatch = None
        if review:
            if not job_id or review.get('job_id') != job_id:
                review_mismatch = 'Visual review job_id differs from the current indexed job.'
            elif not receipt:
                review_mismatch = 'Current receipt identity is not verified.'
        stale_acceptance = review.get('decision') == 'accepted_initial_design' and review_mismatch is not None
        preview_only = setup.get('initialization_preview_only')
        row = {'row_id': row_id, 'document_alias': alias, 'variant_id': variant_id,
            'label': entry.get('label'), **{key: str(path) for key, path in definition_paths.items()}, **hashes,
            'selected_task_sha256': entry.get('task_sha256'), 'selected_scene_sha256': entry.get('scene_sha256'),
            'job_id': job_id, 'state': state, 'queue_status_file': str(status_path) if status_path else None,
            'receipt_file': str(selected_receipt_path) if receipt else None,
            'status_receipt_file': str(selected_receipt_path) if selected_receipt_path else None,
            'evidence_identity_verified': bool(receipt), 'evidence_identity_errors': errors,
            'images_complete': receipt.get('complete') is True,
            'initial_setup_accepted': setup.get('accepted'),
            'initialization_preview_only': preview_only,
            'interaction_setup_kind': 'initialization_preview_only' if preview_only is True else ('interaction_setup' if setup else None),
            'failed_supported_items': _failed_supports(setup, 'supported_items'),
            'failed_inventory_supports': _failed_supports(setup, 'inventory_supports'),
            'robot_initialization': deepcopy(setup.get('robot', setup.get('robot_initialization'))),
            'native_optical_selection_accepted': selection.get('accepted'),
            'native_optical_selection': deepcopy(selection),
            'main_visual_review': None if stale_acceptance else deepcopy(review.get('decision')),
            'main_visual_notes': None if stale_acceptance else deepcopy(review.get('notes')),
            'main_visual_review_job_id': None if stale_acceptance else deepcopy(review.get('job_id')),
            'main_visual_review_mismatch': review_mismatch,
            'recorded_main_visual_review': {key: deepcopy(review.get(key)) for key in ('decision', 'job_id', 'notes')},
            'explicit_hold': review.get('hold') is True or review.get('decision') in HOLD_DECISIONS,
            'human_accepted': None, 'robot_actions': receipt.get('robot_actions'), 'model_calls': receipt.get('model_calls'),
            'images': deepcopy(receipt.get('images', {})),
            'duration_s': receipt.get('duration_s', status.get('duration_s')),
            'timings': deepcopy(receipt.get('timings', status.get('timings'))),
            'failure': deepcopy(status.get('error')), 'failure_stage': status.get('failure_stage'),
            'retryable': status.get('retryable'), 'attempts': status.get('attempts'),
            'indexed_failure_retry_records': {key: deepcopy(value) for key, value in item.items()
                if any(word in key.lower() for word in ('error', 'retry', 'failed', 'superseded'))},
            **FALSE_FLAGS}
        rows.append(row)
    summary = {'selected_versions': len(rows),
        'distinct_proposals': len({row['row_id'] or row['document_alias'] for row in rows}),
        'complete_images': sum(row['images_complete'] for row in rows),
        'initial_setup_accepted': sum(row['initial_setup_accepted'] is True for row in rows),
        'initial_setup_rejected': sum(row['initial_setup_accepted'] is False for row in rows),
        'initialization_preview_only': sum(row['initialization_preview_only'] is True for row in rows),
        'main_visual_accepted': sum(row['main_visual_review'] == 'accepted_initial_design' for row in rows),
        'explicit_holds': sum(row['explicit_hold'] for row in rows),
        'states': dict(Counter(row['state'] for row in rows)),
        'main_visual_decisions': dict(Counter(row['main_visual_review'] for row in rows if isinstance(row['main_visual_review'], str)))}
    result = {'schema': 'wia_selected_revision_review_v1', 'updated_at': datetime.now(timezone.utc).isoformat(),
        'manifest_file': str(manifest_path), 'manifest_sha256': file_hash(manifest_path),
        'visual_reviews_file': str(visual_reviews_path), 'visual_reviews_sha256': file_hash(visual_reviews_path),
        'review_index_file': str(index_path), 'review_index_sha256': file_hash(index_path),
        'receipt_selection_policy': 'Current indexed job status.receipt only for REVIEW_PENDING or CACHED; identity must match the selected exact task.',
        'evidence_scope': 'Initialization, settling and images only. Main visual decisions are copied verbatim; human acceptance, complete task success and scientific qualification are not established.',
        'human_accepted': None, **FALSE_FLAGS, 'summary': summary, 'entries': rows}
    return result, protected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--visual-reviews', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result, protected = build_report(args.manifest, args.visual_reviews, args.project_root)
    output_dir = path_at(args.output_dir, args.project_root.expanduser().resolve())
    output_paths = [output_dir / ('scene_review_status.' + extension) for extension in ('json', 'csv')]
    if any(path.resolve() in protected or path.is_symlink() for path in output_paths):
        raise ValueError('Report outputs must not replace or alias input evidence files.')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths[0].write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with output_paths[1].open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in CSV_FIELDS} for row in result['entries'])
    print(json.dumps(result['summary']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
