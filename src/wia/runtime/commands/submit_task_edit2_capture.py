#!/usr/bin/env python3
"""Validate and submit explicit TaskEdit2 derivatives through the existing bridge.

Run CPU-only validation (no queue writes, freezing or enqueue):
  python -B src/wia/runtime/commands/submit_task_edit2_capture.py --manifest MANIFEST --queue-root QUEUE --dry-run
For authorized submission omit --dry-run; one common freeze_working_code snapshot
is used. --frozen-code-record accepts JSON of that function's complete return
value, including implementation_hashes. No GPU workers or services are started.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

import yaml

import wia.runtime.esi_category_capture as bridge
from wia.runtime.scene_capture_queue import digest, read_json, sha256, write_json
from wia.tasks.loader import UniqueLoader, task_definition
from wia.tasks.schema import SceneDefinition

REVISION = 'task_edit2_capture_submission_20261018_v1'
CAMPAIGN_FIELDS = {'capture_initial_agent_bundle': True, 'task_edit2_capture_campaign': True}


def _identity(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or len(path.parts) != 5 or path.suffix != '.json':
        raise ValueError('Source identity must be an exact safe five-part ESI JSON identity.')
    return value


def _file(path, expected):
    path = Path(path).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError('Exact absolute input file is missing: '+str(path))
    if not expected or sha256(path) != expected:
        raise ValueError('Current input bytes do not match the manifest SHA256: '+str(path))
    return str(path.resolve())


def _modules(value):
    modules = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('module', 'converter_module', 'evaluator_module') and isinstance(item, str) and item.startswith('wia.authoring.converters.'):
                modules.add(item)
            modules.update(_modules(item))
    elif isinstance(value, list):
        for item in value:
            modules.update(_modules(item))
    return modules


def capture_records(manifest):
    """Return one bridge record per explicit derivative, retaining validation errors.

    This function only reads declared files. It never freezes, publishes, creates
    directories, selects final variants or transfers old capture qualifications.
    Rows with status=validation_failed must not be sent to publish_record.
    """
    variants = [row for row in manifest['variants'] if row.get('role') == 'explicit_derivative']
    counts = Counter(row.get('variant_key') for row in variants)
    records = []
    for variant in variants:
        key = variant.get('variant_key')
        alias = 'category_'+key if isinstance(key, str) and re.fullmatch('[0-9a-f]+', key) else None
        record = {'candidate_id': alias, 'actual_candidate_id': variant.get('candidate_id'),
            'task_edit2_variant_key': key, 'role': 'explicit_derivative', **CAMPAIGN_FIELDS,
            'status': 'validation_failed', 'failure_stage': 'capture_record_validation',
            'failure_reason': None, 'selected_final_variant': False, 'native_capture_complete': False,
            'layout_approval_available': False, 'autonomous_agent_performance': False,
            'capture_source_members': [], 'sources': deepcopy(variant.get('sources', [])),
            'membership_class': variant.get('membership_class'),
            'original_core_members': deepcopy(variant.get('original_core_members', [])),
            'additional_source_members': deepcopy(variant.get('additional_source_members', [])),
            'manifest_records': deepcopy(variant.get('manifest_records', [])),
            'baseline_membership': deepcopy(manifest.get('baseline_membership')),
            'warning_codes': deepcopy(variant.get('recorded_warnings', [])),
            'pending_gates': sorted(set(variant.get('pending_gates', [])+variant.get('pending_review_gates', []))),
            'limitations': ['Provisional explicit TaskEdit2 derivative; final selection remains with the user.',
                'Capture is initial-layout evidence, not task success, physical qualification or layout approval.'],
            'reuse_level': variant.get('reuse_level') or 'explicit_derivative',
            'submission_adapter_version': REVISION}
        records.append(record)
        try:
            if alias is None or counts[key] != 1:
                raise ValueError('Capture aliases require unique hexadecimal variant keys.')
            if variant.get('schema_status') != 'schema_valid' or variant.get('source_binding_status') != 'verified':
                raise ValueError('Manifest must declare schema-valid and verified source binding.')
            for kind in ('task', 'scene'):
                record[kind+'_file'] = _file(variant[kind]['path'], variant[kind]['sha256'])
                record[kind+'_sha256'] = variant[kind]['sha256']
            task_raw = yaml.load(Path(record['task_file']).read_text(), Loader=UniqueLoader)
            scene_raw = yaml.load(Path(record['scene_file']).read_text(), Loader=UniqueLoader)
            task, scene = task_definition(task_raw), SceneDefinition.model_validate(scene_raw)
            if scene.status != 'pending':
                raise ValueError('New derivative capture requires a pending scene, never final approval.')
            if task.defaults.seed != scene.seed:
                raise ValueError('Task and scene initialization seeds must match exactly.')
            common = set(task.scene.variants) & set(scene.scene.variants)
            selected = 'adapted' if 'adapted' in common else 'original' if 'original' in common else None
            if selected is None:
                raise ValueError('Task and scene have no common adapted/original capture variant.')
            primary = _identity(task.source.json_path)
            if scene.source.json_path != primary:
                raise ValueError('Task and scene primary source identities differ.')
            seen, members = set(), []
            for source in variant['sources']:
                identity = _identity(source['source_identity'])
                if identity in seen or source.get('provenance_status') != 'verified_current_source_bytes':
                    raise ValueError('Every source member must be unique and current-byte verified.')
                seen.add(identity)
                members.append({'source_identity': identity,
                    'source_file': _file(source['source_file'], source['source_sha256']),
                    'source_sha256': source['source_sha256']})
            primary_member = next((row for row in members if row['source_identity'] == primary), None)
            if primary_member is None:
                raise ValueError('The primary task/scene source is absent from explicit membership.')
            raw = json.loads(Path(primary_member['source_file']).read_text())
            if not isinstance(raw.get('scene'), str) or not isinstance(raw.get('room'), str):
                raise ValueError('Primary source must declare its actual scene and room.')
            evaluation = task_raw.get('private', {}).get('evaluation', {})
            module = variant.get('converter_module') or evaluation.get('module')
            if not isinstance(module, str) or not module.startswith('wia.authoring.converters.'):
                raise ValueError('The actual converter/evaluator module is unavailable.')
            parameters = evaluation.get('parameters', {})
            version = variant.get('converter_version') or parameters.get('task_revision') or parameters.get('definition_revision')
            if not isinstance(version, str) or not version:
                version = REVISION+':definition_version_'+str(task.version)
            record.update(status='generated', failure_stage=None, failure_reason=None,
                task_id=task.id, scene_id=scene.id, seed=scene.seed, variant=selected,
                source_identity=primary, original_source=primary_member['source_file'],
                source_sha256=primary_member['source_sha256'], big_task=Path(primary).parts[0],
                small_task=Path(primary).parts[1], scene_model=raw['scene'], room=raw['room'],
                question=raw.get('question', ''), capture_source_members=members,
                converter_module=module, converter_version=version,
                composite_member_modules=sorted(_modules(evaluation)-{module}),
                diagnostic_dir=str(Path(record['task_file']).parent),
                source_core_metadata=[{'source_identity': row['source_identity'],
                    'original_core_member': row['source_identity'] in record['original_core_members']}
                    for row in members])
            record['revision'] = digest({'adapter': REVISION, 'alias': alias,
                'actual_candidate_id': record['actual_candidate_id'], 'task_sha256': record['task_sha256'],
                'scene_sha256': record['scene_sha256'], 'task_file': record['task_file'], 'scene_file': record['scene_file'],
                'sources': sorted(members, key=lambda row: row['source_identity']), 'variant': selected,
                'seed': scene.seed, 'private_binding_digest': digest(task_raw.get('private', {}).get('bindings', {}))})
        except Exception as error:
            record.update(status='validation_failed', failure_stage='capture_record_validation',
                          failure_reason=type(error).__name__+': '+str(error))
    return records


def _frozen_record(path):
    frozen = read_json(path, {})
    required = ('code_root', 'code_commit', 'esi_commit', 'implementation_hashes', 'code_archive', 'code_archive_sha256')
    if any(key not in frozen for key in required) or not isinstance(frozen['implementation_hashes'], dict):
        raise ValueError('Frozen code record must contain the complete freeze_working_code return value.')
    root = Path(frozen['code_root']).resolve()
    marker = read_json(root/'.capture_revision.json', {})
    if marker.get('commit') != frozen['code_commit'] or marker.get('esi_commit') != frozen['esi_commit']:
        raise ValueError('Frozen code marker identity mismatch.')
    _file(frozen['code_archive'], frozen['code_archive_sha256'])
    for relative, expected in frozen['implementation_hashes'].items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts or '\\' in relative:
            raise ValueError('Frozen implementation path escaped its code root.')
        path = (root/relative).resolve()
        if not path.is_relative_to(root):
            # freeze_working_code deliberately links the exact committed ESI
            # archive beside its working tree. No other external link is valid.
            commit = frozen['esi_commit']
            valid_commit = isinstance(commit, str) and re.fullmatch('[A-Za-z0-9_-]+', commit)
            esi = root.parent/('esi_'+commit) if valid_commit else None
            link = root/'ESI'
            sealed = bool(esi and relative_path.parts[0] == 'ESI' and link.is_symlink()
                and not esi.is_symlink() and link.resolve() == esi
                and path.is_relative_to(esi)
                and read_json(esi/'.capture_revision.json', {}).get('commit') == commit)
            if not sealed:
                raise ValueError('Frozen implementation path escaped its code root.')
        _file(str(path), expected)
    return frozen


def _existing_record(path, record, frozen=None):
    existing = read_json(path)
    if existing is None:
        return None
    if existing.get('record_revision') != record['revision']:
        raise ValueError('A category revision identity cannot be overwritten.')
    fields = ('candidate_id', 'actual_candidate_id', 'task_edit2_variant_key', 'task_file', 'task_sha256',
        'scene_file', 'scene_sha256', 'source_identity', 'original_source', 'source_sha256',
        'seed', 'variant', 'capture_source_members', *CAMPAIGN_FIELDS)
    if any(existing.get(key) != record.get(key) for key in fields):
        raise ValueError('Existing category record does not match the exact TaskEdit2 binding.')
    if frozen and existing.get('code_commit') and existing['code_commit'] != frozen['code_commit']:
        raise ValueError('Existing category record belongs to a different frozen code snapshot.')
    return existing


def submit(manifest, queue_root, *, dry_run=False, frozen_code_record=None, timeout_s=1800,
           max_attempts=1, manifest_path=None):
    """Return a per-alias report; actual mode writes queue/submission_report.json."""
    if timeout_s <= 0 or max_attempts < 1:
        raise ValueError('Capture timeout and maximum attempts must be positive.')
    queue = Path(queue_root).expanduser().resolve()
    records = capture_records(manifest)
    report = {'schema': REVISION, 'dry_run': bool(dry_run), 'manifest_file': str(Path(manifest_path).resolve()) if manifest_path else None,
        'manifest_sha256': sha256(manifest_path) if manifest_path else digest(manifest),
        'queue_root': str(queue), 'frozen_code': None, 'records': [], 'final_selection_made': False,
        'workers_started': False, 'timeout_s': timeout_s, 'max_attempts': max_attempts}
    frozen, freeze_error = None, None
    eligible = [record for record in records if record['status'] == 'generated']
    if not dry_run:
        try:
            if frozen_code_record:
                frozen = _frozen_record(frozen_code_record)
            elif any(not (queue/'category_records'/(record['candidate_id']+'.json')).is_file() for record in eligible):
                frozen = bridge.freeze_working_code(queue)
        except Exception as error:
            freeze_error = type(error).__name__+': '+str(error)
        if frozen:
            report['frozen_code'] = {key: deepcopy(frozen[key]) for key in ('code_commit', 'esi_commit', 'code_archive', 'code_archive_sha256')}
    for record in records:
        entry = {key: deepcopy(record.get(key)) for key in ('candidate_id', 'actual_candidate_id', 'task_edit2_variant_key',
            'revision', 'task_file', 'task_sha256', 'scene_file', 'scene_sha256', 'source_identity',
            'original_source', 'source_sha256', 'capture_source_members', 'seed', 'variant')}
        entry.update(job_id=None, status='error' if record['status'] != 'generated' else 'validated' if dry_run else 'pending',
                     error=record['failure_reason'])
        if record['status'] == 'generated':
            try:
                previous = _existing_record(queue/'category_records'/(record['candidate_id']+'.json'), record, frozen)
                if dry_run:
                    entry['existing_job_id'] = previous.get('job_id') if previous else None
                elif previous is not None:
                    entry.update(status='existing', job_id=previous.get('job_id'), capture_input=previous.get('capture_input'),
                        error=previous.get('capture_submission_error'))
                    if entry['error'] or not entry['job_id']:
                        entry['status'] = 'error'
                        entry['error'] = entry['error'] or 'Existing publication has no capture job; immutable record was preserved.'
                else:
                    if freeze_error or frozen is None:
                        raise ValueError('Common frozen code snapshot unavailable: '+str(freeze_error))
                    record['diagnostic_dir'] = str(queue/'category_diagnostics'/record['candidate_id'])
                    published = bridge.publish_record(record, queue, frozen=frozen, timeout_s=timeout_s,
                        max_attempts=max_attempts, report_file=queue/'submission_report.json', batch_file=manifest_path)
                    error = published.get('capture_submission_error')
                    entry.update(status='error' if error or not published.get('job_id') else 'submitted',
                        job_id=published.get('job_id'), capture_input=published.get('capture_input'),
                        error=error or (None if published.get('job_id') else 'Bridge returned no capture job.'))
            except Exception as error:
                entry.update(status='error', error=type(error).__name__+': '+str(error))
        report['records'].append(entry)
    outcomes = Counter(row['status'] for row in report['records'])
    report['counts'] = {'explicit_derivatives': len(records), 'validated': len(eligible),
        'submitted': outcomes['submitted'], 'existing': outcomes['existing'], 'error': outcomes['error']}
    if not dry_run:
        write_json(queue/'submission_report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--queue-root', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--timeout-s', type=float, default=1800.)
    parser.add_argument('--max-attempts', type=int, default=1)
    parser.add_argument('--frozen-code-record', type=Path)
    args = parser.parse_args(argv)
    report = submit(read_json(args.manifest), args.queue_root, dry_run=args.dry_run,
        frozen_code_record=args.frozen_code_record, timeout_s=args.timeout_s, max_attempts=args.max_attempts,
        manifest_path=args.manifest)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if report['counts'].get('error', 0) else 0


if __name__ == '__main__':
    raise SystemExit(main())
