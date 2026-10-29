#!/usr/bin/env python3
"""Stage a provisional TaskEdit2 overlay for the existing LiveSceneReview app.

Run: python -B src/wia/runtime/commands/publish_task_edit2_review.py --manifest outputs/task_edit2_20261018/review/current_manifest.json --baseline-catalogue /home/zhouxinyang/esi_category_conversion/core_revision_20261018/catalogue/snapshot.json --output-dir outputs/task_edit2_20261018/review/ui
Optional --watch refreshes this staged output only. No queues, historical rows,
source definitions, review decisions, running services or final selections change.
--group-core groups revisions under the original Core sources for browsing;
it is mutually exclusive with --new-only and never selects final Core versions.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

import yaml

from wia.runtime.scene_capture_queue import CaptureQueue, complete_images, read_json, write_json
from wia.tasks.loader import UniqueLoader
from wia.runtime.commands.build_task_edit2_review_manifest import exact_native_match, initialization_checkpoint_proof, sha256
from wia.runtime.commands.update_esi_review_catalogue import task_summary

PUBLIC_FIELDS = ('objective', 'success_criteria', 'environment', 'constraints')
STATUS_FIELDS = ('schema_status', 'source_binding_status', 'definition_review_status',
    'native_initialization_status', 'layout_status', 'execution_status', 'scoring_status')
MEMBERSHIP_EXPLANATIONS = {
    'core_derivative': 'TaskEdit2 provisional Core derivative: every explicitly mapped source belongs to the original Core membership.',
    'mixed_core_and_additional_members': 'TaskEdit2 provisional mixed parent: includes original Core sources and additional sources; additions do not enter the original Core membership.',
    'additional_prototype': 'TaskEdit2 provisional additional prototype: this card is outside the original Core membership.',
}
STATUS_LABELS = {'schema_status': 'Definition schema', 'source_binding_status': 'Source binding',
    'definition_review_status': 'Definition review', 'native_initialization_status': 'Native initialization',
    'layout_status': 'Layout', 'execution_status': 'Execution', 'scoring_status': 'Scoring'}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _category(source):
    path = Path(source)
    if path.is_absolute() or '..' in path.parts or len(path.parts) != 5 or path.suffix != '.json':
        raise ValueError('Review source labels require an exact five-part source identity.')
    return path.parts[:2]


def _current_file(definition):
    path = Path(definition.get('path') or '')
    actual = sha256(path) if path.is_file() else None
    return {'path': str(path.resolve()) if definition.get('path') else None,
        'manifest_sha256': definition.get('sha256'), 'current_sha256': actual,
        'matches': bool(actual and actual == definition.get('sha256'))}


def _public(value):
    return {key: deepcopy(value[key]) for key in PUBLIC_FIELDS if key in value}


def _sources(variant):
    result = []
    for member in variant.get('sources', []):
        name = member['source_identity']
        _category(name)
        current = _current_file({'path': member.get('source_file'), 'sha256': member.get('source_sha256')})
        raw = read_json(current['path'], {}) if current['matches'] else {}
        result.append({'source_identity': name, 'source_sha256': member.get('source_sha256'),
            'file_check': current, 'question': raw.get('question') if isinstance(raw.get('question'), str) else None,
            'scene': raw.get('scene'), 'room': raw.get('room')})
    if not result or len({row['source_identity'] for row in result}) != len(result):
        raise ValueError('Each derivative needs an explicit unique source mapping.')
    return result


def _public_goals(task, sources, public):
    members = task.get('private', {}).get('evaluation', {}).get('parameters', {}).get('members', [])
    index = {member.get('source_identity'): member for member in members if isinstance(member, dict)}
    goals = []
    for number, source in enumerate(sources, 1):
        identity = source['source_identity']
        big, small = _category(identity)
        member = index.get(identity, {})
        member_public = _public(member.get('public', {}))
        goal = {'id': member.get('public_id') or f'goal_{number}', 'source': identity,
            'big_task': big, 'small_task': small,
            'objective': member_public.get('objective') or (public.get('objective') if len(sources) == 1
                else 'Source question: ' + source['question'] if source.get('question') else
                'Source context for the shared public task: ' + identity),
            'success_criteria': member_public.get('success_criteria') or public.get('success_criteria', ''),
            'constraints': member_public.get('constraints', public.get('constraints', []))}
        if source.get('question'):
            goal['original_question'] = source['question']
        goals.append(goal)
    return goals


def _image(path, view, label, role, *, evidence_id=None):
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        return None
    return {'path': str(path.resolve()), 'view': view, 'label': label,
        'kind': 'partial', 'capture_job_id': None, 'evidence_role': role,
        'native_evidence_id': evidence_id, 'queue_capture_certificate': False}


def _evidence(variant, evidence_index, file_checks, sources):
    accepted, rejected, images = [], [], []
    current = all(row['matches'] for row in file_checks.values()) and all(s['file_check']['matches'] for s in sources)
    for evidence_id in variant.get('native_evidence_ids', []):
        evidence = evidence_index.get(evidence_id)
        if not evidence:
            rejected.append({'evidence_id': evidence_id, 'reason': 'manifest_native_evidence_missing'})
            continue
        path = Path(evidence.get('private_result_file') or '')
        if not current or not path.is_file() or sha256(path) != evidence.get('result_sha256'):
            rejected.append({'evidence_id': evidence_id, 'reason': 'current_definition_source_or_result_hash_mismatch'})
            continue
        result = read_json(path, {})
        checkpoint = (result.get('status') == 'stopped_at_requested_initialization_checkpoint'
                      and result.get('diagnostic_checkpoint_complete') is True)
        proof = initialization_checkpoint_proof(path, result) if checkpoint else None
        matched, checks = exact_native_match(variant, result, checkpoint_proof=proof)
        if not matched:
            rejected.append({'evidence_id': evidence_id, 'reason': 'native_result_not_exactly_bound', 'checks': checks})
            continue
        accepted.append({'evidence_id': evidence_id, 'result_file': str(path.resolve()),
            'result_sha256': evidence['result_sha256'],
            'diagnostic_complete': evidence.get('diagnostic_complete'),
            **{field: deepcopy(evidence.get(field)) for field in
               ('native_initialization_status', 'layout_status', 'execution_status', 'scoring_status',
                'public_control_results', 'public_control_success_count')},
            'qualification_status': (evidence.get('qualification') or {}).get('status'),
            'qualification_reason': (evidence.get('qualification') or {}).get('reason'),
            'interaction_setup_results': [{'phase': item.get('phase'),
                **{key: deepcopy((item.get('receipt') or {}).get(key)) for key in
                   ('accepted', 'reason', 'warning_codes')}} for item in evidence.get('interaction_setup_evidence', [])],
            'autonomous_agent_performance': False,
            'interpretation': 'Exact-bound manual native diagnostic; no queue capture certificate or layout approval.'})
        captures = evidence.get('captures') or {}
        named = [('initial_overview_rgb', captures.get('initial_global'))]
        named += list((captures.get('initial_mounted') or {}).items())
        named += [(f'human_review_{index}', path) for index, path in enumerate(captures.get('other_review_images', []))]
        for view, filename in named:
            if filename:
                item = _image(filename, view, f'Exact-bound native diagnostic · {view} · unqualified',
                              'exact_bound_native_diagnostic', evidence_id=evidence_id)
                if item:
                    images.append(item)
    return accepted, rejected, images


def queue_capture_evidence(variant, queue_root):
    """Read one exact new-queue binding without creating or changing queue files."""
    alias = 'category_' + variant['variant_key']
    queue_root = Path(queue_root).expanduser().resolve()
    record_path = queue_root / 'category_records' / (alias + '.json')
    value = {'state':'NOT_SUBMITTED','raw_state':None,'job_id':None,'complete':False,
        'binding_available':False,'reason':'new_queue_record_not_available','images':{},
        'record_file':str(record_path),'record_sha256':None,'manifest_file':None,'manifest_sha256':None,
        'receipt_file':None,'receipt_sha256':None,'record':{},'status':{},'receipt':{},'checks':{}}
    if not record_path.is_file():
        return value
    try:
        record = read_json(record_path,{})
        value.update(record=record,record_sha256=sha256(record_path),job_id=record.get('job_id'))
        checks = value['checks']
        checks['candidate_alias'] = record.get('candidate_id') == alias
        checks['variant_key'] = record.get('task_edit2_variant_key') == variant['variant_key']
        checks['actual_candidate_id'] = record.get('actual_candidate_id') == variant.get('candidate_id')
        checks['new_capture_campaign'] = record.get('task_edit2_capture_campaign') is True
        for kind in ('task','scene'):
            expected = variant[kind]
            current = _current_file(expected)
            checks[kind+'_path_and_sha256'] = (current['matches'] and record.get(kind+'_file') == current['path']
                and record.get(kind+'_sha256') == current['manifest_sha256'])
        sources = _sources(variant)
        primary = (variant['task'].get('source') or {}).get('json')
        source = next((row for row in sources if row['source_identity'] == primary),None)
        checks['primary_source_path_and_sha256'] = bool(source and source['file_check']['matches']
            and record.get('source_identity') == primary and record.get('original_source') == source['file_check']['path']
            and record.get('source_sha256') == source['source_sha256'])
        expected_sources = {(row['source_identity'],row['file_check']['path'],row['source_sha256']) for row in sources}
        recorded_sources = {(row.get('source_identity'),row.get('source_file'),row.get('source_sha256'))
                            for row in record.get('sources',[]) if isinstance(row,dict)}
        checks['all_source_members'] = (recorded_sources == expected_sources
            and all(row['file_check']['matches'] for row in sources))
        if not all(checks.values()):
            raise ValueError('New queue record does not bind the exact current variant definitions and source membership.')
        if not record.get('job_id'):
            value.update(state='FAILED' if record.get('capture_submission_error') else 'SUBMISSION_PENDING',
                binding_available=True,reason=record.get('capture_submission_error') or 'capture_job_not_yet_published')
            return value
        # CaptureQueue.__init__ creates directories. These two methods are read-only.
        queue = CaptureQueue.__new__(CaptureQueue)
        queue.root = queue_root
        job = queue.job_dir(record['job_id'])
        status = queue.status(record['job_id']) or {}
        value.update(status=deepcopy(status),raw_state=status.get('state'),state=status.get('state','QUEUED'))
        checks['status_job_id'] = status.get('job_id',record['job_id']) == record['job_id']
        frozen = queue.manifest(record['job_id'])
        manifest_path = job / 'input/manifest.json'
        checks['frozen_manifest_path'] = (Path(record.get('capture_input') or '').resolve() == manifest_path.resolve())
        checks['frozen_candidate_context'] = (frozen.get('candidate_id') == alias
            and frozen.get('task_edit2_capture_campaign') is True
            and frozen.get('job_id') == record['job_id'] and frozen.get('source_identity') == primary
            and frozen.get('task_file') == variant['task']['path'] and frozen.get('scene_file') == variant['scene']['path']
            and frozen.get('source_json') == source['file_check']['path']
            and frozen.get('capture_signature') == record.get('capture_signature')
            and frozen.get('fingerprint') == record.get('fingerprint')
            and frozen.get('seed') == record.get('seed') and frozen.get('variant') == record.get('variant')
            and frozen.get('code_commit') == record.get('code_commit') and not frozen.get('source_only'))
        snapshot = Path(frozen['snapshot_dir']).resolve()
        checks['frozen_snapshot_directory'] = snapshot == (job/'input').resolve()
        hashes = frozen['input_hashes']
        checks['all_frozen_input_hashes'] = bool(hashes) and all((snapshot/relative).resolve().is_relative_to(snapshot)
            and (snapshot/relative).is_file() and sha256(snapshot/relative) == expected for relative,expected in hashes.items())
        for kind,filename,expected in (
            ('task',frozen.get('snapshot_task_file',str(snapshot/'task.yaml')),variant['task']['sha256']),
            ('scene',frozen['snapshot_scene_file'],variant['scene']['sha256']),
            ('source',frozen['snapshot_source_file'],source['source_sha256'])):
            path = Path(filename).resolve()
            checks['frozen_'+kind+'_bytes'] = (path.is_relative_to(snapshot) and path.is_file()
                and hashes.get(str(path.relative_to(snapshot))) == expected and sha256(path) == expected)
        if not all(checks.values()):
            raise ValueError('Frozen capture manifest or input bytes do not match the current explicit variant.')
        value.update(binding_available=True,reason=None,manifest_file=str(manifest_path.resolve()),
            manifest_sha256=sha256(manifest_path))
        if status.get('receipt'):
            receipt_path = Path(status['receipt']).resolve()
            receipt = read_json(receipt_path,{})
            value.update(receipt_file=str(receipt_path),receipt_sha256=sha256(receipt_path) if receipt_path.is_file() else None)
            complete = (status.get('state') in ('REVIEW_PENDING','CACHED') and complete_images(frozen,receipt)
                and receipt.get('id') == alias and receipt.get('capture_job_id') == record['job_id'])
            checks['complete_capture_receipt'] = complete
            if complete:
                value.update(complete=True,images=deepcopy(receipt['images']),receipt=receipt)
            elif status.get('state') in ('REVIEW_PENDING','CACHED'):
                value.update(state='STALE',reason='capture_receipt_not_complete_or_exactly_bound')
        elif status.get('state') in ('REVIEW_PENDING','CACHED'):
            checks['complete_capture_receipt'] = False
            value.update(state='STALE',reason='completed_queue_status_has_no_capture_receipt')
    except (ValueError,TypeError,KeyError,AttributeError,OSError) as error:
        value.update(state='UNBOUND',binding_available=False,complete=False,images={},receipt={},
                     reason=f'{type(error).__name__}: {error}')
    return value


def make_card(variant, manifest, output, evidence_index, *, new_only=False, capture_queue_root=None):
    key = variant['variant_key']
    if not isinstance(key, str) or not re.fullmatch('[0-9a-f]+', key):
        raise ValueError('Stable Review aliases require the manifest hexadecimal variant_key.')
    alias = 'category_' + key
    directory = Path(output) / 'cards' / alias
    sources = _sources(variant)
    checks = {kind: _current_file(variant[kind]) for kind in ('task', 'scene')}
    task = yaml.load(Path(checks['task']['path']).read_text(), Loader=UniqueLoader) if checks['task']['matches'] else {}
    primary = (task.get('source') or {}).get('json')
    if primary not in {row['source_identity'] for row in sources}:
        primary = sources[0]['source_identity']
    source = next(row for row in sources if row['source_identity'] == primary)
    big, small = _category(primary)
    public = _public(task.get('public') or {})
    if not public.get('objective'):
        public = {'objective': 'The current definition is unavailable or differs from this manifest; regenerate its review manifest.',
                  'success_criteria': 'Resolve the recorded definition provenance before qualification or execution.',
                  'environment': 'Provisional diagnostic context.', 'constraints': []}
    tags = [(big, small)] + [_category(row['source_identity']) for row in sources]
    tags = [{'big_task': big, 'small_task': small} for big, small in dict.fromkeys(tags)]
    native, rejected, images = _evidence(variant, evidence_index, checks, sources)
    if new_only:
        images = []  # Retain native facts below, without importing old screenshots.
    queued = queue_capture_evidence(variant,capture_queue_root) if capture_queue_root is not None else None
    contexts = {'original_esi_images': [], 'historical_review_images': [], 'explicit_source_review_images': []}
    core = {row['source_identity']: row for row in manifest.get('core_sources', [])}
    for member in sources:
        original = core.get(member['source_identity'], {})
        for filename in original.get('original_esi_images', []):
            contexts['original_esi_images'].append(filename)
        if not new_only:
            contexts['historical_review_images'].extend((original.get('historical_review_images') or {}).get('paths', []))
    if not new_only:
        contexts['explicit_source_review_images'].extend((variant.get('artifacts') or {}).get('explicit_original_or_review_images', []))
    labels = {'original_esi_images': 'Original ESI source context · not current derivative physics',
        'historical_review_images': 'Historical baseline reference · not current derivative physics',
        'explicit_source_review_images': 'Explicit source/review context · not current derivative physics'}
    for role, filenames in contexts.items():
        contexts[role] = list(dict.fromkeys(filenames))
        for index, filename in enumerate(contexts[role]):
            item = _image(filename, f'{role}_{index}', labels[role], role)
            if item:
                images.append(item)
    if queued and queued['complete']:
        for view, filename in queued['images'].items():
            item = _image(filename,view,'Current TaskEdit2 capture · ' + view,'current_queue_capture')
            if item:
                item.update(kind='native',capture_job_id=queued['job_id'],queue_capture_certificate=True)
                images.append(item)
    images = list({row['path']: row for row in reversed(images)}.values())[::-1]
    matched_ids = {row['evidence_id'] for row in native}
    statuses = {}
    for field in STATUS_FIELDS:
        if field in ('schema_status', 'source_binding_status', 'definition_review_status'):
            statuses[field] = variant.get(field, 'unknown')
        else:
            values = list(dict.fromkeys(row.get(field, 'unknown') for row in native))
            statuses[field] = values[0] if len(values) == 1 else 'mixed_native_attempts' if values else 'unavailable'
    if not all(check['matches'] for check in checks.values()):
        statuses['schema_status'] = 'unavailable'
    if not all(s['file_check']['matches'] for s in sources):
        statuses['source_binding_status'] = 'unavailable'
    pending = list(dict.fromkeys(variant.get('pending_review_gates', []) + variant.get('pending_gates', [])))
    if rejected or not all(check['matches'] for check in checks.values()) or not all(s['file_check']['matches'] for s in sources):
        pending.append('current_definition_source_and_native_result_provenance')
    membership = variant.get('membership_class', 'unresolved')
    explanation = MEMBERSHIP_EXPLANATIONS.get(membership,
        'TaskEdit2 provisional source membership is unresolved and requires review.')
    change_summary = [explanation,
        '; '.join(f'{STATUS_LABELS[field]}: {statuses[field]}' for field in STATUS_FIELDS[:3]) + '.',
        '; '.join(f'{STATUS_LABELS[field]}: {statuses[field]}' for field in STATUS_FIELDS[3:]) + '.',
        'Pending gates: ' + (', '.join(sorted(set(pending))) if pending else
                            'none explicitly listed; native qualification remains unavailable') + '.']
    fingerprint = _digest({'variant_key': key, 'files': checks,
        'sources': [{k: row[k] for k in ('source_identity', 'source_sha256', 'file_check')} for row in sources],
        'native_result_hashes': [(row['evidence_id'], row['result_sha256']) for row in native], 'public': public})
    record = {'candidate_id': alias, 'actual_candidate_id': variant.get('candidate_id'),
        'task_edit2_variant_key': key, 'fingerprint': fingerprint,
        'source_identity': primary, 'original_source': source['file_check']['path'],
        'big_task': big, 'small_task': small, 'scene_model': source.get('scene') or Path(primary).parts[2],
        'room': source.get('room') or Path(primary).parts[3], 'status': 'pending',
        'task_file': checks['task']['path'], 'scene_file': checks['scene']['path'],
        'diagnostic_dir': str(directory.resolve()), 'job_id': queued['job_id'] if queued and queued['binding_available'] else None,
        'diagnostic_images': images, 'diagnostic_logs': [], 'warning_codes': list(variant.get('recorded_warnings', [])),
        'converter_version': 'task_edit2_provisional_review_overlay_v1',
        'public': public, 'category_tags': tags,
        'composite': {'group_id': key, 'member_count': len(sources),
            'sources': [row['source_identity'] for row in sources], 'public_goals': _public_goals(task, sources, public)},
        'limitations': [explanation, 'Provisional derivative; final selection belongs to the user.',
            'Manual native diagnostics and historical source images do not establish task success, capture certification or layout approval.'],
        'core_selection_status': 'TaskEdit2/provisional/' + membership,
        'change_summary': change_summary,
        'selection_status': 'provisional_user_review', 'selected_final_variant': False,
        'membership_class': membership,
        'original_core_members': deepcopy(variant.get('original_core_members', [])),
        'additional_source_members': deepcopy(variant.get('additional_source_members', [])),
        'pending_gates': sorted(set(pending)), 'current_definition_hash_checks': checks,
        'source_file_checks': [row['file_check'] for row in sources],
        'native_diagnostics': native, 'rejected_native_evidence': rejected,
        'native_evidence_ids': sorted(matched_ids), 'source_context_images': contexts,
        'autonomous_agent_performance': False, 'native_capture_complete': False,
        'layout_approval_available': False, **statuses}
    if new_only or capture_queue_root is not None:
        record['task_edit2_capture_campaign'] = True
    capture_status = {'state':'REVIEW_ONLY'}
    if queued:
        record['capture_queue_evidence'] = {key:deepcopy(value) for key,value in queued.items()
            if key not in ('record','receipt')}
        record['queue_capture_receipt'] = ({key:deepcopy(queued['receipt'].get(key)) for key in
            ('interaction_setup','diagnostic_pending','provenance','capture_signature','complete')}
            if queued['complete'] else None)
        capture_status = {**queued['status'],'state':queued['state'],
            'error':queued['status'].get('error') or queued['reason']}
        record['variant'] = queued['record'].get('variant','adapted')
        record['seed'] = queued['record'].get('seed',0)
    row = task_summary(record, capture_status, public)
    fields = ('actual_candidate_id', 'task_edit2_variant_key', 'selection_status', 'selected_final_variant',
        'core_selection_status', 'change_summary',
        'membership_class', 'original_core_members', 'additional_source_members', 'pending_gates',
        'current_definition_hash_checks', 'source_file_checks', 'native_diagnostics', 'rejected_native_evidence',
        'native_evidence_ids', 'source_context_images', 'autonomous_agent_performance', *STATUS_FIELDS)
    row.update({field: deepcopy(record[field]) for field in fields})
    row.update(instance_status='provisional_definition_review', layout_approval_available=False,
               native_capture_complete=False, physical_qualified=False, has_global_preview=False,
               no_native_image=not any(image['evidence_role'] == 'exact_bound_native_diagnostic' for image in images))
    if new_only or capture_queue_root is not None:
        row['task_edit2_capture_campaign'] = True
    if queued:
        row.update(capture_queue_evidence=deepcopy(record['capture_queue_evidence']),
            queue_capture_receipt=deepcopy(record['queue_capture_receipt']),
            instance_status='capture_complete' if queued['complete'] else 'capture_' + queued['state'].lower(),
            layout_approval_available=queued['complete'],native_capture_complete=queued['complete'],
            has_global_preview=queued['complete'] and any(view.startswith('global_') for view in queued['images']),
            no_native_image=not queued['complete'],images=[deepcopy(item) for item in images
                if item.get('evidence_role')=='current_queue_capture'])
        record.update(native_capture_complete=queued['complete'],layout_approval_available=queued['complete'])
    write_json(directory / 'record.json', record)
    return row


def _historical_baseline_rows(snapshot, core_sources, output_dir):
    """Keep original IDs/history while disabling historical capture eligibility."""
    index = {row['original_candidate_id']: row for row in core_sources}
    for row in snapshot['scenes'] + snapshot['diagnostics']:
        if not re.fullmatch('category_[0-9a-f]+', row['id']):
            raise ValueError('Original baseline cards require exact canonical Review IDs.')
        source = index.get(row['id'], {})
        previous_dir = row.get('diagnostic_dir')
        previous_file = Path(previous_dir)/'record.json' if previous_dir else None
        previous = read_json(previous_file, {}) if previous_file and previous_file.is_file() else {}
        declared = (list(row.get('images') or []) + list(row.get('diagnostic_images') or [])
                    + list(previous.get('diagnostic_images') or []))
        for filenames in (row.get('source_context_images') or {}).values():
            if isinstance(filenames, list):
                declared += [{'path': filename} for filename in filenames if isinstance(filename, str)]
        declared += [{'path': filename} for filename in
            (source.get('historical_review_images') or {}).get('paths', [])]
        declared += [{'path': filename} for filename in source.get('original_esi_images', [])]
        images = {}
        for item in declared:
            if not isinstance(item, dict) or not item.get('path'):
                continue
            path = Path(item['path'])
            if not path.is_absolute() or not path.is_file():
                continue
            image = deepcopy(item)
            image.update(path=str(path.resolve()), kind='historical',
                view=item.get('view') or path.stem,
                label='Original baseline historical context · not current capture',
                evidence_role='original_baseline_historical_context', queue_capture_certificate=False)
            images[image['path']] = image
        row.update(original_baseline=True, native_capture_complete=False, layout_approval_available=False,
            physical_qualified=False, has_global_preview=False, has_final_preview=False,
            preview_is_current=False, no_native_image=True, images=list(images.values()),
            diagnostic_images=list(images.values()), selected_final_variant=False,
            historical_capture_state=(row.get('capture') or {}).get('state'),
            baseline_evidence_role='historical_context_only')
        directory = Path(output_dir)/'cards'/row['id']
        identity = source.get('source_identity') or row.get('source') or previous.get('source_identity')
        public = _public(row.get('robot_task_public') or previous.get('public') or {})
        if not public.get('objective'):
            public['objective'] = row.get('objective') or row.get('question') or 'Original baseline metadata and diagnostics.'
        record = {**deepcopy(previous), 'candidate_id': row['id'], 'original_baseline': True,
            'source_identity': identity, 'original_source': source.get('source_file') or previous.get('original_source') or row.get('original_source'),
            'task_file': source.get('original_task_file', row.get('task_file')),
            'scene_file': source.get('original_scene_file', row.get('scene_file')),
            'status': previous.get('status') or row.get('conversion_result_status') or 'original_baseline',
            'failure_stage': previous.get('failure_stage') or row.get('failure_stage'),
            'failure_reason': previous.get('failure_reason') or row.get('failure_reason') or row.get('failure'),
            'diagnostic_logs': sorted(set(previous.get('diagnostic_logs', [])+row.get('diagnostic_logs', []))),
            'diagnostic_dir': str(directory.resolve()), 'historical_diagnostic_dir': previous_dir,
            'baseline_record_file': str(previous_file.resolve()) if previous_file and previous_file.is_file() else None,
            'public': public, 'diagnostic_images': deepcopy(row['diagnostic_images']),
            'native_capture_complete': False, 'layout_approval_available': False, 'physical_qualified': False,
            'selected_final_variant': False, 'baseline_evidence_role': 'historical_context_only',
            'limitations': list(previous.get('limitations') or []) +
                ['Original baseline context only; historical captures do not establish current initialization or layout approval.']}
        row['diagnostic_dir'] = record['diagnostic_dir']
        write_json(directory/'record.json', record)


def core_groups(manifest, baseline_rows, cards):
    """Return source groups plus additional IDs without copying shared cards/receipts."""
    original = {row['id']: row for row in baseline_rows}
    current = {row['id']: row for row in cards}
    inputs = {str(Path(row['path']).resolve()): index
              for index, row in enumerate(manifest.get('explicit_current_inputs', []))}
    sources = manifest.get('core_sources', [])
    identities = {row['source_identity'] for row in sources}
    if len(identities) != len(sources):
        raise ValueError('Core browsing groups require unique original source identities.')
    members = {identity: [] for identity in identities}
    additional = []
    for variant in manifest.get('variants', []):
        if variant.get('role') != 'explicit_derivative':
            continue
        alias = 'category_' + variant['variant_key']
        if alias not in current:
            raise ValueError('Explicit derivative has no canonical Review card: ' + alias)
        declared = variant.get('original_core_members') or []
        if not declared:
            additional.append(alias)
        ranks = [(inputs[str(Path(record['manifest_file']).resolve())], record['manifest_file'])
                 for record in variant.get('manifest_records', [])
                 if record.get('manifest_file') and str(Path(record['manifest_file']).resolve()) in inputs]
        rank, filename = max(ranks, default=(-1, None))
        card = current[alias]
        version = {'id': alias, 'label': 'Revision · ' + (Path(filename).parent.name if filename else variant['variant_key']),
            'role': 'explicit_derivative', 'capture_state': (card.get('capture') or {}).get('state', 'UNAVAILABLE'),
            'task_file': card.get('task_file'), 'scene_file': card.get('scene_file'),
            'variant_key': variant['variant_key'], 'actual_candidate_id': variant.get('candidate_id'),
            'declared_input_index': rank, 'declared_manifest_file': filename,
            'native_capture_complete': bool(card.get('native_capture_complete')),
            'layout_approval_available': bool(card.get('layout_approval_available')),
            'selected_final_variant': False}
        for identity in dict.fromkeys(declared):
            if identity not in members:
                raise ValueError('Derivative names an unknown original Core source: ' + identity)
            members[identity].append(version)
    groups = []
    for source in sources:
        key, original_id = source['original_variant_key'], source['original_candidate_id']
        if not isinstance(key, str) or not re.fullmatch('[0-9a-f]+', key) or original_id not in original:
            raise ValueError('Original Core source has no exact canonical baseline Review card.')
        baseline = original[original_id]
        versions = [{'id': original_id, 'label': 'Original baseline · historical context',
            'role': 'original_baseline', 'capture_state': 'HISTORICAL', 'variant_key': key,
            'task_file': source.get('original_task_file') or baseline.get('task_file'),
            'scene_file': source.get('original_scene_file') or baseline.get('scene_file'),
            'historical_capture_state': (baseline.get('capture') or {}).get('state'),
            'native_capture_complete': False, 'layout_approval_available': False, 'selected_final_variant': False}]
        revisions = sorted(members[source['source_identity']],
                           key=lambda row: (row['declared_input_index'], row['variant_key']))
        versions.extend(deepcopy(revisions))
        groups.append({'id': 'core_' + key, 'source_identity': source['source_identity'],
            'original_variant_id': original_id, 'original_candidate_id': original_id,
            'default_variant_id': versions[-1]['id'], 'versions': versions,
            'browsing_default_only': True, 'default_policy': 'latest_declared_input_then_variant_key',
            'selected_final_variant': False})
    return groups, additional


def publish(manifest_path, baseline_catalogue, output_dir, *, new_only=False, capture_queue_root=None, group_core=False):
    if new_only and group_core:
        raise ValueError('group_core and new_only are mutually exclusive browsing modes.')
    manifest_path, baseline_catalogue, output_dir = (Path(path).resolve() for path in
                                                    (manifest_path, baseline_catalogue, output_dir))
    if output_dir / 'snapshot.json' in (manifest_path, baseline_catalogue):
        raise ValueError('Staged publication cannot overwrite an input manifest or baseline catalogue.')
    manifest, baseline = read_json(manifest_path), read_json(baseline_catalogue)
    snapshot = deepcopy(baseline)
    if not all(isinstance(snapshot.get(key), list) for key in ('scenes', 'diagnostics', 'categories')):
        raise ValueError('Baseline snapshot requires original scenes, diagnostics and categories lists.')
    if new_only:
        snapshot.update(scenes=[],diagnostics=[],categories=[])
    if group_core:
        _historical_baseline_rows(snapshot, manifest.get('core_sources', []), output_dir)
        snapshot['diagnostics'] = snapshot['scenes'] + snapshot['diagnostics']
        snapshot['scenes'] = []
    original_ids = {row['id'] for row in snapshot['scenes'] + snapshot['diagnostics']}
    evidence_index = {}
    for evidence in manifest.get('native_evidence', []):
        if evidence['evidence_id'] in evidence_index:
            raise ValueError('Ambiguous duplicate native evidence identity.')
        evidence_index[evidence['evidence_id']] = evidence
    derivatives = [row for row in manifest.get('variants', []) if row.get('role') == 'explicit_derivative']
    aliases = ['category_' + row['variant_key'] for row in derivatives]
    if len(set(aliases)) != len(aliases) or set(aliases) & original_ids:
        raise ValueError('Derivative aliases must be unique and cannot replace original rows.')
    cards = [make_card(row, manifest, output_dir, evidence_index,new_only=new_only,
                       capture_queue_root=capture_queue_root) for row in derivatives]
    if new_only:
        snapshot['scenes'] = [row for row in cards if row['native_capture_complete']]
        snapshot['diagnostics'] = [row for row in cards if not row['native_capture_complete']]
    elif group_core:
        snapshot['core_groups'], snapshot['additional_variant_ids'] = core_groups(
            manifest, snapshot['scenes'] + snapshot['diagnostics'], cards)
        snapshot['scenes'].extend(row for row in cards if row['native_capture_complete'])
        snapshot['diagnostics'].extend(row for row in cards if not row['native_capture_complete'])
        snapshot['final_core_selection_made'] = False
    else:
        snapshot['diagnostics'].extend(cards)
    if capture_queue_root is not None:
        states = Counter(row['capture']['state'] for row in cards)
        snapshot['capture_queue'] = {'root':str(Path(capture_queue_root).expanduser().resolve()),
            'candidate_count':len(cards),'counts':dict(states),
            'captured':sum(row['native_capture_complete'] for row in cards),
            'queued':sum(states[state] for state in ('QUEUED','SUBMISSION_PENDING','NOT_SUBMITTED')),
            'running':sum(states[state] for state in ('CLAIMED','RUNNING','CAPTURED')),
            'failed':sum(states[state] for state in ('FAILED','STALE','CANCELLED','UNBOUND'))}
    elif new_only:
        snapshot['capture_queue'] = {'root':None,'candidate_count':len(cards),'counts':{'NOT_SUBMITTED':len(cards)}}
    existing_categories = {(row['big_task'], row['small_task']) for row in snapshot['categories']}
    for tag in sorted({(tag['big_task'], tag['small_task']) for row in cards for tag in row['category_tags']} - existing_categories):
        snapshot['categories'].append({'big_task': tag[0], 'small_task': tag[1], 'source_count': len({
            goal['source'] for row in cards for goal in row['composite']['public_goals']
            if (goal['big_task'], goal['small_task']) == tag}), 'counts': {}})
    snapshot['task_edit2_overlay'] = {'schema': 'task_edit2_provisional_review_overlay_v1',
        'manifest_file': str(manifest_path), 'manifest_sha256': sha256(manifest_path),
        'baseline_catalogue': str(baseline_catalogue), 'baseline_catalogue_sha256': sha256(baseline_catalogue),
        'preserved_baseline_scene_rows': len(baseline['scenes']),
        'preserved_baseline_diagnostic_rows': len(baseline['diagnostics']),
        'new_only':bool(new_only),'baseline_rows_hidden_from_display':bool(new_only),
        'explicit_derivative_cards': len(cards), 'membership_classes': dict(Counter(row['membership_class'] for row in cards)),
        'final_selection_owner': 'user', 'final_core_selection_made': False,
        'native_queue_records_created': False, 'running_services_changed': False}
    if group_core:
        snapshot['task_edit2_overlay'].update(group_core=True, core_group_count=len(snapshot['core_groups']),
            additional_variant_count=len(snapshot['additional_variant_ids']), browsing_default_only=True)
    write_json(output_dir / 'snapshot.json', snapshot)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--baseline-catalogue', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--watch', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--new-only', action='store_true')
    modes.add_argument('--group-core', action='store_true')
    parser.add_argument('--capture-queue-root', type=Path)
    args = parser.parse_args()
    while True:
        # Revalidate exact definition/source/result bytes even if the manifest
        # itself has not changed since the preceding publication.
        snapshot = publish(args.manifest, args.baseline_catalogue, args.output_dir,
            new_only=args.new_only,capture_queue_root=args.capture_queue_root,group_core=args.group_core)
        print(json.dumps(snapshot['task_edit2_overlay']), flush=True)
        if not args.watch:
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
