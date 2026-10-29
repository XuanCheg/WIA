"""Build exact Core Geometry derivatives without native simulation or Review mutation.

Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -B src/wia/runtime/commands/build_task_edit2_geometry_candidates.py
The default source-reset seed is 0. Native support, layout, control and scoring remain pending.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

import yaml

from wia.authoring.conversion import ConversionContext, SourceInstance
from wia.authoring.converters import spatial_relations_geometric_configuration as geometry
from wia.runtime.scene_capture_queue import sha256, write_json
from wia.tasks.loader import UniqueLoader, task_definition
from wia.tasks.schema import SceneDefinition

CORE_MANIFEST = Path('/home/zhouxinyang/esi_category_conversion/core_revision_20261018/core_candidates_for_review.json')
OUTPUT = ROOT / 'outputs/task_edit2_20261018/geometry_core'
SOURCE_ROOT = WIA_ESI_ROOT / 'dataset/json_clean'
PREFIX = 'Spatial Relations/Geometric Configuration/'


def definition_check(filename, kind, identity):
    """Validate actual bytes and source binding independently of native feasibility."""
    if not filename:
        return {'status': 'unavailable', 'sha256': None, 'reason': 'definition_not_generated'}
    path = Path(filename).resolve()
    if not path.is_file():
        return {'status': 'unavailable', 'sha256': None, 'reason': 'definition_file_missing'}
    result = {'status': 'schema_invalid', 'sha256': sha256(path), 'file': str(path)}
    try:
        raw = yaml.load(path.read_text(), Loader=UniqueLoader)
        if not isinstance(raw, dict):
            raise ValueError('Definition must be a mapping.')
        if kind == 'task':
            task_definition(raw)
            parameters = raw['private']['evaluation']['parameters']
            if parameters.get('task_revision') != geometry.TASK_EDIT2_REVISION:
                raise ValueError('Generated task does not declare the requested TaskEdit2 revision.')
        else:
            SceneDefinition.model_validate(raw)
        if raw['source'] != {'provider': 'esi', 'json': identity}:
            raise ValueError('Generated definition changes its exact Core source identity.')
        result.update(status='schema_valid', id=raw['id'])
    except (ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        result['reason'] = f'{type(error).__name__}: {error}'
    return result


def build(output=OUTPUT, seed=0, core_manifest=CORE_MANIFEST):
    """Reuse the owned converter on tasks/source membership, retaining every failure."""
    output = Path(output).resolve()
    core_manifest = Path(core_manifest).resolve()
    original_rows = json.loads(core_manifest.read_text())['tasks']
    selected = [row for row in original_rows if row['source'].startswith(PREFIX)]
    by_source = {row['source']: row for row in selected}
    if len(by_source) != len(selected):
        raise ValueError('Core Geometry source membership must be unique.')
    sources = [SourceInstance(SOURCE_ROOT / row['source'], row['source'], *geometry.CategoryConverter.category)
               for row in selected]
    config = {'task_revision': geometry.TASK_EDIT2_REVISION}
    manifest = geometry.CategoryConverter().convert_all(sources, ConversionContext(output, seed=seed, config=config))
    records = manifest['records']
    if len(records) != len(sources) or {row['source_identity'] for row in records} != set(by_source):
        raise ValueError('Converter results do not match the exact Core Geometry source membership.')
    mappings, pending = [], []
    for record in records:
        identity = record['source_identity']
        original = by_source[identity]
        source_path = SOURCE_ROOT / identity
        source_hash = sha256(source_path) if source_path.is_file() else None
        checks = {kind: definition_check(record.get(kind + '_file'), kind, identity) for kind in ('task', 'scene')}
        provenance_ok = bool(source_hash and record.get('source_sha256') == source_hash)
        converter_status = record.get('converter_status', record['status'])
        complete = converter_status == 'generated' and all(row['status'] == 'schema_valid' for row in checks.values())
        status = ('generated' if complete and provenance_ok else 'pending_source_provenance' if not provenance_ok
                  else 'pending_source_conversion' if converter_status != 'generated' else 'pending_definition_validation')
        record.update(converter_status=converter_status, status=status,
            source_core_candidate_id=original['candidate_id'], core_membership='exact_original_tasks_source',
            task_revision=geometry.TASK_EDIT2_REVISION, definition_checks=checks,
            source_provenance_status='verified_current_source_bytes' if provenance_ok else 'unavailable_or_hash_mismatch',
            final_selection='provisional', final_selection_owner='user', generated_definition=status == 'generated',
            native_initialization_status='unknown', native_layout_status='unknown',
            native_execution_status='unavailable', scoring_status='unavailable', autonomous_agent_performance=False,
            native_gates={name: 'pending' for name in ('floor_supported_robot_start', 'accessible_support_and_fit',
                'seeded_initial_shape_false', 'source_assets_scales_orientations_and_camera',
                'native_grasp_contact_and_causal_rearrangement', 'stable_released_terminal_arrangement')})
        for kind in ('task', 'scene'):
            record[kind + '_sha256'] = checks[kind]['sha256']
        mapping = {'source_identity': identity, 'source_file': str(source_path.resolve()),
            'source_sha256': source_hash, 'source_core_candidate_id': original['candidate_id'],
            'derivative_candidate_id': record['candidate_id'], 'task_revision': geometry.TASK_EDIT2_REVISION,
            'status': status, 'converter_status': converter_status,
            'source_provenance_status': record['source_provenance_status'], 'definition_checks': deepcopy(checks)}
        for kind in ('task', 'scene'):
            historical = original.get(kind + '_file')
            if historical:
                path = Path(historical).resolve()
                mapping['historical_' + kind + '_file'] = str(path)
                mapping['historical_' + kind + '_sha256'] = sha256(path) if path.is_file() else None
            if record.get(kind + '_file'):
                mapping[kind + '_file'] = str(Path(record[kind + '_file']).resolve())
                mapping[kind + '_sha256'] = checks[kind]['sha256']
        mappings.append(mapping)
        if status != 'generated':
            pending.append({**deepcopy(mapping), 'generated_definition': False,
                'failure_stage': record.get('failure_stage'), 'failure_reason': record.get('failure_reason')})
        write_json(output / 'candidates' / record['candidate_id'] / 'record.json', record)
    manifest.update(converter_batch_schema=manifest['schema'], schema='task_edit2_geometry_core_candidates_v1',
        core_membership={'manifest_file': str(core_manifest), 'manifest_sha256': sha256(core_manifest),
            'membership_field': 'tasks/source', 'source_count': len(selected)},
        source_mapping=mappings, pending_records=pending,
        counts=dict(Counter(row['status'] for row in records)),
        definition_status_counts=dict(Counter(check['status'] for row in records for check in row['definition_checks'].values())),
        final_selection='provisional', final_selection_owner='user', native_experiments_run=False,
        native_evidence_status='unavailable', source_reset_seed=seed)
    write_json(output / 'batch_manifest.json', manifest)
    write_json(output / 'source_mapping.json', mappings)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--core-manifest', type=Path, default=CORE_MANIFEST)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    manifest = build(args.output, args.seed, args.core_manifest)
    print(json.dumps({'core_sources': manifest['source_count'], 'counts': manifest['counts'],
        'pending': len(manifest['pending_records']), 'native_evidence_status': 'unavailable'}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
