#!/usr/bin/env python3
"""Build provisional reflection verification variants from current Core members.

Run: python -B src/wia/runtime/commands/build_task_edit2_reflection_candidates.py --output-dir \
outputs/task_edit2_20261018/reflection_candidates
Use --original-image-root ROOT only for immutable original ESI image archives.
No native feasibility, source-view equivalence or final Core membership is claimed.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import yaml

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
from wia.authoring.conversion import ConversionContext, SourceInstance
from wia.runtime.scene_capture_queue import write_json
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition

MODULES = {
    'Correspondence': ('specular_reflection_correspondence', 'correspondence_operation'),
    'Spatial Relations': ('specular_reflection_spatial_relations', 'spatial_relations_operation'),
    'Reflection Authoring': ('specular_reflection_reflection_authoring', 'reflection_authoring_operation'),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/task_edit2_20261018/reflection_candidates')
    parser.add_argument('--core-manifest', type=Path, default=Path(
        '/home/zhouxinyang/esi_category_conversion/core_revision_20261018/core_candidates_for_review.json'))
    parser.add_argument('--original-image-root', type=Path)
    parser.add_argument('--robot-uniform-scale', type=float)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--all-core', action='store_true', help='Generate every current Core reflection member instead of one representative per subclass.')
    args = parser.parse_args(argv)
    manifest = json.loads(args.core_manifest.read_text())
    members = [dict(row, source_identity=row.get('source_identity') or row['source'])
        for row in manifest['tasks']
        if (row.get('source_identity') or row.get('source', '')).startswith('Specular Reflection/')]
    selected = []
    seen = set()
    for row in sorted(members, key=lambda row: row['source_identity']):
        subclass = Path(row['source_identity']).parts[1]
        if subclass not in MODULES or (not args.all_core and subclass in seen):
            continue
        selected.append(row)
        seen.add(subclass)
    records = []
    for row in selected:
        identity = row['source_identity']
        subclass = Path(identity).parts[1]
        module_name, operation_key = MODULES[subclass]
        module = importlib.import_module('wia.authoring.converters.' + module_name)
        source = SourceInstance(WIA_ESI_ROOT / 'dataset/json_clean' / identity,
            identity, 'Specular Reflection', subclass)
        for mode in ('bounded', 'unrestricted'):
            config = {operation_key: module.ACTION_OPERATION, 'reflection_boundary_mode': mode,
                'reflection_robot_uniform_scale': args.robot_uniform_scale}
            if args.original_image_root:
                config['reflection_original_image_root'] = str(args.original_image_root.resolve())
            result = module.CategoryConverter().convert(source, ConversionContext(args.output_dir, args.seed, config))
            task, scene = result['task_definition'], result['scene_definition']
            task_definition(task)
            SceneDefinition.model_validate(scene)
            token = hashlib.sha256((identity + '\0' + module.ACTION_OPERATION + '\0' + mode + '\0' +
                result['task_revision'] + '\0' + json.dumps(result['task_definition']['private']['evaluation']['parameters']['reflection_sensor_profile'],
                    sort_keys=True)).encode()).hexdigest()[:24]
            candidate_id = 'reflection_verify_' + token
            directory = args.output_dir / 'candidates' / candidate_id
            directory.mkdir(parents=True, exist_ok=True)
            task_path, scene_path = directory / 'task.yaml', directory / 'scene.yaml'
            task_path.write_text(yaml.safe_dump(task, sort_keys=False))
            scene_path.write_text(yaml.safe_dump(scene, sort_keys=False))
            record = {key: value for key, value in result.items() if key not in ('task_definition', 'scene_definition')}
            record.update(candidate_id=candidate_id, status='pending', source_identity=identity,
                source_core_candidate_id=row['candidate_id'], small_task=subclass,
                boundary_mode=mode, task_file=str(task_path.resolve()), scene_file=str(scene_path.resolve()),
                native_execution_verified=False)
            write_json(directory / 'record.json', record)
            records.append(record)
    value = {'schema': 'task_edit2_reflection_candidate_manifest_v1', 'status': 'provisional',
        'records': records, 'candidate_count': len(records), 'source_count': len(selected),
        'source_view_failures': [{'candidate_id': row['candidate_id'], 'source_identity': row['source_identity'],
            'reason': row['source_reference_provenance']['reason']} for row in records
            if not row['source_reference_provenance']['available']],
        'mandatory_global_input_condition': 'frozen_initial_reset_world_RGB_for_every_task',
        'all_eight_mounted_and_global_shortcut_audit': 'pending_per_candidate_native_evidence',
        'no_category_wide_equivalence_inference': True, 'final_selection_owner': 'user'}
    write_json(args.output_dir / 'batch_manifest.json', value)
    print(json.dumps({'candidate_count': len(records), 'source_count': len(selected),
        'original_source_image_missing': len(value['source_view_failures']),
        'manifest': str((args.output_dir / 'batch_manifest.json').resolve())}), flush=True)
    return value


if __name__ == '__main__':
    main()
