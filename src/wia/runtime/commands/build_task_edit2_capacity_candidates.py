#!/usr/bin/env python3
"""Generate a provisional joint capacity task from the four current Rs_int sources.

Run: python -B src/wia/runtime/commands/build_task_edit2_capacity_candidates.py --output-dir \
outputs/task_edit2_20261018/capacity_inventory
This writes new candidates only; no source, historical task or approval is changed.
Native calibration and tabletop/label qualification remain explicit pending gates.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
from wia.authoring.conversion import ConversionContext, SourceInstance
from wia.authoring.converters.physical_structure_liquid_volume import _source_data, capacity_evidence
from wia.authoring.converters.liquid_capacity_inventory import build_joint_candidate
from wia.runtime.scene_capture_queue import write_json
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/task_edit2_20261018/capacity_inventory')
    parser.add_argument('--core-records', type=Path,
        default=Path('/home/zhouxinyang/esi_category_conversion/core_revision_20261018/queue/category_records'))
    parser.add_argument('--source-runs', type=int, nargs='+', default=[247, 89, 191, 275])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--calibration-receipt', type=Path,
        help='Optional exact-inventory native diagnostic receipt; validity is rechecked.')
    args = parser.parse_args(argv)
    context = ConversionContext(args.output_dir, seed=args.seed)
    rows = [json.loads(path.read_text()) for path in args.core_records.glob('*.json')]
    selected, evidence = [], []
    for run in args.source_runs:
        identity = f'Physical Structure/Liquid Volume/unknown_scene/unknown_room/run_{run}.json'
        matches = [row for row in rows if row.get('source_identity') == identity]
        if len(matches) != 1:
            raise ValueError('Each selected current Core source needs one explicit record mapping: ' + identity)
        source = SourceInstance(WIA_ESI_ROOT / 'dataset/json_clean' / identity,
            identity, 'Physical Structure', 'Liquid Volume')
        _, _, entries, _ = _source_data(source)
        evidence.append((source, capacity_evidence(source, entries, context)))
        selected.append({'candidate_id': matches[0]['candidate_id'], 'source_identity': identity,
            'historical_task_file': matches[0].get('task_file'), 'historical_scene_file': matches[0].get('scene_file')})
    calibrated = json.loads(args.calibration_receipt.read_text()) if args.calibration_receipt else None
    result = build_joint_candidate(evidence, context, calibration_receipt=calibrated)
    task_definition(result['task_definition'])
    SceneDefinition.model_validate(result['scene_definition'])
    destination = args.output_dir / 'candidates' / result['candidate_id']
    destination.mkdir(parents=True, exist_ok=True)
    task_path, scene_path = destination / 'task.yaml', destination / 'scene.yaml'
    task_path.write_text(yaml.safe_dump(result['task_definition'], sort_keys=False, allow_unicode=True))
    scene_path.write_text(yaml.safe_dump(result['scene_definition'], sort_keys=False, allow_unicode=True))
    receipt = result['inventory_receipt']
    write_json(destination / 'inventory_receipt.json', receipt)
    mapping = {'selected_core_sources': selected, 'source_side_to_joint_identity': result['source_mappings'],
        'all_source_members_retained': True, 'task_file': str(task_path.resolve()), 'scene_file': str(scene_path.resolve())}
    write_json(destination / 'source_mappings.json', mapping)
    record = {key: value for key, value in result.items() if key not in ('task_definition', 'scene_definition')}
    record.update(task_file=str(task_path.resolve()), scene_file=str(scene_path.resolve()),
        selected_core_sources=selected)
    write_json(destination / 'record.json', record)
    manifest = {'schema': 'task_edit2_capacity_candidate_manifest_v1', 'records': [record],
        'candidate_count': 1, 'container_count': len(receipt['inventory']),
        'source_count': len(selected), 'pair_denominator': receipt['truth']['pair_denominator'],
        'unknown_pairs': receipt['truth'].get('missing_pairs', []),
        'native_execution_verified': False, 'status': 'pending'}
    write_json(args.output_dir / 'batch_manifest.json', manifest)
    print(json.dumps({'candidate_id': result['candidate_id'], 'status': 'pending',
        'containers': manifest['container_count'], 'source_members': len(selected),
        'pair_denominator': manifest['pair_denominator'], 'unknown_pairs': len(manifest['unknown_pairs']),
        'manifest': str((args.output_dir / 'batch_manifest.json').resolve())}), flush=True)
    return manifest


if __name__ == '__main__':
    main()
