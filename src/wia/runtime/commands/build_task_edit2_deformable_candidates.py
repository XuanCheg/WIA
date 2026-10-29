#!/usr/bin/env python3
"""Generate pending real-cloth support derivatives for all actual Core sources.

Run: python -B src/wia/runtime/commands/build_task_edit2_deformable_candidates.py --output-dir
 outputs/task_edit2_20261018/deformable_candidates_v1
This script only performs CPU conversion/schema checks; native mesh and actions
remain unavailable until separately measured through the original mechanism.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
from wia.authoring.conversion import ConversionContext, SourceInstance
from wia.authoring.converters.physical_structure_deformable import CategoryConverter, DEFORMABLE_REVISION
from wia.runtime.scene_capture_queue import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-manifest', type=Path, default=Path('/home/zhouxinyang/esi_category_conversion/core_revision_20261018/core_candidates_for_review.json'))
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/task_edit2_20261018/deformable_candidates_v1')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    members = [row for row in json.loads(args.core_manifest.read_text())['tasks'] if
        row.get('category') == 'Physical Structure' and row.get('subcategory') == 'Deformable']
    source_map = {row['source']: row['candidate_id'] for row in members}
    if not members or len(source_map) != len(members):
        raise ValueError('Actual Core deformable membership is missing or ambiguous.')
    sources = [SourceInstance(WIA_ESI_ROOT / 'dataset/json_clean' / identity,
        identity, 'Physical Structure', 'Deformable') for identity in sorted(source_map)]
    helper = ROOT / 'src/wia/tasks/deformable_support.py'
    context = ConversionContext(args.output_dir.resolve(), args.seed, {'task_edit2_deformable': True,
        'deformable_support_helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest()})
    manifest = CategoryConverter().convert_all(sources, context)
    value = {'schema': DEFORMABLE_REVISION, 'status': 'provisional', 'source_count': len(sources),
        'actual_core_manifest': str(args.core_manifest.resolve()),
        'source_mapping': [{'source_identity': row['source_identity'],
            'original_core_candidate_id': source_map[row['source_identity']], 'candidate_id': row['candidate_id'],
            'task_file': row['task_file'], 'scene_file': row['scene_file'], 'generation_status': row['status'],
            'native_executable': False} for row in manifest['records']],
        'mandatory_global_input_condition': 'frozen_initial_reset_world_RGB_for_every_task',
        'native_gates_pending': ['source_world_particles_topology_physics_and_true_cover',
            'full_cloth_and_item_support_footprint', 'pre_physics_particle_translation',
            'post_settle_mesh_contact_cover_and_physics', 'robot_approach', 'actual_hand_sweep_reveal_grasp_lift'],
        'final_selection_owner': 'user'}
    write_json(args.output_dir / 'source_mapping.json', value)
    print(json.dumps({'source_count': len(sources),
        'generated': sum(row['generation_status'] == 'generated' for row in value['source_mapping']),
        'mapping': str((args.output_dir / 'source_mapping.json').resolve())}), flush=True)
    return value


if __name__ == '__main__':
    main()
