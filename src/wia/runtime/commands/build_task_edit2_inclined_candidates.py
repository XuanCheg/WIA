#!/usr/bin/env python3
"""Build pending whole-apparatus derivatives for every actual Core Inclined source.

Run: python -B src/wia/runtime/commands/build_task_edit2_inclined_candidates.py --output-dir
 outputs/task_edit2_20261018/inclined_apparatus_candidates_v1
This CPU generator runs no native physics and claims no support/access/release.
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
from wia.authoring.converters.physical_dynamics_inclined_plane import CategoryConverter, APPARATUS_REVISION
from wia.runtime.scene_capture_queue import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-manifest', type=Path, default=Path('/home/zhouxinyang/esi_category_conversion/core_revision_20261018/core_candidates_for_review.json'))
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/task_edit2_20261018/inclined_apparatus_candidates_v1')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    core = json.loads(args.core_manifest.read_text())
    members = [row for row in core['tasks'] if row.get('category') == 'Physical Dynamics' and row.get('subcategory') == 'Inclined Plane']
    if not members:
        raise ValueError('Actual Core manifest contains no Inclined Plane sources.')
    source_map = {row['source']: row['candidate_id'] for row in members}
    if len(source_map) != len(members):
        raise ValueError('Actual Core source membership is ambiguous.')
    sources = [SourceInstance(WIA_ESI_ROOT / 'dataset/json_clean' / identity, identity, 'Physical Dynamics', 'Inclined Plane')
        for identity in sorted(source_map)]
    helper = ROOT / 'src/wia/tasks/inclined_apparatus.py'
    context = ConversionContext(args.output_dir.resolve(), args.seed, {'task_edit2_inclined': True,
        'inclined_apparatus_contract_sha256': hashlib.sha256(helper.read_bytes()).hexdigest()})
    manifest = CategoryConverter().convert_all(sources, context)
    mappings = [{'source_identity': row['source_identity'], 'original_core_candidate_id': source_map[row['source_identity']],
        'candidate_id': row['candidate_id'], 'task_file': row['task_file'], 'scene_file': row['scene_file'],
        'generation_status': row['status'], 'native_executable': False} for row in manifest['records']]
    value = {'schema': APPARATUS_REVISION, 'status': 'provisional', 'actual_core_manifest': str(args.core_manifest.resolve()),
        'source_count': len(sources), 'source_mapping': mappings,
        'mandatory_global_input_condition': 'frozen_initial_reset_world_RGB_for_every_task',
        'initial_shortcut_audit': 'pending_all_mounted_sensors_and_frozen_global',
        'native_gates_pending': ['pre_translation_source_physics_and_pickup_bounds', 'whole_envelope_deck_fit',
            'initial_native_contacts_and_robot_floor', 'public_approach', 'real_grasp_lift_and_top_release',
            'one_second_contained_motion_and_safe_runoff'], 'final_selection_owner': 'user'}
    write_json(args.output_dir / 'source_mapping.json', value)
    print(json.dumps({'source_count': len(sources), 'generated': sum(row['generation_status'] == 'generated' for row in mappings),
        'mapping': str((args.output_dir / 'source_mapping.json').resolve())}), flush=True)
    return value


if __name__ == '__main__':
    main()
