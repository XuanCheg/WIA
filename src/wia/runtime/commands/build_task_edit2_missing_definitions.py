#!/usr/bin/env python3
"""Build only the actual Core destination/region-label gaps as immutable derivatives.

Run: python src/wia/runtime/commands/build_task_edit2_missing_definitions.py --output-dir outputs/task_edit2_20261018/missing_definitions
This does not render, start a simulator, modify queues/reviews or overwrite sources.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
from collections import Counter
from functools import lru_cache
import json
import hashlib
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.authoring.conversion import ConversionContext, SourceInstance
from wia.authoring.converters.task_edit2_destinations import DefinitionRevisionConverter, REVISION
from wia.runtime.scene_capture_queue import read_json, write_json
from wia.runtime.commands.build_esi_core_revision import converter_for, destination, ASSETS

CORE_ROOT = Path('/home/zhouxinyang/esi_category_conversion/core_revision_20261018')
GAPS = {'placement_destination_pending': 'destination', 'named_placement_destination_required': 'destination',
        'observable_region_labels_pending': 'region_labels'}


def audit_current_gaps(core_path=CORE_ROOT / 'core_candidates_for_review.json',
                       generation_path=CORE_ROOT / 'candidate_generation.json'):
    """Reconcile full current Core membership by source, excluding backup-only rows."""
    core = read_json(core_path)['tasks']
    generation = read_json(generation_path)['records']
    source_rows = {}
    for row in generation:
        source_rows.setdefault(row['source_identity'], []).append(row)
    pending = []
    for member in core:
        rows = source_rows.get(member['source'], [])
        exact = [row for row in rows if row['candidate_id'] == member['candidate_id']]
        if len(exact) != 1:
            if member.get('protected_baseline') and not rows:
                continue
            raise ValueError('Current Core source/candidate membership is ambiguous: ' + member['source'])
        row = exact[0]
        kind = GAPS.get(row.get('failure_reason'))
        if kind is not None:
            pending.append({'source': member['source'], 'gap_kind': kind,
                            'historical_core_member': member, 'historical_generation_record': row})
    return {'core_membership_path': str(core_path), 'generation_path': str(generation_path),
            'core_membership_count': len(core), 'generation_record_count': len(generation),
            'pending_count': len(pending), 'gap_counts': dict(Counter(row['gap_kind'] for row in pending)),
            'pending': pending}


@lru_cache(maxsize=32)
def anchor_source_states(scene):
    path = ASSETS / 'scenes' / scene / 'json' / (scene + '_best.json')
    rows = read_json(path, {}).get('state', {}).get('registry', {}).get('object_registry', {})
    return {name: {'pos': row['root_link']['pos'], 'ori': row['root_link']['ori']}
            for name, row in rows.items() if 'pos' in row.get('root_link', {}) and 'ori' in row.get('root_link', {})}


def build_missing_definitions(output_dir, *, core_path=CORE_ROOT / 'core_candidates_for_review.json',
                              generation_path=CORE_ROOT / 'candidate_generation.json', seed=0):
    audit = audit_current_gaps(core_path, generation_path)
    records, source_mapping = [], []
    output_dir = Path(output_dir).resolve()
    for row in audit['pending']:
        identity = row['source']
        big, small = Path(identity).parts[:2]
        source = SourceInstance(WIA_ESI_ROOT / 'dataset/json_clean' / identity, identity, big, small)
        config = {'definition_revision': REVISION, 'task_revision': REVISION}
        if row['gap_kind'] == 'destination':
            config['placement_destination'] = destination(source, seed, task_edit2=True)
        else:
            config.update(task_edit2_global_region_labels=True,
                          region_anchor_source_states=anchor_source_states(source.raw['scene']))
        converter = DefinitionRevisionConverter(converter_for(source))
        root = output_dir / 'derivatives' / row['gap_kind'] / hashlib.sha256(identity.encode()).hexdigest()[:16]
        manifest = converter.convert_all([source], ConversionContext(root, seed=seed, config=config))
        record = manifest['records'][0]
        records.append(record)
        source_mapping.append({'source': identity, 'gap_kind': row['gap_kind'],
            'historical_candidate_id': row['historical_core_member']['candidate_id'],
            'derivative_candidate_id': record['candidate_id'], 'status': record['status'],
            'task_file': record.get('task_file'), 'scene_file': record.get('scene_file'),
            'failure_stage': record.get('failure_stage'), 'failure_reason': record.get('failure_reason'),
            'native_qualification_run': False})
    manifest = {'schema': 'task_edit2_missing_definitions_v1', 'revision': REVISION, 'seed': seed,
        'audit': {key: value for key, value in audit.items() if key != 'pending'},
        'records': records, 'source_mapping': source_mapping,
        'counts': dict(Counter(row['status'] for row in records)), 'native_qualification_run': False,
        'historical_sources_and_files_preserved': True, 'render_or_review_mutation': False}
    write_json(output_dir / 'missing_definitions_manifest.json', manifest)
    write_json(output_dir / 'source_to_derivative.json', {'revision': REVISION, 'sources': source_mapping})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--core-membership', type=Path, default=CORE_ROOT / 'core_candidates_for_review.json')
    parser.add_argument('--generation', type=Path, default=CORE_ROOT / 'candidate_generation.json')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    result = build_missing_definitions(args.output_dir, core_path=args.core_membership,
                                      generation_path=args.generation, seed=args.seed)
    print(json.dumps({'audit': result['audit'], 'counts': result['counts'],
                      'manifest': str(args.output_dir.resolve() / 'missing_definitions_manifest.json')}))


if __name__ == '__main__':
    main()
