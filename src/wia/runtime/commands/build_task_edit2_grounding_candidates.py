#!/usr/bin/env python3
"""Generate only current Core viewpoint-before-retrieval grounding derivatives.

Run: python src/wia/runtime/commands/build_task_edit2_grounding_candidates.py --output-dir outputs/task_edit2_20261018/grounding
Optional --occluder-resting-supports supplies explicit per-source bound safe-rest
proposals; no simulator, rendering, queue or Review mutation is performed.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import json
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.authoring.conversion import SourceInstance, ConversionContext
from wia.authoring.converters.partial_occlusion import GROUNDING_REVISION
from wia.authoring.converters.task_edit2_destinations import DefinitionRevisionConverter, destination_proposal
from wia.runtime.scene_capture_queue import read_json, write_json
from wia.runtime.commands.build_esi_core_revision import converter_for

CORE = Path('/home/zhouxinyang/esi_category_conversion/core_revision_20261018/core_candidates_for_review.json')


def build_candidates(output_dir, *, core_path=CORE, seed=0, resting_supports=None):
    rows = [row for row in read_json(core_path)['tasks'] if row['category'] == 'Perceptual Grounding'
            and row['subcategory'] in ('Partial Occlusion', 'View Hallucination')]
    records, mappings = [], []
    for row in rows:
        identity = row['source']
        source = SourceInstance(WIA_ESI_ROOT / 'dataset/json_clean' / identity, identity,
                                row['category'], row['subcategory'])
        recipes = [('viewpoint', None)]
        if identity in (resting_supports or {}):
            if row['subcategory'] != 'Partial Occlusion':
                raise ValueError('Only a bound original Partial Occlusion blocker may use the move_occluder recipe.')
            recipes.append(('move_occluder', resting_supports[identity]))
        for recipe, rest in recipes:
            config = {'task_edit2_grounding': True, 'grounding_recipe': recipe,
                      'placement_destination': destination_proposal(source, seed)}
            if rest is not None:
                config['occluder_resting_support'] = rest
            converter = DefinitionRevisionConverter(converter_for(source))
            root = Path(output_dir) / 'derivatives' / hashlib.sha256(identity.encode()).hexdigest()[:16] / recipe
            result = converter.convert_all([source], ConversionContext(root.resolve(), seed=seed, config=config))
            record = result['records'][0]
            records.append(record)
            mappings.append({'source': identity, 'source_core_candidate_id': row['candidate_id'],
                'derivative_candidate_id': record['candidate_id'], 'recipe': recipe, 'status': record['status'],
                'task_file': record.get('task_file'), 'scene_file': record.get('scene_file'),
                'failure_stage': record.get('failure_stage'), 'failure_reason': record.get('failure_reason'),
                'native_qualification_run': False})
    manifest = {'schema': 'task_edit2_grounding_candidates_v1', 'grounding_revision': GROUNDING_REVISION,
        'core_membership': str(core_path), 'core_grounding_source_count': len(rows), 'seed': seed,
        'source_mapping': mappings, 'records': records, 'counts': dict(Counter(row['status'] for row in records)),
        'native_qualification_run': False, 'queue_or_review_mutation': False}
    write_json(Path(output_dir) / 'grounding_manifest.json', manifest)
    write_json(Path(output_dir) / 'source_to_derivative.json', {'revision': GROUNDING_REVISION, 'sources': mappings})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--core-membership', type=Path, default=CORE)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--occluder-resting-supports', type=Path)
    args = parser.parse_args()
    result = build_candidates(args.output_dir.resolve(), core_path=args.core_membership, seed=args.seed,
                              resting_supports=read_json(args.occluder_resting_supports) if args.occluder_resting_supports else None)
    print(json.dumps({'source_count': result['core_grounding_source_count'], 'counts': result['counts'],
                      'manifest': str(args.output_dir.resolve() / 'grounding_manifest.json')}))


if __name__ == '__main__':
    main()
