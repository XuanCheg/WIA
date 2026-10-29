#!/usr/bin/env python3
"""Prepare a source shortlist and versioned repaired definitions without rendering.

Run: python src/wia/runtime/commands/build_esi_core_revision.py shortlist --output-dir OUTPUT
     python src/wia/runtime/commands/build_esi_core_revision.py generate --output-dir OUTPUT
Historical tasks, approvals and source JSON are never overwritten.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

import yaml

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
from wia.runtime.scene_capture_queue import read_json, write_json, digest
from wia.authoring.conversion import ConversionContext, SourceInstance, get_converter

CAMPAIGN = Path('/home/zhouxinyang/esi_category_conversion/20261014_v2')
OLD = CAMPAIGN / 'review_recapture_20261016'
from wia.authoring.scene_inventory import (
    ASSETS, scene_inventory as _scene_inventory,
    scene_object_positions as _scene_object_positions, destination as _destination,
)
REVISION = 'task_edit_20261018_v1'


def members(record):
    evaluation = record.get('goal_predicate') or {}
    children = evaluation.get('parameters', {}).get('members', [])
    return children or [{'source_identity': record['source_identity'], 'evaluation': evaluation}]


@lru_cache(maxsize=64)
def scene_inventory(scene):
    """Forward to canonical inventory logic using this script's configured assets."""
    return _scene_inventory(scene, assets=ASSETS)


@lru_cache(maxsize=64)
def scene_object_positions(scene):
    """Forward to canonical inventory logic using this script's configured assets."""
    return _scene_object_positions(scene, assets=ASSETS)


def destination(source, seed, *, task_edit2=False):
    """Forward with local helpers so historical monkeypatches retain their scope."""
    return _destination(source, seed, task_edit2=task_edit2,
                        _inventory=scene_inventory, _positions=scene_object_positions)


@lru_cache(maxsize=64)
def usable_map_screen(scene):
    """Conservative 2D clearance screen, explicitly not a native route proof."""
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    path = ASSETS / 'scenes' / scene / 'layout/floor_trav_0.png'
    free = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if free is None:
        return {'available': False, 'native_verified': False}
    distance = cv2.distanceTransform(np.pad((free > 0).astype(np.uint8), 1),
        cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1] * .01
    cleared = (distance >= .42).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(cleared, connectivity=8)
    areas = sorted((stats[1:, cv2.CC_STAT_AREA] * .0001).tolist(), reverse=True)
    return {'available': True, 'largest_clear_component_m2': areas[0] if areas else 0.,
        'clear_area_m2': float(cleared.sum() * .0001), 'clearance_radius_m': .42,
        'map_resolution_m': .01, 'native_verified': False,
        'limitation': 'Static whole-floor map excludes live doors, height-dependent obstacles and actual task paths.'}


def shortlist(args):
    records = read_json(args.records)['records']
    reviews = read_json(OLD / 'gallery/reviews.json')['entries']
    by_source = defaultdict(list)
    for key, row in reviews.items():
        by_source[row.get('source')].append({'id': key, **row})
    groups = defaultdict(list)
    index = []
    for record in records:
        ids = [m['source_identity'] for m in members(record)]
        history = [v for source in ids for v in by_source.get(source, [])]
        approved = any(row['decision'] == 'approved' for row in history)
        scene = record.get('scene_model') or Path(record['source_identity']).parts[2]
        row = {'candidate_id': record['candidate_id'], 'source_identity': record['source_identity'],
            'member_sources': ids, 'category': [record['big_task'], record['small_task']],
            'scene_model': scene, 'room': record.get('room') or Path(record['source_identity']).parts[3],
            'status': record['status'], 'historical_layout_approved': approved,
            'prior_feedback': history, 'protected_rigid': record['small_task'] == 'Rigid Containment',
            'source_operation': (record.get('goal_predicate') or {}).get('operation'),
            'semantic_child_operations': sorted({m.get('evaluation', {}).get('operation', 'legacy') for m in members(record)}),
            'space_screen': usable_map_screen(scene), 'release_status': 'unverified',
            'old_scene_file': record.get('scene_file'), 'old_task_file': record.get('task_file')}
        index.append(row)
        groups[tuple(row['category'])].append(row)
    chosen, backups = [], []
    for category, rows in sorted(groups.items()):
        eligible = [r for r in rows if r['status'] == 'generated'] or rows
        eligible.sort(key=lambda r: (not r['historical_layout_approved'],
            -r['space_screen'].get('largest_clear_component_m2', 0.), r['candidate_id']))
        selected, seen = [], set()
        limit = len(eligible) if category[1] == 'Rigid Containment' else 4
        for row in eligible:
            branch = (row['scene_model'], row['room'], tuple(row['semantic_child_operations']))
            if branch in seen and category[1] != 'Rigid Containment':
                continue
            seen.add(branch)
            selected.append(row)
            if len(selected) >= limit:
                break
        # First pool is deliberately provisional; repaired semantic families and
        # native evidence, rather than a fixed quota, govern the final release.
        for row in selected:
            row = {**row, 'selection': 'core_candidate', 'selection_reason':
                'Protected Rigid baseline retained unchanged.' if row['protected_rigid'] else
                'Provisional mechanism/world diversity, prior layout feedback and static usable-space screen.'}
            chosen.append(row)
        selected_ids = {r['candidate_id'] for r in selected}
        backup_scenes = set()
        for row in eligible:
            if row['candidate_id'] in selected_ids or row['scene_model'] in backup_scenes:
                continue
            backup_scenes.add(row['scene_model'])
            backups.append({**row, 'selection': 'backup', 'selection_reason': 'Alternative source world for failed qualification.'})
            if len(backup_scenes) == 2:
                break
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'source_inventory.json', {'schema': 'core_source_inventory_v1', 'records': index})
    write_json(out / 'source_shortlist.json', {'schema': 'provisional_core_sources_v1',
        'revision': REVISION, 'seed': args.seed, 'frozen': False, 'core_candidates': chosen,
        'backups': backups, 'coverage': dict(Counter('/'.join(r['category']) for r in chosen)),
        'release_count': 0, 'native_qualification_run': False,
        'selection_does_not_use_model_performance': True})
    print(json.dumps({'inventory': len(index), 'core_candidates': len(chosen), 'backups': len(backups),
        'subclasses': len(groups), 'native_qualified': 0}), flush=True)


def converter_for(source):
    converter = get_converter(source.big_task, source.small_task)
    if converter is not None:
        return converter
    slug = re.sub(r'[^a-z0-9]+', '_', source.big_task.lower() + '_' + source.small_task.lower()).strip('_')
    module = importlib.import_module('wia.authoring.converters.' + slug)
    return module.CategoryConverter()


def generate(args):
    shortlist_data = read_json(args.output_dir / 'source_shortlist.json')
    inventory = read_json(args.output_dir / 'source_inventory.json')['records']
    rows = shortlist_data['core_candidates'] + shortlist_data['backups'] if args.candidates_only else inventory
    selected_sources = {source for row in rows for source in row['member_sources']}
    originals = {r['candidate_id']: r for r in read_json(args.records)['records']}
    output, seen = [], set()
    for row in rows:
        if row['protected_rigid']:
            output.append({**originals[row['candidate_id']], 'core_revision': REVISION,
                'protected_rigid': True, 'repaired_definition': False, 'preservation': 'byte_identical_source_paths'})
            continue
        for identity in row['member_sources']:
            if identity in seen:
                continue
            seen.add(identity)
            big, small = Path(identity).parts[:2]
            source = SourceInstance(WIA_ESI_ROOT / 'dataset/json_clean' / identity, identity, big, small)
            config = {'task_revision': REVISION}
            if big == 'Specular Reflection':
                # All 300 source records use a native observer height of 0.8 m.
                # Four native pilots verify this proposal's head pose/support;
                # each released instance still needs its own optical/motion proof.
                config['reflection_robot_uniform_scale'] = .5
            dest = destination(source, args.seed)
            if dest is not None:
                config['placement_destination'] = dest
            converter = converter_for(source)
            batch_dir = args.output_dir / 'generated' / re.sub('[^a-z0-9]+', '_', small.lower()) / digest(identity)[:16]
            manifest = converter.convert_all([source], ConversionContext(batch_dir, seed=args.seed, config=config))
            for record in manifest['records']:
                record.update(core_revision=REVISION, historical_parent_candidate=row['candidate_id'],
                    prior_feedback=row['prior_feedback'], historical_layout_approved=row['historical_layout_approved'],
                    native_qualification_run=False)
                output.append(record)
            print(json.dumps({'source': identity, 'status': manifest['records'][0]['status']}), flush=True)
    write_json(args.output_dir / ('candidate_generation.json' if args.candidates_only else 'full_generation.json'),
        {'schema': 'core_revision_generation_v1', 'revision': REVISION, 'seed': args.seed,
            'records': output, 'counts': dict(Counter(r['status'] for r in output)),
            'source_count': len(selected_sources), 'native_qualification_run': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('shortlist', 'generate'))
    parser.add_argument('--records', type=Path, default=OLD / 'records.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--candidates-only', action='store_true')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    (shortlist if args.command == 'shortlist' else generate)(args)


if __name__ == '__main__':
    main()
