"""Materialize the main-authored 32-world furnished temporal control round.

Run: behavior/bin/python -B scripts/design_wia_temporal_round.py [--check]
Transforms and primary review roles are explicit main decisions. Source states,
assets and temporal interventions remain inherited; native controls are unverified.


Canonical invocation: python -m wia.authoring.commands.design_wia_temporal_round --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from wia.resources import ROOT

from wia.authoring.commands.design_wia_furnished_round import OUT, ROWS, dump, inventory, quat
from wia.authoring.commands import build_wia_furnished_temporal as builder

DESIGN_PATH = OUT / 'temporal_design_v1.json'
OUTPUT = OUT / 'temporal'
BASE_ORIGIN = [7.2954, 3.5004, 0.]
BASE_TABLETOP = .8126344680786133
CONFIGURATIONS = {
    'H': {'scene': 'house_single_floor', 'room': 'dining_room_0', 'support': 'breakfast_table_rhjoby_0',
          'center': [7.2954, 3.5004], 'yaw': 0., 'robot': [9.4, 2.7, .02], 'robot_yaw': 160.},
    'B': {'scene': 'Beechwood_0_garden', 'room': 'dining_room_0', 'support': 'coffee_table_dnsjnv_0',
          'center': [-5.1, 16.7], 'yaw': 90., 'robot': [-3.8, 17.3, .02], 'robot_yaw': 180.},
    'M': {'scene': 'Merom_0_garden', 'room': 'utility_room_0', 'support': 'breakfast_table_skczfi_0',
          'center': [4.95, -12.78], 'yaw': 0., 'robot': [4.8, -11.5, .02], 'robot_yaw': -90.},
    'O': {'scene': 'office_cubicles_left', 'room': 'meeting_room_0', 'support': 'conference_table_fpkmyw_0',
          'center': [-7., 13.4], 'yaw': 90., 'robot': [-5.1, 13.8, .02], 'robot_yaw': 180.},
}
PRIMARY = {'T40': ('H', 'a'), 'T41': ('B', None), 'T45': ('B', None),
           'T46': ('M', None), 'T47': ('O', None), 'T48': ('O', None)}


def make_entry(configuration, alias, history=None):
    group = CONFIGURATIONS[configuration]
    inv = inventory(group['scene'])
    support = inv[group['support']]
    floor = next(row['name'] for row in inv.values()
                 if row['category'] == 'floors' and group['room'] in row['rooms'])
    if alias == 'T40':
        base = ROOT / 'explore_to_achieve/wia_v1/temporal_start_v3'
        parent_task = base / 'tasks' / ('history_' + history + '_v3.yaml')
        parent_scene = base / 'scenes' / ('history_' + history + '_v3.yaml')
    else:
        parent_task, parent_scene = (Path(ROWS[alias][kind + '_file']) for kind in ('task', 'scene'))
    variant = alias.lower() + ('_history_' + history if history is not None else '')
    variant += '_' + configuration.lower() + '_furnished_v1'
    transform = {'origin_m': deepcopy(BASE_ORIGIN),
        'translation_m': [group['center'][0] - BASE_ORIGIN[0], group['center'][1] - BASE_ORIGIN[1],
                          support['aabb'][1][2] - BASE_TABLETOP], 'yaw_deg': group['yaw']}
    return {'document_alias': alias, 'variant_id': variant,
        'label': alias + (' history ' + history.upper() if history is not None else '') + ' furnished ' + configuration,
        'parent_task_file': str(parent_task), 'parent_scene_file': str(parent_scene),
        'output_task_file': str(OUTPUT / 'tasks' / (variant + '.yaml')),
        'output_scene_file': str(OUTPUT / 'scenes' / (variant + '.yaml')),
        'source_identity': 'Temporal Understanding/Unobserved Change/' + group['scene'] + '/' + group['room'] + '/' + variant + '.json',
        'scene_model': group['scene'], 'room': group['room'], 'floor_name': floor,
        'transform': transform, 'support_name_map': {'breakfast_table_rhjoby_0': group['support']},
        'robot_start': {'position': deepcopy(group['robot']), 'orientation': quat(group['robot_yaw'])},
        'apparatus_floor_z_m': 0., 'matched_group': 'temporal_furnished_' + configuration.lower(),
        'environment': ('A furnished ' + group['room'].removesuffix('_0').replace('_', ' ')
                        + ' with the original temporal task objects on an existing native table and the inherited physical apparatus.'),
        'configuration': configuration, 'history_variant': history,
        'review_role': 'primary' if PRIMARY[alias] == (configuration, history) else 'control',
        'native_rgb_equivalence_verified': False,
        'support_geometry_proposal': {'support_name': group['support'], 'static_aabb_m': deepcopy(support['aabb']),
            'bounds_source': support['bounds_source'], 'native_measurement': False,
            'source_geometry_and_event_semantics_preserved': True}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    entries = []
    for configuration in CONFIGURATIONS:
        entries.extend(make_entry(configuration, 'T40', history) for history in 'abc')
        entries.extend(make_entry(configuration, alias) for alias in ('T41', 'T45', 'T46', 'T47', 'T48'))
    design = {'schema': 'wia_furnished_temporal_design_v1', 'author': 'main_agent',
        'status': 'pending_native_initialization_review',
        'hypothesis': 'Cross-scene controls preserve each temporal recipe and the matched A/B/C intervention while changing only the explicitly authored furnished world placement and robot start.',
        'primary_review_selection': {alias: {'configuration': group, 'history_variant': history}
                                     for alias, (group, history) in PRIMARY.items()},
        'native_rgb_equivalence_verified': False, 'entries': entries}
    if args.check:
        if not DESIGN_PATH.is_file() or json.loads(DESIGN_PATH.read_text()) != design:
            raise ValueError('The main-authored furnished temporal design differs or is missing.')
    else:
        if DESIGN_PATH.exists() and json.loads(DESIGN_PATH.read_text()) != design:
            raise ValueError('Refusing to overwrite a differing furnished temporal design.')
        if not DESIGN_PATH.exists():
            dump(DESIGN_PATH, design)
    commands = ['--design-file', str(DESIGN_PATH), '--manifest-output', str(OUTPUT / 'manifest.json')]
    return builder.main(commands + (['--check'] if args.check else []))


if __name__ == '__main__':
    raise SystemExit(main())
