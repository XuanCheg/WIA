"""Materialize the main-specified household mirror round without native actions.

Run: behavior/bin/python -B scripts/design_wia_household_round.py [--check]
The layout and optical intersections are proposals from saved static bounds.
Native initialization images and geometry must qualify every authored variant.


Canonical invocation: python -m wia.authoring.commands.design_wia_household_round --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

import numpy as np
import yaml

from wia.resources import ROOT

from wia.authoring.commands.design_wia_furnished_round import INV, OUT, background, dump, inventory, quat
from wia.authoring.commands.build_wia_dimensional import slot_order
from wia.authoring.commands import build_wia_household_mirror as builder

PARENT = ROOT / 'explore_to_achieve/wia_v1/mirror_v1'
DESIGN_PATH = OUT / 'mirror_design_v1.json'
OUTPUT = OUT / 'household_mirror'
GROUPS = {
    'H': {'scene': 'house_single_floor', 'room': 'dining_room_0', 'support': 'breakfast_table_rhjoby_0',
          'center': [7.3, 3.5], 'size': [2.4, 1.0]},
    'B': {'scene': 'Beechwood_0_garden', 'room': 'dining_room_0', 'support': 'coffee_table_dnsjnv_0',
          'center': [-5.1, 16.7], 'size': [.95, 1.9]},
    'O': {'scene': 'office_cubicles_left', 'room': 'meeting_room_0', 'support': 'conference_table_fpkmyw_0',
          'center': [-7.0, 13.4], 'size': [1.2, 2.4]},
}
TASKS = {
    'R-C1': ('B', 'pepper_grinder', 'tfoogy', [0, 0, 0], [-.55, 0., .55], 'pepper grinder', 'pepper grinders'),
    'R-C2': ('O', 'cola_bottle', 'oyqdtz', [0, 0, 0, 0], [-.84, -.28, .28, .84], 'cola bottle', 'cola bottles'),
    'R-C3': ('H', 'banana', 'znakxm', [0, 90, 45], [-.32, 0., .32], 'banana', 'bananas'),
    'R-A1': ('O', 'marjoram_shaker', 'xlspnl', [0, 0, 0], [-.6, 0., .6], 'marjoram shaker', 'marjoram shakers'),
    'R-A2': ('O', 'marjoram_shaker', 'xlspnl', [0, 0, 0], [-.6, 0., .6], 'marjoram shaker', 'marjoram shakers'),
    'R-S1': ('B', 'box_of_oatmeal', 'jtqeef', [90, 90], [-.45, .45], 'oatmeal box', 'oatmeal boxes'),
    'R-S2': ('O', 'bottle_of_orange_juice', 'edltwh', [0, 0], [-.65, .65], 'orange juice bottle', 'orange juice bottles'),
    'R-S3': ('H', 'pepper_grinder', 'tfoogy', [0, 0], [-.32, .32], 'pepper grinder', 'pepper grinders'),
}


def _public(parent, singular, plural, group):
    public = deepcopy(parent['public'])
    def wording(value):
        if isinstance(value, list):
            return [wording(item) for item in value]
        if isinstance(value, dict):
            return {key: wording(item) for key, item in value.items()}
        if not isinstance(value, str):
            return value
        value = re.sub(r'\bgrey blocks\b', plural, value)
        value = re.sub(r'\bgrey block\b', singular, value)
        value = re.sub(r'\bblocks\b', plural, value)
        return re.sub(r'\bblock\b', singular, value)
    public = wording(public)
    room = group['room'].removesuffix('_0').replace('_', ' ')
    public['environment'] = ('A furnished ' + room + ' with ' + plural
        + ' on a fixed deck over an existing native table, a physical mirror, a red rectangular outline, '
          'blue and yellow delivery pads, and a fixed yellow reference pillar.')
    return public


def make_entry(alias, seed, model_metadata):
    group_key, category, model, yaws, offsets, singular, plural = TASKS[alias]
    group = deepcopy(GROUPS[group_key])
    inv = inventory(group['scene'])
    floor = next(row['name'] for row in inv.values()
                 if row['category'] == 'floors' and group['room'] in row['rooms'])
    support = inv[group['support']]
    top = support['aabb'][1][2]
    bounds = np.asarray(model_metadata[category + '/' + model]['collision_bounds_root_link_m'])
    height = float(bounds[1, 2] - bounds[0, 2])
    cx, cy = group['center']
    sx, sy = group['size']
    deck_top = top + .014
    candidate_x = cx - .45 if group_key == 'H' else cx - sx / 2 + .22
    plane_x = cx + sx / 2 + .06
    robot = [cx - sx / 2 - .7, cy, .11]
    eye = np.asarray(robot) + [.111264, -.013035, 1.35]
    order = slot_order(seed, len(offsets))
    centers = {f'candidate_{index}': [candidate_x, cy + offsets[order[index]], deck_top + height / 2]
               for index in range(len(offsets))}
    selected_key = min(centers, key=lambda key: centers[key][1]) if seed == 101 else max(centers, key=lambda key: centers[key][1])
    reflected = alias != 'R-A2'
    cue_x = plane_x - .012 if reflected else candidate_x - .08
    intersections = {}
    for key, center in centers.items():
        virtual = np.array(center)
        if reflected:
            virtual[0] = 2 * plane_x - virtual[0]
        ray = virtual - eye
        t = (cue_x - eye[0]) / ray[0]
        intersections[key] = (eye + t * ray).tolist()
    projected_y = sorted(point[1] for point in intersections.values())
    cue_width = min(.20, min(right - left for left, right in zip(projected_y, projected_y[1:])) * .55)
    cue_y, cue_z = intersections[selected_key][1:]
    inner_height, border, depth = .38, .012, .008
    variant_id = alias.lower().replace('-', '_') + '_seed' + str(seed) + '_furnished_v1'
    parent_stem = alias.lower().replace('-', '_') + '_seed' + str(seed) + '_v1'
    parent_task_file = PARENT / 'tasks' / (parent_stem + '.yaml')
    parent_scene_file = PARENT / 'scenes' / (parent_stem + '.yaml')
    parent = yaml.safe_load(parent_task_file.read_text())
    objects, bindings = [], {}

    def cube(key, dimensions, position, *, name=None, category_name='object', rgba=None):
        name = name or 'wia_mirror_' + key
        obj = {'type': 'PrimitiveObject', 'primitive_type': 'Cube', 'name': name, 'category': category_name,
            'size': 1., 'scale': dimensions, 'position': position, 'orientation': quat(0),
            'fixed_base': True, 'visual_only': False, 'rgba': rgba or [.55, .55, .55, 1.]}
        objects.append(obj)
        bindings[key] = {'name': name, 'category': category_name}

    cube('table', [sx, sy, .014], [cx, cy, top + .007], category_name='table', rgba=[.42, .38, .32, 1.])
    cube('mirror', [.025, sy + .12, 1.25], [plane_x + .0125, cy, top + .62],
         name='wia_mirror_panel', rgba=[.95, .95, .95, 1.])
    for color, sign, rgba in [('blue', -1, [.1, .3, .95, 1.]), ('yellow', 1, [.95, .8, .1, 1.])]:
        cube('pad_' + color, [.28, .28, .012], [cx + sx / 2 - .20, cy + sign * sy * .26, deck_top + .006],
             category_name='table', rgba=rgba)
    sign = -1 if seed == 101 else 1
    if alias == 'R-S2':
        sign *= -1
    anchor_y = cy + sign * (sy / 2 - .08) if alias in ('R-S1', 'R-S2') else cy + sy / 2 - .08
    cube('anchor', [.06, .06, .28], [candidate_x + .12, anchor_y, top + .14],
         name='wia_reference_beacon', rgba=[.95, .8, .1, 1.])
    for key, sign in [('cue_low', -1), ('cue_high', 1)]:
        cube(key, [depth, border, inner_height + 2 * border],
             [cue_x, cue_y + sign * (cue_width / 2 + border / 2), cue_z], rgba=[.95, .04, .03, 1.])
    for key, sign in [('cue_bottom', -1), ('cue_top', 1)]:
        cube(key, [depth, cue_width + 2 * border, border],
             [cue_x, cue_y, cue_z + sign * (inner_height / 2 + border / 2)], rgba=[.95, .04, .03, 1.])
    assets, supported = {}, []
    for index, yaw in enumerate(yaws):
        key, name = 'candidate_' + str(index), 'wia_mirror_candidate_' + str(index)
        asset = {'category': category, 'model': model, 'scale': [1., 1., 1.]}
        assets[key] = asset
        objects.append({'type': 'DatasetObject', 'name': name, **deepcopy(asset),
            'position': [candidate_x, centers[key][1], top + .3], 'orientation': quat(yaw),
            'fixed_base': False, 'visual_only': False})
        bindings[key] = {'name': name, 'category': category, 'model': model}
        supported.append({'name': name, 'support': bindings['table']['name']})
    public = _public(parent, singular, plural, group)
    public_id = ('wia_household_appearance_delivery_seed' + str(seed) + '_v1' if alias in ('R-A1', 'R-A2')
                 else 'wia_household_' + alias.lower().replace('-', '_') + '_seed' + str(seed) + '_v1')
    shared = ('appearance' if alias in ('R-A1', 'R-A2') else alias.lower().replace('-', '_'))
    layout = {'shared_layout_id': 'wia_furnished_mirror_' + shared + '_seed' + str(seed),
        'scene_model': group['scene'], 'rooms': [group['room']], 'floor_name': floor,
        'objects': objects, 'supported_items': supported,
        'robot_start': {'position': robot, 'orientation': quat(0)}}
    return {'document_alias': alias, 'variant_id': variant_id, 'label': alias + ' household mirror seed ' + str(seed),
        'parent_task_file': str(parent_task_file), 'parent_scene_file': str(parent_scene_file),
        'source_identity': background(group['scene'], group['room'], floor),
        'scene_model': group['scene'], 'room': group['room'], 'layout': layout, 'bindings': bindings,
        'candidate_assets': assets, 'public_goal': public['objective'], 'public': public,
        'task_id': public_id, 'scene_id': 'wia_' + variant_id + '_scene',
        'geometry_proposal': {'status': 'pending_native_initialization_review',
            'native_measurement': False, 'support_name': group['support'], 'static_tabletop_world_z_m': top,
            'static_asset_collision_bounds_root_link_m': bounds.tolist(), 'candidate_slot_permutation': order,
            'candidate_center_proposals_m': centers, 'selected_outer_slot_proposal': selected_key,
            'approximate_initial_head_eye_proposal_m': eye.tolist(), 'cue_ray_intersections_proposal_m': intersections,
            'cue_center_proposal_m': [cue_x, cue_y, cue_z], 'cue_inner_width_proposal_m': cue_width,
            'scope': 'Saved static collision bounds, bounding-center height and approximate head eye; actual native collision centroid, projection, support and optics must qualify.'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    metadata = json.loads((INV / 'asset_metadata.json').read_text())
    design = {'schema': 'wia_household_mirror_design_v1', 'author': 'main_agent',
        'status': 'pending_native_initialization_review',
        'scientific_configuration': 'Original mirror operations, scoring, seeds and budgets preserved; household assets and furnished layouts explicitly authored by main.',
        'entries': [make_entry(alias, seed, metadata) for alias in TASKS for seed in (101, 202)]}
    if args.check:
        if not DESIGN_PATH.is_file() or json.loads(DESIGN_PATH.read_text()) != design:
            raise ValueError('The authored household mirror design differs or is missing.')
    else:
        if DESIGN_PATH.exists() and json.loads(DESIGN_PATH.read_text()) != design:
            raise ValueError('Refusing to overwrite a differing household mirror design.')
        if not DESIGN_PATH.exists():
            dump(DESIGN_PATH, design)
    builder_args = ['--design-file', str(DESIGN_PATH), '--output-dir', str(OUTPUT)]
    return builder.main(builder_args + (['--check'] if args.check else []))


if __name__ == '__main__':
    raise SystemExit(main())
