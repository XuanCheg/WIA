"""Materialize the main-authored furnished AP layouts without native execution.

Run: /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B
scripts/design_wia_existing_round.py
Then run scripts/build_wia_furnished_existing.py with the printed design/output
paths. Existing differing designs are never overwritten; no capture is started.


Canonical invocation: python -m wia.authoring.commands.design_wia_existing_round
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import yaml

from wia.resources import ROOT

from wia.tasks.sources import resolve_source
from wia.authoring.commands.design_wia_furnished_round import ROWS, inventory, background, quat, dump

OUT = ROOT / 'explore_to_achieve/wia_v1/furnished_v1'
GROUPS = {
    'K': {'scene': 'Beechwood_0_garden', 'room': 'kitchen_0',
          'robot': [-1.6, 14.98, .02], 'yaw': -90,
          'slots': [('countertop_tpuwys_0', [-1.15, 15.85]),
                    ('countertop_tpuwys_1', [-3.73, 14.0]),
                    ('countertop_tpuwys_2', [-2.5, 11.96]),
                    ('countertop_tpuwys_0', [-.35, 15.85]),
                    ('countertop_tpuwys_2', [-1.95, 11.96])]},
    'U': {'scene': 'house_single_floor', 'room': 'utility_room_0',
          'robot': [23.25, .15, .02], 'yaw': 0,
          'slots': [('countertop_ikwqer_0', [24.51, .15]),
                    ('countertop_fjkase_0', [21.94, .85]),
                    ('countertop_sjxber_0', [23.65, -2.05]),
                    ('countertop_ikwqer_0', [24.51, 1.35]),
                    ('countertop_fjkase_0', [21.94, -.25])]},
    'P': {'scene': 'Merom_0_garden', 'room': 'playroom_0',
          'robot': [-4.8, -11.45, .02], 'yaw': 180,
          'slots': [('bookcase_owvfik_0', [-3.02, -10.61]),
                    ('countertop_tpuwys_0', [-5.82, -12.72]),
                    ('countertop_tpuwys_0', [-5.82, -10.6])]},
}
ASSIGNMENTS = {'T01': 'K', 'T03': 'U', 'T19': 'P', 'T20': 'K', 'T21': 'U'}


def _records(values):
    if isinstance(values, dict):
        return values
    return {row.get('_key', row.get('name')): row for row in values}


def original_target_records(task):
    """Read exact source values without assuming missing roots or orientations."""
    evaluation = task['private']['evaluation']
    if evaluation['module'] == 'wia.authoring.converters.physical_dynamics_stacking_stability':
        return {row['name']: deepcopy(row) for row in evaluation['parameters']['original_objects'].values()}
    if evaluation['module'] != 'wia.authoring.converters.action_order':
        raise ValueError('The main-authored round contains only Action Order and Stacking tasks.')
    source = json.loads(resolve_source(task['source']['json']).read_text())
    metadata = source['metadata_json']
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    objects, poses = _records(metadata['objects']), _records(metadata['initial_poses'])
    result = {}
    for name, row in objects.items():
        pose = poses[name]
        result[name] = {'name': name, 'category': row['cat'], 'model': row['model'],
                        'scale': deepcopy(row['scale']), 'position': deepcopy(pose['position']),
                        'quaternion_xyzw': deepcopy(pose['quaternion_xyzw'])}
    return result


def build_design():
    entries = []
    for alias, group_key in ASSIGNMENTS.items():
        group = deepcopy(GROUPS[group_key])
        parent_task_path = Path(ROWS[alias]['task_file'])
        parent_scene_path = Path(ROWS[alias]['scene_file'])
        if alias == 'T01':
            parent_task_path = ROOT / 'explore_to_achieve/wia_v1/action_order_transfer_v2/t01_task.yaml'
            parent_scene_path = ROOT / 'explore_to_achieve/wia_v1/action_order_transfer_v2/t01_scene.yaml'
        task = yaml.safe_load(parent_task_path.read_text())
        inv = inventory(group['scene'])
        floors = [row['name'] for row in inv.values()
                  if row['category'] == 'floors' and group['room'] in row['rooms']]
        if not floors:
            raise ValueError('The selected room has no inventoried native floor: ' + group['room'])
        floor = floors[0]
        originals = original_target_records(task)
        targets = {key: binding for key, binding in task['private']['bindings'].items()
                   if binding.get('category') and binding['category'] not in ('floor', 'floors')
                   and key not in ('floor', 'source_floor')}
        if len(targets) > len(group['slots']):
            raise ValueError('The explicit main-authored slots do not cover all targets: ' + alias)
        objects, supported, references = [], [], {}
        for (key, binding), (support, xy) in zip(targets.items(), group['slots']):
            original = originals[binding['name']]
            if any(original[field] != binding[field] for field in ('name', 'category', 'model')):
                raise ValueError('Parent source inventory identity mismatch: ' + alias + ':' + key)
            objects.append({'type': 'DatasetObject', **deepcopy(binding),
                'position': [*xy, inv[support]['aabb'][1][2] + .3],
                'orientation': deepcopy(original['quaternion_xyzw']), 'scale': deepcopy(original['scale']),
                'fixed_base': False, 'visual_only': False})
            item = {'name': binding['name'], 'support': support}
            if support == 'bookcase_owvfik_0':
                item.update(support_surface_mode='local_raycast', support_alignment_mode='local_raycast')
            supported.append(item)
            references[binding['name']] = deepcopy(original['position'])
        layout = {'shared_layout_id': 'wia_furnished_existing_' + alias,
            'scene_model': group['scene'], 'rooms': [group['room']], 'floor_name': floor,
            'objects': objects, 'supported_items': supported,
            'robot_start': {'position': group['robot'], 'orientation': quat(group['yaw'])}}
        floor_updates = {key: floor for key, binding in task['private']['bindings'].items()
                         if key in ('floor', 'source_floor') or binding.get('category') in ('floor', 'floors')}
        entries.append({'document_alias': alias, 'variant_id': alias.lower() + '_furnished_existing_v1',
            'label': alias + ' open-furniture retrieval with retained parent goal',
            'parent_task_file': str(parent_task_path), 'parent_scene_file': str(parent_scene_path),
            'source_identity': background(group['scene'], group['room'], floor), 'layout': layout,
            'source_reference_positions': references, 'floor_binding_updates': floor_updates,
            'public': {'objective': 'Retrieve the items from exposed room furniture. ' + task['public']['objective'],
                'environment': 'A furnished ' + group['room'].rsplit('_', 1)[0].replace('_', ' ')
                    + ' with the task items openly dispersed on exposed native furniture surfaces. '
                    + 'The items are not deliberately hidden.'}})
    return {'schema': 'wia_furnished_existing_design_v1', 'entries': entries}


def main():
    design = build_design()
    destination = OUT / 'existing_design_v1.json'
    data = json.dumps(design, indent=2) + '\n'
    if destination.exists() and destination.read_text() != data:
        raise ValueError('Refusing to overwrite a differing existing design: ' + str(destination))
    if not destination.exists():
        dump(destination, design)
    print('Design: ' + str(destination))
    print('Output: ' + str(OUT / 'existing'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
