"""Materialize the exact main-authored furnished collection round on CPU.

Run: /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B \
  scripts/design_wia_collection_round.py
Then check with scripts/build_wia_room_collection.py --design-file
explore_to_achieve/wia_v1/furnished_v1/collection_design_v1.json --output-dir
explore_to_achieve/wia_v1/furnished_v1/collection --check. Static room labels and
definition loaders are checked; native support, access and feasibility remain
pending capture. No native simulation or capture is run by this script.


Canonical invocation: python -m wia.authoring.commands.design_wia_collection_round
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

from wia.resources import ROOT

from wia.authoring.converters.wia_room_collection import ORDERED_OPERATION, SEARCH_OPERATION
from wia.authoring.commands.build_wia_room_collection import main as build_main
from wia.authoring.commands.design_wia_furnished_round import ROWS, INV, inventory, background, quat

OUT = ROOT / 'explore_to_achieve/wia_v1/furnished_v1'
ASSETS = Path('/local_home2/zhouxinyang/data/behavior/behavior-1k-assets')
IDENTITY = [0., 0., 0., 1.]

# Every asset, room, support, position and scale below was chosen by the main
# agent. Do not replace a failed slot with another room, support or coordinate.
NAVIGATION = {
    'T07': [
        ('living_room_0', 'banana', 'znakxm', 1., 'banana', 'breakfast_table_skczfi_0', [2.68, -7.44]),
        ('kitchen_0', 'marjoram_shaker', 'xlspnl', .8, 'small marjoram shaker', 'countertop_tpuwys_4', [-.7, -7.4]),
        ('corridor_0', 'cola_bottle', 'oyqdtz', .65, 'miniature cola bottle', 'wia_t07_corridor_open_shelf', [1.88, -4.3]),
        ('private_office_0', 'box_of_oatmeal', 'jtqeef', .5, 'miniature oatmeal box', 'bottom_cabinet_immwzb_0', [4.85, -4.74]),
        ('utility_room_0', 'pepper_grinder', 'tfoogy', 1., 'pepper grinder', 'bottom_cabinet_slgzfc_0', [4.35, .4]),
        ('storage_room_1', 'bottle_of_orange_juice', 'edltwh', .7, 'miniature orange juice bottle', 'wia_t07_storage_open_shelf', [2.65, -4.8]),
    ],
    'T08': [
        ('bathroom_0', 'bottle_of_lavender_oil', 'csalbx', 1., 'lavender oil bottle', 'countertop_zkhelh_0', [-24.7, -13.28]),
        ('shared_office_0', 'bottle_of_orange_juice', 'edltwh', .7, 'miniature orange juice bottle', 'bottom_cabinet_immwzb_0', [-18.4, -10.15]),
        ('corridor_0', 'cola_bottle', 'oyqdtz', .65, 'miniature cola bottle', 'wia_t08_corridor_open_shelf', [-9.5, 8.7]),
        ('conference_hall_0', 'banana', 'znakxm', 1., 'banana', 'wia_t08_conference_side_table', [-1., 12.]),
    ],
}
SEARCH = {
    'T35': {'scene': 'Merom_0_int', 'room': 'playroom_0', 'robot': [-.8, 6.3, .02],
            'category': 'banana', 'model': 'znakxm', 'scale': 1., 'public_name': 'banana',
            'areas': 'the playroom', 'targets': 'banana',
            'slots': [('countertop_tpuwys_0', [-2., 8.35], 'playroom_0'),
                      ('countertop_tpuwys_0', [-.6, 8.35], 'playroom_0'),
                      ('bookcase_owvfik_0', [-.035, 5.57], 'playroom_0')]},
    'T36': {'scene': 'Ihlen_0_int', 'room': 'living_room_0', 'robot': [-.2, 2.3, .02],
            'category': 'bottle_of_orange_juice', 'model': 'edltwh', 'scale': .7,
            'public_name': 'miniature orange juice bottle', 'areas': 'the living room and garage',
            'targets': 'miniature orange juice bottle',
            'slots': [('coffee_table_fqluyq_0', [.2, 3.47], 'living_room_0'),
                      ('coffee_table_qlmqyy_0', [-1.26, .05], 'living_room_0'),
                      ('bottom_cabinet_jrhgeu_0', [3.6, 4.9], 'garage_0')]},
    'T37': {'scene': 'office_cubicles_left', 'room': 'shared_office_0', 'robot': [-17.5, -11.5, .02],
            'category': 'cola_bottle', 'model': 'oyqdtz', 'scale': .65,
            'public_name': 'miniature cola bottle', 'areas': 'the shared office', 'targets': 'miniature cola bottle',
            'slots': [('bottom_cabinet_immwzb_0', [-18.4, -10.15], 'shared_office_0'),
                      ('bottom_cabinet_immwzb_1', [-15.52, -10.15], 'shared_office_0'),
                      ('bottom_cabinet_immwzb_3', [-16.83, -6.9], 'shared_office_0')]},
}


@lru_cache(maxsize=None)
def _static_map(scene, map_resolution=.1):
    """Mirror the explicitly selected native map resolution and room naming.

    Historical calls default to 0.1 m; those checks do not confirm the live
    native map, which uses 1.0 m for the original navigation configuration.
    """
    import cv2
    if (isinstance(map_resolution, bool) or not isinstance(map_resolution, (int, float))
            or not math.isfinite(map_resolution) or map_resolution <= 0):
        raise ValueError('Static map resolution must be an explicit finite positive number.')
    directory = ASSETS / 'scenes' / scene / 'layout'
    instance = cv2.imread(str(directory / 'floor_insseg_0.png'), cv2.IMREAD_GRAYSCALE)
    semantic = cv2.imread(str(directory / 'floor_semseg_0.png'), cv2.IMREAD_GRAYSCALE)
    if instance is None or semantic is None or instance.shape != semantic.shape or instance.shape[0] != instance.shape[1]:
        raise ValueError('Missing or inconsistent static native room maps: ' + scene)
    size = int(instance.shape[0] * .01 / map_resolution)
    instance = cv2.resize(instance, (size, size), interpolation=cv2.INTER_NEAREST)
    semantic = cv2.resize(semantic, (size, size), interpolation=cv2.INTER_NEAREST)
    categories = (ASSETS / 'metadata/room_categories.txt').read_text().splitlines()
    groups, names = {}, {}
    for value in np.unique(instance):
        if not value:
            continue
        index = np.argwhere(instance == value)[0]
        sem = int(semantic[tuple(index)])
        groups.setdefault(sem, []).append(int(value))
    for sem, instances in groups.items():
        for count, value in enumerate(instances):
            names[value] = categories[sem - 1] + '_' + str(count)
    return instance, names


def static_room(scene, xy, map_resolution=.1):
    instance, names = _static_map(scene, map_resolution)
    # collection.array supplies float64 to native world_to_map; truncate after
    # the same world-to-pixel transform. The default preserves historical 0.1 m.
    indices = (np.asarray(xy, dtype=np.float64) / float(map_resolution) + instance.shape[0] / 2.)[::-1].astype(int)
    if np.any(indices < 0) or np.any(indices >= instance.shape[0]):
        return None
    return names.get(int(instance[tuple(indices)]))


def _cube(name, position, scale, category):
    return {'type': 'PrimitiveObject', 'primitive_type': 'Cube', 'name': name, 'category': category,
            'position': position, 'orientation': deepcopy(IDENTITY), 'scale': scale,
            'fixed_base': True, 'visual_only': False, 'rgba': [.48, .32, .18, 1.]}


def _furniture(alias):
    if alias == 'T07':
        return [_cube('wia_t07_corridor_open_shelf', [1.88, -4.3, .74], [.45, .28, .04], 'shelf'),
                _cube('wia_t07_storage_open_shelf', [2.65, -4.8, .74], [.45, .28, .04], 'shelf')]
    if alias == 'T08':
        objects = [_cube('wia_t08_corridor_open_shelf', [-9.5, 8.7, .74], [.45, .28, .04], 'shelf'),
                   _cube('wia_t08_conference_side_table', [-1., 12., .74], [.7, .5, .04], 'coffee_table')]
        for index, (x, y) in enumerate(((-.31, -.21), (-.31, .21), (.31, -.21), (.31, .21))):
            objects.append(_cube('wia_t08_side_table_leg_' + str(index), [-1. + x, 12. + y, .36],
                                 [.04, .04, .72], 'coffee_table'))
        return objects
    return []


def _target(alias, index, scene, inv, furniture, spec, *, role='target', yaw=0.):
    room, category, model, scale, public_name, support, xy = spec
    measured = static_room(scene, xy)
    if measured != room:
        raise ValueError(f'Main-authored target XY fails static segmentation: {alias} {xy}: expected {room}, got {measured}')
    if support in inv:
        top = inv[support]['aabb'][1][2]
    else:
        match = [obj for obj in furniture if obj['name'] == support]
        if len(match) != 1:
            raise ValueError('Main-authored support is missing: ' + support)
        top = match[0]['position'][2] + match[0]['scale'][2] / 2
    row = {'name': f'wia_{alias.lower()}_{role}_{index:02d}', 'category': category, 'model': model,
           'scale': [scale] * 3, 'role': role, 'room': room, 'public_name': public_name,
           'support': support, 'position': [*xy, float(top) + .25], 'orientation': quat(yaw)}
    if support == 'bookcase_owvfik_0':
        row.update(support_surface_mode='local_raycast', support_alignment_mode='local_raycast')
    return row


def _entry(alias, scene, room, start, orientation, plate):
    parent = ROWS[alias]
    inv = inventory(scene)
    floor_room = static_room(scene, start[:2]) if room == 'full_scene' else room
    floors = [row['name'] for row in inv.values() if row['category'] == 'floors' and floor_room in row['rooms']]
    if not floors:
        raise ValueError('Main-authored starting room has no inventoried native floor: ' + scene + '/' + room)
    return {'document_alias': alias, 'variant_id': alias.lower() + '_furnished_collection_v1',
            'label': alias + ' furnished room collection', 'parent_task_file': parent['task_file'],
            'parent_scene_file': parent['scene_file'], 'scene_inventory_file': str(INV / (scene + '.json')),
            'room': room, 'layout': {'scene_model': scene, 'native_inventory': [], 'added_inventory': [],
                'support_objects': [], 'route': [], 'plate': deepcopy(plate),
                'robot_start': {'position': start, 'orientation': orientation},
                'source_objects_policy': 'replace_authored_source_objects',
                'scene_config': {'load_room_instances': None, 'load_room_types': None,
                                 'load_object_categories': None, 'not_load_object_categories': None},
                'public_environment': 'Original furnished rooms and doors, with objects on openly exposed supporting surfaces.'},
            'main_authored_layout': True, 'static_target_room_segmentation_checked': True,
            'native_validation_run': False, 'native_support_verified': False,
            'native_robot_feasibility_verified': False, 'physical_success_verified': False,
            'original_source_qa_unused': True}, inv, floors[0]


def build_design():
    """Render only the main-selected navigation and search layouts."""
    tray_parent = yaml.safe_load(Path(ROWS['T35']['task_file']).read_text())
    plate = tray_parent['private']['evaluation']['parameters']['plate']
    entries = []
    for alias, selected in NAVIGATION.items():
        parent_task = yaml.safe_load(Path(ROWS[alias]['task_file']).read_text())
        parameters = parent_task['private']['evaluation']['parameters']
        route = parameters['required_order']
        if route != [spec[0] for spec in selected]:
            raise ValueError('Main-authored inventory must preserve the original navigation route: ' + alias)
        scene = parent_task['source']['json'].split('/')[2]
        yaw = parameters['initial_yaw_rad']
        entry, inv, _ = _entry(alias, scene, 'full_scene', [*parameters['start_xy'], .02],
                               [0., 0., math.sin(yaw / 2), math.cos(yaw / 2)], plate)
        entry.update(operation=ORDERED_OPERATION, source_identity=parent_task['source']['json'])
        entry['layout'].update(route=deepcopy(route), allow_authored_navigation_inventory=True,
                               support_objects=_furniture(alias))
        entry['layout']['added_inventory'] = [_target(alias, index, scene, inv, entry['layout']['support_objects'], spec)
                                              for index, spec in enumerate(selected, 1)]
        entries.append(entry)
    for alias, group in SEARCH.items():
        entry, inv, floor = _entry(alias, group['scene'], group['room'], deepcopy(group['robot']), quat(90.), plate)
        entry.update(operation=SEARCH_OPERATION, source_identity=background(group['scene'], group['room'], floor))
        for index, (support, xy, room) in enumerate(group['slots'], 1):
            spec = (room, group['category'], group['model'], group['scale'], group['public_name'], support, xy)
            entry['layout']['added_inventory'].append(_target(alias, index, group['scene'], inv, [], spec,
                                                               yaw=90. if support == 'bookcase_owvfik_0' else 0.))
        constraints = ['Keep the tray in your right hand. Use your left hand to pick and place.',
                       'Collect every target category object you find in the declared search rooms.',
                       'The target count is unknown. Repeated views of one physical object count only once.']
        if alias == 'T37':
            distractor = ('shared_office_0', 'bottle_of_orange_juice', 'edltwh', .7,
                          'miniature orange juice bottle', 'bottom_cabinet_immwzb_2', [-15.39, -6.9])
            entry['layout']['added_inventory'].append(_target(alias, 1, group['scene'], inv, [], distractor, role='distractor'))
            constraints.append('Orange juice bottles are distractors. Do not leave them on the tray.')
        entry['public'] = {'objective': f"Search {group['areas']} and collect every {group['targets']} you find there. "
                                       'Use your left hand to place them on the tray held in your right hand.',
                           'success_criteria': 'Finish with every target physically resting on the tray, '
                                               'your left hand released, and the tray held steadily in your right hand.',
                           'environment': entry['layout']['public_environment'], 'constraints': constraints}
        entries.append(entry)
    return {'schema': 'wia_room_collection_design_v1', 'main_authored': True,
            'native_validation_run': False, 'physical_success_verified': False, 'entries': entries}


def main():
    design = build_design()
    design_path, output_dir = OUT / 'collection_design_v1.json', OUT / 'collection'
    data = (json.dumps(design, indent=2, allow_nan=False) + '\n').encode()
    if design_path.exists() and design_path.read_bytes() != data:
        raise ValueError('Refusing to overwrite a differing authored design: ' + str(design_path))
    if not design_path.exists():
        design_path.parent.mkdir(parents=True, exist_ok=True)
        design_path.write_bytes(data)
    return build_main(['--design-file', str(design_path), '--output-dir', str(output_dir)])


if __name__ == '__main__':
    raise SystemExit(main())
