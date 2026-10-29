"""Author produce, pot and cutting-board mirror variants on native furniture.

Run: behavior/bin/python -B scripts/design_wia_produce_mirror.py
Uses saved asset geometry for proposed layouts; native captures qualify setup.


Canonical invocation: python -m wia.authoring.commands.design_wia_produce_mirror
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from wia.resources import ROOT
from wia.authoring.commands._inputs import ARTIFACT_ROOT, LazyMapping, read_json_input
from wia.authoring.commands.design_wia_household_round import GROUPS, make_entry, TASKS
from wia.authoring.commands.design_wia_furnished_round import inventory, quat
from wia.authoring.commands.design_wia_review_revisions import dump, write, sha
from wia.authoring.commands.build_wia_dimensional import slot_order
from wia.tasks.loader import load_task, load_scene
from wia.tasks.wia_mirror_scoring import MirrorProgress

OUT = ROOT / 'explore_to_achieve/wia_v1/review_20261025/produce_mirror_v2'
META = LazyMapping(lambda: read_json_input(ARTIFACT_ROOT / 'wia_v1_20261025_revision/assets.json'))
OLD_META = LazyMapping(lambda: {k: {'collision_bounds_root_link_m': v['bounds']} for k, v in META.items() if 'bounds' in v})
SPECS = {
    'R-C1': ('B', [('apple', 'agveuv'), ('apple', 'agveuv'), ('lemon', 'bbvonm')]),
    'R-C2': ('O', [('carrot', 'qhmmmx'), ('carrot', 'qhmmmx'), ('cucumber', 'wcvwye'), ('cucumber', 'wcvwye')]),
    'R-C3': ('H', [('banana', 'znakxm')] * 3),
    'R-A1': ('O', [('bell_pepper', 'uqcenz'), ('bell_pepper', 'uqcenz'), ('bell_pepper', 'ukkycp')]),
    'R-A2': ('O', [('bell_pepper', 'uqcenz'), ('bell_pepper', 'uqcenz'), ('bell_pepper', 'ukkycp')]),
    'R-S1': ('B', [('apple', 'agveuv')] * 2),
    'R-S2': ('O', [('lemon', 'bbvonm')] * 2),
    'R-S3': ('H', [('cucumber', 'wcvwye')] * 2),
}
POT = ('stockpot', 'lfnbhc')
BOARD = ('cutting_board', 'tcdrzs')

def build(alias, seed):
    group_key, assets = SPECS[alias]
    group = GROUPS[group_key]
    n = len(assets)
    offsets = np.linspace(-.65, .65, n).tolist() if group_key == 'O' else np.linspace(-.5, .5, n).tolist() if group_key == 'B' else np.linspace(-.32, .32, n).tolist()
    TASKS[alias] = (group_key, *assets[0], [0] * n, offsets, 'produce item', 'produce items')
    entry = make_entry(alias, seed, OLD_META)
    task = yaml.safe_load(Path(entry['parent_task_file']).read_text())
    scene = yaml.safe_load(Path(entry['parent_scene_file']).read_text())
    layout = entry['layout']
    prior_designs = [ROOT / 'explore_to_achieve/wia_v1/furnished_v1/mirror_design_v2.json',
                    ROOT / 'explore_to_achieve/wia_v1/furnished_v1/mirror_design_v3.json']
    for path in prior_designs:
        for prior in json.loads(path.read_text())['entries']:
            if prior['document_alias'] == alias and ('seed' + str(seed)) in prior['variant_id']:
                if 'native_object_poses' in prior['layout']:
                    layout['native_object_poses'] = deepcopy(prior['layout']['native_object_poses'])
    inv = inventory(group['scene'])
    top = inv[group['support']]['aabb'][1][2]
    cx, cy = group['center']
    sx, sy = group['size']
    objects = [o for o in layout['objects'] if o['name'] not in {
        'wia_mirror_table', 'wia_mirror_pad_blue', 'wia_mirror_pad_yellow', 'wia_reference_beacon'}
        and not o['name'].startswith('wia_mirror_candidate_')]
    bindings = {k: v for k, v in entry['bindings'].items() if k in ('mirror', 'cue_low', 'cue_high', 'cue_top', 'cue_bottom')}
    spatial = alias.startswith('R-S')
    board_xy = [cx + sx / 2 - .24, cy - sy * .24]
    pot_xy = [cx + sx / 2 - .24, cy + sy * .24]
    board_scale = [1., 1., 1.]
    board_yaw = 90.
    if spatial:
        board_scale = [1.6, 1.6, 1.]
        board_xy = [cx - .08, cy]
        pot_xy = [cx - .08, cy + .68]
        if group_key == 'H':
            board_xy, pot_xy, board_yaw = [cx - .40, cy], [cx + .40, cy], 0.
    supported = []
    def asset_object(key, asset, xy, scale, yaw, fixed):
        category, model = asset
        bounds = np.asarray(META[category + '/' + model]['bounds'])
        q = quat(yaw)
        # All authored orientations here are yaw-only, so root Z uses native min Z.
        position = [*xy, top - bounds[0, 2] * scale[2] + .002]
        name = 'wia_produce_' + key
        objects.append({'type': 'DatasetObject', 'name': name, 'category': category, 'model': model,
            'scale': scale, 'position': position, 'orientation': q, 'fixed_base': fixed, 'visual_only': False})
        bindings[key] = {'name': name, 'category': category, 'model': model}
        supported.append({'name': name, 'support': group['support'],
                          'support_alignment_mode': 'local_raycast', 'support_surface_mode': 'local_raycast',
                          **({'support_contract': 'fixed_mount'} if fixed else {})})
        centroid = np.asarray(META[category + '/' + model]['centroid']) * scale
        return np.asarray(position) + Rotation.from_quat(q).apply(centroid)
    asset_object('pot', POT, pot_xy, [1., 1., 1.], 0., True)
    asset_object('board', BOARD, board_xy, board_scale, board_yaw, True)
    centers, candidate_assets = {}, {}
    order = slot_order(seed, n)
    for i, asset in enumerate(assets):
        key = 'candidate_' + str(i)
        yaw = [0., 90., 45.][i] if alias == 'R-C3' else 0.
        xy = [cx - sx / 2 + .22, cy + offsets[order[i]]]
        if spatial:
            if alias == 'R-S1':
                pair = [.18, .02]
            elif alias == 'R-S2':
                pair = [-.18, -.02]
            else:
                pair = [-.16, .16]
            xy = [board_xy[0], board_xy[1] + pair[i]]
            if group_key == 'H':
                xy = [board_xy[0] + pair[i], board_xy[1] + (-.055 if i == 0 else .055)]
            yaw = board_yaw
        centers[key] = asset_object(key, asset, xy, [1., 1., 1.], yaw, False)
        candidate_assets[key] = {'category': asset[0], 'model': asset[1], 'scale': [1., 1., 1.]}
        if spatial:
            board_top = top + (META['cutting_board/tcdrzs']['bounds'][1][2] - META['cutting_board/tcdrzs']['bounds'][0][2]) + .002
            objects[-1]['position'][2] += board_top - top
            centers[key][2] += board_top - top
            supported[-1] = {'name': bindings[key]['name'], 'support': bindings['board']['name']}
    selected = 'candidate_0' if spatial else min(centers, key=lambda k: centers[k][1]) if seed == 101 else max(centers, key=lambda k: centers[k][1])
    eye = np.asarray(layout['robot_start']['position']) + [.111264, -.013035, 1.35]
    plane_x = cx + sx / 2 + .06
    reflected = alias != 'R-A2'
    cue_x = plane_x - .012 if reflected else min(p[0] for p in centers.values()) - .08
    rays = {}
    for key, point in centers.items():
        virtual = point.copy()
        if reflected: virtual[0] = 2 * plane_x - virtual[0]
        rays[key] = eye + (cue_x - eye[0]) / (virtual[0] - eye[0]) * (virtual - eye)
    yvalues = sorted(point[1] for point in rays.values())
    width = min(.18, min(b-a for a, b in zip(yvalues, yvalues[1:])) * .55)
    y, z = rays[selected][1:]
    for key in ('cue_low', 'cue_high', 'cue_bottom', 'cue_top'):
        obj = next(o for o in objects if o['name'] == bindings[key]['name'])
        if key in ('cue_low', 'cue_high'):
            obj['position'] = [cue_x, float(y + (-1 if key == 'cue_low' else 1) * (width/2 + .006)), float(z)]
            obj['scale'] = [.008, .012, .404]
        else:
            obj['position'] = [cue_x, float(y), float(z + (-1 if key == 'cue_bottom' else 1) * .196)]
            obj['scale'] = [.008, float(width + .024), .012]
    layout.update(objects=objects, supported_items=supported)
    variant = alias.lower().replace('-', '_') + '_seed' + str(seed) + '_produce_v2'
    layout['shared_layout_id'] = ('appearance' if alias.startswith('R-A') else alias) + '_produce_seed' + str(seed)
    p = task['private']['evaluation']['parameters']
    p.pop('candidate_dimensions_m', None)
    p.update(revision='wia_mirror_v4', scene_contract={'scene_model': group['scene'], 'room': group['room']},
        candidate_assets=candidate_assets, candidates={k: bindings[k]['name'] for k in candidate_assets},
        table_key='board', anchor_key='pot', destinations={'blue': 'pot', 'yellow': 'board'},
        fixture_keys=[k for k in bindings if k not in candidate_assets],
        destination_contracts={
            'pot': {'kind': 'inside', 'usable_bounds_local_m': [[-.10, -.10, -.082], [.10, .10, .09]]},
            'board': {'kind': 'on_top', 'usable_bounds_local_m': [[-.19, -.13, -.011], [.19, .13, .011]]}},
        delivery_destination='yellow' if alias == 'R-C3' else 'blue', spatial_destination='yellow')
    MirrorProgress(p)
    task['private']['bindings'] = bindings
    task['private']['evaluation']['binding_keys'] = list(bindings)
    source_identity = entry['source_identity']
    init = deepcopy(task['scene']['variants']['adapted']['patches'][0]['parameters'])
    init.update(revision='wia_mirror_v4', scene_contract=p['scene_contract'], layout=layout)
    for k in ('table_key', 'destinations', 'fixture_keys', 'destination_contracts'):
        init[k] = deepcopy(p[k])
    patches = [{'operation': 'category_initialization', 'phase': phase,
                'module': 'wia.authoring.converters.wia_mirror', 'parameters': deepcopy(init)}
               for phase in ('before_construction', 'after_postprocessing')]
    task['scene'] = {'policy': 'native_then_minimal_patch_else_skip', 'variants': {'original': {'patches': []}, 'adapted': {'patches': patches}}}
    goal = 'Identify the real produce item corresponding to the reflection inside the red outline in the initial head-camera view. '
    if alias.startswith('R-C'):
        goal += 'Grasp that item and release it ' + ('on the cutting board.' if alias == 'R-C3' else 'inside the stockpot.')
    elif alias.startswith('R-A'):
        goal = ('Inspect the produce appearance inside the red outline in the initial head-camera view. Before grasping, submit ANSWER: reflected or ANSWER: direct. '
                'Deliver the corresponding real item inside the stockpot if reflected, or onto the cutting board if direct.')
    elif alias == 'R-S3':
        goal = ('Use the mirror to identify the two produce items and their initial distance order relative to the fixed stockpot. '
                'Move both items on the cutting board to reverse that order, with at least a 20 cm distance gap. The stockpot is the fixed reference; robot movement does not redefine it.')
    else:
        goal += ('Move that item on the cutting board so it is at least ' + ('20 cm farther from' if alias == 'R-S1' else '10 cm closer to')
                 + ' the fixed stockpot than the other item. Keep the other item in place.')
    task['public'] = {'objective': goal,
        'success_criteria': 'Use genuine grasp and at least 5 cm lift for every required item. Release all required items stably in the specified destination for at least 0.5 seconds, with linear speed at most 0.02 m/s and angular speed at most 0.1 rad/s. Empty and open the working hand, withdraw both hands, and declare DONE.',
        'environment': 'A furnished ' + group['room'].replace('_0', '').replace('_', ' ') + ' with vegetables or fruit on native furniture, a real mirror, a stockpot, and a cutting board.',
        'constraints': ['Keep the mirror, red outline, stockpot, cutting board and furniture fixed. Preserve item shapes and sizes.',
            'Use the initial head-camera view to interpret the red outline; later robot motion does not change the requested identity.',
            'Use guarded public controls and mounted observations. Unselected produce must remain within 2 cm of its initial position.']}
    if alias != 'R-S3': task['public']['constraints'].append('The first candidate grasp commits selection; do not grasp a different candidate.')
    task.update(id=('wia_produce_appearance_seed' + str(seed) if alias.startswith('R-A') else 'wia_' + variant),
                version=4, status='pending', source={'provider': 'esi', 'json': source_identity})
    task['repair'].update(document_alias=alias, variant_id=variant, contract_revision='wia_mirror_v4',
        parent_task_file=entry['parent_task_file'], parent_scene_file=entry['parent_scene_file'],
        native_validation_run=False, physical_success_verified=False, full_reference_execution='pending',
        scene_contract=p['scene_contract'], board_scale_explicit=board_scale,
        asset_measurement_file='outputs/wia_v1_20261025_revision/assets.json',
        design_scope='Produce and native pot/board destinations; fixed board/pot use explicit measured fixed mounts rather than dynamic support velocity. V1 failure and geometry retained.')
    scene.update(id='wia_' + variant + '_scene', version=4, status='pending', source=deepcopy(task['source']),
                 scene=deepcopy(task['scene']), bindings=deepcopy(bindings))
    paths = {kind: OUT / variant / (kind + '.yaml') for kind in ('task', 'scene')}
    for kind, value in (('task', task), ('scene', scene)):
        plain = json.loads(json.dumps(value, default=lambda x: x.item()))
        write(paths[kind], yaml.safe_dump(plain, sort_keys=False).encode())
    load_task(paths['task'], variant='adapted', run_mode='diagnostic')
    load_scene(paths['scene'], variant='adapted', run_mode='diagnostic')
    return {'document_alias': alias, 'variant_id': variant, 'label': alias + ' produce / pot and cutting board / seed ' + str(seed),
        'task_file': str(paths['task']), 'scene_file': str(paths['scene']), 'variant': 'adapted', 'seed': seed,
        'parent_task_file': entry['parent_task_file'], 'parent_scene_file': entry['parent_scene_file'],
        'native_validation_run': False, 'physical_success_verified': False}

if __name__ == '__main__':
    rows = [build(alias, seed) for alias in SPECS for seed in (101, 202)]
    dump(OUT / 'manifest.json', {'schema': 'wia_variant_manifest_v1', 'entries': rows})
    dump(OUT / 'pilot_manifest.json', {'schema': 'wia_variant_manifest_v1',
        'entries': [r for r in rows if r['document_alias'] in ('R-C2', 'R-A1', 'R-S1', 'R-S3') and r['seed'] == 101]})
