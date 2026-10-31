"""Build four compact household geometry definitions without simulator actions.

Run from WIA with behavior Python: PYTHONPATH=src python -B
scripts/build_household_geometry_20261031.py [--check]. The saved asset bounds
are CPU USD measurements; initialization, access and physical success are separate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from wia.resources import ROOT
from wia.tasks.loader import load_scene, load_task


DEST = ROOT / 'explore_to_achieve/candidates/household_geometry_20261031'
ASSETS = DEST / 'asset_bounds_cpu.json'
MODULE = 'wia.authoring.converters.wia_household_fit'
REVISION = 'wia_household_fit_v1'
SCENE, ROOM, FLOOR = 'restaurant_hotel', 'kitchen_0', 'floors_ykbuit_0'
TABLE = 'commercial_kitchen_table_bwlfnu_0'
TOP = .893
COLORS = [[.12, .38, .90, 1.], [.95, .75, .15, 1.], [.65, .28, .80, 1.]]


def build(alias, assets):
    """Author one explicit task; all coordinates are initialization proposals."""
    stem = alias.lower().replace('-', '_')
    objects, bindings, supported, asset_specs, dimensions = [], {}, [], {}, {}

    def cube(key, size, center, color=None):
        name = f'hg_{stem}_{key}'
        objects.append(dict(type='PrimitiveObject', primitive_type='Cube', name=name,
            category='table' if 'floor' in key or key == 'deck' else 'object', size=1.,
            scale=list(size), position=list(center), orientation=[0., 0., 0., 1.],
            fixed_base=True, visual_only=False, rgba=color or [.55, .51, .44, 1.]))
        bindings[key] = {'name': name}
        return key

    def target(key, category, model, scale, center_xy, angles=(0., 0., 0.), support='deck'):
        metadata = assets[category + '/' + model]
        rotation = Rotation.from_euler('xyz', angles, degrees=True)
        bounds = np.asarray(metadata['bounds_root_m']) * scale
        offset = rotation.apply(bounds.mean(0))
        center = np.array([*center_xy, TOP + .25])
        name = f'hg_{stem}_{key}'
        asset = dict(category=category, model=model, scale=[float(scale)] * 3)
        objects.append(dict(type='DatasetObject', name=name, **asset,
            position=(center - offset).tolist(), orientation=rotation.as_quat().tolist(),
            fixed_base=False, visual_only=False))
        bindings[key] = dict(name=name, category=category, model=model)
        supported.append(dict(name=name, support=bindings[support]['name']))
        asset_specs[key] = asset
        dimensions[key] = (np.asarray(metadata['dimensions_m']) * scale).tolist()

    def compartment(index, x0, x1, cy, width, height, *, closed_top, mouth=False):
        # All widths are usable inner dimensions, measured again from native faces.
        prefix = f'bin_{index}'
        floor = cube(prefix + '_floor', [x1 - x0 + .012, width + .024, .008],
                     [(x0 + x1) / 2, cy, TOP + .004], COLORS[index])
        bottom = TOP + .008
        left = cube(prefix + '_left', [x1 - x0 + .012, .012, height],
                    [(x0 + x1) / 2, cy - width / 2 - .006, bottom + height / 2])
        right = cube(prefix + '_right', [x1 - x0 + .012, .012, height],
                     [(x0 + x1) / 2, cy + width / 2 + .006, bottom + height / 2])
        back = cube(prefix + '_back', [.012, width + .024, height],
                    [x1 + .006, cy, bottom + height / 2])
        interior = dict(floor=floor, left=left, right=right, back=back)
        if closed_top:
            interior['roof'] = cube(prefix + '_roof', [x1 - x0 + .024, width + .024, .012],
                                    [(x0 + x1) / 2, cy, bottom + height + .006])
        aperture = None
        if mouth:
            entry, exit = x0 - .018, x0 - .006
            lower = cube(prefix + '_mouth_left', [.012, .012, height],
                         [(entry + exit) / 2, cy - width / 2 - .006, bottom + height / 2])
            upper = cube(prefix + '_mouth_right', [.012, .012, height],
                         [(entry + exit) / 2, cy + width / 2 + .006, bottom + height / 2])
            roof = cube(prefix + '_mouth_roof', [.012, width + .024, .012],
                        [(entry + exit) / 2, cy, bottom + height + .006])
            # A separate thin entry deck reaches the mouth and is coplanar with the bin floor.
            mouth_floor = cube(prefix + '_mouth_floor', [.030, width + .024, .008],
                               [x0 - .009, cy, TOP + .004], COLORS[index])
            aperture = dict(lower_side=lower, upper_side=upper, roof=roof, floor=mouth_floor)
        return floor, interior, aperture

    cube('deck', [.450, 1.60, .016], [4.46, -.088, TOP - .008], [.45, .42, .36, 1.])
    bindings['native_support'] = {'name': TABLE}
    supported.append(dict(name=bindings['deck']['name'], support=TABLE,
        support_contract='fixed_mount', support_surface_mode='local_raycast'))
    destinations, interiors, upright, aperture = [], {}, {}, None
    operation = 'select_pass_aperture' if alias.startswith('GF2') else 'sort_intrinsic_dimension'
    if alias == 'GF2-1':
        title = 'Lay a juice carton into a low pantry cubby'
        # Keep the upright carton above the 87 mm mouth while leaving gripper clearance after pitching.
        scale = .105 / assets['pineapple_juice_carton/barzwx']['dimensions_m'][2]
        target('item_0', 'pineapple_juice_carton', 'barzwx', scale, [4.31, -.43])
        floor, interior, aperture = compartment(0, 4.445, 4.665, .25, .112, .087,
                                                closed_top=True, mouth=True)
        destinations.append(floor); interiors[floor] = interior
        objective = ('Store the juice carton inside the low blue pantry cubby. Turn the carton to fit the opening, '
                     'slide it through the front mouth, and leave it completely inside on the cubby floor.')
        decision = 'A standing carton is too tall; pitching it changes the feasible cross-section.'
        burden = 'One object, one insertion; the low mouth permits object passage, not a full open hand.'
    elif alias == 'GF2-2':
        title = 'Turn a cutting board into a narrow drying slot'
        cube('pickup_floor', [.070, .280, .012], [4.31, -.43, TOP + .006])
        for sign, key in [(-1, 'pickup_guide_a'), (1, 'pickup_guide_b')]:
            cube(key, [.010, .280, .065], [4.31 + sign * .0175, -.43, TOP + .012 + .0325])
        target('item_0', 'cutting_board', 'aibvew', 1., [4.31, -.43],
               angles=(90., 0., 90.), support='pickup_floor')
        floor, interior, aperture = compartment(0, 4.423, 4.675, .25, .055, .205,
                                                closed_top=True, mouth=True)
        destinations.append(floor); interiors[floor] = interior
        objective = ('Move the upright cutting board from its wide pickup stand into the narrow blue drying slot. '
                     'Turn the board to match the slot, pass it through the front mouth, and leave the whole board '
                     'inside, supported on the slot floor. The top of the board is exposed for handling at the pickup stand.')
        decision = 'The board starts upright but broadside to the destination; yaw changes the feasible mouth width.'
        burden = 'One board with an exposed upper gripping region; narrow-slot alignment is the harder insertion.'
    elif alias == 'GF3-1':
        title = 'Match cucumber lengths to produce compartments'
        lengths = [.200, .120, .160]
        for i, (length, cy) in enumerate(zip(lengths, [-.48, 0., .48])):
            target(f'item_{i}', 'cucumber', 'wcvwye', length / assets['cucumber/wcvwye']['dimensions_m'][0],
                   [4.30, cy], angles=(0., 0., 90.))
        for i, (length, cy) in enumerate(zip([.145, .185, .225], [-.48, 0., .48])):
            floor, interior, _ = compartment(i, 4.665 - length, 4.665, cy, .180, .110, closed_top=False)
            destinations.append(floor); interiors[floor] = interior
        objective = ('Put the shortest cucumber in the blue produce compartment, the middle-length cucumber in yellow, '
                     'and the longest in purple. Compare their actual length regardless of their current orientation. '
                     'Leave one cucumber completely inside each compartment, supported on its floor.')
        decision = 'Three-way ordering by intrinsic length, with different compartment lengths and a permuted initial order.'
        burden = 'Three sequential transfers with generous open-top access; shape-based ordering rather than tight insertion.'
    else:
        title = 'Match spice-jar heights to shelf compartments'
        for i, (height, cy) in enumerate(zip([.135, .090], [-.40, .40])):
            target(f'item_{i}', 'marjoram_shaker', 'xlspnl', height / assets['marjoram_shaker/xlspnl']['dimensions_m'][2],
                   [4.30, cy])
            upright[f'item_{i}'] = 2
        for i, (height, cy) in enumerate(zip([.125, .175], [-.35, .35])):
            floor, interior, _ = compartment(i, 4.425, 4.665, cy, .185, height, closed_top=True)
            destinations.append(floor); interiors[floor] = interior
        objective = ('Store the shorter spice jar upright in the low blue shelf compartment and the taller jar upright '
                     'in the high yellow compartment. Compare the jars themselves. Leave one jar completely inside '
                     'each compartment on its floor; keep both upright within 10 degrees of vertical.')
        decision = 'Two-way height assignment with a required upright final pose; laying the tall jar down does not satisfy the goal.'
        burden = 'Two transfers; wide mouths allow access while shelf roofs make the final pose meaningful.'
    scoring = dict(minimum_lift_m=None, stability_duration_s=.5, maximum_linear_speed_m_s=.02,
        maximum_angular_speed_rad_s=.1, max_sample_gap_s=.1, fixture_position_tolerance_m=.002,
        fixture_angle_tolerance_rad=.01, dimension_relative_tolerance=1.e-5,
        aperture_clearance_margin_m=.001, minimum_assignment_gap_m=.015, manipulation_revision='final_state_v1',
        threshold_status='Explicit development thresholds; native access and legal physical completion remain separate.')
    if aperture:
        scoring['aperture_floor_contact_tolerance_m'] = .003
    fixtures = [key for key in bindings if key not in asset_specs and key != 'native_support']
    parameters = dict(revision=REVISION, operation=operation,
        candidates={key: bindings[key]['name'] for key in asset_specs}, candidate_assets=asset_specs,
        candidate_dimensions_m=dimensions, destinations=destinations, destination_interiors=interiors,
        fixture_keys=fixtures, scene_contract=dict(scene_model=SCENE, room=ROOM), scoring=scoring,
        unselected_position_tolerance_m=.02, upright_axes=upright,
        completion_items=[dict(id='passage' if aperture else 'assignment',
                               kind='historical_checkpoint' if aperture else 'terminal_condition'),
                          dict(id='stable_placement', kind='terminal_condition')])
    if aperture: parameters['aperture_bindings'] = aperture
    layout = dict(shared_layout_id=f'household_{stem}_20261031', scene_model=SCENE, rooms=[ROOM],
        floor_name=FLOOR, objects=objects, supported_items=supported, synchronize_authored_poses_before_alignment=True,
        robot_start=dict(position=[3.58, -.088, .02], orientation=[0., 0., 0., 1.]),
        initial_gripper_state='open')
    # This uses the same native room and accessible west-table edge as the measured H2 layout.
    init = dict(revision=REVISION, layout=layout)
    patches = [dict(operation='category_initialization', phase=phase,
                    module='wia.authoring.converters.active_layout', parameters=deepcopy(init))
               for phase in ('before_construction', 'after_postprocessing')]
    patches.append(dict(operation='restore_robot_head_mount', phase='after_postprocessing'))
    scene_runtime = dict(policy='native_then_minimal_patch_else_skip', variants={
        'original': {'patches': []}, 'adapted': {'patches': patches}})
    source_identity = f'Specular Reflection/Correspondence/{SCENE}/{ROOM}/household_{stem}.json'
    success = ('Leave every required item on its assigned compartment floor, entirely within its usable interior, '
               'released without robot contact and stable for at least 0.5 seconds. Keep linear speed at most 0.02 m/s '
               'and angular speed at most 0.1 rad/s. Finish with empty hands, the working gripper open and all hands '
               'at least 2 cm clear of the items. No minimum lift height or prior grasp is required. Declare DONE.')
    if aperture:
        success = ('The object must cross the front mouth from the outside through measured intermediate positions '
                   'before its final placement. Routing over or around the mouth does not count. ' + success)
    public = dict(objective=objective, success_criteria=success,
        environment='A kitchen preparation table with ordinary household items and physical storage compartments.',
        constraints=['Use public robot sensors and legal physical controls.',
                     'Preserve object size and shape and the fixed storage apparatus.',
                     'Ordinary movable background furniture may be repositioned when needed.',
                     'Sliding and reorientation are allowed; the robot hand need not enter a narrow opening.'])
    task_id = f'wia_household_{stem}_v1'
    task = dict(kind='category_robot_task', id=task_id, version=1, status='pending',
        source={'provider': 'esi', 'json': source_identity}, public=public,
        requirements={'capabilities': ['base_translation', 'base_rotation', 'ee_translation', 'ee_rotation', 'trunk_height', 'gripper']},
        private=dict(bindings=bindings, evaluation=dict(type='category_operation', module=MODULE,
            operation=operation, binding_keys=list(bindings), parameters=parameters)), scene=scene_runtime,
        qualification=dict(on_failure='skip', initial_visibility='none', target_movable_collidable=True),
        defaults=dict(seed=7, max_steps=180, wall_timeout_s=5400., world_advances_during_inference=False,
                      observation_condition='current', action_mode='parameterized'))
    scene = dict(kind='scene_patch', id=task_id + '_scene', version=1, status='pending', source=task['source'],
                 bindings=bindings, scene=scene_runtime, seed=7, runtime=dict(preview_fps=0., camera_size=[512, 512]))
    transport = dict(id=task_id + '_transport', big_task='Specular Reflection', small_task='Correspondence',
        runner_task='mirror', scene=SCENE, room=ROOM, question=objective, answer='', answer_type='free_form',
        options_json='[]', image_paths_json='[]', metadata_json=json.dumps({'floor_name': FLOOR,
            'question_data': {'placement': [], 'render': {'camera_pose': {
                'position': [3.70, -.088, 1.30], 'quaternion_xyzw': [.5, -.5, -.5, .5]}}}}),
        objects={}, initial_poses={}, wia_provenance=dict(role='authored_scene_transport_only',
            original_source_answer_used=False, task_definition=str(DEST / 'tasks' / (stem + '.yaml'))))
    entry = dict(alias=alias, id=task_id, task_id=task_id, task_file=str((DEST / 'tasks' / (stem + '.yaml')).relative_to(ROOT)),
        scene_file=str((DEST / 'scenes' / (stem + '.yaml')).relative_to(ROOT)), variant='adapted', family='GF',
        subclass='GF-2' if aperture else 'GF-3', title=title, origin='household_geometry', physical_success_verified=False)
    design = dict(alias=alias, decision=decision, execution_burden=burden,
        initial_asset_dimensions_m=dimensions, native_measurements=False,
        reused_native_layout_evidence='outputs/interactiveworld_20261029/r6/iw_h2_02/inventory.json',
        native_route_and_grasp_verified=False, initial_target_count=len(asset_specs), layout=layout)
    return entry, task, scene, transport, design


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); assets = json.loads(ASSETS.read_text()); entries, designs = [], []
    for alias in ['GF2-1', 'GF2-2', 'GF3-1', 'GF3-2']:
        entry, task, scene, transport, design = build(alias, assets)
        files = {ROOT / entry['task_file']: yaml.safe_dump(task, sort_keys=False, width=110),
                 ROOT / entry['scene_file']: yaml.safe_dump(scene, sort_keys=False, width=110),
                 ROOT / 'explore_to_achieve/sources/json_clean' / task['source']['json']: json.dumps(transport, indent=2) + '\n'}
        for path, content in files.items():
            if args.check:
                if not path.is_file() or path.read_text() != content: raise ValueError(f'Generated definition differs: {path}')
            else:
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
        loaded = load_task(ROOT / entry['task_file'], variant='adapted', run_mode='diagnostic')
        loaded_scene = load_scene(ROOT / entry['scene_file'], variant='adapted', run_mode='diagnostic')
        assert loaded.seed == loaded_scene.seed and loaded.definition.private.bindings == loaded_scene.definition.bindings
        entries.append(entry); designs.append(design)
    for name, value in [('catalogue.json', dict(schema='wia_household_geometry_candidates_v1',
            release_admitted=False, tasks=entries)), ('design.json', dict(schema='wia_household_geometry_design_v1',
            user_request='Two everyday scenes each for GF-2 and GF-3, without restoring D tasks.',
            asset_metadata=assets, scenes=designs, empirical_difficulty_calibrated=False))]:
        path = DEST / name; content = json.dumps(value, indent=2) + '\n'
        if args.check:
            if path.read_text() != content: raise ValueError(f'Generated metadata differs: {path}')
        else: path.write_text(content)
    print(json.dumps({'count': len(entries), 'aliases': [r['alias'] for r in entries], 'native_run': False}))


if __name__ == '__main__':
    main()
