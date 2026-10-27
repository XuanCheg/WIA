"""Build five pending dimensional proposals with two position permutations each.

Run: python -B scripts/build_wia_dimensional.py [--check] [--design-file PATH] [--output-dir PATH]
The main-authored design.json is read-only. No simulator, model or queue runs.


Canonical invocation: python -m wia.authoring.commands.build_wia_dimensional --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import random
import sys

import yaml

from wia.resources import ROOT

from wia.tasks.loader import load_scene, load_task
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition, SceneRuntime
from wia.tasks.sources import resolve_source

DEFAULT_OUTPUT = ROOT / 'explore_to_achieve/wia_v1/dimensional_v1'
DESIGN = DEFAULT_OUTPUT / 'design.json'
SUPPORTED_REVISIONS = {'wia_dimensional_v1': ('v1', 1), 'wia_dimensional_v2': ('v2', 2)}
CAPABILITIES = ['base_translation', 'base_rotation', 'ee_translation', 'ee_rotation', 'trunk_height', 'gripper']
STAGE_IDS = {
    'select_pass_aperture': ('choice', 'grasp_lift', 'passage', 'stable_placement'),
    'sort_intrinsic_dimension': ('grasp_lift', 'assignment', 'stable_placement'),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def quaternion(yaw):
    angle = math.radians(yaw) / 2
    return [0., 0., math.sin(angle), math.cos(angle)]


def slot_order(seed, count):
    """Share one seeded permutation between the three- and four-block layouts."""
    result = list(range(count))
    random.Random(seed).shuffle(result)
    return result


def geometry_proposal(row, design):
    """Enumerate object-only bounds; never certify a native motion or hand path."""
    dimensions = row['candidate_dimensions_m']
    if row['operation'] == 'sort_intrinsic_dimension':
        lengths = [max(size) for size in dimensions]
        gaps = [right - left for left, right in zip(sorted(lengths), sorted(lengths)[1:])]
        return {'scope': 'declared_intrinsic_object_dimensions_only',
            'intrinsic_longest_edges_m': lengths, 'ordered_neighbor_gaps_m': gaps,
            'minimum_assignment_gap_m': design['scoring']['minimum_assignment_gap_m'],
            'declared_gaps_meet_contract': min(gaps) >= design['scoring']['minimum_assignment_gap_m'],
            'world_aabb_used_as_intrinsic_metric': False, 'hand_or_path_certified': False}
    margin = design['scoring']['aperture_clearance_margin_m']
    width, height = row['opening_width_m'], row['opening_height_m']
    objects = {}
    for index, size in enumerate(dimensions):
        orientations = sorted(set(permutations(size)))
        fits = [list(value) for value in orientations if value[1] + 2 * margin <= width
                and value[2] + 2 * margin <= height]
        record = {'declared_dimensions_m': size, 'axis_permutations_xyz_m': [list(value) for value in orientations],
            'object_only_axis_aligned_cross_section_fits_xyz_m': fits}
        if len(set(size)) == 1:
            blocking = [axis for axis, extent in (('world_y', width), ('world_z', height)) if size[0] > extent]
            record['cube_minimum_caliper_bound'] = {'minimum_caliper_m': size[0],
                'blocking_aperture_axes': blocking, 'all_rotations_excluded_by_declared_object_bound': bool(blocking),
                'native_dimensions_verified': False, 'hand_or_path_certified': False}
        objects['candidate_' + str(index)] = record
    return {'scope': 'declared_object_bounds_only_not_native_geometry', 'opening_width_m': width,
        'opening_height_m': height, 'tunnel_depth_m': row['tunnel_depth_m'],
        'clearance_margin_m': margin, 'objects': objects, 'hand_or_path_certified': False}


def make_layout(row, seed, design):
    """Create genuine unit Cube primitives scaled to the exact authored dimensions."""
    common, aperture = design['layout'], design['aperture']
    objects, bindings = [], {}

    def add(key, dimensions, position, *, fixed=True, category='object', rgba=None, yaw=0.):
        name = 'wia_dim_' + key
        default_rgba = common.get('fixture_rgba', common['candidate_rgba']) if fixed else common['candidate_rgba']
        objects.append({'type': 'PrimitiveObject', 'name': name, 'category': category,
            'primitive_type': 'Cube', 'size': 1., 'scale': deepcopy(dimensions),
            'position': deepcopy(position), 'orientation': quaternion(yaw),
            'fixed_base': fixed, 'visual_only': False,
            'rgba': deepcopy(default_rgba if rgba is None else rgba)})
        bindings[key] = {'name': name, 'category': category}

    add('table', common['table_dimensions_m'], common['table_center_m'], category='table', rgba=common.get('table_rgba'))
    for index, xy in enumerate(common['leg_centers_xy_m']):
        add('leg_' + str(index), common['leg_dimensions_m'], [*xy, common['leg_center_z_m']])
    aperture_bindings = None
    if row['operation'] == 'select_pass_aperture':
        add('pad_0', aperture['delivery_pad_dimensions_m'], aperture['delivery_pad_center_m'],
            category='table', rgba=common['pad_rgba'][0])
        width, height = row['opening_width_m'], row['opening_height_m']
        thickness, top = aperture['side_thickness_m'], aperture['top_thickness_m']
        bottom, centre_y = aperture['bottom_z_m'], aperture['center_y_m']
        wall_dimensions = [row['tunnel_depth_m'], thickness, height + top]
        for key, sign in (('aperture_low', -1), ('aperture_high', 1)):
            add(key, wall_dimensions, [aperture['center_x_m'], centre_y + sign * (width / 2 + thickness / 2),
                bottom + (height + top) / 2])
        add('aperture_roof', [row['tunnel_depth_m'], aperture['frame_outer_width_m'], top],
            [aperture['center_x_m'], centre_y, bottom + height + top / 2])
        aperture_bindings = {'lower_side': 'aperture_low', 'upper_side': 'aperture_high',
                             'roof': 'aperture_roof', 'floor': 'table'}
        slots = aperture['start_candidate_y_slots_m']
        order = slot_order(seed, 2)
        destinations = ['pad_0']
    else:
        for index, (y, rgba) in enumerate(zip(common['sorting_pad_y_slots_m'], common['pad_rgba'])):
            add('pad_' + str(index), common['sorting_pad_dimensions_m'],
                [common['sorting_pad_x_m'], y, common['pad_z_m']], category='table', rgba=rgba)
        slots = common['candidate_y_slots_m']
        order = slot_order(seed, 4)
        destinations = ['pad_' + str(index) for index in range(len(row['candidate_dimensions_m']))]
    fixture_keys = list(bindings)
    candidates, supported = {}, []
    for index, (dimensions, yaw) in enumerate(zip(row['candidate_dimensions_m'], row['initial_yaw_deg'])):
        key = 'candidate_' + str(index)
        xy = (common['candidate_xy_slots_m'][order[index]]
              if row['operation'] == 'sort_intrinsic_dimension' and 'candidate_xy_slots_m' in common
              else [common['candidate_x_m'], slots[order[index]]])
        add(key, dimensions, [*xy,
            common['tabletop_z_proposal_m'] + dimensions[2] / 2 + .003], fixed=False, yaw=yaw)
        candidates[key] = bindings[key]['name']
        supported.append({'name': bindings[key]['name'], 'support': bindings['table']['name']})
    layout = {'shared_layout_id': common['shared_layout_id'], 'scene_model': design['scene_model'],
        'rooms': [design['room']], 'floor_name': design['floor_name'], 'objects': objects,
        'supported_items': supported, 'robot_start': {'position': deepcopy(common['robot_position_m']),
            'orientation': quaternion(common['robot_yaw_deg'])}}
    return layout, bindings, candidates, destinations, fixture_keys, aperture_bindings, order


def public_goal(row):
    """Expose the dimensional goal without private identities, positions or answers."""
    success = ('Use a real robot grasp to lift each required block at least 5 cm. '
        'Finish with every required block supported on its assigned pad, released without robot contact, '
        'and stable for at least 0.5 seconds with linear speed at most 0.02 m/s and angular speed '
        'at most 0.1 rad/s. Leave the working hand empty and open, withdraw all robot hands from '
        'the placed blocks, and declare DONE.')
    constraints = ['Preserve all block dimensions and all fixed tables, legs, pads and frames.',
                   'Use the existing guarded public robot controls and the observed block shapes and pad colors.']
    if row['operation'] == 'select_pass_aperture':
        success = ('The selected block must physically enter the near opening, traverse the frame depth, '
            'and fully exit through the far opening before its final supported release on blue. ' + success)
        constraints += ['The first candidate block you grasp commits your selection. Keep the unselected block '
            'within 2 cm of its starting position.',
            'Rotation is allowed. Pass the selected block through the opening; do not route it around the frame.',
            'The selected block may be released before passage is complete if it subsequently completes the '
            'physical traversal and finishes on blue. The robot hand itself need not pass through the opening.']
        environment = 'A hall workstation with grey blocks, a fixed framed opening and a blue delivery pad.'
    else:
        constraints += ['Compare each block by its intrinsic longest physical edge, independent of its current '
            'orientation. Complete a genuine grasp, lift and released placement for every block.']
        environment = 'The shared hall workstation with grey blocks and blue, yellow, purple and cyan pads.'
    return {'objective': row['goal'], 'success_criteria': success, 'environment': environment, 'constraints': constraints}


def build_package(design_path=DESIGN):
    """Validate the immutable design and render ten separate pending seed versions."""
    design_path = Path(design_path).resolve()
    design_bytes = design_path.read_bytes()
    design = json.loads(design_bytes)
    if (design['schema'] != 'wia_dimensional_design_v1' or design['seeds'] != [101, 202]
            or [row['alias'] for row in design['tasks']] != ['D1-1', 'D1-2', 'D1-3', 'D2-1', 'D2-2']):
        raise ValueError('The fixed five-proposal, two-seed design requires main-agent review before revision.')
    if design['revision'] not in SUPPORTED_REVISIONS:
        raise ValueError('Only the explicitly supported dimensional V1 and V2 revisions may be generated.')
    suffix, version = SUPPORTED_REVISIONS[design['revision']]
    if suffix == 'v2':
        parent = design['parent_design']
        if sha256(Path(parent['file']).read_bytes()) != parent['sha256']:
            raise ValueError('The V2 parent design must retain its exact historical bytes.')
    source_file = resolve_source(design['source']['identity'])
    source_hash = sha256(source_file.read_bytes())
    if source_hash != design['source']['sha256']:
        raise ValueError('The original background bootstrap source changed.')
    if any(slot_order(101, count) == slot_order(202, count) for count in (2, 4)):
        raise ValueError('The two explicit seeds must give different candidate slot permutations.')
    definitions, entries, rendered = [], [], {}
    runtime = SceneRuntime().model_dump(mode='json')
    for row in design['tasks']:
        for seed in design['seeds']:
            layout, bindings, candidates, destinations, fixtures, aperture, order = make_layout(row, seed, design)
            initialization = {'revision': design['revision'], 'layout': layout}
            patches = [{'operation': 'category_initialization', 'phase': phase,
                'module': design['initializer_module'], 'parameters': deepcopy(initialization)}
                for phase in ('before_construction', 'after_postprocessing')]
            scene_config = {'policy': 'native_then_minimal_patch_else_skip',
                'variants': {'original': {'patches': []}, 'adapted': {'patches': patches}}}
            parameters = {'revision': design['revision'], 'operation': row['operation'], 'candidates': candidates,
                'destinations': destinations, 'scoring': deepcopy(design['scoring']),
                'unselected_position_tolerance_m': .02, 'fixture_keys': fixtures,
                'candidate_dimensions_m': {'candidate_' + str(index): deepcopy(size)
                    for index, size in enumerate(row['candidate_dimensions_m'])},
                'completion_items': [{'id': stage, 'kind': 'terminal_condition' if stage == 'stable_placement'
                    else 'historical_checkpoint'} for stage in STAGE_IDS[row['operation']]]}
            if aperture is not None:
                parameters['aperture_bindings'] = aperture
            public_id = 'wia_dimensional_' + row['alias'].lower().replace('-', '_') + '_' + suffix
            stem = row['alias'].lower().replace('-', '_') + '_seed' + str(seed) + '_' + suffix
            task = {'kind': 'category_robot_task', 'id': public_id, 'version': version, 'status': 'pending',
                'source': {'provider': 'esi', 'json': design['source']['identity']}, 'public': public_goal(row),
                'requirements': {'capabilities': deepcopy(CAPABILITIES)}, 'private': {'bindings': deepcopy(bindings),
                    'evaluation': {'type': 'category_operation', 'module': design['module'], 'operation': row['operation'],
                        'binding_keys': list(bindings), 'parameters': parameters}},
                'scene': deepcopy(scene_config), 'qualification': {'on_failure': 'skip',
                    'target_movable_collidable': True, 'support': None, 'initial_visibility': 'none'},
                'defaults': {'seed': seed, 'max_steps': 240, 'world_advances_during_inference': True,
                    'observation_condition': 'current', 'action_mode': 'parameterized'},
                'repair': {'contract_revision': design['revision'], 'design_file': design_path.relative_to(ROOT).as_posix(),
                    'design_sha256': sha256(design_bytes), 'source_role': design['source']['role'],
                    'budget_status': 'Explicit provisional 240-decision development budget.',
                    'runtime': deepcopy(runtime), 'native_validation_run': False,
                    'full_reference_execution': 'deferred_until_user_review'}}
            scene = {'kind': 'scene_patch', 'id': 'wia_dimensional_' + stem + '_scene', 'version': version,
                'status': 'pending', 'source': deepcopy(task['source']), 'target': None, 'qualification': None,
                'bindings': deepcopy(bindings), 'scene': deepcopy(scene_config), 'runtime': deepcopy(runtime), 'seed': seed}
            CategoryTaskDefinition.model_validate(task)
            SceneDefinition.model_validate(scene)
            task_file, scene_file = Path('tasks') / (stem + '.yaml'), Path('scenes') / (stem + '.yaml')
            for path, definition in ((task_file, task), (scene_file, scene)):
                rendered[path] = yaml.safe_dump(definition, sort_keys=False, allow_unicode=False, width=88).encode('utf-8')
            definitions.append({'alias': row['alias'], 'seed': seed, 'task': task, 'scene': scene,
                                'task_file': task_file, 'scene_file': scene_file})
            entries.append({'proposal_alias': row['alias'], 'title': row['title'], 'primary_family': 'GF',
                'seed': seed, 'variant': 'adapted', 'task_id': public_id, 'shared_layout_id': layout['shared_layout_id'],
                'task_file': task_file.as_posix(), 'task_sha256': sha256(rendered[task_file]),
                'scene_file': scene_file.as_posix(), 'scene_sha256': sha256(rendered[scene_file]),
                'candidate_slot_permutation': order,
                'candidate_y_positions_m': {key: next(obj['position'][1] for obj in layout['objects'] if obj['name'] == name)
                    for key, name in candidates.items()},
                'geometry_proposal': geometry_proposal(row, design), 'native_validation_run': False,
                'native_dimensions_verified': False, 'legal_reference_execution_verified': False})
            if suffix == 'v2':
                entries[-1]['candidate_xy_positions_m'] = {key: next(obj['position'][:2]
                    for obj in layout['objects'] if obj['name'] == name) for key, name in candidates.items()}
    manifest = {'schema': 'wia_dimensional_manifest_v1', 'revision': design['revision'], 'status': 'pending_native_review',
        'artifact_path_base': 'manifest_directory', 'proposal_count': 5, 'authored_seed_version_count': 10,
        'primary_family': 'GF', 'source_category_is_task_family': False,
        'source': {'identity': design['source']['identity'], 'sha256': source_hash,
            'file': source_file.relative_to(ROOT).as_posix(), 'role': design['source']['role'],
            'category': '/'.join(Path(design['source']['identity']).parts[:2]),
            'original_optical_inventory_replaced': True, 'original_optical_qa_reused': False},
        'design_file': design_path.relative_to(ROOT).as_posix(), 'design_sha256': sha256(design_bytes),
        'shared_layout_id': design['layout']['shared_layout_id'],
        'counterbalance': {'seeds': design['seeds'], 'seed_role': design['seed_role'],
            'four_slot_permutations': {str(seed): slot_order(seed, 4) for seed in design['seeds']},
            'two_slot_permutations': {str(seed): slot_order(seed, 2) for seed in design['seeds']},
            'd2_first_three_positions_preserved_when_adding_fourth': True,
            'generator_seed_changes_only_candidate_slot_assignment': True},
        'completion_item_meaning': {'choice': 'Lawful unique first-grasp commitment checkpoint; passage establishes mechanical correctness.'},
        'fixture_color': 'The neutral grey already declared in design.layout.candidate_rgba; colored pads use design colors.',
        'provisional_budget_max_steps': 240, 'native_validation_run': False, 'native_geometry_verified': False,
        'native_visibility_verified': False, 'native_gripper_path_verified': False, 'continuous_path_certified': False,
        'scientific_controls_verified': False, 'full_reference_execution': 'deferred_until_user_review',
        'parameters_calibrated': False, 'entries': entries}
    if suffix == 'v2':
        manifest['parent_design'] = deepcopy(design['parent_design'])
        manifest['initial_visibility_contract'] = deepcopy(design['initial_visibility_contract'])
        manifest['fixture_color'] = {'table_rgba': deepcopy(design['layout']['table_rgba']),
            'fixture_rgba': deepcopy(design['layout']['fixture_rgba']),
            'candidate_rgba': deepcopy(design['layout']['candidate_rgba']),
            'colored_pad_palette': 'Unchanged from the parent design.'}
    rendered[Path('manifest.json')] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode('utf-8')
    return definitions, manifest, rendered


def validate_outputs(output, definitions):
    """Load every authored task and scene using production diagnostic contracts."""
    for entry in definitions:
        task = load_task(output / entry['task_file'], variant='adapted', run_mode='diagnostic')
        scene = load_scene(output / entry['scene_file'], variant='adapted', run_mode='diagnostic')
        if task.definition.scene != scene.definition.scene or task.seed != scene.seed:
            raise ValueError('Task and scene patches and seeds must be identical.')
        task.verify_unchanged()
        scene.verify_unchanged()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Compare deterministic bytes without writing.')
    parser.add_argument('--design-file', type=Path, default=DESIGN, help='Read the supported V1 or V2 design without modifying it.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    definitions, _, rendered = build_package(args.design_file)
    output = args.output_dir.resolve()
    mismatches = [output / path for path, data in rendered.items() if (output / path).is_symlink()
        or not (output / path).is_file() or (output / path).read_bytes() != data]
    if args.check:
        for path in mismatches:
            print(f'WIA dimensional definition differs or is missing: {path}', file=sys.stderr)
        if mismatches:
            return 1
    else:
        for path, data in rendered.items():
            destination = output / path
            if destination.is_symlink() or (destination.exists() and (not destination.is_file() or destination.read_bytes() != data)):
                raise ValueError(f'Refusing to overwrite a differing generated file: {destination}')
        for path, data in rendered.items():
            destination = output / path
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
    validate_outputs(output, definitions)
    print(f'WIA dimensional definitions are current: {output} (5 proposals, 10 task/scene seed versions)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
