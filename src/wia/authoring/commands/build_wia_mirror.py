"""Build eight pending mirror proposals with two authored seed versions each.

Run: python -B scripts/build_wia_mirror.py [--check] [--output-dir PATH]
The main-authored design.json is immutable. No native optics or actions run.


Canonical invocation: python -m wia.authoring.commands.build_wia_mirror --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import yaml

from wia.resources import ROOT

from wia.authoring.conversion import ConversionContext, SourceInstance, original_scene, physical_task
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition, SceneRuntime
from wia.tasks.sources import resolve_source
from wia.authoring.commands.build_wia_dimensional import CAPABILITIES, quaternion, sha256, slot_order, validate_outputs

DEFAULT_OUTPUT = ROOT / 'explore_to_achieve/wia_v1/mirror_v1'
DESIGN = DEFAULT_OUTPUT / 'design.json'
ALIASES = ('R-C1', 'R-C2', 'R-C3', 'R-A1', 'R-A2', 'R-S1', 'R-S2', 'R-S3')
CUE_KEYS = ('cue_low', 'cue_high', 'cue_top', 'cue_bottom')


def anchor_position(row, seed, design):
    """Apply the main-authored side rule as a proposal, never measured truth."""
    position = deepcopy(design['anchor']['position_m'])
    if row['alias'] in ('R-S1', 'R-S2'):
        negative = seed == design['seeds'][0]
        same_side = row['alias'] == 'R-S1'
        position[1] = -2.04 if negative == same_side else .04
    return position


def make_layout(row, seed, design):
    """Create real collidable Cubes and pass material setup to the native hook."""
    common, mirror, cue = design['layout'], design['mirror'], design['cue']
    objects, bindings = [], {}

    def add(key, dimensions, position, *, name=None, category='object', fixed=True, rgba=None, yaw=0.):
        name = name or 'wia_mirror_' + key
        objects.append({'type': 'PrimitiveObject', 'name': name, 'category': category,
            'primitive_type': 'Cube', 'size': 1., 'scale': deepcopy(dimensions),
            'position': deepcopy(position), 'orientation': quaternion(yaw),
            'fixed_base': fixed, 'visual_only': False,
            'rgba': deepcopy(common['candidate_rgba'] if rgba is None else rgba)})
        bindings[key] = {'name': name, 'category': category}

    add('table', common['table_dimensions_m'], common['table_center_m'], category='table')
    for index, xy in enumerate(common['leg_centers_xy_m']):
        add('leg_' + str(index), common['leg_dimensions_m'], [*xy, common['leg_center_z_m']])
    for index, color in enumerate(('blue', 'yellow')):
        add('pad_' + color, common['delivery_pad_dimensions_m'],
            [common['delivery_pad_x_m'], common['delivery_pad_y_m'][index], common['pad_z_m']],
            category='table', rgba=common['pad_rgba'][index])
    add('mirror', mirror['dimensions_m'], mirror['position_m'], name=mirror['name'], rgba=[.95, .95, .95, 1.])
    for index, y in enumerate(mirror['stand_post_y_m']):
        add('mirror_post_' + str(index), mirror['stand_post_dimensions_m'],
            [mirror['position_m'][0], y, mirror['stand_post_center_z_m']])
    add('anchor', design['anchor']['dimensions_m'], anchor_position(row, seed, design),
        name=design['anchor']['name'], rgba=design['anchor']['rgba'])
    seed_index = design['seeds'].index(seed)
    cue_y = row.get('cue_seed_center_y_m', cue['seed_center_y_m'])[seed_index]
    cue_x = (mirror['position_m'][0] - mirror['dimensions_m'][0] / 2 - cue['mirror_front_clearance_m']
             if row['appearance_mode'] == 'reflected' else cue['direct_x_m'])
    width, height, border, depth = (cue[key] for key in
        ('inner_width_m', 'inner_height_m', 'border_thickness_m', 'depth_m'))
    for key, sign in (('cue_low', -1), ('cue_high', 1)):
        add(key, [depth, border, height + 2 * border],
            [cue_x, cue_y + sign * (width / 2 + border / 2), cue['center_z_m']], rgba=cue['rgba'])
    for key, sign in (('cue_bottom', -1), ('cue_top', 1)):
        add(key, [depth, width, border],
            [cue_x, cue_y, cue['center_z_m'] + sign * (height / 2 + border / 2)], rgba=cue['rgba'])
    fixture_keys = list(bindings)
    order = slot_order(seed, len(row['candidate_y_m']))
    candidates, supported = {}, []
    for index, yaw in enumerate(row['candidate_yaw_deg']):
        key = 'candidate_' + str(index)
        size = row['candidate_dimensions_m']
        add(key, size, [common['candidate_x_m'], row['candidate_y_m'][order[index]],
            common['tabletop_z_proposal_m'] + size[2] / 2 + .003], fixed=False, yaw=yaw)
        candidates[key] = bindings[key]['name']
        supported.append({'name': bindings[key]['name'], 'support': bindings['table']['name']})
    layout = {'shared_layout_id': design['shared_layout_id'], 'scene_model': design['scene_model'],
        'rooms': [design['room']], 'floor_name': design['floor_name'], 'objects': objects,
        'supported_items': supported, 'robot_start': {'position': deepcopy(common['robot_position_m']),
            'orientation': quaternion(common['robot_yaw_deg'])}}
    proposal = {'candidate_slot_permutation': order, 'cue_center_proposal_m': [cue_x, cue_y, cue['center_z_m']],
        'anchor_position_proposal_m': anchor_position(row, seed, design)}
    if row['operation'] == 'mirror_reconfigure':
        proposal.update(intended_cue_side_proposal='negative_y' if seed_index == 0 else 'positive_y',
            native_selected_identity_verified=False, non_goal_initialization_required=True,
            native_non_goal_initialization_verified=False)
    return layout, bindings, candidates, fixture_keys, proposal


def public_goal(row, scoring):
    """Expose only the requested appearance or fixed-reference physical relation."""
    success = (f'Use a genuine robot grasp to lift each required block at least {scoring["minimum_lift_m"] * 100:g} cm. '
        f'Leave all required blocks supported and released without robot contact for at least {scoring["stability_duration_s"]:g} seconds, '
        f'with linear speed at most {scoring["maximum_linear_speed_m_s"]:g} m/s and angular speed at most '
        f'{scoring["maximum_angular_speed_rad_s"]:g} rad/s. Leave the working hand empty and open, '
        'withdraw the robot hands from the blocks, and declare DONE.')
    constraints = ['Keep the table, pads, mirror, red outline and yellow reference pillar fixed; preserve all object dimensions.',
        'Use the existing guarded public controls and robot-mounted observations.',
        'Interpret the requested appearance using the initial head-camera view. Later robot movement does not redefine the requested real block.']
    if row.get('requires_both_objects_moved'):
        constraints += [f'Grasp and lift both real blocks. Move each block at least {scoring["minimum_required_motion_m"] * 100:g} cm from its starting position.',
            'Measure the final distance order in the horizontal plane relative to the fixed yellow pillar; moving the robot does not establish progress.']
    else:
        constraints += ['The first candidate grasp commits the real-object selection. Do not grasp a different candidate.',
            f'Keep each unselected block within {scoring["unselected_position_tolerance_m"] * 100:g} cm of its starting position.']
        if row['operation'] == 'mirror_reconfigure':
            constraints += [f'Move the selected real block at least {scoring["minimum_required_motion_m"] * 100:g} cm from its starting position.',
                'Measure distances in the horizontal plane from each real block to the fixed yellow pillar.']
    if row['operation'] == 'classify_appearance_deliver':
        success = 'Report whether the initial framed appearance is reflected or direct before the first candidate grasp, and deliver the corresponding real block to the declared color. ' + success
    return {'objective': row['goal'], 'success_criteria': success,
        'environment': 'A shared hall workstation with grey blocks, a physical mirror, a red outline, blue and yellow pads, and a fixed yellow reference pillar.',
        'constraints': constraints}


def build_package():
    """Render sixteen pending versions while preserving the fixed design and source."""
    design_bytes = DESIGN.read_bytes()
    design = json.loads(design_bytes)
    if (design['schema'] != 'wia_mirror_design_v1' or design['seeds'] != [101, 202]
            or tuple(row['alias'] for row in design['tasks']) != ALIASES):
        raise ValueError('The fixed eight-proposal mirror design requires main-agent review before revision.')
    source_file = resolve_source(design['source']['identity'])
    source_hash = sha256(source_file.read_bytes())
    if source_hash != design['source']['sha256']:
        raise ValueError('The immutable background bootstrap source changed.')
    source = SourceInstance(source_file, design['source']['identity'], *Path(design['source']['identity']).parts[:2])
    definitions, entries, rendered = [], [], {}
    runtime = SceneRuntime().model_dump(mode='json')
    for row in design['tasks']:
        for seed in design['seeds']:
            layout, bindings, candidates, fixtures, proposal = make_layout(row, seed, design)
            initialization = {'revision': design['revision'], 'layout': layout,
                'mirror_material': {'binding': 'mirror', **deepcopy(design['mirror']['material'])}, 'mirror_key': 'mirror'}
            patches = [{'operation': 'category_initialization', 'phase': phase, 'module': design['module'],
                'parameters': deepcopy(initialization)} for phase in ('before_construction', 'after_postprocessing')]
            p = {'revision': design['revision'], 'operation': row['operation'], 'candidates': candidates,
                'candidate_dimensions_m': {key: deepcopy(row['candidate_dimensions_m']) for key in candidates},
                'fixture_keys': fixtures, 'scoring': deepcopy(design['scoring']), 'table_key': 'table', 'mirror_key': 'mirror',
                'cue_bindings': {'lower_side': 'cue_low', 'upper_side': 'cue_high', 'roof': 'cue_top', 'floor': 'cue_bottom'},
                'anchor_key': 'anchor', 'destinations': {'blue': 'pad_blue', 'yellow': 'pad_yellow'},
                'appearance_mode': row['appearance_mode'],
                'completion_items': [{'id': stage, 'kind': 'terminal_condition' if stage == 'stable_placement'
                    else 'historical_checkpoint'} for stage in design['completion_items'][row['operation']]]}
            for field in ('relation', 'distance_gap_m', 'requires_both_objects_moved', 'non_goal_initialization_required'):
                if field in row:
                    p[field] = deepcopy(row[field])
            evaluation = {'type': 'category_operation', 'module': design['module'], 'operation': row['operation'],
                'binding_keys': list(bindings), 'parameters': p}
            context = ConversionContext(DEFAULT_OUTPUT, seed=seed)
            task = physical_task(source, context, public=public_goal(row, design['scoring']), evaluation=evaluation,
                bindings=deepcopy(bindings), patches=deepcopy(patches), capabilities=deepcopy(CAPABILITIES),
                qualification={'on_failure': 'skip', 'target_movable_collidable': True, 'support': None, 'initial_visibility': 'none'})
            public_id = ('wia_appearance_delivery_v1' if row['alias'].startswith('R-A') else
                'wia_mirror_' + row['alias'].lower().replace('-', '_') + '_v1')
            stem = row['alias'].lower().replace('-', '_') + '_seed' + str(seed) + '_v1'
            task.update(id=public_id, version=1, status='pending')
            task['defaults'].update(max_steps=240, world_advances_during_inference=True,
                observation_condition='current', action_mode='parameterized')
            task['repair'] = {'contract_revision': design['revision'], 'design_file': DESIGN.relative_to(ROOT).as_posix(),
                'design_sha256': sha256(design_bytes), 'source_role': design['source']['role'],
                'budget_status': 'Explicit provisional 240-decision development budget.', 'runtime': deepcopy(runtime),
                'native_optics_verified': False, 'physical_success_verified': False,
                'full_reference_execution': design['full_reference_execution']}
            scene = original_scene(source, context, bindings=deepcopy(bindings), patches=deepcopy(patches))
            scene.update(id='wia_mirror_' + stem + '_scene', runtime=deepcopy(runtime))
            CategoryTaskDefinition.model_validate(task)
            SceneDefinition.model_validate(scene)
            task_file, scene_file = Path('tasks') / (stem + '.yaml'), Path('scenes') / (stem + '.yaml')
            for path, definition in ((task_file, task), (scene_file, scene)):
                rendered[path] = yaml.safe_dump(definition, sort_keys=False, allow_unicode=False, width=88).encode('utf-8')
            definitions.append({'alias': row['alias'], 'seed': seed, 'task': task, 'scene': scene,
                                'task_file': task_file, 'scene_file': scene_file})
            entries.append({'proposal_alias': row['alias'], 'wia_id': 'wia_v1_' + row['alias'].lower().replace('-', '_'),
                'title': row['title'], 'seed': seed, 'variant': 'adapted', 'public_task_id': public_id,
                'task_file': task_file.as_posix(), 'task_sha256': sha256(rendered[task_file]),
                'scene_file': scene_file.as_posix(), 'scene_sha256': sha256(rendered[scene_file]),
                'layout_proposal': proposal, 'appearance_mode_proposal': row['appearance_mode'],
                'native_optics_verified': False, 'native_selected_identity_verified': False,
                'native_non_goal_initialization_verified': False, 'physical_success_verified': False})
    manifest = {'schema': 'wia_mirror_manifest_v1', 'revision': design['revision'], 'status': 'pending_native_review',
        'artifact_path_base': 'manifest_directory', 'proposal_count': 8, 'authored_seed_version_count': 16,
        'source': {'identity': design['source']['identity'], 'sha256': source_hash,
            'file': source_file.relative_to(ROOT).as_posix(), 'role': design['source']['role'],
            'original_optical_inventory_replaced': True, 'original_source_qa_reused': False},
        'design_file': DESIGN.relative_to(ROOT).as_posix(), 'design_sha256': sha256(design_bytes),
        'shared_layout_id': design['shared_layout_id'], 'seeds': design['seeds'],
        'seed_role': 'Deterministic candidate position permutations and explicitly authored cue/anchor proposals; not additional proposal aliases.',
        'matched_appearance_control': {'aliases': ['R-A1', 'R-A2'], 'shared_public_task_id': 'wia_appearance_delivery_v1',
            'same_public_payload': True, 'only_layout_difference': 'The four cue rods change pose; all other objects and robot start remain identical per seed.'},
        'mirror_material': {'binding': 'mirror', **deepcopy(design['mirror']['material'])},
        'mirror_material_application': design['mirror']['material_application'],
        'initial_head_reference': 'Actual initial mounted head eye must be frozen by the native adapter; later robot movement cannot redefine correspondence.',
        'observation_contract': 'Retain the existing eight mounted RGB observations and frozen initial overview; optical visibility and necessity remain unverified.',
        'provisional_budget_max_steps': 240, 'native_optics_verified': False, 'native_visibility_verified': False,
        'native_shape_verified': False, 'native_hand_path_verified': False, 'physical_success_verified': False,
        'scientific_controls_verified': False, 'continuous_path_certified': False,
        'native_non_goal_initialization_verified': False, 'full_reference_execution': design['full_reference_execution'],
        'entries': entries}
    rendered[Path('manifest.json')] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode('utf-8')
    return definitions, manifest, rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Compare exact generated bytes without writing.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    definitions, _, rendered = build_package()
    output = args.output_dir.resolve()
    mismatches = [output / path for path, data in rendered.items() if (output / path).is_symlink()
        or not (output / path).is_file() or (output / path).read_bytes() != data]
    if args.check:
        for path in mismatches:
            print(f'WIA mirror definition differs or is missing: {path}', file=sys.stderr)
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
    print(f'WIA mirror definitions are current: {output} (8 proposals, 16 task/scene seed versions)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
