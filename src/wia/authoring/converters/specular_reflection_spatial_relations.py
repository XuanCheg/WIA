"""Convert mirror distance sources into fixed-reference observation answers.

Run CPU batching with OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m
core.esi_converters.specular_reflection_spatial_relations --assignment ASSIGNMENT.json

Native hooks only read state. Existing guarded controls perform every action.
Source answers require qualified native optics; historical navigation remains explicit.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from wia.tasks.background_policy import legal_background_contact

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from wia.authoring.conversion import (
    ConverterBase, ConversionContext, SourceInstance, original_scene, physical_task,
)
from wia.authoring.io import write_json
from wia.tasks.interaction_setup import array as native_array, verified_repaired_start

MODULE = 'core.esi_converters.specular_reflection_spatial_relations'
OPERATIONS = ('nearer_object_approach', 'nearer_object_approach_return', 'pair_approach_return')
ANSWER_OPERATION = 'answer_reflected_spatial_relation'
ACTION_OPERATION = 'answer_and_verify_reflected_spatial_relation'
OPTICAL_KEYS = ('mirror', 'prop_a', 'prop_b')
BINDING_KEYS = (*OPTICAL_KEYS, 'source_floor')


def _array(value):
    return native_array(value)


def _vector(value, size):
    result = _array(value)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError('missing_or_invalid_geometry')
    return result


def _orientation(item):
    # This is the exact default in specular_reflection/common.py, not a repair.
    value = item.get('quaternion_xyzw', item.get('orientation', [0., 0., 0., 1.]))
    quat = _vector(value, 4)
    if np.linalg.norm(quat) < 1e-8:
        raise ValueError('undefined_orientation')
    return quat.tolist()


def _angle_deg(a, b):
    return abs(math.degrees(math.atan2(math.sin(a - b), math.cos(a - b))))


def distance_order(positions, observer):
    """Source generator uses 3D object-root distance, not XY or mirror depth."""
    eye = _vector(observer, 3)
    distances = {key: float(np.linalg.norm(_vector(point, 3) - eye))
                 for key, point in positions.items()}
    ordered = sorted(distances, key=lambda key: (distances[key], key))
    if len(ordered) != 2:
        raise ValueError('two_original_props_required')
    return ordered[0], abs(distances[ordered[0]] - distances[ordered[1]]), distances


def reflected_point(point, mirror_position, normal):
    """Reflect a real root across the measured vertical mirror plane."""
    result = _vector(point, 3).copy()
    n = _vector(normal, 2)
    n = n / np.linalg.norm(n)
    result[:2] -= 2. * np.dot(result[:2] - _vector(mirror_position, 3)[:2], n) * n
    return result.tolist()


def _source_geometry(source):
    raw, payload = source.raw, source.payload
    if (source.big_task, source.small_task) != CategoryConverter.category or (
            raw.get('big_task'), raw.get('small_task')) != CategoryConverter.category:
        raise ValueError('category_mismatch')
    if not raw.get('scene') or not raw.get('room') or any(
            payload.get(key, raw.get(key)) != raw.get(key) for key in ('scene', 'room')):
        raise ValueError('original_scene_room_mismatch')
    q = payload['question_data']
    if payload.get('task_type') != 'mirror_distance' or q.get('task_type') != 'mirror_distance':
        raise ValueError('unsupported_source_mechanism')
    props = {item['alias']: item for item in q['placement']}
    if set(props) != {'A', 'B'} or len(q['placement']) != 2 or props['A']['name'] == props['B']['name']:
        raise ValueError('two_distinct_original_aliases_required')
    for prop in props.values():
        if not all(prop.get(key) for key in ('name', 'category', 'model')):
            raise ValueError('missing_original_prop_identity')
        _vector(prop['real_position'], 3)
        _orientation(prop)
    camera = q.get('render', {}).get('camera_pose') or q['scene_setup']['camera']
    eye = _vector(camera['position'], 3)
    _orientation(camera)
    if np.linalg.norm(eye - _vector(q['scene_setup']['camera']['position'], 3)) > 1e-6 or (
            np.linalg.norm(eye - _vector(q['evidence']['observation_position'], 3)) > 1e-6):
        raise ValueError('original_named_observer_reference_conflict')
    mirror, setup = q['scene_setup']['mirror'], payload['mirror_setup']
    _vector(mirror['position'], 3)
    _orientation(mirror)
    normal = _vector(mirror['normal_xy'], 2)
    if np.linalg.norm(normal) < 1e-8 or not all(setup.get(key) for key in ('name', 'category', 'model')):
        raise ValueError('missing_original_mirror_identity_or_normal')
    if not payload.get('floor_name'):
        raise ValueError('missing_original_floor')
    height = float(payload['camera_poses']['height_m'])
    if not math.isfinite(height) or height <= 0:
        raise ValueError('invalid_original_observer_height')
    if q.get('evidence', {}).get('temporarily_cleared_scene_objects'):
        raise ValueError('source_cleared_obstacles_require_restoration_review')
    return raw, payload, q, props, camera, mirror, setup


def _stance(q, props, camera, mirror):
    """Point screens propose initialization only; native body clearance is required."""
    rows = [('original_observer', 'scene_setup.camera', camera)]
    render = q.get('render', {})
    rows += [('saved_orbit', str(v.get('image_path', i)), v['camera_pose'])
             for i, v in enumerate(render.get('gt_view_input', {}).get('views', []))]
    corners = [('saved_corner', str(v.get('image_path', i)), v['camera_pose'])
               for i, v in enumerate(render.get('multi_image_input', {}).get('views', []))]
    eye = _vector(camera['position'], 3)
    for candidates in (rows, corners):
        valid = []
        for family, label, pose in candidates:
            point = _vector(pose['position'], 3)
            if (all(np.linalg.norm(point[:2] - _vector(o['real_position'], 3)[:2]) >= .7 for o in props.values())
                    and np.linalg.norm(point[:2] - _vector(mirror['position'], 3)[:2]) >= .8):
                valid.append((float(np.linalg.norm(point[:2] - eye[:2])), family, label, point.tolist()))
        if valid:
            _, family, label, point = min(valid)
            return {'family': family, 'position': point, 'source_view': label,
                    'distance_from_original_observer_xy_m': float(np.linalg.norm(_array(point)[:2] - eye[:2])),
                    'native_clearance_or_route_proven': False}
    raise ValueError('no_source_evidenced_robot_stance_after_point_screen')


class CategoryConverter(ConverterBase):
    category = ('Specular Reflection', 'Spatial Relations')
    version = 'specular_spatial_relations_v2_task_edit_20261018'

    def inspect(self, source):
        try:
            _, _, q, props, camera, mirror, _ = _source_geometry(source)
            proposal = {'family': 'original_observer', 'position': camera['position'],
                'native_clearance_or_route_proven': False}
            return {'eligible': True, 'reason': 'preserved_distance_mechanism_with_physical_navigation',
                    'stance_proposal': proposal}
        except (KeyError, TypeError, ValueError, IndexError) as error:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2: ' + str(error),
                    'warning_codes': ['not_convertible_under_L1_L2', 'source_only_diagnostic_required']}

    def convert(self, source, context):
        operation = context.config.get('spatial_relations_operation', ANSWER_OPERATION)
        if operation in (ANSWER_OPERATION, ACTION_OPERATION):
            from .specular_reflection_reflection_authoring import build_reflection_observation, build_reflection_action
            raw, payload, q, props, camera, mirror, setup = _source_geometry(source)
            builder = build_reflection_action if operation == ACTION_OPERATION else build_reflection_observation
            result = builder(source, context, module=MODULE, operation=operation,
                camera=camera, mirror=mirror, props={'prop_' + alias.lower(): obj for alias, obj in props.items()})
            parameters = result['task_definition']['private']['evaluation']['parameters']
            parameters['answer_reference'] = 'original_source_observer_fixed_after_navigation'
            if operation == ACTION_OPERATION:
                parameters['verification_contract']['required_bindings'] = ['prop_a', 'prop_b']
                parameters['verification_contract']['mechanism'] = 'fixed_reference_relation_verified'
            parameters['observable_candidate_labels'] = {
                'prop_' + alias.lower(): alias + ' (' + obj['category'].replace('_', ' ') + ')'
                for alias, obj in props.items()}
            result['available_operation_recipes'] += list(OPERATIONS)
            return result
        eligibility = self.inspect(source)
        if not eligibility['eligible']:
            return {'scene_definition': original_scene(source, context), 'task_definition': None,
                    'status': 'conversion_failed', 'failure_stage': 'source_inspection',
                    'failure_reason': eligibility['reason'], 'warning_codes': eligibility['warning_codes']}
        raw, payload, q, props, camera, mirror, setup = _source_geometry(source)
        proposal = _stance(q, props, camera, mirror)
        bindings = {'prop_' + alias.lower(): {key: prop[key] for key in ('name', 'category', 'model')}
                    for alias, prop in props.items()}
        bindings.update(mirror={key: setup[key] for key in ('name', 'category', 'model')},
                        source_floor={'name': payload['floor_name']})
        poses = {'prop_' + alias.lower(): {'position': prop['real_position'], 'quaternion_xyzw': _orientation(prop)}
                 for alias, prop in props.items()}
        poses['mirror'] = {'position': mirror['position'], 'quaternion_xyzw': _orientation(mirror)}
        selected, gap, distances = distance_order({key: poses[key]['position'] for key in ('prop_a', 'prop_b')}, camera['position'])
        operation = context.config.get('spatial_relations_operation')
        if operation is None:
            operation = OPERATIONS[hashlib.sha256(f'{context.seed}:{source.relative_path}'.encode()).digest()[0] % 2]
            if proposal['family'] != 'original_observer' or gap <= .025:
                operation = OPERATIONS[2]
        if operation not in OPERATIONS:
            raise ValueError('unsupported_spatial_relations_operation')
        if operation != OPERATIONS[2] and (proposal['family'] != 'original_observer' or gap <= .025):
            raise ValueError('nearer_recipe_requires_original_observer_stance_and_unambiguous_source_margin')
        start = proposal['position']
        floor_z = float(camera['position'][2] - payload['camera_poses']['height_m'])
        yaw = math.degrees(math.atan2(mirror['position'][1] - start[1], mirror['position'][0] - start[0]))
        # common.py uses the installed R1 default builder, so no robot injection is needed.
        patches = [{'operation': 'robot_pose', 'position': [*start[:2], floor_z + .02],
                    'yaw_deg': yaw, 'head_pitch_deg': 0.0}]
        parameters = {'source_poses': poses, 'source_observer_position': camera['position'],
            'source_nearer_binding': selected, 'source_mirror_normal_xy': mirror['normal_xy'],
            'proposed_start_xy': start[:2], 'floor_z_m': floor_z,
            'source_pose_tolerance_m': .01, 'source_angle_tolerance_deg': 1.0,
            'minimum_native_distance_margin_m': .005, 'minimum_travel_m': .75,
            'minimum_excursion_m': .5, 'approach_minimum_m': .6, 'approach_maximum_m': .95,
            'departure_distance_m': 1.1, 'candidate_distance_margin_m': .05,
            'return_tolerance_m': .2, 'heading_tolerance_deg': 25.0,
            'maximum_stopped_speed_m_s': .02, 'maximum_stopped_angular_speed_rad_s': .1,
            'stability_steps': 6, 'visual_asset_clearance_m': .03,
            'floor_height_tolerance_m': .03, 'minimum_up_dot': .95,
            'maximum_root_z_change_m': .05, 'start_tolerance_m': .1,
            'approach_order': ['prop_a', 'prop_b'] if operation == OPERATIONS[2] else [selected],
            'minimum_between_stations_m': .25}
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': operation,
                      'binding_keys': list(BINDING_KEYS), 'parameters': parameters}
        objective = ('Use the original mirror scene to determine which of the two real objects is closer '
            'to the original observation point. Physically drive to that real object, then stop 0.60 to 0.95 m '
            'from its root, facing it and at least 0.05 m closer to it than to the other object. '
            'If initially within 1.10 m of that object, first move to a distance of at least 1.10 m and then approach again.')
        success = ('Travel at least 0.75 m with a 0.50 m excursion from your start. Stop for six consecutive measured control steps '
            'with heading error at most 25 degrees, speed at most 0.02 m/s and angular speed at most 0.1 rad/s. '
            'Remain upright on the original floor and avoid obstacles and all optical assets throughout.')
        if operation == OPERATIONS[1]:
            objective += ' After that stable approach, return within 0.20 m of your starting position and stop facing the mirror.'
            success += ' Complete the approach before the final stable return stop.'
        if operation == OPERATIONS[2]:
            objective = ('Physically patrol the two real objects in the original mirror scene: approach object A, '
                'stop, then approach object B and stop, and finally return within 0.20 m of your starting position '
                'and stop facing the mirror. At each object stop, stay 0.60 to 0.95 m from its root, face it within '
                '25 degrees, and be at least 0.05 m closer to it than to the other object. If initially within '
                '1.10 m of A, first move to a distance of at least 1.10 m and then approach A again.')
            success += (' Complete stable stops at A and B in that order, separated by at least 0.25 m, '
                'before the final stable return stop. Original observer-distance QA remains auxiliary.')
        public = {'objective': objective, 'success_criteria': success,
            'environment': 'The original mirror and two real objects are retained: A ('
                + props['A']['category'].replace('_', ' ') + ') and B (' + props['B']['category'].replace('_', ' ') + ').',
            'constraints': ['Use only robot mounted observations and existing guarded base motion controls.',
                'The original observation point remains the distance reference throughout; your moving base does not redefine it.',
                'Keep all original mirror and object poses and materials unchanged; avoid the visual-only props as physical obstacles.',
                'The original question and answer remain auxiliary; answering alone cannot satisfy the navigation goal.']}
        if operation == OPERATIONS[2]:
            public['constraints'][1] = ('The original observation point remains the source question reference; '
                'the pair patrol uses the named A and B objects and does not reinterpret that question.')
        warnings = ['assumption_review_required', 'physical_qualification_pending', 'native_route_unverified',
                    'mounted_reflection_visibility_unverified', 'source_point_screen_is_not_robot_clearance']
        assumptions = [{'doubt': 'Saved camera locations certify neither a full R1 footprint nor an executable route.',
            'chosen_assumption': 'Use the original observer XY when the point screen passes, otherwise the nearest source-saved orbit point (corner fallback); qualify the unchanged native room, floor and full robot bounds.',
            'alternatives': ['Select another source-evidenced stance after native measurement.', 'Retain a qualification failure for Review.'],
            'scope': source.relative_path, 'risk': 'Furniture, room boundaries or visual-only geometry may block the proposed stance or route.'},
            {'doubt': 'The mounted R1 image and root height are different from the original camera at its declared height.',
             'chosen_assumption': 'Keep the complete original source reference unchanged and private, and declare the new navigation task relative to that fixed point; native prop roots determine ranking, never the QA answer.',
             'alternatives': ['Calibrate a mounted reference separately.', 'Use an optical survey recipe with no distance ranking.'],
             'scope': 'All category candidates', 'risk': 'Mounted reflected visibility and target recognition remain unverified, especially when the stance is shifted.'},
            {'doubt': 'Observer height gives a source floor elevation, not measured robot support.',
             'chosen_assumption': 'Initialize only the existing R1 at source-derived floor height plus 0.02 m; require measured native floor contacts and height consistency.',
             'alternatives': ['Revise the stance from measured original floor geometry.', 'Publish the failed native start.'],
             'scope': source.relative_path, 'risk': 'Uneven floors or startup settling may invalidate the proposal.'}]
        if operation == OPERATIONS[2]:
            assumptions[1]['chosen_assumption'] = ('Keep the complete original source reference unchanged and private. '
                'Use a declared A-then-B physical patrol for shifted starts or close distance margins; closer-object '
                'recipes require the original observer XY and an unambiguous source margin. No QA answer selects success.')
            warnings.append('pair_patrol_preserves_original_distance_qa')
        if gap <= 2 * parameters['source_pose_tolerance_m'] + parameters['minimum_native_distance_margin_m']:
            warnings.append('near_tied_original_observer_distances')
            assumptions.append({'doubt': f'The original root-distance margin is only {gap:.9f} m.',
                'chosen_assumption': 'Keep original distance QA verbatim; use an A-then-B patrol independent of ranking for the default recipe. Explicit nearer recipes require a larger source margin and measured ranking at every sample.',
                'alternatives': ['Use a separately declared pair-survey operation.', 'Keep the ambiguous source as a native qualification diagnostic.'],
                'scope': source.relative_path, 'risk': 'Float precision or pose drift can change the auxiliary QA ranking; patrol success cannot certify that answer.'})
        if proposal['family'] != 'original_observer':
            warnings.append('source_view_stance_fallback')
        expected_answer = selected.rsplit('_', 1)[1].upper()
        options = raw.get('options_json')
        options = json.loads(options) if isinstance(options, str) else options
        qa_consistent = (raw.get('answer') == q.get('answer') == expected_answer
                         and raw.get('question') == q.get('question') and options == ['A', 'B'])
        if not qa_consistent:
            warnings.append('source_qa_geometry_conflict')
            assumptions.append({'doubt': 'Source QA annotations disagree with preserved geometry or each other.',
                'chosen_assumption': 'Preserve annotations verbatim; select the physical goal from independent measured root distances.',
                'alternatives': ['Keep source-only diagnostics.', 'Request source annotation review without editing source data.'],
                'scope': source.relative_path, 'risk': 'Physical goal identity may differ from the historical answer.'})
        task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings,
            patches=patches, capabilities=['base_translation', 'base_rotation'],
            qualification={'on_failure': 'skip', 'initial_visibility': 'none', 'target_movable_collidable': False})
        return {'scene_definition': original_scene(source, context, bindings=bindings, patches=patches),
            'task_definition': task, 'status': 'generated', 'reuse_level': 'L2', 'qa_preserved': True,
            'operation_recipe': operation, 'stance_proposal': proposal,
            'original_reference': {'camera_pose': camera, 'observer_position': camera['position'],
                                   'scope': 'Fixed original image reference; never replaced by the moving robot.'},
            'source_geometry_audit': {'root_distances_m': distances, 'distance_margin_m': gap,
                                      'qa_geometry_consistent': qa_consistent},
            'change_summary': ['Retain original mirror, both props, materials, scales, poses, floor, reference and private QA.',
                               'Declare only a reversible robot initialization pose; preserve the empty original variant.',
                               'Add real nearer-object approach/return or ordered A-then-B patrol/return with private native terminal evidence.'],
            'warning_codes': warnings, 'assumptions': assumptions,
            'limitations': ['CPU generation is not native loading, route, reflection visibility or physical success evidence.',
                'Visual-only optical props remain unchanged for navigation. No grasping, physics activation, support addition or new asset is needed.',
                'Native root distance follows the source formula; it does not measure closest mesh surface or apparent mirror depth.',
                'Conservative native body/footprint AABB clearance may reject otherwise navigable paths.',
                'No physical episode, task latency or execution throughput has been measured by this agent.'],
            'attempted_repairs': ['Source-evidenced robot-only stance; no optical relocation or source mechanism change.'],
            'next_step': 'Main runner captures and publishes every candidate or diagnostic; native qualification and guarded execution remain required.'}

    def convert_all(self, sources, context):
        manifest = super().convert_all(sorted(sources, key=lambda source: source.relative_path), context)
        for row in manifest['records']:
            if row['status'] == 'generated':
                continue
            row['limitations'] = ['No physical task was generated for this revision. Source-only images remain diagnostics.',
                'The absence of a source-evidenced stance under this conservative rule does not prove that every native L1/L2 route is impossible.']
            row['assumptions'] = [{'doubt': 'A source camera point screen is incomplete evidence of native robot stance and route feasibility.',
                'chosen_assumption': 'Retain an explicit diagnostic instead of inventing an unevidenced stance or relocating the optical scene.',
                'alternatives': ['Measure original native floor and robot footprint to propose another reversible stance.',
                                 'Keep this source as a source-only diagnostic if no faithful operation can be qualified.'],
                'scope': row['source_identity'], 'risk': 'A valid native stance may be omitted by the conservative screen; diagnostic status is not a universal impossibility claim.'}]
            if 'assumption_review_required' not in row['warning_codes']:
                row['warning_codes'].append('assumption_review_required')
            row['attempted_repairs'] = ['Inspect original observer, saved orbit and saved corner stances without changing source geometry.']
            row['next_step'] = 'Publish a source-only diagnostic through the main runner. Measure native footprint and floor before revising the stance rule; retain source QA and optical geometry.'
            write_json(Path(row['diagnostic_dir']) / 'record.json', row)
        manifest['failure_distribution'] = dict(Counter(row['failure_reason'] for row in manifest['records'] if row['status'] != 'generated'))
        manifest['operation_distribution'] = dict(Counter(row.get('operation_recipe', 'diagnostic') for row in manifest['records']))
        manifest['warning_distribution'] = dict(Counter(code for row in manifest['records'] for code in row['warning_codes']))
        manifest['sample_ids'] = self.sample(manifest)
        write_json(context.output_root / 'batch_manifest.json', manifest)
        return manifest

    def sample(self, manifest, policy=None):
        policy = policy or {}
        count = min(int(policy.get('count', 5)), len(manifest['records']))
        rows = sorted(manifest['records'], key=lambda row: row['candidate_id'])
        selected = []
        groups = [[r for r in rows if r['status'] != 'generated'],
                  [r for r in rows if r['candidate_id'] in policy.get('previous_failures', [])]]
        groups += [sorted([r for r in rows if r.get('operation_recipe') == operation],
                          key=lambda r: r.get('source_geometry_audit', {}).get('distance_margin_m', math.inf))
                   for operation in OPERATIONS]
        groups += [[r for r in rows if 'source_view_stance_fallback' in r['warning_codes']]]
        for group in groups:
            candidate = next((r['candidate_id'] for r in group if r['candidate_id'] not in selected), None)
            if candidate and len(selected) < count:
                selected.append(candidate)
        for candidate in super().sample(manifest, policy):
            if candidate not in selected and len(selected) < count:
                selected.append(candidate)
        return selected


def _pose(obj):
    from scipy.spatial.transform import Rotation
    position, quaternion = obj.get_position_orientation()
    position, quaternion = _vector(position, 3), _vector(quaternion, 4)
    if np.linalg.norm(quaternion) < 1e-8:
        raise ValueError('undefined_native_orientation')
    return position, Rotation.from_quat(quaternion)


def _bounds(obj):
    low, high = (_vector(value, 3) for value in obj.aabb)
    if np.any(high <= low):
        raise ValueError('invalid_native_bounds')
    return low, high


def _objects(session, config):
    if (config.get('type') != 'category_operation' or module_identity(config.get('module')) != MODULE
            or config.get('operation') not in OPERATIONS
            or (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name)
            or set(config.get('binding_keys', [])) != set(BINDING_KEYS)
            or set(config.get('bindings', {})) != set(BINDING_KEYS)):
        raise ValueError('native_scene_or_operation_contract_mismatch')
    objects = {key: session.env.scene.object_registry('name', config['bindings'][key]) for key in BINDING_KEYS}
    if any(obj is None for obj in objects.values()):
        raise ValueError('missing_native_original_binding')
    return objects


def measure_native_state(session, config):
    """Read transforms, chassis velocities, native contacts, bounds and room map."""
    from scipy.spatial.transform import Rotation
    objects, p = _objects(session, config), config['parameters']
    robot = session.robot
    point, rotation = _pose(robot)
    chassis = robot.links[robot.base_footprint_link_name]
    chassis_low, chassis_high = _bounds(chassis)
    body_low, body_high = _bounds(robot)
    linear, angular = (_vector(value, 3) for value in (chassis.get_linear_velocity(), chassis.get_angular_velocity()))
    optical, clearances, preserved = {}, {}, True
    for key in OPTICAL_KEYS:
        position, orientation = _pose(objects[key])
        source = p['source_poses'][key]
        drift = float(np.linalg.norm(position - _vector(source['position'], 3)))
        angle = math.degrees(float((orientation.inv() * Rotation.from_quat(source['quaternion_xyzw'])).magnitude()))
        preserved &= drift <= p['source_pose_tolerance_m'] and angle <= p['source_angle_tolerance_deg']
        low, high = _bounds(objects[key])
        body_gap = np.maximum(np.maximum(low - body_high, body_low - high), 0.)
        footprint_gap = np.maximum(np.maximum(low[:2] - chassis_high[:2], chassis_low[:2] - high[:2]), 0.)
        clearances[key] = {'body_gap_m': float(np.linalg.norm(body_gap)),
                           'footprint_xy_gap_m': float(np.linalg.norm(footprint_gap))}
        optical[key] = {'position': position.tolist(), 'quaternion_xyzw': orientation.as_quat().tolist(),
                        'source_drift_m': drift, 'source_angle_drift_deg': angle}
    selected, gap, observer_distances = distance_order(
        {key: optical[key]['position'] for key in ('prop_a', 'prop_b')}, p['source_observer_position'])
    ranking_valid = selected == p['source_nearer_binding'] and gap >= p['minimum_native_distance_margin_m']
    yaw = math.atan2(rotation.as_matrix()[1, 0], rotation.as_matrix()[0, 0])
    def heading(key):
        target = _array(optical[key]['position'])
        return _angle_deg(yaw, math.atan2(target[1] - point[1], target[0] - point[0]))
    candidate_distances = {key: float(np.linalg.norm(point[:2] - _array(optical[key]['position'])[:2]))
                           for key in ('prop_a', 'prop_b')}
    source_floor = objects['source_floor']
    if source_floor.visual_only or getattr(source_floor, 'category', None) not in ('floor', 'floors'):
        raise ValueError('original_native_floor_is_not_collidable')
    within = lambda body, root: body == root or body.startswith(root + '/')
    floor_contacts, other_contacts = [], []
    for contact in robot.contact_list():
        if legal_background_contact(session, contact.body0, contact.body1):
            continue
        external = [str(body) for body in (contact.body0, contact.body1) if not within(str(body), robot.prim_path)]
        if not external:
            continue
        if all(within(body, source_floor.prim_path) for body in external):
            pos, normal = _vector(contact.position, 3), _vector(contact.normal, 3)
            floor_contacts.append({'position': pos.tolist(), 'normal': normal.tolist()})
        else:
            other_contacts.extend(external)
    supported = bool(floor_contacts and all(abs(row['position'][2] - p['floor_z_m']) <= p['floor_height_tolerance_m']
                                            and abs(row['normal'][2]) >= .5 for row in floor_contacts))
    footprint_points = [point[:2], chassis_low[:2], chassis_high[:2],
                        [chassis_low[0], chassis_high[1]], [chassis_high[0], chassis_low[1]]]
    rooms = [session.env.scene.seg_map.get_room_instance_by_point(_array(xy)) for xy in footprint_points]
    clear = all(min(row.values()) >= p['visual_asset_clearance_m'] for row in clearances.values())
    normal = _pose(objects['mirror'])[1].as_matrix()[:2, 0]
    source_normal = _vector(p['source_mirror_normal_xy'], 2)
    normal_preserved = abs(float(normal @ (source_normal / np.linalg.norm(source_normal)))) >= math.cos(math.radians(p['source_angle_tolerance_deg']))
    upright = float(rotation.as_matrix()[2, 2]) >= p['minimum_up_dot']
    ranking_required = config['operation'] != OPERATIONS[2]
    valid = bool(supported and not other_contacts and all(room == session.room_name for room in rooms)
                 and upright and clear and preserved and (ranking_valid or not ranking_required) and normal_preserved)
    return {'position': point.tolist(), 'yaw_rad': yaw, 'up_dot': float(rotation.as_matrix()[2, 2]),
        'optical_poses': optical, 'optical_poses_preserved': bool(preserved),
        'fixed_observer_root_distances_m': observer_distances, 'native_nearer_binding': selected,
        'native_distance_margin_m': gap, 'source_distance_ranking_preserved': bool(ranking_valid),
        'native_reflected_roots': {key: reflected_point(optical[key]['position'], optical['mirror']['position'], normal)
                                  for key in ('prop_a', 'prop_b')},
        'candidate_distances_xy_m': candidate_distances, 'target_heading_error_deg': heading(selected),
        'object_heading_errors_deg': {key: heading(key) for key in ('prop_a', 'prop_b')},
        'mirror_heading_error_deg': heading('mirror'), 'linear_speed_m_s': float(np.linalg.norm(linear)),
        'angular_speed_rad_s': float(np.linalg.norm(angular)), 'visual_asset_clearance': clearances,
        'floor_contacts': floor_contacts, 'nonfloor_contacts': other_contacts, 'footprint_rooms': rooms,
        'native_floor_supported': supported, 'state_valid': valid}


def qualify_operation(session, executor, config, bound):
    if config.get('operation') == ACTION_OPERATION:
        from .specular_reflection_reflection_authoring import qualify_reflection_action
        return qualify_reflection_action(session, executor, config, bound)
    if config.get('operation') == ANSWER_OPERATION:
        from .specular_reflection_reflection_authoring import _qualify_observation
        return _qualify_observation(session, executor, config, bound)
    """Native precondition screen only; never step, act, set poses or use QA."""
    try:
        objects = _objects(session, config)
        measured = measure_native_state(session, config)
        p = config['parameters']
        position, quaternion = session.robot.get_position_orientation()
        repair = verified_repaired_start(session, session.room_name, _array(position), _array(quaternion),
            position_tolerance_m=p['start_tolerance_m'], heading_tolerance_deg=p['heading_tolerance_deg'])
        checks = {'bound_native_objects': all(bound.get(key) is objects[key] for key in BINDING_KEYS),
            'native_r1': type(session.robot).__name__ == 'R1' and not session.robot.visual_only,
            'guarded_base_capabilities': bool(executor.capabilities.base_translation and executor.capabilities.base_rotation),
            'native_holonomic_base': type(session.robot.controllers.get('base')).__name__ == 'HolonomicBaseJointController',
            'native_safe_start': measured['state_valid'],
            'declared_or_verified_initialized_stance': bool(math.dist(measured['position'][:2], p['proposed_start_xy'])
                <= p['start_tolerance_m'] or repair['accepted']),
            'facing_original_mirror': measured['mirror_heading_error_deg'] <= p['heading_tolerance_deg']}
        accepted = bool(all(checks.values()))
        return {'accepted': accepted, 'checks': checks, 'native_state': measured, 'verified_repaired_start': repair,
                'reason': None if accepted else 'native_mirror_distance_start_unqualified',
                'route_qualified': False, 'reflected_visibility_qualified': False}
    except (ValueError, RuntimeError, KeyError, AttributeError, TypeError, IndexError) as error:
        return {'accepted': False, 'reason': 'native_mirror_distance_measurement_unavailable',
                'exception_summary': f'{type(error).__name__}: {error}', 'route_qualified': False}



def _completion_result(items):
    """Report declared physical obligations without distance-shaped rewards."""
    completed = sum(bool(item['completed']) for item in items)
    total = len(items)
    score = completed / total
    return {'scoring_version': 'task_completion_v2', 'evaluation_validity': 'complete',
            'completion_items': items, 'completed_goals': completed, 'total_goals': total,
            'completion_score': score, 'score_percent': 100. * score}


class MirrorDistanceEvaluator:
    """Ordered native stops with monotonic physics-time and motion sampling."""
    def __init__(self, session, controller, config, output):
        _objects(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.p, self.output = config['parameters'], Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.active, self.episode = False, 0

    def _time(self):
        now = float(self.session.pipeline.og.sim.current_time)
        if not math.isfinite(now):
            raise ValueError('native_clock_unavailable')
        return now

    def begin_episode(self, observation):
        self.active = False
        qualified = qualify_operation(self.session, self.controller, self.config, _objects(self.session, self.config))
        if not qualified['accepted']:
            raise ValueError(qualified['reason'])
        self.initial = qualified
        state = qualified['native_state']
        self.start = self.previous = _array(state['position'])
        self.previous_yaw, self.previous_time = state['yaw_rad'], self._time()
        self.travel, self.excursion, self.samples = 0., 0., 0
        self.approached, self.faults, self.visits = False, set(), []
        self.completed_targets = []
        self.approach_samples, self.terminal_samples = 0, 0
        first = self.p['approach_order'][0]
        self.left_approach = state['candidate_distances_xy_m'][first] >= self.p['departure_distance_m']
        self.episode += 1
        self.active = True
        return self.record(observation, action=None)

    def _approach(self, state, selected=None):
        distances = state['candidate_distances_xy_m']
        selected = selected or self.p['approach_order'][0]
        target = distances[selected]
        return bool(self.left_approach and state['state_valid']
            and self.p['approach_minimum_m'] <= target <= self.p['approach_maximum_m']
            and all(target + self.p['candidate_distance_margin_m'] <= distance for key, distance in distances.items() if key != selected)
            and state['object_heading_errors_deg'][selected] <= self.p['heading_tolerance_deg'])

    def _settled(self, state):
        return (state['linear_speed_m_s'] <= self.p['maximum_stopped_speed_m_s']
                and state['angular_speed_rad_s'] <= self.p['maximum_stopped_angular_speed_rad_s'])

    def _terminal(self, state):
        if self.config['operation'] == OPERATIONS[0]:
            return self._approach(state)
        return bool(self.approached and state['state_valid']
                    and math.dist(state['position'][:2], self.start[:2]) <= self.p['return_tolerance_m']
                    and state['mirror_heading_error_deg'] <= self.p['heading_tolerance_deg'])

    def sample_step(self):
        if not self.active:
            return
        state, now = measure_native_state(self.session, self.config), self._time()
        point = _array(state['position'])
        elapsed, distance = now - self.previous_time, float(np.linalg.norm(point[:2] - self.previous[:2]))
        cfg = self.controller.config
        if (elapsed <= 0 or distance > cfg.max_translation_speed_m_s * max(elapsed, 0.) + cfg.translation_tolerance_m
                or _angle_deg(state['yaw_rad'], self.previous_yaw) > cfg.max_rotation_speed_deg_s * max(elapsed, 0.) + cfg.rotation_tolerance_deg):
            self.faults.add('native_clock_or_motion_continuity_failed')
        if (abs(point[2] - self.previous[2]) > self.p['maximum_root_z_change_m']
                or abs(point[2] - self.start[2]) > self.p['maximum_root_z_change_m'] or not state['state_valid']):
            self.faults.add('native_support_clearance_or_optical_preservation_failed')
        self.samples += 1
        self.travel += distance
        self.excursion = max(self.excursion, float(np.linalg.norm(point[:2] - self.start[:2])))
        self.left_approach |= state['candidate_distances_xy_m'][self.p['approach_order'][0]] >= self.p['departure_distance_m']
        moved = self.travel >= self.p['minimum_travel_m'] and self.excursion >= self.p['minimum_excursion_m']
        stage = len(self.completed_targets)
        newly_approached = False
        if stage < len(self.p['approach_order']):
            selected = self.p['approach_order'][stage]
            separation = stage == 0 or math.dist(point[:2], self.visits[-1]['position'][:2]) >= self.p['minimum_between_stations_m']
            approach = self._approach(state, selected) and self._settled(state) and moved and separation and not self.faults
            self.approach_samples = self.approach_samples + 1 if approach else 0
            if self.approach_samples >= self.p['stability_steps']:
                self.completed_targets.append(selected)
                self.visits.append({'stage': selected + '_approach', 'time_s': now, 'position': state['position']})
                self.approach_samples = 0
                self.approached = len(self.completed_targets) == len(self.p['approach_order'])
                newly_approached = self.approached
        terminal = self._terminal(state) and self._settled(state) and self.approached and not self.faults
        # Return samples must follow the completed approach in a later physics step.
        if self.config['operation'] != OPERATIONS[0] and newly_approached:
            terminal = False
        self.terminal_samples = self.terminal_samples + 1 if terminal else 0
        self.previous, self.previous_time, self.previous_yaw = point, now, state['yaw_rad']
        with (self.output / 'physical_trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps({'sample': self.samples, 'time_s': now, 'native_state': state,
                'travel_m': self.travel, 'excursion_m': self.excursion, 'faults': sorted(self.faults)}, allow_nan=False) + '\n')

    def record(self, observation, *, action):
        if not self.active:
            return None
        state = measure_native_state(self.session, self.config)
        if (np.linalg.norm(_array(state['position']) - self.previous) > 1e-6
                or _angle_deg(state['yaw_rad'], self.previous_yaw) > 1e-4):
            self.faults.add('unsampled_native_motion')
        if not state['state_valid']:
            self.faults.add('native_support_clearance_or_optical_preservation_failed')
        criteria = {'qualified_start': self.initial['accepted'], 'continuous_safe_native_motion': not self.faults,
            'positive_measured_travel': self.samples > 0 and self.travel >= self.p['minimum_travel_m'],
            'minimum_excursion': self.excursion >= self.p['minimum_excursion_m'],
            'stable_correct_object_approach': self.approached, 'native_terminal_valid': state['state_valid'],
            'requested_terminal_goal': self._terminal(state), 'terminal_at_rest': self._settled(state),
            'terminal_stability': self.terminal_samples >= self.p['stability_steps']}
        value = {'schema': 'private_native_mirror_distance_v1', 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step, 'action': action,
            'task_success': bool(all(criteria.values())), 'criteria': criteria, 'native_state': state,
            'physical_travel_m': self.travel, 'maximum_excursion_m': self.excursion,
            'control_step_samples': self.samples, 'terminal_stability_samples': self.terminal_samples,
            'ordered_visits': self.visits, 'motion_faults': sorted(self.faults), 'configuration': self.config,
            'original_qa_used_for_success': False, 'reflected_visibility_certified': False}
        returning = self.config['operation'] != OPERATIONS[0]
        items = [{'id': f'approach_{index + 1}_{target}',
            'completed': bool(not self.faults and (len(self.completed_targets) > index if returning
                else all(criteria.values()))), 'evaluation_validity': 'valid',
            'kind': 'historical_checkpoint' if returning else 'terminal_condition',
            'evidence': {'visit': self.visits[index] if index < len(self.visits) else None,
                'safety_faults': sorted(self.faults)}}
            for index, target in enumerate(self.p['approach_order'])]
        if returning:
            items.append({'id': 'return_and_stop', 'completed': bool(all(criteria.values())),
                'evaluation_validity': 'valid', 'kind': 'terminal_condition', 'evidence': criteria})
        value.update(_completion_result(items))
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    if config.get('operation') == ACTION_OPERATION:
        from .specular_reflection_reflection_authoring import ReflectionVerificationEvaluator
        return ReflectionVerificationEvaluator(session, controller, config, output)
    if config.get('operation') == ANSWER_OPERATION:
        from .specular_reflection_reflection_authoring import ReflectionObservationEvaluator
        return ReflectionObservationEvaluator(session, controller, config, output)
    return MirrorDistanceEvaluator(session, controller, config, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment', required=True, type=Path)
    args = parser.parse_args()
    assignment = json.loads(args.assignment.read_text())
    corpus, root = Path(assignment['corpus']), Path(assignment['batch_root'])
    sources = [SourceInstance(path, str(path.relative_to(corpus)), *CategoryConverter.category)
               for path in sorted((corpus / CategoryConverter.category[0] / CategoryConverter.category[1]).rglob('*.json'))]
    manifest = CategoryConverter().convert_all(sources, ConversionContext(root, seed=0, advice=assignment['advice']))
    print(json.dumps({key: manifest[key] for key in ('source_count', 'counts', 'failure_distribution', 'operation_distribution', 'sample_ids')}))


if __name__ == '__main__':
    main()
