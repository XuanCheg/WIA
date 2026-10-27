"""Preserve occluded layouts and require discovery of the disclosed full inventory.

Run CPU conversion from the repository root with the behavior Python environment:
CategoryConverter().convert_all(sources, ConversionContext(output_root)).
Native qualification/evaluation hooks are read-only and run in normal episodes.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from wia.authoring.conversion import ConverterBase, map_records, original_scene, physical_task
from wia.tasks.background_policy import legal_background_contact
from wia.authoring.io import write_json


MODULE = 'core.esi_converters.enumerative_perception_counting_w_occlusion'
OPERATIONS = ('occluder_side_patrol', 'occluder_side_patrol_return')
PARAMETERS = {'minimum_standoff_m': 0.55, 'maximum_standoff_m': 1.25,
    'target_side_margin_m': 0.20, 'chassis_anchor_clearance_m': 0.05,
    'minimum_departure_m': 0.50, 'minimum_travel_m': 1.0,
    'return_radius_m': 0.30, 'stability_steps': 3, 'maximum_visit_speed_m_s': 0.05,
    'maximum_visit_angular_speed_rad_s': 0.10,
    'position_preservation_m': 0.02, 'orientation_preservation_deg': 5.0}


def _completion_result(value, items, *, available=True):
    """Expose required physical obligations without averaging safety predicates."""
    completion_items = [{'id': key, 'completed': bool(done),
        'evaluation_validity': 'valid' if available else 'unavailable',
        'kind': kind, 'evidence': evidence} for key, done, kind, evidence in items]
    completed = sum(item['completed'] for item in completion_items)
    total = len(completion_items)
    score = completed / total if total and available else None
    value.update(scoring_version='task_completion_v2',
        evaluation_validity='complete' if available else 'unavailable',
        completion_items=completion_items, completed_goals=completed, total_goals=total,
        completion_score=score, score_percent=100. * score if score is not None else None)
    return value


def _array(value):
    from wia.tasks.interaction_setup import array
    return array(value)


def _vector(value, size):
    array = _array(value)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError('Source/native vectors must have the declared finite shape.')
    return array


def _box(value):
    low, high = (_vector(part, 3) for part in value)
    if np.any(high <= low):
        raise ValueError('Source/native AABB extents must be positive.')
    return low, high


def _public_category(raw):
    match = re.search(r'objects of category "([^"\n]+)"', raw.get('question', ''))
    return match.group(1) if match else None


def _source_layout(source):
    raw, payload = source.raw, source.payload
    question = payload['question_data']
    render = question['render']
    targets = render['resolved_objects']['targets']
    entries = question['case_metadata']['source_entries']
    if question.get('case') != 'hidden_by_others' or not targets or len(targets) != len(entries):
        raise ValueError('Original target/anchor mechanism is incomplete.')
    if render['resolved_objects'].get('confusers') or render['resolved_objects'].get('containers'):
        raise ValueError('This patrol rule requires the original hidden-by-furniture case.')
    public_category = _public_category(raw)
    if public_category is None or question.get('count_target') != public_category:
        raise ValueError('Public target category does not match the original construction.')
    cameras = map_records(render['camera_poses'])
    camera = cameras.get(Path(render['image']).name) or next(iter(cameras.values()))
    _vector(camera['position'], 3)
    if not np.linalg.norm(_vector(camera['quaternion_xyzw'], 4)):
        raise ValueError('Source reference camera quaternion is invalid.')
    bounds = _vector(render['room_bbox']['expanded_bbox_world_xy'], 4)
    if np.any(bounds[2:] <= bounds[:2]):
        raise ValueError('Source room footprint is invalid.')
    # The original generator uses floor_top + BALL_RADIUS (0.035 m), not
    # the rendered mesh's slightly penetrating visual-only AABB bottom.
    floor_levels = [float(_vector(t['requested_position'], 3)[2] - 0.035) for t in targets]
    if max(floor_levels) - min(floor_levels) > 1e-4 or not payload.get('floor_name'):
        raise ValueError('Original common floor level/binding is unresolved.')
    pairs, bindings, expected = [], {}, {}
    names = set()
    for index, target in enumerate(targets):
        key = f'target_{index}'
        if not all(target.get(field) for field in ('name', 'category', 'model')) or target['name'] in names:
            raise ValueError('Original target identity is missing or duplicated.')
        names.add(target['name'])
        if target['category'] != public_category or not np.linalg.norm(_vector(target['quaternion_xyzw'], 4)):
            raise ValueError('Original target category/orientation is unresolved.')
        position = _vector(target['requested_position'], 3)
        matches = [entry for entry in entries if any(np.linalg.norm(_vector(point, 3) - position) < 1e-4
            for point in entry.get('ball_positions', []))]
        if len(matches) != 1:
            raise ValueError('Target-to-original-occluder provenance is ambiguous.')
        anchor = matches[0]['metadata']['anchor_object']
        if not anchor.get('name') or not anchor.get('category'):
            raise ValueError('Original occluder identity is incomplete.')
        _box(anchor['bbox'])
        _box(target['bbox'])
        anchor_key = next((key for key, binding in bindings.items() if binding['name'] == anchor['name']), None)
        if anchor_key is None:
            anchor_key = f'anchor_{index}'
            bindings[anchor_key] = {'name': anchor['name'], 'category': anchor['category']}
            expected[anchor_key] = {'bbox': anchor['bbox']}
        bindings[key] = {field: target[field] for field in ('name', 'category', 'model')}
        expected[key] = {field: target[field] for field in ('position', 'quaternion_xyzw', 'bbox')}
        pairs.append({'target': key, 'anchor': anchor_key})
    bindings['floor'] = {'name': payload['floor_name'], 'category': 'floors'}
    return {'raw': raw, 'question': question, 'camera': camera, 'bounds': bounds,
        'floor_z': floor_levels[0], 'pairs': pairs, 'bindings': bindings, 'expected': expected}


def _start_pose(layout):
    """Screen a deterministic robot-only start; never certify native clearance."""
    bounds, camera = layout['bounds'], layout['camera']
    margin = 0.55
    low, high = bounds[:2] + margin, bounds[2:] - margin
    if np.any(high < low):
        return None
    camera_xy = _vector(camera['position'], 3)[:2]
    points = [np.clip(camera_xy, low, high)]
    points.extend(np.array([x, y]) for x in np.arange(low[0], high[0] + 1e-8, 0.20)
        for y in np.arange(low[1], high[1] + 1e-8, 0.20))
    boxes = [_box(item['bbox']) for item in layout['expected'].values()]
    safe = [point for point in points if all(np.linalg.norm(point - np.clip(point, lo[:2], hi[:2]))
        >= margin for lo, hi in boxes)]
    if not safe:
        return None
    xy = min(safe, key=lambda point: (float(np.linalg.norm(point - camera_xy)), float(point[0]), float(point[1])))
    # Face the same room-centre direction as the original reference, without
    # moving the reference camera or borrowing its ground-level closeups.
    look = _vector(camera.get('look_target', [* ((bounds[:2] + bounds[2:]) / 2), layout['floor_z']]), 3)
    delta = look[:2] - xy
    return {'operation': 'robot_pose', 'position': [float(xy[0]), float(xy[1]), layout['floor_z']],
        'yaw_deg': float(np.rad2deg(np.arctan2(delta[1], delta[0]))), 'head_pitch_deg': 45.0}


class CategoryConverter(ConverterBase):
    category = ('Enumerative Perception', 'Counting w Occlusion')
    version = '2.0.0_task_edit'

    def inspect(self, source):
        if (source.big_task, source.small_task) != self.category or (
                source.raw.get('big_task'), source.raw.get('small_task')) != self.category:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2: category_mismatch'}
        try:
            _source_layout(source)
        except (ValueError, KeyError, TypeError, StopIteration) as error:
            return {'eligible': False, 'reason': f'not_convertible_under_L1_L2: {error}',
                'warning_codes': ['source_mechanism_unresolved']}
        return {'eligible': True, 'reason': 'Original targets and furniture support an access patrol.'}

    def convert(self, source, context):
        if context.config.get('legacy_occlusion_patrol'):
            return self.convert_legacy(source, context)
        from .partial_occlusion import visual_discovery_task
        layout = _source_layout(source)
        targets = [pair['target'] for pair in layout['pairs']]
        category = _public_category(source.raw).replace('_', ' ')
        public = {'objective': f'Find all {len(targets)} distinct {category} objects hidden around the original furniture. '
            + 'Use multiple robot observations as needed and track each distinct object across views.',
            'success_criteria': 'Every original target must appear in a delivered mounted-camera observation; '
                + 'repeated views of the same target count once. The total is supplied, so this task does not score count estimation.',
            'environment': 'Original target locations, supporting floor, occluding furniture and source reference are retained.',
            'constraints': ['Keep semantic occluders, task targets and supports in place.',
                'No prescribed standoff, travel distance or return trip is required.']}
        start = _start_pose(layout)
        patches = [] if start is None else [start]
        for key in targets:
            patches += [{'operation': 'object_physics', 'phase': phase, 'binding': key, 'visual_only': False}
                        for phase in ('before_construction', 'after_postprocessing')]
        result = visual_discovery_task(source, context, bindings=layout['bindings'], target_keys=targets,
            public=public, patches=patches,
            preserved_relations=['target-to-occluder relations', 'source floor support', 'original observer and target scale'],
            source_evidence={'pairs': layout['pairs'], 'expected': layout['expected'], 'source_camera': layout['camera']})
        result['warning_codes'].append('native_fallen_target_check_required')
        return result

    def convert_legacy(self, source, context):
        layout = _source_layout(source)
        start = _start_pose(layout)
        if start is None:
            return {'scene_definition': original_scene(source, context, bindings=layout['bindings']),
                'task_definition': None, 'status': 'conversion_failed', 'failure_stage': 'robot_start_screen',
                'failure_reason': 'not_convertible_under_L1_L2: no_source_screened_robot_start',
                'warning_codes': ['robot_start_unresolved'], 'limitations': ['No native patrol is qualified.'],
                'next_step': 'Inspect the original diagnostic scene and measure a source-faithful R1 stance.'}
        operation = context.config.get('occlusion_operation')
        if operation is None:
            index = int(hashlib.sha256(f'{context.seed}:{source.relative_path}'.encode()).hexdigest()[:8], 16)
            operation = OPERATIONS[index % len(OPERATIONS)]
        if operation not in OPERATIONS:
            raise ValueError('Unsupported occlusion patrol recipe.')
        public_category = _public_category(layout['raw']).replace('_', ' ')
        objective = f'Navigate around the original furniture and approach every original {public_category} '
        objective += 'on its initialized accessible support. Stop briefly at each safe approach.'
        criteria = 'Reach a safe standoff 0.55 to 1.25 m from each object centre, '
        criteria += 'after moving at least 0.50 m from departure. '
        criteria += 'Complete at least 1.0 m of real travel and remain clear of furniture with floor support.'
        if operation == 'occluder_side_patrol_return':
            objective += ' After the patrol, return to your departure position and park.'
            criteria += ' Finish stopped within 0.30 m of your measured departure position.'
        else:
            criteria += ' Finish stopped at a qualified approach to one of the objects.'
        public = {'objective': objective,
            'success_criteria': criteria,
            'environment': 'The original room and occluding furniture are retained. Portable targets are initialized '
                'on measured accessible supports in groups associated with their original furniture regions. '
                'Elevation and support placement may change source occlusion and reference visibility. '
                'Auxiliary question about the initial arrangement: ' + layout['raw']['question'],
            'constraints': ['Use existing guarded robot controls; move noncritical native movable furniture only with arm/gripper actions.',
                'Keep semantic occluders, task supports and the building in place; preserve initialized targets, dimensions and groups. Native movable noncritical furniture may be moved with guarded arm/gripper actions.',
                'Use robot observations to find objects; a count answer alone does not complete the patrol.']}
        parameters = {**PARAMETERS, 'pairs': layout['pairs'], 'expected': layout['expected'], 'accessible_placement': True}
        target_keys = [pair['target'] for pair in layout['pairs']]
        groups = [[pair['target'] for pair in layout['pairs'] if pair['anchor'] == anchor]
                  for anchor in dict.fromkeys(pair['anchor'] for pair in layout['pairs'])]
        patches = [start]
        for key in target_keys:
            patches.extend([{'operation': 'object_physics', 'phase': 'before_construction',
                             'binding': key, 'visual_only': False, 'fixed_base': False},
                            {'operation': 'object_physics', 'phase': 'after_postprocessing',
                             'binding': key, 'visual_only': False}])
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': operation,
            'binding_keys': list(layout['bindings']), 'parameters': parameters}
        task = physical_task(source, context, public=public, evaluation=evaluation,
            bindings=layout['bindings'], patches=patches, capabilities=['base_translation', 'base_rotation'],
            qualification={'on_failure': 'skip', 'initial_visibility': 'none', 'target_movable_collidable': False})
        task['defaults']['max_steps'] = 600
        warnings = ['assumption_review_required', 'native_patrol_unverified', 'source_start_screen_advisory',
            'body_view_count_coverage_unverified', 'source_occlusion_adapted_for_accessibility',
            'accessible_target_support_unverified']
        if str(len(layout['pairs'])) != str(layout['raw'].get('answer')):
            warnings.append('source_count_annotation_conflict')
        return {'scene_definition': original_scene(source, context, bindings=layout['bindings'], patches=patches),
            'task_definition': task, 'status': 'generated', 'reuse_level': 'L2',
            'operation_recipe': operation, 'available_operation_recipes': list(OPERATIONS),
            'target_category': _public_category(layout['raw']), 'target_count_private': len(layout['pairs']),
            'interaction_policy': {'mode': 'inspection', 'group_bindings': target_keys,
                'manipulation_bindings': target_keys, 'relocation_policy': 'independent_groups',
                'placement_groups': groups,
                'support_preferences': {pair['target']: pair['anchor'] for pair in layout['pairs']},
                'support_search': 'measured_surfaces', 'allow_fallback_table': True,
                'source_camera_policy': 'preserve', 'source_region': layout['raw']['room'],
                'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10,
                'maximum_stance_distance_m': .85,
                'warning_codes': ['accessible_target_support_unverified', 'source_occlusion_adapted_for_accessibility']},
            'change_summary': ['Retain native targets, furniture, reference camera and original private QA.',
                'Add a robot-only source-screened initialization and independent physical access patrol.',
                'Initialize dynamic collidable portable targets on accessible measured supports, grouped by original occluder region.',
                'Use measured target standoff after adaptation; historical configurations retain the original furniture-side predicate.'],
            'warning_codes': warnings, 'assumptions': [{
                'doubt': 'A free-camera stance and source AABBs do not certify R1 footprint, arm clearance or a route.',
                'chosen_assumption': 'Use the nearest deterministic room-grid stance with 0.55 m source '
                    'AABB clearance, the source floor level, and unchanged R1 assets/controls. '
                    'Native qualification rejects unsupported floor or body contact.',
                'alternatives': ['Keep the empty original variant for diagnostic review.',
                    'Measure a different robot-only start after native capture while preserving accepted target groups.'],
                'scope': source.relative_path,
                'risk': 'Other loaded furniture instances, walls, room raster boundaries or R1 arms may obstruct the start. '
                    'The fixed-base holonomic R1 does not freely settle; wheel/floor measurements must reject a bad height.'}, {
                'doubt': 'Original occlusion labels use an XY segment/AABB test and do not establish actual body-view visibility.',
                'chosen_assumption': 'Define physical access from measured target standoff after accessible support placement. '
                    'Keep original furniture as collision guards and count QA auxiliary; source occlusion may change.',
                'alternatives': ['Add measured mounted-camera pixel coverage if a read-only native '
                    'sensor protocol becomes available.', 'Retain unreachable instances as explicit Review diagnostics.'],
                'scope': 'All category patrol recipes.',
                'risk': 'A geometrically valid approach may not reveal the floor target to the enabled camera; '
                    'targets near walls may have no valid R1 approach.'}],
            'private_start_evidence': {'reference_camera': layout['camera'], 'proposed_robot_pose': start,
                'source_floor_z_m': layout['floor_z'], 'grid_step_m': 0.20, 'screen_margin_m': 0.55,
                'provenance': 'batch_counting_merge.py BALL_RADIUS=0.035, AGENT_CAMERA_HEIGHT_M=1.25; '
                    'question_data/render/resolved_objects and case_metadata/source_entries.'},
            'limitations': ['CPU generation is not native loading, navigation qualification or physical success.',
                'Patrol success measures access, not pixel visibility or the correctness of the original count answer.',
                'The 600-decision budget and standoff thresholds are declared proposals, not measured runtime guarantees.',
                'Native initial objects must match accepted per-group translated source geometry; later displacement or support loss invalidates patrol success.'],
            'next_step': 'Review all candidates, qualify native start/furniture state, then execute guarded patrols.'}

    def convert_all(self, sources, context):
        # Shared deterministic persistence preserves every failure and previous feedback.
        return super().convert_all(sorted(sources, key=lambda item: item.relative_path), context)

    def sample(self, manifest, policy=None):
        """Cover failures first, then target/count/start/recipe and room strata."""
        count = (policy or {}).get('count', 5)
        base = super().sample(manifest, policy)
        rows = {row['candidate_id']: row for row in manifest['records']}
        previous = set((policy or {}).get('previous_failures', []))
        selected = [key for key in base if rows[key]['status'] != 'generated' or key in previous][:count]
        def features(row):
            return {('category', row.get('target_category')), ('count', row.get('target_count_private')),
                ('recipe', row.get('operation_recipe')), ('scene', row.get('scene_model')),
                ('room', (row.get('scene_model'), row.get('room')))}
        weights = {'category': 5, 'count': 5, 'recipe': 3, 'scene': 1, 'room': 1}
        covered = set().union(*(features(rows[key]) for key in selected))
        remaining = sorted(manifest['records'], key=lambda row: (
            row.get('target_category') == 'club_sandwich', row['candidate_id']))
        while remaining and len(selected) < count:
            row = max(remaining, key=lambda row: sum(weights[kind] for kind, value in features(row) - covered))
            remaining.remove(row)
            if row['candidate_id'] not in selected:
                selected.append(row['candidate_id'])
                covered.update(features(row))
        return selected


def _validate_config(session, config):
    if config.get('operation') not in OPERATIONS or (config.get('scene'), config.get('room')) != (
            session.scene_name, session.room_name):
        raise ValueError('Patrol evaluation must match the original native scene and room.')
    parameters = config['parameters']
    if any(parameters.get(key) != value for key, value in PARAMETERS.items()) or not parameters.get('pairs'):
        raise ValueError('Use the declared conservative access-patrol thresholds and native pairs.')
    return parameters


def _native_objects(session, config):
    objects = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
    if any(obj is None for obj in objects.values()):
        raise ValueError('Original native patrol bindings are unavailable.')
    return objects


def _pose(obj):
    position, quaternion = obj.get_position_orientation()
    position, quaternion = _vector(position, 3), _vector(quaternion, 4)
    norm = np.linalg.norm(quaternion)
    if not norm:
        raise ValueError('Native object quaternion is invalid.')
    return {'position': position.tolist(), 'quaternion_xyzw': (quaternion / norm).tolist()}


def _quaternion_angle(first, second):
    first, second = _vector(first, 4), _vector(second, 4)
    if not np.linalg.norm(first) or not np.linalg.norm(second):
        raise ValueError('Native/source orientations must be nonzero.')
    cosine = np.clip(abs(np.dot(first / np.linalg.norm(first), second / np.linalg.norm(second))), 0.0, 1.0)
    return math.degrees(2 * math.acos(cosine))


def _native_state(session, config, objects):
    """Read actual poses, body footprint, source floor ray and contacts only."""
    from omnigibson.utils.sampling_utils import raytest
    from wia.tasks.scene_only import local_floor_surface

    robot = session.robot
    pose = _pose(robot)
    xy = np.asarray(pose['position'][:2])
    chassis = _box(robot.links[robot.base_footprint_link_name].aabb)
    floor = objects['floor']
    if floor.visual_only:
        raise ValueError('Original source floor must remain collidable.')
    hit = local_floor_surface(floor.prim_path, raytest(
        [*xy, pose['position'][2] + 0.25], [*xy, pose['position'][2] - 0.50], only_closest=False))
    surface_z = float(hit['position'][2])
    floor_contacts, nonfloor_contacts = [], []
    for contact in robot.contact_list():
        if legal_background_contact(session, str(contact.body0), str(contact.body1)):
            continue
        bodies = [str(contact.body0), str(contact.body1)]
        external = [body for body in bodies if not (body == robot.prim_path or body.startswith(robot.prim_path + '/'))]
        if not external:
            continue
        row = {'bodies': bodies}
        if all(body == floor.prim_path or body.startswith(floor.prim_path + '/') for body in external):
            position, normal = _vector(contact.position, 3), _vector(contact.normal, 3)
            row.update(position=position.tolist(), normal=normal.tolist())
            if abs(position[2] - surface_z) <= 0.02 and abs(normal[2]) > 0.5:
                floor_contacts.append(row)
        else:
            nonfloor_contacts.append(row)
    region = session.env.scene.seg_map.get_room_instance_by_point(xy)
    poses = {key: _pose(obj) for key, obj in objects.items() if key != 'floor'}
    boxes = {key: _box(obj.aabb) for key, obj in objects.items() if key != 'floor'}
    pairs = []
    for pair in config['parameters']['pairs']:
        target = np.asarray(poses[pair['target']]['position'][:2])
        low, high = boxes[pair['anchor']]
        face = np.clip(target, low[:2], high[:2])
        direction = target - face
        norm = float(np.linalg.norm(direction))
        separation = np.maximum(np.maximum(chassis[0][:2] - high[:2], low[:2] - chassis[1][:2]), 0.0)
        side = float((xy - face) @ (direction / norm)) if norm > 0.01 else None
        distance = float(np.linalg.norm(xy - target))
        valid = bool((config['parameters'].get('accessible_placement') or
            (side is not None and side >= PARAMETERS['target_side_margin_m']))
            and PARAMETERS['minimum_standoff_m'] <= distance <= PARAMETERS['maximum_standoff_m']
            and np.linalg.norm(separation) >= PARAMETERS['chassis_anchor_clearance_m'])
        pairs.append({**pair, 'distance_m': distance, 'target_side_projection_m': side,
            'chassis_anchor_clearance_m': float(np.linalg.norm(separation)), 'approach_geometry': valid})
    # R1's base_link is the chassis, not the floor-touching geometry. Its
    # three declared wheel links must agree with measured local floor rays.
    wheel_gaps = {}
    for name in robot.floor_touching_base_link_names:
        low, high = _box(robot.links[name].aabb)
        centre = (low + high) / 2
        wheel_hit = local_floor_surface(floor.prim_path, raytest(
            [*centre[:2], float(low[2]) + 0.10], [*centre[:2], float(low[2]) - 0.20], only_closest=False))
        wheel_gaps[name] = float(low[2] - wheel_hit['position'][2])
    supported = bool(floor_contacts and wheel_gaps and all(abs(gap) <= 0.03 for gap in wheel_gaps.values()))
    return {'robot': pose, 'room': region, 'floor_surface_z_m': surface_z,
        'floor_contacts': floor_contacts, 'nonfloor_contacts': nonfloor_contacts,
        'wheel_floor_gaps_m': wheel_gaps,
        'floor_supported': supported, 'safe': bool(supported and not nonfloor_contacts),
        'poses': poses, 'boxes': {key: [low.tolist(), high.tolist()] for key, (low, high) in boxes.items()},
        'pairs': pairs}


def qualify_operation(session, executor, config, bound):
    """Fail closed on missing native evidence; never move or alter the scene."""
    checks = {}
    try:
        from wia.tasks.interaction_setup import reference_position, touching_task_support, on_top_task_support
        parameters = _validate_config(session, config)
        objects = _native_objects(session, config)
        measured = _native_state(session, config, objects)
        checks = {'base_capabilities': bool(executor.capabilities.base_translation and executor.capabilities.base_rotation),
            'source_room_start': measured['room'] == session.room_name,
            'source_floor_support': measured['floor_supported'], 'no_nonfloor_contact': not measured['nonfloor_contacts'],
            'occluders_collidable': all(not objects[pair['anchor']].visual_only for pair in parameters['pairs']),
            'original_target_poses': all(np.linalg.norm(np.asarray(measured['poses'][key]['position'])
                - reference_position(session, objects[key].name, expected['position'])) <= 0.03 for key, expected in parameters['expected'].items()
                if 'position' in expected),
            'original_target_orientations': all(_quaternion_angle(measured['poses'][key]['quaternion_xyzw'],
                expected['quaternion_xyzw']) <= PARAMETERS['orientation_preservation_deg']
                for key, expected in parameters['expected'].items() if 'quaternion_xyzw' in expected),
            'original_object_dimensions': all(np.max(np.abs(np.asarray(measured['boxes'][key])
                - np.asarray([reference_position(session, objects[key].name, corner)
                    for corner in expected['bbox']]))) <= 0.05 for key, expected in parameters['expected'].items()),
            'native_target_side_resolved': bool(parameters.get('accessible_placement') or all(
                row['target_side_projection_m'] is not None for row in measured['pairs']))}
        if parameters.get('accessible_placement'):
            checks['physical_targets_accessibly_supported'] = all(
                not objects[pair['target']].visual_only and not objects[pair['target']].fixed_base and
                touching_task_support(session, objects[pair['target']], objects['floor']) and
                on_top_task_support(session, objects[pair['target']], objects['floor']) and
                measured['boxes'][pair['target']][0][2] - measured['floor_surface_z_m'] >= .35 and
                np.mean(np.asarray(measured['boxes'][pair['target']])[:, 2]) - measured['floor_surface_z_m'] <= 1.10
                for pair in parameters['pairs'])
        accepted = bool(all(checks.values()))
        return {'accepted': accepted, 'checks': checks, 'native_state': measured,
            'reason': None if accepted else 'native_patrol_start_or_source_geometry_precondition_failed',
            'route_qualified': False, 'pixel_visibility_qualified': False}
    except Exception as error:
        return {'accepted': False, 'checks': checks, 'reason': 'native_patrol_evidence_unavailable',
            'exception_summary': f'{type(error).__name__}: {error}', 'route_qualified': False}


class AccessPatrolEvaluator:
    """Private physical visit history; source answers and labels never score."""
    def __init__(self, session, controller, config, output):
        self.parameters = _validate_config(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.objects = _native_objects(session, config)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.maximum_displacement = controller.config.max_translation_speed_m_s * controller.dt + controller.config.translation_tolerance_m
        if not math.isfinite(self.maximum_displacement) or self.maximum_displacement <= 0 or controller.dt <= 0:
            raise ValueError('Actual guarded base controller sampling bounds are required.')
        self.episode, self.active = 0, False

    def _preserved(self, measured):
        if self.parameters.get('accessible_placement'):
            from wia.tasks.interaction_setup import touching_task_support, on_top_task_support
            if not all(not self.objects[pair['target']].visual_only and not self.objects[pair['target']].fixed_base and
                touching_task_support(self.session, self.objects[pair['target']], self.objects['floor']) and
                on_top_task_support(self.session, self.objects[pair['target']], self.objects['floor'])
                for pair in self.parameters['pairs']):
                return False
        for key, initial in self.initial['poses'].items():
            current = measured['poses'][key]
            if np.linalg.norm(np.asarray(current['position']) - initial['position']) > PARAMETERS['position_preservation_m']:
                return False
            if _quaternion_angle(current['quaternion_xyzw'], initial['quaternion_xyzw']) > PARAMETERS['orientation_preservation_deg']:
                return False
            initial_size = np.diff(np.asarray(self.initial['boxes'][key]), axis=0)
            current_size = np.diff(np.asarray(measured['boxes'][key]), axis=0)
            if np.max(np.abs(initial_size - current_size)) > PARAMETERS['position_preservation_m']:
                return False
        return True

    def begin_episode(self, observation):
        self.episode += 1
        self.initial = _native_state(self.session, self.config, self.objects)
        self.previous = np.asarray(self.initial['robot']['position'][:2])
        self.previous_quaternion = self.initial['robot']['quaternion_xyzw']
        self.start = self.previous.copy()
        self.travel, self.excursion, self.sample_count, self.speed = 0.0, 0.0, 0, float('inf')
        self.angular_speed = float('inf')
        qualified = not self.parameters.get('accessible_placement') or qualify_operation(
            self.session, self.controller, self.config, self.objects)['accepted']
        self.safe, self.continuous, self.layout_preserved = bool(self.initial['safe'] and qualified), True, True
        self.streaks = {pair['target']: 0 for pair in self.parameters['pairs']}
        self.visits, self.return_streak = {}, 0
        self.active = True
        return self.record(observation, action=None)

    def sample_step(self):
        if not self.active:
            return
        measured = _native_state(self.session, self.config, self.objects)
        point = np.asarray(measured['robot']['position'][:2])
        distance = float(np.linalg.norm(point - self.previous))
        self.speed = distance / self.controller.dt
        self.angular_speed = math.radians(_quaternion_angle(measured['robot']['quaternion_xyzw'],
            self.previous_quaternion)) / self.controller.dt
        self.travel += distance
        self.excursion = max(self.excursion, float(np.linalg.norm(point - self.start)))
        self.continuous &= distance <= self.maximum_displacement
        self.safe &= measured['safe']
        self.layout_preserved &= self._preserved(measured)
        self.previous = point
        self.previous_quaternion = measured['robot']['quaternion_xyzw']
        self.sample_count += 1
        allowed = bool(self.safe and self.continuous and self.layout_preserved
            and self.excursion >= PARAMETERS['minimum_departure_m']
            and self.speed <= PARAMETERS['maximum_visit_speed_m_s']
            and self.angular_speed <= PARAMETERS['maximum_visit_angular_speed_rad_s']
            and measured['room'] == self.session.room_name)
        for row in measured['pairs']:
            key = row['target']
            self.streaks[key] = self.streaks[key] + 1 if allowed and row['approach_geometry'] else 0
            if self.streaks[key] >= PARAMETERS['stability_steps'] and key not in self.visits:
                self.visits[key] = {'sample': self.sample_count, 'robot_position': measured['robot']['position'], **row}
        returned = bool(allowed and len(self.visits) == len(self.streaks)
            and np.linalg.norm(point - self.start) <= PARAMETERS['return_radius_m'])
        self.return_streak = self.return_streak + 1 if returned else 0

    def record(self, observation, *, action):
        if not self.active:
            return None
        measured = _native_state(self.session, self.config, self.objects)
        at_last_sample = bool(np.linalg.norm(np.asarray(measured['robot']['position'][:2]) - self.previous) <= 1e-6
            and _quaternion_angle(measured['robot']['quaternion_xyzw'], self.previous_quaternion) <= 1e-4)
        returned = self.config['operation'] == 'occluder_side_patrol_return'
        stopped = bool(self.speed <= PARAMETERS['maximum_visit_speed_m_s']
            and self.angular_speed <= PARAMETERS['maximum_visit_angular_speed_rad_s'] and at_last_sample)
        terminal = (self.return_streak >= PARAMETERS['stability_steps'] and np.linalg.norm(
            np.asarray(measured['robot']['position'][:2]) - self.start) <= PARAMETERS['return_radius_m']) if returned else any(
                self.streaks[row['target']] >= PARAMETERS['stability_steps'] and row['approach_geometry'] for row in measured['pairs'])
        checks = {'all_native_target_visits': len(self.visits) == len(self.streaks),
            'real_departure': self.excursion >= PARAMETERS['minimum_departure_m'],
            'real_travel': self.travel >= PARAMETERS['minimum_travel_m'],
            'continuous_native_motion': bool(self.continuous), 'safe_trajectory': bool(self.safe and measured['safe']),
            'source_layout_preserved': bool(self.layout_preserved and self._preserved(measured)),
            'stopped_at_terminal': bool(stopped and terminal), 'source_room_terminal': measured['room'] == self.session.room_name}
        value = {'schema': 'private_occlusion_access_patrol_v1', 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step, 'action': action,
            'operation': self.config['operation'], 'task_success': bool(all(checks.values())),
            'checks': checks, 'native_state': measured, 'visits': self.visits,
            'sampled_travel_m': self.travel, 'maximum_departure_m': self.excursion,
            'sampled_speed_m_s': self.speed if math.isfinite(self.speed) else None,
            'sampled_angular_speed_rad_s': self.angular_speed if math.isfinite(self.angular_speed) else None,
            'sample_count': self.sample_count, 'return_streak': self.return_streak,
            'maximum_sample_displacement_m': self.maximum_displacement,
            'configuration': self.config, 'pixel_visibility_qualified': False}
        items = [(f'{key}:geometric_visit', key in self.visits, 'historical_checkpoint', self.visits.get(key))
                 for key in sorted(self.streaks)]
        if returned:
            items.append(('return_and_park', bool(stopped and terminal and checks['real_travel'] and
                checks['all_native_target_visits'] and checks['source_room_terminal']), 'terminal_condition',
                {'return_streak': self.return_streak, 'sampled_travel_m': self.travel}))
        _completion_result(value, items)
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as handle:
            handle.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    return AccessPatrolEvaluator(session, controller, config, output)
