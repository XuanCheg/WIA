"""Convert source triangles into supported robot perimeter and vertex inspections.

Run CPU generation from the repository root with the behavior Python environment:
CategoryConverter().convert_all(sources, ConversionContext(batch_root)).
Set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2.
The campaign handles screenshots; this module never launches simulation or actions.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from wia.tasks.background_policy import legal_background_contact
from wia.tasks.interaction_setup import array as _array

from collections import Counter
from functools import lru_cache
import json
import hashlib
import math
import os
from pathlib import Path
import re

import numpy as np

from wia.authoring.conversion import ConverterBase, map_records, original_scene, physical_task
from wia.authoring.io import sha256, write_json

MODULE = 'core.esi_converters.spatial_relations_geometric_configuration'
VERTICES = ('vertex_1', 'vertex_2', 'vertex_3')
OPERATIONS = ('triangle_circuit_return', 'ordered_vertex_survey_return')
ARRANGEMENT_OPERATION = 'arrange_released_source_shape'
TASK_REVISION = 'task_edit_20261018_v1'
TASK_EDIT2_REVISION = 'task_edit2_20261018_v1'
LAYOUT_VERSION = 'seeded_aabb_centers_v1'
INITIAL_SHAPE_MARGIN = .025  # Half the existing 5% side-equality scoring tolerance.
MAP_RESOLUTION_M = .01
SPAWN_RADIUS_M = .40


def _vector(value, size):
    return isinstance(value, (list, tuple)) and len(value) == size and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)
        for item in value)


def _geometry(points):
    points = np.asarray(points, dtype=float)
    distances = [float(np.linalg.norm(points[i, :2] - points[j, :2]))
                 for i, j in ((0, 1), (0, 2), (1, 2))]
    a, b = points[1, :2] - points[0, :2], points[2, :2] - points[0, :2]
    area = abs(float(a[0] * b[1] - a[1] * b[0])) / 2
    # Diagnostic only: the source gives no exact classification tolerance.
    tolerance = .05 * max(distances)
    equal = [abs(distances[i] - distances[j]) <= tolerance
             for i, j in ((0, 1), (0, 2), (1, 2))]
    label = 'equilateral' if all(equal) else 'isosceles' if any(equal) else 'random triangle'
    return {'pairwise_xy_m': distances, 'area_m2': area,
            'diagnostic_class_at_five_percent': label, 'diagnostic_tolerance_m': tolerance}


def _source_data(source):
    raw, payload = source.raw, source.payload
    if (source.big_task, source.small_task) != CategoryConverter.category or (
            raw.get('big_task'), raw.get('small_task')) != CategoryConverter.category:
        raise ValueError('category_mismatch')
    if tuple(Path(source.relative_path).parts[-3:-1]) != (raw.get('scene'), raw.get('room')):
        raise ValueError('source_scene_room_hierarchy_mismatch')
    if raw.get('runner_task') != 'triangle' or raw.get('answer') not in (
            'equilateral', 'isosceles', 'random triangle'):
        raise ValueError('unsupported_original_triangle_qa')
    objects = map_records(payload.get('objects'))
    values = [objects.get('obj' + str(index), {}) for index in (1, 2, 3)]
    for index, obj in enumerate(values, 1):
        if not obj.get('category') or not obj.get('model') or obj.get('model') == 'scene_native':
            raise ValueError('missing_original_asset:obj' + str(index))
        if not _vector(obj.get('position'), 3) or not _vector(obj.get('quaternion_xyzw'), 4) or not any(
                obj['quaternion_xyzw']):
            raise ValueError('invalid_original_pose:obj' + str(index))
        scale = obj.get('scale', 1.)
        if isinstance(scale, (int, float)) and not isinstance(scale, bool):
            scale = [scale] * 3
        if not _vector(scale, 3) or min(scale) <= 0:
            raise ValueError('invalid_original_scale:obj' + str(index))
        # The original postprocessor resolves obj1/2/3 even when name is supplied.
        if obj.get('name', 'obj' + str(index)) != 'obj' + str(index):
            raise ValueError('source_builder_object_name_restore_conflict')
    match = re.fullmatch(r'Do the (.+?), (.+?) and (.+?) form an equilateral triangle, '
                         r'an isosceles triangle or a random triangle\?', raw.get('question', ''))
    if not match or list(match.groups()) != [obj['category'].replace('_', ' ') for obj in values]:
        raise ValueError('public_question_vertex_order_unresolved')
    if len({obj['category'] for obj in values}) != 3:
        raise ValueError('public_vertex_identity_ambiguous')
    poses = payload.get('camera_poses', [])
    poses = list(poses.values()) if isinstance(poses, dict) else poses
    if not poses or any(not _vector(pose.get('position'), 3) or not _vector(
            pose.get('quaternion_xyzw'), 4) or not any(pose['quaternion_xyzw']) for pose in poses):
        raise ValueError('missing_original_reference_camera')
    if not payload.get('floor_name'):
        raise ValueError('original_floor_unresolved')
    geometry = _geometry([obj['position'] for obj in values])
    if geometry['area_m2'] <= .001 or min(geometry['pairwise_xy_m']) < .05:
        raise ValueError('degenerate_original_triangle')
    return raw, payload, values, poses, geometry


@lru_cache(maxsize=1)
def _scene_maps(asset_root, scene_name):
    """Original map pixels, matching native BaseMap and SegmentationMap formulas."""
    import cv2
    cv2.setNumThreads(2)
    root = Path(asset_root)
    paths = [root / 'scenes' / scene_name / 'layout' / name for name in (
        'floor_trav_0.png', 'floor_insseg_0.png', 'floor_semseg_0.png')]
    categories = root / 'metadata/room_categories.txt'
    if not all(path.is_file() for path in [*paths, categories]):
        return None
    trav, rooms, semantics = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]
    if any(value is None for value in (trav, rooms, semantics)) or not (
            trav.shape == rooms.shape == semantics.shape and trav.shape[0] == trav.shape[1]):
        return None
    names, counts, room_ids = categories.read_text().splitlines(), Counter(), {}
    instances, first_pixels = np.unique(rooms, return_index=True)
    for instance, first_pixel in zip(instances, first_pixels):
        if instance == 0:
            continue
        semantic = int(semantics.flat[first_pixel])
        if not 0 < semantic <= len(names):
            continue
        name = names[semantic - 1]
        room_ids[f'{name}_{counts[name]}'] = int(instance)
        counts[name] += 1
    return trav, rooms, room_ids, {str(path): sha256(path) for path in [*paths, categories]}


@lru_cache(maxsize=2)
def _room_map(asset_root, scene_name, room_name):
    import cv2
    maps = _scene_maps(asset_root, scene_name)
    if maps is None or room_name not in maps[2]:
        return None
    trav, rooms, room_ids, hashes = maps
    free = ((trav > 0) & (rooms == room_ids[room_name])).astype(np.uint8)
    # Padding makes scene borders obstacles as well.
    padded = np.pad(free, 1)
    clearance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    return clearance * MAP_RESOLUTION_M, hashes


def _robot_start(raw, values, poses, context):
    root = context.config.get('original_asset_root', str(Path(os.environ.get(
        'OMNIGIBSON_DATA_PATH', '/local_home2/zhouxinyang/data/behavior')) / 'behavior-1k-assets'))
    maps = _room_map(str(root), raw['scene'], raw['room'])
    evidence = {'map_asset_root': str(root), 'map_resolution_m': MAP_RESOLUTION_M,
                'body_screen_radius_m': SPAWN_RADIUS_M, 'native_qualified': False}
    if maps is None:
        return None, {**evidence, 'failure_reason': 'original_room_maps_unavailable'}
    clearance, hashes = maps
    vertices = np.asarray([obj['position'][:2] for obj in values])
    centre = vertices.mean(axis=0)

    def valid(xy):
        row, col = (np.asarray(xy)[::-1] / MAP_RESOLUTION_M + clearance.shape[0] / 2).astype(int)
        return (0 <= row < clearance.shape[0] and 0 <= col < clearance.shape[1]
                and clearance[row, col] >= SPAWN_RADIUS_M + .02
                and float(np.min(np.linalg.norm(vertices - xy, axis=1))) >= .65)

    candidates = [(f'original_floor_view_{i}', np.asarray(pose['position'][:2]))
                  for i, pose in enumerate(poses) if pose['position'][2] < .1]
    circle = None
    for radius in (.9, 1.1, 1.3, 1.5):
        angles = np.linspace(0, 2 * math.pi, math.ceil(2 * math.pi * radius / .04), endpoint=False)
        ring = centre + radius * np.column_stack((np.cos(angles), np.sin(angles)))
        if circle is None and all(valid(xy) for xy in ring):
            circle = (radius, ring)
        candidates += [(f'original_cluster_ring_{radius}_{i}', xy) for i, xy in enumerate(ring[::4])]
    good = [(label, xy) for label, xy in candidates if valid(xy)]
    evidence.update(map_sha256=hashes, screened_candidates=len(candidates),
                    valid_candidates=len(good), complete_circle_radius_m=circle[0] if circle else None,
                    floor_z_assumption_m=0., floor_z_provenance='Generator render_and_save uses '
                    'absolute cz=0.025; all current source roots are below 0.282 m. Ground floor '
                    'z=0 is an L2 assumption, requiring native source-floor verification.')
    if not good:
        return None, {**evidence, 'failure_reason': 'no_screened_original_room_robot_start'}
    if circle:
        radius, ring = circle
        first_view = np.asarray(poses[0]['position'][:2])
        xy = min(ring, key=lambda point: float(np.linalg.norm(point - first_view)))
        label, operation = f'complete_screened_circle_{radius}', OPERATIONS[0]
    else:
        # Wall-adjacent formations need accessible views, not an invented
        # requirement to stand beyond each outward corner (possibly in a wall).
        survey = []
        for vertex in vertices:
            options = [point for _, point in good if .65 <= float(np.linalg.norm(point - vertex)) <= 1.5
                       and all(float(np.linalg.norm(point - previous)) >= .4 for previous in survey)]
            if not options:
                return None, {**evidence, 'failure_reason': 'no_screened_three_vertex_survey_positions',
                              'survey_positions_screened': len(survey)}
            survey.append(min(options, key=lambda point: (float(np.linalg.norm(point - vertex)),
                                                          float(point[0]), float(point[1]))))
        evidence['possible_survey_positions_xy'] = [point.tolist() for point in survey]
        evidence['survey_connectivity_verified'] = False
        label, xy = min(good, key=lambda item: (float(np.linalg.norm(item[1] - centre)), item[0]))
        operation = OPERATIONS[1]
    return {'position': [float(xy[0]), float(xy[1]), 0.],
            'yaw_deg': math.degrees(math.atan2(centre[1] - xy[1], centre[0] - xy[0])),
            'selection': label, 'operation': operation}, evidence


def _public(raw, operation):
    if operation == OPERATIONS[0]:
        objective = ('Drive one complete circuit around the outside of the three-object triangle '
                     'in either direction, then return within 0.25 m of your starting position and stop.')
        success = ('Physically circle all three objects in the same direction, travel at least 2 m '
                   'within 2.5 m of the formation centre, return and remain stationary for 0.5 s.')
    else:
        objective = ('Approach the three triangle objects in the order listed in the question, '
                     'then return within 0.25 m of your starting position and stop. At each object, '
                     'stand between 0.65 and 1.5 m from it and face it with the robot front within '
                     '30 degrees. Use three inspection positions each at least 0.4 m from the others.')
        success = ('Physically complete the three ordered object approaches and headings, travel at least 1 m '
                   'with at least 0.5 m excursion, return and remain stationary for 0.5 s.')
    return {'objective': objective, 'success_criteria': success,
            'environment': 'The three original-size objects form a tabletop triangle in the original room. '
                'Distances between their horizontal positions may be uniformly scaled during initialization '
                'without changing individual object sizes; the formation may also be translated. '
                'the original floor layout remains available as a historical variant. '
                'Auxiliary initial question: ' + raw['question'],
            'constraints': ['Keep every triangle object on its starting support and in place throughout the route.',
                'Use mounted robot observations and existing guarded base controls.',
                'Remain on the original floor and avoid base collisions and contact with protected task objects. Legal arm manipulation of noncritical native movable furniture is allowed.',
                'Maintain at least 0.35 m clearance from each object footprint; keep the base within '
                '2.5 m of the formation centre. Initial QA is auxiliary to completing the route.']}


def shape_geometry(centres, shape):
    """Evaluate source-derived shapes in world XY, normalized by longest span."""
    points = np.asarray(centres, dtype=float)
    if points.shape != (3, 2) or not np.isfinite(points).all():
        raise ValueError('Three finite horizontal native centers are required.')
    pairs = ((0, 1), (0, 2), (1, 2))
    lengths = np.array([np.linalg.norm(points[i] - points[j]) for i, j in pairs])
    longest = float(lengths.max())
    if longest <= 0:
        return {'satisfied': False, 'longest_span_m': longest, 'reason': 'collapsed_centers'}
    i, j = pairs[int(lengths.argmax())]
    direction = (points[j] - points[i]) / longest
    residual = float(np.abs((points - points[i]) @ [-direction[1], direction[0]]).max())
    differences = np.array([abs(lengths[i] - lengths[j]) / longest for i, j in pairs])
    area = abs(float(np.cross(points[1] - points[0], points[2] - points[0]))) / 2
    distinct = bool(lengths.min() > max(.03, .05 * longest))
    if shape == 'collinear':
        satisfied = distinct and residual <= min(.02, .02 * longest)
    elif shape == 'equilateral':
        satisfied = distinct and 2 * area / longest ** 2 >= .05 and differences.max() <= .05
    elif shape == 'isosceles':
        satisfied = distinct and 2 * area / longest ** 2 >= .05 and differences.min() <= .05 < differences.max()
    elif shape == 'random triangle':
        satisfied = distinct and 2 * area / longest ** 2 >= .05 and differences.min() > .05
    else:
        raise ValueError('Unsupported declared source shape.')
    return {'satisfied': bool(satisfied), 'shape': shape, 'plane': 'world_xy',
        'pairwise_distances_m': lengths.tolist(), 'longest_span_m': longest,
        'minimum_center_separation_m': max(.03, .05 * longest), 'normalized_area': 2 * area / longest ** 2,
        'normalized_side_differences': differences.tolist(), 'maximum_perpendicular_residual_m': residual,
        'line_angular_residual_rad': math.atan2(residual, longest),
        'line_maximum_angular_residual_rad': math.atan(.02)}


def initial_shape_is_wrong(centres, target_shape, margin=INITIAL_SHAPE_MARGIN):
    """Reject the target by a side-ratio margin, retaining scoring's area bound.

    The margin is half the existing 0.05 equality band. For a scalene target
    (source name ``random triangle``), require a pair inside the equality band
    by that margin. The other two targets start from distinctly scalene sides.
    This changes initialization only; historical scoring tolerances are retained.
    """
    geometry = shape_geometry(centres, target_shape)
    if geometry['satisfied'] or geometry.get('normalized_area', 0.) < .05:
        return False
    differences = geometry.get('normalized_side_differences', [])
    if not differences:
        return False
    return bool(min(differences) <= .05 - margin if target_shape == 'random triangle'
                else min(differences) >= .05 + margin)


def seeded_initial_layout(*, object_keys, aabb_half_extents_xy, root_to_aabb_center_xy,
                          root_z_m, support_aabb_xy, target_shape, seed,
                          source_identity, minimum_clearance_m=.005, max_attempts=4096):
    """Pure supported-layout recipe, called after native asset measurements.

    Shared setup must preserve orientations/scales, supply the horizontal support
    rectangle and measured root-to-AABB offsets, place only during initialization,
    then store this returned receipt under
    ``category_interaction_setup['geometric_initial_layout']``. Add
    ``accepted=True`` only after native support/grasp-access/settling checks.
    ``minimum_clearance_m`` must include the setup's grasp-space requirement;
    5 mm is only the historical object-clearance floor, not a graspability proof.
    The qualifier recomputes the recipe and checks the settled native geometry.
    """
    keys = list(object_keys)
    extents = np.asarray(aabb_half_extents_xy, dtype=float)
    offsets = np.asarray(root_to_aabb_center_xy, dtype=float)
    bounds = np.asarray(support_aabb_xy, dtype=float)
    z = np.asarray(root_z_m, dtype=float)
    if (len(keys) != 3 or len(set(keys)) != 3 or extents.shape != (3, 2) or
            offsets.shape != (3, 2) or bounds.shape != (2, 2) or z.shape != (3,) or
            not all(np.isfinite(a).all() for a in (extents, offsets, bounds, z)) or
            np.any(extents <= 0) or minimum_clearance_m < .005 or
            target_shape not in ('equilateral', 'isosceles', 'random triangle')):
        raise ValueError('Invalid measured seeded-layout inputs.')
    inputs = {'object_keys': keys, 'aabb_half_extents_xy': extents.tolist(),
        'root_to_aabb_center_xy': offsets.tolist(), 'root_z_m': z.tolist(),
        'support_aabb_xy': bounds.tolist(), 'target_shape': target_shape,
        'seed': int(seed), 'source_identity': str(source_identity),
        'minimum_clearance_m': float(minimum_clearance_m), 'max_attempts': int(max_attempts)}
    token = hashlib.sha256(f'{seed}\0{source_identity}\0{LAYOUT_VERSION}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(token[:8], 'big'))
    low, high = bounds[0] + extents + minimum_clearance_m, bounds[1] - extents - minimum_clearance_m
    if np.any(high <= low):
        raise ValueError('Insufficient support footprint for seeded geometry.')
    for attempt in range(max_attempts):
        if target_shape == 'random triangle':
            # Equilateral starts provide a separated, robustly non-scalene class.
            angle = rng.uniform(0., 2 * math.pi)
            radius = rng.uniform(.2, .48) * float(np.min(bounds[1] - bounds[0]))
            center = rng.uniform(np.max(low, axis=0), np.min(high, axis=0))
            angles = angle + np.arange(3) * 2 * math.pi / 3
            centers = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
        else:
            centers = rng.uniform(low, high)
        if np.any(centers < low) or np.any(centers > high):
            continue
        gaps = [float(np.linalg.norm(np.maximum(np.abs(centers[i] - centers[j]) -
            extents[i] - extents[j], 0.))) for i, j in ((0, 1), (0, 2), (1, 2))]
        geometry = shape_geometry(centers, target_shape)
        if (min(gaps) < minimum_clearance_m or
                min(geometry['pairwise_distances_m']) <= geometry['minimum_center_separation_m'] or
                not initial_shape_is_wrong(centers, target_shape)):
            continue
        roots = np.column_stack((centers - offsets, z))
        return {'version': LAYOUT_VERSION, 'inputs': inputs, 'seed': int(seed),
            'source_identity': str(source_identity), 'attempt': attempt,
            'object_keys': keys, 'initial_positions': roots.tolist(),
            'initial_aabb_centers_xy': centers.tolist(), 'initial_geometry': geometry,
            'initial_false_margin': INITIAL_SHAPE_MARGIN, 'horizontal_clearances_m': gaps}
    raise ValueError('No supported nonoverlapping initially-wrong seeded layout found.')


def convert_arrangement(source, context, raw, payload, values, keys, module, shape):
    """Declare actual arranging, without retaining route feasibility as a gate."""
    revised = module == MODULE and context.config.get('task_revision', TASK_EDIT2_REVISION) == TASK_EDIT2_REVISION
    bindings = {key: {'name': f'obj{index}', 'category': obj['category'], 'model': obj['model']}
        for index, (key, obj) in enumerate(zip(keys, values), 1)}
    bindings['source_floor'] = {'name': payload['floor_name']}
    patches = [{'operation': 'object_physics', 'phase': phase, 'binding': key, 'visual_only': False}
        for key in keys for phase in ('before_construction', 'after_postprocessing')]
    contract = {'preserve': ['participating_object_identities', 'asset_scales', 'source_reference_camera'],
        'goal': 'stable_released_arrangement', 'shape': shape, 'plane': 'world_xy',
        'initial_layout_policy': 'common_translation_only', 'source_qa_role': 'historical_initial_state'}
    names = ', '.join(obj['category'].replace('_', ' ') for obj in values)
    geometry_words = ('Keep maximum center-to-longest-baseline distance at most min(2 cm, 2% of span), '
        'equivalent to a normalized angular residual at most atan(0.02).' if shape == 'collinear' else
        'Compare side lengths using 5% of the longest side. Equilateral requires all equal; isosceles '
        'requires an equal pair and a distinct third side; random triangle means all sides distinct. '
        'Twice the horizontal area divided by longest side squared must be at least 0.05.')
    public = {'objective': f'Arrange the {names} into a {shape} arrangement on their starting support. '
        'Use their horizontal bounding-box centers. You may leave an already correct arrangement in place.',
        'success_criteria': geometry_words + ' Every center pair must be farther apart than max(3 cm, 5% of '
        'the longest span). Object bounding boxes must have at least 5 mm horizontal clearance. Preserve '
        'asset scales, release every object, open the gripper, and keep supported objects below 0.02 m/s '
        'and 0.1 rad/s for 15 consecutive native samples.',
        'environment': 'The original objects and room are retained. A qualified common translation to a '
        'support may be used at initialization; formation spacing and object dimensions are preserved.',
        'constraints': ['Use mounted observations and guarded robot controls.',
            'Preserve the starting support and task objects; no resizing or collapsed overlapping formation.',
            'The original initial-state question and answer remain historical evidence.']}
    parameters = {'task_revision': TASK_REVISION, 'mechanism_contract': contract, 'object_keys': list(keys),
        'target_shape': shape, 'source_positions': [obj['position'] for obj in values],
        'source_quaternions': [obj['quaternion_xyzw'] for obj in values],
        'source_scales': [obj.get('scale', 1.) for obj in values], 'release_stability_steps': 15,
        'completion_items': [{'id': 'requested_shape'}, {'id': 'stable_released_support'}]}
    evaluation = {'type': 'category_operation', 'module': module, 'operation': ARRANGEMENT_OPERATION,
        'binding_keys': [*keys, 'source_floor'], 'parameters': parameters}
    revision = TASK_EDIT2_REVISION if revised else TASK_REVISION
    if revised:
        contract.update(initial_layout_policy=LAYOUT_VERSION, already_satisfied_no_op_allowed=False,
            preserve=['participating_object_identities', 'asset_scales', 'asset_orientations', 'source_reference_camera'])
        parameters.update(task_revision=revision, initial_layout={
            'version': LAYOUT_VERSION, 'seed': context.seed, 'source_identity': source.relative_path,
            'reference_points': 'horizontal_aabb_centers', 'initial_false_margin': INITIAL_SHAPE_MARGIN},
            minimum_rearrangement_m=.035,
            source_assets=[{'category': obj['category'], 'model': obj['model']} for obj in values],
            completion_items=[{'id': 'requested_shape'}, {'id': 'robot_caused_rearrangement'},
                {'id': 'stable_released_support'}])
        public['objective'] = (f'Arrange the {names} into a {shape} arrangement on their starting support. '
            'Use their horizontal bounding-box centers and physically rearrange the objects with the robot.')
        public['environment'] = ('The original objects, orientations, scales and room are retained. '
            'Their seeded initial arrangement deliberately fails the requested shape. '
            'Objects share a qualified support with space for grasping.')
        public['success_criteria'] += (' Recorded native robot control and grasp/contact must cause '
            'object displacement exceeding the 3.5 cm initialization pose tolerance. '
            'Simulator state setters and initialization placement do not count as rearrangement.')
    task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings,
        patches=patches, capabilities=['base_translation', 'base_rotation', 'ee_translation', 'ee_rotation',
            'trunk_height', 'gripper'], qualification={'on_failure': 'skip', 'initial_visibility': 'none'})
    task['repair'] = {'task_revision': revision, 'mechanism_contract': contract}
    return {'status': 'generated', 'reuse_level': 'L2', 'task_revision': revision,
        'mechanism_contract': contract, 'scene_definition': original_scene(source, context, bindings=bindings, patches=patches),
        'task_definition': task, 'operation': ARRANGEMENT_OPERATION,
        'interaction_policy': {'mode': 'manipulation', 'group_bindings': list(keys),
            'manipulation_bindings': list(keys), 'relocation_policy': 'rigid_group',
            'source_camera_policy': 'preserve', 'source_region': raw['room'],
            'support_search': 'measured_surfaces', 'allow_fallback_table': bool(revised),
            'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10, 'maximum_stance_distance_m': .85},
        'warning_codes': ['native_arrangement_reach_and_support_pending'],
        'change_summary': ['Replace route-only credit with a specified released geometric arrangement.',
            'Preserve initial formation scale and original assets; require disjoint supported final objects.'],
        'limitations': ['Native robot start, support fit, grasp access and executed arrangement remain unverified.'],
        'private_source_qa': {'question': raw['question'], 'answer': raw['answer']}}


def arrangement_measure(session, controller, config, bound):
    from wia.authoring.converters.spatial_relations_physical_contact import _snapshot, _empty_released, _robot_state
    from wia.tasks.interaction_setup import touching_task_support, on_top_task_support
    keys = config['parameters']['object_keys']
    rows = {key: {**_snapshot(bound[key]), 'scale': _array(bound[key].scale).tolist()} for key in keys}
    centers = [(_array(row['aabb_min']) + _array(row['aabb_max']))[:2] / 2 for row in rows.values()]
    geometry = shape_geometry(centers, config['parameters']['target_shape'])
    clearance = []
    for i, first in enumerate(keys):
        for second in keys[i + 1:]:
            a, b = rows[first], rows[second]
            widths = np.minimum(a['aabb_max'][:2], b['aabb_max'][:2]) - np.maximum(a['aabb_min'][:2], b['aabb_min'][:2])
            clearance.append(float(np.linalg.norm(np.maximum(-widths, 0.))))
    support = all(touching_task_support(session, bound[key], bound['source_floor']) and
        on_top_task_support(session, bound[key], bound['source_floor']) for key in keys)
    robot = _robot_state(session, bound['source_floor'])
    checks = {'nonoverlapping_objects': min(clearance) >= .005, 'native_support': support,
        'movable_collidable': all(not row['visual_only'] and not row['fixed_base'] for row in rows.values()),
        'stationary': all(row['linear_speed_m_s'] <= .02 and row['angular_speed_rad_s'] <= .1 for row in rows.values()),
        'empty_open_released': _empty_released(controller.debug_gripper_state()),
        'source_room': robot['in_source_room'], 'robot_floor_clearance': robot['supported_upright']}
    return {'objects': rows, 'geometry': geometry, 'horizontal_clearances_m': clearance, 'checks': checks}


def qualify_arrangement(session, executor, config, bound):
    from wia.tasks.interaction_setup import reference_position
    try:
        controller = getattr(executor, 'controller', executor)
        measured = arrangement_measure(session, controller, config, bound)
        p = config['parameters']
        if p.get('task_revision') == TASK_EDIT2_REVISION:
            return qualify_seeded_arrangement(session, config, bound, measured)
        deltas = [reference_position(session, bound[key].name, original) - _array(original)
            for key, original in zip(p['object_keys'], p['source_positions'])]
        checks = {**measured['checks'], 'common_translation_only': all(np.allclose(delta, deltas[0], atol=1e-7, rtol=0.) for delta in deltas),
            'scene_and_bindings': (session.scene_name, session.room_name) == (config['scene'], config['room']) and
                all(bound[key].name == config['bindings'][key] for key in config['binding_keys']),
            'source_scales': all(np.allclose(_array(bound[key].scale), scale, rtol=1e-5, atol=1e-6)
                for key, scale in zip(p['object_keys'], p['source_scales'])),
            'initial_reference_positions': all(np.linalg.norm(_array(measured['objects'][key]['position']) -
                reference_position(session, bound[key].name, original)) <= .035
                for key, original in zip(p['object_keys'], p['source_positions'])),
            'initial_reference_orientations': all(_orientation_delta(
                measured['objects'][key]['quaternion_xyzw'], original) <= math.radians(5)
                for key, original in zip(p['object_keys'], p['source_quaternions']))}
        return {'accepted': bool(all(checks.values())), 'checks': checks, 'native_state': measured,
            'reason': None if all(checks.values()) else 'arrangement_initialization_failed', 'native_execution_verified': False}
    except Exception as error:
        return {'accepted': False, 'reason': 'arrangement_measurement_unavailable', 'exception_summary': str(error)}


def qualify_seeded_arrangement(session, config, bound, measured):
    """Validate the authorized recipe after settling; source poses are provenance."""
    p = config['parameters']
    setup = getattr(session, 'category_interaction_setup', {})
    receipt = setup.get('geometric_initial_layout', {})
    expected = p['initial_layout']
    checks = dict(measured['checks'])
    checks['authorized_seeded_layout'] = bool(setup.get('accepted') and receipt.get('accepted') and
        receipt.get('version') == expected['version'] == LAYOUT_VERSION and
        receipt.get('seed') == expected['seed'] and
        receipt.get('source_identity') == expected['source_identity'])
    if not checks['authorized_seeded_layout']:
        return {'accepted': False, 'checks': checks, 'native_state': measured,
            'reason': 'seeded_layout_receipt_missing_or_mismatched', 'native_execution_verified': False}
    recomputed = seeded_initial_layout(**receipt['inputs'])
    checks['seeded_recipe_matches'] = (recomputed == {key: receipt.get(key) for key in recomputed} and
        receipt['inputs']['seed'] == expected['seed'] and
        receipt['inputs']['source_identity'] == expected['source_identity'] and
        receipt['inputs']['target_shape'] == p['target_shape'] and
        receipt['inputs']['object_keys'] == p['object_keys'])
    keys = p['object_keys']
    centers = [((_array(measured['objects'][key]['aabb_min']) +
        _array(measured['objects'][key]['aabb_max'])) / 2)[:2] for key in keys]
    checks.update(scene_and_bindings=(session.scene_name, session.room_name) == (config['scene'], config['room']) and
            all(bound[key].name == config['bindings'][key] for key in config['binding_keys']),
        source_scales=all(np.allclose(_array(bound[key].scale), scale, rtol=1e-5, atol=1e-6)
            for key, scale in zip(keys, p['source_scales'])),
        source_assets=all(all(getattr(bound[key], field) == value for field, value in asset.items())
            for key, asset in zip(keys, p.get('source_assets', [{}] * 3))),
        initial_reference_positions=all(np.linalg.norm(_array(measured['objects'][key]['position']) - position) <= .035
            for key, position in zip(keys, recomputed['initial_positions'])),
        initial_aabb_centers=all(np.linalg.norm(center - _array(expected_center)) <= .035
            for center, expected_center in zip(centers, recomputed['initial_aabb_centers_xy'])),
        initial_reference_orientations=all(_orientation_delta(measured['objects'][key]['quaternion_xyzw'], original) <= math.radians(5)
            for key, original in zip(keys, p['source_quaternions'])),
        initial_target_false_with_margin=initial_shape_is_wrong(centers, p['target_shape']))
    low, high = _array(receipt['inputs']['support_aabb_xy'])
    checks['initial_footprints_within_support'] = all(
        np.all(_array(measured['objects'][key]['aabb_min'])[:2] >= low) and
        np.all(_array(measured['objects'][key]['aabb_max'])[:2] <= high) for key in keys)
    return {'accepted': bool(all(checks.values())), 'checks': checks, 'native_state': measured,
        'reason': None if all(checks.values()) else 'seeded_arrangement_initialization_failed',
        'native_execution_verified': False, 'initial_layout': receipt}


class ShapeArrangementEvaluator:
    """Require stable terminal geometry, with native causal history for TaskEdit2.

    The trusted simulator/controller supplies ``geometric_rearrangement_evidence``
    on every native physics sample. Its dictionary has ``available`` (bool),
    monotonically increasing ``sample_id`` and ``control_sequence`` (integers),
    ``control_active`` (bool), ``controlled_object_names``,
    ``grasped_object_names``, ``contact_object_names`` (native object-name lists),
    and ``state_setter_used`` (bool, latched since episode start). Sequence zero
    means initialization/no executed public robot command. Evidence must derive
    from executed native controls and native grasp/contact state, never an agent
    assertion or this evaluator's ``record(action=...)`` argument. State-setting
    after episode start invalidates rearrangement even when a grasp is present.
    Missing instrumentation stays unavailable. Legacy revisions use their old
    stable-release/no-op contract.
    """
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.bound = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.active = False

    def begin_episode(self, observation):
        self.active, self.stable_steps, self.shape_seen = True, 0, False
        self.revised = self.config['parameters'].get('task_revision') == TASK_EDIT2_REVISION
        self.robot_rearranged, self.state_setter_seen = False, False
        self.native_sample_error, self.previous_sample_id = None, None
        self.initial_control_sequence = None
        self.previous_control_sequence = None
        self.control_displacements = {key: np.zeros(2) for key in self.config['parameters']['object_keys']}
        self.initial_positions, self.previous_positions = None, None
        self.qualification = qualify_arrangement(self.session, self.controller, self.config, self.bound)
        if self.revised and self.qualification.get('native_state'):
            self.initial_positions = {key: ((_array(row['aabb_min']) + _array(row['aabb_max'])) / 2)[:2]
                for key, row in self.qualification['native_state']['objects'].items()}
            self.previous_positions = {key: value.copy() for key, value in self.initial_positions.items()}
        return self.record(observation, action=None)

    def _causal_history(self, measured, count):
        provider = getattr(self.controller, 'geometric_rearrangement_evidence', None)
        if not callable(provider):
            raise ValueError('Native geometric control/grasp/contact history is unavailable.')
        evidence = provider()
        required = ('sample_id', 'control_sequence', 'control_active', 'controlled_object_names',
            'grasped_object_names', 'contact_object_names', 'state_setter_used')
        if evidence.get('available') is not True or any(key not in evidence for key in required):
            raise ValueError('Native geometric control/grasp/contact history is incomplete.')
        sample_id, sequence = evidence['sample_id'], evidence['control_sequence']
        if (not isinstance(sample_id, int) or not isinstance(sequence, int) or sequence < 0 or
                (self.previous_sample_id is not None and sample_id < self.previous_sample_id) or
                (self.previous_control_sequence is not None and sequence < self.previous_control_sequence)):
            raise ValueError('Native geometric history is stale or nonmonotonic.')
        if self.initial_control_sequence is None:
            self.initial_control_sequence = sequence
        self.state_setter_seen |= bool(evidence['state_setter_used'])
        fresh = self.previous_sample_id is None or sample_id > self.previous_sample_id
        positions = {key: ((_array(row['aabb_min']) + _array(row['aabb_max'])) / 2)[:2]
            for key, row in measured['objects'].items()}
        if self.initial_positions is None or self.previous_positions is None:
            raise ValueError('Initial native geometry baseline is unavailable.')
        if fresh and count:
            controlled = set(evidence['controlled_object_names'])
            touching = set(evidence['grasped_object_names']) | set(evidence['contact_object_names'])
            for key, position in positions.items():
                if (evidence['control_active'] and sequence > self.initial_control_sequence and
                        self.bound[key].name in controlled & touching):
                    self.control_displacements[key] += position - self.previous_positions[key]
                    threshold = self.config['parameters']['minimum_rearrangement_m']
                    self.robot_rearranged |= bool(np.linalg.norm(self.control_displacements[key]) > threshold and
                        np.linalg.norm(position - self.initial_positions[key]) > threshold)
        # Every observed sample advances the baseline. A placement while controls
        # are inactive cannot later be credited by grasping the placed object.
        if fresh:
            self.previous_positions = positions
            self.previous_sample_id, self.previous_control_sequence = sample_id, sequence
        measured['native_rearrangement_history'] = evidence
        measured['robot_caused_rearrangement'] = bool(self.robot_rearranged and not self.state_setter_seen)
        measured['checks']['no_simulator_state_setter'] = not self.state_setter_seen
        return fresh

    def _update(self, count=False):
        measured = arrangement_measure(self.session, self.controller, self.config, self.bound)
        p = self.config['parameters']
        measured['checks']['scales_preserved'] = all(np.allclose(_array(self.bound[key].scale), scale,
            rtol=1e-5, atol=1e-6) for key, scale in zip(p['object_keys'], p['source_scales']))
        fresh = self._causal_history(measured, count) if self.revised else True
        shape = measured['geometry']['satisfied'] and measured['checks']['nonoverlapping_objects']
        self.shape_seen |= bool(shape)
        terminal = bool(shape and all(measured['checks'].values()))
        if self.revised:
            terminal &= bool(self.robot_rearranged and not self.state_setter_seen)
        if not terminal:
            self.stable_steps = 0
        elif count and fresh:
            self.stable_steps += 1
        return measured, terminal

    def sample_step(self):
        if self.active:
            if self.revised:
                try:
                    self._update(True)
                except Exception as error:
                    self.native_sample_error, self.stable_steps = str(error), 0
            else:
                self._update(True)

    def record(self, observation, *, action=None):
        if not self.active:
            return None
        error, measured, terminal = None, None, False
        try:
            measured, terminal = self._update()
        except Exception as exception:
            error, self.stable_steps = str(exception), 0
        if self.revised and self.native_sample_error:
            error = self.native_sample_error
        available = error is None and self.qualification['accepted']
        stable = bool(terminal and self.stable_steps >= self.config['parameters']['release_stability_steps'])
        items = [{'id': 'requested_shape', 'completed': self.shape_seen, 'kind': 'checkpoint'},
            {'id': 'stable_released_support', 'completed': stable, 'kind': 'terminal_condition'}]
        if self.revised:
            items.insert(1, {'id': 'robot_caused_rearrangement',
                'completed': bool(self.robot_rearranged and not self.state_setter_seen), 'kind': 'checkpoint'})
        for item in items:
            item['evaluation_validity'] = 'valid' if available else 'unavailable'
        value = {'schema': 'private_source_shape_arrangement_v1', 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'action': action, 'task_success': bool(available and stable),
            'evaluation_validity': 'complete' if available else 'unavailable', 'completion_items': items,
            'native_state': measured, 'stable_steps': self.stable_steps, 'measurement_error': error,
            'initial_qualification': self.qualification, 'task_revision': self.config['parameters'].get('task_revision', TASK_REVISION)}
        from wia.tasks.completion import progress
        value.update(progress(value))
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


class CategoryConverter(ConverterBase):
    category = ('Spatial Relations', 'Geometric Configuration')
    version = 'geometric_configuration_v2.1_task_edit2_20261018_v1'

    def inspect(self, source):
        try:
            _source_data(source)
            return {'eligible': True, 'reason': 'original_triangle_assets_and_relation_retained'}
        except (ValueError, KeyError, TypeError) as error:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2:' + str(error),
                    'warning_codes': ['original_source_precondition_failed']}

    def convert(self, source, context):
        raw, payload, values, poses, geometry = _source_data(source)
        if context.config.get('operation', ARRANGEMENT_OPERATION) == ARRANGEMENT_OPERATION:
            return convert_arrangement(source, context, raw, payload, values, VERTICES, MODULE, raw['answer'])
        bindings = {key: {'name': 'obj' + str(index), 'category': obj['category'], 'model': obj['model']}
                    for index, (key, obj) in enumerate(zip(VERTICES, values), 1)}
        bindings['source_floor'] = {'name': payload['floor_name']}
        robot, start_evidence = _robot_start(raw, values, poses, context)
        warnings = ['assumption_review_required', 'native_route_unverified', 'mounted_visibility_unverified',
                    'layout_uniform_position_scale_requires_review', 'native_navigation_unverified']
        assumptions = [{
            'doubt': 'A source camera is not an R1 footprint; map pixels do not certify full-body '
                     'clearance, wheel support, dynamic stability or a traversable circuit/survey.',
            'chosen_assumption': 'Use a deterministic 0.40 m radius screen in the original room '
                     'and a ground-floor root at z=0. Preserve the existing R1 template and all '
                     'assets. Require actual native robot wheel contact with the source floor before execution.',
            'alternatives': ['Keep an original-only diagnostic if no screened start exists.',
                     'Requalify a different reversible robot start using native evidence.'],
            'scope': source.relative_path,
            'risk': 'Ground height, full-body geometry, furniture, doors or limited floor area '
                    'may invalidate a route. No static map result certifies execution.'}, {
            'doubt': 'The original floor formation may not fit an existing tabletop at unchanged root spacing.',
            'chosen_assumption': 'Request a reversible same-room support layout using one positive uniform '
                'horizontal scale of root positions, a common horizontal translation and a common vertical '
                'translation. Preserve asset sizes, orientations, vertex identities, triangle similarity and '
                'original QA. Require measured support fit, no prop AABB overlap and live upward support contact.',
            'alternatives': ['Retain the historical original variant when no supported similarity layout qualifies.'],
            'scope': source.relative_path,
            'risk': 'Changed spacing and support context require Review and new mounted views; height alone '
                'does not establish gripper fit, reach or graspability. The navigation operation remains unverified.'}]
        if geometry['diagnostic_class_at_five_percent'] != raw['answer'] or (
                payload.get('layout') == 'equilateral_triangle' and raw['answer'] != 'equilateral'):
            warnings.append('source_geometry_annotation_uncertainty')
            assumptions.append({
                'doubt': 'The generic layout metadata or a five-percent root-distance diagnostic '
                         'disagrees with the source answer; no exact source equality tolerance is saved.',
                'chosen_assumption': 'Preserve original QA and triangle similarity; require '
                         'the physical route around the actual native formation independently of classification.',
                'alternatives': ['Request clarification of original label generation and tolerance.',
                         'Retain QA as historical annotation while inspecting the physical geometry.'],
                'scope': source.relative_path,
                'risk': 'The original QA label may be ambiguous; the physical operation neither '
                        'changes that answer nor uses it to establish success.'})
        result = {'status': 'generated', 'reuse_level': 'L2', 'qa_preserved': True,
            'scene_definition': original_scene(source, context, bindings=bindings), 'task_definition': None,
            'warning_codes': warnings, 'assumptions': assumptions, 'source_geometry_evidence': geometry,
            'robot_start_evidence': start_evidence, 'attempted_repairs': [],
            'change_summary': ['Preserve source assets, sizes, orientations, original observer and private QA; '
                'retain source poses in the historical original variant.',
                'Request an existing same-room tabletop with reversible uniformly scaled horizontal root '
                'positions and common translation, preserving triangle similarity.',
                'Retain the native base-navigation operation around the initialized three vertices.'],
            'limitations': ['CPU map screens and static validation do not prove native loading, '
                'mounted visibility or physical route success.', 'Historical assessment reports one '
                'original construction/control example, not a verified patrol or survey.',
                'No grasp recipe, asset resizing, new asset or source-label rewrite is introduced.'],
            'next_step': 'Publish all successes and diagnostics for Review; qualify actual tabletop support, '
                'robot wheel contact with the source floor, triangle similarity and guarded navigation before readiness.'}
        if robot is None:
            result.update(status='conversion_failed', reuse_level='L1', scene_definition=None,
                failure_reason='not_convertible_under_L1_L2:' + start_evidence['failure_reason'],
                diagnostic_stage='robot_start_screen', diagnostic_source_only=True)
            result['warning_codes'].extend(['not_convertible_under_L1_L2', 'source_only_diagnostic'])
            assumptions[0]['chosen_assumption'] = ('Retain an unmodified source-only diagnostic: '
                'the bounded conservative screen does not justify a complete operation recipe '
                'and its initialization. Preserve any possible start/inspection evidence privately for Review.')
            result['change_summary'] = ['Preserve the original source for explicitly labelled source-only Review.']
            result['limitations'].append('A bounded map/recipe search failure does not prove that '
                'every source-faithful physical operation is impossible.')
            result['next_step'] = ('Review the source-only capture and bounded search evidence. '
                'A different native-qualified approach or manipulation recipe may remain possible; '
                'preserve source geometry and QA while considering alternatives.')
            return result
        operation = context.config.get('operation', robot['operation'])
        patches = [{'operation': 'robot_pose', 'position': robot['position'],
                    'yaw_deg': robot['yaw_deg'], 'head_pitch_deg': 45.}]
        # The source has the installed R1 already via pipeline.build_env_config.
        # No robot injection, numerical configuration change or mechanism restoration is needed.
        for key in VERTICES:
            patches += [{'operation': 'object_physics', 'binding': key, 'visual_only': False},
                        {'operation': 'object_physics', 'phase': 'after_postprocessing',
                         'binding': key, 'visual_only': False}]
        parameters = {'source_positions': [obj['position'] for obj in values],
            'source_quaternions': [obj['quaternion_xyzw'] for obj in values],
            'source_floor_z_m': 0., 'initial_pose_tolerance_m': .035,
            'object_preservation_m': .02, 'object_orientation_tolerance_rad': math.radians(5),
            'minimum_object_clearance_m': .35, 'maximum_cluster_distance_m': 2.5,
            'maximum_step_displacement_m': .10, 'return_distance_m': .25,
            'minimum_path_m': 2. if operation == OPERATIONS[0] else 1.,
            'minimum_excursion_m': .5, 'survey_min_distance_m': .65,
            'survey_max_distance_m': 1.5, 'survey_heading_cosine': math.cos(math.radians(30)),
            'survey_separation_m': .4, 'maximum_stop_speed_m_s': .03,
            'maximum_stop_angular_speed_rad_s': .1, 'stop_duration_s': .5}
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': operation,
                      'binding_keys': [*VERTICES, 'source_floor'], 'parameters': parameters}
        result.update(scene_definition=original_scene(source, context, bindings=bindings, patches=patches),
            task_definition=physical_task(source, context, public=_public(raw, operation),
                evaluation=evaluation, bindings=bindings, patches=patches,
                capabilities=['base_translation', 'base_rotation'],
                qualification={'on_failure': 'skip', 'initial_visibility': 'none',
                               'target_movable_collidable': False}), operation=operation,
            robot_start={**robot, 'native_qualified': False})
        result['change_summary'].append('Declare reversible R1 initialization and collidable objects before '
            'construction and after source pose postprocessing; shared native initialization measures the tabletop.')
        return result

    def convert_all(self, sources, context):
        """Reuse shared per-source diagnostics and immutable deterministic candidate IDs."""
        return super().convert_all(sources, context)

    def sample(self, manifest, policy=None):
        count = (policy or {}).get('count', 5)
        base = super().sample(manifest, policy)
        rows = sorted(manifest['records'], key=lambda row: row['source_identity'])
        prior = set((policy or {}).get('previous_failures', ()))
        selected = [row['candidate_id'] for row in rows if row['candidate_id'] in prior][:count]
        strata = [lambda row: row['status'] != 'generated',
            lambda row: row.get('operation') == OPERATIONS[0],
            lambda row: row.get('operation') == OPERATIONS[1],
            lambda row: 'source_geometry_annotation_uncertainty' in row['warning_codes'],
            lambda row: row.get('source_geometry_evidence', {}).get('diagnostic_class_at_five_percent') == 'random triangle']
        for predicate in strata:
            selected_scenes = {row.get('scene_model') for row in rows if row['candidate_id'] in selected}
            matching_rows = [row for row in rows if predicate(row) and row['candidate_id'] not in selected]
            matching_rows.sort(key=lambda row: (row.get('scene_model') in selected_scenes, row['source_identity']))
            matches = [row['candidate_id'] for row in matching_rows]
            if matches and len(selected) < count:
                selected.append(matches[0])
        return (selected + [candidate for candidate in base if candidate not in selected])[:count]


def _object_state(obj):
    position, quaternion = (_array(value) for value in obj.get_position_orientation())
    low, high = (_array(value) for value in obj.aabb)
    if any(value.shape != (size,) or not np.isfinite(value).all() for value, size in (
            (position, 3), (quaternion, 4), (low, 3), (high, 3))) or np.linalg.norm(quaternion) == 0:
        raise ValueError('invalid_native_object_state')
    return {'name': obj.name, 'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
            'aabb_min': low.tolist(), 'aabb_max': high.tolist(), 'visual_only': bool(obj.visual_only)}


def _native_support(session, floor, pose, floor_z):
    """Read native external wheel/floor contacts; XY aliases and hovering fail."""
    robot, root = session.robot, str(session.robot.prim_path)
    owns = lambda path, owner: path == owner or path.startswith(owner + '/')
    wheels = {str(robot.links[name].prim_path) for name in robot.floor_touching_base_link_names}
    supported_wheels, contacts, forbidden = set(), [], []
    for contact in robot.contact_list():
        if legal_background_contact(session, contact.body0, contact.body1):
            continue
        bodies = [str(contact.body0), str(contact.body1)]
        in_robot = [owns(body, root) for body in bodies]
        if all(in_robot):
            continue
        external = [body for body, own in zip(bodies, in_robot) if not own]
        wheel = next((path for path in sorted(wheels) if any(owns(body, path) for body in bodies)), None)
        position, normal = _array(contact.position), _array(contact.normal)
        row = {'bodies': bodies, 'position': position.tolist(), 'normal': normal.tolist()}
        if (any(in_robot) and wheel and external and all(owns(body, str(floor.prim_path)) for body in external)
                and position.shape == normal.shape == (3,) and np.isfinite([position, normal]).all()
                and abs(position[2] - floor_z) <= .015 and abs(normal[2]) >= .5):
            supported_wheels.add(wheel)
            contacts.append(row)
        else:
            forbidden.append(row)
    q = _array(pose['quaternion_xyzw'])
    q /= np.linalg.norm(q)
    upright = bool(1 - 2 * (q[0] ** 2 + q[1] ** 2) >= .9)
    low, _ = (_array(value) for value in robot.aabb)
    return {'source_floor_contacts': contacts, 'other_external_contacts': forbidden,
            'supported_wheels': sorted(supported_wheels),
            'supported_upright': bool(len(supported_wheels) >= 2 and not forbidden and upright
                                      and abs(low[2] - floor_z) < .15)}


def _task_support(session, bound):
    """Keep current upward support separate from the robot's original floor."""
    from wia.tasks.interaction_setup import touching_task_support, on_top_task_support
    floor = bound['source_floor']
    return [{'touching_task_support': touching_task_support(session, bound[key], floor),
             'on_top_task_support': on_top_task_support(session, bound[key], floor)} for key in VERTICES]


def _configuration(session, config, bound=None):
    if (module_identity(config.get('module')) != MODULE or config.get('operation') not in OPERATIONS or
            tuple(config.get('binding_keys', ())) != (*VERTICES, 'source_floor') or
            (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name)):
        raise ValueError('native_operation_configuration_mismatch')
    if bound is None:
        bound = {key: session.env.scene.object_registry('name', config['bindings'][key])
                 for key in config['binding_keys']}
    if any(bound.get(key) is None for key in config['binding_keys']):
        raise ValueError('missing_native_original_binding')
    return bound


def qualify_operation(session, executor, config, bound):
    """Only inspect state. No control, teleport, state setter or QA-based acceptance."""
    if config.get('operation') == ARRANGEMENT_OPERATION:
        return qualify_arrangement(session, executor, config, bound)
    checks = {}
    try:
        from wia.tasks.interaction_setup import reference_position
        bound = _configuration(session, config, bound)
        p, floor = config['parameters'], bound['source_floor']
        if floor.category not in ('floor', 'floors') or floor.visual_only:
            raise ValueError('original_floor_not_collidable')
        states = [_object_state(bound[key]) for key in VERTICES]
        checks['native_vertices'] = states
        references = [reference_position(session, bound[key].name, original)
                      for key, original in zip(VERTICES, p['source_positions'])]
        errors = [math.dist(row['position'], reference) for row, reference in zip(states, references)]
        checks['reference_pose_errors_m'] = errors
        checks['adapted_reference_positions'] = [reference.tolist() for reference in references]
        if any(row['visual_only'] for row in states) or max(errors) > p['initial_pose_tolerance_m']:
            raise ValueError('native_original_triangle_not_reproduced')
        if any(_orientation_delta(row['quaternion_xyzw'], original) > p['object_orientation_tolerance_rad']
               for row, original in zip(states, p['source_quaternions'])):
            raise ValueError('native_original_object_orientation_not_reproduced')
        floor_top = float(_array(floor.aabb[1])[2])
        checks['native_floor_top_m'] = floor_top
        checks['native_object_task_support'] = _task_support(session, bound)
        if (abs(floor_top - p['source_floor_z_m']) > .015 or
                not all(all(row.values()) for row in checks['native_object_task_support'])):
            raise ValueError('native_original_floor_height_or_task_support_mismatch')
        pose = executor.pose()
        checks['native_robot_support'] = _native_support(session, floor, pose, floor_top)
        if not checks['native_robot_support']['supported_upright']:
            raise ValueError('native_robot_source_floor_support_or_clearance_failed')
        xy = np.asarray(pose['position'][:2])
        clearances = [float(np.linalg.norm(xy - np.clip(xy, row['aabb_min'][:2], row['aabb_max'][:2])))
                      for row in states]
        checks['native_initial_object_clearances_m'] = clearances
        if min(clearances) < p['minimum_object_clearance_m'] or float(np.linalg.norm(
                xy - np.mean([row['position'][:2] for row in states], axis=0))) > p['maximum_cluster_distance_m']:
            raise ValueError('native_robot_initial_triangle_neighborhood_or_clearance_failed')
        checks['native_geometry'] = _geometry([row['position'] for row in states])
        checks['route_feasibility_verified'] = False
        return {'accepted': True, 'checks': checks, 'reason': None}
    except (ValueError, RuntimeError, AttributeError, KeyError, TypeError, IndexError) as error:
        return {'accepted': False, 'checks': checks, 'reason': str(error)}


def _orientation_delta(a, b):
    a, b = _array(a), _array(b)
    cosine = abs(float(np.dot(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 2 * math.acos(float(np.clip(cosine, 0., 1.)))



def _completion_result(items):
    """Report declared physical obligations without distance-shaped rewards."""
    completed = sum(bool(item['completed']) for item in items)
    total = len(items)
    score = completed / total
    return {'scoring_version': 'task_completion_v2', 'evaluation_validity': 'complete',
            'completion_items': items, 'completed_goals': completed, 'total_goals': total,
            'completion_score': score, 'score_percent': 100. * score}


class TriangleNavigationEvaluator:
    """Accumulate actual physics-step motion around measured, unchanged vertices."""
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.bound = _configuration(session, config)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.initial, self.episode = None, 0

    def _measure(self):
        p = self.config['parameters']
        pose = self.controller.pose()
        states = [_object_state(self.bound[key]) for key in VERTICES]
        floor = self.bound['source_floor']
        support = _native_support(self.session, floor, pose, float(_array(floor.aabb[1])[2]))
        xy = np.asarray(pose['position'][:2])
        clearances = [float(np.linalg.norm(xy - np.clip(xy, row['aabb_min'][:2], row['aabb_max'][:2])))
                      for row in states]
        speed = float(np.linalg.norm(_array(self.session.robot.get_linear_velocity())))
        angular = float(np.linalg.norm(_array(self.session.robot.get_angular_velocity())))
        valid = support['supported_upright'] and min(clearances) >= p['minimum_object_clearance_m']
        time_s = float(self.session.pipeline.og.sim.current_time)
        if not math.isfinite(time_s):
            raise ValueError('invalid_native_physics_clock')
        return {'base': pose, 'vertices': states, 'support': support, 'time_s': time_s,
                'object_task_support': _task_support(self.session, self.bound),
                'object_clearances_m': clearances, 'safe': bool(valid),
                'speed_m_s': speed, 'angular_speed_rad_s': angular}

    def begin_episode(self, observation):
        qualified = qualify_operation(self.session, self.controller, self.config, self.bound)
        if not qualified['accepted']:
            raise ValueError('Episode native preconditions failed: ' + str(qualified['reason']))
        self.episode += 1
        self.initial = self._measure()
        self.centres = np.asarray([row['position'][:2] for row in self.initial['vertices']])
        self.centre = self.centres.mean(axis=0)
        self.last = np.asarray(self.initial['base']['position'])
        self.angles = np.arctan2(self.last[1] - self.centres[:, 1], self.last[0] - self.centres[:, 0])
        self.winding = np.zeros(3)
        self.path_m, self.excursion_m, self.quiet_seconds = 0., 0., 0.
        self.stage, self.survey_visits = 0, []
        self.circuit_checkpoint = None
        self.trajectory_valid = True
        self.last_time, self.previously_quiet = self.initial['time_s'], False
        self.record(observation, action=None)

    def _checks(self, measured):
        p = self.config['parameters']
        preserved = all(not row['visual_only'] and math.dist(row['position'], original['position']) <= p[
            'object_preservation_m'] and _orientation_delta(row['quaternion_xyzw'], original['quaternion_xyzw']) <= p[
            'object_orientation_tolerance_rad'] and max(math.dist(row[edge], original[edge]) for edge in (
            'aabb_min', 'aabb_max')) <= p['object_preservation_m']
            for row, original in zip(measured['vertices'], self.initial['vertices']))
        position = measured['base']['position']
        cluster_distance = float(np.linalg.norm(np.asarray(position[:2]) - self.centre))
        return {'original_vertices_preserved': bool(preserved and all(
                    all(row.values()) for row in measured['object_task_support'])),
            'source_floor_supported_and_clear': measured['safe'],
            'within_formation_neighborhood': cluster_distance <= p['maximum_cluster_distance_m'],
            'at_return_position': math.dist(position[:2], self.initial['base']['position'][:2]) <= p['return_distance_m'],
            'stopped': measured['speed_m_s'] <= p['maximum_stop_speed_m_s'] and
                       measured['angular_speed_rad_s'] <= p['maximum_stop_angular_speed_rad_s']}

    def _operation_complete(self):
        if self.config['operation'] == OPERATIONS[0]:
            return self.circuit_checkpoint is not None
        return self.stage == 3

    def sample_step(self):
        if self.initial is None:
            return
        measured = self._measure()
        p = self.config['parameters']
        dt = measured['time_s'] - self.last_time
        if dt < -1e-7 or dt > .1 + 1e-7:
            self.trajectory_valid = False
        if dt <= 1e-7:
            return  # A repeated read is not another native time sample.
        self.last_time = measured['time_s']
        position = np.asarray(measured['base']['position'])
        checks = self._checks(measured)
        step = float(np.linalg.norm(position - self.last))
        self.trajectory_valid &= bool(step <= p['maximum_step_displacement_m'] and all(checks[key] for key in (
            'original_vertices_preserved', 'source_floor_supported_and_clear', 'within_formation_neighborhood')))
        self.path_m += float(np.linalg.norm(position[:2] - self.last[:2]))
        self.excursion_m = max(self.excursion_m, math.dist(position[:2], self.initial['base']['position'][:2]))
        angles = np.arctan2(position[1] - self.centres[:, 1], position[0] - self.centres[:, 0])
        self.winding += np.arctan2(np.sin(angles - self.angles), np.cos(angles - self.angles))
        self.angles, self.last = angles, position
        if self.config['operation'] == OPERATIONS[1] and self.stage < 3 and self.trajectory_valid:
            offset = self.centres[self.stage] - position[:2]
            distance = float(np.linalg.norm(offset))
            q = _array(measured['base']['quaternion_xyzw'])
            x, y, z, w = q / np.linalg.norm(q)
            yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
            cosine = float(np.dot([math.cos(yaw), math.sin(yaw)], offset)) / max(distance, 1e-12)
            separated = all(math.dist(position[:2], visit['position_xy']) >= p['survey_separation_m']
                            for visit in self.survey_visits)
            if (p['survey_min_distance_m'] <= distance <= p['survey_max_distance_m'] and
                    cosine >= p['survey_heading_cosine'] and separated):
                self.survey_visits.append({'vertex': VERTICES[self.stage], 'time_s': measured['time_s'],
                    'position_xy': position[:2].tolist(), 'distance_m': distance, 'heading_cosine': cosine})
                self.stage += 1
        if (self.config['operation'] == OPERATIONS[0] and self.trajectory_valid
                and self.circuit_checkpoint is None
                and (np.all(self.winding >= 2 * math.pi - 1e-6)
                     or np.all(self.winding <= -2 * math.pi + 1e-6))):
            self.circuit_checkpoint = {'time_s': measured['time_s'],
                'winding_radians': self.winding.tolist(), 'position': position.tolist()}
        quiet = (self.trajectory_valid and self._operation_complete() and checks['at_return_position'] and checks['stopped']
                 and self.path_m >= p['minimum_path_m'] and self.excursion_m >= p['minimum_excursion_m'])
        # An env.step can encompass several physics steps. Measure native elapsed
        # time instead of counting callbacks or assuming their frequency.
        self.quiet_seconds = self.quiet_seconds + dt if quiet and self.previously_quiet else 0.
        self.previously_quiet = quiet

    def record(self, observation, *, action=None):
        if self.initial is None:
            return None
        measured = self._measure()
        checks = self._checks(measured)
        p = self.config['parameters']
        checks.update(continuous_safe_native_trajectory=bool(self.trajectory_valid),
            recent_monotonic_native_sample=-1e-7 <= measured['time_s'] - self.last_time <= .1 + 1e-7,
            operation_complete=self._operation_complete(), minimum_path_completed=self.path_m >= p['minimum_path_m'],
            minimum_excursion_completed=self.excursion_m >= p['minimum_excursion_m'],
            stationary_duration_completed=self.quiet_seconds >= p['stop_duration_s'])
        value = {'schema': 'private_triangle_navigation_v1', 'episode': self.episode,
            'snapshot_id': getattr(observation, 'snapshot_id', None), 'action_step': getattr(observation, 'action_step', None),
            'action': action, 'task_success': all(checks.values()), 'criteria': checks,
            'path_m': self.path_m, 'excursion_m': self.excursion_m, 'winding_radians': self.winding.tolist(),
            'survey_stage': self.stage, 'survey_visits': self.survey_visits, 'stationary_seconds': self.quiet_seconds,
            'configuration': self.config, 'initial_native_state': self.initial, 'native_state': measured}
        safe = bool(self.trajectory_valid and all(checks[key] for key in (
            'original_vertices_preserved', 'source_floor_supported_and_clear',
            'within_formation_neighborhood', 'recent_monotonic_native_sample')))
        if self.config['operation'] == OPERATIONS[0]:
            items = [{'id': 'complete_circuit', 'completed': bool(safe and self.circuit_checkpoint),
                'evaluation_validity': 'valid', 'kind': 'historical_checkpoint',
                'evidence': self.circuit_checkpoint}]
        else:
            items = [{'id': f'approach_{index + 1}_{vertex}',
                'completed': bool(safe and self.stage > index), 'evaluation_validity': 'valid',
                'kind': 'historical_checkpoint',
                'evidence': self.survey_visits[index] if index < len(self.survey_visits) else None}
                for index, vertex in enumerate(VERTICES)]
        items.append({'id': 'return_and_stop', 'completed': bool(all(checks.values())),
            'evaluation_validity': 'valid', 'kind': 'terminal_condition', 'evidence': checks})
        value.update(_completion_result(items))
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    if config.get('operation') == ARRANGEMENT_OPERATION:
        return ShapeArrangementEvaluator(session, controller, config, output)
    return TriangleNavigationEvaluator(session, controller, config, output)
