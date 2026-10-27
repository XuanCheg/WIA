"""Evaluate a direct line-arrangement goal without changing existing scenes.

Run CPU generation using CategoryConverter().convert_all(sources, context) from
the repository root with the behavior Python environment. Set OMP_NUM_THREADS,
OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to 2. The campaign owns native captures;
this module never starts simulation or issues robot/object control commands.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from wia.tasks.background_policy import legal_background_contact

from collections import Counter
import copy
from functools import lru_cache
import json
import math
import os
from pathlib import Path

import numpy as np

from wia.authoring.conversion import ConverterBase, map_records, original_scene, physical_task
from wia.authoring.io import sha256, write_json

MODULE = 'core.esi_converters.spatial_relations_linear_alignment'
OBJECTS = ('alignment_1', 'alignment_2', 'alignment_3')
OPERATIONS = ('opposite_flank_survey_return', 'endpoint_survey_return', 'parallax_arc_survey_return')
ARRANGEMENT_OPERATION = 'arrange_released_collinear_objects'
RESOLUTION_M = .01


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _vector(value, size):
    return isinstance(value, (list, tuple)) and len(value) == size and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)
        for item in value)


def _baseline(points):
    """Longest pair defines the baseline even for a noncollinear source."""
    points = np.asarray(points, dtype=float)[:, :2]
    pairs = ((0, 1), (0, 2), (1, 2))
    i, j = max(pairs, key=lambda pair: float(np.linalg.norm(points[pair[1]] - points[pair[0]])))
    span = float(np.linalg.norm(points[j] - points[i]))
    if span < .05:
        raise ValueError('degenerate_original_baseline')
    direction = (points[j] - points[i]) / span
    normal = np.array([-direction[1], direction[0]])
    centre = (points[i] + points[j]) / 2
    residual = float(abs(np.dot(points[3 - i - j] - centre, normal)))
    return centre, direction, normal, span, (i, j), residual


def _source_data(source):
    raw, payload = source.raw, source.payload
    if (source.big_task, source.small_task) != CategoryConverter.category or (
            raw.get('big_task'), raw.get('small_task')) != CategoryConverter.category:
        raise ValueError('category_mismatch')
    if tuple(Path(source.relative_path).parts[-3:-1]) != (raw.get('scene'), raw.get('room')):
        raise ValueError('source_scene_room_mismatch')
    if raw.get('runner_task') != 'line' or raw.get('answer') not in ('yes', 'no'):
        raise ValueError('unsupported_original_qa')
    objects = map_records(payload.get('objects'))
    values = [objects.get('obj' + str(i), {}) for i in (1, 2, 3)]
    for i, obj in enumerate(values, 1):
        if not obj.get('category') or not obj.get('model') or obj['model'] == 'scene_native':
            raise ValueError('missing_original_asset:obj' + str(i))
        if obj.get('name', 'obj' + str(i)) != 'obj' + str(i):
            raise ValueError('unexpected_original_object_name')
        if not _vector(obj.get('position'), 3) or not _vector(obj.get('quaternion_xyzw'), 4) or not any(
                obj['quaternion_xyzw']):
            raise ValueError('invalid_original_object_pose')
        scale = obj.get('scale', 1.)
        scale = [scale] * 3 if isinstance(scale, (int, float)) else scale
        if not _vector(scale, 3) or min(scale) <= 0:
            raise ValueError('invalid_original_object_scale')
    if len({obj['category'] for obj in values}) != 3 or not raw.get('question'):
        raise ValueError('original_public_object_identity_unresolved')
    poses = payload.get('camera_poses', [])
    poses = list(poses.values()) if isinstance(poses, dict) else poses
    if not poses or any(not _vector(pose.get('position'), 3) or not _vector(
            pose.get('quaternion_xyzw'), 4) or not any(pose['quaternion_xyzw']) for pose in poses):
        raise ValueError('missing_original_reference_camera')
    if not payload.get('floor_name'):
        raise ValueError('missing_original_floor')
    _baseline([obj['position'] for obj in values])
    return raw, payload, values, poses


@lru_cache(maxsize=2)
def _room_map(asset_root, scene_name, room_name):
    """Read original maps with native BaseMap and SegmentationMap conventions."""
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
    if room_name not in room_ids:
        return None
    free = ((trav > 0) & (rooms == room_ids[room_name])).astype(np.uint8)
    clearance = cv2.distanceTransform(np.pad(free, 1), cv2.DIST_L2,
                                     cv2.DIST_MASK_PRECISE)[1:-1, 1:-1] * RESOLUTION_M
    return clearance, {str(path): sha256(path) for path in [*paths, categories]}


def _gate_masks(xy, baseline, operation):
    centre, direction, normal, span = baseline[:4]
    offset = np.asarray(xy) - centre
    along, across = offset @ direction, offset @ normal
    if operation == OPERATIONS[0]:
        longitudinal = np.abs(along) <= span / 2
        return [(longitudinal & (sign * across >= .8) & (sign * across <= 1.3))
                for sign in (-1, 1)]
    if operation == OPERATIONS[2]:
        radius = np.linalg.norm(offset, axis=-1)
        # Distinct bearing and separation are checked against the first actual
        # native stop by the evaluator, rather than by fixed world targets.
        annulus = (radius >= .9) & (radius <= 1.6)
        return [annulus, annulus]
    return [((sign * along >= span / 2 + .75) & (sign * along <= span / 2 + 1.3)
             & (np.abs(across) <= .5)) for sign in (-1, 1)]


def _parallax_pairs(candidates, labels, centre, reference):
    """Bounded angular representatives screen pairs without quadratic pixel work."""
    offset = candidates - centre
    radius = np.linalg.norm(offset, axis=1)
    valid = (radius >= 1.) & (radius <= 1.5)
    points, components = candidates[valid], labels[valid]
    pairs = {}
    for component in sorted(set(components) - {0}):
        points_here = points[components == component]
        offsets = points_here - centre
        angles = np.arctan2(offsets[:, 1], offsets[:, 0])
        bins = np.floor((angles + math.pi) * 72 / (2 * math.pi)).astype(int) % 72
        representatives = []
        for index in np.unique(bins):
            group = points_here[bins == index]
            representatives.append(group[int(np.argmin(np.linalg.norm(group - reference, axis=1)))])
        qualified = []
        for i, first in enumerate(representatives):
            for second in representatives[i + 1:]:
                a, b = first - centre, second - centre
                bearing = math.acos(float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1., 1.)))
                if bearing >= math.radians(50) and np.linalg.norm(first - second) >= .8:
                    qualified.append((float(np.linalg.norm(first - reference) + np.linalg.norm(second - reference)),
                                      first.tolist(), second.tolist()))
        if qualified:
            _, first, second = min(qualified)
            pairs[int(component)] = [first, second]
    return pairs


def _route_screen(raw, values, poses, context):
    """A reversible start and two connected station regions, never a motion plan."""
    import cv2
    root = context.config.get('original_asset_root', str(Path(os.environ.get(
        'OMNIGIBSON_DATA_PATH', '/local_home2/zhouxinyang/data/behavior')) / 'behavior-1k-assets'))
    maps = _room_map(str(root), raw['scene'], raw['room'])
    evidence = {'map_asset_root': str(root), 'map_resolution_m': RESOLUTION_M,
                'body_screen_radius_m': .4, 'map_clearance_m': .42,
                'original_object_exclusion_radius_m': .7, 'native_qualified': False}
    if maps is None:
        return None, {**evidence, 'failure_reason': 'original_room_maps_unavailable'}
    clearance, hashes = maps
    baseline = _baseline([obj['position'] for obj in values])
    centre = baseline[0]
    size = clearance.shape[0]
    centre_rc = (centre[::-1] / RESOLUTION_M + size / 2).astype(int)
    low, high = np.maximum(centre_rc - 302, 0), np.minimum(centre_rc + 303, size)
    evidence.update(map_sha256=hashes, floor_z_assumption_m=0.,
        floor_z_provenance='Both source generators render saved floor cameras at absolute z=0.025 m.',
        screen_kind='Connected components of original room map with object-root exclusion disks.')
    if np.any(low >= high):
        return None, {**evidence, 'failure_reason': 'source_cluster_outside_original_maps'}
    rr, cc = np.mgrid[low[0]:high[0], low[1]:high[1]]
    xy = np.stack((cc - size / 2, rr - size / 2), axis=-1) * RESOLUTION_M
    valid = clearance[low[0]:high[0], low[1]:high[1]] >= .42
    valid &= np.linalg.norm(xy - centre, axis=-1) <= 3.
    for obj in values:
        valid &= np.linalg.norm(xy - obj['position'][:2], axis=-1) >= .7
    _, components = cv2.connectedComponents(valid.astype(np.uint8), connectivity=4)
    candidates = xy[valid]
    labels = components[valid]
    evidence['safe_start_pixel_count'] = len(candidates)
    evidence['screened_operation_components'] = {}
    for operation in OPERATIONS:
        first_camera = np.asarray(poses[0]['position'][:2])
        gates = _gate_masks(candidates, baseline, operation)
        parallax = (_parallax_pairs(candidates, labels, centre, first_camera)
                    if operation == OPERATIONS[2] else {})
        shared = (sorted(parallax) if operation == OPERATIONS[2] else
                  [int(value) for value in sorted((set(labels[gates[0]]) & set(labels[gates[1]])) - {0})])
        evidence['screened_operation_components'][operation] = shared
        if not shared:
            continue
        choices = candidates[np.isin(labels, shared)]
        start = choices[int(np.argmin(np.linalg.norm(choices - first_camera, axis=1)))]
        start_rc = (start[::-1] / RESOLUTION_M + size / 2).round().astype(int) - low
        component = int(components[tuple(start_rc)])
        selected_gates = [candidates[mask & (labels == component)] for mask in gates]
        stations = [gate[int(np.argmin(np.linalg.norm(gate - start, axis=1)))].tolist()
                    for gate in selected_gates]
        if operation == OPERATIONS[2]:
            stations = parallax[component]
        evidence.update(selected_component=component, example_station_xy_m=stations,
                        start_selection='Nearest screened connected-component pixel to first original camera XY.')
        return {'position': [float(start[0]), float(start[1]), 0.],
                'yaw_deg': math.degrees(math.atan2(centre[1] - start[1], centre[0] - start[0])),
                'operation': operation}, evidence
    return None, {**evidence, 'failure_reason': 'no_connected_source_faithful_survey_stations'}


def _public(raw, operation):
    if operation == OPERATIONS[0]:
        route = ('Drive to both sides of the baseline between the two objects furthest apart, '
                 'in either order. At each stop, stay longitudinally between those endpoints '
                 'and 0.8 to 1.3 m perpendicular to their baseline.')
    elif operation == OPERATIONS[1]:
        route = ('Drive beyond both ends of the baseline between the two objects furthest apart, '
                 'in either order. At each stop, stand 0.75 to 1.3 m beyond the endpoint '
                 'and within 0.5 m sideways of the extended baseline.')
    else:
        route = ('Drive to two distinct survey viewpoints 0.9 to 1.6 m from the midpoint '
                 'between the two objects furthest apart. The viewpoints must be at least '
                 '0.75 m apart and differ by at least 45 degrees in bearing around that midpoint.')
    return {'objective': route + ' At both stations stop for 0.4 s facing the formation midpoint '
            'within 30 degrees. Then return within 0.25 m of your start and stop for 0.5 s.',
        'success_criteria': 'Complete both distinct physical survey stops, travel at least 1.5 m '
            'with at least 0.75 m excursion, and return stationary. Keep all three original '
            'objects in place on their starting support while the robot stays on its original floor throughout.',
        'environment': 'The three original-size objects occupy an existing tabletop in their original room. '
            'Distances between their horizontal positions may be uniformly scaled during initialization '
            'without changing individual object sizes; the formation may also be translated, '
            'preserving collinearity and order. The original floor layout remains a historical variant. '
            'Auxiliary initial question: ' + raw['question'],
        'constraints': ['Use mounted robot observations and existing guarded base controls.',
            'Avoid base collisions and contact with task objects; keep at least 0.4 m from every task object footprint. Legal arm manipulation of noncritical native movable furniture is allowed.',
            'Stay within 3 m of the baseline midpoint. Do not grasp, move or rearrange the objects.',
            'The original initial question is auxiliary; answering it does not complete the route.']}


def _arrangement_public(raw, values):
    original = _public(raw, OPERATIONS[0])
    labels = [obj['category'].replace('_', ' ') for obj in values]
    names = ', '.join(labels[:-1]) + ' and ' + labels[-1]
    return {**original,
        'objective': 'Arrange the ' + names + ' in a straight line on their starting support, in any '
            'object order. If they already satisfy the goal, you may stop without moving anything.',
        'success_criteria': 'Use the horizontal centers of the three native object bounding boxes. '
            'Every pair of centers must be more than 0.0001 m apart. Using the longest pair as a '
            'baseline of length L, the maximum perpendicular center distance from that line must be '
            'at most min(0.02 m, 2% of L). All three objects must currently touch and rest on their '
            'starting support, remain collidable, and have linear speed at most 0.03 m/s and angular '
            'speed at most 0.1 rad/s. Release the objects with no held object or active grasp constraint. '
            'The robot must remain upright with wheels on the original floor and no forbidden external '
            'contact. No travel, grasp, lift or minimum motion is required.',
        'constraints': ['Use mounted robot observations and existing guarded controls to arrange the named objects.',
            'Preserve the starting support and task references. Noncritical native movable furniture may be moved using guarded arm and gripper actions.',
            'The original initial question remains auxiliary and separate from the arrangement goal.']}


def revise_goal(task_definition):
    """Return an idempotent goal-only revision without changing any scene field."""
    revised = copy.deepcopy(task_definition)
    evaluation = revised['private']['evaluation']
    if module_identity(evaluation.get('module')) != MODULE:
        raise ValueError('Goal-only revision requires an existing Linear Alignment task.')
    values = [revised['private']['bindings'][key] for key in OBJECTS]
    public = _arrangement_public({'question': ''}, values)
    for key in ('objective', 'success_criteria', 'constraints'):
        revised['public'][key] = public[key]
    if evaluation['operation'] != ARRANGEMENT_OPERATION:
        revised['version'] += 1
    evaluation['operation'] = ARRANGEMENT_OPERATION
    capabilities = revised['requirements']['capabilities']
    revised['requirements']['capabilities'] = list(dict.fromkeys([*capabilities,
        'base_translation', 'base_rotation', 'ee_translation', 'ee_rotation', 'trunk_height', 'gripper']))
    return revised


def _relation_geometry(centres):
    centres = np.asarray(centres, dtype=float)
    if centres.shape != (3, 2) or not np.isfinite(centres).all():
        raise ValueError('invalid_native_relation_centres')
    pairs = ((0, 1), (0, 2), (1, 2))
    lengths = np.asarray([np.linalg.norm(centres[i] - centres[j]) for i, j in pairs])
    longest = float(lengths.max())
    distinct = bool(np.all(lengths > 1e-4))
    i, j = pairs[int(np.argmax(lengths))]
    direction = (centres[j] - centres[i]) / max(longest, 1e-12)
    normal = np.array([-direction[1], direction[0]])
    residual = float(np.max(np.abs((centres - centres[i]) @ normal)))
    tolerance = min(.02, .02 * longest)
    return {'pairwise_xy_m': lengths.tolist(), 'distinct_centres': distinct,
            'longest_baseline_length_m': longest, 'maximum_perpendicular_distance_m': residual,
            'perpendicular_tolerance_m': tolerance,
            'target_relation_satisfied': bool(distinct and residual <= tolerance)}


def _arrangement_measure(session, executor, config, bound):
    rows = []
    for key in OBJECTS:
        obj = bound[key]
        row = _object_state(obj)
        low, high = _array(row['aabb_min']), _array(row['aabb_max'])
        linear, angular = _array(obj.get_linear_velocity()), _array(obj.get_angular_velocity())
        if (np.any(high <= low) or linear.shape != (3,) or angular.shape != (3,) or
                not np.isfinite(np.r_[linear, angular]).all()):
            raise ValueError('invalid_native_relation_bounds_or_velocity')
        rows.append({**row, 'fixed_base': bool(obj.fixed_base), 'centre_xy': ((low + high)[:2] / 2).tolist(),
            'linear_speed_m_s': float(np.linalg.norm(linear)),
            'angular_speed_rad_s': float(np.linalg.norm(angular))})
    floor = bound['source_floor']
    floor_top = float(_array(floor.aabb[1])[2])
    if not math.isfinite(floor_top):
        raise ValueError('invalid_native_source_floor')
    robot = _native_support(session, floor, executor.pose())
    support = _task_support(session, bound)
    geometry = _relation_geometry([row['centre_xy'] for row in rows])
    gripper = dict(executor.debug_gripper_state())
    checks = {
        'three_distinct_original_objects': len({id(bound[key]) for key in OBJECTS}) == 3 and
            len({row['name'] for row in rows}) == 3 and
            all(bound[key].name == config['bindings'][key] for key in OBJECTS),
        'original_floor_collidable': floor.category in ('floor', 'floors') and not floor.visual_only,
        'objects_collidable': all(not row['visual_only'] for row in rows),
        'distinct_noncoincident_centres': geometry['distinct_centres'],
        'target_relation_satisfied': geometry['target_relation_satisfied'],
        'objects_on_starting_support': all(all(row.values()) for row in support),
        'objects_currently_stationary': all(row['linear_speed_m_s'] <= .03 and
                                           row['angular_speed_rad_s'] <= .1 for row in rows),
        'objects_released': gripper.get('object_name') is None and gripper.get('holding') is False and
                            gripper.get('constraint_active') is False,
        'robot_on_original_floor_upright_and_clear': robot['supported_upright']}
    return {'objects': rows, 'geometry': geometry, 'object_task_support': support,
            'native_gripper': gripper, 'native_robot_support': robot, 'criteria': checks}


def _qualify_arrangement(session, executor, config, bound):
    """Check initialized references and safety, without requiring the desired relation."""
    checks = {}
    try:
        from wia.tasks.interaction_setup import reference_position
        bound = _configuration(session, config, bound)
        measured = _arrangement_measure(session, executor, config, bound)
        checks = {key: bool(value) for key, value in measured['criteria'].items()
                  if key != 'target_relation_satisfied'}
        p = config['parameters']
        references = [reference_position(session, bound[key].name, original)
                      for key, original in zip(OBJECTS, p['source_positions'])]
        checks['initialized_reference_positions_match'] = len(references) == 3 and all(
            math.dist(row['position'], reference) <= p['initial_pose_tolerance_m']
            for row, reference in zip(measured['objects'], references))
        checks['initialized_quaternions_match'] = len(p['source_quaternions']) == 3 and all(
            _orientation_delta(row['quaternion_xyzw'], original) <= p['orientation_tolerance_rad']
            for row, original in zip(measured['objects'], p['source_quaternions']))
        accepted = all(checks.values())
        return {'accepted': accepted, 'checks': checks, 'native_state': measured,
                'reason': None if accepted else 'native_relation_initialization_failed',
                'native_execution_verified': False}
    except Exception as error:
        return {'accepted': False, 'checks': checks, 'reason': 'native_relation_measurement_unavailable',
                'exception_summary': f'{type(error).__name__}: {error}', 'native_execution_verified': False}


class RelationArrangementEvaluator:
    """Read current released relation; a satisfied initial state requires no action."""
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.bound = _configuration(session, config)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.initial, self.episode = None, 0

    def begin_episode(self, observation):
        self.initial_qualification = _qualify_arrangement(self.session, self.controller, self.config, self.bound)
        if not self.initial_qualification['accepted']:
            raise ValueError('Episode native preconditions failed: ' + str(self.initial_qualification['reason']))
        self.episode += 1
        self.initial = _arrangement_measure(self.session, self.controller, self.config, self.bound)
        return self.record(observation, action=None)

    def _state(self):
        try:
            measured = _arrangement_measure(self.session, self.controller, self.config, self.bound)
            checks = {**measured['criteria'], 'initial_native_qualification': self.initial_qualification['accepted']}
            return {'task_success': bool(all(checks.values())), 'criteria': checks, 'native_state': measured}
        except Exception as error:
            return {'task_success': False, 'criteria': {'native_measurements_available': False},
                    'measurement_error': f'{type(error).__name__}: {error}'}

    def sample_step(self):
        if self.initial is not None:
            return self._state()

    def record(self, observation, *, action=None):
        if self.initial is None:
            return None
        value = self._state()
        value.update(schema='private_relation_arrangement_v1', episode=self.episode,
            snapshot_id=getattr(observation, 'snapshot_id', None), action_step=getattr(observation, 'action_step', None),
            action=action, operation=self.config['operation'], configuration=self.config,
            initial_native_state=self.initial, initial_qualification=self.initial_qualification,
            stability_evidence='current_boundary_contact_and_velocity_only')
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


class CategoryConverter(ConverterBase):
    category = ('Spatial Relations', 'Linear Alignment')
    version = 'linear_alignment_v2.1_task_edit_20261018_v1'

    def inspect(self, source):
        try:
            _source_data(source)
            return {'eligible': True, 'reason': 'original_alignment_assets_and_observer_preserved'}
        except (ValueError, KeyError, TypeError) as error:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2:' + str(error),
                    'warning_codes': ['original_source_precondition_failed']}

    def convert(self, source, context):
        raw, payload, values, poses = _source_data(source)
        if context.config.get('operation') not in (*OPERATIONS, ARRANGEMENT_OPERATION):
            from .spatial_relations_geometric_configuration import convert_arrangement
            return convert_arrangement(source, context, raw, payload, values, OBJECTS, MODULE, 'collinear')
        bindings = {key: {'name': 'obj' + str(i), 'category': obj['category'], 'model': obj['model']}
                    for i, (key, obj) in enumerate(zip(OBJECTS, values), 1)}
        bindings['source_floor'] = {'name': payload['floor_name']}
        baseline = _baseline([obj['position'] for obj in values])
        geometry = {'source_answer_private': raw['answer'], 'root_baseline_pair_indices': list(baseline[4]),
                    'root_baseline_span_m': baseline[3], 'root_perpendicular_residual_m': baseline[5],
                    'diagnostic_only': True}
        robot, screen = _route_screen(raw, values, poses, context)
        assumptions = [{'doubt': 'A source camera and a 2D room map do not certify R1 floor height, '
            'full-body clearance, object footprint sizes, camera visibility or an executable route.',
            'chosen_assumption': 'Use a z=0 R1 root nearest the first source camera in a screened '
            'room-map component connecting two survey regions; require native source-floor wheel '
            'contacts, unchanged collidable objects and continuous safe movement before success.',
            'alternatives': ['Keep source-only diagnostics when no connected stations exist.',
                'Review another reversible robot start using native evidence.'],
            'scope': source.relative_path,
            'risk': 'Map obstacles, root-versus-center differences or incorrect floor height can '
                'invalidate qualification or the route. Static screening is not physical certification.'}, {
            'doubt': 'The original root spacing may exceed available same-room tabletops.',
            'chosen_assumption': 'Request a reversible existing-support layout using one positive uniform '
                'horizontal scale of root positions and common horizontal/vertical translation. Keep every '
                'asset size and quaternion, object order, collinearity or noncollinearity, and original QA. '
                'Require measured support fit, disjoint prop AABBs and live upward support contact.',
            'alternatives': ['Retain the historical original variant if no supported similarity layout qualifies.'],
            'scope': source.relative_path,
            'risk': 'Spacing and support context change, so the layout and mounted views need Review. '
                'Height alone certifies neither graspability nor navigation feasibility.'}]
        warnings = ['assumption_review_required', 'native_route_unverified', 'mounted_visibility_unverified',
                    'layout_uniform_position_scale_requires_review', 'native_navigation_unverified']
        if raw['answer'] == 'yes' and baseline[5] > .03:
            warnings.append('source_collinearity_annotation_uncertainty')
            assumptions.append({'doubt': 'The saved yes label has a root-to-longest-baseline '
                'residual above 0.03 m. The positive generator sets equal Y before settling; '
                'root positions, rendered object centers and the annotation may differ.',
                'chosen_assumption': 'Preserve the original answer, order and similarity relation. Define the '
                'physical route from the actual native object AABB centers independently of QA.',
                'alternatives': ['Review the original rendered centers and annotation provenance.',
                    'Retain the source as an uncertain QA annotation.'],
                'scope': source.relative_path,
                'risk': 'Original yes/no interpretation may be ambiguous; this diagnostic threshold '
                    'is not an authoritative source-label correction or a success predicate.'})
        result = {'status': 'generated', 'reuse_level': 'L2', 'qa_preserved': True,
            'scene_definition': original_scene(source, context, bindings=bindings), 'task_definition': None,
            'warning_codes': warnings, 'assumptions': assumptions, 'source_geometry_evidence': geometry,
            'robot_start_evidence': screen, 'attempted_repairs': [],
            'change_summary': ['Preserve original assets, sizes, quaternions, reference camera and private QA; '
                'retain source poses in the historical original variant.',
                'Request an existing same-room tabletop with uniformly scaled horizontal root positions '
                'and common translation, preserving alignment and object order.',
                'Retain the guarded R1 navigation survey and stationary return around the initialized formation.'],
            'limitations': ['CPU construction and map connectivity do not certify native loading, '
                'reachability, mounted visibility or physical completion.',
                'The historical assessment reports one construction/control probe, not route success.',
                'Root-based map screening approximates native AABB centers and object footprints.'],
            'next_step': 'Publish every candidate and diagnostic for Review; qualify actual native '
                'task-support contact, robot wheel support on the source floor and guarded navigation before readiness.'}
        if robot is None:
            result.update(status='conversion_failed', reuse_level='L1',
                failure_reason='not_convertible_under_L1_L2:' + screen['failure_reason'],
                diagnostic_stage='robot_route_screen')
            result['warning_codes'].append('source_only_diagnostic')
            return result
        operation = context.config.get('operation', robot['operation'])
        patches = [{'operation': 'robot_pose', 'position': robot['position'],
                    'yaw_deg': robot['yaw_deg'], 'head_pitch_deg': 45.}]
        # The source uses the installed R1 through pipeline.build_env_config.
        # Retain its numerical settings and explicitly preserve collision physics
        # on either side of source postprocessing. Shared native initialization
        # may subsequently transform root spacing without changing asset sizes.
        for key in OBJECTS:
            patches.extend({'operation': 'object_physics', 'phase': phase,
                            'binding': key, 'visual_only': False}
                           for phase in ('before_construction', 'after_postprocessing'))
        parameters = {'source_positions': [obj['position'] for obj in values],
            'source_quaternions': [obj['quaternion_xyzw'] for obj in values],
            'source_floor_z_m': 0., 'initial_pose_tolerance_m': .04,
            'object_preservation_m': .02, 'orientation_tolerance_rad': math.radians(5),
            'minimum_object_clearance_m': .4, 'maximum_midpoint_distance_m': 3.,
            'maximum_step_displacement_m': .1, 'maximum_native_sample_gap_s': .1,
            'return_distance_m': .25, 'minimum_path_m': 1.5, 'minimum_excursion_m': .75,
            'station_heading_tolerance_rad': math.radians(30), 'station_duration_s': .4,
            'maximum_stop_speed_m_s': .03, 'maximum_stop_angular_speed_rad_s': .1,
            'stop_duration_s': .5}
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': operation,
                      'binding_keys': [*OBJECTS, 'source_floor'], 'parameters': parameters}
        result.update(scene_definition=original_scene(source, context, bindings=bindings, patches=patches),
            task_definition=physical_task(source, context, public=_public(raw, operation),
                evaluation=evaluation, bindings=bindings, patches=patches,
                capabilities=['base_translation', 'base_rotation'],
                qualification={'on_failure': 'skip', 'initial_visibility': 'none',
                               'target_movable_collidable': False}), operation=operation,
            robot_start={**robot, 'native_qualified': False})
        result['change_summary'].append('Declare reversible map-screened R1 initialization '
            'and collidable objects; retain an empty historical original variant.')
        if context.config.get('operation') in OPERATIONS:
            return result
        result['task_definition'] = revise_goal(result['task_definition'])
        result['operation'] = ARRANGEMENT_OPERATION
        result['change_summary'][2] = ('Replace the survey objective and success predicate with a direct '
            'released-line arrangement goal; preserve the existing scene and initialization settings.')
        return result

    def convert_all(self, sources, context):
        """Retain shared immutable candidates, original QA and per-source failures."""
        return super().convert_all(sources, context)

    def sample(self, manifest, policy=None):
        count = (policy or {}).get('count', 5)
        rows = sorted(manifest['records'], key=lambda row: row['source_identity'])
        prior = set((policy or {}).get('previous_failures', ()))
        selected = [row['candidate_id'] for row in rows if row['candidate_id'] in prior][:count]
        strata = [lambda row: row['status'] != 'generated',
            lambda row: row.get('operation') == OPERATIONS[0],
            lambda row: row.get('operation') == OPERATIONS[1],
            lambda row: row.get('operation') == OPERATIONS[2],
            lambda row: 'source_collinearity_annotation_uncertainty' in row['warning_codes'],
            lambda row: row.get('source_geometry_evidence', {}).get('source_answer_private') == 'yes']
        for predicate in strata:
            matches = [row['candidate_id'] for row in rows if predicate(row) and row['candidate_id'] not in selected]
            if matches and len(selected) < count:
                selected.append(matches[0])
        return (selected + [key for key in super().sample(manifest, policy) if key not in selected])[:count]


def _orientation_delta(a, b):
    a, b = _array(a), _array(b)
    cosine = abs(float(np.dot(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 2 * math.acos(float(np.clip(cosine, 0., 1.)))


def _object_state(obj):
    position, quaternion = (_array(value) for value in obj.get_position_orientation())
    low, high = (_array(value) for value in obj.aabb)
    linear, angular = (_array(value) for value in (obj.get_linear_velocity(), obj.get_angular_velocity()))
    if (any(value.shape != (3,) for value in (position, low, high, linear, angular)) or
            quaternion.shape != (4,) or not np.isfinite(np.concatenate(
                (position, quaternion, low, high, linear, angular))).all() or
            np.linalg.norm(quaternion) == 0 or np.any(low > high)):
        raise ValueError('invalid_native_original_object_measurement')
    return {'name': obj.name, 'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
            'aabb_min': low.tolist(), 'aabb_max': high.tolist(), 'centre_xy': ((low + high)[:2] / 2).tolist(),
            'visual_only': bool(obj.visual_only), 'fixed_base': bool(obj.fixed_base),
            'linear_speed_m_s': float(np.linalg.norm(linear)),
            'angular_speed_rad_s': float(np.linalg.norm(angular))}


def _native_support(session, floor, pose):
    """Require native wheel contacts with this floor, rejecting other supports."""
    robot = session.robot
    owns = lambda path, root: path == root or path.startswith(root + '/')
    root, floor_root = str(robot.prim_path), str(floor.prim_path)
    floor_z = float(_array(floor.aabb[1])[2])
    wheels = {str(robot.links[name].prim_path) for name in robot.floor_touching_base_link_names}
    supported_wheels, contacts, forbidden = set(), [], []
    for contact in robot.contact_list():
        if legal_background_contact(session, contact.body0, contact.body1):
            continue
        bodies = [str(contact.body0), str(contact.body1)]
        internal = [owns(body, root) for body in bodies]
        if all(internal):
            continue
        external = [body for body, inside in zip(bodies, internal) if not inside]
        wheel = next((wheel for wheel in wheels if any(owns(body, wheel) for body in bodies)), None)
        position, normal = _array(contact.position), _array(contact.normal)
        row = {'bodies': bodies, 'position': position.tolist(), 'normal': normal.tolist()}
        if (any(internal) and wheel and external and all(owns(body, floor_root) for body in external)
                and position.shape == normal.shape == (3,) and np.isfinite([position, normal]).all()
                and abs(position[2] - floor_z) <= .02 and abs(normal[2]) >= .5):
            supported_wheels.add(wheel)
            contacts.append(row)
        else:
            forbidden.append(row)
    q = _array(pose['quaternion_xyzw'])
    q = q / np.linalg.norm(q)
    upright = 1 - 2 * (q[0] ** 2 + q[1] ** 2) >= .9
    low = _array(robot.aabb[0])
    return {'floor_top_m': floor_z, 'source_floor_wheel_contacts': contacts,
            'supported_wheels': sorted(supported_wheels), 'other_external_contacts': forbidden,
            'supported_upright': bool(len(supported_wheels) >= 2 and not forbidden and upright
                                      and abs(low[2] - floor_z) < .15)}


def _configuration(session, config, bound=None):
    if (module_identity(config.get('module')) != MODULE or config.get('operation') not in (*OPERATIONS, ARRANGEMENT_OPERATION) or
            tuple(config.get('binding_keys', ())) != (*OBJECTS, 'source_floor') or
            (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name)):
        raise ValueError('native_operation_configuration_mismatch')
    if bound is None:
        bound = {key: session.env.scene.object_registry('name', config['bindings'][key])
                 for key in config['binding_keys']}
    if any(bound.get(key) is None for key in config['binding_keys']):
        raise ValueError('missing_native_original_binding')
    return bound


def _task_support(session, bound):
    """Read live support contacts without confusing them with robot floor contact."""
    from wia.tasks.interaction_setup import touching_task_support, on_top_task_support
    floor = bound['source_floor']
    return [{'touching_task_support': touching_task_support(session, bound[key], floor),
             'on_top_task_support': on_top_task_support(session, bound[key], floor)} for key in OBJECTS]


def qualify_operation(session, executor, config, bound):
    """Read native state only; never act, set a pose, step physics or consult QA."""
    if config.get('operation') == 'arrange_released_source_shape':
        from .spatial_relations_geometric_configuration import qualify_arrangement
        return qualify_arrangement(session, executor, config, bound)
    if config.get('operation') == ARRANGEMENT_OPERATION:
        return _qualify_arrangement(session, executor, config, bound)
    checks = {}
    try:
        from wia.tasks.interaction_setup import reference_position
        bound = _configuration(session, config, bound)
        p, floor = config['parameters'], bound['source_floor']
        if floor.category not in ('floor', 'floors') or floor.visual_only:
            raise ValueError('source_floor_not_collidable')
        states = [_object_state(bound[key]) for key in OBJECTS]
        checks['native_objects'] = states
        references = [reference_position(session, bound[key].name, original)
                      for key, original in zip(OBJECTS, p['source_positions'])]
        errors = [math.dist(row['position'], reference) for row, reference in zip(states, references)]
        checks['reference_position_errors_m'] = errors
        checks['adapted_reference_positions'] = [reference.tolist() for reference in references]
        if max(errors) > p['initial_pose_tolerance_m'] or any(row['visual_only'] for row in states):
            raise ValueError('native_original_arrangement_not_reproduced')
        if any(_orientation_delta(row['quaternion_xyzw'], original) > p['orientation_tolerance_rad']
               for row, original in zip(states, p['source_quaternions'])):
            raise ValueError('native_original_orientation_not_reproduced')
        floor_z = float(_array(floor.aabb[1])[2])
        checks['native_object_task_support'] = _task_support(session, bound)
        if (abs(floor_z - p['source_floor_z_m']) > .02 or
                not all(all(row.values()) for row in checks['native_object_task_support']) or
                any(row['linear_speed_m_s'] > .03
                    or row['angular_speed_rad_s'] > .1 for row in states)):
            raise ValueError('native_original_floor_or_task_support_mismatch')
        checks['robot_support'] = _native_support(session, floor, executor.pose())
        if not checks['robot_support']['supported_upright']:
            raise ValueError('native_robot_source_floor_support_or_clearance_failed')
        baseline = _baseline([row['centre_xy'] for row in states])
        xy = np.asarray(executor.pose()['position'][:2])
        clearances = [float(np.linalg.norm(xy - np.clip(xy, row['aabb_min'][:2], row['aabb_max'][:2])))
                      for row in states]
        checks['native_object_clearances_m'] = clearances
        checks['native_baseline_span_m'] = baseline[3]
        if min(clearances) < p['minimum_object_clearance_m'] or np.linalg.norm(xy - baseline[0]) > p[
                'maximum_midpoint_distance_m']:
            raise ValueError('native_robot_outside_safe_survey_neighborhood')
        checks['route_feasibility_verified'] = False
        return {'accepted': True, 'checks': checks, 'reason': None}
    except (ValueError, RuntimeError, AttributeError, KeyError, TypeError, IndexError) as error:
        return {'accepted': False, 'checks': checks, 'reason': str(error)}


class AlignmentSurveyEvaluator:
    """Measure real two-station travel and return on consecutive native states."""
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.bound = _configuration(session, config)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.initial, self.episode = None, 0

    def _measure(self):
        pose = self.controller.pose()
        states = [_object_state(self.bound[key]) for key in OBJECTS]
        floor = self.bound['source_floor']
        support = _native_support(self.session, floor, pose)
        xy = np.asarray(pose['position'][:2])
        clearances = [float(np.linalg.norm(xy - np.clip(xy, row['aabb_min'][:2], row['aabb_max'][:2])))
                      for row in states]
        time_s = float(self.session.pipeline.og.sim.current_time)
        speed = float(np.linalg.norm(_array(self.session.robot.get_linear_velocity())))
        angular = float(np.linalg.norm(_array(self.session.robot.get_angular_velocity())))
        if not all(math.isfinite(value) for value in (time_s, speed, angular)):
            raise ValueError('invalid_native_clock_or_robot_velocity')
        return {'base': pose, 'objects': states, 'support': support, 'time_s': time_s,
                'object_clearances_m': clearances, 'speed_m_s': speed, 'angular_speed_rad_s': angular,
                'object_task_support': _task_support(self.session, self.bound)}

    def begin_episode(self, observation):
        qualified = qualify_operation(self.session, self.controller, self.config, self.bound)
        if not qualified['accepted']:
            raise ValueError('Episode native preconditions failed: ' + str(qualified['reason']))
        self.episode += 1
        self.initial = self._measure()
        self.baseline = _baseline([row['centre_xy'] for row in self.initial['objects']])
        self.last = np.asarray(self.initial['base']['position'])
        self.last_time = self.initial['time_s']
        self.path_m, self.excursion_m, self.station_seconds, self.quiet_seconds = 0., 0., 0., 0.
        self.completed_stations, self.active_station = [], None
        self.station_evidence = []
        self.previously_quiet, self.trajectory_valid = False, True
        self.record(observation, action=None)

    def _checks(self, measured):
        p = self.config['parameters']
        preserved = all(not row['visual_only'] and row['fixed_base'] == original['fixed_base']
            and math.dist(row['position'], original['position']) <= p['object_preservation_m']
            and _orientation_delta(row['quaternion_xyzw'], original['quaternion_xyzw']) <= p['orientation_tolerance_rad']
            and max(math.dist(row[edge], original[edge]) for edge in ('aabb_min', 'aabb_max')) <= p['object_preservation_m']
            and row['linear_speed_m_s'] <= .03 and row['angular_speed_rad_s'] <= .1
            for row, original in zip(measured['objects'], self.initial['objects']))
        position = measured['base']['position']
        return {'original_arrangement_preserved': bool(preserved and all(
                    all(row.values()) for row in measured['object_task_support'])),
            'source_floor_supported_and_clear': bool(measured['support']['supported_upright'] and
                abs(measured['support']['floor_top_m'] - self.initial['support']['floor_top_m']) <= .005 and
                min(measured['object_clearances_m']) >= p['minimum_object_clearance_m']),
            'within_survey_neighborhood': float(np.linalg.norm(np.asarray(position[:2]) - self.baseline[0])) <= p[
                'maximum_midpoint_distance_m'],
            'at_return_position': math.dist(position[:2], self.initial['base']['position'][:2]) <= p['return_distance_m'],
            'stopped': bool(measured['speed_m_s'] <= p['maximum_stop_speed_m_s'] and
                           measured['angular_speed_rad_s'] <= p['maximum_stop_angular_speed_rad_s'])}

    def sample_step(self):
        if self.initial is None:
            return
        measured, p = self._measure(), self.config['parameters']
        dt = measured['time_s'] - self.last_time
        if dt < -1e-7 or dt > p['maximum_native_sample_gap_s'] + 1e-7:
            self.trajectory_valid = False
        if dt <= 1e-7:
            return
        position = np.asarray(measured['base']['position'])
        checks = self._checks(measured)
        self.trajectory_valid &= bool(np.linalg.norm(position - self.last) <= p['maximum_step_displacement_m']
            and all(checks[key] for key in ('original_arrangement_preserved',
                'source_floor_supported_and_clear', 'within_survey_neighborhood')))
        self.path_m += float(np.linalg.norm(position[:2] - self.last[:2]))
        self.excursion_m = max(self.excursion_m, math.dist(position[:2], self.initial['base']['position'][:2]))
        masks = _gate_masks(position[None, :2], self.baseline, self.config['operation'])
        heading = math.atan2(self.baseline[0][1] - position[1], self.baseline[0][0] - position[0])
        delta = heading - measured['base']['yaw_rad']
        facing = abs(math.atan2(math.sin(delta), math.cos(delta))) <= p['station_heading_tolerance_rad']
        station = next((i for i, mask in enumerate(masks) if bool(mask[0]) and i not in self.completed_stations), None)
        if self.config['operation'] == OPERATIONS[2] and self.completed_stations:
            first = np.asarray(self.station_evidence[0]['base']['position'][:2])
            a, b = first - self.baseline[0], position[:2] - self.baseline[0]
            bearing = math.acos(float(np.clip(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)),
                                                               1e-12), -1., 1.)))
            if bearing < math.radians(45) or np.linalg.norm(position[:2] - first) < .75:
                station = None
        if self.trajectory_valid and checks['stopped'] and facing and station is not None:
            self.station_seconds = self.station_seconds + dt if self.active_station == station else 0.
            self.active_station = station
            if self.station_seconds >= p['station_duration_s']:
                self.completed_stations.append(station)
                self.station_evidence.append({'station': station, 'native_time_s': measured['time_s'],
                    'base': measured['base'], 'consecutive_stationary_seconds': self.station_seconds,
                    'heading_error_rad': abs(math.atan2(math.sin(delta), math.cos(delta)))})
                self.station_seconds, self.active_station = 0., None
        else:
            self.station_seconds, self.active_station = 0., None
        quiet = (self.trajectory_valid and len(self.completed_stations) == 2 and checks['stopped']
                 and checks['at_return_position'] and self.path_m >= p['minimum_path_m']
                 and self.excursion_m >= p['minimum_excursion_m'])
        self.quiet_seconds = self.quiet_seconds + dt if quiet and self.previously_quiet else 0.
        self.previously_quiet = quiet
        self.last, self.last_time = position, measured['time_s']

    def record(self, observation, *, action=None):
        if self.initial is None:
            return None
        measured, p = self._measure(), self.config['parameters']
        checks = self._checks(measured)
        checks.update(continuous_safe_native_trajectory=bool(self.trajectory_valid),
            recent_monotonic_native_sample=-1e-7 <= measured['time_s'] - self.last_time <= p['maximum_native_sample_gap_s'] + 1e-7,
            both_survey_stations_completed=len(self.completed_stations) == 2,
            minimum_path_completed=self.path_m >= p['minimum_path_m'],
            minimum_excursion_completed=self.excursion_m >= p['minimum_excursion_m'],
            stationary_return_completed=self.quiet_seconds >= p['stop_duration_s'])
        value = {'schema': 'private_alignment_survey_v1', 'episode': self.episode,
            'snapshot_id': getattr(observation, 'snapshot_id', None), 'action_step': getattr(observation, 'action_step', None),
            'action': action, 'task_success': all(checks.values()), 'criteria': checks,
            'path_m': self.path_m, 'excursion_m': self.excursion_m,
            'completed_stations': self.completed_stations, 'station_seconds': self.station_seconds,
            'survey_stop_evidence': self.station_evidence,
            'stationary_return_seconds': self.quiet_seconds, 'configuration': self.config,
            'initial_native_state': self.initial, 'native_state': measured}
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as handle:
            handle.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    if config.get('operation') == 'arrange_released_source_shape':
        from .spatial_relations_geometric_configuration import ShapeArrangementEvaluator
        return ShapeArrangementEvaluator(session, controller, config, output)
    if config.get('operation') == ARRANGEMENT_OPERATION:
        return RelationArrangementEvaluator(session, controller, config, output)
    return AlignmentSurveyEvaluator(session, controller, config, output)
