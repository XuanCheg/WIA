"""Preserve Liquid Volume apparatus and add measured container docking routes.

Run CPU conversion from the repository root with the behavior Python environment:
CategoryConverter().convert_all(sources, ConversionContext(batch_root)).
Set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2.
The campaign owns native capture; these hooks never issue robot or water actions.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from wia.resources import ESI_ROOT

from wia.tasks.background_policy import legal_background_contact

from collections import Counter
from dataclasses import replace
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np

from wia.authoring.conversion import ROOT, ConverterBase, original_scene, physical_task
from wia.authoring.io import sha256, write_json


MODULE = 'wia.authoring.converters.physical_structure_liquid_volume'
OPERATIONS = ('dock_at_both_containers', 'dock_at_both_and_return')
CAPACITY_OPERATION = 'report_source_capacity_order'
TASK_REVISION = 'task_edit_20261018_v1'
KEYS = ('left', 'right', 'floor')
FLOOR_NAME = 'floors_ptwlei_0'
MAP_RESOLUTION_M = .01
SCREEN_RADIUS_M = .45
PARAMETERS = {
    'lateral_tolerance_m': .08,
    'front_distance_range_m': [.70, .95],
    'facing_tolerance_deg': 12.,
    'stability_seconds': .5,
    'minimum_stability_samples': 10,
    'maximum_sample_gap_s': .1,
    'maximum_linear_speed_m_s': .02,
    'maximum_angular_speed_rad_s': .1,
    'minimum_visit_separation_m': .20,
    'minimum_total_travel_m': .25,
    'apparatus_drift_tolerance_m': .005,
    'apparatus_rotation_tolerance_rad': .02,
    'floor_contact_height_tolerance_m': .02,
}


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite native measurement.')
    return result


def _label(value):
    return ' '.join(str(value).replace('_', ' ').lower().split())


def _source_data(source):
    raw, payload = source.raw, source.payload
    if (source.big_task, source.small_task) != CategoryConverter.category or (
            raw.get('big_task'), raw.get('small_task')) != CategoryConverter.category:
        raise ValueError('source_category_mismatch')
    if raw.get('runner_task') != 'pour' or not raw.get('scene') or not raw.get('room'):
        raise ValueError('missing_original_pour_scene_contract')
    if payload.get('skip_reason'):
        raise ValueError('original_builder_skip:' + str(payload['skip_reason']))
    entries = {}
    # Match source side_entry precedence; do not assume an objN/side mapping.
    for side in ('left', 'right'):
        entry = next((payload[key] for key in ('obj1', 'obj2')
            if isinstance(payload.get(key), dict) and payload[key].get('side') == side), None)
        if entry is None:
            entry = {'category': payload.get(side + '_category'), 'model': payload.get(side + '_model')}
        if any(not isinstance(entry.get(field), str) or not entry[field] for field in ('category', 'model')):
            raise ValueError('missing_original_' + side + '_asset')
        entries[side] = entry
    options = raw.get('options_json', [])
    options = json.loads(options) if isinstance(options, str) else options
    if not isinstance(options, list) or len(options) != 2 or any(not isinstance(x, str) for x in options):
        raise ValueError('missing_public_container_pair')
    if [_label(x) for x in options] != [_label(entries[x]['category']) for x in ('left', 'right')]:
        raise ValueError('public_container_pair_disagrees_with_source_assets')
    expected_question = f'Which container has a larger volume: the {options[0]} or the {options[1]}?'
    if raw.get('question') != expected_question:
        raise ValueError('unsupported_public_volume_question')
    # The actual loader copies the top-level answer into _ground_truth. Metadata
    # counts are historical private measurements, not physical goal predicates.
    if raw.get('answer') not in ('left', 'right'):
        raise ValueError('missing_original_qa_answer')
    if payload.get('_ground_truth', raw['answer']) not in ('left', 'right'):
        raise ValueError('original_preprocess_ground_truth_invalid')
    return raw, payload, entries, options


def _resource_paths(context):
    data = Path(os.environ.get('OMNIGIBSON_DATA_PATH', '/local_home2/zhouxinyang/data/behavior'))
    root = Path(context.config.get('original_asset_root', data / 'behavior-1k-assets'))
    spec = importlib.util.find_spec('bddl')
    inventory = Path(context.config['original_inventory_path']) if 'original_inventory_path' in context.config else (
        Path(spec.origin).parent / 'generated_data/object_inventory.json' if spec and spec.origin else Path('missing_inventory'))
    return root, inventory


@lru_cache(maxsize=4)
def _inventory(path, fingerprint):
    return json.loads(Path(path).read_text()).get('bounding_box_sizes', {})


@lru_cache(maxsize=4)
def _maps(root, scene, fingerprints):
    """Use the original 1 cm map and native instance-name ordering on CPU."""
    import cv2
    cv2.setNumThreads(2)
    layout = Path(root) / 'scenes' / scene / 'layout'
    trav, rooms, semantics = [cv2.imread(str(layout / name), cv2.IMREAD_GRAYSCALE) for name in (
        'floor_trav_0.png', 'floor_insseg_0.png', 'floor_semseg_0.png')]
    if any(x is None for x in (trav, rooms, semantics)) or not (
            trav.shape == rooms.shape == semantics.shape and rooms.shape[0] == rooms.shape[1]):
        raise ValueError('original_room_maps_unavailable_or_inconsistent')
    categories = (Path(root) / 'metadata/room_categories.txt').read_text().splitlines()
    names, counts = {}, Counter()
    for instance in np.unique(rooms):
        if instance == 0:
            continue
        row, col = np.argwhere(rooms == instance)[0]
        category = categories[int(semantics[row, col]) - 1]
        names[f'{category}_{counts[category]}'] = int(instance)
        counts[category] += 1
    return trav, rooms, names


def _robot_start(raw, entries, context):
    root, inventory = _resource_paths(context)
    paths = [root / 'scenes' / raw['scene'] / 'layout' / name for name in (
        'floor_trav_0.png', 'floor_insseg_0.png', 'floor_semseg_0.png')]
    paths += [root / 'metadata/room_categories.txt', inventory]
    evidence = {'method': 'original_1cm_room_map_and_source_inventory_screen',
        'source_origin_xy_m': [0., 0.], 'source_object_gap_m': .3,
        'robot_screen_radius_m': SCREEN_RADIUS_M, 'native_qualified': False,
        'includes_synthetic_pool_collision_screen': False}
    if not all(p.is_file() for p in paths):
        return None, {**evidence, 'reason': 'original_maps_or_inventory_unavailable'}
    hashes = {str(p): sha256(p) for p in paths}
    sizes = _inventory(str(inventory), hashes[str(inventory)])
    extents = {}
    for side in ('left', 'right'):
        size = _array(sizes.get(entries[side]['model'], []))
        if size.shape != (3,) or min(size) <= 0:
            return None, {**evidence, 'resource_sha256': hashes, 'reason': 'original_inventory_bbox_missing'}
        extents[side] = size * min(.3 / max(size), 1.)
    centres = {'left': np.array([0., 0.]), 'right': np.array([
        .5 * extents['left'][0] + .3 + .5 * extents['right'][0], 0.])}
    trav, rooms, names = _maps(str(root), raw['scene'], tuple(hashes.items()))
    if raw['room'] not in names:
        return None, {**evidence, 'resource_sha256': hashes, 'reason': 'original_room_not_in_map'}
    radius = math.ceil(SCREEN_RADIUS_M / MAP_RESOLUTION_M)
    axis = np.arange(-radius, radius + 1)
    disk = axis[:, None] ** 2 + axis[None, :] ** 2 <= radius ** 2
    rejected, choices = Counter(), []
    middle = (centres['left'] + centres['right']) / 2
    # Start behind the source observer's front side, outside either docking band.
    for distance in (1.30, 1.45, 1.60):
        for offset in (0., -.15, .15, -.30, .30):
            xy = middle + [offset, -distance]
            row, col = (xy[::-1] / MAP_RESOLUTION_M + rooms.shape[0] / 2).astype(int)
            if min(row, col) < radius or max(row, col) + radius >= rooms.shape[0]:
                rejected['map_boundary'] += 1
                continue
            region = (slice(row - radius, row + radius + 1), slice(col - radius, col + radius + 1))
            if not np.all(trav[region][disk] > 0) or not np.all(rooms[region][disk] == names[raw['room']]):
                rejected['map_footprint_or_room'] += 1
                continue
            choices.append((distance, abs(offset), float(xy[0]), xy))
    evidence.update(resource_sha256=hashes, source_scaled_bbox_m={k: v.tolist() for k, v in extents.items()},
        predicted_container_centres_xy_m={k: v.tolist() for k, v in centres.items()},
        room_instance_id=names[raw['room']], screened_count=15, feasible_count=len(choices),
        rejections=dict(rejected), map_floor_index=0, proposed_floor_z_m=0.)
    if not choices:
        return None, {**evidence, 'reason': 'no_source_anchored_map_screened_robot_start'}
    xy = min(choices, key=lambda item: item[:3])[3]
    evidence['chosen_xy_m'] = xy.tolist()
    return {'operation': 'robot_pose', 'position': [float(xy[0]), float(xy[1]), 0.],
        'yaw_deg': math.degrees(math.atan2(middle[1] - xy[1], middle[0] - xy[0])),
        'head_pitch_deg': 45.}, evidence


def capacity_evidence(source, entries, context):
    """Ground pairwise capacity in one source fill trial, never final liquid level."""
    raw, payload = source.raw, source.payload
    counts = {side: entries[side].get('capacity_measured') for side in ('left', 'right')}
    evidence = {'metric': 'source_retained_water_particle_capacity', 'units': 'same_trial_water_particle_count',
        'counts': counts, 'valid': False, 'calibration': {'same_water_system_within_trial': True,
            'volume_in_litres_known': False, 'maximum_asset_bbox_m': .3,
            'scale_formula': 'min(0.3/max(native_asset_bbox_dimensions), 1.0)',
            'cross_trial_comparison_calibrated': False}, 'source_identity': source.relative_path}
    if any(type(value) is not int or value <= 0 for value in counts.values()):
        return {**evidence, 'reason': 'missing_positive_source_fill_capacity_counts'}
    delta = counts['left'] - counts['right']
    if payload.get('capacity_diff_measured') != delta:
        return {**evidence, 'reason': 'source_capacity_count_difference_conflict'}
    relation = 'left = right' if delta == 0 else 'right < left' if delta > 0 else 'left < right'
    expected_larger = 'tie' if delta == 0 else 'left' if delta > 0 else 'right'
    evidence.update(expected_order=relation, capacity_tie=delta == 0)
    if expected_larger != raw['answer'] or payload.get('gt_larger', raw['answer']) != raw['answer']:
        return {**evidence, 'reason': 'source_capacity_answer_conflict_or_unresolved_tie'}
    _, inventory_path = _resource_paths(context)
    if not inventory_path.is_file():
        return {**evidence, 'reason': 'source_asset_scale_calibration_unavailable'}
    inventory = _inventory(str(inventory_path), sha256(inventory_path))
    scales = {}
    for side in counts:
        sizes = inventory.get(entries[side]['model'])
        if not sizes or len(sizes) != 3 or min(sizes) <= 0:
            return {**evidence, 'reason': 'source_asset_scale_calibration_unavailable'}
        scales[side] = min(.3 / max(sizes), 1.)
    source_builder = ESI_ROOT / 'src/dataset_generation/task_capacity/batch_pour.py'
    evidence.update(valid=True, reason=None, expected_native_scales=scales,
        source_protocol={'path': str(source_builder), 'sha256': sha256(source_builder),
            'method': 'Both original containers filled and measured in the same water particle system before the pour.',
            'native_scale_match_required': True})
    return evidence


def revise_capacity_goal(result, source, context, entries, labels):
    """Retire docking credit while preserving the original apparatus and budgets."""
    evidence = capacity_evidence(source, entries, context)
    task = result['task_definition']
    parameters = task['private']['evaluation']['parameters']
    contract = {'preserve': ['container_identity', 'source_asset_scale', 'same_trial_capacity_measurement',
        'apparatus_geometry', 'source_reference_camera'], 'goal': 'report_pairwise_capacity_order',
        'metric': evidence['metric'], 'not_metrics': ['fill_height', 'current_liquid_amount', 'aabb_volume', 'docking'],
        'cross_world_pooling_allowed': False}
    parameters.update(task_revision=TASK_REVISION, mechanism_contract=contract,
        capacity_evidence=evidence, expected_order=evidence.get('expected_order'),
        completion_items=[{'id': 'capacity_order', 'kind': 'semantic_answer'}])
    task['repair'] = {'task_revision': TASK_REVISION, 'mechanism_contract': contract}
    task['public'].update(objective=f'Compare the capacities of the left {labels[0]} and right {labels[1]}. '
        'Report their increasing capacity order using "left < right", "right < left", or "left = right" '
        'for equal capacities. Capacity means how much water each original-size container can hold, '
        'not its current fill amount, height or outer bounding-box volume.',
        success_criteria='Report the capacity ordering for the two original containers. Preserve both '
        'containers and their scales while observing through mounted robot cameras. No docking tour or pouring shortcut earns comparison credit.',
        constraints=['Use mounted robot observations and guarded robot controls.',
            'Leave the original containers, pool geometry and water unchanged.',
            'Do not substitute current fill level or external dimensions for capacity.'])
    result.update(task_revision=TASK_REVISION, mechanism_contract=contract,
        available_operation_recipes=[CAPACITY_OPERATION, *OPERATIONS], private_capacity_evidence=evidence,
        change_summary=['Replace docking-only credit with an explicit pairwise container-capacity report.',
            'Ground comparison in original same-trial fill counts and preserve native asset-scale calibration.'],
        next_step='Review native apparatus/scale preservation and capacity evidence before qualifying the semantic order report.')
    result['private_source_conditions']['source_capacity_measurements_are_not_goal_evidence'] = False
    result['assumptions'][1].update(
        doubt='Source fill counts measure a calibrated same-trial capacity proxy, not a current fill level or an absolute litre volume.',
        chosen_assumption='Use the original same-water particle-count ordering only after source count, answer and native asset-scale checks agree.',
        alternatives=['Retain a pending comparison diagnostic when counts or scale evidence disagree.',
            'Add larger joint inventories only with exact-world identity and complete calibrated pairwise evidence.'],
        scope='New pairwise capacity-report revision.',
        risk='Close capacities, missing source calibration or changed native scales can invalidate an ordering; cross-trial pooling remains prohibited.')
    result['limitations'] = ['Source capacity uses retained water-particle counts from one common fill trial; '
        'absolute litre calibration and cross-trial count comparability are unavailable.',
        'Native apparatus identity, source scale, support and informative mounted views require qualification.',
        'This comparison task does not certify pouring or physical container sorting.']
    if not evidence['valid']:
        result.update(status='needs_human_decision', failure_reason=evidence['reason'],
            diagnostic_stage='capacity_comparison_integrity')
        result['warning_codes'].append('capacity_total_order_pending')


class CapacityOrderEvaluator:
    """Score an explicit order report with unchanged native source apparatus."""
    def __init__(self, session, controller, config, output):
        self.parameters = _native_config(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.bound = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.active = False

    def begin_episode(self, observation):
        self.active, self.answer, self.reports = True, None, []
        self.initial = measure_native_state(self.session, self.bound)
        evidence = self.parameters['capacity_evidence']
        self.initial_scale_calibrated = bool(evidence['valid'] and all(np.allclose(
            _array(self.bound[key].scale), scale, rtol=1e-5, atol=1e-6)
            for key, scale in evidence.get('expected_native_scales', {}).items()))
        self.apparatus_preserved = True
        return self.record(observation, action=None)

    def report_answer(self, answer, observation):
        self.answer = ' '.join(str(answer).lower().strip().split())
        self.reports.append({'answer': self.answer, 'action_step': observation.action_step,
            'snapshot_id': observation.snapshot_id})

    def sample_step(self):
        # No route or movement history is required by a comparison answer.
        return None

    def record(self, observation, *, action=None):
        if not self.active:
            return None
        error, measured, unchanged = None, None, False
        evidence = self.parameters['capacity_evidence']
        try:
            measured = measure_native_state(self.session, self.bound)
            unchanged = all(np.allclose(_array(self.bound[key].scale), scale, rtol=1e-5, atol=1e-6)
                and np.linalg.norm(_array(measured['objects'][key]['position']) -
                    _array(self.initial['objects'][key]['position'])) <= .005
                for key, scale in evidence['expected_native_scales'].items())
            for key in ('left', 'right'):
                first, current = self.initial['objects'][key], measured['objects'][key]
                q0, q1 = _array(first['quaternion_xyzw']), _array(current['quaternion_xyzw'])
                unchanged &= abs(float(np.dot(q0, q1))) >= math.cos(.01)
            self.apparatus_preserved &= bool(unchanged)
        except Exception as exception:
            error = str(exception)
        available = bool(evidence['valid'] and self.initial_scale_calibrated and error is None)
        correct = bool(self.answer is not None and self.answer == self.parameters['expected_order'])
        value = {'schema': 'private_capacity_order_v1', 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'action': action,
            'task_success': bool(available and self.apparatus_preserved and measured['safe'] and correct),
            'constraints_satisfied': bool(self.apparatus_preserved and measured and measured['safe']),
            'evaluation_validity': 'complete' if available else 'unavailable',
            'completion_items': [{'id': 'capacity_order', 'completed': correct,
                'kind': 'semantic_answer', 'evaluation_validity': 'valid' if available else 'unavailable'}],
            'answer_history': list(self.reports), 'measurements': measured, 'measurement_error': error,
            'capacity_calibration': evidence, 'task_revision': TASK_REVISION}
        from wia.tasks.completion import progress
        value.update(progress(value))
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


class CategoryConverter(ConverterBase):
    category = ('Physical Structure', 'Liquid Volume')
    version = '2.0.0_capacity_task_edit_20261018_v1'

    def inspect(self, source):
        try:
            _source_data(source)
        except (ValueError, TypeError, KeyError) as error:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2:' + str(error),
                'warning_codes': ['not_convertible_under_L1_L2', 'source_contract_review_required']}
        return {'eligible': True, 'reason': 'Original publicly named container pair preserves a source-capacity comparison contract.'}

    def convert(self, source, context):
        raw, payload, entries, labels = _source_data(source)
        operation = context.config.get('liquid_volume_operation', CAPACITY_OPERATION)
        if operation not in (*OPERATIONS, CAPACITY_OPERATION):
            raise ValueError('Unknown Liquid Volume operation recipe.')
        bindings = {side: {'name': name, 'category': entries[side]['category'], 'model': entries[side]['model']}
            for side, name in (('left', 'obj1'), ('right', 'obj2'))}
        bindings['floor'] = {'name': FLOOR_NAME}
        pose, screen = _robot_start(raw, entries, context)
        assumptions = [{
            'doubt': 'The invisible source pool planes are not scene objects; the original 2D map cannot establish '
                'their compatibility with an added R1, its controller posture or its full body clearance.',
            'chosen_assumption': 'Keep source collisions, fixed-base flags and container geometry. '
                'Request one common vertical translation onto measured accessible support, including all four '
                'pool planes, current particles and cached free/dip references. Preserve left/right XY geometry. '
                'Add the installed R1 with an independently measured original-room floor start. Native '
                'qualification must read floor support and every robot contact, including non-scene geometry.',
            'alternatives': ['Retain source-only diagnostics if initialization or native contact checks fail.',
                'Investigate a measured, separately reviewed scaffold repair before enabling pouring.'],
            'scope': source.relative_path,
            'risk': 'Infinite pool geometry may prevent initialization or every proposed route; CPU map screening '
                'is not evidence of native collision clearance. The proposed Z=0 start needs native floor confirmation.',
        }, {
            'doubt': 'Retained source capacities and scripted pouring do not prove native liquid transfer; '
                'navigation around the apparatus cannot determine the volume answer by itself.',
            'chosen_assumption': 'Add ordered docking or round-trip navigation as an independent physical task '
                'using original public container labels. Keep the initial volume QA private and auxiliary.',
            'alternatives': ['Use movable-container retrieval only after preserving a supported source initialization.',
                'Require a measured robot-driven particle stream for a separately qualified transfer recipe.'],
            'scope': 'Both operation recipes in this category.',
            'risk': 'Physical route success certifies docking only; it does not certify capacity comparison, '
                'grasping, filling or pouring. Docking reach and view coverage remain unverified.',
        }]
        warnings = ['assumption_review_required', 'native_navigation_unverified',
            'synthetic_pool_robot_collision_unverified', 'liquid_transfer_unverified',
            'robot_initialization_unqualified']
        if payload.get('gt_larger') != raw['answer']:
            warnings.append('source_qa_annotation_conflict')
        limits = ['No native episode, fill measurement or robot-driven liquid transfer was run by this converter.',
            'Both containers remain fixed-base and empty; common vertical support adaptation requires verified pool/scaffold anchors.',
            'Original maps omit synthetic pool planes and do not certify complete swept-body clearance.',
            'Docking distances, 0.5 s holds and the 160-decision budget are explicit new task settings, '
            'not historical experiment parameters or calibrated performance estimates.']
        hierarchy_mismatch = tuple(Path(source.relative_path).parts[2:4]) != (raw['scene'], raw['room'])
        if hierarchy_mismatch:
            warnings.append('source_directory_scene_mismatch')
            limits.append('The source directory says unknown_scene/unknown_room while JSON resolves the native scene. '
                'The current generic ready task/scene loader enforces directory/JSON equality and will reject this '
                'source after approval. The existing diagnostic capture path uses raw scene/room and can attempt '
                'Review images. Shared loader integration is required before native episode execution.')
            assumptions.append({
                'doubt': 'Source identity directories and top-level scene/room disagree; the current diagnostic '
                    'capture loader and generic ready execution loader apply different identity rules.',
                'chosen_assumption': 'Preserve the original source identity and JSON. Publish a pending candidate '
                    'through the existing raw-identity diagnostic capture path, retaining this execution blocker.',
                'alternatives': ['Have the shared API owner reconcile identity resolution against immutable original '
                    'JSON before approving execution.', 'Retain source-only diagnostics if shared integration is unavailable.'],
                'scope': source.relative_path,
                'risk': 'Review capture does not establish generic task loadability; approval alone cannot make '
                    'the current strict hierarchy loader accept this candidate.',
            })
        result = {'status': 'generated', 'reuse_level': 'L2', 'operation_recipe': operation,
            'available_operation_recipes': list(OPERATIONS), 'warning_codes': warnings, 'assumptions': assumptions,
            'limitations': limits, 'private_robot_start_evidence': screen,
            'private_source_conditions': {'legacy_gt_larger': payload.get('gt_larger'),
                'legacy_capacity_diff_measured': payload.get('capacity_diff_measured'),
                'directory_scene_room': list(Path(source.relative_path).parts[2:4]),
                'resolved_scene_room': [raw['scene'], raw['room']],
                'source_capacity_measurements_are_not_goal_evidence': True},
            'attempted_repairs': [], 'change_summary': ['Retain original liquid apparatus, reference camera, '
                'fixed-base physics, empty start and private QA.', 'Add installed R1 and source-front map-screened '
                'robot pose.', 'Add a measured ordered docking or round-trip goal through existing guarded base controls.'],
            'next_step': 'Review original/adapted images; qualify native R1 posture, floor support and all scaffold '
                'contacts, then execute the guarded docking route. Retain any failure as a diagnostic Review item.'}
        if pose is None:
            warnings.append('source_start_native_measurement_required')
        patches = [{'operation': 'add_existing_r1'}, *([pose] if pose is not None else [])]
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': operation,
            'binding_keys': list(KEYS), 'parameters': {**PARAMETERS, 'accessible_support_required': True}}
        returning = operation == OPERATIONS[1]
        objective = (f'Navigate to directly in front of the left {labels[0]}, stop and face it. '
            f'Then navigate to directly in front of the right {labels[1]}, stop and face it.')
        if returning:
            objective += f' Return to directly in front of the left {labels[0]}, stop and face it again.'
        public = {'objective': objective,
            'success_criteria': 'Visit the named containers in order from their original front side: align within '
                '8 cm of each centre in the left-right direction, remain 70 to 95 cm away in the front-back direction, '
                'face within 12 degrees and stay stopped for at least 0.5 seconds at each visit. Move at least '
                '20 cm between visits and travel at least 25 cm overall. Keep floor support and avoid apparatus contacts.',
            'environment': f'The left container is the {labels[0]}; the right container is the {labels[1]}. '
                'The original empty containers and liquid apparatus retain their sizes and relative XY '
                'arrangement on measured accessible supports.',
            'constraints': ['Use real guarded robot base motion and rotation.',
                'Leave the original containers, pool geometry and water unchanged.',
                'Use mounted robot observations to find the approach positions.']}
        task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings,
            patches=patches, capabilities=['base_translation', 'base_rotation'],
            qualification={'on_failure': 'skip', 'initial_visibility': 'none', 'target_movable_collidable': False})
        task['defaults']['max_steps'] = 160
        result.update(scene_definition=original_scene(source, context, bindings=bindings, patches=patches),
            task_definition=task)
        result['interaction_policy'] = {
            'mode': 'inspection', 'group_bindings': ['left', 'right'],
            'manipulation_bindings': ['left', 'right'], 'relocation_policy': 'vertical_only',
            # Equal root height does not imply equal bottom height for distinct
            # containers. Keep one common Z shift but measure each support.
            'placement_groups': [['left'], ['right']],
            'support_bindings': ['left', 'right'],
            'anchor_policy': 'liquid_volume_common_translation',
            'source_camera_policy': 'preserve', 'source_region': raw.get('room'),
            'support_search': 'measured_surfaces', 'allow_fallback_table': True,
            'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10,
            'maximum_stance_distance_m': .85,
            'warning_codes': ['liquid_pool_scaffold_translation_requires_native_verification',
                              'fixed_container_accessible_support_required', 'liquid_transfer_unverified'],
        }
        warnings.extend(result['interaction_policy']['warning_codes'])
        result['change_summary'].append('Require accessible physical support and one common vertical '
            'container/pool/particle/cache translation, with separate measured supports at each '
            'container bottom; preserve the docking goal and source capacities.')
        result['private_scaffold_contract'] = {
            'prim_paths': [f'/World/plane_{index}' for index in range(4)],
            'state_location': 'session.task_state', 'particle_system': 'water',
            'translated_state_fields': ['free_pos1', 'free_pos2', 'dip_pos1', 'dip_pos2', 'box_floor_z'],
            'preserved_state_fields': ['box_half_extent', 'box_center_x', 'box_center_y',
                                       'particle_point_offsets', 'n_left', 'n_right'],
            'native_anchor_verification_required': True,
        }
        if operation == CAPACITY_OPERATION:
            revise_capacity_goal(result, source, context, entries, labels)
        return result

    def convert_all(self, sources, context):
        """Delegate all serialization/failures; include CPU screening inputs in revisions."""
        sources = list(sources)
        root, inventory = _resource_paths(context)
        paths = {inventory, root / 'metadata/room_categories.txt'}
        for source in sources:
            try:
                scene = source.raw.get('scene', '')
                paths.update(root / 'scenes' / scene / 'layout' / name for name in (
                    'floor_trav_0.png', 'floor_insseg_0.png', 'floor_semseg_0.png'))
            except (OSError, ValueError, TypeError):
                pass  # ConverterBase retains a result for unreadable sources.
        fingerprints = {str(p): sha256(p) if p.is_file() else 'unavailable' for p in sorted(paths)}
        effective = replace(context, config={**context.config, 'liquid_volume_screening_sha256': fingerprints})
        manifest = super().convert_all(sources, effective)
        manifest['operation_counts'] = dict(Counter(r.get('operation_recipe', 'diagnostic') for r in manifest['records']))
        manifest['failure_distribution'] = dict(Counter(r.get('failure_reason') for r in manifest['records']
            if r['status'] != 'generated'))
        manifest['native_physical_qualification_claimed'] = False
        write_json(context.output_root / 'batch_manifest.json', manifest)
        return manifest

    def sample(self, manifest, policy=None):
        """Keep any failure and cover geometry extremes and both physical recipes."""
        fallback = super().sample(manifest, policy)
        rows = {r['candidate_id']: r for r in manifest['records']}
        successful = [r for r in manifest['records'] if r['status'] == 'generated' and
            r.get('private_robot_start_evidence', {}).get('source_scaled_bbox_m')]
        priorities = [key for key in fallback if rows[key]['status'] != 'generated'][:1]
        if successful:
            metrics = (
                lambda r: min(min(size[:2]) for size in r['private_robot_start_evidence']['source_scaled_bbox_m'].values()),
                lambda r: -max(size[0] * size[1] for size in r['private_robot_start_evidence']['source_scaled_bbox_m'].values()),
                lambda r: min(size[2] for size in r['private_robot_start_evidence']['source_scaled_bbox_m'].values()),
                lambda r: -max(size[2] for size in r['private_robot_start_evidence']['source_scaled_bbox_m'].values()),
            )
            priorities.extend(min(successful, key=lambda r: (metric(r), r['candidate_id']))['candidate_id']
                for metric in metrics)
        selected = list(dict.fromkeys(priorities + fallback))[:len(fallback)]
        for operation in OPERATIONS:
            if any(rows[key].get('operation_recipe') == operation for key in selected):
                continue
            available = sorted(key for key, r in rows.items() if r.get('operation_recipe') == operation
                and r['status'] == 'generated')
            replaceable = [i for i, key in enumerate(selected) if rows[key]['status'] == 'generated' and
                sum(rows[x].get('operation_recipe') == rows[key].get('operation_recipe') for x in selected) > 1]
            if available and replaceable:
                selected[replaceable[-1]] = available[0]
        return selected


def _native_config(session, config):
    if (config.get('type') != 'category_operation' or module_identity(config.get('module')) != MODULE or
            config.get('operation') not in (*OPERATIONS, CAPACITY_OPERATION) or (config.get('scene'), config.get('room')) != (
                session.scene_name, session.room_name) or set(config.get('bindings', {})) != set(KEYS)):
        raise ValueError('Docking requires its original scene, pair and floor bindings.')
    if config.get('operation') == CAPACITY_OPERATION:
        return config['parameters']
    if config.get('parameters') not in (PARAMETERS, {**PARAMETERS, 'accessible_support_required': True}):
        raise ValueError('Use the declared docking protocol and thresholds.')
    return config['parameters']


def _accessible_support_checks(session, bound):
    """Read current physical support and height; no scripted fluid actions."""
    from wia.tasks.interaction_setup import (
        height_screen, support_contact_paths, touching_task_support, on_top_task_support,
    )
    floor = bound['floor']
    rows = height_screen({bound[key].name: bound[key].aabb for key in ('left', 'right')},
                         float(_array(floor.aabb[1])[2]))
    return {key: bool(rows[bound[key].name]['within_height_window']
        and support_contact_paths(session, bound[key].name)
        and touching_task_support(session, bound[key], floor)
        and on_top_task_support(session, bound[key], floor)) for key in ('left', 'right')}


def _under(body, root):
    return body == root or body.startswith(root + '/')


def measure_native_state(session, bound):
    """Read native poses, AABBs, velocities and all contacts, including USD scaffold."""
    robot, scene = session.robot, session.env.scene
    position, quaternion = (_array(x) for x in robot.get_position_orientation())
    if position.shape != (3,) or quaternion.shape != (4,) or not np.linalg.norm(quaternion):
        raise ValueError('Native R1 pose is unavailable.')
    quaternion = quaternion / np.linalg.norm(quaternion)
    x, y, z, w = quaternion
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    objects = {}
    for key in KEYS:
        obj = bound[key]
        low, high = (_array(v) for v in obj.aabb)
        point, rotation = (_array(v) for v in obj.get_position_orientation())
        if low.shape != (3,) or high.shape != (3,) or np.any(high <= low) or point.shape != (3,) or (
                rotation.shape != (4,) or not np.linalg.norm(rotation)):
            raise ValueError('Native bound object pose or AABB is unavailable.')
        objects[key] = {'name': obj.name, 'position': point.tolist(),
            'quaternion_xyzw': (rotation / np.linalg.norm(rotation)).tolist(),
            'aabb_min': low.tolist(), 'aabb_max': high.tolist(), 'centre': ((low + high) / 2).tolist(),
            'fixed_base': bool(obj.fixed_base), 'visual_only': bool(obj.visual_only)}
    floor = bound['floor']
    floor_z = objects['floor']['aabb_max'][2]
    floor_contacts, nonfloor_contacts = [], []
    for contact in robot.contact_list():
        if legal_background_contact(session, contact.body0, contact.body1):
            continue
        bodies = [str(contact.body0), str(contact.body1)]
        outside = [body for body in bodies if not _under(body, robot.prim_path)]
        if not outside:
            continue
        row = {'body0': bodies[0], 'body1': bodies[1]}
        if all(_under(body, floor.prim_path) for body in outside):
            point, normal = _array(contact.position), _array(contact.normal)
            row.update(position=point.tolist(), normal=normal.tolist())
            floor_contacts.append(row)
        else:
            # Never whitelist the string groundPlane: source plane_N/groundPlane
            # is an apparatus wall, not measured ground support.
            nonfloor_contacts.append(row)
    floor_contact = bool(floor_contacts) and all(
        len(row['position']) == 3 and len(row['normal']) == 3 and
        abs(row['position'][2] - floor_z) <= PARAMETERS['floor_contact_height_tolerance_m'] and
        abs(row['normal'][2]) >= .5 for row in floor_contacts)
    lower, upper = (_array(v) for v in robot.links[robot.base_footprint_link_name].aabb)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(upper[:2] <= lower[:2]):
        raise ValueError('Native chassis footprint is unavailable.')
    segmentation = scene.seg_map
    regions = [segmentation.get_room_instance_by_point(np.array([a, b]))
        for a in (lower[0], (lower[0] + upper[0]) / 2, upper[0])
        for b in (lower[1], (lower[1] + upper[1]) / 2, upper[1])]
    room = segmentation.get_room_instance_by_point(position[:2])
    in_room = room == session.room_name and all(r == session.room_name for r in regions)
    linear_speed = float(np.linalg.norm(_array(robot.get_linear_velocity())))
    angular_speed = float(np.linalg.norm(_array(robot.get_angular_velocity())))
    now = float(session.pipeline.og.sim.current_time)
    if not math.isfinite(now):
        raise ValueError('Native simulation clock is unavailable.')
    return {'time_s': now, 'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
        'yaw_rad': yaw, 'linear_speed_m_s': linear_speed, 'angular_speed_rad_s': angular_speed,
        'objects': objects, 'floor_contact': bool(floor_contact), 'floor_contacts': floor_contacts,
        'nonfloor_contacts': nonfloor_contacts, 'chassis_room': room, 'footprint_regions': regions,
        'chassis_in_original_room': bool(in_room),
        'safe': bool(floor_contact and not nonfloor_contacts and in_room)}


def docking_checks(measured, side):
    """Exact measured docking predicate; source capacities and answers are unused."""
    position = np.asarray(measured['position'])
    centre = np.asarray(measured['objects'][side]['centre'])
    difference = centre - position
    facing = math.atan2(difference[1], difference[0])
    error = abs(math.atan2(math.sin(facing - measured['yaw_rad']), math.cos(facing - measured['yaw_rad'])))
    low, high = PARAMETERS['front_distance_range_m']
    return {'safe_native_state': measured['safe'],
        'aligned_with_container': bool(abs(difference[0]) <= PARAMETERS['lateral_tolerance_m']),
        'at_original_front_distance': bool(low <= difference[1] <= high),
        'facing_container': bool(error <= math.radians(PARAMETERS['facing_tolerance_deg'])),
        'stopped': bool(measured['linear_speed_m_s'] <= PARAMETERS['maximum_linear_speed_m_s'] and
            measured['angular_speed_rad_s'] <= PARAMETERS['maximum_angular_speed_rad_s'])}


def qualify_operation(session, executor, config, bound):
    """Read native preconditions only; do not move R1, containers or particles."""
    from wia.authoring.converters.liquid_capacity_inventory import OPERATION
    if config.get('operation') == OPERATION:
        return qualify_joint_inventory(session, executor, config, bound)
    checks = {}
    try:
        parameters = _native_config(session, config)
        if set(bound) != set(KEYS) or {key: obj.name for key, obj in bound.items()} != config['bindings']:
            raise ValueError('Qualification bindings differ from native task bindings.')
        measured = measure_native_state(session, bound)
        left, right = (measured['objects'][side] for side in ('left', 'right'))
        checks = {'base_motion_capabilities': bool(executor.capabilities.base_translation and executor.capabilities.base_rotation),
            'native_floor_contact': measured['floor_contact'],
            'no_nonfloor_robot_contacts': not measured['nonfloor_contacts'],
            'chassis_in_original_room': measured['chassis_in_original_room'],
            'fixed_collidable_apparatus': all(row['fixed_base'] and not row['visual_only'] for row in measured['objects'].values()),
            'distinct_ordered_container_stations': right['centre'][0] - left['centre'][0] >= .30 and
                abs(right['centre'][1] - left['centre'][1]) <= .01,
            'robot_initially_stopped': measured['linear_speed_m_s'] <= PARAMETERS['maximum_linear_speed_m_s'] and
                measured['angular_speed_rad_s'] <= PARAMETERS['maximum_angular_speed_rad_s']}
        if parameters.get('accessible_support_required'):
            checks.update({key + '_current_accessible_support': value
                           for key, value in _accessible_support_checks(session, bound).items()})
        accepted = all(checks.values())
        return {'accepted': bool(accepted), 'checks': checks, 'native_state': measured,
            'reason': None if accepted else 'native_docking_initialization_or_scaffold_contact_failed',
            'route_executed': False, 'liquid_transfer_qualified': False}
    except Exception as error:
        return {'accepted': False, 'checks': checks, 'reason': 'native_docking_measurement_unavailable',
            'exception_summary': f'{type(error).__name__}: {error}', 'route_executed': False}



def _completion_result(items):
    """Report declared physical obligations without distance-shaped rewards."""
    completed = sum(bool(item['completed']) for item in items)
    total = len(items)
    score = completed / total
    return {'scoring_version': 'task_completion_v2', 'evaluation_validity': 'complete',
            'completion_items': items, 'completed_goals': completed, 'total_goals': total,
            'completion_score': score, 'score_percent': 100. * score}


class ContainerDockingEvaluator:
    """Track measured ordered stops and continuous motion without native writes."""
    def __init__(self, session, controller, config, output):
        _native_config(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.bound = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        if any(obj is None for obj in self.bound.values()):
            raise ValueError('Original docking apparatus binding is missing.')
        self.required = ['left', 'right'] + (['left'] if config['operation'] == OPERATIONS[1] else [])
        self.maximum_speed = float(controller.config.max_translation_speed_m_s)
        self.translation_tolerance = float(controller.config.translation_tolerance_m)
        if not math.isfinite(self.maximum_speed) or self.maximum_speed <= 0 or not (
                math.isfinite(self.translation_tolerance) and self.translation_tolerance >= 0):
            raise ValueError('Guarded controller movement bounds are unavailable.')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.initial, self.last = 0, None, None

    def begin_episode(self, observation):
        self.episode += 1
        self.initial = measure_native_state(self.session, self.bound)
        self.last = self.initial
        self.stage, self.stable_samples, self.hold_start = 0, 0, None
        self.travel_m, self.visits, self.invalid_reasons = 0., [], set()
        self.record(observation, action=None)

    def _sample(self, count_step):
        measured = measure_native_state(self.session, self.bound)
        elapsed = measured['time_s'] - self.last['time_s']
        displacement = float(np.linalg.norm(np.asarray(measured['position']) - self.last['position']))
        if elapsed < 0 or elapsed > PARAMETERS['maximum_sample_gap_s']:
            self.invalid_reasons.add('native_clock_gap_or_reset')
        elif elapsed == 0 and displacement > 1e-6:
            self.invalid_reasons.add('pose_change_without_native_time')
        elif elapsed > 0:
            if displacement > self.maximum_speed * elapsed + self.translation_tolerance:
                self.invalid_reasons.add('motion_exceeds_guarded_control_bound')
            self.travel_m += displacement
        if not measured['safe']:
            self.invalid_reasons.add('unsafe_native_contact_or_floor_state')
        if self.config['parameters'].get('accessible_support_required') and not all(
                _accessible_support_checks(self.session, self.bound).values()):
            self.invalid_reasons.add('accessible_container_support_lost')
        for key in KEYS:
            current, initial = measured['objects'][key], self.initial['objects'][key]
            moved = np.linalg.norm(np.asarray(current['position']) - initial['position'])
            dot = abs(float(np.dot(current['quaternion_xyzw'], initial['quaternion_xyzw'])))
            turned = 2 * math.acos(min(1., dot))
            if moved > PARAMETERS['apparatus_drift_tolerance_m'] or turned > PARAMETERS['apparatus_rotation_tolerance_rad']:
                self.invalid_reasons.add('original_apparatus_moved')
            if not current['fixed_base'] or current['visual_only']:
                self.invalid_reasons.add('original_apparatus_physics_changed')
        if count_step and elapsed > 0 and self.stage < len(self.required):
            side = self.required[self.stage]
            valid = all(docking_checks(measured, side).values()) and not self.invalid_reasons
            if self.visits:
                valid = valid and np.linalg.norm(np.asarray(measured['position'][:2]) -
                    self.visits[-1]['position'][:2]) >= PARAMETERS['minimum_visit_separation_m']
            if valid:
                self.hold_start = measured['time_s'] if self.hold_start is None else self.hold_start
                self.stable_samples += 1
                if measured['time_s'] - self.hold_start >= PARAMETERS['stability_seconds'] and (
                        self.stable_samples >= PARAMETERS['minimum_stability_samples']):
                    self.visits.append({'side': side, 'time_s': measured['time_s'], 'position': measured['position'],
                        'stable_samples': self.stable_samples, 'hold_seconds': measured['time_s'] - self.hold_start})
                    self.stage += 1
                    self.stable_samples, self.hold_start = 0, None
            else:
                self.stable_samples, self.hold_start = 0, None
        self.last = measured
        return measured

    def sample_step(self):
        if self.initial is not None:
            self._sample(True)

    def record(self, observation, *, action):
        if self.initial is None:
            return None
        measured = self._sample(False)
        terminal = docking_checks(measured, self.required[-1])
        criteria = {'ordered_visits_complete': self.stage == len(self.required),
            'minimum_real_travel': self.travel_m >= PARAMETERS['minimum_total_travel_m'],
            'uninterrupted_safe_measured_history': not self.invalid_reasons,
            'final_docking_state': all(terminal.values())}
        record = {'schema': 'private_liquid_volume_docking_v1', 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step, 'action': action,
            'task_success': bool(all(criteria.values())), 'criteria': criteria, 'configuration': self.config,
            'required_visits': self.required, 'visits': self.visits, 'travel_m': self.travel_m,
            'invalid_reasons': sorted(self.invalid_reasons), 'terminal_docking_checks': terminal,
            'native_state': measured, 'liquid_transfer_success_claimed': False}
        items = []
        for index, side in enumerate(self.required):
            terminal_item = index == len(self.required) - 1
            completed = self.stage > index and not self.invalid_reasons
            if terminal_item:
                completed = completed and bool(all(criteria.values()))
            items.append({'id': f'dock_{index + 1}_{side}', 'completed': bool(completed),
                'evaluation_validity': 'valid',
                'kind': 'terminal_condition' if terminal_item else 'historical_checkpoint',
                'evidence': {'visit': self.visits[index] if index < len(self.visits) else None,
                    'safety_faults': sorted(self.invalid_reasons),
                    'terminal_checks': criteria if terminal_item else None}})
        record.update(_completion_result(items))
        write_json(self.output / 'state.json', record)
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        return record


def create_evaluator(session, controller, config, output):
    from wia.authoring.converters.liquid_capacity_inventory import OPERATION
    if config.get('operation') == OPERATION:
        return JointCapacityRankingEvaluator(session, controller, config, output)
    if config.get('operation') == CAPACITY_OPERATION:
        return CapacityOrderEvaluator(session, controller, config, output)
    return ContainerDockingEvaluator(session, controller, config, output)


def configure_scene(config, parameters):
    """Construct the declared co-loaded inventory and restore native furniture."""
    from wia.authoring.converters.liquid_capacity_inventory import INITIALIZATION
    if parameters.get('operation') != INITIALIZATION:
        raise ValueError('Unsupported joint capacity initialization.')
    receipt = parameters['inventory_receipt']
    if receipt['scene_model'] != 'Rs_int' or receipt['room'] != 'living_room_0':
        raise ValueError('Joint capacity inventory belongs to another source world.')
    scene = config['scene']
    if scene.get('scene_model') != receipt['scene_model']:
        raise ValueError('Joint capacity initialization cannot switch source buildings.')
    scene.update(load_room_instances=[receipt['room']], not_load_object_categories=[])
    objects = {obj['name']: obj for obj in config.get('objects', [])}
    for index, item in enumerate(receipt['inventory']):
        current = objects.get(item['name'])
        if current is not None and any(current.get(key) != item[key] for key in ('category', 'model')):
            raise ValueError('Joint capacity object name conflicts with a source asset.')
        if current is None:
            current = {'type': 'DatasetObject', 'name': item['name'], 'category': item['category'],
                'model': item['model'], 'kinematic_only': False, 'fixed_base': True,
                'position': [.4 * (index % 3), .4 * (index // 3), .4]}
            config.setdefault('objects', []).append(current)
        current.update(scale=item['scale'], fixed_base=True, visual_only=False)
    return {'schema': INITIALIZATION, 'inventory_id': receipt['inventory_id'],
        'co_loaded_object_names': [item['name'] for item in receipt['inventory']],
        'native_furniture_and_floors_requested': True, 'native_support_pending': True,
        'agent_operation_count': 0}


def initialize_scene(session, parameters):
    """Retire pouring-only pools before comparison support placement, once."""
    from wia.authoring.converters.liquid_capacity_inventory import INITIALIZATION
    from omnigibson.utils.usd_utils import delete_or_deactivate_prim
    if parameters.get('operation') != INITIALIZATION or hasattr(session, 'capacity_inventory_initialization'):
        raise ValueError('Joint inventory initialization is versioned and runs once.')
    receipt = parameters['inventory_receipt']
    if (session.scene_name, session.room_name) != (receipt['scene_model'], receipt['room']):
        raise ValueError('Joint capacity scene/room mismatch.')
    scene, sim = session.env.scene, session.pipeline.og.sim
    for item in receipt['inventory']:
        obj = scene.object_registry('name', item['name'])
        if obj is None or any(getattr(obj, key) != item[key] for key in ('category', 'model')) or not np.allclose(
                _array(obj.scale), item['scale'], atol=1e-6, rtol=1e-5):
            raise ValueError('Every source-sized container must be co-loaded.')
    water = scene.get_system('water')
    water.remove_all_particles()
    removed = []
    for index in range(4):
        path = f'/World/plane_{index}'
        if sim.stage.GetPrimAtPath(path).IsValid():
            delete_or_deactivate_prim(path)
            removed.append(path)
    # Reset source scripted-pour caches; none of these setters are Agent actions.
    session.task_state['n_left'], session.task_state['n_right'] = 0, 0
    evidence = {'schema': INITIALIZATION, 'inventory_id': receipt['inventory_id'],
        'all_inventory_co_loaded': True, 'pool_walls_removed': removed,
        'comparison_only': True, 'agent_operation_count': 0, 'labels': {},
        'native_support_and_label_visibility_pending': True}
    session.capacity_inventory_initialization = evidence
    session.capacity_inventory_receipt = receipt
    return evidence


def on_interaction_initialized(session, evidence):
    """Attach visible stable letter labels after the one-table support succeeds."""
    if not evidence.get('accepted') or not hasattr(session, 'capacity_inventory_receipt'):
        return
    from wia.authoring.converters.liquid_capacity_inventory import attach_native_labels
    session.capacity_inventory_initialization['labels'] = attach_native_labels(
        session, session.capacity_inventory_receipt)


def measure_joint_inventory(session, config, bound):
    """Read capacity apparatus and one shared support without touching particles."""
    from wia.authoring.converters.spatial_relations_physical_contact import _snapshot, _robot_state
    from wia.tasks.interaction_setup import support_contact_paths, touching_task_support, on_top_task_support
    inventory = config['parameters']['inventory_receipt']['inventory']
    rows = {item['id']: {**_snapshot(bound[item['binding']]),
        'scale': _array(bound[item['binding']].scale).tolist()} for item in inventory}
    paths = [tuple(support_contact_paths(session, bound[item['binding']].name)) for item in inventory]
    robot = _robot_state(session, bound['floor'])
    pairs = []
    for index, first in enumerate(inventory):
        for second in inventory[index + 1:]:
            a, b = rows[first['id']], rows[second['id']]
            widths = np.minimum(a['aabb_max'][:2], b['aabb_max'][:2]) - np.maximum(a['aabb_min'][:2], b['aabb_min'][:2])
            pairs.append(float(np.linalg.norm(np.maximum(-widths, 0.))))
    checks = {'one_shared_table': bool(paths and paths[0] and all(path == paths[0] for path in paths)),
        'all_supported': all(touching_task_support(session, bound[item['binding']], bound['floor']) and
            on_top_task_support(session, bound[item['binding']], bound['floor']) for item in inventory),
        'nonoverlap': bool(pairs and min(pairs) >= .005),
        'source_assets_and_scales': all(bound[item['binding']].name == item['name'] and
            all(getattr(bound[item['binding']], key) == item[key] for key in ('category', 'model')) and
            np.allclose(rows[item['id']]['scale'], item['scale'], rtol=1e-5, atol=1e-6) for item in inventory),
        'fixed_collidable_containers': all(row['fixed_base'] and not row['visual_only'] for row in rows.values()),
        'robot_supported_upright': robot['supported_upright'], 'robot_in_source_room': robot['in_source_room']}
    return {'objects': rows, 'checks': checks, 'robot': robot, 'shared_support_paths': paths,
        'horizontal_clearances_m': pairs}


def qualify_joint_inventory(session, executor, config, bound):
    """Require the complete native inventory, accessible support and visible labels."""
    from wia.authoring.converters.liquid_capacity_inventory import OPERATION
    checks = {}
    try:
        if (config.get('operation') != OPERATION or module_identity(config.get('module')) != MODULE or
                (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name)):
            raise ValueError('Joint capacity configuration mismatch.')
        measured = measure_joint_inventory(session, config, bound)
        receipt = config['parameters']['inventory_receipt']
        initialization = getattr(session, 'capacity_inventory_initialization', {})
        labels = initialization.get('labels', {})
        checks = {**measured['checks'], 'co_loaded_shared_reset': initialization.get('all_inventory_co_loaded') is True and
            initialization.get('inventory_id') == receipt['inventory_id'],
            'labels_attached_to_bound_identities': all(labels.get(item['id'], {}).get('object_name') == item['name']
                for item in receipt['inventory']),
            'public_label_visibility_verified': initialization.get('public_label_visibility_verified') is True}
        return {'accepted': bool(all(checks.values())), 'checks': checks, 'native_state': measured,
            'reason': None if all(checks.values()) else 'joint_capacity_native_initialization_pending',
            'calibration_runs_are_agent_operations': False}
    except Exception as error:
        return {'accepted': False, 'checks': checks, 'reason': 'joint_capacity_native_measurement_unavailable',
            'exception_summary': f'{type(error).__name__}: {error}'}


class JointCapacityRankingEvaluator:
    """Compare every declared pair; never replace missing truth with a smaller task."""
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.receipt = config['parameters']['inventory_receipt']
        self.bound = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.active = False

    def begin_episode(self, observation):
        self.active, self.answer, self.reports, self.preserved = True, None, [], True
        self.qualification = qualify_joint_inventory(self.session, self.controller, self.config, self.bound)
        self.initial = self.qualification.get('native_state')
        return self.record(observation, action=None)

    def report_answer(self, answer, observation):
        self.answer = answer
        self.reports.append({'answer': answer, 'action_step': observation.action_step,
            'snapshot_id': observation.snapshot_id})

    def sample_step(self):
        if self.active:
            self._measure_preservation()

    def _measure_preservation(self):
        measured = measure_joint_inventory(self.session, self.config, self.bound)
        if self.initial is None:
            raise ValueError('Initial native inventory is unavailable.')
        for key, row in measured['objects'].items():
            initial = self.initial['objects'][key]
            self.preserved &= bool(np.linalg.norm(_array(row['position']) - initial['position']) <= .005 and
                abs(float(np.dot(row['quaternion_xyzw'], initial['quaternion_xyzw']))) >= math.cos(.01))
        return measured

    def record(self, observation, *, action=None):
        if not self.active:
            return None
        from wia.authoring.converters.composition_manipulation import capacity_ranking_score
        from wia.authoring.converters.liquid_capacity_inventory import validate_truth, REVISION
        from wia.tasks.completion import progress
        error, measured = None, None
        try:
            measured = self._measure_preservation()
        except Exception as exception:
            error = f'{type(exception).__name__}: {exception}'
        truth = validate_truth(self.receipt)
        score = capacity_ranking_score([item['id'] for item in self.receipt['inventory']],
            truth['valid_comparisons'] if truth['accepted'] else [], self.answer)
        available = bool(error is None and self.qualification['accepted'] and truth['accepted'])
        items = [{'id': f"capacity:{row['first']}:{row['second']}", 'kind': 'semantic_answer',
            'completed': row['correct'], 'evaluation_validity': 'valid' if available else 'unavailable'}
            for row in score['pair_results']]
        constraints = bool(self.preserved and measured and all(measured['checks'].values()))
        value = {'schema': REVISION, 'task_revision': REVISION, 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'action': action,
            'evaluation_validity': 'complete' if available else 'unavailable',
            'task_success': bool(available and constraints and score['exact_ranking_success']),
            'constraints_satisfied': constraints, 'completion_items': items,
            'comparison_accuracy_percent': score['comparison_accuracy_percent'] if available else None,
            'correct_pairwise_relations': score['correct_pairwise_relations'] if available else None,
            'total_pairs': score['total_pairs'], 'exact_ranking_success': bool(available and score['exact_ranking_success']),
            'pair_results': score['pair_results'], 'truth_validation': truth, 'answer_history': self.reports,
            'initial_qualification': self.qualification, 'native_state': measured, 'measurement_error': error}
        value.update(progress(value))
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value
