"""Preserve chair-defined spatial groups and add measured R1 access patrols.

Run CPU generation from the repository root with the behavior Python environment:
CategoryConverter().convert_all(sources, ConversionContext(output_root)).
Native hooks run through the existing task runner and never move scene objects.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from wia.authoring.conversion import ConverterBase, map_records, original_scene, physical_task
from wia.tasks.background_policy import legal_background_contact
from wia.authoring.io import write_json

MODULE = 'wia.authoring.converters.enumerative_perception_spatial_segmentation'
OPERATIONS = ('segmented_access_patrol_return', 'segmented_two_approach_patrol_return')
PARAMETERS = {
    'minimum_standoff_m': 0.55, 'maximum_standoff_m': 1.25,
    'maximum_heading_error_deg': 30.0, 'chassis_clearance_m': 0.05,
    'different_group_separation_m': 0.35, 'second_approach_separation_m': 0.50,
    'minimum_departure_m': 0.50, 'minimum_travel_m': 1.0, 'return_radius_m': 0.30,
    'stop_duration_s': 0.25, 'maximum_stop_speed_m_s': 0.05,
    'maximum_stop_angular_speed_rad_s': 0.10,
    'position_preservation_m': 0.02, 'orientation_preservation_deg': 5.0,
}


def global_region_labels(layout, *, band_tolerance_m=.10, minimum_separation_m=.10):
    """Name immutable chair boundaries using global X bands and within-band Y.

    Source targets and answers never enter the ordering. Near-equal X centres
    are one visual band; its bottom/top chairs are resolved by unique Y order.
    """
    anchors = sorted({pair['anchor'] for pair in layout['pairs']})
    centres = {key: np.mean(np.asarray(layout['expected'][key]['bbox'], dtype=float), axis=0)[:2]
               for key in anchors}
    ordered = sorted(anchors, key=lambda key: (float(centres[key][0]), float(centres[key][1]), key))
    bands = []
    for key in ordered:
        if not bands or centres[key][0] - centres[bands[-1][0]][0] > band_tolerance_m:
            bands.append([key])
        else:
            bands[-1].append(key)
    gaps = [float(min(centres[key][0] for key in right) - max(centres[key][0] for key in left))
            for left, right in zip(bands, bands[1:])]
    if any(gap < minimum_separation_m for gap in gaps):
        raise ValueError('Global chair X bands lack uniquely separated visible boundaries.')
    labels, within_gaps = {}, {}
    for index, band in enumerate(bands):
        if len(bands) == 1:
            band_label = 'the chair group'
        elif index == 0:
            band_label = 'the leftmost chair group'
        elif index == len(bands) - 1:
            band_label = 'the rightmost chair group'
        else:
            band_label = f'the chair group numbered {index + 1} from the left'
        ordered_y = sorted(band, key=lambda key: (float(centres[key][1]), key))
        differences = [float(centres[b][1] - centres[a][1]) for a, b in zip(ordered_y, ordered_y[1:])]
        if any(gap < minimum_separation_m for gap in differences):
            raise ValueError('Chairs within one global X band lack unique Y separation.')
        within_gaps[str(index)] = differences
        for y_index, key in enumerate(ordered_y):
            if len(band) == 1:
                chair = ('the leftmost chair' if index == 0 else 'the rightmost chair'
                         if index == len(bands) - 1 else f'the chair numbered {index + 1} from the left')
            else:
                position = ('bottom' if y_index == 0 else 'top' if y_index == len(band) - 1
                            else f'number {y_index + 1} from the bottom')
                chair = f'the {position} chair in {band_label}'
            labels[key] = 'the region beneath ' + chair
    if len(set(labels.values())) != len(anchors):
        raise ValueError('Global chair boundary descriptions are not unique.')
    return labels, {'frame': 'initial_global_world_xy', 'image_right_world_axis': '+X',
        'image_up_world_axis': '+Y', 'band_tolerance_m': band_tolerance_m,
        'minimum_separation_m': minimum_separation_m, 'x_bands': bands,
        'source_anchor_centres_xy_m': {key: point.tolist() for key, point in centres.items()},
        'between_band_separations_x_m': gaps, 'within_band_separations_y_m': within_gaps,
        'ordering_uses_targets_or_answers': False}


def validate_region_labels(session, config, bound=None, *, stage='evaluation'):
    """Read current structural labels, separately from episode visual proof.

    Initialization preserves scene, anchor poses/bounds and unmoved target-region
    checks. It leaves semantic visual distinction pending and never certifies
    visibility. Default evaluation retains all original current-episode gates.
    """
    if stage not in ('initialization', 'evaluation'):
        raise ValueError('Region label validation stage must be initialization or evaluation.')
    visual_gates = ('current_episode_visual_receipt',
        'public_initial_global_labels_visually_distinguishable')
    contract = config.get('parameters', {}).get('region_label_contract')
    if not contract:
        return {'applicable': False, 'accepted': True}
    checks, native = {}, {}
    try:
        scene = session.env.scene
        checks['scene_matches'] = session.scene_name == contract['scene']
        expected = contract['source_expected']
        current_expected = {}
        for key, identity in contract['anchor_bindings'].items():
            obj = scene.object_registry('name', identity['name'])
            checks['anchor_identity:' + key] = bool(obj is not None and obj.category == identity['category'])
            if obj is None:
                raise ValueError('Source boundary chair is missing: ' + key)
            low, high = _box(obj.aabb)
            native[key] = {'name': obj.name, 'bbox': [low.tolist(), high.tolist()]}
            current_expected[key] = {'bbox': native[key]['bbox']}
            checks['source_boundary_preserved:' + key] = bool(np.max(np.abs(
                np.asarray(native[key]['bbox']) - expected[key]['bbox'])) <= contract['preservation_tolerance_m'])
            checks['runtime_binding:' + key] = config.get('bindings', {}).get(key, obj.name) == obj.name
            source_pose = contract.get('source_anchor_states', {}).get(identity['name'])
            position, orientation = obj.get_position_orientation()
            checks['source_anchor_pose:' + key] = bool(source_pose is not None and
                np.linalg.norm(_vector(position, 3) - source_pose['pos']) <= contract['preservation_tolerance_m']
                and _angle(orientation, source_pose['ori']) <= PARAMETERS['orientation_preservation_deg'])
        layout = {'pairs': contract['source_pairs'], 'expected': current_expected}
        labels, geometry = global_region_labels(layout, band_tolerance_m=contract['axis_evidence']['band_tolerance_m'],
            minimum_separation_m=contract['axis_evidence']['minimum_separation_m'])
        checks['current_native_label_mapping_matches'] = labels == contract['anchor_public_labels']
        for pair in contract['source_pairs']:
            key = pair['target']
            obj = scene.object_registry('name', contract['target_bindings'][key]['name'])
            if obj is None:
                raise ValueError('Original region target is missing: ' + key)
            low, high = _box(obj.aabb)
            checks['source_region_membership_preserved:' + key] = bool(np.max(np.abs(
                np.asarray([low, high]) - expected[key]['bbox'])) <= contract['preservation_tolerance_m'])
        if stage == 'initialization':
            accepted = all(checks.values())
            return {'applicable': True, 'accepted': accepted, 'checks': checks,
                'native_anchor_geometry': native, 'native_axis_evidence': geometry,
                'qualification_complete': False, 'pending_qualification': list(visual_gates),
                'evaluation_validity': 'unavailable',
                'reason': 'native_region_visual_qualification_pending' if accepted else
                    'native_region_structural_checks_failed'}
        receipt = getattr(session, 'region_label_qualification', {}).get(contract['source_identity'], {})
        checks['current_episode_visual_receipt'] = (getattr(session, 'episode_id', None) is not None
            and receipt.get('episode_id') == session.episode_id
            and receipt.get('frame') == 'initial_global_world_xy')
        checks['public_initial_global_labels_visually_distinguishable'] = receipt.get(
            'anchor_public_label_associations_verified') is True
        accepted = all(checks.values())
        return {'applicable': True, 'accepted': accepted, 'checks': checks,
            'native_anchor_geometry': native, 'native_axis_evidence': geometry,
            'qualification_complete': accepted,
            'pending_qualification': [name for name in visual_gates if not checks[name]],
            'evaluation_validity': 'complete' if accepted else 'unavailable',
            'reason': None if accepted else 'native_boundary_or_visual_label_qualification_pending'}
    except Exception as error:
        return {'applicable': True, 'accepted': False, 'checks': checks,
                'native_anchor_geometry': native, 'evaluation_validity': 'unavailable',
                'qualification_complete': False,
                'pending_qualification': [name for name in visual_gates if not checks.get(name, False)],
                'reason': f'{type(error).__name__}: {error}'}


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


def _vector(value, length):
    value = _array(value)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError('Missing or nonfinite source/native vector.')
    return value


def _box(value):
    low, high = (_vector(part, 3) for part in value)
    if np.any(high <= low):
        raise ValueError('Source/native bounds must have positive extents.')
    return low, high


def _angle(first, second):
    first, second = _vector(first, 4), _vector(second, 4)
    norms = np.linalg.norm(first) * np.linalg.norm(second)
    if norms <= 0:
        raise ValueError('Source/native quaternion must be nonzero.')
    return math.degrees(2.0 * math.acos(float(np.clip(abs(first @ second) / norms, 0.0, 1.0))))


def _layout(source):
    raw, payload = source.raw, source.payload
    question = payload['question_data']
    render = question['render']
    resolved = render['resolved_objects']
    targets = resolved['targets']
    entries = question['case_metadata']['source_entries']
    if (question.get('case') != 'observation_divided' or not targets
            or len(targets) != len(entries) or resolved.get('containers') or resolved.get('confusers')):
        raise ValueError('Unresolved original chair-defined segmentation mechanism.')
    match = re.search(r'objects of category "([^"\n]+)"', raw.get('question', ''))
    if match is None or match.group(1) != question.get('count_target'):
        raise ValueError('Public target category and original construction disagree.')
    if (question.get('question'), str(question.get('answer'))) != (raw.get('question'), str(raw.get('answer'))):
        raise ValueError('Original public/private QA fields disagree.')
    cameras = map_records(render['camera_poses'])
    camera = cameras.get(Path(render['image']).name) or next(iter(cameras.values()))
    _vector(camera['position'], 3)
    _angle(camera['quaternion_xyzw'], camera['quaternion_xyzw'])
    bounds = _vector(render['room_bbox']['expanded_bbox_world_xy'], 4)
    if np.any(bounds[2:] <= bounds[:2]) or not payload.get('floor_name'):
        raise ValueError('Original room bounds or floor binding are missing.')
    bindings, expected, pairs, names = {}, {}, [], set()
    floor_levels = []
    for index, target in enumerate(targets):
        if not all(target.get(field) for field in ('name', 'category', 'model')) or target['name'] in names:
            raise ValueError('Missing or duplicated original target identity.')
        names.add(target['name'])
        if target['category'] != match.group(1):
            raise ValueError('A resolved target is outside the publicly requested category.')
        position = _vector(target['position'], 3)
        requested = _vector(target['requested_position'], 3)
        if np.linalg.norm(position - requested) > 1e-4:
            raise ValueError('Resolved target does not preserve the requested source position.')
        _box(target['bbox'])
        _angle(target['quaternion_xyzw'], target['quaternion_xyzw'])
        matches = [entry for entry in entries if any(np.linalg.norm(_vector(point, 3) - requested) < 1e-4
                   for point in entry.get('ball_positions', []))]
        if len(matches) != 1:
            raise ValueError('Target-to-chair source provenance is ambiguous.')
        entry = matches[0]
        anchor = entry['metadata']['anchor_object']
        visibility = entry['metadata'].get('visibility', {})
        low, high = _box(anchor['bbox'])
        if (not anchor.get('name') or not anchor.get('category') or 'chair' not in anchor['category']
                or visibility.get('support') != 'placed_on_floor_below_chair'
                or np.linalg.norm(requested[:2] - (low[:2] + high[:2]) / 2.0) > 1e-4):
            raise ValueError('Original under-chair region relation is not reproduced by source geometry.')
        anchor_key, target_key = f'anchor_{index}', f'target_{index}'
        if anchor['name'] in names:
            raise ValueError('Original segmentation regions must have distinct anchors.')
        names.add(anchor['name'])
        bindings[anchor_key] = {'name': anchor['name'], 'category': anchor['category']}
        bindings[target_key] = {field: target[field] for field in ('name', 'category', 'model')}
        expected[anchor_key] = {'bbox': anchor['bbox']}
        expected[target_key] = {field: target[field] for field in ('position', 'quaternion_xyzw', 'bbox')}
        pairs.append({'target': target_key, 'anchor': anchor_key})
        # Verified against _generate_observation_divided: floor top + BALL_RADIUS.
        floor_levels.append(float(requested[2] - 0.035))
    if max(floor_levels) - min(floor_levels) > 1e-4:
        raise ValueError('Original targets do not share the recorded floor height.')
    bindings['floor'] = {'name': payload['floor_name'], 'category': 'floors'}
    return {'raw': raw, 'camera': camera, 'bounds': bounds, 'floor_z': floor_levels[0],
            'bindings': bindings, 'expected': expected, 'pairs': pairs,
            'target_category': match.group(1)}


def _start(layout):
    """Source-only stance screening; actual robot clearance is qualified natively."""
    low, high = layout['bounds'][:2] + 0.55, layout['bounds'][2:] - 0.55
    if np.any(high < low):
        return None
    camera_xy = _vector(layout['camera']['position'], 3)[:2]
    candidates = [np.clip(camera_xy, low, high)]
    candidates += [np.array([x, y]) for x in np.arange(low[0], high[0] + 1e-8, 0.20)
                   for y in np.arange(low[1], high[1] + 1e-8, 0.20)]
    boxes = [_box(item['bbox']) for item in layout['expected'].values()]
    safe = [point for point in candidates if all(np.linalg.norm(point - np.clip(point, lo[:2], hi[:2]))
            >= 0.55 for lo, hi in boxes)]
    if not safe:
        return None
    point = min(safe, key=lambda value: (float(np.linalg.norm(value - camera_xy)), *value))
    look = _vector(layout['camera'].get('look_target', [*(layout['bounds'][:2] + layout['bounds'][2:]) / 2,
                                                        layout['floor_z']]), 3)
    delta = look[:2] - point
    return {'operation': 'robot_pose', 'position': [*point.tolist(), layout['floor_z']],
            'yaw_deg': float(np.rad2deg(np.arctan2(delta[1], delta[0]))), 'head_pitch_deg': 45.0}


class CategoryConverter(ConverterBase):
    category = ('Enumerative Perception', 'Spatial Segmentation')
    version = 'spatial_segmentation_v2_task_edit'

    def inspect(self, source):
        if (source.big_task, source.small_task) != self.category or (
                source.raw.get('big_task'), source.raw.get('small_task')) != self.category:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2: category_mismatch'}
        try:
            _layout(source)
        except (KeyError, ValueError, TypeError, StopIteration) as error:
            return {'eligible': False, 'reason': f'not_convertible_under_L1_L2: {error}',
                    'warning_codes': ['source_segmentation_unresolved']}
        return {'eligible': True, 'reason': 'Source chair-defined regions support a pending physical access patrol.'}

    def convert(self, source, context):
        if context.config.get('legacy_segmentation_patrol'):
            return self.convert_legacy(source, context)
        from .partial_occlusion import visual_discovery_task
        layout = _layout(source)
        labels = context.config.get('region_public_labels', {})
        label_contract = None
        if context.config.get('task_edit2_global_region_labels'):
            labels, evidence = global_region_labels(layout)
            label_contract = {'version': 'task_edit2_20261018_v1', 'source_identity': source.relative_path,
                'scene': source.raw['scene'], 'room': source.raw['room'], 'selection_seed': context.seed,
                'anchor_public_labels': labels, 'axis_evidence': evidence,
                'anchor_bindings': {key: layout['bindings'][key] for key in labels},
                'target_bindings': {pair['target']: layout['bindings'][pair['target']] for pair in layout['pairs']},
                'source_pairs': layout['pairs'], 'source_expected': layout['expected'],
                'source_anchor_states': {layout['bindings'][key]['name']:
                    context.config['region_anchor_source_states'][layout['bindings'][key]['name']]
                    for key in labels if layout['bindings'][key]['name'] in
                    context.config.get('region_anchor_source_states', {})},
                'preservation_tolerance_m': .02, 'native_boundary_mapping_verified': False,
                'initial_global_visual_label_association_verified': False,
                'capability_change': 'Global-frame chair descriptions replace the historical source-observer convention.'}
        anchors = list(dict.fromkeys(pair['anchor'] for pair in layout['pairs']))
        if any(not str(labels.get(key) or '').strip() for key in anchors):
            return {'scene_definition': original_scene(source, context, bindings=layout['bindings']),
                'task_definition': None, 'status': 'needs_human_decision', 'failure_stage': 'public_region_labels',
                'failure_reason': 'observable_region_labels_pending',
                'warning_codes': ['observable_region_labels_pending', 'source_mechanism_native_review_pending'],
                'source_layout_evidence': {'pairs': layout['pairs'], 'source_camera': layout['camera'],
                    'anchor_bindings': {key: layout['bindings'][key] for key in anchors}},
                'next_step': 'Supply qualified observable region_public_labels anchored to the preserved source observer.'}
        if len(set(labels[key] for key in anchors)) != len(anchors):
            raise ValueError('Public region labels must distinguish all original regions.')
        targets = [pair['target'] for pair in layout['pairs']]
        regions = {key: {'public_label': labels[key], 'targets': [pair['target'] for pair in layout['pairs']
                    if pair['anchor'] == key]} for key in anchors}
        category = layout['target_category'].replace('_', ' ')
        public = {'objective': f'Find all {len(targets)} distinct {category} objects in the separated regions: '
                + '; '.join(labels[key] for key in anchors) + '. '
                + 'For each fully inspected region report world_observation as ANSWER: <region label>.',
            'success_criteria': 'Visually discover every distinct target in its original region and report each named region. '
                + 'Facing an anchor or approaching a region does not establish discovery.',
            'environment': 'Original chairs define the same region boundaries and target membership as the source.',
            'constraints': ['Keep targets and divider/anchor geometry in place; preserve the source observer convention for region labels.',
                'No prescribed travel distance or return trip is required.']}
        start = _start(layout)
        if label_contract is not None:
            public['objective'] = ('Use the initial global image with +X pointing right and +Y pointing up to identify '
                'the chair boundaries. ' + public['objective'])
            public['constraints'] = ['Keep targets and chair boundaries in place. Region labels use the initial global '
                'world frame (+X right, +Y up), independently of the robot heading or historical source observer.',
                'No prescribed travel distance or return trip is required.']
        result = visual_discovery_task(source, context, bindings=layout['bindings'], target_keys=targets,
            public=public, patches=[] if start is None else [start], regions=regions,
            preserved_relations=['original chair-defined boundaries', 'target membership'] +
                ([] if label_contract is not None else ['source observer reference']),
            source_evidence={'pairs': layout['pairs'], 'expected': layout['expected'], 'source_camera': layout['camera']})
        if label_contract is not None:
            p = result['task_definition']['private']['evaluation']['parameters']
            p['source_task_revision'] = p.get('task_revision')
            p.update(region_label_contract=label_contract, definition_revision='task_edit2_20261018_v1',
                     task_revision='task_edit2_20261018_v1')
            result['task_definition']['version'] += 1
            result['scene_definition']['version'] += 1
            result['task_definition'].setdefault('repair', {})['region_label_contract'] = label_contract
            result.update(region_label_contract=label_contract, definition_revision='task_edit2_20261018_v1')
            result['change_summary'].append(label_contract['capability_change'])
        return result

    def convert_legacy(self, source, context):
        layout = _layout(source)
        start = _start(layout)
        result = {'scene_definition': original_scene(source, context, bindings=layout['bindings']),
                  'task_definition': None, 'status': 'conversion_failed', 'reuse_level': 'L1',
                  'qa_preserved': True, 'warning_codes': ['native_operation_unverified', 'assumption_review_required'],
                  'assumptions': [{
                      'doubt': 'Saved free-camera positions, coarse room bounds and chair AABBs do not establish R1 support, full-body clearance or a connected route.',
                      'chosen_assumption': 'Screen the nearest robot-only stance with 0.55 m source-geometry clearance and source-derived floor height. Preserve every asset and qualify actual native wheel-floor support, contacts and source geometry before an episode.',
                      'alternatives': ['Keep the unchanged original diagnostic scene.', 'Choose a different robot-only stance after actual native evidence, without moving targets or chairs.'],
                      'scope': source.relative_path,
                      'risk': 'Other loaded chairs, walls, arms, robot frame offsets and room raster errors may invalidate a screened spawn. The rule is advisory and cannot certify native feasibility.'}, {
                      'doubt': 'The original ground-level closeups and visibility annotations do not establish mounted-camera coverage or floor-level grasp reach.',
                      'chosen_assumption': 'Elevate portable targets independently onto measured supports, preferring their original chair seats. Enable dynamic collidable target physics and retain the physical access patrol, without requiring a grasp.',
                      'alternatives': ['Use another measured existing support or the authorized simple support when a chair seat is unsuitable.', 'Retain an explicit native support failure.'],
                      'scope': 'Both category navigation recipes.',
                      'risk': 'Physical access and heading do not guarantee seeing the sandwich beneath its chair or answering the count correctly. Original QA remains separate.'}],
                  'change_summary': ['Retain original targets, all chair groups, dimensions, reference observer and private QA.'],
                  'limitations': ['CPU conversion and source diagrams are not native loading, route qualification or physical execution.',
                                  'Patrol completion certifies physical access only; it does not certify mounted-camera visibility or correct counting.',
                                  'Source room bounds are coarse; native placement and safe floor stance require measured evidence.',
                                  'The declared 600-decision budget and 1.5-hour wall limit are unbenchmarked ceilings.'],
                  'private_source_audit': {'target_count': len(layout['pairs']), 'expected': layout['expected'],
                      'anchor_categories': sorted({layout['bindings'][pair['anchor']]['category'] for pair in layout['pairs']}),
                      'source_reference_camera': layout['camera'], 'room_bounds': layout['bounds'].tolist(),
                      'floor_z_m': layout['floor_z'], 'proposed_robot_pose': start,
                      'provenance': 'question_data/case_metadata/source_entries; question_data/render/resolved_objects; batch_counting_merge.py::_generate_observation_divided'},
                  'next_step': 'Review all candidates and diagnostics, then qualify actual native support, unchanged grouping, and guarded navigation.'}
        outside = [pair['target'] for pair in layout['pairs'] if np.any(
            _vector(layout['expected'][pair['target']]['position'], 3)[:2] < layout['bounds'][:2]) or np.any(
            _vector(layout['expected'][pair['target']]['position'], 3)[:2] > layout['bounds'][2:])]
        if outside:
            result['warning_codes'].append('source_target_outside_coarse_room_bbox')
            result['private_source_audit']['targets_outside_coarse_bbox'] = outside
        if str(len(layout['pairs'])) != str(layout['raw'].get('answer')):
            result['warning_codes'].append('source_count_annotation_conflict')
        if start is None:
            result.update(failure_stage='robot_start_screen',
                          failure_reason='not_convertible_under_L1_L2: no_source_screened_robot_start',
                          next_step='Retain the original scene diagnostic and measure a collision-free robot-only stance; the coarse screening failure is not a proof of global impossibility.')
            result['warning_codes'].append('robot_start_unresolved')
            return result
        operation = context.config.get('spatial_segmentation_operation')
        if operation is None:
            index = int(hashlib.sha256(f'{context.seed}:{source.relative_path}'.encode()).hexdigest()[:8], 16)
            operation = OPERATIONS[index % len(OPERATIONS)]
        if operation not in OPERATIONS:
            raise ValueError('Unsupported Spatial Segmentation operation recipe.')
        paired = operation == OPERATIONS[1]
        category = layout['target_category'].replace('_', ' ')
        objective = f'Patrol the separated accessible regions containing the original {category} objects. '
        objective += ('Reach two distinct safe approaches to each occupied region, at least 0.50 m apart. '
                      if paired else 'Reach a separate safe approach to every occupied region. ')
        objective += 'Face the object region and pause at each approach, then return to your departure position and park.'
        public = {'objective': objective,
                  'success_criteria': 'At every required approach, stop for at least 0.25 s, remain 0.55 to 1.25 m from '
                      'the object centre, face it within 30 degrees, and keep the chassis clear of chairs. '
                      'First approaches to different regions must be at least 0.35 m apart. Move at least 0.50 m '
                      'from departure, travel at least 1.0 m in total and park within 0.30 m of departure for 0.25 s. '
                      'Preserve the original objects and remain floor-supported without chassis contact with furniture.',
                  'environment': 'Original chair-defined groups with portable targets initialized on measured accessible supports, '
                      'preferably their original chair seats. Original chairs remain in place; the source under-chair '
                      'view and occlusion may change. Auxiliary original question: '
                      + layout['raw']['question'],
                  'constraints': ['Use existing guarded controls and mounted robot observations; move noncritical native movable furniture only with arm/gripper actions.',
                                  'Keep chairs in place and preserve initialized target locations, original scales and group identities.',
                                  'Do not manipulate the targets or replace physical patrol with a count answer.']}
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': operation,
                      'binding_keys': list(layout['bindings']),
                      'parameters': {**PARAMETERS, 'pairs': layout['pairs'], 'expected': layout['expected'],
                                     'accessible_placement': True}}
        target_keys = [pair['target'] for pair in layout['pairs']]
        patches = [start]
        for key in target_keys:
            patches.extend([{'operation': 'object_physics', 'phase': 'before_construction',
                             'binding': key, 'visual_only': False, 'fixed_base': False},
                            {'operation': 'object_physics', 'phase': 'after_postprocessing',
                             'binding': key, 'visual_only': False}])
        task = physical_task(source, context, public=public, evaluation=evaluation, bindings=layout['bindings'],
                             patches=patches, capabilities=['base_translation', 'base_rotation'],
                             qualification={'on_failure': 'skip', 'initial_visibility': 'none',
                                            'target_movable_collidable': False})
        task['defaults'].update(max_steps=600, wall_timeout_s=5400.0)
        result.update(scene_definition=original_scene(source, context, bindings=layout['bindings'], patches=patches),
                      task_definition=task, status='generated', reuse_level='L2',
                      operation_recipe=operation, available_operation_recipes=list(OPERATIONS),
                      interaction_policy={'mode': 'inspection', 'group_bindings': target_keys,
                          'manipulation_bindings': target_keys, 'relocation_policy': 'independent_groups',
                          'placement_groups': [[key] for key in target_keys],
                          'support_preferences': {pair['target']: pair['anchor'] for pair in layout['pairs']},
                          'support_search': 'measured_surfaces', 'allow_fallback_table': True,
                          'source_camera_policy': 'preserve', 'source_region': layout['raw']['room'],
                          'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10,
                          'maximum_stance_distance_m': .85,
                          'warning_codes': ['accessible_target_support_unverified', 'source_under_chair_occlusion_adapted']},
                      attempted_repairs=['Measured independent target placement, original-chair seat preference and physical target collision restoration.'])
        result['warning_codes'] += ['accessible_target_support_unverified', 'source_under_chair_occlusion_adapted']
        result['change_summary'] += ['Place dynamic collidable portable targets on real accessible supports while keeping chair anchors fixed.',
                                    'Retain the native ' + operation + ' predicate without requiring target grasp.']
        return result

    def convert_all(self, sources, context):
        sources = sorted(sources, key=lambda source: source.relative_path)
        manifest = super().convert_all(sources, context)
        manifest['failure_distribution'] = dict(Counter(record['failure_reason'] for record in manifest['records']
                                                       if record['status'] != 'generated'))
        manifest['operation_distribution'] = dict(Counter(record.get('operation_recipe', 'diagnostic_only')
                                                         for record in manifest['records']))
        manifest['sample_ids'] = self.sample(manifest, {'count': 5, 'seed': context.seed})
        manifest['physical_execution_verified'] = False
        write_json(context.output_root / 'batch_manifest.json', manifest)
        return manifest

    def sample(self, manifest, policy=None):
        policy = policy or {}
        count = min(policy.get('count', 5), len(manifest['records']))
        rows = sorted(manifest['records'], key=lambda row: row['candidate_id'])
        previous = set(policy.get('previous_failures', ()))
        selected, covered = [], set()

        def features(row):
            audit = row.get('private_source_audit', {})
            pitch = audit.get('source_reference_camera', {}).get('pitch_deg', 0.0)
            values = {('count', audit.get('target_count')), ('operation', row.get('operation_recipe')),
                      ('scene_room', (row.get('scene_model'), row.get('room'))),
                      ('outside_bbox', 'source_target_outside_coarse_room_bbox' in row['warning_codes']),
                      ('camera_pitch_bin', int(pitch // 30.0))}
            values.update(('anchor_category', category) for category in audit.get('anchor_categories', []))
            return values

        # Always retain a failure, when present, before maximizing source strata.
        priorities = [row for row in rows if row['status'] != 'generated'][:1]
        priorities += [row for row in rows if row['candidate_id'] in previous]
        for row in priorities:
            if len(selected) < count and row not in selected:
                selected.append(row)
                covered.update(features(row))
        weights = {'count': 4, 'operation': 3, 'anchor_category': 3, 'outside_bbox': 2,
                   'camera_pitch_bin': 2, 'scene_room': 1}
        while len(selected) < count:
            remaining = [row for row in rows if row not in selected]
            row = max(remaining, key=lambda value: sum(weights[feature[0]] for feature in features(value) - covered))
            selected.append(row)
            covered.update(features(row))
        return [row['candidate_id'] for row in selected]


def _parameters(session, config):
    if (config.get('type') != 'category_operation' or module_identity(config.get('module')) != MODULE
            or config.get('operation') not in OPERATIONS
            or (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name)):
        raise ValueError('Operation configuration does not match the native category scene.')
    parameters = config['parameters']
    if any(parameters.get(key) != value for key, value in PARAMETERS.items()) or not parameters.get('pairs'):
        raise ValueError('Use the declared access predicates and original private pairs.')
    return parameters


def _objects(session, config):
    objects = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
    if any(obj is None for obj in objects.values()):
        raise ValueError('Native source bindings are unavailable.')
    return objects


def _pose(obj):
    position, quaternion = obj.get_position_orientation()
    position, quaternion = _vector(position, 3), _vector(quaternion, 4)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0:
        raise ValueError('Native quaternion is invalid.')
    return {'position': position.tolist(), 'quaternion_xyzw': (quaternion / norm).tolist()}


def _native_state(session, parameters, objects):
    """Read measured native geometry, contacts, wheel rays and robot velocities."""
    from omnigibson.utils.sampling_utils import raytest
    from wia.tasks.scene_only import local_floor_surface

    robot, floor = session.robot, objects['floor']
    if floor.visual_only:
        raise ValueError('The bound native floor is not collidable.')
    pose = _pose(robot)
    xy = np.asarray(pose['position'][:2])
    low, high = _box(robot.base_footprint_link.aabb)
    hit = local_floor_surface(floor.prim_path, raytest([*xy, pose['position'][2] + 0.25],
                              [*xy, pose['position'][2] - 0.50], only_closest=False))
    floor_z = float(hit['position'][2])
    wheel_gaps = {}
    for name in robot.floor_touching_base_link_names:
        lo, hi = _box(robot.links[name].aabb)
        centre = (lo + hi) / 2.0
        wheel_hit = local_floor_surface(floor.prim_path, raytest([*centre[:2], float(lo[2]) + 0.10],
                                        [*centre[:2], float(lo[2]) - 0.20], only_closest=False))
        wheel_gaps[name] = float(lo[2] - wheel_hit['position'][2])
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
            point, normal = _vector(contact.position, 3), _vector(contact.normal, 3)
            if abs(point[2] - floor_z) <= 0.02 and abs(normal[2]) > 0.5:
                floor_contacts.append({**row, 'position': point.tolist(), 'normal': normal.tolist()})
        else:
            nonfloor_contacts.append(row)
    supported = bool(floor_contacts and wheel_gaps and all(abs(gap) <= 0.03 for gap in wheel_gaps.values()))
    poses = {key: _pose(obj) for key, obj in objects.items() if key != 'floor'}
    boxes = {key: _box(obj.aabb) for key, obj in objects.items() if key != 'floor'}
    anchor_clearances = {}
    for pair in parameters['pairs']:
        lo, hi = boxes[pair['anchor']]
        separation = np.maximum(np.maximum(low[:2] - hi[:2], lo[:2] - high[:2]), 0.0)
        anchor_clearances[pair['anchor']] = float(np.linalg.norm(separation))
    x, y, z, w = pose['quaternion_xyzw']
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    approaches = []
    for pair in parameters['pairs']:
        target_xy = np.asarray(poses[pair['target']]['position'][:2])
        delta = target_xy - xy
        distance = float(np.linalg.norm(delta))
        error = abs(math.degrees(math.atan2(math.sin(math.atan2(delta[1], delta[0]) - yaw),
                                            math.cos(math.atan2(delta[1], delta[0]) - yaw))))
        approaches.append({**pair, 'distance_m': distance, 'heading_error_deg': error,
                           'approach_geometry': bool(PARAMETERS['minimum_standoff_m'] <= distance
                               <= PARAMETERS['maximum_standoff_m'] and error <= PARAMETERS['maximum_heading_error_deg'])})
    clear = bool(all(value >= PARAMETERS['chassis_clearance_m'] for value in anchor_clearances.values()))
    return {'robot': pose, 'room': session.env.scene.seg_map.get_room_instance_by_point(xy),
            'floor_supported': supported, 'floor_surface_z_m': floor_z, 'wheel_floor_gaps_m': wheel_gaps,
            'floor_contacts': floor_contacts, 'nonfloor_contacts': nonfloor_contacts,
            'chassis_anchor_clearances_m': anchor_clearances,
            'safe': bool(supported and clear and not nonfloor_contacts),
            'native_linear_speed_m_s': float(np.linalg.norm(_vector(robot.base_footprint_link.get_linear_velocity(), 3))),
            'native_angular_speed_rad_s': float(np.linalg.norm(_vector(robot.base_footprint_link.get_angular_velocity(), 3))),
            'poses': poses, 'boxes': {key: [lo.tolist(), hi.tolist()] for key, (lo, hi) in boxes.items()},
            'approaches': approaches}


def _start_checks(session, config, objects):
    from wia.tasks.interaction_setup import reference_position, touching_task_support, on_top_task_support
    parameters = _parameters(session, config)
    measured = _native_state(session, parameters, objects)
    checks = {'floor_support_and_clearance': measured['safe'],
              'original_room_start': measured['room'] == session.room_name,
              'original_chairs_collidable': all(not objects[pair['anchor']].visual_only for pair in parameters['pairs']),
              'original_target_physics': all((not objects[pair['target']].visual_only and not objects[pair['target']].fixed_base)
                  if parameters.get('accessible_placement') else objects[pair['target']].visual_only for pair in parameters['pairs']),
              'source_object_bounds': all(np.max(np.abs(np.asarray(measured['boxes'][key]) -
                  np.asarray([reference_position(session, objects[key].name, corner) for corner in expected['bbox']]))) <= 0.05
                                          for key, expected in parameters['expected'].items()),
              'native_region_grouping': (all(touching_task_support(session, objects[pair['target']], objects['floor']) and
                  on_top_task_support(session, objects[pair['target']], objects['floor']) and
                  measured['boxes'][pair['target']][0][2] - measured['floor_surface_z_m'] >= .35 and
                  np.mean(np.asarray(measured['boxes'][pair['target']])[:, 2]) - measured['floor_surface_z_m'] <= 1.10
                  for pair in parameters['pairs']) if parameters.get('accessible_placement') else all(np.linalg.norm(
                  np.asarray(measured['poses'][pair['target']]['position'][:2]) - np.mean(
                  np.asarray(measured['boxes'][pair['anchor']])[:, :2], axis=0)) <= 0.05
                  and measured['poses'][pair['target']]['position'][2] < measured['boxes'][pair['anchor']][1][2]
                  for pair in parameters['pairs'])),
              'source_target_poses': all(np.linalg.norm(np.asarray(measured['poses'][key]['position']) -
                  reference_position(session, objects[key].name, expected['position']))
                  <= PARAMETERS['position_preservation_m'] and _angle(measured['poses'][key]['quaternion_xyzw'],
                  expected['quaternion_xyzw']) <= PARAMETERS['orientation_preservation_deg']
                  for key, expected in parameters['expected'].items() if 'position' in expected)}
    return measured, checks


def qualify_operation(session, executor, config, bound):
    """Fail closed on missing native preconditions; never act or write state."""
    checks = {}
    try:
        objects = _objects(session, config)
        measured, checks = _start_checks(session, config, objects)
        checks['guarded_base_capabilities'] = bool(executor.capabilities.base_translation and executor.capabilities.base_rotation)
        checks['bound_native_identity'] = all(objects.get(key) is obj for key, obj in bound.items()) and set(bound) == set(objects)
        accepted = bool(all(checks.values()))
        return {'accepted': accepted, 'checks': checks, 'native_state': measured,
                'reason': None if accepted else 'native_segmented_patrol_start_rejected',
                'route_qualified': False, 'mounted_visibility_qualified': False}
    except Exception as error:
        return {'accepted': False, 'checks': checks, 'reason': 'native_segmented_patrol_evidence_unavailable',
                'exception_summary': f'{type(error).__name__}: {error}', 'route_qualified': False}


class SegmentedPatrolEvaluator:
    """Private real-motion witnesses; no source answer or render flag can score."""

    def __init__(self, session, controller, config, output):
        self.parameters = _parameters(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.objects = _objects(session, config)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.active = 0, False

    def _time(self):
        value = float(self.session.pipeline.og.sim.current_time)
        if not math.isfinite(value):
            raise ValueError('Native simulation clock is unavailable.')
        return value

    def _preserved(self, measured):
        from wia.tasks.interaction_setup import touching_task_support, on_top_task_support
        supported = not self.parameters.get('accessible_placement') or all(
            not self.objects[pair['target']].visual_only and not self.objects[pair['target']].fixed_base and
            touching_task_support(self.session, self.objects[pair['target']], self.objects['floor']) and
            on_top_task_support(self.session, self.objects[pair['target']], self.objects['floor'])
            for pair in self.parameters['pairs'])
        return supported and all(np.linalg.norm(np.asarray(measured['poses'][key]['position']) - initial['position'])
            <= PARAMETERS['position_preservation_m'] and _angle(measured['poses'][key]['quaternion_xyzw'],
            initial['quaternion_xyzw']) <= PARAMETERS['orientation_preservation_deg'] and
            np.max(np.abs(np.diff(np.asarray(measured['boxes'][key]), axis=0)
                - np.diff(np.asarray(self.initial['boxes'][key]), axis=0))) <= PARAMETERS['position_preservation_m']
            for key, initial in self.initial['poses'].items())

    def begin_episode(self, observation):
        self.initial, checks = _start_checks(self.session, self.config, self.objects)
        if not all(checks.values()):
            raise ValueError('Native segmented patrol episode start is unqualified.')
        self.episode += 1
        self.start = np.asarray(self.initial['robot']['position'])
        self.previous = self.start.copy()
        self.previous_quaternion = self.initial['robot']['quaternion_xyzw']
        self.previous_time = self._time()
        self.travel, self.excursion, self.samples, self.return_duration = 0.0, 0.0, 0, 0.0
        self.speed, self.angular_speed = float('inf'), float('inf')
        self.visits = {pair['target']: [] for pair in self.parameters['pairs']}
        self.durations = dict.fromkeys(self.visits, 0.0)
        self.faults, self.active = set(), True
        return self.record(observation, action=None)

    def _required_visits(self):
        return 2 if self.config['operation'] == OPERATIONS[1] else 1

    def _all_visited(self):
        return all(len(visits) >= self._required_visits() for visits in self.visits.values())

    def sample_step(self):
        if not self.active:
            return
        try:
            self._advance(_native_state(self.session, self.parameters, self.objects), self._time())
        except Exception as error:
            self.faults.add(f'native_measurement_unavailable:{type(error).__name__}:{error}')

    def _advance(self, measured, now):
        point = np.asarray(measured['robot']['position'])
        elapsed = now - self.previous_time
        distance = float(np.linalg.norm(point - self.previous))
        rotation = _angle(measured['robot']['quaternion_xyzw'], self.previous_quaternion)
        # Repeated reads at one physics time cannot accumulate stop/visit evidence.
        if elapsed <= 0:
            if distance > 1e-6 or rotation > 1e-4 or elapsed < 0:
                self.faults.add('native_clock_or_unsampled_motion_failed')
            return
        config = self.controller.config
        if (distance > config.max_translation_speed_m_s * elapsed + config.translation_tolerance_m
                or rotation > config.max_rotation_speed_deg_s * elapsed + config.rotation_tolerance_deg):
            self.faults.add('native_motion_continuity_failed')
        self.speed = max(distance / elapsed, measured['native_linear_speed_m_s'])
        self.angular_speed = max(math.radians(rotation) / elapsed, measured['native_angular_speed_rad_s'])
        self.travel += float(np.linalg.norm(point[:2] - self.previous[:2]))
        self.excursion = max(self.excursion, float(np.linalg.norm(point[:2] - self.start[:2])))
        if not measured['safe']:
            self.faults.add('native_floor_or_contact_safety_failed')
        if not self._preserved(measured):
            self.faults.add('source_layout_changed')
        self.previous, self.previous_quaternion, self.previous_time = point, measured['robot']['quaternion_xyzw'], now
        self.samples += 1
        stopped = self.speed <= PARAMETERS['maximum_stop_speed_m_s'] and self.angular_speed <= PARAMETERS['maximum_stop_angular_speed_rad_s']
        allowed = bool(not self.faults and stopped and self.excursion >= PARAMETERS['minimum_departure_m'])
        first_positions = {key: np.asarray(visits[0]['robot_position'][:2]) for key, visits in self.visits.items() if visits}
        for approach in measured['approaches']:
            key = approach['target']
            visits = self.visits[key]
            distinct = (all(np.linalg.norm(point[:2] - other) >= PARAMETERS['different_group_separation_m']
                        for other_key, other in first_positions.items() if other_key != key) if not visits else
                        np.linalg.norm(point[:2] - np.asarray(visits[0]['robot_position'][:2])) >= PARAMETERS['second_approach_separation_m'])
            valid = allowed and approach['approach_geometry'] and distinct and len(visits) < self._required_visits()
            self.durations[key] = self.durations[key] + elapsed if valid else 0.0
            if self.durations[key] >= PARAMETERS['stop_duration_s']:
                visits.append({'native_time_s': now, 'sample': self.samples, 'robot_position': point.tolist(), **approach})
                first_positions[key] = np.asarray(visits[0]['robot_position'][:2])
                self.durations[key] = 0.0
        returned = (allowed and self._all_visited() and self.travel >= PARAMETERS['minimum_travel_m']
                    and np.linalg.norm(point[:2] - self.start[:2]) <= PARAMETERS['return_radius_m']
                    and measured['room'] == self.initial['room'])
        self.return_duration = self.return_duration + elapsed if returned else 0.0

    def record(self, observation, *, action):
        if not self.active:
            return None
        try:
            measured = _native_state(self.session, self.parameters, self.objects)
            current_pose = np.asarray(measured['robot']['position'])
            sampled = (np.linalg.norm(current_pose - self.previous) <= 1e-6
                       and _angle(measured['robot']['quaternion_xyzw'], self.previous_quaternion) <= 1e-4)
            if not sampled:
                self.faults.add('unsampled_native_terminal_motion')
            checks = {'all_separate_native_group_visits': self._all_visited(),
                      'actual_departure': self.excursion >= PARAMETERS['minimum_departure_m'],
                      'actual_travel': self.travel >= PARAMETERS['minimum_travel_m'],
                      'safe_continuous_trajectory': not self.faults and measured['safe'],
                      'original_layout_preserved': self._preserved(measured),
                      'returned_and_parked': bool(sampled and self.return_duration >= PARAMETERS['stop_duration_s']
                          and np.linalg.norm(current_pose[:2] - self.start[:2]) <= PARAMETERS['return_radius_m']
                          and measured['room'] == self.initial['room']
                          and max(self.speed, measured['native_linear_speed_m_s']) <= PARAMETERS['maximum_stop_speed_m_s']
                          and max(self.angular_speed, measured['native_angular_speed_rad_s']) <= PARAMETERS['maximum_stop_angular_speed_rad_s'])}
        except Exception as error:
            self.faults.add(f'native_terminal_evidence_unavailable:{type(error).__name__}:{error}')
            measured, checks = None, {'native_terminal_evidence': False}
        value = {'schema': 'private_segmented_access_patrol_v1', 'episode': self.episode,
                 'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step, 'action': action,
                 'operation': self.config['operation'], 'task_success': bool(all(checks.values()) and not self.faults),
                 'checks': checks, 'native_state': measured, 'visits': self.visits,
                 'faults': sorted(self.faults), 'sampled_travel_m': self.travel,
                 'maximum_departure_m': self.excursion, 'samples': self.samples,
                 'return_stop_duration_s': self.return_duration, 'configuration': self.config,
                 'mounted_visibility_qualified': False, 'qa_used_for_success': False}
        available = not any('unavailable:' in fault for fault in self.faults)
        items = [(f'{key}:approach:{index + 1}', len(visits) > index, 'historical_checkpoint',
                  visits[index] if len(visits) > index else None)
                 for key, visits in sorted(self.visits.items()) for index in range(self._required_visits())]
        items.append(('return_and_park', bool(checks.get('returned_and_park') and self._all_visited()),
            'terminal_condition', {'return_stop_duration_s': self.return_duration}))
        _completion_result(value, items, available=available)
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as handle:
            handle.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    return SegmentedPatrolEvaluator(session, controller, config, output)
