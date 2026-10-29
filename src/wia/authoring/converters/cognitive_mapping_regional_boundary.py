"""Preserve Regional Boundary scenes and measure native room visits and reports.

Run: python src/wia/runtime/commands/run_esi_category_campaign.py --help
CategoryConverter.convert_all performs CPU generation. Native qualification and
episode hooks read the actual robot, anchors and private room map without writes.
"""
from __future__ import annotations

from wia.resources import ESI_ROOT

import ast
from functools import lru_cache
import json
import math
from pathlib import Path
import re

import numpy as np

from wia.tasks.background_policy import legal_background_contact

from wia.authoring.conversion import ROOT, ConverterBase, original_scene, physical_task
from wia.authoring.converters.cognitive_mapping_connectivity import measure_native_state as measure_room_state
from wia.authoring.converters.cognitive_mapping_visit_reports import (
    ARRIVAL_INSTRUCTION, completion_public_goal, NativeVisitReportEvaluator, native_array, settled_native_state, visit_parameters)

from wia.authoring.converters.cognitive_mapping_visit_reports import (
    REVISED_MAPPING_OPERATIONS, revise_mapping_result, legacy_mapping_config,
    qualify_revised_mapping, SourceAnswerMappingEvaluator)

MODULE = 'wia.authoring.converters.cognitive_mapping_regional_boundary'
REVISED_OPERATION = REVISED_MAPPING_OPERATIONS['Regional Boundary']
EXPLORATION_OPERATION = 'anchor_region_exploration'
GENERATOR = ESI_ROOT / 'src/dataset_generation/task_cognitive_map/batch_cognitivemap_region.py'
COMMON = ESI_ROOT / 'src/active_explore/tasks/cognitive_mapping/common.py'
SOURCE_TYPES = ('object_in_region', 'object_closer_region', 'objects_same_region')


def _array(value):
    return native_array(value)


def _region(value):
    return re.sub(r'\s+', '_', str(value).split(' (', 1)[0].strip())


def _options(raw):
    value = raw.get('options_json', [])
    return json.loads(value) if isinstance(value, str) else value


@lru_cache(maxsize=1)
def _geometry():
    """Reuse exact pure generator formulas without its global simulator changes."""
    names = {'_normalize_bbox_xyxy', '_bboxes_intersect_xy', '_expand_bbox_until_wall_touch',
             '_expand_bbox_xyxy', '_room_bbox_center_xy', '_distance_xy', '_distance_point_to_bbox_xy'}
    parsed = ast.parse(GENERATOR.read_text())
    selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise RuntimeError('Original private room-distance formulas are unavailable.')
    scope = {'np': np, 'math': math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(GENERATOR), 'exec'), scope)
    return scope


@lru_cache(maxsize=1)
def _reference_selector():
    parsed = ast.parse(COMMON.read_text())
    selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                and node.name == 'extract_primary_camera_pose']
    scope = {}
    # The official selector's annotations do not affect this pure extraction.
    exec(compile('from __future__ import annotations\n', str(COMMON), 'exec'), scope)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(COMMON), 'exec',
                 flags=__import__('__future__').annotations.compiler_flag), scope)
    return scope['extract_primary_camera_pose']


def _public(raw, operation, source_task_type=None):
    question = raw['question'].strip()
    if operation == EXPLORATION_OPERATION:
        if source_task_type == 'objects_same_region':
            target = 'the region containing each of the two publicly queried objects'
        elif source_task_type == 'object_closer_region':
            target = 'the room closest to the queried object among the original answer choices'
        else:
            target = 'the room containing the publicly queried object'
        return {'objective': question + f' Explore and stably visit {target}. '
                    'Identify the rooms from your mounted observations. ' + ARRIVAL_INSTRUCTION,
            'success_criteria': 'Physically occupy and explicitly report every required anchor-defined room. '
                'Visit them in any order; extra rooms and backtracking are allowed. If already in a '
                'required room, remain stably there and report it without a forced exit or return.',
            'environment': 'The original full indoor scene. Original answer choices: '
                + '; '.join(_options(raw)) + '.',
            'constraints': ['Use mounted observations to identify the original queried furniture and rooms.',
                'Keep region-defining furniture, dimensions and scene topology unchanged.',
                'Use real guarded base motion; no terminal room, return trip or survey order is required.']}
    if operation == 'ordered_anchor_survey_return':
        match = re.fullmatch(r'Are (.+?) and (.+?) in the same region\?', question)
        first, second = match.groups()
        objective = (question + f' Survey the region containing {first}, then the region containing '
                     f'{second}, approaching each object within 1.5 m of its footprint. Use two '
                     'survey positions at least 0.5 m apart, then return to your starting position.')
        success = ('Complete the ordered two-object regional survey through actual robot motion, '
                   'travel at least 1.5 m in total, and return within 0.4 m of the episode-start '
                   'position in its original region. Stop with the original anchors unchanged.')
    else:
        relation = 'belongs to' if operation == 'enter_anchor_region' else 'is closest to'
        objective = (question + f' Inspect the queried object and room boundaries, then enter the '
                     f'room the object {relation}. If already inside that room, first leave it '
                     'through an existing passage and re-enter. Travel at least 0.5 m and stop inside.')
        success = ('The robot physically crosses from outside into the correctly selected room, '
                   'travels at least 0.5 m with an excursion of at least 0.3 m, and stops inside '
                   'with the queried original furniture unchanged.')
    return {'objective': objective, 'success_criteria': success,
            'environment': 'The original full indoor scene. Original answer choices: ' + '; '.join(_options(raw)) + '.',
            'constraints': ['Use mounted robot observations to identify the original queried furniture and rooms.',
                            'Keep the region-defining furniture and scene topology unchanged.',
                            'Use native base motion through existing passages for every route segment.']}


class CategoryConverter(ConverterBase):
    category = ('Cognitive Mapping', 'Regional Boundary')
    version = 'regional_boundary_task_edit_20261018'

    def inspect(self, source):
        if (source.big_task, source.small_task) != self.category:
            return {'eligible': False, 'reason': 'category_mismatch'}
        raw, payload = source.raw, source.payload
        q = payload.get('question_data', {})
        reason = None
        if raw.get('room') != 'full_scene':
            reason = 'source_is_not_full_scene'
        elif (q.get('task_type') or payload.get('task_type')) not in SOURCE_TYPES:
            reason = 'unsupported_regional_mechanism'
        elif not isinstance(raw.get('question'), str) or not raw['question'].strip():
            reason = 'missing_public_regional_question'
        elif not isinstance(_options(raw), list) or not _options(raw):
            reason = 'missing_public_regional_options'
        elif q.get('question') != raw['question'] or q.get('answer') != raw.get('answer'):
            reason = 'source_public_private_qa_disagreement'
        if reason:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2:' + reason,
                    'warning_codes': ['source_regional_contract_unresolved']}
        return {'eligible': True, 'reason': 'preserve_full_native_scene_and_inspect_relation',
                'source_task_type': q.get('task_type') or payload.get('task_type')}

    def convert(self, source, context):
        result = self._convert_legacy(source, context)
        if context.config.get('mapping_revision') == 'legacy' or context.config.get('regional_boundary_operation'):
            return result
        return revise_mapping_result(source, context, result)

    def _convert_legacy(self, source, context):
        raw, payload = source.raw, source.payload
        q = payload.get('question_data', {})
        kind = q.get('task_type') or payload.get('task_type')
        recipes = {kind: EXPLORATION_OPERATION for kind in SOURCE_TYPES}
        result = {'scene_definition': original_scene(source, context), 'task_definition': None,
                  'rule_revision': 'feedback_door_inspection_v1',
                  'interaction_policy': {'mode': 'navigation', 'group_bindings': [], 'manipulation_bindings': [],
                      'relocation_policy': 'preserve', 'source_region': None,
                      'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10,
                      'maximum_stance_distance_m': .85},
                  'status': 'needs_human_decision', 'reuse_level': 'L1', 'qa_preserved': True,
                  'warning_codes': ['native_route_unverified', 'mounted_sensor_coverage_unverified',
                                    'assumption_review_required', 'source_start_region_unspecified',
                                    'source_runtime_door_policy_difference',
                                    'door_opening_and_passage_unverified', 'native_start_support_unverified'],
                  'assumptions': [], 'attempted_repairs': [],
                  'change_summary': ['Preserve the full native floorplan, original furniture, scale, pose, reference and QA.'],
                  'limitations': ['No native scene, route, floor support, passage or terminal success was tested by this CPU converter.',
                                  'The historical report establishes shared construction only; it is not complete-route evidence.',
                                  'The generator excludes doors, while the active loader retains them. Base-only capabilities do not establish door opening or doorway clearance.',
                                  'A diagnostic preview can exist after interaction initialization rejects the spawn; images alone do not establish supported navigation.',
                                  'Private source/map evidence and Human review views are not Agent observations.'],
                  'next_step': 'Inspect actual spawn support and native door state; retain failed diagnostics. Test guarded door opening and passage only in a separately authorized native trial.'}
        reason = None
        if kind not in recipes or not isinstance(_options(raw), list) or not _options(raw):
            reason = 'unsupported_or_missing_public_regional_question'
        elif q.get('question') != raw.get('question') or q.get('answer') != raw.get('answer'):
            reason = 'source_public_private_qa_disagreement'
        elif raw.get('room') != 'full_scene':
            reason = 'source_is_not_full_scene'
        if reason:
            result.update(failure_reason=reason, diagnostic_stage='regional_source_inspection')
            return result
        operation = recipes[kind]
        verification = q.get('verification', {})
        if kind == 'objects_same_region':
            names, categories = q.get('object_instance_names', []), q.get('object_names', [])
            target_regions = [verification.get('object_a_region'), verification.get('object_b_region')]
            relation_valid = (len(target_regions) == 2 and all(target_regions)
                              and (str(raw['answer']).lower() == 'yes') == (target_regions[0] == target_regions[1]))
        else:
            names = [q.get('object_instance_name') or next(iter(q.get('required_objects', [])), None)]
            categories = [q.get('object_name') or q.get('object_category')]
            target_regions = [q.get('answer_region') if kind == 'object_closer_region'
                              else verification.get('assigned_region') or _region(raw.get('answer'))]
            relation_valid = target_regions[0] == _region(raw.get('answer'))
        if not names or len(names) != len(categories) or not all(names) or not all(categories) or not relation_valid:
            result.update(failure_reason='unresolved_original_anchor_or_region_label', diagnostic_stage='regional_source_inspection')
            return result
        if kind == 'objects_same_region' and not re.fullmatch(r'Are (.+?) and (.+?) in the same region\?', raw['question']):
            result.update(failure_reason='unsupported_public_pair_question', diagnostic_stage='regional_source_inspection')
            return result
        bindings = {key: {'name': name, 'category': category} for key, name, category in
                    zip(('anchor_a', 'anchor_b'), names, categories)}
        reference = _reference_selector()(payload)
        result['source_reference'] = {'selection': 'official_extract_primary_camera_pose', 'camera_pose': reference}
        parameters = {'target_regions': target_regions, 'source_task_type': kind,
                      'anchor_preservation_tolerance_m': .02, 'maximum_step_displacement_m': .10,
                      'maximum_stop_speed_m_s': .03,
                      'maximum_stop_angular_speed_rad_s': .10,
                      'floor_idx': 0, 'floor_contact_height_tolerance_m': .005,
                      'minimum_floor_contact_vertical_normal': .5,
                      'stop_samples': 3}
        parameters.update(visit_parameters(target_regions))
        if kind == 'object_in_region' and not verification.get('assigned_region'):
            result['warning_codes'].append('membership_assignment_field_missing')
            containment = verification.get('containment_regions', [])
            if containment and target_regions[0] not in containment:
                result['warning_codes'].append('source_expanded_assignment_conflict')
            result['assumptions'].append({
                'doubt': 'The source omits assigned_region while retaining its public/private QA answer and historical expanded-bbox containment metadata, which can refer to a different region.',
                'chosen_assumption': 'Preserve the original answer as the private intended destination only. Create an unchanged pending scene/task and require actual native anchor-centre segmentation to reproduce that region before physical execution.',
                'alternatives': ['Resolve source cleaning provenance and original assignment parameters.', 'Keep the native region mismatch as an original-scene diagnostic when the label cannot be reproduced.'],
                'scope': source.relative_path,
                'risk': 'Historical expanded-bbox containment can disagree with the current source answer. The answer alone cannot qualify the task or produce terminal success.'})
        if kind == 'object_closer_region':
            distances = verification.get('distance_to_regions', {})
            parameters.update(candidate_regions=[_region(option) for option in _options(raw)],
                              recorded_distances=distances, expansion_ratio=.35, expansion_min_m=.75,
                              metric='source_wall_limited_expanded_bbox_then_centre',
                              distance_reproduction_tolerance_m=.15)
            try:
                ranked = sorted(distances, key=lambda region: (distances[region]['bbox_distance'],
                                                               distances[region]['center_distance'], region))
            except (KeyError, TypeError):
                ranked = []
            if not ranked or ranked[0] != target_regions[0]:
                result.update(failure_reason='source_distance_order_disagrees_with_original_answer',
                              diagnostic_stage='source_distance_consistency')
                return result
            if min(distances, key=lambda name: distances[name]['center_distance']) != target_regions[0]:
                result['warning_codes'].append('region_centre_only_distance_would_change_answer')
            result['warning_codes'].append('source_expansion_configuration_not_saved')
            result['assumptions'].append({
                'doubt': 'The source records the closest-region metric and distances but omits the expansion settings; public MCQ distractors lack recorded distance evidence.',
                'chosen_assumption': 'Use the current official generator defaults of 0.35 relative expansion and 0.75 m minimum expansion, clipped by actual structural walls, only for private qualification. Require native scores to reproduce saved distances and preserve the private selected region across all public options.',
                'alternatives': ['Recover the original generation parameters and requalify without changing the source.', 'Publish the native metric mismatch diagnostic and request guidance if defaults do not reproduce the source.'],
                'scope': source.relative_path,
                'risk': 'Different expansion settings or native geometry can change the closest-region order; such cases are rejected by qualification, never repaired by changing the answer.'})
            if len(_options(raw)) > len(distances):
                result['warning_codes'].append('public_distractor_distance_unverified')
        parameters['source_verification'] = verification
        result['source_mechanism_provenance'] = {
            'generator': str(Path('ESI') / GENERATOR.relative_to(ESI_ROOT)),
            'generator_door_policy': 'exclude_door_and_sliding_door_then_remove_door_named_objects',
            'recorded_traversability_basename': payload.get('trav_map_basename'),
            'active_loader': 'ESI-Bench/src/active_explore/pipeline.py:build_env_config',
            'active_loader_door_policy': 'retain_native_doors',
            'scene_mechanism_modified': False}
        result['assumptions'].append({
            'doubt': 'The original generator excludes doors and attempts a no-door map, but the active loader retains native doors. Neither saved articulation nor a preview proves that R1 can open a door or pass it.',
            'chosen_assumption': 'Retain the current native door inventory and guarded controls for inspection. Record read-only native door joint evidence during qualification, and keep opening, clearance and route feasibility unverified. Do not change the garden spawn or door state based on appearance.',
            'alternatives': ['Review a source-faithful initialization-only door exclusion with explicit L2 provenance.', 'Qualify a compound guarded door-opening and room-entry recipe using the installed doors.', 'Keep a blocked or unsupported start as a diagnostic Review record.'],
            'scope': source.relative_path,
            'risk': 'The active door inventory adds constraints absent during generation; saved joint values and movable limits do not prove opening, traversability or a route.'})
        result['assumptions'].append({
            'doubt': 'Original free-camera positions and map clearance do not establish a collision-free R1 spawn, traversable passage or faithful mounted observation.',
            'chosen_assumption': 'Preserve native construction and the source camera reference. Use the existing L2 navigation initialization to screen a robot-only stance without moving furniture; require actual native support and relation checks separately. Do not derive robot height from camera height or treat grass-name contact as floor support.',
            'alternatives': ['Use an evidence-based robot-only stance in a later reversible converter revision.', 'Retain the unchanged original-scene diagnostic if native support or passage fails.'],
            'scope': source.relative_path,
            'risk': 'The native default spawn may be invalid or may provide different visual evidence; CPU generation does not certify readiness.'})
        if kind == 'objects_same_region':
            result['assumptions'].append({
                'doubt': 'The binary source QA has no destination label suitable for a single room-entry goal.',
                'chosen_assumption': 'Visit the unordered set of regions defined by the two public anchors, and require Agent-authored arrival reports while actually there. Source Yes/No remains private auxiliary QA; no order, separation or return is imposed.',
                'alternatives': ['Review a boundary-crossing inspection route after native passage evidence.', 'Keep an original-scene diagnostic when either anchor region has no accessible survey position.'],
                'scope': source.relative_path,
                'risk': 'Anchor-defined room visits do not alone answer the original binary QA; actual native relation qualification remains required.'})
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': operation,
                      'binding_keys': list(bindings), 'parameters': parameters}
        result.update(scene_definition=original_scene(source, context, bindings=bindings),
                      task_definition=physical_task(source, context, public=completion_public_goal(_public(raw, operation, kind), self.category[1]), evaluation=evaluation,
                          bindings=bindings, capabilities=['base_translation', 'base_rotation'],
                          qualification={'on_failure': 'skip', 'initial_visibility': 'none'}),
                      status='generated', operation=operation, source_task_type=kind,
                      source_region_evidence={'target_regions': target_regions, 'verification': verification},
                      change_summary=result['change_summary'] + ['Add a native ' + operation + ' goal with private map/trajectory measurement.'])
        return result

    def sample(self, manifest, policy=None):
        policy = policy or {}
        count = min(policy.get('count', 5), len(manifest['records']))
        rows = sorted(manifest['records'], key=lambda row: row['candidate_id'])
        groups = [[row for row in rows if row['status'] != 'generated']]
        membership = [row for row in rows if row.get('source_task_type') == 'object_in_region']
        groups += [[row for row in membership if 'membership_assignment_field_missing' not in row.get('warning_codes', [])] or membership]
        groups += [[row for row in rows if row.get('source_task_type') == 'objects_same_region'
                    and len(set(row.get('source_region_evidence', {}).get('target_regions', []))) == 1]]
        closest = [row for row in rows if row.get('source_task_type') == 'object_closer_region']
        garden = sorted((row for row in closest if str(row.get('scene_model', '')).endswith('_garden')),
                        key=lambda row: row['source_identity'])
        groups += [garden or [row for row in closest if 'region_centre_only_distance_would_change_answer' in row.get('warning_codes', [])] or closest]
        groups += [[row for row in rows if row.get('source_task_type') == 'objects_same_region'
                    and len(set(row.get('source_region_evidence', {}).get('target_regions', []))) == 2]]
        groups += [[row for row in rows if 'source_expanded_assignment_conflict' in row.get('warning_codes', [])]]
        selected = []
        for group in groups:
            candidate = next((row['candidate_id'] for row in group if row['candidate_id'] not in selected), None)
            if candidate and len(selected) < count:
                selected.append(candidate)
        for candidate in super().sample(manifest, policy):
            if candidate not in selected and len(selected) < count:
                selected.append(candidate)
        return selected


def _query_region(scene, position):
    import torch
    seg = getattr(scene, 'seg_map', None)
    if seg is None or not callable(getattr(seg, 'get_room_instance_by_point', None)):
        raise ValueError('native_room_segmentation_unavailable')
    value = seg.get_room_instance_by_point(torch.tensor(position[:2], dtype=torch.float32))
    if value is None:
        return None
    return str(value)


def _anchor_state(obj):
    position, quaternion = obj.get_position_orientation()
    low, high = (_array(value).tolist() for value in obj.aabb)
    position, quaternion = _array(position).tolist(), _array(quaternion).tolist()
    if not all(math.isfinite(value) for row in (position, quaternion, low, high) for value in row):
        raise ValueError('nonfinite_native_anchor_geometry')
    return {'name': obj.name, 'position': position, 'quaternion_xyzw': quaternion,
            'aabb_min': low, 'aabb_max': high, 'centre_xy': [(a + b) / 2 for a, b in zip(low[:2], high[:2])]}


def _floor_contact_evidence(session):
    """Require measured external contacts on actual floor objects in map layer 0.

    Installed EntityPrim.contact_list aggregates RigidPrim.contact_list records,
    whose body paths are decoded native rigid bodies. No contact-reporting flag
    is enabled here, and an unavailable/empty report never establishes support.
    """
    scene, robot = session.env.scene, session.robot
    evidence = {'segmentation_floor_idx': 0, 'segmentation_floor_z_m': None,
                'native_floor_layer_known': False, 'native_floor_object_paths': [],
                'native_floor_contacts': [], 'nonfloor_contacts': [],
                'native_floor_contact': False, 'no_nonfloor_robot_contact': True,
                'native_contacts_available': False, 'contact_support_reason': None}
    try:
        heights = _array(scene.seg_map.floor_heights).reshape(-1)
        if not len(heights) or not math.isfinite(float(heights[0])):
            raise ValueError('native_segmentation_floor_0_height_unavailable')
        floor = float(heights[0])
        evidence.update(segmentation_floor_z_m=floor, native_floor_layer_known=True)
        objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
        floor_paths = [str(obj.prim_path) for obj in objects
                       if getattr(obj, 'category', None) in ('floor', 'floors')
                       and getattr(obj, 'prim_path', None)]
        evidence['native_floor_object_paths'] = floor_paths
        if not floor_paths:
            raise ValueError('native_scene_floor_objects_unavailable')
        robot_path = str(robot.prim_path)
        if not robot_path or not callable(getattr(robot, 'contact_list', None)):
            raise ValueError('native_robot_contact_measurement_unavailable')
        contacts = robot.contact_list()
        if contacts is None:
            raise ValueError('native_robot_contact_measurement_unavailable')
        evidence['native_contacts_available'] = True
        owns = lambda body, root: body == root or body.startswith(root + '/')
        for contact in contacts:
            bodies = [str(contact.body0), str(contact.body1)]
            robot_bodies = [owns(body, robot_path) for body in bodies]
            if all(robot_bodies):
                continue  # Self-contact is not an external support/collision.
            external = [body for body, owned in zip(bodies, robot_bodies) if not owned]
            position, normal = getattr(contact, 'position', None), getattr(contact, 'normal', None)
            position = _array(position) if position is not None else np.empty(0)
            normal = _array(normal) if normal is not None else np.empty(0)
            row = {'body0': bodies[0], 'body1': bodies[1],
                   'position': position.tolist() if position.shape == (3,) and np.isfinite(position).all() else None,
                   'normal': normal.tolist() if normal.shape == (3,) and np.isfinite(normal).all() else None}
            is_floor = bool(any(robot_bodies) and external and all(
                any(owns(body, root) for root in floor_paths) for body in external))
            if is_floor:
                evidence['native_floor_contacts'].append(row)
            elif not legal_background_contact(session, *bodies):
                evidence['nonfloor_contacts'].append(row)
        evidence['no_nonfloor_robot_contact'] = not evidence['nonfloor_contacts']
        floor_contacts = evidence['native_floor_contacts']
        evidence['native_floor_contact'] = bool(floor_contacts and all(
            row['position'] is not None and row['normal'] is not None
            and abs(row['position'][2] - floor) <= .005 and abs(row['normal'][2]) >= .5
            for row in floor_contacts))
        if not evidence['no_nonfloor_robot_contact']:
            evidence['contact_support_reason'] = 'native_robot_nonfloor_external_contact'
        elif not evidence['native_floor_contact']:
            evidence['contact_support_reason'] = 'native_robot_floor_0_contact_not_verified'
    except (ValueError, RuntimeError, AttributeError, KeyError, TypeError, IndexError, AssertionError) as error:
        evidence['contact_support_reason'] = f'native_floor_contact_measurement_unavailable:{error}'
        evidence['native_floor_contact'] = False
    return evidence


def _robot_support_evidence(session, pose):
    evidence = _floor_contact_evidence(session)
    quaternion = _array(pose['quaternion_xyzw'])
    norm = float(np.linalg.norm(quaternion))
    evidence.update(robot_upright=False, robot_floor_aabb_screen=False,
                    native_robot_bottom_m=None, robot_supported_upright=False)
    if quaternion.shape != (4,) or norm == 0 or not np.isfinite(quaternion).all():
        return evidence
    x, y, z, w = quaternion / norm
    evidence['robot_upright'] = bool(1 - 2 * (x*x + y*y) >= .9)
    low, high = (_array(value) for value in session.robot.aabb)
    if low.shape == high.shape == (3,) and np.isfinite([low, high]).all():
        evidence['native_robot_bottom_m'] = float(low[2])
        if evidence['native_floor_layer_known']:
            evidence['robot_floor_aabb_screen'] = abs(float(low[2]) - evidence['segmentation_floor_z_m']) <= .15
    evidence['robot_supported_upright'] = bool(evidence['robot_upright']
        and evidence['robot_floor_aabb_screen'] and evidence['native_floor_contact']
        and evidence['no_nonfloor_robot_contact'])
    return evidence


def _robot_screen(session, pose):
    return _robot_support_evidence(session, pose)['robot_supported_upright']


def _native_region_distances(scene, anchor, parameters):
    """Reconstruct the original metric from actual room masks and wall geometry."""
    import torch
    seg, geometry = scene.seg_map, _geometry()
    mapping, room_map = seg.room_ins_id_to_ins_name, _array(seg.room_ins_map)
    raw_objects = scene.objects
    objects = raw_objects.values() if isinstance(raw_objects, dict) else raw_objects
    walls = []
    for obj in objects:
        name, category = str(obj.name).lower(), str(obj.category).lower()
        if 'wall' in name and ('walls' in name or category in ('walls', 'wall')):
            low, high = (_array(value) for value in obj.aabb)
            walls.append(tuple(geometry['_normalize_bbox_xyxy']([*low[:2], *high[:2]])))
    measured = {}
    for room_id, name in mapping.items():
        if name not in parameters['candidate_regions']:
            continue
        pixels = np.argwhere(room_map == int(room_id))
        if not len(pixels):
            continue
        low, high = pixels.min(axis=0), pixels.max(axis=0)
        corners = torch.tensor([[low[0], low[1]], [low[0], high[1]],
                                [high[0], low[1]], [high[0], high[1]]], dtype=torch.float32)
        world = _array(seg.map_to_world(corners))
        bbox = tuple([*world.min(axis=0), *world.max(axis=0)])
        if walls:
            expanded = geometry['_expand_bbox_until_wall_touch'](bbox, walls,
                parameters['expansion_ratio'], parameters['expansion_min_m'])
        else:
            expanded = geometry['_expand_bbox_xyxy'](bbox, parameters['expansion_ratio'], parameters['expansion_min_m'])
        measured[str(name)] = {
            'bbox_distance': geometry['_distance_point_to_bbox_xy'](anchor['centre_xy'], expanded),
            'center_distance': geometry['_distance_xy'](anchor['centre_xy'], geometry['_room_bbox_center_xy'](bbox))}
    if set(measured) != set(parameters['candidate_regions']):
        raise ValueError('native_candidate_room_masks_unavailable')
    return measured


def _binding_objects(session, config):
    bound = {key: session.env.scene.object_registry('name', config['bindings'][key])
             for key in config['binding_keys']}
    if any(obj is None for obj in bound.values()):
        raise ValueError('native_anchor_binding_missing')
    return bound


def _relation_evidence(session, config, bound):
    scene, p = session.env.scene, config['parameters']
    states = [_anchor_state(bound[key]) for key in config['binding_keys']]
    regions = [_query_region(scene, state['centre_xy']) for state in states]
    result = {'native_anchor_states': states, 'native_anchor_seg_regions': regions}
    if config['operation'] == 'enter_closest_region' or (config['operation'] == EXPLORATION_OPERATION
            and p['source_task_type'] == 'object_closer_region'):
        measured = _native_region_distances(scene, states[0], p)
        ranked = sorted(measured, key=lambda name: (measured[name]['bbox_distance'], measured[name]['center_distance'], name))
        result.update(native_region_distances=measured, native_selected_region=ranked[0])
        tolerance = p['distance_reproduction_tolerance_m']
        reproduced = all(name in measured and all(abs(float(record[key]) - measured[name][key]) <= tolerance
            for key in ('bbox_distance', 'center_distance')) for name, record in p['recorded_distances'].items())
        result.update(recorded_metric_reproduced=reproduced,
                      region_relation_consistent=reproduced and ranked[0] == p['target_regions'][0])
    else:
        result['region_relation_consistent'] = regions == p['target_regions']
    return result


def _native_door_evidence(session):
    """Read installed door state; joint ranges never certify opening or passage."""
    evidence = {'schema': 'private_regional_door_inspection_v1',
                'inventory_available': False, 'doors': [],
                'door_opening_verified': False, 'doorway_clearance_verified': False,
                'route_feasibility_verified': False}
    try:
        objects = session.env.scene.objects
        objects = list(objects.values()) if isinstance(objects, dict) else list(objects)
        evidence['inventory_available'] = True
        doors = [obj for obj in objects if getattr(obj, 'category', None) in ('door', 'sliding_door')]
        for obj in sorted(doors, key=lambda item: str(item.name)):
            row = {'name': str(obj.name), 'category': str(obj.category),
                   'joint_state_available': False}
            evidence['doors'].append(row)
            try:
                row.update(prim_path=str(obj.prim_path), fixed_base=bool(obj.fixed_base),
                           visual_only=bool(obj.visual_only), native_joint_count=int(obj.n_joints))
                if not obj.n_joints:
                    row['reason'] = 'native_door_has_no_articulated_joints'
                    continue
                positions, lower, upper = (_array(value).reshape(-1) for value in (
                    obj.get_joint_positions(), obj.joint_lower_limits, obj.joint_upper_limits))
                if not (positions.shape == lower.shape == upper.shape and positions.size
                        and np.isfinite([positions, lower, upper]).all() and np.all(lower <= upper)):
                    raise ValueError('invalid_native_door_joint_arrays')
                row.update(joint_state_available=True, joint_positions=positions.tolist(),
                           joint_lower_limits=lower.tolist(), joint_upper_limits=upper.tolist(),
                           joint_motion_range_observed=bool(np.any(upper - lower > 1e-6)))
            except (ValueError, RuntimeError, AttributeError, KeyError, TypeError, AssertionError) as error:
                row['reason'] = 'native_door_joint_measurement_unavailable:' + str(error)
    except (ValueError, RuntimeError, AttributeError, KeyError, TypeError) as error:
        evidence['inventory_available'] = False
        evidence['reason'] = 'native_door_inventory_unavailable:' + str(error)
    return evidence


def qualify_operation(session, executor, config, bound):
    if config.get('operation') == REVISED_OPERATION:
        return qualify_revised_mapping(session, executor, config, bound, qualify_operation)
    """Inspect actual native preconditions; the executor is not a pose controller."""
    # Door diagnostics are private evidence, including when floor checks reject
    # the start. They do not loosen any precondition or claim a route was run.
    checks = {'native_door_inspection': _native_door_evidence(session),
              'route_feasibility_verified': False}
    try:
        if config['operation'] not in (EXPLORATION_OPERATION, 'enter_anchor_region', 'enter_closest_region', 'ordered_anchor_survey_return'):
            raise ValueError('unsupported_regional_operation')
        if (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name):
            raise ValueError('native_scene_room_mismatch')
        position, quaternion = (_array(value) for value in session.robot.get_position_orientation())
        if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            raise ValueError('invalid_native_robot_pose')
        norm = np.linalg.norm(quaternion)
        if norm == 0:
            raise ValueError('invalid_native_robot_orientation')
        checks.update(_robot_support_evidence(session, {'position': position.tolist(),
                      'quaternion_xyzw': quaternion.tolist()}))
        checks['robot_initial_region'] = _query_region(session.env.scene, position)
        checks.update(_relation_evidence(session, config, bound))
        if not checks['native_floor_contact'] or not checks['no_nonfloor_robot_contact']:
            raise ValueError(checks['contact_support_reason'] or 'native_robot_floor_0_contact_not_verified')
        if not checks['robot_supported_upright']:
            raise ValueError('native_robot_floor_support_or_orientation_failed')
        if checks['robot_initial_region'] is None:
            raise ValueError('native_robot_spawn_outside_named_region')
        if not checks['region_relation_consistent']:
            raise ValueError('native_original_regional_relation_not_reproduced')
        checks['route_feasibility_verified'] = False
        return {'accepted': True, 'checks': checks, 'reason': None}
    except (ValueError, RuntimeError, AttributeError, KeyError, TypeError) as error:
        return {'accepted': False, 'checks': checks, 'reason': str(error)}


class LegacyRegionalOperationEvaluator:
    """Measure real native base motion; source answers never satisfy a predicate."""
    def __init__(self, session, controller, config, output):
        if (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name):
            raise ValueError('Native regional evaluation scene/room mismatch.')
        self.session, self.controller, self.config = session, controller, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.bound = _binding_objects(session, config)
        self.episode, self.initial = 0, None

    def begin_episode(self, observation):
        self.episode += 1
        self.initial = self.controller.pose()
        self.initial_region = _query_region(self.session.env.scene, self.initial['position'])
        self.initial_anchors = [_anchor_state(self.bound[key]) for key in self.config['binding_keys']]
        relation = _relation_evidence(self.session, self.config, self.bound)
        self.relation_valid = relation['region_relation_consistent']
        self.relation_evidence = relation
        self.last = self.initial['position']
        self.path_m, self.excursion_m, self.stage, self.quiet_samples = 0., 0., 0, 0
        self.survey_position = None
        self.outside_selected = self.initial_region != self.config['parameters']['target_regions'][0]
        self.discontinuous, self.anchors_preserved = False, True
        self.contact_safe_trajectory = True
        self.record(observation, action=None)

    def sample_step(self):
        if self.initial is None:
            return
        p, scene = self.config['parameters'], self.session.env.scene
        pose = self.controller.pose()
        position = pose['position']
        if not np.isfinite(position).all():
            raise ValueError('Nonfinite native robot trajectory.')
        delta = math.dist(position, self.last)
        self.discontinuous |= delta > p['maximum_step_displacement_m']
        self.path_m += math.dist(position[:2], self.last[:2])
        self.excursion_m = max(self.excursion_m, math.dist(position[:2], self.initial['position'][:2]))
        self.last = list(position)
        region = _query_region(scene, position)
        support = _robot_support_evidence(self.session, pose)
        robot_supported = support['robot_supported_upright']
        self.contact_safe_trajectory &= robot_supported
        anchors = [_anchor_state(self.bound[key]) for key in self.config['binding_keys']]
        for initial, current in zip(self.initial_anchors, anchors):
            quaternion_change = 1 - abs(float(np.dot(initial['quaternion_xyzw'], current['quaternion_xyzw'])) /
                max(float(np.linalg.norm(initial['quaternion_xyzw']) * np.linalg.norm(current['quaternion_xyzw'])), 1e-12))
            preserved = (math.dist(initial['position'], current['position']) <= p['anchor_preservation_tolerance_m']
                         and math.dist(initial['aabb_min'], current['aabb_min']) <= p['anchor_preservation_tolerance_m']
                         and math.dist(initial['aabb_max'], current['aabb_max']) <= p['anchor_preservation_tolerance_m']
                         and quaternion_change <= 1e-4)
            self.anchors_preserved &= preserved
        if self.config['operation'] == 'ordered_anchor_survey_return':
            if self.stage < 2:
                index = self.stage
                anchor = self.initial_anchors[index]
                distance = math.sqrt(sum(max(anchor['aabb_min'][axis] - position[axis],
                    position[axis] - anchor['aabb_max'][axis], 0.) ** 2 for axis in (0, 1)))
                separated = index == 0 or math.dist(position[:2], self.survey_position) >= p['survey_separation_m']
                if (robot_supported and region == p['target_regions'][index]
                        and distance <= p['survey_distance_m'] and separated):
                    self.survey_position = list(position[:2])
                    self.stage += 1
            endpoint = robot_supported and self.stage == 2 and region == self.initial_region and math.dist(
                position[:2], self.initial['position'][:2]) <= p['return_distance_m']
        else:
            self.outside_selected |= robot_supported and region is not None and region != p['target_regions'][0]
            endpoint = robot_supported and self.outside_selected and region == p['target_regions'][0]
        speed = float(np.linalg.norm(_array(self.session.robot.get_linear_velocity())))
        angular_speed = float(np.linalg.norm(_array(self.session.robot.get_angular_velocity())))
        quiet = (endpoint and robot_supported and speed <= p['maximum_stop_speed_m_s']
                 and angular_speed <= p.get('maximum_stop_angular_speed_rad_s', .10))
        # Repeated decision-boundary reads cannot count as additional physics steps.
        if getattr(self, '_in_record', False):
            if not quiet:
                self.quiet_samples = 0
        else:
            self.quiet_samples = self.quiet_samples + 1 if quiet else 0
        self.measured = {'base': pose, 'native_region': region, 'anchor_states': anchors,
                         'speed_m_s': speed, 'angular_speed_rad_s': angular_speed,
                         'endpoint_reached': bool(endpoint),
                         'robot_supported_upright': robot_supported, 'native_support': support}

    def record(self, observation, *, action=None):
        if self.initial is None:
            return None
        self._in_record = True
        try:
            self.sample_step()
        finally:
            self._in_record = False
        p = self.config['parameters']
        criteria = {'native_source_relation_preserved': bool(self.relation_valid),
                    'anchors_unchanged': bool(self.anchors_preserved),
                    'continuous_actual_base_trajectory': not self.discontinuous,
                    'native_floor_0_supported_trajectory': bool(self.contact_safe_trajectory),
                    'minimum_route_completed': self.path_m >= p['minimum_route_m'],
                    'minimum_excursion_completed': self.excursion_m >= p['minimum_excursion_m'],
                    'native_endpoint_reached': self.measured['endpoint_reached'],
                    'native_robot_supported_upright': self.measured['robot_supported_upright'],
                    'native_stop_measured': self.quiet_samples >= p['stop_samples']}
        record = {'schema': 'private_regional_operation_v1', 'episode': self.episode,
                  'snapshot_id': getattr(observation, 'snapshot_id', None),
                  'action_step': getattr(observation, 'action_step', None), 'action': action,
                  'task_success': all(criteria.values()), 'criteria': criteria,
                  'operation': self.config['operation'], 'path_m': self.path_m,
                  'excursion_m': self.excursion_m, 'survey_stage': self.stage,
                  'quiet_physics_samples': self.quiet_samples, 'initial_base': self.initial,
                  'initial_region': self.initial_region, 'initial_relation': self.relation_evidence,
                  'configuration': self.config, **self.measured}
        temporary = self.output / 'state.json.tmp'
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        temporary.replace(self.output / 'state.json')
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        return record


def create_evaluator(session, controller, config, output):
    if config.get('operation') == REVISED_OPERATION:
        native = create_evaluator(session, controller, legacy_mapping_config(config), output)
        return SourceAnswerMappingEvaluator(session, controller, config, output, native)
    evaluator = RegionalOperationEvaluator if config['operation'] == EXPLORATION_OPERATION else LegacyRegionalOperationEvaluator
    return evaluator(session, controller, config, output)


class RegionalOperationEvaluator(NativeVisitReportEvaluator):
    """Unordered anchor-defined native room visits with explicit Agent reports."""
    def __init__(self, session, controller, config, output):
        if (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name):
            raise ValueError('Native regional evaluation scene/room mismatch.')
        self.bound = _binding_objects(session, config)
        super().__init__(session, controller, config, output)

    def _measure_native(self):
        measured = measure_room_state(self.session, parameters=self.parameters)
        pose = {'position': measured['position'], 'quaternion_xyzw': measured['quaternion_xyzw']}
        support = _robot_support_evidence(self.session, pose)
        if not support.get('native_contacts_available', False):
            raise ValueError(support.get('contact_support_reason') or 'native_contact_measurement_unavailable')
        measured['native_support'] = support
        measured['state_valid'] = bool(measured['state_valid'] and support['robot_supported_upright'])
        return settled_native_state(self.session, measured, minimum_up_dot=.9)

    def _qualify_start(self):
        door_evidence = _native_door_evidence(self.session)
        (self.output / 'initial_native_doors.json').write_text(
            json.dumps(door_evidence, indent=2, allow_nan=False) + '\n')
        self.initial_anchors = [_anchor_state(self.bound[key]) for key in self.config['binding_keys']]
        self.relation_evidence = _relation_evidence(self.session, self.config, self.bound)
        measured = self._measure_native()
        return {'native_anchor_relation_preserved': bool(self.relation_evidence['region_relation_consistent']),
            'native_required_regions_exist': self.required.issubset(measured['known_regions']),
            'native_initial_floor_contact_and_pose': measured['state_valid'],
            'native_initial_region_known': measured['root_region'] in measured['known_regions']}

    def _additional_checks(self, measured):
        unchanged = True
        current = [_anchor_state(self.bound[key]) for key in self.config['binding_keys']]
        tolerance = self.parameters['anchor_preservation_tolerance_m']
        for initial, anchor in zip(self.initial_anchors, current):
            initial_q, current_q = (np.asarray(value) for value in (
                initial['quaternion_xyzw'], anchor['quaternion_xyzw']))
            denominator = float(np.linalg.norm(initial_q) * np.linalg.norm(current_q))
            orientation_error = 1 - abs(float(np.dot(initial_q, current_q))) / max(denominator, 1e-12)
            unchanged &= (all(math.dist(initial[key], anchor[key]) <= tolerance
                for key in ('position', 'aabb_min', 'aabb_max')) and orientation_error <= 1e-4)
        return {'native_source_relation_preserved': bool(self.relation_evidence['region_relation_consistent']),
            'anchors_unchanged': bool(unchanged)}
