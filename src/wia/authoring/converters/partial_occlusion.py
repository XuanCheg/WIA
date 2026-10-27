"""Convert Partial Occlusion into identification and stable destination placement.

Run PartialOcclusionConverter.convert_all(sources, ConversionContext(batch_root))
with the behavior environment and OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=2.
Native hooks use the shared task runtime; they only read measured state. This
module never submits capture jobs, changes source data, or performs robot actions.
"""
from __future__ import annotations

from wia.resources import ESI_ROOT

from collections import Counter
from copy import deepcopy
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import re

import numpy as np

from wia.authoring.conversion import ROOT, ConverterBase, map_records, original_scene, physical_task


MODULE = 'core.esi_converters.partial_occlusion'
LEGACY_RULE_REVISION = 'partial_occlusion_annotation_retrieval_20261015_r3'
RULE_REVISION = 'partial_occlusion_identify_place_20261018_v1'
TASK_REVISION = 'task_edit_20261018_v1'
IDENTIFY_PLACE = 'identify_and_place'
GROUNDING_REVISION = 'task_edit2_grounding_20261018_v1'
GROUNDING_OPERATION = 'viewpoint_identify_and_place_v2'
FEEDBACK_REVISION = 'f4046c9f3ca054f3015e_ad21f219c5500ba609ca898c'
FEEDBACK_SOURCE = ('Perceptual Grounding/Partial Occlusion/Beechwood_0_garden/'
                   'dining_room_0/Beechwood_0_garden__dining_room_0_2.json')


def revise_grounding_result(result, source, context):
    """Opt-in source-faithful viewpoint/reveal identification before retrieval."""
    if result.get('task_definition') is None:
        return result
    task, scene = result['task_definition'], result['scene_definition']
    evaluation, bindings = task['private']['evaluation'], task['private']['bindings']
    p = evaluation['parameters']
    partial = source.small_task == 'Partial Occlusion'
    if partial:
        builder = _source_builder()
        specs = dict(zip(('target', 'occluder'), builder.build_env_objects(source.payload)))
    else:
        from .perceptual_grounding_view_hallucination import source_builder
        builder = source_builder()
        specs = {'target': builder.build_env_objects(source.payload)[0]}
    camera_position, camera_quaternion, reference = builder.initial_camera(source.payload)
    recipe = context.config.get('grounding_recipe', 'viewpoint')
    if recipe not in ('viewpoint', 'move_occluder'):
        raise ValueError('Use an explicit viewpoint or move_occluder grounding recipe.')
    resting = context.config.get('occluder_resting_support')
    if recipe == 'move_occluder':
        if (not partial or not isinstance(resting, dict) or not resting.get('public_label')
                or not resting.get('binding', {}).get('name') or resting.get('relation') != 'on_top'):
            raise ValueError('Occluder manipulation requires an explicit bound named resting-support contract.')
        if resting['binding']['name'] in {value['name'] for value in bindings.values()}:
            raise ValueError('The occluder resting support must be distinct from original actors/support and destination.')
        bindings['occluder_rest'] = _binding(resting['binding'])
        scene['bindings'] = deepcopy(bindings)
    floor_support = bindings['support'].get('category') in ('floor', 'floors', 'wall', 'walls', 'ceiling', 'ceilings')
    group = ['target', 'occluder'] if partial else ['target']
    # A non-task furniture support remains an external reference. In particular,
    # a floor/wall is never admitted as a movable dependency-group actor.
    relocation = 'rigid_group' if partial or floor_support else 'preserve'
    contract = {'version': GROUNDING_REVISION, 'source_identity': source.relative_path,
        'recipe': recipe, 'source_category': source.small_task, 'dependency_group': group,
        'source_actors': {key: {field: deepcopy(value[field]) for field in
            ('name', 'category', 'model', 'position', 'orientation', 'scale') if field in value}
            for key, value in specs.items()},
        'source_reference': {'position_m': np.asarray(camera_position, dtype=float).tolist(),
                             'quaternion_xyzw': np.asarray(camera_quaternion, dtype=float).tolist(),
                             'selection': deepcopy(reference)},
        'source_camera_translation_policy': 'translate_with_group',
        'external_support_binding': deepcopy(bindings['support']),
        'external_support_preserved': True, 'floor_background_context_adapted': not partial and floor_support,
        'recognition_difficulty': 'unavailable_until_all_eight_and_initial_global_are_audited',
        'initial_geometry_translation_only': True, 'target_fixed_base_override_permitted': False,
        'position_tolerance_m': .02, 'orientation_tolerance_deg': 2.,
        'native_material_appearance_review_required': True, 'occluder_motion_permitted': recipe == 'move_occluder',
        'inspection_only_alternative': 'Inspect and report the original candidate without retrieval if native grasp/access '
            'cannot be qualified; this is a separately declared alternative and cannot count as retrieval success.'}
    if recipe == 'move_occluder':
        contract['resting_support_contract'] = deepcopy(resting)
        contract['resting_support_contract'].update(occluder_grasp_verified=False,
            swept_space_verified=False, support_fit_verified=False, public_control_access_verified=False)
    contract = json.loads(json.dumps(contract, default=lambda value: value.tolist(), allow_nan=False))
    p.update(grounding_revision=GROUNDING_REVISION, grounding_contract=contract,
        requires_native_visual_discovery=True, native_visual_adapter='task_visual_evidence',
        identity_before_target_grasp=True,
        completion_item_ids=['informative_inspection', 'identity', 'correct_bound_selection', 'stable_placement']
            + (['safe_occluder_resting'] if recipe == 'move_occluder' else []))
    evaluation.update(module=MODULE, operation=GROUNDING_OPERATION, binding_keys=list(bindings))
    choices = ' or '.join(p['identity_choices'])
    label = p['destination_public_label']
    target_description = ('the candidate behind the foreground blocker' if partial else
                          'the candidate at its original support whose shape may look different across views')
    inspection = ('Change viewpoint using a public base translation or torso-height change until you see a new '
        'diagnostic feature; keep the blocker fixed. ' if recipe == 'viewpoint' else
        'Using the native gripper, move only the original foreground blocker to ' + resting['public_label'] +
        ', leave it stably supported there, and inspect a newly revealed diagnostic feature. ')
    if not partial:
        inspection = inspection.replace('keep the blocker fixed', 'preserve the candidate pose and appearance')
    task['public'] = {'objective': 'Inspect ' + target_description + '. ' + inspection
        + 'Determine whether the candidate is ' + choices + '. Report its category as ANSWER: <category> '
        + 'before grasping it. Then grasp that same original candidate and release it '
        + ('inside ' if p['placement_relation'] == 'inside' else 'on ') + label + '.',
        'success_criteria': 'Complete informative inspection, report the correct identity before target grasp, '
            'select the same bound original candidate, and leave it released and supported at the destination '
            'for 15 consecutive native samples. The final placement must still hold when you finish.'
            + (' The permitted blocker must remain safely released at its named resting support.' if recipe == 'move_occluder' else ''),
        'environment': 'All eight mounted cameras and the frozen initial global image are available. '
            'The source object identity, dimensions, material and appearance-producing orientation are preserved; '
            'the declared complete task-prop group may be translated together onto a measured support.',
        'constraints': ['Preserve target pose/material until informative identification. Never substitute another object.',
            'Do not move the original blocker.' if recipe == 'viewpoint' and partial else
                'Move only the specifically permitted blocker to its distinct named resting support.' if partial else
                'Preserve original external furniture and the building floor.',
            'No minimum excursion, lift threshold or return-to-start trip is required.']}
    result['interaction_policy'] = {'mode': 'manipulation', 'group_bindings': group,
        'manipulation_bindings': ['target'] + (['occluder'] if recipe == 'move_occluder' else []),
        'relocation_policy': relocation, 'allow_fallback_table': relocation != 'preserve',
        'support_search': 'measured_surfaces', 'source_camera_policy': 'translate_with_group',
        'source_region': source.raw['room'],
        'warning_codes': ['grounding_initial_all_inputs_and_native_diagnostic_features_pending']}
    task['version'] = task.get('version', 1) + 1
    scene['version'] = scene.get('version', 1) + 1
    task.setdefault('repair', {})['grounding_contract'] = deepcopy(contract)
    result.update(grounding_revision=GROUNDING_REVISION, grounding_contract=deepcopy(contract),
                  operation_recipe=GROUNDING_OPERATION)
    result['warning_codes'] = sorted(set(result.get('warning_codes', [])) | {
        'grounding_all_inputs_difficulty_audit_pending', 'native_same_object_retrieval_pending'})
    result.setdefault('change_summary', []).append(
        'Require a successful public informative inspection with an exact new diagnostic feature, then identification before original-object retrieval.')
    return result


def grounding_source_geometry(session, config, bound):
    """Share the same initial common-translation contract across qualify/evaluate."""
    contract = config['parameters']['grounding_contract']
    checks, geometry, deltas = {}, {}, []
    for key in contract['dependency_group']:
        obj, expected = bound[key], contract['source_actors'][key]
        if obj.category in ('floor', 'floors', 'wall', 'walls', 'ceiling', 'ceilings'):
            raise ValueError('Building geometry cannot be a movable grounding dependency.')
        position, quaternion = map(_array, obj.get_position_orientation())
        delta = np.asarray(position, dtype=float) - expected['position']
        deltas.append(delta)
        checks['identity:' + key] = all(getattr(obj, name, None) == value for name, value in expected.items()
            if name in ('name', 'category', 'model'))
        checks['orientation:' + key] = _quaternion_angle(quaternion, expected['orientation']) <= contract['orientation_tolerance_deg']
        checks['scale:' + key] = bool(np.allclose(_array(obj.scale), expected.get('scale', [1., 1., 1.]), atol=1e-5))
        geometry[key] = {'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
                         'source_translation_m': delta.tolist()}
    delta = deltas[0]
    checks['common_group_translation'] = all(np.linalg.norm(value - delta) <= contract['position_tolerance_m'] for value in deltas)
    setup = getattr(session, 'category_interaction_setup', {})
    camera_delta = setup.get('source_reference_camera_translation_m')
    if camera_delta is None and np.linalg.norm(delta) <= contract['position_tolerance_m']:
        camera_delta = [0., 0., 0.]
    checks['source_reference_translates_equally'] = bool(camera_delta is not None and np.linalg.norm(
        np.asarray(camera_delta) - delta) <= contract['position_tolerance_m'])
    checks['source_reference_current_position'] = bool(np.linalg.norm(np.asarray(session.scene_camera_position)
        - np.asarray(contract['source_reference']['position_m']) - delta) <= contract['position_tolerance_m'])
    checks['source_reference_orientation_preserved'] = _quaternion_angle(session.scene_camera_quaternion,
        contract['source_reference']['quaternion_xyzw']) <= contract['orientation_tolerance_deg']
    return checks, geometry


def qualify_grounding(session, executor, config, bound):
    """Check common translation and native feasibility without changing the world."""
    contract = config['parameters']['grounding_contract']
    checks, geometry = {}, {}
    try:
        checks.update(_robot_safety(session))
        source_checks, geometry = grounding_source_geometry(session, config, bound)
        checks.update(source_checks)
        checks['target_native_movable_collidable'] = not bound['target'].fixed_base and not bound['target'].visual_only
        checks['native_gripper_capability'] = bool(executor.capabilities.gripper and executor.capabilities.ee_translation)
        checks['native_visual_collector_installed'] = callable(getattr(session, 'task_visual_evidence', None))
        receipt = getattr(session, 'grounding_qualification', {}).get(contract['source_identity'], {})
        checks['current_episode_feasibility_receipt'] = (getattr(session, 'episode_id', None) is not None
            and receipt.get('episode_id') == session.episode_id
            and receipt.get('reset_id') == getattr(session, 'native_reset_id', None))
        for name in ('source_material_appearance_verified', 'initial_all_inputs_audited',
                     'informative_public_inspection_feasible', 'same_object_retrieval_feasible'):
            checks[name] = receipt.get(name) is True
        if contract['recipe'] == 'move_occluder':
            checks['occluder_native_movable_collidable'] = not bound['occluder'].fixed_base and not bound['occluder'].visual_only
            for name in ('occluder_grasp_verified', 'swept_space_verified', 'resting_support_fit_verified',
                         'resting_support_public_access_verified'):
                checks[name] = receipt.get(name) is True
        accepted = all(checks.values())
        return {'accepted': accepted, 'checks': checks, 'native_group_geometry': geometry,
            'evaluation_validity': 'complete' if accepted else 'unavailable',
            'inspection_only_alternative': contract['inspection_only_alternative'],
            'reason': None if accepted else 'grounding_native_geometry_or_feasibility_unavailable'}
    except Exception as error:
        return {'accepted': False, 'checks': checks, 'native_group_geometry': geometry,
                'evaluation_validity': 'unavailable', 'reason': f'{type(error).__name__}: {error}'}


def _quaternion_angle(first, second):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 0:
        raise ValueError('Native/source orientation is unavailable.')
    return math.degrees(2 * math.acos(float(np.clip(abs(first @ second) / denominator, 0., 1.))))


@lru_cache(maxsize=1)
def _source_builder():
    """Read the official pure constructor without importing the simulator."""
    path = ESI_ROOT / 'src/active_explore/tasks/perceptual_grounding/partial_occlusion.py'
    spec = importlib.util.spec_from_file_location('_esi_partial_occlusion_converter_source', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _public_question(raw):
    question = str(raw.get('question') or '').strip()
    value = raw.get('options_json')
    choices = json.loads(value) if isinstance(value, str) else value
    if not choices and ':' in question:
        choices = re.split(r'\s+or\s+', question.split(':', 1)[1].rstrip('?'), flags=re.IGNORECASE)
    return question, [str(value).strip() for value in choices or [] if str(value).strip()]


def _binding(spec):
    return {field: spec[field] for field in ('name', 'category', 'model') if spec.get(field)}


def _bounds(obj):
    low, high = (np.asarray(obj[key], dtype=float) for key in ('aabb_min', 'aabb_max'))
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(high <= low):
        raise ValueError('Original object bounds must be finite with positive extents.')
    return low, high


def _operation(objects):
    """Retrieve the original target; retain size evidence without forcing a patrol."""
    extents = {key: (_bounds(objects[key])[1] - _bounds(objects[key])[0]).tolist()
               for key in ('target', 'occluder')}
    small = {key: min(value) <= .10 and max(value) <= .60 for key, value in extents.items()}
    legacy_recipe = ('uncover_retrieve' if small['occluder'] else 'retrieve') if small['target'] else 'inspection_route'
    return 'retrieve', {'original_aabb_extents_m': extents, 'small_object_screen': small,
        'legacy_dimension_selected_recipe': legacy_recipe,
        'maximum_extent_m': .60, 'maximum_minimum_extent_m': .10,
        'basis': 'Saved original AABBs; existing native R1 receipts report 0.10 m finger opening. '
                 'Large bounds do not establish a grasp or rule out handles; native aperture and IK '
                 'must qualify original-target retrieval independently.',
        'native_grasp_certificate': False}


def _source_stance(objects, view):
    """Source geometry supplies a near-task guess; native maps/contacts decide."""
    boxes = [_bounds(objects[key]) for key in ('target', 'occluder')]
    centre = sum((low + high) / 2 for low, high in boxes) / 2
    direction = np.asarray(view['position'], float)[:2] - centre[:2]
    angle = math.atan2(direction[1], direction[0])
    radius = max(float(np.linalg.norm(np.maximum(abs(low[:2] - centre[:2]), abs(high[:2] - centre[:2]))))
                 for low, high in boxes) + .65
    support_low, support_high = _bounds(objects['support'])
    floor = objects['support'].get('category') in ('floor', 'floors')
    positions = []
    for offset in (0., 45., -45., 90., -90., 135., -135., 180.):
        theta = angle + math.radians(offset)
        xy = centre[:2] + radius * np.array([math.cos(theta), math.sin(theta)])
        clear = all(np.linalg.norm(np.maximum(np.maximum(low[:2] - xy, xy - high[:2]), 0.)) >= .55
                    for low, high in boxes)
        inside = not floor or bool(np.all(xy >= support_low[:2] + .55) and np.all(xy <= support_high[:2] - .55))
        positions.append((clear and inside, xy))
    if floor and not any(passed for passed, _ in positions) and np.all(support_high[:2] - support_low[:2] >= 1.10):
        # A circular orbit can miss valid floor cells in narrow rooms. Check a
        # small deterministic source-floor grid before retaining an inconclusive
        # guess. This still cannot certify furniture or polygonal room clearance.
        grid = [np.array([x, y]) for x in np.linspace(support_low[0] + .55, support_high[0] - .55, 5)
                for y in np.linspace(support_low[1] + .55, support_high[1] - .55, 5)]
        grid.sort(key=lambda xy: float(np.linalg.norm(xy - positions[0][1])))
        for xy in grid:
            clear = all(np.linalg.norm(np.maximum(np.maximum(low[:2] - xy, xy - high[:2]), 0.)) >= .55
                        for low, high in boxes)
            positions.append((clear, xy))
    passed, xy = next((row for row in positions if row[0]), positions[0])
    floor_z = float(support_high[2]) if floor else 0.
    yaw = math.degrees(math.atan2(centre[1] - xy[1], centre[0] - xy[0]))
    return {'operation': 'robot_pose', 'phase': 'before_construction',
            'position': [float(xy[0]), float(xy[1]), floor_z + .005], 'yaw_deg': yaw,
            'head_pitch_deg': 45.}, {'source_pair_floor_screen_passed': bool(passed),
            'source_floor_z_m': floor_z, 'floor_height_inferred_for_table_source': not floor,
            'source_reference_used_for_bearing_only': True, 'native_floor_contact_verified': False}


class PartialOcclusionConverter(ConverterBase):
    category = ('Perceptual Grounding', 'Partial Occlusion')
    version = 'partial_occlusion_v3.0'

    def inspect(self, source):
        if (source.big_task, source.small_task) != self.category:
            return {'eligible': False, 'reason': 'source_category_mismatch'}
        question, choices = _public_question(source.raw)
        if not question or len(set(choices)) < 2:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2:missing_public_question_or_choices'}
        objects = map_records(source.payload.get('objects'))
        if any(not objects.get(key, {}).get('name') for key in ('target', 'occluder', 'support')):
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2:missing_original_binding'}
        raw, payload = source.raw, source.payload
        if any(payload.get(key) is not None and payload[key] != raw.get(key) for key in ('scene', 'room')):
            return {'eligible': False, 'reason': 'source_scene_room_disagree'}
        if payload.get('resolved_room') is not None and payload['resolved_room'] != raw.get('room'):
            return {'eligible': False, 'reason': 'source_resolved_room_disagree'}
        native = _source_builder().build_env_objects(payload)
        position, quaternion, _ = _source_builder().initial_camera(payload)
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all() or not np.linalg.norm(quaternion):
            raise ValueError('Invalid original reference camera.')
        if len({spec['name'] for spec in native} | {objects['support']['name']}) != 3:
            raise ValueError('Original role bindings must identify distinct objects.')
        for key in ('target', 'occluder', 'support'):
            _bounds(objects[key])
        return {'eligible': True, 'reason': 'original_pair_reference_and_physical_recipes_available'}

    def convert_all(self, sources, context):
        """Use the shared per-source exception, revision, QA and YAML writer."""
        from wia.authoring.io import write_json
        manifest = super().convert_all(list(sources), context)
        manifest.update(rule_revision=RULE_REVISION, feedback_revision=FEEDBACK_REVISION,
            operation_counts=dict(Counter(row.get('operation_recipe', 'diagnostic') for row in manifest['records'])),
            failure_distribution=dict(Counter(row.get('failure_reason') for row in manifest['records']
                                              if row['status'] != 'generated')))
        write_json(context.output_root / 'batch_manifest.json', manifest)
        return manifest

    def sample(self, manifest, policy=None):
        """Prioritize failures, human feedback, recipes, and rare table supports."""
        policy = policy or {}
        count = min(policy.get('count', 5), len(manifest['records']))
        rows = sorted(manifest['records'], key=lambda row: row['candidate_id'])
        groups = [[r for r in rows if r['status'] != 'generated'],
                  [r for r in rows if r['source_identity'] == FEEDBACK_SOURCE],
                  [r for r in rows if 'historical_native_load_failure' in r.get('warning_codes', [])]]
        groups += [[r for r in rows if r.get('operation_recipe') == recipe] for recipe in ('uncover_retrieve', 'retrieve')]
        groups += [[r for r in rows if r.get('source_layout_evidence', {}).get('support_category') not in (None, 'floors')]]
        selected = []
        for group in groups:
            choice = next((r['candidate_id'] for r in group if r['candidate_id'] not in selected), None)
            if choice is not None and len(selected) < count:
                selected.append(choice)
        for choice in super().sample(manifest, policy):
            if choice not in selected and len(selected) < count:
                selected.append(choice)
        return selected

    def convert(self, source, context):
        """Preserve the ambiguous source scene and require an explicit destination."""
        if context.config.get('legacy_partial_occlusion'):
            result = self.convert_legacy(source, context)
            result['rule_revision'] = LEGACY_RULE_REVISION
            return result
        question, choices = _public_question(source.raw)
        objects = map_records(source.payload['objects'])
        native = _source_builder().build_env_objects(source.payload)
        _, _, reference = _source_builder().initial_camera(source.payload)
        bindings = {key: _binding(spec) for key, spec in zip(('target', 'occluder'), native)}
        bindings['support'] = _binding(objects['support'])
        answer = _normalize_answer(source.raw.get('answer'))
        if answer not in {_normalize_answer(choice) for choice in choices}:
            raise ValueError('Source answer is not one of the declared category alternatives.')
        if answer != _normalize_answer(bindings['target']['category']):
            raise ValueError('Source answer disagrees with the original target category.')
        contract = {
            'preserved_relations': ['original target identity and scale', 'target/occluder/support poses',
                'source reference camera and external room geometry'],
            'initialization_relocation_policy': 'preserve',
            'required_native_evidence': ['all eight initial robot camera views',
                'reachable informative viewpoint and diagnostic feature visibility',
                'native source support and unchanged mechanism geometry',
                'destination interior geometry, fit and access separately qualified', 'same bound original target manipulation',
                'released native destination relation and consecutive stability'],
            'recognition_difficulty': 'pending_all_camera_review',
            'native_operation_verified': False}
        destination = context.config.get('placement_destination')
        if not destination:
            return {'scene_definition': original_scene(source, context, bindings=bindings),
                'task_definition': None, 'status': 'needs_human_decision', 'reuse_level': 'L1',
                'failure_stage': 'destination_selection', 'failure_reason': 'placement_destination_pending',
                'operation_recipe': IDENTIFY_PLACE, 'rule_revision': RULE_REVISION,
                'task_revision': TASK_REVISION, 'mechanism_contract': contract,
                'warning_codes': ['placement_destination_pending', 'native_operation_pending'],
                'change_summary': ['Preserve the source geometry while awaiting a named destination.'],
                'limitations': ['A named destination with measured fit and access is required before task generation.'],
                'next_step': 'Supply the main-selected seeded placement_destination and qualify it natively.'}
        relation = destination['relation']
        label = str(destination.get('public_label') or '').strip()
        if relation not in ('on_top', 'inside') or not label or not destination.get('binding', {}).get('name'):
            raise ValueError('Destination requires binding, public_label and on_top or inside relation.')
        bindings['destination'] = _binding(destination['binding'])
        if bindings['destination']['name'] in (bindings['target']['name'], bindings['occluder']['name']):
            raise ValueError('The destination cannot be the target or its original occluder.')
        _, dimensions = _operation(objects)
        pose, stance = _source_stance(objects, reference['view'])
        patches = [pose]
        for key in ('target', 'occluder'):
            patches.extend({'operation': 'object_physics', 'phase': phase, 'binding': key, 'visual_only': False}
                           for phase in ('before_construction', 'after_postprocessing'))
        preposition = 'on' if relation == 'on_top' else 'inside'
        public = {
            'objective': 'Inspect the partly hidden candidate behind the original foreground blocker. '
                + 'Determine whether it is ' + ' or '.join(choices) + ', report its category, '
                + f'and place that same candidate {preposition} {label}. '
                + 'Report the category in world_observation as ANSWER: <category>.',
            'success_criteria': 'Report the correct category, select the same original candidate with the gripper, '
                + f'and release it stably {preposition} {label}. The object must remain released and supported '
                + 'for 15 consecutive physics steps; identification and placement are scored separately.',
            'environment': 'The original partly occluded candidate, its foreground blocker and support remain in place at reset.',
            'constraints': ['Track the same candidate across the enabled robot cameras. Category alternatives: '
                + '; '.join(choices) + '.',
                'Use the existing guarded robot controls and mounted cameras; preserve original object sizes.',
                'No minimum lift, travel, return or ordering of the completion items is required.']}
        parameters = {'task_revision': TASK_REVISION, 'mechanism_contract': contract,
            'identity_choices': choices, 'expected_identity': answer,
            'candidate_label': 'the partly hidden candidate behind the original foreground blocker',
            'placement_relation': relation, 'destination_public_label': label,
            'destination_selection_seed': context.seed,
            'release_stability_steps': 15, 'maximum_release_speed_m_s': .02,
            'maximum_release_angular_speed_rad_s': .10,
            'completion_item_ids': ['identity', 'correct_bound_selection', 'stable_placement']}
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': IDENTIFY_PLACE,
            'binding_keys': list(bindings), 'parameters': parameters}
        interaction = {'mode': 'manipulation', 'group_bindings': ['target', 'occluder'],
            'manipulation_bindings': ['target'], 'relocation_policy': 'preserve',
            'allow_fallback_table': False, 'source_region': source.raw['room'],
            'warning_codes': ['source_mechanism_native_review_pending', 'native_grasp_ik_and_aperture_pending']}
        result = {'scene_definition': original_scene(source, context, bindings=bindings, patches=patches),
            'task_definition': physical_task(source, context, public=public, evaluation=evaluation,
                bindings=bindings, patches=patches,
                qualification={'on_failure': 'skip', 'initial_visibility': 'none', 'target_movable_collidable': True},
                capabilities=['base_translation', 'base_rotation', 'ee_translation', 'ee_rotation', 'gripper', 'trunk_height']),
            'status': 'generated', 'reuse_level': 'L2', 'operation_recipe': IDENTIFY_PLACE,
            'rule_revision': RULE_REVISION, 'interaction_policy': interaction,
            'warning_codes': ['native_destination_fit_access_pending', 'initial_all_camera_ambiguity_pending',
                'informative_viewpoint_pending', 'physical_qualification_pending'],
            'change_summary': ['Require explicit category reporting, original candidate selection and stable destination placement.',
                'Preserve source objects, scales, occluder, support and original reference geometry.'],
            'limitations': ['CPU generation does not certify occlusion, recognition, destination access or executed manipulation.',
                'The source-derived initial robot stance requires shared native qualification.',
                'Native Inside and Touching are relation measurements; destination interior geometry, '
                'fit and access require separate qualification.'],
            'source_reference': reference,
            'source_layout_evidence': {'operation_selection': dimensions, 'robot_stance': stance,
                'source_objects': objects, 'original_question': question},
            'next_step': 'Review repaired candidate views and qualify public-control placement before Core release.'}
        return revise_grounding_result(result, source, context) if context.config.get('task_edit2_grounding') else result

    def convert_legacy(self, source, context):
        question, choices = _public_question(source.raw)
        objects = map_records(source.payload['objects'])
        native = _source_builder().build_env_objects(source.payload)
        _, _, reference = _source_builder().initial_camera(source.payload)
        bindings = {key: _binding(spec) for key, spec in zip(('target', 'occluder'), native)}
        bindings['support'] = _binding(objects['support'])
        recipe, recipe_evidence = _operation(objects)
        pose, stance_evidence = _source_stance(objects, reference['view'])
        patches = [pose]
        # The current native pipeline already constructs installed R1; this
        # category is not robot-free and needs no add_existing_r1 patch.
        for key in ('target', 'occluder'):
            patches += [{'operation': 'object_physics', 'phase': 'before_construction', 'binding': key, 'visual_only': False},
                        {'operation': 'object_physics', 'phase': 'after_postprocessing', 'binding': key, 'visual_only': False}]
        public = {'objective': '', 'success_criteria': '',
            'environment': 'An indoor scene containing an obscured item and an original foreground blocker.',
            'constraints': ['Track the same obscured item across robot observations. Original identity question: '
                            + question + ' Public choices: ' + '; '.join(choices) + '.',
                            'Use the existing guarded robot controls and enabled robot cameras.',
                            'Keep original object sizes and use physical motion, grasp and release.']}
        qualification = {'on_failure': 'skip', 'initial_visibility': 'none'}
        if recipe == 'retrieve':
            public.update(objective='Approach and retrieve the partially obscured item described by the original question.',
                          success_criteria='Grasp that item and hold it at least 0.05 m above its initial bottom height.')
            evaluation = {'type': 'grasp_lift', 'target': 'target', 'minimum_lift_m': .05}
            qualification.update(target_movable_collidable=True, support='support')
        else:
            parameters = ({'minimum_lift_m': .05, 'minimum_clearance_m': .03, 'maximum_release_speed_m_s': .02,
                           'release_stability_steps': 15}
                          if recipe == 'uncover_retrieve' else
                          {'minimum_travel_m': .50, 'minimum_viewpoint_separation_deg': 45.,
                           'maximum_surface_distance_m': 1.50, 'maximum_object_drift_m': .05,
                           'maximum_object_rotation_deg': 5., 'maximum_step_translation_m': .15})
            evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': recipe,
                          'binding_keys': ['target', 'occluder', 'support'], 'parameters': parameters}
            if recipe == 'uncover_retrieve':
                public.update(objective='Uncover and retrieve the partially obscured item: grasp and lift the foreground blocker, '
                              'set it aside on the same support, then grasp the originally hidden item.',
                              success_criteria='Lift the blocker at least 0.05 m, release it on support for 15 consecutive physics steps with at least 0.03 m horizontal '
                              'clearance from the item, then hold the item at least 0.05 m above its initial bottom height.')
            else:
                public.update(objective='Drive around the original blocker to inspect the obscured item from two distinct sides. '
                              'Finish stopped with the item in an enabled robot camera.',
                              success_criteria='Visit two unobstructed camera viewpoints at least 45 degrees apart, travel at least '
                              '0.50 m, and finish within 1.50 m of the item footprint. Preserve the item and blocker poses.',
                              constraints=public['constraints'] + ['Leave both original task objects in place. The identity answer is auxiliary.'])
        manipulation = ['target', 'occluder'] if recipe == 'uncover_retrieve' else ['target'] if recipe == 'retrieve' else []
        interaction = {'mode': 'manipulation' if manipulation else 'navigation',
            'group_bindings': ['target', 'occluder'], 'manipulation_bindings': manipulation,
            'relocation_policy': 'rigid_group' if manipulation else 'preserve',
            'support_search': 'measured_surfaces', 'source_region': source.raw['room'],
            'allow_fallback_table': True,
            'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10, 'maximum_stance_distance_m': .85,
            'warning_codes': ['native_grasp_ik_and_aperture_pending'] if manipulation else ['native_inspection_route_pending']}
        visibility = reference['view'].get('visibility', {})
        warnings = ['physical_qualification_pending', 'assumption_review_required', 'source_derived_robot_stance_requires_native_screen']
        if not recipe_evidence['small_object_screen']['target']:
            warnings.append('original_target_aperture_and_ik_unverified')
        for key in ('target', 'occluder'):
            if visibility.get(bindings[key]['name']) is False:
                warnings.append(f'source_reference_{key}_not_visible')
        if bindings['target']['category'] == bindings['occluder']['category']:
            warnings.append('target_and_occluder_share_category')
        if not stance_evidence['source_pair_floor_screen_passed']:
            warnings.append('source_pair_floor_stance_screen_inconclusive')
        if recipe == 'inspection_route':
            warnings.append('geometric_visibility_does_not_certify_recognition')
        if source.relative_path.endswith('Merom_0_int/closet_0/Merom_0_int__closet_0_4.json'):
            warnings.append('historical_native_load_failure')
        assumptions = [
            {'doubt': 'Source AABBs and existing finger opening do not establish handles, reach or executed graspability.',
             'chosen_assumption': 'Keep original-target retrieval as the goal, without a mandatory survey, '
                 'return trip or blocker lift. Retain original-size aperture and IK uncertainty explicitly.',
             'alternatives': ['Native handle qualification may permit original-size retrieval for a large item.',
                              'Mark unsupported instances not_convertible_under_L1_L2 after native diagnosis.'],
             'scope': source.relative_path, 'risk': 'The unchanged original target may have no feasible grasp; CPU conversion does not qualify aperture, IK or executed retrieval.'},
            {'doubt': 'Source camera poses may intersect furniture and are not robot floor poses; source floor edges omit other obstacles.',
             'chosen_assumption': 'Use source viewing bearing and task bounds for an initial R1 guess, then require native same-room wheel support and collision screening.',
             'alternatives': ['Use a measured same-room stance supplied by native initialization.', 'Reject unsafe starts and retain diagnostic images.'],
             'scope': source.relative_path, 'risk': 'No safe room stance or useful camera path may exist. Table sources initially assume floor z=0 before native correction.'},
            {'doubt': 'Small floor objects may be unreachable and source camera recognition differs from embodied cameras.',
             'chosen_assumption': 'Allow only a common target/blocker group translation to a measured existing support, or a user-authorized small simple support if needed; translate the reference equally.',
             'alternatives': ['Keep the original support if qualified.', 'Retain a pending failure if the unchanged group cannot fit a safe measured support.'],
             'scope': source.relative_path, 'risk': 'Relocation preserves pairwise geometry but changes room context and reference-camera relationships to furniture.'}]
        return {'scene_definition': original_scene(source, context, bindings=bindings, patches=patches),
            'task_definition': physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings,
                patches=patches, qualification=qualification,
                capabilities=['base_translation', 'base_rotation', 'ee_translation', 'ee_rotation', 'gripper', 'trunk_height']
                             if manipulation else ['base_translation', 'base_rotation']),
            'status': 'generated', 'reuse_level': 'L2', 'operation_recipe': recipe,
            'rule_revision': RULE_REVISION, 'feedback_revision': FEEDBACK_REVISION, 'interaction_policy': interaction,
            'change_summary': ['Preserve empty original variant, source assets/scales/answers and original private reference.',
                'Require a genuine grasp and 0.05 m lift of the original target, without arbitrary survey or return requirements.',
                'Initialize installed R1 near the source pair; native floor contacts and room screening remain mandatory.',
                'Enable original task-object collisions before construction and after postprocessing; retain original fixed_base settings.'],
            'warning_codes': warnings, 'assumptions': assumptions,
            'limitations': ['Pending conversion is not native loading, physical qualification, recognition or human approval.',
                'Native camera frustum and first-hit rays measure geometric observation opportunity, not semantic recognition.',
                'Historical transform failures and missing historical images are not failures of this unrun revision.',
                'Preserve existing R1/source assets and sizes; only a user-authorized simple support may be added. No runtime teleporting or answer-based operation success.'],
            'attempted_repairs': [RULE_REVISION],
            'next_step': 'Main should freeze/capture this revision, retain all failures in Review, and qualify guarded execution independently.',
            'source_reference': {'selection': reference['selection'], 'view_index': reference['view_index'],
                                 'original_visibility_metadata': visibility},
            'source_layout_evidence': {'native_object_count': len(native), 'support_category': bindings['support'].get('category'),
                'scale_policy': 'Unchanged source builder: applied_scale precedes scale.',
                'operation_selection': recipe_evidence, 'robot_stance': stance_evidence}}


def _array(value):
    from wia.tasks.interaction_setup import array
    return array(value)


def _robot_safety(session):
    """Read current room and contacts; never place or move the robot here."""
    import torch
    from wia.tasks.interaction_setup import _objects, _robot_contact_check
    robot, scene = session.robot, session.env.scene
    floors = [obj for obj in _objects(scene) if obj.category in ('floors', 'floor')]
    position = _array(robot.get_position_orientation()[0])
    floor_z = float(scene.trav_map.floor_heights[0])
    checks = _robot_contact_check(robot, [obj.prim_path for obj in floors], floor_z)
    checks['same_source_room'] = scene.seg_map.get_room_instance_by_point(
        torch.tensor(position[:2], dtype=torch.float32)) == session.room_name
    checks['native_r1'] = type(robot).__name__ == 'R1' and not robot.visual_only
    checks['stopped'] = (float(np.linalg.norm(_array(robot.get_linear_velocity()))) <= .02
                         and float(np.linalg.norm(_array(robot.get_angular_velocity()))) <= .10)
    return checks


def qualify_operation(session, executor, config, bound):
    """Only measured native preconditions; initialization acceptance is separate."""
    if config['operation'] == VISUAL_DISCOVERY:
        return qualify_visual_discovery(session, config, bound)
    if config['operation'] == GROUNDING_OPERATION:
        return qualify_grounding(session, executor, config, bound)
    try:
        recipe = config['operation']
        checks = _robot_safety(session)
        required = ['floor_supported', 'collision_free', 'same_source_room', 'native_r1']
        checks['scene_matches'] = (session.scene_name, session.room_name) == (config['scene'], config['room'])
        checks['bindings_match'] = all(bound[key].name == config['bindings'][key] for key in config['binding_keys'])
        checks['target_pair_collidable'] = all(not bound[key].visual_only for key in ('target', 'occluder') if key in bound)
        required += ['scene_matches', 'bindings_match', 'target_pair_collidable']
        if recipe == IDENTIFY_PLACE:
            checks['movable_target'] = not bound['target'].fixed_base
            checks['destination_collidable'] = not bound['destination'].visual_only
            checks['gripper_capability'] = bool(executor.capabilities.gripper and executor.capabilities.ee_translation)
            required += ['movable_target', 'destination_collidable', 'gripper_capability']
            controller = getattr(executor, 'controller', None)
            if controller is None and callable(getattr(executor, 'debug_gripper_state', None)):
                controller = executor
            if controller is None:
                raise ValueError('Native action executor controller is unavailable.')
            measured = _placement_measurements(session, controller, config, bound['target'], bound['destination'])
            checks['placement_telemetry_available'] = not measured['measurement_errors']
            required.append('placement_telemetry_available')
            checks['native_destination_fit_access_verified'] = False
            checks['destination_interior_geometry_verified'] = False
            checks['source_mechanism_review_verified'] = False
        elif recipe == 'uncover_retrieve':
            from wia.tasks.interaction_setup import touching_task_support, on_top_task_support
            checks['movable_pair'] = all(not bound[key].fixed_base for key in ('target', 'occluder'))
            checks['supported_pair'] = all(touching_task_support(session, bound[key], bound['support'])
                and on_top_task_support(session, bound[key], bound['support']) for key in ('target', 'occluder'))
            checks['gripper_capability'] = bool(executor.capabilities.gripper and executor.capabilities.ee_translation)
            required += ['movable_pair', 'supported_pair', 'gripper_capability']
        elif recipe == 'inspection_route':
            checks['base_capability'] = bool(executor.capabilities.base_translation and executor.capabilities.base_rotation)
            view = _camera_opportunity(session, bound['target'])
            checks['enabled_native_camera'] = view['camera_available']
            checks['initial_geometric_view'] = view
            required += ['base_capability', 'enabled_native_camera']
        else:
            raise ValueError('Unsupported Partial Occlusion operation.')
        failed = [key for key in required if not checks[key]]
        return {'accepted': not failed, 'reason': None if not failed else 'native_operation_precondition_failed:' + ','.join(failed),
                'checks': checks, 'executed_operation_verified': False}
    except (ValueError, KeyError, AttributeError, RuntimeError, AssertionError) as error:
        return {'accepted': False, 'reason': f'native_operation_measurement_unavailable:{type(error).__name__}:{error}'}


def _camera_opportunity(session, target):
    """Read enabled robot camera calibration and native first-hit rays only.

    No modalities, sensor poses, physics steps or renders are changed here.
    A hit must identify the native target body inside the measured frustum.
    """
    from core.camera_geometry import pose_matrix
    from omnigibson.utils.sampling_utils import raytest
    low, high = (_array(value) for value in target.aabb)
    centre = (low + high) / 2
    points = [centre] + [np.where(np.arange(3) == axis, end[axis], centre)
                         for axis in range(3) for end in (low, high)]
    registry = getattr(session, 'robot_camera_registry', {}) or {'agentview_rgb': session.camera_name}
    settings = session.agent_observation_config
    result = {'camera_available': False, 'unobstructed_target_ray': False, 'views': {}}
    for name, sensor_name in registry.items():
        enabled = (settings.head_rgb if name == 'agentview_rgb' else settings.wrist_rgb if name == 'wrist_rgb'
                   else settings.additional_robot_rgb)
        if not enabled:
            continue
        sensor = session.robot.sensors[sensor_name]
        transform = pose_matrix(*[_array(value) for value in sensor.get_position_orientation()])
        intrinsic = _array(sensor.intrinsic_matrix)
        if intrinsic.shape != (3, 3) or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ValueError('Measured robot camera calibration is unavailable.')
        result['camera_available'] = True
        visible = 0
        for point in points:
            local = transform[:3, :3].T @ (point - transform[:3, 3]) * [1., -1., -1.]
            if local[2] <= .01:
                continue
            pixel = intrinsic @ local
            u, v = pixel[:2] / pixel[2]
            if not (0 <= u < sensor.image_width and 0 <= v < sensor.image_height):
                continue
            hit = raytest(transform[:3, 3].tolist(), point.tolist())
            body = str(hit.get('rigidBody', ''))
            visible += bool(hit.get('hit') and (body == target.prim_path or body.startswith(target.prim_path + '/')))
        result['views'][name] = {'target_first_hit_rays': visible}
        result['unobstructed_target_ray'] |= visible > 0
    return result


def _angular_separation(first, second, centre):
    a, b = np.asarray(first)[:2] - centre[:2], np.asarray(second)[:2] - centre[:2]
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0. if scale <= 1e-8 else math.degrees(math.acos(float(np.clip(np.dot(a, b) / scale, -1., 1.))))


class InspectionRouteEvaluator:
    """Private native trajectory, geometric camera opportunity and pair preservation."""

    def __init__(self, session, controller, config, output):
        if (config['scene'], config['room']) != (session.scene_name, session.room_name):
            raise ValueError('Partial Occlusion evaluation must match the loaded source.')
        self.session, self.controller, self.config = session, controller, config
        self.parameters = config['parameters']
        self.objects = {key: session.env.scene.object_registry('name', config['bindings'][key])
                        for key in ('target', 'occluder')}
        if any(obj is None for obj in self.objects.values()):
            raise ValueError('Original native inspection pair is unavailable.')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.initial = None

    def begin_episode(self, observation):
        self.initial = {key: tuple(_array(value).copy() for value in obj.get_position_orientation())
                        for key, obj in self.objects.items()}
        self.previous = _array(self.session.robot.get_position_orientation()[0]).copy()
        self.start_step = observation.action_step
        self.travel = 0.
        self.continuous = self.preserved = True
        self.viewpoints = []
        return self.record(observation, action=None)

    def sample_step(self):
        if self.initial is None:
            return
        position = _array(self.session.robot.get_position_orientation()[0])
        delta = float(np.linalg.norm(position[:2] - self.previous[:2]))
        self.continuous &= delta <= self.parameters['maximum_step_translation_m']
        self.travel += delta
        self.previous = position.copy()
        for key, obj in self.objects.items():
            xyz, q = (_array(value) for value in obj.get_position_orientation())
            start, initial_q = self.initial[key]
            cosine = abs(float(np.dot(q, initial_q) / (np.linalg.norm(q) * np.linalg.norm(initial_q))))
            angle = math.degrees(2 * math.acos(float(np.clip(cosine, 0., 1.))))
            self.preserved &= (np.linalg.norm(xyz - start) <= self.parameters['maximum_object_drift_m']
                               and angle <= self.parameters['maximum_object_rotation_deg'])

    def record(self, observation, *, action):
        if self.initial is None:
            return None
        self.sample_step()
        target = self.objects['target']
        low, high = (_array(value) for value in target.aabb)
        centre = (low + high) / 2
        position = self.previous
        distance = float(np.linalg.norm(np.maximum(np.maximum(low[:2] - position[:2], position[:2] - high[:2]), 0.)))
        view = _camera_opportunity(self.session, target)
        safety = _robot_safety(self.session)
        viewpoint = bool(safety['floor_supported'] and safety['collision_free'] and safety['same_source_room']
                         and safety['stopped'] and view['unobstructed_target_ray']
                         and distance <= self.parameters['maximum_surface_distance_m'])
        if viewpoint and observation.action_step > self.start_step:
            if not self.viewpoints or np.linalg.norm(position[:2] - np.asarray(self.viewpoints[-1])[:2]) >= .10:
                self.viewpoints.append(position.tolist())
        separation = max((_angular_separation(a, position, centre) for a in self.viewpoints), default=0.)
        criteria = {'two_separated_viewpoints': separation >= self.parameters['minimum_viewpoint_separation_deg'],
            'minimum_travel_reached': self.travel >= self.parameters['minimum_travel_m'],
            'continuous_native_motion': bool(self.continuous), 'original_pair_preserved': bool(self.preserved),
            'terminal_viewpoint_safe_and_observable': viewpoint, 'post_action_boundary': observation.action_step > self.start_step}
        result = {'schema': 'private_partial_occlusion_inspection_v1', 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'task_success': bool(all(criteria.values())), 'action': action,
            'criteria': criteria, 'travel_m': self.travel, 'viewpoint_separation_deg': separation,
            'surface_distance_m': distance, 'viewpoints': self.viewpoints.copy(), 'camera_opportunity': view,
            'robot_safety': safety, 'configuration': self.config}
        from wia.authoring.io import write_json
        write_json(self.output / 'state.json', result)
        with (self.output / 'trajectory.jsonl').open('a') as handle:
            handle.write(json.dumps(result, allow_nan=False) + '\n')
        return result


def create_evaluator(session, controller, config, output):
    """Adapt native measurements to the shared private episode envelope."""
    if config['operation'] == VISUAL_DISCOVERY:
        return VisualDiscoveryEvaluator(session, controller, config, output)
    if config['operation'] == GROUNDING_OPERATION:
        return ViewpointIdentificationPlacementEvaluator(session, controller, config, output)
    if config['operation'] == IDENTIFY_PLACE:
        return IdentificationPlacementEvaluator(session, controller, config, output)
    if config['operation'] == 'inspection_route':
        return InspectionRouteEvaluator(session, controller, config, output)
    if config['operation'] != 'uncover_retrieve':
        raise ValueError('Unsupported Partial Occlusion operation.')
    from wia.native.esi_task_evaluation import GraspLiftEvaluator

    class UncoverRetrieveEvaluator(GraspLiftEvaluator):
        def __init__(self):
            native = {'type': 'grasp_lift', 'scene': config['scene'], 'room': config['room'],
                      'target': config['bindings']['target'], 'minimum_lift_m': config['parameters']['minimum_lift_m']}
            super().__init__(session, controller, native, output)
            self.blocker = session.env.scene.object_registry('name', config['bindings']['occluder'])
            self.support = session.env.scene.object_registry('name', config['bindings']['support'])
            if self.blocker is None or self.support is None or self.blocker.visual_only or self.blocker.fixed_base:
                raise ValueError('Uncovering requires the original movable blocker and original support.')

        def begin_episode(self, observation):
            self.blocker_bottom = float(_array(self.blocker.aabb[0])[2])
            self.blocker_initially_unheld = self.controller.debug_gripper_state().get('object_name') != self.blocker.name
            self.blocker_lifted = False
            self.release_stable_steps = 0
            return super().begin_episode(observation)

        def sample_step(self):
            if self.initial is None:
                return
            gripper = self.controller.debug_gripper_state()
            held = (gripper.get('holding') is True and gripper.get('constraint_active') is True
                    and gripper.get('object_name') == self.blocker.name)
            lift = float(_array(self.blocker.aabb[0])[2]) - self.blocker_bottom
            self.blocker_lifted |= held and lift >= config['parameters']['minimum_lift_m']
            checks, _ = self._blocker_state()
            stable = self.blocker_lifted and all(checks.values())
            self.release_stable_steps = self.release_stable_steps + 1 if stable else 0

        def _blocker_state(self):
            from wia.tasks.interaction_setup import touching_task_support, on_top_task_support
            low, high = (_array(value) for value in self.blocker.aabb)
            target_low, target_high = (_array(value) for value in self.target.aabb)
            gap = float(np.max(np.maximum(low[:2] - target_high[:2], target_low[:2] - high[:2])))
            speed = float(np.linalg.norm(_array(self.blocker.get_linear_velocity())))
            angular_speed = float(np.linalg.norm(_array(self.blocker.get_angular_velocity())))
            checks = {'blocker_released': self.controller.debug_gripper_state().get('object_name') != self.blocker.name,
                'blocker_cleared': gap >= config['parameters']['minimum_clearance_m'],
                'blocker_supported': bool(touching_task_support(session, self.blocker, self.support)
                                         and on_top_task_support(session, self.blocker, self.support)),
                'blocker_stopped': speed <= config['parameters']['maximum_release_speed_m_s'] and angular_speed <= .10}
            return checks, {'blocker_clearance_m': gap, 'blocker_linear_speed_m_s': speed,
                            'blocker_angular_speed_rad_s': angular_speed}

        def additional_evidence(self, target):
            checks, evidence = self._blocker_state()
            checks.update(blocker_initially_unheld=self.blocker_initially_unheld,
                blocker_grasped_and_lifted=bool(self.blocker_lifted),
                blocker_release_stable=self.release_stable_steps >= config['parameters']['release_stability_steps'])
            return checks, {**evidence, 'blocker_release_stable_steps': self.release_stable_steps,
                            'operation_configuration': config}

    return UncoverRetrieveEvaluator()


def _normalize_answer(value):
    """Normalize category spelling without guessing a semantic synonym."""
    return ' '.join(str(value or '').lower().replace('_', ' ').split())


def _placement_measurements(session, controller, config, target, destination):
    """Measure the named destination directly, never an adapted initial support."""
    values = {'measurement_errors': {}}

    def read(key, callback):
        try:
            values[key] = callback()
        except (KeyError, ValueError, AttributeError, RuntimeError, TypeError, ImportError) as error:
            values[key] = None
            values['measurement_errors'][key] = f'{type(error).__name__}: {error}'

    def grasp():
        state = controller.debug_gripper_state()
        for key in ('holding', 'constraint_active', 'release_pending'):
            if type(state.get(key)) is not bool:
                raise ValueError(f'Native {key} telemetry unavailable.')
        if 'object_name' not in state:
            raise ValueError('Native bound-object identity telemetry unavailable.')
        return state

    def relation():
        from omnigibson.object_states import Inside, OnTop, Touching
        if config['parameters']['placement_relation'] == 'inside':
            # Native Inside may use spatial sampling or bounding-box approximations.
            # Its conjunction with contact is measured here, while independent
            # interior geometry and fit qualification remains a release gate.
            return bool(target.states[Inside].get_value(destination)
                        and target.states[Touching].get_value(destination))
        return bool(target.states[OnTop].get_value(destination)
                    and target.states[Touching].get_value(destination))

    def withdrawn():
        from wia.tasks.interaction_setup import _body_under, _contact_parts
        return not any(_body_under(body, session.robot.prim_path)
                       for contact in target.contact_list() for body in _contact_parts(contact))

    read('gripper', grasp)
    read('destination_relation', relation)
    read('linear_speed_m_s', lambda: float(np.linalg.norm(_array(target.get_linear_velocity()))))
    read('angular_speed_rad_s', lambda: float(np.linalg.norm(_array(target.get_angular_velocity()))))
    read('robot_withdrawn', withdrawn)
    return values


class IdentificationPlacementEvaluator:
    """Keep semantic reporting independent of native selection and released placement."""

    def __init__(self, session, controller, config, output):
        if (config['scene'], config['room']) != (session.scene_name, session.room_name):
            raise ValueError('Identification placement must match the loaded source scene and room.')
        self.session, self.controller, self.config = session, controller, config
        self.parameters = config['parameters']
        self.target, self.destination = [session.env.scene.object_registry('name', config['bindings'][key])
                                        for key in ('target', 'destination')]
        if self.target is None or self.destination is None:
            raise ValueError('Original target and named destination must be bound natively.')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.active = False

    def measure(self):
        return _placement_measurements(self.session, self.controller, self.config, self.target, self.destination)

    def begin_episode(self, observation):
        self.active = True
        self.start_step = observation.action_step
        self.answer = None
        self.answer_history = []
        self.selected = False
        self.selection_history_complete = True
        self.stable_steps = 0
        self.samples = 0
        self.last = self.measure()
        return self.record(observation, action=None)

    def report_answer(self, answer, observation):
        if not self.active:
            raise RuntimeError('Category reports require an active episode.')
        # The public transport sends the category text. A later explicit correction
        # replaces the current answer while history remains reviewable.
        self.answer = _normalize_answer(answer)
        self.answer_history.append({'answer': str(answer), 'normalized_answer': self.answer,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step})

    def _update(self, *, count_step=False):
        measured = self.measure()
        gripper = measured.get('gripper')
        if gripper is None:
            self.selection_history_complete = False
        elif count_step and (gripper['holding'] is True and gripper['constraint_active'] is True
                             and gripper['object_name'] == self.target.name):
            self.selected = True
        released = None if gripper is None else bool(gripper['holding'] is False
            and gripper['constraint_active'] is False and gripper['release_pending'] is False
            and gripper['object_name'] is None)
        physical_available = all(measured.get(key) is not None for key in (
            'destination_relation', 'linear_speed_m_s', 'angular_speed_rad_s', 'robot_withdrawn')) and released is not None
        stable = bool(physical_available and released and measured['robot_withdrawn']
            and measured['destination_relation']
            and measured['linear_speed_m_s'] <= self.parameters['maximum_release_speed_m_s']
            and measured['angular_speed_rad_s'] <= self.parameters['maximum_release_angular_speed_rad_s'])
        if not stable:
            self.stable_steps = 0
        elif count_step:
            self.stable_steps += 1
        if count_step:
            self.samples += 1
        self.last = measured
        return physical_available, stable

    def sample_step(self):
        if self.active:
            self._update(count_step=True)

    def record(self, observation, *, action=None):
        if not self.active:
            return None
        available, stable = self._update()
        identity = self.answer is not None and self.answer == self.parameters['expected_identity']
        items = [
            {'id': 'identity', 'kind': 'semantic_answer', 'completed': bool(identity),
             'evaluation_validity': 'valid'},
            {'id': 'correct_bound_selection', 'kind': 'checkpoint', 'completed': self.selected,
             'evaluation_validity': 'valid' if self.selected or self.selection_history_complete else 'unavailable'},
            {'id': 'stable_placement', 'kind': 'terminal_condition',
             'completed': bool(stable and self.stable_steps >= self.parameters['release_stability_steps']),
             'evaluation_validity': 'valid' if available else 'unavailable'}]
        valid = all(item['evaluation_validity'] == 'valid' for item in items)
        result = {'schema': 'private_identification_placement_v1', 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'action': action,
            'task_success': bool(valid and all(item['completed'] for item in items)),
            'evaluation_validity': 'complete' if valid else 'partial', 'completion_items': items,
            'answer_history': self.answer_history.copy(), 'release_stable_steps': self.stable_steps,
            'physics_samples': self.samples, 'measurements': self.last, 'configuration': self.config}
        from wia.tasks.completion import progress
        result.update(progress(result))
        from wia.authoring.io import write_json
        write_json(self.output / 'state.json', result)
        with (self.output / 'trajectory.jsonl').open('a') as handle:
            handle.write(json.dumps(result, allow_nan=False) + '\n')
        return result


def _grounding_actor_state(obj):
    """Read appearance-producing pose/scale and available native material fields."""
    position, quaternion = (_array(value).astype(float) for value in obj.get_position_orientation())
    scale = _array(obj.scale).astype(float)
    if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(np.r_[position, quaternion]).all():
        raise ValueError('Native grounding actor pose is unavailable.')
    materials = getattr(obj, 'materials', ())
    materials = materials.values() if isinstance(materials, dict) else materials
    appearance = []
    for material in materials:
        row = {'prim_path': str(getattr(material, 'prim_path', ''))}
        for key in ('diffuse_color_constant', 'reflection_roughness_constant', 'metallic_constant',
                    'opacity_constant', 'albedo_add', 'albedo_brightness'):
            value = getattr(material, key, None)
            if value is not None:
                row[key] = _array(value).tolist()
        appearance.append(row)
    return {'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
            'scale': scale.tolist(), 'materials': appearance,
            'category': obj.category, 'model': getattr(obj, 'model', None), 'name': obj.name}


class ViewpointIdentificationPlacementEvaluator(IdentificationPlacementEvaluator):
    """Require exact diagnostic inspection before original-object identification/grasp."""

    def __init__(self, session, controller, config, output):
        super().__init__(session, controller, config, output)
        self.contract = self.parameters['grounding_contract']
        self.occluder = (session.env.scene.object_registry('name', config['bindings']['occluder'])
                         if 'occluder' in config['bindings'] else None)
        self.resting_support = (session.env.scene.object_registry('name', config['bindings']['occluder_rest'])
                                if self.contract['recipe'] == 'move_occluder' else None)
        if self.contract['recipe'] == 'move_occluder' and (self.occluder is None or self.resting_support is None):
            raise ValueError('The permitted original occluder and distinct resting support must be bound.')

    def begin_episode(self, observation):
        self.initial_audit = self.inspection_event = None
        self.visual_status = {'available': False, 'reason': 'initial_input_audits_missing'}
        self.last_visual = {}
        self.eligible_identity = False
        self.control_events, self.grounding_faults, self.grounding_errors = {}, set(), []
        self.occluder_grasped, self.occluder_stable_steps = False, 0
        self.occluder_stable, self.occluder_available = False, self.contract['recipe'] != 'move_occluder'
        self.initial_actors = {'target': _grounding_actor_state(self.target)}
        if self.occluder is not None:
            self.initial_actors['occluder'] = _grounding_actor_state(self.occluder)
        try:
            bound = {key: self.session.env.scene.object_registry('name', name)
                     for key, name in self.config['bindings'].items()}
            checks, geometry = grounding_source_geometry(self.session, self.config, bound)
            self.initial_source_geometry = {'checks': checks, 'geometry': geometry, 'accepted': all(checks.values())}
            if not all(checks.values()):
                self.grounding_errors.append('native_initial_common_translation_or_source_reference_unqualified')
        except Exception as error:
            self.initial_source_geometry = {'accepted': False, 'error': str(error)}
            self.grounding_errors.append('native_initial_source_geometry_unavailable:' + str(error))
        return super().begin_episode(observation)

    def _preserved(self):
        try:
            for key, initial in self.initial_actors.items():
                current = _grounding_actor_state(self.target if key == 'target' else self.occluder)
                shape = all(current[name] == initial[name] for name in ('name', 'category', 'model', 'scale', 'materials'))
                pose = (np.linalg.norm(np.asarray(current['position_m']) - initial['position_m']) <= self.contract['position_tolerance_m']
                    and _quaternion_angle(current['quaternion_xyzw'], initial['quaternion_xyzw']) <= self.contract['orientation_tolerance_deg'])
                if not shape:
                    self.grounding_faults.add(key + '_appearance_changed')
                if key == 'target' and not self.eligible_identity and not pose:
                    self.grounding_faults.add('target_moved_before_identification')
                if key == 'occluder' and self.contract['recipe'] == 'viewpoint' and not pose:
                    self.grounding_faults.add('protected_occluder_moved')
        except Exception as error:
            self.grounding_errors.append('native_appearance_measurement_unavailable:' + str(error))

    @staticmethod
    def _feature_ids(row):
        evidence = row.get('informative_feature_evidence', {})
        ids = evidence.get('feature_ids')
        return set(ids) if (row.get('informative') is True and row.get('audited') is True
            and row.get('identity_synchronized') is True and row.get('feature_audit_id')
            and evidence.get('semantic_identity_verified') is True and evidence.get('view_kind') == 'direct'
            and isinstance(ids, list) and ids and all(isinstance(value, str) and value for value in ids)) else set()

    def _row_matches(self, row, observation):
        context = row.get('audit_context', {})
        return (row.get('snapshot_id') == observation.snapshot_id
            and row.get('action_step') == observation.action_step
            and row.get('native_name') == self.target.name
            and row.get('native_rgb_matches_delivered') is True
            and context.get('source_identity') == self.contract['source_identity']
            and context.get('reset_id') == getattr(self.session, 'native_reset_id', None)
            and context.get('episode_id') == str(getattr(self.session, 'episode_id', ''))
            and context.get('snapshot_id') == observation.snapshot_id
            and context.get('action_step') == observation.action_step
            and context.get('camera') == row.get('camera')
            and context.get('native_identity', {}).get('name') == self.target.name)

    def _initial_input_audit(self, visual, observation):
        from core.robot_cameras import ROBOT_CAMERA_LABELS
        if observation.action_step != self.start_step:
            return None
        views = visual.get('measurements', {}).get('target', {}).get('views', {})
        if any(name not in views or views[name].get('audited') is not True
               or views[name].get('feature_audit_errors')
               or not self._row_matches(views[name], observation) for name in ROBOT_CAMERA_LABELS):
            return None
        frozen = visual.get('frozen_views', {}).get('initial_overview_rgb', {})
        if frozen.get('initial_input_reviewed') is not True:
            return None
        receipt = next((row for row in getattr(self.session, 'visual_evidence_frozen_input_receipts', ())
            if row.get('schema') == 'native_frozen_initial_input_audit_v1'
            and row.get('audit_id') == frozen.get('audit_id') and row.get('context') == frozen.get('audit_context')), None)
        global_target = (receipt or {}).get('target_identity_audits', {}).get('target', {})
        if global_target.get('native_name') != self.target.name or type(global_target.get('informative')) is not bool:
            return None
        global_ids = global_target.get('feature_ids', [])
        if (not isinstance(global_ids, list) or any(not isinstance(value, str) or not value for value in global_ids)
                or len(set(global_ids)) != len(global_ids)
                or global_target['informative'] and not global_ids):
            return None
        ids = set(global_ids) if global_target['informative'] else set()
        for row in views.values():
            ids |= self._feature_ids(row)
        camera = visual.get('cameras', {}).get('agentview_rgb', {})
        control = visual.get('native_public_control_context', {})
        robot_pose = views['agentview_rgb'].get('audit_context', {}).get('robot_pose')
        if (not camera.get('camera_pose') or control.get('available') is not True
                or type(control.get('control_sequence')) is not int or not robot_pose):
            return None
        return {'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'feature_ids': sorted(ids), 'difficulty': 'easy_initial_identity_available' if ids else 'initially_ambiguous_audited',
            'head_pose': deepcopy(camera['camera_pose']), 'control_sequence': control['control_sequence'],
            'robot_pose': deepcopy(robot_pose),
            'task_context': deepcopy(visual.get('native_capture_context', {})),
            'global_audit_id': frozen['audit_id'], 'mounted_views_audited': list(ROBOT_CAMERA_LABELS)}

    def _observe_inspection(self, observation, action=None):
        provider = getattr(self.session, 'task_visual_evidence', None)
        try:
            if not callable(provider):
                raise ValueError('task_visual_evidence adapter unavailable')
            visual = provider(observation, self.config['bindings'], self.parameters)
            self.last_visual = visual
            if visual.get('schema') != 'native_visual_evidence_v1' or visual.get('recognition_inferred_from_mask_size') is not False:
                raise ValueError('Exact native visual collector contract unavailable')
            if self.initial_audit is None:
                self.initial_audit = self._initial_input_audit(visual, observation)
            if self.initial_audit is None:
                self.visual_status = {'available': False, 'reason': 'initial_eight_plus_global_target_identity_audit_missing'}
                return
            control = visual.get('native_public_control_context', {})
            cameras = visual.get('cameras', {})
            head = cameras.get('agentview_rgb', {}).get('camera_pose', {})
            safe_control = (control.get('available') is True and control.get('state_setter_used') is False
                and type(control.get('control_sequence')) is int)
            if not safe_control or not head:
                raise ValueError('Native public control sequence/head pose is unavailable')
            history = visual.get('public_action_history', [])
            last = history[-1] if history else {}
            token = action if isinstance(action, str) else last.get('action')
            allowed = token in ('BASE_FWD', 'BASE_BACK', 'BASE_LEFT', 'BASE_RIGHT', 'TRUNK_UP', 'TRUNK_DOWN')
            changed = np.linalg.norm(np.asarray(head['position_m']) - self.initial_audit['head_pose']['position_m']) > 1e-6
            changed = changed and (not str(token).startswith('TRUNK_') or abs(
                head['position_m'][2] - self.initial_audit['head_pose']['position_m'][2]) > 1e-6)
            if str(token).startswith('BASE_'):
                robot_pose = visual.get('measurements', {}).get('target', {}).get('views', {}).get(
                    'agentview_rgb', {}).get('audit_context', {}).get('robot_pose')
                if not robot_pose:
                    raise ValueError('Native base pose is missing for the public translation')
                changed = changed and np.linalg.norm(np.asarray(robot_pose['position_m'])
                    - self.initial_audit['robot_pose']['position_m']) > 1e-6
            if (allowed and last.get('action') == token and last.get('result_summary') == 'completed'
                    and observation.action_step > self.start_step and changed
                    and control['control_sequence'] > self.initial_audit['control_sequence']):
                self.control_events.setdefault(observation.action_step, {'action': token,
                    'control_sequence': control['control_sequence'], 'head_pose': deepcopy(head),
                    'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step})
            ready = bool(self.control_events) if self.contract['recipe'] == 'viewpoint' else bool(
                self.occluder_grasped and self.occluder_stable and self.occluder_stable_steps >= self.parameters['release_stability_steps'])
            if not ready:
                self.visual_status = {'available': True, 'reason': 'required_public_inspection_not_completed'}
                return
            views = visual.get('measurements', {}).get('target', {}).get('views', {})
            audited = False
            for camera, row in views.items():
                if camera == 'initial_overview_rgb' or camera not in observation.rgb_images() or not self._row_matches(row, observation):
                    continue
                audited |= row.get('audited') is True
                new_ids = self._feature_ids(row) - set(self.initial_audit['feature_ids'])
                if new_ids:
                    context = row['audit_context']
                    original = self.initial_audit['task_context']
                    if any(context.get(key) != original.get(key) for key in
                           ('task_id', 'task_sha256', 'scene_id', 'scene_sha256', 'source_sha256', 'seed')):
                        raise ValueError('Diagnostic feature belongs to a different task/reset configuration')
                    if not self.grounding_faults:
                        self.inspection_event = self.inspection_event or {'snapshot_id': observation.snapshot_id,
                            'action_step': observation.action_step, 'camera': camera,
                            'feature_ids': sorted(new_ids), 'feature_audit_id': row['feature_audit_id'],
                            'audit_context': deepcopy(context), 'control_events': deepcopy(list(self.control_events.values())),
                            'recipe': self.contract['recipe']}
                    self.visual_status = {'available': True, 'reason': None}
                    return
            self.visual_status = {'available': audited, 'reason': 'new_diagnostic_feature_not_verified'}
        except Exception as error:
            self.visual_status = {'available': False, 'reason': f'{type(error).__name__}: {error}'}

    def report_answer(self, answer, observation):
        self._preserved()
        self._observe_inspection(observation)
        super().report_answer(answer, observation)
        self.eligible_identity = bool(self.inspection_event is not None and not self.grounding_faults
            and self.answer == self.parameters['expected_identity'])
        self.answer_history[-1].update(informative_inspection_preceded=self.inspection_event is not None,
                                       eligible_identity=self.eligible_identity)

    def _update(self, *, count_step=False):
        self._preserved()
        previous = self.selected
        available, stable = super()._update(count_step=count_step)
        gripper = self.last.get('gripper')
        if count_step and gripper and gripper.get('holding') and gripper.get('constraint_active'):
            if gripper.get('object_name') == self.target.name and not self.eligible_identity:
                self.selected = previous
                self.grounding_faults.add('target_grasped_before_informative_identification')
            if self.occluder is not None and gripper.get('object_name') == self.occluder.name:
                if self.contract['recipe'] == 'viewpoint':
                    self.grounding_faults.add('protected_occluder_grasped')
                else:
                    self.occluder_grasped = True
        if self.contract['recipe'] == 'move_occluder':
            config = deepcopy(self.config)
            config['parameters']['placement_relation'] = 'on_top'
            rest = _placement_measurements(self.session, self.controller, config, self.occluder, self.resting_support)
            grip = rest.get('gripper')
            self.occluder_available = all(rest.get(key) is not None for key in (
                'destination_relation', 'linear_speed_m_s', 'angular_speed_rad_s', 'robot_withdrawn')) and grip is not None
            self.occluder_stable = bool(self.occluder_available and grip.get('holding') is False
                and grip.get('constraint_active') is False and grip.get('release_pending') is False
                and grip.get('object_name') is None and rest['destination_relation'] and rest['robot_withdrawn']
                and rest['linear_speed_m_s'] <= self.parameters['maximum_release_speed_m_s']
                and rest['angular_speed_rad_s'] <= self.parameters['maximum_release_angular_speed_rad_s'])
            if not self.occluder_stable:
                self.occluder_stable_steps = 0
            elif count_step:
                self.occluder_stable_steps += 1
        return available, stable

    def record(self, observation, *, action=None):
        if not self.active:
            return None
        available, stable = self._update()
        self._observe_inspection(observation, action)
        semantic_available = self.initial_audit is not None and self.visual_status['available'] and not self.grounding_errors
        items = [{'id': 'informative_inspection', 'kind': 'historical_checkpoint',
            'completed': self.inspection_event is not None,
            'evaluation_validity': 'valid' if self.inspection_event or semantic_available else 'unavailable',
            'evidence': deepcopy(self.inspection_event)},
            {'id': 'identity', 'kind': 'answer_requirement', 'completed': self.eligible_identity,
             'evaluation_validity': 'valid' if semantic_available or self.eligible_identity else 'unavailable'},
            {'id': 'correct_bound_selection', 'kind': 'historical_checkpoint', 'completed': self.selected,
             'evaluation_validity': 'valid' if self.selected or self.selection_history_complete else 'unavailable'},
            {'id': 'stable_placement', 'kind': 'terminal_condition',
             'completed': bool(stable and self.stable_steps >= self.parameters['release_stability_steps']),
             'evaluation_validity': 'valid' if available else 'unavailable'}]
        if self.contract['recipe'] == 'move_occluder':
            items.append({'id': 'safe_occluder_resting', 'kind': 'terminal_condition',
                'completed': bool(self.occluder_grasped and self.occluder_stable and
                                  self.occluder_stable_steps >= self.parameters['release_stability_steps']),
                'evaluation_validity': 'valid' if self.occluder_available else 'unavailable'})
        valid = all(row['evaluation_validity'] == 'valid' for row in items) and not self.grounding_errors
        result = {'schema': 'private_task_edit2_grounding_v1', 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'action': action,
            'task_success': bool(valid and all(row['completed'] for row in items) and not self.grounding_faults),
            'evaluation_validity': 'complete' if valid else 'unavailable', 'completion_items': items,
            'answer_history': deepcopy(self.answer_history), 'initial_input_audit': deepcopy(self.initial_audit),
            'initial_source_geometry': deepcopy(self.initial_source_geometry),
            'recognition_difficulty': self.initial_audit['difficulty'] if self.initial_audit else 'unavailable',
            'inspection_evidence_status': deepcopy(self.visual_status), 'grounding_faults': sorted(self.grounding_faults),
            'measurement_errors': list(self.grounding_errors), 'constraints_satisfied': not self.grounding_faults,
            'physics_samples': self.samples, 'release_stable_steps': self.stable_steps,
            'reported_identity_matches_private_answer': self.answer == self.parameters['expected_identity'],
            'answer_accuracy_percent': (100. if self.eligible_identity else 0.) if semantic_available or self.eligible_identity else None,
            'required_unique_questions': 1, 'correct_answers': int(self.eligible_identity),
            'configuration': self.config, 'measurements': self.last}
        from wia.tasks.completion import progress
        result.update(progress(result))
        result['physical_obligations'] = progress({'task_success': False,
            'completion_items': [row for row in items if row['id'] not in ('identity', 'informative_inspection')]})
        from wia.authoring.io import write_json
        write_json(self.output / 'state.json', result)
        with (self.output / 'trajectory.jsonl').open('a') as handle:
            handle.write(json.dumps(result, allow_nan=False) + '\n')
        return result


VISUAL_DISCOVERY = 'distinct_visual_inventory'


def native_visual_evidence(session, observation, objects):
    """Read identity masks only when native RGB exactly matches delivered RGB.

    Existing private seg_instance must be enabled during initialization. This
    helper never changes sensor modalities, renders, steps physics or infers
    semantic recognition from a pixel gate.
    """
    delivered = observation.rgb_images()
    registry = getattr(session, 'robot_camera_registry', {})
    rows = {key: {'visible': False, 'views': {}} for key in objects}
    errors, available_views = {}, []
    for view, rgb in delivered.items():
        if view not in registry:
            continue
        try:
            native, info = session.robot.sensors[registry[view]].get_obs()
            actual = _array(native['rgb'])[..., :3].astype(np.uint8)
            if not np.array_equal(actual, np.asarray(rgb)):
                raise ValueError('Native segmentation RGB does not match delivered observation.')
            segmentation = _array(native['seg_instance'])
            if segmentation.shape != actual.shape[:2]:
                raise ValueError('Native identity mask shape does not match delivered RGB.')
            labels = info['seg_instance']
            available_views.append(view)
            for key, obj in objects.items():
                ids = [int(label) for label, name in labels.items() if name == obj.name]
                ys, xs = np.nonzero(np.isin(segmentation, ids))
                extent = 0 if not len(xs) else int(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
                gate = len(xs) >= 64 and extent >= 8
                rows[key]['visible'] |= bool(gate)
                rows[key]['views'][view] = {'visible_pixels': int(len(xs)), 'minimum_extent_px': extent,
                    'passes_visibility_gate': bool(gate), 'native_rgb_matches_delivered': True}
        except (KeyError, ValueError, AttributeError, RuntimeError, TypeError) as error:
            errors[view] = f'{type(error).__name__}: {error}'
    if not available_views:
        errors['coverage'] = 'No synchronized mounted RGB/instance segmentation is available.'
    return {'objects': rows, 'available_views': available_views, 'measurement_errors': errors,
            'complete_coverage': bool(available_views) and not errors, 'recognition_verified': False}


def visual_discovery_task(source, context, *, bindings, target_keys, public, patches=None,
                          preserved_relations=(), regions=None, source_evidence=None):
    """Build an explicit source-preserving visual inventory, retaining native gaps."""
    targets = list(dict.fromkeys(target_keys))
    parameters = {'task_revision': TASK_REVISION, 'target_keys': targets,
        'requires_native_visual_discovery': True, 'disclosed_target_count': len(targets),
        'mechanism_contract': {'preserved_relations': list(preserved_relations),
            'initialization_relocation_policy': 'preserve',
            'required_native_evidence': ['source support and no fallen targets',
                'source mechanism geometry', 'all mounted initial camera views',
                'synchronized delivered RGB and private instance identity', 'legal informative access'],
            'native_operation_verified': False}, 'regions': regions or {},
        'source_evidence': source_evidence or {},
        'completion_item_ids': ['discovery:' + key for key in targets]}
    if regions:
        parameters['completion_item_ids'] += ['region_report:' + key for key in regions]
    evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': VISUAL_DISCOVERY,
        'binding_keys': list(bindings), 'parameters': parameters}
    task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings,
        patches=patches, qualification={'on_failure': 'skip', 'initial_visibility': 'none'},
        capabilities=['base_translation', 'base_rotation'])
    return {'scene_definition': original_scene(source, context, bindings=bindings, patches=patches),
        'task_definition': task, 'status': 'generated', 'reuse_level': 'L2' if patches else 'L1',
        'operation_recipe': VISUAL_DISCOVERY, 'measured_capability': 'all_target_visual_search',
        'interaction_policy': {'mode': 'navigation', 'group_bindings': targets,
            'manipulation_bindings': [], 'relocation_policy': 'preserve', 'allow_fallback_table': False,
            'source_region': source.raw['room']},
        'warning_codes': ['source_mechanism_native_review_pending', 'native_visual_discovery_pending'],
        'change_summary': ['Disclose the total and require each distinct original target in delivered native robot observations.',
            'Preserve task objects, occluders, supports, source reference and external relationships.'],
        'limitations': ['Instance pixel evidence measures visible identity, not semantic recognition.',
            'Native reset support, informative access and mechanism qualification remain pending.'],
        'source_layout_evidence': source_evidence or {},
        'next_step': 'Qualify original layout, synchronized mounted views and informative access in the candidate pool.'}


def qualify_visual_discovery(session, config, bound):
    try:
        checks = _robot_safety(session)
        checks['scene_matches'] = (session.scene_name, session.room_name) == (config['scene'], config['room'])
        checks['bindings_match'] = all(bound[key].name == config['bindings'][key] for key in config['binding_keys'])
        required = ['floor_supported', 'collision_free', 'same_source_room', 'native_r1', 'scene_matches', 'bindings_match']
        return {'accepted': all(checks[key] for key in required), 'checks': checks,
            'executed_operation_verified': False, 'source_mechanism_verified': False,
            'reason': None if all(checks[key] for key in required) else 'native_visual_inventory_precondition_failed'}
    except (KeyError, ValueError, AttributeError, RuntimeError) as error:
        return {'accepted': False, 'reason': f'native_visual_inventory_measurement_unavailable:{error}'}


class VisualDiscoveryEvaluator:
    """Count each rendered original target once; retain unavailable coverage."""
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters = config['parameters']
        self.objects = {key: session.env.scene.object_registry('name', config['bindings'][key])
                        for key in self.parameters['target_keys']}
        if any(obj is None for obj in self.objects.values()):
            raise ValueError('Native original discovery target is unavailable.')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.active = False

    def begin_episode(self, observation):
        self.active = True
        self.discoveries, self.region_reports = {}, set()
        return self.record(observation, action=None)

    def sample_step(self):
        # Discovery is tied to delivered observations, never hidden intermediate views.
        return None

    def report_answer(self, answer, observation):
        if not self.active:
            raise RuntimeError('Inventory reports require an active episode.')
        for key, region in self.parameters.get('regions', {}).items():
            if _normalize_answer(answer) == _normalize_answer(region['public_label']):
                self.region_reports.add(key)

    def record(self, observation, *, action=None):
        if not self.active:
            return None
        evidence = native_visual_evidence(self.session, observation, self.objects)
        for key, row in evidence['objects'].items():
            if row['visible'] and key not in self.discoveries:
                self.discoveries[key] = {'snapshot_id': observation.snapshot_id,
                    'action_step': observation.action_step, 'views': row['views']}
        items = [{'id': 'discovery:' + key, 'kind': 'checkpoint', 'completed': key in self.discoveries,
            'evaluation_validity': 'valid' if key in self.discoveries or evidence['complete_coverage'] else 'unavailable'}
            for key in self.objects]
        items += [{'id': 'region_report:' + key, 'kind': 'semantic_output',
            'completed': key in self.region_reports and all(target in self.discoveries for target in region['targets']),
            'evaluation_validity': 'valid' if evidence['complete_coverage'] or
                all(target in self.discoveries for target in region['targets']) else 'unavailable'}
            for key, region in self.parameters.get('regions', {}).items()]
        result = {'schema': 'private_distinct_visual_inventory_v1', 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'completion_items': items,
            'task_success': all(item['completed'] and item['evaluation_validity'] == 'valid' for item in items),
            'discoveries': self.discoveries.copy(), 'visual_evidence': evidence,
            'semantic_recognition_verified': False}
        from wia.tasks.completion import progress
        result.update(progress(result))
        from wia.authoring.io import write_json
        write_json(self.output / 'state.json', result)
        return result
