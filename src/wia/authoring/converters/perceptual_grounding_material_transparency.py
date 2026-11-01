"""Preserve transparent-container resets and score insertion or verified no-op.

Run the category through ConverterBase.convert_all with the shared
SourceInstance/ConversionContext API; conversion is CPU-only.
"""
from __future__ import annotations

import math
import re

from wia.authoring.conversion import ConverterBase, map_records, original_scene, physical_task


OBJECT_KEYS = ('obj_container', 'obj_small')
QUESTION = re.compile(r'Is the (?P<small>.+?) inside the (?P<container>.+?)\?', re.IGNORECASE)
PARKED_CONTAINER_XY = (150., 100.)
PARKED_XY_TOLERANCE_M = .25
FALLEN_PARKING_Z_M = -100.
MINIMUM_LIFT_M = .05


def _text(value):
    return ' '.join(str(value or '').strip().lower().replace('_', ' ').split())


def _vector(value, length):
    return (isinstance(value, (list, tuple)) and len(value) == length
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x) for x in value))


def _scale(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)] * 3 if math.isfinite(value) and value > 0 else None
    return list(value) if _vector(value, 3) and min(value) > 0 else None


def _annotation(value):
    text = _text(value)
    if text in ('yes', 'true', 'inside', 'obj small is inside obj container'):
        return 'yes'
    if text in ('no', 'false', 'outside', 'not inside', 'obj small is not inside obj container'):
        return 'no'
    return None


class CategoryConverter(ConverterBase):
    category = ('Perceptual Grounding', 'Material Transparency')
    version = '2.0.0_task_edit'

    def _source_data(self, source):
        raw, payload = source.raw, source.payload
        if (source.big_task, source.small_task) != self.category or (
                raw.get('big_task'), raw.get('small_task')) != self.category:
            raise ValueError('source_category_mismatch')
        question = raw.get('question')
        match = QUESTION.fullmatch(question.strip()) if isinstance(question, str) else None
        if match is None:
            raise ValueError('unsupported_public_containment_question')
        objects = map_records(payload.get('objects'))
        for key in OBJECT_KEYS:
            obj = objects.get(key)
            if not isinstance(obj, dict) or not obj.get('category') or not obj.get('model'):
                raise ValueError(f'missing_{key}_category_or_model')
            if not _vector(obj.get('position'), 3) or not _vector(obj.get('quaternion_xyzw'), 4):
                raise ValueError(f'missing_or_invalid_{key}_pose')
            if not any(obj['quaternion_xyzw']) or _scale(obj.get('scale', 1.)) is None:
                raise ValueError(f'invalid_{key}_quaternion_or_scale')
            if (not _vector(obj.get('aabb_min'), 3) or not _vector(obj.get('aabb_max'), 3)
                    or any(low >= high for low, high in zip(obj['aabb_min'], obj['aabb_max']))):
                raise ValueError(f'missing_or_invalid_{key}_source_bounds')
        small, container = objects['obj_small'], objects['obj_container']
        for field, expected in (('small_obj_cat', small['category']), ('container_cat', container['category'])):
            if payload.get(field) and _text(payload[field]) != _text(expected):
                raise ValueError(f'private_{field}_mismatch')
        if _annotation(raw.get('answer')) not in ('yes', 'no'):
            raise ValueError('unsupported_original_yes_no_answer')
        cameras = payload.get('camera_poses')
        if isinstance(cameras, dict):
            cameras = list(cameras.values())
        if not isinstance(cameras, list) or not cameras:
            raise ValueError('missing_original_camera_poses')
        if any(not isinstance(camera, dict) or not _vector(camera.get('position'), 3)
               or not _vector(camera.get('quaternion_xyzw'), 4) for camera in cameras):
            raise ValueError('invalid_original_camera_pose')
        return raw, payload, objects

    def inspect(self, source):
        try:
            self._source_data(source)
        except (ValueError, TypeError, KeyError) as error:
            return {'eligible': False, 'reason': str(error),
                    'warning_codes': ['not_convertible_under_L1_L2', 'source_contract_review_required']}
        return {'eligible': True, 'reason': 'Original source construction contract is complete; geometry is screened in convert.'}

    def convert(self, source, context):
        if context.config.get('legacy_transparency_extraction'):
            return self.convert_legacy(source, context)
        from .partial_occlusion import TASK_REVISION
        result = self.convert_legacy(source, context)
        task = result.get('task_definition')
        if task is None:
            return result
        labels = QUESTION.fullmatch(source.raw['question'].strip()).groupdict()
        task['public'] = {
            'objective': f"Place the {labels['small']} inside its original {labels['container']} and release it stably. "
                + 'If it is already inside, leave it there and verify the final state.',
            'success_criteria': 'The original small object is released, contacts its original container, satisfies the '
                + 'native inside relation and stays stable for 15 consecutive physics steps. An already satisfied state permits a no-op.',
            'environment': 'The source inside/outside initial state, original container, object and dimensions are retained.',
            'constraints': ['Use mounted observations and legal robot controls.',
                'Do not extract an already-contained object merely to insert it again.',
                'A visual overlap alone does not establish containment.']}
        task['private']['evaluation'] = {'type': 'category_operation', 'module': __name__,
            'operation': 'insert_or_verify_inside', 'binding_keys': list(task['private']['bindings']),
            'parameters': {'task_revision': TASK_REVISION, 'placement_relation': 'inside',
                'release_stability_steps': 15, 'maximum_release_speed_m_s': .02,
                'maximum_release_angular_speed_rad_s': .10, 'expected_identity': '',
                'completion_item_ids': ['stable_inside'],
                'mechanism_contract': {'preserved_relations': ['original inside/outside reset',
                    'same small object and original container', 'original asset scales and opening geometry'],
                    'required_native_evidence': ['initial native relation', 'independent interior geometry',
                        'opening, fit and gripper clearance', 'stable released relation'],
                    'native_operation_verified': False}}}
        result.update(operation_recipe='insert_or_verify_inside',
            interaction_policy={'mode': 'manipulation', 'group_bindings': list(OBJECT_KEYS),
                'manipulation_bindings': ['obj_small'], 'relocation_policy': 'preserve',
                'allow_fallback_table': False, 'source_region': source.raw['room']},
            change_summary=['Preserve original inside/outside state and original container/object geometry.',
                'Require stable insertion or verification of an already satisfied no-op; no extraction default.'],
            warning_codes=['native_inside_interior_fit_clearance_pending', 'native_stable_release_pending'],
            assumptions=[], limitations=['Native Inside plus contact is measured but does not prove interior geometry or insertion feasibility.',
                'Source yes/no QA remains provenance; the measured original pair determines the final-state goal.'])
        if context.config.get('task_edit2_support'):
            result['interaction_policy'].update(relocation_policy='rigid_group',
                support_search='measured_surfaces', allow_fallback_table=True,
                source_camera_policy='preserve')
            result['support_revision'] = 'task_edit2_support_20261018_v1'
            task['private']['evaluation']['parameters']['support_revision'] = result['support_revision']
            result['change_summary'].append('Elevate the complete original container/object pair on a measured '
                'support, preserving inside/outside initialization and no-op semantics.')
        return result

    def convert_legacy(self, source, context):
        raw, payload, objects = self._source_data(source)
        # The original builder fixes these names; an optional metadata name is ignored.
        bindings = {key: {'name': key, 'category': objects[key]['category'], 'model': objects[key]['model']}
                    for key in OBJECT_KEYS}
        scene = original_scene(source, context, bindings=bindings)
        small, container = objects['obj_small'], objects['obj_container']
        public_labels = QUESTION.fullmatch(raw['question'].strip()).groupdict()
        labels_match = (_text(public_labels['small']) == _text(small['category'])
                        and _text(public_labels['container']) == _text(container['category']))
        source_bbox_inside = all(container['aabb_min'][axis] - .01 <=
            (small['aabb_min'][axis] + small['aabb_max'][axis]) / 2 <= container['aabb_max'][axis] + .01
            for axis in range(3))
        raw_answer, metadata_answer = _annotation(raw['answer']), _annotation(payload.get('answer'))
        disagreement = metadata_answer != raw_answer
        warnings = ['native_extraction_unverified', 'native_grasp_unverified',
                    'mounted_sensor_coverage_unverified', 'assumption_review_required']
        if disagreement:
            warnings.append('source_qa_metadata_disagreement')
        conditions = {'original_qa_answer': raw['answer'], 'metadata_containment_annotation': payload.get('answer'),
                      'annotations_agree': not disagreement, 'original_layout': payload.get('layout'),
                      'source_saved_aabb_center_inside_margin_0_01_m': source_bbox_inside,
                      'source_floor_name': payload.get('floor_name'),
                      'public_question_labels': public_labels,
                      'original_asset_labels': {'small': small['category'], 'container': container['category']},
                      'public_labels_match_original_pair': labels_match,
                      'original_bounds': {key: {field: objects[key][field] for field in
                                               ('position', 'aabb_min', 'aabb_max', 'scale')}
                                          for key in OBJECT_KEYS},
                      'native_inside_measured': False}
        assumptions = [{
            'doubt': 'The original container opening, collidable cavity, support and robot access have not '
                     'been qualified for this source-sized small object.',
            'chosen_assumption': 'Retain the complete original L1 scene as a pending retrieval candidate; '
                                 'native held/lift and measured Inside=false must independently establish terminal success.',
            'alternatives': ['Keep a source-only diagnostic if native containment state or safe extraction is unsupported.',
                             'Skip the physical task after native reach/cavity qualification fails, retaining the source and QA.'],
            'scope': 'This original task/container pair, with no asset, scale, layout, reference-camera or spawn patch.',
            'risk': 'A narrow or closed container, floor-level target, collision geometry or occlusion may prevent retrieval. '
                    'Static conversion is not evidence of physical feasibility.',
        }]
        if disagreement:
            assumptions.append({
                'doubt': 'The top-level source QA answer differs from its retained metadata containment annotation; '
                         'the public question may refer to different object categories than that metadata pair.',
                'chosen_assumption': 'Preserve both verbatim. Existing ESI loading uses the top-level answer for private '
                                     '_ground_truth unless explicitly supplied; neither annotation selects the physical target.',
                'alternatives': ['Inspect the original native Inside relation and reference images before interpreting QA.',
                                 'Treat auxiliary QA as unresolved while still requiring independent native retrieval evidence.'],
                'scope': 'This candidate auxiliary QA; original scene geometry and the native extraction predicate are unchanged.',
                'risk': 'Auxiliary yes/no scoring may conflict with the actual native initial state; do not infer containment '
                        'from metadata or silently replace the original answer.',
            })
        if not labels_match:
            warnings.extend(['not_convertible_under_L1_L2', 'public_question_object_labels_mismatch',
                             'source_only_diagnostic'])
            return {'scene_definition': scene, 'task_definition': None, 'status': 'conversion_failed',
                    'reuse_level': 'L1', 'failure_reason': 'public_question_object_labels_mismatch',
                    'change_summary': ['Preserve the original scene/question as a diagnostic; do not substitute a '
                                       'different privately named target or rewrite the public question.'],
                    'warning_codes': warnings, 'assumptions': assumptions[1:] + [{
                        'doubt': 'The public small-object/container categories do not match the original constructed pair.',
                        'chosen_assumption': 'Retain the original QA and scene but omit the physical task because the '
                                             'authorized publicly named retrieval pair lacks a faithful source binding.',
                        'alternatives': ['Inspect the original native room for a separately qualified public pair.',
                                         'Retain this instance as original observation/QA only, without a retrieval goal.'],
                        'scope': 'This mismatched public question and original obj_small/obj_container pair only.',
                        'risk': 'A similarly named native-room object might exist, but no exact binding or relation evidence '
                                'is supplied. Substituting original assets would change task semantics.',
                    }], 'limitations': ['A physical task for the publicly named pair cannot be created from the supplied '
                                       'original task-object bindings.',
                                       'The original no answer may describe the different public pair; positive metadata '
                                       'for the constructed pair is not necessarily erroneous.',
                                       'No source edit, object substitution, native image or physical operation was performed.'],
                    'private_source_conditions': conditions, 'diagnostic_source_only': True,
                    'attempted_repairs': [], 'next_step': 'Review the exact public question against original native assets; '
                                                        'retain a source-only diagnostic unless a faithful binding is evidenced.'}
        parked_fall = (all(abs(container['position'][axis] - PARKED_CONTAINER_XY[axis]) <=
                           PARKED_XY_TOLERANCE_M for axis in (0, 1))
                       and container['position'][2] < FALLEN_PARKING_Z_M
                       and small['position'][2] < FALLEN_PARKING_Z_M)
        if parked_fall:
            warnings.extend(['not_convertible_under_L1_L2', 'source_geometry_outlier', 'source_only_diagnostic'])
            return {'scene_definition': scene, 'task_definition': None, 'status': 'conversion_failed',
                    'reuse_level': 'L1', 'failure_reason': 'original_objects_at_fallen_generator_parking_location',
                    'change_summary': ['Preserve the original source as a diagnostic overlay; do not invent a physical start.'],
                    'warning_codes': warnings, 'assumptions': assumptions[1:] + [{
                        'doubt': 'The saved container remains near the generator parking XY [150, 100] m and both '
                                 'objects are below -100 m; a valid original room support is not documented.',
                        'chosen_assumption': 'Quarantine the unusable physical start and retain an unchanged source-only diagnostic.',
                        'alternatives': ['Inspect original native room/support evidence before proposing any reversible source repair.',
                                         'Retain this source exclusively for historical QA review.'],
                        'scope': 'Only original container XY within 0.25 m of [150,100], with both original objects below -100 m.',
                        'risk': 'This conservative screen is not a proof of global impossibility; no corrected support or '
                                'relocation is inferred from the generator parking/fall evidence.',
                    }], 'limitations': ['No faithful physical initial state is available under unchanged source poses.',
                                       'Diagnostic rendering may also fail or show empty room/reference views.',
                                       'No native simulation or source repair was performed.'],
                    'private_source_conditions': conditions, 'diagnostic_source_only': True,
                    'attempted_repairs': [], 'next_step': 'Review original source parking/fall and native room support evidence; '
                                                        'retain the failure if no source-faithful repair is supported.'}
        small_label = QUESTION.fullmatch(raw['question'].strip()).group('small')
        container_label = QUESTION.fullmatch(raw['question'].strip()).group('container')
        public = {
            'objective': raw['question'].strip() + ' Determine the initial relation using your observations, '
                         f'then retrieve the {small_label} and hold it outside the {container_label}. '
                         'If it is inside, extract it using real robot actions; otherwise pick it up from its observed location.',
            'success_criteria': 'The named small object starts unheld and ends held by the robot gripper with an active '
                                'native grasp constraint, its lowest bounding-box point rises at least 0.05 m above '
                                'its episode-start height, and it is outside the original named container.',
            'environment': 'Explore the original scene using robot mounted sensors and bounded controls.',
            'constraints': ['Inspect the initial relation before moving either named object; proximity or side-view occlusion '
                            'alone does not establish containment.',
                            'Use the container rim, gaps and interior views as observed evidence, and report uncertainty when needed.',
                            'Keep original assets and use real robot motion and gripper actions under the existing collision '
                            'and controller guards.'],
        }
        task = physical_task(source, context, public=public, bindings=bindings,
                             evaluation={'type': 'extract_lift', 'target': 'obj_small', 'container': 'obj_container',
                                         'minimum_lift_m': MINIMUM_LIFT_M},
                             qualification={'on_failure': 'skip', 'target_movable_collidable': True,
                                            'initial_visibility': 'none'})
        return {'scene_definition': scene, 'task_definition': task, 'status': 'generated', 'reuse_level': 'L1',
                'change_summary': ['Preserve original container/target assets, scales, poses and camera/reference mechanisms.',
                                   'Add native held/lift and actual Inside(container)==false terminal evaluation.',
                                   'Preserve original yes/no QA privately without choosing the initial relation for the agent.'],
                'warning_codes': warnings, 'assumptions': assumptions, 'private_source_conditions': conditions,
                'limitations': ['Pending static conversion does not establish reach, grasp, extraction or mounted visibility.',
                                'The existing native Inside state must be available; source AABB/QA labels are not terminal evidence.',
                                'Initially outside objects still require genuine pickup/holding and 0.05 m lift; no extraction '
                                'transition is claimed for those conditions.',
                                'Original scripted pickup/pour teleports rigid objects and changes visual_only; these '
                                'actions are excluded, and no pouring success is claimed.'],
                'attempted_repairs': [], 'next_step': 'Capture unchanged native scenes, inspect original material/cavity and '
                                                    'mounted-view evidence, then qualify bounded native reach and retrieval.'}


def qualify_operation(session, executor, config, bound):
    from .partial_occlusion import _placement_measurements, _robot_safety
    try:
        controller = getattr(executor, 'controller', None)
        if controller is None:
            raise ValueError('Native executor controller unavailable.')
        context = _robot_safety(session, include_safety=False)
        measured = _placement_measurements(session, controller, config, bound['obj_small'], bound['obj_container'])
        checks = {key: context[key] for key in ('same_source_room', 'native_r1')}
        checks['relation_telemetry_available'] = not measured['measurement_errors']
        checks['movable_collidable_target'] = not bound['obj_small'].visual_only and not bound['obj_small'].fixed_base
        return {'accepted': all(checks.values()), 'checks': checks,
            'initial_inside_native': measured['destination_relation'],
            'interior_fit_access_verified': False, 'executed_operation_verified': False}
    except (ValueError, KeyError, AttributeError, RuntimeError) as error:
        return {'accepted': False, 'reason': f'native_containment_measurement_unavailable:{error}'}


def create_evaluator(session, controller, config, output):
    from copy import deepcopy
    from .partial_occlusion import IdentificationPlacementEvaluator
    from wia.tasks.completion import progress
    from wia.authoring.io import write_json

    class StableInsideEvaluator(IdentificationPlacementEvaluator):
        def begin_episode(self, observation):
            self.initial_relation = self.measure()['destination_relation']
            return super().begin_episode(observation)

        def record(self, observation, *, action=None):
            if not self.active:
                return None
            available, stable = self._update()
            item = {'id': 'stable_inside', 'kind': 'terminal_condition',
                'completed': bool(stable and self.stable_steps >= self.parameters['release_stability_steps']),
                'evaluation_validity': 'valid' if available else 'unavailable'}
            result = {'schema': 'private_stable_inside_v1', 'snapshot_id': observation.snapshot_id,
                'action_step': observation.action_step, 'completion_items': [item],
                'task_success': item['completed'] and item['evaluation_validity'] == 'valid',
                'initial_inside_native': self.initial_relation,
                'release_stable_steps': self.stable_steps, 'measurements': self.last,
                'evaluation_validity': 'complete' if available else 'unavailable'}
            result['no_op_verified'] = bool(result['task_success'] and self.initial_relation is True
                                           and observation.action_step == self.start_step)
            result.update(progress(result))
            write_json(self.output / 'state.json', result)
            return result

    native = deepcopy(config)
    native['bindings']['target'] = config['bindings']['obj_small']
    native['bindings']['destination'] = config['bindings']['obj_container']
    return StableInsideEvaluator(session, controller, native, output)
