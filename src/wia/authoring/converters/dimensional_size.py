"""Convert original size comparisons into pending native grasp/lift tasks.

Run the whole category through ConverterBase.convert_all using SourceInstance
and ConversionContext from wia.authoring.conversion; no simulator is used.
"""
from __future__ import annotations

import math
import re
import json
from pathlib import Path

import numpy as np

from wia.authoring.conversion import ConverterBase, map_records, original_scene, physical_task
from wia.authoring.converters.interaction_profiles import policy_for


OBJECT_KEYS = ('task_obj1', 'task_obj2', 'ref_obj1', 'ref_obj2')
QUESTION = re.compile(
    r'Which (?P<task>.+?) is larger: the one near the (?P<ref1>.+?), '
    r'or the one near the (?P<ref2>.+?)\?', re.IGNORECASE)
MINIMUM_LIFT_M = 0.05
MODULE = 'wia.authoring.converters.dimensional_size'
PLACEMENT_OPERATION = 'compare_select_transport_place'
TASK_REVISION = 'task_edit_20261018_v1'


def _text(value):
    return ' '.join(str(value or '').strip().lower().replace('_', ' ').split())


def _vector(value, length):
    return (isinstance(value, (list, tuple)) and len(value) == length
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x) for x in value))


def _uniform_scale(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(value) and value > 0 else None
    if _vector(value, 3) and min(value) > 0 and math.isclose(min(value), max(value), abs_tol=1e-9):
        return float(value[0])
    return None


def _answer_pair(value, objects):
    """Resolve explicit annotations only; never use the upstream default pair."""
    normalized = _text(value)
    matches = []
    for index in (1, 2):
        reference = _text(objects[f'ref_obj{index}']['category'])
        if normalized in (f'near the {reference}', f'task obj{index}', f'ref obj{index}'):
            matches.append(index)
    return matches[0] if len(matches) == 1 else None


def revise_comparison_placement(result, source, context, *, module, selected_key, expected_answer,
                                source_objects, comparison_kind, metric_description, public_report,
                                reference_evidence=None):
    """Use the reviewed named-destination interface without inventing furniture."""
    destination = context.config.get('placement_destination')
    contract = {'preserve': ['original_target_identity', 'asset_scales', 'comparison_reference_associations',
        'source_metric', 'source_reference_camera'], 'goal': 'comparison_then_transport_and_stable_release',
        'comparison_kind': comparison_kind, 'answer_before_intervention': True}
    result.update(task_revision=TASK_REVISION, mechanism_contract=contract)
    if not isinstance(destination, dict) or not destination.get('binding', {}).get('name') or not destination.get('public_label'):
        result.update(status='needs_human_decision', task_definition=None,
            failure_reason='named_placement_destination_required', diagnostic_stage='transport_destination')
        result['next_step'] = 'Supply placement_destination with a seeded native binding, public label and relation after support/access screening.'
        return result
    relation = destination.get('relation', 'on_top')
    if relation not in ('on_top', 'inside'):
        raise ValueError('Placement destination relation must be on_top or inside.')
    task = result['task_definition']
    bindings = task['private']['bindings']
    bindings['target'] = dict(bindings[selected_key])
    bindings['destination'] = dict(destination['binding'])
    result['scene_definition']['bindings'] = bindings.copy()
    task['public'].update(objective=source.raw['question'] + ' ' + metric_description + ' ' + public_report +
        ' Select the original candidate that satisfies the comparison, transport it to ' + destination['public_label'] +
        (', place it inside, release it and withdraw.' if relation == 'inside' else ', place it on top, release it and withdraw.'),
        success_criteria='Report the correct initial comparison before moving candidates. Acquire the selected '
        'original object with the native gripper, then leave it stably released at the named destination for '
        '15 consecutive native samples below 0.02 m/s and 0.1 rad/s, with the robot withdrawn. '
        'Preserve all comparison references and the other candidate.',
        constraints=['Compare the initial arrangement through mounted robot observations before intervention.',
            'Preserve original asset scales, reference identities and all unselected comparison objects.',
            'Use guarded native manipulation; source answers do not establish physical placement.'])
    parameters = {'task_revision': TASK_REVISION, 'mechanism_contract': contract,
        'expected_identity': expected_answer, 'placement_relation': relation, 'selected_key': selected_key,
        'comparison_kind': comparison_kind, 'source_objects': source_objects,
        'source_reference_evidence': reference_evidence, 'placement_destination': destination,
        'release_stability_steps': 15, 'maximum_release_speed_m_s': .02,
        'maximum_release_angular_speed_rad_s': .1,
        'completion_items': [{'id': 'initial_comparison'}, {'id': 'correct_bound_selection'}, {'id': 'stable_placement'}]}
    task['private']['evaluation'] = {'type': 'category_operation', 'module': module,
        'operation': PLACEMENT_OPERATION, 'binding_keys': list(bindings), 'parameters': parameters}
    task['repair'] = {'task_revision': TASK_REVISION, 'mechanism_contract': contract}
    task['requirements']['capabilities'] = ['base_translation', 'base_rotation', 'ee_translation',
        'ee_rotation', 'trunk_height', 'gripper']
    result['change_summary'].append('Require an initial metric report, original-object selection and stable transport placement at a named destination.')
    result['warning_codes'].append('destination_native_support_fit_and_access_pending')
    return result


def comparison_measurements(session, config):
    """Read source association and scale evidence before any target intervention."""
    from wia.tasks.interaction_setup import array, reference_position
    p = config['parameters']
    values = {}
    for key, original in p['source_objects'].items():
        if key not in config['bindings']:
            continue
        obj = session.env.scene.object_registry('name', config['bindings'][key])
        position, quaternion = (array(value) for value in obj.get_position_orientation())
        scale = array(obj.scale)
        values[key] = {'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(), 'scale': scale.tolist(),
            'source_scale_match': bool(np.allclose(scale, original.get('scale', 1.), atol=1e-6, rtol=1e-5)),
            'source_pose_match': bool(np.linalg.norm(position - reference_position(session, obj.name, original['position'])) <= .02)}
    if len(values) != len(p['source_objects']):
        raise ValueError('Every original comparison object must retain its native binding.')
    checks = {'source_scales': all(row['source_scale_match'] for row in values.values()),
        'source_associations': all(row['source_pose_match'] for row in values.values())}
    if p['comparison_kind'] == 'candidate_root_xy_to_reference_aabb_center':
        from .metric_comparison_spatial_distance import resolve_fixed_distance_reference
        anchor, evidence = resolve_fixed_distance_reference(session, p['source_reference_evidence'])
        checks['source_reference_qualified'] = evidence['accepted']
        if anchor is not None:
            center = (array(anchor.aabb[0]) + array(anchor.aabb[1]))[:2] / 2
            distances = {key: float(np.linalg.norm(np.asarray(row['position'])[:2] - center)) for key, row in values.items()}
            checks['source_distance_order'] = distances[p['selected_key']] < min(
                value for key, value in distances.items() if key != p['selected_key'])
    return {'objects': values, 'checks': checks, 'accepted': bool(all(checks.values()))}


def qualify_operation(session, executor, config, bound):
    try:
        from .partial_occlusion import _placement_measurements
        controller = getattr(executor, 'controller', executor)
        comparison = comparison_measurements(session, config)
        placement = _placement_measurements(session, controller, config, bound['target'], bound['destination'])
        checks = {'source_comparison': comparison['accepted'], 'target_movable_collidable': not bound['target'].visual_only
            and not bound['target'].fixed_base, 'destination_collidable': not bound['destination'].visual_only,
            'placement_measurements': not placement['measurement_errors']}
        return {'accepted': bool(all(checks.values())), 'checks': checks, 'source_comparison': comparison,
            'native_destination_access_verified': False, 'native_execution_verified': False}
    except Exception as error:
        return {'accepted': False, 'reason': 'comparison_placement_measurement_unavailable', 'exception_summary': str(error)}


from wia.authoring.converters.partial_occlusion import IdentificationPlacementEvaluator


class ComparisonPlacementEvaluator(IdentificationPlacementEvaluator):
    """Require the comparison answer before disturbing its original evidence."""
    def begin_episode(self, observation):
        self.initial_comparison = comparison_measurements(self.session, self.config)
        self.intervened_before_answer = False
        self.references_preserved = True
        return super().begin_episode(observation)

    def _update(self, *, count_step=False):
        measured = comparison_measurements(self.session, self.config)
        selected = self.parameters['selected_key']
        for key, row in measured['objects'].items():
            initial = self.initial_comparison['objects'][key]
            moved = np.linalg.norm(np.asarray(row['position']) - initial['position']) > .02
            current_q, initial_q = np.asarray(row['quaternion_xyzw']), np.asarray(initial['quaternion_xyzw'])
            rotated = abs(float(np.dot(current_q, initial_q) / (np.linalg.norm(current_q) * np.linalg.norm(initial_q)))) < math.cos(math.radians(2.5))
            if self.answer is None and moved:
                self.intervened_before_answer = True
            if key != selected and (moved or rotated):
                self.references_preserved = False
            if not row['source_scale_match']:
                self.references_preserved = False
        if self.parameters['comparison_kind'] == 'candidate_root_xy_to_reference_aabb_center':
            self.references_preserved &= measured['checks'].get('source_reference_qualified', False)
        result = super()._update(count_step=count_step)
        gripper = self.last.get('gripper') or {}
        comparison_names = {self.config['bindings'][key] for key in self.parameters['source_objects']}
        if self.answer is None and gripper.get('holding') and gripper.get('object_name') in comparison_names:
            self.intervened_before_answer = True
        return result

    def report_answer(self, answer, observation):
        self._update()
        super().report_answer(answer, observation)

    def record(self, observation, *, action=None):
        if not self.active:
            return None
        error = None
        try:
            available, stable = self._update()
        except Exception as exception:
            available, stable, error = False, False, str(exception)
        valid_comparison = bool(self.initial_comparison['accepted'])
        identity = bool(self.answer == self.parameters['expected_identity'] and not self.intervened_before_answer)
        items = [{'id': 'initial_comparison', 'completed': identity, 'kind': 'semantic_answer',
            'evaluation_validity': 'valid' if valid_comparison else 'unavailable'},
            {'id': 'correct_bound_selection', 'completed': self.selected, 'kind': 'checkpoint',
                'evaluation_validity': 'valid' if self.selected or self.selection_history_complete else 'unavailable'},
            {'id': 'stable_placement', 'completed': bool(stable and self.stable_steps >= self.parameters['release_stability_steps']),
                'kind': 'terminal_condition', 'evaluation_validity': 'valid' if available else 'unavailable'}]
        value = {'schema': 'private_comparison_transport_placement_v1', 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'action': action, 'completion_items': items,
            'evaluation_validity': 'complete' if available and valid_comparison else 'unavailable',
            'task_success': bool(available and valid_comparison and self.references_preserved and all(item['completed'] for item in items)),
            'constraints_satisfied': bool(self.references_preserved and not self.intervened_before_answer),
            'answer_history': list(self.answer_history), 'initial_comparison': self.initial_comparison,
            'intervened_before_answer': self.intervened_before_answer, 'measurement_error': error,
            'measurements': self.last, 'release_stable_steps': self.stable_steps, 'task_revision': TASK_REVISION}
        from wia.tasks.completion import progress
        from wia.authoring.io import write_json
        value.update(progress(value))
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    return ComparisonPlacementEvaluator(session, controller, config, output)


class DimensionalSizeConverter(ConverterBase):
    category = ('Metric Comparison', 'Dimensional Size')
    version = '3.0.0_task_edit_20261018_v1'

    def _evidence(self, source):
        raw, payload = source.raw, source.payload
        if (source.big_task, source.small_task) != self.category or (
                raw.get('big_task'), raw.get('small_task')) != self.category:
            raise ValueError('source_category_mismatch')
        question = raw.get('question')
        match = QUESTION.fullmatch(question.strip()) if isinstance(question, str) else None
        if match is None:
            raise ValueError('unsupported_public_size_question')
        objects = map_records(payload.get('objects'))
        for key in OBJECT_KEYS:
            obj = objects.get(key)
            if not isinstance(obj, dict) or not obj.get('category') or not obj.get('model'):
                raise ValueError(f'missing_{key}_category_or_model')
            if obj['model'] == 'scene_native':
                raise ValueError(f'{key}_omitted_by_original_builder')
            if not _vector(obj.get('position'), 3):
                raise ValueError(f'missing_or_invalid_{key}_position')
            quaternion = obj.get('quaternion_xyzw', [0., 0., 0., 1.])
            if not _vector(quaternion, 4) or not any(quaternion):
                raise ValueError(f'invalid_{key}_quaternion')
        names = [str(objects[key].get('name') or key).strip() for key in OBJECT_KEYS]
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError('ambiguous_original_builder_names')
        first, second = (objects[f'task_obj{index}'] for index in (1, 2))
        if first['category'] != second['category'] or first['model'] != second['model']:
            raise ValueError('unsupported_nonidentical_size_comparison_assets')
        expected = {'task': first['category'], 'ref1': objects['ref_obj1']['category'],
                    'ref2': objects['ref_obj2']['category']}
        if any(_text(match.group(key)) != _text(value) for key, value in expected.items()):
            raise ValueError('public_question_object_labels_mismatch')
        if payload.get('task_category') and _text(payload['task_category']) != _text(first['category']):
            raise ValueError('private_task_category_mismatch')
        if _text(objects['ref_obj1']['category']) == _text(objects['ref_obj2']['category']):
            raise ValueError('ambiguous_identical_reference_labels')
        scales = [_uniform_scale(obj.get('scale', 1.)) for obj in (first, second)]
        if any(scale is None for scale in scales) or math.isclose(*scales, rel_tol=1e-9):
            raise ValueError('unsupported_or_tied_comparison_scales')
        pair = _answer_pair(raw.get('answer'), objects)
        if pair is None:
            raise ValueError('unsupported_private_size_answer')
        for field in ('answer', '_ground_truth'):
            if field in payload and _answer_pair(payload[field], objects) != pair:
                raise ValueError(f'conflicting_private_{field}_annotation')
        geometric_pair = 1 if scales[0] > scales[1] else 2
        if pair != geometric_pair:
            raise ValueError('private_answer_conflicts_with_original_uniform_scales')
        cameras = payload.get('camera_poses')
        if not isinstance(cameras, list) or not cameras:
            raise ValueError('missing_original_camera_poses')
        for camera in cameras:
            if (not isinstance(camera, dict) or not _vector(camera.get('position'), 3)
                    or not _vector(camera.get('quaternion_xyzw'), 4)):
                raise ValueError('invalid_original_camera_pose')
        return raw, objects, pair, scales

    def inspect(self, source):
        try:
            self._evidence(source)
        except (ValueError, KeyError, TypeError) as error:
            return {'eligible': False, 'reason': str(error),
                    'warning_codes': ['not_convertible_under_L1_L2', 'size_annotation_review_required']}
        return {'eligible': True, 'reason': 'Original public comparison and private size annotations agree.'}

    def convert(self, source, context):
        result = self.convert_legacy(source, context)
        if context.config.get('operation') == 'legacy_grasp_lift':
            return result
        raw, objects, pair, scales = self._evidence(source)
        return revise_comparison_placement(result, source, context, module=MODULE,
            selected_key=f'task_obj{pair}', expected_answer='near the ' + _text(objects[f'ref_obj{pair}']['category']),
            source_objects=objects, comparison_kind='identical_asset_uniform_linear_scale',
            metric_description='Compare the identical asset copies by their uniform linear dimensions; all dimensions scale together.',
            public_report='Report "near the ' + _text(objects['ref_obj1']['category']) + '" or "near the ' + _text(objects['ref_obj2']['category']) + '" before moving either copy.')

    def convert_legacy(self, source, context):
        raw, objects, pair, scales = self._evidence(source)
        bindings = {key: {'name': str(objects[key].get('name') or key).strip(),
                          'category': objects[key]['category'], 'model': objects[key]['model']}
                    for key in OBJECT_KEYS}
        # Native head extrinsics are captured before source postprocessing by the
        # shared adapter and restored exactly; no world pose or geometry is guessed.
        patches = [{'operation': 'restore_robot_head_mount', 'phase': 'after_postprocessing'}]
        evaluation = {'type': 'grasp_lift', 'target': f'task_obj{pair}',
                      'minimum_lift_m': MINIMUM_LIFT_M}
        public = {
            'objective': raw['question'].strip() + ' Select the larger copy using your observations, '
                         'then grasp and lift that copy with the robot.',
            'success_criteria': 'The selected larger copy starts unheld and ends held by the robot gripper '
                                'with an active native grasp constraint; its lowest bounding-box point '
                                'must rise at least 0.05 m above its episode-start height.',
            'environment': 'Explore the original scene using the robot mounted sensors and bounded controls.',
            'constraints': ['Preserve both task objects and their reference objects until the comparison is made.',
                            'Use observed size evidence to compare the copies; change viewpoints as needed to resolve perspective distortion.',
                            'Use real robot movement and gripper actions under the existing collision and controller guards.'],
        }
        scene = original_scene(source, context, bindings=bindings, patches=patches)
        task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings,
                             patches=patches, qualification={'on_failure': 'skip',
                                 'target_movable_collidable': True, 'initial_visibility': 'none'})
        # Each task/reference pair retains its geometry; the two pairs can be
        # arranged independently on an existing table without changing sizes.
        interaction_policy = policy_for(source, task)
        # Keep the reversible native source variant alongside the adapted variant.
        for definition in (scene, task):
            definition['scene']['variants'].setdefault('original', {'patches': []})
        extreme = [key for key in OBJECT_KEYS if abs(objects[key]['position'][2]) > 100.]
        warnings = ['assumption_review_required', 'native_grasp_unverified',
                    'mounted_sensor_coverage_unverified', 'source_robot_eye_view_changed']
        if extreme:
            warnings.append('source_reference_geometry_outlier_requires_review')
        return {
            'scene_definition': scene, 'task_definition': task, 'status': 'generated', 'reuse_level': 'L2',
            'interaction_policy': interaction_policy,
            'change_summary': ['Preserve original assets, scales, QA and each task/reference pair geometry.',
                               'Arrange the two unchanged pairs on an existing measured same-room support; '
                               'screen a floor stance for each copy without a whole-group centre restriction.',
                               'Retain the original variant and label adapted source-camera translation as approximate.',
                               'Restore the exact native R1 head sensor parent-frame mount after source postprocessing.',
                               'Add a separate native grasp/lift predicate for the privately annotated larger copy.'],
            'warning_codes': warnings,
            'source_group_geometry_evidence': {
                'original_positions_m': {key: objects[key]['position'] for key in OBJECT_KEYS},
                'root_xy_span_lower_bound_m': [max(objects[key]['position'][axis] for key in OBJECT_KEYS) -
                    min(objects[key]['position'][axis] for key in OBJECT_KEYS) for axis in (0, 1)],
                'extreme_source_world_z_bindings': extreme,
                'source_positions_repaired': False,
                'interpretation': 'Original source roots are preserved verbatim. Extreme reference coordinates '
                                  'require Review; no support fit, native state or corrected reference is inferred.'},
            'assumptions': [{
                'doubt': 'The upstream source moves the robot eye sensor by world [0.05, 0.2, -0.3] m '
                         'and its original QA pickup may rely on that displaced view.',
                'chosen_assumption': 'Restore the exact native sensor mount captured before postprocessing '
                                     'for a consistent physical R1 embodiment; keep all viewer reference poses.',
                'alternatives': ['Use the unpatched original variant and retain the upstream displaced robot eye.',
                                 'Defer adapted operation until mounted sensor coverage is verified.'],
                'scope': 'All Dimensional Size candidates; reversible sensor patch plus a declarative '
                         'pair-preserving support/stance policy, with both task copies screened.',
                'risk': 'The robot may lose an intended inspection viewpoint or remain unable to reach/grasp the '
                        'source-sized floor objects. CPU conversion and historical bounded controls do not establish success.',
            }, {
                'doubt': 'Keeping the original spacing between the two comparison pairs can prevent '
                         'use of a table even though each pair fits and absolute spacing is not the size question.',
                'chosen_assumption': 'Translate each task/reference pair together onto one existing same-room '
                                     'support. Preserve every asset dimension, pair-relative pose and reference '
                                     'association; allow inter-pair spacing to change.',
                'alternatives': ['Keep the original floor layout as a diagnostic variant.',
                                 'Use a larger existing support that fits a whole-group translation.'],
                'scope': 'Only initialization of Dimensional Size; no change to distance or contact categories.',
                'risk': 'The adapted perspective differs from original QA frames; source viewer translation '
                        'by the mean pair displacement is approximate. Robot paths, IK and grasp remain unverified.',
            }],
            'limitations': ['Pending candidates are statically validated, not native load, reach or grasp qualified.',
                            'Historical source pickup teleports the robot/object and creates a mock constraint; '
                            'it is not used for the physical task or accepted as terminal evidence.',
                            'Original QA remains private auxiliary evidence. Human review images are not agent observations.',
                            'Pair placement is a request, not evidence: native support, paths, mounted-view coverage, '
                            'IK, gripper aperture and grasp feasibility remain unverified.'],
            'attempted_repairs': ['Declarative restoration of the captured native head mount; native execution not run.'],
            'next_step': 'Capture original and adapted diagnostic views, inspect mounted-camera coverage and '
                         'qualify unchanged-size reach and native grasp before approving execution.',
            'private_size_evidence': {'target_pair': pair, 'original_uniform_scales': scales,
                                      'larger_to_smaller_scale_ratio': max(scales) / min(scales),
                                      'selection_rule': 'Explicit source annotations agree with identical-asset uniform scales.',
                                      'historical_head_parent_offset_m': [0.23889, -0.24503, -0.12406],
                                      'native_execution_verified': False},
        }
