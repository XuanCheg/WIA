"""Preserve Spatial Distance scenes and add private selected-object lift goals.

Run CategoryConverter().convert_all(sources, ConversionContext(output_root))
from the repository root, or use the shared category campaign CPU runner.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np

from wia.authoring.conversion import (
    ConverterBase, map_records, original_scene, physical_task,
)


def _normalize(value):
    return str(value or '').strip().lower().replace(' ', '_')


def _vector(value, length):
    return isinstance(value, (list, tuple)) and len(value) == length and all(
        isinstance(item, (int, float)) and math.isfinite(item) for item in value)


def _options(raw):
    value = raw.get('options_json', [])
    return json.loads(value) if isinstance(value, str) else value


def _public(raw):
    """Read public source fields only; private labels cannot select a public name."""
    choices = _options(raw)
    return {
        'objective': 'Using the initial arrangement, select the closer object in this comparison, '
            'grasp it and lift it at least 5 cm. ' + raw['question'],
        'success_criteria': 'The correctly selected candidate is held by the robot, and its lowest '
            'point has risen at least 5 cm from its initial height.',
        'environment': 'The original reference and candidate objects are described in the question. '
            'Candidate choices: ' + '; '.join(choices) + '.',
        'constraints': ['Judge the original distances before moving either candidate.',
            'Keep the reference object in its original position.',
            'Use robot motion and gripper interaction for the grasp and lift.'],
    }


def _distance_evidence(payload):
    objects = map_records(payload['objects'])
    reference = objects.get('obj_ref', {})
    evidence = {'reference_category': payload.get('ref_category'),
        'reference_metadata': reference,
        'recorded_near_centre_m': payload.get('dist_near_centre'),
        'recorded_far_centre_m': payload.get('dist_far_centre'),
        'recorded_near_corner_offset_m': payload.get('d_near'),
        'recorded_far_corner_offset_m': payload.get('d_far'),
        'reference_native_name': reference.get('name'),
        'reference_is_bound': False}
    if all(isinstance(payload.get(key), (int, float)) and math.isfinite(payload[key])
            for key in ('d_near', 'd_far')):
        evidence['planned_corner_order_agrees'] = payload['d_near'] < payload['d_far']
    if _vector(reference.get('position'), 3):
        for role in ('near', 'far'):
            evidence[f'saved_root_xy_{role}_m'] = math.dist(
                objects[f'obj_{role}']['position'][:2], reference['position'][:2])
    if all(isinstance(payload.get(key), (int, float)) and math.isfinite(payload[key])
            for key in ('dist_near_centre', 'dist_far_centre')):
        evidence['recorded_centre_order_agrees'] = payload['dist_near_centre'] < payload['dist_far_centre']
    if 'saved_root_xy_near_m' in evidence:
        evidence['saved_root_order_agrees'] = evidence['saved_root_xy_near_m'] < evidence['saved_root_xy_far_m']
    top_down = next((camera for camera in payload.get('camera_poses', [])
        if camera.get('type') == 'top_down' and _vector(camera.get('position'), 3)), None)
    corner = payload.get('near_corner_xy')
    other_corner = payload.get('far_corner_xy')
    if top_down and _vector(corner, 2) and _vector(other_corner, 2):
        centre = top_down['position'][:2]
        half_extent = [abs(corner[axis] - centre[axis]) for axis in range(2)]
        other_half_extent = [abs(other_corner[axis] - centre[axis]) for axis in range(2)]
        # Native AABB tensors are float32; their averaged centre can round by one ULP.
        rounding_tolerance = [2 ** -22 * max(1.0, abs(centre[axis]), abs(corner[axis]),
            abs(other_corner[axis])) for axis in range(2)]
        if all(value > 0 for value in half_extent) and all(abs(value - other) <= tolerance
                for value, other, tolerance in zip(half_extent, other_half_extent, rounding_tolerance)):
            evidence['reference_aabb_xy_reconstruction'] = {
                'centre_xy': centre, 'half_extents_xy': half_extent,
                'corner_symmetry_float32_tolerance_m': rounding_tolerance,
                'provenance': 'The original generator places the top_down camera exactly above '
                    'the reference AABB centre and records its sampled AABB corners.',
                'source_formula': 'batch_distance.py render_and_save: topdown_eye=[cx,cy,ceiling_z]; '
                    'place_comparison_objects: reference centre=(aabb_min+aabb_max)/2.',
                'assumption': 'Saved camera and corner metadata share the unchanged generator '
                    'reference AABB; candidate roots approximate comparison points, not full surfaces.',
            }
            for role in ('near', 'far'):
                point = objects[f'obj_{role}']['position'][:2]
                evidence[f'saved_aabb_centre_{role}_m'] = math.dist(point, centre)
                evidence[f'saved_point_to_aabb_{role}_m'] = math.sqrt(sum(
                    max(abs(point[axis] - centre[axis]) - half_extent[axis], 0.0) ** 2
                    for axis in range(2)))
            evidence['saved_aabb_centre_order_agrees'] = (
                evidence['saved_aabb_centre_near_m'] < evidence['saved_aabb_centre_far_m'])
            evidence['saved_point_to_aabb_order_agrees'] = (
                evidence['saved_point_to_aabb_near_m'] < evidence['saved_point_to_aabb_far_m'])
    return evidence


def resolve_fixed_distance_reference(session, source_evidence, *, position_tolerance_m=.02,
                                     heading_tolerance_deg=3., bounds_tolerance_m=.02):
    """Resolve one unchanged native anchor from private source geometry, never QA.

    Return ``(native_object_or_None, JSON_evidence)``. Missing, moved or ambiguous
    anchors fail closed. The caller must separately check candidate XY ordering
    before/after a common Z-only support lift; this function moves nothing and
    awards no physical goal credit.
    """
    evidence = {'accepted': False, 'candidate_count': 0,
                'reason': 'fixed_distance_reference_unresolved'}
    try:
        from wia.tasks.interaction_setup import array as native_array
        metadata = source_evidence['reference_metadata']
        reconstructed = source_evidence['reference_aabb_xy_reconstruction']
        original_position = np.asarray(metadata['position'], dtype=float)
        original_quaternion = np.asarray(metadata['quaternion_xyzw'], dtype=float)
        centre = np.asarray(reconstructed['centre_xy'], dtype=float)
        half = np.asarray(reconstructed['half_extents_xy'], dtype=float)
        if (original_position.shape != (3,) or original_quaternion.shape != (4,)
                or centre.shape != (2,) or half.shape != (2,) or np.any(half <= 0)
                or not all(np.isfinite(value).all() for value in (
                    original_position, original_quaternion, centre, half))
                or np.linalg.norm(original_quaternion) == 0
                or not all(math.isfinite(value) and value >= 0 for value in (
                    position_tolerance_m, heading_tolerance_deg, bounds_tolerance_m))):
            raise ValueError('Invalid source anchor geometry or resolution tolerance.')
        category = metadata.get('category') or source_evidence['reference_category']
        name = metadata.get('name') or source_evidence.get('reference_native_name')
        values = session.env.scene.objects
        objects = list(values.values()) if isinstance(values, dict) else list(values)
        matches, rejections = [], []
        for obj in objects:
            if _normalize(obj.category) != _normalize(category) or (name and obj.name != name):
                continue
            position, quaternion = (native_array(value) for value in obj.get_position_orientation())
            low, high = (native_array(value) for value in obj.aabb)
            if (position.shape != (3,) or quaternion.shape != (4,) or low.shape != (3,)
                    or high.shape != (3,) or np.any(high <= low)
                    or not all(np.isfinite(value).all() for value in (position, quaternion, low, high))
                    or np.linalg.norm(quaternion) == 0):
                rejections.append({'name': obj.name, 'reason': 'invalid_native_anchor_geometry'})
                continue
            cosine = abs(float(np.dot(quaternion / np.linalg.norm(quaternion),
                                      original_quaternion / np.linalg.norm(original_quaternion))))
            angle = math.degrees(2 * math.acos(float(np.clip(cosine, 0., 1.))))
            position_error = float(np.linalg.norm(position - original_position))
            bounds_error = float(max(np.max(np.abs(low[:2] - (centre - half))),
                                     np.max(np.abs(high[:2] - (centre + half)))))
            row = {'name': obj.name, 'position_m': position.tolist(),
                   'quaternion_xyzw': quaternion.tolist(), 'bounds_m': [low.tolist(), high.tolist()],
                   'source_position_error_m': position_error, 'source_heading_error_deg': angle,
                   'source_xy_bounds_error_m': bounds_error}
            if (position_error <= position_tolerance_m and angle <= heading_tolerance_deg
                    and bounds_error <= bounds_tolerance_m and not obj.visual_only):
                matches.append((obj, row))
            else:
                rejections.append({**row, 'reason': 'native_anchor_differs_from_source'})
        evidence.update(candidate_count=len(matches), rejected_candidates=rejections,
                        source_reference_category=category)
        if len(matches) != 1:
            evidence['reason'] = ('fixed_distance_reference_ambiguous' if matches
                                  else 'fixed_distance_reference_unresolved')
            return None, evidence
        obj, row = matches[0]
        evidence.update(accepted=True, reason=None, reference=row)
        return obj, evidence
    except (KeyError, ValueError, TypeError, AttributeError, RuntimeError, OverflowError) as error:
        evidence.update(reason='fixed_distance_reference_measurement_unavailable',
                        exception_summary=f'{type(error).__name__}: {error}')
        return None, evidence


MODULE = 'core.esi_converters.metric_comparison_spatial_distance'


class CategoryConverter(ConverterBase):
    category = ('Metric Comparison', 'Spatial Distance')
    version = '3.0.0_task_edit_20261018_v1'

    def sample(self, manifest, policy=None):
        selected = super().sample(manifest, policy)
        by_id = {row['candidate_id']: row for row in manifest['records']}
        for warning in ('distance_metric_order_ambiguity', 'comparison_pose_phase_disagreement'):
            matching = sorted(row['candidate_id'] for row in manifest['records'] if
                row['status'] == 'generated' and warning in row['warning_codes'])
            if matching and selected and not set(matching).intersection(selected):
                replaceable = [index for index, candidate_id in enumerate(selected) if
                    by_id[candidate_id]['status'] == 'generated' and not any(code in
                        by_id[candidate_id]['warning_codes'] for code in
                        ('distance_metric_order_ambiguity', 'comparison_pose_phase_disagreement'))]
                if replaceable:
                    selected[replaceable[-1]] = matching[0]
        return selected

    def inspect(self, source):
        raw, payload = source.raw, source.payload
        if (source.big_task, source.small_task) != self.category or (
                raw.get('big_task'), raw.get('small_task')) != self.category:
            return {'eligible': False, 'reason': 'category_mismatch'}
        question = raw.get('question', '')
        match = re.fullmatch(r'Which object is closer to the (.+?): the (.+?) or the (.+?)\?', question)
        if not match:
            return {'eligible': False, 'reason': 'unsupported_public_distance_question'}
        options = _options(raw)
        if not isinstance(options, list) or len(options) != 2 or not all(
                isinstance(item, str) and item.strip() for item in options):
            return {'eligible': False, 'reason': 'invalid_public_candidate_options'}
        public_categories = {_normalize(value) for value in options}
        if len(public_categories) != 2 or public_categories != {_normalize(match[2]), _normalize(match[3])}:
            return {'eligible': False, 'reason': 'public_question_options_disagree'}
        objects = map_records(payload.get('objects'))
        for role in ('near', 'far'):
            info = objects.get(f'obj_{role}', {})
            if not info.get('category') or not info.get('model') or info['model'] == 'scene_native':
                return {'eligible': False, 'reason': f'missing_movable_candidate_asset:{role}'}
            if not _vector(info.get('position'), 3) or not _vector(info.get('quaternion_xyzw'), 4) or not any(
                    info['quaternion_xyzw']):
                return {'eligible': False, 'reason': f'missing_original_candidate_pose:{role}'}
            if _normalize(info['category']) != _normalize(payload.get(f'{role}_category')):
                return {'eligible': False, 'reason': f'candidate_category_annotation_disagreement:{role}'}
        if public_categories != {_normalize(objects[f'obj_{role}']['category']) for role in ('near', 'far')}:
            return {'eligible': False, 'reason': 'public_private_candidate_categories_disagree'}
        if _normalize(match[1]) != _normalize(payload.get('ref_category')):
            return {'eligible': False, 'reason': 'public_private_reference_category_disagree'}
        if _normalize(raw.get('answer')) != _normalize(objects['obj_near']['category']) or payload.get('answer') != 'obj_near':
            return {'eligible': False, 'reason': 'source_answer_near_role_disagreement',
                'warning_codes': ['source_annotation_conflict']}
        if not payload.get('camera_poses'):
            return {'eligible': False, 'reason': 'missing_original_reference_camera'}
        return {'eligible': True, 'reason': 'original_candidates_and_public_comparison_preserved'}

    def convert(self, source, context):
        result = self.convert_legacy(source, context)
        if context.config.get('operation') == 'legacy_grasp_lift' or result['task_definition'] is None:
            return result
        evidence = result['source_distance_evidence']
        if evidence.get('saved_aabb_centre_order_agrees') is not True or evidence.get('recorded_centre_order_agrees') is not True:
            result.update(status='needs_human_decision', task_definition=None,
                failure_reason='source_center_distance_comparison_unresolved', diagnostic_stage='source_distance_consistency')
            result['warning_codes'].append('source_center_distance_comparison_unresolved')
            return result
        from .dimensional_size import revise_comparison_placement
        objects = map_records(source.payload['objects'])
        comparison_objects = {'selected_candidate': objects['obj_near'], 'other_candidate': objects['obj_far']}
        return revise_comparison_placement(result, source, context, module=MODULE,
            selected_key='selected_candidate', expected_answer=objects['obj_near']['category'].replace('_', ' ').lower(),
            source_objects=comparison_objects, comparison_kind='candidate_root_xy_to_reference_aabb_center',
            metric_description='Use horizontal distance from each candidate object root to the center of the original reference furniture bounding box, matching the source comparison convention.',
            public_report='Report the category of the closer candidate before grasping or moving either candidate.',
            reference_evidence=evidence)

    def convert_legacy(self, source, context):
        raw, payload = source.raw, source.payload
        objects = map_records(payload['objects'])
        bindings = {binding: {'name': info.get('name') or source_key,
            'category': info['category'], 'model': info['model']}
            for binding, source_key in (('selected_candidate', 'obj_near'), ('other_candidate', 'obj_far'))
            for info in [objects[source_key]]}
        evidence = _distance_evidence(payload)
        warnings = ['native_grasp_lift_unverified', 'scene_native_reference_unresolved',
            'assumption_review_required']
        assumptions = [{
            'doubt': 'The scene-native reference has category and pose metadata but no recorded '
                'native name; its loaded identity and comparison geometry have not been qualified.',
            'chosen_assumption': 'Retain the original room, fixed reference and candidate XY arrangement. '
                'Permit one common vertical translation onto separate measured accessible supports. '
                'Keep anchor metadata private and bind only the '
                'two original candidate objects actually created by the source builder.',
            'alternatives': ['Resolve the reference against a native inventory using source '
                'category and pose before physical qualification.', 'Defer execution if the '
                'original reference cannot be resolved without changing the comparison.'],
            'scope': source.relative_path,
            'risk': 'A missing, moved or ambiguous reference may invalidate the original distance '
                'comparison. Source visibility flags do not certify native reference visibility.',
        }]
        if 'reference_aabb_xy_reconstruction' not in evidence:
            warnings.append('reference_geometry_reconstruction_unavailable')
        if 'reference_aabb_xy_reconstruction' in evidence:
            assumptions.append({
                'doubt': 'The cleaned source omits the reference AABB bounds and native identity, '
                    'and its public question does not specify centre versus nearest-surface distance.',
                'chosen_assumption': 'Reconstruct only a private planar AABB diagnostic from the '
                    'source top-down camera centre and sampled corners, whose relationship is '
                    'explicit in the generator. Keep intended corner-offset labels authoritative '
                    'unless the saved geometry cannot be reconciled conservatively.',
                'alternatives': ['Measure actual original reference geometry and candidate surfaces '
                    'during native qualification.', 'Keep ambiguous metric cases for human guidance.'],
                'scope': source.relative_path,
                'risk': 'Reconstructed root-point distances do not certify physical surface '
                    'distances or a native reference match; this is not native label revalidation.',
            })
        if evidence.get('recorded_centre_order_agrees') is False:
            warnings.append('pre_render_centre_order_disagreement')
        if evidence.get('saved_aabb_centre_order_agrees') is False and evidence.get(
                'saved_point_to_aabb_order_agrees') is True:
            warnings.append('distance_metric_order_ambiguity')
            assumptions.append({
                'doubt': 'Saved-point distance to the reconstructed AABB supports the near label, '
                    'while distance to its centre reverses that selection.',
                'chosen_assumption': 'Preserve the generator corner-offset selection as a pending '
                    'candidate, warn that the public comparison metric needs Review, and require '
                    'native reference/surface evidence before execution.',
                'alternatives': ['Clarify the original intended distance metric without changing '
                    'source QA.', 'Keep the instance diagnostic-only if the ambiguity is unresolved.'],
                'scope': source.relative_path,
                'risk': 'A centre-distance interpretation would select a different candidate; '
                    'public wording and native geometry have not been revalidated.',
            })
        if 'saved_aabb_centre_order_agrees' in evidence and evidence.get('recorded_centre_order_agrees') != evidence[
                'saved_aabb_centre_order_agrees']:
            warnings.append('comparison_pose_phase_disagreement')
            assumptions.append({
                'doubt': 'The generator centre-distance order recorded after placement differs '
                    'from that calculated from object poses saved after its camera rendering.',
                'chosen_assumption': 'Keep the actual saved source poses used by the active builder '
                    'and retain both phases as private evidence; do not reset an object to an earlier pose.',
                'alternatives': ['Reproduce source settling phases in an independent native '
                    'diagnostic.', 'Keep the source diagnostic-only if its intended geometry is lost.'],
                'scope': source.relative_path,
                'risk': 'Simulation drift during generation can alter the intended comparison '
                    'before active loading. No native reproduction has been run.',
            })
        if evidence.get('saved_root_order_agrees') is False:
            warnings.append('reference_root_distance_order_disagreement')
            assumptions.append({
                'doubt': 'Distances to the saved reference root reverse the source near/far order; '
                    'the generator measures distances to the reference AABB centre instead.',
                'chosen_assumption': 'Preserve intended generator corner-offset labels and retain '
                    'reference-root distances separately from AABB-centre and boundary diagnostics; '
                    'do not change either candidate or the answer.',
                'alternatives': ['Measure the exact original reference AABB in native initialization.',
                    'Treat the instance as diagnostic-only if native geometry contradicts the label.'],
                'scope': source.relative_path,
                'risk': 'The source root is not necessarily its AABB centre, and both the anchor '
                    'identity and current native comparison remain unverified.',
            })
        result = {'scene_definition': original_scene(source, context, bindings=bindings),
            'task_definition': None, 'status': 'generated', 'reuse_level': 'L2',
            'change_summary': ['Preserve original candidate assets, scales, XY positions, reference '
                'furniture and source camera protocol.', 'Keep original question, options and QA; '
                'select the near candidate only in private evaluation.',
                'Request separate physical supports at unchanged candidate XY with one common Z translation; '
                'require unchanged native anchor and source-consistent horizontal distance ordering.'],
            'warning_codes': warnings, 'assumptions': assumptions,
            'source_distance_evidence': evidence,
            'limitations': ['Native reference identity, sensor coverage, reach, grasp and lift '
                'remain unqualified.', 'The source builder skips the scene_native reference; '
                'no invented obj_ref or named furniture binding is introduced.',
                'Original free-camera actions and source answer labels are not physical success.'],
            'attempted_repairs': [],
            'next_step': 'Review the preserved distance geometry, resolve the scene-native reference '
                'and qualify real robot grasp/lift before marking the task ready.'}
        unresolved_geometry = (evidence.get('saved_aabb_centre_order_agrees') is False and
            evidence.get('saved_point_to_aabb_order_agrees') is False)
        if evidence.get('planned_corner_order_agrees') is False or unresolved_geometry:
            result.update(status='needs_human_decision',
                failure_reason='source_saved_geometry_cannot_reconcile_near_label',
                diagnostic_stage='source_distance_consistency')
            warnings.append('source_distance_geometry_conflict')
            assumptions.append({
                'doubt': 'The source answer follows intended near corner offsets, but both saved '
                    'reference-AABB-centre and point-to-AABB distances reverse that selection. '
                    'The physical question cannot be resolved faithfully from current metadata.',
                'chosen_assumption': 'Preserve an original-scene diagnostic record without '
                    'creating a robot task that would privately reward a contradictory selection.',
                'alternatives': ['Obtain native evidence resolving the original comparison metric '
                    'and reference identity.', 'Skip physical conversion if the source contradiction '
                    'cannot be reconciled while preserving the original answer.'],
                'scope': source.relative_path,
                'risk': 'Automatically choosing either label or distance would silently change '
                    'the original question meaning or reward the wrong physical selection.',
            })
            result['next_step'] = ('Review the inconsistent private distance evidence and source '
                'answer. Resolve the original metric and saved geometry before adding a physical task.')
            return result
        if 'recorded_centre_order_agrees' not in evidence:
            warnings.append('recorded_distance_evidence_missing')
        patches = [
            {'operation': 'object_physics', 'phase': 'before_construction',
             'binding': key, 'visual_only': False, 'fixed_base': False}
            for key in bindings
        ] + [
            {'operation': 'object_physics', 'phase': 'after_postprocessing',
             'binding': key, 'visual_only': False}
            for key in bindings
        ]
        result['scene_definition'] = original_scene(source, context, bindings=bindings, patches=patches)
        result['task_definition'] = physical_task(source, context, public=_public(raw),
            evaluation={'type': 'grasp_lift', 'target': 'selected_candidate', 'minimum_lift_m': 0.05},
            bindings=bindings, patches=patches,
            qualification={'on_failure': 'skip', 'initial_visibility': 'none', 'target_movable_collidable': True})
        from .interaction_profiles import policy_for
        result['interaction_policy'] = policy_for(source, result['task_definition'])
        result['interaction_policy'].update(
            relocation_policy='vertical_only', placement_groups=[[key] for key in bindings],
            source_camera_policy='preserve')
        warnings.append('separate_supports_common_vertical_translation_requires_review')
        return result


from .dimensional_size import qualify_operation, create_evaluator
