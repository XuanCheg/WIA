"""Declare category interaction groups without selecting layouts from answers.

Run: python -m pytest -q tests/test_esi_interaction_profiles.py
These CPU policies request native measurement; they do not establish graspability.
"""
from __future__ import annotations


NAVIGATION_CATEGORIES = {
    ('Cognitive Mapping', 'Connectivity'),
    ('Cognitive Mapping', 'Long-Term Navigation'),
    ('Cognitive Mapping', 'Regional Boundary'),
    ('Cognitive Mapping', 'Traversable Passage'),
    ('Enumerative Perception', 'Counting w Occlusion'),
    ('Enumerative Perception', 'Spatial Segmentation'),
    ('Spatial Relations', 'Geometric Configuration'),
    ('Spatial Relations', 'Linear Alignment'),
    ('Specular Reflection', 'Correspondence'),
    ('Specular Reflection', 'Reflection Authoring'),
    ('Specular Reflection', 'Spatial Relations'),
    ('Temporal Understanding', 'Agent Observation'),
    ('Physical Structure', 'Liquid Volume'),
}

MANIPULATION_CATEGORIES = {
    ('Action Sequencing', 'Action Order Inference'),
    ('Enumerative Perception', 'Category Ambiguity'),
    ('Enumerative Perception', 'Illumination Variability'),
    ('Enumerative Perception', 'Merged Observation'),
    ('Metric Comparison', 'Dimensional Size'),
    ('Metric Comparison', 'Spatial Distance'),
    ('Perceptual Grounding', 'Partial Occlusion'),
    ('Perceptual Grounding', 'Material Transparency'),
    ('Perceptual Grounding', 'View Hallucination'),
    ('Physical Dynamics', 'Inclined Plane'),
    ('Physical Dynamics', 'Stacking & Stability'),
    ('Physical Structure', 'Deformable'),
    ('Spatial Relations', 'Physical Contact'),
    ('Temporal Understanding', 'Unobserved Change'),
}

SUPPORTED_CATEGORIES = NAVIGATION_CATEGORIES | MANIPULATION_CATEGORIES | {
    ('Enumerative Perception', 'Structural Enclosure')}


def _nonfloor(bindings):
    """Never translate the original floor or its collision geometry."""
    return sorted(key for key, value in bindings.items()
                  if key not in ('floor', 'source_floor', 'support') and
                  value.get('category') not in ('floor', 'floors'))


def policy_for(source, task):
    """Return an answer-independent measured-initialization request.

    Group membership follows source roles, including comparison alternatives and
    confusers. It never follows the evaluation's privately selected target.
    ``source_region`` is the original room; full-scene navigation has no room.
    Height limits are the existing Rigid screening window, not an IK certificate.
    """
    category = (source.big_task, source.small_task)
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(f'No interaction profile for {category!r}')
    bindings = task['private']['bindings']
    evaluation = task['private']['evaluation']
    parameters = evaluation.get('parameters', {})
    group = _nonfloor(bindings)
    room = source.raw.get('room')
    room = None if room in (None, '', 'full_scene', 'unknown_room') else room
    policy = {'mode': 'manipulation', 'group_bindings': group,
              'manipulation_bindings': group.copy(), 'relocation_policy': 'rigid_group',
              'support_search': 'measured_surfaces', 'allow_fallback_table': True,
              'source_region': room, 'minimum_grasp_height_m': .35,
              'maximum_grasp_height_m': 1.10, 'maximum_stance_distance_m': .85,
              'warning_codes': ['native_grasp_ik_and_aperture_pending']}
    if category in NAVIGATION_CATEGORIES:
        if (source.small_task == 'Reflection Authoring' and
                evaluation.get('operation') == 'discover_and_lift_reflected_prop'):
            policy.update(group_bindings=['prop'], manipulation_bindings=['prop'],
                          relocation_policy='vertical_only',
                          warning_codes=['native_initial_reflection_only_unverified',
                                         'optical_layout_changed_by_target_height'])
            return policy
        policy.update(mode='navigation', manipulation_bindings=[], relocation_policy='preserve',
                      allow_fallback_table=False, warning_codes=[])
        if category[0] == 'Spatial Relations' and category[1] in ('Geometric Configuration', 'Linear Alignment'):
            policy.update(relocation_policy='relation_group', warning_codes=[
                'layout_uniform_position_scale_requires_review', 'native_navigation_unverified'])
        return policy

    small = source.small_task
    if small == 'Dimensional Size':
        required = ['task_obj1', 'task_obj2', 'ref_obj1', 'ref_obj2']
        if any(key not in bindings for key in required):
            raise ValueError('Dimensional Size needs both task and both reference bindings.')
        policy.update(group_bindings=required, manipulation_bindings=required[:2],
                      relocation_policy='comparison_pairs',
                      placement_groups=[['task_obj1', 'ref_obj1'], ['task_obj2', 'ref_obj2']])
        policy['warning_codes'].append('comparison_pair_layout_adapted_requires_review')
    elif small in ('Category Ambiguity', 'Merged Observation', 'Illumination Variability'):
        targets = sorted(key for key in group if key.startswith('target_'))
        policy['manipulation_bindings'] = targets
        if small == 'Illumination Variability':
            policy.update(relocation_policy='vertical_only',
                          anchor_policy='illumination_common_translation')
            policy['warning_codes'].append('native_complete_lighting_anchor_measurement_required')
    elif small == 'Partial Occlusion':
        policy['group_bindings'] = [key for key in ('target', 'occluder') if key in bindings]
        policy['manipulation_bindings'] = [key for key in group if key == 'target']
    elif small == 'Material Transparency':
        policy['manipulation_bindings'] = [key for key in group if key == 'obj_small']
    elif small == 'Physical Contact':
        # The converter chooses this role by original bounding volume, never QA.
        target = parameters.get('target')
        policy['manipulation_bindings'] = [target] if target in group else group.copy()
        policy['support_search'] = 'measured_surfaces'
    elif small == 'Inclined Plane':
        policy['manipulation_bindings'] = [key for key in group if key != 'slope']
        policy.update(group_bindings=policy['manipulation_bindings'].copy(),
                      support_bindings=policy['manipulation_bindings'].copy(),
                      relocation_policy='rigid_group', source_camera_policy='preserve')
        policy['warning_codes'].append('ramp_floor_outcome_reference_preserved')
    elif small == 'Structural Enclosure':
        policy.update(mode='articulation', relocation_policy='vertical_only',
                      manipulation_bindings=sorted(key for key in group if key.startswith('container_')))
        policy['warning_codes'].append('native_handle_height_and_articulation_pending')
    elif small == 'View Hallucination':
        policy.update(group_bindings=[key for key in group if key == 'target'],
                      manipulation_bindings=[key for key in group if key == 'target'])
        policy['warning_codes'].extend(['source_context_changed_by_support_relocation',
                                       'native_original_identity_view_recheck_required'])
    elif small in ('Unobserved Change', 'Spatial Distance'):
        policy['relocation_policy'] = 'preserve'
        if small == 'Spatial Distance':
            from .metric_comparison_spatial_distance import _distance_evidence
            policy.update(relocation_policy='vertical_only',
                          fixed_reference_evidence=_distance_evidence(source.payload))
            policy['warning_codes'].append('fixed_original_distance_reference_preserved')
            return policy
        if small == 'Unobserved Change' and evaluation.get('operation') == 'observe_two_phase_change':
            policy.update(mode='navigation', manipulation_bindings=[],
                          allow_fallback_table=False,
                          group_bindings=sorted(parameters.get('container_keys', []) +
                              parameters.get('phase_keys', {}).get('image1', [])),
                          warning_codes=['native_temporal_execution_unverified'])
            return policy
        policy['warning_codes'].append({
            'Unobserved Change': 'active_phase_original_group_restoration_pending',
            'Spatial Distance': 'fixed_original_distance_reference_preserved',
        }[small])
        if small == 'Unobserved Change':
            patches = task.get('scene', {}).get('variants', {}).get('adapted', {}).get('patches', [])
            phases = {patch.get('parameters', {}).get('phase') for patch in patches
                      if patch.get('operation') == 'category_initialization' and
                      patch.get('module', '').endswith('temporal_understanding_unobserved_change')}
            phase_keys = parameters.get('phase_keys', {})
            if len(phases) == 1 and next(iter(phases)) in phase_keys:
                active = phase_keys[next(iter(phases))]
                group = sorted(set(parameters.get('container_keys', []) + active))
                if not set(group) <= set(bindings):
                    raise ValueError('The declared active temporal phase has incomplete original bindings.')
                policy.update(group_bindings=group, manipulation_bindings=sorted(active))
                if evaluation.get('operation') == 'box_inspection_route':
                    policy.update(mode='navigation', manipulation_bindings=[])
            else:
                policy['warning_codes'].append('active_phase_binding_resolution_requires_review')
    elif small == 'Deformable':
        # The declared operation sweeps the cloth with the hand and grasps the
        # item; cloth solver/contact qualification is separate from rigid grasp.
        policy['manipulation_bindings'] = [key for key in group if key == 'item']
        policy['warning_codes'].append('cloth_initial_transform_and_solver_qualification_pending')
    return policy
