"""Check CPU interaction groups and answer independence.

Run: OMP_NUM_THREADS=2 python -m pytest -q tests/test_esi_interaction_profiles.py
No fixtures here establish native support, IK or grasp success.
"""
from copy import deepcopy
from types import SimpleNamespace
import sys

import pytest

from wia.authoring.converters.interaction_profiles import (
    NAVIGATION_CATEGORIES, SUPPORTED_CATEGORIES, policy_for,
)


def fixture(category, keys, parameters=None, operation=None):
    source = SimpleNamespace(big_task=category[0], small_task=category[1],
                             raw={'room': 'kitchen_0', 'answer': 'first'}, payload={})
    task = {'private': {'bindings': {key: {'name': key} for key in keys},
                        'evaluation': {'target': keys[0] if keys else None,
                                       'parameters': parameters or {}, 'operation': operation}}}
    return source, task


@pytest.mark.parametrize('category', sorted(NAVIGATION_CATEGORIES))
def test_navigation_preserves_room_anchors_and_requests_relational_tabletops(category):
    source, task = fixture(category, ['anchor_0', 'target_0', 'floor'])
    policy = policy_for(source, task)
    assert policy['mode'] == 'navigation'
    relational = category[0] == 'Spatial Relations' and category[1] in ('Geometric Configuration', 'Linear Alignment')
    assert policy['relocation_policy'] == ('relation_group' if relational else 'preserve')
    assert policy['manipulation_bindings'] == []


def test_all_scheduled_categories_have_explicit_profiles():
    from wia.runtime.esi_category_scheduler import scan_inventory
    inventory = scan_inventory()
    found = {(row['big_task'], row['small_task']) for row in inventory['categories']
             if row['small_task'] != 'Rigid Containment'}
    assert len(SUPPORTED_CATEGORIES) == 28
    assert SUPPORTED_CATEGORIES == found


def test_dimensional_screens_both_copies_and_keeps_all_reference_members():
    keys = ['task_obj1', 'task_obj2', 'ref_obj1', 'ref_obj2']
    source, task = fixture(('Metric Comparison', 'Dimensional Size'), keys)
    before = deepcopy(task)
    policy = policy_for(source, task)
    assert policy['group_bindings'] == keys
    assert policy['manipulation_bindings'] == keys[:2]
    assert policy['relocation_policy'] == 'comparison_pairs'
    assert policy['placement_groups'] == [['task_obj1', 'ref_obj1'], ['task_obj2', 'ref_obj2']]
    source.raw['answer'] = 'second'
    task['private']['evaluation']['target'] = 'task_obj2'
    assert policy_for(source, task) == policy
    assert before['private']['bindings'] == task['private']['bindings']


@pytest.mark.parametrize('small,keys', [
    ('Category Ambiguity', ['target_00', 'target_01', 'confuser_00']),
    ('Merged Observation', ['target_00', 'target_01', 'target_02']),
])
def test_inventory_groups_include_all_original_members(small, keys):
    source, task = fixture(('Enumerative Perception', small), [*keys, 'floor'])
    policy = policy_for(source, task)
    assert set(policy['group_bindings']) == set(keys)
    assert 'floor' not in policy['group_bindings']
    assert policy['manipulation_bindings'] == [key for key in keys if key.startswith('target_')]


def test_unbound_stage_lighting_requests_complete_common_vertical_anchors():
    source, task = fixture(('Enumerative Perception', 'Illumination Variability'),
                           ['target_00', 'lamp_00', 'floor'],
                           {'lamp_keys': ['lamp_00'], 'lighting': {'lamps': [{'name': 'lamp_00'}],
                            'original_scene_file': '/original/scene.json'}})
    policy = policy_for(source, task)
    assert policy['relocation_policy'] == 'vertical_only'
    assert set(policy['group_bindings']) == {'target_00', 'lamp_00'}
    assert policy['anchor_policy'] == 'illumination_common_translation'
    assert 'native_complete_lighting_anchor_measurement_required' in policy['warning_codes']


def test_fully_bound_lighting_group_retains_lamp_anchor():
    source, task = fixture(('Enumerative Perception', 'Illumination Variability'),
                           ['target_00', 'lamp_00', 'floor'],
                           {'lamp_keys': ['lamp_00'], 'lighting': {'lamps': [{'name': 'lamp_00'}]}})
    assert policy_for(source, task)['relocation_policy'] == 'vertical_only'


def test_occlusion_and_transparency_keep_relationship_groups():
    source, task = fixture(('Perceptual Grounding', 'Partial Occlusion'), ['target', 'occluder', 'support'])
    assert set(policy_for(source, task)['group_bindings']) == {'target', 'occluder'}
    assert policy_for(source, task)['manipulation_bindings'] == ['target']
    source, task = fixture(('Perceptual Grounding', 'Material Transparency'), ['obj_small', 'obj_container'])
    assert set(policy_for(source, task)['group_bindings']) == {'obj_small', 'obj_container'}
    assert policy_for(source, task)['manipulation_bindings'] == ['obj_small']


def test_existing_floor_relative_ramp_trial_preserves_all_anchors():
    source, task = fixture(('Physical Dynamics', 'Inclined Plane'), ['target', 'slope', 'floor'],
                           operation='place_release_observe')
    policy = policy_for(source, task)
    assert policy['relocation_policy'] == 'rigid_group'
    assert policy['manipulation_bindings'] == ['target']
    assert policy['group_bindings'] == ['target']
    assert policy['support_bindings'] == ['target']
    assert policy['source_camera_policy'] == 'preserve'
    assert 'ramp_floor_outcome_reference_preserved' in policy['warning_codes']


@pytest.mark.parametrize('category', [('Temporal Understanding', 'Unobserved Change')])
def test_source_dependent_layouts_remain_preserved(category):
    source, task = fixture(category, ['target', 'other', 'floor'])
    assert policy_for(source, task)['relocation_policy'] == 'preserve'


def test_distance_preserves_native_anchor_and_requests_common_vertical_candidate_move():
    from wia.authoring.conversion import SourceInstance, ROOT
    path = next((ROOT / 'ESI/dataset/json_clean/Metric Comparison/Spatial Distance').glob('*/*/*.json'))
    source = SourceInstance(path, str(path.relative_to(ROOT / 'ESI/dataset/json_clean')),
                            'Metric Comparison', 'Spatial Distance')
    _, task = fixture(('Metric Comparison', 'Spatial Distance'), ['obj_near', 'obj_far', 'floor'])
    policy = policy_for(source, task)
    assert policy['relocation_policy'] == 'vertical_only'
    assert policy['fixed_reference_evidence']['reference_is_bound'] is False
    assert 'reference_aabb_xy_reconstruction' in policy['fixed_reference_evidence']


def test_identity_target_can_move_without_changing_native_scale_or_orientation():
    source, task = fixture(('Perceptual Grounding', 'View Hallucination'), ['target', 'support'])
    policy = policy_for(source, task)
    assert policy['relocation_policy'] == 'rigid_group'
    assert policy['group_bindings'] == policy['manipulation_bindings'] == ['target']
    assert 'source_context_changed_by_support_relocation' in policy['warning_codes']


def test_enclosure_articulation_keeps_bodies_fixed():
    source, task = fixture(('Enumerative Perception', 'Structural Enclosure'),
                           ['container_00', 'container_01', 'content_00', 'floor'])
    policy = policy_for(source, task)
    assert policy['mode'] == 'articulation'
    assert policy['relocation_policy'] == 'vertical_only'
    assert policy['manipulation_bindings'] == ['container_00', 'container_01']


def test_temporal_profile_uses_declared_active_phase_not_inactive_contents():
    source, task = fixture(('Temporal Understanding', 'Unobserved Change'),
                           ['container_00', 'image1_content_00', 'image2_content_00', 'floor'],
                           {'container_keys': ['container_00'],
                            'phase_keys': {'image1': ['image1_content_00'], 'image2': ['image2_content_00']}})
    task['scene'] = {'variants': {'adapted': {'patches': [
        {'operation': 'category_initialization',
         'module': 'wia.authoring.converters.temporal_understanding_unobserved_change',
         'parameters': {'phase': 'image2'}}]}}}
    policy = policy_for(source, task)
    assert policy['group_bindings'] == ['container_00', 'image2_content_00']
    assert policy['manipulation_bindings'] == ['image2_content_00']
    task['private']['evaluation']['operation'] = 'box_inspection_route'
    policy = policy_for(source, task)
    assert policy['mode'] == 'navigation'
    assert policy['manipulation_bindings'] == []


def test_unknown_category_fails_explicitly_for_review():
    source, task = fixture(('Unknown', 'Unknown'), [])
    with pytest.raises(ValueError, match='No interaction profile'):
        policy_for(source, task)


def test_cluster_support_uses_current_measured_surface_and_keeps_floor_evidence(monkeypatch):
    """Synthetic contact graph: table-rooted chains must lose support on contact loss."""
    touching, on_top = object(), object()
    monkeypatch.setitem(sys.modules, 'omnigibson.object_states',
                        SimpleNamespace(Touching=touching, OnTop=on_top))
    from wia.authoring.converters.enumerative_perception_merged_observation import _support
    from wia.tasks.interaction_setup import reference_position
    floor = SimpleNamespace(name='floor', prim_path='/floor')
    table = SimpleNamespace(name='table', prim_path='/table')

    def obj(name, targets, surface_contact):
        contacts = [SimpleNamespace(body0='/' + name, body1='/table/link', normal=[0., 0., 1.],
                                     position=[0., 0., .7])] if surface_contact else []
        state = SimpleNamespace(get_value=lambda other: other.name in targets)
        return SimpleNamespace(name=name, prim_path='/' + name,
                               states={touching: state, on_top: state}, contact_list=lambda: contacts)

    lower = obj('lower', {'table'}, True)
    upper = obj('upper', {'lower'}, False)
    session = SimpleNamespace(env=SimpleNamespace(scene=SimpleNamespace(objects=[table, lower, upper, floor])),
                              category_interaction_setup={'accepted': True, 'relocated': True,
                                  'object_names': ['lower', 'upper'], 'support_prim_path': '/table',
                                  'support_surface_z_m': .7, 'group_translation_m': [1., 2., .7]})
    result = _support({'lower': lower, 'upper': upper}, floor, session)
    assert result['floor_roots'] == []
    assert result['task_support_roots'] == ['lower']
    assert result['all_grounded']
    assert list(reference_position(session, 'lower', [0., 0., .1])) == pytest.approx([1., 2., .8])
    assert list(reference_position(session, 'unrelated', [0., 0., .1])) == [0., 0., .1]
    lower.contact_list = lambda: []
    assert not _support({'lower': lower, 'upper': upper}, floor, session)['all_grounded']


def test_stack_allows_measured_base_support_and_rejects_upper_furniture_support():
    from wia.authoring.converters.physical_dynamics_stacking_stability import StackOperationEvaluator, OBJECT_KEYS
    evaluator = StackOperationEvaluator.__new__(StackOperationEvaluator)
    bottom, middle, top = OBJECT_KEYS
    evaluator.bound = {key: SimpleNamespace(name=key, prim_path='/' + key) for key in (*OBJECT_KEYS, 'floor')}
    evaluator.session = SimpleNamespace(category_interaction_setup={'accepted': True, 'relocated': True,
        'object_names': list(OBJECT_KEYS), 'support_prim_path': '/table'})
    evaluator.lifted = {middle, top}
    evaluator.released = {middle: bottom, top: middle}
    rows = {key: {'visual_only': False, 'fixed_base': False, 'up_dot': 1.,
                  'linear_speed_m_s': 0., 'angular_speed_rad_s': 0., 'external_contacts': []}
            for key in OBJECT_KEYS}
    rows[bottom]['external_contacts'] = [['/table/link'], ['/' + middle]]
    rows[middle]['external_contacts'] = [['/' + bottom], ['/' + top]]
    rows[top]['external_contacts'] = [['/' + middle]]
    relations = {(a, b): False for a in OBJECT_KEYS for b in (*OBJECT_KEYS, 'task_support') if a != b}
    relations.update({(bottom, 'task_support'): True, (middle, bottom): True, (top, middle): True})
    gripper = {'all_arms_free': True, 'gripper_open': True}
    assert evaluator._terminal_orders(rows, relations, relations, gripper) == [(bottom, middle, top)]
    rows[middle]['external_contacts'].append(['/table/link'])
    assert evaluator._terminal_orders(rows, relations, relations, gripper) == []
