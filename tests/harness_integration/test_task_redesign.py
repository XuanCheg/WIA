"""Check redesigned temporal contracts and physical evidence reducers.

Run: python -B -m pytest -q tests/test_task_redesign.py
These CPU checks do not certify native layout or manipulation feasibility.
"""
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
from types import ModuleType
import pytest

import numpy as np

from wia.authoring.converters.inclined_comparison import qualified_fastest
from wia.authoring.converters.deformable_redrape import projected_cover
from wia.native.esi_temporal_phases import PhysicalTemporalChangeEvaluator, TemporalChangeEvaluator, measured_change_branch
from wia.tasks.temporal_policy import mounted_visibility_rows
from harness_integration.test_annotation_manipulation_categories import NativeObject, native_types, put_on
from harness_integration.test_native_temporal_phases import Object, world


def test_incomplete_survey_can_transition_only_in_explicit_new_revision():
    evaluator = object.__new__(PhysicalTemporalChangeEvaluator)
    evaluator.parameters = {'require_complete_phase_1_coverage': False}
    evaluator.visual_available = {'phase_1': True}
    evaluator.coverage = {'phase_1': set()}
    evaluator._required_coverage = lambda phase: {'unseen'}
    evaluator._visual = lambda observation: {}
    observation = NS(snapshot_id=4, action_step=3)
    with patch.object(TemporalChangeEvaluator, 'prepare_phase_transition') as transition:
        evaluator.prepare_phase_transition(observation)
        transition.assert_called_once_with(observation)
    evaluator.parameters = {}
    with patch.object(TemporalChangeEvaluator, 'prepare_phase_transition') as transition:
        import pytest
        with pytest.raises(ValueError, match='every comparison location'):
            evaluator.prepare_phase_transition(observation)
        transition.assert_not_called()


def test_mounted_exposure_needs_current_native_identity_but_no_semantic_audit():
    observation = NS(snapshot_id=3, rgb_images=lambda: {'head': [], 'initial_overview_rgb': []})
    row = {'snapshot_id': 3, 'camera': 'head', 'native_name': 'item',
        'identity_synchronized': True, 'native_rgb_matches_delivered': True,
        'visible_pixels': 400, 'minimum_extent_px': 20,
        'audit_context': {'native_pose': {'position_m': [1., 0., 1.]},
            'robot_pose': {'position_m': [0., 0., 0.]}}}
    result = {'measurement_available': True,
        'native_public_control_context': {'available': True, 'state_setter_used': False},
        'measurements': {'target': {'views': {'head': row}}}}
    visible = mounted_visibility_rows(None, observation, result, {'target': 'item'}, {})
    assert visible['objects']['target']['informative']
    assert visible['semantic_recognition_verified'] is False
    for changed in ({'snapshot_id': 2}, {'visible_pixels': 1}, {'native_rgb_matches_delivered': False}):
        altered = deepcopy(result)
        altered['measurements']['target']['views']['head'].update(changed)
        assert not mounted_visibility_rows(None, observation, altered, {'target': 'item'}, {})['objects']
    altered = deepcopy(result)
    altered['measurements']['target']['views'] = {'initial_overview_rgb': row}
    assert not mounted_visibility_rows(None, observation, altered, {'target': 'item'}, {})['objects']


def test_partial_mesh_overlap_does_not_pass_full_projected_coverage():
    square = np.array([[[0., 0., 1.], [1., 0., 1.], [1., 1., 1.]],
                       [[0., 0., 1.], [1., 1., 1.], [0., 1., 1.]]])
    assert projected_cover(square, [.1, .1, 0.], [.9, .9, .9])['all_sampled_top_rays_covered']
    assert not projected_cover(square[:1], [.1, .1, 0.], [.9, .9, .9])['all_sampled_top_rays_covered']


def test_fastest_requires_every_controlled_trial_and_successful_receiver_finish():
    records = {'a': {'valid': True, 'arrival_s': .5, 'final_in_receiver': False},
               'b': {'valid': True, 'arrival_s': 1., 'final_in_receiver': True},
               'c': {'valid': True, 'arrival_s': 1.1, 'final_in_receiver': True}}
    assert qualified_fastest(records, ['a', 'b', 'c']) == ['b', 'c']
    records['a']['valid'] = False
    assert qualified_fastest(records, ['a', 'b', 'c']) == []
    assert qualified_fastest({}, ['a', 'b', 'c']) == []


def test_multi_event_is_distinct_from_a_single_displacement():
    before = {'left': 'apple', 'middle': 'bowl', 'right': None}
    assert measured_change_branch(before, {'left': None, 'middle': 'bowl', 'right': 'apple'}) == 'moved'
    assert measured_change_branch(before, {'left': None, 'middle': 'bowl', 'right': 'orange'}) == 'multiple'


def test_ramp_release_reads_native_structured_contacts_and_rejects_side_contact(native_types):
    from wia.authoring.converters.inclined_comparison import InclinedComparisonEvaluator
    obj = NativeObject('trial', [-.1, 0., .06])
    ramp = NativeObject('ramp', [0., 0., 0.], (.4, .3, .02))
    receiver = NativeObject('receiver', [.5, 0., 0.], (.45, .45, .02))
    put_on(obj, ramp, native_types, z=.01)
    dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    obj.contacts[0].position = np.array((-.1, 0., .01), dtype=dtype)
    obj.contacts[0].normal = np.array((0., 0., 1.), dtype=dtype)
    evaluator = object.__new__(InclinedComparisonEvaluator)
    evaluator.objects = {'trial': obj, 'ramp': ramp, 'receiver': receiver}
    evaluator.spares = {}
    evaluator.initial = {'trial': {'quaternion_xyzw': [0., 0., 0., 1.]}}
    evaluator.p = {'ramp_dimensions_m': [.4, .3, .02], 'trial_support_keys': ['ramp', 'receiver'],
        'release_local_x_m': -.1, 'safe_bounds_m': [[-1., -1., -.1], [1., 1., 1.]]}
    assert evaluator._measure('trial')['release_ready']
    obj.contacts[0].normal = np.array((1., 0., 0.), dtype=dtype)
    assert not evaluator._measure('trial')['release_ready']


@pytest.mark.parametrize('publish_geometry', [False, True])
def test_temporal_event_cannot_pass_with_tensor_pose_but_offstage_geometry(world, monkeypatch, publish_geometry):
    class DivergentBody(Object):
        @property
        def aabb(self):
            return self.geometry_position - self.extent / 2, self.geometry_position + self.extent / 2

    c = world.controller
    key = 'image2_content_00'
    body = DivergentBody(key, 'syrup', [1000., 1000., 120.])
    body.geometry_position = body.position.copy()
    body.fixed_base, body.kinematic_only = False, False
    body.root_link, body.links = body, {}
    body.enable_gravity = lambda: None
    old = c.objects[key]
    world.session.env.scene.objects.remove(old)
    world.session.env.scene.objects.append(body)
    c.objects[key] = body
    c.slots['image2']['container_00'] = key
    c.parameters.update(survey_revision='free_survey_10_v1', task_edit2_physical=True,
        physical_target_key=key)
    for obj in c.objects.values():
        obj.links = {}
    world.session.pipeline = NS(og=NS(sim=NS(current_time=1.,
        pi=NS(update_simulation=lambda **kwargs: None))))
    for name in ('omnigibson', 'omnigibson.utils', 'omnigibson.utils.usd_utils',
                 'omnigibson.prims', 'omnigibson.prims.xform_prim'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['omnigibson.utils.usd_utils'].PoseAPI = NS(invalidate=lambda: None, _refresh=lambda: None)
    def publish(root, **kwargs):
        if publish_geometry:
            root.geometry_position = np.array(kwargs['position'])
    sys.modules['omnigibson.prims.xform_prim'].XFormPrim = NS(set_position_orientation=publish)
    c.prepared_boundary = (1, 0)
    if publish_geometry:
        transition = c.advance()
        assert transition['physics_steps'] == 0 and transition['world_unchanged']
        assert transition['active_actor_placement_checks'][key]['geometry_is_onstage']
    else:
        with pytest.raises(ValueError, match='Native temporal event placement failed'):
            c.advance()
        assert c.failed and c.phase == 'phase_1'
