"""Check meaningful-group support placement and the single finalization boundary.

Run: python -m pytest -q tests/test_accessible_shared_groups.py
CPU native-shaped fixtures verify placement contracts, not successful robot actions.
"""
from copy import deepcopy

import numpy as np
import pytest

from wia.tasks import interaction_setup as setup
from wia.tasks.interaction_supports import preload_supports, support_names
from harness_integration.test_esi_interaction_runtime import world_factory


def test_independent_groups_fit_separately_without_scaling_or_overlap(world_factory):
    world = world_factory()
    world.companion.position[0] = 3.
    world.policy.update(mode='inspection', relocation_policy='independent_groups',
                        manipulation_bindings=['target', 'companion'],
                        placement_groups=[['target'], ['companion']], source_camera_policy='preserve')
    result = world.run()
    assert result['accepted'], result
    assert len(result['per_group']) == 2
    assert set(result['support_prim_paths']) == {'target', 'companion'}
    assert np.linalg.norm(world.target.position - world.companion.position) > .1
    assert result['height_screen_after']['target']['within_height_window']
    assert result['height_screen_after']['companion']['within_height_window']
    assert world.session.scene_camera_position == pytest.approx(world.original_camera)
    assert not result['physical_grasp_verified']
    assert all(np.array_equal(obj.scale, np.ones(3)) for obj in (world.target, world.companion))


def test_common_z_uses_per_object_support_evidence_and_keeps_xy(world_factory):
    world = world_factory()
    world.table.position[:2] = [.15, 0.]
    before = [obj.position.copy() for obj in (world.target, world.companion)]
    world.policy.update(mode='inspection', relocation_policy='vertical_only',
                        manipulation_bindings=['target', 'companion'],
                        placement_groups=[['target'], ['companion']], source_camera_policy='preserve')
    result = world.run()
    assert result['accepted'], result
    delta = np.asarray(result['group_translation_m'])
    assert delta[:2] == pytest.approx([0., 0.])
    for obj, original in zip((world.target, world.companion), before):
        assert obj.position == pytest.approx(original + delta)
        assert setup.support_contact_paths(world.session, obj.name) == ('/World/table',)
        assert result['support_surfaces_z_m'][obj.name] == pytest.approx(.7)


def test_final_initializer_is_once_and_trial_requires_owned_context(world_factory):
    world = world_factory()
    with pytest.raises(RuntimeError, match='active owned'):
        setup.initialize_interaction_trial(world.session, world.bindings, world.policy)
    world.session._category_initialization_active = True
    provisional = setup.initialize_interaction_trial(world.session, world.bindings, world.policy)
    assert provisional['accepted']
    steps = world.sim.step_count
    final = setup.initialize_interaction(world.session, world.bindings, world.policy)
    assert final['object_translations_m'] == provisional['object_translations_m']
    assert world.sim.step_count == steps
    with pytest.raises(RuntimeError, match='cannot run twice'):
        setup.initialize_interaction(world.session, world.bindings, world.policy)


def test_multiple_reserved_supports_never_replace_source_assets():
    config = {'objects': [{'name': 'source_chair', 'scale': [1., 1., 1.]}]}
    original = deepcopy(config['objects'][0])
    preload_supports(config, slots=3)
    assert config['objects'][0] == original
    assert len(config['objects']) == 16
    assert set(support_names(0)).isdisjoint(support_names(1))
    assert all(row['fixed_base'] and not row['visual_only'] for row in config['objects'][1:])


def test_fallback_retries_keep_the_owned_support_slot(world_factory, monkeypatch):
    from wia.tasks import interaction_supports as supports
    world = world_factory()
    world.policy.update(support_search='measured_surfaces', allow_fallback_table=True, _support_slot=2)
    monkeypatch.setattr(setup, '_comparison_candidates', lambda *args: iter(()))
    calls = []

    def candidates(session, group, bounds, policy, world_cells, floor, target, evidence):
        delta = np.array([1.35, 0., .7])
        layout = {'new_support': {'centre_xy_m': [1.5, 0.], 'dimensions_xy_m': [1.4, 1.2]}}
        yield None, .7, {name: delta.copy() for name in bounds}, layout

    def activate(session, centre, dimensions, surface, floor, *, slot=0):
        calls.append(slot)
        return world.table

    monkeypatch.setattr(supports, 'support_candidates', candidates)
    monkeypatch.setattr(supports, 'activate_support', activate)
    result = world.run()
    assert result['accepted'], result
    assert len(calls) >= 2
    assert set(calls) == {2}
    assert result['added_support']['asset_names'] == list(support_names(2))


def test_root_reference_uses_accepted_pose_after_preinitialization_fall():
    from types import SimpleNamespace
    session = SimpleNamespace(category_interaction_setup={'accepted': True,
        'object_translations_m': {'prop': [0., 0., 65000.]},
        'adapted_geometry': {'prop': {'position_m': [1., 2., .65]}}})
    assert setup.reference_position(session, 'prop', [1., 2., -65001.], root=True).tolist() == [1., 2., .65]


def test_articulation_cannot_accept_a_floor_appliance_from_handle_height_alone():
    from types import SimpleNamespace
    body = (np.array([0., 0., 0.]), np.array([.5, .4, .6]))
    handle = SimpleNamespace(aabb=(np.array([.45, .1, .45]), np.array([.5, .2, .5])))
    link = SimpleNamespace(prim_path='/World/microwave/door', collision_meshes={'handle': handle})
    obj = SimpleNamespace(name='microwave', aabb=body, links={'door': link})
    policy = {'mode': 'articulation', 'manipulation_bindings': ['container'],
              'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.1}
    bounds = setup._interaction_bounds({'container': obj}, policy)
    screen = setup.height_screen(bounds, 0., .35, 1.1)
    assert not screen['microwave']['within_height_window']
    assert screen['microwave:handle']['within_height_window']
    obj.links = {}
    assert 'microwave' in setup._interaction_bounds({'container': obj}, policy)


def test_independent_groups_report_dynamic_settling_without_moving_fixed_anchors(world_factory):
    world = world_factory(disturbance='background')
    world.policy.update(mode='inspection', relocation_policy='independent_groups',
                        placement_groups=[['target'], ['companion']])
    result = world.run()
    assert result['accepted'], result
    assert 'dynamic_background_settled_during_initialization_requires_review' in result['warning_codes']
    first = result['per_group'][0]['evidence']['robot_spawn']
    assert not first['checks']['background_geometry_preserved']
    assert first['checks']['fixed_background_geometry_preserved']
    assert first['background_pose_changes']['background']['category'] == 'book'
    assert np.array_equal(world.floor.position, world.original['floor'][0])


def test_independent_groups_still_reject_fixed_background_motion(world_factory):
    world = world_factory(disturbance='background')
    world.background.fixed_base = True
    world.policy.update(mode='inspection', relocation_policy='independent_groups',
                        placement_groups=[['target'], ['companion']])
    result = world.run()
    assert not result['accepted']
    assert all(not row['evidence']['accepted'] for row in result['per_group'])


def test_measured_physical_layouts_apply_the_same_dynamic_settling_policy(world_factory):
    world = world_factory(disturbance='background')
    world.policy['support_search'] = 'measured_surfaces'
    result = world.run()
    assert result['accepted'], result
    assert 'dynamic_background_settled_during_initialization_requires_review' in result['warning_codes']
    assert '_allow_dynamic_background_settling' not in world.policy
