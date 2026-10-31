"""Check opt-in reviewed native root placement with CPU native-shaped fixtures.

Run: PYTHONPATH=src:../Show-Harness-Explore python -m pytest -q \
  tests/harness_integration/test_reviewed_group_placement.py
These fixtures exercise initialization acceptance without simulating robot actions.
"""
from copy import deepcopy
import sys

import numpy as np
import pytest

from wia.tasks import interaction_setup as setup
from wia.tasks import interaction_supports as supports
from harness_integration.test_esi_interaction_runtime import Object, world_factory


def reviewed_policy(world, xy=(1.22, .16), support='table', **extra):
    world.policy.update(support_search='measured_surfaces', allow_fallback_table=True,
        layout_parameters={'reviewed_group_placements_v1': {
            'target': {'support_name': support, 'root_xy_m': list(xy)}, **extra}})
    return world.policy


def forbid_fallback(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('A reviewed native placement must not propose or activate an added support.')
    monkeypatch.setattr(supports, 'support_candidates', unexpected)
    monkeypatch.setattr(supports, 'activate_support', unexpected)


class OffsetObject(Object):
    """Expose a native root that differs from the measured collision AABB centre."""
    @property
    def aabb(self):
        centre = self.position + self.aabb_offset
        return centre - self.extent / 2, centre + self.extent / 2


def test_root_target_accounts_for_aabb_offset_and_rigidly_moves_all_members(world_factory, monkeypatch):
    world = world_factory()
    world.target.__class__ = OffsetObject
    world.target.aabb_offset = np.array([.13, -.08, 0.])
    reviewed_policy(world)
    # Only the root object bears the group; the contained member must still get
    # the exact same translation and retain its original pose and dimensions.
    world.policy['support_bindings'] = ['target']
    before = {obj.name: (obj.position.copy(), obj.quaternion.copy(), obj.scale.copy(), obj.extent.copy())
              for obj in (world.target, world.companion)}
    forbid_fallback(monkeypatch)
    result = world.run()
    assert result['accepted'], result
    assert world.target.position[:2] == pytest.approx([1.22, .16])
    delta = world.target.position - before['target'][0]
    assert delta == pytest.approx([1.22, .16, .705])
    for obj in (world.target, world.companion):
        position, quaternion, scale, extent = before[obj.name]
        assert obj.position == pytest.approx(position + delta)
        assert obj.quaternion == pytest.approx(quaternion)
        assert obj.scale == pytest.approx(scale)
        assert obj.extent == pytest.approx(extent)
        assert result['object_translations_m'][obj.name] == pytest.approx(delta)
    assert result['support_name'] == 'table'
    assert result['support_search'][0]['candidate_layouts'] == 1
    assert result['reviewed_group_placement']['root_xy_m'] == [1.22, .16]
    assert result['reviewed_group_placement_measurement']['root_xy_error_m'] < 1e-6
    assert not result['physical_grasp_verified'] and not result['ik_verified']


def test_reviewed_group_moves_even_when_initial_native_support_is_valid(world_factory):
    world = world_factory()
    world.target.position = np.array([1.05, 0., .805])
    world.companion.position = np.array([1.35, 0., .805])
    reviewed_policy(world, xy=(1.4, .18))
    result = world.run()
    assert result['accepted'], result
    assert all(result['initial_support_contacts'].values())
    assert all(row['within_height_window'] for row in result['height_screen_before'].values())
    assert world.target.position[:2] == pytest.approx([1.4, .18])
    assert world.companion.position[:2] == pytest.approx([1.7, .18])
    assert result['relocated'] and result['robot_spawn']['checks']['native_task_support_contact']


@pytest.mark.parametrize('support,xy,rejection', [
    ('missing_table', (1.22, .16), 'requested_native_support_unavailable'),
    ('table', (2.15, .16), 'unchanged_rigid_group_does_not_fit'),
])
def test_requested_support_or_xy_failure_never_uses_other_support_or_grid(
        world_factory, monkeypatch, support, xy, rejection):
    world = world_factory()
    reviewed_policy(world, xy=xy, support=support)
    forbid_fallback(monkeypatch)
    result = world.run()
    assert not result['accepted'] and not result['attempts']
    assert result['failure_stage'] == 'support_candidate_generation'
    assert [row['support'] for row in result['support_search']] == [support]
    assert result['support_search'][0]['rejections'][rejection] > 0
    assert result['reviewed_group_placement']['support_name'] == support
    assert 'added_support' not in result
    assert not result['physical_grasp_verified']


def test_reviewed_proposal_retains_native_footprint_ray_checks(world_factory, monkeypatch):
    world = world_factory()
    reviewed_policy(world)
    native_raytest = world.raytest

    def missing_corner(start, end, only_closest=True):
        hit = native_raytest(start, end, only_closest=only_closest)
        if only_closest and start[2] - end[2] < .05:
            hit['hit'] = False
        return hit

    monkeypatch.setattr(sys.modules['omnigibson.utils.sampling_utils'], 'raytest', missing_corner)
    forbid_fallback(monkeypatch)
    result = world.run()
    assert not result['accepted']
    assert result['support_search'][0]['rejections']['pair_footprint_lacks_local_native_surface'] == 1
    assert not result['attempts']


@pytest.mark.parametrize('mapping', [
    None, [], {},
    {'other_group': {'support_name': 'table', 'root_xy_m': [1., 0.]}},
    {'target': {'support_name': 'table', 'root_xy_m': [1., 0.]},
     'companion': {'support_name': 'table', 'root_xy_m': [1.5, 0.]}},
    {'target': None},
    {'target': {'support_name': '', 'root_xy_m': [1., 0.]}},
    {'target': {'support_name': '   ', 'root_xy_m': [1., 0.]}},
    {'target': {'support_name': 'table', 'root_xy_m': [1.]}},
    {'target': {'support_name': 'table', 'root_xy_m': [1., 0., .7]}},
    {'target': {'support_name': 'table', 'root_xy_m': [1., np.nan]}},
    {'target': {'support_name': 'table', 'root_xy_m': [np.inf, 0.]}},
])
def test_invalid_or_unmatched_reviewed_map_fails_without_default_fallback(world_factory, monkeypatch, mapping):
    world = world_factory()
    world.policy['layout_parameters'] = {'reviewed_group_placements_v1': deepcopy(mapping)}
    forbid_fallback(monkeypatch)
    result = world.run()
    assert not result['accepted'] and not result['attempts']
    assert result['reason'].startswith('ValueError:') or result['reason'].startswith('TypeError:')
    assert 'reviewed_group_placements_v1' in result
    assert world.sim.step_count == 0
    assert not result['physical_grasp_verified']


def test_whole_map_matches_one_binding_per_independent_local_group(world_factory, monkeypatch):
    world = world_factory()
    reviewed_policy(world, xy=(1.05, .15),
        companion={'support_name': 'table', 'root_xy_m': [1.75, -.15]})
    world.policy.update(mode='inspection', relocation_policy='independent_groups',
        manipulation_bindings=['target', 'companion'], placement_groups=[['target'], ['companion']])
    original_policy = deepcopy(world.policy)
    forbid_fallback(monkeypatch)
    result = world.run()
    assert result['accepted'], result
    assert world.policy == original_policy
    assert world.target.position[:2] == pytest.approx([1.05, .15])
    assert world.companion.position[:2] == pytest.approx([1.75, -.15])
    for row, binding in zip(result['per_group'], ('target', 'companion')):
        receipt = row['evidence']
        assert receipt['reviewed_group_placement']['binding'] == binding
        assert receipt['reviewed_group_placement_final_measurement']['accepted']
        assert receipt['support_name'] == 'table'
    assert result['independent_group_clearance']['accepted']


@pytest.mark.parametrize('drift,accepted', [(.01, True), (.02, False)])
def test_native_root_drift_has_its_own_candidate_acceptance_gate(world_factory, monkeypatch, drift, accepted):
    world = world_factory()
    reviewed_policy(world)
    native_step = world.sim.step

    def physics_step():
        native_step()
        if world.steps_this_placement == 1:
            for obj in (world.target, world.companion):
                obj.position[0] += drift

    monkeypatch.setattr(world.sim, 'step', physics_step)
    # Isolate the reviewed-root gate from the existing broader pose gate.
    monkeypatch.setattr(setup, '_pose_unchanged', lambda *args, **kwargs: True)
    forbid_fallback(monkeypatch)
    result = world.run()
    assert result['accepted'] is accepted, result
    assert result['attempts']
    for attempt in result['attempts']:
        measurement = attempt['reviewed_group_placement_measurement']
        assert measurement['root_xy_error_m'] == pytest.approx(drift, abs=1e-6)
        assert measurement['maximum_root_xy_error_m'] == .015
        assert attempt['checks']['reviewed_root_xy_retained'] is accepted
    if not accepted:
        assert any(row['code'] == 'reviewed_group_root_xy_drift' for row in result['failure_reasons'])


def test_later_group_physics_cannot_silently_move_an_earlier_reviewed_root(world_factory, monkeypatch):
    world = world_factory()
    reviewed_policy(world, xy=(1.05, .15),
        companion={'support_name': 'table', 'root_xy_m': [1.75, -.15]})
    world.policy.update(mode='inspection', relocation_policy='independent_groups',
        manipulation_bindings=['target', 'companion'], placement_groups=[['target'], ['companion']])
    initialize_single = setup._initialize_single

    def settle_later_group(session, bindings, policy):
        receipt = initialize_single(session, bindings, policy)
        if policy['group_bindings'] == ['companion'] and receipt['accepted']:
            world.target.position[0] += .02
        return receipt

    monkeypatch.setattr(setup, '_initialize_single', settle_later_group)
    result = world.run()
    assert not result['accepted']
    assert result['failure_stage'] == 'final_reviewed_group_root_xy'
    first = result['per_group'][0]['evidence']
    assert not first['accepted']
    assert first['reviewed_group_placement_measurement']['accepted']
    assert not first['reviewed_group_placement_final_measurement']['accepted']
    assert first['reviewed_group_placement_final_measurement']['root_xy_error_m'] == pytest.approx(.02, abs=1e-6)
    assert result['unrepaired_object_names'] == ['target']
    assert not result['physical_grasp_verified'] and not result['ik_verified']


def test_default_measured_search_retains_original_xy_grid_and_receipts(world_factory):
    world = world_factory()
    world.policy['support_search'] = 'measured_surfaces'
    group = {'target': world.target, 'companion': world.companion}
    bounds = {obj.name: obj.aabb for obj in group.values()}
    trace = {}
    proposals = list(setup._comparison_candidates(world.session, group, bounds, world.policy, 0.,
        np.array([.15, 0., .1]), trace))
    assert len(proposals) == 9
    result = world.run()
    assert result['accepted'], result
    assert world.target.position[:2] == pytest.approx([1.35, 0.])
    assert 'reviewed_group_placement' not in result
    assert 'reviewed_root_xy_retained' not in result['robot_spawn']['checks']

