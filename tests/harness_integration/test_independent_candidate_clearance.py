"""Check independent candidate clearance with source bounds and CPU native APIs.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q \
    tests/test_independent_candidate_clearance.py
No native renderer, GPU workload or robot action is launched.
"""
import numpy as np
import pytest

from wia.tasks import interaction_setup as setup
from wia.tasks import interaction_groups as groups
from harness_integration.test_esi_interaction_runtime import world_factory


def test_real_thin_chopping_block_must_not_hide_under_gift_box():
    # Native preview capture_00a901a1e5d885b9623714eea9589b54, final AABBs.
    gift = ([-17.37403679, -5.30101299, .37135133], [-17.25173569, -5.20863485, .51366711])
    board = ([-17.35983467, -5.29631758, .37233555], [-17.26587296, -5.21331692, .37694278])
    xyz_overlap = np.minimum(gift[1], board[1]) - np.maximum(gift[0], board[0])
    assert not np.all(xyz_overlap > .005), 'The old 5 mm XYZ penetration screen misses this board.'
    result = groups.independent_group_clearance({'middle': {'obj_fixed': gift},
        'base': {'obj_A_wrong': board}})
    assert not result['accepted'] and result['pairs'][0]['xy_gap_m'] == 0.
    assert result['pairs'][0]['objects'] == ['obj_fixed', 'obj_A_wrong']
    assert result['pairs'][0]['minimum_gap_m'] == .03
    assert not result['optical_coverage_verified'] and not result['ik_verified']


def test_adjacent_candidates_on_one_support_allowed_but_vertical_stacking_rejected():
    a = ([0., 0., .7], [.1, .1, .8])
    adjacent = ([.13, 0., .7], [.23, .1, .8])
    high = ([0., 0., .9], [.1, .1, 1.])
    assert groups.independent_group_clearance({'a': {'a': a}, 'b': {'b': adjacent}})['accepted']
    assert not groups.independent_group_clearance({'a': {'a': a}, 'b': {'b': high}})['accepted']
    assert not groups.independent_group_clearance({'a': {'a': a},
        'b': {'b': ([.129, 0., .7], [.229, .1, .8])}})['accepted']


def test_content_and_container_in_one_meaningful_group_are_not_separated():
    assert groups.independent_group_clearance({'contained_pair': {
        'container': ([0., 0., .7], [.3, .3, 1.]),
        'content': ([.1, .1, .75], [.2, .2, .9])}})['accepted']


def test_supported_overlapping_candidates_force_a_new_clear_layout(world_factory):
    world = world_factory()
    for obj in (world.target, world.companion):
        obj.position[:] = [1.5, 0., .805]
    world.policy.update(mode='inspection', relocation_policy='independent_groups',
        manipulation_bindings=['target', 'companion'], placement_groups=[['target'], ['companion']],
        source_camera_policy='preserve', support_search='measured_surfaces')
    result = world.run()
    assert result['accepted'], result
    second = result['per_group'][1]['evidence']
    assert not second['independent_group_clearance_before']['accepted']
    assert 'independent_candidate_footprint_requires_repair' in second['warning_codes']
    rejected = [a for a in second['attempts'] if a.get('failure_stage') == 'projected_independent_group_clearance']
    assert rejected and not rejected[0]['independent_group_clearance']['accepted']
    assert result['independent_group_clearance']['accepted']
    assert result['independent_group_clearance']['pairs'][0]['xy_gap_m'] >= .03 - 1e-8
    assert result['per_group'][1]['evidence']['robot_spawn']['checks']['independent_groups_horizontal_clearance']
    assert np.array_equal(world.target.scale, np.ones(3))
    assert np.array_equal(world.companion.scale, np.ones(3))


def test_measured_settling_collision_rejects_a_projected_clear_attempt(world_factory, monkeypatch):
    world = world_factory()
    world.policy.update(mode='inspection', relocation_policy='independent_groups',
        placement_groups=[['target'], ['companion']], support_search='measured_surfaces')
    original_place = setup._place_robot
    def place(*args, **kwargs):
        original_place(*args, **kwargs)
        if world.placements > 1 and world.on_table(world.companion):
            world.target.position[:2] = world.companion.position[:2]
    monkeypatch.setattr(setup, '_place_robot', place)
    result = world.run()
    assert not result['accepted']
    attempts = result['per_group'][1]['evidence']['attempts']
    assert any(not a['checks'].get('independent_groups_horizontal_clearance', True)
        and a.get('independent_group_clearance', {}).get('applicable') for a in attempts)


def test_final_assembly_remeasures_previously_accepted_groups(world_factory):
    world = world_factory()
    world.policy.update(mode='inspection', relocation_policy='independent_groups',
        placement_groups=[['target'], ['companion']])
    count = 0
    def claimed_success(session, bindings, policy):
        nonlocal count
        count += 1
        name = bindings[policy['group_bindings'][0]].name
        # A later group's settling changes a previously accepted candidate.
        world.target.position[:] = [1.5, 0., .805]
        world.companion.position[:] = [1.5, 0., .805]
        return {'accepted': True, 'relocated': False, 'object_names': [name],
            'object_translations_m': {name: [0., 0., 0.]}, 'adapted_geometry': {}}
    result = groups.initialize_groups(world.session, world.bindings, world.policy, claimed_success)
    assert count == 2 and all(r['evidence']['accepted'] for r in result['per_group'])
    assert not result['accepted'] and result['failure_stage'] == 'final_independent_group_clearance'
    assert not result['independent_group_clearance']['pairs'][0]['accepted']


def test_common_z_does_not_inject_independent_xy_constraints(world_factory):
    world = world_factory()
    world.table.position[:2] = [.15, 0.]
    before = [obj.position.copy() for obj in (world.target, world.companion)]
    world.policy.update(mode='inspection', relocation_policy='vertical_only',
        manipulation_bindings=['target', 'companion'], placement_groups=[['target'], ['companion']],
        source_camera_policy='preserve')
    result = world.run()
    assert result['accepted'], result
    assert 'independent_group_clearance' not in result
    for obj, original in zip((world.target, world.companion), before):
        assert obj.position[:2] == pytest.approx(original[:2])
    assert all('independent_group_clearance_before' not in row['evidence'] for row in result['per_group'])
