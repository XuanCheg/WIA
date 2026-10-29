"""Check measured upright gates and bounded independent-group source proposals.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_wia_stance_upright.py
CPU native-shaped fixtures do not certify a native stance, reach or route.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from wia.tasks import interaction_setup as setup
from harness_integration.test_esi_interaction_runtime import world_factory


@pytest.mark.parametrize('quaternion,vertical,upright', [
    ([0., 0., 0., 1.], 1., True),
    ([0., 0., np.sin(.7), np.cos(.7)], 1., True),
    ([0., 0., -np.sin(.7), -np.cos(.7)], 1., True),
    ([0., 0., 5., 5.], 1., True),
    ([np.sin(.1), 0., 0., np.cos(.1)], np.cos(.2), True),
    ([0., np.sin(.2), 0., np.cos(.2)], np.cos(.4), False),
    ([1., 0., 0., 0.], -1., False),
])
def test_native_upright_uses_normalized_vertical_axis(quaternion, vertical, upright):
    measured = setup._upright_check(quaternion)
    assert measured['upright_measurement_available']
    assert measured['upright'] is upright
    assert measured['upright_vertical_axis_dot'] == pytest.approx(vertical)
    assert measured['upright_quaternion_xyzw'] == quaternion
    assert np.linalg.norm(measured['upright_normalized_quaternion_xyzw']) == pytest.approx(1.)


@pytest.mark.parametrize('quaternion', [None, [], [0., 0., 0., 0.],
    [0., 0., np.nan, 1.], [0., 0., 0., np.inf], [0., 0., 1.]])
def test_unavailable_or_invalid_native_orientation_fails_closed(quaternion):
    measured = setup._upright_check(quaternion)
    assert not measured['upright']
    assert not measured['upright_measurement_available']
    assert measured['upright_vertical_axis_dot'] is None


def test_contact_receipt_keeps_floor_support_separate_from_upright(world_factory):
    world = world_factory()
    world.robot.quaternion = np.array([1., 0., 0., 0.])
    measured = setup._robot_contact_check(world.robot, ['/World/floor'], 0.)
    assert measured['floor_supported'] and measured['collision_free']
    assert measured['upright_vertical_axis_dot'] == -1.
    assert not measured['upright']
    unavailable = SimpleNamespace(prim_path=world.robot.prim_path, contact_list=world.robot.contact_list)
    measured = setup._robot_contact_check(unavailable, ['/World/floor'], 0.)
    assert measured['floor_supported'] and not measured['upright_measurement_available']
    assert not measured['upright']


def force_tilt_after_placement(world, monkeypatch, *, first_placement=1):
    step = world.sim.step

    def tilted_step():
        step()
        if world.placements >= first_placement:
            world.robot.quaternion = np.array([1., 0., 0., 0.])

    monkeypatch.setattr(world.sim, 'step', tilted_step)


def test_generic_floor_contact_cannot_accept_a_toppled_robot(world_factory, monkeypatch):
    world = world_factory()
    force_tilt_after_placement(world, monkeypatch)
    result = world.run()
    assert not result['accepted']
    assert len(result['attempts']) == 2
    assert all(row['checks']['floor_supported'] and row['checks']['collision_free']
               and not row['checks']['upright'] for row in result['attempts'])
    assert 'robot_upright_gate_failed' in {row['code'] for row in setup.interaction_failure_reasons(result)}
    world.assert_restored()


def test_safe_source_and_repaired_spawn_both_reject_toppled_floor_contact(world_factory, monkeypatch):
    world = world_factory()
    world.robot.quaternion = np.array([1., 0., 0., 0.])
    baseline = world.sim.dump_state(serialized=False)
    force_tilt_after_placement(world, monkeypatch)
    result = {}
    setup._repair_safe_spawn(world.session, world.robot, {'target': world.target}, 'room_0',
        lambda: world.sim.load_state(baseline, serialized=False), result)
    assert result['original_robot_start_checks']['floor_supported']
    assert not result['original_robot_start_checks']['upright']
    assert not result.get('original_robot_start_safe')
    assert not result.get('robot_spawn_repaired')
    assert 'upright start' in result['safe_spawn_repair_error']
    assert world.robot.quaternion.tolist() == [1., 0., 0., 0.]


def test_restored_independent_group_stance_still_requires_upright(world_factory, monkeypatch):
    world = world_factory()
    world.companion.position[0] = 3.
    world.policy.update(mode='inspection', relocation_policy='independent_groups',
        manipulation_bindings=['target', 'companion'],
        placement_groups=[['target'], ['companion']], source_camera_policy='preserve')
    force_tilt_after_placement(world, monkeypatch, first_placement=2)
    result = world.run()
    assert not result['accepted']
    assert result['per_group'][0]['evidence']['accepted']
    second = result['per_group'][1]['evidence']
    assert not second['accepted']
    native_attempts = [row for row in second['attempts'] if 'floor_supported' in row['checks']]
    assert native_attempts
    assert all(row['checks']['floor_supported'] and row['checks']['collision_free']
               and not row['checks']['upright'] for row in native_attempts)


def source_proposal_fixture(world, monkeypatch):
    # The supported source cell lies outside the target-nearest four cells.
    cells = np.array([[0., -1.], [1.5, -.60], [1.6, -.60], [1.7, -.60], [1.8, -.60]])
    world.session.env.scene.trav_map = SimpleNamespace(map_resolution=.1)
    world.policy.update(support_search='measured_surfaces', _independent_group_key='group_0')
    monkeypatch.setattr(setup, '_navigation_cells', lambda session, robot, region, excluded:
        (cells.copy(), 0., {'requested_source_region': region, 'candidate_cell_count': len(cells)}))
    return cells


def test_valid_source_proposal_is_first_without_claiming_access(world_factory, monkeypatch):
    world = world_factory()
    cells = source_proposal_fixture(world, monkeypatch)
    result = world.run()
    assert result['accepted'], result
    assert len(result['attempts']) == 1
    assert result['robot_spawn']['robot_position_m'][:2] == cells[0].tolist()
    receipt = result['robot_spawn']['retained_valid_source_stance_proposal']
    assert receipt == result['retained_valid_source_stance_proposal']
    assert receipt['original_native_checks']['upright']
    assert receipt['grid_match_distance_m'] == 0.
    assert receipt['measured_map_resolution_tolerance_m'] == .1
    assert receipt['initial_approach_required']
    assert receipt['proposal_only'] and receipt['native_post_placement_acceptance_required']
    assert not receipt['arm_reach_verified'] and not receipt['route_verified']
    assert 'initial_base_approach_required' in result['warning_codes']


def test_tipped_or_off_grid_source_is_not_retained(world_factory, monkeypatch):
    world = world_factory()
    cells = source_proposal_fixture(world, monkeypatch)
    target = np.array([1.5, 0., .8])
    world.robot.quaternion = np.array([1., 0., 0., 0.])
    assert setup._retained_source_stance_proposal(world.session, world.robot, cells,
        ['/World/floor'], 0., target, world.policy) == (None, None)
    world.robot.quaternion = np.array([0., 0., 0., 1.])
    assert setup._retained_source_stance_proposal(world.session, world.robot, cells[1:],
        ['/World/floor'], 0., target, world.policy) == (None, None)
    for overrides in ({'_independent_group_key': None}, {'support_search': 'bounded_group'},
            {'relocation_policy': 'seeded_geometry'}, {'layout_parameters': {'inclined_apparatus': True}}):
        assert setup._retained_source_stance_proposal(world.session, world.robot, cells,
            ['/World/floor'], 0., target, {**world.policy, **overrides}) == (None, None)


def test_retained_source_does_not_bypass_post_placement_gates_or_four_attempt_limit(world_factory, monkeypatch):
    world = world_factory()
    cells = source_proposal_fixture(world, monkeypatch)
    candidates = setup._comparison_candidates
    monkeypatch.setattr(setup, '_comparison_candidates', lambda *args: [next(iter(candidates(*args)))])
    force_tilt_after_placement(world, monkeypatch)
    result = world.run()
    assert not result['accepted']
    assert len(result['attempts']) == 4
    assert result['attempts'][0]['robot_position_m'][:2] == cells[0].tolist()
    assert all(row['checks']['floor_supported'] and row['checks']['collision_free']
               and not row['checks']['upright'] for row in result['attempts'])
    world.assert_restored()
