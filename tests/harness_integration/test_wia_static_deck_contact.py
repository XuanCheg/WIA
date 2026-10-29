"""Check static-deck contact bounds on CPU without importing a simulator.

Run: python -B -m pytest -q -p no:cacheprovider
tests/test_wia_static_deck_contact.py tests/test_wia_spatial_tray.py
Native interfaces are fakes; these checks do not validate physical operation.
"""
from copy import deepcopy
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from wia.authoring.converters import wia_spatial_tray as adapter
from wia.tasks.wia_spatial_tray_scoring import SpatialTrayProgress
from harness_integration.test_wia_spatial_tray import parameters, sample, retrieved, native_fixture_world


@pytest.fixture
def world(monkeypatch):
    touching = object()
    states = ModuleType('omnigibson.object_states')
    states.Touching = touching
    monkeypatch.setitem(sys.modules, states.__name__, states)
    usd = ModuleType('omnigibson.utils.usd_utils')
    calls = []

    def missing_static_index(scene_idx, *, row_prim_paths, column_prim_paths):
        calls.append((scene_idx, row_prim_paths, column_prim_paths))
        raise KeyError(column_prim_paths[0])

    usd.RigidContactAPI = SimpleNamespace(get_contact_data=missing_static_index)
    monkeypatch.setitem(sys.modules, usd.__name__, usd)
    mesh = SimpleNamespace(prim_path='/World/deck/base_link/collisions', collision_enabled=True,
        aabb=(np.array([-1., -1., -.02]), np.array([1., 1., 0.])),
        get_collision_approximation=lambda: 'boundingCube',
        get_position_orientation=lambda: (np.array([0., 0., -.01]), np.array([0., 0., 0., 1.])))
    link = SimpleNamespace(prim_path='/World/deck/base_link', is_meta_link=False,
                           collision_meshes={'collisions': mesh})
    deck = SimpleNamespace(fixed_base=True, visual_only=False, _primitive_type='Cube',
        links={'base_link': link}, aabb=mesh.aabb)
    target = SimpleNamespace(fixed_base=False, visual_only=False, scene=SimpleNamespace(idx=0),
        aabb=(np.array([-.05, -.05, -.0001]), np.array([.05, .05, .1])),
        states={touching: SimpleNamespace(get_value=lambda other: True)},
        links={'base_link': SimpleNamespace(prim_path='/World/target/base_link')})
    return SimpleNamespace(target=target, deck=deck, mesh=mesh, link=link, touching=touching,
                           api=usd.RigidContactAPI, calls=calls)


def measure(world, *, on_top=True, tolerance=.001):
    return adapter._deck_pair(world.target, world.deck, tolerance, on_top=on_top)


def test_missing_static_pair_index_uses_explicit_geometric_bound(world):
    before = deepcopy((world.target.aabb, world.mesh.aabb))
    result = measure(world)
    assert len(world.calls) == 1
    assert result['available'] and result['relation'] == 'touching' and not result['penetrating']
    assert result['separation_source'] == 'native_fixed_cube_top_bound'
    assert result['penetration_depth_upper_bound_m'] == pytest.approx(.0001)
    assert 'KeyError' in result['native_pair_contact_geometry']['reason']
    assert 'minimum_separation_m' not in result and 'separations_m' not in result
    for old, current in zip(before, (world.target.aabb, world.mesh.aabb)):
        np.testing.assert_array_equal(old, current)


def test_saved_t26_bounds_provide_submicrometre_bound_not_new_native_validation(world):
    # Historical obj2/deck collision bounds; no simulation is replayed here.
    world.target.aabb = (np.array([-3.663146495819092, 13.499387741088867, .8896105885505676]),
                        np.array([-3.6016845703125, 13.533182144165039, 1.043398141860962]))
    world.mesh.aabb = (np.array([-3.9700000286102295, 12.949999809265137, .8696109056472778]),
                      np.array([-3.490000009536743, 14.15000057220459, .8896108865737915]))
    world.deck.aabb = world.mesh.aabb
    result = measure(world)
    assert result['available']
    assert result['penetration_depth_upper_bound_m'] == pytest.approx(2.980232238769531e-7)


def test_empty_sleeping_pair_still_requires_native_contact_and_bound(world):
    world.api.get_contact_data = lambda *args, **kwargs: []
    result = measure(world)
    assert result['available'] and result['penetration_depth_upper_bound_m'] == pytest.approx(.0001)
    assert 'no pair-specific separation samples' in result['native_pair_contact_geometry']['reason']


@pytest.mark.parametrize('depth,passes', [(0., True), (.0009, True), (.001, True), (.001001, False), (.01, False)])
def test_unchanged_penetration_threshold_and_inconclusive_large_bound(world, depth, passes):
    world.target.aabb[0][2] = -depth
    result = measure(world)
    assert result['available'] is passes
    if not passes:
        assert result['penetrating'] is None and result['relation'] == 'unresolved'
        assert result['fixed_deck_bound']['penetration_depth_upper_bound_m'] == pytest.approx(depth)


@pytest.mark.parametrize('fault', ['no_contact', 'not_on_top', 'not_fixed', 'visual_only', 'wrong_shape',
    'disabled_collision', 'wrong_approximation', 'tilted', 'missing_mesh', 'multiple_meshes',
    'outside', 'side_boundary', 'fully_buried', 'nonfinite_bounds', 'nonfinite_pose', 'fixed_target'])
def test_missing_or_inapplicable_evidence_cannot_pass(world, fault):
    on_top = True
    if fault == 'no_contact':
        world.target.states[world.touching].get_value = lambda other: False
    elif fault == 'not_on_top':
        on_top = False
    elif fault == 'not_fixed':
        world.deck.fixed_base = False
    elif fault == 'visual_only':
        world.deck.visual_only = True
    elif fault == 'wrong_shape':
        world.deck._primitive_type = 'Cylinder'
    elif fault == 'disabled_collision':
        world.mesh.collision_enabled = False
    elif fault == 'wrong_approximation':
        world.mesh.get_collision_approximation = lambda: 'convexDecomposition'
    elif fault == 'tilted':
        world.mesh.get_position_orientation = lambda: (np.zeros(3), np.array([.001, 0., 0., 1.]))
    elif fault == 'missing_mesh':
        world.link.collision_meshes = {}
    elif fault == 'multiple_meshes':
        world.link.collision_meshes['extra'] = world.mesh
    elif fault == 'outside':
        world.target.aabb[0][0] = -1.01
    elif fault == 'side_boundary':
        world.target.aabb[0][0] = -1.
    elif fault == 'fully_buried':
        world.target.aabb[1][2] = -.00001
    elif fault == 'nonfinite_bounds':
        world.mesh.aabb[0][2] = np.nan
    elif fault == 'nonfinite_pose':
        world.mesh.get_position_orientation = lambda: (np.zeros(3), np.array([0., 0., 0., np.nan]))
    elif fault == 'fixed_target':
        world.target.fixed_base = True
    result = measure(world, on_top=on_top)
    assert result['available'] is False and result['penetrating'] is None


def test_missing_getter_is_not_false_or_zero_evidence(world):
    del world.target.states[world.touching]
    result = measure(world)
    assert result['available'] is False and result['penetrating'] is None
    assert 'KeyError' in result['reason']


@pytest.mark.parametrize('separation', [-.002, -.0005])
def test_available_native_pair_evidence_always_takes_precedence(world, separation):
    world.api.get_contact_data = lambda *args, **kwargs: [(None, None, None, None, None, separation)]
    result = measure(world)
    assert result['available'] and result['separation_source'] == 'native_physx_contact_data'
    assert result['penetrating'] is (separation < -.001)
    assert 'fixed_deck_bound' not in result


def test_unrelated_pair_still_requires_physx_data(world):
    result = adapter._pair(world.target, world.deck, .001)
    assert not result['available'] and 'KeyError' in result['reason']


def test_native_measurement_path_uses_the_bound_only_for_deck(native_fixture_world, monkeypatch):
    world = native_fixture_world
    states = sys.modules['omnigibson.object_states']
    states.Touching = object()
    deck = world.objects['deck']
    deck._primitive_type = 'Cube'
    mesh = SimpleNamespace(collision_enabled=True, prim_path='/World/deck/collisions', aabb=deck.aabb,
        get_collision_approximation=lambda: 'boundingCube',
        get_position_orientation=lambda: (np.zeros(3), np.array([0., 0., 0., 1.])))
    deck.links = {'base_link': SimpleNamespace(is_meta_link=False, collision_meshes={'collision': mesh})}
    for key, row in sample()['targets'].items():
        obj = world.objects[key]
        obj.aabb[0][:], obj.aabb[1][:] = row['aabb_min'], row['aabb_max']
        obj.states[states.OnTop].get_value = lambda other: other is deck
        obj.states[states.Touching] = SimpleNamespace(get_value=lambda other: other is deck)
    original_pair = adapter._pair

    def pair(first, second, tolerance):
        if second is deck:
            return {'available': False, 'relation': 'unresolved', 'penetrating': None,
                    'reason': 'KeyError: static deck has no matrix column'}
        return original_pair(first, second, tolerance)

    monkeypatch.setattr(adapter, '_pair', pair)
    result = adapter._measure(world.session, None, world.config, world.objects)
    assert result['fixtures_valid'] and result['tray_valid']
    for row in result['targets'].values():
        assert row['on_top_deck']
        assert row['deck_contact_geometry']['separation_source'] == 'native_fixed_cube_top_bound'
        assert row['deck_contact_geometry']['available']
    assert world.velocity_calls == []


def test_full_reducer_still_requires_three_retrievals_release_shape_and_dwell(world):
    geometry = measure(world)
    value = sample(.4)
    for row in value['targets'].values():
        row['deck_contact_geometry'] = deepcopy(geometry)
    scorer = retrieved()
    assert not scorer.update(value)['task_success']
    value['native_time_s'] = .9
    result = scorer.update(value)
    assert result['task_success'] and result['scoring_version'] == 'wia_spatial_tray_progress_v2'
    for fault in ('held', 'robot_support', 'bad_bound', 'no_retrieval'):
        scorer = retrieved() if fault != 'no_retrieval' else SpatialTrayProgress(parameters())
        if fault == 'no_retrieval':
            scorer.begin(sample(initial=True))
        bad = deepcopy(value)
        bad['native_time_s'] = .4
        if fault == 'held':
            bad['grasp'].update(held_names=['native_c'], released=False)
        elif fault == 'robot_support':
            bad['targets']['c']['robot_contact'] = True
        elif fault == 'bad_bound':
            world.target.aabb[0][2] = -.002
            bad['targets']['c']['deck_contact_geometry'] = measure(world)
        scorer.update(bad)
        bad['native_time_s'] = .9
        assert not scorer.update(bad)['task_success']
