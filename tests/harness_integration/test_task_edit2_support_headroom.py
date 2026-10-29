"""CPU native-shaped headroom ray tests; run ``python -B -m pytest -q
tests/test_task_edit2_support_headroom.py tests/test_task_edit2_tabletop.py``.
These fixtures validate proposal rejection, not native physical qualification.
"""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.tasks import support_headroom as module


class Actor:
    def __init__(self, name, low, high, *, path=None):
        self.name, self.prim_path = name, path or '/World/' + name
        self.category, self.model = 'native_' + name, 'model_' + name
        self.low, self.high = np.array(low, dtype=float), np.array(high, dtype=float)

    @property
    def aabb(self):
        return self.low.copy(), self.high.copy()

    def set_position_orientation(self, *args, **kwargs):
        raise AssertionError('Proposal measurement cannot mutate actor poses.')


def ray_hit(body, z, xy=(0., 0.), normal=(0., 0., 1.)):
    return {'hit': True, 'rigidBody': body, 'collision': body + '/collision',
        'position': np.array([*xy, z]), 'normal': np.array(normal)}


@pytest.fixture
def scene(monkeypatch):
    cabinet = Actor('bottom_cabinet', [-.5, -.5, 0.], [.5, .5, .8942])
    bottle = Actor('bottle', [-.04, -.04, .01], [.04, .04, .289])
    wall = Actor('wall', [.3, -.5, 0.], [.5, .5, 2.])
    robot = NS(name='native_robot', prim_path='/World/native_robot')
    sim = NS(current_time=3.25)
    # The actual robot is env.robots[0]; a stale session.robot must not define
    # a larger or different collision exclusion.
    session = NS(env=NS(scene=NS(objects=[cabinet, bottle, wall]), robots=[robot]),
        robot=NS(prim_path='/World/stale_robot'), native_reset_id='reset-a',
        pipeline=NS(og=NS(sim=sim)))
    calls = []
    def native_ray(start, end, *, only_closest):
        calls.append((list(start), list(end), only_closest))
        return [ray_hit(cabinet.prim_path + '/shelf', height, start[:2])
            for height in (.50564, .72774, .8942) if end[2] < height < start[2]]
    monkeypatch.setattr(module, 'native_headroom_ray', native_ray)
    return NS(session=session, support=cabinet, bottle=bottle, robot=robot, wall=wall, calls=calls,
        inner_bounds={'bottle': (np.array([-.04, -.04, .51064]), np.array([.04, .04, .78964]))},
        top_bounds={'bottle': (np.array([-.04, -.04, .8992]), np.array([.04, .04, 1.1782]))})


def measure(scene, *, top=False, excluded=('bottle',)):
    return module.measure_projected_headroom(scene.session, scene.support,
        scene.top_bounds if top else scene.inner_bounds,
        surface_z_m=.8942 if top else .50564, excluded_names=excluded)


def test_native_cabinet_inner_ceiling_rejects_unchanged_tall_bottle(scene):
    result = measure(scene)
    assert result['available'] is True and result['accepted'] is False
    assert result['reason'] == 'native_sampled_headroom_obstruction'
    assert len(result['obstructions']) == 9
    assert all(row['body'] == '/World/bottom_cabinet/shelf' and row['position_m'][2] == .72774
        for row in result['obstructions'])
    assert result['support']['native_bounds_m_provenance'][1][2] == .8942
    assert result['actors']['bottle']['native_bounds_m_provenance'] == [scene.bottle.low.tolist(), scene.bottle.high.tolist()]
    assert result['objects']['bottle']['projected_bounds_m'] == [part.tolist() for part in scene.inner_bounds['bottle']]
    assert result['sample_coverage']['completed_ray_count'] == 9
    assert all(call[2] is False for call in scene.calls)


def test_top_deck_no_sampled_obstruction_has_exact_nine_point_coverage_only(scene):
    result = measure(scene, top=True)
    assert result['available'] and result['accepted']
    assert result['reason'] == 'no_sampled_headroom_obstruction'
    assert 'not_full_collision_fit_grasp_or_path_certificate' in result['scope']
    assert not result['obstructions']
    samples = result['objects']['bottle']['samples']
    assert {tuple(row['xy_m']) for row in samples} == {(x, y) for x in (-.04, 0., .04) for y in (-.04, 0., .04)}
    assert len(samples) == result['sample_coverage']['expected_ray_count'] == 9
    assert samples[0]['ray_start_m'][2] == pytest.approx(1.1782 + .002)
    assert samples[0]['ray_end_m'][2] == pytest.approx(max(.8942 + .002, .8992 + .001))
    assert result['sample_coverage']['probe_endpoint_exclusion_m'] == .002
    assert result['provenance']['reset_id'] == 'reset-a'
    assert result['provenance']['native_time_s'] == 3.25
    json.dumps(result, allow_nan=False)


def test_foreign_wall_collision_with_horizontal_normal_is_definite_obstruction(scene, monkeypatch):
    def ray(start, end, **kwargs):
        return [ray_hit('/World/wall/base_link', 1., start[:2], (1., 0., 0.))] if start[:2] == [0., 0.] else []
    monkeypatch.setattr(module, 'native_headroom_ray', ray)
    result = measure(scene, top=True)
    assert result['available'] and not result['accepted']
    assert len(result['obstructions']) == 1
    assert result['obstructions'][0]['body'] == '/World/wall/base_link'
    assert result['obstructions'][0]['normal'] == [1., 0., 0.]


def test_only_exact_explicit_actor_roots_and_actual_robot_are_ignored(scene, monkeypatch):
    monkeypatch.setattr(module, 'native_headroom_ray', lambda start, end, **kwargs: [
        ray_hit('/World/bottle/base_link', 1., start[:2]),
        ray_hit('/World/native_robot/wheel', 1., start[:2]),
        ray_hit('/World/bottle_other/base_link', 1., start[:2]),
        ray_hit('/World/stale_robot/base_link', 1., start[:2])])
    result = measure(scene, top=True)
    assert result['actual_robot_root'] == scene.robot.prim_path
    assert result['excluded_actor_roots'] == [{'name': 'bottle', 'prim_path': '/World/bottle'}]
    assert result['available'] and not result['accepted']
    assert {row['body'] for row in result['obstructions']} == {'/World/bottle_other/base_link', '/World/stale_robot/base_link'}
    hits = result['objects']['bottle']['samples'][0]['hits']
    assert hits[0]['ignored_root'] == '/World/bottle'
    assert hits[1]['ignored_root'] == '/World/native_robot'
    # Without explicit exclusion, the original task actor is also obstruction.
    result = measure(scene, top=True, excluded=())
    assert '/World/bottle/base_link' in {row['body'] for row in result['obstructions']}


def test_selected_support_cannot_be_excluded_even_explicitly(scene):
    result = measure(scene, excluded=('bottle', 'bottom_cabinet'))
    assert not result['available'] and not result['accepted']
    assert 'must not be excluded' in result['error']
    assert scene.calls == []


def test_upper_shelf_child_is_not_ignored_as_original_task_actor(scene, monkeypatch):
    shelf = Actor('upper_shelf', [-.5, -.5, .72], [.5, .5, .73], path=scene.support.prim_path + '/upper_shelf')
    scene.session.env.scene.objects.append(shelf)
    result = measure(scene, excluded=('bottle', 'upper_shelf'))
    assert not result['available'] and not result['accepted']
    assert 'must not be excluded' in result['error']


def test_missing_native_ray_is_unavailable_not_empty_clearance(scene, monkeypatch):
    def missing(*args, **kwargs):
        raise ImportError('native ray API unavailable')
    monkeypatch.setattr(module, 'native_headroom_ray', missing)
    result = measure(scene)
    assert not result['available'] and not result['accepted']
    assert result['exception_type'] == 'ImportError'
    assert result['sample_coverage']['completed_ray_count'] == 0


@pytest.mark.parametrize('field', ['projected', 'support', 'source', 'surface', 'ray'])
def test_nonfinite_native_or_projected_geometry_stays_unavailable(scene, monkeypatch, field):
    if field == 'projected': scene.inner_bounds['bottle'][1][2] = np.nan
    elif field == 'support': scene.support.high[2] = np.nan
    elif field == 'source': scene.bottle.low[1] = np.nan
    elif field == 'surface':
        result = module.measure_projected_headroom(scene.session, scene.support, scene.inner_bounds, surface_z_m=np.nan)
        assert not result['available'] and not result['accepted']
        json.dumps(result, allow_nan=False)
        return
    else:
        monkeypatch.setattr(module, 'native_headroom_ray', lambda *args, **kwargs: [ray_hit('/World/wall', np.nan)])
    result = measure(scene)
    assert not result['available'] and not result['accepted']
    assert result['reason'] == 'native_headroom_measurement_unavailable'
    json.dumps(result, allow_nan=False)


def test_endpoint_and_outside_volume_hits_are_retained_but_not_obstructions(scene, monkeypatch):
    monkeypatch.setattr(module, 'native_headroom_ray', lambda start, end, **kwargs: [
        ray_hit('/World/bottom_cabinet/shelf', end[2], start[:2]),
        ray_hit('/World/bottom_cabinet/shelf', .8942, start[:2]),
        ray_hit('/World/wall/base_link', start[2] - .001, start[:2])])
    result = measure(scene, top=True)
    assert result['available'] and result['accepted']
    assert all(not hit['within_projected_vertical_probe']
        for row in result['objects']['bottle']['samples'] for hit in row['hits'])


def test_read_only_measurement_keeps_actors_inputs_and_native_clock(scene):
    original = deepcopy(scene.inner_bounds)
    native_before = [(obj.low.copy(), obj.high.copy()) for obj in scene.session.env.scene.objects]
    result = measure(scene)
    assert result['available'] and not result['accepted']
    assert scene.session.pipeline.og.sim.current_time == 3.25
    for obj, (low, high) in zip(scene.session.env.scene.objects, native_before):
        np.testing.assert_array_equal(obj.low, low)
        np.testing.assert_array_equal(obj.high, high)
    for name, bounds in original.items():
        for actual, expected in zip(scene.inner_bounds[name], bounds):
            np.testing.assert_array_equal(actual, expected)
