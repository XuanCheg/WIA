"""Run: python -m pytest tests/harness_integration/test_pickup_edges.py."""
import numpy as np
import pytest
from wia.tasks.pickup_edges import edge_candidates


def test_edge_candidates_use_full_offset_footprint_with_exact_margin():
    support = np.array([[0, 0, .5], [1, 2, .6]])
    bounds = np.array([[.3, .7, .6], [.5, 1., .8]])
    root = np.array([.45, .8, .65])
    for xy, axis, sign in edge_candidates(support, bounds, root, .03):
        assert xy[1-axis] == root[1-axis]
        shifted = bounds[:, :2] + xy-root[:2]
        assert np.all(shifted[0] >= support[0, :2]+.03-1e-9)
        assert np.all(shifted[1] <= support[1, :2]-.03+1e-9)
        edge_gap = shifted[0, axis]-support[0, axis] if sign < 0 else support[1, axis]-shifted[1, axis]
        assert edge_gap == pytest.approx(.03)


def test_oversize_target_has_no_false_supported_edge():
    with pytest.raises(ValueError, match='does not fit'):
        edge_candidates([[0, 0, 0], [.1, .1, .5]], [[0, 0, 0], [.2, .2, .2]], [0, 0, 0], .03)


class Prop:
    def __init__(self, name, low, high, fixed=False):
        self.name = name
        self.position = (np.asarray(low) + high) / 2
        self.offsets = np.array([low, high]) - self.position
        self.quat = np.array([0., 0., 0., 1.])
        self.fixed_base = fixed

    @property
    def aabb(self):
        return self.offsets + self.position

    def get_position_orientation(self):
        return self.position.copy(), self.quat.copy()

    def set_position_orientation(self, position, orientation):
        self.position = np.asarray(position)
        self.quat = np.asarray(orientation)

    def keep_still(self):
        pass


def test_related_group_translation_preserves_containment_and_orientation():
    from wia.tasks.pickup_edges import related_components, _PlacementGroup
    outer = Prop('container', [0., 0., 1.], [.4, .4, 1.4])
    inner = Prop('contents', [.1, .1, 1.1], [.2, .2, 1.2])
    separate = Prop('separate', [1., 1., 1.], [1.2, 1.2, 1.2])
    groups = related_components([outer, inner, separate])
    assert [[obj.name for obj in group] for group in groups] == [['container', 'contents'], ['separate']]
    delta_before = inner.position - outer.position
    group = _PlacementGroup(groups[0])
    group.set_position_orientation(outer.position + [.2, 0., 0.], outer.quat)
    np.testing.assert_allclose(inner.position - outer.position, delta_before)
    np.testing.assert_array_equal(inner.quat, [0., 0., 0., 1.])
    np.testing.assert_allclose(separate.position, [1.1, 1.1, 1.1])


def test_policy_audits_fixed_held_and_failed_placements_without_retries(monkeypatch):
    from types import SimpleNamespace
    from wia.tasks import pickup_edges as edges
    fixed = Prop('fixed', [0., 0., 0.], [.2, .2, .2], fixed=True)
    held = Prop('held', [1., 0., 0.], [1.2, .2, .2])
    free = Prop('free', [2., 0., 0.], [2.2, .2, .2])
    support = SimpleNamespace(name='table', category='table')
    session = SimpleNamespace(_category_initialization_active=True,
        robot=SimpleNamespace(_ag_obj_in_hand={'right': held}),
        env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda *args: support)))
    before = free.position.copy()
    def blocked(*args, **kwargs):
        raise ValueError('footprint_support')
    monkeypatch.setattr(edges, 'place_pickup_near_edge', blocked)
    edges.apply_edge_policy(session, [fixed, held, free], supports={'free': 'table'})
    rows = {row['object']: row for row in session.pickup_edge_placements}
    assert rows['fixed']['reason'] == 'fixed_apparatus_or_visual_only'
    assert rows['held']['reason'] == 'currently_held'
    assert rows['free']['reason'] == 'footprint_support'
    np.testing.assert_array_equal(free.position, before)
    edges.apply_edge_policy(session, [fixed, held, free])
    assert len(session.pickup_edge_placements) == 3


def test_failed_settling_restores_entire_related_group(monkeypatch):
    from types import SimpleNamespace, ModuleType
    import sys
    from wia.tasks.pickup_edges import settle_edge_policy
    state_type = type('Touching', (), {})
    module = ModuleType('omnigibson.object_states')
    module.Touching = state_type
    monkeypatch.setitem(sys.modules, 'omnigibson.object_states', module)
    objects = {name: Prop(name, [i, 0., 1.], [i + .1, .1, 1.1])
               for i, name in enumerate(('container', 'contents'))}
    table = Prop('table', [-2., -2., .5], [3., 3., 1.])
    objects['table'] = table
    for obj in objects.values():
        obj.states = {state_type: SimpleNamespace(get_value=lambda support: False)}
        obj.get_linear_velocity = lambda: np.zeros(3)
    session = SimpleNamespace(env=SimpleNamespace(scene=SimpleNamespace(
        object_registry=lambda _, name: objects[name])),
        pipeline=SimpleNamespace(og=SimpleNamespace(sim=SimpleNamespace(step=lambda: None))))
    rows = [dict(object=name, status='placed', support='table',
        related_objects=['container', 'contents'], before_position_m=[i, .4, 1.05],
        before_quaternion_xyzw=[0., 0., 0., 1.], position_m=[i, .05, 1.05],
        support_surface_z_m=1.)
        for i, name in enumerate(('container', 'contents'))]
    settle_edge_policy(session, rows)
    for row in rows:
        assert row['status'] == 'unchanged'
        assert row['reason'] == 'post_settle_support_check_failed'
        np.testing.assert_allclose(objects[row['object']].position, row['before_position_m'])


def test_surface_search_moves_inward_on_only_the_original_axis():
    from wia.tasks.pickup_edges import surface_edge_candidates
    root = np.array([.45, .8, .65])
    support = [[0, 0, .5], [1, 2, .6]]
    bounds = [[.3, .7, .6], [.5, 1., .8]]
    candidates = list(surface_edge_candidates(support, bounds, root, .03))
    assert len(candidates) == 80
    distances = {}
    for xy, axis, sign, fraction in candidates:
        assert xy[1-axis] == root[1-axis]
        distance = abs(xy[axis] - root[axis])
        face = axis, sign
        assert 0 < fraction <= 1
        if face in distances:
            assert distance < distances[face]
        distances[face] = distance
