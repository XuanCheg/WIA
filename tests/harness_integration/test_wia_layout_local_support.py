"""Check optional bound local support rays using CPU initialization doubles.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_layout_local_support.py
No simulator, GPU, renderer or physical operability workload is started.
"""
from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from wia.authoring.converters import active_layout
from wia.tasks import interaction_setup


LOCAL_Z = .4309856
GLOBAL_TOP_Z = .4632726


class Vector(np.ndarray):
    def clone(self):
        return self.copy().view(Vector)


class Target:
    def __init__(self):
        self.name, self.prim_path = 'target', '/World/target'
        self.position = np.array([1., 2., 2.]).view(Vector)
        self.quaternion = np.array([0., 0., 0., 1.]).view(Vector)
        self.extent = np.array([.2, .2, .2])
        self.speed, self.touching = 0., True
        self.pose_writes, self.still_calls = [], 0

    @property
    def aabb(self):
        return self.position - self.extent / 2, self.position + self.extent / 2

    def get_position_orientation(self):
        return self.position.clone(), self.quaternion.clone()

    def set_position_orientation(self, position, orientation):
        self.position, self.quaternion = position.clone(), orientation.clone()
        self.pose_writes.append((self.position.copy(), self.quaternion.copy()))

    def keep_still(self):
        self.still_calls += 1

    def get_linear_velocity(self):
        return [self.speed, 0., 0.]


def hit(z=LOCAL_Z, *, body='/World/support/base/collision', normal=1., distance=.1):
    return {'hit': True, 'rigidBody': body, 'normal': [0., 0., normal],
            'position': [1., 2., z], 'distance': distance}


def parameters(mode='local_raycast', alignment=None):
    spec = {'name': 'target', 'support': 'support'}
    if mode is not None:
        spec['support_surface_mode'] = mode
    if alignment is not None:
        spec['support_alignment_mode'] = alignment
    return {'revision': 'fixture_support_v1', 'layout': {
        'objects': [{'name': 'target', 'position': [1., 2., 2.], 'orientation': [0., 0., 0., 1.]}],
        'supported_items': [spec], 'floor_name': 'floor'}}


@pytest.fixture
def native(monkeypatch):
    target = Target()
    support = NS(name='support', prim_path='/World/support',
                 aabb=np.array([[0., 1., .10], [2., 3., GLOBAL_TOP_Z]]))
    floor = NS(name='floor', prim_path='/World/floor', aabb=np.array([[-3., -3., -.1], [3., 3., 0.]]))
    objects = {'target': target, 'support': support, 'floor': floor}
    robot = object()
    session = NS(_category_initialization_active=True, robot=robot,
                 env=NS(scene=NS(object_registry=lambda kind, name: objects.get(name))))
    calls = {'rays': [], 'steps': 0, 'robot_checks': []}
    runtime = NS(hits=[[hit()], [hit()]], session=session, target=target, support=support,
                 calls=calls, settled_bottom=None)
    og, states, sampling, torch = (ModuleType(name) for name in (
        'omnigibson', 'omnigibson.object_states', 'omnigibson.utils.sampling_utils', 'torch'))
    states.Touching = object()
    target.states = {states.Touching: NS(get_value=lambda obj: obj is support and target.touching)}

    def step():
        calls['steps'] += 1
        if runtime.settled_bottom is not None:
            target.position[2] = runtime.settled_bottom + target.extent[2] / 2

    def raytest(start, end, **kwargs):
        calls['rays'].append((start, end, kwargs))
        return runtime.hits[len(calls['rays']) - 1]

    def robot_check(*args):
        calls['robot_checks'].append(args)
        return {'floor_supported': True, 'collision_free': True}

    og.sim = NS(step=step)
    sampling.raytest = raytest
    torch.float32 = np.float32
    torch.tensor = lambda value, dtype: np.asarray(value, dtype=dtype).view(Vector)
    for module in (og, states, sampling, torch):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(interaction_setup, '_robot_contact_check', robot_check)
    return runtime


def test_local_mode_uses_exact_bound_upward_surface_for_both_rays(native):
    # Nearer hits on another actor, a similarly prefixed actor, and a downward
    # face must not supply the bound support's local height.
    hits = [hit(.9, body='/World/other', distance=.001),
            hit(.8, body='/World/support_other/base', distance=.002),
            hit(.7, normal=-1., distance=.003), hit(.6, normal=.5, distance=.004),
            hit(.45, distance=.2), hit(LOCAL_Z, distance=.1)]
    native.hits = [deepcopy(hits), deepcopy(hits)]
    p = parameters()
    before = deepcopy(p)
    result = active_layout.initialize_scene(native.session, p)
    receipt = result['supported_items']['target']
    assert result['accepted'] is receipt['accepted'] is True
    assert receipt['support_surface_mode'] == 'local_raycast'
    assert receipt['support_surface_z_m'] == LOCAL_Z
    assert receipt['support_aabb_top_z_m'] == GLOBAL_TOP_Z
    assert receipt['support_ray_distance_m'] == .1
    assert receipt['support_hit_position_m'] == [1., 2., LOCAL_Z]
    assert receipt['bottom_gap_m'] == pytest.approx(.003, abs=1e-7)
    assert receipt['touching'] is receipt['within_footprint'] is True
    assert native.calls['rays'][0] == ([1., 2., GLOBAL_TOP_Z + .10], [1., 2., 0.], {'only_closest': False})
    low, high = native.target.aabb
    xy = ((low + high) / 2)[:2].tolist()
    assert native.calls['rays'][1] == ([*xy, GLOBAL_TOP_Z + .10], [*xy, 0.], {'only_closest': False})
    assert receipt['support_ray_start_m'] == native.calls['rays'][1][0]
    assert receipt['support_ray_end_m'] == native.calls['rays'][1][1]
    assert native.calls['steps'] == 90
    assert native.calls['robot_checks'] == [(native.session.robot, ['/World/floor'], 0.)]
    assert len(native.target.pose_writes) == 2 and native.target.still_calls == 1
    assert native.target.pose_writes[1][0][2] == pytest.approx(LOCAL_Z + .1 + .003, abs=1e-7)
    assert p == before
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('missing_stage', ['initial_alignment', 'final_receipt'])
@pytest.mark.parametrize('invalid_hits', [[], [hit(body='/World/unrelated')], [hit(normal=.5)], [hit(normal=-1.)]])
def test_missing_bound_upward_surface_fails_closed_without_aabb_fallback(native, missing_stage, invalid_hits):
    native.hits[0 if missing_stage == 'initial_alignment' else 1] = invalid_hits
    with pytest.raises(ValueError, match='No local upward-facing support surface was found.') as error:
        active_layout.initialize_scene(native.session, parameters())
    diagnostic = json.loads(str(error.value).removeprefix('Local support ray failed: '))
    assert diagnostic['stage'] == missing_stage
    assert diagnostic['object_name'] == 'target' and diagnostic['support_name'] == 'support'
    assert diagnostic['support_prim_path'] == '/World/support'
    assert diagnostic['object_aabb_m'] == [value.tolist() for value in native.target.aabb]
    assert diagnostic['support_aabb_m'] == native.support.aabb.tolist()
    assert (diagnostic['ray_start_m'], diagnostic['ray_end_m']) == native.calls['rays'][-1][:2]
    assert diagnostic['hit_count'] == len(invalid_hits)
    assert diagnostic['hit_summaries_truncated'] is False
    assert native.calls['steps'] == (0 if missing_stage == 'initial_alignment' else 90)
    assert native.calls['robot_checks'] == []
    assert not hasattr(native.session, 'category_interaction_setup')


def test_aabb_alignment_and_final_local_measurement_are_independent(native):
    native.settled_bottom = LOCAL_Z
    p = parameters(alignment='aabb_top')
    before = deepcopy(p)
    result = active_layout.initialize_scene(native.session, p)
    receipt = result['supported_items']['target']
    assert result['accepted'] is receipt['accepted'] is True
    assert receipt['support_alignment_mode'] == 'aabb_top'
    assert receipt['support_surface_mode'] == 'local_raycast'
    assert receipt['support_surface_z_m'] == LOCAL_Z
    assert receipt['support_aabb_top_z_m'] == GLOBAL_TOP_Z
    assert abs(receipt['bottom_gap_m']) < 1e-7
    # Initial placement follows the exact legacy global-AABB alignment; only
    # the measurement after the 90 existing settle steps casts a local ray.
    assert native.target.pose_writes[1][0][2] == pytest.approx(GLOBAL_TOP_Z + .1 + .003, abs=1e-7)
    assert len(native.calls['rays']) == 1 and native.calls['steps'] == 90
    low, high = native.target.aabb
    xy = ((low + high) / 2)[:2].tolist()
    assert native.calls['rays'][0] == ([*xy, GLOBAL_TOP_Z + .10], [*xy, 0.], {'only_closest': False})
    assert native.calls['robot_checks'] == [(native.session.robot, ['/World/floor'], 0.)]
    assert p == before


def test_missing_final_local_surface_still_fails_with_aabb_alignment(native):
    native.settled_bottom = LOCAL_Z
    native.hits = [[]]
    with pytest.raises(ValueError, match='Local support ray failed: ') as error:
        active_layout.initialize_scene(native.session, parameters(alignment='aabb_top'))
    diagnostic = json.loads(str(error.value).removeprefix('Local support ray failed: '))
    assert diagnostic['stage'] == 'final_receipt' and diagnostic['hit_count'] == 0
    assert native.calls['steps'] == 90 and len(native.calls['rays']) == 1
    assert native.calls['robot_checks'] == []
    assert not hasattr(native.session, 'category_interaction_setup')


@pytest.mark.parametrize('distance', [0., -1., 1e-6, float('inf'), float('nan')])
def test_nonpositive_tiny_or_nonfinite_bound_hit_cannot_pass_final_support(native, distance):
    native.settled_bottom = LOCAL_Z
    native.hits = [[hit(LOCAL_Z + .015, distance=distance)]]
    with pytest.raises(ValueError, match='Local support ray failed: ') as error:
        active_layout.initialize_scene(native.session, parameters(alignment='aabb_top'))
    diagnostic = json.loads(str(error.value).removeprefix('Local support ray failed: '))
    assert diagnostic['stage'] == 'final_receipt'
    assert diagnostic['positive_finite_distance_hit_count'] == 0
    assert native.calls['steps'] == 90 and native.calls['robot_checks'] == []
    assert not hasattr(native.session, 'category_interaction_setup')


def test_zero_distance_hit_is_skipped_before_selecting_positive_bound_surface(native):
    native.settled_bottom = LOCAL_Z
    native.hits = [[hit(LOCAL_Z + .015, distance=0.), hit(LOCAL_Z, distance=.132287)]]
    receipt = active_layout.initialize_scene(native.session, parameters(alignment='aabb_top'))['supported_items']['target']
    assert receipt['accepted'] is True and abs(receipt['bottom_gap_m']) < 1e-7
    assert receipt['support_surface_z_m'] == LOCAL_Z
    assert receipt['support_ray_distance_m'] == .132287
    assert receipt['support_hit_position_m'] == [1., 2., LOCAL_Z]
    assert receipt['support_ray_start_m'] == [1., 2., GLOBAL_TOP_Z + .10]
    assert receipt['support_ray_end_m'] == [1., 2., 0.]


def test_failure_hit_summaries_are_bounded_and_contain_only_finite_numbers(native):
    invalid = [hit(body='/World/unrelated_' + 'x' * 300, distance=float('inf')) for _ in range(20)]
    invalid[0]['position'] = [float('nan'), 2., 3.]
    invalid[1]['normal'] = [0., 0., float('nan')]
    native.hits = [invalid]
    with pytest.raises(ValueError, match='Local support ray failed: ') as error:
        active_layout.initialize_scene(native.session, parameters(alignment='aabb_top'))
    diagnostic = json.loads(str(error.value).removeprefix('Local support ray failed: '))
    assert diagnostic['hit_count'] == 20 and diagnostic['hit_summaries_truncated'] is True
    assert len(diagnostic['hit_summaries']) == 8
    assert diagnostic['hit_summaries'][0]['position'] is None
    assert diagnostic['hit_summaries'][1]['normal'] is None
    assert all(row['distance'] is None and len(row['body']) <= 256 for row in diagnostic['hit_summaries'])
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize('mode', [None, 'aabb_top'])
def test_legacy_aabb_mode_keeps_original_receipt_and_steps_without_rays(native, mode):
    result = active_layout.initialize_scene(native.session, parameters(mode))
    low, high = (np.asarray(value, dtype=float) for value in native.target.aabb)
    expected = {'support': 'support', 'touching': True, 'within_footprint': True,
                'speed_m_s': 0., 'bottom_gap_m': float(low[2] - GLOBAL_TOP_Z),
                'aabb_m': [low.tolist(), high.tolist()], 'accepted': True}
    assert result == {'revision': 'fixture_support_v1', 'accepted': True,
        'supported_items': {'target': expected}, 'robot': {'floor_supported': True, 'collision_free': True},
        'validation_scope': 'Native initial support and floor contacts; route and manipulation remain unverified.'}
    assert native.calls['rays'] == [] and native.calls['steps'] == 90
    assert native.target.pose_writes[1][0][2] == pytest.approx(GLOBAL_TOP_Z + .1 + .003, abs=1e-7)


@pytest.mark.parametrize('condition', ['not_touching', 'speed_at_limit', 'outside_footprint', 'gap_exceeds_limit'])
def test_local_mode_preserves_existing_acceptance_thresholds(native, condition):
    if condition == 'not_touching':
        native.target.touching = False
    elif condition == 'speed_at_limit':
        native.target.speed = .02
    elif condition == 'outside_footprint':
        native.target.extent[0] = 3.
    else:
        native.hits[1] = [hit(LOCAL_Z - .02)]
    result = active_layout.initialize_scene(native.session, parameters())
    assert result['supported_items']['target']['accepted'] is result['accepted'] is False
    assert native.calls['steps'] == 90 and len(native.calls['robot_checks']) == 1


def test_unknown_surface_mode_is_rejected(native):
    with pytest.raises(ValueError, match='support_surface_mode'):
        active_layout.initialize_scene(native.session, parameters('invented_mode'))
    assert native.calls['steps'] == 0 and native.calls['rays'] == []


def test_unknown_alignment_mode_is_rejected(native):
    with pytest.raises(ValueError, match='support_alignment_mode'):
        active_layout.initialize_scene(native.session, parameters(alignment='invented_mode'))
    assert native.calls['steps'] == 0 and native.calls['rays'] == []
