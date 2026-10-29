"""Check explicit pre-alignment physics/Fabric pose synchronization on CPU.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m
pytest -q -p no:cacheprovider tests/test_wia_layout_pose_sync.py
These doubles reproduce a native AABB lag without starting a simulator or GPU.
"""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.authoring.converters import active_layout
from harness_integration.test_wia_layout_local_support import LOCAL_Z, Target, hit, native, parameters
from harness_integration.test_wia_layout_fixed_mount import mount, mount_parameters
from harness_integration.test_wia_native_furniture_poses import Furniture, spec as furniture_spec


class LaggedTarget(Target):
    """Pose writes become visible to native collision AABBs on the next step."""
    @property
    def aabb(self):
        return self.geometry_position - self.extent / 2, self.geometry_position + self.extent / 2

    def set_position_orientation(self, position, orientation):
        super().set_position_orientation(position, orientation)
        self.pose_write_steps.append(self.current_step())


@pytest.fixture
def lagged(native):
    target = native.target
    target.__class__ = LaggedTarget
    target.position[:2] = [.662, -2.163]
    target.geometry_position = target.position.copy()
    target.pose_write_steps = []
    target.current_step = lambda: native.calls['steps']
    native.support.aabb = np.array([[-4.218, 1.751, .10], [-2.595, 2.488, .4632726]])
    native.targets = [target]
    previous = sys.modules['omnigibson'].sim.step

    def step():
        previous()
        for obj in native.targets:
            obj.geometry_position = obj.position.copy()

    def raytest(start, end, **kwargs):
        native.calls['rays'].append((start, end, kwargs))
        low, high = native.support.aabb
        if not all(low[axis] <= start[axis] <= high[axis] for axis in (0, 1)):
            return []
        return [{**hit(), 'position': [*start[:2], LOCAL_Z]}]

    sys.modules['omnigibson'].sim.step = step
    sys.modules['omnigibson.utils.sampling_utils'].raytest = raytest
    return native


def sync_parameters(enabled=True):
    p = parameters()
    p['layout']['objects'][0]['position'] = [-4., 2.18, 2.]
    p['layout']['objects'][0]['orientation'] = [0., 0., 1., 0.]
    p['layout']['synchronize_authored_poses_before_alignment'] = enabled
    return p


def test_opt_in_flushes_stale_geometry_once_before_local_alignment(lagged):
    p = sync_parameters()
    before = deepcopy(p)
    result = active_layout.initialize_scene(lagged.session, p)
    assert result['accepted'] is True
    assert lagged.calls['steps'] == 91 and len(lagged.calls['rays']) == 2
    assert lagged.calls['rays'][0][0][:2] == pytest.approx([-4., 2.18])
    assert lagged.target.pose_write_steps == [0, 1]
    assert len(lagged.target.pose_writes) == 2 and lagged.target.still_calls == 2
    assert lagged.target.pose_writes[0][0] == pytest.approx([-4., 2.18, 2.])
    assert lagged.target.pose_writes[1][0][2] == pytest.approx(LOCAL_Z + .1 + .003)
    sync = result['authored_pose_synchronization']
    assert sync['additional_initialization_steps'] == 1 and sync['evidence_scope'] == 'initialization_only'
    entry, = sync['objects'].values()
    assert entry['authored_pose'] == {'position': [-4., 2.18, 2.], 'orientation': [0., 0., 1., 0.]}
    assert entry['measured_pre_alignment_pose']['position'] == pytest.approx([-4., 2.18, 2.])
    assert entry['measured_pre_alignment_pose']['orientation'] == [0., 0., 1., 0.]
    assert entry['measured_pre_alignment_aabb_m'][0] == pytest.approx([-4.1, 2.08, 1.9])
    assert result['supported_items']['target']['bottom_gap_m'] == pytest.approx(.003, abs=1e-7)
    assert p == before
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('enabled', [False, 'omitted'])
def test_default_retains_original_stale_ray_path_without_extra_steps(lagged, enabled):
    p = sync_parameters(False)
    if enabled == 'omitted':
        del p['layout']['synchronize_authored_poses_before_alignment']
    with pytest.raises(ValueError, match='Local support ray failed'):
        active_layout.initialize_scene(lagged.session, p)
    assert lagged.calls['steps'] == 0 and lagged.target.pose_write_steps == [0]
    assert lagged.calls['rays'][0][0][:2] == pytest.approx([.662, -2.163])
    assert lagged.target.still_calls == 0


def test_all_explicit_supported_poses_are_restored_in_one_batch_with_no_other_pose_writes(lagged):
    second = LaggedTarget()
    second.name, second.prim_path = 'second', '/World/second'
    second.geometry_position = second.position.copy()
    second.pose_write_steps, second.current_step = [], lambda: lagged.calls['steps']
    states = sys.modules['omnigibson.object_states']
    second.states = {states.Touching: NS(get_value=lambda support: support is lagged.support)}
    lagged.targets.append(second)
    untouched = Target()
    untouched.name = 'untouched'
    registry = lagged.session.env.scene.object_registry
    lagged.session.env.scene.object_registry = lambda kind, name: (
        {'second': second, 'untouched': untouched}.get(name) or registry(kind, name))
    p = sync_parameters()
    p['layout']['objects'].extend([
        {'name': 'second', 'position': [-3., 2.18, 2.]},
        {'name': 'untouched', 'position': [50., 50., 50.]}])
    p['layout']['supported_items'].append({'name': 'second', 'support': 'support', 'support_surface_mode': 'local_raycast'})
    result = active_layout.initialize_scene(lagged.session, p)
    assert result['accepted'] is True and lagged.calls['steps'] == 91
    assert lagged.target.pose_write_steps == second.pose_write_steps == [0, 1]
    assert set(result['authored_pose_synchronization']['objects']) == {'target', 'second'}
    assert second.quaternion.tolist() == [0., 0., 0., 1.]  # The existing authored default is retained.
    assert len(lagged.calls['rays']) == 4
    assert untouched.pose_writes == [] and untouched.still_calls == 0


def test_native_furniture_reset_precedes_supported_pose_synchronization(lagged):
    furniture = Furniture()
    registry = lagged.session.env.scene.object_registry
    lagged.session.env.scene.object_registry = lambda kind, name: furniture if name == furniture.name else registry(kind, name)
    p = sync_parameters()
    p['layout']['native_object_poses'] = [furniture_spec()]
    first_step = []
    previous = sys.modules['omnigibson'].sim.step

    def step():
        if lagged.calls['steps'] == 0:
            first_step.append((deepcopy(furniture.position), lagged.target.position.copy()))
        previous()

    sys.modules['omnigibson'].sim.step = step
    result = active_layout.initialize_scene(lagged.session, p)
    furniture_position, target_position = first_step[0]
    assert furniture_position == [3., 4., .5]
    assert target_position == pytest.approx([-4., 2.18, 2.])
    assert len(furniture.writes) == 1 and furniture.still_calls == 1
    assert result['native_object_poses'][0]['final_pose']['position'] == [3., 4., .5]
    assert lagged.calls['steps'] == 91


@pytest.mark.parametrize('invalid', [None, 0, 1, 'true', []])
def test_sync_option_requires_an_exact_boolean_before_pose_or_physics_changes(native, invalid):
    p = parameters()
    p['layout']['synchronize_authored_poses_before_alignment'] = invalid
    with pytest.raises(ValueError, match='must be a boolean'):
        active_layout.initialize_scene(native.session, p)
    assert native.calls['steps'] == 0 and native.target.pose_writes == []


@pytest.mark.parametrize('invalid', ['missing_authored', 'duplicate_authored', 'missing_native', 'duplicate_supported'])
def test_sync_rejects_unsupplied_or_ambiguous_objects_before_batch_writes(native, invalid):
    p = parameters()
    p['layout']['synchronize_authored_poses_before_alignment'] = True
    if invalid == 'missing_authored': p['layout']['objects'] = []
    elif invalid == 'duplicate_authored': p['layout']['objects'].append(deepcopy(p['layout']['objects'][0]))
    elif invalid == 'missing_native':
        p['layout']['objects'][0]['name'] = p['layout']['supported_items'][0]['name'] = 'absent'
    else: p['layout']['supported_items'].append(deepcopy(p['layout']['supported_items'][0]))
    with pytest.raises(ValueError, match='explicitly supported native objects'):
        active_layout.initialize_scene(native.session, p)
    assert native.calls['steps'] == 0 and native.target.pose_writes == []


def test_fixed_mount_remains_a_separate_90_step_post_alignment_stability_contract(mount):
    p = mount_parameters()
    p['layout']['synchronize_authored_poses_before_alignment'] = True
    result = active_layout.initialize_scene(mount.session, p)
    assert result['accepted'] is True and mount.calls['steps'] == 91
    receipt = result['supported_items']['target']
    assert receipt['speed_m_s'] is None and receipt['physical_contact_required'] is False
    assert receipt['fixed_mount_stability']['pose_stable'] is True
    assert receipt['fixed_mount_stability']['settle_steps'] == 90
    aligned = receipt['fixed_mount_stability']['before']['object_pose']['position'][2]
    restored = result['authored_pose_synchronization']['objects']['target']['measured_pre_alignment_pose']['position'][2]
    assert aligned == pytest.approx(LOCAL_Z + .1 + .003) and restored == 2.


def test_explicit_false_keeps_successful_default_receipt_and_90_steps(native):
    p = parameters(mode=None)
    default = active_layout.initialize_scene(native.session, p)
    native.calls['steps'] = 0
    p['layout']['synchronize_authored_poses_before_alignment'] = False
    explicit = active_layout.initialize_scene(native.session, p)
    assert explicit == default and native.calls['steps'] == 90
    assert 'authored_pose_synchronization' not in explicit
