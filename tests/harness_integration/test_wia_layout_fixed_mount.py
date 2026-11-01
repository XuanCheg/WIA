"""Validate explicit fixed mounts with CPU native initialization doubles.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m
pytest -q -p no:cacheprovider tests/test_wia_layout_fixed_mount.py
No simulator process, GPU workload, task scorer or robot action is launched.
"""
from copy import deepcopy
import json
import math
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from wia.authoring.converters import active_layout
from harness_integration.test_wia_layout_local_support import (
    LOCAL_Z, Target, Vector, hit, native, parameters)


class FixedTarget(Target):
    @property
    def get_linear_velocity(self):
        raise AttributeError('The fixed native root has no velocity API.')


class PhysicalLink:
    def __init__(self, target):
        self.target = target
        self.kinematic_only = True
        self.collision_meshes = {'collision': NS(collision_enabled=True)}
        self.position_offset = np.zeros(3)
        self.orientation_override = None

    def get_position_orientation(self):
        position, orientation = self.target.get_position_orientation()
        if self.orientation_override is not None:
            orientation = np.asarray(self.orientation_override).view(Vector)
        return (position + self.position_offset).view(Vector), orientation


@pytest.fixture
def mount(native):
    target = native.target
    target.__class__ = FixedTarget
    target.fixed_base, target.visual_only = True, False
    target.root_link = PhysicalLink(target)
    target.links = {'base_link': target.root_link}
    target.touching = False
    return native


def mount_parameters():
    p = parameters()
    p['layout']['supported_items'][0]['support_contract'] = 'fixed_mount'
    return p


def mutate_at_final_step(runtime, callback):
    sim = sys.modules['omnigibson'].sim
    previous = sim.step

    def step():
        previous()
        if runtime.calls['steps'] == 90:
            callback()

    sim.step = step


def test_stable_mount_without_velocity_or_contact_reports_measured_geometry(mount):
    p = mount_parameters()
    before = deepcopy(p)
    result = active_layout.initialize_scene(mount.session, p)
    receipt = result['supported_items']['target']
    assert result['accepted'] is receipt['accepted'] is True
    assert receipt['support_contract'] == 'fixed_mount'
    assert receipt['physical_contact_required'] is receipt['touching'] is False
    assert receipt['speed_m_s'] is None
    assert receipt['within_footprint'] is True and abs(receipt['bottom_gap_m']) < .015
    assert receipt['support_surface_mode'] == 'local_raycast'
    assert receipt['support_surface_z_m'] == LOCAL_Z
    assert receipt['maximum_absolute_bottom_gap_m'] == .015
    assert receipt['footprint_tolerance_m'] == .003
    stability = receipt['fixed_mount_stability']
    assert stability['pose_stable'] is True and stability['settle_steps'] == 90
    assert stability['maximum_position_drift_m'] == .002
    assert stability['maximum_orientation_drift_rad'] == .01
    for field in ('object_pose', 'root_link_pose'):
        assert stability[field + '_position_drift_m'] == 0.
        assert stability[field + '_orientation_drift_rad'] == 0.
        assert stability['before'][field] == stability['after'][field]
    geometry = receipt['fixed_mount_geometry']
    assert geometry['fixed_base'] is True and geometry['visual_only'] is False
    link, = geometry['collision_links']
    assert link['kinematic_evidence'] == 'native_kinematic_only_boolean'
    assert link['all_collision_meshes_enabled'] is True
    assert mount.calls['steps'] == 90 and len(mount.calls['rays']) == 2
    assert len(mount.calls['robot_checks']) == 0
    assert len(mount.target.pose_writes) == 2 and mount.target.still_calls == 1
    assert p == before
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('change', ['object_position', 'root_position', 'object_orientation', 'root_orientation'])
def test_either_actual_object_or_root_pose_change_rejects_mount(mount, change):
    root = mount.target.root_link

    def mutate():
        if change == 'object_position':
            mount.target.position[0] += .003
        elif change == 'root_position':
            root.position_offset[0] = .003
        elif change == 'object_orientation':
            mount.target.quaternion = np.array([0., 0., math.sin(.02/2), math.cos(.02/2)]).view(Vector)
        else:
            root.orientation_override = [0., 0., math.sin(.02/2), math.cos(.02/2)]

    mutate_at_final_step(mount, mutate)
    result = active_layout.initialize_scene(mount.session, mount_parameters())
    receipt = result['supported_items']['target']
    assert receipt['accepted'] is result['accepted'] is False
    assert receipt['fixed_mount_stability']['pose_stable'] is False
    if change.startswith('root_'):
        assert receipt['fixed_mount_stability']['object_pose_position_drift_m'] == 0.
        assert receipt['fixed_mount_stability']['object_pose_orientation_drift_rad'] == 0.
    assert mount.calls['steps'] == 90


def test_small_motion_and_quaternion_sign_equivalence_stay_within_thresholds(mount):
    def mutate():
        mount.target.position[0] += .0019
        mount.target.root_link.position_offset[1] = .0001
        mount.target.quaternion *= -1

    mutate_at_final_step(mount, mutate)
    result = active_layout.initialize_scene(mount.session, mount_parameters())
    assert result['accepted'] is True
    stability = result['supported_items']['target']['fixed_mount_stability']
    assert stability['object_pose_position_drift_m'] == pytest.approx(.0019, abs=1e-7)
    assert stability['root_link_pose_position_drift_m'] < .002
    assert stability['root_link_pose_orientation_drift_rad'] == 0.


@pytest.mark.parametrize('invalid', ['dynamic_object', 'visual_only', 'dynamic_link', 'second_dynamic_link',
    'missing_kinematic_boolean', 'no_collision_geometry', 'disabled_collision', 'missing_collision_state',
    'missing_root', 'foreign_root'])
def test_fixed_mount_requires_declared_and_native_physical_evidence(mount, monkeypatch, invalid):
    target, root = mount.target, mount.target.root_link
    if invalid == 'dynamic_object': target.fixed_base = False
    elif invalid == 'visual_only': target.visual_only = True
    elif invalid == 'dynamic_link': root.kinematic_only = False
    elif invalid == 'second_dynamic_link':
        other = PhysicalLink(target)
        other.kinematic_only = False
        target.links['other_link'] = other
    elif invalid == 'missing_kinematic_boolean':
        del root.kinematic_only
        module = ModuleType('omnigibson.prims.rigid_kinematic_prim')
        module.RigidKinematicPrim = type('RigidKinematicPrim', (), {})
        monkeypatch.setitem(sys.modules, module.__name__, module)
    elif invalid == 'no_collision_geometry': root.collision_meshes = {}
    elif invalid == 'disabled_collision': root.collision_meshes['collision'].collision_enabled = False
    elif invalid == 'missing_collision_state': del root.collision_meshes['collision'].collision_enabled
    elif invalid == 'missing_root': del target.root_link
    else: target.root_link = PhysicalLink(target)
    with pytest.raises(ValueError, match='fixed_mount'):
        active_layout.initialize_scene(mount.session, mount_parameters())
    assert mount.calls['steps'] == 0 and mount.target.pose_writes == []


def test_exact_installed_kinematic_type_fallback_and_fresh_root_pose_reads(mount, monkeypatch):
    root = mount.target.root_link
    del root.kinematic_only
    module = ModuleType('omnigibson.prims.rigid_kinematic_prim')
    module.RigidKinematicPrim = type(root)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cache_clears = []
    root.clear_kinematic_only_cache = lambda: cache_clears.append(len(cache_clears))
    result = active_layout.initialize_scene(mount.session, mount_parameters())
    assert result['accepted'] is True and len(cache_clears) == 2
    link, = result['supported_items']['target']['fixed_mount_geometry']['collision_links']
    assert link['kinematic_evidence'] == 'exact_native_RigidKinematicPrim_type'


@pytest.mark.parametrize('failure', ['missing_local_mode', 'missing_local_ray', 'footprint', 'surface_gap'])
def test_mount_still_requires_native_local_surface_footprint_and_gap(mount, failure):
    p = mount_parameters()
    if failure == 'missing_local_mode':
        p['layout']['supported_items'][0]['support_surface_mode'] = 'aabb_top'
    elif failure == 'missing_local_ray': mount.hits[1] = []
    elif failure == 'footprint': mount.target.extent[0] = 3.
    else: mount.hits[1] = [hit(LOCAL_Z - .02)]
    if failure.startswith('missing_'):
        with pytest.raises(ValueError):
            active_layout.initialize_scene(mount.session, p)
    else:
        result = active_layout.initialize_scene(mount.session, p)
        assert result['accepted'] is result['supported_items']['target']['accepted'] is False


def test_mount_preserves_independent_authored_aabb_alignment(mount):
    p = mount_parameters()
    p['layout']['supported_items'][0]['support_alignment_mode'] = 'aabb_top'
    # The fixture's global top differs from its local surface. Leave the mount
    # stable after that unchanged AABB alignment and measure the actual local gap.
    mount.hits = [[hit()]]
    result = active_layout.initialize_scene(mount.session, p)
    receipt = result['supported_items']['target']
    assert receipt['support_alignment_mode'] == 'aabb_top'
    assert receipt['support_surface_mode'] == 'local_raycast'
    assert receipt['fixed_mount_stability']['pose_stable'] is True
    assert result['accepted'] is False  # Its measured local gap still exceeds .015 m.
    assert len(mount.calls['rays']) == 1 and mount.calls['steps'] == 90


def test_fixed_object_without_opt_in_still_fails_for_missing_dynamic_velocity(mount):
    mount.target.touching = True
    with pytest.raises(AttributeError, match='no velocity API'):
        active_layout.initialize_scene(mount.session, parameters())
    assert mount.calls['steps'] == 90
    assert not hasattr(mount.session, 'category_interaction_setup')


def test_dynamic_missing_velocity_fails_instead_of_receiving_zero_speed(native):
    native.target.__class__ = FixedTarget
    with pytest.raises(AttributeError, match='no velocity API'):
        active_layout.initialize_scene(native.session, parameters())


def test_explicit_dynamic_contract_keeps_default_receipt_unchanged(native):
    default = active_layout.initialize_scene(native.session, parameters(mode=None))
    p = parameters(mode=None)
    p['layout']['supported_items'][0]['support_contract'] = 'dynamic_support'
    explicit = active_layout.initialize_scene(native.session, p)
    assert explicit == default
    assert 'support_contract' not in explicit['supported_items']['target']


def test_unknown_contract_is_rejected_before_pose_or_physics_changes(native):
    p = parameters()
    p['layout']['supported_items'][0]['support_contract'] = 'invented_static'
    with pytest.raises(ValueError, match='support_contract'):
        active_layout.initialize_scene(native.session, p)
    assert native.calls['steps'] == 0 and native.target.pose_writes == []
