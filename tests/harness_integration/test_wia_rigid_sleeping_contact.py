"""Check exact sleeping-pair retention of genuine native separation witnesses.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_rigid_sleeping_contact.py
These CPU fixtures implement read-only OG accessors; they do not run native physics.
"""
from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from wia.authoring.converters import rigid_partial_insertion as module
from harness_integration.test_wia_rigid_native_adapter import native, supported_initial_state


class AuthoredPrim:
    def __init__(self):
        self.kind = 'Mesh'
        self.attributes = {
            'points': np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]], np.float32),
            'faceVertexCounts': np.array([3], np.int32),
            'faceVertexIndices': np.array([0, 1, 2], np.int32),
        }
        self.reads = []

    def GetTypeName(self):
        return self.kind

    def GetAttribute(self, name):
        self.reads.append(name)
        return NS(Get=lambda: self.attributes.get(name))


class DynamicLink:
    def __init__(self, owner, meshes):
        self.owner, self.collision_meshes = owner, meshes
        self.prim_path = owner.prim_path + '/body'
        self.is_meta_link, self.sleeping = False, False
        self.position_offset = np.zeros(3)
        self.scale = np.ones(3)
        self.attributes = {'physics:kinematicEnabled': False, 'physics:rigidBodyEnabled': True}

    def get_position_orientation(self):
        return self.owner.position + self.position_offset, self.owner.quaternion

    def get_attribute(self, key):
        return self.attributes[key]

    @property
    def is_asleep(self):
        return self.sleeping

    def wake(self):
        raise AssertionError('Measurement must never wake native physics.')

    def sleep(self):
        raise AssertionError('Measurement must never change native sleeping state.')


@pytest.fixture
def sleeping_native(native, monkeypatch):
    prims = ModuleType('omnigibson.prims')
    dynamic = ModuleType('omnigibson.prims.rigid_dynamic_prim')
    dynamic.RigidDynamicPrim = DynamicLink
    monkeypatch.setitem(sys.modules, prims.__name__, prims)
    monkeypatch.setitem(sys.modules, dynamic.__name__, dynamic)
    for obj, mesh in ((native.target, native.mesh), (native.container, native.solid)):
        mesh.prim_path = obj.prim_path + '/body/collision'
        mesh.prim = AuthoredPrim()
        mesh.scale = np.ones(3)
        mesh.position_offset = np.zeros(3)
        mesh.get_position_orientation = lambda obj=obj, mesh=mesh: (
            obj.position + mesh.position_offset, obj.quaternion)
        mesh.get_collision_approximation = lambda: 'convexHull'
        obj.links['body'] = DynamicLink(obj, {'mesh': mesh})
    supported_initial_state(native)
    native.evaluator.begin_episode(native.observation)
    native.target.position[:] = [0., 0., .2]
    native.target.touching['container'] = True
    native.target.contacts = []
    return native


def measure(native, time):
    native.clock.current_time = time
    return native.evaluator._measure()['targets']['item']['containers']['container']


def sleep_pair(native):
    for obj in (native.target, native.container):
        obj.links['body'].sleeping = True
    native.separation[:] = []


def seed(native, *, separation=0.):
    native.separation[:] = [separation]
    return measure(native, 1.)


def test_exact_unchanged_sleeping_pair_retains_native_separations_and_raw_current_receipt(sleeping_native):
    n = sleeping_native
    original = seed(n)
    original_separation = deepcopy(original['contact_geometry'])
    assert original['contact_geometry']['separation_source'] == 'native_physx_contact_data'
    assert 'points' in n.mesh.prim.reads and 'faceVertexIndices' in n.solid.prim.reads
    sleep_pair(n)
    retained = measure(n, 2.)
    receipt = retained['contact_geometry']
    assert retained['penetration_free'] is True
    assert receipt['separation_source'] == 'retained_native_contact_for_unchanged_sleeping_pair'
    assert receipt['native_sample_time_s'] == 1. and receipt['current_native_time_s'] == 2.
    assert receipt['original_contact_geometry'] == original_separation
    assert receipt['original_native_separations_m'] == [0.]
    assert receipt['current_contact_geometry'] == {
        'available': False, 'relation': 'unresolved', 'penetrating': None,
        'reason': 'ValueError: Native contact has no pair-specific separation samples.'}
    assert receipt['exact_geometry_unchanged'] is True
    assert all(item['is_asleep'] is False for item in receipt['original_link_sleeping_flags'])
    assert all(item['dynamic'] and item['is_asleep'] for item in receipt['current_link_sleeping_flags'])
    json.dumps(receipt, allow_nan=False)
    # Duplicate native times retain the same original witness, without creating a new sample.
    repeated = measure(n, 2.)['contact_geometry']
    assert repeated['native_sample_time_s'] == 1. and repeated['current_native_time_s'] == 2.


@pytest.mark.parametrize('change', [
    'target_pose', 'container_pose', 'quaternion_sign', 'object_scale', 'link_pose', 'link_scale',
    'mesh_pose', 'mesh_scale', 'mesh_points', 'mesh_faces', 'mesh_enabled', 'mesh_replaced',
    'link_removed', 'link_replaced', 'new_collision_mesh',
])
def test_any_exact_geometry_stamp_change_invalidates_and_discards_witness(sleeping_native, change):
    n = sleeping_native
    seed(n)
    sleep_pair(n)
    link = n.target.links['body']
    if change == 'target_pose':
        n.target.position[0] += 1e-12
    elif change == 'container_pose':
        n.container.position[0] += 1e-12  # Inside the declared protection tolerance; exact stamp still rejects.
    elif change == 'quaternion_sign':
        n.target.quaternion *= -1
    elif change == 'object_scale':
        n.target.scale[0] += 1e-8
    elif change == 'link_pose':
        link.position_offset[0] += 1e-12
    elif change == 'link_scale':
        link.scale[0] += 1e-8
    elif change == 'mesh_pose':
        n.mesh.position_offset[0] += 1e-12
    elif change == 'mesh_scale':
        n.mesh.scale[0] += 1e-8
    elif change == 'mesh_points':
        n.mesh.prim.attributes['points'][1, 0] += np.float32(.01)
    elif change == 'mesh_faces':
        n.mesh.prim.attributes['faceVertexIndices'][:] = [0, 2, 1]
    elif change == 'mesh_enabled':
        n.mesh.collision_enabled = False
    elif change == 'mesh_replaced':
        from copy import copy
        link.collision_meshes['mesh'] = copy(n.mesh)
    elif change == 'link_removed':
        n.target.links.pop('body')
    elif change == 'link_replaced':
        replacement = DynamicLink(n.target, link.collision_meshes)
        replacement.sleeping = True
        n.target.links['body'] = replacement
    else:
        from copy import copy
        extra = copy(n.mesh)
        extra.prim_path += '_extra'
        link.collision_meshes['extra'] = extra
    # Exercise the exact contact path directly when the separate interior sampler cannot read a removed link.
    current = {'available': False, 'relation': 'unresolved', 'penetrating': None,
               'reason': module._EMPTY_PAIR_CONTACT_REASON}
    result = n.evaluator._contact_geometry_with_sleeping_retention(
        'item', 'container', True, current, 2., {'available': True, 'unchanged': True})
    assert result is current and ('item', 'container') not in n.evaluator._contact_witnesses


@pytest.mark.parametrize('kind,attribute,attributes', [
    ('Sphere', 'radius', {'radius': .1}), ('Cube', 'size', {'size': .2}),
    ('Cylinder', 'height', {'radius': .1, 'height': .3, 'axis': 'Z'}),
])
def test_native_primitive_dimensions_are_in_the_exact_stamp(sleeping_native, kind, attribute, attributes):
    n = sleeping_native
    n.mesh.prim.kind, n.mesh.prim.attributes = kind, attributes
    seed(n)
    sleep_pair(n)
    n.mesh.prim.attributes[attribute] += .001
    relation = measure(n, 2.)
    assert relation['penetration_free'] is None
    assert relation['contact_geometry']['available'] is False


def test_native_touching_loss_discards_the_witness_even_if_contact_returns_later(sleeping_native):
    n = sleeping_native
    seed(n)
    sleep_pair(n)
    n.target.touching['container'] = False
    assert measure(n, 2.)['touching'] is False
    assert not n.evaluator._contact_witnesses
    n.target.touching['container'] = True
    assert measure(n, 3.)['penetration_free'] is None


@pytest.mark.parametrize('state', ['awake', 'unknown_sleeping', 'kinematic', 'disabled_rigid_body'])
def test_awake_or_unknown_or_nondynamic_native_body_cannot_retain_contact(sleeping_native, state):
    n = sleeping_native
    seed(n)
    sleep_pair(n)
    link = n.container.links['body']
    if state == 'awake':
        link.sleeping = False
    elif state == 'unknown_sleeping':
        link.sleeping = None
    elif state == 'kinematic':
        link.attributes['physics:kinematicEnabled'] = True
    else:
        link.attributes['physics:rigidBodyEnabled'] = False
    assert measure(n, 2.)['penetration_free'] is None
    assert not n.evaluator._contact_witnesses
    link.sleeping = True
    link.attributes.update({'physics:kinematicEnabled': False, 'physics:rigidBodyEnabled': True})
    assert measure(n, 3.)['penetration_free'] is None


def test_no_prior_native_pair_witness_never_produces_a_retained_separation(sleeping_native):
    n = sleeping_native
    sleep_pair(n)
    relation = measure(n, 1.)
    assert relation['penetration_free'] is None and relation['contact_geometry']['available'] is False
    assert not n.evaluator._contact_witnesses


def test_cached_penetrating_verdict_is_preserved_unchanged(sleeping_native):
    n = sleeping_native
    seed(n, separation=-.002)
    sleep_pair(n)
    relation = measure(n, 2.)
    receipt = relation['contact_geometry']
    assert relation['penetration_free'] is False
    assert receipt['penetrating'] is True and receipt['minimum_separation_m'] == -.002
    assert receipt['penetration_tolerance_m'] == n.config['parameters']['penetration_tolerance_m'] == .001
    assert receipt['original_native_separations_m'] == [-.002]


@pytest.mark.parametrize('failure', ['api_error', 'shape_missing', 'shape_nan', 'shape_unknown', 'pose_error'])
def test_measurement_errors_discard_cache_instead_of_masking_unavailable_data(sleeping_native, monkeypatch, failure):
    n = sleeping_native
    seed(n)
    sleep_pair(n)
    if failure == 'api_error':
        def unavailable(*args, **kwargs):
            raise RuntimeError('Native contact read failed.')
        monkeypatch.setattr(sys.modules['omnigibson.utils.usd_utils'].RigidContactAPI, 'get_contact_data', unavailable)
    elif failure == 'shape_missing':
        n.mesh.prim.attributes['points'] = None
    elif failure == 'shape_nan':
        n.mesh.prim.attributes['points'][0, 0] = np.nan
    elif failure == 'shape_unknown':
        n.mesh.prim.kind = 'UnknownShape'
    else:
        def unavailable_pose():
            raise RuntimeError('Native link pose unavailable.')
        n.container.links['body'].get_position_orientation = unavailable_pose
    relation = measure(n, 2.)
    assert relation['penetration_free'] is None and relation['contact_geometry']['available'] is False
    assert not n.evaluator._contact_witnesses
    if failure == 'api_error':
        assert 'Native contact read failed' in relation['contact_geometry']['reason']


@pytest.mark.parametrize('clock', [.5, None, float('nan'), float('inf')])
def test_regressed_or_unknown_time_invalidates_and_discards_cache(sleeping_native, clock):
    n = sleeping_native
    seed(n)
    sleep_pair(n)
    current = {'available': False, 'relation': 'unresolved', 'penetrating': None,
               'reason': module._EMPTY_PAIR_CONTACT_REASON}
    result = n.evaluator._contact_geometry_with_sleeping_retention(
        'item', 'container', True, current, clock, {'available': True, 'unchanged': True})
    assert result is current and not n.evaluator._contact_witnesses
    assert measure(n, 3.)['penetration_free'] is None


def test_new_episode_clears_witnesses_and_original_native_time(sleeping_native):
    n = sleeping_native
    seed(n)
    sleep_pair(n)
    assert measure(n, 2.)['penetration_free'] is True
    previous_episode = n.evaluator._contact_witness_episode
    n.evaluator.begin_episode(n.observation)
    assert n.evaluator._contact_witness_episode == previous_episode + 1
    assert not n.evaluator._contact_witnesses
    assert measure(n, 3.)['penetration_free'] is None


@pytest.mark.parametrize('protection', [{'available': False, 'unchanged': None},
                                      {'available': True, 'unchanged': False}])
def test_failed_or_unavailable_protected_geometry_blocks_retention(sleeping_native, protection):
    n = sleeping_native
    seed(n)
    sleep_pair(n)
    current = {'available': False, 'reason': module._EMPTY_PAIR_CONTACT_REASON}
    assert n.evaluator._contact_geometry_with_sleeping_retention(
        'item', 'container', True, current, 2., protection) is current
    assert not n.evaluator._contact_witnesses


@pytest.mark.parametrize('invalid_witness', ['positive_aabb', 'empty', 'nonfinite'])
def test_cache_accepts_only_finite_nonempty_genuine_native_separation_records(sleeping_native, invalid_witness):
    n = sleeping_native
    if invalid_witness == 'positive_aabb':
        current = {'available': True, 'penetrating': False, 'minimum_separation_m': .1,
                   'separation_source': 'positive_aabb_clearance'}
    else:
        current = {'available': True, 'penetrating': False, 'separation_source': 'native_physx_contact_data',
                   'separations_m': [] if invalid_witness == 'empty' else [float('nan')]}
    n.evaluator._contact_geometry_with_sleeping_retention(
        'item', 'container', True, current, 1., {'available': True, 'unchanged': True})
    assert not n.evaluator._contact_witnesses
    sleep_pair(n)
    assert measure(n, 2.)['penetration_free'] is None
