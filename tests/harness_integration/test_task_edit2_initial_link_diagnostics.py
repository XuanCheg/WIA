"""CPU checks of bounded initial native object/link diagnostics.

Run: python -B -m pytest -q tests/test_task_edit2_initial_link_diagnostics.py
Native-shaped fixtures distinguish live OG getters from authored USD metadata;
no simulator, GPU, scene setters or physics operations are used.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

import wia.runtime.sim.esi_initial_link_diagnostics as diagnostics
from wia.tasks.loader import LoadedScene, LoadedTask


class Attribute:
    def __init__(self, name, value):
        self.name, self.value = name, value

    def GetName(self):
        return self.name

    def Get(self):
        return self.value

    def HasAuthoredValueOpinion(self):
        return self.value is not None

    def Set(self, *args):
        raise AssertionError('USD attribute writes forbidden.')


class Prim:
    def __init__(self, path, kind='Xform', **attributes):
        self.path, self.kind, self.children, self.attributes = path, kind, [], attributes
        self.valid = True

    def GetPath(self):
        return self.path

    def GetName(self):
        return self.path.rsplit('/', 1)[-1]

    def GetTypeName(self):
        return self.kind

    def IsValid(self):
        return self.valid

    def GetChildren(self):
        return self.children

    def GetAttributes(self):
        return [Attribute(name, value) for name, value in self.attributes.items()]

    def GetAttribute(self, name):
        return Attribute(name, self.attributes.get(name))

    def GetAppliedSchemas(self):
        return ['PhysicsRigidBodyAPI', 'PhysicsMassAPI'] if self.kind == 'Xform' else ['PhysicsCollisionAPI']

    def GetCustomData(self):
        return {'metalink_role': 'togglebutton'} if 'meta' in self.path else {}

    def GetRelationship(self, name):
        return NS(GetTargets=lambda: ['/World/laptop/base'] if name == 'physics:body0' else [])

    def ApplyAPI(self, *args):
        raise AssertionError('Schema mutation forbidden.')


class Node:
    def __init__(self, path):
        self.prim_path = path
        self.prim = Prim(path, **{'xformOp:translate': [9., 8., 7.], 'xformOp:scale': [1., 1., 1.],
            'physics:rigidBodyEnabled': True, 'physics:kinematicEnabled': False,
            'physics:mass': .001, 'physics:diagonalInertia': [1e-6, 2e-6, 3e-6],
            'physics:centerOfMass': [0., 0., 0.], 'physics:collisionEnabled': True})
        self.scale = np.ones(3)
        self.aabb = (np.zeros(3), np.ones(3))
        self.position = np.array([1., 2., 3.])
        self.quaternion = np.array([0., 0., 0., 1.])
        self.reads = []

    def get_position_orientation(self, *, frame):
        self.reads.append(frame)
        return self.position.copy() if frame == 'world' else np.array([.1, .2, .3]), self.quaternion.copy()

    def set_position_orientation(self, *args, **kwargs):
        raise AssertionError('Pose writes forbidden.')

    def get_obs(self):
        raise AssertionError('Observation reads forbidden.')


class RigidKinematicPrim(Node):
    @property
    def mass(self):
        raise AssertionError('Do not promote installed kinematic mass compatibility zero.')

    @property
    def center_of_mass(self):
        raise AssertionError('Do not promote installed kinematic COM compatibility zero.')


class RigidDynamicPrim(Node):
    def get_position_orientation(self, *, frame):
        assert frame == 'world', 'Installed rigid dynamic getter rejects parent frame.'
        return super().get_position_orientation(frame=frame)


class Session(NS):
    @property
    def observation_source(self):
        raise AssertionError('Observation rig access forbidden.')


def loaded(cls, name):
    return cls(definition=NS(id=name, version='v1'), path=Path('/tmp/'+name+'.yaml'),
        source_path=Path('/tmp/source.json'), file_sha256=name+'-sha', source_sha256='source-sha',
        variant='revised', seed=23, scene_name='building', room_name='bathroom_1')


@pytest.fixture
def native(monkeypatch):
    lazy = ModuleType('omnigibson.lazy')
    class Xformable:
        def __init__(self, prim):
            self.prim = prim

        def ComputeLocalToWorldTransform(self, time):
            assert time == 'default'
            matrix = np.eye(4)
            matrix[3, :3] = [9., 8., 7.]
            return matrix

        def GetLocalTransformation(self, time):
            assert time == 'default'
            return np.eye(4)
    lazy.pxr = NS(UsdGeom=NS(Xformable=Xformable), Usd=NS(TimeCode=NS(Default=lambda: 'default')))
    lazy.isaacsim = NS(core=NS(utils=NS(xforms=NS(get_local_pose=lambda path:
        (np.array([.4, .5, .6]), np.array([np.sqrt(.5), 0., 0., np.sqrt(.5)]))))))
    og = ModuleType('omnigibson')
    og.__path__ = []
    monkeypatch.setitem(sys.modules, 'omnigibson', og)
    monkeypatch.setitem(sys.modules, 'omnigibson.lazy', lazy)
    laptop = Node('/World/laptop')
    laptop.name, laptop.category, laptop.model = 'laptop', 'laptop', 'nvulcs'
    laptop.fixed_base, laptop.visual_only = False, False
    child = Node('/World/laptop/meta__base_link_togglebutton_0_0_link')
    child.mass, child.center_of_mass = 1e-6, np.array([.001, .002, .003])
    child._rigid_prim_view = NS(get_masses=lambda: np.array([1e-6]),
        get_coms=lambda: (np.array([[.001, .002, .003]]), np.array([[1., 0., 0., 0.]])),
        get_inertias=lambda: np.eye(3).reshape(1, 9)*1e-6)
    collision = Prim(child.prim_path+'/collision', kind='Mesh', **{'physics:collisionEnabled': True})
    child.prim.children.append(collision)
    laptop.prim.children.append(child.prim)
    laptop.links = {'base': laptop, 'toggle': child}
    laptop.joints = {'toggle_joint': NS(prim_path='/World/laptop/joint',
        get_state=lambda normalized: (np.array([.1]), np.array([0.]), np.array([.2])))}
    unrelated = Node('/World/unrelated')
    objects = {'laptop': laptop, 'unrelated': unrelated}
    registry_reads = []
    def registry(kind, name):
        registry_reads.append((kind, name))
        return objects.get(name)
    def forbidden(*args, **kwargs):
        raise AssertionError('Physics, render or public actions forbidden.')
    session = Session(env=NS(scene=NS(object_registry=registry), step=forbidden),
        native_reset_id='native_reset', episode_id='native_episode', loaded_task=loaded(LoadedTask, 'task'),
        scene_adapter=NS(task=loaded(LoadedScene, 'scene')), pipeline=NS(og=NS(sim=NS(
            current_time=1.25, render=forbidden, step=forbidden))))
    return NS(session=session, laptop=laptop, child=child, unrelated=unrelated,
              registry_reads=registry_reads, objects=objects, lazy=lazy)


def measure(native, names=('laptop',)):
    return diagnostics.measure_initial_object_links(native.session, names, 'before_source_settle')


def test_finite_root_bad_child_stays_explicit_and_usd_values_are_not_live_replacements(native):
    native.child.position[0] = np.nan
    report = measure(native)
    obj = report['objects'][0]
    assert obj['status'] == 'resolved' and obj['world_pose']['available']
    child = next(link for link in obj['links'] if link['registry_name'] == 'toggle')
    assert child['world_pose']['status'] == 'nonfinite'
    assert child['world_pose']['error_type'] == 'NonFiniteMeasurement'
    assert child['local_to_parent_pose']['available']
    usd = next(prim for prim in obj['usd_descendants']['prims'] if prim['path'] == native.child.prim_path)
    assert usd['attributes_default']['xformOp:translate']['value'] == [9., 8., 7.]
    assert usd['usd_world_transform_default']['available']
    assert usd['usd_world_transform_default']['value']['time'] == 'Usd.TimeCode.Default'
    assert usd['metalink_name_marker']
    assert usd['custom_data']['value']['metalink_role'] == 'togglebutton'
    assert report['native_time_s']['value'] == 1.25
    assert report['provenance']['task']['file_sha256']['value'] == 'task-sha'
    assert report['provenance']['scene']['file_sha256']['value'] == 'scene-sha'
    assert report['provenance']['reset_id']['value'] == 'native_reset'
    assert report['safety_qualification'] is None
    json.dumps(report, allow_nan=False)


def test_supported_existing_runtime_mass_com_inertia_and_joint_state_are_read(native):
    obj = measure(native)['objects'][0]
    link = obj['links'][1]
    assert link['mass']['value'] == 1e-6
    assert link['runtime_rigid_view']['masses']['value'] == [1e-6]
    assert link['runtime_rigid_view']['centres_of_mass']['available']
    assert link['runtime_rigid_view']['inertias']['available']
    assert obj['joints'][0]['state']['value'] == [[.1], [0.], [.2]]
    usd = obj['usd_descendants']['prims'][1]
    assert usd['attributes_default']['physics:diagonalInertia']['value'] == [1e-6, 2e-6, 3e-6]
    assert usd['joint_relationships']['physics:body0']['value'] == ['/World/laptop/base']


def test_missing_mass_api_never_becomes_zero(native):
    del native.child.mass
    del native.child._rigid_prim_view
    link = measure(native)['objects'][0]['links'][1]
    assert not link['mass']['available'] and link['mass']['error_type'] == 'AttributeError'
    assert all(not field['available'] for field in link['runtime_rigid_view'].values())
    assert 'value' not in link['mass']


def test_kinematic_compatibility_zeros_are_not_measured_physical_values(native):
    node = RigidKinematicPrim(native.child.prim_path)
    native.laptop.links['toggle'] = node
    row = measure(native)['objects'][0]['links'][1]
    assert row['kinematic_compatibility_stub_detected']
    assert not row['mass']['available'] and not row['center_of_mass']['available']
    assert 'compatibility zero' in row['mass']['error']
    assert not row['world_pose']['available']
    assert row['local_to_parent_pose']['value']['position_m'] == [.4, .5, .6]


def test_existing_kinematic_pose_cache_is_explicitly_non_live_and_never_refreshed(native):
    node = RigidKinematicPrim(native.child.prim_path)
    node._kinematic_pose_cache = {'world': (np.array([7., 8., 9.]), np.array([0., 0., 0., 1.]))}
    before = deepcopy(node._kinematic_pose_cache)
    native.laptop.links['toggle'] = node
    row = measure(native)['objects'][0]['links'][1]
    assert row['world_pose']['value']['position_m'] == [7., 8., 9.]
    assert 'NOT live PhysX' in row['world_pose']['source']
    assert node.reads == []
    assert all(np.array_equal(first, last) for first, last in zip(before['world'], node._kinematic_pose_cache['world']))


def test_rigid_dynamic_local_pose_uses_installed_supported_local_reader(native):
    node = RigidDynamicPrim(native.child.prim_path)
    native.laptop.links['toggle'] = node
    row = measure(native)['objects'][0]['links'][1]
    assert row['world_pose']['value']['position_m'] == [1., 2., 3.]
    assert row['local_to_parent_pose']['value']['position_m'] == [.4, .5, .6]
    assert row['local_to_parent_pose']['value']['quaternion_xyzw'] == pytest.approx([0., 0., np.sqrt(.5), np.sqrt(.5)])
    assert node.reads == ['world']


@pytest.mark.parametrize('stale', ['missing', 'name', 'invalid_prim', 'path'])
def test_missing_or_stale_exact_registry_object_never_substitutes(native, stale):
    if stale == 'missing':
        native.objects.pop('laptop')
    elif stale == 'name':
        native.laptop.name = 'other'
    elif stale == 'invalid_prim':
        native.laptop.prim.valid = False
    else:
        native.laptop.prim_path = '/World/other'
    row = measure(native)['objects'][0]
    assert row['status'] == 'unavailable' and row['error_type']
    assert row['links'] == [] and native.unrelated.reads == []
    assert native.registry_reads == [('name', 'laptop')]


def test_explicit_scope_deduplicates_names_and_descendant_cap_is_recorded(native):
    native.laptop.prim.children = [Prim('/World/laptop/node_'+str(i)) for i in range(300)]
    report = measure(native, names=['laptop', 'laptop'])
    assert report['requested_names'] == ['laptop', 'laptop'] and len(report['objects']) == 1
    assert native.registry_reads == [('name', 'laptop')]
    usd = report['objects'][0]['usd_descendants']
    assert usd['truncated'] and usd['reported_count'] == 256 and usd['total_count'] is None
    assert all(row['path'].startswith('/World/laptop') for row in usd['prims'])
    assert native.unrelated.reads == []


def test_registry_link_cap_and_foreign_descendant_scope_are_explicit(native):
    native.laptop.links = {str(i): native.child for i in range(300)}
    row = measure(native)['objects'][0]
    assert row['links_inventory']['actual_count'] == 300
    assert row['links_inventory']['truncated'] and len(row['links']) == 256
    native.laptop.prim.children = [Prim('/World/unrelated')]
    row = measure(native)['objects'][0]
    assert not row['usd_descendants']['available']
    assert 'escaped' in row['usd_descendants']['error']


@pytest.mark.parametrize('field', ['scale', 'aabb', 'mass', 'com', 'inertia', 'usd', 'time'])
def test_nonfinite_fields_are_typed_and_json_safe(native, field):
    if field == 'scale':
        native.child.scale[0] = np.inf
        key = 'scale'
    elif field == 'aabb':
        native.child.aabb[1][0] = np.nan
        key = 'aabb_m'
    elif field == 'mass':
        native.child.mass = np.inf
        key = 'mass'
    elif field == 'com':
        native.child.center_of_mass[0] = np.nan
        key = 'center_of_mass'
    elif field == 'inertia':
        native.child._rigid_prim_view.get_inertias = lambda: np.array([np.inf])
    elif field == 'usd':
        native.child.prim.attributes['physics:mass'] = np.nan
    else:
        native.session.pipeline.og.sim.current_time = np.inf
    report = measure(native)
    link = report['objects'][0]['links'][1]
    row = (link[key] if field in ('scale', 'aabb', 'mass', 'com') else
        link['runtime_rigid_view']['inertias'] if field == 'inertia' else
        report['objects'][0]['usd_descendants']['prims'][1]['attributes_default']['physics:mass'] if field == 'usd' else
        report['native_time_s'])
    assert not row['available'] and row['status'] == 'nonfinite'
    json.dumps(report, allow_nan=False)


def test_usd_failure_never_hides_actual_link_inventory(native):
    native.child.prim.GetChildren = lambda: (_ for _ in ()).throw(RuntimeError('USD graph unavailable'))
    row = measure(native)['objects'][0]
    assert row['links_inventory']['available'] and len(row['links']) == 2
    assert row['links'][1]['world_pose']['available']
    assert not row['usd_descendants']['available']


def test_no_physics_render_schema_attribute_pose_or_observation_mutation(native):
    before_position, before_quaternion = native.child.position.copy(), native.child.quaternion.copy()
    before_attrs = deepcopy(native.child.prim.attributes)
    session_fields = set(vars(native.session))
    report = measure(native)
    assert report['objects'][0]['status'] == 'resolved'
    assert np.array_equal(before_position, native.child.position)
    assert np.array_equal(before_quaternion, native.child.quaternion)
    assert native.child.prim.attributes == before_attrs
    assert set(vars(native.session)) == session_fields
    assert native.unrelated.reads == []


def test_no_paths_or_patterns_are_expanded(native):
    for name in ('/World/laptop', 'laptop*'):
        row = measure(native, [name])['objects'][0]
        assert row['status'] == 'unavailable'
    assert ('name', 'laptop*') in native.registry_reads
    assert native.laptop.reads == []
