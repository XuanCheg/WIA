"""CPU camera-mount diagnostics tests.

Run: ``python -B -m pytest -q tests/test_task_edit2_initial_camera_diagnostics.py``.
These native-shaped fixtures prove read contracts, not native camera readiness.
"""
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from wia.runtime.sim.esi_initial_camera_diagnostics import measure_initial_camera_mounts
from wia.tasks.loader import LoadedScene, LoadedTask


class Prim:
    def __init__(self, path, kind='Xform', parent=None):
        self.path, self.kind, self.parent = path, kind, parent

    def GetTypeName(self):
        return self.kind

    def GetPath(self):
        return self.path

    def GetParent(self):
        return self.parent

    def IsValid(self):
        return True

    def GetAttribute(self, name):
        return SimpleNamespace(Get=lambda: {'verticalAperture': 17., 'projection': 'perspective'}.get(name))


class Node:
    def __init__(self, path, *, parent=None, kind='Xform'):
        self.name = path.rsplit('/', 1)[-1]
        self.prim_path = path
        self.prim = Prim(path, kind, parent)
        self.reads = []

    def get_position_orientation(self, *, frame):
        self.reads.append(frame)
        return np.array([1., 2., 3.]) if frame == 'world' else np.array([.1, .2, .3]), np.array([0., 0., 0., 1.])

    def get_obs(self):
        raise AssertionError('Observation capture forbidden.')

    @property
    def intrinsic_matrix(self):
        raise AssertionError('Intrinsics access forbidden.')

    def set_position_orientation(self, *args, **kwargs):
        raise AssertionError('Pose write forbidden.')


@pytest.fixture
def native_session(monkeypatch):
    lazy = ModuleType('omnigibson.lazy')
    class Xformable:
        def __init__(self, prim):
            self.prim = prim

        def ComputeLocalToWorldTransform(self, time):
            matrix = np.eye(4)
            matrix[3, :3] = [4, 5, 6]
            return matrix

        def GetLocalTransformation(self, time):
            return np.eye(4)
    lazy.pxr = SimpleNamespace(UsdGeom=SimpleNamespace(Xformable=Xformable),
        Usd=SimpleNamespace(TimeCode=SimpleNamespace(Default=lambda: 'default')))
    og = ModuleType('omnigibson')
    og.__path__ = []
    monkeypatch.setitem(sys.modules, 'omnigibson', og)
    monkeypatch.setitem(sys.modules, 'omnigibson.lazy', lazy)
    robot = Node('/World/robot')
    eye = Node('/World/robot/eye_link', parent=robot.prim)
    camera = Node('/World/robot/eye_link/Camera', parent=eye.prim, kind='Camera')
    camera.image_width, camera.image_height = 512, 288
    camera.modalities = {'rgb', 'seg_instance_id'}
    camera.focal_length, camera.horizontal_aperture, camera.clipping_range = 10., 20., np.array([.01, 100.])
    robot.links = {'eye_link': eye, 'base_link': robot}
    robot.sensors = {'head': camera}
    return SimpleNamespace(env=SimpleNamespace(robots=[robot]), camera_name='head', native_reset_id='reset-a')


def loaded(cls, name):
    return cls(definition=SimpleNamespace(id=name, version='v1'), path=Path('/tmp/' + name + '.yaml'),
        source_path=Path('/tmp/source.json'), file_sha256=name + '-sha', source_sha256='source-sha',
        variant='original', seed=17, scene_name='house', room_name='bathroom')


def test_actual_env_robot_before_assignment_and_read_only_native_frames(native_session):
    session = native_session
    session.robot = SimpleNamespace(sensors={'wrong': object()})
    session.loaded_task = loaded(LoadedTask, 'task')
    session.scene_adapter = SimpleNamespace(task=loaded(LoadedScene, 'scene'))
    before = session.env.robots[0].sensors['head'].__dict__.copy()
    evidence = measure_initial_camera_mounts(session, 'before_mount')
    assert evidence['robot']['source'] == 'session.env.robots[0]'
    assert evidence['sensor_inventory']['installed_names'] == ['head']
    row = evidence['sensors'][0]
    assert row['world_pose']['value']['position_m'] == [1., 2., 3.]
    assert row['local_to_parent_pose']['value']['position_m'] == [.1, .2, .3]
    assert row['parent_prim_path']['value'] == '/World/robot/eye_link'
    assert row['actual_parent_links'][0]['name'] == 'eye_link'
    assert row['parent_usd_world_transform']['value']['matrix4x4'][3] == [4., 5., 6., 1.]
    assert evidence['head_eye_links'][0]['name'] == 'eye_link'
    assert row['world_camera_forward']['value']['vector_world'] == [0., 0., -1.]
    assert row['properties']['image_width']['value'] == 512
    assert row['properties']['vertical_aperture']['value'] == 17.
    assert evidence['provenance']['task']['id']['value'] == 'task'
    assert evidence['provenance']['scene']['id']['value'] == 'scene'
    assert evidence['provenance']['reset_id']['value'] == 'reset-a'
    assert evidence['provenance']['episode_id']['available'] is False
    # Only read-call tracing changes in this fixture; no camera property is set.
    assert all(row == session.env.robots[0].sensors['head'].__dict__[key]
        for key, row in before.items() if key != 'reads' and not isinstance(row, np.ndarray))
    json.dumps(evidence, allow_nan=False)


def test_scene_is_never_mislabelled_as_task(native_session):
    native_session.scene_adapter = SimpleNamespace(task=loaded(LoadedScene, 'scene-only'))
    evidence = measure_initial_camera_mounts(native_session, 'loaded')
    assert evidence['provenance']['task']['available'] is False
    assert evidence['provenance']['scene']['file_sha256']['value'] == 'scene-only-sha'


def test_installed_measurement_failure_is_not_missing_sensor(native_session):
    sensor = native_session.env.robots[0].sensors['head']
    def fail(*, frame):
        raise RuntimeError('native pose not initialized')
    sensor.get_position_orientation = fail
    sensor.focal_length = float('nan')
    row = measure_initial_camera_mounts(native_session, 'early')['sensors'][0]
    assert row['status'] == 'installed'
    assert not row['world_pose']['available']
    assert 'native pose not initialized' in row['world_pose']['error']
    assert not row['world_camera_forward']['available']
    assert not row['properties']['focal_length']['available']
    json.dumps(row, allow_nan=False)


def test_no_forward_convention_for_non_camera_sensor(native_session):
    native_session.env.robots[0].sensors['head'].prim.kind = 'Xform'
    row = measure_initial_camera_mounts(native_session, 'early')['sensors'][0]
    assert not row['world_camera_forward']['available']
    assert 'USD Camera type is not established' in row['world_camera_forward']['error']


def test_camera_rotation_controls_forward(native_session):
    sensor = native_session.env.robots[0].sensors['head']
    sensor.get_position_orientation = lambda *, frame: (np.zeros(3), np.array([0., np.sqrt(.5), 0., np.sqrt(.5)]))
    row = measure_initial_camera_mounts(native_session, 'early')['sensors'][0]
    np.testing.assert_allclose(row['world_camera_forward']['value']['vector_world'], [-1., 0., 0.], atol=1e-12)


def test_missing_expected_sensor_and_missing_robot(native_session):
    native_session.env.robots[0].sensors = {}
    result = measure_initial_camera_mounts(native_session, 'early')
    assert result['requested_head_sensor']['status'] == 'missing'
    assert result['sensors'] == []
    native_session.env.robots = []
    result = measure_initial_camera_mounts(native_session, 'early')
    assert not result['robot']['available']
    assert not result['sensor_inventory']['available']


def test_usd_failure_does_not_hide_native_pose(native_session, monkeypatch):
    lazy = sys.modules['omnigibson.lazy']
    def fail(prim):
        raise RuntimeError('USD stage unavailable')
    monkeypatch.setattr(lazy.pxr.UsdGeom, 'Xformable', fail)
    row = measure_initial_camera_mounts(native_session, 'early')['sensors'][0]
    assert row['world_pose']['available']
    assert not row['usd_world_transform']['available']
    assert 'USD stage unavailable' in row['usd_world_transform']['error']
    assert row['actual_parent_links'][0]['world_pose']['available']
