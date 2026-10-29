"""Deferred CPU regression checks for native camera-pose synchronization.

Run later with: python -m pytest tests/test_native_camera_freshness.py
These fixtures do not import or start a native simulator.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from core.camera_geometry import pose_matrix
from core.performance import PerformanceCounters
from wia.runtime.sim.esi_camera_freshness import (
    prepare_camera_freshness, check_camera_freshness, synchronized_camera_rendering,
)
from wia.runtime.sim.esi_omnigibson import NativeCameraNotReady


class Camera:
    def __init__(self, name):
        self.name = name
        self.modalities = {"rgb"}
        self._annotators = {}
        self.position = np.array([1., 2., 3.])
        self.quaternion = np.array([0., 0., 0., 1.])
        self.parameters_reads = 0
        self.set_rendered_pose(self.position, self.quaternion)

    def initialize_sensors(self, names):
        self._annotators[names] = object()

    def get_position_orientation(self):
        return self.position, self.quaternion

    def set_rendered_pose(self, position, quaternion):
        self.parameters = {"cameraViewTransform": np.linalg.inv(
            pose_matrix(position, quaternion)).T.reshape(-1)}

    @property
    def camera_parameters(self):
        self.parameters_reads += 1
        return self.parameters


def camera_session():
    cameras = {name: Camera(name) for name in ("head", "wrist", "chassis")}
    return SimpleNamespace(
        _native_camera_freshness_required=True,
        camera_name="head", wrist_camera_name="wrist",
        robot_camera_registry={"agentview_rgb": "head", "wrist_rgb": "wrist",
                               "chassis_front_left_rgb": "chassis"},
        robot=SimpleNamespace(sensors=cameras), performance=PerformanceCounters())


def test_prepare_attaches_all_mounted_cameras_without_reading_or_rendering():
    session = camera_session()
    targets = prepare_camera_freshness(session)
    assert [name for name, _, _ in targets] == ["head", "wrist", "chassis"]
    for _, camera, _ in targets:
        assert "camera_params" in camera._annotators
        assert camera.modalities == {"rgb"}
        assert camera.parameters_reads == 0
    check_camera_freshness(session, targets)
    assert session._native_camera_freshness["status"] == "ready"


@pytest.mark.parametrize("translation,yaw_deg", [(0.01, 0.), (0., 40.)])
def test_valid_but_stale_rendered_pose_is_rejected(translation, yaw_deg):
    session = camera_session()
    camera = session.robot.sensors["head"]
    position = camera.position + np.array([translation, 0., 0.])
    half_angle = np.deg2rad(yaw_deg) / 2.
    quaternion = [0., 0., np.sin(half_angle), np.cos(half_angle)]
    camera.set_rendered_pose(position, quaternion)
    targets = prepare_camera_freshness(session)
    with pytest.raises(NativeCameraNotReady, match="head: rendered_camera_pose_stale"):
        check_camera_freshness(session, targets)
    receipt = session._native_camera_freshness
    assert receipt["status"] == "pending"
    assert receipt["cameras"]["wrist"]["status"] == "ready"
    assert session.performance.snapshot()["camera_pose_sync_retry_count"] == 1
    camera.set_rendered_pose(camera.position, camera.quaternion)
    check_camera_freshness(session, targets)
    assert session._native_camera_freshness["status"] == "ready"


@pytest.mark.parametrize("view", [[], np.zeros(16), np.full(16, np.nan)])
def test_uninitialized_pose_is_unavailable_not_current(view):
    session = camera_session()
    session.robot.sensors["wrist"].parameters = {"cameraViewTransform": view}
    with pytest.raises(NativeCameraNotReady, match="wrist: rendered_camera_pose_"):
        check_camera_freshness(session, prepare_camera_freshness(session))


def test_target_pose_is_frozen_before_graphics_flush():
    session = camera_session()
    targets = prepare_camera_freshness(session)
    session.robot.sensors["head"].position[:] += 1.
    check_camera_freshness(session, targets)
    assert session._native_camera_freshness["status"] == "ready"


def test_diagnostic_receipt_contains_errors_without_world_pose_arrays():
    session = camera_session()
    check_camera_freshness(session, prepare_camera_freshness(session))
    for receipt in session._native_camera_freshness["cameras"].values():
        assert set(receipt) == {"status", "position_error_m", "orientation_error_deg"}


def test_non_native_adapters_do_not_require_native_camera_metadata():
    session = SimpleNamespace()
    assert prepare_camera_freshness(session) == ()
    check_camera_freshness(session, ())


@pytest.mark.parametrize('previous', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_render_synchronization_restores_owner_policy_without_physics(previous, fail):
    policy = [previous]
    sim = SimpleNamespace(get_block_on_render=lambda: policy[-1],
        set_block_on_render=policy.append,
        step=lambda: pytest.fail('Sensor synchronization must not step physics'))
    session = SimpleNamespace(_native_camera_freshness_required=True,
        pipeline=SimpleNamespace(og=SimpleNamespace(sim=sim)))
    try:
        with synchronized_camera_rendering(session):
            assert policy[-1] is True
            if fail:
                raise RuntimeError('Capture failed')
    except RuntimeError:
        assert fail
    assert policy == [previous, True, previous]
