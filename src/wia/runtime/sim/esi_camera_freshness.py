"""Check native rendered camera poses before publishing action-boundary images.

Used by ESIObservationSource; run the deferred CPU checks with:
    python -m pytest tests/test_native_camera_freshness.py
No rendering or physics is performed by these helpers.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from core.camera_geometry import pose_matrix
from wia.runtime.sim.esi_omnigibson import NativeCameraNotReady, _array


POSITION_TOLERANCE_M = 1e-3
ORIENTATION_TOLERANCE_DEG = 0.1


@contextmanager
def synchronized_camera_rendering(session):
    """Wait for native render products during canonical sensor acquisition.

    Camera poses alone cannot detect stale depth when only the gripper moves.
    Isaac Sim's blocking-render setting waits for the renderer; the existing
    graphics flush still drains its documented one-frame pipeline lag. Restore
    the caller's policy after acquisition, including failed captures. No physics
    step is added and human previews retain their existing rendering policy.
    """
    if not getattr(session, '_native_camera_freshness_required', False):
        yield
        return
    sim = session.pipeline.og.sim
    previous = sim.get_block_on_render()
    try:
        sim.set_block_on_render(True)
        yield
    finally:
        sim.set_block_on_render(previous)


def prepare_camera_freshness(session):
    """Attach pose annotators and retain current physical targets privately.

    Camera parameters must be attached before the caller's graphics flush.
    Reading VisionSensor.camera_parameters lazily would otherwise render inside
    the readback itself and could mix images from different graphics updates.
    """
    if not getattr(session, "_native_camera_freshness_required", False):
        return ()
    names = dict.fromkeys((session.camera_name, session.wrist_camera_name,
                          *session.robot_camera_registry.values()))
    targets = []
    for name in names:
        if name is None:
            continue
        sensor = session.robot.sensors[name]
        # Match VisionSensor.camera_parameters: attach the annotator directly,
        # without adding it to image modalities. The installed get_obs path
        # assumes dictionary modalities contain image data, which camera
        # parameters do not, and would otherwise fail on a missing 'data' key.
        if sensor._annotators.get("camera_params") is None:
            sensor.initialize_sensors(names="camera_params")
        position, quaternion = sensor.get_position_orientation()
        targets.append((name, sensor, pose_matrix(_array(position), _array(quaternion))))
    if not targets:
        raise RuntimeError("Native camera freshness requires mounted camera targets.")
    return tuple(targets)


def check_camera_freshness(session, targets):
    """Require rendered camera poses to match this capture's physical targets.

    OmniGibson's native evaluation uses inverse(reshape(view, 4, 4).T)
    to recover the USD camera world pose. Both sides here use USD camera axes;
    no optical-frame conversion or privileged world pose enters Agent input.
    A matching pose supplements the minimum graphics flush; unchanged pixels
    alone never establish a failure, since static scenes can repeat images.
    """
    if not targets:
        return
    cameras, failures = {}, []
    for name, sensor, target in targets:
        parameters = sensor.camera_parameters
        view = _array(parameters.get("cameraViewTransform", ())).astype(float)
        reason = None
        position_error = orientation_error = None
        if view.size != 16 or not np.isfinite(view).all():
            reason = "rendered_camera_pose_unavailable"
        else:
            try:
                rendered = np.linalg.inv(view.reshape(4, 4).T)
            except np.linalg.LinAlgError:
                reason = "rendered_camera_pose_singular"
            else:
                rotation = rendered[:3, :3]
                if (not np.isfinite(rendered).all()
                        or not np.allclose(rendered[3], [0, 0, 0, 1], atol=1e-5)
                        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                        or not np.isclose(np.linalg.det(rotation), 1., atol=1e-4)):
                    reason = "rendered_camera_pose_invalid"
                else:
                    position_error = float(np.linalg.norm(rendered[:3, 3] - target[:3, 3]))
                    cosine = (np.trace(target[:3, :3].T @ rotation) - 1.) / 2.
                    orientation_error = float(np.rad2deg(np.arccos(np.clip(cosine, -1., 1.))))
                    if (position_error > POSITION_TOLERANCE_M
                            or orientation_error > ORIENTATION_TOLERANCE_DEG):
                        reason = "rendered_camera_pose_stale"
        cameras[name] = {"status": "ready" if reason is None else reason,
                         "position_error_m": position_error,
                         "orientation_error_deg": orientation_error}
        if reason is not None:
            failures.append(f"{name}: {reason}")
    session._native_camera_freshness = {
        "status": "pending" if failures else "ready",
        "position_tolerance_m": POSITION_TOLERANCE_M,
        "orientation_tolerance_deg": ORIENTATION_TOLERANCE_DEG,
        "cameras": cameras,
    }
    if failures:
        counters = getattr(session, "performance", None)
        if counters:
            counters.add("camera_pose_sync_retry_count")
        raise NativeCameraNotReady("; ".join(failures))
