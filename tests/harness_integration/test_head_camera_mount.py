"""Check that explicit head repairs keep the camera attached to its robot link.

Run: .venv/bin/python -m pytest -q tests/test_head_camera_mount.py
"""
from types import SimpleNamespace

import pytest

from wia.runtime.sim.esi_robot_cameras import apply_head_camera_mount


def test_head_override_uses_parent_coordinates_and_default_mount_stays_native():
    calls = []
    sensor = SimpleNamespace(set_position_orientation=lambda **kwargs: calls.append(kwargs))
    session = SimpleNamespace(camera_name='head', robot=SimpleNamespace(sensors={'head': sensor}))
    apply_head_camera_mount(session, {})
    assert not calls
    apply_head_camera_mount(session, {'head_camera': {'position_m': [-.047, 0., -.08],
        'quaternion_xyzw': [0., 0., 0., 2.]}})
    assert calls == [{'position': [-.047, 0., -.08], 'orientation': [0., 0., 0., 1.], 'frame': 'parent'}]


def test_invalid_optical_rotation_is_rejected_before_the_mount_changes():
    session = SimpleNamespace()
    with pytest.raises(ValueError, match='nonzero quaternion'):
        apply_head_camera_mount(session, {'head_camera': {'position_m': [0., 0., 0.],
            'quaternion_xyzw': [0., 0., 0., 0.]}})
