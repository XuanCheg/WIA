"""Check native kinematic API handling without inventing velocity measurements.

Run: python -B -m pytest -q tests/test_task_edit2_native_kinematic.py.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from wia.authoring.converters.physical_dynamics_inclined_plane import _pose, _static_or_slow


def test_fixed_kinematic_velocity_is_unavailable_and_pose_is_still_measured():
    obj = SimpleNamespace(kinematic_only=True, fixed_base=True, visual_only=False,
        scale=np.ones(3), aabb=(np.zeros(3), np.ones(3)),
        get_position_orientation=lambda: (np.zeros(3), np.array([0., 0., 0., 1.])))
    row = _pose(obj)
    assert row['linear_speed_m_s'] is row['angular_speed_rad_s'] is None
    assert row['position'] == [0., 0., 0.]
    assert _static_or_slow(row)
    obj.kinematic_only = False
    with pytest.raises(AttributeError):
        _pose(obj)


def test_missing_dynamic_velocity_never_becomes_a_stationary_certificate():
    assert not _static_or_slow({'linear_speed_m_s': None, 'fixed_base': True})
    assert not _static_or_slow({'linear_speed_m_s': .1, 'kinematic_only': False, 'fixed_base': True})


def test_usd_quaternion_reset_compares_numeric_components_with_existing_tolerance():
    from wia.runtime.sim.esi_capacity_calibration import _equal, _plain
    quaternion = SimpleNamespace(GetReal=lambda: .9999999999999999,
        GetImaginary=lambda: [1.4e-8, -2.e-9, 3.8e-9])
    measured = _plain(quaternion)
    assert isinstance(measured, list)
    assert _equal(measured, [1., 0., 0., 0.])
    assert not _equal(measured, [1., .01, 0., 0.])
