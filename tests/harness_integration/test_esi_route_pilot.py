"""Check public-command quantization at short private route waypoints.

Run: python -B -m pytest -q tests/test_esi_route_pilot.py
"""
import math

import pytest

from core.action_compiler import PrimitiveResolution, compile_action
from core.action_interface import BackendCapabilities
from wia.runtime.sim.esi_route_pilot import next_translation


@pytest.mark.parametrize('yaw', [0., math.pi / 2, math.pi])
def test_short_diagonal_waypoint_remains_a_representable_public_command(yaw):
    """A 10 cm radial tolerance can still leave an axis below half a primitive."""
    quaternion = [0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]
    request = next_translation([0., 0., 0.], quaternion, [.08, .08], minimum_m=.125)
    plan = compile_action(request, BackendCapabilities(base_translation=True, base_rotation=True),
                          PrimitiveResolution(base_translation_m=.25, base_rotation_deg=15.))
    assert plan.repeat_count == 1
    assert request['distance_m'] == .125


def test_waypoint_translation_uses_actual_robot_heading():
    request = next_translation([0., 0., 0.], [0., 0., math.sqrt(.5), math.sqrt(.5)],
                               [1., 0.], minimum_m=.125)
    assert request == {'action': 'BASE_RIGHT', 'distance_m': .25}
