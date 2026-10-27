"""Benchmark mounted RGB observation keys. Run: python -m pytest tests/test_extraction.py.

The Harness adapter implementing these keys is core/robot_cameras.py; this
contract describes required evidence without importing its implementation.
"""
MOUNTED_RGB_VIEWS = (
    'agentview_rgb', 'wrist_rgb', 'right_wrist_rgb',
    'chassis_front_left_rgb', 'chassis_front_right_rgb',
    'chassis_left_rgb', 'chassis_right_rgb', 'chassis_rear_rgb',
)
