"""Check late camera RGB advances through real render boundaries.

Run: python -m pytest -q tests/test_capture_camera_refresh.py
The synthetic sensor returns stale pixels until nine render boundaries pass.
"""
from types import SimpleNamespace

import numpy as np

from wia.runtime.scene_capture_adapters import capture_loaded_views


def test_capture_drains_late_render_products_without_physics_or_mount_changes():
    frames = []
    sim = SimpleNamespace(render=lambda: frames.append(True))
    session = SimpleNamespace(scene_camera=object(), global_cameras={'global_diagonal': object()},
        pipeline=SimpleNamespace(og=SimpleNamespace(sim=sim)), invalidate_camera_cache=lambda: None,
        _sensor_rgb=lambda sensor: np.full((64, 64, 3), len(frames), dtype=np.uint8))
    selected = SimpleNamespace()
    views, _ = capture_loaded_views(session, selected, ['global_diagonal'])
    assert len(frames) == 9
    assert np.all(views['global_diagonal'] == 9)


def test_two_phase_human_preview_acquires_fresh_native_phase_frames_without_motion():
    from types import SimpleNamespace as NS
    renders = []
    controller = NS(phase='phase_1', prepared_boundary=None, transition=None)
    def advance():
        assert controller.prepared_boundary == (1, 0)
        controller.phase = 'phase_2'
        controller.transition = {'world_unchanged': True, 'physics_steps': 0}
    controller.advance = advance
    captures = []
    def capture(*, action_boundary):
        assert action_boundary
        value = 1 if controller.phase == 'phase_1' else 2
        captures.append(value)
        return NS(observation=NS(snapshot_id=value, action_step=0,
            agentview_rgb=np.full((4, 4, 3), value, np.uint8),
            wrist_rgb=np.full((4, 4, 3), value + 10, np.uint8)))
    session = NS(scene_camera=object(), global_cameras={'global_diagonal': object()},
        pipeline=NS(og=NS(sim=NS(render=lambda: renders.append(True)))),
        invalidate_camera_cache=lambda: None, _sensor_rgb=lambda sensor: np.zeros((4, 4, 3), np.uint8),
        temporal_phase_controller=controller, observation_source=NS(capture=capture))
    views, _ = capture_loaded_views(session, NS(), ['phase_1_head', 'phase_1_wrist',
        'phase_2_head', 'phase_2_wrist', 'phase_2_global_diagonal'])
    assert captures == [1, 2] and len(renders) == 18
    assert np.all(views['phase_1_head'] == 1) and np.all(views['phase_2_head'] == 2)
    assert session.temporal_capture_evidence['robot_actions'] == 0
    assert session.temporal_capture_evidence['mounted_view_coverage_requires_review'] is True
