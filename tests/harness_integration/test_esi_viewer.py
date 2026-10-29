"""Grouped GUMI cameras, recordings and explicit Agent projections."""
import json
import threading
import time
from datetime import datetime, timezone

import imageio_ffmpeg
import numpy as np
import pytest
import requests

from core.observations import AgentObservation, CameraSnapshot, freeze_rgb, _quaternion_rotation
from wia.runtime.sim.esi_debug_cameras import camera_orientation, depth_rgb, object_camera_target
from core.viewer_config import ViewerConfig
from wia.runtime.web.web_teleop.esi_preview import ESIPreviewBackend
from wia.runtime.web.web_teleop.esi_recording import HumanVideoRecorder, square_frame
from wia.runtime.web.web_teleop.esi_viewer import ESIViewerBackend
from gumi.web_teleop.server import serve
from harness_integration.test_esi_preview import METADATA
from harness_integration.test_agent_observation import source_and_rig
from harness_integration.test_esi_base import make_controller
from core.action_interface import ActionExecutor
from core.observation_config import ObservationConfig
from wia.runtime.sim.esi_actions import create_r1_executor
from wia.runtime.web.web_teleop.esi_control import ESIControlBackend


def snapshot():
    head = np.full((64, 64, 3), 60, np.uint8)
    obs = AgentObservation('Public instruction', head, head+30, (), 1,
                           datetime.now(timezone.utc).isoformat(), 0, True)
    private = {cam.name: head+index for index, cam in enumerate(ViewerConfig().global_views)}
    private.update(head_depth=head+50, wrist_depth=head+80)
    return CameraSnapshot(obs, scene_rgb=head+10, human_rgb=private)


def test_camera_groups_stay_selectable_while_images_are_not_yet_published(tmp_path):
    robot = ESIPreviewBackend(dict(METADATA))
    viewer = ESIViewerBackend(robot, tmp_path, ViewerConfig())
    try:
        before = viewer.state()
        assert [len(group['views']) for group in before['camera_groups']] == [1, 6, 4]
        assert before['available_views'] == []
        assert viewer.camera_group('global')['images'] == {}
        frame = snapshot()
        robot.publish(frame)
        viewer.publish(frame)
        ready = viewer.state()
        assert ready['camera_groups'] == before['camera_groups']
        assert len(viewer.camera_group('global')['images']) == 6
        with viewer.frame_cond:
            viewer.frames = {'head': viewer.frames['head']}
        partial = viewer.state()
        assert partial['camera_groups'] == before['camera_groups']
        assert partial['available_views'] == ['head']
    finally:
        viewer.close()


def decoded(path):
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    metadata = next(reader)
    frames = list(reader)
    return metadata, frames


def test_fixed_depth_scale_preserves_input_and_marks_invalid():
    depth = np.ones((64, 64), np.float32)
    depth[40, :4] = [0, np.nan, np.inf, -1]
    original = depth.copy()
    rgb = depth_rgb(depth)
    assert rgb.dtype == np.uint8 and rgb.shape == (64, 64, 3)
    assert np.array_equal(depth, original, equal_nan=True)
    assert not rgb[40, :4].any()
    another = depth_rgb(np.where(depth == 1, 1, 4))
    assert np.array_equal(rgb[45, 45], another[45, 45])


@pytest.mark.parametrize('position', [(2, 0, 1), (-2, 0, 1), (0, 2, 1), (0, -2, 1), (0, 0, 2)])
def test_global_camera_optical_axis_and_quaternion(position):
    target = np.array([0., 0., 0.])
    quat = camera_orientation(position, target)
    direction = _quaternion_rotation(quat) @ [0, 0, -1]
    expected = target-np.array(position)
    expected /= np.linalg.norm(expected)
    assert np.allclose(direction, expected, atol=1e-6)
    assert np.isclose(np.linalg.norm(quat), 1)


def test_diagnostic_task_framing_is_independent_of_distant_native_robot():
    positions = np.array([[-1.5, -3., .17], [-1.94, -2.65, .19]])
    native_base = np.array([0., 0., 0.])
    framed = object_camera_target(positions, native_base, include_robot=False)
    moved_robot = object_camera_target(positions, native_base + [100., 50., 0.], include_robot=False)
    assert np.array_equal(framed, moved_robot)
    camera = framed + [-.35, -.35, 1.35]
    rotation = _quaternion_rotation(camera_orientation(camera, framed))
    camera_points = (positions-camera) @ rotation
    assert (camera_points[:, 2] < 0).all()
    assert (np.linalg.norm(camera_points[:, :2], axis=1)/-camera_points[:, 2] < .3).all()
    # Existing Rigid Containment calls retain their robot-inclusive framing.
    default = object_camera_target(positions, native_base)
    assert np.array_equal(default, object_camera_target(positions, native_base, include_robot=True))
    assert not np.allclose(default, framed)


def test_configuration_and_square_letterbox():
    config = ViewerConfig.load('configs/esi_viewer.json')
    assert len(config.global_views) == 6 and config.playback_speed == 20
    assert ViewerConfig().playback_speed == config.playback_speed
    assert config.video_size == 512 and config.video_fps == 30
    assert config.recording_layout == ViewerConfig().recording_layout == 'groups'
    frame = square_frame(np.full((40, 80, 3), 100, np.uint8), 96)
    assert frame.shape == (96, 96, 3)
    assert not frame[0].any() and frame[48].min() == 100
    with pytest.raises(ValueError):
        ViewerConfig(playback_speed=0)


def test_human_depth_and_rgb_do_not_enable_disabled_agent_inputs():
    source, rig = source_and_rig()
    original = rig.observe

    def observe(**kwargs):
        raw = original(**kwargs)
        raw.update(head_depth_m=np.ones((12, 16), np.float32),
                   wrist_depth_m=np.full((12, 16), 2, np.float32))
        return raw

    rig.observe = observe
    source.include_scene = True
    source.config = ObservationConfig(head_rgb=False, wrist_rgb=False, head_depth=False, wrist_depth=False)
    frame = source.capture(action_boundary=True)
    assert frame.observation.agentview_rgb is None and frame.observation.wrist_rgb is None
    assert frame.observation.sensors.head_depth_m is None and frame.observation.sensors.wrist_depth_m is None
    assert {'head', 'wrist', 'scene', 'head_depth', 'wrist_depth'} <= set(frame.human_views())
    assert frame.observation.to_payload()['images'] == {'agentview_rgb': None, 'wrist_rgb': None}


def test_human_cancel_stops_pending_robot_action_without_executing(tmp_path):
    controller = make_controller()[0]
    robot = ESIControlBackend(dict(METADATA), create_r1_executor(controller))
    viewer = ESIViewerBackend(robot, tmp_path, ViewerConfig())
    calls = []
    robot.executor = ActionExecutor(lambda token, **kwargs: calls.append(token), capabilities=robot.capabilities)
    response = {}
    worker = threading.Thread(target=lambda: response.update(robot.submit('step', ['BASE_FWD'], timeout=3)))
    worker.start()
    for _ in range(100):
        if robot.busy:
            break
        time.sleep(0.005)
    assert robot.busy
    assert viewer.submit('cancel')['ok']
    assert robot.process_next()
    worker.join(timeout=2)
    assert not calls and not response['ok'] and not robot.busy
    assert not robot.cancel_active_action()
    robot.shutdown()


def test_encoder_failure_is_reported_without_claiming_saved_video(tmp_path, monkeypatch):
    def failure(*args, **kwargs):
        raise RuntimeError('test encoder unavailable')

    monkeypatch.setattr('wia.runtime.web.web_teleop.esi_recording.CompactVideoWriter', failure)
    recorder = HumanVideoRecorder(tmp_path, ViewerConfig())
    recorder.update(snapshot().human_views(), 1)
    assert recorder.start()['ok']
    recorder.worker.join(timeout=2)
    state = recorder.state()
    assert state['recording_status'] == 'error' and 'encoder unavailable' in state['recording_error']
    assert state['rollouts_saved'] == 0 and state['last_recording'] is None
    assert not recorder.finish()['ok']


@pytest.mark.parametrize('playback_speed', [1, 20])
def test_real_all_camera_videos_repeated_save_cancel_and_path_boundary(tmp_path, playback_speed):
    config = ViewerConfig(playback_speed=playback_speed, recording_layout='individual')
    recorder = HumanVideoRecorder(tmp_path, config)
    human = snapshot().human_views()
    recorder.update(human, 1)
    assert recorder.start()['ok']
    assert recorder.start()['status'] == 409
    time.sleep(1.1)
    recorder.update({name: freeze_rgb(np.full_like(rgb, 180)) for name, rgb in human.items()}, 2)
    time.sleep(1.05)
    result = recorder.finish()
    assert result['ok'], result
    record = result['recording']
    assert {video['view'] for video in record['videos']} == set(human)
    assert len(record['videos']) == 11 and record['final_snapshot_id'] == 2
    for video in record['videos']:
        path = tmp_path/record['id']/video['filename']
        metadata, frames = decoded(path)
        assert metadata['codec'] == 'h264' and metadata['size'] == (512, 512)
        assert metadata['fps'] == 30 and len(frames) == video['frames']
        assert abs(len(frames)/30-record['wall_duration_s']/playback_speed) <= 1/30+0.015
        assert video['size_bytes'] < 100_000
    assert recorder.recording_file(record['id']+'/head.mp4')
    timing = [json.loads(line) for line in (tmp_path/record['id']/'samples.jsonl').read_text().splitlines()]
    assert len(timing) == record['videos'][0]['frames']
    assert record['source_snapshot_count'] == 2
    assert recorder.recording_file('../outside.mp4') is None
    assert recorder.recording_file(record['id']+'/head.ffmpeg.log') is None
    assert recorder.start()['ok']
    assert recorder.finish()['ok']
    assert recorder.state()['rollouts_saved'] == 2
    saved = sorted(tmp_path.iterdir())
    assert recorder.start()['ok']
    assert recorder.finish(discard=True)['ok']
    assert sorted(tmp_path.iterdir()) == saved
    assert recorder.state()['rollouts_saved'] == 2 and not recorder.state()['recording']
    assert recorder.finish()['status'] == 409
    recorder.close()


def test_one_gumi_server_groups_recordings_and_agent_routes(tmp_path):
    robot = ESIPreviewBackend(dict(METADATA))
    viewer = ESIViewerBackend(robot, tmp_path, ViewerConfig())
    frame = snapshot()
    before = frame.observation.to_payload()
    robot.publish(frame)
    viewer.publish(frame)
    assert frame.observation.to_payload() == before
    assert set(robot.frames) == {'head', 'wrist', 'agentview'}
    server = serve(viewer, host='127.0.0.1', port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    human = f'http://127.0.0.1:{server.server_port}'
    try:
        state = requests.get(human+'/api/state').json()
        assert state['read_only'] and state['recording_enabled'] and len(state['camera_catalog']) == 11
        assert not requests.get(human+'/api/agent/state').json().get('human_viewer')
        assert [group['id'] for group in state['camera_groups']] == ['esi', 'global', 'robot']
        for group in state['camera_groups']:
            bundle = requests.get(human+'/api/camera-group/'+group['id']).json()
            assert set(bundle['images']) == set(group['views'])
            assert bundle['frame_seq'] == frame.observation.snapshot_id
        assert requests.get(human+'/api/camera-group/missing').status_code == 404
        for camera in state['camera_catalog']:
            assert requests.get(human+'/snapshot/'+camera['view']).status_code == 200
        for view in ('scene', 'global_front', 'head_depth'):
            assert requests.get(human+'/api/agent/snapshot/'+view).status_code == 404
        assert requests.get(human+'/api/agent/snapshot/head').status_code == 200
        assert requests.post(human+'/api/start', json={}).json()['ok']
        stop = requests.post(human+'/api/stop', json={}).json()
        assert stop['ok'] and not stop['state']['recording']
        video_url = stop['recording']['videos'][0]['url']
        response = requests.get(human+video_url)
        assert response.status_code == 200 and response.headers['Content-Type'] == 'video/mp4'
        metadata = json.loads((tmp_path/stop['recording']['id']/'metadata.json').read_text())
        assert {video['view'] for video in metadata['videos']} == {'esi', 'global', 'robot'}
        assert sorted(len(video['camera_views']) for video in metadata['videos']) == [1, 4, 6]
        assert requests.post(human+'/api/start', json={}).json()['ok']
        assert requests.post(human+'/api/cancel', json={}).json()['ok']
    finally:
        viewer.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
