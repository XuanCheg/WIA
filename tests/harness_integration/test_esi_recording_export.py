"""Verify video timing and successful/failed export retention.

Run: python -B -m pytest -q tests/test_esi_recording_export.py
"""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from core.viewer_config import ViewerConfig
from wia.runtime.web.web_teleop.esi_recording import HumanVideoRecorder, recording_frames, recording_layouts
from wia.runtime.commands.recover_esi_snapshot_video import main as recover_recording


def decoded_video(path):
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    metadata = next(reader)
    count, first, last = 0, None, None
    for frame in reader:
        count += 1
        if first is None:
            first = frame
        last = frame
    return metadata, count, first, last


def test_default_snapshot_export_keeps_video_and_timing_without_source_images(tmp_path):
    config = ViewerConfig(video_size=64, recording_mode='snapshots', recording_views=('head', 'wrist'),
                          recording_layout='individual')
    recorder = HumanVideoRecorder(tmp_path, config)
    recorder.recording_id = 'recording_test'
    recorder.pending = tmp_path/'.recording_test.pending'
    recorder.pending.mkdir(parents=True)
    recorder.discard = False
    recorder.started, recorder.ended = 100., 140.
    recorder.started_at = '2026-10-10T00:00:00+00:00'
    recorder.initial_snapshot = 1
    recorder.initial_frames = {view: np.full((64, 64, 3), 50, np.uint8) for view in ('head', 'wrist')}
    recorder.source_queue.put((120., 2, {view: rgb+130 for view, rgb in recorder.initial_frames.items()}))
    recorder.finish_event.set()
    recorder._record_snapshots()
    record = recorder.state()['last_recording']
    assert record is not None, recorder.state()
    assert record['wall_duration_s'] == 40 and record['config']['playback_speed'] == 20
    path = tmp_path/record['id']
    sources = [json.loads(line) for line in (path/'samples.jsonl').read_text().splitlines()]
    assert [row['wall_elapsed_s'] for row in sources] == [0, 20]
    assert not (path/'source_frames').exists()
    assert not (path/'source_metadata.json').exists()
    assert not list(path.glob('*.concat.txt'))
    assert record['source_frames_retained'] is False
    counts = []
    for video in record['videos']:
        metadata, frames, first, last = decoded_video(path/video['filename'])
        assert metadata['fps'] == 30 and metadata['size'] == (64, 64)
        assert frames == video['frames'] and abs(frames/30-2.) <= 1/30
        assert abs(np.frombuffer(first, np.uint8).mean()-50) < 3
        assert abs(np.frombuffer(last, np.uint8).mean()-180) < 3
        counts.append(frames)
    assert counts[0] == counts[1]


@pytest.mark.parametrize('source_speed', [1, 20])
def test_recovery_applies_default_speed_once_to_legacy_and_current_sources(tmp_path, monkeypatch, source_speed):
    (tmp_path/'native_exit_code.txt').write_text('0\n')
    source = tmp_path/'recordings'/'.recording_test.pending'
    frames = source/'source_frames'
    frames.mkdir(parents=True)
    samples = []
    for index, value in enumerate((50, 180)):
        files = {}
        for view in ('head', 'wrist'):
            filename = f'source_frames/{view}_{index:06d}.png'
            assert cv2.imwrite(str(source/filename), np.full((512, 512, 3), value, np.uint8))
            files[view] = filename
        samples.append({'snapshot_id': index+1, 'wall_elapsed_s': index*20., 'files': files})
    (source/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in samples))
    (source/'head.concat.txt').write_text(''.join(
        "file '"+row['files']['head']+"'\nduration "+str(20/source_speed)+'\n' for row in samples))
    if source_speed == 20:
        (source/'source_metadata.json').write_text(json.dumps({
            'config': asdict(ViewerConfig(video_size=512, recording_mode='snapshots', recording_layout='individual')), 'wall_duration_s': 40.,
        }))
    originals = {path: path.read_bytes() for path in source.rglob('*') if path.is_file()}
    monkeypatch.setattr(sys, 'argv', ['recover_esi_snapshot_video.py', str(tmp_path)])
    assert recover_recording() == 0
    destination = source.parent/'recording_test_offline_export'
    record = json.loads((destination/'metadata.json').read_text())
    assert record['wall_duration_s'] == 40 and record['config']['playback_speed'] == 20
    assert record['source_playback_speed'] == source_speed
    for video in record['videos']:
        metadata, count, first, last = decoded_video(destination/video['filename'])
        assert metadata['fps'] == 30 and metadata['size'] == (512, 512)
        assert count == video['frames'] and abs(count/30-2.) <= 1/30
        assert abs(np.frombuffer(first, np.uint8).mean()-50) < 3
        assert abs(np.frombuffer(last, np.uint8).mean()-180) < 3
    assert all(path.read_bytes() == original for path, original in originals.items())


def grouped_recorder(tmp_path):
    config = ViewerConfig(video_size=64, recording_mode='snapshots')
    catalog = ([{'view': 'scene', 'label': 'ESI camera', 'group': 'ESI'}]
               + [{'view': f'global_{name}', 'label': name, 'group': 'Global'} for name in ('front', 'left', 'right')]
               + [{'view': name, 'label': name, 'group': 'Robot'} for name in
                  ('head', 'wrist', 'right_wrist', 'chassis_front_left', 'chassis_front_right',
                   'chassis_left', 'chassis_right', 'chassis_rear')])
    frames = {camera['view']: np.full((64, 64, 3), 40+index*8, np.uint8)
              for index, camera in reversed(list(enumerate(catalog)))}
    recorder = HumanVideoRecorder(tmp_path, config, metadata={'camera_catalog': catalog})
    recorder.recording_id = 'recording_grouped'
    recorder.pending = tmp_path/'.recording_grouped.pending'
    recorder.pending.mkdir(parents=True)
    recorder.discard = False
    recorder.started, recorder.ended = 100., 140.
    recorder.started_at = '2026-10-10T00:00:00+00:00'
    recorder.initial_snapshot = 1
    recorder.initial_frames = frames
    recorder.source_queue.put((120., 2, {name: rgb+80 for name, rgb in frames.items()}))
    recorder.finish_event.set()
    return recorder


def test_grouped_snapshot_export_decodes_synchronized_camera_cells_and_labels(tmp_path):
    recorder = grouped_recorder(tmp_path)
    originals = {name: rgb.copy() for name, rgb in recorder.initial_frames.items()}
    recorder._record_snapshots()
    record = recorder.state()['last_recording']
    assert record is not None, recorder.state()
    assert {video['view'] for video in record['videos']} == {'esi', 'global', 'robot'}
    sources = [json.loads(line) for line in (Path(record['path'])/'samples.jsonl').read_text().splitlines()]
    assert [row['snapshot_id'] for row in sources] == [1, 2]
    assert all(set(row['files']) == {'esi', 'global', 'robot'} for row in sources)
    assert not (Path(record['path'])/'source_frames').exists()
    assert record['source_frames_retained'] is False
    for video in record['videos']:
        metadata, count, first, last = decoded_video(Path(record['path'])/video['filename'])
        width, height = video['size']
        assert metadata['size'] == (width, height) and metadata['codec'] == 'h264'
        assert abs(count/metadata['fps']-2.) <= 1/30
        first = np.frombuffer(first, np.uint8).reshape(height, width, 3)
        last = np.frombuffer(last, np.uint8).reshape(height, width, 3)
        for index, name in enumerate(video['camera_views']):
            row, column = divmod(index, video['columns'])
            y, x = row*88+24+32, column*64+32
            assert abs(first[y, x].mean()-originals[name].mean()) < 4
            assert abs(last[y, x].mean()-originals[name].mean()-80) < 4
        assert first[:24].max() > 150  # Camera identity labels survive encoding.
    assert all(np.array_equal(rgb, originals[name]) for name, rgb in recorder.initial_frames.items())
    robot = next(video for video in record['videos'] if video['view'] == 'robot')
    assert robot['columns'] == 2 and robot['rows'] == 4 and robot['size'] == [128, 352]
    assert recorder.recording_file(record['id']+'/robot.mp4')


def test_group_grid_leaves_unused_cells_black_without_mutating_sensor_images():
    config = ViewerConfig(video_size=64)
    frames = {name: np.full((64, 64, 3), value, np.uint8)
              for name, value in [('head', 50), ('wrist', 100), ('right_wrist', 150)]}
    for rgb in frames.values():
        rgb.flags.writeable = False
    layouts = recording_layouts(frames, config)
    grid = recording_frames(frames, layouts, config)['robot']
    assert grid.shape == (176, 128, 3)
    assert not grid[88:, 64:].any()
    assert np.array_equal(grid[24:88, :64], frames['head'])


def test_failed_group_export_recovers_original_grid_dimensions_and_timing(tmp_path, monkeypatch):
    from unittest.mock import patch
    recorder = grouped_recorder(tmp_path/'recordings')
    with patch('wia.runtime.web.web_teleop.esi_recording.subprocess.run', side_effect=RuntimeError('test encoder failure')):
        recorder._record_snapshots()
    assert recorder.state()['recording_status'] == 'error' and recorder.last_recording is None
    originals = {path: path.read_bytes() for path in recorder.pending.rglob('*') if path.is_file()}
    (tmp_path/'native_exit_code.txt').write_text('0\n')
    monkeypatch.setattr(sys, 'argv', ['recover_esi_snapshot_video.py', str(tmp_path)])
    assert recover_recording() == 0
    recovered = json.loads((tmp_path/'recording_recovery.json').read_text())
    assert recovered['config']['recording_layout'] == 'groups'
    assert {video['view'] for video in recovered['videos']} == {'esi', 'global', 'robot'}
    for video in recovered['videos']:
        metadata, count, _, _ = decoded_video(Path(recovered['path'])/video['filename'])
        assert metadata['size'] == tuple(video['size'])
        assert count == video['frames'] and abs(count/metadata['fps']-2.) <= 1/30
    assert all(path.read_bytes() == original for path, original in originals.items())


def test_later_view_failure_preserves_all_source_images(tmp_path, monkeypatch):
    recorder = grouped_recorder(tmp_path)
    original = __import__('subprocess').run
    calls = []

    def fail_second_view(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 2:
            raise RuntimeError('test second-view encoder failure')
        return original(*args, **kwargs)

    monkeypatch.setattr('wia.runtime.web.web_teleop.esi_recording.subprocess.run', fail_second_view)
    recorder._record_snapshots()
    assert recorder.state()['recording_status'] == 'error'
    assert recorder.last_recording is None
    assert len(list((recorder.pending/'source_frames').glob('*.jpg'))) == 6
    assert (recorder.pending/'source_metadata.json').is_file()
    assert (recorder.pending/'esi.mp4').is_file()


def test_snapshot_video_omits_model_wait_but_retains_wall_timestamps(tmp_path):
    config = ViewerConfig(video_size=64, recording_mode='snapshots', recording_views=('head',),
                          recording_layout='individual', playback_speed=1)
    recorder = HumanVideoRecorder(tmp_path, config)
    recorder.recording_id = 'recording_skip_wait'
    recorder.pending = tmp_path/'.recording_skip_wait.pending'
    recorder.pending.mkdir()
    recorder.discard = False
    recorder.started, recorder.ended = 100., 140.
    recorder.started_at = '2026-10-28T00:00:00+00:00'
    recorder.pause_intervals = [[102., 138.]]
    recorder.initial_snapshot = 1
    recorder.initial_frames = {'head': np.full((64, 64, 3), 50, np.uint8)}
    recorder.source_queue.put((139., 2, {'head': np.full((64, 64, 3), 180, np.uint8)}))
    recorder.finish_event.set()
    recorder._record_snapshots()
    record = recorder.state()['last_recording']
    assert record is not None, recorder.state()
    assert record['wall_duration_s'] == 40
    assert record['recorded_duration_s'] == 4
    assert record['omitted_model_wait_s'] == 36
    samples = [json.loads(line) for line in (tmp_path/record['id']/'samples.jsonl').read_text().splitlines()]
    assert [row['wall_elapsed_s'] for row in samples] == [0, 39]
    assert [row['recorded_elapsed_s'] for row in samples] == [0, 3]
    _, frames, first, last = decoded_video(tmp_path/record['id']/'head.mp4')
    assert abs(frames/config.video_fps - 4) <= 1/config.video_fps
    assert abs(np.frombuffer(first, np.uint8).mean()-50) < 3
    assert abs(np.frombuffer(last, np.uint8).mean()-180) < 3


def test_pause_transitions_freeze_timeline_and_skip_wait_captures(tmp_path, monkeypatch):
    import wia.runtime.web.web_teleop.esi_recording as module
    recorder = HumanVideoRecorder(tmp_path, ViewerConfig(recording_mode='snapshots'))
    recorder.started, recorder.status = 100., 'recording'
    clock = [101.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    recorder.set_model_wait(True)
    clock[0] = 131.
    assert recorder.recorded_elapsed(clock[0]) == 1
    recorder.update({'head': np.zeros((4, 4, 3), np.uint8)}, 2)
    assert recorder.source_queue.empty()
    recorder.set_model_wait(False)
    clock[0] = 133.
    assert recorder.recorded_elapsed(clock[0]) == 3
    recorder.update({'head': np.zeros((4, 4, 3), np.uint8)}, 3)
    assert recorder.source_queue.qsize() == 1
    recorder.set_model_wait(True)
    recorder.ended = 140.
    assert recorder.timeline_metadata()['recorded_duration_s'] == 3
    assert recorder.timeline_metadata()['omitted_model_wait_s'] == 37


def test_legacy_stream_recording_omits_model_wait(tmp_path):
    import time
    config = ViewerConfig(video_size=64, recording_mode='constant_fps', recording_views=('head',),
                          recording_layout='individual', playback_speed=1, video_fps=10)
    recorder = HumanVideoRecorder(tmp_path, config)
    recorder.update({'head': np.zeros((64, 64, 3), np.uint8)}, 1)
    assert recorder.start()['ok']
    time.sleep(.15)
    recorder.set_model_wait(True)
    time.sleep(.5)
    recorder.set_model_wait(False)
    recorder.update({'head': np.full((64, 64, 3), 180, np.uint8)}, 2)
    time.sleep(.15)
    result = recorder.finish()
    assert result['ok'], result
    record = result['recording']
    assert record['omitted_model_wait_s'] >= .5
    assert abs(record['recorded_duration_s'] + record['omitted_model_wait_s'] - record['wall_duration_s']) < 1e-6
    video = record['videos'][0]
    assert abs(video['duration_s'] - record['recorded_duration_s']) <= .1


def test_viewer_tracks_model_wait_phase_without_changing_world_policy(tmp_path):
    from types import SimpleNamespace
    from wia.runtime.sim.esi_world_policy import WorldPolicy
    from wia.runtime.web.web_teleop.esi_viewer import ESIViewerBackend
    policy = WorldPolicy()
    backend = SimpleNamespace(metadata={}, observation_source=None, world_policy=policy)
    viewer = ESIViewerBackend(backend, tmp_path, ViewerConfig())
    for phase, paused in [('awaiting_model', True), ('capturing_observation', False),
                          ('executing', False), ('awaiting_model', True), ('human_idle', False)]:
        policy.transition(phase)
        assert viewer.recorder.model_wait_paused is paused
        assert policy.advances is True
