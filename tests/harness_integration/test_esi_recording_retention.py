"""Validate real-video retention; run: python -B -m pytest -q this file.

Uses tiny CPU-only ffmpeg fixtures with two threads; never touches project runs.
"""
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from wia.runtime.commands.manage_esi_recording_retention import prune_recording


@pytest.fixture(scope='session')
def tiny_video(tmp_path_factory):
    path = tmp_path_factory.mktemp('retention_video') / 'tiny.mp4'
    subprocess.run(['/usr/bin/ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
        '-i', 'color=c=blue:s=64x48:r=10:d=0.5', '-an', '-c:v', 'libx264', '-threads', '2',
        '-pix_fmt', 'yuv420p', str(path)], check=True, timeout=30, capture_output=True)
    return path


@pytest.fixture
def recording(tmp_path, tiny_video):
    root = tmp_path / 'recording_test'
    root.mkdir()
    shutil.copyfile(tiny_video, root / 'robot.mp4')
    frames = root / 'source_frames'
    frames.mkdir()
    (frames / 'robot_000000.png').write_bytes(b'human source image fixture')
    (root / 'robot.ffmpeg.log').write_text('Preserve original encoder evidence.\n')
    (root / 'samples.jsonl').write_text('{"sample":1}\n')
    metadata = {'status': 'saved', 'config': {'playback_speed': 20, 'video_fps': 10},
        'wall_duration_s': 10., 'videos': [{'filename': 'robot.mp4', 'frames': 5,
            'duration_s': .5, 'size': [64, 48], 'size_bytes': (root / 'robot.mp4').stat().st_size}]}
    (root / 'metadata.json').write_text(json.dumps(metadata))
    return root, tmp_path / 'receipts', metadata


def test_happy_prune_keeps_twenty_speed_video_and_all_metadata_logs_and_is_idempotent(recording):
    root, receipts, _ = recording
    retained = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    dry = prune_recording(root, receipts)
    assert dry['status'] == 'verified_dry_run' and (root / 'source_frames').is_dir()
    result = prune_recording(root, receipts, apply=True)
    assert result['status'] == 'pruned' and result['removed_files'] == 1
    assert not (root / 'source_frames').exists()
    assert {path.name: path.read_bytes() for path in root.iterdir()} == retained
    receipt = json.loads(Path(result['receipt']).read_text())
    assert receipt['status'] == 'pruned' and receipt['videos'][0]['frames'] == 5
    again = prune_recording(root, receipts, apply=True)
    assert again['status'] == 'already_pruned' and again['removed_files'] == 0
    assert json.loads(Path(result['receipt']).read_text()) == receipt


@pytest.mark.parametrize('problem', ['missing', 'corrupt', 'non20', 'recording', 'finalizing',
                                     'pending_directory', 'duration', 'frame_count', 'dimensions', 'fps'])
def test_unverified_export_preserves_source_frames(recording, problem):
    root, receipts, metadata = recording
    if problem == 'missing':
        (root / 'robot.mp4').unlink()
    elif problem == 'corrupt':
        (root / 'robot.mp4').write_bytes(b'not a video')
    elif problem == 'non20':
        metadata['config']['playback_speed'] = 10
    elif problem in ('recording', 'finalizing'):
        metadata['status'] = problem
    elif problem == 'pending_directory':
        root = root.rename(root.with_name('.recording_test.pending'))
    elif problem == 'duration':
        metadata['wall_duration_s'] = 200
    elif problem == 'frame_count':
        metadata['videos'][0]['frames'] = 9
    elif problem == 'dimensions':
        metadata['videos'][0]['size'] = [128, 48]
    elif problem == 'fps':
        metadata['config']['video_fps'] = 20
    (root / 'metadata.json').write_text(json.dumps(metadata))
    result = prune_recording(root, receipts, apply=True)
    assert result['status'] == 'failed'
    assert (root / 'source_frames' / 'robot_000000.png').is_file()
    assert result['removed_files'] == 0


@pytest.mark.parametrize('problem', ['escape', 'video_symlink', 'frame_symlink', 'root_symlink'])
def test_escape_and_symlinks_are_refused_without_deleting_targets(recording, problem):
    root, receipts, metadata = recording
    external = root.parent / 'outside.mp4'
    shutil.copyfile(root / 'robot.mp4', external)
    if problem == 'escape':
        metadata['videos'][0]['filename'] = '../outside.mp4'
        (root / 'metadata.json').write_text(json.dumps(metadata))
    elif problem == 'video_symlink':
        (root / 'robot.mp4').unlink()
        (root / 'robot.mp4').symlink_to(external)
    elif problem == 'frame_symlink':
        (root / 'source_frames' / 'linked.mp4').symlink_to(external)
    else:
        link = root.parent / 'alias'
        link.symlink_to(root, target_is_directory=True)
        root = link
    result = prune_recording(root, receipts, apply=True)
    assert result['status'] == 'failed'
    assert (root / 'source_frames' / 'robot_000000.png').is_file()
    assert external.is_file()


def test_one_bad_listed_view_prevents_pruning_any_source_frames(recording):
    root, receipts, metadata = recording
    metadata['videos'].append({'filename': 'missing_global.mp4'})
    (root / 'metadata.json').write_text(json.dumps(metadata))
    result = prune_recording(root, receipts, apply=True)
    assert result['status'] == 'failed'
    assert (root / 'source_frames').is_dir()


def watch_inputs(recording):
    root, receipts, metadata = recording
    workflow = root.parent / 'workflow.json'
    workflow.write_text(json.dumps({'stage': 'completed'}))
    return root.parent, workflow, receipts


def test_watch_waits_for_recording_and_finalizing_then_prunes_once(recording, monkeypatch):
    import wia.runtime.commands.manage_esi_recording_retention as tool
    root, _, _ = recording
    recording_root, workflow, receipts = watch_inputs(recording)
    states = iter(['recording', 'finalizing', 'idle'])
    sleeps, prunes = [], []
    original = tool.prune_recording

    def state(url):
        assert (root / 'source_frames').is_dir()
        status = next(states)
        return {'recording_status': status, 'recording_error': None}

    def prune(*args, **kwargs):
        prunes.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(tool, 'get_state', state)
    monkeypatch.setattr(tool, 'prune_recording', prune)
    monkeypatch.setattr(tool, 'time', SimpleNamespace(sleep=sleeps.append))
    result = tool.watch(recording_root, workflow, 'http://unused', receipts, apply=True)
    assert result['status'] == 'completed' and len(prunes) == 1
    assert sleeps == [30, 30]
    assert not (root / 'source_frames').exists()


def test_watch_retries_transient_get_error_then_prunes(recording, monkeypatch):
    import wia.runtime.commands.manage_esi_recording_retention as tool
    root, _, _ = recording
    recording_root, workflow, receipts = watch_inputs(recording)
    calls, sleeps = [], []

    def state(url):
        calls.append(url)
        if len(calls) == 1:
            raise OSError('Temporary connection reset')
        return {'recording_status': 'idle'}

    def waiting(seconds):
        assert (root / 'source_frames').is_dir()
        receipt = json.loads((receipts / 'watch.json').read_text())
        assert receipt['status'] == 'waiting'
        sleeps.append(seconds)

    monkeypatch.setattr(tool, 'get_state', state)
    monkeypatch.setattr(tool, 'time', SimpleNamespace(sleep=waiting))
    result = tool.watch(recording_root, workflow, 'http://unused', receipts, apply=True)
    assert result['status'] == 'completed' and len(calls) == 2 and sleeps == [30]
    assert len(result['read_retry_diagnostics']) == 1
    assert not (root / 'source_frames').exists()


def test_watch_retries_workflow_partial_write_then_prunes(recording, monkeypatch):
    import wia.runtime.commands.manage_esi_recording_retention as tool
    root, _, _ = recording
    recording_root, workflow, receipts = watch_inputs(recording)
    workflow.write_text('{')
    monkeypatch.setattr(tool, 'get_state', lambda url: {'recording_status': 'idle'})
    sleeps = []

    def finish_write(seconds):
        assert (root / 'source_frames').is_dir()
        sleeps.append(seconds)
        workflow.write_text(json.dumps({'stage': 'completed'}))

    monkeypatch.setattr(tool, 'time', SimpleNamespace(sleep=finish_write))
    result = tool.watch(recording_root, workflow, 'http://unused', receipts, apply=True)
    assert result['status'] == 'completed' and sleeps == [30]
    assert 'JSONDecodeError' in result['read_retry_diagnostics'][0]['error']
    assert not (root / 'source_frames').exists()


@pytest.mark.parametrize('problem', ['export_error', 'idle_pending'])
def test_watch_hard_export_or_pending_failure_preserves_raw_frames(recording, monkeypatch, problem):
    import wia.runtime.commands.manage_esi_recording_retention as tool
    root, _, _ = recording
    recording_root, workflow, receipts = watch_inputs(recording)
    state = {'recording_status': 'idle', 'recording_error': None}
    pending = recording_root / '.recording_abandoned.pending'
    if problem == 'export_error':
        state.update(recording_status='error', recording_error='Encoder failed')
    else:
        (pending / 'source_frames').mkdir(parents=True)
        (pending / 'source_frames' / 'frame.png').write_bytes(b'preserve failed export source')
    monkeypatch.setattr(tool, 'get_state', lambda url: state)
    result = tool.watch(recording_root, workflow, 'http://unused', receipts, apply=True)
    assert result['status'] == 'failed'
    assert (root / 'source_frames' / 'robot_000000.png').is_file()
    if problem == 'idle_pending':
        assert (pending / 'source_frames' / 'frame.png').is_file()
