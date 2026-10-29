#!/usr/bin/env python3
"""Verify saved 20x ESI exports and remove only their human source frames.

Run with the project Python; ffprobe decodes with two threads and a finite
600-second timeout. No encoding, simulator controls or model calls are made.

  python -B src/wia/runtime/commands/manage_esi_recording_retention.py prune RECORDING_DIR \
      --receipt-dir outputs/recording_retention_20261026/current [--apply]
  python -B src/wia/runtime/commands/manage_esi_recording_retention.py watch \
      --recording-root RUN/scene/recordings --workflow RUN/workflow.json \
      --url http://127.0.0.1:8650 --receipt-dir RECEIPTS [--apply]

Both commands are dry runs unless --apply is explicit. Prune requires saved
metadata, exactly 20x playback, and every listed video to pass full frame-count,
size, FPS and wall-duration checks. It preserves videos, samples and all logs.
Receipts are written outside source_frames. Watch polls GET /api/state every
30 seconds, waits for recording/finalizing and terminal workflow, and has no
whole-episode deadline. Export errors or abandoned pending data are reported
rather than silently discarded. Transient GET/workflow read errors are recorded
and retried at the same 30-second interval. Unexpected/missing evidence preserves
raw frames. Old-run media inventory/deletion is separately reviewed by main.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time
from urllib.request import urlopen

FFPROBE = '/usr/bin/ffprobe'
VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
TERMINAL_STAGES = {'complete', 'completed', 'finished', 'failed', 'error', 'cancelled',
                   'agent_finished', 'recording_export_failed'}


def now():
    return datetime.now(timezone.utc).isoformat()


def no_symlinks(path):
    path = Path(os.path.abspath(Path(path).expanduser()))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('Symlink paths are refused: ' + str(part))
    return path


def read_json(path):
    path = no_symlinks(path)
    if not path.is_file():
        raise ValueError('Missing regular JSON file: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path = no_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    no_symlinks(temporary)
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def tree_files(directory):
    directory = no_symlinks(directory)
    rows = []
    for base, dirs, files in os.walk(directory, followlinks=False):
        for name in (*dirs, *files):
            path = no_symlinks(Path(base) / name)
            mode = path.stat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError('Nonregular source frame entry: ' + str(path))
        for name in files:
            path = Path(base) / name
            entry = path.stat()
            rows.append({'path': str(path), 'bytes': entry.st_size, 'mtime_ns': entry.st_mtime_ns})
    return sorted(rows, key=lambda row: row['path'])


def finite_positive(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def verify_recording(directory, *, timeout=600):
    directory = no_symlinks(directory)
    if not directory.is_dir() or directory.name.startswith('.') or '.pending' in directory.name:
        raise ValueError('Only a finalized visible recording directory is eligible.')
    metadata = read_json(directory / 'metadata.json')
    if metadata.get('status') != 'saved':
        raise ValueError('Recording metadata is not saved.')
    config = metadata.get('config', {})
    speed = config.get('playback_speed')
    if isinstance(speed, bool) or speed != 20:
        raise ValueError('Export playback_speed must be exactly 20.')
    wall = metadata.get('wall_duration_s')
    if not finite_positive(wall):
        raise ValueError('Positive wall_duration_s is required.')
    videos = metadata.get('videos')
    if not isinstance(videos, list) or not videos:
        raise ValueError('A nonempty complete video list is required.')
    verified, seen = [], set()
    for video in videos:
        name = video.get('filename')
        if (not isinstance(name, str) or name in ('', '.', '..') or '/' in name or '\\' in name
                or Path(name).name != name or Path(name).suffix.lower() not in VIDEO_SUFFIXES or name in seen):
            raise ValueError('Video filenames must be distinct safe relative basenames.')
        seen.add(name)
        path = no_symlinks(directory / name)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('Missing or empty video: ' + name)
        process = subprocess.run([FFPROBE, '-v', 'error', '-threads', '2', '-select_streams', 'v:0',
            '-count_frames', '-show_entries',
            'stream=width,height,avg_frame_rate,nb_read_frames,duration:format=duration',
            '-of', 'json', str(path)], capture_output=True, text=True, timeout=timeout, check=False)
        if process.returncode or process.stderr.strip():
            raise ValueError('Video decoding failed: ' + name + ': ' + process.stderr[-1000:])
        probe = json.loads(process.stdout)
        stream = probe['streams'][0]
        width, height = int(stream['width']), int(stream['height'])
        frames, fps = int(stream['nb_read_frames']), float(Fraction(stream['avg_frame_rate']))
        duration = float(stream.get('duration', probe.get('format', {}).get('duration', 'nan')))
        if min(width, height, frames) <= 0 or not finite_positive(fps) or not finite_positive(duration):
            raise ValueError('Video lacks positive readable frames, dimensions, FPS or duration: ' + name)
        tolerance = max(2 / fps, .1)
        if abs(duration - wall / 20) > tolerance:
            raise ValueError('Video duration does not match wall_duration_s / 20: ' + name)
        if abs(frames / fps - duration) > tolerance:
            raise ValueError('Video frame count and duration disagree: ' + name)
        if 'video_fps' in config and (not finite_positive(config['video_fps'])
                or not math.isclose(fps, config['video_fps'], rel_tol=1e-5)):
            raise ValueError('Video FPS differs from metadata: ' + name)
        if 'frames' in video and (isinstance(video['frames'], bool) or video['frames'] != frames):
            raise ValueError('Video frame count differs from metadata: ' + name)
        if 'size' in video and video['size'] != [width, height]:
            raise ValueError('Video dimensions differ from metadata: ' + name)
        if 'duration_s' in video and (not finite_positive(video['duration_s'])
                or abs(video['duration_s'] - duration) > tolerance):
            raise ValueError('Video duration differs from metadata: ' + name)
        st = path.stat()
        if 'size_bytes' in video and video['size_bytes'] != st.st_size:
            raise ValueError('Video byte count differs from metadata: ' + name)
        verified.append({'filename': name, 'frames': frames, 'fps': fps, 'duration_s': duration,
                         'size': [width, height], 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns})
    return metadata, verified


def prune_recording(directory, receipt_dir, *, apply=False, timeout=600):
    directory = Path(os.path.abspath(directory))
    receipt_dir = no_symlinks(receipt_dir)
    frames = directory / 'source_frames'
    if receipt_dir == frames or frames in receipt_dir.parents:
        raise ValueError('Retention receipt must be outside deleted source_frames.')
    key = hashlib.sha256(str(directory).encode()).hexdigest()[:20]
    receipt_path = receipt_dir / ('prune_' + key + '.json')
    result = {'schema': 'esi_recording_retention_v1', 'checked_at': now(),
              'recording': str(directory), 'apply': apply, 'receipt': str(receipt_path),
              'removed_bytes': 0, 'removed_files': 0}
    try:
        metadata, videos = verify_recording(directory, timeout=timeout)
        result.update(playback_speed=20, videos=videos, metadata_status=metadata['status'])
        if not frames.exists() and not frames.is_symlink():
            previous = read_json(receipt_path) if receipt_path.exists() else {}
            if previous.get('status') == 'pruned':
                return dict(result, status='already_pruned', previous_prune_at=previous['checked_at'])
            result['status'] = 'verified_no_source_frames'
        else:
            no_symlinks(frames)
            if not frames.is_dir():
                raise ValueError('source_frames is not a directory.')
            files = tree_files(frames)
            result.update(source_bytes=sum(row['bytes'] for row in files), source_files=len(files))
            if apply:
                # Recheck exact metadata/video identities and frame membership
                # immediately before removing the sole authorized subtree.
                current = read_json(directory / 'metadata.json')
                if current != metadata or tree_files(frames) != files:
                    raise ValueError('Recording changed during verification.')
                for row in videos:
                    st = no_symlinks(directory / row['filename']).stat()
                    if (st.st_size, st.st_mtime_ns) != (row['bytes'], row['mtime_ns']):
                        raise ValueError('Verified video changed before pruning.')
                result.update(status='prune_started', source_frames=str(frames))
                write_json(receipt_path, result)
                shutil.rmtree(frames)
                result.update(status='pruned', removed_bytes=result['source_bytes'], removed_files=len(files))
            else:
                result['status'] = 'verified_dry_run'
    except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.SubprocessError) as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    write_json(receipt_path, result)
    return result


def get_state(url):
    url = url.rstrip('/')
    if url.endswith('/gumi'):
        url = url[:-5]
    with urlopen(url + '/api/state', timeout=15) as response:
        return json.load(response)


def watch(recording_root, workflow, url, receipt_dir, *, apply=False):
    root, workflow, receipt_dir = map(no_symlinks, (recording_root, workflow, receipt_dir))
    receipt = {'schema': 'esi_recording_retention_watch_v1', 'recording_root': str(root),
               'workflow': str(workflow), 'url': url, 'apply': apply, 'started_at': now()}
    while True:
        try:
            try:
                state, flow = get_state(url), read_json(workflow)
            except (OSError, json.JSONDecodeError) as error:
                diagnostic = {'at': now(), 'error': f'{type(error).__name__}: {error}'}
                receipt.setdefault('read_retry_diagnostics', []).append(diagnostic)
                receipt.update(status='waiting', checked_at=now(), waiting_for='transient_read_retry')
                write_json(receipt_dir / 'watch.json', receipt)
                time.sleep(30)
                continue
            stage = flow.get('stage', flow.get('status'))
            terminal = stage in TERMINAL_STAGES
            status = state.get('recording_status')
            if state.get('recording_error') or status == 'error':
                raise ValueError('Recorder export error; raw source frames preserved: ' + str(state.get('recording_error')))
            if status not in ('recording', 'finalizing', 'idle'):
                raise ValueError('Recorder status is missing or unknown.')
            results, pending = [], []
            if status == 'idle' and root.exists():
                paths = sorted(root.iterdir())
                for path in paths:
                    no_symlinks(path)
                    if path.is_dir() and (path.name.startswith('.') or '.pending' in path.name):
                        pending.append(str(path))
                if pending:
                    raise ValueError('Idle recorder has abandoned pending export; raw data preserved.')
                for path in paths:
                    if not path.is_dir():
                        continue
                    if (path / 'metadata.json').is_file():
                        results.append(prune_recording(path, receipt_dir, apply=apply))
                    elif path.name.startswith('recording_'):
                        raise ValueError('Final recording metadata is missing; raw data preserved.')
            if any(row['status'] == 'failed' for row in results):
                raise ValueError('Finalized recording verification failed; inspect per-record receipts.')
            receipt.update(checked_at=now(), workflow_stage=stage, recording_status=status,
                           recordings=results, pending=pending, status='waiting')
            if terminal and status == 'idle':
                if not results:
                    raise ValueError('Terminal workflow has no finalized recording to verify.')
                receipt.update(status='completed', finished_at=now())
            write_json(receipt_dir / 'watch.json', receipt)
            if receipt['status'] == 'completed':
                return receipt
        except (OSError, ValueError, TypeError, KeyError) as error:
            receipt.update(status='failed', finished_at=now(), error=f'{type(error).__name__}: {error}')
            write_json(receipt_dir / 'watch.json', receipt)
            return receipt
        time.sleep(30)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prune = commands.add_parser('prune')
    prune.add_argument('recording', type=Path)
    for command in (prune, commands.add_parser('watch')):
        command.add_argument('--receipt-dir', type=Path, required=True)
        command.add_argument('--apply', action='store_true')
        if command.prog.endswith(' watch'):
            command.add_argument('--recording-root', type=Path, required=True)
            command.add_argument('--workflow', type=Path, required=True)
            command.add_argument('--url', required=True)
    args = parser.parse_args(argv)
    if args.command == 'prune':
        result = prune_recording(args.recording, args.receipt_dir, apply=args.apply)
    else:
        result = watch(args.recording_root, args.workflow, args.url, args.receipt_dir, apply=args.apply)
    print(json.dumps(result, indent=2))
    return int(result['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
