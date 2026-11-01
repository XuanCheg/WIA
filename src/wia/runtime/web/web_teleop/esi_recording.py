"""Bounded-memory human video recorder. Use run_esi_robot.py --web --human-recording."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
import time
import queue
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from core.viewer_config import ViewerConfig


def square_frame(rgb, size):
    height, width = rgb.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(rgb, (max(1, round(width*scale)), max(1, round(height*scale))),
                         interpolation=cv2.INTER_AREA)
    frame = np.zeros((size, size, 3), np.uint8)
    h, w = resized.shape[:2]
    frame[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = resized
    return frame


def recording_layouts(frames, config, catalog=()):
    """Use the viewer's group order and grid columns for synchronized exports."""
    cameras = {item['view']: item for item in catalog}
    names = [item['view'] for item in catalog if item['view'] in frames]
    names.extend(name for name in frames if name not in names)
    if config.recording_layout == 'individual':
        return {name: {'camera_views': [name], 'label': cameras.get(name, {}).get('label', name),
                       'columns': 1, 'rows': 1, 'size': [config.video_size, config.video_size]}
                for name in names}
    groups = {}
    for name in names:
        fallback = 'ESI' if name == 'scene' else 'Global' if name.startswith('global_') else 'Robot'
        group = cameras.get(name, {}).get('group', fallback)
        groups.setdefault(group, []).append(name)
    layouts = {}
    for group, key, columns in (('ESI', 'esi', 1), ('Global', 'global', 3), ('Robot', 'robot', 2)):
        views = groups.get(group, [])
        if not views:
            continue
        columns = min(columns, len(views))
        rows = math.ceil(len(views) / columns)
        layouts[key] = {'group': key, 'label': group, 'camera_views': views,
                        'camera_labels': [cameras.get(name, {}).get('label', name) for name in views],
                        'columns': columns, 'rows': rows, 'tile_size': config.video_size,
                        'label_height': 24, 'size': [columns * config.video_size,
                                                  rows * (config.video_size + 24)]}
    return layouts


def recording_frames(frames, layouts, config):
    """Compose copied human frames without modifying any robot sensor buffer."""
    outputs = {}
    for name, layout in layouts.items():
        if config.recording_layout == 'individual':
            outputs[name] = square_frame(frames[name], config.video_size)
            continue
        width, height = layout['size']
        canvas = np.zeros((height, width, 3), np.uint8)
        size, label_height = layout['tile_size'], layout['label_height']
        for index, view in enumerate(layout['camera_views']):
            row, column = divmod(index, layout['columns'])
            y, x = row * (size + label_height), column * size
            canvas[y+label_height:y+label_height+size, x:x+size] = square_frame(frames[view], size)
            cv2.putText(canvas, layout['camera_labels'][index], (x+5, y+17),
                        cv2.FONT_HERSHEY_SIMPLEX, min(.55, max(.3, size/900)),
                        (235, 235, 235), 1, cv2.LINE_AA)
        outputs[name] = canvas
    return outputs


class CompactVideoWriter:
    def __init__(self, path, config, *, frame_size=None):
        self.path, self.config = Path(path), config
        self.frame_size = frame_size or (config.video_size, config.video_size)
        self.frames = 0
        self.latest_rgb = None
        self.latest_bytes = None
        self.log = self.path.with_suffix('.ffmpeg.log').open('wb')
        width, height = self.frame_size
        try:
            self.process = subprocess.Popen([
                imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
                '-framerate', str(config.video_fps), '-i', 'pipe:0', '-an',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', str(config.video_crf),
                '-threads', '1', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.path),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        except BaseException:
            self.log.close()
            raise

    def append(self, rgb):
        if rgb is not self.latest_rgb:
            self.latest_rgb = rgb
            if rgb.shape[:2] != tuple(reversed(self.frame_size)):
                raise ValueError('Recording frame dimensions changed during export.')
            self.latest_bytes = np.ascontiguousarray(rgb).tobytes()
        self.process.stdin.write(self.latest_bytes)
        self.frames += 1

    def close(self):
        try:
            self.process.stdin.close()
            code = self.process.wait()
            if code != 0 or not self.path.is_file() or self.path.stat().st_size == 0:
                raise RuntimeError(f'Cannot finalize {self.path.name}; see encoder log.')
        finally:
            self.log.close()


class HumanVideoRecorder:
    """Sample latest frames at output_fps / speed, excluding model-wait intervals.

    At the default 30 fps and 20x, each output frame spans 2/3 wall-clock seconds.
    Selected cameras share timestamps. Legacy mode samples constant-FPS ticks;
    Snapshot mode temporarily stores each capture once and exports on stop.
    After every selected video is encoded successfully, source images and
    encoder input lists are removed; timing, metadata and videos remain.
    """
    def __init__(self, root, config: ViewerConfig, *, metadata=None, performance=None):
        self.root = Path(root).expanduser().resolve()
        self.config = config
        self.metadata = metadata or {}
        self.performance = performance
        self.lock = threading.RLock()
        self.latest = {}
        self.latest_snapshot = None
        self.status = 'idle'
        self.saved = 0
        self.last_recording = None
        self.recording_id = None
        self.error = None
        self.worker = None
        self.finish_event = threading.Event()
        self.source_queue = queue.Queue(maxsize=8)
        self.dropped_snapshots = 0
        self.model_wait_paused = False
        self.pause_intervals = []

    def set_model_wait(self, paused):
        """Exclude model waiting from video time without changing simulation time."""
        with self.lock:
            if paused == self.model_wait_paused:
                return
            self.model_wait_paused = paused
            if self.status == 'recording':
                if paused:
                    self.pause_intervals.append([time.monotonic(), None])
                elif self.pause_intervals and self.pause_intervals[-1][1] is None:
                    self.pause_intervals[-1][1] = time.monotonic()

    def recorded_elapsed(self, stamp):
        with self.lock:
            omitted = sum(max(0., min(stamp, end if end is not None else stamp) - max(self.started, begin))
                          for begin, end in self.pause_intervals)
            return max(0., stamp - self.started - omitted)

    def timeline_metadata(self):
        elapsed = self.recorded_elapsed(self.ended)
        return {'recorded_duration_s': elapsed,
                'omitted_model_wait_s': max(0., self.ended-self.started-elapsed),
                'skip_model_wait': True}

    def update(self, frames, snapshot_id, captured_at=None):
        with self.lock:
            self.latest = {name: rgb for name, rgb in frames.items()
                           if not self.config.recording_views or name in self.config.recording_views}
            previous = self.latest_snapshot
            self.latest_snapshot = snapshot_id
            now = time.monotonic()
            if captured_at is not None:
                age = max(0., (datetime.now(timezone.utc)-datetime.fromisoformat(captured_at)).total_seconds())
                now -= age
            self.latest_timestamp = now
            if self.status == 'recording' and self.config.recording_mode == 'snapshots' and snapshot_id != previous and not self.model_wait_paused:
                try:
                    self.source_queue.put_nowait((self.latest_timestamp, snapshot_id, dict(self.latest)))
                except queue.Full:
                    self.dropped_snapshots += 1

    def state(self):
        with self.lock:
            return {'recording': self.status == 'recording', 'recording_status': self.status,
                    'rollout': self.recording_id, 'rollouts_saved': self.saved,
                    'can_stop': self.status == 'recording', 'recording_error': self.error,
                    'recording_elapsed_s': max(0., time.monotonic()-self.started) if self.status == 'recording' else 0.,
                    'recording_settings': {'aspect_ratio': 'per-group' if self.config.recording_layout == 'groups' else '1:1',
                                           'size': self.config.video_size,
                                           'fps': self.config.video_fps, 'playback_speed': self.config.playback_speed,
                                           'codec': 'H.264', 'crf': self.config.video_crf,
                                           'layout': self.config.recording_layout},
                    'recording_layouts': recording_layouts(self.latest, self.config,
                                                          self.metadata.get('camera_catalog', ())),
                    'recording_source_mode': self.config.recording_mode,
                    'recording_views': list(self.config.recording_views),
                    'last_recording': self.last_recording}

    def start(self):
        with self.lock:
            if self.status in ('recording', 'finalizing'):
                return {'ok': False, 'status': 409, 'message': 'A recording is already active.'}
            if not self.latest:
                return {'ok': False, 'status': 503, 'message': 'Camera frames are not ready.'}
            self.root.mkdir(parents=True, exist_ok=True)
            self.recording_id = 'recording_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            self.pending = self.root / ('.' + self.recording_id + '.pending')
            self.pending.mkdir()
            self.started = time.monotonic()
            self.pause_intervals = [[self.started, None]] if self.model_wait_paused else []
            self.started_at = datetime.now(timezone.utc).isoformat()
            self.initial_frames = dict(self.latest)
            self.initial_snapshot = self.latest_snapshot
            self.error, self.discard = None, False
            self.status = 'recording'
            self.finish_event.clear()
            self.source_queue = queue.Queue(maxsize=8)
            self.dropped_snapshots = 0
            if self.config.recording_views and set(self.config.recording_views)-set(self.latest):
                self.status = 'idle'
                shutil.rmtree(self.pending)
                return {'ok': False, 'status': 503, 'message': 'Selected recording cameras are not ready.'}
            worker_target = self._record_snapshots if self.config.recording_mode == 'snapshots' else self._record
            self.worker = threading.Thread(target=worker_target, name='esi-video', daemon=True)
            self.worker.start()
            return {'ok': True, 'message': f'Recording all {len(self.initial_frames)} cameras · {self.config.playback_speed:g}× playback.'}

    def _record_snapshots(self):
        """Spool unique captures, export synchronized videos, then remove sources."""
        sources, videos, error = [], [], None
        source_names = tuple(self.initial_frames)
        layouts = recording_layouts(self.initial_frames, self.config, self.metadata.get('camera_catalog', ()))
        names = tuple(layouts)
        try:
            directory = self.pending / 'source_frames'
            directory.mkdir()
            def append(stamp, snapshot_id, frames):
                if sources and snapshot_id == sources[-1]['snapshot_id']:
                    return
                if any(name not in frames for name in source_names):
                    self.dropped_snapshots += 1
                    return
                index = len(sources)
                files = {}
                started = time.perf_counter()
                for name, frame in recording_frames(frames, layouts, self.config).items():
                    path = directory / f'{name}_{index:06d}.jpg'
                    if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 80]):
                        raise RuntimeError('Cannot save source recording frame.')
                    files[name] = str(path.relative_to(self.pending))
                if self.performance:
                    self.performance.add('recording_source_write_wall_time', time.perf_counter()-started)
                sources.append({'sample': index+1, 'wall_elapsed_s': max(0., stamp-self.started),
                                'recorded_elapsed_s': self.recorded_elapsed(stamp),
                                'snapshot_id': snapshot_id, 'files': files})
            append(self.started, self.initial_snapshot, self.initial_frames)
            while not self.finish_event.is_set() or not self.source_queue.empty():
                try:
                    stamp, snapshot_id, frames = self.source_queue.get(timeout=.1)
                except queue.Empty:
                    continue
                append(stamp, snapshot_id, frames)
            duration = max(.001, self.recorded_elapsed(self.ended))
            (self.pending / 'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in sources))
            if not self.discard:
                # Preserve the unscaled duration and settings before encoding,
                # so a failed export can be recovered without scaling twice.
                (self.pending / 'source_metadata.json').write_text(json.dumps({
                    'wall_duration_s': self.ended-self.started, **self.timeline_metadata(), 'config': asdict(self.config),
                    'started_at': self.started_at, 'recording_layouts': layouts,
                    'metadata': self.metadata,
                }, indent=2)+'\n')
                for name in names:
                    listing = self.pending / f'{name}.concat.txt'
                    lines = []
                    for index, row in enumerate(sources):
                        end = sources[index+1]['recorded_elapsed_s'] if index+1 < len(sources) else duration
                        lines += ["file '"+row['files'][name]+"'", 'duration '+str(max(.001, end-row['recorded_elapsed_s'])/self.config.playback_speed)]
                    lines += ["file '"+sources[-1]['files'][name]+"'"]
                    listing.write_text('\n'.join(lines)+'\n')
                    path = self.pending / f'{name}.mp4'
                    started = time.perf_counter()
                    with path.with_suffix('.ffmpeg.log').open('wb') as log:
                        process = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                            '-progress', 'pipe:1', '-nostats',
                            '-f', 'concat', '-safe', '0', '-i', str(listing), '-vf', f'fps={self.config.video_fps}',
                            '-t', str(max(1/self.config.video_fps, duration/self.config.playback_speed)),
                            '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', str(self.config.video_crf),
                            '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
                            stdout=subprocess.PIPE, stderr=log, text=True)
                    if self.performance:
                        self.performance.add('video_encode_wall_time', time.perf_counter()-started)
                    if process.returncode or not path.exists() or not path.stat().st_size:
                        raise RuntimeError('Offline recording export failed; see encoder log.')
                    counts = [int(line.split('=', 1)[1]) for line in process.stdout.splitlines() if line.startswith('frame=')]
                    if not counts or counts[-1] < 1:
                        raise RuntimeError('Offline recording export produced no video frames.')
                    frames = counts[-1]
                    videos.append({'view': name, 'filename': path.name, 'frames': frames,
                        'size_bytes': path.stat().st_size, 'duration_s': frames/self.config.video_fps,
                        'url': f'/recordings/{self.recording_id}/{path.name}', **layouts[name]})
                # Keep every input until all selected views have exported.
                # A failed view leaves the pending sources recoverable.
                if len(videos) != len(names) or not videos:
                    raise RuntimeError('Cannot remove sources before all recording views are exported.')
                shutil.rmtree(directory)
                for name in names:
                    (self.pending / f'{name}.concat.txt').unlink()
                (self.pending / 'source_metadata.json').unlink()
            record = {'id': self.recording_id, 'status': 'saved', 'started_at': self.started_at,
                'wall_duration_s': self.ended-self.started, **self.timeline_metadata(), 'initial_snapshot_id': self.initial_snapshot,
                'final_snapshot_id': sources[-1]['snapshot_id'], 'config': asdict(self.config),
                'videos': videos, 'metadata': self.metadata, 'recording_layouts': layouts,
                'source_snapshot_count': len(sources),
                'source_frames_retained': False,
                'source_retention_policy': 'remove_after_all_selected_videos_export_successfully',
                'dropped_snapshots': self.dropped_snapshots, 'timing_file': 'samples.jsonl',
                'frame_policy': 'unique captured snapshots with synchronized timestamps; offline constant-FPS export'}
            with self.lock:
                if self.discard:
                    shutil.rmtree(self.pending)
                else:
                    (self.pending/'metadata.json').write_text(json.dumps(record, indent=2)+'\n')
                    destination = self.root/self.recording_id
                    self.pending.rename(destination)
                    self.last_recording = dict(record, path=str(destination))
                    self.saved += 1
        except Exception as exc:
            error = f'Snapshot recording failed: {exc}'
        finally:
            with self.lock:
                self.status = 'error' if error else 'idle'
                self.error, self.recording_id = error, None

    def finish(self, *, discard=False):
        with self.lock:
            if self.status != 'recording':
                return {'ok': False, 'status': 409, 'message': self.error or 'No recording is active.'}
            self.discard = discard
            self.ended = time.monotonic()
            self.status = 'finalizing'
            worker = self.worker
            self.finish_event.set()
        worker.join()
        with self.lock:
            if self.error:
                return {'ok': False, 'status': 500, 'message': self.error}
            return {'ok': True, 'message': 'Recording discarded.' if discard else 'All camera videos saved.',
                    'recording': None if discard else self.last_recording}

    def _record(self):
        writers = {}
        frames = self.initial_frames
        source_names = tuple(frames)
        layouts = recording_layouts(frames, self.config, self.metadata.get('camera_catalog', ()))
        interval = self.config.playback_speed / self.config.video_fps
        samples = 0
        seen_snapshots = set()
        snapshot_id = self.initial_snapshot
        timing = None
        error = None
        try:
            timing = (self.pending / 'samples.jsonl').open('w')
            writers = {}
            for name, layout in layouts.items():
                writers[name] = CompactVideoWriter(self.pending / f'{name}.mp4', self.config,
                                                  frame_size=tuple(layout['size']))
            while True:
                outputs = recording_frames(frames, layouts, self.config)
                for name, writer in writers.items():
                    writer.append(outputs[name])
                samples += 1
                seen_snapshots.add(snapshot_id)
                timing.write(json.dumps({'sample': samples, 'scheduled_recorded_elapsed_s': (samples-1)*interval,
                                          'snapshot_id': snapshot_id})+'\n')
                while not self.finish_event.is_set() and self.recorded_elapsed(time.monotonic()) < samples*interval:
                    self.finish_event.wait(.05)
                if self.finish_event.is_set():
                    break
                with self.lock:
                    frames = {name: self.latest.get(name, frames[name]) for name in source_names}
                    snapshot_id = self.latest_snapshot
            # Fill missed wall-clock ticks if encoding was delayed. At most one
            # output frame of quantization; even immediate STOP has a valid MP4.
            needed = max(1, math.ceil(self.recorded_elapsed(self.ended)/interval))
            with self.lock:
                frames = {name: self.latest.get(name, frames[name]) for name in source_names}
            outputs = recording_frames(frames, layouts, self.config)
            while samples < needed:
                for name, writer in writers.items():
                    writer.append(outputs[name])
                samples += 1
                seen_snapshots.add(self.latest_snapshot)
                timing.write(json.dumps({'sample': samples, 'scheduled_recorded_elapsed_s': (samples-1)*interval,
                                          'snapshot_id': self.latest_snapshot})+'\n')
        except Exception as exc:
            error = f'Video recording failed: {exc}'
        finally:
            if timing is not None:
                timing.close()
            for writer in writers.values():
                try:
                    writer.close()
                except Exception as exc:
                    error = error or str(exc)
        with self.lock:
            try:
                if self.discard:
                    shutil.rmtree(self.pending)
                    error = None
                elif error is None:
                    videos = [{'view': name, 'filename': writer.path.name, 'frames': writer.frames,
                               'size_bytes': writer.path.stat().st_size,
                               'duration_s': writer.frames/self.config.video_fps,
                               'url': f'/recordings/{self.recording_id}/{writer.path.name}', **layouts[name]}
                              for name, writer in writers.items()]
                    record = {'id': self.recording_id, 'status': 'saved', 'started_at': self.started_at,
                              'wall_duration_s': self.ended-self.started, **self.timeline_metadata(), 'initial_snapshot_id': self.initial_snapshot,
                              'final_snapshot_id': self.latest_snapshot, 'config': asdict(self.config),
                              'videos': videos, 'metadata': self.metadata, 'recording_layouts': layouts}
                    record.update(source_snapshot_count=len(seen_snapshots), timing_file='samples.jsonl',
                                  frame_policy='constant output fps; repeated last observed frame between native captures')
                    (self.pending/'metadata.json').write_text(json.dumps(record, indent=2)+'\n')
                    destination = self.root/self.recording_id
                    self.pending.rename(destination)
                    self.last_recording = dict(record, path=str(destination))
                    self.saved += 1
            except Exception as exc:
                error = f'Cannot save recording: {exc}'
            self.status = 'error' if error else 'idle'
            self.error = error
            self.recording_id = None

    def recording_file(self, path):
        target = (self.root/path).resolve()
        if not target.is_relative_to(self.root) or len(target.relative_to(self.root).parts) != 2:
            return None
        manifest = target.parent/'metadata.json'
        if not manifest.is_file():
            return None
        record = json.loads(manifest.read_text())
        allowed = {'metadata.json', 'samples.jsonl', *(video['filename'] for video in record.get('videos', []))}
        return target if record.get('status') == 'saved' and target.name in allowed and target.is_file() else None

    def close(self):
        if self.state()['recording']:
            self.finish()
        elif self.worker is not None and self.worker.is_alive():
            self.worker.join()
