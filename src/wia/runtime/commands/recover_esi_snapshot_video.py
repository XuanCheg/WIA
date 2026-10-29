#!/usr/bin/env python3
"""Offline export of complete native snapshot recordings whose encoder timed out."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--playback-speed', type=float, default=None,
                        help='Playback speed override; use stored settings, or 20x for legacy recordings.')
    args = parser.parse_args()
    run = args.run.resolve()
    if not (run/'native_exit_code.txt').exists():
        parser.error('Wait until the original native process has stopped.')
    recordings = run/'recordings'
    pending = list(recordings.glob('.*.pending'))
    if len(pending) != 1:
        parser.error('Exactly one preserved pending snapshot recording is required.')
    source = pending[0]
    samples = [json.loads(x) for x in (source/'samples.jsonl').read_text().splitlines() if x.strip()]
    source_metadata_path = source/'source_metadata.json'
    source_metadata = json.loads(source_metadata_path.read_text()) if source_metadata_path.exists() else {}
    config = source_metadata.get('config', {})
    layouts = source_metadata.get('recording_layouts', {})
    source_speed = config.get('playback_speed', 1.)  # Legacy pending recordings used 1x.
    speed = args.playback_speed if args.playback_speed is not None else config.get('playback_speed', 20.)
    if not math.isfinite(speed) or not 1 <= speed <= 60:
        parser.error('Playback speed must be finite and between 1 and 60.')
    fps, size, crf = config.get('video_fps', 30.), config.get('video_size', 512), config.get('video_crf', 26)
    destination = recordings/(source.name.removeprefix('.').removesuffix('.pending')+'_offline_export')
    if (destination/'metadata.json').exists():
        existing = json.loads((destination/'metadata.json').read_text())
        if existing.get('config', {}).get('playback_speed') != speed:
            destination = destination.with_name(destination.name+f'_{speed:g}x')
    if (destination/'metadata.json').exists():
        print('Already exported:', destination)
        return 0
    destination.mkdir(exist_ok=True)
    report_path = run/'recording_recovery.json'
    job_path = run/'recording_recovery_job.json'
    if report_path.exists():
        report_path = run/f'recording_recovery_{speed:g}x.json'
        job_path = run/f'recording_recovery_{speed:g}x_job.json'
    job = json.loads(job_path.read_text()) if job_path.exists() else {}
    job.update(pid=os.getpid(), status='running', cpu_only=True,
               source='complete_native_snapshot_recording')
    job_path.write_text(json.dumps(job, indent=2)+'\n')
    original_listing = next(source.glob('*.concat.txt'))
    durations = [float(line.split(' ', 1)[1]) for line in original_listing.read_text().splitlines()
                 if line.startswith('duration ')]
    if len(durations) != len(samples):
        raise RuntimeError('Native timing list does not cover every saved snapshot.')
    # Concat holds already include the native export's playback-speed divisor.
    # Undo that divisor before applying the requested speed to the saved timeline.
    duration = source_metadata.get('recorded_duration_s', source_metadata.get('wall_duration_s', sum(durations)*source_speed))
    output_duration = max(1/fps, duration/speed)
    videos = []
    hashes = {}
    for view in samples[0]['files']:
        lines = []
        for sample, hold in zip(samples, durations):
            frame = source/sample['files'][view]
            if not frame.is_file():
                raise RuntimeError('Missing original native snapshot: '+str(frame))
            hashes[str(frame.relative_to(run))] = hashlib.sha256(frame.read_bytes()).hexdigest()
            # Source paths are controlled recording paths; quote FFmpeg's own
            # concat grammar, without passing any content through a shell.
            lines.extend(["file '"+str(frame).replace("'", "'\\''")+"'",
                          'duration '+str(hold*source_speed/speed)])
        lines.append("file '"+str(source/samples[-1]['files'][view]).replace("'", "'\\''")+"'")
        listing = destination/(view+'.concat.txt')
        listing.write_text('\n'.join(lines)+'\n')
        output = destination/(view+'.mp4')
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
            '-progress', str(destination/(view+'.progress.log')), '-nostats',
            '-f', 'concat', '-safe', '0', '-i', str(listing), '-vf', f'fps={fps}', '-t', str(output_duration),
            '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', str(crf), '-threads', '2',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
        started = time.perf_counter()
        with (destination/(view+'.ffmpeg.log')).open('w') as log:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=log, check=True, timeout=1800)
        progress = (destination/(view+'.progress.log')).read_text().splitlines()
        frames = [int(x.split('=', 1)[1]) for x in progress if x.startswith('frame=')][-1]
        probe = imageio_ffmpeg.read_frames(str(output))
        try:
            metadata = next(probe)
        finally:
            probe.close()
        expected_size = tuple(layouts.get(view, {}).get('size', (size, size)))
        if frames < 1 or tuple(metadata['size']) != expected_size or abs(metadata['fps']-fps) > .001:
            raise RuntimeError('Recovered video has unexpected dimensions or frame rate.')
        videos.append({'view': view, 'filename': output.name, 'frames': frames, 'duration_s': frames/fps,
            'size_bytes': output.stat().st_size, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'probe': metadata, 'encoding_wall_time_s': time.perf_counter()-started, 'command': command,
            **layouts.get(view, {})})
        print('Exported',view,'frames',frames,flush=True)
    record = {'schema': 'offline_native_snapshot_export_v1', 'status': 'saved',
        'exported_at_utc': datetime.now(timezone.utc).isoformat(), 'path': str(destination),
        'original_pending_path': str(source), 'source_snapshot_count': len(samples),
        'source_frame_hashes': hashes, 'native_timing_file': str(source/'samples.jsonl'),
        'wall_duration_s': source_metadata.get('wall_duration_s', duration), 'videos': videos,
        'recorded_duration_s': duration,
        'omitted_model_wait_s': source_metadata.get('omitted_model_wait_s', 0.),
        'skip_model_wait': source_metadata.get('skip_model_wait', False),
        'config': {'video_size': size, 'video_fps': fps, 'playback_speed': speed, 'codec': 'H.264', 'crf': crf,
                   'recording_layout': config.get('recording_layout', 'individual')},
        'recording_layouts': layouts, 'metadata': source_metadata.get('metadata', {}),
        'source_playback_speed': source_speed,
        'frame_policy': 'Original native boundary captures held until the next recorded timestamp; CPU-only re-export.',
        'original_failure_preserved': True, 'no_simulator_or_model_rerun': True}
    (destination/'metadata.json').write_text(json.dumps(record, indent=2)+'\n')
    report_path.write_text(json.dumps(record, indent=2)+'\n')
    job.update(status='finished', metadata=str(destination/'metadata.json'),
               finished_at_utc=record['exported_at_utc'])
    job_path.write_text(json.dumps(job, indent=2)+'\n')
    print('Recovered recording:', destination,flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
