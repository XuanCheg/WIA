#!/usr/bin/env python3
"""Record disk measurements without changing capture workloads or cache files.

Run one sample: python src/wia/runtime/commands/monitor_esi_capture_disk.py --once
Run every half hour: python src/wia/runtime/commands/monitor_esi_capture_disk.py --interval-s 1800
The default output is /home/zhouxinyang/esi_category_conversion/20261014_v2/disk_monitor.
A lifetime advisory lock permits one monitor per output root. Only this monitor's
output files are written. There are no free-space thresholds or automatic cache,
worker, GPU, or campaign actions. SIGTERM requests a clean exit; an in-flight
read-only du helper has a 120-second timeout and may finish before exit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time


DEFAULT_OUTPUT = Path('/home/zhouxinyang/esi_category_conversion/20261014_v2/disk_monitor')
FILESYSTEM_PATHS = ('/tmp', '/home/zhouxinyang', '/local_home2/zhouxinyang')
CACHE_PATHS = tuple('/tmp/show-harness-zhouxinyang/worker_caches/worker' + str(i)
                    for i in range(8))
DU_TIMEOUT_S = 120


def _utc(value=None):
    return (value or datetime.now(timezone.utc)).isoformat(timespec='seconds')


def _text(value):
    return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else (value or '')


def _cache_allocation():
    """Measure only the eight owned cache paths in one hardlink-aware du call."""
    command = ['du', '-sk', '--', *CACHE_PATHS]
    result = {'paths': list(CACHE_PATHS), 'allocated_bytes_by_path': dict.fromkeys(CACHE_PATHS),
              'total_allocated_bytes': None, 'command': command, 'timeout_s': DU_TIMEOUT_S,
              'shared_hardlinks_counted_once': True, 'returncode': None, 'errors': []}
    started = time.monotonic()
    environment = dict(os.environ, LC_ALL='C', LANG='C')
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=DU_TIMEOUT_S, env=environment, check=False)
        output, error_text = completed.stdout, completed.stderr
        result['returncode'] = completed.returncode
        result['status'] = 'ok' if completed.returncode == 0 else 'error'
    except subprocess.TimeoutExpired as error:
        output, error_text = _text(error.stdout), _text(error.stderr)
        result['status'] = 'timeout'
        result['errors'].append('The owned read-only du helper exceeded its 120-second timeout.')
    except OSError as error:
        output = ''
        error_text = 'Cannot run the read-only du helper (errno ' + str(error.errno) + ').'
        result['status'] = 'error'
    result['duration_s'] = round(time.monotonic() - started, 3)
    result['errors'].extend(line for line in _text(error_text).splitlines() if line)
    for line in _text(output).splitlines():
        try:
            blocks, path = line.split('\t', 1)
            if path not in result['allocated_bytes_by_path'] or int(blocks) < 0:
                raise ValueError('Unexpected du output.')
            result['allocated_bytes_by_path'][path] = int(blocks) * 1024
        except (ValueError, TypeError):
            result['errors'].append('Could not parse a du measurement line.')
    values = list(result['allocated_bytes_by_path'].values())
    if all(value is not None for value in values):
        result['total_allocated_bytes'] = sum(values)
    return result


def _sample(previous, interval_s):
    previous_devices = (previous or {}).get('devices', {})
    result = {'schema': 'esi_capture_disk_sample_v1', 'timestamp_utc': _utc(),
              'host': socket.gethostname(), 'monitor_pid': os.getpid(),
              'interval_s': interval_s, 'filesystems': [], 'devices': {}, 'errors': []}
    for path in FILESYSTEM_PATHS:
        try:
            info, device = os.statvfs(path), os.stat(path).st_dev
            block_bytes = info.f_frsize or info.f_bsize
            device_key = str(os.major(device)) + ':' + str(os.minor(device))
            total, free, available = (info.f_blocks * block_bytes,
                                      info.f_bfree * block_bytes,
                                      info.f_bavail * block_bytes)
            result['filesystems'].append({
                'path': path, 'device': device_key, 'total_bytes': total,
                'free_bytes': free, 'available_bytes': available,
                'used_bytes': total - free, 'block_bytes': block_bytes})
            if device_key not in result['devices']:
                old = previous_devices.get(device_key, {}).get('available_bytes')
                result['devices'][device_key] = {
                    'paths': [], 'available_bytes': available,
                    'available_change_bytes': available - old if isinstance(old, int) else None}
            result['devices'][device_key]['paths'].append(path)
        except OSError as error:
            result['errors'].append('Cannot measure filesystem ' + path
                                    + ' (errno ' + str(error.errno) + ').')
    result['cache_allocation'] = _cache_allocation()
    result['measurement_finished_utc'] = _utc()
    return result


def _write_sample(root, sample):
    """Append history and atomically publish latest, preserving prior history."""
    with (root / 'history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(sample, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary = root / 'latest.json.tmp'
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(sample, indent=2, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, root / 'latest.json')
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    summary = {
        'timestamp_utc': sample['timestamp_utc'], 'monitor_pid': sample['monitor_pid'],
        'next_sample_utc': sample['next_sample_utc'], 'output_root': str(root),
        'available_bytes': {row['path']: row['available_bytes'] for row in sample['filesystems']},
        'available_change_bytes_by_device': {
            device: row['available_change_bytes'] for device, row in sample['devices'].items()},
        'cache_allocated_bytes': sample['cache_allocation']['total_allocated_bytes'],
        'du_status': sample['cache_allocation']['status'],
        'errors': sample['errors'] + sample['cache_allocation']['errors'],
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--interval-s', type=float, default=1800)
    parser.add_argument('--once', action='store_true', help='Record one immediate sample and exit.')
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval_s) or args.interval_s <= 0:
        parser.error('--interval-s must be a finite positive duration.')
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def request_exit(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, request_exit)
    signal.signal(signal.SIGINT, request_exit)
    with (root / 'monitor.lock').open('a+', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('A disk monitor already holds ' + str(root / 'monitor.lock') + '.', file=sys.stderr)
            return 1
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps({'monitor_pid': os.getpid(), 'started_utc': _utc(),
                               'output_root': str(root), 'interval_s': args.interval_s}) + '\n')
        lock.flush()
        os.fsync(lock.fileno())
        previous = None
        latest = root / 'latest.json'
        if latest.exists():
            try:
                previous = json.loads(latest.read_text(encoding='utf-8'))
                if not isinstance(previous, dict):
                    previous = None
            except (OSError, ValueError):
                print('The previous latest.json could not be read; history will be preserved.',
                      file=sys.stderr, flush=True)
        deadline = time.monotonic()
        while not stop.is_set():
            sample = _sample(previous, args.interval_s)
            deadline += args.interval_s
            now = time.monotonic()
            if deadline <= now:
                deadline += (math.floor((now - deadline) / args.interval_s) + 1) * args.interval_s
            remaining = max(0, deadline - now)
            sample['next_sample_utc'] = None if args.once or stop.is_set() else _utc(
                datetime.now(timezone.utc) + timedelta(seconds=remaining))
            sample['next_sample_in_s'] = None if args.once or stop.is_set() else round(remaining, 3)
            sample['output_root'] = str(root)
            sample['latest_path'] = str(latest)
            sample['history_path'] = str(root / 'history.jsonl')
            sample['lock_path'] = str(root / 'monitor.lock')
            _write_sample(root, sample)
            previous = sample
            if args.once:
                break
            stop.wait(max(0, deadline - time.monotonic()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
