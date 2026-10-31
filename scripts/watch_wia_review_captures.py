"""Refill authorized extra GPUs for an existing, explicitly selected screenshot refresh.

Run in a retained codex-gpu terminal with behavior Python:
  python scripts/watch_wia_review_captures.py --refresh-manifest PATH --output PATH --gpus 6 7
The existing queue and its retry budgets are retained. Explicitly selected GPUs
receive one native capture per worker and exit afterwards. Existing processes
are retained; the queue authorization policy independently limits new claims.
--allow-shared-gpu explicitly permits non-capture clients only while fresh
measured headroom meets --shared-gpu-min-free-mib (default 14336 MiB).
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import launch_wia_review_captures as launch
from wia.runtime.scene_capture_queue import CaptureQueue, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', choices=range(8), default=[6, 7])
    parser.add_argument('--allow-shared-gpu', action='store_true')
    parser.add_argument('--shared-gpu-min-free-mib', type=launch.positive_free_mib,
                        default=launch.DEFAULT_SHARED_GPU_MIN_FREE_MIB)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    refresh = json.loads(args.refresh_manifest.read_text())
    queue = CaptureQueue(refresh['queue_root'])
    launched, errors = [], []
    while True:
        refresh = json.loads(args.refresh_manifest.read_text())
        if Path(refresh['queue_root']).resolve() != queue.root.resolve():
            raise ValueError('The watched manifest must retain its original queue root.')
        rows = []
        for alias, job in refresh['new_jobs'].items():
            status = queue.status(job['job_id'])
            maximum = queue.manifest(job['job_id'])['max_attempts']
            if status['state'] == 'FAILED' and status.get('retryable') and status['attempts'] < maximum:
                queue.retry(job['job_id'])
                status = queue.status(job['job_id'])
            terminal = status['state'] in {'REVIEW_PENDING', 'CACHED', 'STALE', 'CANCELLED', 'FAILED'}
            rows.append({'alias': alias, 'job_id': job['job_id'], 'state': status['state'],
                         'attempts': status['attempts'], 'terminal': terminal})
        all_terminal = all(row['terminal'] for row in rows)
        live_extra = [worker for worker in launched if
            (status := json.loads(Path(worker['status_file']).read_text()) if Path(worker['status_file']).exists() else {})
            .get('state') not in {'STOPPED', 'ERROR'} and launch._process_ticks(worker.get('pid')) is not None]
        state = 'complete' if all_terminal and not live_extra else 'draining_extra_workers' if all_terminal else 'waiting_or_capturing'
        write_json(args.output / 'status.json', {'pid': os.getpid(), 'updated_at': time.time(),
            'state': state, 'jobs': rows, 'extra_workers': launched, 'recent_errors': errors[-5:],
            'refresh_manifest': str(args.refresh_manifest.resolve()), 'allowed_physical_gpus': args.gpus,
            'allow_shared_gpu': args.allow_shared_gpu, 'shared_gpu_min_free_mib': args.shared_gpu_min_free_mib,
            'gpu_authorization_mode': 'shared_with_headroom' if args.allow_shared_gpu else 'exclusive_idle',
            'model_calls': 0, 'robot_actions': 0})
        if state == 'complete':
            return
        if not all_terminal and any(row['state'] == 'QUEUED' for row in rows):
            try:
                devices = launch.gpu_snapshot(args.gpus)
                pending = sum(row['state'] == 'QUEUED' for row in rows)
                available = [gpu for gpu, device in devices.items() if not launch.device_errors(device,
                    allow_shared_gpu=args.allow_shared_gpu, shared_gpu_min_free_mib=args.shared_gpu_min_free_mib)[0]][:pending]
                if available:
                    plan = launch.build_plan(available, allow_shared_gpu=args.allow_shared_gpu,
                        shared_gpu_min_free_mib=args.shared_gpu_min_free_mib)
                    if plan['ready']:
                        result = launch.apply_plan(plan, allow_shared_gpu=args.allow_shared_gpu,
                            shared_gpu_min_free_mib=args.shared_gpu_min_free_mib)
                        launched.extend({'gpu': worker['gpu']['physical_index'], 'worker_id': worker['worker_id'],
                            'status_file': worker['queue_worker_status_file'], 'pid': worker.get('worker_script_pid'),
                            'manifest': result['manifest_file']} for worker in result['workers'])
            except (OSError, ValueError, RuntimeError) as error:
                errors.append({'time': time.time(), 'error': f'{type(error).__name__}: {error}'})
        time.sleep(30)


if __name__ == '__main__':
    main()
