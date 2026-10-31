"""Run bounded CPU-only geometry audits for an explicit immutable input list.

Run in a retained terminal with behavior Python and PYTHONPATH=src:
  python scripts/run_review_static_followup.py --manifest PATH
No simulator or robot controls run. Collision/IK proxies do not certify grasping.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time

from wia.resources import ROOT
from wia.runtime.scene_capture_queue import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    state_path = args.manifest.parent / 'state.json'
    if state_path.exists():
        raise FileExistsError('Preserve the previous audit state; choose a new output directory.')
    state = {'started_at': time.time(), 'pid': os.getpid(), 'jobs': [], 'robot_actions': 0}
    lock = Lock()

    def audit(job):
        row = {**job, 'state': 'starting', 'started_at': time.time()}
        folder = Path(job['output'])
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-B', str(ROOT / 'scripts/audit_interactiveworld_candidates.py'),
                   job['inventory'], str(folder), '--navigation-mode', manifest['navigation_mode']]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        with (folder / 'audit.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT)
            row.update(pid=process.pid, state='running')
            with lock:
                state['jobs'].append(row)
                write_json(state_path, state)
            code = process.wait()
        report_path = folder / 'reachability.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        with lock:
            row.update(state='complete' if code == 0 and report else 'failed', exit_code=code,
                       finished_at=time.time(), report=str(report_path), report_status=report.get('status'))
            write_json(state_path, state)

    with ThreadPoolExecutor(max_workers=manifest['concurrency']) as pool:
        list(pool.map(audit, manifest['jobs']))
    state['finished_at'] = time.time()
    write_json(state_path, state)


if __name__ == '__main__':
    main()
