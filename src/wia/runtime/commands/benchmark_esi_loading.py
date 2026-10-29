#!/usr/bin/env python3
"""Measure optimized native loading in one retained allocation.

Run: python src/wia/runtime/commands/benchmark_esi_loading.py --output-dir /absolute/OUT
     --scene-file /absolute/ordinary.yaml --scene-file /absolute/conference.yaml
Add --include-baseline to request paired baseline/cache measurements. Workers
run sequentially with frozen descriptors and seed 0. This
performs initialization qualification only, with no model or manipulation.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scene-file', type=Path, action='append', required=True)
    parser.add_argument('--include-baseline', action='store_true')
    parser.add_argument('--ordinary-wall-budget-s', type=int, default=600)
    parser.add_argument('--conference-wall-budget-s', type=int, default=1800)
    args = parser.parse_args()
    if not os.environ.get('PBS_JOBID'):
        parser.error('Run inside the retained interactive GPU allocation.')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = args.output_dir / 'inputs'
    inputs.mkdir()
    state = {'status': 'running', 'job_id': os.environ['PBS_JOBID'], 'node': socket.gethostname(),
             'started_at_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.executable,
             'seed': 0, 'runs': [], 'current': None, 'model_calls': 0, 'physical_grasp_verified': False,
             'measurement_mode': 'paired' if args.include_baseline else 'optimized_only',
             'independent_variable': 'SHOWHARNESS_ESI_MESH_CACHE; loading profiling enabled for every worker.'}
    save(args.output_dir / 'status.json', state)
    (inputs / 'code.diff').write_bytes(subprocess.check_output(['git', 'diff'], cwd=ROOT))
    (inputs / 'code_commit.txt').write_bytes(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT))
    frozen = []
    for index, path in enumerate(args.scene_file):
        contents = path.read_bytes()
        destination = inputs / f'scene_{index:02d}.yaml'
        destination.write_bytes(contents)
        frozen.append((path, destination, hashlib.sha256(contents).hexdigest()))
    save(inputs / 'scene_inputs.json', [{'source': str(path), 'snapshot': str(snapshot), 'sha256': digest}
                                      for path, snapshot, digest in frozen])
    for index, (original, snapshot, digest) in enumerate(frozen):
        for treatment in (('baseline', 'mesh_cache') if args.include_baseline else ('mesh_cache',)):
            name = f'{index:02d}_{treatment}'
            output = args.output_dir / name
            output.mkdir()
            command = [sys.executable, '-u', str(ROOT / 'src/wia/runtime/commands/diagnose_esi_rigid_scene.py'),
                       '--base-scene', str(snapshot), '--output-dir', str(output),
                       '--profile-loading', '--snapshot-physics']
            if treatment == 'mesh_cache':
                command.append('--cache-mesh-geometry')
            budget = args.conference_wall_budget_s if 'hall_conference' in str(original) else args.ordinary_wall_budget_s
            state['current'] = {'name': name, 'scene': str(original), 'started_at_utc': datetime.now(timezone.utc).isoformat()}
            save(args.output_dir / 'status.json', state)
            environment = os.environ.copy()
            environment.pop('SHOWHARNESS_ESI_LOAD_PROFILE', None)
            environment.pop('SHOWHARNESS_ESI_MESH_CACHE', None)
            began = time.perf_counter()
            with (output / 'native.log').open('w') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                state['current']['pid'] = process.pid
                save(args.output_dir / 'status.json', state)
                try:
                    code = process.wait(timeout=budget)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = -999
            row = {'name': name, 'scene': str(original), 'scene_sha256': digest, 'command': command,
                   'started_at_utc': state['current']['started_at_utc'],
                   'worker_elapsed_s': time.perf_counter() - began, 'exit_code': code,
                   'validation_complete': (output / 'validation_complete.json').is_file(), 'output': str(output)}
            save(output / 'worker_exit.json', row)
            state['runs'].append(row)
            if code or not row['validation_complete']:
                state.update(status='needs_diagnosis', current=None)
                save(args.output_dir / 'status.json', state)
                return 1
            save(args.output_dir / 'status.json', state)
    state.update(status='complete', current=None, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 source_descriptors_unchanged=all(hashlib.sha256(path.read_bytes()).hexdigest() == digest
                                                 for path, _, digest in frozen))
    save(args.output_dir / 'status.json', state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
