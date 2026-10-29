"""Measure native depth refresh without model calls or extra physics steps.

Run: WIA_DEPTH_PROBE_OUTPUT=/path/to/output python scripts/probe_depth_readback.py
     NATIVE_RUN_ESI_ROBOT_ARGS (including --port and --episode-manifest).
The output manifest supplies the control lease identity. This diagnostic replays
three public commands and retains numerical comparisons plus selected arrays.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import requests

from wia.runtime.commands.run_esi_robot import main
from wia.runtime.sim.esi_observations import ESIObservationSource, _render
from wia.tasks.loader import load_task


OUTPUT = Path(os.environ['WIA_DEPTH_PROBE_OUTPUT'])
PORT = sys.argv[sys.argv.index('--port') + 1]
ORIGINAL_CAPTURE = ESIObservationSource._capture
DEPTH_NAMES = ('head_depth_m', 'wrist_depth_m')
ROWS = []


def difference(first, last):
    finite = np.isfinite(first) & np.isfinite(last)
    error = np.abs(first[finite].astype(np.float64) - last[finite])
    return {
        'pixels': int(first.size), 'joint_finite_pixels': int(finite.sum()),
        'finite_mask_changed': int(np.count_nonzero(np.isfinite(first) != np.isfinite(last))),
        'changed_finite_pixels': int(np.count_nonzero(error)),
        'absolute_error_m': dict(zip(('median', 'p95', 'p99', 'max'),
            np.quantile(error, [.5, .95, .99, 1.]).tolist())) if error.size else None,
        'pixels_above_m': {str(limit): int(np.count_nonzero(error > limit))
                          for limit in (1e-6, 1e-4, 1e-3, .01, .1)},
    }


def read_depths(session):
    raw = session.observe(include_wrist=True, include_sensors=True)
    return {name: np.array(raw[name], copy=True) for name in DEPTH_NAMES}


def capture(self, *, action_boundary=False, action_result=None):
    snapshot = ORIGINAL_CAPTURE(self, action_boundary=action_boundary, action_result=action_result)
    if not action_boundary or not action_result:
        return snapshot
    session = self.session
    generation = session._physical_step_generation
    physical_time = session.pipeline.og.sim.current_time
    started = time.perf_counter()
    frames = [{name: np.array(getattr(snapshot.observation.sensors, name), copy=True)
               for name in DEPTH_NAMES}, read_depths(session)]
    for _ in range(6):
        _render(session)
        frames.append(read_depths(session))
    ROWS.append({
        'action': action_result['action'], 'snapshot_id': snapshot.observation.snapshot_id,
        'physical_generation_before': generation,
        'physical_generation_after': session._physical_step_generation,
        'physical_time_before': physical_time, 'physical_time_after': session.pipeline.og.sim.current_time,
        'block_on_render_during_capture': session.pipeline.og.sim.get_block_on_render(),
        'probe_wall_time_s': time.perf_counter() - started,
        'normal_to_final': {name: difference(frames[0][name], frames[-1][name]) for name in DEPTH_NAMES},
        'readbacks': [{'extra_renders': max(0, index - 1),
            'after_normal_capture': index > 0,
            'to_final': {name: difference(frame[name], frames[-1][name]) for name in DEPTH_NAMES}}
            for index, frame in enumerate(frames)],
        'camera_pose_receipt': getattr(session, '_native_camera_freshness', None),
    })
    np.savez_compressed(OUTPUT / f'boundary_{len(ROWS):02d}.npz',
        **{f'{stage}_{name}': frame[name]
           for stage, frame in (('normal', frames[0]), ('final', frames[-1])) for name in DEPTH_NAMES})
    (OUTPUT / 'depth_readback.json').write_text(json.dumps({
        'model_calls': 0, 'extra_physics_steps': 0, 'boundaries': ROWS,
        'protocol': 'T01 adapted seed 0; public commands replayed without model delays; '
                    'compare normal capture, immediate readback and six graphics-only updates.'}, indent=2) + '\n')
    return snapshot


def control():
    token = None
    url = f'http://127.0.0.1:{PORT}'
    try:
        deadline = time.monotonic() + 500
        while True:
            try:
                requests.get(url + '/api/agent/state', timeout=3).raise_for_status()
                break
            except requests.RequestException:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Native controller did not become ready.')
                time.sleep(1)
        manifest = json.loads((OUTPUT / 'episode_manifest.json').read_text())
        task = load_task(manifest['task_file'], variant=manifest['scene_variant'], run_mode='diagnostic')
        response = requests.post(url + '/api/control/claim', json={
            'run_id': manifest['episode_id'],
            'task_text': task.task_text,
            'world_advances_during_inference': True, 'action_mode': 'parameterized'}, timeout=30)
        response.raise_for_status()
        token = response.json()['control_token']
        receipts = []
        for command in ({'action': 'BASE_BACK', 'distance_m': .1},
                        {'action': 'MV_FWD', 'distance_m': .08}, {'action': 'RELEASE'}):
            response = requests.post(url + '/api/agent/command',
                json={'control_token': token, 'command': command}, timeout=150)
            response.raise_for_status()
            receipts.append({'command': command, 'response': response.json()})
            (OUTPUT / 'control_receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
    except Exception as error:
        (OUTPUT / 'control_error.json').write_text(json.dumps({
            'error': str(error), 'type': type(error).__name__}, indent=2) + '\n')
    finally:
        if token:
            try:
                requests.post(url + '/api/control/release', json={'control_token': token}, timeout=15)
            except requests.RequestException:
                pass


if __name__ == '__main__':
    ESIObservationSource._capture = capture
    threading.Thread(target=control, daemon=True).start()
    raise SystemExit(main(sys.argv[1:]))
