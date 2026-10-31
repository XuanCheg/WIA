"""Retain one Kit application while serving complete native episode worlds.

Run through WIA's resident episode client, never beside an existing GPU owner.
Each request invokes the original native runner; its HTTP server, controllers,
sensors and evaluator close before the existing whole-stage discard lifecycle.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-dir', type=Path, required=True)
    parser.add_argument('--task-profiles', type=Path, required=True)
    args = parser.parse_args()
    root = WIA_RUNTIME_ROOT
    sys.path.insert(0, str(root))
    control = args.control_dir.resolve()
    profiles = json.loads(args.task_profiles.read_text())
    import wia.runtime.scene_capture_adapters as adapters
    from wia.runtime.sim.esi_omnigibson import ESISession, _import_pipeline
    from wia.runtime.commands.run_esi_capture_service import install_lifecycle, native_log, reset_discarded_sensor_graphs, discard_whole_stage
    import wia.runtime.commands.run_esi_robot as run_esi_robot
    pipeline = _import_pipeline(WIA_ESI_ROOT)
    from omnigibson.macros import gm
    keys = ('USE_GPU_DYNAMICS', 'ENABLE_FLATCACHE', 'ENABLE_OBJECT_STATES', 'ENABLE_TRANSITION_RULES', 'USE_ENCRYPTED_ASSETS')
    defaults = {key: bool(gm[key]) for key in keys}
    # Import macro-setting task modules before the first native application.
    with gm.unlocked():
        for module in sorted({row['native_module'] for row in profiles.values()}):
            importlib.import_module(module)
    def reset_macros(profile):
        values = dict(defaults, USE_GPU_DYNAMICS=profile['gpu_dynamics'],
                      **{key: value for key, value in profile.get('macro_overrides', {}).items()
                         if key != 'USE_GPU_DYNAMICS'})
        changed = any(bool(getattr(gm, key)) != value for key, value in values.items())
        with gm.unlocked():
            for key, value in values.items():
                setattr(gm, key, value)
        if changed and pipeline.og.app is not None:
            if pipeline.og.sim.scenes:
                raise RuntimeError('Cannot change native physics profile while a world is loaded.')
            from omni.syntheticdata import SyntheticData
            reset_discarded_sensor_graphs(pipeline.og.sim, SyntheticData.Get())
            with discard_whole_stage(pipeline.og):
                pipeline.og.clear()
        return values
    state = {'reusable': True, 'application_identity': None, 'scene_clear_s': 0.}
    def reset_graphs():
        from omni.syntheticdata import SyntheticData
        reset_discarded_sensor_graphs(pipeline.og.sim, SyntheticData.Get())
    install_lifecycle(adapters, ESISession, state, before_stage_discard=reset_graphs)
    active = {'request_id': None}
    stopping = threading.Event()
    def watcher():
        seen = set()
        while not stopping.wait(.2):
            path = control / 'stop_scene.json'
            if path.exists():
                try:
                    value = json.loads(path.read_text())
                except (ValueError, FileNotFoundError):
                    continue
                request_id = value.get('request_id')
                if request_id == active['request_id'] and request_id not in seen:
                    seen.add(request_id)
                    os.kill(os.getpid(), signal.SIGINT)
    thread = threading.Thread(target=watcher, daemon=True)
    thread.start()
    completed = {path.name for path in (control / 'responses').glob('*.json')}
    count = 0
    atomic_json(control / 'service.json', {'state': 'IDLE', 'pid': os.getpid(), 'application_identity': None})
    try:
        while not (control / 'stop_service.json').exists():
            for path in sorted((control / 'requests').glob('*.json')):
                if path.name in completed:
                    continue
                request = json.loads(path.read_text())
                completed.add(path.name)
                argv = request['native_argv']
                output = Path(request['output_dir'])
                task_path = str(Path(argv[argv.index('--task-file') + 1]).resolve())
                profile = profiles[task_path]
                active['request_id'] = request['request_id']
                before = state['application_identity']
                state.update(reusable=True, scene_clear_s=0., output=str(output), request_started_at=time.monotonic())
                response = {'request_id': request['request_id'], 'native_pid': os.getpid(), 'request_index': count,
                            'application_identity_before': before, 'world_rebuilt': True}
                atomic_json(control / 'service.json', {**response, 'state': 'SCENE_RUNNING', 'pid': os.getpid()})
                code = 1
                with native_log(output / 'simulator.log'):
                    try:
                        response['native_macro_profile'] = reset_macros(profile)
                        if request.get('camera_sync_diagnostic'):
                            from wia.runtime.sim.esi_camera_sync_diagnostic import run
                            code = run(argv, output)
                        else:
                            code = run_esi_robot.main(argv)
                        if (output / 'initialization_failure.json').exists():
                            raise RuntimeError('Failed native initialization cannot be reused.')
                        app = pipeline.og.app
                        after = id(app) if app is not None else None
                        if before is not None and after != before:
                            raise RuntimeError('The persistent Kit application identity changed.')
                        if app is not None and pipeline.og.sim.scenes:
                            raise RuntimeError('Native world was not cleared after the episode.')
                        response.update(reusable=state['reusable'], application_identity_after=after,
                                        scene_clear_s=state['scene_clear_s'], engine_reused=count > 0)
                    except BaseException as error:
                        response.update(reusable=False, error=f'{type(error).__name__}: {error}')
                        traceback.print_exc()
                    finally:
                        active['request_id'] = None
                        response['scene_exit_code'] = code
                        atomic_json(output / 'persistent_scene_receipt.json', response)
                        (output / 'native_exit_code.txt').write_text(str(code) + '\n')
                atomic_json(control / 'responses' / path.name, response)
                gc.collect()
                import torch
                torch.cuda.empty_cache()
                if not response.get('reusable'):
                    atomic_json(control / 'service.json', {'state': 'FAILED', 'pid': os.getpid(),
                                                          'error': response.get('error')})
                    return 1
                count += 1
                atomic_json(control / 'service.json', {'state': 'IDLE', 'pid': os.getpid(),
                    'application_identity': response['application_identity_after'], 'completed_scenes': count})
            time.sleep(.2)
    finally:
        stopping.set()
        thread.join(timeout=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
