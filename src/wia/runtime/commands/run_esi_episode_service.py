"""Retain one Kit application while serving complete native episode worlds.

Run through WIA's resident episode client, never beside an existing GPU owner.
Each request invokes the original native runner. Completed worlds restore their
initial native state and remain paused on the GPU; a different world request
releases the old world through the official full scene cleanup lifecycle.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
import gc
import faulthandler
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
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    profiles = json.loads(args.task_profiles.read_text())
    import wia.runtime.scene_capture_adapters as adapters
    from wia.runtime.sim.esi_omnigibson import ESISession, _import_pipeline
    from wia.runtime.resident_episode import native_macro_settings
    from wia.runtime.commands.run_esi_capture_service import install_lifecycle, native_log
    import wia.runtime.commands.run_esi_robot as run_esi_robot
    pipeline = _import_pipeline(WIA_ESI_ROOT)
    pipeline.og._wia_resident_episode = True
    from omnigibson.macros import gm
    keys = ('USE_GPU_DYNAMICS', 'ENABLE_FLATCACHE', 'ENABLE_OBJECT_STATES', 'ENABLE_TRANSITION_RULES', 'USE_ENCRYPTED_ASSETS')
    defaults = {key: bool(gm[key]) for key in keys}
    # Import macro-setting task modules before the first native application.
    with gm.unlocked():
        for module in sorted({row['native_module'] for row in profiles.values()}):
            importlib.import_module(module)
    def reset_macros(profile):
        values = native_macro_settings(defaults, profile)
        changed = any(bool(getattr(gm, key)) != value for key, value in values.items())
        if changed and pipeline.og.app is not None:
            raise RuntimeError('Native macro profile changes require a fresh Kit process; client must recycle its owner.')
        with gm.unlocked():
            for key, value in values.items():
                setattr(gm, key, value)
        return values
    state = {'reusable': True, 'application_identity': None, 'scene_clear_s': 0.}
    install_lifecycle(adapters, ESISession, state)
    from wia.runtime.loaded_scene import LoadedScene
    loaded_scene = LoadedScene(ESISession.close)
    ESISession.close = lambda session: loaded_scene.finish(session)
    run_esi_robot.resident_scene = loaded_scene
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
    atomic_json(control / 'service.json', {'state': 'IDLE', 'pid': os.getpid(), 'application_identity': None,
                                         'macro_defaults': defaults})
    try:
        while not (control / 'stop_service.json').exists():
            for check in sorted((control / 'retained_checks').glob('*.json')):
                destination = control / 'retained_responses' / check.name
                if destination.exists():
                    continue
                destination.parent.mkdir(exist_ok=True)
                prior = json.loads((control / 'service.json').read_text())
                try:
                    request = json.loads(check.read_text())
                    if not loaded_scene.retained or not callable(loaded_scene.validate):
                        raise ValueError('This resident world does not support repeated fixed-sequence validation.')
                    active['request_id'] = check.stem
                    atomic_json(control / 'service.json', {**prior, 'state': 'SCENE_RUNNING', 'request_id': check.stem})
                    directory = Path(request['output_dir'])
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    # The callback creates a fresh result directory before native logging.
                    session = loaded_scene.session
                    previous_prechecks = session._wia_prechecks_enabled
                    try:
                        session._wia_prechecks_enabled = request.get('prechecks_enabled', previous_prechecks)
                        report = loaded_scene.validate(directory, request['sequence'])
                    finally:
                        session._wia_prechecks_enabled = previous_prechecks
                    atomic_json(destination, {'completed': report['completed'], 'errors': len(report['errors']),
                        'native_pid': os.getpid(), 'world_rebuilt': False, 'output_dir': str(directory)})
                except Exception as error:
                    atomic_json(destination, {'completed': False, 'error': f'{type(error).__name__}: {error}'})
                finally:
                    active['request_id'] = None
                    atomic_json(control / 'service.json', prior)
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
                            'application_identity_before': before, 'world_rebuilt': True, 'output_dir': str(output)}
                atomic_json(control / 'service.json', {**response, 'state': 'SCENE_RUNNING', 'pid': os.getpid()})
                code = 1
                with native_log(output / 'simulator.log'):
                    try:
                        loaded_scene.release()
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
                        if app is not None and pipeline.og.sim.scenes and not loaded_scene.retained:
                            raise RuntimeError('Native world was not cleared after the episode.')
                        state['application_identity'] = after
                        response.update(reusable=state['reusable'], application_identity_after=after,
                                        scene_clear_s=state['scene_clear_s'], engine_reused=count > 0,
                                        world_retained=loaded_scene.retained,
                                        physics_paused=loaded_scene.retained,
                                        initial_native_state_restored=loaded_scene.retained)
                    except BaseException as error:
                        response.update(reusable=False, error=f'{type(error).__name__}: {error}')
                        traceback.print_exc()
                    finally:
                        active['request_id'] = None
                        response['scene_exit_code'] = code
                        atomic_json(output / 'persistent_scene_receipt.json', response)
                        (output / 'native_exit_code.txt').write_text(str(code) + '\n')
                gc.collect()
                import torch
                torch.cuda.empty_cache()
                if not response.get('reusable'):
                    atomic_json(control / 'service.json', {'state': 'FAILED', 'pid': os.getpid(),
                                                          'error': response.get('error')})
                    atomic_json(control / 'responses' / path.name, response)
                    return 1
                count += 1
                atomic_json(control / 'service.json', {'state': 'SCENE_RETAINED' if loaded_scene.retained else 'IDLE', 'pid': os.getpid(),
                    'application_identity': response['application_identity_after'], 'completed_scenes': count,
                    'retained_output_dir': str(output) if loaded_scene.retained else None,
                    'retained_validation': loaded_scene.retained and callable(loaded_scene.validate),
                    'retained_prechecks_switch': True,
                    'macro_defaults': defaults, 'native_macro_profile': response['native_macro_profile']})
                atomic_json(control / 'responses' / path.name, response)
            time.sleep(.2)
    finally:
        stopping.set()
        thread.join()
        loaded_scene.release()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
