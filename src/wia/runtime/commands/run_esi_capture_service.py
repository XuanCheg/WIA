#!/usr/bin/env python3
"""Serve immutable screenshot requests while retaining one native Kit engine.

Run only through run_esi_capture_worker.py --persistent-simulator. The worker
pins this lifecycle adapter separately from each request's frozen capture code.
Every request rebuilds its world; no previous physical episode is continued.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import faulthandler
import gc
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


@contextmanager
def native_log(path):
    """Route Python and native console output to the current durable attempt."""
    sys.stdout.flush()
    sys.stderr.flush()
    previous = [os.dup(1), os.dup(2)]
    try:
        with Path(path).open('ab', buffering=0) as destination:
            os.dup2(destination.fileno(), 1)
            os.dup2(destination.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for descriptor, saved in zip((1, 2), previous):
            os.dup2(saved, descriptor)
            os.close(saved)


@contextmanager
def native_stage_cleanup(og):
    """Use complete SDK cleanup with the PR #2114 fix for OmniGibson 3.7.1.

    The temporary stop callback runs inside the original _partial_clear, after
    its existing stop callbacks and before scene removal. No second stop, scene
    method replacement or premature SyntheticData reset is performed. Other SDK
    versions retain their native implementation.
    """
    simulator = og.sim
    missing = object()
    previous = vars(simulator).get('_partial_clear', missing)
    patched = getattr(og, '__version__', None) == '3.7.1'
    unsubscribed = []
    if patched:
        original = simulator._partial_clear
        handles = ('_pre_physics_step_callback', '_post_physics_step_callback', '_simulation_event_callback')
        # Verify the exact API before attempting cleanup. Unsupported teardown
        # must fail visibly so its owner retires the process.
        subscriptions = [(name, getattr(simulator, name)) for name in handles]
        if not all(callable(getattr(handle, 'unsubscribe', None)) for _, handle in subscriptions):
            raise RuntimeError('OmniGibson 3.7.1 PhysX subscriptions do not support the official clear fix.')

        def partial_clear():
            callback_name = 'wia_physx_clear_' + str(id(simulator))

            def unsubscribe():
                for name, handle in subscriptions:
                    if name not in unsubscribed:
                        handle.unsubscribe()
                        unsubscribed.append(name)

            simulator.add_callback_on_stop(callback_name, unsubscribe)
            try:
                return original()
            finally:
                simulator.remove_callback_on_stop(callback_name)

        simulator._partial_clear = partial_clear
    print('[native-stage-cleanup] ' + json.dumps({'stage': 'started', 'scene_count': len(simulator.scenes),
        'policy': 'official_full_cleanup', 'physx_subscription_fix': patched}), flush=True)
    try:
        yield
    except BaseException as error:
        print('[native-stage-cleanup] ' + json.dumps({'stage': 'failed',
            'error': f'{type(error).__name__}: {error}', 'unsubscribed': unsubscribed}), flush=True)
        raise
    else:
        print('[native-stage-cleanup] ' + json.dumps({'stage': 'completed',
            'unsubscribed': unsubscribed}), flush=True)
    finally:
        if patched:
            if previous is missing:
                del simulator._partial_clear
            else:
                simulator._partial_clear = previous


def load_asset_runtime(path, hash_file):
    """Load the separately pinned CPU validator and native preparation adapter."""
    value = json.loads(Path(path).read_text())
    names = {'src/wia/runtime/scene_capture_assets.py': 'wia.runtime.scene_capture_assets',
             'src/wia/runtime/sim/esi_prepared_assets.py': 'wia.runtime.sim.esi_prepared_assets'}
    if value.get('schema') != 'native_asset_preparation_runtime_v1' or set(value.get('modules', {})) != set(names):
        raise ValueError('Asset preparation runtime must pin both declared helpers.')
    for relative, module_name in names.items():
        row = value['modules'][relative]
        if hash_file(row['path']) != row['sha256']:
            raise ValueError('Pinned asset preparation helper changed: ' + relative)
        spec = importlib.util.spec_from_file_location(module_name, row['path'])
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return value, sys.modules['wia.runtime.sim.esi_prepared_assets'].prepared_asset_loading


def install_lifecycle(adapters, session_class, state):
    """Replace process shutdown only; retain the frozen scene/camera code."""
    original_selection = adapters.install_gpu_selection
    original_texture = adapters.isolate_texture_cache
    selected, isolated = False, False

    def selection(gpu):
        nonlocal selected
        if not selected:
            original_selection(gpu)
            selected = True

    def texture(simulator, appdata):
        nonlocal isolated
        if not isolated:
            original_texture(simulator, appdata)
            isolated = True

    def close(session):
        if session.closed:
            return
        session.closed = True
        pipeline = session.pipeline
        if pipeline is None or pipeline.og.app is None:
            return
        if sys.exc_info()[0] is not None:
            # The frozen capture code has already attempted partial evidence.
            # A partially loaded native world is disposed by its owning worker.
            state['reusable'] = False
            return
        og = pipeline.og
        application, simulator = og.app, og.sim
        began = time.monotonic()
        if state.get('output'):
            adapters.write_json(Path(state['output']) / 'progress.json', {'stage': 'scene_cleanup',
                'elapsed_s': time.monotonic() - state['request_started_at']})
        state['reusable'] = False
        with native_stage_cleanup(og):
            if session.task_module is not None and session.env is not None:
                pipeline.task_cleanup_runtime(session.task_module, session.env, session.payload, session.task_state)
            # Some original cleanup hooks already clear the stage.
            if og.sim is simulator:
                og.clear()
        if og.app is not application or og.sim is None or og.sim.scenes:
            raise RuntimeError('Native scene cleanup did not preserve an empty, live application.')
        state['scene_clear_s'] = time.monotonic() - began
        state['application_identity'] = id(application)
        state['reusable'] = True

    adapters.install_gpu_selection = selection
    adapters.isolate_texture_cache = texture
    session_class.close = close


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--control-dir', type=Path, required=True)
    parser.add_argument('--gpu-info', type=Path, required=True)
    parser.add_argument('--runtime-key', required=True)
    parser.add_argument('--asset-runtime', type=Path,
                        help='Separately pinned optional CPU-prepared asset consumption helpers.')
    args = parser.parse_args()
    code = args.code_root.resolve()
    sys.path.insert(0, str(code))
    sys.dont_write_bytecode = True
    from wia.runtime.scene_capture_queue import complete_images, read_json, sha256, verify_snapshot, write_json
    import wia.runtime.scene_capture_adapters as adapters
    from wia.runtime.sim.esi_omnigibson import ESISession

    if not Path(adapters.__file__).resolve().is_relative_to(code):
        raise RuntimeError('Resident capture imported code outside the frozen revision.')
    control = args.control_dir.resolve()
    gpu = read_json(args.gpu_info)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    lifecycle = {'reusable': True, 'application_identity': None, 'scene_clear_s': 0.}

    install_lifecycle(adapters, ESISession, lifecycle)
    asset_runtime, asset_context = (load_asset_runtime(args.asset_runtime, sha256)
                                    if args.asset_runtime else (None, None))
    write_json(control / 'service.json', {'state': 'READY', 'pid': os.getpid(),
        'runtime_key': args.runtime_key, 'code_root': str(code), 'asset_runtime': asset_runtime,
        'timestamp': time.time()})
    handled, count = set(), 0
    while not (control / 'stop.json').exists():
        for request_path in sorted((control / 'requests').glob('*.json')):
            if request_path.name in handled:
                continue
            handled.add(request_path.name)
            request = read_json(request_path)
            output = Path(request['output'])
            output.mkdir(parents=True, exist_ok=True)
            response = {'request_id': request['request_id'], 'job_id': request['job_id'],
                'native_pid': os.getpid(), 'runtime_key': args.runtime_key, 'reusable': False}
            manifest = {}
            with native_log(output / 'native.log'):
                try:
                    manifest = read_json(request['manifest'])
                    verify_snapshot(manifest)
                    if (request['runtime_key'] != args.runtime_key
                            or manifest['job_id'] != request['job_id']
                            or Path(manifest['code_root']).resolve() != code):
                        raise ValueError('Resident request does not match its pinned runtime or immutable job.')
                    import random
                    import numpy as np
                    import torch
                    random.seed(manifest['seed'])
                    np.random.seed(manifest['seed'])
                    torch.manual_seed(manifest['seed'])
                    torch.cuda.manual_seed_all(manifest['seed'])
                    lifecycle.update(reusable=True, scene_clear_s=0., output=str(output),
                                     request_started_at=time.monotonic())
                    before_application = lifecycle['application_identity']
                    preparation = None
                    if asset_runtime:
                        from wia.runtime.scene_capture_queue import digest
                        from wia.runtime.scene_capture_resident import runtime_identity
                        expected_key = digest({'native_runtime': runtime_identity(manifest),
                            'asset_preparation_runtime': {name: row['sha256']
                                for name, row in asset_runtime['modules'].items()}})
                        if expected_key != args.runtime_key:
                            raise ValueError('Prepared asset runtime does not match the immutable native request.')
                        preload = read_json(output / 'cpu_preload.json', {})
                        if preload.get('manifest_sha256') == sha256(request['manifest']):
                            preparation = preload.get('prepared_assets')
                        if preparation is None:
                            write_json(output / 'native_asset_preparation.json', {'enabled': False,
                                'warnings': ['CPU asset preparation unavailable; original native loader retained.']})
                    with asset_context(preparation, output) if preparation is not None else nullcontext():
                        adapters.capture_snapshot(request['manifest'], output, gpu)
                    gc.collect()
                    torch.cuda.empty_cache()
                    receipt_path = output / 'preview_complete.json'
                    receipt = read_json(receipt_path)
                    if not lifecycle['reusable'] or receipt is None:
                        raise RuntimeError('Native capture did not finish with a reusable application.')
                    reused = count > 0
                    if reused and before_application != lifecycle['application_identity']:
                        raise RuntimeError('A resident request changed its native application.')
                    initial = receipt['timings']['cold_initialization_s']
                    receipt['timings'].update(scene_initialization_s=initial,
                        cold_initialization_s=0. if reused else initial,
                        scene_clear_s=lifecycle['scene_clear_s'])
                    receipt['persistent_runtime'] = {'native_pid': os.getpid(), 'request_index': count,
                        'engine_reused': reused, 'world_rebuilt': True,
                        'application_identity': lifecycle['application_identity'],
                        'lifecycle_file': str(Path(__file__).resolve()),
                        'lifecycle_sha256': sha256(Path(__file__)), 'runtime_key': args.runtime_key,
                        'seed_reset': manifest['seed'], 'frozen_code_root': str(code)}
                    if asset_runtime:
                        receipt['native_asset_preparation'] = read_json(output / 'native_asset_preparation.json', {})
                        receipt['asset_preparation_runtime'] = asset_runtime
                    receipt['provenance'] += ' Native application retained; world rebuilt by the declared lifecycle adapter.'
                    write_json(receipt_path, receipt)
                    count += 1
                    response.update(state='CAPTURED', reusable=True, receipt=str(receipt_path))
                except BaseException as error:
                    response.update(state='FAILED', error=f'{type(error).__name__}: {error}')
                    write_json(output / 'failure.json', {'error': str(error), 'error_type': type(error).__name__,
                        'failure_stage': read_json(output / 'progress.json', {}).get('stage', 'resident_capture'),
                        'retryable': type(error).__name__ == 'NativeCameraNotReady', 'complete': False})
                    traceback.print_exc()
                    receipt = read_json(output / 'preview_complete.json', {})
                    if (manifest.get('diagnostic_pending') and complete_images(manifest, receipt)):
                        # Valid screenshots survive a later native teardown failure.
                        # Dispose this engine, retaining the failure and warning.
                        receipt['native_cleanup_warning'] = response['error']
                        receipt['provenance'] += ' Native teardown failed; engine will restart. ' + response['error']
                        write_json(output / 'preview_complete.json', receipt)
                        response.update(state='CAPTURED', reusable=False)
            write_json(control / 'responses' / request_path.name, response)
            if not response['reusable']:
                return 1
        time.sleep(.2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
