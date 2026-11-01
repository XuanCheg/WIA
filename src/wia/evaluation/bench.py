"""Parallel bench supervisor. Run `python -m wia bench --config FILE --gpus 0,1,2,3`."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
import math
from urllib.request import urlopen

from .runner import run
from .telemetry import totals, format_totals


def gateway_preflight(policy):
    """Check exact public model metadata without a paid model request."""
    if policy.get('base_url', '').rstrip('/') != 'https://ai-gateway.vercel.sh/v1':
        return None
    model = policy['model']
    with urlopen('https://ai-gateway.vercel.sh/v1/models/' + model + '/endpoints', timeout=30) as response:
        metadata = json.load(response)['data']
    errors = []
    if 'image' not in metadata.get('architecture', {}).get('input_modalities', []):
        errors.append('the exact model does not declare image input required by WIA')
    provider = model.split('/', 1)[0]
    endpoints = [e for e in metadata.get('endpoints', []) if e['provider_name'] == provider]
    if not endpoints:
        available = ', '.join(e['provider_name'] for e in metadata.get('endpoints', []))
        errors.append(f'no official {provider} endpoint; available providers: {available or "none"}')
    effort = policy.get('reasoning_effort')
    supported = metadata.get('reasoning', {}).get('supported_efforts', [])
    if effort and effort not in supported:
        errors.append(f'reasoning effort {effort!r} is not declared supported')
    for endpoint in endpoints:
        maximum = endpoint.get('max_completion_tokens')
        if maximum and policy.get('max_output_tokens', 1024) > maximum:
            errors.append(f'max_output_tokens exceeds official endpoint limit {maximum}')
    if errors:
        raise ValueError('Model preflight failed: ' + '; '.join(errors) + '. No simulator or model call started.')
    return metadata


def idle_devices():
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'], text=True)
    busy = {line.split(',')[0].strip() for line in processes.splitlines() if line.strip()}
    devices = []
    for line in query.splitlines():
        index, uuid, utilization, memory = [part.strip() for part in line.split(',')]
        devices.append({'index': int(index), 'uuid': uuid,
            'idle': float(utilization) < 5 and float(memory) < 512 and uuid not in busy})
    return devices


def inventory_devices():
    """Read device identity only, without utilization, memory or process queries."""
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                    '--format=csv,noheader,nounits'], text=True)
    return [{'index': int(row.split(',')[0]), 'uuid': row.split(',')[1].strip()}
            for row in query.splitlines() if row.strip()]


def visible_devices(devices):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        return devices
    tokens = [token.strip() for token in visible.split(',') if token.strip()]
    return [d for token in tokens for d in devices
            if str(d['index']) == token or d['uuid'].startswith(token)]


def shards(prepared, gpus, port_base):
    if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise ValueError('GPU indices must be unique nonnegative integers.')
    if not 1024 <= port_base <= 65535 - len(gpus):
        raise ValueError('Invalid worker port range.')
    if not prepared['plans']:
        raise ValueError('Bench requires at least one trial.')
    output = Path(prepared['output_dir'])
    result = []
    for worker, gpu in enumerate(gpus):
        plans = deepcopy(prepared['plans'][worker::len(gpus)])
        if not plans:
            continue
        directory = output / f'worker_{worker:02d}'
        for plan in plans:
            if plan['environment'].get('mode', 'launch') != 'launch':
                raise ValueError('Parallel bench requires independently launched scenes.')
            if plan['evaluation'].get('execution_mode') == 'debug':
                raise ValueError('Bench requires batch execution, not retained debug scenes.')
            plan['environment'].update(gpu=gpu, port=port_base + worker)
            plan['output_dir'] = str(directory / f"trial_{plan['index']:04d}")
            if plan['policy'].get('backend') in ('openai_chat_completions', 'openai_responses', 'local_http'):
                plan['policy'].setdefault('history_policy', 'full')
        result.append({**deepcopy(prepared), 'output_dir': str(directory), 'plans': plans})
    return result


def progress(done, total, active, accounting):
    filled = int(24 * done / total) if total else 24
    return f"[bench {'#' * filled}{'-' * (24-filled)}] {done}/{total} active={active} " + format_totals(accounting)


def snapshot(output, workers):
    trials, records = [], []
    for worker in workers:
        path = Path(worker['output_dir']) / 'results.json'
        if path.exists():
            try:
                trials.extend(json.loads(path.read_text())['trials'])
            except json.JSONDecodeError:
                pass  # A serial worker may be replacing its progress snapshot.
    for path in output.glob('worker_*/**/api_usage.jsonl'):
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # Ignore only an incomplete trailing write; retry next refresh.
    return trials, totals(records)


def launch(prepared, *, gpus=None, worker_count=4, port_base=8800, dry_run=False,
           log_interval=30, persistent_simulator=True):
    formal = all(plan['evaluation'].get('run_mode', 'diagnostic') == 'benchmark' for plan in prepared['plans'])
    skip_gpu_check = formal or os.environ.get('WIA_SKIP_GPU_CHECK') == '1'
    if worker_count < 1:
        raise ValueError('Worker count must be positive.')
    if not math.isfinite(log_interval) or log_interval <= 0:
        raise ValueError('Log interval must be finite and positive.')
    if dry_run:
        workers = shards(prepared, gpus if gpus is not None else list(range(worker_count)), port_base)
        return {'schema': 'wia_bench_plan_v1', 'workers': workers,
                'persistent_simulator': persistent_simulator, 'log_interval_s': log_interval,
                'gpu_idle_check': not skip_gpu_check,
                'note': ('Dry run only; auto GPU indices are placeholders. Device identities resolve at execution; GPU occupancy checks are disabled.'
                         if skip_gpu_check else 'Dry run only; auto GPU indices are placeholders. Live checks occur before execution.')}
    receipts = {}
    for plan in prepared['plans']:
        if plan['evaluation'].get('run_mode', 'diagnostic') == 'benchmark':
            continue
        policy = plan['policy']
        key_env = policy.get('api_key_env')
        if key_env and not os.environ.get(key_env):
            raise ValueError(f'Required credential environment variable {key_env} is empty.')
        signature = json.dumps(policy, sort_keys=True)
        if signature not in receipts:
            receipts[signature] = gateway_preflight(policy)
    devices = inventory_devices() if skip_gpu_check else idle_devices()
    if gpus is None:
        candidates = visible_devices(devices) if skip_gpu_check else [d for d in devices if d['idle']]
        gpus = [d['index'] for d in candidates][:worker_count]
        if len(gpus) < worker_count:
            kind = 'visible' if skip_gpu_check else 'idle'
            raise ValueError(f'Need {worker_count} {kind} GPUs; found {len(gpus)}.')
    workers = shards(prepared, gpus, port_base)
    selected = [d for d in devices if d['index'] in gpus and (skip_gpu_check or d['idle'])]
    if len(selected) != len(gpus):
        if skip_gpu_check:
            raise ValueError('A selected GPU index is absent from the device inventory.')
        raise ValueError('Every selected GPU must currently be idle; existing jobs will not be displaced.')
    for worker in (() if formal else workers):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', worker['plans'][0]['environment']['port']))
    output = Path(prepared['output_dir'])
    existing = output.parent
    while not existing.exists():
        existing = existing.parent
    if not formal:
        free = shutil.disk_usage(existing).free
        print(f'[bench] disk_free_GiB={free/2**30:.1f} load_average={os.getloadavg()} '
              f'memory={Path("/proc/meminfo").read_text().split("MemAvailable:")[1].splitlines()[0].strip()}',
              file=sys.stderr, flush=True)
    output.mkdir(parents=True, exist_ok=False)
    (output/'bench_plan.json').write_text(json.dumps({'workers': workers, 'devices': selected,
        'persistent_simulator': persistent_simulator, 'log_interval_s': log_interval,
        'gpu_idle_check': not skip_gpu_check,
        'model_preflight': list(receipts.values())}, indent=2) + '\n')
    processes, logs = [], []
    from tqdm import tqdm
    from wia.runtime.lifecycle_log import read_events, format_event
    terminal = sys.stderr.isatty()
    bar = tqdm(total=len(prepared['plans']), desc='bench', unit='task', dynamic_ncols=True,
               mininterval=log_interval, disable=not terminal)
    offsets = {}
    started = time.monotonic()
    try:
        for index, worker in enumerate(workers):
            gpu = worker['plans'][0]['environment']['gpu']
            # Recheck immediately before each independent worker launch.
            current_devices = selected if skip_gpu_check else idle_devices()
            device = next((d for d in current_devices if d['index'] == gpu and (skip_gpu_check or d['idle'])), None)
            if device is None:
                raise RuntimeError(f'GPU {gpu} became busy before launch.')
            path = output/f'worker_{index:02d}_plan.json'
            path.write_text(json.dumps(worker, indent=2) + '\n')
            log = (output/f'worker_{index:02d}.log').open('w')
            logs.append(log)
            env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'CUDA_VISIBLE_DEVICES': device['uuid'],
                   'WIA_BENCH_EVENTS': str((output/f'worker_{index:02d}_events.jsonl').resolve()),
                   'WIA_BENCH_GPU': str(gpu),
                   'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
            # A run owns its services; never inherit an unrelated resident owner.
            env.pop('WIA_RESIDENT_CONTROL_DIR', None)
            env.pop('WIA_RESIDENT_TASK_PROFILES', None)
            if persistent_simulator:
                env['WIA_RESIDENT_CONTROL_DIR'] = str((output/f'resident_{index:02d}').resolve())
                env['WIA_RESIDENT_TASK_PROFILES'] = str((output/f'resident_{index:02d}_profiles.json').resolve())
            process = subprocess.Popen([sys.executable, '-m', 'wia.evaluation.bench', '--worker', str(path)],
                stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            processes.append(process)
            print(f"[bench] worker={index} pid={process.pid} physical_gpu={gpu} uuid={device['uuid']} "
                  f"log={log.name}", file=sys.stderr, flush=True)
        last_refresh = -math.inf
        while True:
            active = sum(p.poll() is None for p in processes)
            for index in range(len(workers)):
                path = output/f'worker_{index:02d}_events.jsonl'
                events, offsets[index] = read_events(path, offsets.get(index, 0))
                for event in events:
                    tqdm.write(format_event(event), file=sys.stderr)
            now = time.monotonic()
            if now - last_refresh >= log_interval or not active:
                trials, accounting = snapshot(output, workers)
                if terminal:
                    bar.n = len(trials)
                    bar.set_postfix_str(f'active={active} ' + format_totals(accounting), refresh=False)
                    bar.refresh()
                else:
                    print(progress(len(trials), len(prepared['plans']), active, accounting), file=sys.stderr, flush=True)
                last_refresh = now
            if not active:
                break
            time.sleep(1)
    except BaseException:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        raise
    finally:
        if persistent_simulator:
            from wia.runtime.resident_episode import stop_unless_retained
            for index in range(len(workers)):
                stop_unless_retained(output/f'resident_{index:02d}')
        bar.close()
        for log in logs:
            log.close()
    completed = {t['index'] for t in trials}
    for plan in prepared['plans']:
        if plan['index'] not in completed:
            trials.append({'index': plan['index'], 'task': plan['task'], 'status': 'runtime_error',
                           'error': 'Worker exited without a trial result.'})
    from .activity import aggregate_activity
    result = {'activity': aggregate_activity(trials), 'schema': 'wia_bench_results_v1', 'trials': sorted(trials, key=lambda t: t['index']),
              'request_accounting': accounting, 'elapsed_s': time.monotonic()-started,
              'worker_exit_codes': [p.returncode for p in processes]}
    (output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    prepared = json.loads(Path(args.worker).read_text())
    control = os.environ.get('WIA_RESIDENT_CONTROL_DIR')
    try:
        if control:
            from wia.runtime.resident_episode import task_profiles
            Path(os.environ['WIA_RESIDENT_TASK_PROFILES']).write_text(
                json.dumps(task_profiles(prepared['plans']), indent=2) + '\n')
        result = run(prepared)
    finally:
        if control:
            from wia.runtime.resident_episode import stop_unless_retained
            stop_unless_retained(control)
    return int(any(t['status'] not in ('completed_success', 'completed_unsuccessful') for t in result['trials']))


if __name__ == '__main__':
    raise SystemExit(main())
