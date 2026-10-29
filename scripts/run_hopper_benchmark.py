"""Run a bounded four-worker campaign through the existing WIA/Harness runner.

Prepare: python scripts/run_hopper_benchmark.py prepare --output-dir OUTPUT --deadline ISO_TIME
Worker: python scripts/run_hopper_benchmark.py worker --output-dir OUTPUT --gpu 0
Status: python scripts/run_hopper_benchmark.py status --output-dir OUTPUT
Workers claim distinct tasks transactionally. Only physical GPUs 0-3 are allowed.
The global deadline interrupts unfinished trials; it never changes task budgets.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

import yaml

from wia.evaluation.runner import prepare

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT.parent / 'Show-Harness-Explore'
SNAPSHOT = ROOT / 'explore_to_achieve/wia_v1/release/catalogue.json'


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def database(out):
    return sqlite3.connect(out / 'queue.sqlite', timeout=30)


def prepare_campaign(out, deadline, policy_config=None):
    if deadline is None:
        raise ValueError('Provide the explicit serving-window deadline with --deadline.')
    parsed_deadline = datetime.fromisoformat(deadline)
    if parsed_deadline.tzinfo is None or parsed_deadline.timestamp() <= time.time():
        raise ValueError('The deadline must be a future timezone-aware ISO timestamp.')
    out.mkdir(parents=True, exist_ok=True)
    policy_path = policy_config or ROOT / 'configs/hopper_qwen38_27b.yaml'
    policy = yaml.safe_load(policy_path.read_text())['policy']
    catalogue = json.loads((ROOT / 'explore_to_achieve/wia_v1/release/catalogue.json').read_text())
    buckets = defaultdict(deque)
    excluded = list(catalogue['excluded'])
    for row in catalogue['tasks']:
        selection = {'path': str(ROOT / row['task_file']), 'variant': row['variant']}
        buckets[row['family']].append({'alias': row['alias'], 'family': row['family'],
            'variant_id': row['variant_id'], 'selection': selection})
    ordered = []
    families = ('AP', 'NT', 'GF', 'SR', 'GG', 'TM')
    while any(buckets.values()):
        for family in families:
            if buckets[family]:
                ordered.append(buckets[family].popleft())
    connection = database(out)
    connection.execute('CREATE TABLE jobs (id INTEGER PRIMARY KEY, alias TEXT, family TEXT, '
                       'config TEXT, status TEXT, gpu INTEGER, pid INTEGER, started REAL, '
                       'finished REAL, exit_code INTEGER, note TEXT)')
    valid = []
    for entry in ordered:
        index = len(valid)
        folder = out / f'job_{index:03d}_{entry["alias"].lower()}'
        folder.mkdir()
        config = {'tasks': [entry['selection']], 'policy': policy,
                  'environment': {'mode': 'launch', 'python': sys.executable,
                                  'record': True, 'preview_fps': 0},
                  'evaluation': {'run_mode': 'diagnostic'},
                  'output_dir': str(folder / 'run')}
        path = folder / 'config.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        try:
            frozen = prepare(path)
        except Exception as exc:
            excluded.append({**entry, 'reason': f'{type(exc).__name__}: {exc}'})
            continue
        write_json(folder / 'prepared_plan.json', frozen)
        connection.execute('INSERT INTO jobs (id,alias,family,config,status) VALUES (?,?,?,?,?)',
                           (index, entry['alias'], entry['family'], str(path), 'queued'))
        valid.append({**entry, 'index': index, 'config': str(path)})
    connection.commit()
    manifest = {'schema': 'wia_hopper_stress_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'deadline_utc': parsed_deadline.astimezone(timezone.utc).isoformat(), 'physical_gpus': [0, 1, 2, 3],
        'selection_source': str(SNAPSHOT), 'curation_source': str(ROOT / 'docs/research/world_interactive_agency_v1/review_evidence.json'),
        'selection_policy': 'One current preferred presentation per proposal; round-robin families; no historical-version duplication.',
        'run_mode': 'diagnostic', 'model_policy': policy,
        'policy_config': str(policy_path),
        'scientific_settings': 'Task defaults retained; only global campaign deadline can censor a trial.',
        'recording': 'Native synchronized recording enabled; continuous human preview disabled.',
        'jobs': valid, 'excluded': excluded}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({'queued': len(valid), 'excluded': excluded, 'deadline_utc': manifest['deadline_utc']}), flush=True)


def claim(out, gpu):
    with database(out) as connection:
        connection.execute('BEGIN IMMEDIATE')
        row = connection.execute("SELECT id,alias,config FROM jobs WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
        if row:
            connection.execute("UPDATE jobs SET status='running',gpu=?,pid=?,started=? WHERE id=?",
                               (gpu, os.getpid(), time.time(), row[0]))
        return row


def gpu_snapshot():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used',
                                     '--format=csv,noheader,nounits'], text=True)
    return {int(parts[0]): {'uuid': parts[1].strip(), 'utilization': int(parts[2]), 'memory_mib': int(parts[3])}
            for line in result.splitlines() if (parts := line.split(','))}


def stop_owned_simulators(folder, gpu):
    for runtime in folder.glob('run/trial_*/local_runtime.json'):
        mapping = json.loads(runtime.read_text())
        pid_file = runtime.parent / 'sim_pid.txt'
        if mapping.get('physical_index') != gpu or not pid_file.exists():
            continue
        pid = int(pid_file.read_text())
        if mapping.get('persistent_simulator') and (runtime.parent / 'native_exit_code.txt').exists():
            continue
        proc = Path(f'/proc/{pid}')
        if not proc.exists():
            continue
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if proc.stat().st_uid == os.getuid() and 'run_esi_robot.py' in command and str(runtime) in command:
            os.kill(pid, signal.SIGINT)


def clear_preloaded_scene(folder, gpu):
    """Close an attached world's web loop, retaining its resident application."""
    path = folder / 'preloaded_scene.json'
    if not path.exists():
        return
    from wia.runtime.resident_episode import read_service
    value = json.loads(path.read_text())
    if value['gpu'] != gpu:
        raise RuntimeError('Preloaded scene GPU ownership differs from this worker.')
    control = Path(value['control_dir'])
    service = read_service(control)
    if service is None or service['pid'] != value['native_pid']:
        raise RuntimeError('Preloaded native service identity changed.')
    response = control / 'responses' / (value['request_id'] + '.json')
    if not response.exists():
        if service.get('request_id') != value['request_id']:
            raise RuntimeError('Refusing to clear a different resident request.')
        write_json(control / 'stop_scene.json', {'request_id': value['request_id']})
        expires = time.monotonic() + 180
        while not response.exists():
            if read_service(control) is None or time.monotonic() >= expires:
                raise RuntimeError('Preloaded scene did not clear; resident process was preserved.')
            time.sleep(.5)
    receipt = json.loads(response.read_text())
    if not receipt.get('reusable'):
        raise RuntimeError('Preloaded scene cleanup did not retain a reusable native service.')
    write_json(folder / 'preloaded_cleanup.json', receipt)


class AdoptedProcess:
    """Monitor an existing owned trial; infer completion from its durable result."""
    def __init__(self, pid, config, start_ticks):
        self.pid, self.results, self.returncode = pid, config.parent / 'run/results.json', None
        self.start_ticks = start_ticks
        proc = Path(f'/proc/{pid}')
        fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if (proc.stat().st_uid != os.getuid() or fields[19] != start_ticks
                or 'wia eval --config' not in command or str(config) not in command):
            raise RuntimeError('Adopted trial ownership or process identity does not match.')

    def poll(self):
        try:
            fields = Path(f'/proc/{self.pid}/stat').read_text().split(') ', 1)[1].split()
            if fields[0] != 'Z' and fields[19] == self.start_ticks:
                return None
        except FileNotFoundError:
            pass
        self.returncode = 0 if self.results.exists() else 1
        return self.returncode

    def wait(self, timeout=None):
        began = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic() - began >= timeout:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(.2)
        return self.returncode


def worker(out, gpu, adopt_pid=None, adopt_job=None, adopt_start_ticks=None):
    if gpu not in (0, 1, 2, 3):
        raise ValueError('Only explicitly authorized physical GPUs 0-3 may be used.')
    manifest = json.loads((out / 'manifest.json').read_text())
    deadline = datetime.fromisoformat(manifest['deadline_utc']).timestamp()
    while time.time() < deadline - 120:
        observed = gpu_snapshot()[gpu]
        write_json(out / f'gpu_{gpu}_latest.json', {'at': time.time(), **observed})
        owned_resident = False
        resident_control = os.environ.get('WIA_RESIDENT_CONTROL_DIR')
        if resident_control:
            from wia.runtime.resident_episode import owns_gpu
            owned_resident = owns_gpu(resident_control, gpu, observed['uuid'])
        if adopt_pid is None and (observed['memory_mib'] >= 512 or observed['utilization'] >= 5) and not owned_resident:
            print(f'GPU {gpu} is not idle; waiting without launching.', flush=True)
            time.sleep(10)
            continue
        if adopt_pid is not None:
            with database(out) as connection:
                row = connection.execute("SELECT id,alias,config FROM jobs WHERE id=? AND gpu=? AND status='running'",
                                         (adopt_job, gpu)).fetchone()
                if row is None:
                    raise RuntimeError('The adopted job is not running on the selected GPU.')
                connection.execute('UPDATE jobs SET pid=? WHERE id=?', (os.getpid(), adopt_job))
        else:
            row = claim(out, gpu)
        if row is None:
            break
        index, alias, config_path = row
        config_path = Path(config_path)
        folder = config_path.parent
        config = yaml.safe_load(config_path.read_text())
        if adopt_pid is None:
            config['environment'].update(gpu=gpu, port=8700 + gpu * 10)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            write_json(folder / 'prepared_plan.json', prepare(config_path))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=observed['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   NUMEXPR_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
                   OMNIGIBSON_DATA_PATH='/local_home2/zhouxinyang/data/behavior',
                   VK_ICD_FILENAMES=str(ROOT / 'configs/esi_vulkan_egl.json'),
                   LD_LIBRARY_PATH='/local_home2/zhouxinyang/runtime_libs' +
                       (':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''))
        print(f'Starting {alias} on physical GPU {gpu}, UUID {observed["uuid"]}', flush=True)
        if adopt_pid is None:
            write_json(folder / 'worker_launch.json', {'physical_gpu': gpu, 'cuda_visible_devices': observed['uuid'],
                       'started_utc': datetime.now(timezone.utc).isoformat(), 'worker_pid': os.getpid()})
        note = None
        with (folder / 'driver.log').open('a' if adopt_pid is not None else 'w') as stream:
            if adopt_pid is not None:
                process = AdoptedProcess(adopt_pid, config_path, adopt_start_ticks)
                write_json(folder / 'worker_adoption.json', {'trial_pid': adopt_pid, 'worker_pid': os.getpid(),
                    'at_utc': datetime.now(timezone.utc).isoformat(), 'deadline_utc': manifest['deadline_utc'],
                    'exit_status_source': 'Inferred from process exit and durable WIA results; not waitpid.'})
                adopt_pid = None
            else:
                process = subprocess.Popen([sys.executable, '-u', '-m', 'wia', 'eval', '--config', str(config_path)],
                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            started = time.time()
            while process.poll() is None:
                if time.time() >= deadline:
                    note = 'global_deadline_censored'
                elif (config['environment'].get('mode') != 'attach'
                      and time.time() - started > 1200 and not list(folder.glob('run/trial_*/metadata.json'))):
                    note = 'native_initialization_watchdog_1200s'
                if note:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=150)
                    except subprocess.TimeoutExpired:
                        stop_owned_simulators(folder, gpu)
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    break
                time.sleep(3)
        stop_owned_simulators(folder, gpu)
        clear_preloaded_scene(folder, gpu)
        results_path = folder / 'run/results.json'
        statuses = []
        if results_path.exists():
            results = json.loads(results_path.read_text())
            statuses = [trial['status'] for trial in results.get('trials', [])]
        state = 'censored' if note == 'global_deadline_censored' else 'finished' if process.returncode == 0 else 'error'
        note = note or ','.join(statuses)
        with database(out) as connection:
            connection.execute('UPDATE jobs SET status=?,finished=?,exit_code=?,note=? WHERE id=?',
                               (state, time.time(), process.returncode, note, index))
        print(f'Finished {alias}: {state}, exit={process.returncode}, {note}', flush=True)
        status(out)
        time.sleep(3)
    status(out)
    resident_control = os.environ.get('WIA_RESIDENT_CONTROL_DIR')
    if resident_control:
        write_json(Path(resident_control) / 'worker_completed.json', {
            'reason': 'Campaign ended; the reusable simulator remains running.', 'at': time.time()})
    print(f'GPU {gpu} worker completed.', flush=True)


def status(out):
    with database(out) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute('SELECT * FROM jobs ORDER BY id')]
    data = {'at_utc': datetime.now(timezone.utc).isoformat(),
            'counts': dict(Counter(row['status'] for row in rows)), 'jobs': rows}
    # Each worker owns its own snapshot; the shared database remains authoritative.
    write_json(out / f'status_{os.getpid()}.json', data)
    print(json.dumps(data), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'worker', 'status'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--deadline', help='Explicit timezone-aware serving-window deadline for preparation.')
    parser.add_argument('--policy-config', type=Path, help='Explicit policy profile; task selection still follows current curation.')
    parser.add_argument('--adopt-pid', type=int, help='Existing paused WIA trial to adopt after a window extension.')
    parser.add_argument('--adopt-job', type=int)
    parser.add_argument('--adopt-start-ticks')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.command == 'prepare':
        prepare_campaign(out, args.deadline, args.policy_config)
    elif args.command == 'worker':
        worker(out, args.gpu, args.adopt_pid, args.adopt_job, args.adopt_start_ticks)
    else:
        status(out)


if __name__ == '__main__':
    main()
