"""Launch one owned diagnostic and clean its native/model descendants afterwards.

Run in a persistent tmux shell with behavior Python:
  PYTHONPATH=src python scripts/launch_privileged_grasp_validation.py --case CASE --gpu 4 --output OUTPUT
Rechecks the explicitly selected idle GPU, freezes WIA Python code, retains all
logs and removes only owned temporary cache after verified process exit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

import psutil

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    case = json.loads(args.case.read_text())
    if case['start_condition'] == 'near_target' and not case.get('near_pose'):
        raise ValueError('No reviewed nearby initial pose in case.')
    if case['model'] != 'gpt-6.1-sol' or case['reasoning_effort'] != 'high':
        raise ValueError('Use the exact authorized model and effort.')
    case_path = output / 'case.json'
    write(case_path, case)
    code = output / 'code'
    shutil.copytree(ROOT / 'src/wia', code / 'src/wia', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT / 'scripts/run_privileged_grasp_validation.py', code / 'run_privileged_grasp_validation.py')
    hashes = {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*.py')}
    write(output / 'code_hashes.json', hashes)
    cache = Path('/dev/shm') / ('wia_grasp_' + str(os.getuid()) + '_' + uuid.uuid4().hex)
    if shutil.disk_usage('/dev/shm').free < 12 * 2**30 or psutil.virtual_memory().available < 24 * 2**30:
        raise RuntimeError('Insufficient verified native scratch/RAM headroom.')
    if shutil.disk_usage(output).free < 2 * 2**30:
        raise RuntimeError('Insufficient durable output space.')
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    row = next(line.split(',') for line in query.splitlines() if int(line.split(',')[0]) == args.gpu)
    gpu_uuid = row[1].strip()
    clients = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if int(row[2]) != 0 or int(row[3]) > 512 or gpu_uuid in clients:
        raise RuntimeError('Selected GPU is occupied; no existing process will be displaced.')
    (cache / 'tmp').mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID', OMNIGIBSON_GPU_ID='0',
        OMNIGIBSON_HEADLESS='1', OMNI_KIT_ACCEPT_EULA='YES', OMNIGIBSON_DATA_PATH='/local_home2/zhouxinyang/data/behavior',
        VK_ICD_FILENAMES=str(ROOT / 'configs/esi_vulkan_egl.json'), LD_LIBRARY_PATH='/local_home2/zhouxinyang/runtime_libs',
        WIA_ROOT=str(ROOT), PYTHONPATH=str(code / 'src') + ':' + str(ROOT.parent / 'Show-Harness-Explore'),
        OMNIGIBSON_APPDATA_PATH=str(cache), TMPDIR=str(cache / 'tmp'), TMP=str(cache / 'tmp'), TEMP=str(cache / 'tmp'),
        XDG_CACHE_HOME=str(cache / 'xdg'), CUDA_CACHE_PATH=str(cache / 'cuda'), __GL_SHADER_DISK_CACHE_PATH=str(cache / 'gl'),
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
        TORCHINDUCTOR_COMPILE_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-u', str(code / 'run_privileged_grasp_validation.py'), '--case', str(case_path), '--output', str(output)]
    receipt = dict(case_file=str(args.case.resolve()), gpu=args.gpu, gpu_uuid=gpu_uuid, started_at=time.time(),
                   command=command, cache=str(cache), state='starting', model=case['model'], reasoning_effort=case['reasoning_effort'])
    owned = {}
    interrupted = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *unused: interrupted.__setitem__(0, True))
    with (output / 'native.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt.update(pid=process.pid, state='running')
        write(output / 'launch.json', receipt)
        try:
            while process.poll() is None:
                parent = psutil.Process(process.pid)
                for child in [parent, *parent.children(recursive=True)]:
                    try:
                        owned[child.pid] = child.create_time()
                    except psutil.NoSuchProcess:
                        pass
                if interrupted[0] or time.time() - receipt['started_at'] >= case['process_timeout_s']:
                    receipt['stop_reason'] = 'interrupted' if interrupted[0] else 'process_wall_budget'
                    break
                time.sleep(2)
        finally:
            def living():
                result = []
                for pid, created in owned.items():
                    try:
                        child = psutil.Process(pid)
                        if child.create_time() == created and child.status() != psutil.STATUS_ZOMBIE:
                            result.append(child)
                    except psutil.NoSuchProcess:
                        pass
                return result
            live = living()
            for child in live:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, remaining = psutil.wait_procs(live, timeout=10)
            for child in remaining:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(remaining, timeout=5)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            receipt.update(returncode=process.poll(), wall_time_s=time.time() - receipt['started_at'],
                           owned_processes=[dict(pid=p, creation_time=t) for p,t in owned.items()],
                           remaining_owned_pids=[p.pid for p in living()])
            if not receipt['remaining_owned_pids']:
                shutil.rmtree(cache)
                receipt['temporary_cache_removed'] = True
            else:
                receipt['temporary_cache_removed'] = False
            summary = output / 'summary.json'
            result = json.loads(summary.read_text()) if summary.exists() else {}
            receipt.update(state='finished', diagnostic_state=result.get('state', 'initialization_failed'),
                           diagnostic_success=result.get('success', False))
            write(output / 'launch.json', receipt)
    print(json.dumps({k:receipt.get(k) for k in ('gpu','pid','state','returncode','diagnostic_state','diagnostic_success','remaining_owned_pids','temporary_cache_removed')}), flush=True)


if __name__ == '__main__':
    main()
