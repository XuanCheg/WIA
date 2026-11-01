"""Native release smoke test, no model. Run: python scripts/smoke_release_native.py --task t01 --gpu 4 --output OUTPUT.

Run in a persistent terminal with the behavior environment. Existing GPU owners
are never displaced. The output retains device mapping, command and exit status.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from wia.resources import ROOT
from wia.tasks.loader import load_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--actions', action='store_true')
    parser.add_argument('--scorer', action='store_true', help='Begin a no-action episode and validate native score availability.')
    parser.add_argument('--timeout', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    row = next(r.split(',') for r in query.splitlines() if int(r.split(',')[0]) == args.gpu)
    if int(row[2]) >= 5 or int(row[3]) >= 512:
        raise RuntimeError('Selected GPU is occupied; no workload launched.')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalogue = json.loads((ROOT / 'explore_to_achieve/wia_v1/release/catalogue.json').read_text())
    selected = next(item for item in catalogue['tasks'] if item['alias'].lower() == args.task)
    task = load_task(ROOT / selected['task_file'], variant=selected['variant'])
    defaults = task.definition.defaults
    manifest = task.manifest('native-smoke-' + args.task)
    manifest.update(max_steps=defaults.max_steps, world_mode='pause', world_advances_during_inference=False,
        observation_condition=defaults.observation_condition, action_mode='parameterized', model_name='no_model_native_smoke',
        reasoning_effort=None, wall_timeout_s=None, agent_session='local_http_persistent', world_observation_log=True)
    (output / 'episode_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    gpu = {'execution_mode': 'local', 'CUDA_VISIBLE_DEVICES': row[1].strip(), 'uuid': row[1].strip(),
           'renderer_index': args.gpu, 'physical_index': args.gpu, 'device_uuid': row[1].strip(),
           'service': {}, 'observed_gpu_query': query}
    (output / 'gpu.json').write_text(json.dumps(gpu, indent=2) + '\n')
    cache = Path('/tmp') / ('wia-release-smoke-' + args.task + '-' + str(os.getpid()))
    (cache / 'native_tmp').mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=row[1].strip(), CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMNIGIBSON_GPU_ID='0', OMNIGIBSON_HEADLESS='1', OMNI_KIT_ACCEPT_EULA='YES',
        OMNIGIBSON_APPDATA_PATH=str(cache), XDG_CACHE_HOME=str(cache / 'xdg_cache'),
        CUDA_CACHE_PATH=str(cache / 'cuda_cache'), __GL_SHADER_DISK_CACHE_PATH=str(cache / 'nvidia_shader_cache'),
        TMPDIR=str(cache / 'native_tmp'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1')
    env['OMNIGIBSON_DATA_PATH'] = '/local_home2/zhouxinyang/data/behavior'
    env['VK_ICD_FILENAMES'] = str(ROOT / 'configs/esi_vulkan_egl.json')
    env['LD_LIBRARY_PATH'] = '/local_home2/zhouxinyang/runtime_libs' + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    command = [sys.executable, '-u', '-m', 'wia.runtime.commands.run_esi_robot', '--task-file', str(task.path),
        '--variant', task.variant, '--scene-seed', str(task.seed), '--episode-manifest', str(output / 'episode_manifest.json'),
        '--gpu-info', str(output / 'gpu.json'), '--output-dir', str(output), '--gripper-control', '--rotation-control',
        '--trunk-control', '--parameterized-actions', '--primitive-profile', 'fine', '--performance-profile', 'benchmark',
        '--preview-fps', '0', '--capture-agent-observation']
    if args.actions:
        command.append('--validate-base')
    else:
        command += ['--web', '--host', '127.0.0.1', '--port', str(9800 + args.gpu)]
    if args.scorer:
        command.append('--validate-scorer')
    receipt = {'schema': 'release_native_smoke_v1', 'task': args.task, 'gpu': gpu, 'command': command,
               'pid': os.getpid(), 'model_calls': 0, 'VK_ICD_FILENAMES': env['VK_ICD_FILENAMES'], 'LD_LIBRARY_PATH': env['LD_LIBRARY_PATH'], 'cache': str(cache), 'started': time.time()}
    (output / 'smoke.json').write_text(json.dumps(receipt, indent=2) + '\n')
    with (output / 'native.log').open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt['returncode'] = result.returncode
    receipt['wall_time_s'] = time.time() - receipt['started']
    receipt['task_status'] = json.loads((output / 'task_status.json').read_text()) if (output / 'task_status.json').exists() else None
    receipt['observation_saved'] = (output / 'agent_observation/observation.json').is_file()
    if args.scorer:
        receipt['scorer_validation'] = (json.loads((output/'scorer_validation.json').read_text())
                                        if (output/'scorer_validation.json').exists() else None)
    (output / 'smoke.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)
    return 0 if (receipt.get('returncode') == 0 and receipt['observation_saved']
                 and (not args.scorer or (receipt['scorer_validation'] or {}).get('evaluation_validity') == 'complete')) else 1


if __name__ == '__main__':
    raise SystemExit(main())
