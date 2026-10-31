"""Load release scenes serially and execute one fixed native sequence, without a model.

Run in an existing persistent terminal:
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python scripts/check_all_scenes.py --gpu 4 --output outputs/scene_checks_RUN
Use --dry-run to inspect the ordered task list and commands without accessing GPUs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from wia.evaluation import prepare
from wia.evaluation.bench import idle_devices
from wia.resources import ROOT
from wia.runtime.resident_episode import ResidentEpisodeClient, read_service, stop_service, task_profiles
from wia.runtime.resources import execution_resource_environment
from wia.runtime.scene_capture_queue import write_json
from wia.runtime.scene_capture_resident import RendererLogMonitor
from wia.tasks.loader import load_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/bench_deepseek_60.yaml')
    parser.add_argument('--sequence', type=Path, default=ROOT/'configs/native_scene_validation.json')
    parser.add_argument('--scene-timeout', type=float, default=1200)
    parser.add_argument('--tasks', nargs='+', help='Optional catalogue aliases; default: all configured tasks in order.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    prepared = prepare(args.config)
    catalogue = json.loads((ROOT/'explore_to_achieve/wia_v1/release/catalogue.json').read_text())['tasks']
    aliases = {str((ROOT/row['task_file']).resolve()): row['alias'].lower() for row in catalogue}
    plans = [p for p in prepared['plans'] if not args.tasks or aliases[p['task']['path']] in args.tasks]
    if args.tasks and {aliases[p['task']['path']] for p in plans} != set(args.tasks):
        parser.error('One or more selected aliases are absent from the configured catalogue.')
    sequence = json.loads(args.sequence.read_text())
    if not plans or args.scene_timeout <= 0:
        parser.error('Select at least one task and a positive scene timeout.')
    preview = {'schema': 'wia_scene_sweep_plan_v1', 'gpu': args.gpu, 'model_calls': 0,
               'loading': 'serial', 'persistent_simulator': True, 'sequence': sequence,
               'tasks': [aliases[p['task']['path']] for p in plans]}
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return 0
    device = next((d for d in idle_devices() if d['index'] == args.gpu and d['idle']), None)
    if device is None:
        raise RuntimeError('Selected GPU is not idle; no existing owner will be displaced.')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sequence_path = output/'sequence.json'
    write_json(sequence_path, sequence)
    profiles_path = output/'task_profiles.json'
    write_json(profiles_path, task_profiles(plans))
    os.environ['WIA_RESIDENT_TASK_PROFILES'] = str(profiles_path)
    os.environ['WIA_BENCH_EVENTS'] = str(output/'lifecycle_events.jsonl')
    os.environ['WIA_BENCH_GPU'] = str(args.gpu)
    environment = {**execution_resource_environment(),
        'LD_LIBRARY_PATH': '/local_home2/zhouxinyang/runtime_libs:' + os.environ.get('LD_LIBRARY_PATH', ''),
        'OMNIGIBSON_DATA_PATH': os.environ.get('OMNIGIBSON_DATA_PATH', '/local_home2/zhouxinyang/data/behavior'),
        'VK_ICD_FILENAMES': str(ROOT/'configs/esi_vulkan_egl.json'),
        'OMNIGIBSON_HEADLESS': '1', 'OMNIGIBSON_GPU_ID': '0', 'OMNI_KIT_ACCEPT_EULA': 'YES'}
    gpu = {'physical_index': args.gpu, 'renderer_index': args.gpu, 'uuid': device['uuid'],
           'device_uuid': device['uuid'], 'CUDA_VISIBLE_DEVICES': device['uuid'], 'service': {}}
    write_json(output/'gpu.json', gpu)
    report = {**preview, 'pid': os.getpid(), 'gpu_mapping': gpu,
              'started_at': datetime.now(timezone.utc).isoformat(),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'results': [], 'completed': False}
    write_json(output/'results.json', report)
    controls = []
    generation = 0
    try:
        for index, plan in enumerate(plans):
            alias = aliases[plan['task']['path']]
            directory = output/f'{index:02d}_{alias}'
            directory.mkdir()
            task = load_task(plan['task']['path'], variant=plan['task']['variant'], seed=plan['task']['seed'], run_mode='diagnostic')
            manifest = task.manifest(f'fixed-scene-{index:02d}-{alias}')
            manifest.update(max_steps=task.definition.defaults.max_steps, world_mode='run',
                world_advances_during_inference=True, observation_condition='current', action_mode='parameterized',
                model_name='no_model_fixed_native_sequence', reasoning_effort=None,
                wall_timeout_s=None, agent_session='local_http_persistent', world_observation_log=True)
            write_json(directory/'episode_manifest.json', manifest)
            control = output/f'resident_{generation:03d}'
            if control not in controls:
                controls.append(control)
            client = ResidentEpisodeClient(control, gpu, environment, sys.executable)
            record = {'index': index, 'task': alias, 'task_file': str(task.path),
                      'output_dir': str(directory), 'started_at': datetime.now(timezone.utc).isoformat()}
            print(f'[sweep] {index+1}/{len(plans)} task={alias} stage=loading gpu={args.gpu}', flush=True)
            started = time.monotonic()
            try:
                argv = ['--task-file', str(task.path), '--variant', task.variant, '--scene-seed', str(task.seed),
                    '--episode-manifest', str(directory/'episode_manifest.json'), '--gpu-info', str(output/'gpu.json'),
                    '--output-dir', str(directory), '--run-mode', 'diagnostic', '--gripper-control', '--rotation-control',
                    '--trunk-control', '--parameterized-actions', '--primitive-profile', 'fine',
                    '--performance-profile', 'benchmark', '--macro-execution', 'metric_target',
                    '--observation-pipeline', 'optimized', '--preview-fps', '0', '--no-human-recording',
                    '--validation-sequence', str(sequence_path), '--capture-agent-observation']
                client.submit(argv, directory)
                monitor = RendererLogMonitor(directory/'simulator.log')
                deadline = time.monotonic() + args.scene_timeout
                response_path = control/'responses'/f'{client.request_id}.json'
                last_progress = time.monotonic()
                while not response_path.exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Native load/sequence deadline exceeded.')
                    if read_service(control) is None:
                        raise RuntimeError('Owned native service exited before its response.')
                    fatal = monitor.failure()
                    if fatal:
                        raise RuntimeError(fatal)
                    if time.monotonic()-last_progress >= 30:
                        stage = 'sequence' if (directory/'scorer_validation.json').exists() else 'loading'
                        print(f'[sweep] task={alias} stage={stage} elapsed_s={time.monotonic()-started:.0f}', flush=True)
                        last_progress = time.monotonic()
                    time.sleep(1)
                receipt = json.loads(response_path.read_text())
                record['native_receipt'] = receipt
                validation = directory/'scorer_validation.json'
                record['validation'] = json.loads(validation.read_text()) if validation.exists() else None
                check = record['validation'] or {}
                record['status'] = ('passed' if receipt.get('scene_exit_code') == 0 and check.get('completed')
                    and not check.get('errors') else 'issues_found')
                if not receipt.get('reusable'):
                    stop_service(control)
                    generation += 1
            except Exception as error:
                record.update(status='runtime_error', error=f'{type(error).__name__}: {error}')
                stop_service(control)
                generation += 1
            record['elapsed_s'] = time.monotonic()-started
            report['results'].append(record)
            write_json(output/'results.json', report)
            print(f'[sweep] {index+1}/{len(plans)} task={alias} status={record["status"]} elapsed_s={record["elapsed_s"]:.1f}', flush=True)
        report['completed'] = True
    except KeyboardInterrupt:
        report['interrupted'] = True
    finally:
        for control in controls:
            stop_service(control)
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(output/'results.json', report)
    return int(not report['completed'] or any(r['status'] != 'passed' for r in report['results']))


if __name__ == '__main__':
    raise SystemExit(main())
