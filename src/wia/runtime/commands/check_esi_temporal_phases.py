#!/usr/bin/env python3
"""Check a real two-phase scene without running a model or a motion benchmark.

Run in the behavior environment on an explicitly selected CUDA_VISIBLE_DEVICES:
  python src/wia/runtime/commands/check_esi_temporal_phases.py --scene-file SCENE --task-file TASK
    --output-dir ~/esi_category_conversion/CAMPAIGN/two_phase_20261015/native_check
This diagnostic keeps pending task status, uses actual mounted RGB observations,
and submits a known answer only to check private scorer plumbing, not reasoning.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-file', type=Path, required=True)
    parser.add_argument('--task-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    from wia.runtime.scene_capture_queue import read_json, write_json
    from wia.runtime.commands.run_esi_capture_worker import allocation_gpu
    from wia.runtime.scene_capture_adapters import gpu_evidence, install_gpu_selection, isolate_texture_cache
    gpu = allocation_gpu(0, execution_mode='local', renderer_order='pci')
    install_gpu_selection(gpu)
    from omnigibson import simulator
    isolate_texture_cache(simulator, os.environ['OMNIGIBSON_APPDATA_PATH'])
    import yaml
    from wia.tasks.loader import LoadedScene, LoadedTask, sha256, task_definition
    from wia.tasks.schema import SceneDefinition
    from wia.tasks.scene import ScenePatches
    from wia.runtime.sim.esi_omnigibson import ESISession
    from wia.runtime.sim.esi_robot_cameras import configure_robot_cameras
    from wia.native.esi_category_evaluation import CategoryEvaluationEnvelope
    from wia.native.esi_temporal_phases import TemporalChangeEvaluator, qualify_operation
    from core.record.images import save_png

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_path, task_path = args.scene_file.resolve(), args.task_file.resolve()
    definition = SceneDefinition.model_validate(yaml.safe_load(scene_path.read_text()))
    task_def = task_definition(yaml.safe_load(task_path.read_text()))
    from wia.tasks.sources import resolve_source
    source = resolve_source(definition.source.json_path)
    raw = read_json(source)
    loaded = LoadedScene(definition, scene_path, source, sha256(scene_path), sha256(source),
                         'adapted', definition.seed, raw['scene'], raw['room'])
    task = LoadedTask(task_def, task_path, source, sha256(task_path), sha256(source),
                      'adapted', definition.seed, raw['scene'], raw['room'])
    adapter = ScenePatches(loaded, definition.bindings)
    settings = adapter.observation_settings()
    settings['camera_size'] = list(definition.runtime.camera_size)
    session = ESISession(source, require_robot=True, grasp_mode='assisted', **settings)
    session.scene_adapter = adapter
    began = time.monotonic()
    result = {'complete': False, 'model_calls': 0, 'robot_actions': 0,
              'task_status': task_def.status, 'gpu': gpu, 'scene_file': str(scene_path),
              'task_file': str(task_path), 'test_kind': 'native_phase_continuity_and_private_scorer_plumbing'}
    def progress(stage):
        write_json(output / 'progress.json', {'stage': stage, 'elapsed_s': time.monotonic() - began,
                                            'pid': os.getpid()})
        print(stage, flush=True)
    try:
        progress('native_scene_loading')
        session.reset()
        result['gpu'] = gpu_evidence(session, gpu)
        configure_robot_cameras(session, ROOT / definition.runtime.robot_camera_config,
                                resolution=definition.runtime.camera_size)
        class Readback:
            def debug_gripper_state(self):
                robot = session.robot
                grasp = int(robot.is_grasping('left'))
                obj = robot._ag_obj_in_hand['left']
                return {'holding': {-1: False, 1: True}.get(grasp),
                        'object_name': obj.name if obj is not None else None,
                        'constraint_active': robot._ag_obj_constraints['left'] is not None}
        executor = Readback()
        config = task_def.private.evaluation.model_dump()
        config.update(scene=raw['scene'], room=raw['room'],
                      bindings={key: binding.name for key, binding in task_def.private.bindings.items()})
        source_observation = session.observation_source
        source_observation.task_text = task.task_text
        source_observation.include_scene = False
        progress('phase_1_native_qualification')
        result['qualification'] = qualify_operation(session, executor, config, {})
        result['interaction_setup'] = getattr(session, 'category_interaction_setup', {})
        write_json(output / 'qualification.json', result)
        first_preview = source_observation.capture(action_boundary=True).observation
        save_png(output / 'phase_1_head.png', first_preview.agentview_rgb)
        save_png(output / 'phase_1_wrist.png', first_preview.wrist_rgb)
        if not result['qualification']['accepted']:
            raise ValueError('Native Phase 1 qualification failed; inspect the stored checks.')
        native = TemporalChangeEvaluator(session, executor, config, output / 'private_native')
        envelope = CategoryEvaluationEnvelope(native, task, output / 'private_evaluation')
        session.task_evaluator = envelope
        first = source_observation.begin_episode(task.task_text)
        first.save(output / 'phase_1_observation')
        progress('phase_1_observed_read_only')
        receipt = source_observation.report_temporal('phase_1_complete', None,
            snapshot_id=first.snapshot_id, action_step=first.action_step)
        second = source_observation.agent_observation()
        second.save(output / 'phase_2_observation')
        save_png(output / 'phase_2_head.png', second.agentview_rgb)
        save_png(output / 'phase_2_wrist.png', second.wrist_rgb)
        progress('phase_2_observed_same_episode')
        answer_receipt = source_observation.report_temporal('answer', config['parameters']['original_answer'],
            snapshot_id=second.snapshot_id, action_step=second.action_step)
        sealed = read_json(output / 'private_evaluation/state.json')
        criteria = sealed['native_result']['criteria']
        plumbing = all(criteria.get(key) is True for key in (
            'native_source_state_valid', 'phase_1_observed', 'source_event_preserved_world',
            'phase_2_observed_after_phase_1', 'answer_at_current_boundary',
            'answer_is_original_public_option', 'measured_answer_matches_original_qa', 'agent_answer_correct'))
        result.update(complete=True, phase_1={'snapshot_id': first.snapshot_id, 'action_step': first.action_step},
            phase_2={'snapshot_id': second.snapshot_id, 'action_step': second.action_step},
            phase_transition_receipt=receipt, answer_receipt=answer_receipt,
            known_answer_plumbing_passed=plumbing, physical_task_success=sealed['task_success'],
            autonomous_robot_reasoning_tested=False,
            same_robot_and_furniture=session.temporal_phase_controller.transition['world_unchanged'],
            transition_physics_steps=session.temporal_phase_controller.transition['physics_steps'])
        if not plumbing:
            raise ValueError('Measured phase history or known-answer plumbing failed.')
        progress('native_continuity_check_complete')
    except Exception as error:
        result.update(complete=False, error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    finally:
        result['elapsed_s'] = time.monotonic() - began
        write_json(output / 'result.json', result)
        session.close()
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
