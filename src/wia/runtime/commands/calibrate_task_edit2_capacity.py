#!/usr/bin/env python3
"""Calibrate exact joint-inventory pairs in one disposable native diagnostic.

Launch only with an explicitly selected idle GPU and private appdata, e.g.:
 CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_APPDATA_PATH=/tmp/capacity_calibration_ab_ac \
 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B \
 src/wia/runtime/commands/calibrate_task_edit2_capacity.py --task-file TASK --scene-file SCENE \
 --inventory-receipt RECEIPT --pairs A:B,A:C --output-dir OUTPUT

Omit --pairs to measure missing pairs. At 0.3 m height the unchanged 0.01 m/s
source lift alone needs about 63 simulated seconds per pair, plus source settle
steps; wall time must be measured natively. No Agent/model/evaluator is created.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def _receipts(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'inventory_receipt':
                yield item
            else:
                yield from _receipts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _receipts(item)


def validate_inputs(task_path, scene_path, receipt_path):
    """Validate actual task/scene embedded inventory before importing native OG."""
    import yaml
    from wia.authoring.converters.liquid_capacity_inventory import SOURCE_PROTOCOL, validate_truth
    paths = [Path(path).resolve(strict=True) for path in (task_path, scene_path, receipt_path)]
    task, scene = [yaml.safe_load(path.read_text()) for path in paths[:2]]
    receipt = json.loads(paths[2].read_text())
    ids = [item['id'] for item in receipt['inventory']]
    if ids != list('ABCDEF') or len({item['name'] for item in receipt['inventory']}) != 6:
        raise ValueError('This driver requires the explicit six-container A..F task inventory.')
    immutable = ('schema', 'inventory_id', 'inventory', 'source_mappings', 'source_protocol_sha256', 'scene_model', 'room')
    for definition in (task, scene):
        matches = list(_receipts(definition))
        if not matches or any(any(entry.get(key) != receipt.get(key) for key in immutable) for entry in matches):
            raise ValueError('Task/scene/receipt inventories do not match exactly.')
    if hashlib.sha256(SOURCE_PROTOCOL.read_bytes()).hexdigest() != receipt['source_protocol_sha256']:
        raise ValueError('The original source protocol does not match the inventory receipt.')
    truth = validate_truth(receipt)
    if truth['invalid_evidence'] or truth.get('reason') not in (None, 'incomplete_capacity_comparison_graph'):
        raise ValueError('The input receipt already contains invalid or inconsistent capacity evidence.')
    if task.get('source') != scene.get('source'):
        raise ValueError('Task and scene do not bind the same exact original source.')
    source = WIA_ESI_ROOT / 'dataset/json_clean' / scene['source']['json']
    mappings = receipt['source_mappings']
    if str(source.relative_to(WIA_ESI_ROOT / 'dataset/json_clean')) not in {row['source_identity'] for row in mappings}:
        raise ValueError('The loaded original source is not an inventory member.')
    for row in mappings:
        path = WIA_ESI_ROOT / 'dataset/json_clean' / row['source_identity']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['source_sha256']:
            raise ValueError('An original source inventory member changed.')
    return paths, task, scene, receipt, source



class _OriginalIsolatedSceneAdapter:
    """Script-private source-protocol diagnostic; never a benchmark scene variant."""

    def __init__(self, loaded_scene, inventory_receipt):
        self.task = loaded_scene  # Existing driver provenance interface.
        self.receipt = deepcopy(inventory_receipt)
        self.applied = []
        self._realized = ()
        self._before_applied = self._after_applied = False
        self.source_env_config = None

    def observation_settings(self):
        return {}

    def apply_to_env_config(self, config):
        if self._before_applied:
            raise RuntimeError('Original isolated inventory configuration is initialization-only.')
        if config.get('robots') != []:
            raise ValueError('Original source isolation requires its unchanged robot-free configuration.')
        if config.get('scene', {}).get('scene_model') != self.receipt['scene_model']:
            raise ValueError('Original isolated source scene and validated inventory disagree.')
        self.source_env_config = deepcopy(config)
        objects = config.get('objects')
        if not isinstance(objects, list):
            raise ValueError('The original liquid source must provide its object list.')
        source_pair = {row['name']: row for row in objects}
        if set(source_pair) != {'obj1', 'obj2'} or len(objects) != 2:
            raise ValueError('Original isolated loading requires exactly the original obj1/obj2 constructors.')
        template = deepcopy(source_pair['obj1'])
        for index, item in enumerate(self.receipt['inventory']):
            current = source_pair.get(item['name'])
            if current is not None:
                if any(current.get(key) != item[key] for key in ('category', 'model')):
                    raise ValueError('Validated inventory conflicts with the original source pair.')
            else:
                current = {**deepcopy(template), 'name': item['name'],
                    'category': item['category'], 'model': item['model'],
                    # Fixed source-sized bodies are separated and staged above
                    # the source scene; no table/support or collision override.
                    'position': [float(index), 0., float(template['position'][2]) + 2.]}
                objects.append(current)
            current.update(scale=deepcopy(item['scale']), fixed_base=True, visual_only=False)
        self._before_applied = True

    def source_postprocess_override(self, session):
        # Returning None deliberately retains liquid_volume.postprocess_env for
        # obj1/obj2, source pool construction, water and initial joint behavior.
        return None

    def apply_after_postprocessing(self, session):
        if not self._before_applied or self._after_applied:
            raise RuntimeError('Original isolated cleanup must follow original source postprocessing once.')
        from wia.authoring.converters.liquid_capacity_inventory import INITIALIZATION
        from wia.authoring.converters.physical_structure_liquid_volume import initialize_scene
        evidence = initialize_scene(session, {'operation': INITIALIZATION,
            'inventory_receipt': deepcopy(self.receipt)})
        self.applied.append({'operation': 'original_source_pool_cleanup',
            'initialization_evidence': deepcopy(evidence)})
        self._after_applied = True

    def diagnostic_context(self, session=None):
        original = self.source_env_config
        context = {'mode': 'original_isolated_scene_source_protocol_diagnostic',
            'benchmark_world_mode': False,
            'source_scene_configuration': deepcopy(original.get('scene')) if original else None,
            'source_robot_configuration': deepcopy(original.get('robots')) if original else None,
            'original_source_postprocessor_retained': True,
            'category_initialization_or_interaction_patches_applied': False,
            'support_table_or_label_patches_applied': False,
            'mounted_camera_capture_requested': False,
            'validated_inventory': deepcopy(self.receipt['inventory']),
            'extra_fixed_inventory_staging': 'source obj1 orientation and staging Z + 2 m; distinct integer X',
            'native_inventory': [], 'native_measurement_available': False}
        if session is not None and getattr(session, 'env', None) is not None:
            context['robot_absent'] = not session.env.robots and getattr(session, 'robot', None) is None
            for item in self.receipt['inventory']:
                obj = session.env.scene.object_registry('name', item['name'])
                if obj is None:
                    raise ValueError('Isolated native diagnostic inventory is incomplete.')
                position, orientation = obj.get_position_orientation()
                plain = lambda value: value.detach().cpu().tolist() if hasattr(value, 'detach') else value.tolist() if hasattr(value, 'tolist') else list(value)
                context['native_inventory'].append({'name': obj.name, 'category': obj.category,
                    'model': obj.model, 'scale': plain(obj.scale), 'fixed_base': bool(obj.fixed_base),
                    'visual_only': bool(obj.visual_only), 'position': plain(position),
                    'quaternion_xyzw': plain(orientation)})
            context['native_measurement_available'] = True
        return context

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-file', type=Path, required=True)
    parser.add_argument('--scene-file', type=Path, required=True)
    parser.add_argument('--inventory-receipt', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pairs', help='Ordered source-left:source-right pairs, e.g. A:B,A:C; default missing pairs.')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--original-isolated-scene', action='store_true',
        help='Explicit diagnostic isolation: retain original source scene filters and robot-free physics; '
             'omit benchmark support/labels/Agent/camera patches.')
    parser.add_argument('--disable-render-on-step', action='store_true',
        help='Opt in only when installed native scope/step preserve all original physics and nonphysics updates.')
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    if not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
        parser.error('An explicit CUDA_VISIBLE_DEVICES selection is required; the driver does not choose idle GPUs.')
    if not os.environ.get('OMNIGIBSON_APPDATA_PATH', '').strip():
        parser.error('A private OMNIGIBSON_APPDATA_PATH is required to preserve other native processes.')
    if not 1 <= args.threads <= 4:
        parser.error('--threads must be between 1 and 4.')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    from wia.runtime.sim.esi_capacity_calibration import NativeCapacityCalibration, ProgressJournal, select_pairs
    journal = ProgressJournal(args.output_dir)
    session = None
    adapter = None
    result = None
    failure = None
    try:
        paths, task_raw, scene_raw, receipt, source = validate_inputs(args.task_file, args.scene_file, args.inventory_receipt)
        selected = select_pairs(receipt, args.pairs)
        launch_metadata = {'task_file': str(paths[0]), 'scene_file': str(paths[1]),
            'inventory_receipt': str(paths[2]), 'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'selected_pairs': selected, 'diagnostic_only': True, 'model_calls': 0, 'agent_operation_count': 0,
            'source_lift_speed_m_per_s': .01, 'estimated_lift_sim_seconds_at_0_3m': 63.,
            'threads': args.threads, 'render_optimization_requested': args.disable_render_on_step}
        if args.original_isolated_scene:
            launch_metadata['diagnostic_context'] = {
                'mode': 'original_isolated_scene_source_protocol_diagnostic',
                'benchmark_world_mode': False, 'actual_native_context_pending': True,
                'validated_inventory': receipt['inventory']}
        journal.write('launch.json', launch_metadata)
        journal.event('native_loading')
        from wia.runtime.commands.run_esi_capture_worker import allocation_gpu
        from wia.runtime.scene_capture_adapters import install_gpu_selection, isolate_texture_cache, gpu_evidence
        gpu = allocation_gpu(0, execution_mode='local', renderer_order='pci')
        journal.event('gpu_selected', gpu=gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
        install_gpu_selection(gpu)
        from omnigibson import simulator
        isolate_texture_cache(simulator, os.environ['OMNIGIBSON_APPDATA_PATH'])
        from omnigibson.macros import gm
        with gm.unlocked():
            gm.USE_ENCRYPTED_ASSETS = True
            gm.USE_GPU_DYNAMICS = True
            gm.ENABLE_FLATCACHE = False
            gm.HEADLESS = True  # Diagnostic display only; source physics remains unchanged.
        import torch
        torch.set_num_threads(args.threads)
        from wia.tasks.loader import LoadedScene, LoadedTask, sha256, task_definition
        from wia.tasks.schema import SceneDefinition
        from wia.tasks.scene import ScenePatches
        from wia.runtime.sim.esi_omnigibson import ESISession
        from wia.runtime.sim.esi_robot_cameras import configure_robot_cameras
        definition = SceneDefinition.model_validate(scene_raw)
        task_def = task_definition(task_raw)
        raw = json.loads(source.read_text())
        loaded = LoadedScene(definition, paths[1], source, sha256(paths[1]), sha256(source),
            'adapted', definition.seed, raw['scene'], raw['room'], run_mode='diagnostic')
        task = LoadedTask(task_def, paths[0], source, sha256(paths[0]), sha256(source),
            'adapted', definition.seed, raw['scene'], raw['room'], run_mode='diagnostic')
        if args.original_isolated_scene:
            adapter = _OriginalIsolatedSceneAdapter(loaded, receipt)
            session = ESISession(source, require_robot=False)
        else:
            adapter = ScenePatches(loaded, definition.bindings)
            settings = adapter.observation_settings()
            settings['camera_size'] = list(definition.runtime.camera_size)
            session = ESISession(source, require_robot=True, grasp_mode='assisted', **settings)
        session.scene_adapter, session.loaded_task = adapter, task
        source_initial_joints = {}
        session._initialization_stage_observer = NativeCapacityCalibration.capture_source_initialization(receipt, source_initial_joints)
        session.reset()
        if args.original_isolated_scene:
            context = adapter.diagnostic_context(session)
            if not context['robot_absent']:
                raise ValueError('Original isolated diagnostic unexpectedly contains a robot.')
            launch_metadata['diagnostic_context'] = context
            journal.write('launch.json', launch_metadata)
            journal.write('isolation_context.json', context)
        else:
            configure_robot_cameras(session, ROOT / definition.runtime.robot_camera_config, resolution=definition.runtime.camera_size)
        journal.write('gpu.json', gpu_evidence(session, gpu))
        driver = NativeCapacityCalibration(session, receipt, source_initial_joints, args.output_dir,
            render_without_images=args.disable_render_on_step)
        if args.original_isolated_scene:
            result = driver.run(pairs=selected)
            result['diagnostic_context'] = adapter.diagnostic_context(session)
            journal.write('calibration_receipt.json', result)
        else:
            # Save direct native mounted-camera renders; never instantiate observation/history.
            from PIL import Image
            sensor = session.robot.sensors[session.camera_name]
            Image.fromarray(session._sensor_rgb(sensor)).save(args.output_dir / 'baseline_native_head.png')
            result = driver.run(pairs=selected)
            Image.fromarray(session._sensor_rgb(sensor)).save(args.output_dir / 'restored_native_head.png')
        return 0 if (result['post_calibration_inventory_reset']['accepted'] and
            not result['calibration_recipe'].get('native_error')) else 1
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        if args.original_isolated_scene and adapter is not None:
            journal.write('isolation_context.json', adapter.diagnostic_context())
        journal.event('driver_failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        return 1
    finally:
        def file_hash(path):
            return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        summary = {'schema': 'native_capacity_calibration_diagnostic_v1',
            'task_file': str(args.task_file.resolve()), 'scene_file': str(args.scene_file.resolve()),
            'task_sha256': file_hash(args.task_file), 'scene_sha256': file_hash(args.scene_file),
            'diagnostic_only': True, 'model_calls': 0, 'autonomous_agent_performance': False,
            'benchmark_world_mode': not args.original_isolated_scene, 'error': failure,
            'complete': bool(result and not failure and not result['calibration_recipe'].get('native_error')
                and result.get('post_calibration_inventory_reset', {}).get('accepted')),
            'truth_available': bool(result and result.get('truth', {}).get('accepted')),
            'calibration_receipt': str(args.output_dir / 'calibration_receipt.json') if result else None}
        if session is not None and getattr(session, 'loaded_task', None) is not None:
            summary['source_sha256'] = session.loaded_task.source_sha256
            summary['source_identity'] = session.loaded_task.definition.source.json_path
        if result:
            summary['native_protocol_error'] = result.get('calibration_recipe', {}).get('native_error')
            summary['inventory_reset'] = result.get('post_calibration_inventory_reset')
        journal.write('result.json', summary)
        if session is not None:
            session.close()


if __name__ == '__main__':
    raise SystemExit(main())
