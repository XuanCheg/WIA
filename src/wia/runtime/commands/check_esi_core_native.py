#!/usr/bin/env python3
"""Inspect a repaired Core candidate and exercise the public native controller.

Run with explicit CUDA_VISIBLE_DEVICES and isolated OMNIGIBSON_APPDATA_PATH:
  python src/wia/runtime/commands/check_esi_core_native.py --scene-file SCENE --task-file TASK
    --output-dir OUTPUT [--mobility] [--manipulation] [--serve]
In serve mode place numbered JSON requests in OUTPUT/requests. Supported keys
are commands (public ParameterizedAction payloads), capture, and close. This
diagnostic never accepts direct joint targets, object setters, or teleportation.
Use --initial-controls-file FILE for declared empty-hand public controls before
the initial condition, observation episode, qualification and scorer begin.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


class NativeInitializationCheckpoint(RuntimeError):
    """End a requested diagnostic before any later native initialization step."""


class NativeInitialControlsError(RuntimeError):
    """Abort initialization while retaining its exact attempted public controls."""

    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


def save_diagnostic_capture(session, output, *, snapshot=None):
    """Save a boundary, recording a fresh observation in an already active scorer.

    Scene-created evaluators use a positive episode counter after begin_episode.
    Supplied snapshots have already crossed their owning production boundary and
    must not be sampled again. An observation-only record credits no action.
    """
    if snapshot is None:
        snapshot = session.observation_source.capture(action_boundary=True)
        evaluator = getattr(session, 'task_evaluator', None)
        if evaluator is not None and getattr(evaluator, 'episode', 0):
            evaluator.record(snapshot.observation, action=None)
    snapshot.observation.save(output)
    return snapshot


def measure_initial_control_hands(controller):
    """Read native attachment and grasp state for every arm without releasing it."""
    arms, errors = {}, []
    try:
        robot = controller.robot
        arm_names = tuple(robot.arm_names)
        if not arm_names:
            raise ValueError('Native arm identities are unavailable.')
    except (AttributeError, ValueError, RuntimeError, TypeError) as error:
        return {'available': False, 'empty': None, 'arms': arms,
                'measurement_errors': [f'{type(error).__name__}: {error}']}
    for arm in arm_names:
        try:
            obj = robot._ag_obj_in_hand[arm]
            name = obj.name if obj is not None else None
            if obj is not None and (not isinstance(name, str) or not name):
                raise ValueError('Native held-object identity is unavailable.')
            holding = {-1: False, 1: True}.get(int(robot.is_grasping(arm)))
            if holding is None:
                raise ValueError('Native is_grasping reports an unknown state.')
            constraint = robot._ag_obj_constraints[arm] is not None
            pending = robot._ag_release_counter[arm] is not None
            empty = bool(holding is False and name is None and not constraint and not pending)
            arms[arm] = {'available': True, 'holding': holding, 'object_name': name,
                         'constraint_active': constraint, 'release_pending': pending, 'empty': empty}
        except (AttributeError, ValueError, RuntimeError, KeyError, TypeError) as error:
            reason = f'{type(error).__name__}: {error}'
            arms[arm] = {'available': False, 'empty': None, 'reason': reason}
            errors.append(arm + ': ' + reason)
    empty = (False if any(row['empty'] is False for row in arms.values()) else
             None if errors else True)
    return {'available': not errors, 'empty': empty, 'arms': arms, 'measurement_errors': errors}


def run_initial_controls(path, executor, controller, *, limits, state_reader, output_file,
                         primitive_profile):
    """Execute a hashed JSON list through the existing public macro boundary.

    No command is credited as an episode action. Invalid command lists execute
    nothing; failed macros or unavailable/nonempty hands abort the sequence.
    NativeInitialControlsError.record contains the same journal written to disk.
    """
    from core.action_compiler import ParameterizedAction, unique_json_object
    from core.action_units import BASE_ATOMS, MOVE_ATOMS, ROTATE_ATOMS, TRUNK_ATOMS, RELEASE_ATOM
    from wia.runtime.scene_capture_queue import write_json

    path = Path(path).expanduser().resolve()
    record = {'schema': 'native_initial_controls_v1', 'input_file': str(path),
              'input_sha256': None, 'commands': None, 'records': [],
              'primitive_profile': primitive_profile,
              'credited_as_episode_action': False, 'complete': False, 'completed_commands': 0}

    def publish():
        write_json(output_file, record)

    def snapshot():
        value = {'hands': measure_initial_control_hands(controller)}
        try:
            value['state'] = deepcopy(state_reader())
        except Exception as error:
            value['state_error'] = f'{type(error).__name__}: {error}'
        return value

    def require_empty(value, stage):
        if value['hands']['available'] is not True or value['hands']['empty'] is not True:
            raise ValueError(f'Initial controls require measured empty hands {stage}; no held object may be dropped.')
        if value.get('state_error'):
            raise ValueError(f'Initial control state is unavailable {stage}: ' + value['state_error'])

    def reject_constant(value):
        raise ValueError('Initial controls require finite JSON numbers; invalid constant: ' + value)

    try:
        raw = path.read_bytes()
        record['input_sha256'] = hashlib.sha256(raw).hexdigest()
        payloads = json.loads(raw, object_pairs_hook=unique_json_object, parse_constant=reject_constant)
        json.dumps(payloads, allow_nan=False)
        record['commands'] = deepcopy(payloads)
        if not isinstance(payloads, list):
            raise ValueError('Initial controls must be a JSON list of public ParameterizedAction commands.')
        allowed = set(BASE_ATOMS + MOVE_ATOMS + ROTATE_ATOMS + TRUNK_ATOMS + (RELEASE_ATOM,))
        actions = []
        for payload in payloads:
            if not isinstance(payload, dict) or payload.get('action') not in allowed:
                raise ValueError('Initial controls allow only BASE, MV, TRUNK, ROTATE and RELEASE public commands.')
            actions.append(ParameterizedAction.from_payload(payload, executor.available_actions))
        resolution = executor.primitive_resolution
        record.update(primitive_resolution=resolution.to_dict(), macro_limits=limits.to_dict())
        publish()
        for index, (payload, action) in enumerate(zip(payloads, actions)):
            row = {'index': index, 'command': deepcopy(payload), 'before': snapshot(),
                   'receipt': None, 'after': None, 'credited_as_episode_action': False}
            record['records'].append(row)
            publish()
            require_empty(row['before'], 'before command')
            try:
                row['receipt'] = executor.execute_macro(action, resolution=resolution, limits=limits)
            except Exception as error:
                row['execution_error'] = f'{type(error).__name__}: {error}'
                row['after'] = snapshot()
                raise
            row['after'] = snapshot()
            publish()
            require_empty(row['after'], 'after command')
            receipt = row['receipt']
            if (not isinstance(receipt, dict) or receipt.get('success') is not True
                    or receipt.get('completed') is not True or receipt.get('timeout') is not False
                    or receipt.get('cancelled') is True):
                raise RuntimeError(f'Initial public command {index} failed; initialization stopped.')
            record['completed_commands'] += 1
            publish()
        record['complete'] = True
        publish()
        return record
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
        publish()
        raise NativeInitialControlsError(record['error'], record) from error


def select_native_variant(scene, task, requested=None):
    """Select a declared common variant without inventing an adapted layout."""
    common = set(scene.scene.variants) & set(task.scene.variants)
    if requested is not None:
        if requested not in common:
            raise ValueError('Requested variant must be declared by both task and scene.')
        return requested
    for name in ('adapted', 'original'):
        if name in common:
            return name
    raise ValueError('An explicit shared task/scene variant is required.')


def load_native_control_configs(runtime, *, manipulation, primitive_profile):
    """Resolve the scene trunk settings and existing fixed/fine control settings."""
    from wia.runtime.sim.esi_base import BaseControlConfig

    configs = {'base': BaseControlConfig.load(), 'arm': None, 'rotation': None,
               'gripper': None, 'trunk': None}
    if manipulation:
        from wia.runtime.sim.esi_arm import ArmControlConfig
        from wia.runtime.sim.esi_rotation import RotationControlConfig
        from wia.runtime.sim.esi_gripper import GripperControlConfig
        from wia.runtime.sim.esi_trunk import TrunkControlConfig

        configs.update(arm=ArmControlConfig.load(), rotation=RotationControlConfig.load(),
                       gripper=GripperControlConfig.load(),
                       trunk=TrunkControlConfig.load(ROOT / runtime.trunk_config))
        if primitive_profile == 'fine':
            profile = json.loads((ROOT / 'configs/esi_primitives_fine.json').read_text())
            for name in ('base', 'arm', 'rotation', 'trunk'):
                configs[name] = replace(configs[name], **profile[name])
    return configs


def prepare_native_runtime_configuration(adapter, session, configs, *, primitive_profile,
                                         result, output_file):
    """Record effective controls and reuse the production raw-physics settling.

    SceneOnly's method only reads the loaded scene/runtime and sets its applied
    step count, so it can run on the retained ScenePatches adapter unchanged.
    Its N raw og.sim.step calls are not N environment/control steps. Controller
    targets already exist; no neutral action, teleport or extra step is added.
    """
    from wia.runtime.scene_capture_queue import write_json
    from wia.tasks.scene_only import SceneOnly

    runtime = adapter.task.definition.runtime
    sim = session.pipeline.og.sim
    record = {
        'schema': 'native_runtime_configuration_v1', 'primitive_profile': primitive_profile,
        'scene_trunk_config_file': str((ROOT / runtime.trunk_config).resolve()),
        'primitive_profile_config_file': (str(ROOT / 'configs/esi_primitives_fine.json')
                                          if primitive_profile == 'fine' else None),
        'effective_configs': {name: asdict(config) if config is not None else None
                              for name, config in configs.items()},
        'post_controller_settle': {
            'declared_steps': runtime.post_controller_settle_steps,
            'applied_steps': 0, 'physics_step_method': 'og.sim.step',
            'native_time_before_s': float(sim.current_time), 'native_time_after_s': None,
            'complete': False, 'credited_as_episode_action': False,
        },
    }
    result['runtime_configuration'] = record
    write_json(output_file, record)
    settling = record['post_controller_settle']
    try:
        SceneOnly.prepare_after_controller_initialization(adapter, session)
        settling.update(applied_steps=adapter.post_controller_settle_steps, complete=True)
    except Exception as error:
        settling.update(applied_steps=None, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        settling['native_time_after_s'] = float(sim.current_time)
        write_json(output_file, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-file', type=Path, required=True)
    parser.add_argument('--task-file', type=Path, required=True)
    parser.add_argument('--variant', help='Declared shared variant; defaults to adapted when present, otherwise original.')
    parser.add_argument('--interaction-attempt-limit', type=int,
        help='Diagnostic-only cap on started native placement trials; never an infeasibility certificate.')
    parser.add_argument('--native-stance-proposal-fallback', action='store_true',
        help='Diagnostic revised-Inclined proposal filter comparison; all native acceptance gates remain required.')
    parser.add_argument('--diagnose-object', action='append', default=[],
        help='Exact native object name for bounded link/physics diagnostics at initialization stages.')
    parser.add_argument('--stop-after-native-stage', choices=('constructed', 'source_postprocessed', 'adapted'),
        help='Stop at a read-only initialization checkpoint; never a task or access qualification.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mobility', action='store_true')
    parser.add_argument('--source-route-pilot', action='store_true',
        help='Privately follow an unchanged Navigation source path with guarded public base commands; no qualification claim.')
    parser.add_argument('--manipulation', action='store_true')
    parser.add_argument('--initial-release', action='store_true',
        help='Open the empty working gripper through public control before scoring begins; record as initialization.')
    parser.add_argument('--initial-controls-file', type=Path,
        help='JSON list of declared empty-hand public controls before qualification and scoring; abort on any failure.')
    parser.add_argument('--primitive-profile', choices=('fixed', 'fine'), default='fixed',
        help='Use the same explicit fixed/fine public increments as run_esi_robot.py.')
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--evaluate', action='store_true', help='Record native evaluator outcomes; no model calls.')
    parser.add_argument('--review-views', action='store_true', help='Capture independent human Global/Object/ESI views at initial reset.')
    parser.add_argument('--precontroller-hold-steps', type=int, default=0,
        help='Diagnostic neutral hold before startup posture; separate settling from controller effects.')
    parser.add_argument('--check-render-products', action='store_true',
        help='Compare frames before/after rebinding unchanged native camera render products.')
    args = parser.parse_args()
    if args.initial_release and not args.manipulation:
        parser.error('--initial-release requires --manipulation')
    from wia.runtime.scene_capture_queue import read_json, write_json
    from wia.runtime.commands.run_esi_capture_worker import allocation_gpu
    from wia.runtime.scene_capture_adapters import gpu_evidence, install_gpu_selection, isolate_texture_cache
    gpu = allocation_gpu(0, execution_mode='local', renderer_order='pci')
    install_gpu_selection(gpu)
    from omnigibson import simulator
    isolate_texture_cache(simulator, os.environ['OMNIGIBSON_APPDATA_PATH'])
    import numpy as np
    import yaml
    from wia.tasks.loader import LoadedScene, LoadedTask, sha256, task_definition, source_scene_room
    from wia.tasks.schema import SceneDefinition
    from wia.tasks.scene import ScenePatches, DeclarativeScene
    from wia.runtime.sim.esi_omnigibson import ESISession
    from wia.runtime.sim.esi_robot_cameras import configure_robot_cameras
    from wia.runtime.sim.esi_base import R1BaseController, BaseControlConfig
    from wia.runtime.sim.esi_actions import create_r1_executor
    from wia.runtime.sim.esi_reflection import measure_reflection
    from wia.runtime.sim.esi_safety import SafetyObserver
    from wia.tasks.interaction_setup import array
    from core.action_compiler import ParameterizedAction, MacroLimits

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.interaction_attempt_limit is not None and args.interaction_attempt_limit <= 0:
        raise ValueError('Diagnostic interaction attempt limit must be positive.')
    import faulthandler
    import signal
    stack_file = (output / 'native_stack.log').open('a')
    faulthandler.register(signal.SIGUSR2, file=stack_file, all_threads=True)
    scene_path, task_path = args.scene_file.resolve(), args.task_file.resolve()
    definition = SceneDefinition.model_validate(yaml.safe_load(scene_path.read_text()))
    task_def = task_definition(yaml.safe_load(task_path.read_text()))
    variant = select_native_variant(definition, task_def, args.variant)
    if task_def.source.json_path != definition.source.json_path:
        raise ValueError('Native task and scene must declare the same exact source identity.')
    from wia.tasks.sources import resolve_source
    source = resolve_source(definition.source.json_path)
    raw = read_json(source)
    scene_name, room_name = source_scene_room(definition.source.json_path, raw)
    loaded = LoadedScene(definition, scene_path, source, sha256(scene_path), sha256(source),
        variant, definition.seed, scene_name, room_name)
    task = LoadedTask(task_def, task_path, source, sha256(task_path), sha256(source),
        variant, definition.seed, scene_name, room_name, run_mode='diagnostic')
    adapter = ScenePatches(loaded, definition.bindings)
    settings = adapter.observation_settings()
    settings['camera_size'] = list(definition.runtime.camera_size)
    session = ESISession(source, require_robot=True, grasp_mode='assisted', **settings)
    session.scene_adapter = adapter
    session.loaded_task = task
    session.public_task_text = task.task_text
    session.diagnostic_interaction_attempt_limit = args.interaction_attempt_limit
    session.diagnostic_native_stance_fallback = args.native_stance_proposal_fallback
    began = time.monotonic()
    result = {'complete': False, 'model_calls': 0, 'public_controls_only': True,
        'primitive_profile': args.primitive_profile,
        'task_file': str(task_path), 'scene_file': str(scene_path), 'pid': os.getpid(),
        'task_sha256': sha256(task_path), 'scene_sha256': sha256(scene_path), 'gpu': gpu,
        'source_identity': definition.source.json_path, 'source_file': str(source),
        'source_sha256': sha256(source),
        'diagnostic_overrides': {'interaction_attempt_limit': args.interaction_attempt_limit,
            'native_stance_proposal_fallback': args.native_stance_proposal_fallback,
            'initial_controls_file': str(args.initial_controls_file.resolve()) if args.initial_controls_file else None,
            'diagnose_objects': args.diagnose_object, 'stop_after_native_stage': args.stop_after_native_stage},
        'autonomous_agent_performance': False, 'commands': []}
    startup_trace = []

    def startup_sample(stage):
        rows = {}
        names = {binding.name for binding in definition.bindings.values()}
        for obj in session.env.scene.objects:
            if obj.name not in names and obj is not session.robot:
                continue
            rows[obj.name] = {'pose': [array(v).tolist() for v in obj.get_position_orientation()],
                'contacts': [{'body0': str(c.body0), 'body1': str(c.body1)} for c in obj.contact_list()]}
        startup_trace.append({'stage': stage, 'sample': len(startup_trace),
            'native_time_s': float(session.pipeline.og.sim.current_time), 'objects': rows})

    def progress(stage):
        write_json(output / 'progress.json', {'stage': stage, 'pid': os.getpid(),
            'elapsed_s': time.monotonic() - began, 'gpu': gpu})
        print(stage, flush=True)

    from wia.tasks.interaction_diagnostics import install_interaction_diagnostics
    install_interaction_diagnostics(session, output, progress=progress)

    def initialization_state(native, stage, config):
        from wia.runtime.sim.esi_initial_camera_diagnostics import measure_initial_camera_mounts
        from wia.runtime.sim.esi_initial_link_diagnostics import measure_initial_object_links
        objects = native.env.scene.objects
        objects = list(objects.values()) if isinstance(objects, dict) else list(objects)
        rows = []
        for obj in objects:
            row = {'name': obj.name, 'category': obj.category,
                'in_rooms': list(getattr(obj, 'in_rooms', ()) or ())}
            try:
                row.update(pose=[array(v).tolist() for v in obj.get_position_orientation()],
                    aabb=[array(v).tolist() for v in obj.aabb],
                    fixed_base=bool(obj.fixed_base), visual_only=bool(obj.visual_only))
                if obj in native.env.robots:
                    row['joints'] = array(obj.get_joint_positions()).tolist()
                    row['contacts'] = [{'body0': str(c.body0), 'body1': str(c.body1),
                        'position': array(c.position).tolist(), 'normal': array(c.normal).tolist()}
                        for c in obj.contact_list()]
            except Exception as error:
                row['measurement_error'] = str(error)
            rows.append(row)
        write_json(output / ('initialization_' + stage + '.json'),
            {'stage': stage, 'elapsed_s': time.monotonic() - began,
             'source_constructor': native.task_module.__name__, 'scene_config': config.get('scene'),
             'robot_config': config.get('robots'), 'objects': rows,
             'camera_mounts': measure_initial_camera_mounts(native, stage),
             'selected_object_links': (measure_initial_object_links(native, args.diagnose_object, stage)
                 if args.diagnose_object else None),
             'interaction_setup': getattr(native, 'category_interaction_setup', None)})
        if args.stop_after_native_stage == stage:
            raise NativeInitializationCheckpoint(stage)

    session._initialization_stage_observer = initialization_state

    def state():
        objects = session.env.scene.objects
        objects = list(objects.values()) if isinstance(objects, dict) else list(objects)
        names = {binding.name for binding in definition.bindings.values()}
        rows = {}
        for obj in objects:
            if obj.name not in names and obj.category != 'door':
                continue
            try:
                p, q = (array(v).tolist() for v in obj.get_position_orientation())
                rows[obj.name] = {'category': obj.category, 'position_m': p, 'quaternion_xyzw': q,
                    'aabb_m': [array(v).tolist() for v in obj.aabb], 'scale': array(obj.scale).tolist(),
                    'links': {name: {'aabb_m': [array(v).tolist() for v in link.aabb]} for name, link in obj.links.items()},
                    'joints': {name: {'position': array(joint.get_state()[0]).tolist(),
                        'lower_limit': array(joint.lower_limit).tolist(),
                        'upper_limit': array(joint.upper_limit).tolist()}
                        for name, joint in obj.joints.items()}}
            except Exception as error:
                rows[obj.name] = {'measurement_error': str(error)}
        cameras = {name: session.robot.sensors[sensor] for name, sensor in session.robot_camera_registry.items()}
        if session.scene_camera is not None:
            cameras['original_esi_reference'] = session.scene_camera
        camera_state = {}
        for name, sensor in cameras.items():
            parameters = sensor.camera_parameters
            projection = array(parameters['cameraProjection']).reshape(4, 4)
            width, height = array(parameters['renderProductResolution'])
            intrinsics = [[float(projection[0, 0] * width / 2), 0.,
                           float((1. - projection[0, 2]) * width / 2)],
                          [0., float(projection[1, 1] * height / 2),
                           float((1. + projection[1, 2]) * height / 2)], [0., 0., 1.]]
            camera_state[name] = {'pose_world': [array(v).tolist() for v in sensor.get_position_orientation()],
                'intrinsics': intrinsics,
                'intrinsics_source': 'installed_OG_formula_applied_to_single_native_camera_params_read',
                'resolution': [int(sensor.image_width), int(sensor.image_height)],
                'focal_length': float(sensor.focal_length), 'horizontal_aperture': float(sensor.horizontal_aperture)}
            camera_state[name]['render_parameters'] = {key: array(parameters[key]).tolist()
                for key in ('cameraViewTransform', 'cameraProjection', 'cameraAperture', 'cameraFocalLength',
                            'renderProductResolution')}
        footprint = session.robot.links[session.robot.base_footprint_link_name]
        low, high = (array(v) for v in footprint.aabb)
        footprint_points = [[float(x), float(y)] for x in (low[0], (low[0] + high[0]) / 2, high[0])
                            for y in (low[1], (low[1] + high[1]) / 2, high[1])]
        floor_room = {'expected_room': session.room_name, 'footprint_link': footprint.name,
            'footprint_aabb_m': [low.tolist(), high.tolist()], 'points_xy_m': footprint_points,
            'native_room_labels': [session.env.scene.seg_map.get_room_instance_by_point(np.asarray(point))
                                   for point in footprint_points]}
        return {'robot': controller.pose(), 'native_floor_room': floor_room, 'objects': rows, 'cameras': camera_state,
            'end_effector': controller.ee_pose() if callable(getattr(controller, 'ee_pose', None)) else None,
            'gripper': controller.debug_gripper_state() if callable(getattr(controller, 'debug_gripper_state', None)) else None,
            'head_pose': [array(v).tolist() for v in session.robot.sensors[session.camera_name].get_position_orientation()],
            'reflection': measure_reflection(session), 'interaction_setup': getattr(session, 'category_interaction_setup', None),
            'loaded_room_start': getattr(session, 'loaded_room_start', None),
            'initial_overview_input_status': getattr(session, 'initial_overview_input_status', None),
            'initial_overview_camera': getattr(session, 'initial_overview_camera_evidence', None),
            'temporal_initialization': (session.temporal_phase_controller.measure(controller)
                if getattr(session, 'temporal_phase_controller', None) is not None else None)}

    def capture(label, *, snapshot=None):
        save_diagnostic_capture(session, output / label, snapshot=snapshot)
        if getattr(session, 'reflection_sensor_profile', None) or args.review_views:
            from PIL import Image
            if session.scene_camera is None:
                session.configure_scene_camera()
            for _ in range(3):
                session.pipeline.og.sim.render()
            session.invalidate_camera_cache()
            Image.fromarray(session._sensor_rgb(session.scene_camera)).save(output / label / 'original_esi_reference.png')
        if args.review_views and label == 'initial':
            from wia.runtime.scene_capture_adapters import capture_loaded_views
            from PIL import Image
            views, location = capture_loaded_views(session, loaded,
                ['scene', 'object_overview', 'object_top', 'object_side'])
            directory = output / label / 'human_review'
            directory.mkdir(exist_ok=True)
            for name, rgb in views.items():
                Image.fromarray(rgb).save(directory / (name + '.png'))
            write_json(directory / 'receipt.json', {'views': list(views), 'target_location': location,
                'human_only': True, 'agent_inputs_unchanged': True,
                'camera_evidence': getattr(session, 'human_review_camera_evidence', None)})
        write_json(output / (label + '_state.json'), state())

    def public_macro_limits():
        return (MacroLimits(**read_json(ROOT / 'configs/esi_parameterized_fine.json'))
                if args.primitive_profile == 'fine' else MacroLimits())

    def command(payload):
        action = ParameterizedAction.from_payload(payload, executor.available_actions)
        before = state()
        limits = public_macro_limits()
        receipt = executor.execute_macro(action, resolution=executor.primitive_resolution, limits=limits)
        record = {'command': payload, 'before': before, 'result': receipt, 'after': state()}
        result['commands'].append(record)
        write_json(output / 'commands.json', result['commands'])
        return record

    try:
        progress('native_loading')
        session.reset()
        result['gpu'] = gpu_evidence(session, gpu)
        progress('native_camera_setup')
        configure_robot_cameras(session, ROOT / definition.runtime.robot_camera_config,
            resolution=definition.runtime.camera_size)
        progress('native_controller_setup')
        startup_stage = 'precontroller_hold'
        startup_sample('before_controller')
        native_step = session.step_control
        def observed_step(action):
            value = native_step(action)
            if len(startup_trace) < 512:
                startup_sample(startup_stage)
            return value
        session.step_control = observed_step
        if args.precontroller_hold_steps:
            neutral = R1BaseController(session, BaseControlConfig.load())
            for _ in range(args.precontroller_hold_steps):
                neutral.step_base_debug()
        startup_stage = 'controller_initialization'
        control_configs = load_native_control_configs(definition.runtime,
            manipulation=args.manipulation, primitive_profile=args.primitive_profile)
        if args.manipulation:
            from wia.runtime.sim.esi_dual_arm import R1DualManipulationController
            controller = R1DualManipulationController(session, control_configs['base'],
                arm_config=control_configs['arm'], gripper_config=control_configs['gripper'],
                rotation_config=control_configs['rotation'])
            from wia.runtime.sim.esi_trunk import R1TrunkController
            controller.trunk_controller = R1TrunkController(
                controller, control_configs['trunk'])
        else:
            controller = R1BaseController(session, control_configs['base'])
        startup_sample('after_controller')
        session.step_control = native_step
        executor = create_r1_executor(controller)
        executor.macro_execution = 'metric_target'
        prepare_native_runtime_configuration(adapter, session, control_configs,
            primitive_profile=args.primitive_profile, result=result,
            output_file=output / 'runtime_configuration.json')
        if args.initial_release:
            before = controller.debug_gripper_state()
            if (before.get('holding') is not False or before.get('object_name') is not None
                    or before.get('constraint_active') is not False or before.get('release_pending') is not False):
                raise ValueError('Initial release requires a measured empty hand; no held object may be dropped.')
            receipt = executor.execute_macro(ParameterizedAction(action='RELEASE'),
                resolution=executor.primitive_resolution, limits=MacroLimits())
            after = controller.debug_gripper_state()
            result['initial_release'] = {'before': before, 'receipt': receipt, 'after': after,
                'credited_as_episode_action': False}
            write_json(output / 'initial_release.json', result['initial_release'])
            if after.get('gripper_open') is not True or after.get('holding') is not False:
                raise ValueError('Initial public release did not produce an open empty hand.')
        if args.initial_controls_file:
            try:
                result['initial_controls'] = run_initial_controls(
                    args.initial_controls_file, executor, controller, limits=public_macro_limits(),
                    state_reader=state, output_file=output / 'initial_controls.json',
                    primitive_profile=args.primitive_profile)
            except NativeInitialControlsError as error:
                result['initial_controls'] = error.record
                raise
        session.public_task_text = task.task_text
        session.safety_observer = SafetyObserver(session, controller, output / 'private_safety')
        trace = []
        def sample():
            session.safety_observer.sample_step()
            trace.append({'sample': len(trace), 'robot': controller.pose(),
                'head_position_m': array(session.robot.sensors[session.camera_name].get_position_orientation()[0]).tolist(),
                'doors': {obj.name: {name: array(joint.get_state()[0]).tolist() for name, joint in obj.joints.items()}
                    for obj in session.env.scene.objects if obj.category == 'door'}})
        controller.on_step = sample
        sample()
        session.episode_id = 'native_' + output.name
        from wia.runtime.sim.esi_native_qualification import condition_snapshot
        result['initial_condition'] = condition_snapshot(session, controller, task)
        initial_observation = session.observation_source.begin_episode(task.task_text)
        result['initial_overview_input_status'] = getattr(session, 'initial_overview_input_status', None)
        result['mapping_joint_world_initial_identity'] = getattr(session, 'mapping_joint_world_initial_identity', None)
        result['mapping_joint_world_initial_identity_status'] = getattr(session, 'mapping_joint_world_initial_identity_status', None)
        if args.evaluate:
            evaluation_scene = DeclarativeScene(task)
            evaluation_scene.applied = adapter.applied
            result['qualification'] = vars(evaluation_scene.qualify(
                session, executor, output, observation=initial_observation))
            session.task_evaluator = evaluation_scene.create_evaluator(session, controller, output / 'private_evaluation')
            session.task_evaluator.begin_episode(initial_observation)
        progress('saving_initial_diagnostic_capture')
        capture('initial', snapshot=session.observation_source.latest_snapshot)
        if args.check_render_products:
            sensors = [session.robot.sensors[name] for name in session.robot_camera_registry.values()]
            if session.scene_camera is not None:
                sensors.append(session.scene_camera)
            from wia.runtime.sim.esi_robot_cameras import rebind_render_product
            for sensor in sensors:
                rebind_render_product(session, sensor)
            for _ in range(12):
                session.pipeline.og.sim.render()
            session.invalidate_camera_cache()
            capture('after_render_product_refresh')
        session.safety_observer.begin_episode()
        progress('native_initial_state_ready')
        if args.mobility:
            for token in ('BASE_FWD', 'BASE_BACK'):
                command({'action': token, 'distance_m': .25})
            capture('after_mobility')
            reflection = measure_reflection(session)
            if reflection is not None:
                reflection['legal_mobility_verified'] = all(c['result']['success'] for c in result['commands'])
                reflection['mobility_test'] = {'requested_world_m': .25, 'commands': len(result['commands']),
                    'trace_samples': len(trace), 'initial_pose_and_scale_must_also_qualify': True}
        if args.source_route_pilot:
            from wia.runtime.sim.esi_route_pilot import source_route_pilot
            progress('source_route_public_control_pilot')
            result['route_pilot'] = source_route_pilot(session, controller, task, command, output)
            capture('after_route_pilot')
        if args.evaluate:
            for _ in range(15):
                session.step_control({session.robot.name: controller.hold_action.copy()})
                if controller.on_step:
                    controller.on_step()
            capture('after_passive_hold')
        if args.serve:
            requests = output / 'requests'
            requests.mkdir(exist_ok=True)
            progress('serving_public_control_requests')
            closing = False
            while not closing:
                for path in sorted(requests.glob('*.json')):
                    reply = output / 'responses' / path.name
                    if reply.exists():
                        continue
                    request = read_json(path)
                    try:
                        responses = [command(p) for p in request.get('commands', [])]
                        if request.get('answer') is not None:
                            observed = session.observation_source.capture(action_boundary=True).observation
                            session.task_evaluator.report_answer(request['answer'], observed)
                        if request.get('temporal_event') is not None:
                            observed = session.observation_source.agent_observation()
                            session.observation_source.report_temporal(request['temporal_event'],
                                request.get('temporal_answer'), snapshot_id=observed.snapshot_id,
                                action_step=observed.action_step)
                        capture(path.stem)
                        payload = {'complete': True, 'commands': responses, 'state': state()}
                        if request.get('contact_pairs'):
                            from wia.runtime.sim.wia_contact_diagnostics import measure_contact_pair
                            payload['native_contact_diagnostics'] = []
                            for first_name, second_name in request['contact_pairs']:
                                first = session.env.scene.object_registry('name', first_name)
                                second = session.env.scene.object_registry('name', second_name)
                                if first is None or second is None:
                                    raise ValueError('Contact diagnostic pair must name existing native objects.')
                                payload['native_contact_diagnostics'].append(measure_contact_pair(first, second))
                        write_json(reply, payload)
                    except Exception as error:
                        write_json(reply, {'complete': False, 'error': str(error), 'traceback': traceback.format_exc()})
                    closing = request.get('close') is True
                time.sleep(.5)
        result.update(complete=True, final_state=state(), control_trace=trace,
            safety=session.safety_observer.result())
        progress('native_diagnostic_complete')
    except NativeInitializationCheckpoint as checkpoint:
        result.update(status='stopped_at_requested_initialization_checkpoint',
            diagnostic_checkpoint_complete=True, checkpoint_stage=str(checkpoint),
            benchmark_execution_verified=False)
        progress('requested_native_initialization_checkpoint_complete')
    except Exception as error:
        result.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        progress('native_diagnostic_failed')
        print(result['traceback'], flush=True)
    finally:
        result['elapsed_s'] = time.monotonic() - began
        result['initialization_diagnostics'] = {
            'interaction_setup': getattr(session, 'category_interaction_setup', None),
            'trunk_startup': getattr(session, 'initial_trunk_diagnostics', None),
            'delivery_support': getattr(session, 'delivery_support_initialization', None),
            'camera_recovery': getattr(session, 'camera_render_recovery', None)}
        write_json(output / 'controller_startup_trace.json', startup_trace)
        write_json(output / 'result.json', result)
        session.close()
    return 0 if result['complete'] or result.get('diagnostic_checkpoint_complete') else 1


if __name__ == '__main__':
    raise SystemExit(main())
