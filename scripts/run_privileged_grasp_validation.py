"""Run one isolated privileged model pickup diagnostic in the behavior environment.

Run: PYTHONPATH=src:../Show-Harness-Explore python scripts/run_privileged_grasp_validation.py
--case CASE.json --output OUTPUT. Caller selects an idle CUDA UUID and native cache.
The existing native loop builds all controllers; only this process's web callback
is replaced. Scene assets, task definitions, physics and motion guards are unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import os
import selectors
import subprocess
import sys
from dataclasses import fields
import traceback
import uuid


def write(path, value):
    from wia.diagnostics.privileged_grasp import jsonable
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _prepare_manipulation_phase(session, case, output):
    """Use the authored temporal event before an explicitly phase-2 pickup check."""
    from wia.diagnostics.privileged_grasp import jsonable
    import numpy as np
    temporal = getattr(session, 'temporal_phase_controller', None)
    if temporal is None:
        return {'temporal_scene': False}
    if case.get('native_manipulation_phase') != 'phase_2' or temporal.phase != 'phase_1':
        raise ValueError('Temporal pickup requires an explicit authored phase-2 diagnostic setup.')
    target = session.env.scene.object_registry('name', case['target'])
    before = [jsonable(value) for value in target.get_position_orientation()]
    first = session.observation_source.capture(action_boundary=True).observation
    temporal.prepared_boundary = (first.snapshot_id, first.action_step)
    transition = temporal.advance()
    after = [jsonable(value) for value in target.get_position_orientation()]
    target_unchanged = all(np.allclose(a, b, rtol=0, atol=1e-6) for a, b in zip(before, after))
    receipt = {'from_phase': 'phase_1', 'to_phase': temporal.phase,
        'target_pose_before': before, 'target_pose_after': after, 'target_pose_unchanged': target_unchanged,
        'authored_transition': transition, 'survey_success_claimed': False, 'task_success_claimed': False}
    write(output / 'native_phase_setup.json', receipt)
    if (temporal.phase != 'phase_2' or not target_unchanged or not transition['world_unchanged']
            or transition['physics_steps'] != 0 or transition['robot_actions'] != 0):
        raise ValueError('Authored phase transition did not preserve the diagnostic target/world contract.')
    return {key: receipt[key] for key in ('from_phase', 'to_phase', 'target_pose_unchanged',
                                         'survey_success_claimed', 'task_success_claimed')}


def _relocate(session, controller, executor, case, output):
    """Apply the explicitly selected initial base pose, then rebuild hold targets."""
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from wia.diagnostics.privileged_grasp import jsonable
    before = controller.pose()
    intervention = {'mode': case['start_condition'], 'original_robot_pose': before,
                    'target_or_furniture_teleported': False, 'robot_pose_changed': False}
    if case['start_condition'] == 'original_start':
        return controller, executor, intervention
    pose = case.get('near_pose')
    if not pose:
        raise ValueError('No reviewed collision-clear nearby initial pose; do not force a colliding spawn.')
    objects = {obj.name: jsonable(obj.get_position_orientation()[0])
               for obj in session.env.scene.objects if not obj.fixed_base and obj is not session.robot}
    payloads = []
    for arm in ('left', 'right'):
        with controller._select_arm(arm):
            grasp = controller.debug_gripper_state()
        if grasp.get('object_name') is None:
            continue
        if not case.get('relocate_preheld_payloads', False):
            raise ValueError('Robot-only relocation cannot preserve an existing payload; use an explicitly reviewed assembly start.')
        if (grasp.get('holding') is not True or grasp.get('constraint_active') is not True
                or grasp['object_name'] == case['target']):
            raise ValueError('Assembly relocation requires an existing native grasp of a nontarget payload.')
        obj = session.env.scene.object_registry('name', grasp['object_name'])
        position, quaternion = map(jsonable, obj.get_position_orientation())
        delta = Rotation.from_quat(pose['quaternion_xyzw']) * Rotation.from_quat(before['quaternion_xyzw']).inv()
        selected_position = np.asarray(pose['position']) + delta.apply(np.asarray(position) - before['position'])
        selected_quaternion = (delta * Rotation.from_quat(quaternion)).as_quat()
        payloads.append({'name': obj.name, 'arm': arm, 'object': obj, 'prim_path': str(obj.prim_path),
            'original_position': position, 'original_quaternion_xyzw': quaternion,
            'selected_position': selected_position.tolist(), 'selected_quaternion_xyzw': selected_quaternion.tolist(),
            'grasp_before': grasp})
    def belongs(path, prefix):
        return path == prefix or path.startswith(prefix + '/')
    allowed_payload_contacts = {tuple(sorted((str(c.body0), str(c.body1))))
        for c in session.robot.contact_list()
        if any(belongs(str(body), payload['prim_path']) for payload in payloads for body in (c.body0, c.body1))}
    session.robot.set_position_orientation(position=torch.tensor(pose['position'], dtype=torch.float32),
        orientation=torch.tensor(pose['quaternion_xyzw'], dtype=torch.float32))
    session.robot.keep_still()
    for payload in payloads:
        payload['object'].set_position_orientation(
            position=torch.tensor(payload['selected_position'], dtype=torch.float32),
            orientation=torch.tensor(payload['selected_quaternion_xyzw'], dtype=torch.float32))
        payload['object'].keep_still()
    from wia.runtime.sim.esi_dual_arm import R1DualManipulationController
    from wia.runtime.sim.esi_trunk import R1TrunkController
    from wia.runtime.sim.esi_actions import create_r1_executor
    fresh = R1DualManipulationController(session, controller.config, arm_config=controller.arm_config,
        gripper_config=controller.gripper_config, rotation_config=controller.rotation_config,
        on_result=controller.on_result)
    fresh.trunk_controller = R1TrunkController(fresh, controller.trunk_controller.config)
    new_executor = create_r1_executor(fresh)
    new_executor.macro_execution = executor.macro_execution
    new_executor.macro_limits = executor.macro_limits
    new_executor.on_macro_result = executor.on_macro_result
    for _ in range(12):
        fresh.step_base_debug()
    payload_names = {payload['name'] for payload in payloads}
    drifts = {obj.name: float(np.linalg.norm(np.asarray(jsonable(obj.get_position_orientation()[0])) - objects[obj.name]))
              for obj in session.env.scene.objects if obj.name in objects and obj.name not in payload_names}
    contacts = []
    floor_paths = [str(obj.prim_path) for obj in session.env.scene.objects if obj.category in ('floor', 'floors', 'ground')]
    robot_path = str(session.robot.prim_path)
    for contact in session.robot.contact_list():
        pair = [str(contact.body0), str(contact.body1)]
        if tuple(sorted(pair)) in allowed_payload_contacts:
            continue
        external = [path for path in pair if not path.startswith(robot_path + '/')]
        if external and not all(any(path.startswith(floor + '/') or path == floor for floor in floor_paths) for path in external):
            contacts.append(pair)
    payload_checks = []
    for payload in payloads:
        obj = payload.pop('object')
        position, quaternion = map(jsonable, obj.get_position_orientation())
        with fresh._select_arm(payload['arm']):
            grasp = fresh.debug_gripper_state()
        external = [[str(c.body0), str(c.body1)] for c in obj.contact_list()
            if not all(belongs(str(body), str(obj.prim_path)) or belongs(str(body), robot_path)
                       for body in (c.body0, c.body1))]
        payload_checks.append(dict(payload, settled_position=position, settled_quaternion_xyzw=quaternion,
            grasp_after=grasp, external_contacts=external,
            position_error_m=float(np.linalg.norm(np.asarray(position) - payload['selected_position'])),
            orientation_error_deg=float(np.rad2deg((Rotation.from_quat(quaternion) *
                Rotation.from_quat(payload['selected_quaternion_xyzw']).inv()).magnitude()))))
    intervention.update(robot_pose_changed=True, selected_pose=pose, settled_robot_pose=fresh.pose(),
        settling_steps=12, movable_object_displacement_m=drifts, external_nonfloor_contacts=contacts,
        maximum_movable_object_displacement_m=max(drifts.values(), default=0.),
        preheld_payloads_transformed=payload_checks, injected_attachment=False,
        original_start_navigation_certified=False)
    write(output / 'initialization_intervention.json', intervention)
    if contacts or max(drifts.values(), default=0.) > .01:
        raise ValueError('Nearby initialization has nonfloor robot contact or moves scene objects over 1 cm; no model action allowed.')
    for payload in payload_checks:
        grasp = payload['grasp_after']
        if (payload['external_contacts'] or payload['position_error_m'] > .01 or payload['orientation_error_deg'] > 5.
                or grasp.get('object_name') != payload['name'] or grasp.get('holding') is not True
                or grasp.get('constraint_active') is not True):
            raise ValueError('Existing payload grasp/pose or contact checks failed after assembly relocation.')
    if np.linalg.norm(np.asarray(fresh.pose()['position']) - pose['position']) > .03:
        raise ValueError('Nearby robot pose did not remain within 3 cm after settling.')
    session.invalidate_camera_cache()
    return fresh, new_executor, intervention


MODEL_PROCESS_ENV = dict(os.environ)
MODEL_EXECUTABLE = sys.executable


class ModelProcess:
    """Keep model dependencies and async I/O outside Kit; retain native main-thread control."""

    def __init__(self, output, configuration):
        self.output = output
        config = output / 'model_configuration.json'
        write(config, configuration)
        self.log = (output / 'model_worker.log').open('w')
        self.process = subprocess.Popen([MODEL_EXECUTABLE, '-u', '-m', 'wia.diagnostics.model_worker',
            '--output', str(output), '--configuration', str(config)],
            env=dict(MODEL_PROCESS_ENV, CUDA_VISIBLE_DEVICES=''), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        try:
            if self._receive(30).get('state') != 'ready':
                raise RuntimeError('Model worker did not become ready.')
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError('Isolated model worker response timed out.')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Isolated model worker exited; see model_worker.log.')
        result = json.loads(line)
        if result.get('state') == 'error':
            raise RuntimeError(result['error'])
        return result

    def decide(self, request, timeout):
        from core.agent.decision import ModelReply
        data = {field.name: getattr(request, field.name) for field in fields(request)
                if field.name != 'observation'}
        if request.sensor_observation is not None or request.previous_phase_observation is not None:
            raise ValueError('This diagnostic transports a single current observation bundle.')
        data['observation'] = request.observation.to_payload()
        path = self.output / 'model_request.json'
        write(path, data)
        self.process.stdin.write(json.dumps({'op': 'decide', 'request': str(path)}) + '\n')
        self.process.stdin.flush()
        result = self._receive(timeout)
        if result.get('state') != 'reply':
            raise RuntimeError('Invalid model worker response.')
        reply = json.loads(Path(result['path']).read_text())
        path.unlink()
        return ModelReply(reply['raw_text'], reply['metadata'])

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.stdin.write(json.dumps({'op': 'close'}) + '\n')
                self.process.stdin.flush()
                self.process.wait(timeout=30)
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process.stdin.close()
            self.process.stdout.close()
            self.log.close()

    def provenance(self):
        path = self.output / 'model/native_session.json'
        return json.loads(path.read_text()) if path.exists() else {'model_session_started': False}


def run_agent(session, controller, executor, case, output, intervention):
    from wia.diagnostics.privileged_grasp import GraspProbe, world_snapshot, jsonable
    from core.agent.decision import DecisionInput
    from core.action_compiler import ParameterizedAction
    from core.episode_archive import EpisodeArchive
    from core.record.images import save_png
    probe = GraspProbe(session, controller, case['target'], output / 'grasp_evidence',
        minimum_lift_m=case['minimum_lift_m'], required_hold_sim_s=case['required_hold_sim_s'])
    previous_step = controller.on_step
    def sample():
        if previous_step:
            previous_step()
        probe.sample_step()
    controller.on_step = sample
    session.episode_id = 'privileged_grasp_' + uuid.uuid4().hex
    observation = session.observation_source.begin_episode(
        'Privileged diagnostic: grasp ' + case['target'] + ' and lift its bottom at least 0.05 m for a 0.5 s held interval.')
    episode_id = observation.require_initial_overview().episode_id
    configuration = {key: case[key] for key in ('model', 'reasoning_effort', 'max_steps', 'start_condition',
        'model_timeout_s', 'action_timeout_s', 'loop_timeout_s', 'minimum_lift_m', 'required_hold_sim_s')}
    configuration.update(episode_id=episode_id, diagnostic=True, benchmark_score=False,
                         task_deadline_replaced_for_diagnostic=True,
                         diagnostic_hints=case.get('diagnostic_hints', []),
                         primitive_profile='fine', macro_execution=executor.macro_execution,
                         controller_configuration={'base': vars(controller.config), 'arm': vars(controller.arm_config),
                             'trunk': vars(controller.trunk_controller.config), 'gripper': vars(controller.gripper_config),
                             'rotation': vars(controller.rotation_config)})
    report = {'schema': 'wia_privileged_grasp_run_v1', 'case': case, 'configuration': configuration,
              'initialization_intervention': intervention, 'state': 'running', 'decisions': 0,
              'robot_actions': 0, 'records': [], 'success': False, 'full_task_success_claimed': False}
    write(output / 'summary.json', report)
    started = time.monotonic()
    feedback = None
    source = session.observation_source
    agent = None
    # The shared executor normalizes exceptions to controller_error. Preserve the
    # original traceback locally before that normalization, without retrying it.
    for attribute in ('_execute_action', '_execute_metric'):
        original_execute = getattr(executor, attribute)
        if original_execute is None:
            continue
        def traced_execute(*args, _original=original_execute, _attribute=attribute, **kwargs):
            try:
                return _original(*args, **kwargs)
            except Exception as error:
                with (output / 'controller_errors.jsonl').open('a') as log:
                    log.write(json.dumps({'interface': _attribute, 'arguments': jsonable(args),
                        'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}) + '\n')
                raise
        setattr(executor, attribute, traced_execute)
    try:
        agent = ModelProcess(output, configuration)
        for index in range(case['max_steps']):
            if time.monotonic() - started > case['loop_timeout_s']:
                report['termination'] = 'diagnostic_wall_budget'
                break
            folder = output / 'steps' / f'{index:03d}'
            folder.mkdir(parents=True)
            observation = source.capture(action_boundary=True).observation
            EpisodeArchive(output).save(observation, folder / 'observation')
            world = world_snapshot(session, controller, case['target'])
            write(folder / 'world.json', world)
            context = {'parameterized_actions': {'resolution': executor.primitive_resolution.to_dict(),
                'limits': executor.macro_limits.to_dict()}, 'privileged_diagnostic': {
                'target_name': case['target'], 'start_condition': case['start_condition'],
                'initialization_intervention': intervention, 'world_state': world,
                'current_grasp_evidence': probe.evidence()}}
            request = DecisionInput(observation, tuple(executor.available_actions), feedback=feedback,
                action_context=context, steps_remaining=case['max_steps'] - index,
                action_mode='parameterized', decision_step=index, world_observation_log=True)
            call_start = time.monotonic()
            # Only immutable captured sensor/state data crosses this boundary.
            # Native capture, control and physics remain on this main thread.
            reply = agent.decide(request, case['model_timeout_s'] + 120)
            row = {'index': index, 'model_wall_s': time.monotonic() - call_start,
                   'raw_command': reply.raw_text, 'model_metadata': reply.metadata}
            report['decisions'] += 1
            write(folder / 'reply.json', row)
            try:
                command = ParameterizedAction.parse(reply.raw_text, executor.available_actions)
            except ValueError as error:
                row['parse_error'] = str(error)
                feedback = str(error)
                report['records'].append(row)
                continue
            row['command'] = command.to_dict()
            if command.action == 'DONE':
                report['termination'] = 'model_done'
                report['records'].append(row)
                break
            action_start = time.monotonic()
            result = executor.execute_macro(command, resolution=executor.primitive_resolution,
                limits=executor.macro_limits,
                should_cancel=lambda: time.monotonic() - action_start > case['action_timeout_s'])
            report['robot_actions'] += 1
            row.update(result=jsonable(result), action_wall_s=time.monotonic() - action_start)
            if probe.evidence()['qualifying_hold_now'] and not probe.success:
                # Confirm sustained holding with the same native hold command.
                for _ in range(math.ceil(case['required_hold_sim_s'] / controller.dt) + 2):
                    controller.step_base_debug()
                    if probe.success or not probe.evidence()['qualifying_hold_now']:
                        break
                row['passive_hold_confirmation'] = True
            row['grasp_evidence'] = probe.summary()
            report['records'].append(row)
            report['success'] = probe.success
            write(folder / 'result.json', row)
            write(output / 'summary.json', report)
            print(json.dumps({'decision': index, 'action': command.to_dict(), 'completed': result.get('completed'),
                'reason': result.get('reason'), 'held': probe.evidence()['target_held'],
                'max_held_lift_m': probe.maximum_held_lift_m, 'success': probe.success}), flush=True)
            if probe.success:
                report['termination'] = 'measured_grasp_and_sustained_lift'
                break
            feedback = json.dumps(jsonable(result), separators=(',', ':'))
            if (result.get('motion_stopped') is False or result.get('outcome_known') is False
                    or result.get('physics_steps_known') is False):
                report['termination'] = 'execution_state_uncertain'
                break
        else:
            report['termination'] = 'decision_budget'
        report['state'] = 'completed'
    except Exception as error:
        report.update(state='error', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        controller.on_step = previous_step
        report['grasp_evidence'] = probe.summary()
        report['success'] = probe.success
        report['loop_wall_s'] = time.monotonic() - started
        try:
            final = source.capture(action_boundary=True).observation
            EpisodeArchive(output).save(final, output / 'final_observation')
            save_png(output / 'final_head.png', final.rgb_images()['agentview_rgb'])
        except Exception as error:
            report['final_capture_error'] = str(error)
        try:
            if agent is not None:
                agent.close()
        finally:
            report['native_model'] = agent.provenance() if agent is not None else {'model_session_started': False}
            write(output / 'summary.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    case = json.loads(args.case.read_text())
    if case['model'] != 'gpt-6.1-sol' or case['reasoning_effort'] != 'high':
        raise ValueError('This authorized diagnostic uses exactly gpt-6.1-sol/high.')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise FileExistsError('Retain previous attempts; use a new output directory.')
    from wia.runtime.commands.run_esi_capture_worker import allocation_gpu
    from wia.runtime.scene_capture_adapters import install_gpu_selection, isolate_texture_cache
    import os
    gpu = allocation_gpu(0, execution_mode='local', renderer_order='pci')
    install_gpu_selection(gpu)
    from omnigibson import simulator
    isolate_texture_cache(simulator, os.environ['OMNIGIBSON_APPDATA_PATH'])
    from wia.runtime.commands.run_esi_robot import main as robot_main
    import wia.runtime.sim.esi_web as web
    def callback(session, metadata, controller, executor, **kwargs):
        try:
            phase_setup = _prepare_manipulation_phase(session, case, output)
            controller, executor, intervention = _relocate(session, controller, executor, case, output)
            intervention['native_phase_setup'] = phase_setup
            write(output / 'initialization_intervention.json', intervention)
            run_agent(session, controller, executor, case, output, intervention)
        except Exception as error:
            write(output / 'diagnostic_error.json', {'error': f'{type(error).__name__}: {error}',
                'traceback': traceback.format_exc(), 'model_success_inferred': False})
            raise
    original = web.run_esi_control
    web.run_esi_control = callback
    try:
        return robot_main(['--scene-file', case['scene_file'], '--variant', case['variant'],
            '--run-mode', 'diagnostic', '--output-dir', str(output / 'native'), '--web',
            '--base-control', '--arm-control', '--gripper-control', '--rotation-control', '--trunk-control',
            '--parameterized-actions', '--primitive-profile', 'fine', '--macro-execution', 'metric_target',
            '--camera-size', '512', '512', '--preview-fps', '0', '--no-human-preview', '--no-human-recording',
            '--no-debug-cameras'])
    finally:
        web.run_esi_control = original


if __name__ == '__main__':
    raise SystemExit(main())
