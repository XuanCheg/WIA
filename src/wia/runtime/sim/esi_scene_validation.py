"""Run a fixed native diagnostic. Use `python scripts/check_all_scenes.py --gpu 4 --output DIR`."""
from copy import deepcopy
import json
from pathlib import Path
import time

from core.action_compiler import ParameterizedAction
from core.agent.depth_inputs import DepthSnapshot
from wia.runtime.scene_capture_queue import write_json


def run(session, executor, output, sequence):
    output = Path(output)
    report = {'schema': 'wia_native_scene_validation_v1', 'model_calls': 0,
              'benchmark_score': False, 'sequence': deepcopy(sequence), 'observations': [],
              'actions': [], 'events': [], 'pending_coverage': [], 'errors': [], 'started_at': time.time()}
    path = output/'scorer_validation.json'
    def save():
        report['elapsed_s'] = time.time()-report['started_at']
        write_json(path, report)
    def capture(label, observation):
        item = {'label': label, 'snapshot_id': observation.snapshot_id,
                'action_step': observation.action_step}
        try:
            observation.require_initial_overview(session.initial_overview_identity)
            item['rgb_views'] = {key: list(value.shape) for key, value in observation.rgb_images().items()}
            snapshot = DepthSnapshot.from_observation(observation)
            if not snapshot.arrays:
                raise ValueError('Native head/wrist depth is unavailable.')
            item['depth_tools'] = {}
            for sensor, array in snapshot.arrays.items():
                height, width = array.shape
                item['depth_tools'][sensor] = {
                    'pixels': snapshot.pixels(sensor, [[width//2, height//2]]),
                    'region': snapshot.region(sensor, width//2, height//2, 1, 1)}
        except Exception as error:
            item['observation_error'] = f'{type(error).__name__}: {error}'
            report['errors'].append({'stage': label, 'error': item['observation_error']})
        try:
            value = session.task_evaluator.record(observation, action=None)
            item['evaluation'] = {key: value.get(key) for key in
                ('schema', 'evaluation_validity', 'score_percent', 'task_success', 'failure_stage',
                 'failure_reason', 'measurement_error', 'checks', 'mapping_joint_world_qualification')}
            native = value.get('native_result') or {}
            item['evaluation']['native_details'] = {key: native.get(key) for key in
                ('schema', 'checks', 'measurement_error', 'failure_stage', 'failure_reason',
                 'mapping_joint_world_qualification', 'phase')}
            report['evaluation_validity'] = value.get('evaluation_validity')
            validity = value.get('evaluation_validity')
            phase = getattr(getattr(session, 'temporal_phase_controller', None), 'phase', None)
            pending = validity in ('pending_done', 'partial') or (
                phase == 'phase_1' and validity == 'unavailable'
                and not native.get('measurement_error') and not native.get('failure_reason'))
            if pending:
                report['pending_coverage'].append({'stage': label, 'validity': validity, 'phase': phase})
            elif validity != 'complete':
                report['errors'].append({'stage': label, 'error': 'score_unavailable',
                                         'evaluation': item['evaluation']})
        except Exception as error:
            item['evaluation_error'] = f'{type(error).__name__}: {error}'
            report['errors'].append({'stage': label, 'error': item['evaluation_error']})
        report['observations'].append(item)
        save()
    save()
    try:
        source = session.observation_source
        initial = source.begin_episode(session.public_task_text)
        capture('initial', initial)
        for index, payload in enumerate(sequence['actions']):
            started = time.monotonic()
            record = {'index': index, 'command': payload}
            try:
                phase = getattr(getattr(session, 'temporal_phase_controller', None), 'phase', None)
                if phase == 'phase_1':
                    from wia.tasks.temporal_policy import survey_action_allowed
                    if not survey_action_allowed(payload['action']):
                        boundary = source.boundary_observation
                        receipt = source.report_temporal('phase_1_complete', None,
                            snapshot_id=boundary.snapshot_id, action_step=boundary.action_step)
                        report['events'].append({'event': 'phase_1_complete', 'before_action': index,
                                                 'receipt': receipt})
                        capture('phase_2_start', source.boundary_observation)
                command = ParameterizedAction.from_payload(payload, executor.available_actions)
                result = executor.execute_macro(command, resolution=executor.primitive_resolution,
                                                limits=executor.macro_limits)
                record.update(success=result['success'], reason=result.get('reason'),
                              completed=result['completed'], timeout=result['timeout'])
                # The existing executor callback advances observation and private scoring.
                capture(f'action_{index:02d}', source.boundary_observation)
            except Exception as error:
                record['error'] = f'{type(error).__name__}: {error}'
                report['errors'].append({'stage': f'action_{index:02d}', 'error': record['error']})
            record['elapsed_s'] = time.monotonic()-started
            report['actions'].append(record)
            save()
        done = getattr(source, 'report_done', None)
        if callable(done):
            boundary = source.boundary_observation
            try:
                receipt = done(snapshot_id=boundary.snapshot_id, action_step=boundary.action_step)
                report['events'].append({'event': 'DONE', 'receipt': receipt})
            except Exception as error:
                report['errors'].append({'stage': 'DONE', 'error': f'{type(error).__name__}: {error}'})
        capture('final', source.boundary_observation)
        report['completed'] = True
    except Exception as error:
        report['errors'].append({'stage': 'begin_episode', 'error': f'{type(error).__name__}: {error}'})
        report['completed'] = False
    save()
    print('[scene-validation] ' + json.dumps({'completed': report['completed'],
          'errors': len(report['errors']), 'output': str(path)}), flush=True)
    return report
