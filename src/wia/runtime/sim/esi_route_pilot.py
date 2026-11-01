"""Run a bounded private source-route diagnostic through public base commands.

Run via src/wia/runtime/commands/check_esi_core_native.py --source-route-pilot. Source waypoints
guide this private feasibility probe only; they never enter Agent observations.
No actor setters, door edits, arrival claims or qualification receipts are used.
"""
from __future__ import annotations

import math
import time

import numpy as np

from wia.runtime.scene_capture_queue import write_json


def next_translation(position, quaternion, target, *, maximum_m=.25, minimum_m=0.):
    """Resolve a world waypoint into one public body-relative translation."""
    delta = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)[:2]
    x, y, z, w = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    forward = math.cos(yaw) * delta[0] + math.sin(yaw) * delta[1]
    left = -math.sin(yaw) * delta[0] + math.cos(yaw) * delta[1]
    if abs(forward) >= abs(left):
        token, amount = ('BASE_FWD' if forward >= 0 else 'BASE_BACK'), abs(forward)
    else:
        token, amount = ('BASE_LEFT' if left >= 0 else 'BASE_RIGHT'), abs(left)
    return {'action': token, 'distance_m': max(minimum_m, min(maximum_m, float(amount)))}


def source_route_pilot(session, controller, task, command, output):
    """Measure legal-control progress; a blocked source route is not infeasibility."""
    import json
    import torch

    parameters = task.definition.private.evaluation.parameters
    source = json.loads(task.source_path.read_text())
    question = json.loads(source['metadata_json'])['question_data']
    path = np.asarray(question['path_world'], dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or not np.isfinite(path).all():
        raise ValueError('A finite source XY waypoint path is required.')
    if list(question['region_sequence']) != list(parameters['required_order']):
        raise ValueError('Source and task ordered room routes differ.')
    budget = task.definition.defaults
    began = time.monotonic()
    report = {'schema': 'private_source_route_pilot_v1',
        'source_sha256': task.source_sha256, 'task_sha256': task.file_sha256,
        'required_order': list(parameters['required_order']),
        'waypoint_count': len(path), 'waypoints_reached': 0, 'commands': [],
        'observed_room_sequence': [], 'ordered_rooms_observed': 0,
        'public_controls_only': True, 'object_pose_commands': 0,
        'source_geometry_is_advisory': True, 'execution_qualified': False,
        'autonomous_agent_performance': False, 'budget_calibrated': False,
        'status': 'running', 'waypoint_tolerance_m': .10,
        'minimum_command_m': controller.config.translation_m / 2}
    output.mkdir(parents=True, exist_ok=True)

    def save():
        report['elapsed_s'] = time.monotonic() - began
        write_json(output / 'route_pilot.json', report)

    def observe_room(position):
        region = session.env.scene.seg_map.get_room_instance_by_point(
            torch.tensor(list(position[:2]), dtype=torch.float32))
        if not report['observed_room_sequence'] or report['observed_room_sequence'][-1] != region:
            report['observed_room_sequence'].append(region)
        index = report['ordered_rooms_observed']
        if index < len(report['required_order']) and region == report['required_order'][index]:
            report['ordered_rooms_observed'] += 1

    save()
    for index, target in enumerate(path):
        while True:
            pose = controller.pose()
            observe_room(pose['position'])
            error = float(np.linalg.norm(target - np.asarray(pose['position'])[:2]))
            if error <= report['waypoint_tolerance_m']:
                report['waypoints_reached'] = index + 1
                break
            if len(report['commands']) >= budget.max_steps:
                report.update(status='pilot_budget_exhausted', stopped_waypoint=index)
                save()
                return report
            request = next_translation(pose['position'], pose['quaternion_xyzw'], target,
                minimum_m=report['minimum_command_m'])
            try:
                trial = command(request)
            except Exception as error:
                report.update(status='diagnostic_command_failed', stopped_waypoint=index,
                              error=f'{type(error).__name__}: {error}')
                save()
                raise
            report['commands'].append({'waypoint': index, 'command': request,
                'result': trial['result'], 'before_position_m': pose['position'],
                'after_position_m': controller.pose()['position']})
            if not trial['result']['success']:
                report.update(status='public_controller_guard_stopped', stopped_waypoint=index,
                    limitation='This particular source path failed; alternative legal routes remain untested.')
                save()
                return report
            save()
    report.update(status='source_waypoints_traversed',
        limitation='Observed room occupancy is not a stable arrival report, source-relation proof, door qualification, or model-latency calibration.')
    save()
    return report
