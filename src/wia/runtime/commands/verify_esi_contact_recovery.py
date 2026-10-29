#!/usr/bin/env python3
"""Validate contact-stop recovery with native physics and no model calls.

Run in the retained GPU shell with run_esi_robot.py arguments, plus
--replay-run /absolute/path/to/completed/model/run and optionally --keep-viewer.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import json
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def verify_contact_recovery(session, controller, executor, output, replay):
    """Replay real contact commands and require a certified stop before retreat."""
    from core.action_compiler import ParameterizedAction
    from core.feedback_semantics import CONTINUABLE_MOTION_ERRORS
    from wia.runtime.sim.esi_omnigibson import _array

    commands = [json.loads(line)['parsed_command'] for line in
                (replay / 'episode/steps.jsonl').read_text().splitlines()]
    source = session.observation_source
    report = {'schema': 'native_contact_recovery_v1', 'model_calls': 0,
              'private_technical_check': True, 'replay_source': str(replay),
              'commands': [], 'checks': {}, 'verification_success': False}
    checks = report['checks']

    def save():
        (output / 'contact_recovery_verification.json').write_text(json.dumps(
            report, indent=2, allow_nan=False,
            default=lambda value: _array(value).tolist()) + '\n')

    def execute(command):
        before = source.agent_observation()
        result = executor.execute_macro(ParameterizedAction(**command),
            resolution=executor.primitive_resolution, limits=executor.macro_limits)
        after = source.agent_observation()
        feedback = source.last_action_feedback.to_dict()
        row = {'command': command, 'result': result, 'public_feedback': feedback,
               'action_step_delta': after.action_step - before.action_step,
               'snapshot_delta': after.snapshot_id - before.snapshot_id}
        report['commands'].append(row)
        after.save(output / f"boundary_{len(report['commands']):02d}")
        save()
        if row['action_step_delta'] != 1 or row['snapshot_delta'] != 1:
            raise RuntimeError('One command must produce exactly one canonical boundary.')
        print('[contact-check]', command, result['reason'],
              result.get('stop_stability_verified'), flush=True)
        return result, feedback

    try:
        source.begin_episode(session.public_task_text)
        contact_result = None
        for command in commands:
            result, feedback = execute(command)
            if result['reason'] == 'arm_step_limit':
                contact_result = result
                checks['measured_step_limit_stops_motion'] = not result['success'] and not result['completed']
                checks['measured_stop_verified'] = result.get('stop_stability_verified') is True
                checks['public_stop_certified'] = feedback['motion_stopped'] is True
                checks['failure_category_is_continuable'] = result['reason'] in CONTINUABLE_MOTION_ERRORS
                if not all(checks.values()):
                    raise RuntimeError('Excessive measured motion did not produce a verified recoverable stop.')
                break
            if not result['success'] and not (
                    result['reason'] in CONTINUABLE_MOTION_ERRORS and feedback['motion_stopped'] is True):
                raise RuntimeError('Replay stopped with ' + str(result['reason']))
        if contact_result is None:
            raise RuntimeError('The replay did not produce a measured per-step motion limit stop.')
        faults = output / 'private_motion_diagnostics/faults.jsonl'
        diagnostics = [json.loads(line) for line in faults.read_text().splitlines()]
        contacts = [contact for event in diagnostics if event['reason'] == 'arm_step_limit'
                    for contact in event['contacts']
                    if 'task_glass_box/lid_link' in contact['body0']
                    or 'task_glass_box/lid_link' in contact['body1']]
        report['native_lid_contacts'] = contacts
        checks['real_lid_contact'] = bool(contacts)
        result, feedback = execute({'action': 'MV_UP', 'distance_m': .04})
        checks['new_retreat_command_completes'] = result['success'] and feedback['execution_status'] == 'completed'
        checks['failed_command_not_replayed'] = report['commands'][-1]['command']['action'] == 'MV_UP'
        report['verification_success'] = all(checks.values())
    except Exception as error:
        report['error'] = str(error)
    finally:
        save()
        print('[contact-check] passed=', report['verification_success'], flush=True)
    return report


def main(argv=None):
    import wia.runtime.sim.esi_web as web
    from wia.runtime.commands.run_esi_robot import main as robot_main

    args = list(sys.argv[1:] if argv is None else argv)
    replay_index = args.index('--replay-run')
    replay = Path(args[replay_index + 1])
    del args[replay_index:replay_index + 2]
    keep_viewer = '--keep-viewer' in args
    args = [argument for argument in args if argument != '--keep-viewer']
    output = Path(args[args.index('--output-dir') + 1])
    original_web = web.run_esi_control

    def verify(session, metadata, controller, executor, **kwargs):
        verify_contact_recovery(session, controller, executor, output, replay)
        if keep_viewer:
            return original_web(session, metadata, controller, executor, **kwargs)

    web.run_esi_control = verify
    return robot_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
