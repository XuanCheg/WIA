"""Replay T07 arm commands with native physics and no model calls.

Run with run_esi_robot.py arguments plus --replay-run /absolute/trial/path
and optional --retreat-distance-m 0.01 (default: 0.02).
This diagnostic omits model waiting time and does not score task completion.
"""
import json
from pathlib import Path
import sys


def main():
    import numpy as np
    from core.action_compiler import ParameterizedAction
    from core.feedback_semantics import IK_REJECTION_ERRORS
    from wia.runtime.sim.esi_omnigibson import _array
    import wia.runtime.sim.esi_web as web
    from wia.runtime.commands.run_esi_robot import main as robot_main

    args = sys.argv[1:]
    index = args.index('--replay-run')
    replay = Path(args[index + 1])
    del args[index:index + 2]
    retreat_distance = .02
    if '--retreat-distance-m' in args:
        index = args.index('--retreat-distance-m')
        retreat_distance = float(args[index + 1])
        del args[index:index + 2]
    output = Path(args[args.index('--output-dir') + 1])

    def verify(session, metadata, controller, executor, **kwargs):
        source = session.observation_source
        report = {'schema': 't07_limit_stop_v1', 'model_calls': 0,
                  'replay_source': str(replay), 'model_waits_replayed': False,
                  'retreat_distance_m': retreat_distance,
                  'records': [], 'verification_success': False}

        def save():
            (output / 'limit_stop_verification.json').write_text(json.dumps(
                report, indent=2, default=lambda value: _array(value).tolist()) + '\n')

        def execute(command):
            before = source.agent_observation()
            result = executor.execute_macro(ParameterizedAction(**command),
                resolution=executor.primitive_resolution, limits=executor.macro_limits)
            after = source.agent_observation()
            feedback = source.last_action_feedback.to_dict()
            row = {'command': command, 'result': result, 'feedback': feedback,
                   'action_step_delta': after.action_step-before.action_step,
                   'snapshot_delta': after.snapshot_id-before.snapshot_id}
            report['records'].append(row)
            save()
            print('[limit-check]', command, result['reason'], feedback.get('motion_stopped'), flush=True)
            assert row['action_step_delta'] == 1 and row['snapshot_delta'] == 1
            return result, feedback

        try:
            source.begin_episode(session.public_task_text)
            commands = [json.loads(line)['parsed_command'] for line in
                        (replay / 'episode/steps.jsonl').read_text().splitlines()]
            for command in commands:
                result, feedback = execute(command)
                if result['reason'] in IK_REJECTION_ERRORS:
                    break
            else:
                # Reach the same boundary without relying on model-time drift.
                for _ in range(12):
                    result, feedback = execute(commands[-1])
                    if result['reason'] in IK_REJECTION_ERRORS:
                        break
            assert result['reason'] in IK_REJECTION_ERRORS, 'No native IK rejection reproduced.'
            assert result.get('stop_stability_verified') is True, 'Stop not measured stable.'
            assert feedback.get('motion_stopped') is True, 'Public stop not certified.'
            held = np.asarray(controller.ee_pose()['base_position'])
            for _ in range(12):
                controller.step_base_debug()
            drift = float(np.linalg.norm(np.asarray(controller.ee_pose()['base_position'])-held))
            report['held_pose_drift_m'] = drift
            assert drift < .001, 'Stopped arm continued moving.'
            # A fresh command retreats from the reached pose, never replays the failed request.
            result, feedback = execute({'action': 'MV_FWD', 'distance_m': retreat_distance})
            assert result['success'], 'Fresh retreat failed.'
            assert feedback.get('motion_stopped') is True
            result, feedback = execute({'action': 'MV_BACK', 'distance_m': .005})
            assert result['success'], 'Follow-up motion after retreat failed.'
            report['verification_success'] = True
        except Exception as error:
            report['error'] = f'{type(error).__name__}: {error}'
        finally:
            save()
            print('[limit-check] passed=', report['verification_success'], flush=True)

    web.run_esi_control = verify
    return robot_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
