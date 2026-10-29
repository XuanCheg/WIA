#!/usr/bin/env python3
"""Offline native stop verification; never used in a measured model episode."""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import json
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from core.action_compiler import ParameterizedAction
    from wia.runtime.sim.esi_explore_scenes import TASK_INSTRUCTION
    import wia.runtime.sim.esi_web as web
    from wia.runtime.commands.run_esi_robot import main as robot_main

    args = sys.argv[1:] if argv is None else argv
    output = Path(args[args.index('--output-dir')+1])

    def verify(session, metadata, controller, executor, **kwargs):
        source = session.observation_source
        source.begin_episode(TASK_INSTRUCTION)
        records = []
        summary = {'schema': 'native_motion_recovery_v1', 'model_calls': 0,
                   'private_technical_check': True, 'verification_success': False}

        def execute(command, fixed=False, blocked=False):
            before = source.agent_observation()
            step = controller.step_base_debug
            try:
                if blocked:
                    # Controlled input fault: keep native physics and all other
                    # controllers running, but suppress the BASE offsets only.
                    controller.step_base_debug = lambda *args, **kwargs: step()
                result = (executor.execute(command['action']) if fixed else
                          executor.execute_macro(ParameterizedAction(**command),
                              resolution=executor.primitive_resolution, limits=executor.macro_limits))
            finally:
                controller.step_base_debug = step
            after = source.agent_observation()
            feedback = source.last_action_feedback.to_dict()
            row = {'command': command, 'fixed': fixed,
                   'input_fault': 'suppress_base_offsets' if blocked else None,
                   'native_physics_enabled': True, 'input_restored': True,
                   'action_step_delta': after.action_step-before.action_step,
                   'snapshot_delta': after.snapshot_id-before.snapshot_id,
                   'public_feedback': feedback, 'result': result}
            records.append(row)
            if row['action_step_delta'] != 1 or row['snapshot_delta'] != 1:
                raise RuntimeError('Action must produce exactly one canonical boundary.')
            if blocked:
                if (result['reason'] != 'base_motion_blocked'
                        or result.get('stop_stability_verified') is not True
                        or feedback['motion_stopped'] is not True):
                    raise RuntimeError('Native stalled BASE did not certify a measured stop.')
            elif not result['success']:
                raise RuntimeError('Native follow-up motion failed: '+str(result.get('reason')))
            print('[motion-check] '+json.dumps({'command': command, 'fixed': fixed,
                'reason': result['reason'], 'verified': result.get('stop_stability_verified')}), flush=True)

        try:
            execute({'action': 'BASE_FWD'}, fixed=True, blocked=True)
            execute({'action': 'BASE_FWD', 'distance_m': .05}, blocked=True)
            execute({'action': 'BASE_BACK', 'distance_m': .05})
            execute({'action': 'TRUNK_DOWN', 'distance_m': .04})
            execute({'action': 'TRUNK_UP', 'distance_m': .04})
            summary['verification_success'] = True
        except Exception as error:
            summary['error'] = str(error)
        finally:
            summary['records'] = records
            summary['qualification_success'] = summary['verification_success']
            (output/'motion_recovery_verification.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
            print('[motion-check] passed='+str(summary['verification_success']), flush=True)

    web.run_esi_control = verify
    return robot_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
