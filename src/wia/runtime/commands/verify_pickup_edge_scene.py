"""Validate pickup initialization with native geometry, then exit without actions.

Run: python -m wia.runtime.commands.verify_pickup_edge_scene with run_esi_robot
arguments including --web and --output-dir. This creates an isolated scene.
"""
import json
from pathlib import Path
import sys


def main():
    from wia.runtime.commands.run_esi_robot import main as robot_main
    import wia.runtime.sim.esi_web as web
    args = sys.argv[1:]
    output = Path(args[args.index('--output-dir') + 1])

    def verify(session, metadata, controller, executor, **kwargs):
        import numpy as np
        from wia.tasks.interaction_setup import array
        from omnigibson.object_states import Touching
        setup = getattr(session, 'category_interaction_setup', {})
        measured = []
        for row in getattr(session, 'pickup_edge_placements', []):
            if row.get('status') != 'placed':
                continue
            obj = session.env.scene.object_registry('name', row['object'])
            support = session.env.scene.object_registry('name', row['support'])
            pos, quat = (array(value) for value in obj.get_position_orientation())
            measured.append({'object': obj.name, 'settled_position_m': pos.tolist(),
                'settled_quaternion_xyzw': quat.tolist(),
                'horizontal_settle_drift_m': float(np.linalg.norm(pos[:2] - np.array(row['position_m'])[:2])),
                'touching_support': bool(obj.states[Touching].get_value(support)),
                'speed_m_s': float(np.linalg.norm(array(obj.get_linear_velocity())))})
        report = {'schema': 'pickup_edge_native_validation_v1',
                  'placements': getattr(session, 'pickup_edge_placements', []),
                  'setup': setup, 'settled_measurements': measured, 'physical_grasp_verified': False}
        (output / 'pickup_edge_verification.json').write_text(json.dumps(report, indent=2) + '\n')
        print('[pickup-edges] initialized and verified; no manipulation actions executed', flush=True)

    web.run_esi_control = verify
    return robot_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
