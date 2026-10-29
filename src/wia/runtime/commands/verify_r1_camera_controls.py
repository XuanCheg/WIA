"""Run isolated R1 optics/rotation diagnostics, then retain the web scene.

Run: python -m wia.runtime.commands.verify_r1_camera_controls with run_esi_robot
arguments including --web and --output-dir. Never attaches to a live episode.
"""
from pathlib import Path
import json
import sys
import traceback

import numpy as np


def main():
    from wia.runtime.commands.run_esi_robot import main as robot_main
    from wia.runtime.sim.esi_omnigibson import _array
    import wia.runtime.sim.esi_web as web
    original = web.run_esi_control
    args = sys.argv[1:]
    output = Path(args[args.index('--output-dir') + 1])
    diagnose_edges = '--diagnostic-edge-loop' in args
    if diagnose_edges:
        args.remove('--diagnostic-edge-loop')
        import wia.tasks.pickup_edges as edges
        initial_edge_errors = []
        original_place = edges.place_pickup_near_edge
        def collect_failure(*a, **kw):
            try:
                return original_place(*a, **kw)
            except ValueError as error:
                initial_edge_errors.append(str(error))
                print('[pickup-edge-diagnostic] ' + str(error), flush=True)
                return {'diagnostic_only_failure': str(error)}
        edges.place_pickup_near_edge = collect_failure

    def verify(session, metadata, controller, executor, **kwargs):
        report = {'schema': 'r1_camera_controls_v1', 'cameras': {}, 'objects': {}, 'motions': [], 'passed': False}
        try:
            for role, name in session.robot_camera_registry.items():
                sensor = session.robot.sensors[name]
                k = _array(sensor.intrinsic_matrix)
                measured = np.rad2deg(2 * np.arctan(np.array([sensor.image_width, sensor.image_height]) / (2 * np.diag(k)[:2])))
                report['cameras'][role] = {'size': [sensor.image_width, sensor.image_height], 'K': k.tolist(), 'fov_deg': measured.tolist()}
            for obj in session.env.scene.objects:
                if obj.name.startswith('wia_t07_') or obj.name in ('coffee_table_qlmqyy_0', 'countertop_tpuwys_0', 'desk_rzyfxk_1', 'bottom_cabinet_slgzfc_0'):
                    report['objects'][obj.name] = {'aabb': [_array(v).tolist() for v in obj.aabb], 'pose': [_array(v).tolist() for v in obj.get_position_orientation()]}
            for token in ('ROTATE_PITCH_POS', 'ROTATE_PITCH_NEG', 'ROTATE_ROLL_NEG', 'ROTATE_ROLL_POS'):
                receipt = controller.execute_metric(token, 5)
                report['motions'].append(receipt)
                if not receipt.get('completed'):
                    break
            desired = session.robot_camera_config.get('camera_fov_deg', {})
            report['optics_passed'] = all(np.allclose(row['fov_deg'], desired['head' if role == 'agentview_rgb' else 'wrist' if 'wrist' in role else 'chassis'], atol=.2) for role, row in report['cameras'].items())
            report['passed'] = report['optics_passed'] and len(report['motions']) == 4 and all(r['completed'] for r in report['motions'])
        except Exception:
            report['error'] = traceback.format_exc()
        (output / 'camera_controls_verification.json').write_text(json.dumps(report, indent=2) + '\n')
        if diagnose_edges:
            import time
            import importlib
            report['initial_edge_errors'] = initial_edge_errors
            report['diagnostic_only'] = True
            (output / 'camera_controls_verification.json').write_text(json.dumps(report, indent=2) + '\n')
            print('[edge-loop] ready for diagnostic commands', flush=True)
            sequence = 0
            while True:
                command_file = output / 'diagnostic_command.json'
                if not command_file.exists():
                    time.sleep(.2)
                    continue
                command = json.loads(command_file.read_text())
                command_file.unlink()
                if command['action'] == 'finish':
                    break
                result = {'command': command, 'sequence': sequence}
                try:
                    if command['action'] == 'place_edges':
                        edges = importlib.reload(importlib.import_module('wia.tasks.pickup_edges'))
                        session._category_initialization_active = True
                        placements = []
                        try:
                            for name, support in command['objects'].items():
                                obj = session.env.scene.object_registry('name', name)
                                sup = session.env.scene.object_registry('name', support)
                                try:
                                    placements.append(edges.place_pickup_near_edge(session, obj, sup))
                                except Exception as error:
                                    placements.append({'object': name, 'error': str(error)})
                            for _ in range(60):
                                controller.step_base_debug()
                        finally:
                            session._category_initialization_active = False
                        result['placements'] = placements
                    elif command['action'] == 'optics':
                        cameras = importlib.reload(importlib.import_module('wia.runtime.sim.esi_robot_cameras'))
                        result['cameras'] = {}
                        for role, name in session.robot_camera_registry.items():
                            sensor = session.robot.sensors[name]
                            kind = 'head' if role == 'agentview_rgb' else 'wrist' if 'wrist' in role else 'chassis'
                            fov = session.robot_camera_config['camera_fov_deg'][kind]
                            cameras._resize_camera(session, sensor, cameras.fov_render_dimensions((1024,576), *fov))
                            cameras.configure_camera_fov(session, sensor, *fov)
                        for _ in range(6):
                            session.pipeline.og.sim.render()
                        for role,name in session.robot_camera_registry.items():
                            sensor=session.robot.sensors[name]
                            k=_array(sensor.intrinsic_matrix)
                            result['cameras'][role]={'size':[sensor.image_width,sensor.image_height], 'K':k.tolist(),
                                'fov_deg':np.rad2deg(2*np.arctan(np.array([sensor.image_width,sensor.image_height])/(2*np.diag(k)[:2]))).tolist()}
                    else:
                        raise ValueError('Unknown isolated diagnostic command.')
                except Exception:
                    result['error'] = traceback.format_exc()
                (output / f'diagnostic_result_{sequence}.json').write_text(json.dumps(result, indent=2)+'\n')
                sequence += 1
        print('[camera-controls] verification saved; retaining isolated web scene', flush=True)
        return original(session, metadata, controller, executor, **kwargs)

    web.run_esi_control = verify
    return robot_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
