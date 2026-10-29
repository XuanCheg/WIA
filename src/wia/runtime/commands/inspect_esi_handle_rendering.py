#!/usr/bin/env python3
"""Inspect native handle geometry and side views without model calls.

Run in the retained GPU shell with the arguments of run_esi_robot.py.
The additional views and stage diagnostics are private validation artifacts.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import json
from pathlib import Path
import sys

import numpy as np

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def capture_handle_views(session, output):
    """Save private oblique views of the real rendered handle."""
    from core.record.images import save_png
    from wia.runtime.sim.esi_debug_cameras import camera_orientation
    from wia.runtime.sim.esi_omnigibson import _array

    box = session.task_evaluator.box
    handle_link = box.links.get('handle_link', box.links['lid_link'])
    mesh = next(mesh for mesh in handle_link.visual_meshes.values()
                if mesh.prim.GetName() == 'handle_bar')
    centre = np.mean(np.array([_array(value) for value in mesh.aabb]), axis=0)
    views = [('global_front', [-.40, -.45, .18]),
             ('global_left', [-.45, 0, .04]), ('global_right', [.35, -.25, .25])]
    for name, offset in views:
        sensor = session.global_cameras[name]
        position = centre + offset
        sensor.set_position_orientation(position=position.tolist(),
            orientation=camera_orientation(position, centre).tolist(), frame='world')
    for _ in range(5):
        session.pipeline.og.sim.render()
    for name, _ in views:
        save_png(output / (name + '.png'), session._sensor_rgb(session.global_cameras[name]))


def main(argv=None):
    from wia.native.esi_glass_box import hinge_angle_deg
    from wia.runtime.sim.esi_omnigibson import _array
    import wia.runtime.sim.esi_web as web
    import omnigibson.lazy as lazy
    from wia.runtime.commands.run_esi_robot import main as robot_main

    args = list(sys.argv[1:] if argv is None else argv)
    output = Path(args[args.index('--output-dir') + 1])
    original_web = web.run_esi_control

    def inspect(session, metadata, controller, executor, **kwargs):
        box = session.task_evaluator.box
        report = {'model_calls': 0, 'private_validation': True, 'samples': []}
        for steps in (0, 120):
            for _ in range(steps):
                controller.step_base_debug()
            sample = {'physics_steps_since_previous': steps,
                      'door_angle_deg': hinge_angle_deg(box, 'lid_hinge'),
                      'handle_angle_deg': hinge_angle_deg(box, 'handle_hinge'), 'meshes': {}}
            for link in box.links.values():
                for mesh in link.visual_meshes.values():
                    name = mesh.prim.GetName()
                    if name not in ('glass_lid', 'handle_bar', 'handle_left_post', 'handle_right_post'):
                        continue
                    geom = lazy.pxr.UsdGeom.Mesh(mesh.prim)
                    usd_matrix = geom.ComputeLocalToWorldTransform(lazy.pxr.Usd.TimeCode.Default())
                    points = geom.GetPointsAttr().Get()
                    usd_points = np.asarray([tuple(usd_matrix.Transform(lazy.pxr.Gf.Vec3d(*point))) for point in points])
                    sample['meshes'][name] = {
                        'native_aabb': [_array(value).tolist() for value in mesh.aabb],
                        'usd_aabb': [usd_points.min(axis=0).tolist(), usd_points.max(axis=0).tolist()],
                        'link_pose': [_array(value).tolist() for value in link.get_position_orientation()],
                        'mesh_pose': [_array(value).tolist() for value in mesh.get_position_orientation()],
                        'purpose': geom.ComputePurpose(), 'visibility': geom.ComputeVisibility(),
                        'double_sided': geom.GetDoubleSidedAttr().Get(),
                        'extent': str(geom.GetExtentAttr().Get()),
                        'material': lazy.pxr.UsdShade.MaterialBindingAPI(mesh.prim).GetDirectBinding().GetMaterialPath().pathString,
                        'xform_ops': [op.GetOpName() for op in geom.GetOrderedXformOps()],
                    }
            report['samples'].append(sample)
        session.invalidate_camera_cache()
        session.observation_source.capture(action_boundary=True).observation.save(output / 'robot_views')
        capture_handle_views(session, output)
        (output / 'render_inspection.json').write_text(json.dumps(report, indent=2) + '\n')
        print('[handle-render] native and USD geometry plus three side views saved', flush=True)
        return original_web(session, metadata, controller, executor, **kwargs)

    web.run_esi_control = inspect
    return robot_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
