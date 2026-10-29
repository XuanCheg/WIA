#!/usr/bin/env python3
"""Private native geometry inventory, inside the retained GPU interactive shell."""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import json
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from wia.runtime.sim.esi_omnigibson import _array
    import wia.runtime.sim.esi_web as web
    from wia.runtime.commands.run_esi_robot import main as robot_main

    args = sys.argv[1:] if argv is None else argv
    output = Path(args[args.index("--output-dir") + 1])

    def inspect(session, metadata, controller, executor, **kwargs):
        objects = {}
        for obj in session.env.scene.objects:
            position, quaternion = obj.get_position_orientation()
            lower, upper = obj.aabb
            objects[obj.name] = {
                "category": obj.category, "model": getattr(obj, "model", None),
                "fixed_base": obj.fixed_base, "visual_only": obj.visual_only,
                "position": _array(position).tolist(), "quaternion_xyzw": _array(quaternion).tolist(),
                "aabb_min": _array(lower).tolist(), "aabb_max": _array(upper).tolist(),
            }
        cameras = {}
        for name, sensor_name in session.robot_camera_registry.items():
            position, quaternion = session.robot.sensors[sensor_name].get_position_orientation()
            cameras[name] = {"position": _array(position).tolist(), "quaternion_xyzw": _array(quaternion).tolist()}
        data = {"scene": session.scene_name, "room": session.room_name, "objects": objects,
                "cameras": cameras, "base": controller.pose(), "eef": controller.ee_pose(),
                "simulation_dt_s": controller.dt}
        (output / "private_scene_inventory.json").write_text(json.dumps(data, indent=2) + "\n")
        print("[inspect] private inventory saved", flush=True)

    web.run_esi_control = inspect
    return robot_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
