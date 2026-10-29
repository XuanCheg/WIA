"""Inspect native room furniture and render support-layout proposals.

Run with an explicit GPU UUID in the behavior environment:
python -B src/wia/runtime/commands/audit_hidden_collection_rooms.py --scene house_single_floor
    --room living_room_0 --room kitchen_0 --output DIR
This read-only scene inspection is not a navigation or grasp certificate.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(WIA_RUNTIME_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--room", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from wia.runtime.commands.run_esi_capture_worker import allocation_gpu
    from wia.runtime.scene_capture_adapters import install_gpu_selection, isolate_texture_cache
    gpu = allocation_gpu(0, execution_mode="local", renderer_order="pci")
    install_gpu_selection(gpu)
    import omnigibson as og
    from omnigibson import simulator
    isolate_texture_cache(simulator, os.environ["OMNIGIBSON_APPDATA_PATH"])
    import numpy as np
    from scipy.spatial.transform import Rotation
    from wia.runtime.sim.esi_omnigibson import _array
    from core.record.images import save_png
    config = {"scene": {"type": "InteractiveTraversableScene", "scene_model": args.scene,
        "load_room_instances": args.room, "load_object_categories": None,
        "not_load_object_categories": None},
        "robots": [], "task": {"type": "DummyTask"},
        "render": {"viewer_width": 1280, "viewer_height": 960}}
    result = {"gpu": gpu, "config": config, "complete": False}
    try:
        env = og.Environment(configs=config)
        result["objects"] = []
        for obj in env.scene.objects:
            result["objects"].append({"name": obj.name, "category": obj.category,
                "model": getattr(obj, "model", None), "rooms": obj.in_rooms,
                "pose": [_array(v).tolist() for v in obj.get_position_orientation()],
                "aabb": [_array(v).tolist() for v in obj.aabb]})
        camera = og.sim.viewer_camera
        camera.focal_length = float(camera.horizontal_aperture) / (2. * np.tan(np.deg2rad(60.)))
        for room in args.room:
            floors = [r for r in result["objects"] if room in (r["rooms"] or [])
                      and r["category"] == "floors"]
            if not floors:
                continue
            low = np.min([r["aabb"][0] for r in floors], axis=0)
            high = np.max([r["aabb"][1] for r in floors], axis=0)
            centre = (low + high) / 2
            for tag, delta in (("overhead", [0., 0., 2.4]),
                               ("oblique", [1., -1., 2.2])):
                eye = centre + delta
                z = eye - centre
                z /= np.linalg.norm(z)
                up = [0., 1., 0.] if tag == "overhead" else [0., 0., 1.]
                x = np.cross(up, z)
                x /= np.linalg.norm(x)
                camera.set_position_orientation(position=eye.tolist(),
                    orientation=Rotation.from_matrix(
                        np.column_stack([x, np.cross(z, x), z])).as_quat().tolist())
                for _ in range(8):
                    og.sim.render()
                save_png(args.output / f"{room}_{tag}.png", _array(camera.get_obs()[0]["rgb"]))
        result["complete"] = True
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        (args.output / "rooms.json").write_text(json.dumps(result, indent=2) + "\n")
        og.shutdown()


if __name__ == "__main__":
    main()
