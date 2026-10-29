"""Measure and render native collection assets without changing dataset files.

Run in the behavior environment with an explicit CUDA_VISIBLE_DEVICES UUID:
python -B src/wia/runtime/commands/audit_hidden_collection_assets.py --shard 0 --shards 3 --output DIR
Fixed roots are used only for this geometry audit, never as grasp qualification.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--asset", action="append",
                        help="Explicit category/model; repeat to audit a bounded candidate set.")
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
    dataset = Path(os.environ["OMNIGIBSON_DATA_PATH"]) / "behavior-1k-assets/objects"
    assets = [("tray", p.name) for p in sorted((dataset / "tray").iterdir())]
    assets += [("plate", p.name) for p in sorted((dataset / "plate").iterdir())[:9]]
    assets += [("club_sandwich", "ablfyd"), ("hamburger", "fyswch")]
    if args.asset:
        assets = [tuple(value.split("/", 1)) for value in args.asset]
    selected = assets[args.shard::args.shards]
    config = {"scene": {"type": "Scene"}, "robots": [], "task": {"type": "DummyTask"},
        "render": {"viewer_width": 640, "viewer_height": 480},
        "objects": [{"type": "DatasetObject", "name": f"audit_{category}_{model}",
            "category": category, "model": model, "position": [3. * i, 0., 1.],
            "orientation": [0., 0., 0., 1.], "fixed_base": True, "visual_only": False}
            for i, (category, model) in enumerate(selected)]}
    report = {"gpu": gpu, "fixed_roots_for_geometry_only": True, "assets": [], "complete": False}

    def save():
        (args.output / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        env = og.Environment(configs=config)
        camera = og.sim.viewer_camera
        for category, model in selected:
            obj = env.scene.object_registry("name", f"audit_{category}_{model}")
            low, high = (_array(v) for v in obj.aabb)
            centre = (low + high) / 2
            extent = high - low
            distance = max(.4, float(max(extent)) * 1.9)
            eye = centre + np.array([.7, -.85, 1.]) * distance
            backward = eye - centre
            backward /= np.linalg.norm(backward)
            right = np.cross([0., 0., 1.], backward)
            right /= np.linalg.norm(right)
            up = np.cross(backward, right)
            camera.set_position_orientation(position=eye.tolist(),
                orientation=Rotation.from_matrix(np.column_stack([right, up, backward])).as_quat().tolist())
            for _ in range(8):
                og.sim.render()
            frame = camera.get_obs()[0]["rgb"]
            path = args.output / f"{category}_{model}.png"
            save_png(path, _array(frame))
            report["assets"].append({"category": category, "model": model,
                "extent_m": extent.tolist(), "aabb": [low.tolist(), high.tolist()],
                "scale": _array(obj.scale).tolist(), "root_mass_kg": float(obj.root_link.mass),
                "links": list(obj.links), "image": str(path)})
            save()
            print(json.dumps(report["assets"][-1]), flush=True)
        report["complete"] = True
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        save()
        og.shutdown()


if __name__ == "__main__":
    main()
