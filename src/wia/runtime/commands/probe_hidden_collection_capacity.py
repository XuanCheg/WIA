"""Measure loaded tray capacity and carrying, separately from collection success.

Run with an explicit GPU UUID in the behavior environment:
python -B src/wia/runtime/commands/probe_hidden_collection_capacity.py --output DIR
Hamburgers are positioned only in diagnostic initialization. This is not a
five-object pickup certificate and never produces task-success evidence.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from wia.runtime.commands.run_esi_capture_worker import allocation_gpu
    from wia.runtime.scene_capture_adapters import install_gpu_selection, isolate_texture_cache
    gpu = allocation_gpu(0, execution_mode="local", renderer_order="pci")
    install_gpu_selection(gpu)
    from wia.runtime.sim.esi_omnigibson import _import_pipeline, DEFAULT_ESI_ROOT, _array
    pipeline = _import_pipeline(DEFAULT_ESI_ROOT)
    og = pipeline.og
    from omnigibson import simulator
    isolate_texture_cache(simulator, os.environ["OMNIGIBSON_APPDATA_PATH"])
    import numpy as np
    import yaml
    from scipy.spatial.transform import Rotation
    from omnigibson.utils.sampling_utils import raytest
    from wia.runtime.sim.esi_collection_setup import initialize_collection_tray
    from wia.runtime.sim.esi_gripper import R1ManipulationController
    from wia.authoring.converters.hidden_object_collection import _state
    from core.record.images import save_png
    config = yaml.safe_load((Path(og.example_config_path) / "r1_primitives.yaml").read_text())
    config["scene"] = {"type": "Scene"}
    robot_config = config["robots"][0]
    robot_config.update(name="robot", grasping_mode="assisted", position=[0., 0., .02])
    robot_config["reset_joint_pos"][6:10] = [.6981317007977318, -1.3962634015954636, -.6981317007977318, 0.]
    config["objects"] = [
        {"type": "DatasetObject", "name": "collection_plate", "category": "tray", "model": "gsxbym",
         "position": [3., 0., 1.], "fixed_base": False, "visual_only": False},
        {"type": "PrimitiveObject", "name": "collection_initial_support", "primitive_type": "Cube",
         "scale": [.25, .36, .04], "position": [3., 0., .8], "fixed_base": True},
    ]
    names = [f"hamburger_{i}" for i in range(5)]
    config["objects"].extend({"type": "DatasetObject", "name": name, "category": "hamburger",
        "model": "fyswch", "position": [4. + i, 0., 1.], "fixed_base": False, "visual_only": False}
        for i, name in enumerate(names))
    parameters = {"plate": {"name": "collection_plate", "right_eef_position_base_m": [.4, -.25, .8],
        "rim_axis": "y", "rim_penetration_m": .025,
        "usable_bounds_local_m": [[-.147, -.21, -.0049], [.147, .21, .17]]},
        "maximum_plate_tilt_deg": 10., "maximum_relative_speed_m_s": .03,
        "maximum_plate_speed_m_s": .03, "maximum_plate_angular_speed_rad_s": .1,
        "targets": names, "distractors": []}
    report = {"gpu": gpu, "stage": "constructing", "five_object_pickup_certified": False,
        "manual_target_attachment": False, "commands": []}
    save = lambda: (args.output / "capacity.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        env = og.Environment(configs=config)
        session = SimpleNamespace(robot=env.robots[0], env=env, pipeline=pipeline,
            _category_initialization_active=True, step_control=env.step)
        report["initialization"] = initialize_collection_tray(session, parameters)
        session._category_initialization_active = False
        controller = R1ManipulationController(session)
        plate = env.scene.object_registry("name", "collection_plate")
        position, quaternion = (_array(v) for v in plate.get_position_orientation())
        rotation = Rotation.from_quat(quaternion)
        heights = []
        report["deck_samples"] = []
        for x, y in ((-.06, -.12), (.06, -.12), (-.06, 0.), (.06, 0.), (-.06, .12), (.06, .12),
                     (-.14, 0.), (.14, 0.), (0., -.2), (0., .2)):
            hits = raytest((position + rotation.apply([x, y, .15])).tolist(),
                          (position + rotation.apply([x, y, -.05])).tolist(), only_closest=False)
            for hit in hits:
                if hit.get("hit") and str(hit["rigidBody"]).startswith(plate.prim_path + "/"):
                    normal = rotation.inv().apply(_array(hit["normal"]))
                    if normal[2] > .95:
                        height = float(rotation.inv().apply(_array(hit["position"]) - position)[2])
                        heights.append(height)
                        report["deck_samples"].append({"xy": [x, y], "height": height,
                                                       "up_dot": float(normal[2])})
        if not heights:
            raise ValueError("Native tray deck could not be measured.")
        deck = float(np.median(heights))
        parameters["plate"]["usable_bounds_local_m"][0][2] = deck
        report["deck_heights_local_m"] = heights
        report["measured_deck_local_z_m"] = deck
        report["stage"] = "initializing_capacity_inventory"
        for name, (x, y) in zip(names, ((-.061, -.12), (.061, -.12), (-.061, 0.), (.061, 0.), (-.061, .12))):
            burger = env.scene.object_registry("name", name)
            offset = _array(burger.get_position_orientation()[0])[2] - _array(burger.aabb[0])[2]
            burger.set_position_orientation(position=(position + rotation.apply([x, y, deck + offset + .012])).tolist(),
                orientation=quaternion.tolist())
            burger.keep_still()
        for _ in range(90):
            controller.step_base_debug()
        report["loaded_state"] = _state(session, parameters)
        report["stage"] = "loaded_carry"
        save()
        for token, value in (("BASE_FWD", .4), ("BASE_LEFT", .4), ("BASE_TURN_LEFT", 60.),
                             ("BASE_BACK", .4), ("BASE_RIGHT", .4), ("BASE_TURN_RIGHT", 60.)):
            command = controller.execute_metric(token, value)
            report["commands"].append(command)
            save()
            if not command["completed"]:
                break
        for _ in range(45):
            controller.step_base_debug()
        report["final_state"] = _state(session, parameters)
        report["all_five_retained"] = set(report["final_state"]["supported_objects"]) == set(names)
        report["stage"] = "complete"
        camera = og.sim.viewer_camera
        centre = _array(plate.get_position_orientation()[0])
        eye = centre + [.6, -.5, .75]
        z = eye - centre
        z /= np.linalg.norm(z)
        x = np.cross([0., 0., 1.], z)
        x /= np.linalg.norm(x)
        camera.set_position_orientation(position=eye.tolist(),
            orientation=Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z])).as_quat().tolist())
        for _ in range(8):
            og.sim.render()
        save_png(args.output / "loaded_tray.png", _array(camera.get_obs()[0]["rgb"]))
    except Exception as error:
        report.update(stage="failed", error=repr(error))
        if "session" in locals():
            report["initialization"] = getattr(session, "hidden_collection_initialization", None)
        raise
    finally:
        save()
        og.shutdown()


if __name__ == "__main__":
    main()
