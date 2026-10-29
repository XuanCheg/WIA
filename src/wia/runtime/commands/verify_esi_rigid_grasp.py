#!/usr/bin/env python3
"""Probe a Rigid Containment target with normal, contact-qualified R1 commands.

Run inside the retained GPU shell with run_esi_robot.py arguments, optionally
adding --keep-viewer and --target-scale 0.35 for a target-only size comparison.
Use --viewer-only --target-scale 0.35 to open that scene without probe actions.
Prefer --scene-file /absolute/path/to/scene.yaml --variant tabletop for generated
layouts; scaling then comes from the YAML rather than a second CLI multiplier.
Add --scene-patch /absolute/path/to/patch.json for initialization-only object
poses and fixed collidable support primitives; omit --viewer-only to probe it.
This private diagnostic preserves the original question and creates no
achievement evaluator or model calls.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import json
import copy
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def main(argv=None, *, initial_observation_capture=None):
    from core.action_compiler import ParameterizedAction
    from core.feedback_semantics import CONTINUABLE_MOTION_ERRORS
    from core.record.images import save_png
    from wia.runtime.sim.esi_omnigibson import ESISession, _array
    import wia.runtime.sim.esi_web as web
    import wia.runtime.sim.esi_arm as arm_control
    from wia.runtime.commands.run_esi_robot import main as robot_main

    args = list(sys.argv[1:] if argv is None else argv)
    viewer_only = "--viewer-only" in args
    keep_viewer = "--keep-viewer" in args
    args = [arg for arg in args if arg not in ("--keep-viewer", "--viewer-only")]
    target_scale = 1.0
    if "--target-scale" in args:
        index = args.index("--target-scale")
        target_scale = float(args[index + 1])
        del args[index:index + 2]
    scene_patch = None
    if "--scene-patch" in args:
        index = args.index("--scene-patch")
        scene_patch = json.loads(Path(args[index + 1]).read_text())
        del args[index:index + 2]
    output = Path(args[args.index("--output-dir") + 1])
    scale_override = target_scale
    scene_selection = None
    if "--scene-file" in args:
        from wia.tasks.loader import load_scene
        scene_path = args[args.index("--scene-file") + 1]
        variant = args[args.index("--variant") + 1] if "--variant" in args else "tabletop"
        scene_selection = load_scene(scene_path, variant=variant)
        target_key = scene_selection.definition.target
        for patch in scene_selection.definition.scene.variants[variant].patches:
            if patch.operation == "object_scale" and patch.binding == target_key:
                target_scale *= patch.multiplier
    original_web = web.run_esi_control
    original_configure = ESISession._configure_robot_observation
    original_initialize_trunk = arm_control._initialize_trunk

    def configure(session, config):
        if scale_override != 1.0:
            target_config = next(obj for obj in config["objects"] if obj["name"] == "obj_containee")
            target_config["scale"] = [float(value) * scale_override for value in target_config["scale"]]
        if scene_patch is not None:
            # This source reconstructs object poses during postprocessing.
            # Patch its in-memory initialization payload before construction;
            # the source JSON and all episode-time object motion stay intact.
            poses = session.payload["initial_poses"]
            poses = {pose["_key"]: pose for pose in poses} if isinstance(poses, list) else poses
            for name, pose in scene_patch["object_poses"].items():
                poses[name].update(copy.deepcopy(pose))
            config["objects"].extend(copy.deepcopy(scene_patch["added_primitives"]))
        original_configure(session, config)

    ESISession._configure_robot_observation = configure

    def initialize_trunk(session, base_config, arm_config):
        robot = session.robot
        dofs = _array(robot.controllers["trunk"].dof_idx).astype(int)
        initial = _array(robot.get_joint_positions())[dofs].copy()
        start = time.perf_counter()
        startup = {"original_wall_timeout_s": arm_config.timeout_wall_s,
                   "initialization_wall_timeout_s": 180.,
                   "original_simulation_timeout_s": arm_config.timeout_sim_s,
                   "initialization_simulation_timeout_s": 15.,
                   "target_joints_rad": arm_config.initial_trunk_joint_positions_rad,
                   "before_joints_rad": initial.tolist(),
                   "action_timeouts_unchanged": True}
        try:
            return original_initialize_trunk(session, base_config,
                replace(arm_config, timeout_wall_s=180., timeout_sim_s=15.))
        except Exception as error:
            startup["error"] = str(error)
            raise
        finally:
            startup["after_joints_rad"] = _array(robot.get_joint_positions())[dofs].tolist()
            startup["wall_elapsed_s"] = time.perf_counter() - start
            startup["base_position"] = _array(robot.get_position_orientation()[0]).tolist()
            startup["contacts"] = [{"body0": contact.body0, "body1": contact.body1}
                                   for contact in robot.contact_list()]
            (output / "trunk_initialization.json").write_text(json.dumps(startup, indent=2) + "\n")

    arm_control._initialize_trunk = initialize_trunk

    def verify(session, metadata, controller, executor, **kwargs):
        target = session.env.scene.object_registry("name", "obj_containee")
        source = session.observation_source
        robot, arm = controller.robot, controller.working_arm
        probe_scene_patch = scene_patch
        support_name = "probe_table_top"
        if scene_selection is not None:
            adapter = session.scene_adapter
            support_name = adapter._binding(scene_selection.definition.qualification.support).name
            probe_scene_patch = {"scene_file": str(scene_selection.path),
                "added_primitives": [item for item in adapter.applied if item["operation"] == "add_primitive"]}
        if viewer_only:
            if not getattr(session, '_capture_initial_agent_bundle_complete', False):
                source.begin_episode(metadata["question"])
            low, high = (_array(value) for value in target.aabb)
            initialization = {"mode": "viewer_only", "model_calls": 0,
                "probe_actions": 0, "original_question": metadata["question"],
                "target_scale_multiplier": target_scale,
                "target_scale": _array(target.scale).tolist(),
                "target_extent_m": (high - low).tolist(),
                "robot_scale": _array(robot.scale).tolist(),
                "container_scale_multiplier": 1.0}
            (output / "viewer_initialization.json").write_text(
                json.dumps(initialization, indent=2) + "\n")
            print("[rigid-grasp] viewer-only scene ready", flush=True)
            return original_web(session, metadata, controller, executor, **kwargs)
        report = {"schema": "private_rigid_grasp_probe_v1", "model_calls": 0,
                  "original_question": metadata["question"], "actions": [],
                  "snapshots": [], "grasp_attempts": [], "grasp_acquired": False,
                  "lift_verified": False, "diagnostic_only": True,
                  "target_scale_multiplier": target_scale,
                  "container_scale_multiplier": 1.0, "qa_scoring_enabled": False,
                  "scene_patch": probe_scene_patch}
        started = time.perf_counter()

        def save():
            report["wall_elapsed_s"] = time.perf_counter() - started
            (output / "grasp_probe.json").write_text(json.dumps(report, indent=2,
                allow_nan=False, default=lambda value: _array(value).tolist()) + "\n")

        def object_state(obj):
            pos, quat = obj.get_position_orientation()
            low, high = (_array(value) for value in obj.aabb)
            return {"name": obj.name, "category": obj.category,
                    "scale": _array(obj.scale).tolist(), "position": _array(pos).tolist(),
                    "quaternion_xyzw": _array(quat).tolist(),
                    "aabb_min": low.tolist(), "aabb_max": high.tolist(),
                    "extent_m": (high - low).tolist(), "visual_only": obj.visual_only,
                    "fixed_base": obj.fixed_base, "root_mass_kg": float(obj.root_link.mass)}

        def contacts():
            candidates, touching = robot._find_gripper_contacts(arm=arm)
            rays = robot._find_gripper_raycast_collisions(arm=arm)
            return {"contacted_link_paths": sorted(candidates),
                    "touching_fingers": {key: sorted(value) for key, value in touching.items()},
                    "raycast_link_paths": sorted(rays),
                    "finger_link_paths": [link.prim_path for link in robot.finger_links[arm]]}

        def snapshot(label):
            session.invalidate_camera_cache()
            captured = source.capture(action_boundary=False)
            observation = captured.observation
            folder = output / label
            observation.save(folder)
            row = {"label": label, "target": object_state(target),
                   "eef": controller.ee_pose(), "base": controller.pose(),
                   "gripper": controller.debug_gripper_state(), "contacts": contacts()}
            report["snapshots"].append(row)
            human_views = captured.human_views()
            for name in session.global_cameras:
                if name in human_views:
                    save_png(folder / (name + ".png"), human_views[name])
            save()
            return row

        def move(action, magnitude=None):
            command = {"action": action}
            if magnitude is not None:
                command["distance_m"] = float(magnitude)
            result = executor.execute_macro(ParameterizedAction(**command),
                resolution=executor.primitive_resolution, limits=executor.macro_limits)
            row = {"command": command, "result": result,
                   "target": object_state(target), "contacts": contacts(),
                   "gripper": controller.debug_gripper_state(),
                   "public_feedback": source.last_action_feedback.to_dict()}
            report["actions"].append(row)
            save()
            print("[rigid-grasp]", command, result["reason"], row["gripper"], flush=True)
            if not result["success"] and not (
                    result["reason"] in CONTINUABLE_MOTION_ERRORS
                    and source.last_action_feedback.motion_stopped is True):
                raise RuntimeError("Guarded action stopped: " + result["reason"])
            return row

        def align(point, axes=(0, 1, 2)):
            directions = [("MV_FWD", "MV_BACK"), ("MV_LEFT", "MV_RIGHT"), ("MV_UP", "MV_DOWN")]
            for axis in axes:
                maximum_step = .08
                for _ in range(64):
                    pose = controller.pose()
                    desired = Rotation.from_quat(pose["quaternion_xyzw"]).inv().apply(
                        np.asarray(point) - pose["position"])
                    delta = float(desired[axis] - controller.ee_pose()["base_position"][axis])
                    if abs(delta) < .003:
                        break
                    row = move(directions[axis][delta < 0], min(maximum_step, abs(delta)))
                    if not row["result"]["success"]:
                        if maximum_step <= .005:
                            raise RuntimeError("The approach remains blocked at the 5 mm action quantum.")
                        maximum_step = max(.005, maximum_step / 2)
                else:
                    raise RuntimeError("Private alignment exceeded its bounded budget.")

        try:
            if target is None or target.visual_only or target.fixed_base:
                raise RuntimeError("The source target must be movable and collidable.")
            source.begin_episode(metadata["question"])
            report["initial_objects"] = {name: object_state(session.env.scene.object_registry("name", name))
                                         for name in session.task_object_names}
            report["robot_scale"] = _array(robot.scale).tolist()
            report["eef_to_fingertip_lengths_m"] = robot.eef_to_fingertip_lengths[arm]
            if probe_scene_patch is not None:
                report["support_objects"] = {
                    item["name"]: object_state(session.env.scene.object_registry("name", item["name"]))
                    for item in probe_scene_patch["added_primitives"]}
                support = session.env.scene.object_registry("name", support_name)
                support_top = float(_array(support.aabb[1])[2])
                support_rows = {}
                for name in session.task_object_names:
                    obj = session.env.scene.object_registry("name", name)
                    support_rows[name] = {
                        "bottom_minus_table_top_m": float(_array(obj.aabb[0])[2] - support_top),
                        "linear_velocity_m_s": _array(obj.get_linear_velocity()).tolist(),
                        "contacts": [{"body0": contact.body0, "body1": contact.body1}
                                     for contact in obj.contact_list()]}
                report["initial_support_check"] = support_rows
                (output / "table_setup.json").write_text(json.dumps({
                    "configuration": probe_scene_patch, "objects": report["initial_objects"],
                    "support_objects": report["support_objects"], "support": support_rows}, indent=2) + "\n")
                if any(abs(row["bottom_minus_table_top_m"]) > .015
                       or np.linalg.norm(row["linear_velocity_m_s"]) > .02
                       for row in support_rows.values()):
                    raise RuntimeError("Task objects must settle on the support table before grasp probing.")
            snapshot("initial")
            move("RELEASE")
            report["full_open_aperture_m"] = controller.gripper_state()["gripper_width_m"]
            length = float(np.mean(list(robot.eef_to_fingertip_lengths[arm].values())))
            top = np.array(target.aabb[1].cpu().numpy())
            centre = (_array(target.aabb[0]) + top) / 2
            tip_offset = Rotation.from_quat(controller.ee_pose()["world_quaternion_xyzw"]).apply([0., 0., length])
            approach = np.array([centre[0], centre[1], top[2] + .02]) - tip_offset
            # Lowering the existing trunk first improves low-object arm reach.
            trunk = controller.trunk_controller
            if trunk is not None:
                for _ in range(8):
                    height = float(trunk.kinematics.pose(trunk.joints())[0][2])
                    remaining = height - trunk.config.minimum_height_m - .01
                    if remaining < .02:
                        break
                    row = move("TRUNK_DOWN", min(.03, remaining))
                    if row["result"]["reason"] == "trunk_height_limit":
                        break
            align(approach, axes=(0, 1, 2))
            snapshot("above_target")
            # Test a shallow local surface patch; global AABB is not a grasp width.
            for index, penetration in enumerate((.005, .020, .040)):
                move("RELEASE")
                low, high = (_array(value) for value in target.aabb)
                centre = (low + high) / 2
                tip_offset = Rotation.from_quat(controller.ee_pose()["world_quaternion_xyzw"]).apply([0., 0., length])
                point = np.array([centre[0], centre[1], high[2] - penetration]) - tip_offset
                align(point, axes=(0, 1, 2))
                snapshot(f"attempt_{index + 1}_open")
                row = move("GRASP")
                report["grasp_attempts"].append({"penetration_m": penetration,
                    "gripper": row["gripper"], "contacts": row["contacts"]})
                snapshot(f"attempt_{index + 1}_closed")
                held = row["gripper"]["holding"] is True and row["gripper"]["object_name"] == target.name
                if held:
                    report["grasp_acquired"] = True
                    before = object_state(target)
                    move("MV_UP", .075)
                    after = snapshot("lifted")
                    report["lift_m"] = after["target"]["aabb_min"][2] - before["aabb_min"][2]
                    report["lift_verified"] = (after["gripper"]["object_name"] == target.name
                                               and report["lift_m"] >= .05)
                    break
            report["probe_completed"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            report["probe_completed"] = False
            try:
                snapshot("stopped")
            except Exception as snapshot_error:
                report["snapshot_error"] = str(snapshot_error)
        finally:
            save()
            print("[rigid-grasp] result", {key: report.get(key) for key in
                ("grasp_acquired", "lift_verified", "lift_m", "probe_completed", "error")}, flush=True)
            if keep_viewer:
                return original_web(session, metadata, controller, executor, **kwargs)

    web.run_esi_control = verify
    return robot_main(args, initial_observation_capture=initial_observation_capture)


if __name__ == "__main__":
    raise SystemExit(main())
