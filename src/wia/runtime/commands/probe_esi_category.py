#!/usr/bin/env python3
"""Probe one private ESI sample with the existing native R1 stack.

Run inside the interactive GPU shell:
python src/wia/runtime/commands/probe_esi_category.py --sample SAMPLE.json --output-dir RUN
Optional --add-robot diagnoses robot-free task builders without editing ESI.
All scene truth and scripted measurements stay in private research artifacts.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"native_type": type(value).__name__}


def write(path, data):
    path.write_text(json.dumps(plain(data), indent=2) + "\n")


def inventory(session):
    from wia.runtime.sim.esi_omnigibson import _array
    objects = {}
    names = set(session.task_object_names)
    def velocity(obj, method):
        if obj.fixed_base or obj.visual_only:
            return None
        try:
            return _array(getattr(obj, method)())
        except AttributeError:
            return None
    for obj in session.env.scene.objects:
        if obj in session.env.robots:
            continue
        position, quat = obj.get_position_orientation()
        low, high = obj.aabb
        task_object = obj.name in names or obj.name.startswith(("render_item_", "render_mirror_", "cover_small_item_render_", "hidden_obj_"))
        contacts = None
        if task_object and not obj.visual_only:
            try:
                contacts = [{"body0": c.body0, "body1": c.body1} for c in obj.contact_list()]
            except AttributeError:
                pass
        objects[obj.name] = {"category": obj.category, "model": getattr(obj, "model", None),
            "task_object": task_object, "prim_path": obj.prim_path, "contacts": contacts,
            "position": _array(position), "quaternion_xyzw": _array(quat),
            "aabb_min": _array(low), "aabb_max": _array(high),
            "scale": _array(obj.scale), "fixed_base": obj.fixed_base, "visual_only": obj.visual_only,
            "linear_velocity": velocity(obj, "get_linear_velocity"),
            "angular_velocity": velocity(obj, "get_angular_velocity")}
    return objects


def place_robot(session):
    """Choose an approximate free stance near the unchanged source camera."""
    import torch
    from wia.runtime.sim.esi_omnigibson import _array
    from wia.tasks.support_sampling import chassis_footprint, footprint_clearance

    scene, robot = session.env.scene, session.env.robots[0]
    trav, segmentation = scene.trav_map, scene.seg_map
    free = _array(trav.floor_map[0]) == 255
    indices = np.stack(np.indices(free.shape), axis=-1).reshape(-1, 2)
    world = _array(trav.map_to_world(torch.as_tensor(indices)))
    if session.room_name != "full_scene" and session.room_name in segmentation.room_ins_name_to_ins_id:
        seg_indices = (world[:, ::-1] / segmentation.map_resolution + segmentation.map_size / 2).astype(int)
        valid = np.all((seg_indices >= 0) & (seg_indices < segmentation.map_size), axis=1)
        room = np.zeros(len(indices), dtype=bool)
        room[valid] = _array(segmentation.room_ins_map)[tuple(seg_indices[valid].T)] == segmentation.room_ins_name_to_ins_id[session.room_name]
        free &= room.reshape(free.shape)
    floor_z = float(trav.floor_heights[0])
    _, extent, radius = chassis_footprint(robot)
    for obj in scene.objects:
        if obj is robot or obj.visual_only or obj.category in ("floors", "floor", "ceilings", "ceiling", "walls", "wall"):
            continue
        low, high = (_array(v) for v in obj.aabb)
        if high[2] <= floor_z + .04 or low[2] >= floor_z + 1.5:
            continue
        corners = _array(trav.world_to_map(np.asarray([low[:2], high[:2]])))
        begin, end = np.maximum(np.min(corners, axis=0), 0), np.minimum(np.max(corners, axis=0) + 1, free.shape)
        free[begin[0]:end[0], begin[1]:end[1]] = False
    free = footprint_clearance(free, trav.map_resolution, radius)
    cells = np.argwhere(free)
    if not len(cells):
        return {"accepted": False, "reason": "no_approximate_free_chassis_stance",
                "chassis_extent_m": extent, "clearance_radius_m": radius}
    points = _array(trav.map_to_world(torch.as_tensor(cells)))
    reference = np.asarray(session.scene_camera_position)
    choice = points[int(np.argmin(np.linalg.norm(points - reference[:2], axis=1)))]
    direction = Rotation.from_quat(session.scene_camera_quaternion).apply([0, 0, -1])
    yaw = float(np.arctan2(direction[1], direction[0]))
    old_pos, old_quat = robot.get_position_orientation()
    position = [float(choice[0]), float(choice[1]), floor_z + .1]
    robot.set_position_orientation(position=torch.tensor(position), orientation=torch.tensor([0., 0., np.sin(yaw/2), np.cos(yaw/2)], dtype=torch.float32))
    robot.keep_still()
    return {"accepted": True, "old_position": old_pos, "old_quaternion": old_quat,
            "position": position, "yaw_deg": np.rad2deg(yaw), "reference_camera_position": reference,
            "chassis_extent_m": extent, "clearance_radius_m": radius,
            "reference_xy_offset_m": float(np.linalg.norm(choice - reference[:2])),
            "scope": "Approximate chassis clearance only; not a grasp or navigation certificate."}


def grasp_probe(session, controller, executor, target_name, output, *, vertical_first=False, base_forward_m=.08):
    """Bounded private alignment through existing macros; no object teleporting."""
    from core.action_compiler import ParameterizedAction, command_bounds
    from core.record.images import save_png
    from wia.runtime.sim.esi_omnigibson import _array

    target = session.env.scene.object_registry("name", target_name)
    result = {"diagnostic_only": True, "target": target_name, "actions": [],
              "grasp_acquired": False, "lift_verified": False, "object_pose_commands": 0,
              "release_verified": False, "grasp_mode": controller.debug_gripper_state()["grasp_mode"],
              "approach_axis_order": [2, 0, 1] if vertical_first else [0, 1, 2]}

    def move(action, distance=None):
        command = {"action": action}
        if distance is not None:
            command["distance_m"] = float(distance)
        feedback = executor.execute_macro(ParameterizedAction(**command),
            resolution=executor.primitive_resolution, limits=executor.macro_limits)
        result["actions"].append({"command": command, "feedback": feedback,
                                  "private_gripper": controller.debug_gripper_state()})
        write(output / "grasp_probe.json", result)
        if not feedback["success"]:
            raise RuntimeError("Guarded action stopped: " + feedback["reason"])

    try:
        if target is None or target.fixed_base or target.visual_only:
            raise RuntimeError("Target must be movable and collidable.")
        low, high = (_array(v) for v in target.aabb)
        result["initial_aabb_min"] = low
        result["source_scale"] = _array(target.scale)
        move("RELEASE")
        if session.task_module.TASK_NAME == "action_order_inference":
            maximum = command_bounds("BASE_FWD", executor.primitive_resolution, executor.macro_limits)["maximum"]
            remaining = base_forward_m
            while remaining > 1e-8:
                chunk = min(maximum, remaining)
                move("BASE_FWD", chunk)
                remaining -= chunk
        length = float(np.mean(list(session.robot.eef_to_fingertip_lengths[controller.working_arm].values())))
        offset = Rotation.from_quat(controller.ee_pose()["world_quaternion_xyzw"]).apply([0., 0., length])
        center = (low + high) / 2
        point = np.array([center[0], center[1], high[2] + .025]) - offset

        def align(destination):
            directions = [("MV_FWD", "MV_BACK"), ("MV_LEFT", "MV_RIGHT"), ("MV_UP", "MV_DOWN")]
            for axis in result["approach_axis_order"]:
                for _ in range(24):
                    base = controller.pose()
                    desired = Rotation.from_quat(base["quaternion_xyzw"]).inv().apply(destination-base["position"])
                    delta = desired[axis] - controller.ee_pose()["base_position"][axis]
                    if abs(delta) <= .003:
                        break
                    move(directions[axis][delta < 0], min(.05, abs(delta)))
                else:
                    raise RuntimeError("Bounded alignment budget exhausted.")
        align(point)
        for penetration in (.005, .020):
            low, high = (_array(v) for v in target.aabb)
            center = (low + high) / 2
            offset = Rotation.from_quat(controller.ee_pose()["world_quaternion_xyzw"]).apply([0., 0., length])
            align(np.array([center[0], center[1], high[2]-penetration])-offset)
            move("GRASP")
            held = controller.debug_gripper_state()
            if held["holding"] is True and held["object_name"] == target_name:
                result["grasp_acquired"] = True
                before = _array(target.aabb[0])
                move("MV_UP", .06)
                result["lift_m"] = float(_array(target.aabb[0])[2]-before[2])
                result["lift_verified"] = result["lift_m"] >= .04 and controller.debug_gripper_state()["object_name"] == target_name
                move("MV_DOWN", .06)
                move("RELEASE")
                for _ in range(30):
                    controller.step_base_debug()
                result["release_verified"] = controller.debug_gripper_state()["holding"] is False
                result["final_aabb_min"] = _array(target.aabb[0])
                result["final_linear_speed_m_s"] = float(np.linalg.norm(_array(target.get_linear_velocity())))
                result["final_contacts"] = [{"body0": c.body0, "body1": c.body1} for c in target.contact_list()]
                break
            move("RELEASE")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for name, rgb in session.observe(include_wrist=True, include_scene=True).items():
            save_png(output / f"grasp_final_{name}.png", rgb)
        write(output / "grasp_probe.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--add-robot", action="store_true")
    parser.add_argument("--upstream-effect", action="store_true")
    parser.add_argument("--skip-upstream-effect", action="store_true")
    parser.add_argument("--tabletop", action="store_true")
    parser.add_argument("--robot-position", nargs=3, type=float)
    parser.add_argument("--robot-yaw-deg", type=float)
    parser.add_argument("--late-robot-position", nargs=3, type=float)
    parser.add_argument("--late-robot-yaw-deg", type=float, default=0.)
    parser.add_argument("--grasp-target")
    parser.add_argument("--grasp-vertical-first", action="store_true")
    parser.add_argument("--grasp-base-forward-m", type=float, default=.08)
    parser.add_argument("--cloth-assisted", action="store_true")
    parser.add_argument("--cloth-attachment-config-fix", action="store_true")
    parser.add_argument("--skip-redundant-cloth-restart", action="store_true")
    parser.add_argument("--refresh-native-render-products", action="store_true")
    parser.add_argument("--filter-fluid-scaffold-robot", action="store_true")
    parser.add_argument("--trunk-timeout-wall-s", type=float)
    parser.add_argument("--retain-trunk-posture", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("Native probes require an interactive PBS compute shell.")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "starting", "source": str(args.sample), "job_id": os.environ["PBS_JOBID"],
              "pid": os.getpid(), "model_calls": 0, "add_robot": args.add_robot, "tabletop": args.tabletop,
              "robot_camera_resolution": [256, 256], "changes": [], "native_actions": []}
    write(output / "probe.json", report)
    from wia.runtime.sim.esi_omnigibson import ESISession
    import wia.runtime.sim.esi_web as web
    from wia.runtime.commands.run_esi_robot import main as robot_main
    import wia.runtime.sim.esi_arm as arm_module
    from dataclasses import replace
    original_trunk = arm_module._initialize_trunk

    def initialize_trunk(session, base_config, arm_config):
        from wia.runtime.sim.esi_omnigibson import _array
        if args.trunk_timeout_wall_s:
            arm_config = replace(arm_config, timeout_wall_s=args.trunk_timeout_wall_s)
            report["changes"].append(f"Initialization-only trunk wall timeout {args.trunk_timeout_wall_s:g} s; motion limits unchanged.")
        dofs = _array(session.robot.controllers["trunk"].dof_idx).astype(int)
        before = _array(session.robot.get_joint_positions())[dofs]
        began = time.monotonic()
        try:
            result = original_trunk(session, base_config, arm_config)
            report["trunk_initialization"] = result
            return result
        except Exception as exc:
            report["trunk_initialization_failure"] = {"error": str(exc), "before": before,
                "after": _array(session.robot.get_joint_positions())[dofs],
                "target": arm_config.initial_trunk_joint_positions_rad,
                "elapsed_wall_s": time.monotonic()-began,
                "timeout_wall_s": arm_config.timeout_wall_s, "timeout_sim_s": arm_config.timeout_sim_s,
                "contacts": [{"body0": c.body0, "body1": c.body1} for c in session.robot.contact_list()]}
            write(output / "probe.json", report)
            raise

    arm_module._initialize_trunk = initialize_trunk
    original_configure = ESISession._configure_robot_observation
    original_initialize = ESISession._initialize_scene
    original_close = ESISession.close

    def close(session):
        error_type, error, _ = sys.exc_info()
        if error is not None:
            report.update(status="failed", error=f"{error_type.__name__}: {error}",
                          traceback=traceback.format_exc(), elapsed_wall_s=time.monotonic()-started)
            write(output / "probe.json", report)
        return original_close(session)

    def configure(session, config):
        if args.cloth_assisted:
            from omnigibson.macros import gm
            with gm.unlocked():
                gm.AG_CLOTH = True
            report["changes"].append("Enable the installed native cloth assisted-grasp facility before cloth creation.")
        if args.cloth_attachment_config_fix:
            from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
            original_link_init = RigidDynamicPrim.__init__
            def attachment_link_init(link, relative_prim_path, name, load_config=None):
                if name.endswith(":attachment_point"):
                    load_config = dict(load_config or {})
                    load_config.setdefault("xform_props_pre_loaded", False)
                original_link_init(link, relative_prim_path, name, load_config=load_config)
            RigidDynamicPrim.__init__ = attachment_link_init
            report["changes"].append("Private compatibility fix: supply missing xform_props_pre_loaded=False only to the native cloth attachment link.")
        report["original_robot_count"] = len(config.get("robots", []))
        report["original_object_specs"] = config.get("objects", [])
        report["original_scene_config"] = config.get("scene", {})
        if not config.get("robots") and args.add_robot:
            config["robots"] = session.pipeline.build_env_config(session.scene_name, session.room_name, "R1", [])['robots']
            report["changes"].append("Inject existing R1 configuration into the originally robot-free task builder.")
        direction = Rotation.from_quat(session.scene_camera_quaternion).apply([0, 0, -1])
        session.robot_position = (float(session.scene_camera_position[0]), float(session.scene_camera_position[1]), .1)
        session.robot_yaw_deg = float(np.rad2deg(np.arctan2(direction[1], direction[0])))
        if args.tabletop:
            session.robot_position, session.robot_yaw_deg = (-5.11, 15.10, .1), 90.
        if args.robot_position is not None:
            session.robot_position = tuple(args.robot_position)
            report["changes"].append("Use a private alternative robot stance; task-object layout remains unchanged.")
        if args.robot_yaw_deg is not None:
            session.robot_yaw_deg = args.robot_yaw_deg
        original_configure(session, config)
        write(output / "probe.json", report)

    def initialize(session, pipeline):
        original_postprocess = pipeline.task_postprocess_env
        def postprocess(module, env, payload, camera_info, task_state):
            original_cloth_clock = None
            original_fluid_box = None
            if args.filter_fluid_scaffold_robot and module.TASK_NAME == "liquid_volume":
                import omnigibson.lazy as lazy
                original_fluid_box = module.generate_box
                def generate_filtered_box(*box_args, **box_kwargs):
                    result = original_fluid_box(*box_args, **box_kwargs)
                    offset = box_kwargs.get("index_offset", 0)
                    paths = []
                    for index in range(offset, offset+4):
                        root = pipeline.og.sim.stage.GetPrimAtPath(f"/World/plane_{index}")
                        paths.extend(str(prim.GetPath()) for prim in lazy.pxr.Usd.PrimRange(root)
                                     if prim == root or prim.HasAPI(lazy.pxr.UsdPhysics.CollisionAPI))
                    for robot in env.robots:
                        for link in robot.links.values():
                            pairs = lazy.pxr.UsdPhysics.FilteredPairsAPI.Apply(link.prim).CreateFilteredPairsRel()
                            for path in paths:
                                pairs.AddTarget(lazy.pxr.Sdf.Path(path))
                    report["fluid_scaffold_robot_filtered_paths"] = paths
                    return result
                module.generate_box = generate_filtered_box
                report["changes"].append("Filter only R1 versus synthetic fluid-pool planes; retain cup, water and room collision geometry.")
            if args.skip_redundant_cloth_restart and module.TASK_NAME == "deformable":
                original_cloth_clock = module._configure_sim_for_cloth_drop
                def configure_cloth_clock():
                    sim = pipeline.og.sim
                    current = [sim.get_physics_dt(), sim.get_rendering_dt(), sim.get_sim_step_dt()]
                    already_matching = np.allclose(current, [1./240, 1./60, 1./60], rtol=0., atol=1e-9)
                    report["cloth_clock_restart"] = {"dt_s": current, "redundant_restart_skipped": bool(already_matching)}
                    if not already_matching:
                        original_cloth_clock()
                module._configure_sim_for_cloth_drop = configure_cloth_clock
                report["changes"].append("Skip the source cloth stop/dt/play restart only when all native clock values already match exactly.")
            try:
                result = original_postprocess(module, env, payload, camera_info, task_state)
            finally:
                if original_cloth_clock is not None:
                    module._configure_sim_for_cloth_drop = original_cloth_clock
                if original_fluid_box is not None:
                    module.generate_box = original_fluid_box
            if env.robots and module.TASK_NAME != "agent_observation" and not args.tabletop:
                try:
                    report["robot_initialization"] = place_robot(session)
                    report["changes"].append("Initialize R1 at approximate same-room free stance near source reference camera.")
                except Exception as exc:
                    report["robot_initialization"] = {"accepted": False, "reason": f"{type(exc).__name__}: {exc}"}
            report["postprocess"] = result
            if module.TASK_NAME == "deformable":
                report["cloth_attachment_inventory"] = [{"name": obj.name, "links": list(obj.links),
                    "has_attachment_point": "attachment_point" in obj.links}
                    for obj in env.scene.objects if obj.name.startswith("cover_small_item_render_cloth_")]
            if args.retain_trunk_posture and env.robots:
                for _ in range(30):
                    pipeline.og.sim.step()
                report["standing_settle_steps"] = 30
            write(output / "inventory_after_postprocess.json", inventory(session))
            write(output / "probe.json", report)
            return result
        pipeline.task_postprocess_env = postprocess
        result = original_initialize(session, pipeline)
        if args.refresh_native_render_products:
            refreshed = []
            for name, sensor in session.robot.sensors.items():
                if "rgb" in sensor.modalities:
                    height, width = sensor.image_height, sensor.image_width
                    sensor.image_height = height
                    refreshed.append({"name": name, "unchanged_resolution": [height, width]})
            report["native_render_product_refresh"] = refreshed
            report["changes"].append("Recreate original RGB render products through the native same-height setter after source postprocessing; mounts and resolutions unchanged.")
            session.invalidate_camera_cache()
            write(output / "probe.json", report)
        return result

    ESISession._configure_robot_observation = configure
    ESISession._initialize_scene = initialize
    ESISession.close = close

    def probe(session, metadata, controller, executor, **kwargs):
        from core.record.images import save_png
        if args.late_robot_position is not None:
            import torch
            yaw = np.deg2rad(args.late_robot_yaw_deg)
            before_pose = controller.pose()
            session.robot.set_position_orientation(position=torch.tensor(args.late_robot_position, dtype=torch.float32),
                orientation=torch.tensor([0., 0., np.sin(yaw/2), np.cos(yaw/2)], dtype=torch.float32))
            session.robot.keep_still()
            for _ in range(30):
                controller.step_base_debug()
            report["late_robot_initialization"] = {"before": before_pose, "after": controller.pose(),
                "scope": "Private robot initialization after the working posture is stable; before diagnostic actions."}
            report["changes"].append("Initialize the alternative robot stance after the original tabletop and working arm qualify; no task-object poses change.")
            write(output / "probe.json", report)
            session.scene_adapter.qualify(session)
            for name, rgb in session.observe(include_wrist=True).items():
                save_png(output / f"late_stance_{name}.png", rgb)
        session.configure_scene_camera()
        save_png(output / "initial_reference.png", session.observe(include_scene=True)["scene_rgb"])
        report.update(status="initialized", ready_wall_s=time.monotonic()-started,
                      task=session.task_module.TASK_NAME, scene=session.scene_name, room=session.room_name,
                      postprocess=session.postprocess)
        sim = session.pipeline.og.sim
        report["simulation_timing"] = {"physics_dt_s": sim.get_physics_dt(),
                                       "rendering_dt_s": sim.get_rendering_dt(),
                                       "action_dt_s": controller.dt}
        if session.task_module.TASK_NAME == "inclined_plane":
            report["native_physical_parameters"] = {}
            for name in ("slope", "task_obj"):
                obj = session.env.scene.object_registry("name", name)
                parameters = {"root_link_mass_kg": float(obj.root_link.mass), "collision_materials": []}
                for link in obj.links.values():
                    for mesh in link.collision_meshes.values():
                        material = mesh.get_applied_physics_material()
                        try:
                            friction = {"static_friction": material.get_static_friction(),
                                        "dynamic_friction": material.get_dynamic_friction()}
                        except (AttributeError, RuntimeError) as exc:
                            friction = {"unavailable": str(exc)}
                        parameters["collision_materials"].append({"mesh": mesh.prim_path, **friction})
                report["native_physical_parameters"][name] = parameters
        if session.task_module.TASK_NAME == "deformable":
            from omnigibson.macros import gm
            report["cloth_assisted_grasp_mode"] = gm.AG_CLOTH
            report["cloth_attachment_inventory"] = [{"name": obj.name, "links": list(obj.links),
                "has_attachment_point": "attachment_point" in obj.links}
                for obj in session.env.scene.objects if obj.name.startswith("cover_small_item_render_cloth_")]
        before = inventory(session)
        write(output / "inventory_before.json", before)
        if session.task_module.TASK_NAME == "physical_contact":
            first, second = before.get("obj1"), before.get("obj2")
            if first and second and first["contacts"] is not None:
                report["native_pair_contact"] = any(c[k].startswith(second["prim_path"] + "/")
                    or c[k] == second["prim_path"] for c in first["contacts"] for k in ("body0", "body1"))
        write(output / "source_camera.json", {"position": session.scene_camera_position,
              "quaternion_xyzw": session.scene_camera_quaternion, "camera_info": session.camera_info})
        try:
            answer = json.loads(args.sample.read_text())["answer"]
            try:
                answer = json.loads(answer)
            except (TypeError, json.JSONDecodeError):
                pass
            if isinstance(answer, (dict, list)):
                answer = json.dumps(answer)
            report["source_answer_scorer_check"] = session.pipeline.task_score(session.task_module,
                session.payload, {"answer": answer, "confidence": 1.}, session.camera_info, session.task_state)
        except Exception as exc:
            report["source_answer_scorer_error"] = f"{type(exc).__name__}: {exc}"
        for token in ["RELEASE", "MV_UP", "MV_DOWN", "GRASP", "RELEASE", "BASE_TURN_LEFT", "BASE_TURN_RIGHT"]:
            try:
                result = executor.execute(token)
                report["native_actions"].append(plain(result))
            except Exception as exc:
                report["native_actions"].append({"action": token, "exception": f"{type(exc).__name__}: {exc}"})
                break
            write(output / "probe.json", report)
            if not result.get("completed", False) and token not in ("GRASP",):
                break
        if args.upstream_effect and not args.skip_upstream_effect:
            module = session.task_module
            action = {"inclined_plane": "put_on_slope", "liquid_volume": "fill left"}.get(module.TASK_NAME)
            if action:
                original_step = sim.step
                trace = []
                def traced_step(*step_args, **step_kwargs):
                    result = original_step(*step_args, **step_kwargs)
                    if module.TASK_NAME == "inclined_plane":
                        obj = session.env.scene.object_registry("name", "task_obj")
                        trace.append({"step": len(trace)+1, "position": plain(obj.get_position_orientation()[0]),
                                      "linear_velocity": plain(obj.get_linear_velocity())})
                    return result
                try:
                    sim.step = traced_step
                    report["upstream_effect"] = session.pipeline.call_with_supported_args(module.execute_task_action,
                        session.env, session.payload, session.camera_info, action,
                        np.asarray(session.scene_camera_position), np.asarray(session.scene_camera_quaternion),
                        task_state=session.task_state, step=0, step_image_dir=output / "upstream_effect")
                    report["upstream_effect_semantics"] = "ESI scripted scene effect; not a physical R1 manipulation success."
                    report["upstream_effect_trace"] = trace
                except Exception as exc:
                    report["upstream_effect_error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    sim.step = original_step
        after = inventory(session)
        write(output / "inventory_after.json", after)
        obs = session.observe(include_wrist=True, include_scene=True)
        for name, rgb in obs.items():
            save_png(output / f"after_{name}.png", rgb)
        task_objects = {n: o for n, o in before.items() if o["task_object"]}
        report["task_object_count"] = len(task_objects)
        report["visual_only_task_count"] = sum(o["visual_only"] for o in task_objects.values())
        report["fixed_task_count"] = sum(o["fixed_base"] for o in task_objects.values())
        report["task_state_keys"] = sorted(session.task_state)
        report["reference_image_paths"] = session.pipeline.task_reference_image_paths(session.task_module, session.payload, session.task_state)
        report["reference_images_available"] = [path.is_file() for path in report["reference_image_paths"]]
        report["task_object_displacements_m"] = {n: float(np.linalg.norm(np.asarray(after[n]["position"])-o["position"])) for n, o in task_objects.items() if n in after}
        if args.grasp_target:
            report["grasp_probe"] = grasp_probe(session, controller, executor, args.grasp_target, output,
                                                vertical_first=args.grasp_vertical_first,
                                                base_forward_m=args.grasp_base_forward_m)
        report.update(status="complete", elapsed_wall_s=time.monotonic()-started)
        write(output / "probe.json", report)
        print("[category-probe] " + json.dumps({k: report[k] for k in ("task", "status", "ready_wall_s", "task_object_count", "visual_only_task_count")} ), flush=True)

    web.run_esi_control = probe
    cli = ["--metadata", str(args.sample), "--output-dir", str(output), "--upstream-observation", "--web",
           "--gripper-control", "--rotation-control", "--primitive-profile", "fine", "--camera-size", "256", "256",
           "--no-debug-cameras", "--no-human-recording", "--capture-agent-observation"]
    if args.grasp_target:
        cli += ["--parameterized-actions", "--macro-execution", "metric_target"]
    if args.retain_trunk_posture:
        arm_config = json.loads((ROOT / "configs/esi_arm.json").read_text())
        arm_config["initial_trunk_joint_positions_rad"] = None
        write(output / "arm_standing.json", arm_config)
        cli += ["--arm-config", str(output / "arm_standing.json")]
        report["changes"].append("Retain the existing native trunk posture; skip the optional working-posture initialization.")
    if args.tabletop:
        cli += ["--action-order-tabletop-config", str(ROOT / "configs/esi_action_order_tabletop.json")]
    try:
        result = robot_main(cli)
        if report["status"] != "complete":
            report.update(status="failed", cli_exit_code=result, elapsed_wall_s=time.monotonic()-started)
            write(output / "probe.json", report)
        return result
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(), elapsed_wall_s=time.monotonic()-started)
        write(output / "probe.json", report)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
