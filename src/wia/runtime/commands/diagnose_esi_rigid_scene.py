#!/usr/bin/env python3
"""Trace native initialization of one private Rigid Containment scene.

Run in the retained GPU shell: python src/wia/runtime/commands/diagnose_esi_rigid_scene.py
    --base-scene /absolute/scene.yaml --output-dir /absolute/diagnostics
The wrapper reuses native validation, records phase states and periodic stacks,
and performs no navigation, grasp, task evaluation, or model call.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import faulthandler
import json
import os
from pathlib import Path
import sys
import time

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-scene", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "explore_to_achieve/scenes")
    parser.add_argument("--stack-interval-s", type=int, default=120)
    parser.add_argument("--cache-assets", action="store_true")
    parser.add_argument("--cache-mesh-geometry", action="store_true")
    parser.add_argument("--profile-loading", action="store_true")
    parser.add_argument("--snapshot-physics", action="store_true")
    args = parser.parse_args(argv)
    if not os.environ.get("PBS_JOBID"):
        parser.error("Native diagnostics require the retained interactive GPU allocation.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SHOWHARNESS_ESI_MESH_CACHE"] = "1" if args.cache_mesh_geometry else "0"
    if args.profile_loading:
        os.environ["SHOWHARNESS_ESI_LOAD_PROFILE"] = str(args.output_dir)
    started = time.monotonic()
    trace = (args.output_dir / "stack_samples.log").open("wb", buffering=0)
    faulthandler.enable(file=trace, all_threads=True)
    faulthandler.dump_traceback_later(args.stack_interval_s, repeat=True, file=trace)

    def phase(name):
        with (args.output_dir / "phase_log.jsonl").open("a") as stream:
            stream.write(json.dumps({"phase": name, "wall_elapsed_s": time.monotonic()-started}) + "\n")
        print("[rigid-diagnostic]", name, flush=True)

    phase("before_native_imports")
    import numpy as np
    from wia.runtime.sim.esi_omnigibson import _array
    import wia.runtime.sim.esi_omnigibson as session_module
    import wia.tasks.scene as scene_module
    import wia.tasks.scene_only as scene_only_module
    import wia.runtime.sim.esi_arm as arm_module
    from wia.runtime.commands.sample_esi_supports import main as worker_main
    original_import = session_module._import_pipeline

    def imported(esi_root):
        pipeline = original_import(esi_root)
        from omnigibson.objects.usd_object import USDObject, log
        from wia.runtime.sim.esi_asset_cache import instrument_assets
        instrument_assets(USDObject, log, args.output_dir, cache=args.cache_assets)
        phase("native_pipeline_imported")
        return pipeline

    session_module._import_pipeline = imported

    def state(adapter, session, name, *, geometry=False):
        phase(name)
        robot = session.env.robots[0]
        keys = list(adapter.bindings)
        objects = {adapter._binding(key).name for key in keys}
        objects.add(robot.name)
        value = {"phase": name, "wall_elapsed_s": time.monotonic()-started, "objects": []}
        if name == "after_pose_application":
            value["physics_frame_note"] = "Immediate post-teleport AABBs/contacts may be cached at the previous physics frame; use settled snapshots for physical checks."
        for obj in session.env.scene.objects:
            if not geometry and obj.name not in objects:
                continue
            try:
                low, high = (_array(x) for x in obj.aabb)
                if not np.isfinite(low).all() or not np.isfinite(high).all():
                    continue
                position, quaternion = (_array(x).tolist() for x in obj.get_position_orientation())
                row = {"name": obj.name, "category": obj.category, "prim_path": obj.prim_path,
                    "fixed_base": obj.fixed_base, "visual_only": obj.visual_only,
                    "aabb_min": low.tolist(), "aabb_max": high.tolist(),
                    "position": position, "quaternion_xyzw": quaternion}
                if obj.name in objects:
                    row["contacts"] = [{"body0": c.body0, "body1": c.body1} for c in obj.contact_list()]
                    if not obj.fixed_base:
                        row["linear_velocity"] = _array(obj.get_linear_velocity()).tolist()
                        row["angular_velocity"] = _array(obj.get_angular_velocity()).tolist()
                value["objects"].append(row)
            except Exception as error:
                value["objects"].append({"name": obj.name, "error": str(error)})
        value["robot_name"] = robot.name
        value["source_room"] = session.room_name
        value["floor_name"] = session.payload.get("floor_name")
        if geometry:
            segmentation = session.env.scene.seg_map
            room_id = segmentation.room_ins_name_to_ins_id[session.room_name]
            room_mask = _array(segmentation.room_ins_map) == room_id
            indices = np.argwhere(room_mask)
            segments = []
            for row_index, row in enumerate(room_mask):
                boundaries = np.flatnonzero(np.diff(np.r_[False,row,False].astype(np.int8)))
                segments.extend([[row_index,int(start),int(end)] for start,end in boundaries.reshape(-1,2)])
            value["room_grid"] = {"map_size":segmentation.map_size,
                "map_resolution_m":segmentation.map_resolution,"room_id":room_id,"row_segments":segments}
            value["room_map_path"] = str(Path(segmentation.scene_dir)/"layout/floor_insseg_0.png")
            if len(indices):
                selected = indices[np.linspace(0, len(indices)-1, min(1024,len(indices))).astype(int)]
                value["room_sample_xy_m"] = ((selected[:, ::-1]-segmentation.map_size/2)*segmentation.map_resolution).tolist()
            floor = session.env.scene.object_registry("name", value["floor_name"])
            if floor is not None:
                from omnigibson.utils.sampling_utils import raytest
                from wia.tasks.scene_only import local_floor_surface
                low, high = (_array(x) for x in floor.aabb)
                points = sorted(value.get("room_sample_xy_m", []), key=lambda point: np.linalg.norm(np.asarray(point)-_array(robot.get_position_orientation()[0])[:2]))[:128]
                value["local_floor_samples"] = []
                for point in points:
                    try:
                        hit = local_floor_surface(floor.prim_path, raytest([*point,float(high[2])+.1], [*point,float(low[2])-.1], only_closest=False))
                        value["local_floor_samples"].append({"xy":point,"z":float(hit["position"][2])})
                    except ValueError:
                        continue
        destination = args.output_dir / ("native_geometry.json" if geometry else name+".json")
        destination.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")

    original_apply = scene_module.ScenePatches.apply_after_postprocessing
    original_settle = scene_only_module.SceneOnly.apply_after_postprocessing
    original_trunk = arm_module._initialize_trunk
    original_qualify = scene_only_module.SceneOnly.qualify
    original_final_settle = scene_only_module.SceneOnly.prepare_after_controller_initialization

    def applied(adapter, session):
        original_apply(adapter, session)
        state(adapter, session, "after_pose_application", geometry=True)

    def settled(adapter, session):
        original_settle(adapter, session)
        state(adapter, session, "after_patch_settling")

    def trunk(session, base_config, arm_config):
        adapter = session.scene_adapter
        state(adapter, session, "before_trunk_initialization")
        simulator = session.pipeline.og.sim
        original_step = simulator.step
        watched = [session.env.robots[0]] + [session.env.scene.object_registry("name",adapter._binding(key).name)
            for key in adapter.task.definition.qualification.supported_objects]
        previous_contacts = None

        def observed_step(*step_args, **step_kwargs):
            nonlocal previous_contacts
            result = original_step(*step_args, **step_kwargs)
            rows = [{"name":obj.name,"contacts":sorted(set((c.body0,c.body1) for c in obj.contact_list()))}
                for obj in watched]
            signature = tuple((row["name"],tuple(row["contacts"])) for row in rows)
            if signature != previous_contacts:
                for row,obj in zip(rows,watched):
                    row["position"] = _array(obj.get_position_orientation()[0]).tolist()
                with (args.output_dir/"trunk_contact_changes.jsonl").open("a") as stream:
                    stream.write(json.dumps({"wall_elapsed_s":time.monotonic()-started,"objects":rows})+"\n")
                previous_contacts = signature
            return result

        simulator.step = observed_step
        try:
            return original_trunk(session, base_config, arm_config)
        finally:
            simulator.step = original_step
            state(adapter, session, "after_trunk_initialization")

    def qualified(adapter, session, executor, output, **kwargs):
        result = original_qualify(adapter, session, executor, output, **kwargs)
        if args.snapshot_physics:
            snapshot = []
            for obj in session.env.scene.objects:
                links = []
                for link in obj.links.values():
                    if not hasattr(type(link), "mass"):
                        continue
                    links.append({"name": link.name, "mass": float(link.mass), "density": float(link.density),
                                  "center_of_mass": _array(link.center_of_mass).tolist(),
                                  "collision_meshes": {name: {"approximation": mesh.get_collision_approximation(),
                                      "contact_offset": float(mesh.get_contact_offset()), "rest_offset": float(mesh.get_rest_offset())}
                                      for name, mesh in link.collision_meshes.items()}})
                snapshot.append({"name": obj.name, "category": obj.category,
                    "fixed_base": obj.fixed_base, "visual_only": obj.visual_only,
                    "position": _array(obj.get_position_orientation()[0]).tolist(), "links": links})
            (args.output_dir / "physics_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        if result.status not in ("accepted_original","accepted_minimal_patch"):
            return result
        target = result.checks["supported_objects"][adapter.task.definition.target]
        centre = (np.asarray(target["aabb_min"])+np.asarray(target["aabb_max"]))/2
        base = _array(session.robot.get_position_orientation()[0])
        distance = float(np.linalg.norm(centre[:2]-base[:2]))
        receipt = {"initial_stance_distance_m":distance,"physical_grasp_verified":False,
            "route_connectivity_screen_passed":False,"source_json":adapter.task.definition.source.json_path,
            "scene_file":str(adapter.task.path)}
        if distance <= .90:
            receipt.update(status="initial_stance_within_coarse_reach",stance_xy_m=base[:2].tolist())
        else:
            phase("before_coarse_stance_check")
            from types import SimpleNamespace
            from wia.tasks.support_sampling import SupportSampler, SamplingConfig
            controller = SimpleNamespace(pose=lambda:{"position":_array(session.robot.get_position_orientation()[0]).tolist()})
            sampler = SupportSampler(session,controller,executor,SamplingConfig(),args.output_dir,adapter.task)
            try:
                route,stance,connected = sampler.route_to_target(maximum_stance_distance_m=.90)
                receipt.update(status="screened_nearby_stance",stance_xy_m=stance,route=route,
                    route_connectivity_screen_passed=connected,robot_footprint_radius_m=sampler.report["robot_footprint_radius_m"])
            except ValueError as error:
                receipt.update(status="no_screened_stance",reason=str(error))
            phase("after_coarse_stance_check")
        (args.output_dir/"coarse_stance.json").write_text(json.dumps(receipt,indent=2)+"\n")
        return result

    def final_settle(adapter, session):
        if not adapter.task.definition.runtime.post_controller_settle_steps:
            return original_final_settle(adapter,session)
        state(adapter,session,"before_controller_settling")
        result = original_final_settle(adapter,session)
        state(adapter,session,"after_controller_settling")
        return result

    scene_module.ScenePatches.apply_after_postprocessing = applied
    scene_only_module.SceneOnly.apply_after_postprocessing = settled
    arm_module._initialize_trunk = trunk
    scene_only_module.SceneOnly.qualify = qualified
    scene_only_module.SceneOnly.prepare_after_controller_initialization = final_settle
    phase("before_native_validation")
    try:
        return worker_main(["--worker", "--validate-only", "--base-scene", str(args.base_scene),
            "--output-dir", str(args.output_dir), "--data-root", str(args.data_root), "--variant", "tabletop"])
    finally:
        phase("native_validation_returned")
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    raise SystemExit(main())
