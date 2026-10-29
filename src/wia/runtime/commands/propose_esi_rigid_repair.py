#!/usr/bin/env python3
"""Propose a same-room table/robot translation from a native diagnostic snapshot.

Run on CPU: .venv/bin/python src/wia/runtime/commands/propose_esi_rigid_repair.py
    --base-scene FILE --geometry native_geometry.json --output-dir DIRECTORY
The original task, scales and relative layout are retained. Proposals require a
fresh native validation; this script never publishes a runnable catalogue entry.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import yaml
from PIL import Image

from wia.tasks.schema import SceneDefinition
from wia.tasks.scene_only import horizontal_box_overlap
from wia.tasks.tabletop import source_geometry


def relocate(value, base, floor_delta, yaw_delta):
    result = copy.deepcopy(value)
    patches = result["scene"]["variants"]["tabletop"]["patches"]
    previous = next(p for p in patches if p["operation"] == "robot_pose")
    origin = np.asarray(previous["position"], float)
    rotation = Rotation.from_euler("z", yaw_delta, degrees=True)
    floor_objects = {key for key,support in result["qualification"].get("object_supports",{}).items()
        if result["bindings"][support].get("category") in ("floor","floors")}
    for patch in patches:
        if "position" not in patch:
            continue
        if patch.get("binding") in floor_objects:
            continue
        position = np.asarray(patch["position"], float)
        xy = np.asarray(base) + rotation.apply(position-origin)[:2]
        patch["position"] = [*xy.tolist(), float(position[2]+floor_delta)]
        if patch["operation"] == "robot_pose":
            patch["yaw_deg"] += yaw_delta
        elif patch.get("quaternion_xyzw") is not None:
            patch["quaternion_xyzw"] = (rotation*Rotation.from_quat(patch["quaternion_xyzw"])).as_quat().tolist()
    result["id"] += "_free_floor_repair"
    SceneDefinition.model_validate(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-scene", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--room-grid", type=Path, help="Exact room-grid supplement for earlier sparse snapshots.")
    parser.add_argument("--data-root", type=Path, default=ROOT/"explore_to_achieve/scenes")
    parser.add_argument("--room-map", type=Path, help="Original room map for resolving a coarse native grid.")
    parser.add_argument("--dense-base-grid", action="store_true", help="Try nearby 0.2 m base locations; floor heights remain estimates.")
    parser.add_argument("--robot-reference", type=Path, help="Use a contact-free native R1 snapshot if the failed spawn deforms its arms.")
    parser.add_argument("--robot-footprint-m", nargs=2, type=float,
        help="Optional measured chassis XY dimensions for a coarse proposal; arm clearance must pass native initialization.")
    parser.add_argument("--base-grid-spacing-m",type=float,default=.2)
    parser.add_argument("--base-search-radius-m",type=float,default=3.)
    parser.add_argument("--xy-padding-m",type=float,default=.05,help="Extra clearance per side in the proposal screen.")
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args(argv)
    value = yaml.safe_load(args.base_scene.read_text())
    _, _, source_objects = source_geometry(args.data_root/value["source"]["json"],args.data_root)
    geometry = json.loads(args.geometry.read_text())
    patches = value["scene"]["variants"]["tabletop"]["patches"]
    base_patch = next(p for p in patches if p["operation"] == "robot_pose")
    base = np.asarray(base_patch["position"], float)
    by_name = {o["name"]: o for o in geometry["objects"] if "aabb_min" in o}
    reference = json.loads(args.robot_reference.read_text()) if args.robot_reference else geometry
    native_robot = next(o for o in reference["objects"] if o["name"] == reference["robot_name"])
    if args.robot_reference and any(not any("/floors_" in body or "groundPlane" in body for body in (c["body0"],c["body1"]))
            for c in native_robot.get("contacts",[])):
        raise ValueError("The robot reference must have no native non-floor contact.")
    native_low, native_high = (np.asarray(native_robot[k]) for k in ("aabb_min", "aabb_max"))
    robot_centre_offset = (native_low+native_high)/2-np.asarray(native_robot["position"])
    # Use the full observed articulation envelope, including the tucked arms.
    robot_extent = native_high-native_low
    if args.robot_footprint_m:
        robot_extent[:2] = args.robot_footprint_m
        robot_centre_offset[:2] = 0.
    ignored = {geometry["robot_name"]} | {b["name"] for b in value["bindings"].values()}
    ignored |= {p["name"] for p in patches if p["operation"] == "add_primitive"}
    obstacles = [(o["name"], np.asarray(o["aabb_min"]), np.asarray(o["aabb_max"]))
        for o in by_name.values() if o["name"] not in ignored and not o["visual_only"]
        and o["category"] not in ("floor", "floors", "ceiling", "ceilings")]
    floor_objects = {key for key,support in value["qualification"].get("object_supports",{}).items()
        if value["bindings"][support].get("category") in ("floor","floors")}
    for key in floor_objects:
        original = source_objects[key]["pose"]
        obstacles.append((value["bindings"][key]["name"],np.asarray(original["aabb_min"]),np.asarray(original["aabb_max"])))
    grid = json.loads(args.room_grid.read_text()) if args.room_grid else geometry["room_grid"]
    room_map = args.room_map or geometry.get("room_map_path")
    if room_map and grid["map_resolution_m"] > .25:
        # The native scene uses a 1 m room map. Preserve the environment and
        # improve only this offline proposal heuristic using the source map.
        Image.MAX_IMAGE_PIXELS = None  # Trusted, local scene maps can be very large.
        original_map = np.asarray(Image.open(room_map))
        size = int(original_map.shape[0]*.01/.1)
        indices = (np.arange(size)*original_map.shape[0]/size).astype(int)
        fine_mask = original_map[indices[:,None],indices[None,:]] == grid["room_id"]
        segments = []
        for row_index,row in enumerate(fine_mask):
            boundaries = np.flatnonzero(np.diff(np.r_[False,row,False].astype(np.int8)))
            segments.extend([[row_index,int(start),int(end)] for start,end in boundaries.reshape(-1,2)])
        grid = {"map_size":size,"map_resolution_m":.1,"room_id":grid["room_id"],"row_segments":segments}
    room_mask = np.zeros((grid["map_size"],grid["map_size"]),dtype=bool)
    for row,start,end in grid["row_segments"]:
        room_mask[row,start:end] = True
    floor_samples = geometry["local_floor_samples"]
    floor_tree = cKDTree([s["xy"] for s in floor_samples])
    candidate_samples = list(floor_samples)
    if args.dense_base_grid:
        pixels = np.argwhere(room_mask)
        stride = max(1,int(round(args.base_grid_spacing_m/grid["map_resolution_m"])))
        pixels = pixels[np.all(pixels % stride == 0,axis=1)]
        points = (pixels[:,::-1]-grid["map_size"]/2)*grid["map_resolution_m"]
        points = points[np.linalg.norm(points-base[:2],axis=1) <= args.base_search_radius_m]
        _, neighbours = floor_tree.query(points)
        candidate_samples.extend({"xy":point.tolist(),"z":floor_samples[int(n)]["z"]}
            for point,n in zip(points,neighbours))
    legs = [p for p in patches if p["operation"] == "add_primitive" and "leg" in p["name"]]
    floor_before = np.median([p["position"][2]-p["dimensions_m"][2]/2 for p in legs])
    candidates = []
    rejected = {"room": 0, "collision": 0, "uneven_floor": 0}
    for sample in sorted(candidate_samples, key=lambda s: np.linalg.norm(np.asarray(s["xy"])-base[:2])):
        for yaw in (0.,90.,-90.,180.):
            delta = float(sample["z"]-floor_before)
            proposed = relocate(value, sample["xy"], delta, yaw)
            rotation = Rotation.from_euler("z", yaw, degrees=True)
            robot_rotation = Rotation.from_euler("z",base_patch["yaw_deg"]+yaw,degrees=True) if args.robot_reference else rotation
            robot_centre = np.r_[sample["xy"], base[2]+delta] + robot_rotation.apply(robot_centre_offset)
            shapes = [("robot", robot_centre, robot_rotation.as_quat(), robot_extent)]
            proposed_patches = proposed["scene"]["variants"]["tabletop"]["patches"]
            for p in proposed_patches:
                if p["operation"] == "add_primitive":
                    shapes.append((p["name"],np.asarray(p["position"]),p["quaternion_xyzw"],np.asarray(p["dimensions_m"])))
                elif p["operation"] == "object_pose":
                    if p["binding"] in floor_objects:
                        continue
                    # Native AABBs can remain cached at the pre-teleport pose until
                    # a physics step. Use the retained source-frame bounds here.
                    original = source_objects[p["binding"]]["pose"]
                    low, high = (np.asarray(original[k]) for k in ("aabb_min","aabb_max"))
                    offset = (low+high)/2-np.asarray(original["position"])
                    object_rotation = Rotation.from_quat(p["quaternion_xyzw"])*Rotation.from_quat(original["quaternion_xyzw"]).inv()
                    shapes.append((p["binding"],np.asarray(p["position"])+object_rotation.apply(offset),object_rotation.as_quat(),high-low))
            outside = collision = False
            for _, centre, quat, extent in shapes:
                r = Rotation.from_quat(quat)
                padded_extent = extent+np.array([2*args.xy_padding_m,2*args.xy_padding_m,0.])
                half_aabb = np.abs(r.as_matrix()) @ (padded_extent/2)
                corners = r.apply([[x,y,0.] for x in (-extent[0]/2,extent[0]/2) for y in (-extent[1]/2,extent[1]/2)])[:,:2]+centre[:2]
                pixels = (corners[:,::-1]/grid["map_resolution_m"]+grid["map_size"]/2).astype(int)
                if (np.any(pixels < 0) or np.any(pixels >= grid["map_size"])
                        or not np.all(room_mask[pixels[:,0],pixels[:,1]])):
                    outside = True
                    break
                for _, low, high in obstacles:
                    if (np.all(np.minimum(centre+half_aabb,high)-np.maximum(centre-half_aabb,low) > .003)
                            and horizontal_box_overlap(centre,quat,padded_extent,low,high)):
                        collision = True
                        break
                if collision:
                    break
            if outside or collision:
                rejected["room" if outside else "collision"] += 1
                continue
            proposed_legs = [p for p in proposed_patches if p["operation"] == "add_primitive" and "leg" in p["name"]]
            distances, neighbours = floor_tree.query([p["position"][:2] for p in proposed_legs])
            if max(abs(floor_samples[int(n)]["z"]-sample["z"]) for n in neighbours) > .008:
                rejected["uneven_floor"] += 1
                continue
            candidates.append((np.linalg.norm(np.asarray(sample["xy"])-base[:2])+.002*abs(yaw), proposed,
                {"base_xy": sample["xy"], "floor_delta_m": delta,"yaw_delta_deg":yaw,
                 "maximum_floor_estimate_distance_m":float(max(distances))}))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    selected = []
    for _, proposed, details in sorted(candidates,key=lambda c:c[0]):
        if any(np.linalg.norm(np.asarray(details["base_xy"])-np.asarray(p["base_xy"])) < .25 for p in selected):
            continue
        path = args.output_dir/f"candidate_{len(selected)+1:02d}.yaml"
        path.write_text(yaml.safe_dump(proposed,sort_keys=False))
        selected.append({**details,"scene_file":str(path.resolve())})
        if len(selected) >= args.count:
            break
    report = {"source_scene":str(args.base_scene.resolve()),"native_geometry":str(args.geometry.resolve()),
        "original_scales_and_question_retained":True,"native_validation_required":True,
        "room_resolution_m":grid["map_resolution_m"],"robot_envelope_m":robot_extent.tolist(),
        "robot_reference_file":str(args.robot_reference.resolve()) if args.robot_reference else None,
        "full_articulation_envelope_screened":args.robot_footprint_m is None,
        "xy_padding_m":args.xy_padding_m,"base_grid_spacing_m":args.base_grid_spacing_m if args.dense_base_grid else None,
        "base_search_radius_m":args.base_search_radius_m if args.dense_base_grid else None,
        "rejected":rejected,"proposals":selected}
    (args.output_dir/"repair_proposals.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
