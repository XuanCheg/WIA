#!/usr/bin/env python3
"""Screen saved Action Order geometry with CPU planning, without robot commands.

Run with the existing behavior environment Python and its USD library directory
in LD_LIBRARY_PATH (bin and bin/deps), then:
  python src/wia/runtime/commands/plan_esi_action_order_offline.py --output OUTPUT_DIRECTORY

This imports CPU USD readers, NumPy and SciPy, never OmniGibson or SimulationApp.
Original licensed assets are read through temporary decrypted files, following
the installed asset reader's Fernet protocol; keys and meshes are not exported.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import heapq
import json
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import yaml

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
from core.camera_geometry import RobotKinematics, axis_rotation, pose_matrix

ASSETS = Path.home() / "CFP05-CF-143/esi-bench/data/omnigibson"
BEHAVIOR = Path.home() / "CFP05-CF-143/esi-bench/BEHAVIOR-1K/OmniGibson"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def import_usd():
    directory = next((Path(sys.prefix) / "lib/python3.10/site-packages/isaacsim/extscache").glob("omni.usd.libs-*"))
    sys.path.insert(0, str(directory))
    from pxr import Usd, UsdGeom, UsdPhysics
    return Usd, UsdGeom, UsdPhysics


def quat_xyzw(value):
    return [*value.GetImaginary(), value.GetReal()]


class Robot:
    def __init__(self, Usd, UsdPhysics, trunk):
        directory = ASSETS / "omnigibson-robot-assets/models/r1"
        self.urdf = directory / "urdf/r1.urdf"
        self.kin = RobotKinematics(self.urdf, root="base_link")
        self.calibration = {}
        self.limits = {}
        for joint in ET.parse(self.urdf).getroot().findall("joint"):
            limit = joint.find("limit")
            if limit is not None:
                self.limits[joint.get("name")] = [float(limit.get(x)) for x in ("lower", "upper")]
        stage = Usd.Stage.Open(str(directory / "usd/r1.usda"))
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            bodies = [joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()]
            if not all(len(x) == 1 for x in bodies):
                continue
            parent, child = [x[0].name for x in bodies]
            before = pose_matrix(joint.GetLocalPos0Attr().Get(), quat_xyzw(joint.GetLocalRot0Attr().Get()))
            after = np.linalg.inv(pose_matrix(joint.GetLocalPos1Attr().Get(), quat_xyzw(joint.GetLocalRot1Attr().Get())))
            kind = "revolute" if prim.IsA(UsdPhysics.RevoluteJoint) else "prismatic" if prim.IsA(UsdPhysics.PrismaticJoint) else "fixed"
            axis = np.array({"X": [1., 0., 0.], "Y": [0., 1., 0.], "Z": [0., 0., 1.]}.get(prim.GetAttribute("physics:axis").Get(), [1., 0., 0.]))
            name = prim.GetName()
            self.kin.children[parent] = [x for x in self.kin.children.get(parent, []) if x[0] != name]
            self.kin.children[parent].append((name, kind, child, before, axis, after))
            self.kin.parents[child] = parent
            self.calibration[name] = {"parent": parent, "child": child, "kind": kind,
                                      "before": before.tolist(), "after": after.tolist(), "axis": axis.tolist()}
        reset = yaml.safe_load((BEHAVIOR / "omnigibson/configs/r1_primitives.yaml").read_text())["robots"][0]["reset_joint_pos"]
        self.seed = {f"torso_joint{i+1}": x for i, x in enumerate(trunk)}
        for arm, offset in (("left", 0), ("right", 1)):
            self.seed.update({f"{arm}_arm_joint{i+1}": reset[10+2*i+offset] for i in range(6)})
            self.seed.update({f"{arm}_gripper_axis{i+1}": .05 for i in range(2)})
        self.arm_names = [f"left_arm_joint{i}" for i in range(1, 7)]
        self.lower, self.upper = np.array([self.limits[n] for n in self.arm_names]).T
        self.lower += .02
        self.upper -= .02
        path, child = [], "left_eef_link"
        while child != "base_link":
            parent = self.kin.parents[child]
            info = next(x for x in self.kin.children[parent] if x[2] == child)
            path.append(info)
            child = parent
        self.chain = path[::-1]
        self.reach_bound = sum(np.linalg.norm(x[3][:3, 3]) + np.linalg.norm(x[5][:3, 3])
                               for x in self.chain if x[0] not in {"torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4", "left_arm_joint1"})
        cfg = yaml.safe_load((directory / "curobo/r1_description_curobo_arm_no_torso.yaml").read_text())
        self.spheres = cfg["robot_cfg"]["kinematics"]["collision_spheres"]
        self.spheres = {k: v for k, v in self.spheres.items() if not k.startswith("attached_object")}
        self.sphere_links = set(self.spheres)
        self.self_ignored = {frozenset((a, b)) for a, others in cfg["robot_cfg"]["kinematics"]["self_collision_ignore"].items() for b in others}

    def values(self, arm=None, trunk=None, width=.1):
        values = dict(self.seed)
        if arm is not None:
            values.update(zip(self.arm_names, arm))
        if trunk is not None:
            values.update({f"torso_joint{i+1}": x for i, x in enumerate(trunk)})
        values.update({f"left_gripper_axis{i+1}": width/2 for i in range(2)})
        return values

    def fk(self, arm=None, trunk=None):
        values, result = self.values(arm, trunk), np.eye(4)
        for name, kind, _, origin, axis, after in self.chain:
            motion = np.eye(4)
            if kind == "revolute":
                motion[:3, :3] = axis_rotation(axis, values[name])
            elif kind == "prismatic":
                motion[:3, 3] = axis * values[name]
            result = result @ origin @ motion @ after
        return result

    def sphere_cloud(self, arm=None, trunk=None, width=.1):
        frames = self.kin.frames(self.values(arm, trunk, width), links=self.sphere_links)
        centers, radii, links = [], [], []
        for link, rows in self.spheres.items():
            for row in rows:
                centers.append(frames[link][:3, :3] @ row["center"] + frames[link][:3, 3])
                radii.append(row["radius"])
                links.append(link)
        return np.array(centers), np.array(radii), links

    def solve(self, target, orientation, seed, trunk=None):
        def residual(q):
            frame = self.fk(q, trunk)
            return np.r_[frame[:3, 3]-target,
                         .25*Rotation.from_matrix(orientation.T @ frame[:3, :3]).as_rotvec()]
        result = least_squares(residual, np.clip(seed, self.lower+1e-5, self.upper-1e-5),
                               bounds=(self.lower, self.upper), max_nfev=65,
                               ftol=1e-8, xtol=1e-8, gtol=1e-8)
        frame = self.fk(result.x, trunk)
        pos_error = float(np.linalg.norm(frame[:3, 3]-target))
        angle_error = float(np.rad2deg(Rotation.from_matrix(orientation.T @ frame[:3, :3]).magnitude()))
        return result.x, pos_error, angle_error


class Geometry:
    def __init__(self, Usd, UsdGeom, UsdPhysics, inventory):
        from cryptography.fernet import Fernet
        self.models, self.hulls, self.validation = {}, [], {}
        self.Usds = Usd, UsdGeom, UsdPhysics
        self.reader = Fernet((ASSETS / "omnigibson.key").read_bytes())
        self.inventory = inventory

    def model(self, category, model):
        key = category, model
        if key in self.models:
            return self.models[key]
        Usd, UsdGeom, UsdPhysics = self.Usds
        path = ASSETS / f"behavior-1k-assets/objects/{category}/{model}/usd/{model}.encrypted.usd"
        with tempfile.NamedTemporaryFile(suffix=".usd") as temporary:
            temporary.write(self.reader.decrypt(path.read_bytes()))
            temporary.flush()
            stage = Usd.Stage.Open(temporary.name)
            cache = UsdGeom.XformCache()
            inverse = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(stage.GetDefaultPrim())).T)
            collisions, all_points = [], []
            for prim in stage.Traverse():
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                transform = inverse @ np.array(cache.GetLocalToWorldTransform(prim)).T
                points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
                points = points @ transform[:3, :3].T + transform[:3, 3]
                all_points.append(points)
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    collisions.append(points)
        self.models[key] = collisions, np.concatenate(all_points)
        return self.models[key]

    def add(self, name):
        row = self.inventory[name]
        collisions, visible = self.model(row["category"], row["model"])
        rotation = Rotation.from_quat(row["quaternion_xyzw"]).as_matrix()
        transform = rotation @ np.diag(row["scale"])
        position = np.array(row["position"])
        visible = visible @ transform.T + position
        error = float(np.max(np.abs(np.r_[visible.min(axis=0)-row["aabb_min"], visible.max(axis=0)-row["aabb_max"]])))
        self.validation[name] = {"visible_vs_saved_aabb_max_error_m": error,
                                 "collision_meshes": len(collisions)}
        for points in collisions:
            points = points @ transform.T + position
            try:
                hull = ConvexHull(points)
            except Exception:
                hull = ConvexHull(points, qhull_options="QJ")
            self.hulls.append({"name": name, "points": points, "planes": hull.equations,
                               "geometry_uncertainty_m": error,
                               "low": points.min(axis=0), "high": points.max(axis=0)})

    def colliders(self, centers, radii, *, exclude=(), margin=.005):
        hits = []
        for hull in self.hulls:
            if hull["name"] in exclude:
                continue
            r = radii+margin+hull["geometry_uncertainty_m"]
            near = np.all((centers >= hull["low"]-r[:, None]) & (centers <= hull["high"]+r[:, None]), axis=1)
            if not near.any():
                continue
            planes = hull["planes"]
            values = centers[near] @ planes[:, :3].T + planes[:, 3]
            if np.any(np.all(values <= r[near, None], axis=1)):
                hits.append(hull["name"])
        return sorted(set(hits))


def graph_search(free, turning_free, start, resolution, yaw_step):
    distances = np.full(free.shape, np.inf)
    previous = np.full(free.size, -1, dtype=np.int32)
    if not free[start]:
        return distances, previous
    distances[start] = 0.
    queue = [(0., start)]
    shape = free.shape
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distances[node]:
            continue
        k, i, j = node
        edges = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= i+di < shape[1] and 0 <= j+dj < shape[2]:
                edges.append(((k, i+di, j+dj), resolution))
        for dk, edge in ((1, k), (-1, (k-1) % shape[0])):
            if turning_free[edge, i, j]:
                edges.append((((k+dk) % shape[0], i, j), .20*yaw_step))
        for other, step in edges:
            new_cost = cost+step
            if free[other] and new_cost < distances[other]:
                distances[other] = new_cost
                previous[np.ravel_multi_index(other, shape)] = np.ravel_multi_index(node, shape)
                heapq.heappush(queue, (new_cost, other))
    return distances, previous


def navigation(robot, geometry, floor, base, resolution=.03, yaws=12, excluded=()):
    low, high = np.array(floor["aabb_min"]), np.array(floor["aabb_max"])
    x0 = base[0]-np.ceil((base[0]-low[0])/resolution)*resolution
    y0 = base[1]-np.ceil((base[1]-low[1])/resolution)*resolution
    xs, ys = np.arange(x0, high[0], resolution), np.arange(y0, high[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    positions = np.c_[xx.ravel(), yy.ravel()]
    points, radii, links = robot.sphere_cloud()
    raw = np.array(__import__("PIL.Image", fromlist=["Image"]).open(
        ASSETS / "behavior-1k-assets/scenes/Beechwood_0_garden/layout/floor_trav_no_obj_0.png"))
    clearance = distance_transform_edt(raw == 255)*.01
    shape, yaw_step = xx.shape, 2*np.pi/yaws
    # This margin covers the distance from cell center to its farthest corner.
    margin = .012 + resolution/np.sqrt(2)

    def valid(angle, extra_margin=0.):
        rotation = Rotation.from_euler("z", angle).as_matrix()
        offsets = points @ rotation.T
        result = np.ones(len(positions), bool)
        for offset, radius, link in zip(offsets, radii, links):
            centers = np.c_[positions+offset[:2], np.full(len(positions), base[2]+offset[2])]
            if link == "base_link" or link.startswith("wheel"):
                ij = np.rint(centers[:, [1, 0]]/.01 + np.array(raw.shape)/2).astype(int)
                inside = np.all((ij >= 0) & (ij < raw.shape), axis=1)
                d = np.zeros(len(positions))
                d[inside] = clearance[tuple(ij[inside].T)]
                result &= d > radius+margin+extra_margin
                result &= np.all((centers[:, :2] > low[:2]+radius+margin)
                                 & (centers[:, :2] < high[:2]-radius-margin), axis=1)
            for hull in geometry.hulls:
                if hull["name"] in excluded or hull["name"].startswith("obj_"):
                    continue
                padded_radius = radius+margin+extra_margin+hull["geometry_uncertainty_m"]
                if centers[0, 2]+padded_radius < hull["low"][2] or centers[0, 2]-padded_radius > hull["high"][2]:
                    continue
                near = result & np.all((centers >= hull["low"]-padded_radius)
                                       & (centers <= hull["high"]+padded_radius), axis=1)
                indices = np.flatnonzero(near)
                if not len(indices):
                    continue
                planes = hull["planes"]
                collide = np.all(centers[indices] @ planes[:, :3].T+planes[:, 3] <= padded_radius, axis=1)
                result[indices[collide]] = False
        return result.reshape(shape)

    free = np.stack([valid(k*yaw_step) for k in range(yaws)])
    # Bound each rigid sphere center's displacement over a complete turn edge.
    angular_margin = 2*np.max(np.linalg.norm(points[:, :2], axis=1))*np.sin(yaw_step/4)
    turning = np.stack([free[k] & free[(k+1) % yaws] & valid((k+.5)*yaw_step, angular_margin) for k in range(yaws)])
    start = (round(base[3]/yaw_step) % yaws, round((base[1]-y0)/resolution), round((base[0]-x0)/resolution))
    distances, previous = graph_search(free, turning, start, resolution, yaw_step)
    return {"free": free, "distance": distances, "previous": previous, "xs": xs, "ys": ys,
            "yaw_step": yaw_step, "start": start, "resolution": resolution,
            "collision_margin_m": margin, "angular_sweep_margin_m": float(angular_margin)}


def section_intervals(hulls, origin, direction):
    intervals = []
    for hull in hulls:
        planes = hull["planes"]
        slope = planes[:, :3] @ direction
        value = planes[:, :3] @ origin + planes[:, 3]
        parallel = abs(slope) < 1e-10
        if np.any(value[parallel] > 1e-7):
            continue
        lower, upper = -np.inf, np.inf
        if np.any(slope < -1e-10):
            lower = max(lower, np.max(-value[slope < -1e-10]/slope[slope < -1e-10]))
        if np.any(slope > 1e-10):
            upper = min(upper, np.min(-value[slope > 1e-10]/slope[slope > 1e-10]))
        if lower <= upper:
            intervals.append([float(lower), float(upper)])
    merged = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1]+1e-5:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return merged


def grasp_candidates(name, row, geometry, max_width=.1):
    hulls = [h for h in geometry.hulls if h["name"] == name]
    points = np.concatenate([h["points"] for h in hulls])
    low, high = np.array(row["aabb_min"]), np.array(row["aabb_max"])
    center = (low+high)/2
    polygon = ConvexHull(points[:, :2])
    vertices = points[polygon.vertices, :2]
    edges = np.roll(vertices, -1, axis=0)-vertices
    # Exact convex minimum width occurs normal to an edge; coarse angle grids
    # incorrectly reject the narrowly fitting rotated small cutting board.
    angles = (np.arctan2(edges[:, 1], edges[:, 0])+np.pi/2) % np.pi
    spans = [np.ptp(points[:, :2] @ [np.cos(a), np.sin(a)]) for a in angles]
    best = int(np.argmin(spans))
    candidates = []
    if spans[best] <= max_width-.002:
        candidates.append({"type": "whole_horizontal_span", "position": center.tolist(),
                           "angle": float(angles[best]), "width": float(spans[best])})
    for fraction in (.8, .95):
        origin = center.copy()
        origin[2] = low[2]+fraction*(high[2]-low[2])
        for angle in np.r_[angles[best], np.deg2rad(np.arange(0, 180, 30))]:
            axis = np.array([np.cos(angle), np.sin(angle), 0.])
            for begin, end in section_intervals(hulls, origin, axis):
                width = end-begin
                if .003 <= width <= max_width-.002:
                    position = origin+axis*(begin+end)/2
                    candidates.append({"type": "local_section_candidate", "position": position.tolist(),
                                       "angle": float(angle), "width": width})
    return {"minimum_whole_horizontal_span_m": float(min(spans)), "open_width_m": max_width,
            "whole_span_fits": bool(min(spans) <= max_width-.002),
            "candidates": candidates[:16],
            "limit": "Local sections suggest rim/cap grasps, not force closure, friction, holding or lift success."}


def reconstruct_route(nav, node):
    shape, previous = nav["free"].shape, nav["previous"]
    current, nodes = np.ravel_multi_index(node, shape), []
    while current != -1:
        k, i, j = np.unravel_index(current, shape)
        nodes.append([float(nav["xs"][j]), float(nav["ys"][i]), float(k*nav["yaw_step"])])
        current = previous[current]
    return nodes[::-1]


def screen_target(name, row, candidates, robot, geometry, nav, base_z, *, excluded=()):
    reach = np.argwhere(np.isfinite(nav["distance"]))
    seed = np.array([robot.seed[n] for n in robot.arm_names])
    target = np.array(row["position"])
    best, attempted, errors, rejections = None, 0, [], {}
    sample_trunks = [None, np.deg2rad([55., -110., -55., 0.])]
    sample_trunks = [q for q in sample_trunks if q is None or all(
        robot.limits[f"torso_joint{i+1}"][0]+.02 < value < robot.limits[f"torso_joint{i+1}"][1]-.02
        for i, value in enumerate(q))]
    for trunk in sample_trunks:
        shoulder = robot.kin.frames(robot.values(trunk=trunk), links={"left_arm_link1"})["left_arm_link1"][:3, 3]
        positions = np.c_[nav["xs"][reach[:, 2]], nav["ys"][reach[:, 1]]]
        angles = reach[:, 0]*nav["yaw_step"]
        cos, sin = np.cos(angles), np.sin(angles)
        shoulder_world = np.c_[positions[:, 0]+cos*shoulder[0]-sin*shoulder[1],
                               positions[:, 1]+sin*shoulder[0]+cos*shoulder[1],
                               np.full(len(reach), base_z+shoulder[2])]
        bound_distance = np.linalg.norm(shoulder_world-target, axis=1)
        within = bound_distance <= robot.reach_bound+.10
        available = reach[within]
        score = bound_distance[within] + .06*nav["distance"][tuple(available.T)]
        selected = available[np.argsort(score)[:12]]
        if not len(selected):
            rejections["no_sampled_stance_within_chain_bound"] = rejections.get("no_sampled_stance_within_chain_bound", 0)+1
        for candidate in candidates["candidates"]:
            world = np.array(candidate["position"])
            world_orientation = Rotation.from_euler("z", candidate["angle"]+np.pi/2).as_matrix() @ Rotation.from_euler("x", np.pi).as_matrix()
            for node in selected:
                attempted += 1
                k, i, j = node
                yaw = k*nav["yaw_step"]
                rotation = Rotation.from_euler("z", yaw).as_matrix()
                base = np.array([nav["xs"][j], nav["ys"][i], base_z])
                destination = rotation.T @ (world-base)
                orientation = rotation.T @ world_orientation
                solution, pos_error, angle_error = robot.solve(destination, orientation, seed, trunk)
                errors.append(pos_error)
                if pos_error > .003 or angle_error > 2.:
                    rejections["ik_residual"] = rejections.get("ik_residual", 0)+1
                    continue
                stages, previous = [], solution
                failed = False
                for offset in (.08, .0, .10):
                    position = destination + rotation.T @ [0., 0., offset]
                    q, pe, ae = robot.solve(position, orientation, previous, trunk)
                    if pe > .003 or ae > 2.:
                        rejections["pregrasp_or_lift_ik"] = rejections.get("pregrasp_or_lift_ik", 0)+1
                        failed = True
                        break
                    centers, radii, links = robot.sphere_cloud(q, trunk, width=.1)
                    centers = centers @ rotation.T+base
                    contacts = geometry.colliders(centers, radii, exclude=(*excluded, name), margin=.006)
                    if contacts:
                        for contact in contacts:
                            rejections[f"collision:{contact}"] = rejections.get(f"collision:{contact}", 0)+1
                        failed = True
                        break
                    nonfingers = np.array([not link.startswith("left_gripper") for link in links])
                    target_hits = geometry.colliders(centers[nonfingers], radii[nonfingers],
                        exclude=tuple(n for n in geometry.inventory if n != name), margin=.002)
                    if target_hits:
                        rejections["nonfinger_target_collision"] = rejections.get("nonfinger_target_collision", 0)+1
                        failed = True
                        break
                    stages.append({"vertical_offset_m": offset, "left_arm_rad": q.tolist(),
                                   "position_error_m": pe, "orientation_error_deg": ae})
                    previous = q
                if failed:
                    continue
                # Validate the planned joint approach and Cartesian descent/lift
                # at fine numerical samples. This does not run a controller.
                trajectory = []
                waypoints = [seed, *[np.array(s["left_arm_rad"]) for s in stages]]
                for before, after in zip(waypoints[:-1], waypoints[1:]):
                    count = max(2, int(np.ceil(np.max(abs(after-before))/.04)))
                    for fraction in np.linspace(0., 1., count+1):
                        q = before+(after-before)*fraction
                        centers, radii, _ = robot.sphere_cloud(q, trunk, width=.1)
                        centers = centers @ rotation.T+base
                        contacts = geometry.colliders(centers, radii, exclude=(*excluded, name), margin=.012)
                        if contacts:
                            rejections["sampled_arm_path_collision"] = rejections.get("sampled_arm_path_collision", 0)+1
                            failed = True
                            break
                        trajectory.append(q.tolist())
                    if failed:
                        break
                if failed:
                    continue
                best = {"base_world_xyz": base.tolist(), "yaw_rad": float(yaw), "trunk_rad":
                        [robot.seed[f"torso_joint{x}"] for x in range(1, 5)] if trunk is None else trunk.tolist(),
                        "grasp_candidate": candidate, "stages": stages,
                        "route": reconstruct_route(nav, tuple(node)),
                        "sampled_arm_joint_path_rad": trajectory,
                        "sampled_arm_path_count": len(trajectory),
                        "configuration_route_cost": float(nav["distance"][tuple(node)])}
                break
            if best:
                break
        if best:
            break
    return {"name": name, "search_status": "candidate_found" if best else "no_candidate_in_bounded_search",
            "ik_attempts": attempted, "minimum_ik_position_error_m": min(errors) if errors else None,
            "rejection_counts": rejections, "grasp_geometry": candidates, "plan": best,
            "scope": "Pose and sampled joint-path screening; exact continuous arm swept-path, self-collision and force closure are not certified."}


def validate_candidates(assessment, robot, geometry):
    """Check open-finger interference, proxy self collision and original chairs."""
    def self_pairs(centers, radii, links):
        left, right = np.triu_indices(len(centers), 1)
        gap = np.linalg.norm(centers[left]-centers[right], axis=1)-radii[left]-radii[right]
        return {tuple(sorted((links[i], links[j]))) for i, j, g in zip(left, right, gap)
                if g < -.001 and links[i] != links[j]
                and frozenset((links[i], links[j])) not in robot.self_ignored}

    cloud, radii, links = robot.sphere_cloud()
    baseline_self = self_pairs(cloud, radii, links)
    results = []
    for obj in assessment["objects"]:
        plan = obj["plan"]
        if plan is None:
            continue
        rotation = Rotation.from_euler("z", plan["yaw_rad"]).as_matrix()
        base = np.array(plan["base_world_xyz"])
        grip = next(stage for stage in plan["stages"] if stage["vertical_offset_m"] == 0.)
        centers, radii, links = robot.sphere_cloud(grip["left_arm_rad"], plan["trunk_rad"], width=.1)
        fingers = np.array([link.startswith("left_gripper") for link in links])
        world = centers @ rotation.T+base
        open_hits = geometry.colliders(world[fingers], radii[fingers],
            exclude=tuple(name for name in geometry.inventory if name != obj["name"]), margin=0.)
        added_self, original_obstacles = set(), set()
        for q in plan["sampled_arm_joint_path_rad"]:
            centers, radii, links = robot.sphere_cloud(q, plan["trunk_rad"], width=.1)
            added_self |= self_pairs(centers, radii, links)-baseline_self
            original_obstacles.update(geometry.colliders(centers @ rotation.T+base, radii,
                                                        exclude=(obj["name"],), margin=.012))
        # Include turn-edge sweep bounds, rather than checking just route nodes.
        route = plan["route"]
        samples = [(pose, 0.) for pose in route]
        for before, after in zip(route[:-1], route[1:]):
            delta = float(np.arctan2(np.sin(after[2]-before[2]), np.cos(after[2]-before[2])))
            middle = [*(.5*(np.array(before[:2])+after[:2])), before[2]+delta/2]
            samples.append((middle, assessment["navigation"]["angular_sweep_margin_m"] if abs(delta) > 1e-6 else 0.))
        seed_cloud, seed_radii, _ = robot.sphere_cloud()
        for pose, extra in samples:
            r = Rotation.from_euler("z", pose[2]).as_matrix()
            p = np.r_[pose[:2], base[2]]
            original_obstacles.update(geometry.colliders(seed_cloud @ r.T+p, seed_radii+extra,
                                                        margin=assessment["navigation"]["collision_margin_m"]))
        excluded = set(assessment["virtual_excluded_obstacles"])
        unaccounted = sorted(original_obstacles-excluded)
        passed = not open_hits and not added_self and not unaccounted
        results.append({"name": obj["name"], "open_finger_target_interference": open_hits,
                        "new_proxy_self_collision_pairs": sorted(added_self),
                        "original_layout_obstacles_on_proposed_route_or_arm_path": sorted(original_obstacles),
                        "unaccounted_obstacles": unaccounted,
                        "passes_additional_proxy_screen": passed})
    return {"baseline_proxy_self_collision_pairs": sorted(baseline_self), "candidates": results,
            "scope": "Numerical proxy validation only; no robot operation, force-closure proof, carried-object path or controller tracking test."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--virtual-clear-chairs", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Validate saved candidate paths without repeating navigation search.")
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    initialization = json.loads((ROOT / "docs/esi_action_order_initialization_latest.json").read_text())
    snapshot = Path(initialization["run_root"])
    exploration = Path(json.loads((ROOT / "docs/esi_category_exploration_latest.json").read_text())["run_root"])
    source = exploration / "native/action_order_inference/tabletop_grasp"
    inventory = json.loads((source / "inventory_before.json").read_text())
    metadata = json.loads((source / "metadata.json").read_text())
    trunk = metadata["arm_control"]["initial_trunk_posture"]["joint_positions_after"]
    Usd, UsdGeom, UsdPhysics = import_usd()
    robot = Robot(Usd, UsdPhysics, trunk)
    observed = metadata["arm_control"]["initial_trunk_posture"]["ee_after"]
    native = pose_matrix(observed["position"], observed["quaternion_xyzw"])
    frame = robot.fk()
    fk_position_error = float(np.linalg.norm(frame[:3, 3]-native[:3, 3]))
    fk_angle_error = float(np.rad2deg(Rotation.from_matrix(native[:3, :3].T @ frame[:3, :3]).magnitude()))
    if fk_position_error > .01 or fk_angle_error > 2.:
        dump(args.output / "fk_failure.json", {"position_error_m": fk_position_error,
             "orientation_error_deg": fk_angle_error, "predicted": frame.tolist(), "saved": native.tolist()})
        raise RuntimeError("Offline model does not reproduce saved native FK; reach conclusions withheld.")
    dump(args.output / "fk_validation.json", {"position_error_m": fk_position_error,
         "orientation_error_deg": fk_angle_error, "predicted": frame.tolist(), "saved": native.tolist()})
    base = metadata["robot_world_pose"]
    position = np.array(base["position"])
    yaw = Rotation.from_quat(base["quaternion_xyzw"]).as_euler("xyz")[2]
    floor = inventory["floors_jymrik_0"]
    low, high = np.array(floor["aabb_min"]), np.array(floor["aabb_max"])
    geometry = Geometry(Usd, UsdGeom, UsdPhysics, inventory)
    for name, row in inventory.items():
        if row["visual_only"] or row["category"] in ("floors", "ceilings", "electric_switch", "picture", "openable_window"):
            continue
        a, b = np.array(row["aabb_min"]), np.array(row["aabb_max"])
        if np.any(a[:2] > high[:2]+.1) or np.any(b[:2] < low[:2]-.1) or a[2] > 1.8:
            continue
        geometry.add(name)
    excluded = tuple(n for n, x in inventory.items() if x["category"] == "straight_chair") if args.virtual_clear_chairs else ()
    dump(args.output / "geometry_validation.json", geometry.validation)
    dump(args.output / "robot_joint_calibration.json", robot.calibration)
    print(json.dumps({"stage": "geometry_ready", "fk_position_error_m": fk_position_error,
                      "fk_orientation_error_deg": fk_angle_error, "collision_hulls": len(geometry.hulls)}), flush=True)
    if args.validate_only:
        assessment = json.loads((args.output / "assessment.json").read_text())
        validation = validate_candidates(assessment, robot, geometry)
        validation["elapsed_wall_s"] = time.monotonic()-started
        dump(args.output / "candidate_validation.json", validation)
        print(json.dumps(validation), flush=True)
        return
    nav = navigation(robot, geometry, floor, [*position, yaw], excluded=excluded)
    reachable = np.isfinite(nav["distance"])
    print(json.dumps({"stage": "navigation_ready", "start_free": bool(nav["free"][nav["start"]]),
                      "reachable_states": int(reachable.sum()), "elapsed_s": time.monotonic()-started}), flush=True)
    objects = []
    for name, row in inventory.items():
        if not row["task_object"]:
            continue
        candidates = grasp_candidates(name, row, geometry)
        result = screen_target(name, row, candidates, robot, geometry, nav, position[2], excluded=excluded)
        objects.append(result)
        print(json.dumps({"stage": "target_complete", "name": name, "status": result["search_status"],
                          "whole_span_m": candidates["minimum_whole_horizontal_span_m"],
                          "ik_attempts": result["ik_attempts"]}), flush=True)
        dump(args.output / "targets.json", objects)
    result = {"algorithm": "CPU USD joint FK, convex collision proxies, 3 cm/30 degree SE(2) Dijkstra, bounded IK and local jaw sections",
              "simulation_started": False, "robot_commands": 0, "object_pose_changes": 0, "model_calls": 0,
              "source_snapshot": str(snapshot), "geometry_inventory": str(source / "inventory_before.json"),
              "virtual_excluded_obstacles": list(excluded), "virtual_variant_only": bool(excluded),
              "fk_validation": {"position_error_m": fk_position_error, "orientation_error_deg": fk_angle_error},
              "navigation": {"resolution_m": nav["resolution"], "yaw_step_deg": 30,
                             "collision_margin_m": nav["collision_margin_m"],
                             "angular_sweep_margin_m": nav["angular_sweep_margin_m"],
                             "start_free": bool(nav["free"][nav["start"]]),
                             "free_states": int(nav["free"].sum()), "connected_states": int(reachable.sum())},
              "objects": objects, "elapsed_wall_s": time.monotonic()-started,
              "limits": ["No end-to-end physical grasp or manipulation result.",
                         "A failed bounded conservative search is not a universal impossibility proof.",
                         "Collision uses installed robot spheres and expanded native asset convex planes; excluded original-object contact is intentional at grasp.",
                         "Local rim/cap contact candidates have not been checked for force closure or native assisted-grasp eligibility.",
                         "Arm joint paths are sampled but exact continuous self/carry collision and native IK controller tracking are not certified."]}
    dump(args.output / "assessment.json", result)
    plot(args.output, inventory, nav, objects)
    print(json.dumps({"output": str(args.output), "elapsed_wall_s": result["elapsed_wall_s"]}), flush=True)


def plot(output, inventory, nav, objects):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    figure, ax = plt.subplots(figsize=(9, 8))
    reachable = np.isfinite(nav["distance"]).any(axis=0)
    image = np.where(reachable, 2, np.where(nav["free"].any(axis=0), 1, 0))
    ax.imshow(image, origin="lower", extent=[nav["xs"][0], nav["xs"][-1], nav["ys"][0], nav["ys"][-1]],
              cmap="Greens", alpha=.65, vmin=0, vmax=2)
    for name, row in inventory.items():
        if row["category"] not in ("coffee_table", "straight_chair", "floor_lamp") and not row["task_object"]:
            continue
        low, high = np.array(row["aabb_min"]), np.array(row["aabb_max"])
        ax.add_patch(Rectangle(low[:2], *(high-low)[:2], fill=False,
                              edgecolor="red" if row["task_object"] else "black", linewidth=1.))
        ax.text(*((low+high)/2)[:2], name.replace("straight_chair_dmcixv_", "chair "), fontsize=7)
    for obj in objects:
        if obj["plan"]:
            route = np.array(obj["plan"]["route"])
            ax.plot(route[:, 0], route[:, 1], label=obj["name"])
            ax.scatter(*obj["plan"]["base_world_xyz"][:2], marker="x", s=70)
    _, i, j = nav["start"]
    ax.scatter(nav["xs"][j], nav["ys"][i], c="blue", marker="*", s=150, label="Saved robot start")
    ax.set(xlabel="World X (m)", ylabel="World Y (m)", title="Offline Action Order configuration-space screen")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "navigation.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
