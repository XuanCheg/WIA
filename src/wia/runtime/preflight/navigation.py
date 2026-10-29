"""Bounded native-map navigation diagnostics. Run preflight_esi_tasks.py --mode full."""
from __future__ import annotations

import numpy as np

from wia.tasks.support_sampling import grid_route, navigation_grid
from wia.runtime.sim.esi_omnigibson import _array
from wia.runtime.preflight.models import Check


def screen_navigation(context, session, controller, options):
    """Reuse the support sampler's map and footprint; failed screens prove nothing."""
    import torch

    details = {"policy": "conservative_native_traversability_and_object_bounds",
               "arm_sweep_certified": False, "dependencies": list(context.dependencies)}
    trav, free = navigation_grid(session, session.robot, details)
    start = _array(trav.world_to_map(controller.pose()["position"][:2])).astype(int)
    details["initial_cell"] = start.tolist()
    details["initial_cell_free"] = bool(np.all(start >= 0) and np.all(start < free.shape) and free[tuple(start)])
    details["resolution_m"] = float(trav.map_resolution)
    doors = []
    for obj in session.env.scene.objects:
        if obj.category in ("door", "doors"):
            doors.append({"name": obj.name, "joint_count": len(obj.joints),
                          "conditional_obstacle": True})
    details["dynamic_obstacles"] = doors
    checks, rows = [], {}
    for key in dict.fromkeys(context.targets + context.containers):
        obj = session.env.scene.object_registry("name", context.bindings[key].name)
        if obj is None:
            checks.append(Check("navigation", "navigation." + key, "FAIL", "Bound navigation object is missing.", [key]))
            continue
        centre = (_array(obj.aabb[0]) + _array(obj.aabb[1])) / 2
        candidates, seen = [], set()
        # Include the measured stance first; never snap a blocked start to free space.
        positions = [np.asarray(controller.pose()["position"][:2])]
        for distance in (.55, .70, .85):
            positions.extend(centre[:2] + distance * np.array([np.cos(a), np.sin(a)])
                             for a in np.linspace(0, 2 * np.pi, options.stance_count, endpoint=False))
        for desired in positions:
            if np.linalg.norm(desired - centre[:2]) > 1.05:
                continue
            cell = _array(trav.world_to_map(desired)).astype(int)
            if tuple(cell) in seen:
                continue
            seen.add(tuple(cell))
            route = grid_route(free, start, cell)
            if route is None:
                continue
            world = _array(trav.map_to_world(torch.as_tensor(route))).tolist()
            stance = world[-1]
            yaw = float(np.arctan2(centre[1] - stance[1], centre[0] - stance[0]))
            candidates.append({"stance_xy": stance, "yaw_rad": yaw, "route_xy": world,
                               "distance_m": float((len(route) - 1) * trav.map_resolution),
                               "turning_clearance": "chassis_bounding_circle"})
        candidates.sort(key=lambda row: row["distance_m"])
        candidates = candidates[:options.stance_count]
        row = {"object": obj.name, "candidate_count": len(candidates), "candidates": candidates,
               "connected": bool(candidates), **details}
        rows[key] = row
        checks.append(Check("navigation", "navigation." + key, "PASS" if candidates else "INCONCLUSIVE",
            "Connected chassis-circle grid stances found; this certifies only the geometric screen." if candidates
            else "No connected stance in the conservative grid; dynamic interactions or unsampled routes may permit access.",
            [key], row, verification_level="geometric_screen"))
        # A chassis-only map cannot certify the envelope of a held object/arm.
        if key in context.targets:
            checks.append(Check("navigation", "carried_clearance." + key, "INCONCLUSIVE",
                "Loaded-arm sweep and transport to each destination require physical replay.", [key],
                {"target_extent_m": (_array(obj.aabb[1]) - _array(obj.aabb[0])).tolist(),
                 "destinations": list(context.containers)}, verification_level="geometric_screen"))
    return checks, rows
