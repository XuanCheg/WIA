"""Read-only background and fully-open-container base-route screening.

Run from the native simulator owner thread after loading a scene::

    nav, evidence = build_background_navigation(session, identity="candidate_id")
    route = connected_open_route(nav, stance_xy, face_yaw, open_mesh_aabbs)

CPU syntax check: ``python -m py_compile src/wia/candidates/container_navigation.py``.
This module never steps physics, sets robot/object state, renders, or calls a model.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from wia.candidates.inventory import export_inventory
from wia.candidates.reachability import (
    AuditOptions, AuditUnavailable, _Robot, _SE2Navigation, _angle_delta, _box,
    _convex_hull_2d, _navigation_evidence, polygon_aabb_intersections,
)


_CANDIDATE_PREFIXES = ("item_", "unit_", "zone_", "anchor_")
_SCOPE = ("Base-footprint-only geometric route in the measured background with supplied fully-open "
          "container AABBs. Full-body/arm clearance, door-opening motion, carrying, controller tracking "
          "and task success are untested; no robot or model commands are executed.")


def _candidate_row(row):
    return any(str(row.get(key, "")).startswith(_CANDIDATE_PREFIXES)
               for key in ("name", "parent_object"))


def build_background_navigation(session, identity):
    """Return ``(navigator, provenance)`` using native read-only measurements.

    Candidate prefixed rows are omitted only from this background navigator.
    Their actual open-state mesh AABBs must be supplied to ``connected_open_route``.
    Missing measurements or a blocked/unconnected native start raise explicitly.
    """
    inventory = export_inventory(session, identity=identity, target_names=[])
    floor_z = float(inventory["floor_z"])
    if not np.isfinite(floor_z):
        raise AuditUnavailable("A finite measured supporting floor height is required.")
    objects, candidate_rows, supporting_floor_rows = [], [], []
    for row in inventory["objects"]:
        box = _box(row["aabb"])
        if _candidate_row(row):
            candidate_rows.append(row["name"])
        elif (str(row.get("role", "")).lower() in {"floor", "ground"}
              and box[1, 2] <= floor_z + .03):
            # Match audit_scene's supporting-plane filter exactly.
            supporting_floor_rows.append(row["name"])
        else:
            objects.append(row)
    options = AuditOptions(navigation_mode="se2_rectangle")
    robot = _Robot(inventory["robot"], options)
    nav = _SE2Navigation(robot, objects, [], options)
    if nav.failure:
        raise AuditUnavailable(f"Native background navigation unavailable: {nav.failure}")
    base_z_range = robot.base_box[:, 2].copy()
    nav.container_base_z_range_m = base_z_range
    provenance = {
        "schema": "wia_container_background_navigation_v1", "identity": str(identity),
        "native_robot": deepcopy(inventory["robot"]), "floor_z": floor_z,
        "floor_measurement": deepcopy(inventory.get("floor_measurement")),
        "base_z_range_m": base_z_range.tolist(), "obstacle_row_count": len(objects),
        "base_height_obstacle_count": len(nav.obstacles), "lattice_obstacle_count": len(nav.boxes),
        "removed_candidate_collision_rows": candidate_rows,
        "removed_supporting_floor_plane_rows": supporting_floor_rows,
        "navigation": _navigation_evidence(nav, options),
        "base_geometry_validated": nav.footprint_evidence.get("base_geometry_validated") is True,
        "scope": _SCOPE, "robot_commands_executed": False, "model_calls_executed": False,
    }
    return nav, provenance


def _extra_base_height_boxes(extra_world_aabbs, base_z_range):
    boxes, retained, omitted = [], [], []
    for index, row in enumerate(extra_world_aabbs):
        box = _box(row["aabb"] if isinstance(row, dict) else row)
        name = str(row.get("name", index)) if isinstance(row, dict) else str(index)
        if box[0, 2] <= base_z_range[1] and box[1, 2] >= base_z_range[0]:
            boxes.append(box[:, :2])
            retained.append(name)
        else:
            omitted.append(name)
    return np.asarray(boxes, float).reshape(-1, 2, 2), retained, omitted


def _segment_intersects_extra(nav, before, after, boxes):
    begin, end = np.asarray(before, float), np.asarray(after, float)
    first = nav.local_polygon @ nav._rotation(begin[2]).T + begin[:2]
    last = nav.local_polygon @ nav._rotation(end[2]).T + end[:2]
    hull = _convex_hull_2d(np.vstack((first, last)))
    delta = _angle_delta(end[2], begin[2])
    rotation_inflation = 2 * nav.radius * np.sin(abs(delta) / 4)
    inflation = nav.options.collision_margin_m + rotation_inflation
    return bool(np.any(polygon_aabb_intersections(hull, boxes, inflation)))


def connected_open_route(nav, stance_xy, face_yaw, extra_world_aabbs, *, base_z_range=None):
    """Return an exact base route with evidence, or ``None`` when none is found.

    The measured vertical range comes from ``build_background_navigation`` unless
    supplied explicitly as ``[native_base_aabb_low_z, native_base_aabb_high_z]``.
    Extra obstacles are retained whenever their closed world-Z intervals overlap
    that range. No initializer connectivity hint or candidate polygon is trusted.
    """
    point, yaw = np.asarray(stance_xy, float), float(face_yaw)
    if point.shape != (2,) or not np.isfinite(point).all() or not np.isfinite(yaw):
        raise AuditUnavailable("A finite exact world-XY stance and facing yaw are required.")
    if base_z_range is None:
        base_z_range = getattr(nav, "container_base_z_range_m", None)
    vertical = np.asarray(base_z_range, float)
    if vertical.shape != (2,) or not np.isfinite(vertical).all() or vertical[0] > vertical[1]:
        raise AuditUnavailable("The measured native base AABB vertical extent is required.")
    boxes, retained, omitted = _extra_base_height_boxes(extra_world_aabbs, vertical)
    if nav.failure:
        return None
    connection = nav._stance_connection(point, yaw, {})
    if connection is None:
        return None
    node, exact_point, exact_yaw = connection
    route = deepcopy(nav.route(node))
    # route() already includes these two connectors for an existing canonical
    # stance mapping. Reuse them; never overwrite the navigator's node mapping.
    if node not in nav.stance_positions:
        route.extend([[*np.asarray(exact_point).tolist(), nav.heading(node[0])],
                      [*np.asarray(exact_point).tolist(), float(exact_yaw)]])
    if not route:
        return None
    for index in range(1, len(route)):
        route[index][2] = route[index - 1][2] + _angle_delta(route[index][2], route[index - 1][2])
    # Include a stationary segment so even a zero-length route checks its pose.
    segments = list(zip(route[:-1], route[1:])) or [(route[0], route[0])]
    if any(_segment_intersects_extra(nav, before, after, boxes) for before, after in segments):
        return None
    return {
        "route_world_xy_yaw_rad": route, "lattice_node": list(node),
        "exact_stance_world_xy_m": np.asarray(exact_point).tolist(), "face_yaw_rad": float(exact_yaw),
        "base_z_range_m": vertical.tolist(), "extra_world_aabb_count": len(retained) + len(omitted),
        "base_height_extra_aabb_names": retained, "extra_aabb_names_outside_base_height": omitted,
        "checked_segment_count": len(segments), "collision_margin_m": nav.options.collision_margin_m,
        "rotation_inflation_formula": "2 * maximum_corner_radius_m * sin(abs(wrapped_segment_delta_yaw_rad) / 4)",
        "scope": _SCOPE, "robot_commands_executed": False, "model_calls_executed": False,
    }
