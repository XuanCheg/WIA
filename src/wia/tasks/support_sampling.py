"""Screen private pickup placements on native same-room support surfaces.

Run through src/wia/runtime/commands/sample_esi_supports.py inside the retained GPU allocation.
This module never supplies a navigation or manipulation policy to the Agent.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import hashlib
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

from wia.native.utils import _array
from .pickup_policy import separate_pickup_support


@dataclass(frozen=True)
class SamplingConfig:
    seed: int = 0
    attempts: int = 12
    placements: int = 3
    wall_budget_s: float = 600.
    minimum_height_m: float = .35
    maximum_height_m: float = 1.10
    edge_margin_m: float = .03
    require_connected_route: bool = False
    allow_added_support: bool = True
    added_support_candidates: int = 3
    minimum_added_support_distance_m: float = 1.25
    preferred_support_names: tuple[str, ...] = ()
    reuse_current_placement: bool = True
    targeted_support_sampling: bool = False
    require_planar_footprint: bool = False
    maximum_stance_distance_m: float = .85
    head_camera_diagnostics: bool = False
    head_camera_trial_mounts: tuple[tuple[float, float, float], ...] = ()


def grid_route(free, start, goal):
    """Find a four-connected route without snapping blocked endpoints to free space."""
    start, goal = tuple(start), tuple(goal)
    for point in (start, goal):
        if not all(0 <= v < size for v, size in zip(point, free.shape)) or not free[point]:
            return None
    queue, previous = deque([start]), {start: None}
    while queue:
        point = queue.popleft()
        if point == goal:
            path = []
            while point is not None:
                path.append(point)
                point = previous[point]
            return path[::-1]
        for delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            other = (point[0] + delta[0], point[1] + delta[1])
            if (other not in previous and 0 <= other[0] < free.shape[0]
                    and 0 <= other[1] < free.shape[1] and free[other]):
                previous[other] = point
                queue.append(other)
    return None


def footprint_clearance(free, resolution, radius):
    """Erode by a full circular robot footprint, including map-edge clearance."""
    padded = np.pad(free, 1, constant_values=False)
    distances = distance_transform_edt(padded)[1:-1, 1:-1] * resolution
    return free & (distances > radius + resolution / 2)


def chassis_footprint(robot):
    """Measure R1's physical chassis rather than its empty articulation root."""
    name = robot.base_footprint_link_name
    low, high = (_array(value) for value in robot.links[name].aabb)
    extent = high - low
    if not np.isfinite(extent).all() or np.any(extent[:2] <= 0):
        raise ValueError("Physical chassis bounds are unavailable.")
    return name, extent, float(np.linalg.norm(extent[:2]) / 2 + .03)


def choose_stance(free, start, cells, *, require_connected_route):
    """Keep an unverified nearby stance distinct from a connected grid route."""
    choices, nearby = [], []
    for cell in cells:
        if all(0 <= v < size for v, size in zip(cell, free.shape)) and free[tuple(cell)]:
            nearby.append(cell)
        path = grid_route(free, start, cell)
        if path is not None:
            choices.append(path)
    if choices:
        path = min(choices, key=len)
        return path, path[-1], True
    if require_connected_route or not nearby:
        raise ValueError("No approximate same-room route or nearby free grasp stance.")
    cell = min(nearby, key=lambda candidate: np.linalg.norm(np.asarray(candidate) - start))
    return [], cell, False


def same_body(path, body):
    return body == path or body.startswith(path + "/")


def support_footprint_policy(name, category, preferred_names, require_planar_footprint=False):
    """Require stable native support; retain the legacy planar screen as opt-in."""
    if require_planar_footprint:
        return "planar_local_raycast"
    if name in preferred_names and any(word in category for word in ("sofa", "chair")):
        return "native_contact_for_named_seating"
    return "native_contact_and_settling"


def upward_layers(hits, support_path, minimum, maximum):
    """Keep actual upward-facing layers rather than the furniture AABB maximum."""
    heights = sorted(float(_array(hit["position"])[2]) for hit in hits
        if hit.get("hit") and same_body(support_path, hit["rigidBody"])
        and float(_array(hit["normal"])[2]) > .95
        and minimum <= float(_array(hit["position"])[2]) <= maximum)
    return [z for i, z in enumerate(heights) if i == 0 or z - heights[i - 1] > .005]


def support_edge_points(low, high, margin, stance_xy, robot_xy, maximum_distance=.85):
    """Project free stance cells onto the usable footprint of a support surface."""
    lower, upper = np.asarray(low)[:2] + margin, np.asarray(high)[:2] - margin
    if np.any(lower >= upper) or not len(stance_xy):
        return np.empty((0, 2)), float('inf')
    points = np.clip(stance_xy, lower, upper)
    distances = np.linalg.norm(points - stance_xy, axis=1)
    valid = distances <= maximum_distance
    order = np.argsort(np.linalg.norm(stance_xy - robot_xy, axis=1), kind='stable')
    points = points[order[valid[order]]]
    if len(points):
        _, indices = np.unique(np.round(points, 6), axis=0, return_index=True)
        points = points[np.sort(indices)][:12]
    return points, float(distances.min())


def navigation_grid(session, robot, report, *, extra_radius=0.):
    """Screen native traversability using the existing conservative chassis policy."""
    import torch

    trav, segmentation = session.env.scene.trav_map, session.env.scene.seg_map
    free = _array(trav.floor_map[0]) == 255
    indices = np.stack(np.indices(free.shape), axis=-1).reshape(-1, 2)
    world = _array(trav.map_to_world(torch.as_tensor(indices)))
    # The installed world_to_map reverses the batch axis too; convert explicitly.
    seg_indices = (world[:, ::-1] / segmentation.map_resolution + segmentation.map_size / 2).astype(int)
    valid = np.all((seg_indices >= 0) & (seg_indices < segmentation.map_size), axis=1)
    room = np.zeros(len(indices), dtype=bool)
    room_id = segmentation.room_ins_name_to_ins_id[session.room_name]
    room[valid] = _array(segmentation.room_ins_map)[tuple(seg_indices[valid].T)] == room_id
    free &= room.reshape(free.shape)
    floor_z = float(trav.floor_heights[0])
    robot_high = _array(robot.aabb[1])
    base_link, base_extent, radius = chassis_footprint(robot)
    for obj in session.env.scene.objects:
        if obj is robot or obj.visual_only or obj.category in ("floors", "floor", "ceilings", "ceiling", "walls", "wall"):
            continue
        low, high = (_array(value) for value in obj.aabb)
        if high[2] <= floor_z + .04 or low[2] >= robot_high[2]:
            continue
        corners = _array(trav.world_to_map(np.array([low[:2], high[:2]])))
        begin, end = np.min(corners, axis=0), np.max(corners, axis=0) + 1
        begin = np.maximum(begin, 0)
        end = np.minimum(end, free.shape)
        free[begin[0]:end[0], begin[1]:end[1]] = False
    report["robot_footprint_radius_m"] = radius
    report["robot_chassis_link"] = base_link
    report["robot_chassis_extent_m"] = base_extent.tolist()
    report["footprint_screen_scope"] = "Chassis bounding circle; arm clearance is not certified."
    return trav, footprint_clearance(free, trav.map_resolution, radius + extra_radius)


class SupportSampler:
    def __init__(self, session, controller, executor, config, output, selection, *, on_accepted_placement=None):
        self.session, self.controller, self.executor = session, controller, executor
        self.config, self.output, self.selection = config, output, selection
        self.on_accepted_placement = on_accepted_placement
        self.scene, self.robot = session.env.scene, session.robot
        self.target = self.scene.object_registry("name", "obj_containee")
        self.rng = np.random.default_rng(config.seed)
        self.started = time.monotonic()
        self.report = {"schema": "screened_support_bank_v1", "seed": config.seed,
            "source_json": selection.definition.source.json_path, "model_calls": 0,
            "source_question_unchanged": True,
            "target_scale_policy": "Declared target scale; no additional scaling during sampling.",
            "verification_level": "approximate_reachability",
            "support_candidate_policy": "native_same_room_surfaces_excluding_initial_work_table_then_separate_added_support",
            "navigation_policy": "nearby_stance_with_optional_route_screen", "physical_grasp_verified": False,
            "config": vars(config), "attempts": [], "accepted": [], "fallback_candidates": [], "status": "running"}
        if config.head_camera_diagnostics:
            from wia.runtime.sim.esi_head_diagnostics import diagnose_head_camera

            self.report["head_camera_diagnostic"] = diagnose_head_camera(session, self.output / "head_camera",
                trial_mounts=config.head_camera_trial_mounts)

    def save(self):
        self.report["wall_elapsed_s"] = time.monotonic() - self.started
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "support_bank.json").write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")

    def check_budget(self):
        if time.monotonic() - self.started >= self.config.wall_budget_s:
            raise TimeoutError("Per-scene sampling wall budget exhausted.")

    def eligible_supports(self, initial_support):
        """Honor ordered named supports without bypassing same-room exclusions."""
        preferred = self.config.preferred_support_names
        if preferred:
            existing = {obj.name: obj for obj in self.scene.objects}
            self.report['preferred_support_diagnostics'] = {name: {
                'present': name in existing,
                **({'declared_rooms': getattr(existing[name], 'in_rooms', None),
                    'centre_room': self.scene.seg_map.get_room_instance_by_point(
                        (_array(existing[name].aabb[0]) + _array(existing[name].aabb[1]))[:2] / 2),
                    'visual_only': existing[name].visual_only,
                    'task_object': name in self.session.task_object_names}
                    if name in existing else {})} for name in preferred}
        supports = [obj for obj in self.scene.objects if obj is not self.target and not obj.visual_only
            and obj.name not in self.session.task_object_names and obj.name != initial_support
            and not obj.name.startswith("pickup_surface_") and "lamp" not in obj.category
            and self.support_in_room(obj)
            and (obj.name in preferred if preferred else any(word in obj.category
                for word in ("table", "shelf", "counter", "cabinet", "desk", "bench")))]
        if preferred:
            supports.sort(key=lambda obj: preferred.index(obj.name))
            self.report["unavailable_preferred_supports"] = [name for name in preferred
                if name not in {obj.name for obj in supports}]
        return supports

    def support_in_room(self, obj):
        """Use native furniture membership before the coarse room raster fallback."""
        rooms = getattr(obj, 'in_rooms', None)
        if rooms:
            return self.session.room_name in rooms
        centre = (_array(obj.aabb[0]) + _array(obj.aabb[1]))[:2] / 2
        return self.scene.seg_map.get_room_instance_by_point(centre) == self.session.room_name

    def rank_supports(self, supports, extent):
        """Prefer furniture with usable edges near actual free same-room floor cells."""
        import torch

        trav, free = self.navigation_grid()
        cells = np.argwhere(free)
        stance_xy = _array(trav.map_to_world(torch.as_tensor(cells))) if len(cells) else np.empty((0, 2))
        robot_xy = np.asarray(self.controller.pose()['position'][:2])
        points, scores = {}, {}
        for obj in supports:
            low, high = (_array(value) for value in obj.aabb)
            points[obj.name], distance = support_edge_points(low, high,
                extent[:2] / 2 + self.config.edge_margin_m, stance_xy, robot_xy,
                maximum_distance=self.config.maximum_stance_distance_m)
            scores[obj.name] = (not len(points[obj.name]), distance)
        if not self.config.preferred_support_names:
            supports = [supports[int(i)] for i in self.rng.permutation(len(supports))]
            supports.sort(key=lambda obj: scores[obj.name])
        self.report['support_edge_candidate_counts'] = {name: len(value) for name, value in points.items()}
        self.report['ranked_support_names'] = [obj.name for obj in supports]
        return supports, points

    def navigation_grid(self):
        return navigation_grid(self.session, self.robot, self.report)

    def route_to_target(self, *, maximum_stance_distance_m=None):
        import torch

        trav, free = self.navigation_grid()
        start = _array(trav.world_to_map(self.controller.pose()["position"][:2])).astype(int)
        centre = (_array(self.target.aabb[0]) + _array(self.target.aabb[1])) / 2
        cells = []
        self.report["initial_grid_cell_free"] = bool(all(0 <= v < size for v, size in zip(start, free.shape))
            and free[tuple(start)])
        distances = (.55,.70,.85) if maximum_stance_distance_m is None else tuple(
            d for d in (.55,.70,.85,maximum_stance_distance_m) if d <= maximum_stance_distance_m)
        if self.config.targeted_support_sampling:
            available = np.argwhere(free)
            if len(available):
                world = _array(trav.map_to_world(torch.as_tensor(available)))
                radius = self.config.maximum_stance_distance_m if maximum_stance_distance_m is None else maximum_stance_distance_m
                actual_distances = np.linalg.norm(world - centre[:2], axis=1)
                nearby = available[actual_distances <= radius]
                self.report["last_stance_screen"] = {
                    "nearest_free_cell_distance_m": float(np.min(actual_distances)),
                    "maximum_stance_distance_m": radius, "nearby_free_cell_count": len(nearby)}
                order = np.argsort(np.linalg.norm(nearby - start, axis=1), kind='stable')
                cells = [cell for cell in nearby[order[:36]]]
            else:
                self.report["last_stance_screen"] = {
                    "nearest_free_cell_distance_m": None, "nearby_free_cell_count": 0}
        else:
            for distance in distances:
                for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    desired = centre[:2] + distance * np.array([np.cos(angle), np.sin(angle)])
                    cell = _array(trav.world_to_map(desired)).astype(int)
                    if maximum_stance_distance_m is not None:
                        actual = _array(trav.map_to_world(torch.as_tensor(cell)))
                        if np.linalg.norm(actual-centre[:2]) > maximum_stance_distance_m:
                            continue
                    cells.append(cell)
        path, cell, connected = choose_stance(free, start, cells,
            require_connected_route=self.config.require_connected_route)
        route = _array(trav.map_to_world(torch.as_tensor(path))).tolist() if path else []
        stance = _array(trav.map_to_world(torch.as_tensor(cell))).tolist()
        return route, stance, connected

    def draft_added_supports(self, target_geometry):
        """Write bounded empty-space drafts; only an independent native restart can accept them."""
        import torch
        import yaml
        from omnigibson.utils.sampling_utils import raytest
        from .scene_only import local_floor_surface

        if not self.config.allow_added_support or self.config.require_connected_route:
            return
        trav, free = self.navigation_grid()
        cells = np.argwhere(free)
        if not len(cells):
            self.report["added_support_reason"] = "No free same-room floor cells for separate furniture."
            return
        floor = self.scene.object_registry("name", self.session.payload.get("floor_name"))
        if floor is None:
            self.report["added_support_reason"] = "Source floor is unavailable."
            return
        floor_low, floor_high = (_array(v) for v in floor.aabb)
        # Use one coherent snapshot; restored root and collision-link reads can
        # otherwise refer to different simulation states before synchronization.
        position, quaternion, target_low, target_high = target_geometry
        extent = target_high - target_low
        dimensions = np.maximum(extent[:2] + .08, .30)
        robot_xy = np.asarray(self.controller.pose()["position"][:2])
        obstacles = [obj for obj in self.scene.objects if not obj.visual_only
            and obj.category not in ("floors", "floor", "ceilings", "ceiling")]
        for cell in cells[self.rng.permutation(len(cells))[:96]]:
            self.check_budget()
            xy = _array(trav.map_to_world(torch.as_tensor(cell)))
            if np.linalg.norm(xy - robot_xy) < self.config.minimum_added_support_distance_m:
                continue
            corners = [xy + np.array([dx, dy]) for dx in (-dimensions[0] / 2, dimensions[0] / 2)
                       for dy in (-dimensions[1] / 2, dimensions[1] / 2)]
            if any(self.scene.seg_map.get_room_instance_by_point(c) != self.session.room_name for c in corners):
                continue
            try:
                heights = [float(_array(local_floor_surface(floor.prim_path, raytest(
                    [*c, float(floor_high[2]) + .1], [*c, float(floor_low[2]) - .1],
                    only_closest=False))["position"])[2]) for c in corners]
            except ValueError:
                continue
            if max(heights) - min(heights) > .008:
                continue
            floor_z = float(np.mean(heights))
            low = np.r_[xy - dimensions / 2 - .03, floor_z + .01]
            high = np.r_[xy + dimensions / 2 + .03, floor_z + .55]
            if any(np.all(np.minimum(high, _array(obj.aabb[1])) - np.maximum(low, _array(obj.aabb[0])) > .003)
                   for obj in obstacles):
                continue
            index = len(self.report["fallback_candidates"])
            try:
                definition, details = separate_pickup_support(self.selection.definition, position_xy=xy,
                    floor_z=floor_z, target_extent=extent, target_bottom_offset=position[2] - target_low[2],
                    target_quaternion=quaternion, index=index)
            except ValueError as error:
                self.report["added_support_reason"] = str(error)
                break
            filename = f"separate_support_{index:03d}.yaml"
            (self.output / filename).write_text(yaml.safe_dump(definition.model_dump(mode="json", by_alias=True), sort_keys=False))
            self.report["fallback_candidates"].append({**details, "scene_file": filename, "accepted": False,
                "initial_robot_distance_m": float(np.linalg.norm(xy - robot_xy)),
                "candidate_space_screen": "Same-room floor, level corners and conservative furniture/robot AABBs."})
            if len(self.report["fallback_candidates"]) >= self.config.added_support_candidates:
                break
        if not self.report["fallback_candidates"]:
            self.report.setdefault("added_support_reason", "No separate pickup furniture passed the bounded space screen.")

    def accept_placement(self, row, support, position, quaternion):
        """Export the exact measured target pose and its declared native support."""
        import yaml

        value = self.selection.definition.model_dump(mode="json", by_alias=True)
        value["id"] += f"_support_{len(self.report['accepted']):03d}"
        variant = value["scene"]["variants"][self.selection.variant]
        variant["patches"] = [patch for patch in variant["patches"]
            if not (patch["operation"] == "object_pose" and patch["binding"] == value["target"])]
        variant["patches"].append({"operation": "object_pose", "phase": "after_postprocessing",
            "binding": value["target"], "position": position, "quaternion_xyzw": quaternion})
        value["bindings"]["pickup_support"] = {"name": support.name, "category": support.category}
        value["qualification"]["object_supports"][value["target"]] = "pickup_support"
        value["seed"] = self.config.seed
        value["status"] = "pending"
        value["qualification"]["support_surface_mode"] = "local_raycast"
        filename = f"placement_{len(self.report['accepted']):03d}.yaml"
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / filename).write_text(yaml.safe_dump(value, sort_keys=False))
        row.update(accepted=True, scene_file=filename,
            scene_sha256=hashlib.sha256((self.output / filename).read_bytes()).hexdigest())
        self.report["accepted"].append(dict(row))
        if self.on_accepted_placement is not None:
            try:
                self.on_accepted_placement(self.output / filename)
            except Exception as error:
                # Preserve the completed support screen if ancillary rendering
                # fails; the review coordinator can capture this layout later.
                self.report.setdefault("review_capture_failures", []).append({
                    "scene_file": filename, "error": f"{type(error).__name__}: {error}"})

    def run(self):
        from omnigibson.utils.sampling_utils import raytest

        sim = self.session.pipeline.og.sim
        baseline = sim.dump_state(serialized=False)
        target_position, target_quaternion = (_array(value).copy() for value in self.target.get_position_orientation())
        target_low, target_high = (_array(value).copy() for value in self.target.aabb)
        initial_geometry = (target_position, target_quaternion, target_low, target_high)
        extent = target_high - target_low
        bottom_offset = target_position[2] - float(target_low[2])
        self.report["initial_target_geometry"] = {"position_m": target_position.tolist(),
            "quaternion_xyzw": target_quaternion.tolist(), "aabb_min_m": target_low.tolist(),
            "aabb_max_m": target_high.tolist(), "origin_above_bottom_m": float(bottom_offset)}
        initial_support = self.selection.definition.bindings[self.selection.definition.qualification.support].name
        supports = self.eligible_supports(initial_support)
        self.report["support_names"] = [obj.name for obj in supports]
        self.report["initial_robot_pose"] = self.controller.pose()
        self.report["excluded_initial_work_support"] = initial_support

        def restore():
            from omnigibson.utils.usd_utils import PoseAPI

            sim.load_state(baseline, serialized=False)
            PoseAPI.invalidate()
            self.target.get_position_orientation()
            self.session.invalidate_camera_cache()

        try:
            if not supports:
                raise ValueError("No native support furniture matches the current candidate categories.")
            edge_points = {}
            if self.config.targeted_support_sampling:
                supports, edge_points = self.rank_supports(supports, extent)
            current_key = self.selection.definition.qualification.object_supports.get(
                self.selection.definition.target, self.selection.definition.qualification.support)
            current_name = self.selection.definition.bindings[current_key].name
            current_support = next((s for s in supports if s.name == current_name), None)
            if current_support is not None and self.config.reuse_current_placement:
                # Native initialization has already qualified this off-table
                # pose. A missing-object note does not require moving it again.
                row = {"index": -1, "support": current_name, "accepted": False,
                    "placement_reused": True, "position": target_position.tolist(),
                    "quaternion_xyzw": target_quaternion.tolist()}
                try:
                    low, high = (_array(v) for v in self.target.aabb)
                    xy = ((low + high) / 2)[:2]
                    layers = upward_layers(raytest([*xy, float(low[2]) + .03],
                        [*xy, float(low[2]) - .10], only_closest=False), current_support.prim_path,
                        self.config.minimum_height_m, self.config.maximum_height_m)
                    if not layers:
                        raise ValueError("Existing pickup support lies outside the configured height window.")
                    contacts = self.target.contact_list()
                    if not contacts or not all(any(same_body(current_support.prim_path, getattr(c, side))
                        for side in ("body0", "body1")) for c in contacts):
                        raise ValueError("Existing pickup placement contacts unexpected geometry.")
                    route, stance, connected = self.route_to_target()
                    row.update(route_xy_m=route, approximate_grasp_stance_xy_m=stance,
                        route_connectivity_screen_passed=connected, minimum_target_extent_m=float(np.min(extent)),
                        aperture_warning=bool(np.min(extent) > .1), support_surface_z_m=layers[-1])
                    self.accept_placement(row, current_support, target_position.tolist(), target_quaternion.tolist())
                except (ValueError, RuntimeError) as error:
                    row["reason"] = str(error)
                self.report["attempts"].append(row)
            order = np.arange(len(supports)) if self.config.targeted_support_sampling else self.rng.permutation(len(supports))
            for attempt in range(self.config.attempts):
                if len(self.report["accepted"]) >= self.config.placements:
                    break
                if (self.config.allow_added_support and not self.report["accepted"]
                        and time.monotonic() - self.started >= .8 * self.config.wall_budget_s):
                    self.report["native_sampling_stop_reason"] = "Reserve remaining wall budget for separate-support drafts."
                    break
                self.check_budget()
                restore()
                if self.config.preferred_support_names:
                    # Exhaust each preference before its explicitly requested fallback.
                    per_support = int(np.ceil(self.config.attempts / len(supports)))
                    support = supports[min(attempt // per_support, len(supports) - 1)]
                    local_attempt = attempt % per_support
                else:
                    support = supports[int(order[attempt % len(order)])]
                    local_attempt = attempt // len(order)
                row = {"index": attempt, "support": support.name, "accepted": False}
                try:
                    low, high = (_array(value) for value in support.aabb)
                    margin = extent[:2] / 2 + self.config.edge_margin_m
                    if np.any(high[:2] - low[:2] <= 2 * margin):
                        raise ValueError("Support footprint is smaller than the target and edge margin.")
                    proposed = edge_points.get(support.name, ())
                    if len(proposed) and local_attempt % 2 == 0:
                        xy = proposed[(local_attempt // 2) % len(proposed)]
                    elif self.config.targeted_support_sampling and local_attempt == 1:
                        xy = (low[:2] + high[:2]) / 2
                    else:
                        xy = self.rng.uniform(low[:2] + margin, high[:2] - margin)
                    row["proposed_xy_m"] = np.asarray(xy).tolist()
                    if (not getattr(support, 'in_rooms', None)
                            and self.scene.seg_map.get_room_instance_by_point(xy) != self.session.room_name):
                        raise ValueError("Sampled support patch is outside the source room.")
                    hits = raytest([*xy, high[2] + .10], [*xy, low[2] - .10], only_closest=False)
                    layers = upward_layers(hits, support.prim_path, self.config.minimum_height_m, self.config.maximum_height_m)
                    if not layers:
                        raise ValueError("No horizontal layer within the configured height window.")
                    z = float(self.rng.choice(layers))
                    row["support_surface_z_m"] = z
                    policy = support_footprint_policy(support.name, support.category,
                        self.config.preferred_support_names, self.config.require_planar_footprint)
                    row["support_footprint_policy"] = policy
                    if policy == "planar_local_raycast":
                        for dx in (-margin[0], 0., margin[0]):
                            for dy in (-margin[1], 0., margin[1]):
                                corner = raytest([xy[0] + dx, xy[1] + dy, z + .01],
                                    [xy[0] + dx, xy[1] + dy, z - .02])
                                if (not corner["hit"] or not same_body(support.prim_path, corner["rigidBody"])
                                        or float(_array(corner["normal"])[2]) < .95
                                        or abs(float(_array(corner["position"])[2]) - z) > .005):
                                    raise ValueError("Local footprint or edge-margin support check failed.")
                    self.target.set_position_orientation(position=np.r_[xy, z + bottom_offset + .01], orientation=target_quaternion)
                    self.target.keep_still()
                    for _ in range(60):
                        sim.step()
                    contacts = self.target.contact_list()
                    def support_contact(contact):
                        return any(same_body(support.prim_path, getattr(contact, side)) for side in ("body0", "body1"))
                    if not contacts or not all(support_contact(contact) for contact in contacts):
                        raise ValueError("Target contacts unexpected geometry or lacks support contact.")
                    if (np.linalg.norm(_array(self.target.get_linear_velocity())) > .02
                            or np.linalg.norm(_array(self.target.get_angular_velocity())) > .1
                            or abs(float(_array(self.target.aabb[0])[2]) - z) > .015):
                        raise ValueError("Target settling or local support gap check failed.")
                    if policy != "planar_local_raycast":
                        settled_low, settled_high = (_array(value) for value in self.target.aabb)
                        if np.any(settled_low[:2] < low[:2]) or np.any(settled_high[:2] > high[:2]):
                            raise ValueError("Settled target footprint leaves the support furniture bounds.")
                    row["physical_support_and_settling_passed"] = True
                    position, quaternion = (_array(value).tolist() for value in self.target.get_position_orientation())
                    row.update(position=position, quaternion_xyzw=quaternion)
                    route, stance, connected = self.route_to_target()
                    row.update(position=position, quaternion_xyzw=quaternion, support_surface_z_m=z, route_xy_m=route)
                    row["approximate_grasp_stance_xy_m"] = stance
                    row["route_connectivity_screen_passed"] = connected
                    row["minimum_target_extent_m"] = float(np.min(extent))
                    row["aperture_warning"] = bool(np.min(extent) > .1)
                    self.accept_placement(row, support, position, quaternion)
                except (ValueError, RuntimeError) as error:
                    row["reason"] = f"{type(error).__name__}: {error}"
                if row.get("physical_support_and_settling_passed") and self.config.targeted_support_sampling:
                    row["stance_screen"] = self.report.get("last_stance_screen", {})
                self.report["attempts"].append(row)
                self.save()
                print("[support-sampler]", attempt, support.name, row.get("reason", "screened"), flush=True)
                if len(self.report["accepted"]) >= self.config.placements:
                    break
            self.report["status"] = "screened" if self.report["accepted"] else "not_screened"
        except (ValueError, RuntimeError, TimeoutError) as error:
            self.report.update(status="screened" if self.report["accepted"] else "not_screened", reason=str(error))
        finally:
            restore()
            if not self.report["accepted"]:
                try:
                    self.draft_added_supports(initial_geometry)
                except (ValueError, RuntimeError, TimeoutError) as error:
                    self.report["added_support_reason"] = str(error)
            self.save()
        return self.report
