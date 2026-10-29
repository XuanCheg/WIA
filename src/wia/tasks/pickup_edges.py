"""Place movable task props near their current support edges during initialization.

Run through release scene initialization or an explicit pickup_edge_margin_m.
This uses private scene geometry only during initialization, never agent inputs.
"""
from itertools import product
from collections import Counter

import numpy as np

from wia.tasks.interaction_setup import array as _array


def edge_candidates(support_bounds, object_bounds, position, margin):
    """Return root XY proposals keeping the complete AABB inside the support."""
    slo, shi = np.asarray(support_bounds, dtype=float)
    low, high = np.asarray(object_bounds, dtype=float)
    position = np.asarray(position, dtype=float)
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError('Pickup edge margin must be finite and positive.')
    lower = slo[:2] + margin + position[:2] - low[:2]
    upper = shi[:2] - margin + position[:2] - high[:2]
    if np.any(lower > upper):
        raise ValueError('Pickup object does not fit on the support with the requested margin.')
    rows = []
    for axis, sign in product((0, 1), (-1, 1)):
        other = 1 - axis
        # Preserve the authored along-edge coordinate: only one axis moves.
        if not lower[other] <= position[other] <= upper[other]:
            continue
        xy = position[:2].copy()
        xy[axis] = lower[axis] if sign < 0 else upper[axis]
        rows.append((xy, axis, sign))
    return rows


def surface_edge_candidates(support_bounds, object_bounds, position, margin):
    """Search from each AABB edge inward, without any along-edge movement."""
    faces = edge_candidates(support_bounds, object_bounds, position, margin)
    root = np.asarray(position)[:2]
    for fraction in (1., *np.arange(.95, 0., -.05)):
        for xy, axis, sign in faces:
            yield root + fraction * (xy - root), axis, sign, fraction


def place_pickup_near_edge(session, obj, support, margin=.03, room=None, approach_xy=None, *, moving_objects=()):
    """Require stable support; prefer screened stances and report their absence."""
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Pickup edge placement is initialization-only.')
    from omnigibson.utils.sampling_utils import raytest
    from wia.tasks.support_sampling import same_body, upward_layers, chassis_footprint, footprint_clearance

    scene, robot = session.env.scene, session.robot
    pos, quat = (_array(v) for v in obj.get_position_orientation())
    bounds = np.array([_array(v) for v in obj.aabb])
    sbounds = np.array([_array(v) for v in support.aabb])
    slo, shi = sbounds
    chassis = robot.links[robot.base_footprint_link_name]
    hull = _array(chassis.collision_boundary_points_world)
    chassis_origin = _array(chassis.get_position_orientation()[0])
    radius = float(np.linalg.norm(hull[:, :2]-chassis_origin[:2], axis=1).max() + .03)
    floor_z = float(scene.trav_map.floor_heights[0])
    robot_xy = _array(robot.get_position_orientation()[0])[:2]
    room = room or scene.seg_map.get_room_instance_by_point(pos[:2])
    pending_bounds = getattr(session, '_edge_placement_bounds', {})
    obstacles = []
    for other in scene.objects:
        if other is obj or any(other is member for member in moving_objects) or other is robot or getattr(other, 'visual_only', False) or other.category in ('floors', 'floor', 'ceilings', 'ceiling', 'walls', 'wall'):
            continue
        obstacles.append((other, np.asarray(pending_bounds[other.name]) if other.name in pending_bounds
                          else np.array([_array(v) for v in other.aabb])))
    free = footprint_clearance(_array(scene.trav_map.floor_map[0]) == 255, scene.trav_map.map_resolution, radius)
    if approach_xy is None:
        import torch
        cells = np.argwhere(free)
        stances = _array(scene.trav_map.map_to_world(torch.as_tensor(cells)))
        nearby = np.all((stances >= slo[:2]-1.) & (stances <= shi[:2]+1.), axis=1)
        stances = stances[nearby]
        stances = np.array([point for point in stances if scene.seg_map.get_room_instance_by_point(point) == room]).reshape(-1, 2)
        for other, b in obstacles:
            if b[1, 2] > floor_z+.04 and b[0, 2] < floor_z+1.4:
                stances = stances[np.linalg.norm(stances-np.clip(stances, b[0,:2], b[1,:2]), axis=1) >= radius]
    rectangular_screen = approach_xy is None and not len(stances)
    if rectangular_screen:
        raw_cells = np.argwhere(_array(scene.trav_map.floor_map[0]) == 255)
        raw_world = _array(scene.trav_map.map_to_world(torch.as_tensor(raw_cells)))
        raw_world = raw_world[np.all((raw_world >= slo[:2]-1.) & (raw_world <= shi[:2]+1.), axis=1)]
        raw_world = np.array([point for point in raw_world if scene.seg_map.get_room_instance_by_point(point) == room]).reshape(-1,2)
        stances = raw_world
        from scipy.spatial.transform import Rotation
        base_pos, base_quat = (_array(v) for v in robot.get_position_orientation())
        local_hull = Rotation.from_quat(base_quat).inv().apply(hull-base_pos)
        raw_free = _array(scene.trav_map.floor_map[0]) == 255
    proposals = []
    rejected = Counter()
    # AABB edges are proposals, not necessarily physical surface boundaries.
    # Search inward only on faces whose outer footprint has no supporting surface.
    # Keep the other coordinate fixed and stop at the first supported footprint.
    surface_faces = set()
    for xy, axis, sign, fraction in surface_edge_candidates(sbounds, bounds, pos, margin):
        # Preserve the already validated direct-edge rule whenever it succeeds.
        if fraction < 1. and any(proposal[-1] == 1. for proposal in proposals):
            break
        if (axis, sign) in surface_faces:
            continue
        if scene.seg_map.get_room_instance_by_point(xy) != room:
            rejected['object_room'] += 1
            continue
        if approach_xy is not None:
            stance = np.asarray(approach_xy, dtype=float)
            if scene.seg_map.get_room_instance_by_point(stance) != room or np.linalg.norm(xy-stance) > .8:
                rejected['authored_approach'] += 1
                continue
        else:
            edge = slo[axis] if sign < 0 else shi[axis]
            eligible = stances[(sign*(stances[:,axis]-edge) > 0) & (np.linalg.norm(stances-xy, axis=1) <= .8)]
            if rectangular_screen:
                yaw = (0 if sign < 0 else np.pi) if axis == 0 else (np.pi/2 if sign < 0 else -np.pi/2)
                offsets = Rotation.from_euler('z', yaw).apply(local_hull)[:, :2]
                foot_low, foot_high = offsets.min(axis=0)-.02, offsets.max(axis=0)+.02
                clear = []
                for point in eligible:
                    low_xy, high_xy = point+foot_low, point+foot_high
                    if any(b[1,2] > floor_z+.04 and b[0,2] < floor_z+1.4
                           and np.all(high_xy > b[0,:2]) and np.all(low_xy < b[1,:2]) for other,b in obstacles):
                        continue
                    corners = _array(scene.trav_map.world_to_map(np.array([low_xy,high_xy]))).astype(int)
                    first,last = corners.min(axis=0),corners.max(axis=0)+1
                    if np.any(first < 0) or np.any(last > raw_free.shape) or not raw_free[first[0]:last[0], first[1]:last[1]].all():
                        continue
                    clear.append(point)
                eligible = np.array(clear).reshape(-1,2)
            if not len(eligible):
                rejected['no_nearby_stance'] += 1
                stance = None
            else:
                distances = np.linalg.norm(eligible-xy, axis=1) + .01*np.linalg.norm(eligible-robot_xy, axis=1)
                stance = eligible[np.argmin(distances)]
        centre = xy + (bounds[0, :2] + bounds[1, :2]) / 2 - pos[:2]
        layers = upward_layers(raytest([*centre, shi[2]+.10], [*centre, slo[2]-.10], only_closest=False), support.prim_path, slo[2], shi[2]+.001)
        if not layers:
            rejected['surface_layer'] += 1
            continue
        z = min(layers, key=lambda value: abs(value-bounds[0, 2]))
        if abs(z - bounds[0, 2]) > .03:
            rejected['different_surface_level'] += 1
            continue
        half = (bounds[1, :2]-bounds[0, :2])/2
        if fraction < 1.:
            half = half + margin
        supported = True
        for dx, dy in product((-1, 0, 1), repeat=2):
            point = centre + half * [dx, dy]
            hits = raytest([*point, z+.015], [*point, z-.015], only_closest=False)
            if not any(h.get('hit') and same_body(support.prim_path, h['rigidBody'])
                       and float(_array(h['normal'])[2]) > .95 and abs(float(_array(h['position'])[2])-z) < .005 for h in hits):
                supported = False
                break
        if not supported:
            rejected['footprint_support'] += 1
            continue
        surface_faces.add((axis, sign))
        target = np.r_[xy, pos[2] + z - bounds[0, 2] + .003]
        shifted = bounds + target-pos
        if any(other is not support and np.all(shifted[1] > b[0]+.001) and np.all(shifted[0] < b[1]-.001) for other, b in obstacles):
            rejected['object_overlap'] += 1
            continue
        score = np.linalg.norm(xy-pos[:2]) + (.1*np.linalg.norm(xy-stance) if approach_xy is not None else 0.)
        proposals.append((stance is None, score, target, stance, z, fraction))
    if not proposals:
        raise ValueError('No supported accessible edge placement found for ' + obj.name + ': ' + str(dict(rejected)) + '; room=' + str(room) + '; radius=' + str(radius) + '; stances=' + str(stances[:8].tolist() if approach_xy is None else [approach_xy]))
    missing_stance, _, target, stance, z, fraction = min(proposals, key=lambda item: item[:2])
    members = tuple(moving_objects) or (obj,)
    member_bounds = {member.name: np.array([_array(v) for v in member.aabb]) for member in members}
    obj.set_position_orientation(position=target.tolist(), orientation=quat.tolist())
    obj.keep_still()
    # Native AABB caches can lag pose writes until the next physics step.
    # Screen later placements against the already committed proposals.
    for name, previous in member_bounds.items():
        pending_bounds[name] = previous + target - pos
    session._edge_placement_bounds = pending_bounds
    receipt = {'object': obj.name, 'support': support.name, 'before_position_m': pos.tolist(),
               'position_m': target.tolist(), 'edge_margin_m': margin,
               'surface_edge_search_fraction': float(fraction),
               'surface_edge_search_step_fraction': .05,
               'surface_edge_search_used': bool(fraction < 1.), 'chassis_radius_m': radius, 'screened_stance_xy_m': stance.tolist() if stance is not None else None,
               'same_room_stance_found': not missing_stance,
               'support_surface_z_m': z, 'authored_approach': approach_xy is not None, 'rectangular_stance_screen': rectangular_screen, 'physical_grasp_verified': False, 'route_connectivity_verified': False}
    return receipt


class _PlacementGroup:
    """Translate a related set rigidly without altering its internal relations."""
    def __init__(self, members):
        self.members = members
        self.name = members[0].name

    @property
    def aabb(self):
        bounds = np.array([[_array(v) for v in obj.aabb] for obj in self.members])
        return bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)

    def get_position_orientation(self):
        return self.members[0].get_position_orientation()

    def set_position_orientation(self, position, orientation):
        delta = np.asarray(position) - _array(self.get_position_orientation()[0])
        poses = [(obj, *(_array(v) for v in obj.get_position_orientation())) for obj in self.members]
        for obj, pos, quat in poses:
            obj.set_position_orientation(position=(pos + delta).tolist(), orientation=quat.tolist())

    def keep_still(self):
        for obj in self.members:
            obj.keep_still()


def related_components(objects):
    """Conservatively keep overlapping/touching task props in one rigid group."""
    bounds = [np.array([_array(v) for v in obj.aabb]) for obj in objects]
    remaining = set(range(len(objects)))
    groups = []
    while remaining:
        pending = [min(remaining)]
        remaining.remove(pending[0])
        component = []
        while pending:
            i = pending.pop()
            component.append(objects[i])
            neighbors = [j for j in sorted(remaining)
                         if np.all(bounds[i][1] + .005 >= bounds[j][0])
                         and np.all(bounds[j][1] + .005 >= bounds[i][0])]
            remaining.difference_update(neighbors)
            pending.extend(neighbors)
        groups.append(component)
    return groups


def _current_support(session, obj, excluded):
    """Read the current upward surface under the footprint, never choose new furniture."""
    from omnigibson.utils.sampling_utils import raytest
    from wia.tasks.support_sampling import same_body
    low, high = (_array(v) for v in obj.aabb)
    centre = (low + high) / 2
    hits = raytest([*centre[:2], low[2] + .02], [*centre[:2], low[2] - .04], only_closest=False)
    choices = []
    for hit in hits:
        if not hit.get('hit') or float(_array(hit['normal'])[2]) < .95:
            continue
        for support in session.env.scene.objects:
            if (any(support is member for member in excluded) or support is session.robot
                    or getattr(support, 'visual_only', False)):
                continue
            if same_body(support.prim_path, hit['rigidBody']):
                choices.append((abs(float(_array(hit['position'])[2]) - low[2]), support))
    return min(choices, key=lambda item: item[0])[1] if choices else None


def apply_edge_policy(session, objects, *, supports=None, protected_reason=None):
    """Apply the release policy with explicit per-object unchanged reasons.

    Fixed apparatus and held objects are retained. Related movable props translate
    together. Unsupported or obstructed proposals leave the original poses intact.
    """
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Edge policy is initialization-only.')
    receipts = getattr(session, 'pickup_edge_placements', None)
    if receipts is None:
        receipts = session.pickup_edge_placements = []
    seen = {row['object'] for row in receipts}
    held = tuple(getattr(session.robot, '_ag_obj_in_hand', {}).values())
    movable = []
    for obj in objects:
        if obj is None or obj.name in seen or obj is session.robot:
            continue
        seen.add(obj.name)
        reason = protected_reason
        if getattr(obj, 'fixed_base', False) or getattr(obj, 'visual_only', False):
            reason = 'fixed_apparatus_or_visual_only'
        elif any(obj is value for value in held):
            reason = 'currently_held'
        if reason:
            receipts.append({'object': obj.name, 'status': 'unchanged', 'reason': reason})
        else:
            movable.append(obj)
    new = []
    for members in related_components(movable):
        proxy = members[0] if len(members) == 1 else _PlacementGroup(members)
        before = {obj.name: [_array(v).copy() for v in obj.get_position_orientation()] for obj in members}
        declared = {supports[obj.name] for obj in members if supports and obj.name in supports}
        support = None
        reason = None
        if len(declared) > 1:
            reason = 'related_group_has_multiple_declared_supports'
        elif declared:
            support = session.env.scene.object_registry('name', next(iter(declared)))
        else:
            support = _current_support(session, proxy, members)
        if not reason and (support is None or support.category in ('floor', 'floors', 'wall', 'walls')):
            reason = 'no_current_nonfloor_support'
        if not reason:
            try:
                receipt = place_pickup_near_edge(session, proxy, support, moving_objects=members)
            except ValueError as error:
                reason = str(error)
        for obj in members:
            if reason:
                row = {'object': obj.name, 'status': 'unchanged', 'reason': reason}
            else:
                row = dict(receipt, object=obj.name, status='placed',
                           before_position_m=before[obj.name][0].tolist(),
                           before_quaternion_xyzw=before[obj.name][1].tolist(),
                           position_m=_array(obj.get_position_orientation()[0]).tolist(),
                           related_objects=[member.name for member in members])
            receipts.append(row)
            new.append(row)
    return new


def refresh_edge_references(evidence, placements, session):
    """Update existing measured offsets after legacy setup, before baseline capture."""
    if not isinstance(evidence, dict):
        return
    translations = evidence.get('object_translations_m', {})
    geometry = evidence.get('adapted_geometry', {})
    for row in placements:
        if row.get('status') != 'placed':
            continue
        name = row['object']
        obj = session.env.scene.object_registry('name', name)
        delta = _array(obj.get_position_orientation()[0]) - row['before_position_m']
        if name in translations:
            translations[name] = (np.asarray(translations[name]) + delta).tolist()
            evidence['relocated'] = True
        if name in geometry:
            pos, quat = (_array(v) for v in obj.get_position_orientation())
            low, high = (_array(v) for v in obj.aabb)
            geometry[name].update(position_m=pos.tolist(), quaternion_xyzw=quat.tolist(),
                                  aabb_min_m=low.tolist(), aabb_max_m=high.tolist())
    for value in evidence.values():
        if isinstance(value, dict):
            refresh_edge_references(value, placements, session)
        elif isinstance(value, list):
            for item in value:
                refresh_edge_references(item, placements, session)


def settle_edge_policy(session, placements):
    """Retain only stable legacy placements; restore failed groups before capture."""
    from omnigibson.object_states import Touching
    placed = [row for row in placements if row.get('status') == 'placed']
    if not placed:
        return
    sim = session.pipeline.og.sim
    for _ in range(90):
        sim.step()
    checked = set()
    restored = False
    for row in placed:
        names = tuple(row.get('related_objects', [row['object']]))
        if names in checked:
            continue
        checked.add(names)
        objects = [session.env.scene.object_registry('name', name) for name in names]
        support = session.env.scene.object_registry('name', row['support'])
        low, high = _PlacementGroup(objects).aabb
        slo, shi = (_array(value) for value in support.aabb)
        stable = (np.all(low[:2] >= slo[:2] - .003) and np.all(high[:2] <= shi[:2] + .003)
            and abs(float(low[2]) - row['support_surface_z_m']) < .015
            and any(obj.states[Touching].get_value(support) for obj in objects)
            and all(np.linalg.norm(_array(obj.get_linear_velocity())) < .02 for obj in objects))
        group_rows = [item for item in placed if item['object'] in names]
        if stable:
            for item in group_rows:
                item['native_support_stable'] = True
            continue
        restored = True
        for item in group_rows:
            obj = session.env.scene.object_registry('name', item['object'])
            obj.set_position_orientation(position=item['before_position_m'],
                                         orientation=item['before_quaternion_xyzw'])
            obj.keep_still()
            item.update(status='unchanged', reason='post_settle_support_check_failed',
                        attempted_position_m=item['position_m'],
                        position_m=item['before_position_m'], native_support_stable=False)
    if restored:
        for _ in range(90):
            sim.step()
