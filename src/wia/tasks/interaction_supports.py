"""Propose user-authorized simple supports after existing furniture is exhausted.

Run through scripts/run_esi_capture_worker.py. Geometry proposals alone do not
certify support, floor contact, robot access, articulation, IK or graspability.
Only the new table primitives are resized; source assets retain their geometry.
"""
from __future__ import annotations

import numpy as np

PREFIX = 'esi_interaction_support_'
NAMES = (PREFIX + 'top', *(PREFIX + f'leg_{index}' for index in range(4)))


def support_names(slot=0):
    prefix = PREFIX if not slot else PREFIX + f'group_{slot}_'
    return (prefix + 'top', *(prefix + f'leg_{index}' for index in range(4)))


def preload_supports(config, slots=1, *, start_slot=0, rgba=None):
    """Preload five fixed colliders away from the scene before construction."""
    objects = config.setdefault('objects', [])
    names = [name for slot in range(start_slot, start_slot + slots) for name in support_names(slot)]
    color = [.46, .32, .20, 1.] if rgba is None else list(rgba)
    if len(color) != 4 or not np.isfinite(color).all() or any(value < 0 or value > 1 for value in color):
        raise ValueError('Support RGBA must contain four finite values in [0, 1].')
    if any(obj.get('name') in names for obj in objects):
        raise ValueError('The reserved support names cannot replace source objects.')
    for index, name in enumerate(names):
        objects.append({'type': 'PrimitiveObject', 'primitive_type': 'Cube',
            'name': name, 'category': 'table' if index % 5 == 0 else 'object',
            'size': 1., 'scale': [.05, .05, .05], 'fixed_base': True,
            'visual_only': False, 'visible': False, 'rgba': color,
            'position': [2000. + index, 2000., 5.], 'orientation': [0., 0., 0., 1.]})


def table_dimensions(bounds, policy):
    """Find a modest table footprint without scaling any original object."""
    if policy.get('_inclined_envelope'):
        from .inclined_apparatus import required_support_dimensions
        needed = required_support_dimensions(policy['_inclined_envelope'])
        size = np.asarray(needed['dimensions_xy_m']) + .06 if needed['available'] else None
        return [np.maximum(size, [.45, .40])] if size is not None and np.all(size <= [2., 1.4]) else []
    if policy['relocation_policy'] in ('shared_tabletop', 'seeded_geometry'):
        from .shared_tabletop import tabletop_dimensions
        return tabletop_dimensions(bounds)
    lows = np.stack([np.asarray(row[0], dtype=float) for row in bounds.values()])
    highs = np.stack([np.asarray(row[1], dtype=float) for row in bounds.values()])
    span = (highs.max(axis=0) - lows.min(axis=0))[:2] + .10
    if policy['relocation_policy'] == 'comparison_pairs':
        pairs = policy['placement_groups']
        sizes = [np.maximum.reduce([bounds[name][1] for name in pair])[:2]
                 - np.minimum.reduce([bounds[name][0] for name in pair])[:2] for pair in pairs]
        options = [np.array([sizes[0][0] + sizes[1][0] + .16, max(s[1] for s in sizes) + .10]),
                   np.array([max(s[0] for s in sizes) + .10, sizes[0][1] + sizes[1][1] + .16])]
    else:
        options = [span]
    result = []
    for option in options:
        size = np.maximum(option, [.45, .40])
        if np.all(np.isfinite(size)) and np.all(size <= [2., 1.4]):
            if not any(np.allclose(size, old) for old in result):
                result.append(size)
    return result


def footprint_clear(scene, centre, dimensions, floor_z, excluded, region):
    """Conservatively reject occupied floor volume and out-of-room footprints."""
    import torch
    from .interaction_setup import array, _objects
    centre, half = array(centre), array(dimensions) / 2
    offsets = np.array([[x, y] for x in (-1., 0., 1.) for y in (-1., 0., 1.)]) * half
    points = centre + offsets
    trav, seg = scene.trav_map, scene.seg_map
    indices = array(trav.world_to_map(torch.tensor(points, dtype=torch.float32))).astype(int)
    free = array(trav.floor_map[0]) == 255
    if np.any(indices < 0) or np.any(indices >= free.shape) or not np.all(free[tuple(indices.T)]):
        return False
    if region and any(seg.get_room_instance_by_point(point) != region for point in points):
        return False
    low, high = centre - half - .02, centre + half + .02
    for obj in _objects(scene):
        if (obj.name in excluded or obj.visual_only
                or obj.category in ('floors', 'floor', 'ceilings', 'ceiling')):
            continue
        a, b = (array(value) for value in obj.aabb)
        if b[2] <= floor_z + .03 or a[2] >= floor_z + .85:
            continue
        if np.all(a[:2] < high) and np.all(b[:2] > low):
            return False
    return True


def activate_support(session, centre, dimensions, surface_z, floor_z, *, slot=0):
    """Place only the five preloaded support primitives during initialization."""
    import torch
    from omnigibson.utils.usd_utils import PoseAPI
    scene = session.env.scene
    centre, dimensions = np.asarray(centre), np.asarray(dimensions)
    thickness, leg_width = .04, .05
    height = float(surface_z - floor_z - thickness)
    if height <= .1:
        raise ValueError('The new support requires positive floor-reaching legs.')
    offsets = [(x, y) for x in (-1, 1) for y in (-1, 1)]
    names = support_names(slot)
    for index, name in enumerate(names):
        obj = scene.object_registry('name', name)
        if obj is None or not obj.fixed_base or obj.visual_only:
            raise ValueError('The initialization-only support was not preloaded.')
        if index == 0:
            scale = [*dimensions, thickness]
            position = [*centre, surface_z - thickness / 2]
        else:
            offset = np.array(offsets[index - 1]) * (dimensions / 2 - leg_width / 2)
            scale = [leg_width, leg_width, height]
            position = [*(centre + offset), floor_z + height / 2]
        obj.scale = torch.tensor(scale, dtype=torch.float32)
        obj.set_position_orientation(position=torch.tensor(position, dtype=torch.float32),
                                     orientation=torch.tensor([0., 0., 0., 1.]))
        obj.visible = True
    PoseAPI.invalidate()
    return scene.object_registry('name', names[0])


def hide_supports(session, *, slot=0):
    for name in support_names(slot):
        obj = session.env.scene.object_registry('name', name)
        if obj is not None:
            obj.visible = False


def floor_support_centres(session, target, region):
    """Propose table centres on raw floor cells, independently of chassis stances.

    A table may fit near a wall where a complete robot cannot stand. Actual
    table footprint, loaded-floor rays and a separate robot stance still decide.
    """
    import torch
    from .interaction_setup import array
    scene = session.env.scene
    cells = np.argwhere(array(scene.trav_map.floor_map[0]) == 255)
    if not len(cells):
        return []
    points = array(scene.trav_map.map_to_world(torch.as_tensor(cells)))
    if region:
        points = np.array([point for point in points
            if scene.seg_map.get_room_instance_by_point(point) == region])
    if not len(points):
        return []
    order = np.argsort(np.linalg.norm(points - array(target)[:2], axis=1), kind='stable')
    centres = []
    for index in order:
        point = points[index]
        if all(np.linalg.norm(point - old) >= .20 for old in centres):
            centres.append(point)
        if len(centres) >= 48:
            break
    return centres


def floor_feet_supported(session, xy, size, floor_z):
    """Check actual loaded floor shapes under each proposed table foot centre."""
    from omnigibson.utils.sampling_utils import raytest
    from .interaction_setup import array, _objects
    from .support_sampling import same_body
    floors = [obj for obj in _objects(session.env.scene) if obj.category in ('floor', 'floors') and not obj.visual_only]
    rows = []
    for x in (-1, 1):
        for y in (-1, 1):
            point = array(xy) + np.array([x, y]) * (array(size) / 2 - .025)
            hit = raytest([*point, floor_z + .10], [*point, floor_z - .10])
            accepted = bool(hit['hit'] and any(same_body(obj.prim_path, hit['rigidBody']) for obj in floors)
                and float(array(hit['normal'])[2]) >= .95 and abs(float(array(hit['position'])[2]) - floor_z) <= .006)
            rows.append({'xy_m': point.tolist(), 'accepted': accepted,
                'floor_body': str(hit.get('rigidBody')) if hit['hit'] else None})
    return {'accepted': all(row['accepted'] for row in rows), 'native_floor_ray_samples': rows,
        'evidence_kind': 'native_geometric_floor_foot_support', 'static_contact_stream_measured': False}


def support_candidates(session, group, bounds, policy, world, floor_z, target, evidence):
    """Yield table proposals only after all existing furniture proposals fail."""
    from .interaction_setup import array, group_translation
    from .comparison_support_layout import comparison_support_layouts
    names = {obj.name for obj in group.values()}
    pairs_policy = dict(policy)
    if policy['relocation_policy'] == 'comparison_pairs':
        pairs_policy['placement_groups'] = [[group[key].name for key in keys]
                                           for keys in policy['placement_groups']]
    dimensions = table_dimensions(bounds, pairs_policy)
    evidence['fallback_support_search'] = {'authorized': True, 'candidate_dimensions_m': [d.tolist() for d in dimensions],
        'source_assets_rescaled': False, 'rule': 'Existing same-room supports were attempted first.', 'proposals': []}
    vertical = policy['relocation_policy'] == 'vertical_only'
    low = np.minimum.reduce([row[0] for row in bounds.values()])
    high = np.maximum.reduce([row[1] for row in bounds.values()])
    centre = (low[:2] + high[:2]) / 2
    if vertical:
        centres = [centre]
    else:
        centres = floor_support_centres(session, target, policy.get('source_region'))
        evidence['fallback_support_search']['centre_proposal_source'] = 'raw_floor_cells_before_chassis_erosion'
        evidence['fallback_support_search']['centre_count'] = len(centres)
    for size in dimensions:
        for xy in centres:
            if not footprint_clear(session.env.scene, xy, size, floor_z,
                                   names | set(policy.get('_pending_group_names', ()))
                                   | {session.env.robots[0].name}, policy.get('source_region')):
                continue
            feet = floor_feet_supported(session, xy, size, floor_z)
            if not feet['accepted']:
                continue
            required_z = policy.get('_required_translation_z_m')
            surfaces = (float(low[2] + required_z - .005),) if required_z is not None else (floor_z + .50, floor_z + .65)
            for surface_z in surfaces:
                if surface_z - floor_z < .30:
                    continue
                support_bounds = (np.r_[xy - size / 2, surface_z - .04], np.r_[xy + size / 2, surface_z])
                if policy.get('_inclined_envelope'):
                    from .interaction_setup import inclined_support_layouts
                    layouts = inclined_support_layouts(bounds, support_bounds, surface_z,
                        policy['_inclined_envelope'], vertical_only=vertical)
                elif policy['relocation_policy'] in ('shared_tabletop', 'seeded_geometry'):
                    from .shared_tabletop import tabletop_layouts
                    layouts = tabletop_layouts(group, bounds, support_bounds, surface_z, policy)
                elif policy['relocation_policy'] == 'comparison_pairs':
                    layouts = comparison_support_layouts(bounds, pairs_policy['placement_groups'], support_bounds, surface_z)
                else:
                    try:
                        delta = group_translation(bounds, support_bounds, centre if vertical else xy, surface_z)
                        if vertical and not np.allclose(delta[:2], 0., atol=1e-7):
                            continue
                        layouts = [{'translations': {name: delta.copy() for name in bounds},
                                    'bounds': {name: (a + delta, b + delta) for name, (a, b) in bounds.items()}}]
                    except ValueError:
                        continue
                for layout in layouts:
                    table = {'centre_xy_m': array(xy).tolist(), 'dimensions_xy_m': size.tolist(),
                             'surface_z_m': float(surface_z), 'floor_z_m': floor_z,
                             'floor_foot_support': feet}
                    evidence['fallback_support_search']['proposals'].append(table)
                    layout['new_support'] = table
                    yield None, surface_z, layout['translations'], layout
