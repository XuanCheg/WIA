"""Initialize only a reserved blue delivery table on measured clear native floor.

Run: python -B -m pytest -q tests/test_task_edit2_missing_definitions.py
Native use is through task_edit2_destinations category initialization hooks.
No source actor, furniture, robot, asset scale or source boundary is changed.
"""
from __future__ import annotations

import math
import numpy as np

from .interaction_supports import support_names

SLOT = 99
BLUE_RGBA = [.05, .20, .95, 1.]
PUBLIC_LABEL = 'the blue delivery table in this room'


def _box(obj):
    from .interaction_setup import array
    low, high = (array(value) for value in obj.aabb)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(high <= low):
        raise ValueError('native_aabb_unavailable')
    return low, high


def native_floor_geometry(session, centre, dimensions, expected_floor_z):
    """Measure floor bodies and nine upward-facing native ray intersections."""
    from omnigibson.utils.sampling_utils import raytest
    from .interaction_setup import array, _objects, _body_under
    floors = [obj for obj in _objects(session.env.scene) if obj.category in ('floor', 'floors')
              and not obj.visual_only]
    inventory = [{'name': obj.name, 'prim_path': obj.prim_path,
                  'aabb_m': [value.tolist() for value in _box(obj)]} for obj in floors]
    points, heights = [], []
    half = np.asarray(dimensions) / 2
    for x in (-1., 0., 1.):
        for y in (-1., 0., 1.):
            point = np.asarray(centre) + np.array([x, y]) * half
            hits = raytest([*point, expected_floor_z + .08], [*point, expected_floor_z - .08], only_closest=False)
            hits = [hit for hit in hits if hit.get('hit') and any(
                _body_under(hit['rigidBody'], obj.prim_path) for obj in floors)
                and float(array(hit['normal'])[2]) >= .95
                and abs(float(array(hit['position'])[2]) - expected_floor_z) <= .03]
            if not hits:
                raise ValueError('native_floor_coverage_ray_missing')
            hit = min(hits, key=lambda row: abs(float(array(row['position'])[2]) - expected_floor_z))
            heights.append(float(array(hit['position'])[2]))
            points.append({'xy_m': point.tolist(), 'body': str(hit['rigidBody']),
                           'position_m': array(hit['position']).tolist(), 'normal': array(hit['normal']).tolist()})
    if max(heights) - min(heights) > .01:
        raise ValueError('native_delivery_floor_not_level')
    return float(np.mean(heights)), {'floor_bodies': inventory, 'floor_rays': points}


def delivery_dimensions(targets, margin=.04):
    """Leave room for all unchanged targets simultaneously in a simple grid."""
    sizes = [high - low for low, high in map(_box, targets)]
    if not sizes:
        raise ValueError('delivery_targets_missing')
    columns = math.ceil(math.sqrt(len(sizes)))
    rows = math.ceil(len(sizes) / columns)
    maximum = np.max(sizes, axis=0)
    dimensions = np.maximum([columns * maximum[0] + (columns + 1) * margin,
                             rows * maximum[1] + (rows + 1) * margin], [.45, .40])
    if np.any(dimensions > [2., 1.4]):
        raise ValueError('unchanged_targets_exceed_delivery_table_footprint')
    return dimensions, float(maximum[2])


def initialize_delivery_support(session, contract):
    """Propose/activate fixed slot 99; fail closed before any public controls."""
    from .interaction_setup import _navigation_cells, _objects
    from .interaction_supports import footprint_clear, activate_support, hide_supports, floor_support_centres
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Delivery table placement is initialization-only.')
    evidence = {'accepted': False, 'stage': 'native_target_geometry', 'slot': SLOT,
        'source_identity': contract['source_identity'], 'selection_seed': contract['selection_seed'],
        'source_actors_moved': False, 'source_assets_rescaled': False, 'public_control_access_verified': False,
        'public_label': PUBLIC_LABEL, 'proposals': []}
    session.delivery_support_initialization = evidence
    scene = session.env.scene
    try:
        targets = [scene.object_registry('name', row['name']) for row in contract['target_bindings'].values()]
        if any(obj is None for obj in targets):
            raise ValueError('native_delivery_target_missing')
        dimensions, height = delivery_dimensions(targets, contract['fit_margin_m'])
        evidence['target_geometry'] = {obj.name: [value.tolist() for value in _box(obj)] for obj in targets}
        evidence['loaded_floor_geometry'] = [{'name': obj.name, 'prim_path': getattr(obj, 'prim_path', None),
            'aabb_m': [value.tolist() for value in _box(obj)]} for obj in _objects(scene)
            if obj.category in ('floor', 'floors') and not obj.visual_only]
        evidence['preloaded_support_geometry'] = {name: [value.tolist() for value in _box(
            scene.object_registry('name', name))] for name in support_names(SLOT)}
        evidence['stage'] = 'raw_floor_support_proposals'
        excluded = set(support_names(SLOT))
        target_xy = np.mean([(a + b)[:2] / 2 for a, b in map(_box, targets)], axis=0)
        floor_z = float(scene.trav_map.floor_heights[0])
        if not np.isfinite(floor_z):
            raise ValueError('native_floor_height_unavailable')
        centres = floor_support_centres(session, target_xy, contract['room'])
        world = np.asarray(centres, dtype=float).reshape(-1, 2)
        evidence.update(centre_proposal_source='raw_floor_cells_before_chassis_erosion',
            source_target_xy_m=target_xy.tolist(), raw_floor_proposal_centres_xy_m=world.tolist(),
            proposal_floor_index=0, proposal_floor_z_m=floor_z)
        # Table proposals and complete chassis stances are different domains.
        order = np.lexsort((world[:, 1], world[:, 0], np.linalg.norm(world - target_xy, axis=1)))
        selected = []
        for index in order:
            xy = np.asarray(world[index], dtype=float)
            if any(np.linalg.norm(xy - old) < .20 for old in selected):
                continue
            selected.append(xy)
            if len(selected) > 128:
                break
            proposal = {'centre_xy_m': xy.tolist(), 'dimensions_xy_m': dimensions.tolist(), 'accepted': False}
            evidence['proposals'].append(proposal)
            evidence['stage'] = 'clear_delivery_footprint'
            if not footprint_clear(scene, xy, dimensions, floor_z, excluded, contract['room']):
                proposal['failure'] = 'native_occupied_or_wrong_room_footprint'
                continue
            # The fixed table and future unchanged target volume must not collide
            # with any source body. No task actor is removed to clear this space.
            low, high = np.r_[xy - dimensions / 2, floor_z + .61], np.r_[xy + dimensions / 2, floor_z + .65 + height + .04]
            blocked = []
            for obj in _objects(scene):
                if obj.name in excluded or obj.visual_only or obj.category in ('floor', 'floors', 'ceiling', 'ceilings'):
                    continue
                a, b = _box(obj)
                if np.all(a < high) and np.all(b > low):
                    blocked.append(obj.name)
            if blocked:
                proposal.update(failure='native_delivery_envelope_collision', blocking_bodies=blocked)
                continue
            evidence['stage'] = 'native_floor_coverage'
            try:
                measured_z, floor = native_floor_geometry(session, xy, dimensions, floor_z)
            except ValueError as error:
                proposal['failure'] = str(error)
                continue
            evidence['stage'] = 'activate_reserved_support'
            top = activate_support(session, xy, dimensions, measured_z + .65, measured_z, slot=SLOT)
            geometry = {name: [value.tolist() for value in _box(scene.object_registry('name', name))]
                        for name in support_names(SLOT)}
            legs_reach = all(abs(geometry[name][0][2] - measured_z) <= .01
                             for name in support_names(SLOT)[1:])
            top_low, top_high = _box(top)
            fits = np.all((top_high - top_low)[:2] >= dimensions - .01)
            if not legs_reach or not fits:
                hide_supports(session, slot=SLOT)
                raise ValueError('activated_delivery_support_geometry_mismatch')
            evidence['stage'] = 'post_activation_robot_stance'
            try:
                # No physical table part is excluded: its real colliders must
                # participate in this independent complete-chassis screen.
                stances, stance_floor_z, grid = _navigation_cells(session, session.robot, contract['room'], set())
                stances = np.asarray(stances, dtype=float)
                if stances.ndim != 2 or stances.shape[1] != 2 or not len(stances) or not np.isfinite(stances).all():
                    raise ValueError('No complete native robot stance remains after table activation.')
            except ValueError as error:
                hide_supports(session, slot=SLOT)
                proposal.update(failure='native_delivery_robot_stance_unavailable',
                    stance_rejection=f'{type(error).__name__}: {error}',
                    robot_stance_screen={'accepted': False, 'excluded_object_names': [],
                        'table_parts_participate_in_collision_screen': True})
                continue
            stance_index = int(np.argmin(np.linalg.norm(stances - xy, axis=1)))
            proposal['robot_stance_screen'] = {'accepted': True, 'excluded_object_names': [],
                'table_parts_participate_in_collision_screen': True, 'native_navigation_screen': grid,
                'candidate_cell_count': len(stances), 'nearest_stance_xy_m': stances[stance_index].tolist(),
                'floor_z_m': float(stance_floor_z), 'route_and_arm_clearance_verified': False}
            proposal.update(accepted=True, surface_z_m=float(top_high[2]), floor_z_m=measured_z)
            evidence.update(accepted=True, stage='geometry_initialized_access_pending', floor_geometry=floor,
                            support_geometry=geometry, selected=proposal, legs_reach_measured_floor=legs_reach,
                            target_fit_geometry_verified=bool(fits), navigation_screen=grid,
                            independent_complete_robot_stance_available=True)
            return evidence
        raise ValueError('no_safe_loaded_room_delivery_table_site')
    except Exception as error:
        evidence['failure'] = f'{type(error).__name__}: {error}'
        evidence['failed_stage'] = evidence['stage']
        raise ValueError('Blue delivery table initialization failed: ' + evidence['failure']) from error
