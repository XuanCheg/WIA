"""Restore declared background loading and report native measurements.

Run through category_initialization patches in src/wia/runtime/commands/check_esi_core_native.py.
CPU checks: python -B -m pytest -q -p no:cacheprovider tests/test_wia_background.py
configure_scene either clears room filters or opts into complete ground loading;
initialize_scene optionally changes the existing Kit viewport grid setting.
Diagnostics never step or modify physics.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

REVISION = 'wia_background_v1'
GRID_SETTING = '/app/viewport/grid/enabled'
# Inspected in the installed OmniGibson utils/constants.py. Singular aliases
# also permit explicit floor / ground primitives authored by source tasks.
GROUND_CATEGORIES = frozenset({'floors', 'lawn', 'driveway', 'carpet',
                               'floor', 'ground', 'ground_plane'})
GRID_SOURCE = ('Installed Isaac Sim omni.physx.demos/omni/physxdemos/utils/'
               'room_helper.py uses carb.settings.get_settings().set_bool('
               '"/app/viewport/grid/enabled", False).')


def _contract(parameters):
    if (parameters.get('revision') != REVISION or parameters.get('mode') not in ('full_scene', 'complete_ground')
            or not isinstance(parameters.get('expected_scene'), str)
            or not parameters['expected_scene'].strip()
            or type(parameters.get('disable_grid')) is not bool):
        raise ValueError('Background initialization requires its revision, full_scene or complete_ground mode, '
                         'nonempty expected_scene and boolean disable_grid.')


def configure_scene(config, parameters):
    """Apply only the declared loading policy to the existing native background."""
    _contract(parameters)
    scene = config.get('scene', {})
    if scene.get('scene_model') != parameters['expected_scene']:
        raise ValueError('Background repair cannot replace the existing native scene.')
    keys = ('load_room_instances', 'load_room_types', 'load_object_categories',
            'not_load_object_categories')
    previous = {key: {'present': key in scene, 'value': deepcopy(scene.get(key))}
                for key in keys}
    if parameters['mode'] == 'complete_ground':
        if scene.get('type') not in ('InteractiveTraversableScene', 'WiaCompleteGroundScene'):
            raise ValueError('Complete ground loading requires the native InteractiveTraversableScene or its owned subclass.')
        # Native-only import registers the subclass through OmniGibson's registry.
        from wia.authoring.converters.wia_ground_scene import WiaCompleteGroundScene
        previous['type'] = {'present': 'type' in scene, 'value': scene.get('type')}
        scene['type'] = WiaCompleteGroundScene.__name__
        return {'revision': REVISION, 'mode': 'complete_ground',
                'expected_scene': parameters['expected_scene'], 'previous_selection': previous,
                'changed_fields': ['type'], 'physical_success_verified': False}
    scene.update(load_room_instances=None, load_room_types=None)
    return {'revision': REVISION, 'mode': 'full_scene',
            'expected_scene': parameters['expected_scene'], 'previous_selection': previous,
            'changed_fields': ['load_room_instances', 'load_room_types'],
            'physical_success_verified': False}


def _json(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json(item) for item in sorted(value)]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError('Measurement is not a finite JSON value.')


def _measure(callback):
    try:
        return {'available': True, 'value': _json(callback())}
    except Exception as error:
        return {'available': False, 'value': None,
                'error': f'{type(error).__name__}: {error}'}


def _values(value):
    return list(value.values()) if isinstance(value, dict) else list(value)


def _aabb(obj):
    low, high = obj.aabb
    bounds = np.asarray(_json([low, high]), dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0]):
        raise ValueError('AABB requires finite ordered world-space bounds of shape (2, 3).')
    return bounds.tolist()


def _pose(obj):
    position, quaternion = obj.get_position_orientation()
    position, quaternion = np.asarray(_json(position)), np.asarray(_json(quaternion))
    if (position.shape != (3,) or quaternion.shape != (4,)
            or not np.isfinite(position).all() or not np.isfinite(quaternion).all()):
        raise ValueError('Pose requires finite world-space position and xyzw quaternion.')
    return {'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist()}


def _collision_evidence(obj):
    rows = []
    for link in _values(obj.links):
        row = {'prim_path': _measure(lambda: link.prim_path),
               'visual_only': _measure(lambda: link.visual_only),
               'contact_reporting_enabled': _measure(lambda: link.contact_reporting_enabled),
               'meshes': []}
        meshes = _measure(lambda: len(link.collision_meshes))
        row['collision_mesh_count'] = meshes
        if meshes['available']:
            for mesh in _values(link.collision_meshes):
                row['meshes'].append({'prim_path': _measure(lambda: mesh.prim_path),
                                     'collision_enabled': _measure(lambda: mesh.collision_enabled)})
        rows.append(row)
    return rows


def _object_record(obj):
    return {'name': _measure(lambda: obj.name),
            'category': _measure(lambda: obj.category),
            'in_rooms': _measure(lambda: obj.in_rooms),
            'prim_path': _measure(lambda: obj.prim_path),
            'visual_only': _measure(lambda: obj.visual_only),
            'aabb_m': _measure(lambda: _aabb(obj)),
            'collision_evidence': _measure(lambda: _collision_evidence(obj))}


def _contacts(robot):
    return [{'body0': str(contact.body0), 'body1': str(contact.body1)}
            for contact in robot.contact_list()]


def _global_ground(session):
    # Installed simulator.py exposes the XFormPrim as the floor_plane property.
    plane = session.pipeline.og.sim.floor_plane
    if plane is None:
        return {'present': False}
    return {'present': True, 'object': _object_record(plane),
            'visible': _measure(lambda: plane.visible),
            'pose': _measure(lambda: _pose(plane)),
            'source': 'Installed Simulator.floor_plane getter; simulator-global plane is outside scene.objects.'}


def _footprints(robot_row, floors):
    """Report geometry and current contacts without certifying physical support."""
    result = []
    for floor in floors:
        row = {'floor_name': floor['name'], 'floor_prim_path': floor['prim_path'],
               'contains_robot_origin_xy': None, 'robot_aabb_bottom_gap_m': None,
               'current_contact_pairs': None, 'physical_support_verified': False}
        if robot_row['pose']['available'] and floor['aabb_m']['available']:
            position = robot_row['pose']['value']['position_m']
            low, high = floor['aabb_m']['value']
            row['contains_robot_origin_xy'] = bool(all(low[i] <= position[i] <= high[i] for i in (0, 1)))
        if robot_row['aabb_m']['available'] and floor['aabb_m']['available']:
            row['robot_aabb_bottom_gap_m'] = float(
                robot_row['aabb_m']['value'][0][2] - floor['aabb_m']['value'][1][2])
        if robot_row['contacts']['available'] and floor['prim_path']['available']:
            path = floor['prim_path']['value']
            row['current_contact_pairs'] = [pair for pair in robot_row['contacts']['value']
                if any(body == path or body.startswith(path + '/') for body in pair.values())]
        row['missing_measurements'] = [key for key in (
            'contains_robot_origin_xy', 'robot_aabb_bottom_gap_m', 'current_contact_pairs') if row[key] is None]
        result.append(row)
    return result


def diagnose_scene(session):
    """Read loaded ground inventory, robot/base poses and descriptive support data."""
    result = {'revision': REVISION, 'physical_success_verified': False,
              'measurement_scope': 'Current native getters only; no simulation step or physical qualification.',
              'scene_model': _measure(lambda: session.env.scene.scene_model),
              'ground_category_selection': sorted(GROUND_CATEGORIES), 'floors': [], 'robots': [],
              'simulator_global_ground': _measure(lambda: _global_ground(session))}
    try:
        objects = _values(session.env.scene.objects)
        result['scene_object_count'] = len(objects)
        result['category_read_errors'] = []
        for obj in objects:
            category = _measure(lambda: obj.category)
            if not category['available']:
                result['category_read_errors'].append({'name': _measure(lambda: obj.name), 'category': category})
            elif category['value'] in GROUND_CATEGORIES:
                result['floors'].append(_object_record(obj))
        result['inventory_available'] = True
    except Exception as error:
        result.update(inventory_available=False, inventory_error=f'{type(error).__name__}: {error}')
    try:
        robots = _values(session.env.robots)
        for robot in robots:
            row = {'name': _measure(lambda: robot.name), 'prim_path': _measure(lambda: robot.prim_path),
                   'pose': _measure(lambda: _pose(robot)), 'aabb_m': _measure(lambda: _aabb(robot)),
                   'base_pose': _measure(lambda: _pose(robot.root_link)),
                   'contacts': _measure(lambda: _contacts(robot)),
                   'collision_evidence': _measure(lambda: _collision_evidence(robot))}
            row['ground_support_observations'] = _footprints(row, result['floors'])
            row['contact_limitation'] = 'An empty contact list does not prove absence of contact; native reporting may be disabled.'
            result['robots'].append(row)
        result['robot_inventory_available'] = True
    except Exception as error:
        result.update(robot_inventory_available=False, robot_inventory_error=f'{type(error).__name__}: {error}')
    return result


def _disable_grid(requested):
    receipt = {'requested': requested, 'setting': GRID_SETTING,
               'source': GRID_SOURCE, 'available': False, 'changed': False,
               'disabled_verified': False,
               'scope': 'Kit viewport guide only; grid-textured ground materials are not changed.'}
    if not requested:
        receipt['reason'] = 'Grid change was not requested.'
        return receipt
    try:
        import carb.settings
        settings = carb.settings.get_settings()
        previous = settings.get(GRID_SETTING)
        if type(previous) is not bool:
            receipt['reason'] = 'Existing native viewport grid boolean setting is unavailable.'
            return receipt
        receipt.update(available=True, previous=previous)
        settings.set_bool(GRID_SETTING, False)
        receipt['readback'] = _json(settings.get(GRID_SETTING))
        receipt['changed'] = type(receipt['readback']) is bool and receipt['readback'] != previous
        receipt['disabled_verified'] = receipt['readback'] is False
    except Exception as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
    return receipt


def initialize_scene(session, parameters):
    """Run guarded presentation initialization and return independent diagnostics."""
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Background presentation changes are initialization-only.')
    _contract(parameters)
    if session.env.scene.scene_model != parameters['expected_scene']:
        raise ValueError('Background diagnostics require the declared actual native scene.')
    if parameters['mode'] == 'complete_ground':
        from wia.authoring.converters.wia_ground_scene import WiaCompleteGroundScene
        if not isinstance(session.env.scene, WiaCompleteGroundScene):
            raise ValueError('Complete ground diagnostics require the actual registered WiaCompleteGroundScene.')
        diagnostics = diagnose_scene(session)
        diagnostics['scene_class'] = _measure(lambda: type(session.env.scene).__name__)
        return {'revision': REVISION, 'mode': parameters['mode'],
                'physical_success_verified': False,
                'grid': _disable_grid(parameters['disable_grid']), 'diagnostics': diagnostics}
    return {'revision': REVISION, 'mode': parameters['mode'],
            'physical_success_verified': False,
            'grid': _disable_grid(parameters['disable_grid']),
            'diagnostics': diagnose_scene(session)}
