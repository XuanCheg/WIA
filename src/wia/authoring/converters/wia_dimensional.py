"""Read native geometry for the five WIA dimensional review candidates.

Run generated definitions with scripts/check_esi_core_native.py --evaluate;
CPU protocol checks: python -B -m pytest -q -p no:cacheprovider
tests/test_wia_dimensional_adapter.py. Initialization remains in active_layout.
This adapter never constructs scenes, changes physics, steps time or supplies
private coordinates to the public observation/prompt channel.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from wia.authoring.converters.rigid_partial_insertion import (
    _aggregate_grasp, _contact_bodies, _native_shape_stamp, _under,
)
from wia.authoring.io import write_json
from wia.tasks.completion import progress
from wia.tasks.wia_dimensional_scoring import APERTURE_SCORING_VERSION
from wia.tasks.interaction_setup import array

MODULE = 'core.esi_converters.wia_dimensional'
REVISION = 'wia_dimensional_v1'
REVISIONS = (REVISION, 'wia_dimensional_v2', 'wia_dimensional_v3')
OPERATIONS = ('select_pass_aperture', 'sort_intrinsic_dimension')
SCENE = 'hall_glass_ceiling'
ROOM = 'empty_room_0'
INITIAL_DIMENSION_TOLERANCE_M = .002
ERRORS = (AttributeError, ValueError, RuntimeError, TypeError, KeyError, IndexError, AssertionError, ImportError)


def _boolean(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('A measured native boolean is required.')
    return bool(value)


def _vector(value, size, *, positive=False):
    value = array(value)
    if value.shape != (size,) or (positive and np.any(value <= 0)):
        raise ValueError('Native vector shape or positive dimensions are unavailable.')
    return value


def _all(values):
    values = list(values)
    return False if any(value is False for value in values) else None if any(value is None for value in values) else True


def _scene_contract(parameters):
    """Read the explicit scene identity without supplying a default location."""
    contract = parameters.get('scene_contract')
    if (not isinstance(contract, dict)
            or any(not isinstance(contract.get(key), str) or not contract[key].strip()
                   for key in ('scene_model', 'room'))):
        raise ValueError('An explicit scene_contract with nonempty scene_model and room strings is required.')
    return contract['scene_model'], contract['room']


def _validate_scene_identity(session, scene_model, room):
    if (session.scene_name != scene_model or session.room_name != room
            or session.env.scene.scene_model != scene_model):
        raise ValueError('The declared scene contract and current native scene/room must match exactly.')


def _config(session, config):
    p = config['parameters']
    if (module_identity(config.get('module')) != MODULE or p.get('revision') not in REVISIONS
            or config.get('operation') not in OPERATIONS
            or p.get('operation', config['operation']) != config['operation']):
        raise ValueError('Use the explicit WIA dimensional revision and operation.')
    if p['revision'] == 'wia_dimensional_v3':
        scene_model, room = _scene_contract(p)
        if config.get('scene') != scene_model or config.get('room') != room:
            raise ValueError('The dimensional task must match its explicit scene contract.')
        _validate_scene_identity(session, scene_model, room)
    elif (config.get('scene') != SCENE or config.get('room') != ROOM
            or session.scene_name != config['scene'] or session.room_name != config['room']
            or session.env.scene.scene_model != config['scene']):
        raise ValueError('The dimensional task, current native scene and room must match.')
    if (not isinstance(p['candidates'], dict) or len(p['candidates']) < 2
            or any(config['bindings'].get(key) != name for key, name in p['candidates'].items())
            or len(set(p['candidates'].values())) != len(p['candidates'])):
        raise ValueError('Candidates require distinct exact native binding identities.')
    if (not p['fixture_keys'] or len(set(p['fixture_keys'])) != len(p['fixture_keys'])
            or not p['destinations'] or len(set(p['destinations'])) != len(p['destinations'])
            or not set(p['destinations']).issubset(p['fixture_keys'])
            or not set(p['fixture_keys']).issubset(config['bindings'])):
        raise ValueError('Every destination and fixed fixture needs an explicit binding.')
    if set(p['candidate_dimensions_m']) != set(p['candidates']):
        raise ValueError('All candidate dimensions must be explicitly proposed.')
    for value in p['candidate_dimensions_m'].values():
        _vector(value, 3, positive=True)
    for key in ('minimum_lift_m', 'stability_duration_s', 'maximum_linear_speed_m_s',
                'maximum_angular_speed_rad_s', 'max_sample_gap_s', 'fixture_position_tolerance_m',
                'fixture_angle_tolerance_rad', 'dimension_relative_tolerance',
                'aperture_clearance_margin_m', 'minimum_assignment_gap_m'):
        value = p['scoring'][key]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError('All dimensional scoring thresholds must be explicit finite positive values.')
    if config['operation'] == 'sort_intrinsic_dimension':
        value = p['unselected_position_tolerance_m']
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError('The unselected position tolerance must be explicit and positive.')
    if config['operation'] == 'select_pass_aperture':
        if (len(p['destinations']) != 1 or set(p['aperture_bindings']) != {'lower_side', 'upper_side', 'roof', 'floor'}
                or not set(p['aperture_bindings'].values()).issubset(p['fixture_keys'])):
            raise ValueError('Aperture operations require four bound frame parts and one destination.')
    elif len(p['destinations']) != len(p['candidates']):
        raise ValueError('Sorting requires one ordered destination for each candidate.')
    return p


def _geometry(obj):
    """Keep native world vertices and measure body shape in local USD space.

    USD matrices use row-vector convention. The relative transform excludes the
    root link's transform/scale; apply its measured scale exactly once, in
    float64, without reconstructing body points from large world coordinates.
    """
    import omnigibson.lazy as lazy

    if getattr(obj, '_primitive_type', None) != 'Cube' or _boolean(obj.articulated):
        raise ValueError('Local intrinsic geometry supports only nonarticulated native Cube primitives.')
    root = obj.root_link
    links = [link for link in obj.links.values() if not _boolean(link.is_meta_link)
             and any(_boolean(mesh.collision_enabled) for mesh in link.collision_meshes.values())]
    if len(links) != 1 or links[0] is not root:
        raise ValueError('Local intrinsic geometry requires one enabled collision link equal to root_link.')
    root_path = str(root.prim.GetPath())
    if not root_path.startswith('/') or root_path != str(root.prim_path):
        raise ValueError('The native root-link USD identity is unavailable.')
    root_scale = _vector(root.scale, 3, positive=True)
    cache = lazy.pxr.UsdGeom.XformCache()
    position, quaternion = obj.get_position_orientation()
    position, quaternion = _vector(position, 3), _vector(quaternion, 4)
    if np.linalg.norm(quaternion) == 0:
        raise ValueError('Native quaternion is zero.')
    scale = _vector(obj.scale, 3, positive=True)
    points, body_points, meshes = [], [], []
    for mesh in root.collision_meshes.values():
        if not _boolean(mesh.collision_enabled):
            continue
        local = mesh.points
        raw = array(local)
        if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw):
            raise ValueError('Actual native collision vertices are unavailable.')
        shape_stamp = _native_shape_stamp(mesh)
        if shape_stamp['type'] != 'Mesh':
            raise ValueError('The native Cube collision geometry must expose uncached USD Mesh points and topology.')
        current_points = array(mesh.prim.GetAttribute('points').Get())
        if (current_points.shape != raw.shape or
                not np.array_equal(current_points.astype(np.float32), raw.astype(np.float32))):
            raise ValueError('Cached wrapper collision points do not match current USD Mesh points at float32 storage accuracy.')
        mesh_path = str(mesh.prim.GetPath())
        if mesh_path != str(mesh.prim_path) or not mesh_path.startswith(root_path + '/'):
            raise ValueError('Collision mesh must be a measured descendant of the root link.')
        matrix, reset = cache.ComputeRelativeTransform(mesh.prim, root.prim)
        if _boolean(reset):
            raise ValueError('resetXformStack breaks the collision-mesh-to-root-link hierarchy.')
        matrix = array(matrix)
        if (matrix.shape != (4, 4)
                or not np.allclose(matrix[:, 3], [0., 0., 0., 1.], rtol=0., atol=1e-12)
                or np.linalg.det(matrix[:3, :3]) == 0):
            raise ValueError('Native mesh-to-root transform must be a finite nonsingular affine 4 x 4 matrix.')
        homogeneous = np.concatenate((raw, np.ones((len(raw), 1))), axis=1) @ matrix
        if not np.allclose(homogeneous[:, 3], 1., rtol=0., atol=1e-12):
            raise ValueError('Native mesh-to-root transform did not preserve homogeneous coordinates.')
        body = homogeneous[:, :3] * root_scale
        if not np.isfinite(body).all():
            raise ValueError('Native local collision vertices are nonfinite.')
        faces = mesh.faces
        if hasattr(faces, 'detach'):
            faces = faces.detach().cpu().numpy()
        faces = np.asarray(faces)
        if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in 'iu'
                or np.any(faces < 0) or np.any(faces >= len(raw))):
            raise ValueError('Native collision triangle topology is unavailable or invalid.')
        world = array(mesh.transform_local_points_to_world(local))
        if world.shape != raw.shape:
            raise ValueError('Native collision vertex transformation changed its array shape.')
        points.append(world)
        body_points.append(body)
        meshes.append({'prim_path': str(mesh.prim_path), 'local_vertices_m': raw.tolist(),
                       'faces': faces.tolist(), 'mesh_to_root_link_transform': matrix.tolist(),
                       'reset_xform_stack': False, 'native_shape_stamp': shape_stamp})
    if not points:
        raise ValueError('No enabled non-meta native collision geometry is measurable.')
    vertices = np.concatenate(points)
    body = np.concatenate(body_points)
    dimensions = np.ptp(body, axis=0)
    if np.any(dimensions <= 0):
        raise ValueError('Native collision geometry must have three positive intrinsic dimensions.')
    low, high = (_vector(value, 3) for value in obj.aabb)
    if np.any(high <= low):
        raise ValueError('Native collision AABB is unavailable.')
    return {'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
            'scale': scale.tolist(), 'vertices_m': vertices.tolist(),
            'body_vertices_m': body.tolist(), 'dimensions_m': dimensions.tolist(),
            'root_link_prim_path': root_path, 'root_link_scale': root_scale.tolist(),
            'intrinsic_geometry_source': 'Float64 mesh-to-root-link local USD transform followed by measured root-link scale once.',
            'aabb_m': [low.tolist(), high.tolist()], 'collision_meshes': meshes}


def _collidable(obj, *, dynamic):
    if _boolean(obj.visual_only) or _boolean(obj.fixed_base) == dynamic:
        return False
    found = False
    for link in obj.links.values():
        if _boolean(link.is_meta_link):
            continue
        if not any(_boolean(mesh.collision_enabled) for mesh in link.collision_meshes.values()):
            continue
        found = True
        if dynamic and (not _boolean(link.get_attribute('physics:rigidBodyEnabled'))
                        or _boolean(link.get_attribute('physics:kinematicEnabled'))):
            return False
    return found


def _relative_geometry(current, initial, tolerance):
    scale, original_scale = array(current['scale']), array(initial['scale'])
    if np.any(np.abs(scale / original_scale - 1.) > tolerance):
        return False
    if (current['root_link_prim_path'] != initial['root_link_prim_path'] or
            np.any(np.abs(array(current['root_link_scale']) / initial['root_link_scale'] - 1.) > tolerance)):
        return False
    if np.any(np.abs(array(current['dimensions_m']) / initial['dimensions_m'] - 1.) > tolerance):
        return False
    first, second = array(current['body_vertices_m']), array(initial['body_vertices_m'])
    if first.shape != second.shape:
        return False
    # Normalize vertex differences by the original body's longest extent;
    # this is rotation independent and also detects shape changes at zero axes.
    extent = max(initial['dimensions_m'])
    return bool(np.max(np.abs(first - second)) <= tolerance * extent
                and [(row['prim_path'], row['faces'], row['native_shape_stamp']) for row in current['collision_meshes']]
                == [(row['prim_path'], row['faces'], row['native_shape_stamp']) for row in initial['collision_meshes']])


def _fixture_valid(current, initial, scoring):
    position = np.linalg.norm(array(current['position_m']) - initial['position_m'])
    angle = (Rotation.from_quat(current['quaternion_xyzw']).inv()
             * Rotation.from_quat(initial['quaternion_xyzw'])).magnitude()
    return bool(position <= scoring['fixture_position_tolerance_m']
                and angle <= scoring['fixture_angle_tolerance_rad']
                and _relative_geometry(current, initial, scoring['dimension_relative_tolerance']))


def _axis_alignment(quaternion, tolerance):
    matrix = np.abs(Rotation.from_quat(quaternion).as_matrix())
    axes = matrix.argmax(axis=1)
    angles = np.arccos(np.clip(matrix[np.arange(3), axes], 0., 1.))
    return {'axis_angles_rad': angles.tolist(), 'tolerance_rad': tolerance,
            'accepted': bool(len(set(axes)) == 3 and np.all(angles <= tolerance))}


def _aperture(fixtures, p):
    keys, tolerance = p['aperture_bindings'], p['scoring']['fixture_position_tolerance_m']
    parts = {name: fixtures[key] for name, key in keys.items()}
    alignments = {name: _axis_alignment(row['quaternion_xyzw'], p['scoring']['fixture_angle_tolerance_rad'])
                  for name, row in parts.items()}
    lower, upper = (array(parts[name]['aabb_m']) for name in ('lower_side', 'upper_side'))
    floor, roof = (array(parts[name]['aabb_m']) for name in ('floor', 'roof'))
    measured = {'entry_x_m': float(min(lower[0, 0], upper[0, 0])),
                'exit_x_m': float(max(lower[1, 0], upper[1, 0])),
                'y_min_m': float(lower[1, 1]), 'y_max_m': float(upper[0, 1]),
                'z_min_m': float(floor[1, 2]), 'z_max_m': float(roof[0, 2])}
    pair_depth_difference = np.abs(lower[:, 0] - upper[:, 0])
    checks = {'axis_aligned': all(row['accepted'] for row in alignments.values()),
              'sidewall_depth_consistent': bool(np.all(pair_depth_difference <= tolerance)),
              'positive_opening': all(measured[high] > measured[low] for low, high in (
                  ('entry_x_m', 'exit_x_m'), ('y_min_m', 'y_max_m'), ('z_min_m', 'z_max_m')))}
    margin = p['scoring']['aperture_clearance_margin_m']
    checks['positive_inset_opening'] = all(measured[high] - measured[low] >= 2 * margin
        for low, high in (('y_min_m', 'y_max_m'), ('z_min_m', 'z_max_m')))
    return {'available': True, 'accepted': all(checks.values()), 'bounds': measured,
            'checks': checks, 'axis_alignment': alignments,
            'sidewall_depth_difference_m': pair_depth_difference.tolist(),
            'source': 'Actual bound sidewall, roof and table native AABBs with collision geometry and quaternion readback.'}


def _grasp(session, controller):
    try:
        raw = _aggregate_grasp(session.robot, controller)
    except ERRORS as error:
        raw = {'available': False, 'measurement_error': f'{type(error).__name__}: {error}'}
    try:
        arms = raw['arms']
        available = raw['available'] is True and set(arms) == set(session.robot.arm_names)
        held = []
        for arm, row in arms.items():
            available &= row.get('available') is True
            if row.get('holding') is True and row.get('constraint_active') is True:
                actual = session.robot._ag_obj_in_hand[arm]
                if (actual is None or not str(actual.prim_path).startswith('/')
                        or actual.name != row.get('object_name')):
                    available = False
                else:
                    held.append(actual.name)
        released = bool(available and raw.get('gripper_open') is True and arms and all(
            row.get('holding') is False and row.get('constraint_active') is False
            and row.get('release_pending') is False and row.get('object_name') is None for row in arms.values()))
        return {'available': bool(available), 'held_names': sorted(set(held)), 'released': released}, raw
    except ERRORS as error:
        return {'available': False, 'held_names': [], 'released': False}, {
            'aggregate': raw, 'measurement_error': f'{type(error).__name__}: {error}'}


def _control(session, controller):
    """An unavailable guard is not an invented invalid-controller event."""
    evidence = {'available': False, 'state_setter_used': getattr(session, '_episode_state_setter_used', None)}
    provider = getattr(controller, 'geometric_rearrangement_evidence', None)
    if callable(provider):
        try:
            evidence['native_guard'] = deepcopy(provider())
            evidence['available'] = evidence['native_guard'].get('available') is True
        except ERRORS as error:
            evidence['measurement_error'] = f'{type(error).__name__}: {error}'
    guard = evidence.get('native_guard')
    guard = guard if isinstance(guard, dict) else {}
    invalid = (evidence['state_setter_used'] is True
               or guard.get('state_setter_used') is True or guard.get('control_valid') is False)
    return not invalid, evidence


def _relation(target, destination, vertices):
    from omnigibson.object_states import OnTop, Touching
    result, errors = {'touching': None, 'on_top': None, 'footprint_inside': None}, []
    for key, state in (('touching', Touching), ('on_top', OnTop)):
        try:
            result[key] = _boolean(target.states[state].get_value(destination))
        except ERRORS as error:
            errors.append(f'{key}: {type(error).__name__}: {error}')
    try:
        if vertices is None:
            raise ValueError('Target collision vertices are unavailable.')
        low, high = (_vector(value, 3) for value in destination.aabb)
        if np.any(high <= low):
            raise ValueError('Native pad bounds must have positive extent.')
        points = array(vertices)
        result['footprint_inside'] = bool(np.all(points[:, :2] >= low[:2]) and np.all(points[:, :2] <= high[:2]))
    except ERRORS as error:
        errors.append(f'footprint: {type(error).__name__}: {error}')
    result['measurement_errors'] = errors
    return result


def _measure(session, controller, config, objects, baseline=None):
    p, errors = config['parameters'], []
    control_valid, control = _control(session, controller)
    grasp, raw_grasp = _grasp(session, controller)
    sample = {'native_time_s': None, 'control_valid': control_valid, 'control_evidence': control,
              'fixtures_valid': None, 'geometry_valid': None, 'grasp': grasp,
              'native_grasp_diagnostics': raw_grasp, 'targets': {}, 'fixtures': {}, 'measurement_errors': errors}
    try:
        value = session.pipeline.og.sim.current_time
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError('Native physics time must be finite.')
        sample['native_time_s'] = float(value)
    except ERRORS as error:
        errors.append(f'clock: {type(error).__name__}: {error}')
    fixture_flags, geometry_flags = [], []
    for key in p['fixture_keys']:
        row, flag = {}, None
        try:
            obj = objects[key]
            fixed = _collidable(obj, dynamic=False)
            flag = fixed
            row = _geometry(obj)
            row['fixed_collidable'] = fixed
            flag = fixed
            if baseline is not None:
                reference = baseline['fixtures'].get(key)
                if not reference or 'measurement_error' in reference:
                    flag = False if flag is False else None
                else:
                    flag = bool(fixed and _fixture_valid(row, reference, p['scoring']))
        except ERRORS as error:
            flag = False if flag is False else None
            row['measurement_error'] = f'{type(error).__name__}: {error}'
        row['preserved'] = flag
        sample['fixtures'][key] = row
        fixture_flags.append(flag)
    sample['fixtures_valid'] = _all(fixture_flags)
    if config['operation'] == 'select_pass_aperture':
        try:
            sample['measured_aperture'] = _aperture(sample['fixtures'], p)
            geometry_flags.append(sample['measured_aperture']['accepted'])
        except ERRORS as error:
            sample['measured_aperture'] = {'available': False, 'accepted': False,
                                            'reason': f'{type(error).__name__}: {error}'}
            geometry_flags.append(None)
    for key, name in p['candidates'].items():
        row = {'object_name': name, 'vertices_m': None, 'dimensions_m': None,
               'position_m': None, 'quaternion_xyzw': None, 'lowest_z_m': None,
               'linear_speed_m_s': None, 'angular_speed_rad_s': None,
               'dynamic_collidable': None, 'robot_contact': None, 'destinations': {}, 'measurement_errors': []}
        obj, geometry_flag = objects[key], None
        try:
            row['dynamic_collidable'] = _collidable(obj, dynamic=True)
            primitive = getattr(obj, '_primitive_type', None)
            row['native_cube_primitive'] = None if primitive is None else primitive == 'Cube'
            if row['native_cube_primitive'] is not True:
                geometry_flag = row['native_cube_primitive']
                raise ValueError('Intrinsic dimensions require a measured native Cube primitive.')
            measured = _geometry(obj)
            row.update({field: measured[field] for field in (
                'vertices_m', 'dimensions_m', 'position_m', 'quaternion_xyzw')})
            row['native_geometry'] = measured
            row['lowest_z_m'] = float(array(measured['vertices_m'])[:, 2].min())
            if baseline is None:
                difference = np.abs(array(row['dimensions_m']) - p['candidate_dimensions_m'][key])
                row['initial_dimension_difference_m'] = difference.tolist()
                geometry_flag = bool(np.all(difference <= INITIAL_DIMENSION_TOLERANCE_M))
            else:
                initial = baseline['targets'].get(key, {}).get('native_geometry')
                geometry_flag = (_relative_geometry(measured, initial, p['scoring']['dimension_relative_tolerance'])
                                 if initial else None)
        except ERRORS as error:
            row['measurement_errors'].append(f'geometry: {type(error).__name__}: {error}')
        row['fixed_shape_preserved'] = geometry_flag
        geometry_flags.append(_all((geometry_flag, row['dynamic_collidable'])))
        for field, getter in (('linear_speed_m_s', 'get_linear_velocity'), ('angular_speed_rad_s', 'get_angular_velocity')):
            try:
                row[field] = float(np.linalg.norm(_vector(getattr(obj, getter)(), 3)))
            except ERRORS as error:
                row['measurement_errors'].append(f'{field}: {type(error).__name__}: {error}')
        try:
            contacts = list(obj.contact_list())
            row['robot_contact'] = any(_under(body, session.robot.prim_path)
                                       for contact in contacts for body in _contact_bodies(contact))
        except ERRORS as error:
            row['measurement_errors'].append(f'contact: {type(error).__name__}: {error}')
        for destination in p['destinations']:
            try:
                row['destinations'][destination] = _relation(obj, objects[destination], row['vertices_m'])
            except ERRORS as error:
                row['destinations'][destination] = {'touching': None, 'on_top': None, 'footprint_inside': None,
                    'measurement_errors': [f'{type(error).__name__}: {error}']}
        sample['targets'][key] = row
    sample['geometry_valid'] = _all(geometry_flags)
    return sample


def _qualification(session, config, objects, sample):
    setup = deepcopy(getattr(session, 'category_interaction_setup', {}))
    setup = setup if isinstance(setup, dict) else {}
    p = config['parameters']
    robot = setup.get('robot', {})
    robot = robot if isinstance(robot, dict) else {}
    checks = {'active_layout_native_initial_support': setup.get('accepted') is True,
              'native_robot_upright': robot.get('upright') is True,
              'native_robot_upright_measurement_available': robot.get('upright_measurement_available') is True,
              'bound_native_identities': all(objects.get(key) is not None and objects[key].name == name
                                            for key, name in config['bindings'].items()),
              'native_candidate_geometry': sample['geometry_valid'] is True,
              'native_fixed_fixtures': sample['fixtures_valid'] is True,
              'native_initial_contacts_available': all(row['robot_contact'] is False
                  and row['linear_speed_m_s'] is not None and row['angular_speed_rad_s'] is not None
                  for row in sample['targets'].values()),
              'initial_hands_released': sample['grasp']['available'] is True and sample['grasp']['released'] is True,
              'initial_control_guard': sample['control_valid'] is True,
              'native_time_available': sample['native_time_s'] is not None}
    if config['operation'] == 'select_pass_aperture':
        checks['native_aperture'] = sample.get('measured_aperture', {}).get('accepted') is True
    return {'accepted': all(checks.values()), 'checks': checks,
            'initial_support': setup, 'native_measurements': deepcopy(sample),
            'candidate_dimensions_proposal_m': deepcopy(p['candidate_dimensions_m']),
            'initial_dimension_tolerance_m': INITIAL_DIMENSION_TOLERANCE_M,
            'route_verified': False, 'manipulation_verified': False,
            'optical_verified': False, 'complete_reference_execution_verified': False,
            'scientific_controls_verified': False,
            'validation_scope': 'Native initialization, measured robot upright gate, geometry and aggregate released hands only.'}


def qualify_operation(session, controller, config, bound):
    try:
        _config(session, config)
        controller = getattr(controller, 'controller', controller)
        return _qualification(session, config, bound, _measure(session, controller, config, bound))
    except ERRORS as error:
        return {'accepted': False, 'checks': {}, 'reason': 'native_dimensional_measurement_unavailable',
                'measurement_error': f'{type(error).__name__}: {error}',
                'route_verified': False, 'manipulation_verified': False, 'optical_verified': False,
                'complete_reference_execution_verified': False}


def _progress(parameters):
    from wia.tasks.wia_dimensional_scoring import DimensionalProgress
    return DimensionalProgress(parameters)


class DimensionalEvaluator:
    def __init__(self, session, controller, config, output):
        self.config = deepcopy(config)
        self.p = _config(session, self.config)
        self.session, self.controller = session, getattr(controller, 'controller', controller)
        self.output = Path(output)
        self.objects = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        self.active, self.scorer, self.initial, self.last_sample = False, None, None, None
        self.initial_qualification, self.pure_parameters, self.measurement_error = None, None, None
        self._done_declared, self._terminal_record = False, None

    @property
    def requires_done_declaration(self):
        return self.config['operation'] == 'select_pass_aperture'

    def declare_done(self, observation):
        """Sample current native evidence once without advancing physics, then seal."""
        if not self.active:
            return None
        if not self.requires_done_declaration:
            return self.record(observation)
        if self._terminal_record is not None:
            return deepcopy(self._terminal_record)
        self.sample_step()
        if self.scorer is not None:
            self.scorer.declare_done()
        self._done_declared = True
        self._terminal_record = self.record(observation)
        return deepcopy(self._terminal_record)

    def begin_episode(self, observation):
        if self.active:
            raise RuntimeError('The dimensional episode has already begun.')
        self.initial = _measure(self.session, self.controller, self.config, self.objects)
        self.last_sample = deepcopy(self.initial)
        self.initial_qualification = _qualification(self.session, self.config, self.objects, self.initial)
        self.pure_parameters = deepcopy(self.p)
        self.pure_parameters['operation'] = self.config['operation']
        aperture = self.initial.get('measured_aperture')
        if self.config['operation'] != 'select_pass_aperture' or aperture.get('accepted') is True:
            if aperture is not None:
                self.pure_parameters['aperture'] = deepcopy(aperture['bounds'])
            self.scorer = _progress(self.pure_parameters)
            self.scorer.begin(self.initial)
        else:
            self.measurement_error = 'Initial native aperture is unavailable or invalid; no substitute aperture was constructed.'
        self.active = True
        return self.record(observation)

    def sample_step(self):
        if not self.active or self._terminal_record is not None:
            return
        self.last_sample = _measure(self.session, self.controller, self.config, self.objects, self.initial)
        if self.scorer is not None:
            self.scorer.update(self.last_sample)

    def record(self, observation, *, action=None):
        """Persist the latest physics sample without advancing any reducer state."""
        if not self.active:
            return None
        if self._terminal_record is not None:
            return deepcopy(self._terminal_record)
        measured = (self.scorer.result() if self.scorer is not None else {
            'schema': APERTURE_SCORING_VERSION if self.requires_done_declaration else self.p['revision'],
            'evaluation_validity': 'unavailable', 'physical_success': False,
            'task_success': False, 'completion_items': [{'id': 'native_dimensional_evidence',
                'completed': False, 'evaluation_validity': 'unavailable', 'kind': 'terminal_condition'}]})
        qualified = self.initial_qualification['accepted'] is True
        result = {**deepcopy(measured), 'schema': self.p['revision'], 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'action': action,
            'physical_scorer_schema': measured.get('schema'),
            'physical_scorer_version': measured.get('scoring_version', measured.get('schema')),
            'initial_qualification': deepcopy(self.initial_qualification),
            'configuration': deepcopy(self.config), 'pure_parameters': deepcopy(self.pure_parameters),
            'configured_proposal': {'candidate_dimensions_m': deepcopy(self.p['candidate_dimensions_m']),
                                    'aperture': deepcopy(self.p.get('aperture'))},
            'native_measurements': deepcopy(self.last_sample), 'measurement_error': self.measurement_error,
            'diagnostic_reducer_physical_success': measured.get('physical_success') is True,
            'evaluation_validity': ('invalid' if measured.get('evaluation_validity') == 'invalid' else
                                    measured.get('evaluation_validity', 'unavailable') if qualified else 'unavailable'),
            'physical_success': qualified and measured.get('physical_success') is True,
            'task_success': qualified and measured.get('task_success', measured.get('physical_success')) is True,
            'evidence_scope': 'Private native dimensional scoring; no route, manipulation or optical qualification is inferred.'}
        native_validity = result['evaluation_validity']
        native_evidence_validity = measured.get('native_evaluation_validity', native_validity)
        if not qualified and native_evidence_validity != 'invalid':
            native_evidence_validity = 'unavailable'
        if self.requires_done_declaration and not self._done_declared:
            result.update(evaluation_validity='pending_done', evaluation_finalized=False,
                task_success=False, physical_success=False, constraints_satisfied=None,
                selected_candidate=None, completion_items=None, completed_goals=None,
                total_goals=None, completion_score=None, score_percent=None, diagnostic_lower_bound=None,
                native_evaluation_validity=native_evidence_validity)
        else:
            result.update(progress(result))
            if native_validity == 'invalid':
                result.update(evaluation_validity='invalid', completion_score=None, score_percent=None)
            if result['evaluation_validity'] != 'complete':
                result.update(task_success=False, physical_success=False)
            if self.requires_done_declaration:
                result['evaluation_finalized'] = True
        write_json(self.output / 'state.json', result)
        with (self.output / 'trajectory.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result, allow_nan=False) + '\n')
        return result


def create_evaluator(session, controller, config, output):
    return DimensionalEvaluator(session, controller, config, output)
