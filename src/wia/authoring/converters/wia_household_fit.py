"""Measure native household passage and compartment placement without scene writes.

Run: PYTHONPATH=src python -B -m pytest -q -p no:cacheprovider
tests/harness_integration/test_wia_household_fit.py. Definitions initialize with
active_layout; this adapter only reads native geometry, contacts and robot state.
"""
from __future__ import annotations

from wia.runtime.prechecks import episode_qualification, permits_episode

from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from wia.authoring.converters import wia_dimensional as native
from wia.authoring.converters.rigid_partial_insertion import _box_gap
from wia.authoring.converters.wia_mirror import _household_geometry
from wia.authoring.modules import module_identity

MODULE = 'wia.authoring.converters.wia_household_fit'
REVISION = 'wia_household_fit_v1'
OPERATIONS = native.OPERATIONS
SIDE_TOP_INSET_M = .001
FLOOR_CONTACT_TOLERANCE_M = .003
HAND_CLEARANCE_M = .02
UPRIGHT_TOLERANCE_RAD = float(np.deg2rad(10.))
ERRORS = native.ERRORS


def household_geometry(obj):
    """Read all fixed-connected rigid DatasetObject collision meshes in root space.

    Single-root assets retain the existing household helper's behavior. Sibling
    collision links require an actual enabled USD PhysicsFixedJoint path to the
    root. World matrices use USD row vectors; root scale is applied once after
    cancelling the root's complete world transform.
    """
    import omnigibson.lazy as lazy
    from omnigibson.objects.dataset_object import DatasetObject

    if not isinstance(obj, DatasetObject):
        raise ValueError('Household geometry requires an actual native DatasetObject.')
    root = obj.root_link
    links = [link for link in obj.links.values() if not native._boolean(link.is_meta_link)
             and any(native._boolean(mesh.collision_enabled) for mesh in link.collision_meshes.values())]
    if len(links) == 1 and links[0] is root:
        return _household_geometry(obj)
    counts = {}
    for field in ('n_dof', 'n_joints'):
        value = getattr(obj, field)
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value != 0):
            raise ValueError('Fixed-link household geometry requires measured native n_dof == n_joints == 0.')
        counts[field] = int(value)
    articulated = native._boolean(obj.articulated)
    object_path, root_path = str(obj.prim.GetPath()), str(root.prim.GetPath())
    if (not object_path.startswith('/') or object_path != str(obj.prim_path)
            or not root_path.startswith(object_path + '/') or root_path != str(root.prim_path)
            or not any(link is root for link in links)):
        raise ValueError('All fixed-connected collision geometry must include the native root link.')
    link_paths = {}
    for link in obj.links.values():
        path = str(link.prim.GetPath())
        if (not path.startswith(object_path + '/') or path != str(link.prim_path) or path in link_paths):
            raise ValueError('Native collision-link USD identities are malformed or duplicated.')
        link_paths[path] = link
    graph, joints = {path: set() for path in link_paths}, []
    for prim in lazy.pxr.Usd.PrimRange(obj.prim):
        if not prim.IsA(lazy.pxr.UsdPhysics.Joint):
            continue
        if not prim.IsA(lazy.pxr.UsdPhysics.FixedJoint):
            raise ValueError('A household rigid asset must not contain movable USD joints.')
        joint = lazy.pxr.UsdPhysics.Joint(prim)
        enabled = native._boolean(joint.GetJointEnabledAttr().Get())
        bodies = [joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()]
        if (any(len(body) != 1 for body in bodies)
                or any(str(body[0]) not in link_paths for body in bodies)
                or str(bodies[0][0]) == str(bodies[1][0])):
            raise ValueError('Every native fixed joint must bind two distinct measured object links.')
        first, second = (str(body[0]) for body in bodies)
        joints.append({'prim_path': str(prim.GetPath()), 'type': 'PhysicsFixedJoint',
                       'body0': first, 'body1': second, 'enabled': enabled})
        if enabled:
            graph[first].add(second)
            graph[second].add(first)
    connected, pending = set(), [root_path]
    while pending:
        path = pending.pop()
        if path not in connected:
            connected.add(path)
            pending.extend(graph[path] - connected)
    collision_paths = sorted(str(link.prim_path) for link in links)
    if not set(collision_paths).issubset(connected):
        raise ValueError('Every enabled non-meta collision link needs an enabled fixed-joint path to root.')
    topology = {'root_link_prim_path': root_path, 'collision_link_prim_paths': collision_paths,
                'fixed_joints': sorted(joints, key=lambda row: row['prim_path']),
                'all_collision_links_connected': True}
    cache = lazy.pxr.UsdGeom.XformCache()

    def affine(value):
        matrix = native.array(value)
        if (matrix.shape != (4, 4)
                or not np.allclose(matrix[:, 3], [0., 0., 0., 1.], rtol=0., atol=1e-12)
                or np.linalg.det(matrix[:3, :3]) == 0):
            raise ValueError('Native collision transforms must be finite nonsingular affine row-vector matrices.')
        return matrix

    root_inverse = np.linalg.inv(affine(cache.GetLocalToWorldTransform(root.prim)))
    root_scale = native._vector(root.scale, 3, positive=True)
    position, quaternion = obj.get_position_orientation()
    position, quaternion = native._vector(position, 3), native._vector(quaternion, 4)
    if np.linalg.norm(quaternion) == 0:
        raise ValueError('Native quaternion is zero.')
    scale = native._vector(obj.scale, 3, positive=True)
    points, body_points, meshes = [], [], []
    for path in collision_paths:
        for mesh in link_paths[path].collision_meshes.values():
            if not native._boolean(mesh.collision_enabled):
                continue
            raw = native.array(mesh.points)
            if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw):
                raise ValueError('Actual native collision vertices are unavailable.')
            stamp = native._native_shape_stamp(mesh)
            if stamp['type'] != 'Mesh':
                raise ValueError('Fixed-connected household geometry requires native USD Mesh points and topology.')
            current = native.array(mesh.prim.GetAttribute('points').Get())
            if (current.shape != raw.shape
                    or not np.array_equal(current.astype(np.float32), raw.astype(np.float32))):
                raise ValueError('Cached collision points do not match current USD points at float32 accuracy.')
            mesh_path = str(mesh.prim.GetPath())
            if mesh_path != str(mesh.prim_path) or not mesh_path.startswith(path + '/'):
                raise ValueError('Each collision mesh must descend from its measured native link.')
            matrix = affine(affine(cache.GetLocalToWorldTransform(mesh.prim)) @ root_inverse)
            homogeneous = np.column_stack((raw, np.ones(len(raw)))) @ matrix
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
            world = native.array(mesh.transform_local_points_to_world(mesh.points))
            if world.shape != raw.shape:
                raise ValueError('Native collision vertex transformation changed its array shape.')
            points.append(world)
            body_points.append(body)
            meshes.append({'prim_path': mesh_path, 'link_prim_path': path, 'local_vertices_m': raw.tolist(),
                           'faces': faces.tolist(), 'mesh_to_root_link_transform': matrix.tolist(),
                           'native_shape_stamp': stamp})
    if not points:
        raise ValueError('No enabled non-meta native collision geometry is measurable.')
    vertices, body = np.concatenate(points), np.concatenate(body_points)
    dimensions = np.ptp(body, axis=0)
    if np.any(dimensions <= 0):
        raise ValueError('Native collision geometry must have three positive intrinsic dimensions.')
    low, high = (native._vector(value, 3) for value in obj.aabb)
    if np.any(high <= low):
        raise ValueError('Native collision AABB is unavailable.')
    return {'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist(), 'scale': scale.tolist(),
            'vertices_m': vertices.tolist(), 'body_vertices_m': body.tolist(), 'dimensions_m': dimensions.tolist(),
            'root_link_prim_path': root_path, 'root_link_scale': root_scale.tolist(),
            'n_dof': counts['n_dof'], 'n_joints': counts['n_joints'], 'articulated': articulated,
            'intrinsic_geometry_source': 'Float64 USD mesh-world times inverse root-world; measured root scale applied once.',
            'aabb_m': [low.tolist(), high.tolist()], 'collision_meshes': meshes, 'fixed_link_topology': topology,
            'geometry_scope': 'All enabled fixed-connected native collision Mesh vertices; world AABB is diagnostic only.'}


def _config(session, config):
    p, operation = config['parameters'], config['operation']
    if (module_identity(config.get('module')) != MODULE or p.get('revision') != REVISION
            or operation not in OPERATIONS or p.get('operation', operation) != operation):
        raise ValueError('Use the explicit household fit revision and operation.')
    scene, room = native._scene_contract(p)
    if config.get('scene') != scene or config.get('room') != room:
        raise ValueError('The task must match its explicit scene contract.')
    native._validate_scene_identity(session, scene, room)
    candidates, bindings = p['candidates'], config['bindings']
    if (not isinstance(candidates, dict) or not candidates
            or any(not isinstance(key, str) or not key or not isinstance(name, str) or not name
                   or bindings.get(key) != name for key, name in candidates.items())
            or len(set(candidates.values())) != len(candidates)):
        raise ValueError('Candidates require distinct exact native binding identities.')
    if operation == 'sort_intrinsic_dimension' and len(candidates) not in (2, 3):
        raise ValueError('Household sorting requires two or three targets.')
    fixtures, destinations = p['fixture_keys'], p['destinations']
    if (not isinstance(fixtures, (list, tuple)) or not fixtures
            or not isinstance(destinations, (list, tuple)) or not destinations
            or any(not isinstance(key, str) or not key for key in (*fixtures, *destinations))
            or len(set(fixtures)) != len(fixtures) or len(set(destinations)) != len(destinations)
            or not set(destinations).issubset(fixtures) or not set(fixtures).issubset(bindings)
            or set(candidates).intersection(fixtures)
            or len(set(bindings.values())) != len(bindings)):
        raise ValueError('Every fixed fixture and ordered floor destination needs a distinct binding.')
    if set(p['candidate_dimensions_m']) != set(candidates) or set(p['candidate_assets']) != set(candidates):
        raise ValueError('All targets need explicit asset contracts and dimension proposals.')
    for key in candidates:
        native._vector(p['candidate_dimensions_m'][key], 3, positive=True)
        asset = p['candidate_assets'][key]
        if (not isinstance(asset, dict)
                or any(not isinstance(asset.get(field), str) or not asset[field].strip()
                       for field in ('category', 'model'))):
            raise ValueError('Every target needs its exact native category and model.')
        native._vector(asset['scale'], 3, positive=True)
    scoring = p['scoring']
    if scoring.get('manipulation_revision') != 'final_state_v1' or scoring['minimum_lift_m'] is not None:
        raise ValueError('Household fit requires final_state_v1 and an explicit null minimum lift.')
    for key in ('stability_duration_s', 'maximum_linear_speed_m_s', 'maximum_angular_speed_rad_s',
                'max_sample_gap_s', 'fixture_position_tolerance_m', 'fixture_angle_tolerance_rad',
                'dimension_relative_tolerance', 'aperture_clearance_margin_m', 'minimum_assignment_gap_m'):
        value = scoring[key]
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
            raise ValueError('Household scoring thresholds must be explicit finite positive numbers.')
    if 'aperture_floor_contact_tolerance_m' in scoring:
        value = scoring['aperture_floor_contact_tolerance_m']
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value < 0):
            raise ValueError('aperture_floor_contact_tolerance_m must be an explicit finite nonnegative number.')
    if p['unselected_position_tolerance_m'] != .02:
        raise ValueError('The inert legacy unselected position tolerance must remain 0.02 m.')
    expected_items = (['passage', 'stable_placement'] if operation == 'select_pass_aperture'
                      else ['assignment', 'stable_placement'])
    items = p['completion_items']
    if (not isinstance(items, list) or any(not isinstance(item, dict) for item in items)
            or [item.get('id') for item in items] != expected_items
            or any(item.get('kind') != ('historical_checkpoint' if item['id'] == 'passage' else 'terminal_condition')
                   for item in items)):
        raise ValueError('Completion items must match the final-state dimensional checkpoints.')
    if operation == 'select_pass_aperture':
        aperture = p['aperture_bindings']
        if (len(destinations) != 1 or not isinstance(aperture, dict)
                or set(aperture) != {'lower_side', 'upper_side', 'roof', 'floor'}
                or len(set(aperture.values())) != 4 or not set(aperture.values()).issubset(fixtures)):
            raise ValueError('Aperture passage requires four distinct bound mouth-frame parts and one floor.')
    elif len(destinations) != len(candidates):
        raise ValueError('Sorting requires one ordered floor destination per target.')
    interiors = p['destination_interiors']
    if not isinstance(interiors, dict) or set(interiors) != set(destinations):
        raise ValueError('Every destination needs one measured compartment contract.')
    for key, contract in interiors.items():
        required = {'floor', 'left', 'right', 'back'}
        if (not isinstance(contract, dict) or not required.issubset(contract)
                or not set(contract).issubset(required | {'front', 'roof'})
                or contract['floor'] != key or not set(contract.values()).issubset(fixtures)
                or len(set(contract.values())) != len(contract)):
            raise ValueError('Compartment floors, walls and optional front/roof must bind distinct fixed fixtures.')
    upright = p.get('upright_axes', {})
    if (not isinstance(upright, dict) or not set(upright).issubset(candidates)
            or any(isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer))
                   or axis not in (0, 1, 2) for axis in upright.values())):
        raise ValueError('Optional upright axes must map target keys to local axis indices 0, 1 or 2.')
    return p


def _interior(fixtures, contract, scoring):
    """Measure usable world bounds from fixed, axis-aligned native Cube faces."""
    parts = {role: fixtures[key] for role, key in contract.items()}
    alignments = {role: native._axis_alignment(row['quaternion_xyzw'], scoring['fixture_angle_tolerance_rad'])
                  for role, row in parts.items()}
    boxes = {role: native.array(row['aabb_m']) for role, row in parts.items()}
    if any(box.shape != (2, 3) or np.any(box[1] <= box[0]) for box in boxes.values()):
        raise ValueError('Native compartment fixture bounds require positive three-dimensional extent.')
    low = np.array([boxes['front'][1, 0] if 'front' in boxes else boxes['floor'][0, 0],
                    boxes['left'][1, 1], boxes['floor'][1, 2]])
    high = np.array([boxes['back'][0, 0], boxes['right'][0, 1],
                     boxes['roof'][0, 2] if 'roof' in boxes
                     else min(boxes[role][1, 2] for role in ('left', 'right', 'back'))])
    checks = {'axis_aligned': all(row['accepted'] for row in alignments.values()),
              'positive_inset_interior': bool(np.all(high - low > 2 * SIDE_TOP_INSET_M))}
    return {'available': True, 'accepted': all(checks.values()), 'bounds_m': [low.tolist(), high.tolist()],
            'checks': checks, 'axis_alignment': alignments, 'bindings': deepcopy(contract),
            'side_top_inset_m': SIDE_TOP_INSET_M, 'floor_contact_tolerance_m': FLOOR_CONTACT_TOLERANCE_M,
            'source': 'Actual bound floor, wall, optional front and roof native collision AABB faces.'}


def _contained(vertices, interior):
    if vertices is None or interior.get('available') is not True:
        return None
    points = native.array(vertices)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError('Full containment requires measured target collision vertices.')
    if interior.get('accepted') is not True:
        return False
    bounds = native.array(interior['bounds_m'])
    low, high = bounds[0].copy(), bounds[1].copy()
    low[:2] += SIDE_TOP_INSET_M
    low[2] -= FLOOR_CONTACT_TOLERANCE_M
    high -= SIDE_TOP_INSET_M
    return bool(np.all(points >= low) and np.all(points <= high))


def _hand_clearance(robot, target):
    """Use the same all-arm eef/finger links as rigid_partial_insertion."""
    arms = tuple(robot.arm_names)
    if not arms or len(set(arms)) != len(arms):
        raise ValueError('Native arm identities are unavailable.')
    links = []
    for arm in arms:
        fingers = list(robot.finger_links[arm])
        if not fingers:
            raise ValueError('Native finger links are unavailable for an arm.')
        links.append(robot.eef_links[arm])
        links.extend(fingers)
    gaps = [_box_gap(link, target) for link in links]
    return {'available': True, 'accepted': bool(min(gaps) >= HAND_CLEARANCE_M),
            'minimum_gap_m': min(gaps), 'required_clearance_m': HAND_CLEARANCE_M,
            'link_gaps_m': [{'prim_path': str(link.prim_path), 'gap_m': gap}
                            for link, gap in zip(links, gaps)],
            'source': 'Every native arm eef_links entry and finger_links entry; Euclidean AABB separation.'}


def _upright(obj, axis):
    if axis is None:
        return {'available': True, 'accepted': True, 'required': False}
    _, quaternion = obj.root_link.get_position_orientation()
    quaternion = native._vector(quaternion, 4)
    if np.linalg.norm(quaternion) == 0:
        raise ValueError('The measured root-link quaternion is zero.')
    world_axis = Rotation.from_quat(quaternion).as_matrix()[:, axis]
    angle = float(np.arccos(np.clip(world_axis[2], -1., 1.)))
    return {'available': True, 'accepted': bool(angle <= UPRIGHT_TOLERANCE_RAD), 'required': True,
            'local_axis': axis, 'angle_from_positive_world_z_rad': angle,
            'tolerance_rad': UPRIGHT_TOLERANCE_RAD, 'root_quaternion_xyzw': quaternion.tolist()}


def _measure(session, controller, config, objects, baseline=None):
    p, errors = config['parameters'], []
    control_valid, control = native._control(session, controller)
    grasp, raw_grasp = native._grasp(session, controller)
    sample = {'native_time_s': None, 'control_valid': control_valid, 'control_evidence': control,
              'fixtures_valid': None, 'geometry_valid': None, 'grasp': grasp,
              'native_grasp_diagnostics': raw_grasp, 'targets': {}, 'fixtures': {},
              'destination_interiors': {}, 'measurement_errors': errors}
    try:
        value = session.pipeline.og.sim.current_time
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(float(value)):
            raise ValueError('Native physics time must be finite.')
        sample['native_time_s'] = float(value)
    except ERRORS as error:
        errors.append(f'clock: {type(error).__name__}: {error}')
    fixture_flags, geometry_flags = [], []
    for key in p['fixture_keys']:
        row, flag = {}, None
        try:
            obj = objects[key]
            fixed = native._collidable(obj, dynamic=False)
            flag = bool(fixed and obj.name == config['bindings'][key])
            row = native._geometry(obj)
            row['fixed_collidable'] = fixed
            if baseline is not None:
                reference = baseline['fixtures'].get(key)
                if not reference or 'measurement_error' in reference:
                    flag = False if flag is False else None
                else:
                    flag = bool(flag and native._fixture_valid(row, reference, p['scoring']))
        except ERRORS as error:
            flag = False if flag is False else None
            row['measurement_error'] = f'{type(error).__name__}: {error}'
        row['preserved'] = flag
        sample['fixtures'][key] = row
        fixture_flags.append(flag)
    sample['fixtures_valid'] = native._all(fixture_flags)
    for destination, contract in p['destination_interiors'].items():
        try:
            measured = _interior(sample['fixtures'], contract, p['scoring'])
            geometry_flags.append(measured['accepted'])
        except ERRORS as error:
            measured = {'available': False, 'accepted': False, 'measurement_error': f'{type(error).__name__}: {error}'}
            geometry_flags.append(None)
        sample['destination_interiors'][destination] = measured
    if config['operation'] == 'select_pass_aperture':
        try:
            sample['measured_aperture'] = native._aperture(sample['fixtures'], p)
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
        obj, geometry_flag = objects.get(key), None
        try:
            row['dynamic_collidable'] = native._collidable(obj, dynamic=True)
            asset = p['candidate_assets'][key]
            geometry_flag = bool(obj.name == name and obj.category == asset['category'] and obj.model == asset['model']
                                 and np.allclose(native._vector(obj.scale, 3, positive=True), asset['scale'],
                                                 rtol=1e-6, atol=1e-6))
            if not geometry_flag:
                raise ValueError('The native target identity/category/model/scale mismatches its explicit contract.')
            measured = household_geometry(obj)
            row.update({field: measured[field] for field in (
                'vertices_m', 'dimensions_m', 'position_m', 'quaternion_xyzw')})
            row['native_geometry'] = measured
            row['lowest_z_m'] = float(native.array(measured['vertices_m'])[:, 2].min())
            row['dimension_proposal_difference_m'] = (
                native.array(measured['dimensions_m']) - p['candidate_dimensions_m'][key]).tolist()
            if baseline is not None:
                initial = baseline['targets'].get(key, {}).get('native_geometry')
                geometry_flag = (native._relative_geometry(measured, initial, p['scoring']['dimension_relative_tolerance'])
                                 if initial else None)
                if initial and measured.get('fixed_link_topology') != initial.get('fixed_link_topology'):
                    geometry_flag = False
        except ERRORS as error:
            geometry_flag = False if geometry_flag is False else None
            row['measurement_errors'].append(f'geometry: {type(error).__name__}: {error}')
        row['fixed_shape_preserved'] = geometry_flag
        geometry_flags.append(native._all((geometry_flag, row['dynamic_collidable'])))
        for field, getter in (('linear_speed_m_s', 'get_linear_velocity'), ('angular_speed_rad_s', 'get_angular_velocity')):
            try:
                row[field] = float(np.linalg.norm(native._vector(getattr(obj, getter)(), 3)))
            except ERRORS as error:
                row['measurement_errors'].append(f'{field}: {type(error).__name__}: {error}')
        try:
            row['robot_contact'] = any(native._under(body, session.robot.prim_path)
                for contact in obj.contact_list() for body in native._contact_bodies(contact))
        except ERRORS as error:
            row['measurement_errors'].append(f'contact: {type(error).__name__}: {error}')
        for field, measure in (('hand_clearance', lambda: _hand_clearance(session.robot, obj)),
                               ('upright', lambda: _upright(obj, p.get('upright_axes', {}).get(key)))):
            try:
                row[field] = measure()
            except ERRORS as error:
                row[field] = {'available': False, 'accepted': None, 'measurement_error': f'{type(error).__name__}: {error}'}
        for destination in p['destinations']:
            try:
                relation = native._relation(obj, objects[destination], row['vertices_m'])
                inside = _contained(row['vertices_m'], sample['destination_interiors'][destination])
                relation['native_floor_footprint_inside'] = relation['footprint_inside']
                relation['full_vertices_inside'] = inside
                relation['hand_clear'] = row['hand_clearance']['accepted']
                relation['upright'] = row['upright']['accepted']
                # The inherited reducer consumes these three native boolean fields.
                # Keep missing evidence unavailable even when another gate is false.
                gates = [relation['footprint_inside'], inside, relation['hand_clear'], relation['upright']]
                relation['footprint_inside'] = None if any(value is None for value in gates) else all(gates)
            except ERRORS as error:
                relation = {'touching': None, 'on_top': None, 'footprint_inside': None,
                            'measurement_errors': [f'{type(error).__name__}: {error}']}
            row['destinations'][destination] = relation
        sample['targets'][key] = row
    sample['geometry_valid'] = native._all(geometry_flags)
    return sample


def _qualification(session, config, objects, sample):
    qualification = native._qualification(session, config, objects, sample)
    checks = qualification['checks']
    checks['native_compartment_interiors'] = all(
        row.get('available') is True and row.get('accepted') is True
        for row in sample['destination_interiors'].values())
    checks['native_hand_clearance_available'] = all(
        row['hand_clearance']['available'] is True for row in sample['targets'].values())
    checks['native_required_pose_available'] = all(
        row['upright']['available'] is True for row in sample['targets'].values())
    checks['native_destination_evidence_available'] = all(
        relation.get(field) is not None for row in sample['targets'].values()
        for relation in row['destinations'].values() for field in ('touching', 'on_top', 'footprint_inside'))
    qualification['accepted'] = all(checks.values())
    qualification['candidate_dimensions_are_proposals'] = True
    qualification['validation_scope'] = (
        'Native active_layout, exact identities, geometry, compartment/frame faces, robot upright and released hands; '
        'no route or manipulation execution is inferred. Dimension proposals are diagnostic only.')
    return qualification


def qualify_operation(session, controller, config, bound):
    """Return read-only initial native qualification; unavailable evidence never passes."""
    try:
        _config(session, config)
        controller = getattr(controller, 'controller', controller)
        return _qualification(session, config, bound, _measure(session, controller, config, bound))
    except ERRORS as error:
        return {'accepted': False, 'checks': {}, 'reason': 'native_household_fit_measurement_unavailable',
                'measurement_error': f'{type(error).__name__}: {error}',
                'route_verified': False, 'manipulation_verified': False, 'optical_verified': False,
                'complete_reference_execution_verified': False}


class HouseholdFitEvaluator(native.DimensionalEvaluator):
    """Reuse dimensional progress and DONE persistence with household measurements."""

    def __init__(self, session, controller, config, output):
        self.config = deepcopy(config)
        self.p = _config(session, self.config)
        self.session, self.controller = session, getattr(controller, 'controller', controller)
        self.output = Path(output)
        self.objects = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        self.active, self.scorer, self.initial, self.last_sample = False, None, None, None
        self.initial_qualification, self.pure_parameters, self.measurement_error = None, None, None
        self._done_declared, self._terminal_record = False, None

    def begin_episode(self, observation):
        if self.active:
            raise RuntimeError('The household fit episode has already begun.')
        self.initial = _measure(self.session, self.controller, self.config, self.objects)
        self.last_sample = deepcopy(self.initial)
        self.initial_qualification = episode_qualification(self.session, _qualification, self.session, self.config, self.objects, self.initial)
        self.pure_parameters = deepcopy(self.p)
        self.pure_parameters['operation'] = self.config['operation']
        aperture = self.initial.get('measured_aperture')
        if not self.requires_done_declaration or aperture.get('accepted') is True:
            if aperture is not None:
                self.pure_parameters['aperture'] = deepcopy(aperture['bounds'])
            self.scorer = native._progress(self.pure_parameters)
            self.scorer.begin(self.initial)
        else:
            self.measurement_error = 'Initial native mouth frame is unavailable or invalid; no substitute aperture was constructed.'
        self.active = True
        return self.record(observation)

    def sample_step(self):
        if not self.active or self._terminal_record is not None:
            return
        self.last_sample = _measure(self.session, self.controller, self.config, self.objects, self.initial)
        if self.scorer is not None:
            self.scorer.update(self.last_sample)


def create_evaluator(session, controller, config, output):
    """Create the read-only native household evaluator for one authorized episode."""
    return HouseholdFitEvaluator(session, controller, config, output)
