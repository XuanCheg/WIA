"""Read-only mesh translation and settled cloth-cover support receipts.

Run CPU checks: python -B -m pytest -q tests/test_task_edit2_deformable.py.
Root captures before relocation, verifies world particles immediately after the
one root translation without physics, stores that proof in the receipt, then
measures after settling. No function writes poses, particles or simulator state.
"""
from __future__ import annotations

import hashlib
import json
import threading
import traceback
from types import SimpleNamespace

import numpy as np

REVISION = 'task_edit2_deformable_20261018_v1'
TRANSLATION_ATOL_M = 1e-5
ITEM_POSITION_TOLERANCE_M = .015  # Existing shared rigid-item guard.
ITEM_ORIENTATION_TOLERANCE_RAD = .035  # Existing shared rigid-item guard.
DECK_PENETRATION_TOLERANCE_M = .03  # Existing deformable evaluator floor guard.
PENDING_ACTION_GATES = ['actual_working_hand_sweep', 'real_mesh_reveal_before_grasp', 'native_item_grasp_and_sustained_lift']


def _read_field(callback, field):
    """Attach a field name to a failed read without replacing its exception."""
    try:
        return callback()
    except Exception as error:
        error.measurement_field = field
        raise


def _read_pair(callback, field):
    def read():
        first, second = callback()
        return first, second
    return _read_field(read, field)


def _array(value, field='unlabelled_numeric_evidence', *, dtype=None):
    """Validate the same numeric evidence while reporting bounded metadata.

    ``dtype=float`` is used only at reads whose existing consumer already cast
    to float (not to accept additional inputs). No raw values enter diagnostics.
    """
    diagnostic = {'field': field, 'supplied_type': type(value).__module__ + '.' + type(value).__name__,
        'dtype': None, 'shape': None, 'nonfinite_count': None}
    try:
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        raw = np.asarray(value)
        diagnostic.update(dtype=str(raw.dtype), shape=list(raw.shape))
        if raw.dtype.names:
            if set(raw.dtype.names) != {'x', 'y', 'z'}:
                raise ValueError('Unsupported structured native vector fields.')
            # Native PhysX scalar/batched contact vectors retain XYZ order.
            # This matches the shared interaction adapter, without substituting
            # coordinates or relaxing any mesh/support acceptance condition.
            raw = np.stack([raw[axis] for axis in ('x', 'y', 'z')], axis=-1)
        result = raw if dtype is None else np.asarray(raw, dtype=dtype)
        if result.dtype.kind in 'biuf':
            diagnostic['nonfinite_count'] = int(np.count_nonzero(~np.isfinite(result)))
        if result.dtype.kind not in 'biuf' or diagnostic['nonfinite_count']:
            raise ValueError('invalid_numeric_evidence')
        return result
    except Exception as cause:
        diagnostic['conversion_exception_type'] = type(cause).__name__
        error = ValueError('Native numeric field ' + field + ': type=' + diagnostic['supplied_type'] +
            ', dtype=' + str(diagnostic['dtype']) + ', shape=' + str(diagnostic['shape']) +
            ', nonfinite_count=' + str(diagnostic['nonfinite_count']))
        error.measurement_field = field
        error.numeric_field_diagnostic = diagnostic
        raise error from cause


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _clock(session):
    value = float(_array(_read_field(lambda: session.pipeline.og.sim.current_time, 'native_time_s'), 'native_time_s', dtype=float))
    if not np.isfinite(value):
        raise ValueError('Native simulation time is unavailable.')
    return value


def _context(session, parameters):
    reset = getattr(session, 'native_reset_id', None)
    if not reset or not parameters.get('source_identity') or not parameters.get('source_sha256'):
        raise ValueError('Exact source and native reset provenance are unavailable.')
    task = getattr(session, 'loaded_task', None)
    return {'source_identity': parameters['source_identity'], 'source_sha256': parameters['source_sha256'],
        'native_reset_id': reset, 'scene': session.scene_name, 'room': session.room_name,
        'task_sha256': getattr(task, 'file_sha256', None), 'task_seed': getattr(task, 'seed', None)}


def _receipt_context(session, receipt):
    if receipt.get('schema') != REVISION or receipt.get('available') is not True or receipt.get('accepted') is not True:
        raise ValueError('A verified original native cloth-cover receipt is required.')
    context = receipt['context']
    if _context(session, context) != context:
        raise ValueError('Cloth-cover receipt belongs to another source/reset/task context.')
    if receipt.get('owner_thread_id') != threading.get_ident():
        raise ValueError('Native mesh reads must remain on the capture owner thread.')
    return context


def _plain(value, field):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    try:
        converted = np.asarray(value)
    except (ValueError, TypeError):
        return str(value)
    if converted.dtype.kind in 'biuf':
        return _array(value, field).tolist()
    return str(value)


def _material_prim(stage, path):
    prim = _read_field(lambda: stage.GetPrimAtPath(path), 'material.' + path + '.prim')
    if not prim.IsValid():
        raise ValueError('Native bound material prim is missing: ' + path)
    # Include descendant shaders and relationships, not only a material name.
    rows = {}
    pending = [prim]
    while pending:
        current = pending.pop()
        rows[str(current.GetPath())] = {'type': str(current.GetTypeName()),
            'attributes': {attr.GetName(): _plain(_read_field(attr.Get,
                'material.' + str(current.GetPath()) + '.' + attr.GetName()),
                'material.' + str(current.GetPath()) + '.' + attr.GetName()) for attr in current.GetAttributes()},
            'relationships': {rel.GetName(): [str(target) for target in rel.GetTargets()]
                for rel in current.GetRelationships()}}
        pending.extend(current.GetChildren())
    return rows


def _materials(session, obj, *, cloth=False):
    stage = session.pipeline.og.sim.stage
    prefix = 'cloth' if cloth else 'item'
    paths = _read_field(lambda: sorted(str(material.prim_path) for material in obj.materials), prefix + '.material_bindings')
    if not paths:
        raise ValueError('Actual native object material bindings are unavailable.')
    result = {'visual_bindings': paths, 'material_prims': {path: _material_prim(stage, path) for path in paths}}
    if cloth:
        import omnigibson.lazy as lazy
        root = obj.root_link
        path = str(root.cloth_system.mat_path)
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise ValueError('Native cloth PBD material is missing.')
        api = lazy.pxr.PhysxSchema.PhysxPBDMaterialAPI(prim)
        physical = {name: _read_field(lambda name=name: getattr(api, 'Get' + name.title() + 'Attr')().Get(),
            'cloth.pbd_physics.' + name) for name in ('friction', 'drag', 'lift')}
        for name, value in physical.items():
            # Existing physical values were converted to float before testing.
            _array(value, 'cloth.pbd_physics.' + name, dtype=float)
        if any(value is None or not np.isfinite(float(value)) for value in physical.values()):
            raise ValueError('Native cloth PBD material values are unavailable.')
        springs = {name: float(_array(_read_field(lambda name=name: getattr(root, name), 'cloth.spring_settings.' + name),
            'cloth.spring_settings.' + name, dtype=float)) for name in ('bend_stiffness', 'shear_stiffness', 'stretch_stiffness', 'damping')}
        if not np.isfinite(list(springs.values())).all():
            raise ValueError('Native cloth spring settings are unavailable.')
        result.update(pbd_material_path=path, pbd_material_prims=_material_prim(stage, path),
            pbd_physics={key: float(value) for key, value in physical.items()}, spring_settings=springs)
    return result


def _object(session, obj, *, cloth=False):
    prefix = 'cloth' if cloth else 'item'
    position_value, quaternion_value = _read_pair(obj.get_position_orientation, prefix + '.root_pose')
    position, quaternion = [_array(value, prefix + '.' + field).astype(float)
        for field, value in (('root_position_m', position_value), ('root_quaternion_xyzw', quaternion_value))]
    low_value, high_value = _read_pair(lambda: obj.aabb, prefix + '.aabb_m')
    low, high = [_array(value, prefix + '.aabb_' + bound + '_m').astype(float)
        for bound, value in (('low', low_value), ('high', high_value))]
    mass = float(_array(_read_field(lambda: obj.root_link.mass if cloth else obj.mass, prefix + '.mass_kg'),
        prefix + '.mass_kg', dtype=float))
    if mass <= 0 or not np.isfinite(mass):
        raise ValueError('Actual native mass must be positive; cloth uses root_link.mass.')
    return {'name': obj.name, 'prim_path': obj.prim_path, 'category': obj.category, 'model': obj.model,
        'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist(), 'aabb_m': [low.tolist(), high.tolist()],
        'scale': _array(_read_field(lambda: obj.scale, prefix + '.scale'), prefix + '.scale').tolist(),
        'mass_kg': mass, 'materials': _materials(session, obj, cloth=cloth),
        'visual_only': bool(obj.visual_only), 'fixed_base': bool(obj.fixed_base),
        **({'root_scale': _array(_read_field(lambda: obj.root_link.scale, 'cloth.root_scale'), 'cloth.root_scale').tolist()} if cloth else {})}


def _mesh_state(cloth):
    from wia.authoring.converters.physical_structure_deformable import _mesh
    # Keep the original _mesh validation and native read order. Its float casts
    # are instrumented at their actual native sources; no replacement geometry
    # is supplied and neither solver points nor the legacy validator is changed.
    class MeshReads:
        def compute_particle_positions(self):
            return _array(_read_field(lambda: cloth.root_link.compute_particle_positions(),
                'cloth.world_particle_positions_m'), 'cloth.world_particle_positions_m', dtype=float)
        @property
        def faces(self):
            return _array(_read_field(lambda: cloth.root_link.faces, 'cloth.faces'), 'cloth.faces', dtype=float)
    points, triangles = _mesh(SimpleNamespace(root_link=MeshReads()))
    points = _array(points, 'cloth.world_particle_positions_m')
    triangles = _array(triangles, 'cloth.world_triangles_m')
    raw_faces = _array(_read_field(lambda: cloth.root_link.faces, 'cloth.faces'), 'cloth.faces')
    if not np.equal(raw_faces, np.floor(raw_faces)).all():
        raise ValueError('Native face connectivity is not integer-valued.')
    faces = raw_faces.astype(np.int64)
    velocities = {'available': False, 'reason': 'native_particle_velocities_unavailable'}
    try:
        value = _read_field(lambda: cloth.root_link.particle_velocities, 'cloth.particle_velocities_m_s')
    except (AttributeError, NotImplementedError):
        value = None
    if value is not None:
        velocity = _array(value, 'cloth.particle_velocities_m_s').astype(float)
        if velocity.shape == points.shape:
            speeds = np.linalg.norm(velocity, axis=1)
            velocities = {'available': True, 'maximum_speed_m_s': float(speeds.max()),
                'mean_speed_m_s': float(speeds.mean()), 'particle_count': len(velocity),
                'stationarity_certified': False, 'reason': 'measured_velocity_only_no_new_settling_threshold'}
        elif velocity.size:
            raise ValueError('Native particle velocity count differs from mesh particles.')
    return points, triangles, faces, {'point_count': len(points), 'face_count': len(faces),
        'topology_sha256': _hash(faces.tolist()), 'aabb_m': [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        'centroid_m': points.mean(axis=0).tolist(), 'particle_velocities': velocities}


def native_cloth_contact_paths(cloth, bodies, *, points=None):
    """Query every relevant particle with the SDK's native cloth contact radius.

    The SDK's default contact_list samples keypoints. An AABB broad phase keeps
    all particles that could overlap the requested rigid bodies, then performs
    the same native sphere-overlap query without changing the scene.
    """
    import omnigibson as og
    points = (_array(cloth.root_link.compute_particle_positions()) if points is None else _array(points))
    radius = float(cloth.root_link.cloth_system.particle_contact_offset)
    nearby = np.zeros(len(points), dtype=bool)
    for body in bodies:
        low, high = (_array(value) for value in body.aabb)
        nearby |= np.all(points >= low - radius, axis=1) & np.all(points <= high + radius, axis=1)
    paths = set()
    def hit_contact(hit):
        paths.add(str(hit.rigid_body))
        return True
    for point in points[nearby]:
        og.sim.psqi.overlap_sphere(radius, point.tolist(), hit_contact, False)
    return paths


def _cover(item, cloth, triangles, *, allow_native_bridge=False, full_contact_query=False):
    from omnigibson.object_states import Touching
    from wia.authoring.converters.physical_structure_deformable import _overlap_count
    low_value, high_value = _read_pair(lambda: item.aabb, 'cover.item_aabb_m')
    low, high = [_array(value, 'cover.item_aabb_' + bound + '_m').astype(float)
        for bound, value in (('low', low_value), ('high', high_value))]
    above = triangles[triangles[:, :, 2].max(axis=1) >= high[2] - TRANSLATION_ATOL_M]
    overlap = _overlap_count(above, low, high)
    touching = bool(cloth.states[Touching].get_value(item))
    sampled_touching = touching
    if full_contact_query and not touching:
        from wia.tasks.support_sampling import same_body
        touching = any(same_body(item.prim_path, path) for path in native_cloth_contact_paths(cloth, [item]))
    result = {'native_item_cloth_touching': touching, 'mesh_faces_above_item': len(above),
        'above_item_overlap_faces': overlap, 'mesh_overlaps_item': bool(_overlap_count(triangles, low, high)),
        'accepted': bool(touching and overlap > 0)}
    if full_contact_query:
        result.update(full_particle_contact_query=True,
            full_particle_contact_query_performed=not sampled_touching,
            sampled_item_cloth_touching=sampled_touching)
    if allow_native_bridge:
        from wia.authoring.converters.deformable_redrape import projected_cover
        coverage = projected_cover(triangles, low, high)
        result.update(native_bridge_allowed=True, projected_coverage=coverage,
            accepted=bool(overlap > 0 and coverage['all_sampled_top_rays_covered']))
    return result


def _unavailable(error, *, stage, function):
    return {'schema': REVISION, 'available': False, 'accepted': False,
        'reason': f'{type(error).__name__}: {error}',
        'measurement_function': function, 'measurement_stage': stage, 'exception_type': type(error).__name__,
        'measurement_field': getattr(error, 'measurement_field', None),
        'numeric_field_diagnostic': getattr(error, 'numeric_field_diagnostic', None),
        'traceback': [{'filename': frame.filename, 'line_number': frame.lineno, 'function': frame.name}
            for frame in traceback.extract_tb(error.__traceback__, limit=-6)],
        'pending_native_action_gates': list(PENDING_ACTION_GATES)}


def capture_deformable_source(session, item, cloth, parameters):
    """Capture full native source mesh/physics, before any candidate translation."""
    stage = 'imports'
    try:
        from omnigibson.utils.constants import PrimType
        stage = 'source_reset_context'
        context = _context(session, parameters)
        stage = 'native_cloth_type'
        if cloth.prim_type != PrimType.CLOTH:
            raise ValueError('A real native cloth prim is required; rigid mesh proxies are forbidden.')
        stage = 'source_binding_identity'
        expected = parameters['deformable_cover']['bindings']
        for key, obj in (('item', item), ('cloth', cloth)):
            if any(getattr(obj, field) != value for field, value in expected[key].items()):
                raise ValueError('Source native cloth/item identity differs from the explicit binding.')
        stage = 'native_mesh'
        points, triangles, faces, mesh = _mesh_state(cloth)
        stage = 'native_cover'
        allow_bridge = (parameters.get('cover_revision') == 'full_cover_redrape_v1' and
            bool(parameters.get('protected_extra_keys')))
        cover = _cover(item, cloth, triangles, allow_native_bridge=allow_bridge,
            full_contact_query=parameters.get('cover_revision') == 'full_cover_redrape_v1')
        stage = 'native_item_physics'
        item_row = _object(session, item)
        stage = 'native_cloth_physics'
        cloth_row = _object(session, cloth, cloth=True)
        stage = 'source_group_footprint'
        lows, highs = _array(item_row['aabb_m'][0], 'item.aabb_low_m'), _array(item_row['aabb_m'][1], 'item.aabb_high_m')
        footprint = [np.minimum(lows, points.min(axis=0)).tolist(), np.maximum(highs, points.max(axis=0)).tolist()]
        stage = 'native_clock_and_receipt'
        return {'schema': REVISION, 'available': True, 'accepted': cover['accepted'],
            'context': context, 'owner_thread_id': threading.get_ident(), 'source_time_s': _clock(session),
            'item': item_row, 'cloth': cloth_row, 'points_world_m': points.tolist(), 'faces': faces.tolist(),
            'mesh': mesh, 'source_cover': cover, 'group_bounds_m': footprint,
            'reason': None if cover['accepted'] else 'original_native_mesh_cover_or_contact_missing',
            'pending_native_action_gates': list(PENDING_ACTION_GATES)}
    except Exception as error:
        return _unavailable(error, stage=stage, function='capture_deformable_source')


def verify_initial_mesh_translation(session, cloth, receipt, delta):
    """Require exact world-particle translation BEFORE any native physics step."""
    stage = 'source_receipt_context'
    try:
        context = _receipt_context(session, receipt)
        stage = 'receipt_translation'
        delta = _array(delta, 'receipt.common_translation_m').astype(float)
        if delta.shape != (3,) or cloth.name != receipt['cloth']['name']:
            raise ValueError('Cloth identity or common translation is invalid.')
        stage = 'native_mesh'
        points, _, faces, mesh = _mesh_state(cloth)
        stage = 'expected_translated_source_mesh'
        expected = _array(receipt['points_world_m'], 'receipt.points_world_m') + delta
        shape_matches = points.shape == expected.shape
        # Native poses and particle transforms use float32. At room coordinates
        # tens of metres from the origin, a transform round trip exceeds 10 um.
        # Bound this numerical allowance below 0.1 mm; topology remains exact.
        translation_atol = max(TRANSLATION_ATOL_M, min(1e-4,
            16 * np.finfo(np.float32).eps * max(1., float(np.max(np.abs(expected))))))
        stage = 'initial_translation_checks'
        checks = {'same_native_time_before_physics': abs(_clock(session) - receipt['source_time_s']) <= 1e-9,
            'same_topology': mesh['topology_sha256'] == receipt['mesh']['topology_sha256'] and faces.tolist() == receipt['faces'],
            'exact_world_particle_translation': bool(shape_matches and np.allclose(points, expected, atol=translation_atol, rtol=0))}
        return {'schema': REVISION, 'available': True, 'accepted': all(checks.values()), 'checks': checks,
            'context': context, 'cloth_name': cloth.name, 'translation_m': delta.tolist(),
            'native_time_s': _clock(session), 'topology_sha256': mesh['topology_sha256'],
            'maximum_particle_error_m': float(np.max(np.linalg.norm(points - expected, axis=1))) if shape_matches else None,
            'numerical_tolerance_m': translation_atol,
            'reason': None if all(checks.values()) else 'initial_native_mesh_translation_unverified'}
    except Exception as error:
        return _unavailable(error, stage=stage, function='verify_initial_mesh_translation')


def _same_physics(current, original, *, cloth=False):
    fields = ('name', 'prim_path', 'category', 'model', 'scale', 'mass_kg', 'materials', 'visual_only', 'fixed_base')
    if cloth:
        fields += ('root_scale',)
    return all(current[key] == original[key] for key in fields)


def measure_deformable_placement(session, item, cloth, support, receipt, delta, surface_z):
    """Allow solver deformation while measuring topology, cover and real support.

    Root must store the pre-step result as receipt['initial_mesh_translation'].
    Particle velocities are reported when available; this function never invents
    zero velocity or introduces an unmeasured cloth stationarity certificate.
    """
    stage = 'imports'
    try:
        from omnigibson.object_states import Touching, OnTop
        from omnigibson.utils.constants import PrimType
        stage = 'source_receipt_context'
        context = _receipt_context(session, receipt)
        stage = 'receipt_translation'
        delta = _array(delta, 'receipt.common_translation_m').astype(float)
        stage = 'prephysics_translation_proof'
        initial = receipt.get('initial_mesh_translation', {})
        if (initial.get('accepted') is not True or initial.get('context') != context or
                initial.get('cloth_name') != cloth.name or not np.allclose(initial.get('translation_m', []), delta, atol=TRANSLATION_ATOL_M, rtol=0)):
            raise ValueError('Pre-physics exact world-mesh translation proof is missing or stale.')
        stage = 'native_mesh'
        points, triangles, faces, mesh = _mesh_state(cloth)
        stage = 'native_cover'
        cover = _cover(item, cloth, triangles,
            allow_native_bridge=receipt.get('source_cover', {}).get('native_bridge_allowed') is True,
            full_contact_query=receipt.get('source_cover', {}).get('full_particle_contact_query') is True)
        stage = 'native_item_physics'
        item_row = _object(session, item)
        stage = 'native_cloth_physics'
        cloth_row = _object(session, cloth, cloth=True)
        stage = 'item_orientation_guard'
        before = receipt['item']
        quaternion, original_q = _array(item_row['quaternion_xyzw'], 'item.root_quaternion_xyzw'), _array(before['quaternion_xyzw'], 'receipt.item.root_quaternion_xyzw')
        angle = 2 * np.arccos(np.clip(abs(float(quaternion @ original_q)) / (np.linalg.norm(quaternion) * np.linalg.norm(original_q)), 0., 1.))
        stage = 'selected_support_bounds'
        low_value, high_value = _read_pair(lambda: support.aabb, 'support.aabb_m')
        support_low, support_high = [_array(value, 'support.aabb_' + bound + '_m').astype(float)
            for bound, value in (('low', low_value), ('high', high_value))]
        stage = 'selected_support_surface'
        _array(surface_z, 'support.surface_z_m')
        if not np.isfinite(surface_z):
            raise ValueError('Measured selected support surface is nonfinite.')
        below = points[:, 2] < float(surface_z) - DECK_PENETRATION_TOLERANCE_M
        stage = 'selected_support_contact_state'
        selected_contact = bool(item.states[Touching].get_value(support) and item.states[OnTop].get_value(support))
        native_support_contacts = []
        stage = 'selected_support_native_contacts'
        for contact in item.contact_list():
            bodies = (str(contact.body0), str(contact.body1))
            if not any(body == support.prim_path or body.startswith(support.prim_path + '/') for body in bodies):
                continue
            point, normal = _array(contact.position, 'item.support_contact.position_m'), _array(contact.normal, 'item.support_contact.normal')
            if abs(float(normal[2])) >= .5 and abs(float(point[2]) - surface_z) <= .015:
                native_support_contacts.append(point.tolist())
        stage = 'original_group_bounds'
        low_value, high_value = receipt['group_bounds_m']
        original_low, original_high = [_array(value, 'receipt.group_bounds_' + bound + '_m') + delta
            for bound, value in (('low', low_value), ('high', high_value))]
        stage = 'placement_checks'
        checks = {'actual_native_cloth_type': cloth.prim_type == PrimType.CLOTH,
            'selected_support_collidable': not support.visual_only,
            'same_particle_and_face_connectivity': mesh['point_count'] == receipt['mesh']['point_count'] and
                mesh['topology_sha256'] == receipt['mesh']['topology_sha256'] and faces.tolist() == receipt['faces'],
            'cloth_physics_preserved': _same_physics(cloth_row, receipt['cloth'], cloth=True),
            'item_identity_and_physics_preserved': _same_physics(item_row, before),
            'intended_item_common_translation': bool(np.linalg.norm(_array(item_row['position_m'], 'item.root_position_m') - _array(before['position_m'], 'receipt.item.root_position_m') - delta) <= ITEM_POSITION_TOLERANCE_M and angle <= ITEM_ORIENTATION_TOLERANCE_RAD),
            'full_original_mesh_footprint_fits_selected_support': bool(np.all(original_low[:2] >= support_low[:2] - TRANSLATION_ATOL_M) and
                np.all(original_high[:2] <= support_high[:2] + TRANSLATION_ATOL_M)),
            'no_significant_mesh_particles_below_selected_deck': not bool(below.any()),
            'selected_deck_surface_matches_native_bounds': abs(float(support_high[2]) - surface_z) <= .015,
            'actual_item_selected_support_contact': selected_contact and bool(native_support_contacts),
            'native_contact_and_mesh_cover_preserved': cover['accepted'],
            'original_item_and_cloth_movable_collidable': not any(row[key] for row in (item_row, cloth_row) for key in ('visual_only', 'fixed_base'))}
        accepted = all(checks.values())
        stage = 'final_native_receipt'
        return {'schema': REVISION, 'available': True, 'accepted': accepted, 'context': context,
            'translation_m': delta.tolist(), 'native_time_s': _clock(session), 'checks': checks,
            'native_item': item_row, 'native_cloth': cloth_row, 'native_mesh': mesh, 'native_cover': cover,
            'native_item_support_contact_positions_m': native_support_contacts,
            'below_deck_particle_count': int(below.sum()), 'minimum_particle_z_m': float(points[:, 2].min()),
            'selected_support': {'name': support.name, 'prim_path': support.prim_path,
                'bounds_m': [support_low.tolist(), support_high.tolist()], 'surface_z_m': float(surface_z)},
            'deformed_from_translated_source': not np.allclose(points, _array(receipt['points_world_m'], 'receipt.points_world_m') + delta, atol=TRANSLATION_ATOL_M, rtol=0),
            'cloth_root_pose_is_rigid_invariant': False, 'pending_native_action_gates': list(PENDING_ACTION_GATES),
            'reason': None if accepted else 'native_deformable_support_or_cover_failed'}
    except Exception as error:
        return _unavailable(error, stage=stage, function='measure_deformable_placement')
