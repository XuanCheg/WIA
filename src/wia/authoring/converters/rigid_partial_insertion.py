"""Measure versioned Rigid partial insertion without changing world state.

Run generated tasks with src/wia/runtime/commands/check_esi_core_native.py --manipulation
--evaluate --serve. CPU checks: python -m pytest -q
tests/test_wia_rigid_native_adapter.py tests/test_rigid_insertion_scoring.py.
Native container volume meshes and real target surface witnesses replace the
centre-only Inside predicate; missing geometric evidence remains unavailable.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from wia.authoring.io import write_json
from wia.tasks.interaction_setup import array
from wia.tasks.rigid_insertion_scoring import RigidInsertionProgress

MODULE = 'wia.authoring.converters.rigid_partial_insertion'
REVISION = 'wia_rigid_partial_insertion_v1'
_EMPTY_PAIR_CONTACT_REASON = 'ValueError: Native contact has no pair-specific separation samples.'
_RETAINED_CONTACT_SOURCE = 'retained_native_contact_for_unchanged_sleeping_pair'


@dataclass(frozen=True)
class _NativeContactWitness:
    episode: int
    native_time_s: float
    geometry_stamp: str
    sleeping_flags: str
    contact_geometry: str


def _native_bool(value, label):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(label + ' must be a measured native boolean.')
    return bool(value)


def _native_identity(prim):
    path = str(prim.prim_path)
    if not path.startswith('/'):
        raise ValueError('Native prim identity must be an absolute prim path.')
    return {'prim_path': path, 'wrapper_identity': id(prim),
            'class': type(prim).__module__ + '.' + type(prim).__qualname__}


def _native_pose(prim):
    position, quaternion = (array(value) for value in prim.get_position_orientation())
    if position.shape != (3,) or quaternion.shape != (4,) or np.linalg.norm(quaternion) == 0:
        raise ValueError('Exact contact geometry requires a finite native world pose.')
    # No normalization or quaternion-sign equivalence: equality is exact.
    return {'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist()}


def _native_scale(prim):
    try:
        value = prim.scale
    except AttributeError:
        if hasattr(type(prim), 'scale') or 'scale' in getattr(prim, '__dict__', {}):
            raise
        return {'exposed': False}
    scale = array(value)
    if scale.shape != (3,) or np.any(scale <= 0):
        raise ValueError('Exact contact geometry requires positive finite native scales.')
    return {'exposed': True, 'value': scale.tolist()}


def _shape_array_stamp(value, *, points=False):
    if value is None:
        raise ValueError('Authored native collision shape data is unavailable.')
    values = np.asarray(value)
    if points:
        if values.ndim != 2 or values.shape[1] != 3 or not len(values) or values.dtype.kind not in 'fiu':
            raise ValueError('Native collision mesh points must be numeric nonempty N x 3 values.')
    elif values.ndim != 1 or not len(values) or values.dtype.kind not in 'iu':
        raise ValueError('Native collision topology must be a nonempty integer array.')
    if not np.isfinite(values).all():
        raise ValueError('Native collision shape measurements must be finite.')
    values = np.ascontiguousarray(values)
    return {'shape': list(values.shape), 'dtype': values.dtype.str,
            'sha256': hashlib.sha256(values.tobytes()).hexdigest()}


def _native_shape_stamp(mesh):
    """Read authored USD values rather than GeomPrim's cached points/faces."""
    prim = mesh.prim
    kind = str(prim.GetTypeName())
    if kind == 'Mesh':
        points = prim.GetAttribute('points').Get()
        counts = prim.GetAttribute('faceVertexCounts').Get()
        indices = prim.GetAttribute('faceVertexIndices').Get()
        shape = {'type': kind, 'points': _shape_array_stamp(points, points=True),
                 'faceVertexCounts': _shape_array_stamp(counts),
                 'faceVertexIndices': _shape_array_stamp(indices)}
        if (np.any(np.asarray(counts) < 3) or int(np.sum(counts)) != len(indices)
                or np.any(np.asarray(indices) < 0) or np.any(np.asarray(indices) >= len(points))):
            raise ValueError('Native collision mesh topology is inconsistent with its points.')
        approximation = mesh.get_collision_approximation()
        if not isinstance(approximation, str) or not approximation:
            raise ValueError('Native collision mesh approximation is unavailable.')
        shape['collision_approximation'] = approximation
        return shape
    attributes = {'Sphere': ('radius',), 'Cube': ('size',),
                  'Cylinder': ('radius', 'height', 'axis'), 'Cone': ('radius', 'height', 'axis'),
                  'Capsule': ('radius', 'height', 'axis')}.get(kind)
    if attributes is None:
        raise ValueError('Unsupported native collision shape for exact retention: ' + kind)
    shape = {'type': kind}
    for name in attributes:
        value = prim.GetAttribute(name).Get()
        if name == 'axis':
            if value not in ('X', 'Y', 'Z'):
                raise ValueError('Native primitive axis is unavailable.')
            shape[name] = str(value)
        else:
            if isinstance(value, (bool, np.bool_)) or value is None:
                raise ValueError('Native primitive dimensions are unavailable.')
            dimension = float(value)
            if not np.isfinite(dimension) or dimension <= 0:
                raise ValueError('Native primitive dimensions must be positive and finite.')
            shape[name] = dimension
    return shape


def _pair_geometry_stamp(target, container):
    """Capture immutable exact geometry plus separately measured sleep flags."""
    context = {'stage': 'import_dynamic_link_class'}
    try:
        from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
        objects, sleeping = [], []
        for role, obj in (('target', target), ('container', container)):
            context = {'stage': 'object_links', 'object_role': role,
                       'object_class': type(obj).__module__ + '.' + type(obj).__qualname__}
            links = []
            for name, link in sorted(obj.links.items()):
                context.pop('mesh_name', None)
                context.pop('mesh_class', None)
                context.update(stage='link_meta_flag', link_name=name,
                    link_class=type(link).__module__ + '.' + type(link).__qualname__)
                if _native_bool(link.is_meta_link, 'Native meta-link flag'):
                    continue
                context['stage'] = 'link_collision_meshes'
                meshes = link.collision_meshes
                if not meshes:
                    continue
                context['stage'] = 'link_identity'
                identity = _native_identity(link)
                dynamic = isinstance(link, RigidDynamicPrim)
                if dynamic:
                    context['stage'] = 'link_dynamic_flags'
                    dynamic = (not _native_bool(link.get_attribute('physics:kinematicEnabled'), 'Native kinematic flag')
                               and _native_bool(link.get_attribute('physics:rigidBodyEnabled'), 'Native rigid-body flag'))
                context['stage'] = 'link_sleeping_flag'
                asleep = _native_bool(link.is_asleep, 'Native sleeping flag') if dynamic else None
                sleeping.append({'object_prim_path': str(obj.prim_path), 'link_prim_path': str(link.prim_path),
                                 'dynamic': dynamic, 'is_asleep': asleep})
                enabled_meshes = []
                for mesh_name, mesh in sorted(meshes.items()):
                    context.update(stage='mesh_collision_enabled', mesh_name=mesh_name,
                        mesh_class=type(mesh).__module__ + '.' + type(mesh).__qualname__)
                    enabled = _native_bool(mesh.collision_enabled, 'Native collision-enabled flag')
                    context['stage'] = 'mesh_identity'
                    entry = {'name': mesh_name, **_native_identity(mesh), 'enabled': enabled}
                    if enabled:
                        context['stage'] = 'mesh_world_pose'
                        pose = _native_pose(mesh)
                        context['stage'] = 'mesh_scale'
                        scale = _native_scale(mesh)
                        context['stage'] = 'mesh_authored_shape'
                        entry.update(world_pose=pose, scale=scale, shape=_native_shape_stamp(mesh))
                    enabled_meshes.append(entry)
                context['stage'] = 'link_world_pose'
                pose = _native_pose(link)
                context['stage'] = 'link_scale'
                links.append({'name': name, **identity, 'world_pose': pose,
                              'scale': _native_scale(link), 'collision_meshes': enabled_meshes})
            context['stage'] = 'object_collision_links_validation'
            if not links:
                raise ValueError('Each object needs measured non-meta collision-bearing links.')
            if not any(mesh['enabled'] for link in links for mesh in link['collision_meshes']):
                raise ValueError('Each object needs at least one measured enabled collision mesh.')
            context['stage'] = 'object_identity'
            identity = _native_identity(obj)
            context['stage'] = 'object_world_pose'
            pose = _native_pose(obj)
            context['stage'] = 'object_scale'
            objects.append({**identity, 'world_pose': pose, 'scale': _native_scale(obj), 'links': links})
        context['stage'] = 'geometry_serialization'
        return {'available': True, 'geometry_stamp': json.dumps(objects, sort_keys=True, allow_nan=False),
                'sleeping_flags': sleeping, 'all_dynamic': all(item['dynamic'] for item in sleeping),
                'all_asleep': all(item['is_asleep'] is True for item in sleeping)}
    except (AttributeError, ImportError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as error:
        return {'available': False, 'reason': f'{type(error).__name__}: {error}',
                'diagnostic_context': context}


def _under(path, parent):
    return str(path) == str(parent) or str(path).startswith(str(parent) + '/')


def _contact_bodies(contact):
    bodies = tuple(getattr(contact, key) for key in ('body0', 'body1'))
    if any(body is None or not str(body).startswith('/') for body in bodies):
        raise ValueError('Native contact body identities are unavailable.')
    return tuple(str(body) for body in bodies)


def _bounds_gap(first, second):
    lo, hi = (array(v) for v in first)
    other_lo, other_hi = (array(v) for v in second)
    if (any(value.shape != (3,) for value in (lo, hi, other_lo, other_hi))
            or np.any(hi < lo) or np.any(other_hi < other_lo)):
        raise ValueError('Native bounding boxes must have ordered finite 3D corners.')
    return float(np.linalg.norm(np.maximum(np.maximum(lo - other_hi, other_lo - hi), 0.)))


def _box_gap(first, second):
    return _bounds_gap(first.aabb, second.aabb)


def _aggregate_grasp(robot, controller):
    """Read every arm; the public working hand must be open and all hands empty.

    Inactive finger apertures are diagnostic: the public controller cannot open
    those grippers, so their measured half-open initial posture is not a failure.
    Native attachment, pending release and target contact remain required for
    every arm independently of aperture.
    """
    arms, errors = {}, []
    try:
        arm_names = tuple(robot.arm_names)
        working_arm = controller.working_arm
        if not arm_names or working_arm not in arm_names:
            raise ValueError('The working arm must belong to the native robot arm list.')
        positions = array(robot.get_joint_positions())
        lower, upper = (array(value) for value in robot.control_limits['position'])
        tolerance = float(controller.gripper_config.joint_tolerance_m)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('Measured gripper readback tolerance is unavailable.')
    except (AttributeError, ValueError, RuntimeError, KeyError, TypeError) as error:
        return {'available': False, 'holding': None, 'object_name': None,
                'constraint_active': None, 'release_pending': None, 'gripper_open': None,
                'arms': arms, 'measurement_errors': [f'{type(error).__name__}: {error}']}
    for arm in arm_names:
        try:
            obj = robot._ag_obj_in_hand[arm]
            name = obj.name if obj is not None else None
            constraint = robot._ag_obj_constraints[arm] is not None
            pending = robot._ag_release_counter[arm] is not None
            reading = int(robot.is_grasping(arm))
            holding = {-1: False, 1: True}.get(reading)
            if holding is None:
                raise ValueError('Native is_grasping reports an unknown state.')
            dofs = array(robot.controllers[f'gripper_{arm}'].dof_idx).astype(int)
            fingers = robot.finger_joint_names[arm]
            if (len(dofs) != 2 or len(fingers) != 2 or any('prismatic' not in
                    str(robot.joints[name].joint_type).lower() for name in fingers)):
                raise ValueError('Both native grippers require two measured prismatic finger joints.')
            measured, closed, opened = positions[dofs], lower[dofs], upper[dofs]
            if np.any(opened <= closed):
                raise ValueError('Native open gripper limits must exceed closed limits.')
            if name is not None and (not isinstance(name, str) or not name):
                raise ValueError('Native held-object identity is unavailable.')
            if holding and name is None:
                raise ValueError('Native grasp has no measured held-object identity.')
            if constraint and (name is None or not holding):
                raise ValueError('Native attachment and grasp identity are inconsistent.')
            if name is not None and not holding and not pending:
                raise ValueError('Native held-object identity has no grasp or release transition.')
            arms[arm] = {'available': True, 'object_name': name, 'holding': holding,
                         'constraint_active': constraint, 'release_pending': pending,
                         'gripper_open': bool(np.max(np.abs(measured - opened)) <= tolerance),
                         'joint_positions_m': measured.tolist(), 'open_joint_limits_m': opened.tolist(),
                         'closed_joint_limits_m': closed.tolist(), 'is_grasping_state': reading}
        except (AttributeError, ValueError, RuntimeError, KeyError, TypeError, IndexError) as error:
            reason = f'{type(error).__name__}: {error}'
            arms[arm] = {'available': False, 'reason': reason}
            errors.append(f'{arm}: {reason}')
    if errors:
        return {'available': False, 'holding': None, 'object_name': None,
                'constraint_active': None, 'release_pending': None, 'gripper_open': None,
                'working_arm': working_arm, 'arms': arms, 'measurement_errors': errors}
    identities = {row['object_name'] for row in arms.values() if row['object_name'] is not None}
    working = arms[working_arm]
    holding = any(row['holding'] for row in arms.values())
    constraint = any(row['constraint_active'] for row in arms.values())
    if len(identities) > 1:
        errors.append('Multiple native held identities cannot fit one working-object sample.')
    if (holding or constraint) and working['object_name'] is None:
        errors.append('A nonworking-arm attachment cannot supply working-arm pickup evidence.')
    return {'available': not errors, 'object_name': working['object_name'],
            'holding': holding, 'constraint_active': constraint,
            'release_pending': any(row['release_pending'] for row in arms.values()),
            'gripper_open': working['gripper_open'],
            'all_grippers_open': all(row['gripper_open'] for row in arms.values()),
            'working_arm': working_arm, 'arms': arms, 'measurement_errors': errors}


def _surface_points(target):
    """Use actual native collision vertices and triangle centres, not box corners."""
    import torch
    points = []
    for link in target.links.values():
        if link.is_meta_link:
            continue
        for mesh in link.collision_meshes.values():
            if not mesh.collision_enabled:
                continue
            local = mesh.points
            if local is not None and len(local):
                points.append(mesh.transform_local_points_to_world(local))
            faces = mesh.faces
            if faces is not None and len(faces):
                points.append(mesh.transform_local_points_to_world(mesh.mesh_face_centroids))
    if not points:
        raise ValueError('Native target collision surface points are unavailable.')
    result = torch.cat(points, dim=0)
    if result.ndim != 2 or result.shape[1] != 3 or not torch.isfinite(result).all():
        raise ValueError('Native target surface samples must be finite 3D world points.')
    return result


def partial_interior_evidence(target, container, margin):
    """Find an inset target-surface witness in a real container cavity.

    Positive evidence uses container metadata meshes independently of Inside,
    then excludes the container's enabled collision solids. A negative finite
    surface sample is inconclusive unless the complete target AABB is disjoint
    from every volume AABB; this avoids certifying impossibility from sampling.
    """
    import torch
    from omnigibson.macros import macros
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError('The native interior margin must be finite and positive.')
    types = macros.object_states.contains.CONTAINER_META_LINK_TYPES
    volumes = [link for link in container.links.values()
               if link.is_meta_link and link.meta_link_type in types and link.visual_meshes]
    if not volumes:
        return {'interior_valid': None, 'partial_inside': None,
                'reason': 'Native container interior volume meshes are unavailable.'}
    points = _surface_points(target)
    offsets = torch.tensor([[0., 0., 0.], [margin, 0., 0.], [-margin, 0., 0.],
                            [0., margin, 0.], [0., -margin, 0.],
                            [0., 0., margin], [0., 0., -margin]],
                           dtype=points.dtype, device=points.device)
    query = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    witnesses = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    for volume in volumes:
        inside = volume.check_points_in_volume(query)
        witnesses |= inside.reshape(len(points), len(offsets)).all(dim=1)
    solids = [mesh for link in container.links.values() if not link.is_meta_link
              for mesh in link.collision_meshes.values() if mesh.collision_enabled]
    if not solids:
        return {'interior_valid': None, 'partial_inside': None,
                'reason': 'Container collision solids are unavailable.'}
    in_solid = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    for mesh in solids:
        in_solid |= mesh.check_points_in_volume(points)
    witnesses &= ~in_solid
    indices = torch.where(witnesses)[0]
    # Native meta links often have no collision meshes: RigidPrim.aabb then
    # degenerates to the link position. Their volume uses visual meshes, whose
    # actual native visual_aabb is the corresponding conservative bound.
    disjoint = bool(not len(indices) and
                   all(_bounds_gap(target.aabb, volume.visual_aabb) > 0. for volume in volumes))
    return {'interior_valid': True, 'partial_inside': True if len(indices) else False if disjoint else None,
            'method': 'native_surface_in_inset_container_volume_excluding_collision_solids',
            'surface_sample_count': len(points), 'positive_witness_count': len(indices),
            'witness_world_m': array(points[indices[:8]]).tolist(),
            'interior_margin_m': margin, 'volume_prim_paths': [str(v.prim_path) for v in volumes],
            'reason': None if len(indices) or disjoint else 'Finite sampling found no conclusive interior witness.'}


def qualify_operation(session, controller, config, bound):
    """Qualify declared per-object supports with current native contact and rays."""
    from omnigibson.object_states import Touching
    from omnigibson.utils.sampling_utils import raytest
    from wia.tasks.scene_only import local_support_surface
    p, checks, rows = config['parameters'], {}, {}
    errors = []
    try:
        required = tuple(dict.fromkeys([*p['targets'], *p['containers']]))
        supports = p['initial_supports']
        for key in ('maximum_initial_linear_speed_m_s', 'maximum_initial_angular_speed_rad_s',
                    'initial_support_tolerance_m'):
            if isinstance(p[key], bool) or not np.isfinite(p[key]) or p[key] < 0:
                raise ValueError(f'{key} must be an explicit finite nonnegative threshold.')
        tolerance = p['initial_support_tolerance_m']
        if tolerance == 0:
            raise ValueError('initial_support_tolerance_m must be positive for native support rays.')
        checks.update(explicit_revision=p.get('scorer_version') == REVISION,
                      bound_objects_exist=all(obj is not None for obj in bound.values()),
                      declared_supports_complete=set(supports) == set(required) and
                          all(value in bound and bound[value] is not None for value in supports.values()),
                      target_dynamic_collidable=all(bound.get(key) is not None and
                          not bound[key].fixed_base and not bound[key].visual_only for key in p['targets']))
        grasp = _aggregate_grasp(session.robot, getattr(controller, 'controller', controller))
        checks['initial_hands_empty_released'] = bool(grasp['available'] and
            grasp['holding'] is False and grasp['object_name'] is None and
            grasp['constraint_active'] is False and grasp['release_pending'] is False and
            grasp['gripper_open'] is True)
        for key in required:
            row = {'available': False, 'passed': False}
            try:
                obj, support_key = bound[key], supports[key]
                support = bound[support_key]
                low = array(obj.aabb[0])
                native_touching = obj.states[Touching].get_value(support)
                if native_touching is None:
                    raise ValueError('Native support Touching evidence is unavailable.')
                touching = bool(native_touching)
                contacts = list(obj.contact_list())
                robot_contact = any(_under(body, session.robot.prim_path)
                    for contact in contacts for body in _contact_bodies(contact))
                linear = float(np.linalg.norm(array(obj.get_linear_velocity())))
                angular = float(np.linalg.norm(array(obj.get_angular_velocity())))
                witnesses, unavailable_witnesses = [], []
                for contact in contacts:
                    bodies = _contact_bodies(contact)
                    if not (any(_under(body, obj.prim_path) for body in bodies) and
                            any(_under(body, support.prim_path) for body in bodies)):
                        continue
                    try:
                        position, normal = array(contact.position), array(contact.normal)
                        if position.shape != (3,) or normal.shape != (3,):
                            raise ValueError('Native support contacts require 3D position and normal.')
                        if abs(float(normal[2])) < .5:
                            continue
                        start, end = position.copy(), position.copy()
                        start[2] += tolerance
                        end[2] -= tolerance
                        hits = raytest(start.tolist(), end.tolist(), only_closest=False)
                        if not isinstance(hits, (list, tuple)):
                            raise TypeError('Native support ray must return an all-hit list.')
                        for hit in hits:
                            if hit.get('hit') and _under(hit['rigidBody'], support.prim_path):
                                if (array(hit['position']).shape != (3,) or array(hit['normal']).shape != (3,)
                                        or not np.isfinite(float(hit['distance']))):
                                    raise ValueError('Native support ray measurements must be finite.')
                        try:
                            hit = local_support_surface(str(support.prim_path), hits)
                        except ValueError:
                            continue
                        surface = array(hit['position'])
                        surface_normal = array(hit['normal'])
                        if surface.shape != (3,) or surface_normal.shape != (3,):
                            raise ValueError('Native support ray requires finite 3D position and normal.')
                        bottom_gap = float(low[2] - surface[2])
                        contact_gap = float(position[2] - surface[2])
                        if abs(bottom_gap) <= tolerance and abs(contact_gap) <= tolerance:
                            witnesses.append({'contact_position_m': position.tolist(),
                                'contact_normal': normal.tolist(), 'surface_position_m': surface.tolist(),
                                'surface_normal': surface_normal.tolist(), 'support_body': str(hit['rigidBody']),
                                'bottom_surface_offset_m': bottom_gap, 'contact_surface_offset_m': contact_gap})
                    except (AttributeError, ValueError, RuntimeError, KeyError, TypeError) as error:
                        unavailable_witnesses.append(f'{type(error).__name__}: {error}')
                available = bool(witnesses or not unavailable_witnesses)
                supported = bool(touching and witnesses)
                row.update(available=available, name=obj.name, support_binding=support_key,
                           support_name=support.name, native_touching=touching,
                           upward_support_contact=supported if available else None,
                           linear_speed_m_s=linear, angular_speed_rad_s=angular,
                           robot_contact=robot_contact, support_witnesses=witnesses,
                           measurement_errors=unavailable_witnesses,
                           passed=bool(available and supported and not robot_contact and
                               not obj.visual_only and not support.visual_only and
                               linear <= p['maximum_initial_linear_speed_m_s'] and
                               angular <= p['maximum_initial_angular_speed_rad_s']))
            except (AttributeError, ValueError, RuntimeError, KeyError, TypeError) as error:
                row['measurement_errors'] = [f'{type(error).__name__}: {error}']
            rows[key] = row
        checks['initial_native_support_and_stability'] = all(row['passed'] for row in rows.values())
        available = all(row['available'] for row in rows.values()) and grasp['available']
        accepted = bool(available and all(checks.values()))
        return {'accepted': accepted, 'checks': checks,
                'evaluation_validity': 'complete' if available else 'unavailable',
                'initial_supports': rows, 'initial_grasp': grasp,
                'reason': None if accepted else 'native_initial_support_or_stability_failed',
                'complete_physical_execution_verified': False,
                'interior_geometry_status': 'measured_per_physics_sample'}
    except (AttributeError, ValueError, RuntimeError, KeyError, TypeError) as error:
        errors.append(f'{type(error).__name__}: {error}')
        return {'accepted': False, 'checks': checks, 'evaluation_validity': 'unavailable',
                'initial_supports': rows, 'measurement_errors': errors,
                'reason': 'native_initial_support_measurement_unavailable',
                'complete_physical_execution_verified': False,
                'interior_geometry_status': 'measured_per_physics_sample'}


class RigidPartialInsertionEvaluator:
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.p, self.output = config['parameters'], Path(output)
        self.objects = {key: session.env.scene.object_registry('name', name)
                        for key, name in config['bindings'].items()}
        self.targets = {key: config['bindings'][key] for key in self.p['targets']}
        # These thresholds are part of the declared task, never inherited from
        # an unrelated experiment or a hardcoded quaternion-dot heuristic.
        for key in ('maximum_container_rotation_rad', 'scale_relative_tolerance',
                    'container_pose_tolerance_m'):
            value = self.p[key]
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be an explicit finite nonnegative threshold.')
        if self.p['maximum_container_rotation_rad'] > np.pi:
            raise ValueError('maximum_container_rotation_rad must not exceed pi radians.')
        self.scorer = RigidInsertionProgress(self.targets, self.p['containers'],
            **{key: self.p[key] for key in ('minimum_lift_m', 'stability_duration_s',
                'maximum_linear_speed_m_s', 'maximum_angular_speed_rad_s', 'max_sample_gap_s')},
            expected_containers=self.p['expected_containers'],
            manipulation_revision=self.p.get('manipulation_revision'))
        self.active = False
        self.answer, self.initial, self.measurement_error = {}, {}, None
        self.initial_qualification = None
        self.last_sample = None
        self._contact_witnesses = {}
        self._contact_witness_episode = 0
        self._contact_witness_time_s = None
        self._contact_retention_diagnostics = {}

    def begin_episode(self, observation):
        self._contact_witnesses.clear()
        self._contact_witness_episode += 1
        self._contact_witness_time_s = None
        self._contact_retention_diagnostics.clear()
        self.initial_qualification = qualify_operation(
            self.session, self.controller, self.config, self.objects)
        self.initial = {}
        for key in dict.fromkeys([*self.p['targets'], *self.p['containers']]):
            try:
                obj = self.objects[key]
                scale = array(obj.scale)
                if scale.shape != (3,) or np.any(scale <= 0):
                    raise ValueError('Native object scale must have three positive components.')
                reference = {'scale': scale.tolist()}
                if key in self.p['containers']:
                    position, quaternion = (array(v) for v in obj.get_position_orientation())
                    if position.shape != (3,) or quaternion.shape != (4,) or np.linalg.norm(quaternion) == 0:
                        raise ValueError('Native container pose must contain a position and nonzero quaternion.')
                    reference.update(position=position.tolist(), quaternion=quaternion.tolist())
                self.initial[key] = reference
            except (AttributeError, ValueError, RuntimeError, KeyError, TypeError) as error:
                self.initial[key] = {'measurement_error': f'{type(error).__name__}: {error}'}
        try:
            sample = self._measure()
            self.measurement_error = None
        except (AttributeError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as error:
            self.measurement_error = f'{type(error).__name__}: {error}'
            sample = self._unavailable_sample(self.measurement_error)
        self.scorer.begin(sample)
        self.last_sample = sample
        self.active = True
        return self.record(observation)

    def _protected_geometry(self):
        """Read initial-relative scales and container poses without any setters."""
        rows, errors, violations = {}, [], []
        for key in dict.fromkeys([*self.p['targets'], *self.p['containers']]):
            row = {}
            try:
                initial, obj = self.initial[key], self.objects[key]
                if initial.get('measurement_error'):
                    raise ValueError('Initial reference is unavailable: ' + initial['measurement_error'])
                baseline, current = array(initial['scale']), array(obj.scale)
                if baseline.shape != (3,) or current.shape != (3,) or np.any(baseline <= 0) or np.any(current <= 0):
                    raise ValueError('Native scale comparison requires positive three-component scales.')
                relative = np.abs(current / baseline - 1.)
                scale_valid = bool(np.all(relative <= self.p['scale_relative_tolerance']))
                row.update(baseline_scale=baseline.tolist(), current_scale=current.tolist(),
                           scale_relative_change=relative.tolist(), scale_unchanged=scale_valid)
                if not scale_valid:
                    violations.append(key + ':scale_changed')
                if key in self.p['containers']:
                    position, quaternion = (array(v) for v in obj.get_position_orientation())
                    reference_position, reference_quaternion = array(initial['position']), array(initial['quaternion'])
                    if position.shape != (3,) or quaternion.shape != (4,) or np.linalg.norm(quaternion) == 0:
                        raise ValueError('Native container pose is unavailable.')
                    quaternion = quaternion / np.linalg.norm(quaternion)
                    reference_quaternion = reference_quaternion / np.linalg.norm(reference_quaternion)
                    translation = float(np.linalg.norm(position - reference_position))
                    rotation = float(2. * np.arccos(np.clip(abs(np.dot(quaternion, reference_quaternion)), 0., 1.)))
                    pose_valid = bool(translation <= self.p['container_pose_tolerance_m'] and
                                      rotation <= self.p['maximum_container_rotation_rad'])
                    row.update(position_change_m=translation, rotation_change_rad=rotation,
                               pose_unchanged=pose_valid)
                    if not pose_valid:
                        violations.append(key + ':container_pose_changed')
                row['available'] = True
            except (AttributeError, ValueError, RuntimeError, KeyError, TypeError) as error:
                reason = f'{type(error).__name__}: {error}'
                row.update(available=False, reason=reason)
                errors.append(key + ': ' + reason)
            rows[key] = row
        return {'available': not errors, 'unchanged': False if violations else None if errors else True,
                'objects': rows, 'violations': violations, 'measurement_errors': errors,
                'maximum_container_rotation_rad': self.p['maximum_container_rotation_rad'],
                'scale_relative_tolerance': self.p['scale_relative_tolerance'],
                'container_pose_tolerance_m': self.p['container_pose_tolerance_m']}

    def _contact_failure_diagnostic(self, pair, reason, error, native_time_s, previous):
        """Describe an existing discard path without changing the witness cache."""
        diagnostic = self._contact_retention_diagnostics.setdefault(pair, {
            'native_samples_stored': 0, 'retained_samples': 0, 'refusal_count': 0,
            'discard_count': 0, 'refusal_reasons': {}})
        diagnostic['refusal_count'] += 1
        diagnostic['refusal_reasons'][reason] = diagnostic['refusal_reasons'].get(reason, 0) + 1
        diagnostic.update(last_reason=reason, last_error=error,
            previous_witness_present=previous is not None,
            previous_native_sample_time_s=previous.native_time_s if previous else None,
            current_native_time_s=native_time_s, current_stamp_attempted=False,
            current_stamp_available=None, current_stamp_error=None, stamp_failure_context=None,
            all_dynamic=None, all_asleep=None, exact_geometry_equal=None)
        if previous is not None:
            diagnostic['discard_count'] += 1
            diagnostic['last_discard_reason'] = reason
        return diagnostic

    def _unavailable_sample(self, reason):
        previous = dict(self._contact_witnesses)
        self._contact_witnesses.clear()
        try:
            now = float(self.session.pipeline.og.sim.current_time)
            now = now if np.isfinite(now) else None
        except (AttributeError, ValueError, RuntimeError, TypeError):
            now = None
        for pair in ((key, container) for key in self.p['targets'] for container in self.p['containers']):
            self._contact_failure_diagnostic(pair, 'native_measurement_unavailable', reason, now, previous.get(pair))
        sample = {'native_time_s': now, 'measurement_errors': [reason]}
        # A failed geometry read must not erase an independently observed
        # prohibited setter or scale/pose mutation from the reducer's latch.
        try:
            protection = self._protected_geometry()
            sample.update(protected_geometry=protection,
                          control_valid=False if protection['unchanged'] is False else None)
            control = self.controller.geometric_rearrangement_evidence()
            setter = control.get('state_setter_used')
            public_valid = (not setter if control.get('available') is True and type(setter) is bool else None)
            controls = (public_valid, protection['unchanged'])
            sample.update(control_evidence=control,
                          control_valid=False if any(value is False for value in controls) else
                          None if any(value is None for value in controls) else True)
        except (AttributeError, ValueError, RuntimeError, KeyError, TypeError) as error:
            sample['measurement_errors'].append(f'{type(error).__name__}: {error}')
        return sample

    def _observe_contact_time(self, native_time_s):
        try:
            if isinstance(native_time_s, (bool, np.bool_)) or native_time_s is None:
                raise ValueError('Native contact sample time is unavailable.')
            now = float(native_time_s)
            if not np.isfinite(now) or (self._contact_witness_time_s is not None and
                                       now < self._contact_witness_time_s):
                raise ValueError('Native contact sample time is unavailable or regressed.')
            self._contact_witness_time_s = now
            return True
        except (ValueError, TypeError, OverflowError):
            self._contact_witnesses.clear()
            return False

    def _contact_geometry_with_sleeping_retention(self, target_key, container_key, touching,
                                                  current, native_time_s, protection):
        """Retain actual separation only for an exactly unchanged sleeping pair."""
        pair = (target_key, container_key)
        previous_at_entry = self._contact_witnesses.get(pair)
        diagnostic = self._contact_retention_diagnostics.setdefault(pair, {
            'native_samples_stored': 0, 'retained_samples': 0, 'refusal_count': 0,
            'discard_count': 0, 'refusal_reasons': {}})
        try:
            current_time = float(native_time_s)
            current_time = current_time if np.isfinite(current_time) else None
        except (ValueError, TypeError, OverflowError):
            current_time = None
        diagnostic.update(previous_witness_present=previous_at_entry is not None,
            previous_native_sample_time_s=previous_at_entry.native_time_s if previous_at_entry else None,
            current_native_time_s=current_time, current_stamp_attempted=False,
            current_stamp_available=None, current_stamp_error=None, stamp_failure_context=None,
            all_dynamic=None, all_asleep=None, exact_geometry_equal=None,
            current_contact_reason=current.get('reason'), touching=touching,
            control_protection_available=protection.get('available'),
            control_protection_unchanged=protection.get('unchanged'))

        def finish(reason, result=current, *, discarded=False, error=None):
            diagnostic.update(last_reason=reason, last_error=error)
            if reason == 'native_sample_stored':
                diagnostic['native_samples_stored'] += 1
            elif reason == 'retained_sample':
                diagnostic['retained_samples'] += 1
            else:
                diagnostic['refusal_count'] += 1
                reasons = diagnostic['refusal_reasons']
                reasons[reason] = reasons.get(reason, 0) + 1
            if discarded:
                diagnostic['discard_count'] += 1
                diagnostic['last_discard_reason'] = reason
            if result.get('available') is False or result.get('separation_source') == _RETAINED_CONTACT_SOURCE:
                result['retention_diagnostics'] = deepcopy(diagnostic)
            return result

        if not self._observe_contact_time(native_time_s):
            return finish('sample_time_unavailable_or_regressed', discarded=previous_at_entry is not None)
        now = float(native_time_s)
        if (touching is not True or protection.get('available') is not True
                or protection.get('unchanged') is not True):
            self._contact_witnesses.pop(pair, None)
            reason = ('native_touching_not_true' if touching is not True else
                      'control_protection_unavailable' if protection.get('available') is not True else
                      'control_protection_changed')
            return finish(reason, discarded=previous_at_entry is not None)
        native_sample = (current.get('available') is True and
                         current.get('separation_source') == 'native_physx_contact_data')
        empty_native_sample = (current.get('available') is False and
                               current.get('reason') == _EMPTY_PAIR_CONTACT_REASON)
        if not native_sample and not empty_native_sample:
            self._contact_witnesses.pop(pair, None)
            return finish('current_contact_not_native_or_empty', discarded=previous_at_entry is not None)
        previous = self._contact_witnesses.get(pair)
        if empty_native_sample and previous is None:
            return finish('no_previous_native_sample')
        measured = _pair_geometry_stamp(self.objects[target_key], self.objects[container_key])
        diagnostic.update(current_stamp_attempted=True,
            current_stamp_available=measured.get('available') is True,
            current_stamp_error=measured.get('reason'),
            stamp_failure_context=deepcopy(measured.get('diagnostic_context')),
            all_dynamic=measured.get('all_dynamic'), all_asleep=measured.get('all_asleep'),
            exact_geometry_equal=(measured['geometry_stamp'] == previous.geometry_stamp)
                if measured.get('available') is True and previous is not None else None)
        if measured.get('available') is not True or measured.get('all_dynamic') is not True:
            self._contact_witnesses.pop(pair, None)
            return finish('stamp_read_failure' if measured.get('available') is not True else 'pair_not_dynamic',
                discarded=previous is not None)
        if native_sample:
            try:
                separations = tuple(float(value) for value in current['separations_m'])
                if not separations or not all(np.isfinite(value) for value in separations):
                    raise ValueError('A retained contact witness needs nonempty finite native separations.')
                if type(current.get('penetrating')) is not bool:
                    raise ValueError('Native penetration verdict is unavailable.')
                serialized = json.dumps(current, sort_keys=True, allow_nan=False)
                self._contact_witnesses[pair] = _NativeContactWitness(
                    self._contact_witness_episode, now, measured['geometry_stamp'],
                    json.dumps(measured['sleeping_flags'], sort_keys=True, allow_nan=False), serialized)
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                self._contact_witnesses.pop(pair, None)
                return finish('invalid_native_separation_witness', discarded=previous is not None,
                    error=f'{type(error).__name__}: {error}')
            return finish('native_sample_stored')
        if (previous.episode != self._contact_witness_episode or now < previous.native_time_s
                or measured['all_asleep'] is not True
                or measured['geometry_stamp'] != previous.geometry_stamp):
            self._contact_witnesses.pop(pair, None)
            reason = ('previous_episode_mismatch' if previous.episode != self._contact_witness_episode else
                      'sample_time_before_native_witness' if now < previous.native_time_s else
                      'current_pair_awake' if measured['all_asleep'] is not True else 'exact_geometry_changed')
            return finish(reason, discarded=True)
        original = json.loads(previous.contact_geometry)
        return finish('retained_sample', {**original, 'separation_source': _RETAINED_CONTACT_SOURCE,
                'retention_schema': 'wia_rigid_sleeping_contact_retention_v1',
                'native_sample_time_s': previous.native_time_s, 'current_native_time_s': now,
                'original_native_separations_m': original['separations_m'],
                'original_contact_geometry': original, 'current_contact_geometry': deepcopy(current),
                'exact_geometry_unchanged': True,
                'geometry_stamp_sha256': hashlib.sha256(previous.geometry_stamp.encode()).hexdigest(),
                'geometry_stamp': json.loads(previous.geometry_stamp),
                'original_link_sleeping_flags': json.loads(previous.sleeping_flags),
                'current_link_sleeping_flags': measured['sleeping_flags']})

    def _measure(self):
        from omnigibson.object_states import Touching
        from wia.authoring.converters.spatial_relations_physical_contact import _contact_geometry
        native_time = float(self.session.pipeline.og.sim.current_time)
        if not np.isfinite(native_time):
            raise ValueError('Native rigid contact sample time must be finite.')
        self._observe_contact_time(native_time)
        control = self.controller.geometric_rearrangement_evidence()
        setter_used = control.get('state_setter_used')
        public_control_valid = (not setter_used if control.get('available') is True and
                                type(setter_used) is bool else None)
        protection = self._protected_geometry()
        controls = (public_control_valid, protection['unchanged'])
        control_valid = (False if any(value is False for value in controls) else
                         None if any(value is None for value in controls) else True)
        grasp = _aggregate_grasp(self.session.robot, self.controller)
        hand_links = [self.session.robot.eef_links[arm] for arm in self.session.robot.arm_names]
        hand_links += [link for arm in self.session.robot.arm_names for link in self.session.robot.finger_links[arm]]
        withdrawn = all(_box_gap(hand, self.objects[key]) >= self.p['withdrawal_clearance_m']
                        for hand in hand_links for key in self.p['targets'])
        rows = {}
        for key in self.p['targets']:
            target = self.objects[key]
            contacts = target.contact_list()
            robot_contact = any(_under(body, self.session.robot.prim_path)
                                for row in contacts for body in _contact_bodies(row))
            relations = {}
            for container_key in self.p['containers']:
                container = self.objects[container_key]
                try:
                    relation = partial_interior_evidence(target, container, self.p['interior_margin_m'])
                except (AttributeError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as error:
                    relation = {'interior_valid': None, 'partial_inside': None,
                                'reason': f'{type(error).__name__}: {error}'}
                try:
                    native_touching = target.states[Touching].get_value(container)
                    if native_touching is None:
                        raise ValueError('Native target-container Touching evidence is unavailable.')
                    touching = bool(native_touching)
                    contact = _contact_geometry({'obj1': target, 'obj2': container}, touching,
                                                _box_gap(target, container), self.p['penetration_tolerance_m'])
                    contact = self._contact_geometry_with_sleeping_retention(
                        key, container_key, touching, contact, native_time, protection)
                    relation.update(touching=touching,
                        penetration_free=not contact['penetrating'] if contact['available'] else None,
                        contact_geometry=contact)
                except (AttributeError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as error:
                    pair = (key, container_key)
                    previous = self._contact_witnesses.get(pair)
                    diagnostic = self._contact_failure_diagnostic(pair, 'native_contact_read_failure',
                        f'{type(error).__name__}: {error}', native_time, previous)
                    self._contact_witnesses.pop((key, container_key), None)
                    relation.update(touching=None, penetration_free=None,
                                    contact_error=f'{type(error).__name__}: {error}',
                                    contact_retention_diagnostics=deepcopy(diagnostic))
                relations[container_key] = relation
            rows[key] = {'object_name': target.name, 'lowest_z_m': float(array(target.aabb[0])[2]),
                         'linear_speed_m_s': float(np.linalg.norm(array(target.get_linear_velocity()))),
                         'angular_speed_rad_s': float(np.linalg.norm(array(target.get_angular_velocity()))),
                         'dynamic_collidable': not target.fixed_base and not target.visual_only,
                         'robot_contact': robot_contact, 'containers': relations}
        return {'native_time_s': native_time,
                'control_valid': control_valid, 'control_evidence': control,
                'protected_geometry': protection, 'grasp': grasp,
                'hand_clear': withdrawn, 'targets': rows}

    def sample_step(self):
        if self.active:
            try:
                self.last_sample = self._measure()
                self.scorer.update(self.last_sample)
                self.measurement_error = None
            except (AttributeError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as error:
                self.measurement_error = f'{type(error).__name__}: {error}'
                self.last_sample = self._unavailable_sample(self.measurement_error)
                self.scorer.update(self.last_sample)

    def report_answer(self, answer, observation):
        if len(self.targets) == 1:
            self.answer = {next(iter(self.targets)): str(answer).strip()}
        else:
            value = json.loads(answer)
            if not isinstance(value, dict):
                raise ValueError('Multi-target answers require the declared label-to-container JSON mapping.')
            self.answer = {key: value.get(label, '') for key, label in self.p['target_labels'].items()}

    def record(self, observation, *, action=None):
        self.sample_step()
        measured = self.scorer.result()
        initially_qualified = (isinstance(self.initial_qualification, dict) and
                               self.initial_qualification.get('accepted') is True)
        normalize = lambda text: ' '.join(str(text).lower().replace('_', ' ').split())
        answers = {key: normalize(self.answer.get(key, '')) == normalize(expected)
                   for key, expected in self.p['expected_answers'].items()}
        items = deepcopy(measured['completion_items'])
        items += [{'id': 'answer:' + key, 'kind': 'answer_requirement',
                   'completed': correct, 'evaluation_validity': 'valid'} for key, correct in answers.items()]
        result = {**measured, 'schema': REVISION, 'snapshot_id': observation.snapshot_id,
                  'action_step': observation.action_step, 'completion_items': items,
                  'evaluation_validity': measured['evaluation_validity'] if initially_qualified else 'unavailable',
                  'physical_success': bool(initially_qualified and measured['physical_success']),
                  'answer_correct': all(answers.values()),
                  'task_success': bool(initially_qualified and measured['physical_success'] and all(answers.values())),
                  'initial_qualification': deepcopy(self.initial_qualification),
                  'diagnostic_reducer_physical_success': measured['physical_success'],
                  'physical_scorer_schema': measured['schema'],
                  'physical_scorer_version': measured['scoring_version'],
                  'reported_answers': self.answer, 'native_measurements': self.last_sample,
                  'contact_retention_diagnostics': [
                      {'target_key': pair[0], 'container_key': pair[1], **deepcopy(diagnostic)}
                      for pair, diagnostic in sorted(self._contact_retention_diagnostics.items())],
                  'measurement_error': self.measurement_error, 'scorer_version': REVISION,
                  'evidence_scope': 'Native physical scoring; QA labels preserved separately.'}
        from wia.tasks.completion import progress
        result.update(progress(result))
        if result['evaluation_validity'] != 'complete':
            result['task_success'] = False
            result['physical_success'] = False
        write_json(self.output / 'state.json', result)
        return result


def create_evaluator(session, controller, config, output):
    return RigidPartialInsertionEvaluator(session, controller, config, output)
