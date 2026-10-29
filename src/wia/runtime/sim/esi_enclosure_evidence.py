"""Collect private enclosure evidence without rendering, stepping or native writes.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_task_edit2_enclosure_evidence.py
Install before begin_episode with install_enclosure_evidence(session, controller).
Root must call collector.sample_step() at native physics samples. The first
provider call must receive the retained initial Global-plus-eight observation.

Independent interior receipts are separate private native reviews, supplied as
session.enclosure_interior_reviews (a list of per-container records). Each uses
INTERIOR_REVIEW_SCHEMA, the exact emitted interior_review_contexts[key], a
review_id, review={reviewer, reviewed_at, method}, and independent_cavity_evidence:
{measurement_source: 'independent_native_cavity_measurement', measurement_id,
 method, interior_geometry_verified: True, inside_or_aabb_only: False,
 cavity_points_local_m: [[x,y,z], ...], content_clearance_m: {content_key: value}}.
Points and positive clearances must come from the independently reviewed native
cavity measurement. The collector never derives this proof from Inside or AABBs.
The geometry fingerprint is the initially closed mesh capture; articulated door
motion is permitted while native mesh identities/local geometry remain fixed.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math

import numpy as np

from core.action_units import ROBOT_ACTIONS
from core.observations import AgentObservation
from core.robot_cameras import ROBOT_CAMERA_LABELS
from wia.authoring.converters import enumerative_perception_structural_enclosure as enclosure
from wia.runtime.sim.esi_visual_evidence import _identity, _native_context, _owner, _retained_capture_matches
from wia.tasks.interaction_setup import _robot_contact_check

COLLECTOR_SCHEMA = 'native_enclosure_evidence_v1'
INTERIOR_REVIEW_SCHEMA = 'native_enclosure_interior_review_v1'
WORLD_FIELDS = ('task_id', 'task_sha256', 'scene_id', 'scene_sha256', 'source_identity',
                'source_sha256', 'episode_id', 'reset_id', 'seed')
ERRORS = (ValueError, TypeError, KeyError, AttributeError, RuntimeError, OSError)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _all(values):
    values = list(values)
    return False if any(value is False for value in values) else None if any(value is None for value in values) else True


def _under(path, root):
    return path == root or path.startswith(root + '/')


def _clock(session):
    value = float(session.pipeline.og.sim.current_time)
    if not math.isfinite(value):
        raise ValueError('Native physics time is unavailable.')
    return value


def _world(session, observation):
    context = _native_context(session, observation)
    # Reuse real loaded-file provenance, rather than trusting task labels alone.
    owners = (getattr(session, 'loaded_task', None), getattr(getattr(session, 'scene_adapter', None), 'task', None))
    for owner in owners:
        verify = getattr(owner, 'verify_unchanged', None)
        if callable(verify):
            verify()
    if any(context.get(key) is None for key in ('task_id', 'task_sha256', 'source_identity', 'source_sha256',
                                               'episode_id', 'reset_id', 'seed')):
        raise ValueError('Exact loaded task/source/reset/seed provenance is unavailable.')
    return {key: context.get(key) for key in WORLD_FIELDS}


def _meshes(obj):
    rows = []
    for link_name, link in sorted(obj.links.items()):
        for name, mesh in sorted(link.collision_meshes.items()):
            box = [enclosure._vector(value, 3).tolist() for value in mesh.aabb]
            if np.any(np.asarray(box[1]) < np.asarray(box[0])):
                raise ValueError('Native collision mesh AABBs must have nonnegative extents.')
            path = getattr(mesh, 'prim_path', None) or str(mesh.prim.GetPath())
            if not path:
                raise ValueError('Native collision mesh identity is unavailable.')
            enabled = getattr(mesh, 'collision_enabled', None)
            shape = None
            prim = getattr(mesh, 'prim', None)
            if prim is not None and callable(getattr(prim, 'GetTypeName', None)):
                kind = prim.GetTypeName()
                fields = {'Mesh': ('points', 'faceVertexCounts', 'faceVertexIndices'), 'Cube': ('size',),
                          'Sphere': ('radius',), 'Cylinder': ('radius', 'height'),
                          'Cone': ('radius', 'height')}.get(kind)
                if fields:
                    values = {field: prim.GetAttribute(field).Get() for field in fields}
                    if all(value is not None for value in values.values()):
                        shape = {'type': kind, 'attributes': {field: enclosure._array(value).tolist()
                                                            for field, value in values.items()}}
            rows.append({'link': str(link_name), 'link_prim_path': str(link.prim_path), 'name': str(name),
                'prim_path': str(path), 'aabb_m': box, 'collision_enabled': bool(enabled) if enabled is not None else None,
                'local_geometry_sha256': _digest(shape) if shape is not None else None,
                'local_geometry': shape})
    if not rows:
        raise ValueError('Actual native container collision mesh inventory is unavailable.')
    return rows


def _mesh_structure(rows):
    # World AABBs of door child meshes change during the permitted joint motion.
    return [{key: row[key] for key in row if key not in ('aabb_m', 'local_geometry')} for row in rows]


def _door_physics(row, meshes):
    values = [row['fixed_base'], not row['visual_only'], row['root_collision_mesh_count'] > 0,
              not row['both_sides'], bool(row['joints'])]
    for joint in row['joints']:
        enabled = [mesh['collision_enabled'] for mesh in meshes if mesh['link_prim_path'] == joint['door_link']]
        collidable = True if any(value is True for value in enabled) else None if any(value is None for value in enabled) else False
        values += [not joint['driven'], joint['visible'], joint['contact_reporting'],
                   joint['collision_mesh_count'] > 0, collidable]
    return _all(values)


def _robot(session, controller, objects, doors, measured_robot):
    robot, floor = session.robot, objects['floor']
    position, quaternion = [enclosure._vector(value, size) for value, size in
        zip(robot.get_position_orientation(), (3, 4))]
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError('Native robot orientation is unavailable.')
    quaternion = quaternion / norm
    floor_z = float(enclosure._vector(floor.aabb[1], 3)[2])
    contacts = list(robot.contact_list())
    contact_details = []
    floor_details_available = True
    hand_paths = {str(robot.eef_links[controller.working_arm].prim_path),
                  *(str(link.prim_path) for link in robot.finger_links[controller.working_arm])}
    door_paths = {str(joint['door_link']) for door in doors.values() for joint in door['joints']}
    body_obstacles = []
    for contact in contacts:
        bodies = [str(contact.body0), str(contact.body1)]
        floor_contact = any(_under(body, str(floor.prim_path)) for body in bodies)
        normal, point = getattr(contact, 'normal', None), getattr(contact, 'position', None)
        if floor_contact and (normal is None or point is None):
            floor_details_available = False
        contact_details.append({'body0': bodies[0], 'body1': bodies[1],
            'normal': enclosure._vector(normal, 3).tolist() if normal is not None else None,
            'position_m': enclosure._vector(point, 3).tolist() if point is not None else None})
        hand_door = any(_under(body, hand) for body in bodies for hand in hand_paths) and any(
            _under(body, door) for body in bodies for door in door_paths)
        for body in bodies:
            if not _under(body, str(robot.prim_path)) and not _under(body, str(floor.prim_path)) and not hand_door:
                body_obstacles.append(body)
    floor_reading = _robot_contact_check(robot, [str(floor.prim_path)], floor_z)
    linear = enclosure._vector(robot.get_linear_velocity(), 3)
    angular = enclosure._vector(robot.get_angular_velocity(), 3)
    checks = {'floor_supported': floor_reading['floor_supported'] if floor_details_available else None,
        'upright': bool(1 - 2 * (quaternion[0] ** 2 + quaternion[1] ** 2) >= .95),
        'in_original_room': measured_robot['in_original_room'], 'body_collision_free': not body_obstacles,
        'stationary': bool(np.linalg.norm(linear) <= .03 and np.linalg.norm(angular) <= .1)}
    return {'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
        'linear_velocity_m_s': linear.tolist(), 'angular_velocity_rad_s': angular.tolist(),
        'floor_top_m': floor_z, 'contacts': contact_details, 'floor_contact_reading': floor_reading,
        'body_obstacles': sorted(set(body_obstacles)), 'checks': checks, 'stable': _all(checks.values())}


class EnclosureEvidenceCollector:
    """Freeze one initial native world and accumulate causally bounded inspections."""

    def __init__(self, session, controller):
        self.session, self.controller = session, controller
        self.baseline = None
        self.baseline_attempted = False
        self.measurement_error = None
        self.config = None
        self.last_sample_time = None
        self.sample_count = 0
        self.openings, self.discoveries, self.strokes = {}, {}, {}
        self.preserved = True
        self.physics_preserved = True
        self.context_invalid = False
        self.current = None

    def _configuration(self, config):
        if (config.get('type') != 'category_operation' or config.get('module') != enclosure.MODULE
                or config.get('operation') != 'open_and_discover_contents'):
            raise ValueError('The private provider supports the owned enclosure discovery operation only.')
        # Existing _configuration validates the same source binding/layout contract.
        enclosure._configuration(self.session, {**config, 'operation': enclosure.OPERATIONS[0]})
        p = config['parameters']
        if p.get('curation_revision') != enclosure.TASK_EDIT2_CURATION_REVISION:
            raise ValueError('This evidence provider is for the explicit curated enclosure additions.')
        objects = {key: self.session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        if any(obj is None or obj.name != config['bindings'][key] for key, obj in objects.items()):
            raise ValueError('An exact bound native enclosure/content/floor identity is unavailable.')
        return objects

    def _read(self):
        before = _clock(self.session)
        objects = self._configuration(self.config)
        if self.baseline is not None and any(objects[key] is not self.objects[key] for key in objects):
            self.preserved = False
            raise ValueError('A bound native actor instance changed after baseline freezing.')
        doors, contents, gripper, robot = enclosure._measure(self.session, self.controller, self.config, objects)
        keys = self.config['parameters']['container_keys'] + self.config['parameters']['content_keys'] + ['floor']
        identities = {key: _identity(objects[key]) for key in keys}
        meshes = {key: _meshes(objects[key]) for key in doors}
        measured = {'physics_time_s': before, 'doors': doors, 'contents': contents,
            'gripper': gripper, 'identities': identities, 'collision_meshes': meshes,
            'mesh_fingerprints': {key: _digest(rows) for key, rows in meshes.items()},
            'door_physics': {key: _door_physics(row, meshes[key]) for key, row in doors.items()},
            'robot': _robot(self.session, self.controller, objects, doors, robot)}
        if _clock(self.session) != before:
            raise ValueError('Native physics advanced during enclosure measurement.')
        return measured, objects

    def _freeze(self, observation, config):
        self.baseline_attempted = True
        self.config = deepcopy(config)
        self.config_sha256 = _digest(config)
        source = getattr(self.session, '_observation_source', None)
        if (type(observation) is not AgentObservation or observation.action_step != 0
                or getattr(source, 'initial_observation', None) is not observation
                or not _retained_capture_matches(self.session, observation)):
            raise ValueError('Baseline requires the complete retained initial action-step-zero observation.')
        context = _world(self.session, observation)
        overview = observation.require_initial_overview({key: context[key] for key in ('episode_id', 'reset_id', 'seed')})
        if overview.snapshot_id != observation.snapshot_id or overview.captured_at != observation.captured_at:
            raise ValueError('Initial Global and mounted images do not share one retained boundary.')
        images = observation.rgb_images()
        registry = getattr(self.session, 'robot_camera_registry', {})
        if set(images) != {'initial_overview_rgb', *ROBOT_CAMERA_LABELS} or set(registry) != set(ROBOT_CAMERA_LABELS) or len(set(registry.values())) != 8:
            raise ValueError('The complete initial Global-plus-eight mounted input is unavailable.')
        for name in registry.values():
            sensor = self.session.robot.sensors[name]
            if not _under(str(sensor.prim_path), str(self.session.robot.prim_path)):
                raise ValueError('An initial camera is not native-mounted on this robot.')
        public = getattr(self.controller, '_public_action_context', {})
        if public.get('active') or public.get('sequence', 0) != 0 or getattr(self.controller, '_public_action_sequence', 0) != 0:
            raise ValueError('A baseline cannot be created after public controls started.')
        if getattr(self.session, '_episode_state_setter_used', False):
            raise ValueError('An episode state setter invalidates native enclosure evidence.')
        measured, self.objects = self._read()
        self.initial_observation = observation
        self.world_context = context
        self.baseline = deepcopy(measured)
        self.baseline.update(context=deepcopy(context), initial_snapshot_id=observation.snapshot_id,
            initial_captured_at=observation.captured_at,
            initial_doors_closed={key: not door['open'] and all(abs(joint['velocity']) <= .03 and
                abs(joint['position'] - (joint['lower_limit'] if joint['open_direction'] > 0 else joint['upper_limit'])) <= joint['closed_tolerance']
                for joint in door['joints']) for key, door in measured['doors'].items()})
        self.last_sample_time = measured['physics_time_s']
        self.current = measured
        self.strokes = {key: {joint['name']: None for joint in door['joints']} for key, door in measured['doors'].items()}
        self.last_sample_positions = {key: {joint['name']: joint['position'] for joint in door['joints']}
                                      for key, door in measured['doors'].items()}

    def _protect(self, current):
        baseline = self.baseline
        same_identities = current['identities'] == baseline['identities']
        stationary_roots = all(enclosure._near(row, baseline['doors'][key]) for key, row in current['doors'].items())
        contents_preserved = all(enclosure._near(row, baseline['contents'][key]) and row['inside_original_container']
                                 and all(row[field] == baseline['contents'][key][field] for field in ('fixed_base', 'visual_only'))
                                 for key, row in current['contents'].items())
        mesh_structure = all(_mesh_structure(rows) == _mesh_structure(baseline['collision_meshes'][key])
                             for key, rows in current['collision_meshes'].items())
        self.preserved &= bool(same_identities and stationary_roots and contents_preserved and mesh_structure
                               and not getattr(self.session, '_episode_state_setter_used', False))
        if any(value is False for value in current['door_physics'].values()):
            self.physics_preserved = False
        for key, door in current['doors'].items():
            fields = ('name', 'kind', 'door_link', 'lower_limit', 'upper_limit', 'open_direction',
                      'minimum_travel', 'closed_tolerance')
            initial = [{field: joint[field] for field in fields} for joint in baseline['doors'][key]['joints']]
            actual = [{field: joint[field] for field in fields} for joint in door['joints']]
            self.physics_preserved &= initial == actual

    def sample_step(self):
        """Read one real advanced physics sample; repeated callbacks earn no travel."""
        _owner()
        if self.baseline is None or self.context_invalid:
            return
        try:
            if _world(self.session, self.initial_observation) != self.world_context:
                self.context_invalid = True
                raise ValueError('Task/scene/source/reset/seed context changed after freezing.')
            current, _ = self._read()
            self._protect(current)
            self.current = current
            now = current['physics_time_s']
            if now <= self.last_sample_time:
                if now < self.last_sample_time:
                    self.context_invalid = True
                    raise ValueError('Native time moved backwards after baseline freezing.')
                if any(abs(joint['position'] - self.last_sample_positions[key].get(joint['name'], float('inf'))) > 1e-9
                       for key, door in current['doors'].items() for joint in door['joints']):
                    self.context_invalid = True
                    raise ValueError('Door state changed without an advancing native physics sample.')
                return
            self.last_sample_time = now
            self.last_sample_positions = {key: {joint['name']: joint['position'] for joint in door['joints']}
                                          for key, door in current['doors'].items()}
            self.sample_count += 1
            public = getattr(self.controller, '_public_action_context', {})
            sequence = public.get('sequence')
            active = (public.get('active') is True and type(sequence) is int and sequence > 0
                      and sequence == getattr(self.controller, '_public_action_sequence', None)
                      and public.get('action') in ROBOT_ACTIONS)
            for key, door in current['doors'].items():
                original = self.baseline['doors'][key]
                initial_joints = {joint['name']: joint for joint in original['joints']}
                if set(initial_joints) != {joint['name'] for joint in door['joints']}:
                    self.physics_preserved = False
                    continue
                for joint in door['joints']:
                    previous = self.strokes[key][joint['name']]
                    initial = initial_joints[joint['name']]
                    excursion = (joint['position'] - initial['position']) * joint['open_direction']
                    valid = active and self.preserved and self.physics_preserved and current['door_physics'][key] is True and self.baseline['initial_doors_closed'][key]
                    valid &= joint['hand_contact'] is True and joint['open_direction'] == initial['open_direction']
                    if not valid:
                        self.strokes[key][joint['name']] = None
                        continue
                    if previous is None or previous['sequence'] != sequence:
                        # Contact must first be observed while closed, before movement.
                        self.strokes[key][joint['name']] = ({'sequence': sequence, 'contact_time_s': now,
                            'last_time_s': now, 'last_position': joint['position'], 'contact_open_travel': 0.}
                            if not door['open'] and abs(excursion) <= joint['closed_tolerance'] else None)
                        continue
                    delta = (joint['position'] - previous['last_position']) * joint['open_direction']
                    if delta < -1e-8:
                        self.strokes[key][joint['name']] = None
                        continue
                    previous.update(last_time_s=now, last_position=joint['position'],
                        contact_open_travel=previous['contact_open_travel'] + max(0., delta))
                    if (now > previous['contact_time_s'] and door['open'] and excursion >= joint['minimum_travel']
                            and previous['contact_open_travel'] >= joint['minimum_travel']):
                        self.openings.setdefault(key, {'container': key, 'joint': joint['name'], 'control_sequence': sequence,
                            'action': public['action'], 'contact_time_s': previous['contact_time_s'], 'opened_time_s': now,
                            'contact_open_travel': previous['contact_open_travel'], 'native_open': True})
            self.measurement_error = None
        except ERRORS as error:
            self.measurement_error = f'{type(error).__name__}: {error}'
            for joints in self.strokes.values():
                for name in joints:
                    joints[name] = None

    def _review_contexts(self):
        return {key: {**deepcopy(self.world_context), 'initial_snapshot_id': self.baseline['initial_snapshot_id'],
            'container_binding': key, 'native_identity': deepcopy(self.baseline['identities'][key]),
            'collision_geometry_sha256': self.baseline['mesh_fingerprints'][key]}
            for key in self.config['parameters']['container_keys']}

    def _interior(self, contexts):
        reviews = getattr(self.session, 'enclosure_interior_reviews', ())
        reviews = list(reviews.values()) if isinstance(reviews, dict) else list(reviews or ())
        accepted = {}
        for key, context in contexts.items():
            if not self.preserved or any(context.get(field) is None for field in WORLD_FIELDS):
                continue
            contents = {content for content, container in self.config['parameters']['relations'].items() if container == key}
            for receipt in reviews:
                if not isinstance(receipt, dict) or receipt.get('schema') != INTERIOR_REVIEW_SCHEMA or receipt.get('context') != context or not receipt.get('review_id'):
                    continue
                review = receipt.get('review', {})
                evidence = receipt.get('independent_cavity_evidence', {})
                if not isinstance(review, dict) or not isinstance(evidence, dict):
                    continue
                try:
                    if (not all(isinstance(review.get(field), str) and review[field].strip() for field in ('reviewer', 'reviewed_at', 'method'))
                            or datetime.fromisoformat(review['reviewed_at'].replace('Z', '+00:00')).tzinfo is None):
                        continue
                    if (evidence.get('measurement_source') != 'independent_native_cavity_measurement'
                            or evidence.get('interior_geometry_verified') is not True or evidence.get('inside_or_aabb_only') is not False
                            or not isinstance(evidence.get('measurement_id'), str) or not evidence['measurement_id']
                            or not isinstance(evidence.get('method'), str) or not evidence['method']):
                        continue
                    points = enclosure._array(evidence.get('cavity_points_local_m'))
                    clearances = evidence.get('content_clearance_m', {})
                    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or set(clearances) != contents:
                        continue
                    if not all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in clearances.values()):
                        continue
                except ERRORS:
                    continue
                accepted[key] = deepcopy(receipt)
                break
        return True if len(accepted) == len(contexts) else None, accepted

    def __call__(self, observation, config):
        _owner()
        unavailable = {gate: None for gate in enclosure.TASK_EDIT2_ENCLOSURE_GATES}
        try:
            if self.baseline is None:
                if self.baseline_attempted:
                    raise ValueError('The initial baseline was unavailable; late baseline freezing is forbidden.')
                self._freeze(observation, config)
            if self.context_invalid or _digest(config) != self.config_sha256 or _world(self.session, observation) != self.world_context:
                self.context_invalid = True
                raise ValueError('Enclosure evidence belongs to a different task/config/reset context.')
            if not _retained_capture_matches(self.session, observation):
                raise ValueError('Enclosure discovery requires the current retained native observation.')
            current, _ = self._read()
            if current['physics_time_s'] < self.last_sample_time:
                self.context_invalid = True
                raise ValueError('A retained discovery view precedes the latest native physics sample.')
            self._protect(current)
            self.current = current
            contexts = self._review_contexts()
            interior, reviews = self._interior(contexts)
            provider = getattr(self.session, 'task_visual_evidence', None)
            try:
                visual = provider(observation, {key: self.config['bindings'][key] for key in self.config['parameters']['content_keys']},
                                  self.config['parameters']) if callable(provider) else {}
                if not isinstance(visual, dict):
                    raise ValueError('Shared visual provider did not return measured evidence.')
            except ERRORS as error:
                visual = {'available': False, 'objects': {}, 'measurement_errors': {'provider': f'{type(error).__name__}: {error}'}}
            if _clock(self.session) != current['physics_time_s']:
                raise ValueError('Native time advanced between enclosure and visual evidence reads.')
            rows = {}
            for key in self.config['parameters']['content_keys']:
                row = {'native_name': current['identities'][key]['name'], 'snapshot_id': observation.snapshot_id,
                    'action_step': observation.action_step, 'camera': None, 'identity_synchronized': False,
                    'native_rgb_matches_delivered': False, 'informative_feature_evidence': {},
                    **deepcopy(visual.get('objects', {}).get(key, {}))}
                container = self.config['parameters']['relations'][key]
                feature = row.get('informative_feature_evidence', {})
                audit = row.get('audit_context', {})
                synchronized = (row.get('native_name') == self.config['bindings'][key]
                    and row.get('snapshot_id') == observation.snapshot_id and row.get('action_step') == observation.action_step
                    and row.get('camera') in ROBOT_CAMERA_LABELS and row.get('camera') in observation.rgb_images()
                    and row.get('identity_synchronized') is True and row.get('native_rgb_matches_delivered') is True
                    and audit.get('physics_time_s') == current['physics_time_s']
                    and all(audit.get(field) == self.world_context[field] for field in WORLD_FIELDS))
                informative = bool(synchronized and row.get('informative') is True and isinstance(feature, dict)
                    and feature.get('semantic_identity_verified') is True
                    and isinstance(feature.get('feature_ids'), (list, tuple)) and bool(feature['feature_ids'])
                    and all(isinstance(value, str) and value for value in feature['feature_ids'])
                    and key in current['contents'] and current['contents'][key]['inside_original_container']
                    and container in self.openings and current['doors'][container]['open']
                    and current['door_physics'][container] is True and self.preserved and self.physics_preserved)
                row.update(informative=informative, post_open_view_verified=informative,
                    public_opening_receipt=deepcopy(self.openings.get(container)), original_container_binding=container)
                if not informative:
                    row['informative_feature_evidence'] = {}
                if informative:
                    self.discoveries.setdefault(key, {'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                        'camera': row['camera'], 'native_name': row['native_name'], 'physics_time_s': current['physics_time_s'],
                        'feature_evidence': deepcopy(feature), 'opening': deepcopy(self.openings[container])})
                rows[key] = row
            p = self.config['parameters']
            containment = bool(self.preserved and all(row['inside_original_container'] for row in self.baseline['contents'].values())
                               and all(row['inside_original_container'] for row in current['contents'].values()))
            opening = bool(self.preserved and self.physics_preserved and all(key in self.openings for key in p['container_keys']))
            visible = (True if self.preserved and self.physics_preserved and all(key in self.discoveries for key in p['content_keys'])
                       else False if not self.preserved or not self.physics_preserved or visual.get('available') is True else None)
            gates = {'actual_containment': containment, 'independent_interior_geometry': interior,
                'public_control_contact_opening': opening, 'post_open_informative_visibility': visible,
                'access_to_every_content': visible,
                'stable_robot_start': _all((self.baseline['robot']['stable'], current['robot']['stable']))}
            self.measurement_error = None
            return {'schema': COLLECTOR_SCHEMA, 'available': bool(visual.get('available') is True and self.preserved
                and self.physics_preserved), 'objects': rows, 'gates': gates, 'baseline': deepcopy(self.baseline),
                'native': deepcopy(current), 'opening_receipts': deepcopy(self.openings), 'discovery_receipts': deepcopy(self.discoveries),
                'interior_review_contexts': contexts, 'accepted_interior_reviews': reviews,
                'native_sample_count': self.sample_count, 'measurement_error': None,
                'shared_visual_measurement_errors': deepcopy(visual.get('measurement_errors', {})),
                'semantic_evidence_source': 'shared_exact_audited_native_visual_features',
                'interior_inferred_from_inside_or_aabb': False}
        except ERRORS as error:
            self.measurement_error = f'{type(error).__name__}: {error}'
            return {'schema': COLLECTOR_SCHEMA, 'available': False, 'objects': {}, 'gates': unavailable,
                'measurement_error': self.measurement_error, 'baseline_frozen': self.baseline is not None,
                'native_sample_count': self.sample_count, 'interior_inferred_from_inside_or_aabb': False}


def install_enclosure_evidence(session, controller):
    """Install only a private provider; root wires sample_step into native control."""
    _owner()
    collector = EnclosureEvidenceCollector(session, controller)
    session.enumerative_enclosure_evidence = collector
    session.enclosure_evidence_collector = collector
    return collector
