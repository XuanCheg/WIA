"""Read-only native identities for pending canonical Mapping worlds.

Run: python -B -m pytest -q tests/test_task_edit2_mapping_native_identity.py
Install before begin_episode. The first provider call must occur after the
observation source retains its complete initial Global-plus-eight observation,
and before public actions. This module never renders, steps, edits native state,
or supplies a mapping_joint_world_qualification receipt.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from wia.authoring.converters.task_edit2_mapping_world import REVISION, digest
from core.observations import AgentObservation
from core.robot_cameras import ROBOT_CAMERA_LABELS
from wia.runtime.sim.esi_native_qualification import condition_snapshot, digest_file
from wia.tasks.interaction_setup import _objects, array
from wia.tasks.native_floor_contacts import measure_native_floor_contacts
from wia.tasks.loader import source_scene_room

BOUNDARIES = frozenset(('floor', 'floors', 'wall', 'walls', 'ceiling', 'ceilings'))
DOORS = frozenset(('door', 'sliding_door', 'gate'))


def _json(value):
    if is_dataclass(value):
        return _json(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    return array(value).tolist()


def _box(value):
    box = np.stack([array(v) for v in value])
    if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[1] < box[0]):
        raise ValueError('A loaded native actor has unavailable or invalid AABB geometry.')
    return box.tolist()


def _collision_rows(obj):
    rows = []
    for link_name, link in sorted(getattr(obj, 'links', {}).items()):
        for name, mesh in sorted(getattr(link, 'collision_meshes', {}).items()):
            rows.append({'link': str(link_name), 'name': str(name),
                'prim_path': str(mesh.prim_path), 'aabb_m': _box(mesh.aabb),
                'collision_enabled': _json(getattr(mesh, 'collision_enabled', None))})
    return rows


def _native_rows(session):
    rows = []
    for obj in sorted(_objects(session.env.scene), key=lambda obj: obj.name):
        if obj is session.robot:
            continue
        position, quaternion = (array(v).tolist() for v in obj.get_position_orientation())
        meshes = _collision_rows(obj)
        enabled = getattr(obj, 'collision_enabled', None)
        rows.append({'name': obj.name, 'category': obj.category, 'model': getattr(obj, 'model', None),
            'prim_path': str(obj.prim_path), 'scale': array(obj.scale).tolist(),
            'fixed_base': bool(obj.fixed_base), 'visual_only': bool(obj.visual_only),
            'position_m': position, 'quaternion_xyzw': quaternion, 'aabb_m': _box(obj.aabb),
            'collision_enabled': _json(enabled), 'collision_meshes': meshes,
            'room_memberships': sorted(str(room) for room in (getattr(obj, 'in_rooms', ()) or ())),
            'room_membership_source': 'native_object_in_rooms_metadata_not_a_geometric_boundary_proof',
            'joints': {str(name): array(joint.get_state()[0]).tolist()
                       for name, joint in sorted(obj.joints.items())}})
    if not rows or len({row['name'] for row in rows}) != len(rows):
        raise ValueError('A complete unique native object inventory is unavailable.')
    return rows


def _constructor(session):
    scene = session.env.scene
    names = ('load_room_instances', 'load_room_types', 'load_object_categories', 'not_load_object_categories')
    configured = getattr(session.env, 'config', {})
    configured = configured.get('scene', {}) if isinstance(configured, dict) else {}
    return {'environment_scene_config': _json(configured),
        'native_filter_attributes': {name: _json(getattr(scene, '_' + name, getattr(scene, name, None)))
                                     for name in names},
        'limitation': 'Absent constructor fields are unknown; this record does not prove full-building loading.'}


def _geometry_provenance(scene, boundaries):
    files = {}
    attributes = {}
    for name in ('scene_file', 'scene_dir', 'scene_dirpath', 'usd_path'):
        value = getattr(scene, name, None)
        if isinstance(value, (str, Path)):
            attributes[name] = str(value)
            path = Path(value)
            if path.is_file():
                files[str(path.resolve())] = digest_file(path)
    measured = {'native_boundaries': deepcopy(boundaries), 'scene_model': scene.scene_model,
        'native_floor_heights_m': array(scene.trav_map.floor_heights).tolist(),
        'native_scene_path_attributes': attributes, 'existing_file_sha256': files,
        'geometry_source': 'loaded_native_world_transforms_aabbs_and_collision_meshes',
        'limitation': 'AABB/mesh identities describe actual loaded geometry; external full-scene equivalence remains unqualified.'}
    return {**measured, 'sha256': digest(measured)}


def _controller_configuration(session, controller, condition):
    config = {name: _json(getattr(controller, name)) for name in
        ('config', 'arm_config', 'rotation_config', 'gripper_config')
        if getattr(controller, name, None) is not None}
    trunk = getattr(controller, 'trunk_controller', None)
    if trunk is not None:
        config['trunk_config'] = _json(trunk.config)
    return {'type': type(controller).__module__ + '.' + type(controller).__qualname__,
        'runtime_configs': config, 'controller_files': condition['controller_files'],
        'action_indices': _json(getattr(controller, 'indices', {})),
        'native_controllers': {str(name): {'type': type(native).__module__ + '.' + type(native).__qualname__,
            'dof_indices': array(native.dof_idx).astype(int).tolist()}
            for name, native in sorted(session.robot.controllers.items())},
        'mounted_sensor_configuration': condition['sensors']}


def _pixel_identity(image):
    return {'sha256': hashlib.sha256(image.tobytes()).hexdigest(),
            'shape': list(image.shape), 'dtype': str(image.dtype)}


def _initial_inputs(session):
    source = getattr(session, 'observation_source', None)
    observation = getattr(source, 'initial_observation', None)
    if type(observation) is not AgentObservation:
        raise ValueError('The complete typed initial AgentObservation is not retained.')
    overview = observation.require_initial_overview(session.initial_overview_identity)
    if (observation.action_step != 0 or observation.snapshot_id != overview.snapshot_id
            or observation.captured_at != overview.captured_at):
        raise ValueError('Initial Global and eight views do not share the initial zero-action boundary.')
    images = observation.rgb_images()
    registry = getattr(session, 'robot_camera_registry', {})
    if (set(registry) != set(ROBOT_CAMERA_LABELS) or len(set(registry.values())) != 8
            or any(name not in session.robot.sensors for name in registry.values())):
        raise ValueError('The actual eight native mounted sensor identities are unavailable.')
    cameras = {name: _pixel_identity(images[name]) for name in ROBOT_CAMERA_LABELS}
    return observation, {'camera_ids': list(ROBOT_CAMERA_LABELS), 'snapshot_id': observation.snapshot_id,
        'captured_at': observation.captured_at, 'rgb_identities': cameras,
        'native_sensor_names': dict(registry)}, {
        'snapshot_id': overview.snapshot_id, 'rgb_identity': _pixel_identity(overview.rgb),
        **_json(overview.metadata())}


def _start(session, controller, observation):
    scene, robot = session.env.scene, session.robot
    floors = [obj for obj in _objects(scene) if obj.category in ('floor', 'floors')]
    if not floors:
        raise ValueError('No loaded native floor actor is available for initial contact readings.')
    heights = array(scene.trav_map.floor_heights).reshape(-1)
    if not len(heights) or not np.isfinite(heights).all():
        raise ValueError('Loaded native floor heights are unavailable.')
    robot_box = np.asarray(_box(robot.aabb))
    floor_index = int(np.argmin(abs(heights - robot_box[0, 2])))
    floor_z = float(heights[floor_index])
    contacts = measure_native_floor_contacts(session, robot,
        [obj.prim_path for obj in floors], nominal_floor_z=floor_z)
    distances = []
    for obj in _objects(scene):
        if obj is robot or obj.category in ('floor', 'floors') or obj.visual_only:
            continue
        box = np.asarray(_box(obj.aabb))
        gap = np.maximum(np.maximum(box[0] - robot_box[1], robot_box[0] - box[1]), 0.)
        distances.append({'native_object': obj.name, 'aabb_gap_m': float(np.linalg.norm(gap))})
    return {'robot_pose': _json(controller.pose()), 'snapshot_id': observation.snapshot_id,
        'support_and_clearance_evidence': {'floor_paths': [obj.prim_path for obj in floors],
            'floor_index': floor_index, 'floor_z_m': floor_z, 'contact_readings': contacts,
            'robot_aabb_m': robot_box.tolist(),
            'nearest_native_aabb_gaps': sorted(distances, key=lambda r: (r['aabb_gap_m'], r['native_object']))[:12],
            'clearance_method': 'native_wheel_contacts_verified_against_local_floor_shapes_and_conservative_AABB_gaps_not_a_swept_route_proof'}}


class MappingWorldIdentity:
    def __init__(self, session, controller, task):
        self.session, self.controller, self.task = session, controller, task
        self.baseline = None

    def _verify_source_and_task(self, contract):
        session, task = self.session, self.task
        loaded = getattr(session, 'loaded_task', None) or getattr(getattr(session, 'scene_adapter', None), 'task', None)
        if loaded is not task:
            raise ValueError('The loaded canonical task changed or is unavailable.')
        task.verify_unchanged()
        p = task.definition.private.evaluation.parameters
        if contract.get('revision') != REVISION or contract != p.get('mapping_joint_world_contract'):
            raise ValueError('Requested joint-world contract differs from the actual loaded task.')
        # The source room identifies the representative question. full_scene is
        # a separate declared loading scope, not a replacement source identity.
        if contract.get('room') != 'full_scene' or contract.get('full_building_required') is not True:
            raise ValueError('Canonical Mapping requires an explicit full-building loading scope.')
        if (session.scene_name != contract['scene'] or session.env.scene.scene_model != contract['scene']
                or task.scene_name != contract['scene'] or task.seed != contract['seed']):
            raise ValueError('The loaded task/scene/start seed differs from the canonical contract.')
        identity = dict(session.initial_overview_identity)
        if identity.get('seed') != task.seed or any(not identity.get(k) for k in ('episode_id', 'reset_id')):
            raise ValueError('Actual native reset/seed identity is unavailable or differs.')
        source_identity = task.definition.source.json_path
        if source_identity != contract['representative_source']:
            raise ValueError('The loaded source is not the declared representative.')
        source_path = Path(task.source_path).resolve()
        root = source_path
        for _ in Path(source_identity).parts:
            root = root.parent
        if (root / source_identity).resolve() != source_path or Path(session.question_path).resolve() != source_path:
            raise ValueError('The native source file differs from the loaded task source.')
        source_raw = json.loads(source_path.read_text(encoding='utf-8'))
        source_scene, source_room = source_scene_room(source_identity, source_raw)
        if (source_scene != session.scene_name or source_scene != task.scene_name
                or source_room != session.room_name or source_room != task.room_name):
            raise ValueError('Loaded scene/room identity differs from the actual representative source JSON.')
        members = p['members']
        indexed = {m['source_identity']: m for m in members}
        if len(indexed) != len(members) or set(indexed) != {m['source_identity'] for m in contract['source_members']}:
            raise ValueError('Loaded source-member membership differs from the canonical contract.')
        evidence = []
        for expected in contract['source_members']:
            name = expected['source_identity']
            source = (root / name).resolve()
            if not source.is_relative_to(root) or digest_file(source) != expected['source_sha256']:
                raise ValueError('A contract source file is missing, changed or outside the source root: ' + name)
            member = indexed[name]
            if member['source_sha256'] != expected['source_sha256']:
                raise ValueError('A loaded member source hash differs: ' + name)
            bound = {}
            for key, definition in member['bindings'].items():
                obj = session.env.scene.object_registry('name', definition['name'])
                if obj is None or any(definition.get(field) is not None and getattr(obj, field, None) != definition[field]
                                      for field in ('category', 'model')):
                    raise ValueError('An original source anchor is absent or changed: ' + name + ':' + key)
                bound[key] = {'name': obj.name, 'category': obj.category, 'model': getattr(obj, 'model', None),
                              'scale': array(obj.scale).tolist(), 'prim_path': str(obj.prim_path)}
            evidence.append({'source_identity': name, 'source_sha256': expected['source_sha256'],
                'actual_source_file': str(source), 'native_bindings': bound,
                'source_door_policy': deepcopy(member['evaluation']['parameters'].get('source_door_policy')),
                'relation_truth': 'not_measured_by_identity_collector'})
        return identity, evidence

    def __call__(self, contract):
        identity, sources = self._verify_source_and_task(contract)
        observation, cameras, overview = _initial_inputs(self.session)
        rows = _native_rows(self.session)
        condition = condition_snapshot(self.session, self.controller, self.task)
        configuration = _controller_configuration(self.session, self.controller, condition)
        boundaries = [row for row in rows if row['category'] in BOUNDARIES]
        if not any(row['category'] in ('floor', 'floors') for row in boundaries):
            raise ValueError('Native loaded boundary geometry has no floor.')
        constructor = _constructor(self.session)
        assets = [{**{key: row[key] for key in ('name', 'category', 'model', 'prim_path', 'scale',
                    'fixed_base', 'visual_only', 'collision_enabled', 'room_memberships')},
                   'collision_mesh_identities': [{key: mesh[key] for key in
                       ('link', 'name', 'prim_path', 'collision_enabled')} for mesh in row['collision_meshes']]}
                  for row in rows]
        room_metadata = {'source': 'native_segmentation_metadata_only',
            'room_ins_name_to_ins_id': _json(getattr(getattr(self.session.env.scene, 'seg_map', None), 'room_ins_name_to_ins_id', {}))}
        protected = [deepcopy(row) for row in boundaries]
        static = {'assets': assets, 'boundaries': protected, 'constructor': constructor,
                  'controller': configuration, 'sources': sources, 'identity': identity,
                  'initial_cameras': cameras, 'initial_overview': overview,
                  'boundary_provenance': _geometry_provenance(self.session.env.scene, boundaries),
                  'room_metadata': room_metadata,
                  'task_sha256': self.task.file_sha256, 'source_sha256': self.task.source_sha256}
        if self.baseline is None:
            if getattr(self.session.observation_source, 'action_step', None) != 0:
                raise ValueError('Initial native world identity cannot be captured after public controls.')
            initial_doors = [deepcopy(row) for row in rows if row['category'] in DOORS]
            measured = {'geometry': {'loaded_rooms': sorted({room for row in rows for room in row['room_memberships']}),
                'object_aabbs': rows, 'boundary_identity': static['boundary_provenance'],
                'constructor_filters': constructor,
                'loading_scope': {'declared_scope': contract['room'],
                    'full_building_required': contract['full_building_required'],
                    'representative_source_room': self.task.room_name,
                    'complete_native_loading': 'pending_external_qualification'},
                'collision_enabled_categories': sorted({row['category'] for row in rows if row['collision_enabled'] is True
                    or any(mesh['collision_enabled'] is True for mesh in row['collision_meshes'])}),
                'room_metadata': room_metadata,
                'room_geometry_limitation': 'Object memberships with measured AABBs are loaded observations; complete room boundary equivalence is not established.'},
                'inventory': {'native_objects': assets, 'source_inventory_evidence': sources},
                'doors': {'native_door_states': initial_doors, 'native_door_count': len(initial_doors),
                    'retention_evidence': {'status': 'measured_initial_inventory_only',
                        'source_runtime_mismatch': 'Source removed doors; this runtime retains the actually loaded native door inventory.',
                        'complete_building_retention': 'pending'}, 'source_policy_reconciliation': None},
                'start': _start(self.session, self.controller, observation),
                'controller': {'configuration': configuration, 'public_control_trial_evidence': None},
                'mounted_cameras': cameras, 'initial_overview': overview}
            self.baseline = {'static': deepcopy(static), 'measurements': deepcopy(measured)}
        elif static != self.baseline['static']:
            changed = [key for key in static if static[key] != self.baseline['static'][key]]
            raise ValueError('Frozen canonical native identity changed: ' + ', '.join(changed))
        measured = deepcopy(self.baseline['measurements'])
        measured['doors']['source_policy_reconciliation'] = deepcopy(
            getattr(self.session, 'mapping_source_policy_evidence', None)) or {
                'status': 'pending', 'reason': 'No reviewed source/runtime door-policy evidence supplied.'}
        measured['controller']['public_control_trial_evidence'] = deepcopy(
            getattr(self.session, 'mapping_controller_trial_evidence', None))
        return {**identity, 'scene': self.session.scene_name, 'measurements': measured,
            'task_identity': {'task_id': self.task.definition.id, 'task_sha256': self.task.file_sha256,
                'source_identity': self.task.definition.source.json_path, 'source_sha256': self.task.source_sha256,
                'contract_sha256': digest(contract)},
            'current_state': {'robot': condition['robot'],
                'door_joint_states': {row['name']: row['joints'] for row in rows if row['category'] in DOORS},
                'native_actor_poses': {row['name']: {'position_m': row['position_m'],
                    'quaternion_xyzw': row['quaternion_xyzw']} for row in rows},
                'note': 'Current robot motion and door joints do not replace the frozen initial measurements.'}}


def install_mapping_world_identity(session, controller, task):
    """Install only a lazy read-only identity provider; never install qualification."""
    provider = MappingWorldIdentity(session, controller, task)
    session.mapping_joint_world_identity = provider
    return provider
