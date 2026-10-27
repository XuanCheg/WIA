"""Transfer temporal worlds onto explicitly selected existing furnished supports.

Run: python -B scripts/build_wia_furnished_temporal.py --design-file PATH
  --manifest-output PATH [--check]
Input: wia_furnished_temporal_design_v1 with explicit rigid transforms and exact
support-name maps. This CPU builder never simulates, renders or operates robots.


Canonical invocation: python -m wia.authoring.commands.build_wia_furnished_temporal --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from wia.resources import ROOT

from wia.tasks import sources
from wia.tasks.loader import UniqueLoader, load_scene, load_task, source_scene_room
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition
from wia.authoring.commands import build_wia_temporal_scene_variants as legacy
from wia.authoring.commands.build_wia_workstation_variants import NATIVE_FLAGS, _text, _vector, repo_path, sha256

MODULE = legacy.MODULE
NATIVE_FURNITURE_MODULE = 'core.esi_converters.wia_native_furniture'
POINT_FIELDS = {'position', 'look_target', 'target', 'world_position_m', 'eye_m'}
DIRECTION_FIELDS = {'linear_velocity', 'angular_velocity', 'linear_velocity_m_s', 'angular_velocity_rad_s',
                    'direction', 'normal', 'up', 'forward', 'world_direction'}
QUATERNION_FIELDS = {'orientation', 'quaternion_xyzw'}
LOCAL_VECTOR_FIELDS = {'bbox_size_m', 'dimensions_m', 'scale', 'rgba', 'indices', '_indices', 'bbox_map_rc',
    'controller_initial_trunk_joint_positions_rad', 'jitter_xy_m', 'camera_size', 'image_size',
    'intrinsic_matrix', 'reset_joint_pos', 'root_offset_from_aabb_min_m'}
HISTORICAL_FIELDS = {'authored_revision', 'provenance', 'parent_provenance', 'transfer_provenance'}


def _historical(key):
    return key in HISTORICAL_FIELDS or key.startswith('parent_') or key.endswith('_provenance')


class RigidTransform:
    """Apply p' = origin + Rz(yaw) * (p - origin) + translation in float64."""
    def __init__(self, specification):
        if not isinstance(specification, dict) or set(specification) != {'origin_m', 'translation_m', 'yaw_deg'}:
            raise ValueError('transform must contain exactly origin_m, translation_m and yaw_deg.')
        self.specification = deepcopy(specification)
        self.origin = np.asarray(_vector(specification['origin_m'], 3, 'origin_m'), dtype=float)
        self.translation = np.asarray(_vector(specification['translation_m'], 3, 'translation_m'), dtype=float)
        yaw = specification['yaw_deg']
        if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not np.isfinite(yaw):
            raise ValueError('yaw_deg must be an explicit finite angle.')
        self.yaw = float(yaw)
        self.rotation = Rotation.from_euler('z', yaw, degrees=True)
        self.matrix = self.rotation.as_matrix()
        # Exact cardinal coefficients avoid tiny off-axis drift in matched copies.
        self.matrix[np.abs(self.matrix) < 1e-14] = 0.
        self.matrix[np.abs(self.matrix - 1.) < 1e-14] = 1.
        self.matrix[np.abs(self.matrix + 1.) < 1e-14] = -1.

    @staticmethod
    def parked(vector):
        return bool(np.any(np.abs(np.asarray(vector, dtype=float)) >= 900))

    def point(self, value):
        point = np.asarray(_vector(value, 3, 'world point'), dtype=float)
        return deepcopy(value) if self.parked(point) else (self.origin + self.matrix @ (point - self.origin) + self.translation).tolist()

    def direction(self, value):
        return (self.matrix @ _vector(value, 3, 'world direction')).tolist()

    def quaternion(self, value):
        quaternion = np.asarray(_vector(value, 4, 'world quaternion'), dtype=float)
        if not np.any(quaternion):
            raise ValueError('A world quaternion must be nonzero.')
        if self.yaw % 360 == 0:
            return deepcopy(value)
        return (self.rotation * Rotation.from_quat(quaternion)).as_quat().tolist()

    def bounds(self, value):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError('World bbox requires two three-dimensional corners.')
        low, high = (np.asarray(_vector(row, 3, 'world bbox corner'), dtype=float) for row in value)
        if np.any(high < low):
            raise ValueError('World bbox corners must be ordered.')
        if self.parked(value):
            return deepcopy(value)
        corners = np.array([self.point(list(corner)) for corner in product(*zip(low, high))])
        return [corners.min(axis=0).tolist(), corners.max(axis=0).tolist()]

    def xy_bounds(self, value):
        box = _vector(value, 4, 'world XY bbox')
        if self.parked(box):
            return deepcopy(value)
        corners = np.array([self.point([x, y, self.origin[2]])[:2]
                            for x, y in product((box[0], box[2]), (box[1], box[3]))])
        return [*corners.min(axis=0).tolist(), *corners.max(axis=0).tolist()]


def _numeric_vector(value):
    return (isinstance(value, list) and bool(value)
            and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value))


def transform_world(value, transform, path='world'):
    """Transform reviewed world semantics and reject ambiguous numeric vectors."""
    rigid = transform if isinstance(transform, RigidTransform) else RigidTransform(transform)
    if isinstance(value, list):
        return [transform_world(item, rigid, path + '[]') for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    if value.get('frame', value.get('pose_frame')) in ('parent', 'local'):
        return deepcopy(value)
    if _numeric_vector(value.get('position')) and rigid.parked(value['position']):
        return deepcopy(value)
    result = {}
    if ('aabb_min' in value) != ('aabb_max' in value):
        raise ValueError('Paired aabb_min/aabb_max are required at ' + path)
    aabb = rigid.bounds([value['aabb_min'], value['aabb_max']]) if 'aabb_min' in value else None
    for key, item in value.items():
        child = path + '.' + key
        if _historical(key):
            result[key] = deepcopy(item)
        elif key in POINT_FIELDS and isinstance(item, list):
            result[key] = rigid.point(item)
        elif key in DIRECTION_FIELDS and isinstance(item, list):
            result[key] = rigid.direction(item)
        elif key in QUATERNION_FIELDS and isinstance(item, list):
            result[key] = rigid.quaternion(item)
        elif key == 'yaw_deg':
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not np.isfinite(item):
                raise ValueError('A finite world yaw_deg is required at ' + child)
            result[key] = float(item + rigid.yaw)
        elif key in legacy.XY_BOX_FIELDS:
            result[key] = rigid.xy_bounds(item)
        elif key == 'bbox':
            result[key] = rigid.bounds(item)
        elif key in ('aabb_min', 'aabb_max'):
            result[key] = aabb[key == 'aabb_max']
        elif key == 'source_target_positions':
            if not isinstance(item, dict):
                raise ValueError('source_target_positions must map native names to world positions.')
            result[key] = {name: rigid.point(point) for name, point in item.items()}
        elif key == 'bbox_size_m' and '.source_objects.' in child:
            result[key] = (np.abs(rigid.matrix) @ _vector(item, 3, 'world expected bbox_size_m')).tolist()
        elif key in LOCAL_VECTOR_FIELDS:
            result[key] = deepcopy(item)
        elif _numeric_vector(item):
            raise ValueError('Unsupported ambiguous numeric vector at ' + child)
        else:
            result[key] = transform_world(item, rigid, child)
    return result


def rename_references(value, names):
    """Replace exact native identifiers everywhere active, retaining provenance."""
    if isinstance(value, list):
        return [rename_references(item, names) for item in value]
    if not isinstance(value, dict):
        return names.get(value, value) if isinstance(value, str) else deepcopy(value)
    result = {}
    for key, item in value.items():
        renamed = names.get(key, key)
        if renamed in result:
            raise ValueError('Native support renaming creates duplicate dictionary identities.')
        result[renamed] = deepcopy(item) if _historical(key) else rename_references(item, names)
    return result


def _entry_contract(entry):
    for key in ('document_alias', 'variant_id', 'label', 'source_identity', 'scene_model', 'room', 'floor_name'):
        _text(entry, key)
    transform = RigidTransform(entry['transform'])
    names = entry.get('support_name_map')
    if (not isinstance(names, dict) or not names
            or any(not isinstance(key, str) or not key or not isinstance(value, str) or not value for key, value in names.items())
            or len(set(names.values())) != len(names)):
        raise ValueError('support_name_map must explicitly map distinct source supports to existing native supports.')
    if 'background_support_objects' in entry:
        raise ValueError('Furnished transfer must use existing supports, without replacement table constructors.')
    if 'apparatus_translation_world_m' in entry:
        translation = _vector(entry['apparatus_translation_world_m'], 3, 'apparatus_translation_world_m')
        if translation[2] != 0:
            raise ValueError('apparatus_translation_world_m must have Z == 0 to preserve floor and height.')
    if ('interaction_relocation_policy' in entry
            and entry['interaction_relocation_policy'] not in ('independent_groups', 'vertical_only')):
        raise ValueError('interaction_relocation_policy must be independent_groups or vertical_only.')
    if 'apparatus_floor_z_m' in entry:
        floor = entry['apparatus_floor_z_m']
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not np.isfinite(floor):
            raise ValueError('apparatus_floor_z_m must be an explicit finite world floor height.')
    if 'native_furniture_hook_boundary' in entry:
        raise ValueError('Use explicit per-variant native_furniture_hook_boundaries.')
    if 'native_furniture_hook_boundaries' in entry:
        boundaries = entry['native_furniture_hook_boundaries']
        if (not isinstance(boundaries, dict)
                or any(not isinstance(key, str) or not key or value not in
                       ('before_source_restore', 'after_temporal_initialization')
                       for key, value in boundaries.items())):
            raise ValueError('native_furniture_hook_boundaries must map variants to supported explicit boundaries.')
    if 'native_object_poses' in entry:
        poses = entry['native_object_poses']
        if not isinstance(poses, list):
            raise ValueError('native_object_poses must be an explicit list.')
        names = set()
        for pose in poses:
            if not isinstance(pose, dict) or set(pose) != {'name', 'category', 'model', 'position', 'orientation'}:
                raise ValueError('Native furniture poses require exact name/category/model and pose fields only.')
            for key in ('name', 'category', 'model'):
                _text(pose, key)
            if pose['name'] in names:
                raise ValueError('Native furniture pose names must be distinct.')
            names.add(pose['name'])
            _vector(pose['position'], 3, 'native furniture position')
            orientation = _vector(pose['orientation'], 4, 'native furniture orientation')
            if not any(orientation):
                raise ValueError('Native furniture orientation must be nonzero.')
    if 'robot_start' in entry:
        robot = entry['robot_start']
        if not isinstance(robot, dict) or set(robot) != {'position', 'orientation'}:
            raise ValueError('robot_start must provide final world position and quaternion orientation.')
        _vector(robot['position'], 3, 'robot_start position')
        quaternion = _vector(robot['orientation'], 4, 'robot_start orientation')
        matrix = Rotation.from_quat(quaternion).as_matrix()
        if not np.allclose(matrix[:, 2], [0., 0., 1.], atol=1e-8, rtol=0):
            raise ValueError('The native robot_pose hook supports an upright yaw-only robot_start.')
    return transform


def ground_apparatus_legs(value, floor_z):
    """Keep designated transformed leg tops fixed and place only their feet on floor."""
    if floor_z is None:
        return deepcopy(value)
    if isinstance(value, list):
        return [ground_apparatus_legs(item, floor_z) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    result = {}
    for key, item in value.items():
        if _historical(key):
            result[key] = deepcopy(item)
        elif key == 'primitives':
            rows = deepcopy(item)
            for row in rows:
                if not row['name'].startswith(('change_collection_table_leg_', 'change_spare_table_leg_')):
                    continue
                position = _vector(row['position'], 3, 'apparatus leg world position')
                dimensions = _vector(row['dimensions_m'], 3, 'apparatus leg local dimensions')
                top_z = position[2] + dimensions[2] / 2
                height = top_z - floor_z
                if not np.isfinite(height) or height <= 0:
                    raise ValueError('The explicit apparatus floor produces a nonpositive leg height.')
                row['position'][2] = (top_z + floor_z) / 2
                row['dimensions_m'][2] = height
            result[key] = rows
        else:
            result[key] = ground_apparatus_legs(item, floor_z)
    return result


def translate_active_apparatus(value, translation_world_m, *, floor_z=None):
    """Shift only active apparatus in final world axes, never its historical copies."""
    if translation_world_m is None:
        return deepcopy(value)
    if isinstance(value, list):
        return [translate_active_apparatus(item, translation_world_m, floor_z=floor_z) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    result = {}
    for key, item in value.items():
        if _historical(key):
            result[key] = deepcopy(item)
        elif key == 'apparatus':
            # Source copies may contain their own active apparatus. Ground only
            # that dictionary before applying the same final-world translation.
            apparatus = deepcopy(item)
            for field in ('primitives', 'dataset_objects'):
                if field not in item:
                    continue
                constructors = ground_apparatus_legs({field: item[field]}, floor_z)[field]
                apparatus[field] = transform_world(constructors,
                    {'origin_m': [0., 0., 0.], 'translation_m': translation_world_m, 'yaw_deg': 0.},
                    'active.apparatus.' + field)
            result[key] = apparatus
        else:
            result[key] = translate_active_apparatus(item, translation_world_m, floor_z=floor_z)
    return result


def _restore_poses(variant):
    """Restore rotated poses after source postprocessing, before phase installation."""
    patches, result = variant['patches'], []
    for patch in patches:
        if patch.get('operation') == 'category_initialization' and patch.get('phase') == 'after_postprocessing':
            p = patch['parameters']
            poses = {}
            for key in [*p['container_keys'], *p['phase_keys'][p['phase']]]:
                row = p['source_objects'][key]
                poses[key] = {'position': row['position'], 'quaternion_xyzw': row['quaternion_xyzw']}
            for item in p.get('apparatus', {}).get('primitives', []):
                key = next(key for key, binding in p['bindings'].items() if binding['name'] == item['name'])
                poses[key] = {'position': item['position'], 'quaternion_xyzw': item['quaternion_xyzw']}
            for key, pose in poses.items():
                result.append({'operation': 'object_pose', 'phase': 'after_postprocessing', 'binding': key, **deepcopy(pose)})
        result.append(patch)
    variant['patches'] = result


def _insert_native_furniture_hook(variant, entry, variant_name):
    """Use only the authored boundary for this particular diagnostic/active world."""
    poses = entry.get('native_object_poses')
    patches = variant['patches']
    boundary = entry.get('native_furniture_hook_boundaries', {}).get(variant_name, 'before_source_restore')
    if boundary == 'after_temporal_initialization':
        temporal = [index for index, patch in enumerate(patches)
            if patch.get('phase') == 'after_postprocessing'
            and patch.get('operation') == 'category_initialization' and patch.get('module') == MODULE]
        prepare = [index for index, patch in enumerate(patches)
                   if patch.get('operation') == 'prepare_category_interaction']
        if (not temporal or len(prepare) != 1 or max(temporal) >= prepare[0]
                or patches[prepare[0]].get('phase') != 'after_postprocessing'):
            raise ValueError('after_temporal_initialization requires a two_phase initializer followed by one interaction preparation boundary in variant ' + variant_name)
        before = max(temporal) + 1
    else:
        if not poses or not patches:
            return
        before = next((index for index, patch in enumerate(patches)
            if patch.get('phase') == 'after_postprocessing'
            and patch.get('operation') in ('object_pose', 'category_initialization')), None)
        if before is None:
            raise ValueError('A nonempty furnished variant needs an initialization boundary for native furniture reset.')
    if not poses:
        return
    patches.insert(before, {'operation': 'category_initialization', 'phase': 'after_postprocessing',
        'module': NATIVE_FURNITURE_MODULE, 'parameters': {
            'scene_model': entry['scene_model'], 'native_object_poses': deepcopy(poses)}})


def _prepare_primitives(value):
    """Serialize formerly implicit world identity orientations before rotation."""
    if isinstance(value, list):
        return [_prepare_primitives(item) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    result = {}
    for key, item in value.items():
        if _historical(key):
            result[key] = deepcopy(item)
        elif key == 'primitives':
            result[key] = [{**deepcopy(row), 'quaternion_xyzw': row.get('quaternion_xyzw', [0., 0., 0., 1.])}
                           for row in item]
        else:
            result[key] = _prepare_primitives(item)
    return result


def translate_source(parent_source, entry, provenance):
    _entry_contract(entry)
    zero = {**entry, 'translation_m': [0., 0., 0.]}
    raw = legacy.translate_source(parent_source, zero, provenance)
    encoded = isinstance(raw['metadata_json'], str)
    if encoded:
        raw['metadata_json'] = json.loads(raw['metadata_json'])
    raw = rename_references(transform_world(raw, entry['transform'], 'source'), entry['support_name_map'])
    raw = translate_active_apparatus(raw, entry.get('apparatus_translation_world_m'),
                                     floor_z=entry.get('apparatus_floor_z_m'))
    if encoded:
        raw['metadata_json'] = json.dumps(raw['metadata_json'], ensure_ascii=False, allow_nan=False)
    source_scene_room(entry['source_identity'], raw)
    return raw


def translate_definitions(parent_task, parent_scene, entry, provenance, runtime):
    rigid = _entry_contract(entry)
    zero = {**entry, 'translation_m': [0., 0., 0.], 'background_support_objects': []}
    task, scene = legacy.translate_definitions(parent_task, parent_scene, zero, provenance, runtime)
    old_floor = parent_task['private']['bindings']['floor']['name']
    names = {**entry['support_name_map'], old_floor: entry['floor_name']}
    variants = task['scene']['variants']
    if set(entry.get('native_furniture_hook_boundaries', {})) - set(variants):
        raise ValueError('Native furniture hook boundary keys must name existing scene variants.')
    original = variants.get('original')
    if original is not None and original.get('patches'):
        raise ValueError('A nonempty original variant requires explicit review before furnished transfer.')
    task['private']['evaluation']['parameters'] = rename_references(legacy._active_identities(
        ground_apparatus_legs(transform_world(_prepare_primitives(task['private']['evaluation']['parameters']), rigid, 'evaluation'),
                             entry.get('apparatus_floor_z_m')), entry), names)
    task['private']['evaluation']['parameters'] = translate_active_apparatus(
        task['private']['evaluation']['parameters'], entry.get('apparatus_translation_world_m'))
    task['private']['bindings'] = rename_references(task['private']['bindings'], names)
    for key, variant in list(variants.items()):
        variant = legacy._active_identities(ground_apparatus_legs(
            transform_world(_prepare_primitives(variant), rigid, 'variant.' + key), entry.get('apparatus_floor_z_m')), entry)
        variant = rename_references(variant, names)
        variant = translate_active_apparatus(variant, entry.get('apparatus_translation_world_m'))
        for patch in variant['patches']:
            if patch.get('operation') == 'category_initialization':
                p = patch['parameters']
                p.update(scene_model=entry['scene_model'], room=entry['room'], source_identity=entry['source_identity'])
                p['bindings'] = deepcopy(task['private']['bindings'])
            elif patch.get('operation') == 'prepare_category_interaction':
                patch['source_region'] = entry['room']
                if 'interaction_relocation_policy' in entry:
                    patch['relocation_policy'] = entry['interaction_relocation_policy']
            elif patch.get('operation') == 'robot_pose' and 'robot_start' in entry:
                patch['position'] = deepcopy(entry['robot_start']['position'])
                patch['yaw_deg'] = float(Rotation.from_quat(entry['robot_start']['orientation']).as_euler('xyz', degrees=True)[2])
        # These patches correct replay's hardcoded identity orientations. They run
        # after source postprocessing and before initialization installs phases.
        _restore_poses(variant)
        _insert_native_furniture_hook(variant, entry, key)
        variants[key] = variant
    if 'environment' in entry:
        task['public']['environment'] = _text(entry, 'environment')
    task['repair'].pop('translation_m', None)
    task['repair'].update(transform=deepcopy(entry['transform']), support_name_map=deepcopy(entry['support_name_map']),
        original_variant_policy='Retained the schema-required empty original placeholder with the new authored source; physical review uses the transformed saved/adapted variants.',
        rotation_pose_restoration='Initialization-only object_pose patches run before temporal phase installation.',
        world_bbox_expectations='source_objects bbox_size_m stores rotated world AABB extents; constructor scales/local dimensions are preserved.')
    if 'apparatus_floor_z_m' in entry:
        task['repair'].update(apparatus_floor_z_m=entry['apparatus_floor_z_m'],
            apparatus_leg_policy='Only collection/spare table leg bottoms are aligned to the explicit floor; transformed tops and XY are preserved.')
    for field in ('apparatus_translation_world_m', 'interaction_relocation_policy'):
        if field in entry:
            task['repair'][field] = deepcopy(entry[field])
    scene.update(scene=deepcopy(task['scene']), bindings=deepcopy(task['private']['bindings']))
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    return task, scene


def build_package(design_file, manifest_output):
    """Render fresh authored sources, task/scene copies and a pending manifest."""
    design_file, manifest_output = repo_path(design_file), repo_path(manifest_output)
    design_bytes = design_file.read_bytes()
    design = json.loads(design_bytes)
    if design.get('schema') != 'wia_furnished_temporal_design_v1' or not design.get('entries'):
        raise ValueError('An explicit wia_furnished_temporal_design_v1 entries design is required.')
    definitions, records, rendered, identities, groups = [], [], {}, set(), {}
    protected = {design_file, manifest_output}
    for entry in design['entries']:
        _entry_contract(entry)
        group = entry.get('matched_group')
        if group is not None:
            controls = {key: entry.get(key) for key in ('transform', 'support_name_map', 'robot_start', 'scene_model', 'room', 'floor_name', 'apparatus_floor_z_m', 'native_object_poses', 'native_furniture_hook_boundaries', 'apparatus_translation_world_m', 'interaction_relocation_policy')}
            if group in groups and groups[group] != controls:
                raise ValueError('Matched-group transform/support/robot/scene controls must be identical.')
            groups[group] = deepcopy(controls)
        task_path, scene_path = (repo_path(entry['parent_' + kind + '_file']) for kind in ('task', 'scene'))
        loaded_task, loaded_scene = load_task(task_path, variant='adapted', run_mode='diagnostic'), load_scene(scene_path, variant='adapted', run_mode='diagnostic')
        if loaded_task.source_path != loaded_scene.source_path:
            raise ValueError('Parent task and scene must resolve the same immutable source.')
        protected.update((task_path, scene_path, loaded_task.source_path))
        parent_task, parent_scene = (yaml.load(path.read_text(), Loader=UniqueLoader) for path in (task_path, scene_path))
        source_identity = Path(entry['source_identity'])
        source_path = (sources.AUTHORED_ROOT / source_identity).resolve()
        if (source_identity.is_absolute() or source_identity.suffix != '.json'
                or not source_path.is_relative_to(sources.AUTHORED_ROOT.resolve())
                or (sources.UPSTREAM_ROOT / source_identity).is_file()):
            raise ValueError('New source identity must remain project-owned and cannot shadow upstream sources.')
        provenance = {'design_file': str(design_file), 'design_sha256': sha256(design_bytes),
            'parent_task_file': str(task_path), 'parent_task_sha256': loaded_task.file_sha256,
            'parent_scene_file': str(scene_path), 'parent_scene_sha256': loaded_scene.file_sha256,
            'parent_source_identity': parent_task['source']['json'], 'parent_source_file': str(loaded_task.source_path),
            'parent_source_sha256': loaded_task.source_sha256}
        raw = translate_source(json.loads(loaded_task.source_path.read_bytes()), entry, provenance)
        runtime = loaded_scene.definition.runtime.model_dump(mode='json')
        task, scene = translate_definitions(parent_task, parent_scene, entry, provenance, runtime)
        identity = (entry['document_alias'], entry['variant_id'])
        if identity in identities:
            raise ValueError('Every document_alias/variant_id pair must be unique.')
        identities.add(identity)
        outputs = {kind: repo_path(entry['output_' + kind + '_file']) for kind in ('task', 'scene')}
        artifacts = [(source_path, (json.dumps(raw, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())]
        artifacts.extend((outputs[kind], yaml.safe_dump(value, sort_keys=False, allow_unicode=False, width=88).encode())
                         for kind, value in (('task', task), ('scene', scene)))
        for path, data in artifacts:
            if path in rendered or path in protected:
                raise ValueError('Every output must be distinct from all immutable inputs and other outputs.')
            rendered[path] = data
        definitions.append({'task': task, 'scene': scene, 'source': raw, 'task_file': outputs['task'],
                            'scene_file': outputs['scene'], 'source_file': source_path})
        records.append({**deepcopy(entry), **provenance, 'task_file': str(outputs['task']), 'scene_file': str(outputs['scene']),
            'output_task_file': str(outputs['task']), 'output_scene_file': str(outputs['scene']),
            'source_file': str(source_path), 'source_sha256': sha256(rendered[source_path]),
            'task_sha256': sha256(rendered[outputs['task']]), 'scene_sha256': sha256(rendered[outputs['scene']]),
            'variant': 'adapted', 'seed': loaded_task.seed, 'task_id': task['id'], 'task_version': task['version'],
            'module': MODULE, 'revision': task['private']['evaluation']['parameters']['task_revision'],
            'full_reference_execution': 'pending', **{key: False for key in NATIVE_FLAGS}})
        loaded_task.verify_unchanged()
        loaded_scene.verify_unchanged()
    if protected.intersection(rendered):
        raise ValueError('A generated output collides with an immutable input.')
    manifest = {'schema': 'wia_variant_manifest_v1', 'artifact_path_base': 'absolute', 'status': 'pending_native_review',
        'design_file': str(design_file), 'design_sha256': sha256(design_bytes), 'full_reference_execution': 'pending',
        **{key: False for key in NATIVE_FLAGS}, 'entries': records}
    rendered[manifest_output] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    return definitions, manifest, rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design-file', required=True, type=Path)
    parser.add_argument('--manifest-output', required=True, type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    definitions, manifest, rendered = build_package(args.design_file, args.manifest_output)
    if args.check:
        differences = [path for path, data in rendered.items() if path.is_symlink() or not path.is_file() or path.read_bytes() != data]
        if differences:
            for path in differences:
                print('Furnished temporal output differs or is missing: ' + str(path), file=sys.stderr)
            return 1
    else:
        for path, data in rendered.items():
            if path.is_symlink() or (path.exists() and (not path.is_file() or path.read_bytes() != data)):
                raise ValueError('Refusing to overwrite a differing generated file: ' + str(path))
        for path, data in rendered.items():
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    legacy.validate_outputs(definitions)
    print('Furnished temporal variants are current: ' + str(len(manifest['entries'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
