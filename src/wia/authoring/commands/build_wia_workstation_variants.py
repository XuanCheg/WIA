"""Translate explicitly authored workstation variants without running simulation.

Run: python -B scripts/build_wia_workstation_variants.py --design-file PATH
  --manifest-output PATH [--check]
Design paths are repository-relative or absolute; emitted manifest paths are
absolute. Existing differing outputs are never overwritten.


Canonical invocation: python -m wia.authoring.commands.build_wia_workstation_variants --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml

from wia.resources import ROOT
from wia.authoring.commands._inputs import project_path

from wia.tasks.loader import UniqueLoader, load_scene, load_task, source_scene_room
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition, SceneRuntime
from wia.tasks.sources import resolve_source

CONTRACTS = {
    'core.esi_converters.wia_dimensional': ('wia_dimensional_v3', 'core.esi_converters.active_layout'),
    'core.esi_converters.wia_mirror': ('wia_mirror_v2', 'core.esi_converters.wia_mirror'),
    'core.esi_converters.inclined_comparison': ('controlled_ramp_comparison_v1', 'core.esi_converters.active_layout'),
}
RAMP_MODULE = 'core.esi_converters.inclined_comparison'
RAMP_PARAMETERS = {'revision', 'candidate_pairs', 'ramp_dimensions_m', 'orientation_free_keys',
    'release_local_x_m', 'trial_duration_s', 'arrival_tie_tolerance_s', 'safe_bounds_m',
    'apparatus_keys', 'trial_support_keys', 'static_friction', 'dynamic_friction'}
RAMP_PROVENANCE = {'source_ramp_geometry_and_friction_preserved', 'original_source_room', 'authored_room'}
NATIVE_FLAGS = ('native_validation_run', 'native_geometry_verified', 'native_visibility_verified',
                'native_shape_verified', 'native_hand_path_verified', 'native_gripper_path_verified',
                'native_optics_verified', 'native_selected_identity_verified',
                'native_non_goal_initialization_verified', 'physical_success_verified',
                'scientific_controls_verified', 'continuous_path_certified')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def repo_path(value):
    path = Path(value).expanduser()
    return project_path(path, root=ROOT)


def _vector(value, size, field):
    if (not isinstance(value, list) or len(value) != size
            or any(isinstance(number, bool) or not isinstance(number, (int, float))
                   or not math.isfinite(number) for number in value)):
        raise ValueError(field + ' must be an explicit finite numeric vector.')
    return value


def _text(entry, key):
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Every variant requires an explicit nonempty ' + key + '.')
    return value


def _validate_ramp_contract(evaluation, patches):
    """Reject unreviewed coordinate fields rather than leave them untranslated."""
    p = evaluation['parameters']
    if (evaluation.get('operation') != 'controlled_trials_then_spare_delivery'
            or p.get('revision') != 'controlled_ramp_comparison_v1'
            or not RAMP_PARAMETERS.issubset(p)
            or set(p) - RAMP_PARAMETERS - RAMP_PROVENANCE - {'scene_contract'}):
        raise ValueError('Ramp transfer requires the exact reviewed controlled comparison parameter format.')
    bounds = p['safe_bounds_m']
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError('Ramp safe_bounds_m must contain exactly two world-space corners.')
    low, high = (_vector(corner, 3, 'safe_bounds_m corner') for corner in bounds)
    if any(lo >= hi for lo, hi in zip(low, high)):
        raise ValueError('Ramp safe_bounds_m corners must define positive extents.')
    for pair in p['candidate_pairs']:
        if set(pair) != {'label', 'trial', 'spare'}:
            raise ValueError('Ramp candidate pairs must contain only label, trial and spare identities.')
    for patch in patches:
        parameters = patch['parameters']
        if set(parameters) - {'revision', 'layout', 'scene_contract'}:
            raise ValueError('Ramp initialization contains unsupported fields that may require translation.')
        layout = parameters['layout']
        if set(layout) - {'scene_model', 'rooms', 'objects', 'supported_items', 'floor_name',
                          'friction', 'robot_start', 'shared_layout_id'}:
            raise ValueError('Ramp layout contains unsupported fields that may require translation.')
        if set(layout['robot_start']) != {'position', 'orientation'}:
            raise ValueError('Ramp robot_start must use the reviewed position and orientation format.')
        for obj in layout['objects']:
            if set(obj) - {'type', 'name', 'category', 'model', 'primitive_type', 'size', 'scale',
                           'position', 'orientation', 'fixed_base', 'visual_only', 'rgba', 'radius', 'mass'}:
                raise ValueError('Ramp object contains unsupported fields that may require translation.')
        for support in layout['supported_items']:
            if set(support) != {'name', 'support'}:
                raise ValueError('Ramp supported items must contain only native identities.')
        for material in layout['friction'].values():
            if set(material) - {'static_friction', 'dynamic_friction', 'restitution'}:
                raise ValueError('Ramp material contains unsupported fields that may require translation.')


def translate_definitions(parent_task, parent_scene, entry, provenance):
    """Copy the parent and apply only the declared translation and named overrides."""
    task, scene = deepcopy(parent_task), deepcopy(parent_scene)
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    evaluation = task['private']['evaluation']
    contract = CONTRACTS.get(evaluation.get('module'))
    ramp = evaluation.get('module') == RAMP_MODULE
    if evaluation.get('type') != 'category_operation' or contract is None:
        raise ValueError('Only existing dimensional, mirror or reviewed ramp category operations may be transferred.')
    if ramp:
        # Historical ramp definitions omit runtime; retain their effective defaults.
        runtime = SceneRuntime.model_validate(scene.get('runtime', {}))
        task.setdefault('repair', {}).setdefault('runtime', runtime.model_dump(mode='json'))
    if (task['scene'] != scene['scene'] or task['private']['bindings'] != scene['bindings']
            or task['defaults']['seed'] != scene['seed']
            or SceneRuntime.model_validate(task['repair']['runtime']) != SceneRuntime.model_validate(scene.get('runtime', {}))):
        raise ValueError('Parent task and scene must have identical patches, bindings, seed and runtime.')
    for key in ('document_alias', 'variant_id', 'label', 'source_identity', 'scene_model', 'room', 'floor_name'):
        _text(entry, key)
    delta = _vector(entry['translation_m'], 3, 'translation_m')
    overrides = entry.get('object_overrides', {})
    if not isinstance(overrides, dict):
        raise ValueError('object_overrides must explicitly map existing native object names.')
    if ramp and overrides:
        raise ValueError('Ramp transfer supports only rigid translation; object_overrides are not supported.')
    revision, initializer = contract
    identity = {'scene_model': entry['scene_model'], 'room': entry['room']}
    patches = task['scene']['variants']['adapted']['patches']
    if (len(patches) != 2 or {patch.get('phase') for patch in patches}
            != {'before_construction', 'after_postprocessing'}
            or any(patch.get('operation') != 'category_initialization' or patch.get('module') != initializer
                   for patch in patches)):
        raise ValueError('The adapted parent must retain both exact owned category initialization hooks.')
    if patches[0]['parameters'] != patches[1]['parameters']:
        raise ValueError('Both parent initialization hooks must have identical parameters.')
    if ramp:
        _validate_ramp_contract(evaluation, patches)
    for patch in patches:
        parameters = patch['parameters']
        if not ramp:
            parameters.update(revision=revision, scene_contract=deepcopy(identity))
        elif 'scene_contract' in parameters:
            parameters['scene_contract'] = deepcopy(identity)
        layout = parameters['layout']
        shared_layout_id = layout.get('shared_layout_id', 'wia_ramp_workstation') if ramp else layout['shared_layout_id']
        layout.update(scene_model=entry['scene_model'], rooms=[entry['room']], floor_name=entry['floor_name'],
                      shared_layout_id=shared_layout_id + '__' + entry['variant_id'])
        objects = {obj['name']: obj for obj in layout['objects']}
        if len(objects) != len(layout['objects']) or set(overrides) - objects.keys():
            raise ValueError('Overrides require distinct, explicitly existing native object names.')
        for obj in layout['objects']:
            obj['position'] = [value + shift for value, shift in
                               zip(_vector(obj['position'], 3, 'object position'), delta)]
        robot = layout['robot_start']
        robot['position'] = [value + shift for value, shift in
                             zip(_vector(robot['position'], 3, 'robot position'), delta)]
        for name, override in overrides.items():
            if (not isinstance(override, dict) or not override or set(override) - {'position', 'rgba'}):
                raise ValueError('Object overrides may change only explicit position and/or rgba.')
            for field, value in override.items():
                _vector(value, 3 if field == 'position' else 4, 'override ' + field)
                if field == 'rgba' and any(number < 0 or number > 1 for number in value):
                    raise ValueError('Override rgba components must be within [0, 1].')
                objects[name][field] = deepcopy(value)
    if not ramp:
        evaluation['parameters'].update(revision=revision, scene_contract=deepcopy(identity))
    else:
        evaluation['parameters']['safe_bounds_m'] = [[value + shift for value, shift in zip(corner, delta)]
                                                    for corner in evaluation['parameters']['safe_bounds_m']]
        if 'scene_contract' in evaluation['parameters']:
            evaluation['parameters']['scene_contract'] = deepcopy(identity)
    task.update(version=parent_task['version'] + 1, status='pending')
    task['source']['json'] = entry['source_identity']
    task['repair'].update(deepcopy(provenance), contract_revision=revision,
                         document_alias=entry['document_alias'], variant_id=entry['variant_id'], label=entry['label'],
                         translation_m=deepcopy(delta), object_overrides=deepcopy(overrides),
                         scene_contract=deepcopy(identity), full_reference_execution='pending')
    for key in tuple(task['repair']):
        if key.startswith('native_') and isinstance(task['repair'][key], bool):
            task['repair'][key] = False
    task['repair'].update({key: False for key in NATIVE_FLAGS})
    scene.update(id=parent_scene['id'] + '__' + entry['variant_id'], version=parent_scene['version'] + 1,
                 status='pending', source=deepcopy(task['source']), scene=deepcopy(task['scene']))
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    return task, scene


def build_package(design_file, manifest_output):
    """Render a deterministic package using only supplied sources and placements."""
    design_file, manifest_output = repo_path(design_file), repo_path(manifest_output)
    design_bytes = design_file.read_bytes()
    design = json.loads(design_bytes)
    if design.get('schema') != 'wia_workstation_transfer_design_v1' or not design.get('entries'):
        raise ValueError('An explicit wia_workstation_transfer_design_v1 entries manifest is required.')
    definitions, entries, rendered, identities = [], [], {}, set()
    for entry in design['entries']:
        parent_task_path, parent_scene_path = (repo_path(entry['parent_' + kind + '_file']) for kind in ('task', 'scene'))
        parent_task, parent_scene = (yaml.load(path.read_text(encoding='utf-8'), Loader=UniqueLoader)
                                     for path in (parent_task_path, parent_scene_path))
        loaded_task = load_task(parent_task_path, variant='adapted', run_mode='diagnostic')
        loaded_scene = load_scene(parent_scene_path, variant='adapted', run_mode='diagnostic')
        source_path = resolve_source(_text(entry, 'source_identity'))
        raw_source = json.loads(source_path.read_bytes())
        source_identity = source_scene_room(entry['source_identity'], raw_source)
        if source_identity != (_text(entry, 'scene_model'), _text(entry, 'room')):
            raise ValueError('The target source top-level scene and room must match the explicit variant.')
        provenance = {'design_file': str(design_file), 'design_sha256': sha256(design_bytes),
            'parent_task_file': str(parent_task_path), 'parent_task_sha256': loaded_task.file_sha256,
            'parent_scene_file': str(parent_scene_path), 'parent_scene_sha256': loaded_scene.file_sha256,
            'source_file': str(source_path), 'source_sha256': sha256(source_path.read_bytes())}
        task, scene = translate_definitions(parent_task, parent_scene, entry, provenance)
        identity = (entry['document_alias'], entry['variant_id'])
        if identity in identities:
            raise ValueError('Each document_alias and variant_id pair must be unique.')
        identities.add(identity)
        task_path, scene_path = (repo_path(entry['output_' + kind + '_file']) for kind in ('task', 'scene'))
        protected = {design_file, source_path, parent_task_path, parent_scene_path, manifest_output}
        for path, definition in ((task_path, task), (scene_path, scene)):
            if path in rendered or path in protected:
                raise ValueError('Every output must be distinct from other outputs and immutable inputs.')
            rendered[path] = yaml.safe_dump(definition, sort_keys=False, allow_unicode=False, width=88).encode('utf-8')
        row = {**deepcopy(entry), **provenance, 'task_file': str(task_path), 'scene_file': str(scene_path),
            'output_task_file': str(task_path), 'output_scene_file': str(scene_path),
            'task_sha256': sha256(rendered[task_path]), 'scene_sha256': sha256(rendered[scene_path]),
            'variant': 'adapted', 'seed': task['defaults']['seed'], 'task_id': task['id'], 'task_version': task['version'],
            'module': task['private']['evaluation']['module'], 'revision': task['private']['evaluation']['parameters']['revision'],
            'full_reference_execution': 'pending', **{key: False for key in NATIVE_FLAGS}}
        entries.append(row)
        definitions.append({'task': task, 'scene': scene, 'task_file': task_path, 'scene_file': scene_path})
        loaded_task.verify_unchanged()
        loaded_scene.verify_unchanged()
    manifest = {'schema': 'wia_variant_manifest_v1', 'artifact_path_base': 'absolute',
        'status': 'pending_native_review', 'design_file': str(design_file), 'design_sha256': sha256(design_bytes),
        'full_reference_execution': 'pending', **{key: False for key in NATIVE_FLAGS}, 'entries': entries}
    rendered[manifest_output] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode('utf-8')
    return definitions, manifest, rendered


def validate_outputs(definitions):
    """Load translated definitions through the existing diagnostic loaders."""
    for entry in definitions:
        task = load_task(entry['task_file'], variant='adapted', run_mode='diagnostic')
        scene = load_scene(entry['scene_file'], variant='adapted', run_mode='diagnostic')
        if (task.definition.scene != scene.definition.scene or task.seed != scene.seed
                or task.definition.private.bindings != scene.definition.bindings
                or task.scene_name != scene.scene_name or task.room_name != scene.room_name):
            raise ValueError('Translated task and scene must load identical initialization contracts.')
        task.verify_unchanged()
        scene.verify_unchanged()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design-file', type=Path, required=True)
    parser.add_argument('--manifest-output', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Compare generated bytes without writing.')
    args = parser.parse_args(argv)
    definitions, manifest, rendered = build_package(args.design_file, args.manifest_output)
    mismatches = [path for path, data in rendered.items() if path.is_symlink()
                  or not path.is_file() or path.read_bytes() != data]
    if args.check:
        if mismatches:
            for path in mismatches:
                print('WIA workstation variant differs or is missing: ' + str(path), file=sys.stderr)
            return 1
    else:
        for path, data in rendered.items():
            if path.is_symlink() or (path.exists() and (not path.is_file() or path.read_bytes() != data)):
                raise ValueError('Refusing to overwrite a differing generated file: ' + str(path))
        for path, data in rendered.items():
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    validate_outputs(definitions)
    print(f'WIA workstation variants are current: {len(manifest["entries"])} explicit variants.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
