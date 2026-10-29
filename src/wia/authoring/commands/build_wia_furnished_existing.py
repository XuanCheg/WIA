"""Version existing Action Order/Stacking tasks on explicit native furniture.

Run: python -B scripts/build_wia_furnished_existing.py --design-file PATH
  --output-dir PATH [--check]
The design owns every placement and original root reference. Only CPU schema
and diagnostic-loader validation runs; no native actions or captures are used.


Canonical invocation: python -m wia.authoring.commands.build_wia_furnished_existing --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

import yaml

from wia.resources import ROOT
from wia.authoring.commands._inputs import project_path

from wia.authoring.converters.wia_furnished_layout import MODULE, REVISION, validate_parameters
from wia.tasks.loader import UniqueLoader, load_scene, load_task, source_scene_room
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition, SceneRuntime
from wia.tasks.sources import resolve_source

SUPPORTED_MODULES = {'wia.authoring.converters.action_order', 'wia.authoring.converters.physical_dynamics_stacking_stability'}


def repo_path(value):
    path = Path(value).expanduser()
    return project_path(path, root=ROOT)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _text(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Every entry requires an explicit ' + key + '.')
    return value


def translate_definitions(parent_task, parent_scene, entry, provenance):
    """Retain the complete parent task contract while changing initialization."""
    CategoryTaskDefinition.model_validate(parent_task)
    SceneDefinition.model_validate(parent_scene)
    evaluation = parent_task['private']['evaluation']
    if evaluation.get('type') != 'category_operation' or evaluation.get('module') not in SUPPORTED_MODULES:
        raise ValueError('Only existing AP-1 Action Order and AP-2 Stacking evaluators are supported.')
    task, scene = deepcopy(parent_task), deepcopy(parent_scene)
    layout = deepcopy(entry['layout'])
    if len(layout['rooms']) != 1:
        raise ValueError('The design must explicitly select one native scene room.')
    identity = {'scene_model': layout['scene_model'], 'room': layout['rooms'][0]}
    parameters = {'revision': REVISION, 'scene_contract': identity, 'layout': layout,
                  'source_reference_positions': deepcopy(entry['source_reference_positions'])}
    validate_parameters(parameters)
    objects = {row['name']: row for row in layout['objects']}
    bindings = task['private']['bindings']
    refs = set(entry['source_reference_positions'])
    bound_targets = {binding['name'] for binding in bindings.values() if binding['name'] in refs}
    if bound_targets != refs:
        raise ValueError('Original root references must identify exact existing parent task targets.')
    for key, name in entry.get('floor_binding_updates', {}).items():
        if (key not in bindings or (key not in ('floor', 'source_floor')
                and bindings[key].get('category') not in ('floor', 'floors')) or not isinstance(name, str) or not name):
            raise ValueError('Only existing explicit native floor bindings may be updated.')
        bindings[key]['name'] = name
    for binding in bindings.values():
        if binding['name'] not in refs:
            continue
        obj = objects[binding['name']]
        if any(binding.get(key) is not None and obj.get(key) != binding[key] for key in ('category', 'model')):
            raise ValueError('Authored target inventory must preserve the exact parent category and model.')
    # Replacing old source-postprocessing and search patches must not undo the
    # declared movable/collidable construction flags of the existing task.
    for patch in parent_task['scene']['variants']['adapted']['patches']:
        if patch.get('operation') != 'object_physics':
            continue
        name = parent_task['private']['bindings'][patch['binding']]['name']
        if name not in objects:
            raise ValueError('Every parent physics-patched object needs its explicit authored layout row.')
        for key in ('visual_only', 'fixed_base'):
            if patch.get(key) is not None and objects[name].get(key) is not patch[key]:
                raise ValueError('Authored layout must preserve parent object physics: ' + name + ':' + key)
    public = entry.get('public', {})
    if set(public) - {'environment', 'objective'}:
        raise ValueError('Only main-authored public environment/objective overrides are allowed.')
    task['public'].update(deepcopy(public))
    background = {'revision': 'wia_background_v1', 'mode': 'complete_ground',
                  'expected_scene': identity['scene_model'], 'disable_grid': False}
    patches = []
    for phase in ('before_construction', 'after_postprocessing'):
        patches.append({'operation': 'category_initialization', 'phase': phase,
                        'module': MODULE, 'parameters': deepcopy(parameters)})
        patches.append({'operation': 'category_initialization', 'phase': phase,
                        'module': 'wia.authoring.converters.wia_background', 'parameters': deepcopy(background)})
    task['scene'] = {'policy': 'native_then_minimal_patch_else_skip',
                     'variants': {'original': {'patches': []}, 'adapted': {'patches': patches}}}
    task.update(id=parent_task['id'] + '__' + entry['variant_id'], version=parent_task['version'] + 1,
                status='pending', source={'provider': 'esi', 'json': entry['source_identity']})
    runtime = SceneRuntime.model_validate(parent_scene.get('runtime', {})).model_dump(mode='json')
    task['repair'] = {**deepcopy(parent_task.get('repair', {})), **deepcopy(provenance),
        'contract_revision': REVISION, 'document_alias': entry['document_alias'],
        'variant_id': entry['variant_id'], 'label': entry['label'], 'scene_contract': deepcopy(identity),
        'runtime': runtime, 'native_validation_run': False, 'physical_success_verified': False,
        'full_reference_execution': 'deferred_until_user_review',
        'parent_evaluator_parameters_preserved': True, 'parent_seed_and_budget_preserved': True}
    for key, value in list(task['repair'].items()):
        if key.startswith('native_') and type(value) is bool:
            task['repair'][key] = False
    scene.update(id=parent_scene['id'] + '__' + entry['variant_id'], version=parent_scene['version'] + 1,
                 status='pending', source=deepcopy(task['source']), scene=deepcopy(task['scene']),
                 bindings=deepcopy(bindings), target=None, target_bindings=[], qualification=None,
                 runtime=runtime, seed=task['defaults']['seed'])
    if task['private']['evaluation'] != evaluation or task['defaults'] != parent_task['defaults']:
        raise ValueError('The parent evaluator, scoring, seed and budget must remain byte-for-byte equivalent as values.')
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    return task, scene


def build_package(design_file, output_dir):
    design_file, output_dir = repo_path(design_file), repo_path(output_dir)
    design_bytes = design_file.read_bytes()
    design = json.loads(design_bytes)
    if design.get('schema') != 'wia_furnished_existing_design_v1' or not design.get('entries'):
        raise ValueError('An explicit wia_furnished_existing_design_v1 entries design is required.')
    definitions, entries, rendered, identities, protected = [], [], {}, set(), {design_file}
    for row in design['entries']:
        for key in ('document_alias', 'variant_id', 'label', 'parent_task_file', 'parent_scene_file', 'source_identity'):
            _text(row, key)
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', row['variant_id']):
            raise ValueError('variant_id must be an explicit safe filename token.')
        alias_token = row['document_alias'].lower().replace('-', '_')
        if not re.fullmatch(r'[a-z0-9_]+', alias_token):
            raise ValueError('document_alias must be a safe task alias.')
        identity = (row['document_alias'], row['variant_id'])
        if identity in identities:
            raise ValueError('Every alias/version pair must be unique.')
        identities.add(identity)
        parent_task_path, parent_scene_path = (repo_path(row['parent_' + key + '_file']) for key in ('task', 'scene'))
        parent_task, parent_scene = (yaml.load(path.read_text(encoding='utf-8'), Loader=UniqueLoader)
                                     for path in (parent_task_path, parent_scene_path))
        loaded_task = load_task(parent_task_path, variant='adapted', run_mode='diagnostic')
        loaded_scene = load_scene(parent_scene_path, variant='adapted', run_mode='diagnostic')
        source_path = resolve_source(row['source_identity'])
        source_bytes = source_path.read_bytes()
        if source_scene_room(row['source_identity'], json.loads(source_bytes)) != (
                row['layout']['scene_model'], row['layout']['rooms'][0]):
            raise ValueError('The explicit background source must match the native scene and room.')
        protected.update((parent_task_path, parent_scene_path, source_path))
        provenance = {'design_file': str(design_file), 'design_sha256': sha256(design_bytes),
            'parent_task_file': str(parent_task_path), 'parent_task_sha256': loaded_task.file_sha256,
            'parent_scene_file': str(parent_scene_path), 'parent_scene_sha256': loaded_scene.file_sha256,
            'source_file': str(source_path), 'source_sha256': sha256(source_bytes),
            'source_role': 'Native background and provenance only; immutable parent task contract retained.'}
        task, scene = translate_definitions(parent_task, parent_scene, row, provenance)
        stem = alias_token + '__' + row['variant_id']
        task_path, scene_path = output_dir / 'tasks' / (stem + '.yaml'), output_dir / 'scenes' / (stem + '.yaml')
        for path, value in ((task_path, task), (scene_path, scene)):
            if path in rendered:
                raise ValueError('Output paths must be distinct.')
            rendered[path] = yaml.safe_dump(value, sort_keys=False, allow_unicode=False, width=88).encode('utf-8')
        definitions.append({'task_file': task_path, 'scene_file': scene_path})
        entries.append({**provenance, 'document_alias': row['document_alias'], 'variant_id': row['variant_id'],
            'label': row['label'], 'variant': 'adapted', 'task_file': str(task_path), 'scene_file': str(scene_path),
            'task_sha256': sha256(rendered[task_path]), 'scene_sha256': sha256(rendered[scene_path]),
            'task_version': task['version'], 'seed': task['defaults']['seed'],
            'module': task['private']['evaluation']['module'], 'native_validation_run': False,
            'physical_success_verified': False, 'full_reference_execution': 'deferred_until_user_review'})
        loaded_task.verify_unchanged()
        loaded_scene.verify_unchanged()
    manifest = {'schema': 'wia_variant_manifest_v1', 'artifact_path_base': 'absolute',
        'status': 'pending_initialization_image_review', 'design_file': str(design_file),
        'design_sha256': sha256(design_bytes), 'revision': REVISION, 'entries': entries,
        'native_validation_run': False, 'physical_success_verified': False,
        'full_reference_execution': 'deferred_until_user_review'}
    rendered[output_dir / 'manifest.json'] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if protected & rendered.keys():
        raise ValueError('Generated definitions must never overwrite immutable inputs.')
    return definitions, manifest, rendered


def validate_outputs(definitions):
    for row in definitions:
        task = load_task(row['task_file'], variant='adapted', run_mode='diagnostic')
        scene = load_scene(row['scene_file'], variant='adapted', run_mode='diagnostic')
        if (task.definition.scene != scene.definition.scene or task.seed != scene.seed
                or task.definition.private.bindings != scene.definition.bindings
                or (task.scene_name, task.room_name) != (scene.scene_name, scene.room_name)):
            raise ValueError('Generated task and scene must load identical initialization contracts.')
        task.verify_unchanged()
        scene.verify_unchanged()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    definitions, manifest, rendered = build_package(args.design_file, args.output_dir)
    mismatches = [path for path, data in rendered.items() if path.is_symlink()
                  or not path.is_file() or path.read_bytes() != data]
    if args.check:
        if mismatches:
            for path in mismatches:
                print('Furnished existing-task definition differs or is missing: ' + str(path), file=sys.stderr)
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
    print(f'Furnished existing-task definitions are current: {len(manifest["entries"])} variants.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
