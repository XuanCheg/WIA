"""Build deterministic pending spatial-tray versions from an explicit design.

Run: python -B scripts/build_wia_spatial_tray.py --design-file PATH
  --output-dir PATH [--check]
The design supplies every scene, native binding, layout and scoring threshold.
This builder runs CPU schema/loader checks only and preserves immutable parents.


Canonical invocation: python -m wia.authoring.commands.build_wia_spatial_tray --help
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

from wia.authoring.converters.wia_spatial_tray import MODULE, OPERATION, _layout_contract, validate_contract
from wia.tasks.loader import UniqueLoader, load_scene, load_task, source_scene_room
from wia.tasks.schema import Binding, CategoryTaskDefinition, SceneDefinition, SceneRuntime
from wia.tasks.sources import resolve_source
from wia.tasks.wia_spatial_tray_scoring import REVISION

CAPABILITIES = ['base_translation', 'base_rotation', 'ee_translation', 'ee_rotation', 'trunk_height', 'gripper']


def repo_path(value):
    path = Path(value).expanduser()
    return project_path(path, root=ROOT)


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def _text(entry, key):
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Every entry requires explicit nonempty ' + key + '.')
    return value


def translate_definitions(parent_task, parent_scene, entry, provenance):
    """Copy parents and install only the explicit new task and initialization."""
    CategoryTaskDefinition.model_validate(parent_task)
    SceneDefinition.model_validate(parent_scene)
    task, scene = deepcopy(parent_task), deepcopy(parent_scene)
    p, layout = (deepcopy(entry[key]) for key in ('parameters', 'layout'))
    bindings = {key: Binding.model_validate({'name': value} if isinstance(value, str) else value).model_dump(
                    mode='json', exclude_none=True) for key, value in entry['bindings'].items()}
    native_names = {key: value['name'] for key, value in bindings.items()}
    identity = p['scene_contract']
    evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': OPERATION,
                  'binding_keys': list(bindings), 'parameters': p}
    validate_contract({**evaluation, 'bindings': native_names,
                       'scene': identity['scene_model'], 'room': identity['room']})
    initialization = {'revision': REVISION, 'scene_contract': deepcopy(identity), 'layout': layout}
    _layout_contract(initialization)
    expected = {name: native_names[p['initial_supports'][key]] for key, name in p['targets'].items()}
    actual = {row['name']: row['support'] for row in layout.get('supported_items', [])}
    if any(actual.get(name) != support for name, support in expected.items()):
        raise ValueError('Every target must initialize on its explicitly bound separate native support.')
    patches = [{'operation': 'category_initialization', 'phase': phase, 'module': MODULE,
                'parameters': deepcopy(initialization)}
               for phase in ('before_construction', 'after_postprocessing')]
    scene_config = {'policy': 'native_then_minimal_patch_else_skip',
                    'variants': {'original': {'patches': []}, 'adapted': {'patches': patches}}}
    public = entry.get('public')
    if not isinstance(public, dict) or not public.get('objective') or not public.get('success_criteria'):
        raise ValueError('Explicit public objective and success_criteria overrides are required.')
    task['public'].update(deepcopy(public))
    task['id'] = parent_task['id'] + '__' + entry['variant_id']
    task.update(version=entry.get('version', parent_task['version'] + 1), status='pending', scene=scene_config)
    if task['version'] <= parent_task['version']:
        raise ValueError('A spatial-tray version must be newer than its immutable parent.')
    task['source'] = {'provider': 'esi', 'json': entry['source_identity']}
    task['private'] = {'bindings': bindings, 'evaluation': evaluation}
    task['requirements'] = {'capabilities': deepcopy(CAPABILITIES)}
    task['qualification'] = {'on_failure': 'skip', 'target_movable_collidable': True,
                             'support': None, 'initial_visibility': 'none'}
    task['defaults'].update(deepcopy(entry.get('defaults', {})))
    if 'seed' in entry:
        task['defaults']['seed'] = entry['seed']
    runtime = SceneRuntime.model_validate(entry.get('runtime', parent_scene.get('runtime', {}))).model_dump(mode='json')
    # Parent qualification receipts concern a different task; preserve them only
    # in the immutable parent, and retain source/parent hashes in this version.
    task['repair'] = {**deepcopy(provenance), 'contract_revision': REVISION,
        'document_alias': entry['document_alias'], 'variant_id': entry['variant_id'], 'label': entry['label'],
        'scene_contract': deepcopy(identity), 'runtime': deepcopy(runtime),
        'native_validation_run': False, 'native_visibility_verified': False,
        'manipulation_verified': False, 'full_reference_execution': 'deferred_until_user_review'}
    scene.update(id=parent_scene['id'] + '__' + entry['variant_id'], version=task['version'], status='pending',
                 source=deepcopy(task['source']), target=None, target_bindings=[], qualification=None,
                 bindings=deepcopy(bindings), scene=deepcopy(scene_config), runtime=runtime,
                 seed=task['defaults']['seed'])
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    return task, scene


def build_package(design_file, output_dir):
    """Return deterministic bytes; never choose native scenes or placements."""
    design_file, output_dir = repo_path(design_file), repo_path(output_dir)
    design_bytes = design_file.read_bytes()
    design = json.loads(design_bytes)
    if design.get('schema') != 'wia_spatial_tray_design_v1' or not design.get('entries'):
        raise ValueError('Provide explicit wia_spatial_tray_design_v1 entries.')
    definitions, manifest_entries, rendered, identities = [], [], {}, set()
    protected = {design_file}
    for entry in design['entries']:
        for key in ('document_alias', 'variant_id', 'label', 'parent_task_file', 'parent_scene_file', 'source_identity'):
            _text(entry, key)
        if not re.fullmatch(r'T(?:2[6-9]|3[0-4])', entry['document_alias']):
            raise ValueError('This bounded adapter supports only document aliases T26 through T34.')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', entry['variant_id']):
            raise ValueError('variant_id must be a safe explicit filename token.')
        identity = (entry['document_alias'], entry['variant_id'])
        if identity in identities:
            raise ValueError('Every document_alias and variant_id pair must be unique.')
        identities.add(identity)
        parent_task_path, parent_scene_path = (repo_path(entry['parent_' + kind + '_file']) for kind in ('task', 'scene'))
        protected.update((parent_task_path, parent_scene_path))
        parent_task, parent_scene = (yaml.load(path.read_text(encoding='utf-8'), Loader=UniqueLoader)
                                     for path in (parent_task_path, parent_scene_path))
        loaded_task = load_task(parent_task_path, variant='adapted', run_mode='diagnostic')
        loaded_scene = load_scene(parent_scene_path, variant='adapted', run_mode='diagnostic')
        source_path = resolve_source(entry['source_identity'])
        protected.add(source_path)
        source_bytes = source_path.read_bytes()
        native_identity = source_scene_room(entry['source_identity'], json.loads(source_bytes))
        contract = entry['parameters']['scene_contract']
        if native_identity != (contract['scene_model'], contract['room']):
            raise ValueError('The explicitly selected source must match the declared scene and room.')
        provenance = {'design_file': str(design_file), 'design_sha256': sha256(design_bytes),
            'parent_task_file': str(parent_task_path), 'parent_task_sha256': loaded_task.file_sha256,
            'parent_scene_file': str(parent_scene_path), 'parent_scene_sha256': loaded_scene.file_sha256,
            'source_file': str(source_path), 'source_sha256': sha256(source_bytes)}
        task, scene = translate_definitions(parent_task, parent_scene, entry, provenance)
        stem = entry['document_alias'].lower() + '__' + entry['variant_id']
        task_path, scene_path = output_dir / 'tasks' / (stem + '.yaml'), output_dir / 'scenes' / (stem + '.yaml')
        for path, value in ((task_path, task), (scene_path, scene)):
            if path in rendered:
                raise ValueError('Every generated file must have a distinct path.')
            rendered[path] = yaml.safe_dump(value, sort_keys=False, allow_unicode=False, width=88).encode('utf-8')
        manifest_entries.append({**provenance, 'document_alias': entry['document_alias'],
            'variant_id': entry['variant_id'], 'label': entry['label'],
            'task_file': str(task_path), 'scene_file': str(scene_path), 'variant': 'adapted',
            'task_sha256': sha256(rendered[task_path]), 'scene_sha256': sha256(rendered[scene_path]),
            'task_id': task['id'], 'task_version': task['version'], 'seed': task['defaults']['seed'],
            'revision': REVISION, 'native_validation_run': False,
            'manipulation_verified': False, 'full_reference_execution': 'deferred_until_user_review'})
        definitions.append({'task_file': task_path, 'scene_file': scene_path})
        loaded_task.verify_unchanged()
        loaded_scene.verify_unchanged()
    manifest = {'schema': 'wia_variant_manifest_v1', 'artifact_path_base': 'absolute',
        'status': 'pending_initialization_image_review', 'design_file': str(design_file),
        'design_sha256': sha256(design_bytes), 'revision': REVISION, 'native_validation_run': False,
        'manipulation_verified': False, 'full_reference_execution': 'deferred_until_user_review',
        'entries': manifest_entries}
    rendered[output_dir / 'manifest.json'] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if protected & rendered.keys():
        raise ValueError('Generated paths must never overwrite source, parent or design inputs.')
    return definitions, manifest, rendered


def validate_outputs(definitions):
    """Use production diagnostic loaders, without starting native simulation."""
    for entry in definitions:
        task = load_task(entry['task_file'], variant='adapted', run_mode='diagnostic')
        scene = load_scene(entry['scene_file'], variant='adapted', run_mode='diagnostic')
        if (task.definition.scene != scene.definition.scene or task.seed != scene.seed
                or task.definition.private.bindings != scene.definition.bindings
                or task.scene_name != scene.scene_name or task.room_name != scene.room_name):
            raise ValueError('Task and scene must load identical native contracts.')
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
                print('Spatial-tray output differs or is missing: ' + str(path), file=sys.stderr)
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
    print(f'Spatial-tray definitions are current: {len(manifest["entries"])} explicit variants.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
