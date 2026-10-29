"""Build explicit native-cloth variants without running simulation.

Run: python -B scripts/build_wia_cover_variants.py --design-file PATH
  --output-dir PATH [--check]
Inputs are repository-relative or absolute. Outputs are immutable, deterministic
tasks/<variant_id>.yaml, scenes/<variant_id>.yaml and manifest.json in output-dir.
Native asset availability, cooking, coverage and physical success remain pending.


Canonical invocation: python -m wia.authoring.commands.build_wia_cover_variants --help
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import yaml

from wia.resources import ROOT, ESI_ROOT
from wia.authoring.commands._inputs import project_path

from wia.tasks.loader import UniqueLoader, load_scene, load_task
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition, SceneRuntime

INITIALIZER = 'wia.authoring.converters.deformable_redrape'
CONTRACTS = {
    ('wia.authoring.converters.physical_structure_deformable', 'sweep_cloth_retrieve_item'),
    (INITIALIZER, 'uncover_select_and_deliver'),
}
NATIVE_FLAGS = ('native_validation_run', 'physical_success_verified')
FIELDS = {'document_alias', 'variant_id', 'label', 'parent_task_file', 'parent_scene_file',
          'cloth_asset', 'cloth_scale_multiplier', 'cloth_mass_kg', 'public_cover_label',
          'cloth_drop_clearance_m', 'cloth_drop_xy_offset_m'}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def repo_path(value):
    path = Path(value).expanduser()
    return project_path(path, root=ROOT)


def _text(entry, key):
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Every variant requires an explicit nonempty ' + key + '.')
    return value


def _number(value, field, *, positive=True):
    if (type(value) not in (float, int) or not math.isfinite(value)
            or (positive and value <= 0)):
        raise ValueError(field + ' must be an explicit finite' + (' positive' if positive else '') + ' number.')


def _validate_entry(entry):
    if not isinstance(entry, dict) or set(entry) - FIELDS:
        raise ValueError('Cover variants support only the explicitly reviewed design fields.')
    for field in ('document_alias', 'variant_id', 'label', 'parent_task_file', 'parent_scene_file', 'public_cover_label'):
        _text(entry, field)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', entry['variant_id']):
        raise ValueError('variant_id must be a single safe filename stem.')
    asset = entry.get('cloth_asset')
    if not isinstance(asset, dict) or set(asset) != {'category', 'model'}:
        raise ValueError('cloth_asset must contain exactly category and model.')
    for field in ('category', 'model'):
        _text(asset, field)
    for field in ('cloth_scale_multiplier', 'cloth_mass_kg'):
        _number(entry.get(field), field)
    if 'cloth_drop_clearance_m' in entry:
        _number(entry['cloth_drop_clearance_m'], 'cloth_drop_clearance_m')
    if 'cloth_drop_xy_offset_m' in entry:
        offset = entry['cloth_drop_xy_offset_m']
        if not isinstance(offset, list) or len(offset) != 2:
            raise ValueError('cloth_drop_xy_offset_m must be an explicit two-number vector.')
        for value in offset:
            _number(value, 'cloth_drop_xy_offset_m', positive=False)


def _initializers(definition):
    return [patch for patch in definition['scene']['variants']['adapted']['patches']
            if patch.get('operation') == 'category_initialization']


def validate_bootstrap():
    """Reject source construction changes that need a runtime integration review."""
    path = ESI_ROOT / 'src/active_explore/tasks/physical_structure/deformable.py'
    functions = {node.name: node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef)}
    body = functions['build_env_objects'].body
    if (len(body) != 1 or not isinstance(body[0], ast.Return)
            or not isinstance(body[0].value, ast.List) or body[0].value.elts):
        raise ValueError('Source bootstrap conflict: deformable build_env_objects is no longer empty; runtime review is required.')
    return {'source_builder_file': str(path), 'source_builder_sha256': sha256(path.read_bytes()),
            'source_bootstrap_object_inventory': 'empty',
            'source_postprocess_owner': INITIALIZER,
            'source_postprocess_policy': 'existing_joint_world_override'}


def _replace_authored_assets(value, cloth_name, asset):
    """Update named live specs; immutable source-asset provenance stays verbatim."""
    if isinstance(value, dict):
        if value.get('name') == cloth_name and 'category' in value and 'model' in value:
            value.update(asset)
        for key, item in value.items():
            if key != 'parent_source_cloth_asset':
                _replace_authored_assets(item, cloth_name, asset)
    elif isinstance(value, list):
        for item in value:
            _replace_authored_assets(item, cloth_name, asset)


def _public_label(value, label):
    if isinstance(value, str):
        return re.sub(r'\bcloth\b|\bcover\b', lambda match: label.capitalize()
                      if match.group().istitle() else label, value, flags=re.IGNORECASE)
    if isinstance(value, list):
        return [_public_label(item, label) for item in value]
    if isinstance(value, dict):
        return {key: _public_label(item, label) for key, item in value.items()}
    return value


def translate_definitions(parent_task, parent_scene, entry, provenance):
    """Preserve the parent contract except explicit cover initialization and naming."""
    _validate_entry(entry)
    CategoryTaskDefinition.model_validate(parent_task)
    SceneDefinition.model_validate(parent_scene)
    evaluation = parent_task['private']['evaluation']
    if ((evaluation.get('module'), evaluation.get('operation')) not in CONTRACTS
            or evaluation.get('type') != 'category_operation'):
        raise ValueError('Only reviewed native cloth reveal/select/retrieve parents are supported.')
    if (parent_task['scene'] != parent_scene['scene']
            or parent_task['private']['bindings'] != parent_scene['bindings']
            or parent_task['source'] != parent_scene['source']
            or parent_task['defaults']['seed'] != parent_scene['seed']):
        raise ValueError('Parent task and scene must have identical patches, bindings, source and seed.')
    runtime = parent_task.get('repair', {}).get('runtime')
    if runtime is not None and SceneRuntime.model_validate(runtime) != SceneRuntime.model_validate(parent_scene.get('runtime', {})):
        raise ValueError('Parent task and scene runtime differs.')
    hooks = _initializers(parent_task)
    if (len(hooks) != 2 or {hook.get('phase') for hook in hooks} != {'before_construction', 'after_postprocessing'}
            or any(hook.get('module') != INITIALIZER for hook in hooks)
            or hooks[0]['parameters'] != hooks[1]['parameters']):
        raise ValueError('Both identical native deformable_redrape initialization phases are required; source bootstrap conflict needs runtime review.')
    cloth = parent_task['private']['bindings']['cloth']
    parameters = hooks[0]['parameters']
    if parameters.get('cloth_name') != cloth['name'] or parameters.get('cloth_asset') != cloth:
        raise ValueError('The named cloth binding and authored initialization asset must agree.')
    task, scene = deepcopy(parent_task), deepcopy(parent_scene)
    _replace_authored_assets(task['private'], cloth['name'], entry['cloth_asset'])
    _replace_authored_assets(task['scene'], cloth['name'], entry['cloth_asset'])
    for hook in _initializers(task):
        for field in ('cloth_scale_multiplier', 'cloth_mass_kg', 'cloth_drop_clearance_m', 'cloth_drop_xy_offset_m'):
            if field in entry:
                hook['parameters'][field] = deepcopy(entry[field])
    task['private']['evaluation']['parameters']['authored_cloth_scale_multiplier'] = entry['cloth_scale_multiplier']
    task['public'] = _public_label(task['public'], entry['public_cover_label'])
    task.update(version=parent_task['version'] + 1, status='pending')
    repair = task.setdefault('repair', {})
    repair.update(deepcopy(provenance), document_alias=entry['document_alias'], variant_id=entry['variant_id'],
                  label=entry['label'], public_cover_label=entry['public_cover_label'],
                  initialized_cloth_asset_override={**cloth, **entry['cloth_asset']},
                  full_reference_execution='pending')
    for key in tuple(repair):
        if key.startswith('native_') and isinstance(repair[key], bool):
            repair[key] = False
    repair.update({key: False for key in NATIVE_FLAGS})
    scene.update(id=parent_scene['id'] + '__' + entry['variant_id'], version=parent_scene['version'] + 1,
                 status='pending', scene=deepcopy(task['scene']), bindings=deepcopy(task['private']['bindings']))
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    return task, scene


def build_package(design_file, output_dir):
    """Return definitions, manifest and immutable output bytes without writing."""
    design_file, output_dir = repo_path(design_file), repo_path(output_dir)
    design_bytes = design_file.read_bytes()
    design = json.loads(design_bytes)
    if (design.get('schema') != 'wia_cover_variant_design_v1'
            or not isinstance(design.get('entries'), list) or not design['entries']):
        raise ValueError('An explicit wia_cover_variant_design_v1 entries manifest is required.')
    bootstrap = validate_bootstrap()
    definitions, entries, rendered, identities = [], [], {}, set()
    protected = {design_file}
    for entry in design['entries']:
        _validate_entry(entry)
        if entry['variant_id'] in identities:
            raise ValueError('Every variant_id must be unique across the output package.')
        identities.add(entry['variant_id'])
        parent_task_path, parent_scene_path = (repo_path(entry['parent_' + kind + '_file']) for kind in ('task', 'scene'))
        parent_task, parent_scene = (yaml.load(path.read_text(encoding='utf-8'), Loader=UniqueLoader)
                                     for path in (parent_task_path, parent_scene_path))
        loaded_task = load_task(parent_task_path, variant='adapted', run_mode='diagnostic')
        loaded_scene = load_scene(parent_scene_path, variant='adapted', run_mode='diagnostic')
        raw = json.loads(loaded_scene.source_path.read_bytes())
        if (raw.get('big_task'), raw.get('small_task')) != ('Physical Structure', 'Deformable'):
            raise ValueError('Source bootstrap conflict: only the existing native deformable source is supported.')
        provenance = {'design_file': str(design_file), 'design_sha256': sha256(design_bytes),
            'parent_task_file': str(parent_task_path), 'parent_task_sha256': loaded_task.file_sha256,
            'parent_scene_file': str(parent_scene_path), 'parent_scene_sha256': loaded_scene.file_sha256,
            'source_identity': parent_task['source']['json'], 'source_file': str(loaded_task.source_path),
            'source_sha256': loaded_task.source_sha256, **bootstrap}
        task, scene = translate_definitions(parent_task, parent_scene, entry, provenance)
        task_path = output_dir / 'tasks' / (entry['variant_id'] + '.yaml')
        scene_path = output_dir / 'scenes' / (entry['variant_id'] + '.yaml')
        protected.update((parent_task_path, parent_scene_path, loaded_task.source_path, loaded_scene.source_path))
        for path, definition in ((task_path, task), (scene_path, scene)):
            rendered[path] = yaml.safe_dump(definition, sort_keys=False, allow_unicode=False, width=88).encode('utf-8')
        record = {**deepcopy(entry), **provenance, 'task_file': str(task_path), 'scene_file': str(scene_path),
            'output_task_file': str(task_path), 'output_scene_file': str(scene_path),
            'task_sha256': sha256(rendered[task_path]), 'scene_sha256': sha256(rendered[scene_path]),
            'task_source_identity': task['source']['json'], 'scene_source_identity': scene['source']['json'],
            'task_source_sha256': loaded_task.source_sha256, 'scene_source_sha256': loaded_scene.source_sha256,
            'initialized_cloth_asset_override': deepcopy(task['repair']['initialized_cloth_asset_override']),
            'source_roles': {'task': 'immutable_parent_source', 'scene': 'immutable_parent_source',
                             'cloth': 'explicit_deformable_redrape_initialization_override'},
            'variant': 'adapted', 'seed': task['defaults']['seed'], 'task_id': task['id'], 'task_version': task['version'],
            'scene_model': loaded_task.scene_name, 'room': loaded_task.room_name,
            'module': task['private']['evaluation']['module'], 'operation': task['private']['evaluation']['operation'],
            'full_reference_execution': 'pending', **{key: False for key in NATIVE_FLAGS}}
        entries.append(record)
        definitions.append({'task': task, 'scene': scene, 'task_file': task_path, 'scene_file': scene_path})
        loaded_task.verify_unchanged()
        loaded_scene.verify_unchanged()
    manifest = {'schema': 'wia_variant_manifest_v1', 'artifact_path_base': 'absolute',
        'status': 'pending_native_review', 'design_file': str(design_file), 'design_sha256': sha256(design_bytes),
        'full_reference_execution': 'pending', **{key: False for key in NATIVE_FLAGS}, 'entries': entries}
    rendered[output_dir / 'manifest.json'] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if protected & rendered.keys():
        raise ValueError('Output files must be distinct from immutable parent inputs and sources.')
    return definitions, manifest, rendered


def validate_outputs(definitions):
    """Check the actual diagnostic loaders and matching source roles after writing."""
    for entry in definitions:
        task = load_task(entry['task_file'], variant='adapted', run_mode='diagnostic')
        scene = load_scene(entry['scene_file'], variant='adapted', run_mode='diagnostic')
        if (task.definition.scene != scene.definition.scene
                or task.definition.private.bindings != scene.definition.bindings
                or task.seed != scene.seed or task.source_path != scene.source_path):
            raise ValueError('Variant task and scene must load identical source and initialization contracts.')
        runtime = task.definition.repair.get('runtime')
        if runtime is not None and SceneRuntime.model_validate(runtime) != scene.definition.runtime:
            raise ValueError('Variant task and scene runtime differs.')
        task.verify_unchanged()
        scene.verify_unchanged()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    definitions, manifest, rendered = build_package(args.design_file, args.output_dir)
    mismatches = [path for path, data in rendered.items()
                  if path.is_symlink() or not path.is_file() or path.read_bytes() != data]
    if args.check:
        if mismatches:
            for path in mismatches:
                print('WIA cover variant differs or is missing: ' + str(path), file=sys.stderr)
            return 1
    else:
        for path, data in rendered.items():
            if path.is_symlink() or (path.exists() and (not path.is_file() or path.read_bytes() != data)):
                raise ValueError('Refusing to overwrite a differing generated file: ' + str(path))
        for path, data in rendered.items():
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    stream.write(data)
    validate_outputs(definitions)
    print(f'WIA cover variants are current: {len(manifest["entries"])} explicit variants.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
