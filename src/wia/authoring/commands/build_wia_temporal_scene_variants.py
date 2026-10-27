"""Transfer existing temporal tasks with an explicitly authored rigid translation.

Run: python -B scripts/build_wia_temporal_scene_variants.py --design-file PATH
  --manifest-output PATH [--check]
Only adapted variants are transferred. Existing differing files are refused;
no replay regeneration, native simulation, rendering or qualification is run.


Canonical invocation: python -m wia.authoring.commands.build_wia_temporal_scene_variants --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import yaml

from wia.resources import ROOT

from wia.tasks import sources
from wia.tasks.loader import UniqueLoader, load_scene, load_task, source_scene_room
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition, SceneRuntime
from wia.authoring.commands.build_wia_workstation_variants import NATIVE_FLAGS, _text, _vector, repo_path, sha256

MODULE = 'core.esi_converters.temporal_understanding_two_phase'
OPERATION = 'compare_change_and_restore_or_collect'
VECTOR_FIELDS = {'position', 'look_target', 'target', 'aabb_min', 'aabb_max'}
XY_BOX_FIELDS = {'room_bbox_world_xy', 'expanded_bbox_world_xy', 'bbox_world_xy'}


def translate_world(value, delta):
    """Translate reviewed semantic world fields, leaving local geometry unchanged."""
    def shifted(vector, offset, field):
        vector = _vector(vector, len(offset), field)
        # Native replay parks inactive actors far outside the scene. Preserve
        # these sentinels rather than turning a parked actor into scene content.
        if any(abs(number) >= 900 for number in vector):
            return deepcopy(vector)
        return [number + shift for number, shift in zip(vector, offset)]

    if isinstance(value, list):
        return [translate_world(item, delta) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    result = {}
    for key, item in value.items():
        if key in VECTOR_FIELDS and isinstance(item, list):
            result[key] = shifted(item, delta, key)
        elif key in XY_BOX_FIELDS:
            result[key] = shifted(item, [delta[0], delta[1], delta[0], delta[1]], key)
        elif key == 'bbox':
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError('World bbox must contain exactly two three-dimensional corners.')
            corners = [_vector(corner, 3, 'bbox corner') for corner in item]
            result[key] = (deepcopy(corners) if any(abs(number) >= 900 for corner in corners for number in corner)
                           else [shifted(corner, delta, 'bbox corner') for corner in corners])
        else:
            result[key] = translate_world(item, delta)
    return result


def _active_identities(value, entry):
    """Replace active scene metadata while preserving historical provenance."""
    if isinstance(value, list):
        return [_active_identities(item, entry) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    result = {}
    replacements = {'scene': entry['scene_model'], 'scene_model': entry['scene_model'],
                    'room': entry['room'], 'room_instance': entry['room'],
                    'floor_name': entry['floor_name'], 'source_identity': entry['source_identity']}
    for key, item in value.items():
        if key in ('authored_revision', 'provenance', 'parent_provenance', 'transfer_provenance'):
            result[key] = deepcopy(item)
        elif key in replacements:
            result[key] = deepcopy(replacements[key])
        else:
            result[key] = _active_identities(item, entry)
    return result


def translate_source(parent_source, entry, provenance):
    """Copy the raw parent source without invoking its historical generator."""
    raw = deepcopy(parent_source)
    metadata = raw.get('metadata_json')
    encoded = isinstance(metadata, str)
    if encoded:
        metadata = json.loads(metadata)
    if not isinstance(metadata, dict):
        raise ValueError('Temporal source requires dictionary or JSON-object-string metadata_json.')
    raw['metadata_json'] = metadata
    raw = _active_identities(translate_world(raw, entry['translation_m']), entry)
    stem = Path(entry['source_identity']).stem
    for container in (raw, raw['metadata_json']):
        container.update(scene=entry['scene_model'], room=entry['room'], floor_name=entry['floor_name'])
        for key in ('id', 'question_id'):
            if key in container:
                container[key] = stem
    raw['metadata_json']['source_identity'] = entry['source_identity']
    raw['transfer_provenance'] = deepcopy(provenance)
    if encoded:
        raw['metadata_json'] = json.dumps(raw['metadata_json'], ensure_ascii=False, allow_nan=False)
    source_scene_room(entry['source_identity'], raw)
    return raw


def _binding_updates(bindings, entry, additions):
    result = deepcopy(bindings)
    if 'floor' not in result:
        raise ValueError('Temporal transfer requires an explicit existing floor binding.')
    result['floor']['name'] = entry['floor_name']
    names = {binding['name'] for binding in result.values()}
    for obj in additions:
        key = obj['name']
        if key in result or key in names:
            raise ValueError('Background supports cannot replace existing binding keys or native names.')
        result[key] = {field: obj[field] for field in ('name', 'category', 'model')}
        names.add(key)
    return result


def _support_objects(entry):
    additions = entry.get('background_support_objects')
    if not isinstance(additions, list):
        raise ValueError('background_support_objects must be an explicit list of destination DatasetObject configs.')
    names = set()
    for obj in additions:
        if not isinstance(obj, dict) or obj.get('type') != 'DatasetObject':
            raise ValueError('Every background support must use an explicit DatasetObject constructor.')
        for key in ('name', 'category', 'model'):
            _text(obj, key)
        _vector(obj.get('position'), 3, 'background support destination position')
        if obj['name'] in names:
            raise ValueError('Background support names must be unique.')
        names.add(obj['name'])
    return deepcopy(additions)


def translate_definitions(parent_task, parent_scene, entry, provenance, runtime):
    """Transfer the active complete world while preserving temporal mechanisms."""
    CategoryTaskDefinition.model_validate(parent_task)
    SceneDefinition.model_validate(parent_scene)
    if (parent_task['scene'] != parent_scene['scene']
            or parent_task['private']['bindings'] != parent_scene['bindings']
            or parent_task['source'] != parent_scene['source']
            or parent_task['defaults']['seed'] != parent_scene['seed']):
        raise ValueError('Parent task and scene must have identical scene, bindings, source and seed.')
    evaluation = parent_task['private']['evaluation']
    if (evaluation.get('module') != MODULE or evaluation.get('operation') != OPERATION
            or not evaluation['parameters'].get('task_edit2_physical')):
        raise ValueError('Only existing physical TaskEdit2 two-phase temporal tasks may be transferred.')
    task, scene = deepcopy(parent_task), deepcopy(parent_scene)
    # Rebuild active structures to detach YAML aliases shared with saved_image*.
    task['private'] = deepcopy(parent_task['private'])
    additions = _support_objects(entry)
    bindings = _binding_updates(parent_task['private']['bindings'], entry, additions)
    task['private']['bindings'] = bindings
    patches = parent_task['scene']['variants']['adapted']['patches']
    hooks = [patch for patch in patches if patch['operation'] == 'category_initialization']
    if (len(hooks) != 2 or {hook['phase'] for hook in hooks} != {'before_construction', 'after_postprocessing'}
            or any(hook['module'] != MODULE for hook in hooks)
            or hooks[0]['parameters'] != hooks[1]['parameters']):
        raise ValueError('Adapted temporal transfer requires both identical original temporal hooks.')
    adapted = translate_world(parent_task['scene']['variants']['adapted'], entry['translation_m'])
    for patch in adapted['patches']:
        if patch['operation'] == 'category_initialization':
            p = _active_identities(patch['parameters'], entry)
            p.update(scene_model=entry['scene_model'], room=entry['room'], source_identity=entry['source_identity'])
            p['parent_provenance'] = {'source_identity': hooks[0]['parameters']['source_identity'],
                                      'provenance': deepcopy(hooks[0]['parameters'].get('provenance'))}
            p['provenance'] = 'Rigidly translated copy of the existing temporal world; original producer evidence is retained in parent_provenance.'
            p['bindings'] = deepcopy(bindings)
            p['apparatus']['dataset_objects'].extend(deepcopy(additions))
            patch['parameters'] = p
        elif patch['operation'] == 'prepare_category_interaction':
            patch['source_region'] = entry['room']
    task['scene']['variants']['adapted'] = adapted
    evaluation = task['private']['evaluation']
    evaluation['parameters'] = _active_identities(
        translate_world(parent_task['private']['evaluation']['parameters'], entry['translation_m']), entry)
    evaluation['parameters']['apparatus']['dataset_objects'].extend(deepcopy(additions))
    identities = evaluation['parameters'].get('identities')
    if identities is not None:
        if 'floor' in identities:
            identities['floor']['name'] = entry['floor_name']
        for obj in additions:
            identities[obj['name']] = {field: obj[field] for field in ('name', 'category', 'model')}
    evaluation['binding_keys'] = [*evaluation['binding_keys'], *(obj['name'] for obj in additions)]
    task.update(version=parent_task['version'] + 1, status='pending')
    task['source']['json'] = entry['source_identity']
    old_repair = deepcopy(parent_task.get('repair', {}))
    if 'runtime' in old_repair and SceneRuntime.model_validate(old_repair['runtime']) != SceneRuntime.model_validate(runtime):
        raise ValueError('Existing task repair runtime conflicts with the effective scene runtime.')
    repair = {key: value for key, value in old_repair.items()
              if not key.startswith(('parent_', 'source_', 'design_'))}
    repair.update(deepcopy(provenance), parent_provenance=old_repair,
                  runtime=deepcopy(runtime), document_alias=entry['document_alias'], variant_id=entry['variant_id'],
                  label=entry['label'], translation_m=deepcopy(entry['translation_m']),
                  full_reference_execution='pending')
    for key in tuple(repair):
        if (key.startswith('native_') or key.endswith('_verified')) and isinstance(repair[key], bool):
            repair[key] = False
    repair.update({key: False for key in NATIVE_FLAGS})
    task['repair'] = repair
    scene.update(id=parent_scene['id'] + '__' + entry['variant_id'], version=parent_scene['version'] + 1,
                 status='pending', source=deepcopy(task['source']), scene=deepcopy(task['scene']),
                 bindings=deepcopy(bindings), runtime=deepcopy(runtime))
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    return task, scene


def build_package(design_file, manifest_output):
    """Render explicit task, scene and copied source artifacts entirely in memory."""
    design_file, manifest_output = repo_path(design_file), repo_path(manifest_output)
    design_bytes = design_file.read_bytes()
    design = json.loads(design_bytes)
    if design.get('schema') != 'wia_temporal_transfer_design_v1' or not design.get('entries'):
        raise ValueError('Explicit wia_temporal_transfer_design_v1 entries are required.')
    rendered, definitions, records, identities = {}, [], [], set()
    protected = {design_file, manifest_output}
    for entry in design['entries']:
        for key in ('document_alias', 'variant_id', 'label', 'source_identity', 'scene_model', 'room', 'floor_name'):
            _text(entry, key)
        _vector(entry['translation_m'], 3, 'translation_m')
        task_path, scene_path = (repo_path(entry['parent_' + kind + '_file']) for kind in ('task', 'scene'))
        loaded_task = load_task(task_path, variant='adapted', run_mode='diagnostic')
        loaded_scene = load_scene(scene_path, variant='adapted', run_mode='diagnostic')
        if loaded_task.source_path != loaded_scene.source_path:
            raise ValueError('Parent task and scene must resolve the same immutable source.')
        protected.update((task_path, scene_path, loaded_task.source_path))
        parent_task, parent_scene = (yaml.load(path.read_text(encoding='utf-8'), Loader=UniqueLoader)
                                     for path in (task_path, scene_path))
        source_identity = Path(entry['source_identity'])
        source_path = (sources.AUTHORED_ROOT / source_identity).resolve()
        if (source_identity.is_absolute() or source_identity.suffix != '.json'
                or not source_path.is_relative_to(sources.AUTHORED_ROOT.resolve())
                or (sources.UPSTREAM_ROOT / source_identity).is_file()):
            raise ValueError('New source identity must remain project-owned and cannot shadow upstream sources.')
        provenance = {'design_file': str(design_file), 'design_sha256': sha256(design_bytes),
            'parent_task_file': str(task_path), 'parent_task_sha256': loaded_task.file_sha256,
            'parent_scene_file': str(scene_path), 'parent_scene_sha256': loaded_scene.file_sha256,
            'parent_source_identity': parent_task['source']['json'],
            'parent_source_file': str(loaded_task.source_path), 'parent_source_sha256': loaded_task.source_sha256}
        raw = translate_source(json.loads(loaded_task.source_path.read_bytes()), entry, provenance)
        runtime = loaded_scene.definition.runtime.model_dump(mode='json')
        task, scene = translate_definitions(parent_task, parent_scene, entry, provenance, runtime)
        key = (entry['document_alias'], entry['variant_id'])
        if key in identities:
            raise ValueError('Each document_alias and variant_id pair must be unique.')
        identities.add(key)
        outputs = {kind: repo_path(entry['output_' + kind + '_file']) for kind in ('task', 'scene')}
        for path, data in ((source_path, (json.dumps(raw, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()),
                           *((outputs[kind], yaml.safe_dump(value, sort_keys=False, allow_unicode=False, width=88).encode())
                             for kind, value in (('task', task), ('scene', scene)))):
            if path in rendered or path in protected:
                raise ValueError('Every output must be distinct from all immutable inputs and other outputs.')
            rendered[path] = data
        row = {**deepcopy(entry), **provenance,
               'task_file': str(outputs['task']), 'scene_file': str(outputs['scene']),
               'output_task_file': str(outputs['task']), 'output_scene_file': str(outputs['scene']),
               'source_file': str(source_path), 'source_sha256': sha256(rendered[source_path]),
               'task_sha256': sha256(rendered[outputs['task']]), 'scene_sha256': sha256(rendered[outputs['scene']]),
               'variant': 'adapted', 'seed': loaded_task.seed, 'task_id': task['id'], 'task_version': task['version'],
               'module': MODULE, 'revision': task['private']['evaluation']['parameters']['task_revision'],
               'full_reference_execution': 'pending', **{key: False for key in NATIVE_FLAGS}}
        records.append(row)
        definitions.append({'task': task, 'scene': scene, 'source': raw, 'task_file': outputs['task'],
                            'scene_file': outputs['scene'], 'source_file': source_path})
        loaded_task.verify_unchanged()
        loaded_scene.verify_unchanged()
    if protected.intersection(rendered):
        raise ValueError('An output collides with another variant immutable input.')
    manifest = {'schema': 'wia_variant_manifest_v1', 'artifact_path_base': 'absolute',
                'status': 'pending_native_review', 'design_file': str(design_file), 'design_sha256': sha256(design_bytes),
                'full_reference_execution': 'pending', **{key: False for key in NATIVE_FLAGS}, 'entries': records}
    rendered[manifest_output] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    return definitions, manifest, rendered


def validate_outputs(definitions):
    """Verify both emitted definitions with the existing diagnostic loaders."""
    for row in definitions:
        task = load_task(row['task_file'], variant='adapted', run_mode='diagnostic')
        scene = load_scene(row['scene_file'], variant='adapted', run_mode='diagnostic')
        if (task.definition.scene != scene.definition.scene or task.seed != scene.seed
                or task.definition.private.bindings != scene.definition.bindings
                or task.source_path != scene.source_path or task.scene_name != scene.scene_name
                or task.room_name != scene.room_name
                or SceneRuntime.model_validate(task.definition.repair['runtime']) != scene.definition.runtime):
            raise ValueError('Transferred task and scene must load identical world and runtime contracts.')
        task.verify_unchanged()
        scene.verify_unchanged()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design-file', type=Path, required=True)
    parser.add_argument('--manifest-output', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Compare generated bytes without writing.')
    args = parser.parse_args(argv)
    definitions, manifest, rendered = build_package(args.design_file, args.manifest_output)
    if args.check:
        mismatches = [path for path, data in rendered.items() if path.is_symlink()
                      or not path.is_file() or path.read_bytes() != data]
        if mismatches:
            for path in mismatches:
                print('WIA temporal variant differs or is missing: ' + str(path), file=sys.stderr)
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
    print(f'WIA temporal variants are current: {len(manifest["entries"])} explicit variants.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
