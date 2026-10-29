"""Build explicit native room collection and independent search variants.

Run: python -B scripts/build_wia_room_collection.py --design-file PATH \
  --output-dir PATH [--check]
Input schema is wia_room_collection_design_v1. Each entry supplies parent task
and scene paths, alias, variant ID, label, operation and layout. Native inventory
requires scene_inventory_file: a named native scene JSON snapshot. No native
simulation, registry edits, capture submission or publication is performed.


Canonical invocation: python -m wia.authoring.commands.build_wia_room_collection --help
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

from wia.authoring.converters.wia_room_collection import (
    MODULE, REVISION, ORDERED_OPERATION, SEARCH_OPERATION,
    resolve_native_inventory, validate_layout)
from wia.tasks.loader import UniqueLoader, load_scene, load_task, source_scene_room
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition, SceneRuntime
from wia.tasks.sources import resolve_source


def _path(value):
    path = Path(value).expanduser()
    return project_path(path, root=ROOT)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path):
    return yaml.load(path.read_text(encoding='utf-8'), Loader=UniqueLoader)


def _public(layout, operation):
    targets = [row for row in layout.get('native_inventory', []) + layout.get('added_inventory', [])
               if row['role'] == 'target']
    choices = '; '.join(f"{row['public_name']} in {row['room'].replace('_', ' ')}" for row in targets)
    if operation == ORDERED_OPERATION:
        route = ', then '.join(room.replace('_', ' ') for room in layout['route'])
        objective = f'Visit {route}, in that order. In each designated room, pick up the selected object: {choices}. '
    else:
        categories = ', '.join(dict.fromkeys(row['category'].replace('_', ' ') for row in targets))
        rooms = ', '.join(dict.fromkeys(row['room'].replace('_', ' ') for row in targets))
        objective = f'Search {rooms} and collect every {categories} you find there. '
    constraints = ['Keep the tray in your right hand and use your left hand to pick and place.',
                   'Pick up each selected object while you and the object are in its designated room.',
                   'Only the selected physical objects count; repeated views do not create additional objects.']
    if operation == ORDERED_OPERATION:
        constraints.append('Complete both the room visits and the object pickups in the declared room order.')
    return {'objective': objective + 'Place every selected object on the tray held in your right hand.',
            'success_criteria': 'Finish with all selected objects physically resting on the tray, your left hand released, '
                                'and the tray held steadily in your right hand.',
            'environment': layout['public_environment'], 'constraints': constraints}


def build_definitions(parent_task, parent_scene, entry, provenance):
    """Preserve source identity and budgets except explicit authored overrides."""
    CategoryTaskDefinition.model_validate(parent_task)
    SceneDefinition.model_validate(parent_scene)
    if (parent_task['scene'] != parent_scene['scene']
            or parent_task['private']['bindings'] != parent_scene.get('bindings', {})
            or parent_task['defaults']['seed'] != parent_scene.get('seed', 0)):
        raise ValueError('Parent task and scene must have identical patches, bindings and seed.')
    if ('runtime' in parent_task.get('repair', {})
            and SceneRuntime.model_validate(parent_task['repair']['runtime'])
            != SceneRuntime.model_validate(parent_scene.get('runtime', {}))):
        raise ValueError('Parent task and scene must have identical runtime settings.')
    operation, layout = entry['operation'], deepcopy(entry['layout'])
    rows = validate_layout(layout, operation)
    task, scene = deepcopy(parent_task), deepcopy(parent_scene)
    if entry.get('source_identity') is not None:
        task['source']['json'] = entry['source_identity']
    parameters = {'revision': REVISION, 'operation': operation, 'layout': layout,
                  'plate': deepcopy(layout['plate']), 'targets': [row['name'] for row in rows if row['role'] == 'target'],
                  'distractors': [row['name'] for row in rows if row['role'] == 'distractor'],
                  'terminal_dwell_s': 1., 'maximum_relative_speed_m_s': .03,
                  'maximum_plate_tilt_deg': 10., 'maximum_plate_speed_m_s': .03,
                  'maximum_plate_angular_speed_rad_s': .1}
    allowed = set(parameters) - {'revision', 'operation', 'layout', 'plate', 'targets', 'distractors'}
    if set(entry.get('evaluation_settings', {})) - allowed:
        raise ValueError('Unsupported collection evaluation override.')
    parameters.update(deepcopy(entry.get('evaluation_settings', {})))
    if any(isinstance(parameters[key], bool) or not isinstance(parameters[key], (int, float))
           or not 0 < parameters[key] < float('inf') for key in allowed):
        raise ValueError('Collection physical thresholds must be positive finite numbers.')
    inventory = rows + [layout['plate']]
    bindings = {row['name']: {key: row[key] for key in ('name', 'category', 'model')} for row in inventory}
    # Preserve the separately owned native background loading configuration.
    # Retired parent task mechanisms and object relocation patches are replaced.
    background = {'revision': 'wia_background_v1', 'mode': 'complete_ground',
                  'expected_scene': layout['scene_model'], 'disable_grid': False}
    patches = []
    for phase in ('before_construction', 'after_postprocessing'):
        patches.append({'operation': 'category_initialization', 'phase': phase,
                        'module': MODULE, 'parameters': deepcopy(parameters)})
        patches.append({'operation': 'category_initialization', 'phase': phase,
                        'module': 'wia.authoring.converters.wia_background', 'parameters': deepcopy(background)})
    task['scene'] = {'policy': 'native_then_minimal_patch_else_skip',
                     'variants': {'original': {'patches': []}, 'adapted': {'patches': patches}}}
    task.update(id=parent_task['id'] + '__' + entry['variant_id'], version=parent_task['version'] + 1,
                status='pending', public=deepcopy(entry.get('public', _public(layout, operation))))
    task['private'] = {'bindings': bindings, 'evaluation': {'type': 'category_operation', 'module': MODULE,
                       'operation': operation, 'binding_keys': list(bindings), 'parameters': parameters}}
    task['requirements'] = {'capabilities': ['base_translation', 'base_rotation', 'ee_translation', 'gripper']}
    task['qualification'] = {'on_failure': 'skip', 'initial_visibility': 'none'}
    task['defaults'].update(deepcopy(entry.get('task_defaults', {})))
    runtime = SceneRuntime.model_validate(scene.get('runtime', {})).model_dump(mode='json')
    task['repair'] = {**deepcopy(provenance), 'document_alias': entry['document_alias'],
                      'variant_id': entry['variant_id'], 'label': entry['label'], 'contract_revision': REVISION,
                      'runtime': runtime, 'native_validation_run': False, 'physical_success_verified': False,
                      'native_inventory_spawned': False, 'added_inventory_explicit': bool(layout.get('added_inventory')),
                      'original_source_qa_unused': True,
                      'full_reference_execution': 'pending', 'recommended_primitive_profile': 'fine',
                      'controller_profile_file': 'configs/esi_primitives_fine.json',
                      'parent_task_defaults': deepcopy(parent_task['defaults']),
                      'declared_task_default_overrides': deepcopy(entry.get('task_defaults', {}))}
    scene.update(id=parent_scene['id'] + '__' + entry['variant_id'], version=parent_scene['version'] + 1,
                 status='pending', source=deepcopy(task['source']), bindings=deepcopy(bindings),
                 scene=deepcopy(task['scene']), target=None, target_bindings=[], qualification=None,
                 seed=task['defaults']['seed'])
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    return task, scene


def build_package(design_file, output_dir):
    """Render deterministic definitions and wia_variant_manifest_v1 metadata."""
    design_file, output_dir = _path(design_file), _path(output_dir)
    design_bytes = design_file.read_bytes()
    design = json.loads(design_bytes)
    if design.get('schema') != 'wia_room_collection_design_v1' or not design.get('entries'):
        raise ValueError('Expected explicit wia_room_collection_design_v1 entries.')
    rendered, manifest_entries, definitions, seen = {}, [], [], set()
    for entry in design['entries']:
        for key in ('document_alias', 'variant_id', 'label', 'parent_task_file', 'parent_scene_file', 'operation'):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError('Explicit variant entry missing ' + key)
        identity = (entry['document_alias'], entry['variant_id'])
        if identity in seen or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', value) for value in identity):
            raise ValueError('Alias/variant pairs must be distinct safe identifiers.')
        seen.add(identity)
        parent_paths = [_path(entry['parent_' + kind + '_file']) for kind in ('task', 'scene')]
        provenance = {'design_file': str(design_file), 'design_sha256': _sha(design_bytes)}
        for kind, path in zip(('task', 'scene'), parent_paths):
            provenance.update({f'parent_{kind}_file': str(path), f'parent_{kind}_sha256': _sha(path.read_bytes())})
        layout = entry['layout']
        validate_layout(layout, entry['operation'])
        if layout.get('native_inventory'):
            snapshot_path = _path(entry['scene_inventory_file'])
            snapshot_bytes = snapshot_path.read_bytes()
            resolve_native_inventory(layout, json.loads(snapshot_bytes))
            provenance.update(scene_inventory_file=str(snapshot_path), scene_inventory_sha256=_sha(snapshot_bytes))
        parent_task, parent_scene = [_read(path) for path in parent_paths]
        source_identity = entry.get('source_identity', parent_task['source']['json'])
        source_path = resolve_source(source_identity)
        source_bytes = source_path.read_bytes()
        source_scene, source_room = source_scene_room(source_identity, json.loads(source_bytes))
        if source_scene != layout['scene_model'] or ('room' in entry and entry['room'] != source_room):
            raise ValueError('Resolved source scene/room must match the authored collection scene/room.')
        provenance.update(source_file=str(source_path), source_sha256=_sha(source_bytes),
                          source_identity=source_identity, scene_model=source_scene, room=source_room)
        task, scene = build_definitions(parent_task, parent_scene, entry, provenance)
        stem = '__'.join(identity)
        task_path, scene_path = [output_dir / (stem + '.' + kind + '.yaml') for kind in ('task', 'scene')]
        for path, definition in ((task_path, task), (scene_path, scene)):
            if path in parent_paths or path == design_file or path in rendered:
                raise ValueError('Generated definitions must not overwrite immutable inputs.')
            rendered[path] = yaml.safe_dump(definition, sort_keys=False, allow_unicode=False, width=100).encode()
        definitions.append({'task_file': task_path, 'scene_file': scene_path})
        manifest_entries.append({**deepcopy(entry), **provenance, 'task_file': str(task_path), 'scene_file': str(scene_path),
                                 'task_sha256': _sha(rendered[task_path]), 'scene_sha256': _sha(rendered[scene_path]),
                                 'variant': 'adapted', 'seed': scene['seed'], 'module': MODULE, 'revision': REVISION,
                                 'task_id': task['id'], 'task_version': task['version'],
                                 'native_validation_run': False, 'physical_success_verified': False,
                                 'full_reference_execution': 'pending'})
    manifest = {'schema': 'wia_variant_manifest_v1', 'artifact_path_base': 'absolute',
                'status': 'pending_native_review', 'design_file': str(design_file), 'design_sha256': _sha(design_bytes),
                'native_validation_run': False, 'physical_success_verified': False, 'entries': manifest_entries}
    rendered[output_dir / 'manifest.json'] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    return definitions, manifest, rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    definitions, manifest, rendered = build_package(args.design_file, args.output_dir)
    mismatches = [path for path, data in rendered.items() if path.is_symlink()
                  or not path.is_file() or path.read_bytes() != data]
    if args.check and mismatches:
        for path in mismatches:
            print('Room collection output missing or differs: ' + str(path), file=sys.stderr)
        return 1
    if not args.check:
        for path, data in rendered.items():
            if path.is_symlink() or path.exists() and (not path.is_file() or path.read_bytes() != data):
                raise ValueError('Refusing to overwrite a differing output: ' + str(path))
        for path, data in rendered.items():
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    for definition in definitions:
        task = load_task(definition['task_file'], variant='adapted', run_mode='diagnostic')
        scene = load_scene(definition['scene_file'], variant='adapted', run_mode='diagnostic')
        if task.definition.scene != scene.definition.scene or task.seed != scene.seed:
            raise ValueError('Generated task/scene contracts differ.')
    print(f'Room collection definitions are current: {len(manifest["entries"])} variants.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
