"""Build explicit household mirror definitions without simulation or robot actions.

Run: python -B scripts/build_wia_household_mirror.py --design-file PATH
  --output-dir PATH [--check]
Input: wia_household_mirror_design_v1 with complete main-authored layouts,
bindings, category/model/scale candidate_assets, parent files and public goals.
Existing differing outputs and source artifacts are never overwritten.


Canonical invocation: python -m wia.authoring.commands.build_wia_household_mirror --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import sys

import yaml

from wia.resources import ROOT

from wia.authoring.converters.wia_mirror import HOUSEHOLD_REVISION, MATERIAL, MODULE
from wia.tasks.loader import UniqueLoader, load_scene, load_task, source_scene_room
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition, SceneRuntime
from wia.tasks.sources import resolve_source
from wia.tasks.wia_mirror_scoring import MirrorProgress
from wia.authoring.commands.build_wia_workstation_variants import NATIVE_FLAGS, repo_path, sha256, validate_outputs


def _text(entry, key):
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError('An explicit nonempty ' + key + ' is required.')
    return value


def _vector(value, size, *, positive=False):
    if (not isinstance(value, list) or len(value) != size
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(item) or (positive and item <= 0) for item in value)):
        raise ValueError('An explicit finite numeric vector is required.')


def _bindings(entry):
    result = {}
    for key, value in entry['bindings'].items():
        result[key] = {'name': value} if isinstance(value, str) else deepcopy(value)
        _text(result[key], 'name')
    if len({row['name'] for row in result.values()}) != len(result):
        raise ValueError('Bindings must have distinct native object identities.')
    return result


def translate_definitions(parent_task, parent_scene, entry, provenance):
    """Apply complete explicit household layouts while preserving parent science."""
    CategoryTaskDefinition.model_validate(parent_task)
    SceneDefinition.model_validate(parent_scene)
    task, scene = deepcopy(parent_task), deepcopy(parent_scene)
    evaluation = task['private']['evaluation']
    if (evaluation.get('module') != MODULE or evaluation.get('type') != 'category_operation'
            or evaluation['parameters'].get('revision') not in ('wia_mirror_v1', 'wia_mirror_v2')):
        raise ValueError('A legacy owned mirror parent is required.')
    if (task['scene'] != scene['scene'] or task['private']['bindings'] != scene['bindings']
            or task['defaults']['seed'] != scene['seed']
            or SceneRuntime.model_validate(task['repair']['runtime'])
            != SceneRuntime.model_validate(scene.get('runtime', {}))):
        raise ValueError('Parent task/scene layout, bindings, seed and runtime must agree.')
    for field in ('document_alias', 'variant_id', 'label', 'task_id', 'scene_id', 'source_identity',
                  'scene_model', 'room', 'public_goal'):
        _text(entry, field)
    if (entry['task_id'] == parent_task['id'] or entry['scene_id'] == parent_scene['id']
            or not re.fullmatch(r'[A-Za-z0-9_-]+', entry['variant_id'])):
        raise ValueError('Fresh task/scene identities and a safe variant_id are required.')
    identity = {'scene_model': entry['scene_model'], 'room': entry['room']}
    layout, bindings, assets = deepcopy(entry['layout']), _bindings(entry), deepcopy(entry['candidate_assets'])
    if (layout['scene_model'] != entry['scene_model'] or layout['rooms'] != [entry['room']]
            or not assets or not set(assets).issubset(bindings)):
        raise ValueError('Layout identity and explicitly bound household candidates must agree.')
    objects = {row['name']: row for row in layout['objects']}
    if len(objects) != len(layout['objects']) or set(objects) != {row['name'] for row in bindings.values()}:
        raise ValueError('The complete layout must exactly match the explicit distinct bindings.')
    for key, asset in assets.items():
        if not isinstance(asset, dict) or set(asset) != {'category', 'model', 'scale'}:
            raise ValueError('Every candidate asset must specify category/model/scale exactly.')
        for field in ('category', 'model'):
            _text(asset, field)
        _vector(asset['scale'], 3, positive=True)
        obj = objects[bindings[key]['name']]
        if (obj.get('type') != 'DatasetObject' or obj.get('fixed_base') is not False
                or obj.get('visual_only') is not False or any(obj.get(field) != asset[field] for field in asset)):
            raise ValueError('Candidate constructor must match its explicit dynamic collidable DatasetObject asset contract.')
        bindings[key].update(category=asset['category'], model=asset['model'])
    for obj in objects.values():
        _vector(obj['position'], 3)
        _vector(obj['orientation'], 4)
        if not any(obj['orientation']):
            raise ValueError('An explicit nonzero object quaternion is required.')
    _vector(layout['robot_start']['position'], 3)
    _vector(layout['robot_start']['orientation'], 4)
    p = evaluation['parameters']
    p.pop('candidate_dimensions_m')
    p.update(revision=HOUSEHOLD_REVISION, scene_contract=identity, candidate_assets=assets,
             candidates={key: bindings[key]['name'] for key in assets},
             fixture_keys=[key for key in bindings if key not in assets])
    required = {p['mirror_key'], p['table_key'], p['anchor_key'], *p['destinations'].values(), *p['cue_bindings'].values()}
    if not required.issubset(p['fixture_keys']):
        raise ValueError('All parent optical apparatus and destination semantic keys must be preserved.')
    MirrorProgress(p)
    for key in p['fixture_keys']:
        obj = objects[bindings[key]['name']]
        if (obj.get('type') != 'PrimitiveObject' or obj.get('primitive_type') != 'Cube'
                or obj.get('fixed_base') is not True or obj.get('visual_only') is not False):
            raise ValueError('Apparatus must retain the measured fixed collidable Cube contract.')
    patches = task['scene']['variants']['adapted']['patches']
    if (len(patches) != 2 or {row.get('phase') for row in patches} != {'before_construction', 'after_postprocessing'}
            or any(row.get('operation') != 'category_initialization' or row.get('module') != MODULE for row in patches)
            or patches[0]['parameters'] != patches[1]['parameters']):
        raise ValueError('Both exact owned mirror initialization hooks must agree.')
    for patch in patches:
        init = patch['parameters']
        if init['mirror_key'] != p['mirror_key'] or init['mirror_material'] != {'binding': p['mirror_key'], **MATERIAL}:
            raise ValueError('The fixed exact parent mirror PBR contract must be preserved.')
        init.update(revision=HOUSEHOLD_REVISION, scene_contract=deepcopy(identity), layout=deepcopy(layout))
    if 'public' in entry:
        task['public'] = deepcopy(entry['public'])
        if task['public'].get('objective') != entry['public_goal']:
            raise ValueError('public_goal must exactly match the explicit public objective.')
    else:
        # Preserve all numeric thresholds and behavioral requirements in the parent wording.
        task['public'] = json.loads(re.sub(r'\bblocks\b', 'household objects',
            re.sub(r'\bblock\b', 'household object', json.dumps(task['public']))))
        task['public']['objective'] = entry['public_goal']
    if re.search(r'\bblocks?\b', json.dumps(task['public']), flags=re.IGNORECASE):
        raise ValueError('Public household wording must not retain legacy block terminology.')
    evaluation['binding_keys'] = list(bindings)
    task['private']['bindings'] = bindings
    task.update(id=entry['task_id'], version=1, status='pending')
    task['source']['json'] = entry['source_identity']
    task['repair'].update(deepcopy(provenance), contract_revision=HOUSEHOLD_REVISION,
        document_alias=entry['document_alias'], variant_id=entry['variant_id'], label=entry['label'],
        scene_contract=deepcopy(identity), full_reference_execution='pending',
        geometry_scope='Measured native household collision bounds and optical vertex centroid only; no physical feasibility certificate.')
    task['repair'].update({key: False for key in NATIVE_FLAGS})
    for key, value in task['repair'].items():
        if key.startswith('native_') and isinstance(value, bool):
            task['repair'][key] = False
    scene.update(id=entry['scene_id'], version=1, status='pending', source=deepcopy(task['source']),
        bindings=deepcopy(bindings), scene=deepcopy(task['scene']))
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    return task, scene


def _matched_appearance_pair(first, second):
    """Allow only the explicitly matched same-seed R-A1/R-A2 public identity."""
    a, b = first['task'], second['task']
    ea, eb = first['entry'], second['entry']
    if ({ea['document_alias'], eb['document_alias']} != {'R-A1', 'R-A2'}
            or a['id'] != b['id'] or a['defaults'] != b['defaults']
            or a['public'] != b['public'] or a['requirements'] != b['requirements']
            or a['private']['bindings'] != b['private']['bindings']):
        return False
    pa, pb = (task['private']['evaluation']['parameters'] for task in (a, b))
    if (pa['operation'] != 'classify_appearance_deliver'
            or {pa['appearance_mode'], pb['appearance_mode']} != {'direct', 'reflected'}
            or {key: value for key, value in pa.items() if key != 'appearance_mode'}
            != {key: value for key, value in pb.items() if key != 'appearance_mode'}):
        return False
    layouts = []
    for task, p in ((a, pa), (b, pb)):
        layout = deepcopy(task['scene']['variants']['adapted']['patches'][0]['parameters']['layout'])
        cue_names = {task['private']['bindings'][key]['name'] for key in p['cue_bindings'].values()}
        layout['objects'] = [obj for obj in layout['objects'] if obj['name'] not in cue_names]
        layouts.append(layout)
    return layouts[0] == layouts[1]


def build_package(design_file, output_dir):
    """Render fresh definitions and manifest without claiming native acceptance."""
    design_file, output_dir = repo_path(design_file), repo_path(output_dir)
    design_bytes = design_file.read_bytes()
    design = json.loads(design_bytes)
    if design.get('schema') != 'wia_household_mirror_design_v1' or not design.get('entries'):
        raise ValueError('An explicit wia_household_mirror_design_v1 entries design is required.')
    definitions, entries, rendered, identities, task_ids, scene_ids = [], [], {}, set(), {}, set()
    protected = {design_file}
    for entry in design['entries']:
        task_path, scene_path = (repo_path(entry['parent_' + kind + '_file']) for kind in ('task', 'scene'))
        source_path = resolve_source(_text(entry, 'source_identity'))
        protected.update((task_path, scene_path, source_path))
        raw = json.loads(source_path.read_bytes())
        if source_scene_room(entry['source_identity'], raw) != (entry['scene_model'], entry['room']):
            raise ValueError('The immutable source identity must match the explicit native scene/room.')
        parent_task, parent_scene = (yaml.load(path.read_text(), Loader=UniqueLoader) for path in (task_path, scene_path))
        loaded_task, loaded_scene = load_task(task_path, variant='adapted', run_mode='diagnostic'), load_scene(scene_path, variant='adapted', run_mode='diagnostic')
        provenance = {'design_file': str(design_file), 'design_sha256': sha256(design_bytes),
            'parent_task_file': str(task_path), 'parent_task_sha256': loaded_task.file_sha256,
            'parent_scene_file': str(scene_path), 'parent_scene_sha256': loaded_scene.file_sha256,
            'source_file': str(source_path), 'source_sha256': sha256(source_path.read_bytes())}
        task, scene = translate_definitions(parent_task, parent_scene, entry, provenance)
        identity = (entry['document_alias'], entry['variant_id'])
        if identity in identities or scene['id'] in scene_ids:
            raise ValueError('Each entry must have distinct variant and scene identities.')
        current = {'task': task, 'entry': entry}
        if task['id'] in task_ids:
            previous = task_ids[task['id']]
            if len(previous) != 1 or not _matched_appearance_pair(previous[0], current):
                raise ValueError('Duplicate task IDs are restricted to identical same-seed matched R-A1/R-A2 controls.')
            previous.append(current)
        else:
            task_ids[task['id']] = [current]
        identities.add(identity)
        scene_ids.add(scene['id'])
        task_file, scene_file = (output_dir / entry['variant_id'] / (kind + '.yaml') for kind in ('task', 'scene'))
        for path, value in ((task_file, task), (scene_file, scene)):
            if path in rendered:
                raise ValueError('Duplicate output path.')
            rendered[path] = yaml.safe_dump(value, sort_keys=False, allow_unicode=False, width=88).encode()
        definitions.append({'task': task, 'scene': scene, 'task_file': task_file, 'scene_file': scene_file})
        entries.append({**deepcopy(entry), **provenance, 'task_file': str(task_file), 'scene_file': str(scene_file),
            'output_task_file': str(task_file), 'output_scene_file': str(scene_file),
            'task_sha256': sha256(rendered[task_file]), 'scene_sha256': sha256(rendered[scene_file]),
            'variant': 'adapted', 'seed': task['defaults']['seed'], 'task_id': task['id'], 'task_version': task['version'],
            'module': MODULE, 'revision': HOUSEHOLD_REVISION, 'full_reference_execution': 'pending',
            **{key: False for key in NATIVE_FLAGS}})
        loaded_task.verify_unchanged()
        loaded_scene.verify_unchanged()
    manifest = {'schema': 'wia_variant_manifest_v1', 'artifact_path_base': 'absolute',
        'status': 'pending_native_review', 'design_file': str(design_file), 'design_sha256': sha256(design_bytes),
        'full_reference_execution': 'pending', **{key: False for key in NATIVE_FLAGS}, 'entries': entries}
    rendered[output_dir / 'manifest.json'] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    if set(rendered) & protected:
        raise ValueError('Generated outputs must be distinct from all immutable inputs.')
    return definitions, manifest, rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    definitions, manifest, rendered = build_package(args.design_file, args.output_dir)
    differences = [path for path, data in rendered.items() if path.is_symlink() or not path.is_file() or path.read_bytes() != data]
    if args.check:
        if differences:
            for path in differences:
                print('Household mirror output differs or is missing: ' + str(path), file=sys.stderr)
            return 1
    else:
        for path, data in rendered.items():
            if path.is_symlink() or (path.exists() and (not path.is_file() or path.read_bytes() != data)):
                raise ValueError('Refusing to overwrite a differing file: ' + str(path))
        for path, data in rendered.items():
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    validate_outputs(definitions)
    print(json.dumps({'schema': manifest['schema'], 'entries': len(manifest['entries']), 'check': args.check}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
