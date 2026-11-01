"""Build the two authored historical multi-object collection variants on CPU.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B
scripts/build_wia_history_collection_variants.py [--output-dir PATH] [--check]
The fixed reviewed design preserves each parent's world, seed, runtime and
budgets; only food inventory/poses and the historical collection task change.
Native initialization, visibility, physics and budget sufficiency are unverified.


Canonical invocation: python -m wia.authoring.commands.build_wia_history_collection_variants --help
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

from wia.tasks.loader import UniqueLoader, load_scene, load_task
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition, resolve_episode_defaults
from wia.tasks.wia_history_collection import OPERATION, validate_collection_parameters

DEFAULT_OUTPUT = ROOT / 'explore_to_achieve/wia_v1/review_20261025/history_collection_v2'
INITIALIZER = 'wia.authoring.converters.active_layout'
MODULE = 'wia.authoring.converters.temporal_understanding_agent_observation'
ASSETS = (
    ('hamburger', 'fyswch', 'hamburgers with filling', 'hamburger with filling'),
    ('hamburger_bun', 'fqxumh', 'plain buns', 'plain bun'),
    ('club_sandwich', 'ablfyd', 'club sandwiches', 'club sandwich'))
PARENTS = (
    {'document_alias': 'T38', 'variant_id': 't38_historical_multi_collection_v2',
     'parent_task_file': 'explore_to_achieve/wia_v1/background_ramp_floor_v3/tasks/t38_complete_ground_v3.yaml',
     'parent_scene_file': 'explore_to_achieve/wia_v1/background_ramp_floor_v3/scenes/t38_complete_ground_v3.yaml'},
    {'document_alias': 'T39', 'variant_id': 't39_historical_multi_collection_v2',
     'parent_task_file': 'explore_to_achieve/tasks/Temporal Understanding/Agent Observation/Beechwood_0_int/redesign_20261020/category_ebb26cc29316e397864246bd.yaml',
     'parent_scene_file': 'explore_to_achieve/scenes/Temporal Understanding/Agent Observation/Beechwood_0_int/redesign_20261020/category_ebb26cc29316e397864246bd.yaml'})
NATIVE_FLAGS = ('native_validation_run', 'native_initialization_run', 'native_initialization_verified',
    'native_geometry_verified', 'native_support_verified', 'native_visibility_verified',
    'native_history_verified', 'physical_success_verified', 'scientific_validation_run',
    'scientific_configuration_verified', 'budget_sufficiency_verified')
LOCAL_X = (-.18, 0., .18)
LOCAL_Y = -.10


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def repo_path(value):
    path = Path(value).expanduser()
    return project_path(path, root=ROOT)


def _layout_hooks(definition):
    return [patch for patch in definition['scene']['variants']['adapted']['patches']
            if patch.get('module') == INITIALIZER and patch.get('operation') == 'category_initialization']


def _rotate(vector, quaternion):
    """Rotate a local deck offset with its declared normalized XYZW orientation."""
    norm = math.sqrt(sum(value * value for value in quaternion))
    x, y, z, w = (value / norm for value in quaternion)
    vx, vy, vz = vector
    return [(1 - 2 * (y*y + z*z)) * vx + 2 * (x*y - z*w) * vy + 2 * (x*z + y*w) * vz,
        2 * (x*y + z*w) * vx + (1 - 2 * (x*x + z*z)) * vy + 2 * (y*z - x*w) * vz,
        2 * (x*z - y*w) * vx + 2 * (y*z + x*w) * vy + (1 - 2 * (x*x + y*y)) * vz]


def _inventory(layout):
    """Reuse original asset root heights and cyclically distribute each category."""
    objects = {obj['name']: obj for obj in layout['objects']}
    templates, heights = {}, {}
    for index, (category, model, _, _) in enumerate(ASSETS, 1):
        original = objects['pantry_item_' + str(index)]
        deck = objects[f'pantry_shelf_{index}_deck']
        if (original.get('type') != 'DatasetObject' or original['category'] != category
                or original['model'] != model or original.get('scale', [1., 1., 1.]) != [1., 1., 1.]
                or original.get('fixed_base') is not False or original.get('visual_only') is not False):
            raise ValueError('Parent food assets must match the exact reviewed dynamic unit-scale inventory.')
        templates[category] = original
        heights[category] = original['position'][2] - (deck['position'][2] + deck['scale'][2] / 2)
    items, supports, categories, labels, placements = [], {}, {}, {}, []
    for station in range(1, 4):
        deck_name = f'pantry_shelf_{station}_deck'
        deck = objects[deck_name]
        for slot in range(3):
            asset_index = (station - 1 + slot) % 3
            category, _, public_category, description = ASSETS[asset_index]
            name = (f'pantry_item_{station}' if asset_index == station - 1
                    else f'pantry_item_s{station}_{category}')
            local = [LOCAL_X[slot], LOCAL_Y, deck['scale'][2] / 2 + heights[category]]
            offset = _rotate(local, deck['orientation'])
            item = deepcopy(templates[category])
            item.update(name=name, position=[center + shift for center, shift in zip(deck['position'], offset)],
                orientation=deepcopy(deck['orientation']), scale=[1., 1., 1.])
            items.append(item)
            supports[name] = deck_name
            categories[name] = public_category
            labels[name] = f'the {description} at pantry shelf {station}'
            placements.append({'name': name, 'category': category, 'model': item['model'],
                'support': deck_name, 'station': station, 'local_x_m': LOCAL_X[slot],
                'local_y_m': LOCAL_Y, 'root_above_deck_top_m': heights[category]})
    return items, supports, categories, labels, placements, heights


def translate_definitions(parent_task, parent_scene, entry, provenance):
    """Apply the fixed nine-item design while retaining all other world settings."""
    CategoryTaskDefinition.model_validate(parent_task)
    SceneDefinition.model_validate(parent_scene)
    if (parent_task['scene'] != parent_scene['scene']
            or parent_task['private']['bindings'] != parent_scene['bindings']
            or parent_task['source'] != parent_scene['source']
            or parent_task['defaults']['seed'] != parent_scene['seed']):
        raise ValueError('Parent task and scene must share exact patches, bindings, source and seed.')
    evaluation = parent_task['private']['evaluation']
    if (evaluation.get('module'), evaluation.get('operation')) != (MODULE, 'survey_history_retrieve_place'):
        raise ValueError('Only the reviewed pantry history parents are supported.')
    hooks = _layout_hooks(parent_task)
    if (len(hooks) != 2 or {hook['phase'] for hook in hooks} != {'before_construction', 'after_postprocessing'}
            or hooks[0]['parameters'] != hooks[1]['parameters']):
        raise ValueError('Both identical active_layout initialization phases must be preserved.')
    layout = hooks[0]['parameters']['layout']
    items, supports, categories, labels, placements, heights = _inventory(layout)
    task, scene = deepcopy(parent_task), deepcopy(parent_scene)
    originals = set(evaluation['parameters']['target_keys'])
    if originals != {'pantry_item_1', 'pantry_item_2', 'pantry_item_3'}:
        raise ValueError('The parent must declare exactly its three original pantry instances.')
    by_name = {obj['name']: obj for obj in items}
    added = [obj for obj in items if obj['name'] not in originals]
    for hook in _layout_hooks(task):
        current = hook['parameters']['layout']
        current['objects'] = [deepcopy(by_name.get(obj['name'], obj)) for obj in current['objects']] + deepcopy(added)
        current['supported_items'] = [{'name': row['name'], 'support': supports[row['name']]}
            for row in items]
    bindings = task['private']['bindings']
    for obj in items:
        bindings[obj['name']] = {field: obj[field] for field in ('name', 'category', 'model')}
    evaluation = task['private']['evaluation']
    evaluation['operation'] = OPERATION
    evaluation['binding_keys'].extend(obj['name'] for obj in added)
    parameters = evaluation['parameters']
    parameters.pop('followup_rule', None)
    parameters.update(target_keys=[obj['name'] for obj in items], public_target_labels=labels,
        target_categories=categories, query_rule='earliest_per_category', require_history_subset=True,
        query_min_category_instances=2, query_min_total=1, query_max_total=3,
        history_collection_revision=OPERATION)
    parameters['apparatus']['target_support_bindings'] = supports
    validate_collection_parameters(parameters)
    public = task['public']
    public['objective'] = (
        'Inspect all nine food items on the pantry shelves: three hamburgers with filling, three plain buns, '
        'and three club sandwiches. Choose your own route and remember the order in which you first '
        'inspect the distinct instances within each category. Do not grasp or move any item before submitting '
        'survey_complete=true with command DONE and answer=null after all nine have been inspected. A delayed delivery request will then '
        'ask for one to three items in total, using the first one or two instances you inspected in each '
        'requested category. Retrieve those history-selected instances and leave every requested item '
        'stably released on the same blue-rimmed serving tray.')
    public['success_criteria'] = (
        'Survey all nine distinct items, receive the delayed category-count request, physically grasp every '
        'requested history-selected instance, and leave all of them stably supported on the serving tray '
        'with the robot withdrawn and its hand empty.')
    public['constraints'] = [
        'Preserve all food items during the survey; after survey completion, move only the requested instances.',
        parent_task['public']['constraints'][1],
        'Simultaneous first inspections of two instances in the same requested category do not establish '
        'an order and make that category history ambiguous. Different categories may be first inspected '
        'together. Repeated views of an instance count once.',
        parent_task['public']['constraints'][-1]]
    variant_id = entry['variant_id']
    task.update(id=parent_task['id'] + '__' + variant_id,
        version=parent_task['version'] + 1, status='pending')
    repair = task.setdefault('repair', {})
    repair.update(deepcopy(provenance), document_alias=entry['document_alias'], variant_id=variant_id,
        history_collection_revision=OPERATION, full_reference_execution='pending',
        budget_status='Inherited pilot budgets; sufficiency for nine-item survey and up to three deliveries is unqualified.',
        inherited_episode_defaults=resolve_episode_defaults(parent_task).model_dump(mode='json'),
        inventory_placement=placements, original_asset_root_height_m=heights,
        initialization_status='authored_not_native_validated')
    for key, value in tuple(repair.items()):
        if (key.startswith('native_') or key.endswith('_verified')) and type(value) is bool:
            repair[key] = False
    repair.update({key: False for key in NATIVE_FLAGS})
    scene.update(id=parent_scene['id'] + '__' + variant_id,
        version=parent_scene['version'] + 1, status='pending',
        scene=deepcopy(task['scene']), bindings=deepcopy(bindings))
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    return task, scene


def build_package(output_dir=DEFAULT_OUTPUT):
    """Return both definitions, the standard variant manifest, and output bytes."""
    output_dir = repo_path(output_dir)
    script_path = Path(__file__).resolve()
    definitions, entries, rendered, protected = [], [], {}, {script_path}
    for entry in PARENTS:
        parent_paths = [repo_path(entry['parent_' + kind + '_file']) for kind in ('task', 'scene')]
        parent_task, parent_scene = [yaml.load(path.read_text(), Loader=UniqueLoader) for path in parent_paths]
        loaded_task = load_task(parent_paths[0], variant='adapted', run_mode='diagnostic')
        loaded_scene = load_scene(parent_paths[1], variant='adapted', run_mode='diagnostic')
        provenance = {'builder_file': str(script_path), 'builder_sha256': sha256(script_path.read_bytes()),
            'parent_task_file': str(parent_paths[0]), 'parent_task_sha256': loaded_task.file_sha256,
            'parent_scene_file': str(parent_paths[1]), 'parent_scene_sha256': loaded_scene.file_sha256,
            'source_identity': parent_task['source']['json'], 'source_file': str(loaded_task.source_path),
            'source_sha256': loaded_task.source_sha256}
        task, scene = translate_definitions(parent_task, parent_scene, entry, provenance)
        task_path = output_dir / 'tasks' / (entry['variant_id'] + '.yaml')
        scene_path = output_dir / 'scenes' / (entry['variant_id'] + '.yaml')
        for path, value in ((task_path, task), (scene_path, scene)):
            rendered[path] = yaml.safe_dump(value, sort_keys=False, allow_unicode=False, width=88).encode()
        record = {**deepcopy(entry), **provenance, 'label': entry['document_alias'] + ' historical multi-object collection',
            'task_file': str(task_path), 'scene_file': str(scene_path),
            'output_task_file': str(task_path), 'output_scene_file': str(scene_path),
            'task_sha256': sha256(rendered[task_path]), 'scene_sha256': sha256(rendered[scene_path]),
            'task_source_identity': task['source']['json'], 'scene_source_identity': scene['source']['json'],
            'task_source_sha256': loaded_task.source_sha256, 'scene_source_sha256': loaded_scene.source_sha256,
            'variant': 'adapted', 'seed': task['defaults']['seed'], 'task_id': task['id'], 'task_version': task['version'],
            'scene_model': loaded_task.scene_name, 'room': loaded_task.room_name, 'module': MODULE, 'operation': OPERATION,
            'episode_defaults': resolve_episode_defaults(task).model_dump(mode='json'),
            'budget_status': task['repair']['budget_status'], 'initialization_status': 'authored_not_native_validated',
            'full_reference_execution': 'pending', 'inventory_placement': task['repair']['inventory_placement'],
            'source_roles': {'task': 'immutable_parent_source', 'scene': 'immutable_parent_source',
                'inventory': 'explicit_active_layout_initialization_override'},
            **{key: False for key in NATIVE_FLAGS}}
        entries.append(record)
        definitions.append({'task': task, 'scene': scene, 'task_file': task_path, 'scene_file': scene_path})
        protected.update((*parent_paths, loaded_task.source_path, loaded_scene.source_path))
        loaded_task.verify_unchanged()
        loaded_scene.verify_unchanged()
    manifest = {'schema': 'wia_variant_manifest_v1', 'artifact_path_base': 'absolute',
        'status': 'pending_native_review', 'builder_file': str(script_path),
        'builder_sha256': sha256(script_path.read_bytes()), 'full_reference_execution': 'pending',
        'initialization_status': 'authored_not_native_validated',
        **{key: False for key in NATIVE_FLAGS}, 'entries': entries}
    rendered[output_dir / 'manifest.json'] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    if protected.intersection(rendered):
        raise ValueError('Generated outputs must not overwrite parent definitions or immutable sources.')
    return definitions, manifest, rendered


def validate_outputs(definitions):
    """Load task/scene pairs through the diagnostic runtime input schemas on CPU."""
    for definition in definitions:
        task = load_task(definition['task_file'], variant='adapted', run_mode='diagnostic')
        scene = load_scene(definition['scene_file'], variant='adapted', run_mode='diagnostic')
        if (task.definition.scene != scene.definition.scene
                or task.definition.private.bindings != scene.definition.bindings
                or task.source_path != scene.source_path or task.seed != scene.seed):
            raise ValueError('Generated task and scene must load the same world, bindings, source and seed.')
        task.verify_unchanged()
        scene.verify_unchanged()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    definitions, manifest, rendered = build_package(args.output_dir)
    mismatches = [path for path, data in rendered.items()
                  if path.is_symlink() or not path.is_file() or path.read_bytes() != data]
    if args.check:
        if mismatches:
            for path in mismatches:
                print('WIA historical collection variant differs or is missing: ' + str(path), file=sys.stderr)
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
    print(f'WIA historical collection variants are current: {len(manifest["entries"])} authored variants.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
