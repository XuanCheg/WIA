"""Build the explicitly selected T22/T23 native furniture support revisions.

Run: OMP_NUM_THREADS=2 python -B scripts/build_wia_rigid_native_support_revisions.py
  [--output-dir PATH] [--check]
This CPU-only builder retains separate task QA and native scene source roles.
Native support, robot placement, manipulation and physical success remain pending.


Canonical invocation: python -m wia.authoring.commands.build_wia_rigid_native_support_revisions --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import yaml

from wia.resources import ROOT
from wia.authoring.commands._inputs import project_path

from wia.tasks.loader import UniqueLoader, load_scene, load_task
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition

OUTPUT_DIR = ROOT / 'explore_to_achieve/wia_v1/review_20261025/rigid_support_v2'
INITIALIZER = 'core.esi_converters.active_layout'
REMOVED_PRIMITIVES = {'support_table_top', *(f'support_table_leg_{i}' for i in range(1, 5))}
NATIVE_FLAGS = ('native_validation_run', 'native_support_verified', 'native_robot_placement_verified',
                'physical_success_verified', 'physically_qualified', 'legal_robot_reference_verified')
DESIGNS = (
    {'document_alias': 'T22', 'variant_id': 't22_native_table_support_v2',
     'label': 'T22 native breakfast table support',
     'parent_task_file': 'explore_to_achieve/wia_v1/rigid_family_v1/tasks/rigid_t22_v1.yaml',
     'parent_scene_file': 'explore_to_achieve/wia_v1/rigid_family_v1/scenes/rigid_t22_v1.yaml',
     'inventory_file': 'outputs/wia_v1_20261024_furnished/asset_inventory/Beechwood_0_int.json',
     'support_name': 'breakfast_table_skczfi_1', 'support_category': 'breakfast_table',
     'support_top_m': .75877580735, 'previous_table_top_m': .45,
     'container_xy': [[-4., 2.18], [-3.48, 2.18], [-2.94, 2.18]],
     'target_xy': [-3.75, 1.93], 'robot_position': [-3.4, .85, .1], 'robot_yaw_deg': 90.,
     'floor_name': 'floors_gjpvfd_0',
     'environment': 'The gift box and the three packing boxes are exposed on the native white breakfast table. Retrieve the target using the real R1 robot.'},
    {'document_alias': 'T23', 'variant_id': 't23_native_cabinet_support_v2',
     'label': 'T23 native cabinet support with original sofa pickups',
     'parent_task_file': 'explore_to_achieve/wia_v1/rigid_t23_support_v5/task.yaml',
     'parent_scene_file': 'explore_to_achieve/wia_v1/rigid_t23_support_v5/scene.yaml',
     'inventory_file': 'outputs/wia_v1_20261025_revision/rigid_inventory/Beechwood_1_int.json',
     'support_name': 'bottom_cabinet_jhymlr_0', 'support_category': 'bottom_cabinet',
     'support_top_m': .69620362193, 'previous_table_top_m': .45,
     'container_xy': [[3.78, -5.], [3.78, -4.4], [3.78, -3.85]],
     'floor_name': 'floors_esoefs_0',
     'environment': 'The mug, wicker basket and bucket are exposed on the native bottom cabinet. The granola box, spray can and colored pencil remain on the native sofa. Retrieve the targets using the real R1 robot.'},
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def repo_path(value):
    path = Path(value).expanduser()
    return project_path(path, root=ROOT)


def patches(definition):
    return definition['scene']['variants']['tabletop']['patches']


def _initializer(definition):
    return [patch for patch in patches(definition) if patch['operation'] == 'category_initialization']


def _inventory_support(design):
    path = repo_path(design['inventory_file'])
    inventory = json.loads(path.read_bytes())
    rows = [row for row in inventory['objects'] if row.get('name') == design['support_name']]
    if len(rows) != 1 or rows[0]['category'] != design['support_category'] or not rows[0].get('model'):
        raise ValueError('The explicitly selected native furniture must resolve to one exact inventory asset.')
    return rows[0], {'inventory_file': str(path), 'inventory_sha256': sha256(path.read_bytes())}


def revise_definitions(parent_task, parent_scene, design, support, provenance, scene_file):
    """Apply only the selected native support bindings, object starts and T22 robot start."""
    CategoryTaskDefinition.model_validate(parent_task)
    SceneDefinition.model_validate(parent_scene)
    if (parent_task['scene'] != parent_scene['scene']
            or parent_task['private']['bindings'] != parent_scene['bindings']
            or parent_task['defaults']['seed'] != parent_scene['seed']):
        raise ValueError('Parent task and scene must have paired patches, bindings and seeds.')
    if (parent_task['private']['evaluation']['module'] != 'core.esi_converters.rigid_partial_insertion'
            or parent_task['private']['evaluation']['operation'] != 'pick_release_partial_insert'):
        raise ValueError('Only the unchanged native partial-insertion scorer is supported.')
    task, scene = deepcopy(parent_task), deepcopy(parent_scene)
    bound_support = {key: support[key] for key in ('name', 'category', 'model')}
    task['private']['bindings']['table'] = deepcopy(bound_support)
    if design['document_alias'] == 'T22':
        task['private']['bindings']['pickup_support'] = deepcopy(bound_support)
    removed = [row['name'] for row in patches(task)
               if row['operation'] == 'add_primitive' and row['name'] in REMOVED_PRIMITIVES]
    if set(removed) != REMOVED_PRIMITIVES or len(removed) != 5:
        raise ValueError('Expected exactly the five named fabricated support primitives.')
    kept = [row for row in patches(task)
            if not (row['operation'] == 'add_primitive' and row['name'] in REMOVED_PRIMITIVES)]
    task['scene']['variants']['tabletop']['patches'] = kept
    poses = {row['binding']: row for row in kept if row['operation'] == 'object_pose'}
    for index, xy in enumerate(design['container_xy']):
        row = poses['container_' + str(index)]
        row['position'] = [*xy, row['position'][2] - design['previous_table_top_m'] + design['support_top_m']]
    if 'target_xy' in design:
        poses['target']['position'][:2] = design['target_xy']
        robots = [row for row in kept if row['operation'] == 'robot_pose' and row['phase'] == 'before_construction']
        if len(robots) != 1:
            raise ValueError('T22 requires exactly the retained native robot construction pose.')
        robots[0].update(position=deepcopy(design['robot_position']), yaw_deg=design['robot_yaw_deg'])
    hooks = _initializer(task)
    if design['document_alias'] == 'T22':
        if hooks:
            raise ValueError('T22 parent must not contain an unreviewed initialization hook.')
        hook = {'operation': 'category_initialization', 'phase': 'after_postprocessing', 'module': INITIALIZER,
                'parameters': {'revision': design['variant_id'], 'layout': {'objects': [], 'supported_items': []}}}
        kept.append(hook)
        layout = hook['parameters']['layout']
        keys = ['container_0', 'container_1', 'container_2', 'target']
        layout['objects'] = [{'name': task['private']['bindings'][key]['name'],
            'position': deepcopy(poses[key]['position']), 'orientation': deepcopy(poses[key]['quaternion_xyzw'])} for key in keys]
        layout['supported_items'] = [{'name': task['private']['bindings'][key]['name'],
            'support': support['name'], 'support_alignment_mode': 'local_raycast',
            'support_surface_mode': 'local_raycast'} for key in keys]
    else:
        if len(hooks) != 1 or hooks[0].get('module') != INITIALIZER or hooks[0].get('phase') != 'after_postprocessing':
            raise ValueError('T23 requires its exact existing after-postprocessing active_layout hook.')
        hook = hooks[0]
        hook['parameters']['revision'] = design['variant_id']
        layout = hook['parameters']['layout']
        for key in ('container_0', 'container_1', 'container_2'):
            name = task['private']['bindings'][key]['name']
            authored = [row for row in layout['objects'] if row['name'] == name]
            supported = [row for row in layout['supported_items'] if row['name'] == name]
            if len(authored) != 1 or len(supported) != 1:
                raise ValueError('Each T23 container requires one existing layout pose and support row.')
            authored[0]['position'] = deepcopy(poses[key]['position'])
            supported[0].update(support=support['name'], support_alignment_mode='local_raycast', support_surface_mode='local_raycast')
    layout['floor_name'] = design['floor_name']
    layout['synchronize_authored_poses_before_alignment'] = True
    task.update(version=parent_task['version'] + 1, status='pending')
    task['public']['environment'] = design['environment']
    repair = task.setdefault('repair', {})
    repair.update(deepcopy(provenance), support_revision=design['variant_id'], variant_id=design['variant_id'],
        document_alias=design['document_alias'], scene_file=str(scene_file),
        support_binding_override=deepcopy(bound_support),
        validation_status='pending-native-support-and-robot-review',
        repair_reason='Replace only the five fabricated support-table primitives with the explicitly selected retained native furniture. Preserve original source roles, asset scales and orientations, insertion scoring, budgets and sofa pickup starts. Native support and robot placement remain unqualified.',
        full_reference_execution='pending', reference_execution_status='deferred_until_user_review')
    for key in tuple(repair):
        if key.startswith('native_') and isinstance(repair[key], bool):
            repair[key] = False
    repair.update({key: False for key in NATIVE_FLAGS})
    scene.update(id=parent_scene['id'] + '__' + design['variant_id'], version=parent_scene['version'] + 1,
        status='pending', bindings=deepcopy(task['private']['bindings']), scene=deepcopy(task['scene']))
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    return task, scene


def build_package(output_dir=OUTPUT_DIR):
    """Render the fixed main-authored pair without running or modifying source worlds."""
    output_dir = repo_path(output_dir)
    definitions, entries, rendered, protected = [], [], {}, set()
    for design in DESIGNS:
        parent_task_file, parent_scene_file = (repo_path(design['parent_' + kind + '_file']) for kind in ('task', 'scene'))
        parent_task, parent_scene = (yaml.load(path.read_text(), Loader=UniqueLoader) for path in (parent_task_file, parent_scene_file))
        loaded_task = load_task(parent_task_file, variant='tabletop', run_mode='diagnostic')
        loaded_scene = load_scene(parent_scene_file, variant='tabletop', run_mode='diagnostic')
        support, inventory_provenance = _inventory_support(design)
        provenance = {'parent_task_file': str(parent_task_file), 'parent_task_sha256': loaded_task.file_sha256,
            'parent_scene_file': str(parent_scene_file), 'parent_scene_sha256': loaded_scene.file_sha256,
            'task_source_identity': parent_task['source']['json'], 'task_source_file': str(loaded_task.source_path),
            'task_source_sha256': loaded_task.source_sha256, 'scene_source_identity': parent_scene['source']['json'],
            'scene_source_file': str(loaded_scene.source_path), 'scene_source_sha256': loaded_scene.source_sha256,
            **inventory_provenance}
        task_file = output_dir / 'tasks' / (design['variant_id'] + '.yaml')
        scene_file = output_dir / 'scenes' / (design['variant_id'] + '.yaml')
        task, scene = revise_definitions(parent_task, parent_scene, design, support, provenance, scene_file)
        for path, definition in ((task_file, task), (scene_file, scene)):
            rendered[path] = yaml.safe_dump(definition, sort_keys=False, allow_unicode=False, width=88).encode()
        definitions.append({'task_file': task_file, 'scene_file': scene_file, 'task': task, 'scene': scene})
        entries.append({**deepcopy(design), **provenance, 'task_file': str(task_file), 'scene_file': str(scene_file),
            'task_sha256': sha256(rendered[task_file]), 'scene_sha256': sha256(rendered[scene_file]),
            'source_identity': parent_scene['source']['json'], 'source_file': str(loaded_scene.source_path),
            'source_sha256': loaded_scene.source_sha256, 'variant': 'tabletop', 'seed': loaded_task.seed,
            'scene_model': loaded_scene.scene_name, 'room': loaded_scene.room_name,
            'task_id': task['id'], 'task_version': task['version'],
            'support_binding_override': deepcopy(task['private']['bindings']['table']),
            'source_roles': {'task': 'original_criterion_qa', 'scene': 'original_native_world'},
            'full_reference_execution': 'pending', **{key: False for key in NATIVE_FLAGS}})
        protected.update((parent_task_file, parent_scene_file, loaded_task.source_path, loaded_scene.source_path,
                          repo_path(design['inventory_file'])))
        loaded_task.verify_unchanged()
        loaded_scene.verify_unchanged()
    manifest = {'schema': 'wia_variant_manifest_v1', 'artifact_path_base': 'absolute', 'status': 'pending_native_review',
        'full_reference_execution': 'pending', **{key: False for key in NATIVE_FLAGS}, 'entries': entries}
    rendered[output_dir / 'manifest.json'] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    if protected & rendered.keys():
        raise ValueError('Generated outputs must not replace immutable input files.')
    return definitions, manifest, rendered


def validate_outputs(definitions):
    for definition in definitions:
        task = load_task(definition['task_file'], variant='tabletop', run_mode='diagnostic')
        scene = load_scene(definition['scene_file'], variant='tabletop', run_mode='diagnostic')
        if (task.definition.scene != scene.definition.scene or task.seed != scene.seed
                or task.definition.private.bindings != scene.definition.bindings):
            raise ValueError('Generated task and scene must load matching patches, bindings and seeds.')
        if task.definition.source.json_path != definition['task']['source']['json'] or scene.definition.source.json_path != definition['scene']['source']['json']:
            raise ValueError('Task QA and native scene source roles must remain exact.')
        task.verify_unchanged()
        scene.verify_unchanged()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    definitions, manifest, rendered = build_package(args.output_dir)
    if args.check:
        mismatches = [path for path, data in rendered.items()
                      if path.is_symlink() or not path.is_file() or path.read_bytes() != data]
        if mismatches:
            for path in mismatches:
                print('Native support revision differs or is missing: ' + str(path), file=sys.stderr)
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
    print(f'Native support revisions are current: {len(manifest["entries"])} variants.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
