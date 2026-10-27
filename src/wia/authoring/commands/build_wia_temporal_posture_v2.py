"""Clone pending TM-2 histories with an explicit production construction posture.

Run: python -B scripts/build_wia_temporal_posture_v2.py [--check] [--output-dir PATH]
This CPU-only builder tests a declared hypothesis; it does not run native trials.


Canonical invocation: python -m wia.authoring.commands.build_wia_temporal_posture_v2 --help
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

from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition
from wia.tasks.sources import resolve_source

REVISION = 'wia_tm2_posture_initialization_v2'
PUBLIC_TASK_ID = 'wia_tm2_matched_history_v2'
PARENT = ROOT / 'explore_to_achieve/wia_v1/temporal'
POSTURE_CONFIG = ROOT / 'configs/esi_arm.json'
DEFAULT_OUTPUT = ROOT / 'explore_to_achieve/wia_v1/temporal_posture_v2'
POSTURE_KEY = 'controller_initial_trunk_joint_positions_rad'
VARIANTS = ('a', 'b', 'c')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_posture(path=POSTURE_CONFIG):
    """Read four finite joint positions from the current production arm config."""
    value = json.loads(Path(path).read_text(encoding='utf-8'))['initial_trunk_joint_positions_rad']
    if (not isinstance(value, list) or len(value) != 4
            or any(isinstance(item, bool) or not isinstance(item, (float, int))
                   or not math.isfinite(item) for item in value)):
        raise ValueError('Production arm posture must contain exactly four finite joint positions.')
    return deepcopy(value)


def _repo_path(path):
    return path.relative_to(ROOT).as_posix()


def build_package():
    """Retain complete v1 contracts and change only adapted posture plus version IDs."""
    posture = load_posture()
    definitions, records, rendered, full_public = {}, [], {}, []
    parent_manifest_path = PARENT / 'manifest.json'
    parent_manifest_bytes = parent_manifest_path.read_bytes()
    parent_manifest = json.loads(parent_manifest_bytes)
    for variant in VARIANTS:
        parent_task_file = PARENT / 'tasks' / f'history_{variant}_v1.yaml'
        parent_scene_file = PARENT / 'scenes' / f'history_{variant}_v1.yaml'
        parent_task_bytes, parent_scene_bytes = parent_task_file.read_bytes(), parent_scene_file.read_bytes()
        original_task = yaml.safe_load(parent_task_bytes)
        original_scene = yaml.safe_load(parent_scene_bytes)
        CategoryTaskDefinition.model_validate(original_task)
        SceneDefinition.model_validate(original_scene)
        task, scene = deepcopy(original_task), deepcopy(original_scene)
        task.update(id=PUBLIC_TASK_ID, version=2, status='pending')
        scene.update(id=f'wia_tm2_history_{variant}_v2_scene', version=2, status='pending')
        changes = []
        for kind, definition in (('task', task), ('scene', scene)):
            patches = definition['scene']['variants']['adapted']['patches']
            selected = [(index, patch) for index, patch in enumerate(patches)
                        if patch['operation'] == 'prepare_category_interaction']
            if not selected:
                raise ValueError('Each parent adapted definition must declare its interaction patch.')
            for index, patch in selected:
                if patch.get(POSTURE_KEY) is not None:
                    raise ValueError('The selected v1 baseline must retain its undeclared construction posture.')
                changes.append({'definition': kind,
                    'field': f'/scene/variants/adapted/patches/{index}/{POSTURE_KEY}',
                    'before': patch.get(POSTURE_KEY), 'after': deepcopy(posture)})
                patch[POSTURE_KEY] = deepcopy(posture)
        if task['scene'] != scene['scene'] or task['source'] != scene['source']:
            raise ValueError('Task and scene must retain their same source and complete scene contract.')
        validated = CategoryTaskDefinition.model_validate(task)
        SceneDefinition.model_validate(scene)
        full_public.append({'task_id': validated.id, 'task_version': validated.version,
                            **validated.public.model_dump(mode='json')})
        task_file, scene_file = Path('tasks') / f'history_{variant}_v2.yaml', Path('scenes') / f'history_{variant}_v2.yaml'
        for filename, definition in ((task_file, task), (scene_file, scene)):
            rendered[filename] = yaml.safe_dump(definition, sort_keys=False,
                                                allow_unicode=False, width=88).encode('utf-8')
        source_file = resolve_source(task['source']['json'])
        records.append({
            'history_variant': variant,
            'parent_task_file': _repo_path(parent_task_file), 'parent_task_sha256': sha256(parent_task_bytes),
            'parent_scene_file': _repo_path(parent_scene_file), 'parent_scene_sha256': sha256(parent_scene_bytes),
            'source_identity': task['source']['json'], 'source_file': _repo_path(source_file),
            'source_sha256': sha256(source_file.read_bytes()),
            'task_file': _repo_path(DEFAULT_OUTPUT / task_file), 'task_sha256': sha256(rendered[task_file]),
            'scene_file': _repo_path(DEFAULT_OUTPUT / scene_file), 'scene_sha256': sha256(rendered[scene_file]),
            'matched_changes': changes,
            'native_initialization': 'PENDING', 'robot_stability': 'PENDING',
            'native_phase_2_equivalence': 'PENDING', 'reference_execution': 'PENDING',
        })
        definitions[variant] = (task, scene)
    if not all(value == full_public[0] for value in full_public):
        raise ValueError('The full public task payload must remain identical across histories.')
    manifest = {
        'schema': REVISION, 'status': 'pending-native-validation',
        'parent_manifest_file': _repo_path(parent_manifest_path),
        'parent_manifest_sha256': sha256(parent_manifest_bytes),
        'posture_configuration': {'file': _repo_path(POSTURE_CONFIG),
                                  'sha256': sha256(POSTURE_CONFIG.read_bytes()),
                                  'initial_trunk_joint_positions_rad': posture},
        'hypothesis': (
            'Declaring the production arm controller trunk posture during supported native construction '
            'may avoid later posture movement and improve robot stability. The cause is unproven.'),
        'hypothesis_status': 'UNPROVEN',
        'scientific_change': 'Only the adapted interaction patch construction posture changes from null to the current production arm posture.',
        'allowed_definition_changes': ['id', 'version', 'status',
            'scene.variants.adapted.patches[prepare_category_interaction].' + POSTURE_KEY],
        'v1_definitions_and_sources_modified': False,
        'public_text_unchanged_from_v1': True, 'full_public_payload_equal_across_histories': True,
        'common_public_payload': full_public[0],
        'inherited_repair_metadata_retained': True,
        'observation_contract': deepcopy(parent_manifest['observation_contract']),
        'native_experiments_run': False,
        'native_initialization': 'PENDING', 'robot_stability': 'PENDING',
        'native_phase_2_equivalence': 'PENDING', 'observation_shortcut_audit': 'PENDING',
        'reference_execution': 'PENDING', 'hypothesis_retention_decision': 'PENDING',
        'instances': records,
    }
    rendered[Path('manifest.json')] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode('utf-8')
    return definitions, manifest, rendered


def _require_owned_output(path, expected):
    """Refuse unrelated files before writing any member of this versioned package."""
    if not path.exists() and not path.is_symlink():
        return
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'Refusing to overwrite an unrelated file or symlink: {path}')
    try:
        previous = yaml.safe_load(path.read_bytes())
        current = yaml.safe_load(expected)
        if path.suffix == '.json':
            owned = isinstance(previous, dict) and previous.get('schema') == REVISION
        else:
            owned = (isinstance(previous, dict)
                     and all(previous.get(key) == current[key] for key in ('kind', 'id', 'version', 'source')))
        if not owned:
            raise ValueError('Unrelated output identity.')
    except (ValueError, yaml.YAMLError, UnicodeError) as error:
        raise ValueError(f'Refusing to overwrite an unrelated file: {path}') from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true', help='Compare exact bytes without writing.')
    args = parser.parse_args(argv)
    rendered = build_package()[2]
    output = args.output_dir.resolve()
    if args.check:
        mismatches = [output / name for name, data in rendered.items()
                      if not (output / name).is_file() or (output / name).read_bytes() != data]
        for path in mismatches:
            print(f'WIA temporal posture v2 differs or is missing: {path}', file=sys.stderr)
        if mismatches:
            return 1
        print(f'WIA temporal posture v2 is current: {output}')
        return 0
    for name, data in rendered.items():
        _require_owned_output(output / name, data)
    for name, data in rendered.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(f'Wrote pending WIA temporal posture v2: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
