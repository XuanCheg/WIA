"""Build a pending T11 revision with neutral room choices and unchanged physics.

Run: python -B scripts/build_wia_regional_t11.py [--check] [--output-dir PATH]
Existing differing outputs are refused. No native qualification is performed.


Canonical invocation: python -m wia.authoring.commands.build_wia_regional_t11 --help
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

from wia.tasks.loader import load_scene, load_task
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition
from wia.tasks.sources import resolve_source

PARENT_RELATIVE = Path('Cognitive Mapping/Regional Boundary/school_gym/full_scene/'
                       'corev2__category_1a55282aea78514f749fe4e0.yaml')
PARENT_TASK = ROOT / 'explore_to_achieve/tasks' / PARENT_RELATIVE
PARENT_SCENE = ROOT / 'explore_to_achieve/scenes' / PARENT_RELATIVE
DEFAULT_OUTPUT = ROOT / 'explore_to_achieve/wia_v1/regional_t11_v1'
REVISION = 'wia_regional_neutral_labels_v1'
TASK_ID = 'wia_regional_t11_v1'
SCENE_ID = 'wia_regional_t11_v1_scene'
QUESTION = 'Which room contains the paper towel dispenser?'
NEUTRAL_OPTIONS = ('gym 0', 'locker room 1', 'locker room 0', 'bathroom 0')
ANCHOR_INSTRUCTION = ('Inspect each queried anchor using its public description '
                      'and relate it to the named regions.')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def repo_path(path):
    return path.relative_to(ROOT).as_posix()


def build_package():
    """Change public aliases while preserving the complete native task contract."""
    parent_task_bytes, parent_scene_bytes = PARENT_TASK.read_bytes(), PARENT_SCENE.read_bytes()
    original_task, original_scene = yaml.safe_load(parent_task_bytes), yaml.safe_load(parent_scene_bytes)
    CategoryTaskDefinition.model_validate(original_task)
    SceneDefinition.model_validate(original_scene)
    if original_task['source'] != original_scene['source']:
        raise ValueError('Historical T11 task and scene sources must agree.')
    source_file = resolve_source(original_task['source']['json'])
    source_bytes = source_file.read_bytes()
    source = json.loads(source_bytes)
    original_qa = original_task['private']['evaluation']['parameters']['source_qa']
    if (original_qa['question'] != source['question'] or original_qa['answer'] != source['answer']
            or original_qa['options'] != json.loads(source['options_json'])):
        raise ValueError('The parent source QA must match the immutable original JSON.')
    old_options = original_qa['options']
    if (len(old_options) != 4 or len(set(old_options)) != 4
            or tuple(option.split(' (', 1)[0] for option in old_options) != NEUTRAL_OPTIONS
            or original_qa['answer'] != old_options[1]):
        raise ValueError('The parent must retain the reviewed four-option T11 mapping.')
    aliases = dict(zip(old_options, NEUTRAL_OPTIONS))
    if len(set(aliases.values())) != len(aliases):
        raise ValueError('Room option aliases must form a bijection.')
    task, scene = deepcopy(original_task), deepcopy(original_scene)
    task.update(id=TASK_ID, version=1, status='pending')
    scene.update(id=SCENE_ID, version=1, status='pending')
    if not original_task['public']['objective'].endswith(ANCHOR_INSTRUCTION):
        raise ValueError('The original observed-anchor instruction must remain explicit.')
    task['public']['objective'] = QUESTION + ' Choose from: ' + ', '.join(NEUTRAL_OPTIONS) + '. ' + ANCHOR_INSTRUCTION
    success = original_task['public']['success_criteria']
    for original, neutral in aliases.items():
        success = success.replace(original, neutral)
    task['public']['success_criteria'] = success.replace(
        'Report a standalone ANSWER: <option> line in world_observation',
        'Return the selected room label in the answer field').replace(
        'Return the selected option in the answer field',
        'Return the selected room label in the answer field')
    qa = task['private']['evaluation']['parameters']['source_qa']
    qa.update(question=QUESTION, options=list(NEUTRAL_OPTIONS), answer=aliases[original_qa['answer']])
    provenance = {
        'contract_revision': REVISION, 'visibility': 'private',
        'validation_status': 'pending-native-validation',
        'parent_task_file': repo_path(PARENT_TASK), 'parent_task_sha256': sha256(parent_task_bytes),
        'parent_scene_file': repo_path(PARENT_SCENE), 'parent_scene_sha256': sha256(parent_scene_bytes),
        'source_file': repo_path(source_file), 'source_sha256': sha256(source_bytes),
        'original_source_qa': deepcopy(original_qa),
        'original_to_neutral_option_labels': aliases,
        'neutral_to_original_option_labels': {neutral: original for original, neutral in aliases.items()},
        'answer_region': 'locker_room_1', 'option_order_preserved': True,
        'native_validation_run': False, 'reference_execution_verified': False,
        'change': 'Remove answer-bearing room descriptions from public choices and match the private answer aliases.',
    }
    task.setdefault('repair', {})['regional_t11_neutral_labels'] = deepcopy(provenance)
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    # Preserve the complete historical scene representation outside metadata.
    scene_text = parent_scene_bytes.decode('utf-8')
    for key in ('id', 'version', 'status'):
        scene_text, count = re.subn(rf'^{key}: [^\n]*$', f'{key}: {scene[key]}',
                                   scene_text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f'The parent scene must have one top-level {key}.')
    if yaml.safe_load(scene_text) != scene:
        raise ValueError('Only the scene identity metadata may change.')
    rendered = {
        Path('task.yaml'): yaml.safe_dump(task, sort_keys=False, allow_unicode=False, width=88).encode('utf-8'),
        Path('scene.yaml'): scene_text.encode('utf-8'),
    }
    manifest = {
        'schema': REVISION, 'document_alias': 'T11', 'status': 'pending-native-validation',
        'visibility': 'private-provenance', **deepcopy(provenance),
        'task_file': repo_path(DEFAULT_OUTPUT / 'task.yaml'), 'task_sha256': sha256(rendered[Path('task.yaml')]),
        'scene_file': repo_path(DEFAULT_OUTPUT / 'scene.yaml'), 'scene_sha256': sha256(rendered[Path('scene.yaml')]),
        'allowed_task_changes': ['id', 'version', 'status', 'public.objective', 'public.success_criteria',
            'private.evaluation.parameters.source_qa.question', 'private.evaluation.parameters.source_qa.options',
            'private.evaluation.parameters.source_qa.answer', 'repair.regional_t11_neutral_labels'],
        'scene_changes': ['id', 'version', 'status'],
        'source_json_modified': False, 'geometry_modified': False, 'scorer_modified': False,
        'native_qualification': 'PENDING', 'anchor_observation_validation': 'PENDING',
        'reference_execution': 'PENDING',
    }
    rendered[Path('manifest.json')] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode('utf-8')
    return task, scene, manifest, rendered


def validate_outputs(output):
    """Use production loaders in diagnostic mode without starting a native world."""
    task = load_task(output / 'task.yaml', variant='adapted', run_mode='diagnostic')
    scene = load_scene(output / 'scene.yaml', variant='adapted', run_mode='diagnostic')
    if task.definition.scene != scene.definition.scene or task.source_path != scene.source_path:
        raise ValueError('Versioned T11 task and scene must load the same source world.')
    task.verify_unchanged()
    scene.verify_unchanged()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true', help='Compare deterministic bytes without writing.')
    args = parser.parse_args(argv)
    rendered = build_package()[3]
    output = args.output_dir.resolve()
    differing = [output / name for name, data in rendered.items()
                 if (output / name).is_symlink() or not (output / name).is_file()
                 or (output / name).read_bytes() != data]
    if args.check:
        for path in differing:
            print(f'WIA regional T11 differs or is missing: {path}', file=sys.stderr)
        if differing:
            return 1
        validate_outputs(output)
        print(f'WIA regional T11 is current: {output}')
        return 0
    for name, data in rendered.items():
        path = output / name
        if path.is_symlink() or (path.exists() and (not path.is_file() or path.read_bytes() != data)):
            raise ValueError(f'Refusing to overwrite a differing output: {path}')
    for name, data in rendered.items():
        path = output / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    validate_outputs(output)
    print(f'Wrote pending WIA regional T11: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
