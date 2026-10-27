"""Build pending T22-T24 Rigid definitions while preserving selected native scenes.

Run: python -B scripts/build_wia_rigid_family.py [--check] [--output-dir PATH]
CPU checks: python -B -m pytest -q -p no:cacheprovider tests/test_wia_rigid_family.py
This generator performs schema/loader checks, not native calibration or execution.


Canonical invocation: python -m wia.authoring.commands.build_wia_rigid_family --help
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

from wia.tasks.loader import load_scene, load_task  # noqa: E402
from wia.tasks.schema import CategoryTaskDefinition, RigidContainmentTaskDefinition, SceneDefinition  # noqa: E402
from wia.tasks.sources import resolve_source  # noqa: E402

DEFAULT_OUTPUT = ROOT / 'explore_to_achieve/wia_v1/rigid_family_v1'
REVISION = 'wia_rigid_partial_insertion_v1'
FAMILY_REVISION = 'wia_rigid_family_v1'
CAPABILITIES = ('base_translation', 'base_rotation', 'ee_translation',
                'ee_rotation', 'trunk_height', 'gripper')
PHYSICAL_PARAMETERS = {
    'minimum_lift_m': .05, 'stability_duration_s': .5,
    'maximum_linear_speed_m_s': .02, 'maximum_angular_speed_rad_s': .1,
    'max_sample_gap_s': .1, 'interior_margin_m': .002,
    'penetration_tolerance_m': .001, 'withdrawal_clearance_m': .02,
    'container_pose_tolerance_m': .02, 'maximum_container_rotation_rad': .05,
    'scale_relative_tolerance': .0001,
}
CASES = {
    'T22': {
        'task_relative': 'Physical Structure/Rigid Containment/Beechwood_0_int/living_room_1/Beechwood_0_int_scene_dict_storage2_0_revised.yaml',
        'scene_relative': 'Physical Structure/Rigid Containment/Beechwood_0_int/living_room_1/Beechwood_0_int_scene_dict_storage2_0_revised.yaml',
        'target_labels': {'target': 'gift box'},
        'expected_containers': {'target': 'container_1'},
        'expected_answers': {'target': 'medium packing box'},
    },
    'T23': {
        'task_relative': 'Physical Structure/Rigid Containment/Beechwood_1_int/television_room_0/Beechwood_1_int_scene_dict_storage2_1_revised__criterion_4.yaml',
        'scene_relative': 'Physical Structure/Rigid Containment/Beechwood_1_int/television_room_0/Beechwood_1_int_scene_dict_storage2_1_revised__multi_extra1_extra2.yaml',
        'target_labels': {'target': 'granola box', 'extra1': 'spray can', 'extra2': 'colored pencil'},
        'expected_containers': {'target': 'container_1', 'extra1': 'container_2', 'extra2': 'container_2'},
        'expected_answers': {'target': 'wicker basket', 'extra1': 'bucket', 'extra2': 'bucket'},
    },
    'T24': {
        'task_relative': 'Physical Structure/Rigid Containment/Ihlen_1_int/living_room_0/Ihlen_1_int_scene_dict_storage2_1_revised.yaml',
        'scene_relative': 'Physical Structure/Rigid Containment/Ihlen_1_int/living_room_0/Ihlen_1_int_scene_dict_storage2_1_revised.yaml',
        'target_labels': {'target': 'first aid kit'},
        'expected_containers': {'target': 'container_1'},
        'expected_answers': {'target': 'packing box'},
    },
}


def parent_paths(alias):
    case = CASES[alias]
    return (ROOT / 'explore_to_achieve/tasks' / case['task_relative'],
            ROOT / 'explore_to_achieve/scenes' / case['scene_relative'])


def filenames(alias):
    return Path(f'tasks/rigid_{alias.lower()}_v1.yaml'), Path(f'scenes/rigid_{alias.lower()}_v1.yaml')


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_original_qa(alias, task, scene):
    """Verify assigned answers using source QA and named target/container identities."""
    case = CASES[alias]
    question_path = resolve_source(task['source']['json'])
    scene_source_path = resolve_source(scene['source']['json'])
    raw = json.loads(question_path.read_text(encoding='utf-8'))
    metadata = raw['metadata_json']
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    if raw.get('small_task') != 'Rigid Containment' or raw.get('question') != task['public']['objective']:
        raise ValueError(alias + ': the original QA question must match the selected task.')
    if list(task['targets']) != list(case['target_labels']):
        raise ValueError(alias + ': selected native targets changed.')
    answer = raw['answer']
    if raw['answer_type'] == 'sequence':
        sequence = json.loads(answer) if isinstance(answer, str) else answer
        actual_answers = {item['_key']: item['value'] for item in sequence}
        if len(actual_answers) != len(sequence):
            raise ValueError(alias + ': source QA contains duplicate object labels.')
    else:
        actual_answers = {next(iter(case['target_labels'].values())): answer}
    expected = {label: case['expected_answers'][key] for key, label in case['target_labels'].items()}
    if actual_answers != expected:
        raise ValueError(alias + ': assigned answers differ from the original QA.')
    source_objects = {'target': metadata['containee']}
    source_objects.update({item['_key']: item for item in metadata.get('extras', [])})
    containers = {item['_key']: item for item in metadata['containers']}
    for key, label in case['target_labels'].items():
        item = source_objects[key]
        if item['cat'].replace('_', ' ') != label or scene['bindings'][key]['category'] != item['cat']:
            raise ValueError(alias + ': named target identity differs from source QA.')
        selected = containers[item['target_container']]
        container_key = f"container_{selected['idx']}"
        if (container_key != case['expected_containers'][key]
                or scene['bindings'][container_key]['name'] != f"obj_container_{selected['idx']}"
                or scene['bindings'][container_key]['category'] != selected['cat']):
            raise ValueError(alias + ': assigned container differs from source QA identity.')
    return {
        'question_source_file': str(question_path), 'question_source_identity': task['source']['json'],
        'question_source_sha256': _sha256(question_path),
        'scene_source_file': str(scene_source_path), 'scene_source_identity': scene['source']['json'],
        'scene_source_sha256': _sha256(scene_source_path),
        'original_qa_id': raw.get('id'), 'original_question': raw['question'],
        'original_answer': deepcopy(raw['answer']), 'answer_type': raw['answer_type'],
        'original_options': json.loads(raw['options_json']),
        'verified_answer_by_public_label': actual_answers,
        'verification_basis': 'Original QA answer and named source target_container/idx metadata; no AABB inference.',
    }


def build_definitions(alias, *, output_dir=DEFAULT_OUTPUT):
    case = CASES[alias]
    parent_task, parent_scene = parent_paths(alias)
    task_original = yaml.safe_load(parent_task.read_text(encoding='utf-8'))
    scene_original = yaml.safe_load(parent_scene.read_text(encoding='utf-8'))
    resolved_parent = RigidContainmentTaskDefinition.model_validate(task_original)
    SceneDefinition.model_validate(scene_original)
    provenance = _verify_original_qa(alias, task_original, scene_original)
    scene = deepcopy(scene_original)
    identifier = f'wia_rigid_{alias.lower()}_family_v1'
    scene.update(id=identifier + '_scene', version=1, status='pending')
    targets = list(task_original['targets'])
    containers = ['container_0', 'container_1', 'container_2']
    qualification = scene['qualification']
    supports = {key: qualification.get('object_supports', {}).get(key, qualification['support'])
                for key in [*targets, *containers]}
    if any(support not in scene['bindings'] for support in supports.values()):
        raise ValueError(alias + ': every initial support must remain explicitly bound.')
    parameters = {
        'scorer_version': REVISION, 'targets': targets, 'containers': containers,
        'expected_containers': deepcopy(case['expected_containers']),
        'target_labels': deepcopy(case['target_labels']), 'expected_answers': deepcopy(case['expected_answers']),
        **PHYSICAL_PARAMETERS, 'allow_remove_after_verification': True,
        'initial_supports': supports,
        'initial_support_tolerance_m': qualification['support_tolerance_m'],
        'maximum_initial_linear_speed_m_s': qualification['maximum_linear_speed_m_s'],
        'maximum_initial_angular_speed_rad_s': qualification['maximum_angular_speed_rad_s'],
    }
    public = deepcopy(task_original['public'])
    public['objective'] += (
        ' For every named object, perform a real robot grasp, lift it at least 5 cm, '
        'and partially insert it into the container selected for that object. Release '
        'it and withdraw the robot hands at least 2 cm clear.')
    if len(targets) > 1:
        answer_instruction = 'Submit ANSWER: ' + json.dumps(
            {label: '<container label>' for label in case['target_labels'].values()}) + ' and declare DONE.'
    else:
        answer_instruction = 'Submit ANSWER: <container label> and declare DONE.'
    public['success_criteria'] = (
        'Answer the original containment question for every named object. For each object, '
        'complete an actual grasp and a lift of at least 5 cm, then release it partially '
        'inside its selected container while touching that container. With no robot contact '
        'or support, and the robot hands at least 2 cm clear, maintain this state for at '
        'least 0.5 seconds with linear speed at most 0.02 m/s and angular speed at most '
        '0.1 rad/s. An object resting entirely on top does not count as inside. Finish '
        'with empty hands and the working gripper open. ' + answer_instruction)
    public['constraints'].extend([
        'Use the existing guarded public robot controls and visible object/container labels.',
        'Preserve all asset scales and keep every container within 2 cm and 0.05 radians of its starting pose.',
        'Targets may be verified sequentially. An object may be removed after its released, '
        'stable partial insertion has been completely verified, to free a shared container. '
        'Simultaneous packing is not required.',
    ])
    _, scene_filename = filenames(alias)
    task = {
        'kind': 'category_robot_task', 'id': identifier, 'version': 1, 'status': 'pending',
        'source': deepcopy(task_original['source']), 'public': public,
        'requirements': {'capabilities': list(CAPABILITIES)},
        'private': {'bindings': deepcopy(scene['bindings']), 'evaluation': {
            'type': 'category_operation', 'module': 'core.esi_converters.rigid_partial_insertion',
            'operation': 'pick_release_partial_insert', 'binding_keys': list(scene['bindings']),
            'parameters': parameters}},
        'scene': deepcopy(scene['scene']),
        'qualification': {'on_failure': 'skip', 'support': None, 'initial_visibility': 'none'},
        'defaults': {**resolved_parent.defaults.model_dump(mode='json'), 'action_mode': 'parameterized'},
        'repair': {'family_revision': FAMILY_REVISION, 'contract_revision': REVISION,
            'document_alias': alias, 'validation_status': 'pending-native-calibration-and-scene-review',
            'parent_task_file': str(parent_task), 'parent_scene_file': str(parent_scene),
            'scene_file': str(Path(output_dir).resolve() / scene_filename),
            'original_scene_qualification': deepcopy(qualification),
            'original_qa_provenance': provenance,
            'parameters_provisional': True, 'parameters_calibrated': False,
            'threshold_basis': 'Explicit common provisional Rigid family criteria; native calibration is pending.',
            'native_validation_run': False, 'physically_qualified': False,
            'legal_robot_reference_verified': False,
            'reference_execution_status': 'Deferred until user scene review.'},
    }
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    return task, scene


def render_definitions(*, output_dir=DEFAULT_OUTPUT):
    output = Path(output_dir).resolve()
    rendered, rows = {}, []
    for alias in CASES:
        task, scene = build_definitions(alias, output_dir=output)
        parent_task, parent_scene = parent_paths(alias)
        task_filename, scene_filename = filenames(alias)
        scene_text = parent_scene.read_text(encoding='utf-8')
        for key in ('id', 'version', 'status'):
            scene_text, count = re.subn(rf'^{key}: [^\n]*$', f'{key}: {scene[key]}',
                                       scene_text, count=1, flags=re.MULTILINE)
            if count != 1:
                raise ValueError(f'{alias}: the parent scene must contain a top-level {key}.')
        if yaml.safe_load(scene_text) != scene:
            raise ValueError(alias + ': scene serialization changed more than version metadata.')
        rendered[task_filename] = yaml.safe_dump(task, sort_keys=False, allow_unicode=False, width=88).encode()
        rendered[scene_filename] = scene_text.encode()
        rows.append({'document_alias': alias, 'task_file': str(output / task_filename),
            'scene_file': str(output / scene_filename), 'parent_task_file': str(parent_task),
            'parent_scene_file': str(parent_scene), 'parent_task_sha256': _sha256(parent_task),
            'parent_scene_sha256': _sha256(parent_scene),
            'task_sha256': hashlib.sha256(rendered[task_filename]).hexdigest(),
            'scene_sha256': hashlib.sha256(rendered[scene_filename]).hexdigest(),
            'original_qa_provenance': deepcopy(task['repair']['original_qa_provenance'])})
    manifest = {'schema': 'wia_rigid_family_manifest_v1', 'family_revision': FAMILY_REVISION,
        'status': 'pending_scene_review_and_native_calibration', 'output_dir': str(output),
        'parameters_provisional': True, 'parameters_calibrated': False,
        'physical_parameters': PHYSICAL_PARAMETERS, 'native_validation_run': False,
        'legal_robot_reference_verified': False, 'physical_success_verified': False,
        'reference_execution_status': 'Deferred until user scene review.', 'tasks': rows}
    rendered[Path('manifest.json')] = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    return rendered


def validate_outputs(output_dir):
    for alias in CASES:
        task_filename, scene_filename = filenames(alias)
        task = load_task(Path(output_dir) / task_filename, variant='tabletop', run_mode='diagnostic')
        scene = load_scene(Path(output_dir) / scene_filename, variant='tabletop', run_mode='diagnostic')
        if task.definition.scene != scene.definition.scene or task.scene_name != scene.scene_name or task.room_name != scene.room_name:
            raise ValueError(alias + ': generated task and scene no longer describe the same native world.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true', help='Compare exact output bytes without writing.')
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    rendered = render_definitions(output_dir=output)
    conflicts = [output / name for name, data in rendered.items()
                 if (output / name).exists() and (output / name).read_bytes() != data]
    missing = [output / name for name in rendered if not (output / name).is_file()]
    if conflicts or (args.check and missing):
        for path in [*conflicts, *(missing if args.check else [])]:
            print(f'WIA Rigid family differs or is missing; refusing overwrite: {path}', file=sys.stderr)
        return 1
    if not args.check:
        for name, data in rendered.items():
            path = output / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                print(f'Wrote pending WIA Rigid definition: {path}')
    validate_outputs(output)
    print(f'WIA Rigid family schema and diagnostic loader checks passed: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
