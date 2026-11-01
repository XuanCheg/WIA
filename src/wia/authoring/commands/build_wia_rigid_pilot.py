"""Build the pending, separately versioned WIA T25 banana containment pilot.

Run: python -B scripts/build_wia_rigid_pilot.py [--check] [--output-dir PATH]
This CPU-only generator does not run native qualification or calibrate thresholds.


Canonical invocation: python -m wia.authoring.commands.build_wia_rigid_pilot --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import re
import sys

import yaml

from wia.resources import ROOT

from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition

PARENT_RELATIVE = Path(
    'Physical Structure/Rigid Containment/hall_arch_wood/empty_room_0/'
    'corev2__category_654f8bfa37612193a879dd59__pillar_repair.yaml')
PARENT_TASK = ROOT / 'explore_to_achieve/tasks' / PARENT_RELATIVE
PARENT_SCENE = ROOT / 'explore_to_achieve/scenes' / PARENT_RELATIVE
DEFAULT_OUTPUT = ROOT / 'explore_to_achieve/wia_v1'
TASK_FILENAME = Path('tasks/rigid_t25_v1.yaml')
SCENE_FILENAME = Path('scenes/rigid_t25_v1.yaml')
REVISION = 'wia_rigid_partial_insertion_v1'
CAPABILITIES = ('base_translation', 'base_rotation', 'ee_translation',
                'ee_rotation', 'trunk_height', 'gripper')


def build_definitions():
    """Retain the source world and declare only the pilot's scored task contract."""
    original_task = yaml.safe_load(PARENT_TASK.read_text(encoding='utf-8'))
    original_scene = yaml.safe_load(PARENT_SCENE.read_text(encoding='utf-8'))
    if original_task['source'] != original_scene['source']:
        raise ValueError('The historical T25 task and scene sources must agree.')
    scene = deepcopy(original_scene)
    scene.update(id='wia_rigid_t25_v1_scene', version=1, status='pending')
    public = deepcopy(original_task['public'])
    public['objective'] += (
        ' Grasp the banana with the robot, lift it at least 5 cm, and partially '
        'insert it into your chosen container. Release it and withdraw the gripper '
        'at least 2 cm clear of the banana.')
    public['success_criteria'] = (
        'Answer the original containment question and complete a real grasp, a lift '
        'of at least 5 cm, and stable partial insertion into the chosen container. '
        'After release, keep the banana partially inside with the gripper at least '
        '2 cm clear for at least 0.5 seconds, with linear speed at most 0.02 m/s '
        'and angular speed at most 0.1 rad/s. An object resting entirely on top '
        'does not count as inside. Submit the container label in the answer field and declare DONE.')
    public['constraints'].extend([
        'Use the existing guarded robot controls and visible container labels.',
        'Preserve all asset scales and keep every container within 2 cm and '
        '0.05 radians of its starting pose.',
        'The banana may be removed after its released, stable partial insertion '
        'has been verified.',
    ])
    bindings = deepcopy(scene['bindings'])
    task = {
        'kind': 'category_robot_task', 'id': 'wia_rigid_t25_v1',
        'version': 1, 'status': 'pending', 'source': deepcopy(original_task['source']),
        'public': public, 'requirements': {'capabilities': list(CAPABILITIES)},
        'private': {
            'bindings': bindings,
            'evaluation': {
                'type': 'category_operation',
                'module': 'wia.authoring.converters.rigid_partial_insertion',
                'operation': 'pick_release_partial_insert',
                'binding_keys': list(bindings),
                'parameters': {
                    'scorer_version': REVISION,
                    'targets': ['target'],
                    'containers': ['container_0', 'container_1', 'container_2'],
                    'expected_containers': {'target': 'container_1'},
                    'target_labels': {'target': 'banana'},
                    'expected_answers': {'target': 'tupperware'},
                    'minimum_lift_m': 0.05,
                    'stability_duration_s': 0.5,
                    'maximum_linear_speed_m_s': 0.02,
                    'maximum_angular_speed_rad_s': 0.1,
                    'max_sample_gap_s': 0.1,
                    'interior_margin_m': 0.002,
                    'penetration_tolerance_m': 0.001,
                    'withdrawal_clearance_m': 0.02,
                    'container_pose_tolerance_m': 0.02,
                    'maximum_container_rotation_rad': 0.05,
                    'scale_relative_tolerance': 0.0001,
                    'allow_remove_after_verification': True,
                    'initial_supports': {
                        'target': 'pickup_support', 'container_0': 'table',
                        'container_1': 'table', 'container_2': 'table',
                    },
                    'initial_support_tolerance_m': 0.015,
                    'maximum_initial_linear_speed_m_s': 0.02,
                    'maximum_initial_angular_speed_rad_s': 0.2,
                },
            },
        },
        'scene': deepcopy(scene['scene']),
        # Generic category qualification cannot encode SceneQualification. The
        # operation checks the retained native support contract independently.
        'qualification': {'on_failure': 'skip', 'support': None,
                          'initial_visibility': 'none'},
        'defaults': {**deepcopy(original_task['defaults']), 'action_mode': 'parameterized'},
        'repair': {
            'contract_revision': REVISION,
            'validation_status': 'pending-native-validation',
            'parent_task_file': PARENT_TASK.relative_to(ROOT).as_posix(),
            'parent_scene_file': PARENT_SCENE.relative_to(ROOT).as_posix(),
            'scene_file': (DEFAULT_OUTPUT / SCENE_FILENAME).relative_to(ROOT).as_posix(),
            'original_scene_qualification': deepcopy(scene['qualification']),
            'native_validation_run': False,
            'physically_qualified': False,
            'threshold_basis': 'Explicit pilot thresholds; native calibration is pending.',
        },
    }
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    return task, scene


def render_definitions():
    """Serialize deterministically, preserving all other historical scene bytes."""
    task, scene = build_definitions()
    scene_text = PARENT_SCENE.read_text(encoding='utf-8')
    for key in ('id', 'version', 'status'):
        scene_text, count = re.subn(rf'^{key}: [^\n]*$', f'{key}: {scene[key]}',
                                   scene_text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f'The historical scene must have a top-level {key}.')
    if yaml.safe_load(scene_text) != scene:
        raise ValueError('Scene serialization must change only its version metadata.')
    task_text = yaml.safe_dump(task, sort_keys=False, allow_unicode=False, width=88)
    return {TASK_FILENAME: task_text.encode('utf-8'),
            SCENE_FILENAME: scene_text.encode('utf-8')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true',
                        help='Compare exact output bytes without writing.')
    args = parser.parse_args(argv)
    rendered = render_definitions()
    output = args.output_dir.resolve()
    if args.check:
        mismatches = [output / name for name, data in rendered.items()
                      if not (output / name).is_file() or (output / name).read_bytes() != data]
        if mismatches:
            for path in mismatches:
                print(f'WIA Rigid pilot differs or is missing: {path}', file=sys.stderr)
            return 1
        print(f'WIA Rigid pilot is current: {output}')
        return 0
    for name, data in rendered.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f'Wrote pending WIA Rigid pilot: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
