"""Generate three pending TM-2 histories with one declared active Phase-2 state.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B \
  scripts/build_wia_temporal_pairs.py [--check] [--output-root PATH]
Generation uses the existing CPU converter; native equivalence remains pending.


Canonical invocation: python -m wia.authoring.commands.build_wia_temporal_pairs --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import yaml

from wia.resources import ROOT

from wia.authoring.conversion import ConversionContext, SourceInstance, map_records
from wia.authoring.converters.temporal_understanding_two_phase import CategoryConverter, PHYSICAL_OPERATION
from wia.tasks.interaction_setup import prepare_definitions
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition

REVISION = 'wia_tm2_matched_history_v1'
PUBLIC_TASK_ID = 'wia_tm2_matched_history_v1'
DEFAULT_OUTPUT_ROOT = ROOT
OUTPUT_RELATIVE = Path('explore_to_achieve/wia_v1/temporal')
SOURCE_RELATIVE = Path('Temporal Understanding/Unobserved Change/house_single_floor/dining_room_0')
AUTHORED_RELATIVE = Path('explore_to_achieve/sources/json_clean')
BASE_SOURCE = ROOT / AUTHORED_RELATIVE / SOURCE_RELATIVE / 'redesign_pair_01.json'
BUTTER_SOURCE = ROOT / AUTHORED_RELATIVE / SOURCE_RELATIVE / 'redesign_pair_03.json'
BASE_TASK = ROOT / (
    'explore_to_achieve/tasks/Temporal Understanding/Unobserved Change/'
    'house_single_floor/redesign_20261020/category_006a82c80075ae1fb18c5bc6.yaml')
BASE_SCENE = ROOT / 'explore_to_achieve/scenes' / BASE_TASK.relative_to(
    ROOT / 'explore_to_achieve/tasks')
ASSET_ROOT = Path('/local_home2/zhouxinyang/data/behavior/behavior-1k-assets')
ENCOUNTER_KEYS = ('encounter_policy', 'encounter_min_pixels',
                  'encounter_min_extent_px', 'encounter_max_distance_m')
VARIANTS = ('a', 'b', 'c')


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n').encode('utf-8')


def authored_sources():
    """Change only the declared histories and coherent authored-source metadata."""
    base = json.loads(BASE_SOURCE.read_text(encoding='utf-8'))
    donor = json.loads(BUTTER_SOURCE.read_text(encoding='utf-8'))
    base_boxes = base['metadata_json']['question_data']['boxes']
    shaker = deepcopy(base_boxes[0]['phase1_content'])
    butter = deepcopy(donor['metadata_json']['question_data']['boxes'][0]['phase2_content'])
    if (shaker['category'], shaker['model']) != ('shaker', 'hbheag'):
        raise ValueError('The baseline first-phase shaker identity changed.')
    if (butter['category'], butter['model']) != ('butter', 'ovirmc'):
        raise ValueError('The declared butter donor must be butter/ovirmc.')
    if [box['phase2_content']['category'] if box['phase2_content'] else None
            for box in base_boxes] != [None, 'jar_of_cumin', None]:
        raise ValueError('The baseline must declare the selected common Phase-2 state.')
    result = {}
    for index, (variant, first) in enumerate(zip(VARIANTS, (shaker, butter, None))):
        raw = deepcopy(base)
        identity = f'wia_history_{variant}_v1'
        answer = 'no' if first is None else 'yes'
        branch = 'no_change' if first is None else 'removed'
        raw.update(id=identity, answer=answer, options_json=json.dumps(['yes', 'no']))
        metadata = raw['metadata_json']
        metadata.update(question_id=identity, question_index=index, seed=0,
                        task_type='change_detection')
        question = metadata['question_data']
        question.update(question=raw['question'], answer=answer, options=['yes', 'no'],
                        box_count=len(base_boxes), task_type='change_detection', candidate_index=0)
        question.pop('question_metadata', None)
        question['boxes'][0].update(phase1_content=deepcopy(first), phase2_content=None,
                                    change_type='no_change' if first is None else 'remove')
        for box in question['boxes']:
            box.pop('phase_content', None)
        identity_for = lambda content: ({key: content[key] for key in ('category', 'model')}
                                       if content else None)
        metadata['authored_revision'] = {
            'revision': REVISION, 'history_variant': variant,
            'parent_source': BASE_SOURCE.relative_to(ROOT / AUTHORED_RELATIVE).as_posix(),
            'butter_donor_source': BUTTER_SOURCE.relative_to(ROOT / AUTHORED_RELATIVE).as_posix(),
            'event_branch': branch,
            'initial_inventory': [identity_for(box['phase1_content']) for box in question['boxes']],
            'final_inventory': [identity_for(box['phase2_content']) for box in question['boxes']],
            'original_source_qa_reused': False,
            'inherited_render_metadata_is_new_agent_observation': False,
        }
        result[variant] = raw
    return result


def declared_phase_2_state(raw, task, scene):
    """Describe active state/config equality without claiming native equivalence."""
    p = task['private']['evaluation']['parameters']
    render = raw['metadata_json']['question_data']['render']
    primary = map_records(render['image'])
    runtime = SceneDefinition.model_validate(scene).runtime.model_dump(mode='json')
    config_files = {key: {
        'file': runtime[key],
        'sha256': hashlib.sha256((ROOT / runtime[key]).read_bytes()).hexdigest(),
    } for key in ('robot_camera_config', 'trunk_config', 'viewer_config')}
    return {
        'scene': raw['scene'], 'room': raw['room'],
        'boxes': [{
            'binding_key': key, 'label': p['box_labels'][key],
            'identity': deepcopy(p['identities'][key]),
            'source_geometry': deepcopy(p['source_objects'][key]),
        } for key in p['container_keys']],
        'active_contents': [{
            'binding_key': key, 'container_binding': p['relations'][key],
            'identity': deepcopy(p['identities'][key]),
            'source_geometry': deepcopy(p['source_objects'][key]),
        } for key in p['phase_keys']['image2']],
        'public_spare_inventory': deepcopy(p['public_spare_inventory']),
        'apparatus': deepcopy(p['apparatus']),
        'sensor_and_robot_start_configuration': {
            'scene_runtime': runtime, 'configuration_files': config_files,
            'robot_pose_patches': {
                name: [deepcopy(patch) for patch in variant['patches']
                       if patch['operation'] == 'robot_pose']
                for name, variant in scene['scene']['variants'].items()},
            'source_camera_policy': deepcopy(raw['metadata_json']['camera_poses']),
            'source_primary_camera_poses': {
                phase: deepcopy(primary[phase]['camera_poses']) for phase in ('image1', 'image2')},
            'source_gt_camera_metadata': deepcopy(render['gt_view']),
        },
    }


def build_package():
    """Regenerate authored inputs and owned converter outputs entirely in staging."""
    baseline = yaml.safe_load(BASE_TASK.read_text(encoding='utf-8'))
    baseline_scene = yaml.safe_load(BASE_SCENE.read_text(encoding='utf-8'))
    baseline_p = baseline['private']['evaluation']['parameters']
    encounter = {key: deepcopy(baseline_p[key]) for key in ENCOUNTER_KEYS}
    if encounter != dict(zip(ENCOUNTER_KEYS, ('mounted_visibility_v1', 144, 12, 2.0))):
        raise ValueError('The selected T40 encounter thresholds changed; review the baseline explicitly.')
    if len(baseline_p['public_spare_inventory']) != 4:
        raise ValueError('The complete four-item spare inventory must be retained.')
    config = {
        'task_edit2_temporal': True, 'survey_revision': 'free_survey_10_v1',
        'allow_multiple_changes': True, 'original_asset_root': str(ASSET_ROOT),
        'change_destination': {'binding_key': 'change_collection_tray',
                               'public_label': 'the collection and inspection tray', 'relation': 'on_top'},
        'change_apparatus': deepcopy(baseline_p['apparatus']),
        'change_spares': deepcopy(baseline_p['public_spare_inventory']),
    }
    sources = authored_sources()
    records, definitions, common_states, full_public_payloads, rendered = [], {}, [], [], {}
    with TemporaryDirectory(prefix='wia_temporal_pairs_', dir='/tmp') as staging:
        for variant in VARIANTS:
            raw = sources[variant]
            identity = SOURCE_RELATIVE / f'wia_history_{variant}_v1.json'
            source_file = AUTHORED_RELATIVE / identity
            source_data = json_bytes(raw)
            staged = Path(staging) / identity
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(source_data)
            source = SourceInstance(staged, identity.as_posix(), 'Temporal Understanding', 'Unobserved Change')
            result = CategoryConverter().convert(source, ConversionContext(ROOT, seed=0, config=deepcopy(config)))
            task, scene = result.get('task_definition'), result.get('scene_definition')
            if task is None or scene is None:
                raise ValueError(f'Existing converter did not build history {variant}: {result}')
            p = task['private']['evaluation']['parameters']
            p.update(deepcopy(encounter))
            prepare_definitions(source, scene, task, result.get('interaction_policy'))
            expected_target = {'a': 'public_spare_1', 'b': 'public_spare_3', 'c': None}[variant]
            expected_branch = 'no_change' if variant == 'c' else 'removed'
            if (p['physical_target_key'] != expected_target or p['physical_branch'] != expected_branch
                    or p['physical_destination_key'] != (None if variant == 'c' else 'container_00')
                    or p['original_answer'] != raw['answer']):
                raise ValueError(f'History {variant} does not produce the specified native binding/answer.')
            if task['public'] != baseline['public']:
                raise ValueError('The existing converter must preserve the identical generic public payload.')
            # Exact instance identity is private provenance. All path-selected
            # histories intentionally share the same neutral public task ID.
            task.update(id=PUBLIC_TASK_ID, version=1, status='pending')
            scene.update(id=f'wia_tm2_history_{variant}_v1_scene', version=1, status='pending')
            task['repair'] = {
                'contract_revision': REVISION, 'validation_status': 'pending-native-validation',
                'parent_task_file': BASE_TASK.relative_to(ROOT).as_posix(),
                'parent_source_file': BASE_SOURCE.relative_to(ROOT).as_posix(),
                'encounter_threshold_basis': 'Preserved selected T40 baseline; not newly calibrated.',
                'frozen_initial_overview_retained': True,
                'full_input_history_status': 'history_available',
                'native_validation_run': False,
            }
            validated_task = CategoryTaskDefinition.model_validate(task)
            SceneDefinition.model_validate(scene)
            full_public_payloads.append({
                'task_id': validated_task.id, 'task_version': validated_task.version,
                **validated_task.public.model_dump(mode='json')})
            if (SceneDefinition.model_validate(scene).runtime !=
                    SceneDefinition.model_validate(baseline_scene).runtime):
                raise ValueError('The selected baseline sensor/runtime configuration must be retained.')
            robot_poses = lambda value: {
                name: [patch for patch in variant['patches'] if patch['operation'] == 'robot_pose']
                for name, variant in value['scene']['variants'].items()}
            if robot_poses(scene) != robot_poses(baseline_scene):
                raise ValueError('The selected baseline robot-start configuration must be retained.')
            task_file = OUTPUT_RELATIVE / 'tasks' / f'history_{variant}_v1.yaml'
            scene_file = OUTPUT_RELATIVE / 'scenes' / f'history_{variant}_v1.yaml'
            rendered[source_file] = source_data
            for filename, definition in ((task_file, task), (scene_file, scene)):
                rendered[filename] = yaml.safe_dump(
                    definition, sort_keys=False, allow_unicode=False, width=88).encode('utf-8')
            definitions[variant] = (task, scene)
            common_states.append(declared_phase_2_state(raw, task, scene))
            groups = next(patch for patch in scene['scene']['variants']['adapted']['patches']
                          if patch['operation'] == 'prepare_category_interaction')
            records.append({
                'history_variant': variant, 'source_file': source_file.as_posix(),
                'source_identity': identity.as_posix(),
                'task_file': task_file.as_posix(), 'scene_file': scene_file.as_posix(),
                'source_file_sha256': hashlib.sha256(source_data).hexdigest(),
                'task_file_sha256': hashlib.sha256(rendered[task_file]).hexdigest(),
                'scene_file_sha256': hashlib.sha256(rendered[scene_file]).hexdigest(),
                'source_id': raw['id'], 'status': 'pending',
                'private_answer': raw['answer'], 'physical_branch': p['physical_branch'],
                'physical_target_key': p['physical_target_key'],
                'physical_destination_key': p['physical_destination_key'],
                'placement_relation': p['placement_relation'],
                'source_event_recipe': deepcopy(p['source_event_recipe']),
                'phase_1_inventory': deepcopy(raw['metadata_json']['authored_revision']['initial_inventory']),
                'phase_2_inventory': deepcopy(raw['metadata_json']['authored_revision']['final_inventory']),
                'phase_1_dependent_support_groups': deepcopy(groups['placement_groups']),
                'offstage_phase_1_bindings': {
                    key: deepcopy(p['identities'][key]) for key in p['phase_keys']['image1']},
                'native_gates': deepcopy(result['native_gates']),
            })
    if not all(state == common_states[0] for state in common_states):
        raise ValueError('Declared active Phase-2 state/configuration differs across histories.')
    if not all(payload == full_public_payloads[0] for payload in full_public_payloads):
        raise ValueError('Full public task payloads must be identical across histories.')
    if len({json.dumps(row['phase_1_inventory'], sort_keys=True) for row in records}) != 3:
        raise ValueError('The three declared first-phase histories must differ.')
    if len({row['physical_target_key'] for row in records}) != 3:
        raise ValueError('The three history-dependent physical target choices must differ.')
    manifest = {
        'schema': REVISION, 'status': 'pending-native-validation',
        'baseline_task_file': BASE_TASK.relative_to(ROOT).as_posix(),
        'baseline_scene_file': BASE_SCENE.relative_to(ROOT).as_posix(),
        'baseline_source_file': BASE_SOURCE.relative_to(ROOT).as_posix(),
        'butter_donor_source_file': BUTTER_SOURCE.relative_to(ROOT).as_posix(),
        'converter_module': CategoryConverter.__module__, 'converter_operation': PHYSICAL_OPERATION,
        'seed': 0, 'encounter_policy': encounter,
        'encounter_threshold_basis': 'Preserved selected T40 baseline; not newly calibrated.',
        'public_payload_equal': True, 'common_public_payload': full_public_payloads[0],
        'common_public_task_section': deepcopy(baseline['public']),
        'public_task_identity_policy': (
            'A deliberate shared neutral task ID/version is public. Exact instance identity '
            'is the private source identity, variant key, task path and task-file SHA256.'),
        'declared_active_phase_2_state_equal': True,
        'common_declared_active_phase_2_state': common_states[0],
        'phase_1_histories_differ': True, 'expected_physical_targets_differ': True,
        'observation_contract': {
            'frozen_initial_overview_retained': True,
            'full_input_history_status': 'history_available',
            'memory_free_claim': False,
            'inherited_render_metadata_is_new_agent_observation': False,
        },
        'native_phase_2_equivalence': 'PENDING',
        'observation_shortcut_audit': 'PENDING', 'reference_execution': 'PENDING',
        'native_experiments_run': False,
        'limitations': [
            'Declared active Phase-2 equality does not establish native world or delivered-image equality.',
            'Inactive source actors and history-dependent support rebasing may differ; native equivalence is pending.',
            'The retained frozen initial overview makes full input history available; no memory-free claim is made.',
            'No native phase observations, reference execution, shortcut audit or physical success were produced.',
        ],
        'instances': records,
    }
    rendered[OUTPUT_RELATIVE / 'manifest.json'] = json_bytes(manifest)
    return sources, definitions, manifest, rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--check', action='store_true', help='Compare exact bytes without writing outputs.')
    args = parser.parse_args(argv)
    rendered = build_package()[3]
    output = args.output_root.resolve()
    if args.check:
        mismatches = [output / name for name, data in rendered.items()
                      if not (output / name).is_file() or (output / name).read_bytes() != data]
        for path in mismatches:
            print(f'WIA temporal definition differs or is missing: {path}', file=sys.stderr)
        if mismatches:
            return 1
        print(f'WIA temporal histories are current: {output / OUTPUT_RELATIVE}')
        return 0
    for name, data in rendered.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(f'Wrote three pending WIA temporal histories: {output / OUTPUT_RELATIVE}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
