"""Derive T37's same-desk aisle-facing pickup repair from reviewed static evidence.

Run: PYTHONPATH=src:../Show-Harness-Explore python scripts/revise_t37_aisle_pickup.py
Native settling, carried-tray access and acquisition remain pending. The capture
refresh is a separate explicit operation; old definitions are retained verbatim.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

from revise_wia_final_state_tasks import replace_parameters
from wia.resources import ROOT
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition
from wia.authoring.converters.wia_room_collection import validate_layout


def main():
    catalogue_path = ROOT / 'explore_to_achieve/wia_v1/release/catalogue.json'
    catalogue = json.loads(catalogue_path.read_text())
    row = next(row for row in catalogue['tasks'] if row['alias'] == 'T37')
    target_path = ROOT / 'explore_to_achieve/wia_v1/t37_access_20261030'
    if target_path.exists():
        raise FileExistsError('Preserve the existing derived revision.')
    evidence_path = ROOT / 'outputs/t37_access_repair_20261030/local_0.json'
    evidence = json.loads(evidence_path.read_text())
    witness = evidence['result']['witness']
    if evidence['result']['status'] != 'found_witness' or witness['arm'] != 'left':
        raise ValueError('The proposed layout requires its reviewed left-arm geometric witness.')
    task_path, scene_path = ROOT / row['task_file'], ROOT / row['scene_file']
    task_bytes, scene_bytes = task_path.read_bytes(), scene_path.read_bytes()
    task, scene = yaml.safe_load(task_bytes), yaml.safe_load(scene_bytes)
    old_parameters = task['private']['evaluation']['parameters']
    parameters = deepcopy(old_parameters)
    target = next(item for item in parameters['layout']['added_inventory'] if item['name'] == 'wia_t37_target_03')
    original_target = deepcopy(target)
    target.update(position=[-16.22, -7.55, .837230920791626],
                  pickup_xy_policy='preserve_authored_v1', maximum_pickup_xy_drift_m=.015)
    validate_layout(parameters['layout'], parameters['operation'])
    task = replace_parameters(task, old_parameters, parameters)
    scene = replace_parameters(scene, old_parameters, parameters)
    task['version'] += 1
    scene['version'] += 1
    task['repair']['aisle_pickup_revision'] = {'revision': 't37_same_desk_aisle_v1',
        'target': target['name'], 'original_target': original_target,
        'geometric_evidence': str(evidence_path), 'evidence_sha256': hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        'local_open_gripper_witness': True, 'original_start_route_verified': False,
        'carried_tray_verified': False, 'native_initialization_verified': False, 'native_acquisition_verified': False,
        'scope': 'One target moved on the same native cubicle desk; room, furniture, assets, scale, other objects and collection evaluator retained.'}
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    if task['scene'] != scene['scene']:
        raise ValueError('Task and scene initialization must match.')
    target_path.mkdir(parents=True)
    (target_path / 'parent.task.yaml').write_bytes(task_bytes)
    (target_path / 'parent.scene.yaml').write_bytes(scene_bytes)
    old_selection = deepcopy(row)
    for name, content, active in [('task', task, task_path), ('scene', scene, scene_path)]:
        data = yaml.safe_dump(content, sort_keys=False, width=100).encode()
        derived = target_path / f't37.{name}.yaml'
        derived.write_bytes(data)
        active.write_bytes(data)
        row[f'source_{name}_file'] = str(derived.relative_to(ROOT))
        row[f'source_{name}_file_sha256'] = hashlib.sha256(data).hexdigest()
    (target_path / 'manifest.json').write_text(json.dumps({'previous_selection': old_selection,
        'current_selection': row, 'changes': task['repair']['aisle_pickup_revision'],
        'native_validation_pending': True}, indent=2) + '\n')
    catalogue_path.write_text(json.dumps(catalogue, indent=2) + '\n')
    print('T37 same-desk pickup revision derived; native validation remains pending.')


if __name__ == '__main__':
    main()
