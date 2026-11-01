"""Derive active outcome-based placement definitions without changing old sources.

Run with behavior Python and PYTHONPATH=src:../Show-Harness-Explore:
  python scripts/revise_wia_final_state_tasks.py --apply
Without --apply, print the selected aliases only. Screenshots are refreshed by
scripts/refresh_wia_captures.py after evaluator review and validation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import yaml

from wia.resources import ROOT
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition


RELEASE = ROOT / 'explore_to_achieve/wia_v1/release'
DERIVED = ROOT / 'explore_to_achieve/wia_v1/final_state_20261030'
MODULES = {'action_order', 'physical_dynamics_stacking_stability',
           'rigid_partial_insertion', 'wia_dimensional', 'wia_mirror', 'wia_spatial_tray'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_parameters(value, old, new):
    if value == old:
        return deepcopy(new)
    if isinstance(value, dict):
        return {key: replace_parameters(child, old, new) for key, child in value.items()}
    if isinstance(value, list):
        return [replace_parameters(child, old, new) for child in value]
    return value


def revise(task):
    module = task['private']['evaluation']['module'].rsplit('.', 1)[-1]
    old = task['private']['evaluation']['parameters']
    p, public = deepcopy(old), deepcopy(task['public'])
    if module == 'action_order':
        bindings = task['private']['bindings']
        base = bindings['base_candidate_0']['category'].replace('_', ' ')
        middle = bindings['middle']['category'].replace('_', ' ')
        public['objective'] = f'Build a stable stack with the {middle} called B supported by either original {base}.'
        if p['top_keys']:
            top = bindings['top_candidate_0']['category'].replace('_', ' ')
            public['objective'] += f' Place either original {top} on top of B.'
        public['objective'] += ' You may complete the stack anywhere on a stable supporting surface. The chosen base may remain where it starts.'
        public['success_criteria'] = ('The selected original objects must form the requested support order, '
            'with each upper object supported only by its predecessor and the base supported by a stable surface. '
            'Leave the stack released, clear of robot contact, and stable for at least 0.5 seconds. '
            'Success depends on the final structure; there is no minimum lift height, required grasp history, '
            'intermediate placement checkpoint or prescribed action order.')
        public['constraints'] = ['Use guarded public robot controls to arrange the objects.',
            'Preserve all original object sizes.', 'Leave all grippers released and clear of the finished stack.']
        p.update(manipulation_revision='final_state_v1', minimum_lift_m=None,
                 task_revision='wia_stack_final_state_20261030_v1',
                 mechanism_contract={'goal': 'achieve_declared_final_support_order',
                     'preserve': ['original_candidates', 'asset_scales', 'released_stable_support'],
                     'inference_claimed': False})
    elif module == 'physical_dynamics_stacking_stability':
        # Keep the original object list, source support and auxiliary question.
        start = public['objective'].split(' Choose a bottom object')[0]
        question = public['objective'].split('Original auxiliary question: ', 1)
        public['objective'] = (start + ' Arrange the three objects into any upright, stable tower on the starting support surface. '
            'Leave all hands released and withdrawn and the tower stable for at least one second.')
        if len(question) == 2:
            public['objective'] += ' Original auxiliary question: ' + question[1]
        public['success_criteria'] = ('All three original objects are upright and form a supported tower in any stable order. '
            'The bottom touches the starting support surface; each upper touches and rests on its predecessor, with no other support. '
            'All robot hands are detached and withdrawn; the tower stays still for one continuous second. '
            'No minimum lift height, grasp history or release sequence is required. The original answer remains auxiliary QA.')
        public['constraints'] = [c for c in public['constraints'] if not c.startswith('Use real pick-and-place;')]
        p.update(manipulation_revision='final_state_v1', task_revision='wia_tower_final_state_20261030_v1')
        p['mechanism_contract'] = {'goal': 'achieve_released_stable_tower',
            'preserve': ['original_objects', 'asset_scales', 'source_support', 'native_stability']}
    elif module == 'rigid_partial_insertion':
        # Preserve the public containment question and the explicit sequential protocol.
        objective = re.split(r' For every named object,| Grasp the banana with the robot,', public['objective'])[0]
        public['objective'] = (objective + ' Place each named object partially inside its selected container. '
            'Leave it released and withdraw the robot hands at least 2 cm clear.')
        public['success_criteria'] = ('Answer the original containment question for every named object. '
            'For each object, verify partial insertion into its selected container with actual container contact and no interpenetration. '
            'With both hands empty, the working gripper open, no robot support or contact and both hands at least 2 cm clear, maintain this state for at least '
            '0.5 seconds with linear speed at most 0.02 m/s and angular speed at most 0.1 rad/s. '
            'An object resting entirely on top does not count. No minimum lift height or prior grasp is required. '
            'Targets may be verified sequentially and removed after verification to free a shared container. '
            'Finish with both hands empty and the working gripper open. Submit the container label in the answer field and declare DONE.')
        p.update(manipulation_revision='final_state_v1', minimum_lift_m=None,
                 scorer_version='rigid_insertion_final_state_v1')
    elif module == 'wia_dimensional':
        p['scoring'].update(manipulation_revision='final_state_v1', minimum_lift_m=None)
        for before, after in [('Grasp and lift it, ', 'Move it to '),
                              (' Grasp, lift, release and withdraw for every block.', ''),
                              ('Use a real robot grasp to lift each required block at least 5 cm. ', '')]:
            for key in ('objective', 'success_criteria'):
                public[key] = public[key].replace(before, after)
        # Write fluent aperture instructions without inherited first-grasp commitments.
        if p['operation'] == 'select_pass_aperture':
            public['objective'] = ('Choose a block and orientation that fit the entire framed opening. '
                'Move the block from the near side through the opening and out beyond the far side, '
                'then leave it supported on the blue pad. Rotation is allowed.')
            public['constraints'] = [c for c in public['constraints'] if not c.startswith('The first candidate block')]
            p['completion_items'] = [{'id': 'passage', 'kind': 'historical_checkpoint'},
                                     {'id': 'stable_placement', 'kind': 'terminal_condition'}]
        else:
            public['constraints'] = [c.replace(' Complete a genuine grasp, lift and released placement for every block.', '')
                                     for c in public['constraints']]
            p['completion_items'] = [{'id': key, 'kind': 'terminal_condition'} for key in ('assignment', 'stable_placement')]
        public['success_criteria'] += ' No minimum lift height or prior grasp is required.'
    elif module == 'wia_mirror':
        p['scoring'].update(manipulation_revision='final_state_v1', minimum_lift_m=None, minimum_required_motion_m=None)
        public['success_criteria'] = ('Leave all required original items in the branch-correct destination and requested spatial relation, '
            'released without robot support and stable for at least 0.5 seconds, with linear speed at most 0.02 m/s '
            'and angular speed at most 0.1 rad/s. Finish with empty open hands, withdrawn from the objects. '
            'No minimum lift height, travel distance, prior grasp or first-grasp commitment is required. Declare DONE.')
        public['objective'] = public['objective'].replace('Before grasping, submit', 'Submit').replace(
            'Move both items on the cutting board to reverse that order', 'Arrange the items on the cutting board to reverse that order')
        public['constraints'] = [c.replace('Unselected produce must remain within', 'At completion, unselected produce must be within')
            for c in public['constraints'] if not c.startswith('The first candidate grasp')]
        ids = ['selected_original_at_destination', 'stable_placement']
        if p['operation'] == 'classify_appearance_deliver':
            ids.insert(0, 'appearance_report')
        if p['operation'] == 'mirror_reconfigure':
            ids.insert(1, 'reconfiguration')
            p['requires_both_objects_moved'] = False
        p['completion_items'] = [{'id': key, 'kind': 'terminal_condition'} for key in ids]
    elif module == 'wia_spatial_tray':
        p['scoring'].update(manipulation_revision='final_state_v1', minimum_lift_m=None, minimum_displacement_m=None)
        public['success_criteria'] = ('Leave every original target fully inside the usable tray deck, '
            'in the requested relation, with actual nonpenetrating deck support. Leave hands empty, open and clear '
            'and the objects stable for at least 0.5 seconds. No minimum lift height, travel distance or prior grasp is required.')
    task = replace_parameters(task, old, p)
    task['public'] = public
    return task, old, p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    catalogue_path = RELEASE / 'catalogue.json'
    catalogue = json.loads(catalogue_path.read_text())
    plans = []
    for row in catalogue['tasks']:
        task_path, scene_path = ROOT / row['task_file'], ROOT / row['scene_file']
        raw_task, raw_scene = task_path.read_bytes(), scene_path.read_bytes()
        task, scene = yaml.safe_load(raw_task), yaml.safe_load(raw_scene)
        if task['private']['evaluation'].get('module', '').rsplit('.', 1)[-1] not in MODULES:
            continue
        if task.get('repair', {}).get('final_state_revision'):
            continue
        revised, old, new = revise(task)
        revised['version'] += 1
        revised.setdefault('repair', {})['final_state_revision'] = {
            'revision': 'final_state_v1', 'date': '2026-10-30',
            'parent_task_sha256': sha(raw_task), 'parent_scene_sha256': sha(raw_scene),
            'parent_source_task_file': row['source_task_file'],
            'parent_source_scene_file': row['source_scene_file'],
            'scene_geometry_changed': False, 'native_execution_verified': False,
            'scope': 'Outcome-based placement; essential passage and sequential insertion goals remain explicit.'}
        new_scene = replace_parameters(scene, old, new)
        if new_scene != scene:
            new_scene['version'] += 1
        CategoryTaskDefinition.model_validate(revised)
        SceneDefinition.model_validate(new_scene)
        if revised['scene'] != new_scene['scene']:
            raise ValueError('Task/scene initialization mismatch: ' + row['alias'])
        if re.search(r'(?:lift[^.!?]{0,70}5 cm|5 cm lift)', json.dumps(revised['public'])):
            raise ValueError('Stale public lift gate: ' + row['alias'])
        plans.append((row, task_path, scene_path, revised, new_scene, raw_task, raw_scene))
    print(json.dumps({'aliases': [p[0]['alias'] for p in plans], 'count': len(plans), 'apply': args.apply}))
    if not args.apply or not plans:
        return
    if (DERIVED / 'manifest.json').exists():
        raise FileExistsError('Do not overwrite a prior derived revision.')
    records = []
    for row, task_path, scene_path, task, scene, raw_task, raw_scene in plans:
        stem = row['alias'].lower()
        for name, content in [(f'parents/tasks/{stem}.yaml', raw_task), (f'parents/scenes/{stem}.yaml', raw_scene)]:
            path = DERIVED / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        previous = deepcopy(row)
        for kind, value, destination in [('tasks', task, task_path), ('scenes', scene, scene_path)]:
            content = yaml.safe_dump(value, sort_keys=False, allow_unicode=False, width=100).encode()
            path = DERIVED / kind / f'{stem}.yaml'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            destination.write_bytes(content)
            key = 'task' if kind == 'tasks' else 'scene'
            row[f'source_{key}_file'] = str(path.relative_to(ROOT))
            row[f'source_{key}_file_sha256'] = sha(content)
        records.append({'alias': row['alias'], 'previous_selection': previous,
                        'current_selection': deepcopy(row), 'geometry_changed': False})
    (DERIVED / 'manifest.json').write_text(json.dumps({'revision': 'final_state_v1', 'tasks': records,
        'native_execution_verified': False, 'screenshot_refresh_required': True}, indent=2) + '\n')
    catalogue_path.write_text(json.dumps(catalogue, indent=2) + '\n')


if __name__ == '__main__':
    main()
