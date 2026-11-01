"""Add one source-preserving already-contained control; leave T15--T17 unchanged.

Run: PYTHONPATH=src python scripts/build_inside_control_20261031.py
The control requires a public inside declaration and active DONE in the same episode.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

from wia.resources import ROOT, ESI_ROOT
from wia.authoring.conversion import SourceInstance, ConversionContext
from wia.authoring.converters.perceptual_grounding_material_transparency import CategoryConverter
from wia.tasks.interaction_setup import prepare_definitions
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition


def require_explicit_confirmation(task):
    """Apply the user-requested T52-only response gate without changing its world."""
    task['version'] = 3
    task['private']['evaluation']['parameters']['require_inside_declaration_and_done'] = True
    task['public']['objective'] = (
        'Verify that the whiskey stone is inside its original hingeless jar. If it is outside, '
        'place it inside and release it stably. Return "inside" in the answer field, then '
        'issue DONE. The declaration may precede DONE in this same episode.')
    task['public']['success_criteria'] = (
        'At DONE, the original small object must be released, contact its original container, '
        'satisfy the native inside relation and remain stable for 15 consecutive physics steps. '
        'Credit also requires your explicit answer="inside" declaration and your active DONE command. '
        'An already-contained state requires no manipulation, but still requires both declarations.')
    task['public']['constraints'].append(
        'A correct physical state alone earns no credit. Automatic termination or budget exhaustion '
        'does not replace your active DONE command.')
    task['repair'].update(revision='explicit_inside_confirmation_20261101',
        response_policy='T52 only: explicit inside declaration and active DONE; separate decisions are allowed.')


def main():
    release = ROOT / 'explore_to_achieve/wia_v1/release'
    evidence = ROOT / 'explore_to_achieve/wia_v1/review_followup_20261031'
    evidence.mkdir(parents=True, exist_ok=True)
    preserved = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for name in ('t15', 't16', 't17') for p in
                 (release / 'tasks' / (name + '.yaml'), release / 'scenes' / (name + '.yaml'))}
    identity = ('Perceptual Grounding/Material Transparency/office_large/meeting_room_1/'
                'office_large__meeting_room_1.json')
    source = SourceInstance(ESI_ROOT / 'dataset/json_clean' / identity, identity,
                            'Perceptual Grounding', 'Material Transparency')
    assert source.raw['answer'] == 'yes'
    result = CategoryConverter().convert(source, ConversionContext(evidence, seed=0))
    task, scene = result['task_definition'], result['scene_definition']
    policy = deepcopy(result['interaction_policy'])
    policy.update(relocation_policy='preserve', allow_fallback_table=False,
                  support_search='bounded_group', source_camera_policy='translate_with_group')
    prepare_definitions(source, scene, task, policy)
    # Measured room geometry provides a free corner beyond the default nearest
    # 32 spawn proposals. Keep the original object/container poses unchanged.
    start = {'operation': 'robot_pose', 'position': [-12.6, 2.1, 0.01],
             'yaw_deg': -45.6, 'head_pitch_deg': 45.}
    for definition in (task, scene):
        definition['scene']['variants']['adapted']['patches'].insert(0, deepcopy(start))
    task['repair'] = {'document_alias': 'T52', 'revision': 'already_contained_control_20261031',
        'source_initial_answer': 'yes', 'source_geometry_preserved': True,
        'native_verification_at_definition_time': 'pending',
        'robot_start': 'A measured room corner; object and container source geometry remains unchanged.',
        'scope': 'An additional already-contained branch; T15, T16 and T17 are unchanged.'}
    require_explicit_confirmation(task)
    task_definition(task)
    SceneDefinition.model_validate(scene)
    for folder, value in [('tasks', task), ('scenes', scene)]:
        (release / folder / 't52.yaml').write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=100))
    path = release / 'catalogue.json'
    catalogue = json.loads(path.read_text())
    if not any(row['alias'] == 'T52' for row in catalogue['tasks']):
        catalogue['tasks'].append({'alias': 'T52', 'task_id': task['id'], 'variant_id': 'already_contained_control_20261031',
            'task_file': str((release / 'tasks/t52.yaml').relative_to(ROOT)),
            'scene_file': str((release / 'scenes/t52.yaml').relative_to(ROOT)), 'variant': 'adapted',
            'family': 'GG', 'subclass': 'GG-4', 'origin': 'original',
            'title': 'Verify a whiskey stone already inside its original jar'})
        catalogue['active_family_counts']['GG'] += 1
        for subclass in catalogue['taxonomy']['subclasses']:
            if subclass['id'] == 'GG-4':
                subclass['coverage'].append('T52')
        catalogue['selection_revision'] = 'review_followup_inside_control_20261031'
        path.write_text(json.dumps(catalogue, indent=2, ensure_ascii=False) + '\n')
    else:
        row = next(row for row in catalogue['tasks'] if row['alias'] == 'T52')
        row.update(task_id=task['id'], title='Verify a whiskey stone already inside its original jar')
        path.write_text(json.dumps(catalogue, indent=2, ensure_ascii=False) + '\n')
    assert len(catalogue['tasks']) == 60
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha for name, sha in preserved.items())
    (evidence / 'inside_control_definition.json').write_text(json.dumps({
        'alias': 'T52', 'source_identity': identity, 'source_sha256': hashlib.sha256(source.path.read_bytes()).hexdigest(),
        'preserved_definitions': preserved, 'source_answer_is_not_native_verification': True,
        'source_assets_scales_poses_preserved': True,
        'response_requirement': 'T52-only explicit inside declaration and active DONE in the same episode'}, indent=2) + '\n')
    print('Added T52; active catalogue contains 60 instances. Native declaration/DONE validation is pending.')


if __name__ == '__main__':
    main()
