"""CPU checks for pending shared worlds and exact reset-bound native receipts.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests/test_task_edit2_mapping_world.py
No simulator, service, native truth inference, or GPU is used.
"""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import yaml

from wia.authoring.converters import task_edit2_mapping_world as world
import wia.native.esi_category_evaluation as runtime
from wia.tasks.answer_sheet import mapping_questions
from wia.runtime.scene_capture_queue import read_json
from wia.runtime.commands.build_task_edit2_mapping_candidates import (
    CORE, GROUPS, budget_proposal, build_candidates, manhattan_mst, select_groups, source_points)


def member(index, *, duplicate=False):
    return {'id': f'native_{index}', 'public_id': f'goal_{index}',
        'source_identity': f'private_source_{index}', 'source_sha256': f'sha_{index}',
        'small_task': 'Connectivity', 'bindings': {},
        'public': {'objective': 'Inspect the regions.'},
        'evaluation': {'type': 'category_operation',
            'module': 'wia.authoring.converters.cognitive_mapping_connectivity',
            'operation': 'answer_direct_adjacency_with_exploration', 'binding_keys': [],
            'parameters': {'task_revision': 'task_edit_20261018_v1',
                'source_path': f'private_source_{index}', 'required_regions': ['room_0'],
                'required_order': [], 'arrival_report_required': True, 'stability_steps': 3,
                'mapping_composition_contract': None,
                'source_qa': {'scored': True, 'question': 'Same?' if duplicate else f'Question {index}',
                              'answer': 'Yes', 'options': ['Yes', 'No']}}}}


@pytest.fixture
def native():
    members = [member(1), member(2)]
    contract = {'revision': world.REVISION, 'scene': 'synthetic', 'room': 'full_scene', 'seed': 0,
        'source_members': [{'source_identity': m['source_identity'], 'source_sha256': m['source_sha256']}
                           for m in members], 'episode_budget': {'max_steps': 101, 'wall_timeout_s': 303}}
    config = {'parameters': {'mapping_joint_world_contract': contract, 'members': members,
                            'answer_sheet': True, 'semantic_questions': mapping_questions(members)}}
    reset = {'episode_id': 'episode_1', 'reset_id': 'reset_1', 'seed': 0}
    measured = {name: {field: 'measured_' + field for field in fields}
                for name, fields in world.MEASUREMENTS.items()}
    ids = [f'camera_{i}' for i in range(8)]
    measured['mounted_cameras'].update(camera_ids=ids,
        rgb_identities={key: 'rgb_' + key for key in ids}, snapshot_id='initial_snapshot')
    measured['initial_overview'].update(reset_id=reset['reset_id'], snapshot_id='initial_snapshot')
    live = {**reset, 'scene': 'synthetic', 'measurements': measured}
    proof = {**deepcopy(live), 'contract_sha256': world.digest(contract),
        'source_members': deepcopy(contract['source_members']),
        'gates': {gate: True for gate in world.GATES},
        'source_relations': [{**source, 'verified': True, 'native_evidence': {'trial': 'synthetic_only'}}
                             for source in contract['source_members']],
        'budget_calibration': {'verified': True, 'budget_sha256': world.digest(contract['episode_budget']),
                               'native_trial_evidence': {'synthetic_cpu_test': True}}}
    session = NS(initial_overview_identity=reset, mapping_joint_world_identity=lambda contract: live,
        mapping_joint_world_qualification=proof, scene_name='synthetic', room_name='full_scene',
        env=NS(scene=NS(object_registry=lambda *args: None)))
    return NS(session=session, config=config, proof=proof, live=live,
        controller=NS(physical=True), observation=NS(snapshot_id=4, action_step=0))


@pytest.mark.parametrize('mutation', [
    lambda n: setattr(n.session, 'mapping_joint_world_qualification', None),
    lambda n: setattr(n.session, 'mapping_joint_world_identity', None),
    lambda n: n.proof.update(reset_id='other_reset'),
    lambda n: n.proof.update(seed=7),
    lambda n: n.proof.update(source_members=n.proof['source_members'][:-1]),
    lambda n: n.proof['source_members'].append(deepcopy(n.proof['source_members'][0])),
    lambda n: n.proof['source_relations'][0].update(verified=False),
    lambda n: n.live['measurements']['doors'].update(native_door_states={'changed': 'current'}),
    lambda n: n.proof['budget_calibration'].update(verified=False),
    lambda n: n.config['parameters']['members'].pop(),
    lambda n: n.live['measurements']['initial_overview'].update(snapshot_id='different_boundary'),
])
def test_missing_stale_or_incomplete_native_proof_unavailable(native, mutation):
    mutation(native)
    row = world.joint_world_qualification(native.session, native.controller, native.config)
    assert row['accepted'] is False and row['evaluation_validity'] == 'unavailable'
    assert row['reason']


def test_valid_receipt_delegates_all_original_qualifiers(native, monkeypatch):
    calls = []
    def qualify(session, controller, config, objects):
        calls.append(deepcopy(config))
        return {'accepted': True}
    monkeypatch.setattr(runtime, 'qualify_category_operation', qualify)
    assert world.qualify_operation(native.session, native.controller, native.config, {})['accepted']
    assert len(calls) == 2
    assert [c['parameters'] for c in calls] == [m['evaluation']['parameters'] for m in native.config['parameters']['members']]
    monkeypatch.setattr(runtime, 'qualify_category_operation', lambda *args: {'accepted': False})
    assert not world.qualify_operation(native.session, native.controller, native.config, {})['accepted']


def test_zero_native_doors_requires_explicit_measured_zero_and_retention_evidence(native):
    doors = native.live['measurements']['doors']
    doors.update(native_door_states=[], native_door_count=0)
    native.proof['measurements'] = deepcopy(native.live['measurements'])
    assert world.joint_world_qualification(native.session, native.controller, native.config)['accepted']
    doors.pop('native_door_count')
    native.proof['measurements'] = deepcopy(native.live['measurements'])
    assert not world.joint_world_qualification(native.session, native.controller, native.config)['accepted']


@pytest.mark.parametrize('alias', ['t05', 't06'])
def test_fixed_release_joint_world_and_every_declared_member_use_definition_policy(alias):
    from wia.resources import ROOT
    from wia.tasks.loader import load_task
    from wia.tasks.release_policy import fixed_release_evaluation
    from wia.authoring.converters.cognitive_mapping_visit_reports import mapping_qualification
    task = load_task(ROOT / f'explore_to_achieve/wia_v1/release/tasks/{alias}.yaml', variant='adapted')
    session = NS(loaded_task=task)
    config = task.definition.private.evaluation.model_dump()
    config.update(scene=task.scene_name, room=task.room_name, bindings={})
    result = world.joint_world_qualification(session, None, config)
    assert result['accepted'] and result['policy'] == 'fixed_v1_definition'
    assert result['physical_checks_run'] is False
    assert 'native_receipt_sha256' not in result
    for member in config['parameters']['members']:
        child = deepcopy(member['evaluation'])
        child.update(scene=task.scene_name, room=task.room_name, bindings={})
        child['parameters']['mapping_composition_contract'] = {'compatibility_verified': True}
        proof = mapping_qualification(session, child)
        assert proof['accepted'] and proof['policy'] == 'fixed_v1_definition'
        assert not proof['physical_checks_run']
        modified = deepcopy(child)
        modified['parameters']['source_qa']['answer'] = 'tampered'
        assert not fixed_release_evaluation(session, modified)
    changed = deepcopy(config)
    changed['parameters']['members'].pop()
    assert not world.joint_world_qualification(session, None, changed)['accepted']


class Child:
    def __init__(self, controller, config):
        self.controller, self.config, self.answer, self.calls = controller, config, None, []

    def begin_episode(self, observation):
        self.answer = None
        return self.record(observation)

    def report_answer(self, answer, observation):
        self.answer = answer
        self.calls.append((answer, observation.snapshot_id, observation.action_step))

    def record(self, observation, *, action=None):
        answer = self.answer == self.config['parameters']['source_qa']['answer']
        return {'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'task_success': answer and self.controller.physical,
            'evaluation_validity': 'complete', 'completion_items': [
                {'id': 'room:room_0', 'kind': 'historical_checkpoint',
                 'completed': self.controller.physical, 'evaluation_validity': 'valid'},
                {'id': 'source_relation_answer', 'kind': 'answer_requirement',
                 'completed': answer, 'evaluation_validity': 'valid'}]}


def test_real_release_joint_reducer_scores_answers_without_inventing_visits(tmp_path, monkeypatch):
    from wia.resources import ROOT
    from wia.tasks.loader import load_task
    task = load_task(ROOT/'explore_to_achieve/wia_v1/release/tasks/t05.yaml', variant='adapted')
    config = task.definition.private.evaluation.model_dump()
    config.update(scene=task.scene_name, room=task.room_name, bindings={})
    session = NS(loaded_task=task, scene_name=task.scene_name, room_name=task.room_name,
                 env=NS(scene=NS(object_registry=lambda *args: None)))
    controller = NS(physical=False)
    monkeypatch.setattr(runtime, 'create_category_operation_evaluator',
                        lambda session, controller, config, output: Child(controller, config))
    evaluator = world.create_evaluator(session, controller, config, tmp_path)
    observation = NS(snapshot_id=1, action_step=0)
    initial = evaluator.begin_episode(observation)
    assert initial['evaluation_validity'] == 'complete' and initial['score_percent'] == 0
    import json
    answers = {m['public_id']:m['evaluation']['parameters']['source_qa']['answer']
               for m in config['parameters']['members']}
    evaluator.report_answer(json.dumps(answers), observation)
    row = evaluator.record(observation)
    assert row['evaluation_validity'] == 'complete' and row['score_percent'] == 100
    assert not row['task_success']  # Correct answers do not fabricate native visits.
    assert row['mapping_joint_world_qualification']['physical_checks_run'] is False


def test_one_boundary_sheet_shared_visit_and_live_proof_revocation(native, monkeypatch, tmp_path):
    children = []
    def factory(session, controller, config, output):
        child = Child(controller, config)
        children.append(child)
        return child
    monkeypatch.setattr(runtime, 'create_category_operation_evaluator', factory)
    original = deepcopy(native.config)
    evaluator = world.create_evaluator(native.session, native.controller, native.config, tmp_path)
    evaluator.begin_episode(native.observation)
    evaluator.report_answer('{"goal_1":"Yes","goal_2":"Yes"}', native.observation)
    result = evaluator.record(native.observation)
    assert result['task_success'] and result['answer_accuracy_percent'] == 100
    rooms = [item for item in result['completion_items'] if ':room:' in item['id']]
    assert len(rooms) == 1 and len(rooms[0]['sources']) == 2
    assert all(c.calls == [('Yes', 4, 0)] for c in children)
    assert native.config == original
    native.proof['reset_id'] = 'stale'
    unavailable = evaluator.record(native.observation)
    assert not unavailable['task_success'] and unavailable['score_percent'] is None
    assert unavailable['required_unique_questions'] == 2
    assert unavailable['per_subclass']['Connectivity']['answer_evaluation_validity'] == 'unavailable'
    assert all(q['evaluation_validity'] == 'unavailable' for q in unavailable['semantic_questions'])


def test_semantic_duplicates_losslessly_keep_both_sources():
    members = [member(1, duplicate=True), member(2, duplicate=True)]
    frozen = deepcopy(members)
    questions = mapping_questions(members)
    assert len(questions) == 1
    assert len(questions[0]['sources']) == 2
    assert questions[0]['equivalence'] == 'exact_native_semantic_contract'
    assert members == frozen


def test_full_building_hook_does_not_touch_objects_doors_or_robot():
    config = {'scene': {'scene_model': 'synthetic', 'load_room_instances': ['room_0'],
                       'load_object_categories': ['chair'], 'not_load_object_categories': ['door']},
              'objects': [{'name': 'native_door'}], 'robots': [{'position': [1, 2, 3]}]}
    original = deepcopy(config)
    world.configure_scene(config, {'mapping_joint_world_contract': {'revision': world.REVISION, 'room': 'full_scene'}})
    assert config['scene'] == {'scene_model': 'synthetic'}
    assert config['objects'] == original['objects'] and config['robots'] == original['robots']


def test_budget_is_new_geometric_proposal_and_never_child_sum():
    inputs = {'required_distinct_rooms': ['room_a', 'room_b'], 'required_unique_questions': 3,
        'all_source_points': [{'xy': [0, 0]}, {'xy': [1, 0]}, {'xy': [1, 2]}],
        'missing_camera_coverage': [], 'invalid_point_pointers': []}
    assert manhattan_mst([[0, 0], [1, 0], [1, 2]]) == 3
    proposal = budget_proposal(inputs)
    assert proposal['max_steps'] == 24 + 24 + 12
    assert proposal['wall_timeout_s'] == 180 and proposal['calibrated'] is False
    inputs['missing_camera_coverage'] = [{'missing': 'anchor'}]
    assert budget_proposal(inputs)['requires_budget_file']
    with pytest.raises(ValueError):
        budget_proposal(inputs, {'max_steps': 600, 'wall_timeout_s': 1800})
    assert budget_proposal(inputs, {'max_steps': 100, 'wall_timeout_s': 300,
        'derivation': 'Explicit new proposal', 'evidence': {'measurement': 'pending'}})['max_steps'] == 100


def test_saved_points_are_explicit_and_preserve_json_pointer():
    points, errors = source_points({'metadata_json': '{"room_views":[{"room_instance":"room_0",'
        '"camera_pose":{"position":[1,2,3]}}],"path_world_xy":[[4,5],[6,7]]}'}, 'source')
    assert not errors and len(points) == 3
    assert points[0]['context_labels'] == ['room_0']
    assert points[0]['json_pointer'] == '/metadata_json/room_views/0/camera_pose/position'
    _, errors = source_points({'camera_pose': {'position': ['unknown', 2]}}, 'source')
    assert errors == ['/camera_pose/position']


def test_exact_current_five_plus_five_no_route_pooling_and_definitions(tmp_path):
    rows = read_json(CORE)['tasks']
    selected = select_groups(rows)
    assert set(selected) == set(GROUPS)
    assert all(len(group) == 5 for group in selected.values())
    assert all(row['subcategory'] not in ('Long-Term Navigation', 'Traversable Passage')
               for group in selected.values() for row in group)
    manifest = build_candidates(tmp_path)
    assert manifest['source_count'] == 10 and manifest['task_count'] == 2
    for record in manifest['records']:
        assert record['status'] == 'pending_diagnostic' and record['execution_certificate'] is None
        task = yaml.safe_load(open(record['task_file']))
        scene = yaml.safe_load(open(record['scene_file']))
        assert task['status'] == scene['status'] == 'pending'
        p = task['private']['evaluation']['parameters']
        assert len(p['members']) == 5 and p['mapping_joint_world_contract'] == record['world_contract']
        assert not record['budget_inputs']['missing_camera_coverage']
        assert record['budget_inputs']['child_budgets_used'] is False
        assert task['defaults']['max_steps'] == record['world_contract']['episode_budget']['max_steps']
        assert scene['source']['json'] == min(row['source'] for row in selected[record['group_key']])
        poses = [patch for patch in scene['scene']['variants']['adapted']['patches']
                 if patch['operation'] == 'robot_pose']
        assert len(poses) == (1 if record['group_key'].startswith('connectivity') else 0)
        for child, provenance in zip(p['members'], record['sources']):
            source_task = yaml.safe_load(open(provenance['task_file']))
            assert child['evaluation'] == source_task['private']['evaluation']
            assert child['bindings'] == source_task['private']['bindings']
            assert child['source_task_definition'] == source_task
            assert child['source_sha256'] == provenance['source_sha256']
        assert task['public']['success_criteria'].count('ANSWER: {') == 1
        assert not any(patch['operation'] in ('object_pose', 'object_scale', 'add_primitive')
                       for patch in scene['scene']['variants']['adapted']['patches'])
    with pytest.raises(ValueError):
        select_groups(rows + [deepcopy(selected['connectivity_school_biology'][0])])
