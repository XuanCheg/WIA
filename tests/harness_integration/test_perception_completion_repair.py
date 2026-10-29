"""Replay native-shaped perception events; run: python -m pytest -q tests/test_perception_completion_repair.py.

These deterministic CPU traces verify scoring, not native reachability or vision.
"""
import copy
from types import SimpleNamespace as NS

import pytest

from wia.authoring.converters import enumerative_perception_category_ambiguity as ambiguity
from wia.authoring.converters import enumerative_perception_illumination_variability as illumination
from wia.authoring.converters import enumerative_perception_merged_observation as merged
from wia.authoring.converters import enumerative_perception_structural_enclosure as enclosure
from wia.authoring.converters import enumerative_perception_spatial_segmentation as segmentation
from harness_integration.test_annotation_reflection_temporal import retrieval_world, patrol


def row(name):
    return dict(name=name, position=[0., 0., .1], quaternion_xyzw=[0., 0., 0., 1.],
                aabb_min=[0., 0., 0.], fixed_base=False, visual_only=False,
                touching_task_support=True, on_top_task_support=True,
                linear_speed_m_s=0., angular_speed_rad_s=0.)


def ambiguity_replay():
    evaluator = ambiguity.InventoryEvaluator.__new__(ambiguity.InventoryEvaluator)
    evaluator.targets = {f'target_{i}' for i in range(5)}
    evaluator.confusers = set()
    evaluator.objects = {key: NS(name=key) for key in evaluator.targets}
    evaluator.initial = {key: row(key) for key in evaluator.targets}
    rows = copy.deepcopy(evaluator.initial)
    gripper = dict(object_name=None, holding=False, constraint_active=False, gripper_open=True)
    evaluator.measure = lambda: (rows, gripper)
    evaluator.config = {'operation': 'inventory_return'}
    evaluator.initially_empty = evaluator.initial_layout_valid = evaluator.confusers_undisturbed = True
    evaluator.wrong_object_grasped = False
    evaluator.previous_held = evaluator.pending_release = evaluator.stability_binding = None
    evaluator.lifted, evaluator.released = set(), set()
    evaluator.stable_steps = 0
    evaluator.return_stability = dict.fromkeys(evaluator.targets, 0)
    return evaluator, rows, gripper


def lift_return(evaluator, rows, gripper, key):
    gripper.update(object_name=key, holding=True, constraint_active=True, gripper_open=False)
    rows[key]['aabb_min'][2] = .06
    evaluator._update(count_step=True)
    rows[key]['aabb_min'][2] = 0.
    gripper.update(object_name=None, holding=False, constraint_active=False, gripper_open=True)
    for _ in range(ambiguity.STABILITY_STEPS):
        result = evaluator._update(count_step=True)
    return result


def test_five_inventory_obligations_four_returns_score_eighty_and_duplicate_cannot_score():
    evaluator, rows, gripper = ambiguity_replay()
    for key in sorted(evaluator.targets)[:4]:
        result = lift_return(evaluator, rows, gripper, key)
    assert result['completed_goals'] == 4 and result['total_goals'] == 5
    assert result['score_percent'] == 80. and not result['task_success']
    result = lift_return(evaluator, rows, gripper, 'target_0')
    assert result['completed_goals'] == 4
    rows['target_1']['linear_speed_m_s'] = .03
    assert evaluator._update()['completed_goals'] == 3


def test_terminal_release_stability_cannot_be_accumulated_by_record_reads():
    evaluator, rows, gripper = ambiguity_replay()
    gripper.update(object_name='target_0', holding=True, constraint_active=True)
    rows['target_0']['aabb_min'][2] = .06
    evaluator._update(count_step=True)
    gripper.update(object_name=None, holding=False, constraint_active=False, gripper_open=True)
    rows['target_0']['aabb_min'][2] = 0.
    for _ in range(100):
        result = evaluator._update()
    assert result['completed_goals'] == 0


def test_all_inventory_completion_does_not_override_confuser_safety():
    evaluator, rows, gripper = ambiguity_replay()
    for key in sorted(evaluator.targets):
        result = lift_return(evaluator, rows, gripper, key)
    assert result['task_success'] and result['score_percent'] == 100.
    evaluator.confusers_undisturbed = False
    assert not evaluator._update()['task_success']


def test_agent_visits_retrieval_and_return_are_separate_geometric_obligations(tmp_path):
    result = patrol(retrieval_world(tmp_path / 'without'), retrieve=False)
    assert result['completed_goals'] == 1 and result['total_goals'] == 3
    assert result['visual_discovery_verified'] is False
    result = patrol(retrieval_world(tmp_path / 'with'), retrieve=True)
    assert result['completed_goals'] == result['total_goals'] == 3
    assert result['task_success']


def test_unavailable_measurement_retains_all_required_approaches(tmp_path, monkeypatch):
    evaluator = segmentation.SegmentedPatrolEvaluator.__new__(segmentation.SegmentedPatrolEvaluator)
    evaluator.active = True
    evaluator.config = {'operation': segmentation.OPERATIONS[1]}
    evaluator.session = evaluator.parameters = evaluator.objects = None
    evaluator.faults = set()
    evaluator.visits = {'a': [], 'b': []}
    evaluator.travel = evaluator.excursion = evaluator.samples = evaluator.return_duration = evaluator.episode = 0
    evaluator.output = tmp_path
    def unavailable(*args):
        raise RuntimeError('contacts unavailable')
    monkeypatch.setattr(segmentation, '_native_state', unavailable)
    result = evaluator.record(NS(snapshot_id=0, action_step=0), action=None)
    assert result['evaluation_validity'] == 'unavailable'
    assert result['total_goals'] == 5 and result['completion_score'] is None
    assert all(item['evaluation_validity'] == 'unavailable' for item in result['completion_items'])


def test_enclosure_oscillation_cannot_replace_one_real_open_close_cycle():
    progress = enclosure._joint_cycle_progress()
    joint = dict(position=0., open_direction=1., closed_tolerance=.01, minimum_travel=.5, hand_contact=True)
    for position in [.2, 0., .2, 0., .2, 0.]:
        joint['position'] = position
        enclosure._advance_joint_cycle(progress, joint, {'position': 0.}, enclosure_open=position > .1, count_step=True)
    assert not progress['completed']
    for position in [.6, 0.]:
        joint['position'] = position
        enclosure._advance_joint_cycle(progress, joint, {'position': 0.}, enclosure_open=position > .1, count_step=True)
    assert progress['completed']


def illumination_replay(monkeypatch):
    evaluator = illumination.IlluminationInventoryEvaluator.__new__(illumination.IlluminationInventoryEvaluator)
    evaluator.targets = {'a', 'b'}
    evaluator.initial = {key: row(key) for key in evaluator.targets}
    rows = copy.deepcopy(evaluator.initial)
    gripper = dict(object_name=None, holding=False, constraint_active=False, gripper_open=True)
    lighting = {'measured_phase': illumination.PHASES[0]}
    evaluator.measure = lambda: (rows, gripper, lighting)
    evaluator.config = {'operation': 'two_phase_inventory_return'}
    evaluator.parameters = {}
    evaluator.objects = evaluator.session = None
    evaluator.initially_empty = evaluator.initial_valid = evaluator.history_valid = evaluator.can_acquire = True
    evaluator.wrong_grasp = False
    evaluator.previous_held = evaluator.event = evaluator.pending_release = evaluator.phase = None
    evaluator.phase_samples = evaluator.stable_steps = 0
    evaluator.completed, evaluator.observed_phases = set(), set()
    monkeypatch.setattr(illumination, '_lamp_evidence', lambda *args: {})
    return evaluator, rows, gripper, lighting


def test_illumination_grasp_spanning_phases_does_not_duplicate_event(monkeypatch):
    evaluator, rows, gripper, lighting = illumination_replay(monkeypatch)
    for _ in range(2):
        evaluator._update(count_step=True)
    gripper.update(object_name='a', holding=True, constraint_active=True)
    rows['a']['aabb_min'][2] = .06
    evaluator._update(count_step=True)
    lighting['measured_phase'] = illumination.PHASES[1]
    for _ in range(2):
        evaluator._update(count_step=True)
    rows['a']['aabb_min'][2] = 0.
    gripper.update(object_name=None, holding=False, constraint_active=False)
    result = evaluator._update(count_step=True)
    assert result['completed_goals'] == 0 and result['total_goals'] == 4
    gripper.update(object_name='a', holding=True, constraint_active=True)
    rows['a']['aabb_min'][2] = .06
    evaluator._update(count_step=True)
    rows['a']['aabb_min'][2] = 0.
    gripper.update(object_name=None, holding=False, constraint_active=False)
    result = evaluator._update(count_step=True)
    assert result['completed_goals'] == 1 and result['score_percent'] == 25.


def test_merged_retrieve_return_stays_one_indivisible_obligation():
    evaluator = merged.ClusterEvaluator.__new__(merged.ClusterEvaluator)
    evaluator.config = {'operation': 'retrieve_return'}
    evaluator.objects = {key: NS(name=key) for key in ['a', 'b']}
    evaluator.name_to_key = {key: key for key in evaluator.objects}
    evaluator.initial = {key: row(key) for key in evaluator.objects}
    rows = copy.deepcopy(evaluator.initial)
    support = dict(all_grounded=True, grounded_bindings=['a', 'b'], task_support_roots=['floor'],
                   target_support_edges={'a': [], 'b': []})
    evaluator.initial_support = copy.deepcopy(support)
    evaluator.initial_qualification = {'accepted': True}
    gripper = dict(object_name=None, holding=False, constraint_active=False, gripper_open=True, release_pending=False)
    evaluator.measure = lambda: (rows, support, gripper)
    evaluator.lifted, evaluator.returned = set(), set()
    evaluator.active_target = None
    evaluator.wrong_object_grasped = False
    evaluator.stable_steps = 0
    assert evaluator._update()['total_goals'] == 1
    gripper.update(object_name='a', holding=True, constraint_active=True, gripper_open=False)
    rows['a']['aabb_min'][2] = .06
    evaluator._update(count_step=True)
    rows['a']['aabb_min'][2] = 0.
    gripper.update(object_name=None, holding=False, constraint_active=False, gripper_open=True)
    for _ in range(merged.STABILITY_STEPS):
        result = evaluator._update(count_step=True)
    assert result['task_success'] and result['completed_goals'] == result['total_goals'] == 1
    assert evaluator.returned == {'a'}


def test_single_illumination_recipe_cannot_mix_phase_returns_from_different_targets(monkeypatch):
    evaluator, rows, gripper, lighting = illumination_replay(monkeypatch)
    evaluator.config['operation'] = 'two_phase_single_return'
    evaluator.completed = {('a', illumination.PHASES[0]), ('b', illumination.PHASES[1])}
    result = evaluator._update(count_step=True)
    assert result['total_goals'] == 2 and result['completed_goals'] == 1
    assert not result['task_success']
