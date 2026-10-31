"""Check opt-in final stack scoring with native-shaped CPU readbacks.

Run from WIA: PYTHONPATH=src:../Show-Harness-Explore:tests \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B -m pytest \
  -q -p no:cacheprovider tests/harness_integration/test_wia_stack_final_state.py
These fixtures verify scoring predicates, not simulated robot feasibility.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from wia.authoring.converters import action_order as order
from wia.authoring.converters import physical_dynamics_stacking_stability as tower
from harness_integration.test_annotation_manipulation_categories import NativeObject, native_types, session_with
from harness_integration.test_wia_ordered_stack_transfer import StackTrial


def final_order(tmp_path, states, *, top=False):
    return StackTrial(tmp_path, states, top=top, revision='final_state_v1')


@pytest.mark.parametrize('top', [False, True])
def test_action_final_arrangement_needs_no_grasp_lift_or_intermediate_placement(tmp_path, native_types, top):
    trial = final_order(tmp_path, native_types, top=top)
    base_position = trial.objects['base_candidate_1'].position.copy()
    if top:
        # The top reaches B before B reaches its base, without a prior middle checkpoint.
        trial.support('top_candidate_1', [trial.objects['middle']])
        trial.step(3)
    trial.support('middle', [trial.objects['base_candidate_1']])
    trial.step(6)
    result = trial.result()
    assert result['task_success'] is True
    assert result['schema'] == 'private_ordered_stack_final_state_v1'
    assert result['manipulation_revision'] == 'final_state_v1'
    assert result['task_revision'] == 'wia_final_stack_test_v1'
    assert result['selected_base'] == 'base_candidate_1'
    assert result['selected_top'] == ('top_candidate_1' if top else None)
    assert np.array_equal(trial.objects['base_candidate_1'].position, base_position)
    assert not any('grasp' in key or 'lift' in key or 'placement' in key for key in result['checks'])
    assert not {'histories', 'middle_placement', 'events'} & result.keys()
    assert result['completion_items'] == [{'id': 'released_final_stack', 'kind': 'terminal_condition',
        'completed': True, 'evaluation_validity': 'valid'}]


def test_action_final_stack_can_use_native_external_support_anywhere(tmp_path, native_types):
    trial = final_order(tmp_path, native_types)
    remote = NativeObject('remote_table', [8., -4., .4], (2., 2., .8), category='table')
    trial.session.env.scene.objects[remote.name] = remote
    trial.objects['base_candidate_0'].position[:2] = [8., -4.]
    trial.objects['middle'].position[:2] = [8., -4.]
    trial.support('base_candidate_0', [remote])
    trial.support('middle', [trial.objects['base_candidate_0']])
    trial.step(6)
    assert trial.result()['task_success'] is True


@pytest.mark.parametrize('interference', ['held', 'other_hand', 'robot_contact', 'unstable',
    'external_prop', 'other_candidate_prop', 'wrong_support', 'base_unsupported', 'moving_support'])
def test_action_final_mode_keeps_native_release_support_and_stability_guards(tmp_path, native_types, interference):
    trial = final_order(tmp_path, native_types)
    trial.support('middle', [trial.objects['base_candidate_0']])
    if interference == 'held':
        trial.hold('middle')
    elif interference == 'other_hand':
        trial.robot._ag_obj_in_hand['right'] = trial.objects['base_candidate_1']
    elif interference == 'robot_contact':
        trial.objects['middle'].contacts.append(SimpleNamespace(
            body0=trial.objects['middle'].prim_path + '/body', body1=trial.robot.prim_path + '/body'))
    elif interference == 'unstable':
        trial.objects['middle'].get_linear_velocity = lambda: [.03, 0., 0.]
    elif interference in ('external_prop', 'other_candidate_prop'):
        prop = trial.table if interference == 'external_prop' else trial.objects['base_candidate_1']
        trial.support('middle', [trial.objects['base_candidate_0'], prop])
    elif interference == 'wrong_support':
        trial.support('middle', [trial.table])
    elif interference == 'base_unsupported':
        trial.support('base_candidate_0', [])
    else:
        trial.table.get_linear_velocity = lambda: [.03, 0., 0.]
    trial.step(12)
    assert trial.result()['task_success'] is False


def test_action_record_revokes_dwell_without_adding_callback_time(tmp_path, native_types):
    trial = final_order(tmp_path, native_types)
    trial.support('middle', [trial.objects['base_candidate_0']])
    trial.step(5)
    for _ in range(10):
        assert trial.result()['task_success'] is False
    assert trial.evaluator.sample_count == 5
    trial.step()
    assert trial.result()['task_success'] is True
    trial.support('middle', [trial.table])
    assert trial.result()['task_success'] is False
    trial.support('middle', [trial.objects['base_candidate_0']])
    trial.step(5)
    assert trial.result()['task_success'] is False
    trial.step()
    assert trial.result()['task_success'] is True


def test_action_missing_native_measurement_revokes_completed_stack(tmp_path, native_types):
    trial = final_order(tmp_path, native_types)
    trial.support('middle', [trial.objects['base_candidate_0']])
    trial.step(6)
    assert trial.result()['task_success'] is True
    del trial.objects['middle'].states[native_types.OnTop]
    result = trial.result()
    assert result['task_success'] is False
    assert result['evaluation_validity'] == 'unavailable'
    assert result['stable_controller_duration_s'] == 0.


def test_action_final_mode_retains_initial_incomplete_scene_requirement(tmp_path, native_types):
    trial = final_order(tmp_path, native_types)
    trial.support('middle', [trial.objects['base_candidate_0']])
    trial.evaluator.begin_episode(trial.observation)
    trial.step(12)
    result = trial.result()
    assert result['checks']['initial_incomplete_supported_scene'] is False
    assert result['task_success'] is False
    assert result['completion_items'][0]['completed'] is False


@pytest.mark.parametrize('change', ['missing_null', 'non_null', 'unknown_revision'])
def test_action_final_mode_is_explicit_and_validated(tmp_path, native_types, change):
    trial = final_order(tmp_path, native_types)
    config = deepcopy(trial.config)
    if change == 'missing_null':
        del config['parameters']['minimum_lift_m']
    elif change == 'non_null':
        config['parameters']['minimum_lift_m'] = .05
    else:
        config['parameters']['manipulation_revision'] = 'final_state_v2'
    with pytest.raises(ValueError):
        order._validate_operation(trial.session, config)


class TowerTrial:
    def __init__(self, tmp_path, states, monkeypatch, *, final=True):
        self.states, self.now = states, 0.
        self.objects = {key: NativeObject(key, [index * .3, 0., .05])
            for index, key in enumerate(tower.OBJECT_KEYS)}
        self.floor = NativeObject('floor', [0., 0., -.05], (10., 10., .1), category='floors')
        self.robot = NativeObject('robot', [1., 1., .4])
        self.robot._ag_obj_in_hand = {'left': None, 'right': None}
        self.robot._ag_obj_constraints = {'left': None, 'right': None}
        self.robot._ag_release_counter = {'left': None, 'right': None}
        self.supports = {}
        for key in self.objects:
            self.support(key, [self.floor])
        self.session = session_with([self.floor, self.robot, *self.objects.values()], list(self.objects.values()))
        self.session.category_interaction_setup = {}
        self.session.robot = self.robot
        self.controller = SimpleNamespace(debug_gripper_state=self.gripper)
        parameters = {'original_objects': {key: {'name': obj.name, 'category': obj.category,
            'model': obj.model, 'scale': obj.scale.tolist(), 'position': obj.position.tolist(),
            'quaternion_xyzw': obj.quaternion.tolist()} for key, obj in self.objects.items()},
            'legacy_accepted_category_orders': [['prop', 'prop', 'prop']]}
        if final:
            parameters.update(manipulation_revision='final_state_v1', task_revision='wia_final_tower_test_v1')
        self.config = {'type': 'category_operation', 'module': tower.MODULE, 'operation': tower.OPERATION,
            'scene': self.session.scene_name, 'room': self.session.room_name,
            'bindings': {**{key: obj.name for key, obj in self.objects.items()}, 'floor': self.floor.name},
            'parameters': parameters}
        monkeypatch.setattr(tower, '_native_time', lambda: (self.now, .05))
        monkeypatch.setattr(tower, '_robot_start_checks', lambda session: {
            'in_original_room': True, 'has_floor_contact': True, 'no_obstacle_contact': True})
        self.evaluator = tower.create_evaluator(self.session, self.controller, self.config, tmp_path)
        self.observation = SimpleNamespace(snapshot_id='fixture', action_step=0)
        self.evaluator.begin_episode(self.observation)
        assert self.evaluator.preconditions['accepted']

    def gripper(self):
        held = self.robot._ag_obj_in_hand['left']
        return {'object_name': held.name if held else None, 'holding': held is not None,
            'constraint_active': self.robot._ag_obj_constraints['left'] is not None,
            'release_pending': False, 'gripper_open': held is None}

    def support(self, key, supports):
        obj = self.objects[key]
        self.supports[key] = list(supports)
        obj.states = {self.states.OnTop: SimpleNamespace(get_value=lambda other: other in self.supports[key]),
            self.states.Touching: SimpleNamespace(get_value=lambda other: other in self.supports[key])}
        obj.contacts = [SimpleNamespace(body0=obj.prim_path + '/body', body1=other.prim_path + '/top')
            for other in supports]

    def arrange(self):
        bottom, middle, top = tower.OBJECT_KEYS
        self.support(top, [self.objects[middle]])
        self.support(middle, [self.objects[bottom]])

    def step(self, count=1, *, dt=.05):
        for _ in range(count):
            self.now += dt
            self.evaluator.sample_step()

    def result(self):
        return self.evaluator.record(self.observation, action=None)


def test_tower_final_mode_accepts_no_lift_no_grasp_history_and_top_first_order(tmp_path, native_types, monkeypatch):
    trial = TowerTrial(tmp_path, native_types, monkeypatch)
    # Unrelated historical grasps cannot veto a correct final tower.
    trial.robot._ag_obj_in_hand['left'] = NativeObject('unrelated', [2., 0., .1])
    trial.robot._ag_obj_constraints['left'] = object()
    trial.step()
    trial.robot._ag_obj_in_hand['left'] = None
    trial.robot._ag_obj_constraints['left'] = None
    trial.support('object_2', [trial.objects['object_1']])
    trial.step(3)
    trial.support('object_1', [trial.objects['object_0']])
    trial.step(21)
    result = trial.result()
    assert result['task_success'] is True
    assert result['schema'] == 'private_final_stack_v1'
    assert result['manipulation_revision'] == 'final_state_v1'
    assert result['task_revision'] == 'wia_final_tower_test_v1'
    assert result['measured_order'] == list(tower.OBJECT_KEYS)
    assert not any('lift' in key or 'grasp' in key for key in result['criteria'])
    assert 'release_events' not in result
    assert result['completion_items'] == [{'id': 'withdrawn_stable_tower', 'kind': 'terminal_condition',
        'completed': True, 'evaluation_validity': 'valid'}]


def test_legacy_tower_still_requires_real_lift_release_history(tmp_path, native_types, monkeypatch):
    trial = TowerTrial(tmp_path, native_types, monkeypatch, final=False)
    trial.arrange()
    trial.step(25)
    result = trial.result()
    assert result['task_success'] is False
    assert result['schema'] == 'private_real_stack_v1'
    assert result['criteria']['two_upper_real_lift_and_supported_release_events'] is False
    assert 'manipulation_revision' not in result


@pytest.mark.parametrize('interference', ['held', 'other_hand', 'robot_contact', 'external_prop',
    'wrong_order', 'wrong_source_support', 'unstable', 'tilted', 'rescaled'])
def test_tower_final_mode_retains_physical_guards(tmp_path, native_types, monkeypatch, interference):
    trial = TowerTrial(tmp_path, native_types, monkeypatch)
    trial.arrange()
    if interference in ('held', 'other_hand'):
        arm = 'left' if interference == 'held' else 'right'
        trial.robot._ag_obj_in_hand[arm] = trial.objects['object_2']
    elif interference in ('robot_contact', 'external_prop'):
        contact = trial.robot if interference == 'robot_contact' else trial.floor
        trial.objects['object_1'].contacts.append(SimpleNamespace(
            body0=trial.objects['object_1'].prim_path + '/body', body1=contact.prim_path + '/body'))
    elif interference == 'wrong_order':
        trial.support('object_2', [trial.objects['object_0']])
    elif interference == 'wrong_source_support':
        trial.support('object_0', [NativeObject('unrelated_table', [0., 0., .1])])
    elif interference == 'unstable':
        trial.objects['object_1'].get_linear_velocity = lambda: [.03, 0., 0.]
    elif interference == 'tilted':
        trial.objects['object_1'].quaternion = np.array([.70710678, 0., 0., .70710678])
    else:
        trial.objects['object_1'].scale *= 1.1
    trial.step(25)
    assert trial.result()['task_success'] is False


def test_tower_final_record_does_not_advance_native_dwell_and_revokes_interruption(tmp_path, native_types, monkeypatch):
    trial = TowerTrial(tmp_path, native_types, monkeypatch)
    trial.arrange()
    for _ in range(25):
        trial.now += .05
        assert trial.result()['task_success'] is False
    assert trial.evaluator.trace == []
    trial.step(21)
    assert trial.result()['task_success'] is True
    trial.support('object_2', [trial.floor])
    assert trial.result()['task_success'] is False
    trial.support('object_2', [trial.objects['object_1']])
    trial.step(20)
    assert trial.result()['task_success'] is False
    trial.step()
    assert trial.result()['task_success'] is True


@pytest.mark.parametrize('interruption', ['measurement', 'sample_gap'])
def test_tower_final_native_measurement_or_sample_gap_restarts_window(tmp_path, native_types, monkeypatch, interruption):
    trial = TowerTrial(tmp_path, native_types, monkeypatch)
    trial.arrange()
    trial.step(21)
    assert trial.result()['task_success'] is True
    if interruption == 'measurement':
        del trial.objects['object_1'].states[native_types.OnTop]
        result = trial.result()
        assert result['evaluation_validity'] == 'unavailable'
        trial.support('object_1', [trial.objects['object_0']])
    else:
        trial.step(dt=.15)
        result = trial.result()
    assert result['task_success'] is False
    assert result['stability_duration_s'] == 0.
    trial.step(21)
    assert trial.result()['task_success'] is True


def test_tower_rejects_unknown_explicit_revision(tmp_path, native_types, monkeypatch):
    trial = TowerTrial(tmp_path, native_types, monkeypatch)
    config = deepcopy(trial.config)
    config['parameters']['manipulation_revision'] = 'final_state_v2'
    with pytest.raises(ValueError):
        tower._native_config(trial.session, config)


def test_tower_final_mode_retains_initial_identity_scale_preconditions(tmp_path, native_types, monkeypatch):
    trial = TowerTrial(tmp_path, native_types, monkeypatch)
    trial.objects['object_0'].scale *= 1.1
    trial.evaluator.begin_episode(trial.observation)
    assert trial.evaluator.preconditions['accepted'] is False
    trial.objects['object_0'].scale /= 1.1
    trial.arrange()
    trial.step(25)
    result = trial.result()
    assert result['task_success'] is False
    assert result['completion_items'][0]['completed'] is False
