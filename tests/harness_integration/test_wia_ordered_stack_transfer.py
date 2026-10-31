"""Exercise native-shaped ordered support transfer without a simulator or GPU.

Run: /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B -m pytest \
  -q -p no:cacheprovider tests/test_wia_ordered_stack_transfer.py \
  tests/test_annotation_manipulation_categories.py
The readback fixtures test evaluator predicates, not physical robot feasibility.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from wia.authoring.converters import action_order as order
from harness_integration.test_annotation_manipulation_categories import NativeObject, native_types, session_with


class StackTrial:
    def __init__(self, tmp_path, states, *, top=False, transfer=True, revision=None):
        self.states = states
        self.keys = ['middle', 'base_candidate_0', 'base_candidate_1']
        if top:
            self.keys += ['top_candidate_0', 'top_candidate_1']
        self.objects = {key: NativeObject(key, [index * .1, 0., .75], (.02, .02, .02))
            for index, key in enumerate(self.keys)}
        self.table = NativeObject('table', [0., 0., .37], (3., 3., .74), category='table')
        self.robot = NativeObject('robot', [1., 1., .4])
        self.robot.arm_names = ('left', 'right')
        self.robot._ag_obj_in_hand = {arm: None for arm in self.robot.arm_names}
        self.robot._ag_obj_constraints = {arm: None for arm in self.robot.arm_names}
        self.robot._ag_release_counter = {arm: None for arm in self.robot.arm_names}
        self.robot.is_grasping = lambda arm: 1 if self.robot._ag_obj_in_hand[arm] is not None else -1
        self.supports = {}
        for key in self.keys:
            self.support(key, [self.table])
        self.session = session_with([self.table, self.robot, *self.objects.values()], list(self.objects.values()))
        self.session.robot = self.robot
        self.session.env.env_config = {'action_frequency': 10}
        self.controller = SimpleNamespace(dt=.1, working_arm='left', debug_gripper_state=self.gripper)
        parameters = {'middle_key': 'middle', 'base_keys': self.keys[1:3],
            'top_keys': self.keys[3:], 'minimum_lift_m': None if transfer else .05,
            'minimum_stability_duration_s': .5, 'maximum_linear_speed_m_s': .02,
            'maximum_angular_speed_rad_s': .1}
        if transfer:
            parameters.update(manipulation_revision='support_transfer_v1', task_revision='wia_ordered_transfer_test_v1')
        if revision is not None:
            parameters.update(manipulation_revision=revision, task_revision='wia_final_stack_test_v1')
        self.config = {'type': 'category_operation', 'module': order.MODULE, 'operation': order.OPERATION,
            'scene': self.session.scene_name, 'room': self.session.room_name, 'binding_keys': self.keys,
            'bindings': {key: obj.name for key, obj in self.objects.items()}, 'parameters': parameters}
        self.evaluator = order.create_evaluator(self.session, self.controller, self.config, tmp_path)
        self.observation = SimpleNamespace(snapshot_id='fixture', action_step=0)
        self.evaluator.begin_episode(self.observation)

    def gripper(self):
        held = self.robot._ag_obj_in_hand['left']
        return {'object_name': held.name if held else None, 'holding': held is not None,
            'constraint_active': self.robot._ag_obj_constraints['left'] is not None,
            'release_pending': self.robot._ag_release_counter['left'] is not None,
            'gripper_open': held is None}

    def support(self, key, supports):
        obj = self.objects[key]
        self.supports[key] = list(supports)
        obj.states = {
            self.states.OnTop: SimpleNamespace(get_value=lambda other: other in self.supports[key]),
            self.states.Touching: SimpleNamespace(get_value=lambda other: other in self.supports[key]
                or (other is self.robot and self.robot._ag_obj_in_hand['left'] is obj))}
        obj.contacts = [SimpleNamespace(body0=obj.prim_path + '/body', body1=other.prim_path + '/top')
            for other in supports]

    def hold(self, key):
        self.robot._ag_obj_in_hand['left'] = self.objects[key] if key else None
        self.robot._ag_obj_constraints['left'] = object() if key else None

    def step(self, count=1):
        for _ in range(count):
            self.evaluator.sample_step()

    def move_release(self, key, support, *, lift=.02, separate=True):
        self.hold(key)
        self.step()
        if separate:
            self.support(key, [])
        self.objects[key].position[2] += lift
        self.step()
        self.hold(None)
        self.support(key, [support])
        self.step()

    def result(self):
        return self.evaluator.record(self.observation, action=None)


def test_genuine_sub_five_cm_support_transfer_and_ordered_release_succeeds(tmp_path, native_types):
    trial = StackTrial(tmp_path, native_types, top=True)
    trial.move_release('middle', trial.objects['base_candidate_0'])
    trial.step(6)
    assert trial.evaluator.middle_placement is not None
    trial.move_release('top_candidate_0', trial.objects['middle'])
    trial.step(6)
    result = trial.result()
    assert result['task_success'] is True
    assert result['schema'] == 'private_ordered_stack_support_transfer_v1'
    assert result['manipulation_revision'] == 'support_transfer_v1'
    assert result['task_revision'] == 'wia_ordered_transfer_test_v1'
    assert result['checks']['middle_grasp_support_transfer_release'] is True
    assert 'middle_grasp_lift_release' not in result['checks']
    for key in ('middle', 'top_candidate_0'):
        history = result['histories'][key]
        assert history['unsupported_while_acquired'] is True
        assert 0 < history['peak_lift_m'] < .05
        assert history['acquired_step'] <= history['separation_step'] < history['release_step']


@pytest.mark.parametrize('retained_support', ['external', 'candidate'])
def test_grasp_release_without_support_separation_never_counts(tmp_path, native_types, retained_support):
    trial = StackTrial(tmp_path, native_types)
    if retained_support == 'candidate':
        trial.support('middle', [trial.objects['base_candidate_0']])
    trial.move_release('middle', trial.objects['base_candidate_0'], separate=False)
    trial.step(15)
    result = trial.result()
    assert result['task_success'] is False
    assert result['histories']['middle']['unsupported_while_acquired'] is False
    assert result['middle_placement'] is None


def test_separation_before_genuine_acquisition_does_not_count(tmp_path, native_types):
    trial = StackTrial(tmp_path, native_types)
    trial.support('middle', [])
    trial.step()
    trial.support('middle', [trial.table])
    trial.move_release('middle', trial.objects['base_candidate_0'], separate=False)
    trial.step(15)
    assert trial.result()['task_success'] is False


def test_acquired_support_separation_requires_actual_subsequent_release(tmp_path, native_types):
    trial = StackTrial(tmp_path, native_types)
    trial.hold('middle')
    trial.step()
    trial.support('middle', [])
    trial.step()
    trial.support('middle', [trial.objects['base_candidate_0']])
    trial.step(15)
    result = trial.result()
    assert result['histories']['middle']['unsupported_while_acquired'] is True
    assert result['histories']['middle']['release_step'] is None
    assert result['task_success'] is False


def test_top_grasp_before_middle_placement_fails_order(tmp_path, native_types):
    trial = StackTrial(tmp_path, native_types, top=True)
    trial.move_release('top_candidate_0', trial.table)
    trial.move_release('middle', trial.objects['base_candidate_0'])
    trial.step(6)
    assert trial.evaluator.middle_placement is not None
    trial.support('top_candidate_0', [trial.objects['middle']])
    trial.step(10)
    result = trial.result()
    assert result['checks']['top_grasp_after_middle_placement'] is False
    assert result['task_success'] is False


@pytest.mark.parametrize('interference', ['hand', 'unstable', 'external_prop', 'wrong_support'])
def test_final_support_hand_clear_and_stability_guards_remain(tmp_path, native_types, interference):
    trial = StackTrial(tmp_path, native_types)
    trial.move_release('middle', trial.objects['base_candidate_0'])
    if interference == 'hand':
        trial.hold('middle')
    elif interference == 'unstable':
        trial.objects['middle'].get_linear_velocity = lambda: [.03, 0., 0.]
    elif interference == 'external_prop':
        trial.support('middle', [trial.objects['base_candidate_0'], trial.table])
    else:
        trial.support('middle', [trial.table])
    trial.step(15)
    assert trial.result()['task_success'] is False


def test_missing_native_support_evidence_is_unavailable_and_cannot_earn_separation(tmp_path, native_types):
    trial = StackTrial(tmp_path, native_types)
    trial.hold('middle')
    trial.step()
    del trial.objects['middle'].states[native_types.OnTop]
    trial.step()
    trial.hold(None)
    trial.support('middle', [trial.objects['base_candidate_0']])
    trial.step(15)
    result = trial.result()
    assert result['task_success'] is False
    assert result['measurement_error'] is not None
    assert result['evaluation_validity'] == 'unavailable'
    assert result['histories']['middle']['unsupported_while_acquired'] is False


@pytest.mark.parametrize('lift,success', [(.04, False), (.05, True)])
def test_legacy_five_cm_gate_schema_and_histories_are_unchanged(tmp_path, native_types, lift, success):
    trial = StackTrial(tmp_path, native_types, transfer=False)
    trial.move_release('middle', trial.objects['base_candidate_0'], lift=lift)
    trial.step(15)
    result = trial.result()
    assert result['task_success'] is success
    assert result['schema'] == 'private_ordered_stack_operation_v2'
    assert result['task_revision'] == order.TASK_REVISION
    assert 'middle_grasp_lift_release' in result['checks']
    assert 'manipulation_revision' not in result
    assert 'unsupported_while_acquired' not in result['histories']['middle']


@pytest.mark.parametrize('changes', [
    {'manipulation_revision': 'unknown'}, {'minimum_lift_m': .05},
    {'minimum_stability_duration_s': .1}, {'maximum_linear_speed_m_s': .2},
])
def test_opt_in_does_not_relax_unrelated_thresholds_or_accept_unknown_revision(tmp_path, native_types, changes):
    trial = StackTrial(tmp_path, native_types)
    config = deepcopy(trial.config)
    config['parameters'].update(changes)
    with pytest.raises(ValueError):
        order._validate_operation(trial.session, config)
