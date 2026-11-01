"""Run: PYTHONPATH=src:tests:../Show-Harness-Explore python -m pytest -q tests/test_safety_independence.py.

Safety-only perturbations must leave task evidence and quality unchanged.
Fixtures exercise real reducers and owner boundaries without GPU simulation.
"""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.candidates.high import evaluate
from wia.authoring.converters.temporal_understanding_agent_observation import NativePlacementProgress
from wia.authoring.converters.cognitive_mapping_visit_reports import settled_native_state
from wia.evaluation.activity import aggregate_activity
from wia.evaluation.safety import SafetyEvaluator, record_safety, run_safety_hook, snapshot_safety_log, write_safety_log
from harness_integration.test_native_temporal_phases import world as temporal_world
from harness_integration.test_agent_observation import source_and_rig
from harness_integration.test_wia_layout_fixed_mount import mount, mount_parameters
from harness_integration.test_wia_layout_local_support import native
from test_candidate_high import h4


@pytest.mark.parametrize('safety', [False, None, 'missing'])
def test_h4_ignores_legacy_collision_flags_and_keeps_actual_hand_goal(safety):
    contract, frames = h4()
    baseline = evaluate('H4', contract, frames[-1], frames[:-1])
    assert baseline['task_success']
    frames[-1]['process_evidence'].update(approach_clear=safety, withdrawal_clear=safety)
    assert evaluate('H4', contract, frames[-1], frames[:-1]) == baseline
    frames[-1]['process_evidence']['hands_withdrawn'] = False
    assert not evaluate('H4', contract, frames[-1], frames[:-1])['task_success']


@pytest.mark.parametrize('safety', [True, False, None, 'missing'])
def test_temporal_placement_ignores_safety_availability_but_requires_real_release(safety):
    tracker = NativePlacementProgress({'release_stability_steps': 2,
        'maximum_release_speed_m_s': .02, 'maximum_release_angular_speed_rad_s': .1})
    measured = {'gripper': {'holding': False, 'constraint_active': False,
        'release_pending': False, 'object_name': None}, 'destination_relation': True,
        'linear_speed_m_s': 0., 'angular_speed_rad_s': 0., 'robot_withdrawn': True,
        'target_dynamic_collidable': True}
    if safety != 'missing':
        measured['robot_terminal_safe'] = safety
    assert tracker.update(measured, 'target', count_step=True) == (True, False)
    assert tracker.update(measured, 'target', count_step=True) == (True, True)
    measured['robot_withdrawn'] = False
    assert tracker.update(measured, 'target', count_step=True) == (True, False)
    measured['destination_relation'] = None
    assert tracker.update(measured, 'target', count_step=True) == (False, False)


@pytest.mark.parametrize('supported,quaternion', [(True, [0., 0., 0., 1.]),
    (False, [1., 0., 0., 0.]), (None, [0., 0., 0., 1.])])
def test_room_occupancy_uses_actual_footprint_instead_of_robot_safety(supported, quaternion):
    link = NS(get_linear_velocity=lambda: np.zeros(3), get_angular_velocity=lambda: np.zeros(3))
    session = NS(robot=NS(base_footprint_link_name='base', links={'base': link}))
    state = {'chassis_region': 'room_0', 'known_regions': ['room_0'], 'floor_layer_known': True,
             'quaternion_xyzw': quaternion, 'floor_contact': supported, 'state_valid': False}
    measured = settled_native_state(session, state)
    assert measured['occupancy_valid'] and measured['state_valid']
    state['chassis_region'] = None
    assert not settled_native_state(session, state)['occupancy_valid']


def test_temporal_source_validity_does_not_read_generic_robot_contact_telemetry(temporal_world):
    world = temporal_world
    baseline = world.controller.measure(world.executor)
    assert baseline['valid']
    def unavailable():
        raise RuntimeError('Generic contact telemetry unavailable')
    world.session.robot.contact_list = unavailable
    measured = world.controller.measure(world.executor)
    assert measured == baseline
    assert not any(key in measured['checks'] for key in ('robot_clear', 'robot_floor_support', 'robot_upright'))


def test_safety_hooks_and_output_failures_cannot_interrupt_owner_boundaries(tmp_path):
    source, session = source_and_rig()
    calls = []
    class BrokenSafety:
        unavailable = set()
        def begin_episode(self):
            calls.append('begin')
            raise OSError('Safety log unavailable')
        def record(self, observation):
            calls.append('record')
            raise RuntimeError('Safety telemetry unavailable')
    session.safety_evaluator = BrokenSafety()
    source.capture(action_boundary=True)
    before = source.agent_observation().to_payload()
    record_safety(session, source.agent_observation())
    run_safety_hook(session, 'begin_episode')
    assert source.agent_observation().to_payload() == before
    assert calls == ['record', 'record', 'begin']
    assert len(session.safety_evaluator.unavailable) == 2


def test_safety_hook_failure_does_not_interrupt_a_real_r1_adapter_action():
    from harness_integration.test_esi_gripper import make_manipulation_controller
    from wia.runtime.sim.esi_actions import create_r1_executor
    controller, _, _ = make_manipulation_controller()
    source, session = source_and_rig()
    controller.session.observation_source = source
    source.capture(action_boundary=True)
    class BrokenSafety:
        unavailable = set()
        action = None
        def begin_action(self, action):
            raise RuntimeError('Safety start failure')
        def end_action(self, result):
            raise RuntimeError('Safety end failure')
        def record(self, observation):
            raise OSError('Safety log failure')
    controller.session.safety_evaluator = session.safety_evaluator = BrokenSafety()
    result = create_r1_executor(controller).execute('RELEASE')
    assert result['completed'] and source.agent_observation().action_step == 1
    assert len(session.safety_evaluator.unavailable) == 3
    assert 'safety' not in json.dumps(source.agent_observation().metadata()).lower()


def test_active_layout_qualifies_actual_item_support_without_reading_robot_safety(mount, monkeypatch):
    from wia.authoring.converters import active_layout
    from wia.tasks import interaction_setup
    def unavailable():
        raise RuntimeError('Generic robot contacts unavailable')
    monkeypatch.setattr(interaction_setup, '_robot_contact_check', unavailable)
    value = active_layout.initialize_scene(mount.session, mount_parameters())
    assert value['accepted'] and 'robot' not in value


def test_generic_safety_logs_are_separate_from_quality_results_and_activity(tmp_path):
    session = NS()
    evaluator = SafetyEvaluator(session, NS(), tmp_path/'trial/private_safety')
    evaluator.samples = 2
    evaluator.episode_started = evaluator.started
    evaluator._event('contact', attribution='agent', counterpart='chair', counterpart_category='chair',
        intended_support=False, classification='contact_requires_task_context')
    session.safety_evaluator = evaluator
    record_safety(session, NS(snapshot_id=4, action_step=2))
    result = json.loads((evaluator.output/'safety.json').read_text())
    assert result['statistics']['furniture_contact_intervals'] == 1
    assert result['snapshot_id'] == 4 and result['action_step'] == 2
    assert result['mode'] == 'statistics_only' and result['safety_score'] is None
    assert (evaluator.output/'events.jsonl').is_file() and (evaluator.output/'boundaries.jsonl').is_file()
    snapshot_safety_log(tmp_path/'trial')
    assert json.loads((tmp_path/'trial/final_safety.json').read_text()) == result
    trial = {'index': 0, 'output_dir': str(tmp_path/'trial'), 'task_success': True,
             'completion_score': 1., 'score_percent': 100.}
    before = deepcopy(trial)
    logged = write_safety_log([trial, {'index': 1, 'output_dir': str(tmp_path/'missing')}], tmp_path/'safety_results.json')
    assert trial == before
    assert logged['event_counts'] == {'contact': 1} and logged['missing_trials'] == 1
    assert not any('safety' in key for key in aggregate_activity([trial]))
    malformed = tmp_path/'trial/private_safety/safety.json'
    malformed.write_text('[]')
    assert write_safety_log([trial], tmp_path/'safety_results.json')['missing_trials'] == 1
    assert trial == before


def test_safety_aggregate_write_failure_does_not_modify_a_task_result(tmp_path, monkeypatch):
    from wia.evaluation import safety
    trial = {'index': 0, 'task_success': True, 'score_percent': 100.}
    before = deepcopy(trial)
    def unavailable(*args, **kwargs):
        raise OSError('Safety logging filesystem unavailable')
    monkeypatch.setattr(safety, 'write_json', unavailable)
    value = safety.write_safety_log([trial], tmp_path/'safety_results.json')
    assert value['missing_trials'] == 1 and trial == before
    safety.snapshot_safety_log(tmp_path)


@pytest.mark.parametrize('legacy_safety', [True, False, None])
@pytest.mark.parametrize('interference', [False, True])
def test_ramp_trials_ignore_safety_envelope_but_require_unassisted_physics(
        tmp_path, monkeypatch, legacy_safety, interference):
    from wia.authoring.converters import inclined_comparison as ramp
    objects = {key: NS(name=key, position=np.zeros(3)) for key in ('trial', 'spare')}
    session = NS(env=NS(scene=NS(object_registry=lambda field, name: objects[name])))
    controller = NS(geometric_rearrangement_evidence=lambda: {'available': True, 'state_setter_used': False},
        debug_gripper_state=lambda: {'holding': False, 'constraint_active': False,
                                    'object_name': None, 'release_pending': False})
    config = {'bindings': {key: key for key in objects}, 'parameters': {
        'candidate_pairs': [{'trial': 'trial', 'spare': 'spare'}], 'apparatus_keys': [],
        'trial_duration_s': 3., 'arrival_tie_tolerance_s': .15}}
    instance = ramp.InclinedComparisonEvaluator(session, controller, config, tmp_path)
    instance.active = True
    instance.initial = {'spare': {'position': [0., 0., 0.]}}
    instance.traces = {'trial': {'release_time': 0., 'last_time': 0., 'samples': 0,
        'valid': True, 'arrival_s': None, 'complete': False}}
    now = [0.]
    monkeypatch.setattr(ramp, '_native_time', lambda: {'time_s': now[0]})
    monkeypatch.setattr(ramp, '_pose', lambda obj: {'position': obj.position.tolist()})
    instance._measure = lambda key: {'release_ready': False, 'in_receiver': key == 'trial',
        'safe': legacy_safety, 'interference': ['/robot/hand'] if interference else [],
        'linear_speed_m_s': 0., 'angular_speed_rad_s': 0.}
    for index in range(1, 63):
        now[0] = index * .05
        instance.sample_step()
    value = instance.record(NS(snapshot_id=1, action_step=0))
    assert value['trial_records']['trial']['valid'] is (not interference)
    assert value['completion_score'] == pytest.approx(0. if interference else 1 / 3)
    assert value['evaluation_validity'] == 'complete'


@pytest.mark.parametrize('incidental_contacts', [[], ['/robot/forearm'], None])
def test_cloth_incidental_contacts_do_not_erase_a_valid_reveal_and_grasp(
        tmp_path, monkeypatch, incidental_contacts):
    from wia.authoring.converters import physical_structure_deformable as cloth
    instance = cloth.SweepRetrieveEvaluator.__new__(cloth.SweepRetrieveEvaluator)
    instance.session, instance.controller, instance.parameters = NS(), NS(), {}
    instance.config, instance.output, instance.episode = {}, tmp_path, 1
    instance.objects = {key: NS(name=key, visual_only=False, fixed_base=False) for key in ('item', 'cloth')}
    instance.initial = {'item_aabb_min': [0., 0., 0.], 'item_aabb_max': [.1, .1, .1],
        'item_position': [0., 0., .05], 'cloth_centroid': [0., 0., .2],
        'item_scale': [1., 1., 1.], 'cloth_scale': [1., 1., 1.],
        'cloth_point_count': 4, 'cloth_face_count': 2, 'floor_top_m': 0.}
    instance.hand_contact_seen = instance.revealed_before_grasp = instance.invalid = False
    instance.last_time, instance.hold_samples, instance.hold_start = 0., 0, None
    instance.measurement_error = None
    row = {**deepcopy(instance.initial), 'time_s': .02, 'sim_step_dt_s': .02,
        'gripper': {'holding': False, 'constraint_active': False, 'object_name': None, 'release_pending': False},
        'cloth_centroid': [.2, 0., .2], 'initial_region_overlap_faces': 0,
        'item_touching_task_support': True, 'item_touching_cloth': False,
        'working_hand_cloth_contacts': ['/robot/left_gripper'], 'other_robot_cloth_contacts': incidental_contacts,
        'cloth_min_z': -1., 'rigid_assisted_grasp_mode': True, 'robot_in_original_room': True}
    monkeypatch.setattr(cloth, '_state', lambda *a, **k: deepcopy(row))
    instance._update()
    assert instance.revealed_before_grasp and not instance.invalid
    row.update(item_position=[0., 0., .15], item_aabb_min=[0., 0., .1], item_touching_task_support=False,
        gripper={'holding': True, 'constraint_active': True, 'object_name': 'item', 'release_pending': False})
    for index in range(2, 55):
        row['time_s'] = index * .02
        instance.sample_step()
    value = instance.record(NS(snapshot_id=2, action_step=1), action='GRASP')
    assert value['task_success'] and value['evaluation_validity'] == 'complete'
    assert value['criteria']['working_hand_contact_seen'] and value['criteria']['sustained_native_hold']
