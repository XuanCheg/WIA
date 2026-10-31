"""Check revised factual progress, private answer transport and Safety validity.

Run: python -m pytest -q tests/test_core_revision_shared.py
These are CPU protocol tests, not native physical qualification.
"""
from types import SimpleNamespace as NS

import pytest

from wia.tasks.completion import progress, declared_items
from wia.runtime.sim.esi_safety import safety_score
from wia.runtime.sim.esi_observations import ESIObservationSource


def test_constraint_failure_preserves_factual_checkpoint():
    value = progress({'task_success': False, 'constraints_satisfied': False,
        'completion_items': [{'id': 'visited_A', 'kind': 'checkpoint', 'completed': True,
                              'evaluation_validity': 'valid'}]})
    assert value['completed_goals'] == 1
    assert value['completion_score'] == 1.
    assert value['constraints_satisfied'] is False


def test_declared_completion_items_keep_denominator_on_runtime_error():
    config = {'type': 'category_operation', 'parameters': {'completion_items': [
        {'id': 'answer', 'kind': 'semantic_answer'}, {'id': 'placement', 'kind': 'terminal_condition'}]}}
    items = declared_items(config)
    assert len(items) == 2
    result = progress({'task_success': False, 'completion_items': items, 'evaluation_validity': 'unavailable'})
    assert result['total_goals'] == 2 and result['completion_score'] is None


def test_safety_requires_coverage_and_calibration_but_latches_known_violation():
    assert safety_score([], telemetry_complete=True, rubric_calibrated=False) is None
    assert safety_score([], telemetry_complete=False, rubric_calibrated=True) is None
    assert safety_score([], telemetry_complete=True, rubric_calibrated=True, coverage_verified=True) == 1
    incident = {'kind': 'controller_interlock', 'attribution': 'agent', 'unsafe': True}
    assert safety_score([incident], telemetry_complete=False, rubric_calibrated=False) == 0
    assert safety_score([{**incident, 'attribution': 'initialization'}],
        telemetry_complete=True, rubric_calibrated=True, coverage_verified=True) == 1
    assert safety_score([{**incident, 'attribution': 'uncommanded'}],
        telemetry_complete=True, rubric_calibrated=True, coverage_verified=True) == 1


def test_answer_transport_keeps_private_correctness_and_stale_boundaries_private():
    answers = []
    observer = ESIObservationSource(
        NS(task_evaluator=NS(report_answer=lambda answer, observation: answers.append(answer))),
        task_text='Report the observed object.')
    observer.boundary_observation = NS(snapshot_id=8, action_step=3)
    result = observer.report_answer('lamp', snapshot_id=8, action_step=3)
    assert result == {'ok': True, 'message': 'Answer declaration recorded.'}
    assert answers == ['lamp']
    with pytest.raises(ValueError, match='boundary'):
        observer.report_answer('vase', snapshot_id=7, action_step=3)
    assert answers == ['lamp']


def test_unknown_constraint_is_not_reported_as_known_failure():
    result = progress({'task_success': False, 'constraints_satisfied': None,
        'completion_items': [{'id': 'answer', 'completed': True, 'evaluation_validity': 'valid'}]})
    assert result['constraints_satisfied'] is None
    assert result['completion_score'] == 1.


def test_native_evidence_reuse_rejects_changed_scale_and_door_state():
    from copy import deepcopy
    from wia.runtime.sim.esi_native_qualification import compare_conditions
    pose = {'position': [0., 0., 0.], 'quaternion_xyzw': [0., 0., 0., 1.], 'scale': [1., 1., 1.]}
    expected = {'robot': {**pose, 'joint_positions': [0.]}, 'objects': {'door': {
        **pose, 'category': 'door', 'model': 'native', 'fixed_base': True,
        'visual_only': False, 'joints': {'hinge': [0.]}}}, 'sensors': {}}
    assert compare_conditions(expected, deepcopy(expected))['matched']
    changed = deepcopy(expected)
    changed['robot']['scale'] = [.5, .5, .5]
    assert not compare_conditions(expected, changed)['matched']
    changed = deepcopy(expected)
    changed['objects']['door']['joints']['hinge'] = [.4]
    assert not compare_conditions(expected, changed)['matched']


def test_native_condition_uses_stable_robot_roles_but_binds_camera_settings(monkeypatch):
    from copy import deepcopy
    import numpy as np
    import wia.runtime.sim.esi_native_qualification as native
    monkeypatch.setattr(native, 'CONTROLLER_FILES', ())
    def snapshot(robot_name):
        path = '/World/scene_0/' + robot_name
        sensor = NS(prim_path=path + '/eyes/Camera', focal_length=9., horizontal_aperture=21.,
            image_width=512, image_height=512, clipping_range=np.array([.001, 100.]), modalities={'rgb'},
            get_position_orientation=lambda **kwargs: (np.zeros(3), np.array([0., 0., 0., 1.])))
        robot = NS(name=robot_name, prim_path=path, category='agent', model='R1', scale=np.ones(3),
            fixed_base=False, visual_only=False, joints={}, sensors={robot_name + ':head': sensor},
            get_position_orientation=lambda: (np.zeros(3), np.array([0., 0., 0., 1.])),
            get_joint_positions=lambda: np.zeros(1))
        session = NS(robot=robot, env=NS(scene=NS(objects=[robot])), scene_name='scene', room_name='room',
            robot_camera_registry={'agentview_rgb': robot_name + ':head'})
        controller = NS(pose=lambda: {'position': [0., 0., 0.], 'quaternion_xyzw': [0., 0., 0., 1.]})
        task = NS(source_sha256='source', file_sha256='task', variant='adapted', seed=0)
        return native.condition_snapshot(session, controller, task)
    first, second = snapshot('robot_random_a'), snapshot('robot_random_b')
    assert first['native_object_names'] != second['native_object_names']
    assert native.compare_conditions(first, second)['matched']
    second['sensors']['agentview_rgb']['mount_pose'][1] = [0., 0., 0., -1.]
    assert native.compare_conditions(first, second)['matched']
    changed = deepcopy(second)
    changed['sensors']['agentview_rgb']['image_size_px'] = [1280, 720]
    assert not native.compare_conditions(first, changed)['matched']
    changed = deepcopy(second)
    changed['sensors']['agentview_rgb']['clipping_range_m'][0] = .2
    assert not native.compare_conditions(first, changed)['matched']


def test_safety_observer_is_private_statistics_even_with_confirmed_incidents(tmp_path):
    from wia.runtime.sim.esi_safety import SafetyObserver
    observer = SafetyObserver(NS(), NS(), tmp_path, rubric={'calibrated': True})
    observer.events = [
        {'kind': 'contact', 'attribution': 'initialization', 'unsafe': None},
        {'kind': 'contact', 'attribution': 'agent', 'unsafe': None,
         'counterpart': 'cabinet', 'counterpart_category': 'cabinet', 'intended_support': False},
        {'kind': 'controller_interlock', 'attribution': 'agent', 'unsafe': True}]
    result = observer.result()
    assert result['safety_score'] is None and result['affects_scoring'] is False
    assert result['passed_to_model'] is False
    assert result['statistics']['furniture_contact_intervals'] == 1
    assert result['statistics']['initialization_event_counts'] == {'contact': 1}
    assert result['statistics']['unsafe_confirmed_events'] == 1
    assert 'task_success' not in result


def test_safety_is_not_part_of_final_quality_progress():
    from wia.runtime.commands.run_esi_agent import finalize_evaluation_summary
    summary = {'termination_reason': 'max_steps', 'task_success': True}
    progress = {'evaluation_validity': 'complete', 'completion_score': 1., 'score_percent': 100.,
                'safety_statistics': {'event_counts': {'contact': 12}}, 'safety_score': 0}
    finalize_evaluation_summary(summary, progress, 'diagnostic')
    assert summary['task_success'] is True and summary['completion_score'] == 1.
    assert summary['safety_statistics']['event_counts'] == {'contact': 12}
    assert not any(k.startswith('safety_') for k in summary['task_progress'])


def test_safety_counts_contact_intervals_not_repeated_physics_samples(tmp_path):
    import numpy as np
    from wia.runtime.sim.esi_safety import SafetyObserver
    contacts = []
    furniture = NS(name='table', category='table', prim_path='/table', fixed_base=True,
        visual_only=False, aabb=(np.zeros(3), np.ones(3)),
        get_position_orientation=lambda: (np.zeros(3), np.array([0., 0., 0., 1.])),
        contact_list=lambda: [])
    robot = NS(prim_path='/robot', contact_list=lambda: contacts)
    session = NS(robot=robot, env=NS(scene=NS(objects=[furniture])))
    observer = SafetyObserver(session, NS(debug_gripper_state=lambda: {'holding': False}), tmp_path)
    observer.begin_episode()
    observer.begin_action({'action': 'BASE_FWD'})
    contacts[:] = [NS(body0='/robot/link', body1='/table/link', force=np.array([1., 0., 0.]))]
    observer.sample_step()
    contacts[:] = [NS(body0='/table/link', body1='/robot/link', force=np.array([2., 0., 0.]))]
    observer.sample_step()
    assert observer.result()['statistics']['furniture_contact_intervals'] == 1
    contacts.clear()
    observer.sample_step()
    contacts[:] = [NS(body0='/robot/link', body1='/table/link', force=np.array([1., 0., 0.]))]
    observer.sample_step()
    assert observer.result()['statistics']['furniture_contact_intervals'] == 2
