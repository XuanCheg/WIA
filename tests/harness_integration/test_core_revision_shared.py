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
