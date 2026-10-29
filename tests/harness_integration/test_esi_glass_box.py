"""Check physical task semantics. Run .venv/bin/python -m pytest -q this file."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from wia.native.esi_glass_box import OpenExtractLiftEvaluator, SlideExtractLiftEvaluator
from harness_integration.task_examples import load_example_task as load_task
from wia.tasks.scene import DeclarativeScene
from harness_integration.test_esi_task_evaluation import make_evaluator


def test_task_projects_only_public_goal_and_keeps_original_esi_objects():
    task = load_task('retrieve_apple_and_lift', variant='closed_sliding_box')
    public = json.dumps(task.public_payload)
    assert all(hint not in public.lower() for hint in ('glass', 'box', 'lid', 'hinge', 'door', '60 degrees'))
    assert 'apple' in public and 'task_apple' not in public and 'ymhxqk' not in public
    assert '0.166' not in public and 'lid_hinge' not in public
    config = {'robots': [{'type': 'R1'}], 'objects': [{'name': 'original_esi_object'}]}
    DeclarativeScene(task).apply_to_env_config(config)
    assert config['objects'][0] == {'name': 'original_esi_object'}
    box, apple = config['objects'][1:]
    assert box['type'] == 'USDObject' and box['fixed_base'] and not box['visual_only']
    assert apple['category'] == 'apple' and not apple['fixed_base'] and not apple['visual_only']


def test_success_requires_closed_start_opening_and_complete_held_extraction(tmp_path):
    base, target, gripper, observation = make_evaluator(tmp_path / 'base')
    state = {'angle': 0.}
    joint = SimpleNamespace(get_state=lambda: (np.array([np.deg2rad(state['angle'])]), None, None))
    box = SimpleNamespace(joints={'lid_hinge': joint},
                          get_position_orientation=lambda: (np.array([0., 0., .75]), None))
    base.session.env.scene.object_registry = lambda _, name: box if name == 'box' else target
    evaluator = OpenExtractLiftEvaluator(base.session, base.controller,
        {'type': 'open_extract_lift', 'scene': 'scene', 'room': 'room', 'target': 'obj_small',
         'container': 'box', 'hinge_joint': 'lid_hinge', 'minimum_open_angle_deg': 60.,
         'rim_height_m': .166, 'extraction_clearance_m': .005, 'minimum_lift_m': .05}, tmp_path / 'box')
    evaluator.begin_episode(observation)
    gripper.update(object_name='obj_small', holding=True, constraint_active=True)
    target.height = .87  # Raised by 7 cm, but still below the rim.
    state['angle'] = 80.
    evaluator.sample_step()
    state['angle'] = 0.  # Opening between observation boundaries must still count.
    result = evaluator.record(observation, action='MV_UP')
    assert result['criteria']['door_opened'] and not result['task_success']
    target.height = .93
    assert evaluator.record(observation, action='MV_UP')['task_success']
    gripper['object_name'] = 'box'
    assert not evaluator.record(observation, action='GRASP')['task_success']
    gripper['object_name'] = 'obj_small'
    state['angle'] = 0.
    evaluator.begin_episode(observation)
    target.height = 1.1
    assert not evaluator.record(observation, action='MV_UP')['task_success']


def test_initially_open_door_cannot_satisfy_closed_door_task(tmp_path):
    base, target, gripper, observation = make_evaluator(tmp_path / 'base')
    joint = SimpleNamespace(get_state=lambda: (np.array([np.deg2rad(100.)]), None, None))
    box = SimpleNamespace(joints={'lid_hinge': joint},
                          get_position_orientation=lambda: (np.array([0., 0., .75]), None))
    base.session.env.scene.object_registry = lambda _, name: box if name == 'box' else target
    evaluator = OpenExtractLiftEvaluator(base.session, base.controller,
        {'type': 'open_extract_lift', 'scene': 'scene', 'room': 'room', 'target': 'obj_small',
         'container': 'box', 'hinge_joint': 'lid_hinge', 'minimum_open_angle_deg': 60.,
         'rim_height_m': .166, 'extraction_clearance_m': .005, 'minimum_lift_m': .05}, tmp_path / 'box')
    evaluator.begin_episode(observation)
    gripper.update(object_name='obj_small', holding=True, constraint_active=True)
    target.height = 1.
    assert not evaluator.record(observation, action='MV_UP')['task_success']
    assert evaluator.max_angle_deg == pytest.approx(100.)


def test_sliding_gate_uses_metres_and_requires_closed_start_and_held_extraction(tmp_path):
    base, target, gripper, observation = make_evaluator(tmp_path / 'base')
    state = {'position_m': 0.}
    joint = SimpleNamespace(get_state=lambda: (np.array([state['position_m']]), None, None))
    box = SimpleNamespace(joints={'lid_slide': joint},
                          get_position_orientation=lambda: (np.array([0., 0., .75]), None))
    base.session.env.scene.object_registry = lambda _, name: box if name == 'box' else target
    evaluator = SlideExtractLiftEvaluator(base.session, base.controller,
        {'type': 'slide_extract_lift', 'scene': 'scene', 'room': 'room', 'target': 'obj_small',
         'container': 'box', 'slide_joint': 'lid_slide', 'minimum_open_displacement_m': .20,
         'rim_height_m': .166, 'extraction_clearance_m': .005, 'minimum_lift_m': .05}, tmp_path / 'box')
    evaluator.begin_episode(observation)
    gripper.update(object_name='obj_small', holding=True, constraint_active=True)
    target.height = .93
    state['position_m'] = .199
    assert not evaluator.record(observation, action='MV_FWD')['task_success']
    state['position_m'] = .20
    evaluator.sample_step()
    state['position_m'] = 0.  # Opening between boundaries is retained.
    result = evaluator.record(observation, action='MV_UP')
    assert result['task_success'] and result['maximum_door_slide_displacement_m'] == pytest.approx(.20)
    assert 'maximum_door_angle_deg' not in result
    target.height = .87
    assert not evaluator.record(observation, action='MV_UP')['task_success']
    target.height = .93
    gripper['object_name'] = 'box'
    assert not evaluator.record(observation, action='GRASP')['task_success']
    gripper['object_name'] = 'obj_small'
    state['position_m'] = .20
    evaluator.begin_episode(observation)
    assert not evaluator.record(observation, action='MV_UP')['task_success']


def test_sliding_task_declares_prismatic_initialization_in_metres():
    task = load_task('retrieve_apple_and_lift', variant='closed_sliding_box')
    box = next(patch for patch in task.definition.scene.variants[task.variant].patches
               if patch.operation == 'add_usd_object')
    assert box.initial_joint_positions_m == {'lid_slide': 0.}
    assert box.initial_joint_positions_rad == {}
    assert task.definition.private.evaluation.minimum_open_displacement_m == .20
    with pytest.raises(ValueError, match='one position unit'):
        type(box).model_validate(dict(box.model_dump(), initial_joint_positions_rad={'lid_slide': 0.}))
