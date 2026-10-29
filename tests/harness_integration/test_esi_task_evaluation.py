"""Private success requires target attachment AND lift, never a model DONE."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from wia.native.esi_task_evaluation import GraspLiftEvaluator
from harness_integration.test_agent_observation import source_and_rig


class Target:
    name, category, visual_only, fixed_base = "obj_small", "pool_ball", False, False
    height = 0.8

    @property
    def aabb(self):
        return np.array([0, 0, self.height]), np.array([0.04, 0.04, self.height+0.04])

    def get_position_orientation(self):
        return np.array([0.02, 0.02, self.height+0.02]), np.array([0, 0, 0, 1])


def make_evaluator(tmp_path):
    target = Target()
    gripper = {"object_name": None, "holding": False, "constraint_active": False}
    controller = SimpleNamespace(debug_gripper_state=lambda: dict(gripper), pose=lambda: {}, ee_pose=lambda: {})
    session = SimpleNamespace(scene_name="scene", room_name="room", env=SimpleNamespace(
        scene=SimpleNamespace(object_registry=lambda *args: target)))
    evaluator = GraspLiftEvaluator(session, controller,
        {"type": "grasp_lift", "scene": "scene", "room": "room", "target": "obj_small", "minimum_lift_m": 0.05}, tmp_path)
    source, _ = source_and_rig()
    observation = source.capture(action_boundary=True).observation
    evaluator.begin_episode(observation)
    return evaluator, target, gripper, observation


def test_target_must_be_attached_and_lifted_and_private_state_is_atomically_saved(tmp_path):
    evaluator, target, gripper, observation = make_evaluator(tmp_path)
    assert not evaluator.record(observation, action="GRASP")["task_success"]
    target.height += 0.1
    assert not evaluator.record(observation, action="MV_UP")["task_success"]  # Lift without holding fails.
    gripper.update(object_name="wrong_object", holding=True, constraint_active=True)
    assert not evaluator.record(observation, action="GRASP")["task_success"]
    gripper["object_name"] = "obj_small"
    target.height = 0.81
    assert not evaluator.record(observation, action="GRASP")["task_success"]  # Attached without clearance fails.
    target.height = 0.86
    result = evaluator.record(observation, action="MV_UP")
    assert result["task_success"] and result["lift_m"] == pytest.approx(0.06)
    assert json.loads((tmp_path / "state.json").read_text()) == result
    assert not (tmp_path / "state.json.tmp").exists()
    assert "task_success" not in observation.to_payload() and "obj_small" not in json.dumps(observation.to_payload())
    gripper.update(object_name=None, holding=False, constraint_active=False)
    assert not evaluator.record(observation, action="RELEASE")["task_success"]


def test_initially_held_object_does_not_count_as_a_new_pickup(tmp_path):
    evaluator, target, gripper, observation = make_evaluator(tmp_path)
    gripper.update(object_name="obj_small", holding=True, constraint_active=True)
    evaluator.begin_episode(observation)
    target.height += 0.1
    assert not evaluator.record(observation, action="MV_UP")["task_success"]


def test_manual_motion_before_episode_does_not_create_a_task_baseline(tmp_path):
    evaluator, target, gripper, observation = make_evaluator(tmp_path)
    evaluator.initial = None
    before = (tmp_path / "trajectory.jsonl").read_text()
    target.height += .1
    assert evaluator.record(observation, action="TRUNK_UP") is None
    assert evaluator.initial is None and (tmp_path / "trajectory.jsonl").read_text() == before


def test_episode_source_evaluator_hook_does_not_change_agent_payload(tmp_path):
    evaluator, _, _, _ = make_evaluator(tmp_path)
    source, _ = source_and_rig()
    source.session.task_evaluator = evaluator
    observation = source.begin_episode("Pick up the pool ball.")
    assert json.loads((tmp_path / "state.json").read_text())["task_text"] == observation.task_text
    from core.action_interface import action_result
    source.after_action(action_result("MV_DOWN", completed=True, timeout=False, requested={}, actual={}, message="completed"))
    private = json.loads((tmp_path / "state.json").read_text())
    assert private["action_step"] == source.agent_observation().action_step == 1
    assert private["snapshot_id"] == source.agent_observation().snapshot_id
    assert "task_success" not in json.dumps(source.agent_observation().metadata())
