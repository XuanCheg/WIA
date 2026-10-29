"""Check conservative diagnostic conclusions on CPU.

Run: .venv/bin/python -m pytest -q tests/test_preflight_native_semantics.py
"""
from types import SimpleNamespace

import numpy as np
import pytest

from wia.runtime.preflight.goals import validate_goal
from wia.runtime.preflight.models import Options
from wia.runtime.preflight.physics import check_object_contacts
from wia.runtime.preflight.probes import ProbeRunner, local_grasp_candidates


def test_local_grasp_candidates_do_not_reject_a_narrow_neck_by_global_width():
    def box(low, high):
        return np.array([[x, y, z] for x in (low[0], high[0])
                         for y in (low[1], high[1]) for z in (low[2], high[2])])
    obj = SimpleNamespace(aabb=(np.array([-.1, -.1, 0]), np.array([.1, .1, .2])), links={
        "body": SimpleNamespace(collision_boundary_points_world=box([-.1, -.1, 0], [.1, .1, .12])),
        "neck": SimpleNamespace(collision_boundary_points_world=box([-.02, -.02, .12], [.02, .02, .2]))})
    controller = SimpleNamespace(open_target=np.array([.05, .05]),
        ee_pose=lambda: {"world_quaternion_xyzw": [0., 0., 0., 1.]})
    candidates, geometry = local_grasp_candidates(obj, controller, Options())
    assert geometry["object_extent_m"][0] > geometry["maximum_joint_opening_m"]
    assert candidates and all(row["local_width_m"] < .1 for row in candidates)
    assert len({row["yaw_delta_deg"] for row in candidates[:4]}) == 4


@pytest.mark.parametrize("other_min, expected", [(.05, "INCONCLUSIVE"), (.2, "PASS")])
def test_object_bounds_do_not_turn_overlap_into_a_collision_proof(other_min, expected):
    target = SimpleNamespace(name="target", prim_path="/World/target", category="fruit", visual_only=False,
        aabb=(np.zeros(3), np.full(3, .1)), contact_list=lambda: [])
    furniture = SimpleNamespace(name="furniture", category="table", visual_only=False,
        aabb=(np.full(3, other_min), np.full(3, other_min + .1)))
    context = SimpleNamespace(targets=("target",), containers=(), object_supports={})
    session = SimpleNamespace(robot=object(), env=SimpleNamespace(scene=SimpleNamespace(objects=[target, furniture])))
    emitted = []
    check_object_contacts(context, session, {"target": target}, emitted.append)
    assert len(emitted) == 1 and emitted[0].status == expected


def test_finite_failed_motion_and_grasp_attempts_remain_inconclusive():
    runner = object.__new__(ProbeRunner)
    runner.context = SimpleNamespace(goal="grasp_lift")
    runner.options = Options(mode="full")
    emitted = []
    runner.emit = emitted.append
    runner.publish("target", {"attempts": [{"failure_reason": "ik_joint_jump", "approach_verified": False}],
                              "geometry": {"generated_candidate_count": 1}, "grasp_lift_verified": False})
    assert {check.group for check in emitted} == {"manipulation", "grasp"}
    assert all(check.status == "INCONCLUSIVE" for check in emitted)


def test_unknown_goal_is_not_inferred_from_a_successful_grasp():
    emitted = []
    context = SimpleNamespace(goal="future_stack", evaluation={"type": "future_stack"})
    validate_goal(context, None, Options(), {"target": {"grasp_lift_verified": True}}, emitted.append)
    assert len(emitted) == 1 and emitted[0].status == "INCONCLUSIVE"


def test_open_goal_requires_existing_evaluator_and_stable_holding():
    emitted = []
    context = SimpleNamespace(goal="open_extract_lift", targets=("target",), dependencies=())
    partial = {"target": {"attempts": [{"grasp_lift_verified": True,
                 "goal_evaluation": {"task_success": False, "criteria": {"door_opened": False}}}]}}
    validate_goal(context, None, Options(), partial, emitted.append)
    assert emitted[-1].status == "INCONCLUSIVE"
    partial["target"]["attempts"][0]["goal_evaluation"]["task_success"] = True
    validate_goal(context, None, Options(), partial, emitted.append)
    assert emitted[-1].status == "PASS" and emitted[-1].verification_level == "goal_verified"
