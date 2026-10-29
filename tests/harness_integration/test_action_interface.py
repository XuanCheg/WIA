"""Backend-independent capabilities and shared fixed-action execution outcomes."""
import json
from dataclasses import replace

import pytest

from core.action_interface import ActionExecutor, BackendCapabilities, action_result
from core.action_units import (
    BASE_ATOMS, CANONICAL_ACTIONS, GRIPPER_ATOMS, MOVE_ATOMS, ROTATE_ATOMS, TERMINAL_ATOMS,
)
from wia.runtime.sim.esi_actions import create_r1_executor
from wia.runtime.sim.esi_rotation import RotationControlConfig
from wia.runtime.web.web_teleop.esi_control import ESIControlBackend
from harness_integration.test_esi_base import make_controller
from harness_integration.test_esi_gripper import make_manipulation_controller
from harness_integration.test_esi_preview import METADATA


def test_capabilities_express_fixed_mobile_and_partial_backends_without_simulator_names():
    fixed = BackendCapabilities(ee_translation=True, gripper=True)
    mobile = replace(fixed, base_translation=True, base_rotation=True)
    yaw_only = BackendCapabilities(base_rotation=True, ee_rotation=True)
    assert fixed.available_actions == MOVE_ATOMS + GRIPPER_ATOMS
    assert mobile.available_actions == BASE_ATOMS + MOVE_ATOMS + GRIPPER_ATOMS
    assert yaw_only.available_actions == BASE_ATOMS[4:] + ROTATE_ATOMS
    assert BackendCapabilities().available_actions == ()
    assert CANONICAL_ACTIONS == BASE_ATOMS + MOVE_ATOMS + GRIPPER_ATOMS + TERMINAL_ATOMS
    assert TERMINAL_ATOMS == ("STOP", "DONE")
    assert json.loads(json.dumps(mobile.to_dict())) == {
        "base_translation": True, "base_rotation": True, "ee_translation": True,
        "gripper": True, "ee_rotation": False, "trunk_height": False,
        "right_ee_translation": False, "right_ee_rotation": False, "right_gripper": False,
    }
    with pytest.raises(ValueError, match="booleans"):
        BackendCapabilities(gripper="false")


@pytest.mark.parametrize("action", [
    {"action": "MV_UP", "distance_m": 0.1}, ["MV_UP"], "MV_UP(0.1)", "MV_UP*2",
    "mv_up", "BASE_FWD", "ROTATE_CW", "RT_YAW_CW", "STOP", "DONE", None,
])
def test_invalid_or_unsupported_actions_never_reach_a_backend(action):
    calls = []
    executor = ActionExecutor(lambda *args, **kwargs: calls.append(args),
                              capabilities=BackendCapabilities(ee_translation=True))
    with pytest.raises(ValueError, match="Unsupported fixed action"):
        executor.execute(action)
    assert not calls


def test_generic_executor_passes_cancellation_and_preserves_backend_result():
    marker = object()
    calls = []
    result = action_result("MV_UP", completed=False, timeout=False,
                           requested={"delta_m": 0.02}, actual={"delta_m": 0.005},
                           message="cancelled", cancelled=True, backend_detail=marker)

    def execute(action, *, should_cancel):
        calls.append((action, should_cancel()))
        return result

    executor = ActionExecutor(execute, capabilities=BackendCapabilities(ee_translation=True))
    assert executor.execute("MV_UP", should_cancel=lambda: True) is result
    assert not result["success"] and calls == [("MV_UP", True)]
    assert result["backend_detail"] is marker


@pytest.mark.parametrize("patch", [
    {"action": "MV_DOWN"}, {"success": "true"}, {"success": False},
    {"requested": None}, {"actual": None}, {"message": None}, {"timeout": 0},
])
def test_backend_must_return_matching_and_consistent_action_result(patch):
    result = action_result("MV_UP", completed=True, timeout=False, requested={}, actual={}, message="completed")
    result.update(patch)
    executor = ActionExecutor(lambda *args, **kwargs: result,
                              capabilities=BackendCapabilities(ee_translation=True))
    with pytest.raises(RuntimeError, match="invalid ActionResult"):
        executor.execute("MV_UP")


def test_r1_result_fields_cover_all_existing_actions_and_late_failures():
    controller, robot, env = make_manipulation_controller()
    controller.rotation_config = RotationControlConfig.load()
    logged = []
    controller.on_result = logged.append
    executor = create_r1_executor(controller)
    assert executor.available_actions == BASE_ATOMS + MOVE_ATOMS + GRIPPER_ATOMS + ROTATE_ATOMS
    for action in ("BASE_FWD", "BASE_TURN_LEFT", "MV_UP", "ROTATE_CW", "GRASP", "GRASP", "RELEASE", "RELEASE"):
        result = executor.execute(action)
        assert result["success"] and result["completed"] and not result["timeout"]
        assert logged[-1] is result
        assert all(key in result for key in ("action", "success", "completed", "timeout", "requested", "actual", "message"))
        json.dumps(result, allow_nan=False)
        if action == "BASE_FWD":
            assert result["requested"] == {"distance_m": controller.config.translation_m}
            assert result["actual"]["distance_m"] == result["actual_m"]
        elif action == "MV_UP":
            assert result["requested"]["delta_m"] == controller.arm_config.translation_m
            assert result["actual"]["delta_m"] == result["actual_m"]
        elif action == "GRASP":
            assert result["holding"] and result["actual"]["holding"]
    env.available_object = False
    empty = executor.execute("GRASP")
    # GRASP completes a finite finger-close command. Actual attachment remains
    # a separate measured outcome, as in the native manipulation controller.
    assert empty["completed"] and empty["success"] and not empty["holding"]
    executor.execute("RELEASE")
    env.available_object = True
    executor.execute("GRASP")
    env.drop_on_step = len(env.actions) + 1
    lost = executor.execute("MV_UP")
    assert lost["reason"] == "grasp_lost" and not lost["success"] and not lost["completed"]
    assert logged[-1] is lost and not lost["gripper_after"]["holding"]


def test_timeout_and_immediate_cancellation_are_action_failures_with_measurements():
    c, robot, env = make_controller(timeout_sim_s=0.1)
    env.response = 0
    executor = create_r1_executor(c)
    timeout = executor.execute("BASE_FWD")
    assert timeout["timeout"] and not timeout["success"] and not timeout["completed"]
    assert timeout["actual"] == {"distance_m": 0.0}
    cancelled = executor.execute("BASE_BACK", should_cancel=lambda: True)
    assert cancelled["cancelled"] and not cancelled["success"] and not cancelled["timeout"]
    assert cancelled["requested"] == {"distance_m": c.config.translation_m}


def test_gumi_uses_executor_capabilities_even_when_presentation_metadata_disagrees():
    calls = []
    cap = BackendCapabilities(ee_translation=True, gripper=True)

    def execute(action, **kwargs):
        calls.append(action)
        return action_result(action, completed=True, timeout=False, requested={}, actual={}, message="completed")

    executor = ActionExecutor(execute, capabilities=cap)
    backend = ESIControlBackend(dict(METADATA, robot_type="Franka", rotation_control={"enabled": True}), executor)
    assert backend.executor is executor
    assert backend.capabilities is cap
    assert backend.state()["step_tokens"] == list(executor.available_actions)
    assert backend.state()["action_groups"] == {"base": False, "arm": True, "gripper": True}
    assert not backend.rotation_enabled and not backend.action_allowed("ROTATE_CW")
    assert backend.submit("move", "BASE_FWD")["status"] == 405 and not calls
