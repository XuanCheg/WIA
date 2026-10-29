"""Adapt the verified R1 macros to the simulator-independent action contract."""
from __future__ import annotations

from core.action_interface import ActionExecutor, ActionResult, action_result

STOP_MESSAGES = {
    "base_motion_blocked": "Base made no sufficient progress; stopped. Inspect and choose a new direction or smaller motion.",
    "arm_joint_limit": "Arm target rejected at a joint limit; current command stopped. Inspect the new observation and choose an adjustment.",
    "ik_joint_jump": "Arm target rejected because the IK solution jumped; current command stopped. Inspect the new observation and choose an adjustment.",
    "ik_nonfinite": "Arm solution failed; stopped.",
    "ik_solver_failure": "Arm solution failed; stopped.",
    "trunk_height_limit": "Trunk height target outside the configured workspace; stopped.",
    "arm_step_limit": "Measured arm motion exceeded the configured per-step motion limit; stopped.",
    "arm_motion_blocked": "Arm motion blocked; stopped.",
    "empty_grasp": "No object held; stopped. Release and align before grasping again.",
    "grasp_lost": "Object no longer held; motion stopped.",
    "gripper_timeout": "Gripper did not settle; stopped.",
    "release_incomplete": "Release not confirmed; stopped.",
}


def r1_action_result(result: dict) -> ActionResult:
    """Add the uniform fields without dropping established diagnostic feedback.

    Each outer controller finalizes again after its hold/grasp checks, so a
    late grasp_lost cannot retain an earlier success=True.
    requested/actual contain robot feedback only; privileged debug stays in
    the existing explicitly named extra fields.
    """
    motion = result["motion"]
    if motion == "gripper":
        requested = {"state": result["requested_state"]}
        after = result["gripper_after"]
        actual = {key: after[key] for key in
                  ("gripper_width_m", "gripper_closed", "gripper_open", "holding")}
        result["holding"] = after["holding"]
    elif motion == "arm_translation":
        requested = {"delta_m": result["requested_m"], "delta_base_m": result.get("requested_delta_m")}
        actual = {"delta_m": result["actual_m"], "delta_base_m": result["actual_delta_m"]}
    elif motion == "trunk_translation":
        requested = {"delta_m": result["requested_m"], "delta_base_m": result["requested_delta_m"]}
        actual = {"delta_m": result["actual_m"], "delta_base_m": result["actual_delta_m"]}
    elif motion in ("rotation", "arm_rotation"):
        requested, actual = {"angle_deg": result["requested_deg"]}, {"angle_deg": result["actual_deg"]}
        if motion == "arm_rotation":
            requested["signed_angle_deg"] = result["signed_requested_deg"]
            actual["signed_angle_deg"] = result["signed_actual_deg"]
    elif motion == "translation":
        requested, actual = {"distance_m": result["requested_m"]}, {"distance_m": result["actual_m"]}
    else:
        raise ValueError(f"Unknown R1 result motion: {motion!r}.")
    details = {key: value for key, value in result.items() if key not in ActionResult.__annotations__}
    normalized = action_result(result["action"], completed=result["completed"], timeout=result["timeout"],
                               requested=requested, actual=actual,
                               message=STOP_MESSAGES.get(result["reason"], result["reason"]), **details)
    result.update(normalized)
    return result


def create_r1_executor(controller) -> ActionExecutor:
    """Capabilities come from the constructed controller, never a UI mode name."""
    profile = getattr(controller.session, 'reflection_sensor_profile', None)
    if isinstance(profile, dict) and profile.get('version') == 'reflection_sensor_profile_v3':
        from wia.runtime.sim.esi_reflection import measure_reflection
        measure_reflection(controller.session, controller=controller, finalize_initial=True)
    source = getattr(controller.session, "observation_source", None)
    if source is not None:
        source.controller_state_provider = controller.controller_state
    from wia.runtime.sim.esi_interaction_history import public_action_begin, public_action_end, rearrangement_evidence
    controller.geometric_rearrangement_evidence = lambda: rearrangement_evidence(controller)
    from wia.runtime.sim.esi_visual_evidence import install_visual_evidence
    visual_evidence = install_visual_evidence(controller.session, controller)
    def synchronized_visual_evidence(observation, bindings, parameters):
        if (getattr(controller.session, "action_executing", False)
                and (not getattr(controller.session, "_action_boundary_ready", False)
                     or source is None or source.boundary_observation is not observation
                     or getattr(source, "_captured_physical_generation", None)
                     != getattr(controller.session, "_physical_step_generation", 0))):
            raise RuntimeError("Visual evidence requires a fresh synchronized action-boundary observation.")
        return visual_evidence(observation, bindings, parameters)
    controller.session.task_visual_evidence = synchronized_visual_evidence
    task = getattr(controller.session, 'loaded_task', None)
    evaluation = getattr(getattr(getattr(task, 'definition', None), 'private', None), 'evaluation', None)
    if getattr(evaluation, 'parameters', {}).get('mapping_joint_world_contract'):
        from wia.runtime.sim.esi_mapping_world import install_mapping_world_identity
        install_mapping_world_identity(controller.session, controller, task)
    if getattr(evaluation, 'parameters', {}).get('curation_revision') == 'task_edit2_enclosure_additions_20261018_v1':
        from wia.runtime.sim.esi_enclosure_evidence import install_enclosure_evidence
        install_enclosure_evidence(controller.session, controller)
    def guard(action):
        phase = getattr(getattr(controller.session, 'temporal_phase_controller', None), 'phase', None)
        if phase == 'phase_1':
            from wia.tasks.temporal_policy import survey_action_allowed
            if not survey_action_allowed(action):
                raise ValueError('Phase 1 forbids object interaction.')
            state = controller.debug_gripper_state()
            if state.get('holding') or state.get('constraint_active'):
                raise ValueError('Phase 1 requires empty arms.')
            if controller.capabilities.right_gripper:
                with controller._select_arm("right"):
                    right = controller.debug_gripper_state()
                if right.get("holding") or right.get("constraint_active"):
                    raise ValueError("Phase 1 requires empty arms.")
    def execute(action, **kwargs):
        guard(action)
        public_action_begin(controller, action)
        safety = getattr(controller.session, 'safety_observer', None)
        if safety is not None:
            safety.begin_action(action)
        try:
            result = controller.execute(action, **kwargs)
            if safety is not None:
                safety.end_action(result)
            return result
        finally:
            public_action_end(controller)
            if safety is not None:
                safety.action = None
    def execute_metric(action, magnitude, **kwargs):
        guard(action)
        public_action_begin(controller, action)
        safety = getattr(controller.session, 'safety_observer', None)
        if safety is not None:
            safety.begin_action({'action': action, 'magnitude': magnitude})
        try:
            result = controller.execute_metric(action, magnitude, **kwargs)
            if safety is not None:
                safety.end_action(result)
            return result
        finally:
            public_action_end(controller)
            if safety is not None:
                safety.action = None
    executor = ActionExecutor(execute, capabilities=controller.capabilities,
                              execute_metric=execute_metric,
                              verify_stopped=controller.verify_stopped_result,
                              execution_context=getattr(controller.session, "action_execution", None),
                              on_result=source.after_action if source is not None else None)
    executor.controller_state_provider = controller.controller_state
    executor.controller = controller
    executor.performance = getattr(controller.session, "performance", None)
    from core.action_compiler import PrimitiveResolution
    base, arm = controller.config, getattr(controller, "arm_config", None)
    rotation, trunk = getattr(controller, "rotation_config", None), getattr(controller, "trunk_controller", None)
    executor.primitive_resolution = PrimitiveResolution(
        base_translation_m=base.translation_m, base_rotation_deg=base.rotation_deg,
        arm_translation_m=arm.translation_m if arm else None,
        arm_rotation_deg=rotation.rotation_deg if rotation else None,
        trunk_translation_m=trunk.config.translation_m if trunk else None)
    return executor
