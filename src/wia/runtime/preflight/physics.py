"""Native physical necessary-condition checks. Run preflight_esi_tasks.py --mode full."""
from __future__ import annotations

import numpy as np

from wia.runtime.sim.esi_omnigibson import _array
from wia.runtime.preflight.models import Check


def object_state(obj):
    position, quaternion = (_array(v) for v in obj.get_position_orientation())
    low, high = (_array(v) for v in obj.aabb)
    mass = None if obj.fixed_base else float(obj.root_link.mass)
    return {"name": obj.name, "category": obj.category, "position": position.tolist(),
            "quaternion_xyzw": quaternion.tolist(), "aabb_min": low.tolist(), "aabb_max": high.tolist(),
            "extent_m": (high - low).tolist(), "mass_kg": mass, "fixed_base": bool(obj.fixed_base),
            "visual_only": bool(obj.visual_only),
            "linear_speed_m_s": 0. if obj.fixed_base else float(np.linalg.norm(_array(obj.get_linear_velocity()))),
            "angular_speed_rad_s": 0. if obj.fixed_base else float(np.linalg.norm(_array(obj.get_angular_velocity())))}


def external_contacts(session, *, allowed=()):
    """Read native robot contacts, excluding only floor and explicit grasp surfaces."""
    robot = session.robot
    ignored = [obj.prim_path for obj in session.env.scene.objects
               if obj.category in ("floor", "floors") or obj.name in allowed]
    rows = []
    for contact in robot.contact_list():
        bodies = [str(contact.body0), str(contact.body1)]
        external = [p for p in bodies if not (p == robot.prim_path or p.startswith(robot.prim_path + "/"))]
        if not external or all("groundPlane" in p or any(p == root or p.startswith(root + "/")
                                                       for root in ignored) for p in external):
            continue
        row = {"body0": bodies[0], "body1": bodies[1]}
        if hasattr(contact, "position"):
            row["position"] = _array(contact.position).tolist()
        rows.append(row)
    return rows


def check_object_contacts(context, session, bound, emit):
    """Expose native support contacts and conservative inter-object overlap evidence."""
    obstacles = [(obj, *(_array(value) for value in obj.aabb)) for obj in session.env.scene.objects
                 if obj is not session.robot and not obj.visual_only
                 and obj.category not in ("floor", "floors", "ceiling", "ceilings")]
    for key in dict.fromkeys(context.targets + context.containers):
        obj = bound.get(key)
        if obj is None:
            continue
        contacts = [{"body0": str(contact.body0), "body1": str(contact.body1)} for contact in obj.contact_list()]
        support_key = context.object_supports.get(key)
        support = bound.get(support_key)
        if support is not None:
            supported = any(row[side] == support.prim_path or row[side].startswith(support.prim_path + "/")
                            for row in contacts for side in ("body0", "body1"))
            emit(Check("physics", "support_contact." + key, "PASS" if supported else "INCONCLUSIVE",
                "Native contact with the declared support is present." if supported else
                "No declared-support contact was observed; inspect settling and the existing support screen.",
                [key, support_key], {"contacts": contacts}, verification_level="native_initialization"))
        low, high = (_array(value) for value in obj.aabb)
        overlaps = []
        for other, other_low, other_high in obstacles:
            if other is obj:
                continue
            penetration = np.minimum(high, other_high) - np.maximum(low, other_low)
            if np.all(penetration > .003):
                overlaps.append({"object": other.name, "aabb_overlap_m": penetration.tolist()})
        emit(Check("physics", "object_overlap." + key, "INCONCLUSIVE" if overlaps else "PASS",
            "Object bounds overlap; mesh penetration, intended containment and empty furniture space are not distinguished by this screen." if overlaps else
            "No non-floor object bounds overlap by more than the screening tolerance on all axes.",
            [key], {"tolerance_m": .003, "candidates": overlaps, "native_contacts": contacts,
                    "scope": "conservative object bounds; no mesh penetration depth is available from native contact records"},
            verification_level="geometric_screen"))


def check_physics(context, session, controller, executor, adapter, output, options, emit):
    """Keep conservative qualification rejections distinct from proved violations."""
    level = "native_initialization"
    bound, before = {}, {}
    for key, binding in context.bindings.items():
        obj = session.env.scene.object_registry("name", binding.name)
        mismatch = obj is None or any(getattr(binding, field, None) is not None
            and getattr(obj, field, None) != getattr(binding, field) for field in ("category", "model"))
        if mismatch:
            emit(Check("physics", "binding." + key, "FAIL", "Native object is missing or its category/model differs from the binding.",
                [key], {"expected": binding.model_dump(mode="json"), "present": obj is not None},
                "correct_object_binding", level))
            continue
        bound[key] = obj
        try:
            state = object_state(obj)
            numbers = state["position"] + state["quaternion_xyzw"] + state["extent_m"]
            valid = np.isfinite(numbers).all() and min(state["extent_m"]) > 0 and abs(np.linalg.norm(state["quaternion_xyzw"]) - 1) < .01
            if not obj.fixed_base:
                valid = valid and np.isfinite(state["mass_kg"]) and state["mass_kg"] > 0
            movable = key not in context.targets or context.kind == "scene" or not (obj.fixed_base or obj.visual_only)
            emit(Check("physics", "object." + key, "PASS" if valid and movable else "FAIL",
                "Native pose, collision extent, mass and required mobility are valid." if valid and movable
                else "Native object has invalid geometry/mass/pose or a physical target is fixed/visual-only.",
                [key], state, None if valid and movable else "inspect_object_physics", level))
            before[key] = state
        except (ValueError, RuntimeError, AttributeError) as error:
            emit(Check("physics", "object." + key, "INCONCLUSIVE", "Native physical readback failed: " + str(error), [key], verification_level=level))
    initial_contacts = external_contacts(session)
    for _ in range(options.settle_steps):
        controller.step_base_debug()
    final_contacts = external_contacts(session)
    initial_pairs = {tuple(sorted((c["body0"], c["body1"]))) for c in initial_contacts}
    repeated = [c for c in final_contacts if tuple(sorted((c["body0"], c["body1"]))) in initial_pairs]
    emit(Check("physics", "robot_initial_collision", "FAIL" if repeated else "INCONCLUSIVE" if final_contacts else "PASS",
        "Repeated non-floor robot contact at the unchanged initial pose." if repeated else
        "Transient contact requires further inspection." if final_contacts else "No native external non-floor contact after settling.",
        evidence={"before": initial_contacts, "after": final_contacts, "repeated": repeated,
                  "settle_steps": options.settle_steps}, recommended_action="adjust_initial_robot_pose" if repeated else None,
        verification_level=level))
    q = _array(session.robot.get_joint_positions())
    low, high = (_array(v) for v in session.robot.control_limits["position"])
    invalid = np.where(~np.isfinite(q) | (q < low - .005) | (q > high + .005))[0]
    emit(Check("physics", "robot_joint_limits", "FAIL" if len(invalid) else "PASS",
        "Native robot joints exceed their position limits." if len(invalid) else "Native robot joints are within position limits.",
        evidence={"invalid_indices": invalid.tolist(), "positions": q.tolist(), "tolerance": .005}, verification_level=level))
    if context.goal in ("grasp_lift", "open_extract_lift", "slide_extract_lift"):
        initial_gripper = controller.debug_gripper_state()
        initially_held = initial_gripper["object_name"] in {context.bindings[k].name for k in context.targets}
        emit(Check("physics", "initial_goal_state", "FAIL" if initially_held else "PASS",
            "Target is already held, violating the existing evaluator's initially-unheld precondition." if initially_held else
            "Target is initially unheld; the relative lift goal is not already satisfied.",
            list(context.targets), {"gripper": initial_gripper}, verification_level=level))
    elif context.goal is not None:
        emit(Check("physics", "initial_goal_state", "INCONCLUSIVE",
            "The task has no configured native final-state evaluator for detecting accidental initial completion.",
            list(context.targets), {"goal": context.goal}, verification_level=level))
    config = context.scene.definition.qualification
    max_linear = getattr(config, "maximum_linear_speed_m_s", .02)
    max_angular = getattr(config, "maximum_angular_speed_rad_s", .1)
    for key, old in before.items():
        obj = bound[key]
        state = object_state(obj)
        displacement = float(np.linalg.norm(np.asarray(state["position"]) - old["position"]))
        stable = state["linear_speed_m_s"] <= max_linear and state["angular_speed_rad_s"] <= max_angular and displacement <= .02
        emit(Check("physics", "settling." + key, "PASS" if stable else "INCONCLUSIVE",
            "Object remains stable during the bounded settling window." if stable else "Object did not settle within the bounded window.",
            [key], {"before": old, "after": state, "displacement_m": displacement, "maximum_displacement_m": .02,
                    "maximum_linear_speed_m_s": max_linear, "maximum_angular_speed_rad_s": max_angular},
            None if stable else "inspect_support_and_contacts", level))
    check_object_contacts(context, session, bound, emit)
    if context.goal in ("open_extract_lift", "slide_extract_lift"):
        name = context.evaluation.get("hinge_joint", "lid_hinge") if context.goal == "open_extract_lift" else context.evaluation.get("slide_joint", "lid_slide")
        key = context.evaluation["container"]
        obj = bound.get(key)
        joint = obj.joints.get(name) if obj is not None else None
        valid = joint is not None and joint.upper_limit > joint.lower_limit
        evidence = {"joint": name}
        if joint is not None:
            evidence.update(lower=float(joint.lower_limit), upper=float(joint.upper_limit),
                            position=_array(joint.get_state()[0]).tolist(), type=str(joint.joint_type))
        emit(Check("physics", "required_articulation", "PASS" if valid else "FAIL",
            "Required articulation exists with a nonempty range; movement is checked separately." if valid
            else "Required articulated joint is absent or locked by its limits.", [key], evidence, verification_level=level))
    observation = session.observation_source.agent_observation()
    result = adapter.qualify(session, executor, output, observation=observation)
    # Existing qualification deliberately rejects broad AABB/visibility screens.
    # Preserve its evidence, but do not turn those screens into impossibility claims.
    definite = {"source_scene_mismatch", "private_binding_mismatch", "object_binding_mismatch",
                "target_not_movable_collidable", "primitive_geometry_failed", "unsupported_robot_capabilities"}
    status = "PASS" if result.status != "skipped" else "FAIL" if result.reason in definite else "INCONCLUSIVE"
    emit(Check("physics", "native_qualification", status,
        "Existing scene qualification accepted the initialized world." if status == "PASS" else
        "Existing scene qualification rejected a precondition: " + str(result.reason),
        evidence={"qualification_status": result.status, "reason": result.reason, "checks": result.checks},
        recommended_action=None if status == "PASS" else "inspect_qualification_evidence", verification_level=level))
    return bound
