"""Explicit goal plugins, never text-inferred success. Run preflight_esi_tasks.py --mode full."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from wia.runtime.sim.esi_omnigibson import _array
from wia.runtime.preflight.models import Check
from wia.runtime.preflight.physics import object_state

_VALIDATORS = {}


def register_goal(name, validator):
    """Register validator(context, session, options, probe_results, emit)."""
    if name in _VALIDATORS:
        raise ValueError("Goal validator already exists: " + name)
    _VALIDATORS[name] = validator


def validate_goal(context, session, options, results, emit):
    validator = _VALIDATORS.get(context.goal)
    if validator is None:
        emit(Check("goal", "unsupported_goal", "INCONCLUSIVE", "No physical validator is registered for this goal.",
            evidence={"goal": context.goal, "evaluation": context.evaluation}))
        return
    validator(context, session, options, results, emit)


def _lift_goal(context, session, options, results, emit):
    for key in context.targets:
        attempts = results.get(key, {}).get("attempts", [])
        verified = [row for row in attempts if row.get("grasp_lift_verified")
                    and row.get("goal_evaluation", {}).get("task_success") is True and not row.get("collisions")]
        emit(Check("goal", "goal." + key, "PASS" if verified else "INCONCLUSIVE",
            "Existing native task evaluator passed after correct-object lift and stable hold." if verified else
            "The existing evaluator has not verified the final goal in a complete physical attempt.",
            [key], {"goal": context.goal, "attempt_count": len(attempts),
                    "evaluations": [row.get("goal_evaluation") for row in attempts],
                    "dependencies": list(context.dependencies)},
            verification_level="goal_verified" if verified else "native_initialization"))


def _placement_goal(context, session, options, results, emit):
    for key in context.targets:
        obj = session.env.scene.object_registry("name", context.bindings[key].name)
        target_extent = _array(obj.aabb[1]) - _array(obj.aabb[0])
        screens = []
        for container_key in context.containers:
            container = session.env.scene.object_registry("name", context.bindings[container_key].name)
            if container is None:
                continue
            extent = _array(container.aabb[1]) - _array(container.aabb[0])
            screens.append({"container": container_key, "outer_extent_m": extent.tolist(),
                            "target_extent_m": target_extent.tolist(),
                            "sorted_outer_dimensions_fit": bool(np.all(np.sort(target_extent) < np.sort(extent))),
                            "interior_or_insertion_certified": False})
        attempts = results.get(key, {}).get("attempts", [])
        placements = [row["placement"] for row in attempts if row.get("placement")]
        # The current task loader explicitly has no answer_and_place evaluator.
        # Native Inside evidence is retained, not substituted for original QA rules.
        emit(Check("goal", "goal." + key, "INCONCLUSIVE",
            "Geometry and released-state probes are necessary evidence; the original answer-and-place success contract has no configured final evaluator.",
            [key, *context.containers], {"geometry_screens": screens, "placement_probes": placements,
                "allow_remove_after_verification": True, "all_targets_enumerated": list(context.targets),
                "complete_goal_verified": False}, "configure_original_protocol_compatible_placement_evaluator",
            "physical_manipulation" if placements else "geometric_screen"))


def open_articulation(probe):
    """Try real handle motions using the declared joint and native USD geometry."""
    from wia.runtime.preflight.probes import ProbeStopped

    context, session, controller = probe.context, probe.session, probe.controller
    key = context.evaluation["container"]
    box = session.env.scene.object_registry("name", context.bindings[key].name)
    sliding = context.goal == "slide_extract_lift"
    joint_name = context.evaluation.get("slide_joint", "lid_slide") if sliding else context.evaluation.get("hinge_joint", "lid_hinge")
    joint = box.joints[joint_name]
    handles = [(link, mesh) for link in box.links.values() for mesh in link.collision_meshes.values()
               if mesh.prim.GetName() == "collision_handle_bar"]
    if not handles:
        raise ProbeStopped("No declared-asset handle collision bar is available for the bounded articulation strategy.")
    link, mesh = handles[0]
    handle = (_array(mesh.aabb[0]) + _array(mesh.aabb[1])) / 2
    probe.allowed_object = box
    probe.move("RELEASE")
    length = float(np.mean(list(session.robot.eef_to_fingertip_lengths[controller.working_arm].values())))
    tip_offset = Rotation.from_quat(controller.ee_pose()["world_quaternion_xyzw"]).apply([0., 0., length])
    tcp = handle - tip_offset
    safe = tcp.copy()
    safe[2] = max(tcp[2] + .05, controller.ee_pose()["world_position"][2])
    probe.align(safe, axes=(2, 0, 1))
    probe.align(tcp)
    probe.move("GRASP")
    attachment = session.robot._ag_obj_constraint_params[controller.working_arm]
    if attachment.get("ag_link_prim_path") != link.prim_path:
        raise ProbeStopped("Native grasp did not attach to the intended articulation handle.")
    threshold = (context.evaluation.get("minimum_open_displacement_m", .2) if sliding
                 else np.deg2rad(context.evaluation.get("minimum_open_angle_deg", 60.)))
    prim = joint.prim
    body_path = str(prim.GetRelationship("physics:body0").GetTargets()[0])
    body = next((item for item in box.links.values() if item.prim_path == body_path), box.root_link)
    body_pos, body_quat = (_array(x) for x in body.get_position_orientation())
    local = prim.GetAttribute("physics:localRot0").Get()
    local_rotation = Rotation.from_quat([*local.GetImaginary(), local.GetReal()])
    orientation = Rotation.from_quat(body_quat)
    axis = orientation.apply(local_rotation.apply(np.eye(3)["XYZ".index(prim.GetAttribute("physics:axis").Get())]))
    pivot = body_pos + orientation.apply(np.asarray(prim.GetAttribute("physics:localPos0").Get()))
    initial = float(_array(joint.get_state()[0]).reshape(-1)[0])
    ee = np.asarray(controller.ee_pose()["world_position"])
    for _ in range(48):
        value = float(_array(joint.get_state()[0]).reshape(-1)[0])
        if value >= threshold:
            break
        step = min(.025 if sliding else np.deg2rad(5), threshold + (.01 if sliding else .03) - value)
        point = (np.asarray(controller.ee_pose()["world_position"]) + axis * step if sliding
                 else pivot + Rotation.from_rotvec(axis * step).apply(np.asarray(controller.ee_pose()["world_position"]) - pivot))
        probe.align(point, axes=(2, 0, 1))
    final = float(_array(joint.get_state()[0]).reshape(-1)[0])
    probe.move("RELEASE")
    if final < threshold:
        raise ProbeStopped("Bounded real handle motions did not reach the required joint opening.")
    return {"joint": joint_name, "initial_position": initial, "final_position": final,
            "threshold": float(threshold), "unit": "m" if sliding else "rad",
            "handle_link": link.prim_path, "initial_ee": ee.tolist(), "real_robot_motion": True}


def attempt_placement(probe, target):
    """Probe insertion and native released-state relations, without defining QA success."""
    from wia.runtime.preflight.probes import ProbeStopped
    from omnigibson.object_states import Inside

    rows = []
    extent = _array(target.aabb[1]) - _array(target.aabb[0])
    candidates = []
    for key in probe.context.containers:
        obj = probe.session.env.scene.object_registry("name", probe.context.bindings[key].name)
        low, high = (_array(v) for v in obj.aabb)
        if np.all(np.sort(extent) < np.sort(high - low)):
            candidates.append((float(np.prod(high-low)), key, obj))
    for _, key, container in sorted(candidates, key=lambda row: row[0]):
        row = {"container": key, "released_inside_stable": False}
        rows.append(row)
        # One successful grasp is consumed by the first release; additional
        # containers require a fresh isolated grasp attempt.
        try:
            low, high = (_array(v) for v in container.aabb)
            # Target-container contact is intended during insertion. Robot
            # contacts and contacts with all other objects remain monitored.
            probe.destination_object = container
            offset = np.asarray(probe.controller.ee_pose()["world_position"]) - _array(target.get_position_orientation()[0])
            centre = (low + high) / 2
            above = centre.copy()
            above[2] = high[2] + extent[2] / 2 + .04
            probe.align(above + offset, axes=(2, 0, 1))
            # Slow insertion keeps all robot contacts monitored. This is a
            # candidate insertion pose; dimensions alone never pass the goal.
            centre[2] = low[2] + extent[2] / 2 + .02
            probe.align(centre + offset)
            probe.move("RELEASE")
            probe.carried_object = None
            probe.hold()
            state = object_state(target)
            relation = bool(target.states[Inside].get_value(container)) if Inside in target.states else None
            held = probe.controller.debug_gripper_state()
            released = held["object_name"] is None and not held["constraint_active"]
            stable = state["linear_speed_m_s"] <= .02 and state["angular_speed_rad_s"] <= .1
            row.update(native_inside=relation, released=bool(released), stable=stable, target=state,
                       released_inside_stable=bool(relation and released and stable))
        except (ProbeStopped, ValueError, RuntimeError, KeyError) as error:
            row["failure_reason"] = f"{type(error).__name__}: {error}"
        finally:
            probe.destination_object = None
        break
    return {"candidates": len(candidates), "attempts": rows, "original_qa_goal_verified": False}


for _goal in ("grasp_lift", "open_extract_lift", "slide_extract_lift"):
    register_goal(_goal, _lift_goal)
register_goal("answer_and_place", _placement_goal)
