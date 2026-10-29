"""Bounded, isolated physical probes. Run preflight_esi_tasks.py --mode full."""
from __future__ import annotations

from copy import deepcopy
import json
import time

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from core.action_compiler import ParameterizedAction
from wia.runtime.sim.esi_omnigibson import _array
from wia.runtime.preflight.models import Check
from wia.runtime.preflight.physics import external_contacts, object_state


class ProbeStopped(RuntimeError):
    """A bounded attempt ended; this is never an impossibility certificate."""


class WorldSnapshot:
    """Restore native physics, controller holds, attachments and diagnostic history."""

    def __init__(self, session, controller, context):
        self.session, self.controller, self.context = session, controller, context
        self.state = deepcopy(session.pipeline.og.sim.dump_state(serialized=False))
        self.hold = controller.hold_action.copy()
        self.inactive = deepcopy(controller.inactive_targets)
        self.on_step = controller.on_step
        self.gripper_command = controller.gripper_command
        self.arm_target = deepcopy(getattr(controller, "_arm_target_pose", None))
        self.joints = _array(session.robot.get_joint_positions()).copy()
        self.poses = {obj.name: tuple(_array(x).copy() for x in obj.get_position_orientation())
                      for obj in session.env.scene.objects}
        self.restores = 0

    def restore(self):
        session, controller = self.session, self.controller
        controller.on_step = self.on_step
        session.task_evaluator = None
        session.pipeline.og.sim.load_state(deepcopy(self.state), serialized=False)
        controller.hold_action[:] = self.hold
        controller.inactive_targets = deepcopy(self.inactive)
        controller.gripper_command = self.gripper_command
        controller._arm_target_pose = deepcopy(self.arm_target)
        controller.ik_guard.reset()
        controller._active_control = {"execution_status": "idle", "action": None}
        controller._last_control_result = None
        session.invalidate_camera_cache()
        max_position = max_rotation = 0.
        for name, (position, quaternion) in self.poses.items():
            obj = session.env.scene.object_registry("name", name)
            actual, quat = (_array(x) for x in obj.get_position_orientation())
            max_position = max(max_position, float(np.linalg.norm(actual - position)))
            max_rotation = max(max_rotation, float((Rotation.from_quat(quaternion).inv() * Rotation.from_quat(quat)).magnitude()))
        joint_error = float(np.max(np.abs(_array(session.robot.get_joint_positions()) - self.joints)))
        held = controller.debug_gripper_state()
        clean = max_position < .0001 and max_rotation < .001 and joint_error < .0001 and held["object_name"] is None and not held["constraint_active"]
        evidence = {"max_object_position_error_m": max_position, "max_object_rotation_error_rad": max_rotation,
                    "max_robot_joint_error_rad_or_m": joint_error, "gripper": held,
                    "physics_objects_checked": len(self.poses), "restored": clean}
        if not clean:
            raise RuntimeError("Native state restoration failed; remaining destructive probes are unsafe: " + json.dumps(evidence))
        session.observation_source.begin_episode(self.context.loaded.task_text)
        self.restores += 1
        return evidence


def local_grasp_candidates(obj, controller, options):
    """Intersect native collision hulls with jaw lines; widths are only candidates."""
    from wia.runtime.commands.plan_esi_action_order_offline import section_intervals

    hulls = []
    for link in obj.links.values():
        values = link.collision_boundary_points_world
        if values is None:
            continue
        points = _array(values)
        if len(points) < 4 or not np.isfinite(points).all():
            continue
        try:
            hull = ConvexHull(points)
        except QhullError:
            continue
        hulls.append({"planes": hull.equations, "points": points})
    width = float(np.sum(controller.open_target))
    low, high = (_array(x) for x in obj.aabb)
    centre = (low + high) / 2
    initial = Rotation.from_quat(controller.ee_pose()["world_quaternion_xyzw"])
    candidates = []
    for fraction in (.8, .5, .95):
        for angle in options.directions_deg:
            orientation = initial * Rotation.from_euler("z", angle, degrees=True)
            jaw = orientation.apply([0., 1., 0.])
            origin = centre.copy()
            origin[2] = low[2] + fraction * (high[2] - low[2])
            tangent = orientation.apply([1., 0., 0.])
            radius = float(np.abs(tangent) @ ((high - low) / 2))
            for offset in (0., .94 * radius, -.94 * radius):
                section = origin + tangent * offset
                for begin, end in section_intervals(hulls, section, jaw):
                    span = end - begin
                    if .003 <= span <= width - .002:
                        candidates.append({"point": (section + jaw * (begin + end) / 2).tolist(),
                            "yaw_delta_deg": float((angle + 180) % 360 - 180), "local_width_m": float(span),
                            "quaternion_xyzw": orientation.as_quat().tolist(), "height_fraction": fraction,
                            "tangent_offset_m": offset})
    # Interleave directions so a small budget is not consumed by one rim line.
    by_direction = {float((angle + 180) % 360 - 180): [] for angle in options.directions_deg}
    for candidate in candidates:
        by_direction[candidate["yaw_delta_deg"]].append(candidate)
    ordered = [rows[index] for index in range(max(map(len, by_direction.values()), default=0))
               for rows in by_direction.values() if index < len(rows)]
    return ordered[:options.candidate_count], {
        "native_collision_hull_count": len(hulls), "generated_candidate_count": len(candidates),
        "configured_candidate_budget": options.candidate_count, "maximum_joint_opening_m": width,
        "object_extent_m": (high - low).tolist(),
        "scope": "Local line sections of native link collision hulls; finite-width fingers and holding require physical probes."}


class ProbeRunner:
    def __init__(self, context, session, controller, executor, adapter, options, output, navigation, emit, perception=None):
        self.context, self.session, self.controller, self.executor = context, session, controller, executor
        self.adapter, self.options, self.output, self.navigation = adapter, options, output, navigation
        self.emit, self.perception = emit, perception
        self.snapshot = WorldSnapshot(session, controller, context)
        self.allowed_object = None
        self.carried_object = None
        self.destination_object = None
        self.departure_supports = ()
        self.departure_height = None
        self.actions, self.collisions = [], []
        self.deadline = float("inf")

    def check_budget(self):
        if time.monotonic() >= self.deadline:
            raise ProbeStopped("Per-target search wall budget exhausted.")

    def _contacts(self):
        rows = external_contacts(self.session)
        if self.allowed_object is not None:
            root = self.allowed_object.prim_path
            fingers = [link.prim_path for link in self.session.robot.finger_links[self.controller.working_arm]]
            rows = [row for row in rows if not (any(row[side] == root or row[side].startswith(root + "/") for side in ("body0", "body1"))
                    and any(row[side] == path or row[side].startswith(path + "/") for side in ("body0", "body1") for path in fingers))]
        carried = self.carried_object
        if carried is not None:
            elevated = float(_array(carried.aabb[0])[2]) > self.departure_height + .01
            for contact in carried.contact_list():
                paths = [str(contact.body0), str(contact.body1)]
                other = [p for p in paths if not p.startswith(carried.prim_path + "/") and p != carried.prim_path]
                for path in other:
                    if path.startswith(self.session.robot.prim_path + "/"):
                        continue
                    destination = self.destination_object
                    if destination is not None and (path == destination.prim_path or path.startswith(destination.prim_path + "/")):
                        continue
                    if not elevated and any(path == root or path.startswith(root + "/") for root in self.departure_supports):
                        continue
                    rows.append({"body0": paths[0], "body1": paths[1], "carried_object": carried.name})
        return rows

    def move(self, token, magnitude=None):
        self.check_budget()
        value = {"action": token}
        if magnitude is not None:
            value["angle_deg" if "TURN" in token or "ROT" in token else "distance_m"] = float(magnitude)
        current_contacts = []
        previous_step = self.controller.on_step

        def monitor():
            if previous_step:
                previous_step()
            current_contacts.extend(self._contacts())

        self.controller.on_step = monitor
        try:
            result = self.executor.execute_macro(ParameterizedAction(**value),
                resolution=self.executor.primitive_resolution, limits=self.executor.macro_limits,
                should_cancel=lambda: time.monotonic() >= self.deadline or bool(current_contacts))
        finally:
            self.controller.on_step = previous_step
        row = {"command": value, "result": result, "contacts": current_contacts,
               "ik_guard": self.controller.ik_guard.diagnostics(), "gripper": self.controller.debug_gripper_state()}
        self.actions.append(row)
        self.collisions.extend(current_contacts)
        with (self.output / "probe_actions.jsonl").open("a") as stream:
            stream.write(json.dumps(row, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
        if not result["success"] or current_contacts:
            if not current_contacts and time.monotonic() >= self.deadline:
                raise ProbeStopped("Per-target search wall budget exhausted during guarded motion.")
            raise ProbeStopped("Unexpected native contact." if current_contacts else "Guarded motion stopped: " + str(result.get("reason")))
        return row

    def align(self, world_point, *, axes=(0, 1, 2)):
        for axis in axes:
            for _ in range(32):
                self.check_budget()
                base = self.controller.pose()
                desired = Rotation.from_quat(base["quaternion_xyzw"]).inv().apply(np.asarray(world_point) - base["position"])
                delta = float(desired[axis] - self.controller.ee_pose()["base_position"][axis])
                if abs(delta) < .011:
                    break
                token = (("MV_FWD", "MV_BACK"), ("MV_LEFT", "MV_RIGHT"), ("MV_UP", "MV_DOWN"))[axis][delta < 0]
                self.move(token, min(.08, abs(delta)))
            else:
                raise ProbeStopped("Bounded Cartesian alignment did not converge.")

    def follow_route(self, candidate):
        path = candidate["route_xy"]
        # Preserve each grid corner. Only remove collinear intermediate points.
        route = [p for i, p in enumerate(path) if i == len(path)-1 or i > 0 and
                 not np.allclose(np.asarray(p)-path[i-1], np.asarray(path[i+1])-p)]
        for point in route:
            for _ in range(48):
                base = self.controller.pose()
                error = Rotation.from_euler("z", -base["yaw_rad"]).apply([point[0]-base["position"][0], point[1]-base["position"][1], 0])
                if np.linalg.norm(error[:2]) < .035:
                    break
                axis = int(np.argmax(np.abs(error[:2])))
                self.move((("BASE_FWD", "BASE_BACK"), ("BASE_LEFT", "BASE_RIGHT"))[axis][error[axis] < 0],
                          min(.15, abs(float(error[axis]))))
            else:
                raise ProbeStopped("Bounded navigation replay did not converge.")
        yaw = np.degrees((candidate["yaw_rad"] - self.controller.pose()["yaw_rad"] + np.pi) % (2*np.pi) - np.pi)
        while abs(yaw) >= 2.5:
            self.move("BASE_TURN_LEFT" if yaw > 0 else "BASE_TURN_RIGHT", min(30., abs(yaw)))
            yaw = np.degrees((candidate["yaw_rad"] - self.controller.pose()["yaw_rad"] + np.pi) % (2*np.pi) - np.pi)

    def init_evaluator(self, key, attempt):
        evaluator = None
        if self.context.goal in ("grasp_lift", "open_extract_lift", "slide_extract_lift"):
            evaluator = self.adapter.create_evaluator(self.session, self.controller, self.output / "evaluation" / key / str(attempt))
            self.session.task_evaluator = evaluator
            evaluator.begin_episode(self.session.observation_source.agent_observation())
        return evaluator

    def hold(self):
        for _ in range(self.options.hold_steps):
            self.check_budget()
            self.controller.step_base_debug()
            contacts = self._contacts()
            self.collisions.extend(contacts)
            if contacts:
                raise ProbeStopped("Unexpected contact during holding.")

    def attempt_grasp(self, key, candidate, row):
        target = self.session.env.scene.object_registry("name", self.context.bindings[key].name)
        row["initial_target"] = object_state(target)
        self.allowed_object = target
        self.move("RELEASE")
        row["measured_opening_m"] = self.controller.gripper_state()["gripper_width_m"]
        desired_orientation = Rotation.from_quat(candidate["quaternion_xyzw"])
        for _ in range(72):
            orientation = Rotation.from_quat(self.controller.ee_pose()["world_quaternion_xyzw"])
            relative = (orientation.inv() * desired_orientation).as_matrix()
            angle = float(np.degrees(np.arctan2(relative[1, 0], relative[0, 0])))
            if abs(angle) < 2.5:
                break
            self.move("ROTATE_CW" if angle > 0 else "ROTATE_CCW", min(abs(angle), 20.))
        else:
            raise ProbeStopped("Bounded wrist alignment did not converge.")
        length = float(np.mean(list(self.session.robot.eef_to_fingertip_lengths[self.controller.working_arm].values())))
        orientation = Rotation.from_quat(self.controller.ee_pose()["world_quaternion_xyzw"])
        tcp = np.asarray(candidate["point"]) - orientation.apply([0., 0., length])
        pregrasp = tcp - orientation.apply([0., 0., .05])
        via = pregrasp.copy()
        via[2] = max(via[2], self.controller.ee_pose()["world_position"][2])
        self.align(via, axes=(2, 0, 1))
        self.align(pregrasp, axes=(2,))
        self.align(tcp)
        row["approach_verified"] = True
        row["ik_diagnostics"] = self.controller.ik_guard.diagnostics()
        if "grasp" not in self.options.enabled and "goal" not in self.options.enabled:
            self.align(pregrasp)
            row["retreat_verified"] = True
            return
        self.move("GRASP")
        state = self.controller.debug_gripper_state()
        row["grasp_state"] = state
        if state["holding"] is not True or state["object_name"] != target.name or not state["constraint_active"]:
            raise ProbeStopped("The designated object was not acquired by the native contact-qualified grasp.")
        self.carried_object = target
        self.departure_height = row["initial_target"]["aabb_min"][2]
        support_key = self.context.object_supports.get(key)
        support = self.session.env.scene.object_registry("name", self.context.bindings[support_key].name) if support_key else None
        self.departure_supports = (support.prim_path,) if support is not None else ()
        threshold = max(self.options.lift_m, float(self.context.evaluation.get("minimum_lift_m", .05)) + .01)
        if self.context.goal in ("open_extract_lift", "slide_extract_lift"):
            box = self.session.env.scene.object_registry("name", self.context.bindings[self.context.evaluation["container"]].name)
            rim = float(_array(box.get_position_orientation()[0])[2]) + self.context.evaluation["rim_height_m"]
            threshold = max(threshold, rim + self.context.evaluation.get("extraction_clearance_m", .005) - self.departure_height + .01)
        remaining = threshold
        while remaining > .005:
            step = min(.08, remaining)
            self.move("MV_UP", step)
            remaining -= step
        self.hold()
        state = self.controller.debug_gripper_state()
        final = object_state(target)
        lifted = final["aabb_min"][2] - row["initial_target"]["aabb_min"][2]
        held = state["holding"] is True and state["object_name"] == target.name and state["constraint_active"]
        stable = final["linear_speed_m_s"] <= .02 and final["angular_speed_rad_s"] <= .1
        row.update(final_target=final, final_gripper=state, lift_m=lifted,
                   hold_steps=self.options.hold_steps, held=bool(held), stable=stable,
                   grasp_lift_verified=bool(held and stable and lifted >= float(self.context.evaluation.get("minimum_lift_m", .05))))
        evaluator = getattr(self.session, "task_evaluator", None)
        if evaluator is not None:
            observation = self.session.observation_source.capture(action_boundary=True).observation
            row["goal_evaluation"] = evaluator.record(observation, action="private_hold")

    def run(self):
        results = {}
        for key in self.context.targets:
            self.deadline = time.monotonic() + self.options.search_budget_s
            target = self.session.env.scene.object_registry("name", self.context.bindings[key].name)
            rows, geometry = [], {}
            results[key] = {"attempts": rows, "geometry": geometry, "grasp_lift_verified": False}
            try:
                candidates, geometry = local_grasp_candidates(target, self.controller, self.options)
                results[key]["geometry"] = geometry
                stances = self.navigation.get(key, {}).get("candidates", [])
                # Start with the measured pose; subsequent attempts replay planned stances.
                for index, candidate in enumerate(candidates):
                    row = {"index": index, "candidate": candidate, "approach_verified": False, "grasp_lift_verified": False}
                    rows.append(row)
                    self.actions, self.collisions = [], []
                    try:
                        self.carried_object = None
                        self.destination_object = None
                        row["restoration_before"] = self.snapshot.restore()
                        self.check_budget()
                        self.init_evaluator(key, index)
                        if index and stances:
                            stance = stances[(index-1) % len(stances)]
                            row["navigation_candidate"] = stance
                            self.follow_route(stance)
                            # Align the stored world orientation after any base motion.
                        if self.context.goal in ("open_extract_lift", "slide_extract_lift"):
                            from wia.runtime.preflight.goals import open_articulation
                            row["articulation"] = open_articulation(self)
                        self.attempt_grasp(key, candidate, row)
                        if self.perception is not None:
                            self.perception.inspect(phase=f"probe_{key}_{index}", keys=(key,))
                        if row.get("grasp_lift_verified") and self.context.goal == "answer_and_place" and "goal" in self.options.enabled:
                            from wia.runtime.preflight.goals import attempt_placement
                            row["placement"] = attempt_placement(self, target)
                    except (ProbeStopped, ValueError, RuntimeError) as error:
                        row["failure_reason"] = f"{type(error).__name__}: {error}"
                    finally:
                        row["actions"] = self.actions
                        row["collisions"] = self.collisions
                        self.carried_object = None
                        self.destination_object = None
                        # Restoration failure aborts the whole probe stage; never reuse dirty state.
                        row["restoration_after"] = self.snapshot.restore()
                    results[key]["grasp_lift_verified"] |= row.get("grasp_lift_verified", False)
                    self.publish(key, results[key])
                    if row.get("grasp_lift_verified") or time.monotonic() >= self.deadline:
                        break
            except Exception as error:
                results[key]["error"] = f"{type(error).__name__}: {error}"
                self.publish(key, results[key])
                # Recheck isolation before another target. If this fails, stop.
                self.snapshot.restore()
            self.publish(key, results[key])
        self.emit(Check("physics", "probe_state_isolation", "PASS", "Every destructive probe restored and checked native state before reuse; worker world is discarded.",
            evidence={"restores": self.snapshot.restores, "full_scene_pose_checks": len(self.snapshot.poses)}, verification_level="native_initialization"))
        return results

    def publish(self, key, result):
        attempts = result["attempts"]
        approached = any(row.get("approach_verified") for row in attempts)
        grasped = result["grasp_lift_verified"]
        reached = grasped or any(row.get("retreat_verified") for row in attempts)
        evidence = {**result, "attempted_candidate_count": len(attempts), "grasp_mode": "native_assisted_contact_qualified",
                    "search_budget_s": self.options.search_budget_s, "finite_search": True}
        if "manipulation" in self.options.enabled:
            self.emit(Check("manipulation", "reachability." + key, "PASS" if reached else "INCONCLUSIVE",
                "Native guarded approach and retreat/lift completed with joint continuity and contact monitoring." if reached else
                "Approach evidence is partial; bounded motion failures do not prove the target unreachable.", [key], evidence,
                verification_level="physical_manipulation" if approached else "native_initialization"))
        if "grasp" in self.options.enabled:
            self.emit(Check("grasp", "grasp." + key, "PASS" if grasped else "INCONCLUSIVE",
                "Correct object remained attached and stable after the measured lift and hold." if grasped else
                "Finite local grasp search did not establish a stable correct-object lift.", [key], evidence,
                verification_level="physical_manipulation" if grasped else "native_initialization"))
        if grasped and self.context.goal in ("grasp_lift", "open_extract_lift", "slide_extract_lift") and "navigation" in self.options.enabled:
            self.emit(Check("navigation", "carried_clearance." + key, "PASS", "The task's bounded carried lift path was physically executed with native contact monitoring.",
                [key], {"scope": "measured lift path only; no distant transport certified"}, verification_level="physical_manipulation"))
