#!/usr/bin/env python3
"""Private native feasibility replay. Never call during a measured Agent episode."""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from core.action_compiler import ParameterizedAction
    from wia.runtime.sim.esi_explore_scenes import TASK_INSTRUCTION
    from wia.runtime.sim.esi_omnigibson import _array
    import wia.runtime.sim.esi_web as web
    from wia.runtime.commands.run_esi_robot import main as robot_main

    args = sys.argv[1:] if argv is None else argv
    output = Path(args[args.index("--output-dir") + 1])

    def qualify(session, metadata, controller, executor, **kwargs):
        source = session.observation_source
        checker = session.private_visibility_checker
        target = session.task_evaluator.target
        start = _array(target.get_position_orientation()[0]).copy()
        for _ in range(30):
            controller.step_base_debug()
        stability = float(np.linalg.norm(_array(target.get_position_orientation()[0])-start))
        lower, upper = (_array(value) for value in session.robot.aabb)
        source.begin_episode(TASK_INSTRUCTION)
        initial = checker.check_initial(source.agent_observation())
        source.agent_observation().save(output / "qualification_initial")
        unoccluded = None
        if session.explore_scene.occluders:
            # Offline diagnostic ONLY: verify a genuine occluder-induced deficit,
            # with the same cameras/poses and without a physics step or teleport.
            occluders = [session.env.scene.object_registry("name", item["name"])
                         for item in session.explore_scene.occluders]
            before_diagnostic = float(session.pipeline.og.sim.current_time)
            try:
                for obj in occluders:
                    obj.visible = False
                unoccluded = checker.record(phase="unoccluded_diagnostic", observation=source.agent_observation())
            finally:
                for obj in occluders:
                    obj.visible = True
                session.invalidate_camera_cache()
            if float(session.pipeline.og.sim.current_time) != before_diagnostic:
                raise RuntimeError("Unoccluded setup diagnostic unexpectedly advanced physics.")
        rows, path, contacts = [], [], []
        previous_step = controller.on_step
        phase = "approach"
        def current_contacts(phase):
            found = []
            for name, link in session.robot.links.items():
                if not link.contact_reporting_enabled:
                    continue
                for contact in link.contact_list():
                    bodies = [str(getattr(contact, key, "")) for key in ("body0", "body1")]
                    external = [body for body in bodies if not body.startswith(session.robot.prim_path)]
                    if external and not all("floor" in body for body in external):
                        found.append({"env_step": len(path), "phase": phase, "link": name, "bodies": bodies})
            return found
        spawn_contacts = current_contacts("spawn")
        def sample():
            if previous_step:
                previous_step()
            path.append(controller.pose()["position"])
            contacts.extend(current_contacts(phase))
        controller.on_step = sample

        def move(action, distance=None, angle=None):
            command = {"action": action}
            if distance is not None:
                command["distance_m"] = distance
            if angle is not None:
                command["angle_deg"] = angle
            at = time.perf_counter()
            result = executor.execute_macro(ParameterizedAction(**command), resolution=executor.primitive_resolution,
                                            limits=executor.macro_limits)
            rows.append({"command": command, "result": result, "wall_time_s": time.perf_counter()-at})
            with (output / "qualification_actions.jsonl").open("a") as stream:
                stream.write(json.dumps(rows[-1]) + "\n")
            print("[qualify] " + json.dumps({"command": command, "success": result["success"],
                                            "reason": result.get("reason")}), flush=True)
            if not result["success"]:
                raise RuntimeError("Private feasibility replay failed: " + str(result.get("reason")))

        def target_in_base():
            base = controller.pose()
            point = _array(target.get_position_orientation()[0]).copy()
            point[2] += float(session.explore_scene.config.get("qualification_grasp_height_offset_m", .002))
            # Measured base roll/pitch matter at tabletop height. This private
            # oracle point never enters an Agent prompt or controller shortcut.
            return Rotation.from_quat(base["quaternion_xyzw"]).inv().apply(point-np.asarray(base["position"]))

        def align(axis):
            for _ in range(8):
                delta = float(target_in_base()[axis]-controller.ee_pose()["base_position"][axis])
                if abs(delta) <= .00325:
                    return
                action = (("MV_FWD", "MV_BACK"), ("MV_LEFT", "MV_RIGHT"))[axis][delta < 0]
                move(action, min(.08, abs(delta)))
            raise RuntimeError("Private alignment did not settle within its quantized bounds.")

        summary = {"schema": "private_explore_qualification_v1", "seed": session.explore_scene.seed,
                   "variant": session.explore_scene.config["variant"], "initial_visibility": initial,
                   "target_stability_displacement_m": stability, "oracle_replay": True,
                   "robot_actual_aabb": [lower.tolist(), upper.tolist()],
                   "spawn_external_contacts": spawn_contacts,
                   "unoccluded_setup_diagnostic": unoccluded,
                   "qualification_success": False}
        try:
            near_offset = float(session.explore_scene.config.get("qualification_approach_reduction_m", 0.))
            if session.explore_scene.config["variant"] == "multi_region_search":
                # Separate offline route: translate beside the near table edge
                # before approaching it. No oracle route is exposed to the Agent.
                yaw = np.degrees(controller.pose()["yaw_rad"])
                turn = (90. - yaw + 180.) % 360. - 180.
                if abs(turn) >= 2.5:
                    move("BASE_TURN_LEFT" if turn > 0 else "BASE_TURN_RIGHT", angle=abs(turn))
                move("BASE_FWD", .30 - near_offset)
                right_inspection = checker.record(phase="right_region_inspection", observation=source.agent_observation())
                source.agent_observation().save(output / "qualification_right_inspection")
                summary["right_candidate_viewpoint"] = right_inspection
                if session.explore_scene.selected_index == 1:
                    move("BASE_BACK", .30 - near_offset)
                    move("BASE_LEFT", .40)
                    move("BASE_LEFT", .10)
                    # Stand back from the center edge and present the goal in
                    # the already validated planar arm workspace. These are
                    # separate bounded macros, not an oracle Agent action.
                    move("BASE_TURN_RIGHT", angle=10.)
                    move("BASE_FWD", float(session.explore_scene.config.get("qualification_left_approach_m", .25 - near_offset)))
            elif session.explore_scene.occluders:
                # Pose variants start within five degrees of the validated
                # corner approach. This offline replay uses ordinary bounded
                # turning before translation; actor episodes choose freely.
                yaw = np.degrees(controller.pose()["yaw_rad"])
                turn = (90. - yaw + 180.) % 360. - 180.
                if abs(turn) >= 2.5:
                    move("BASE_TURN_LEFT" if turn > 0 else "BASE_TURN_RIGHT", angle=abs(turn))
                move("BASE_FWD", .30 - near_offset)
            revealing = checker.record(phase="revealing", observation=source.agent_observation())
            source.agent_observation().save(output / "qualification_revealing")
            summary["revealing_visibility"] = revealing
            if session.explore_scene.config.get("qualification_motion_probes", False):
                # Extra offline controller checks for the near-scene revision;
                # no probe actions occur in the measured model episodes.
                move("TRUNK_DOWN", .01)
                move("TRUNK_UP", .01)
                summary["trunk_motion_probe_passed"] = True
            align(0)
            align(1)
            phase = "grasp_alignment"
            dz = float(controller.ee_pose()["base_position"][2]-target_in_base()[2])
            while dz >= .0025:
                length = min(.08, dz)
                move("MV_DOWN", length)
                dz -= round(length/.005)*.005
            move("GRASP")
            move("MV_UP", .06)
            held = session.task_evaluator.record(source.agent_observation(), action=None)
            hold_samples = []
            for _ in range(15):
                controller.step_base_debug()
                hold_samples.append(session.task_evaluator.record(source.agent_observation(), action=None)["task_success"])
            held_after = session.task_evaluator.record(source.agent_observation(), action=None)
            unintended = [row for row in contacts if any(
                not body.startswith(session.robot.prim_path) and not body.startswith(target.prim_path + "/")
                for body in row["bodies"])]
            source.agent_observation().save(output / "qualification_final")
            gates = {"target_stable": stability < .005,
                "revealing_view_recognizable": revealing["any_recognizably_visible"],
                "same_pose_unoccluded_recognizable": unoccluded is None or unoccluded["any_recognizably_visible"],
                "correct_attachment_and_lift": held_after["task_success"],
                "native_contact_reporting": any(link.contact_reporting_enabled for link in session.robot.links.values()),
                "spawn_clear": not spawn_contacts, "whole_route_clear": not unintended}
            summary.update(evaluation=held_after, legacy_grasp_lift_success=held["task_success"],
                unintended_contact_count=len(unintended),
                qualification_gates=gates,
                stable_hold_0_5_sim_s=all(hold_samples),
                stable_hold_criterion={"schema": "private_stable_hold_v1", "duration_sim_s": 15*controller.dt,
                    "sample_period_sim_s": controller.dt, "all_samples_target_attached_and_lifted": all(hold_samples)},
                qualification_success=all(gates.values()))
        except Exception as error:
            summary["error"] = str(error)
        finally:
            controller.on_step = previous_step
            summary["unintended_contact_count"] = sum(any(
                not body.startswith(session.robot.prim_path) and not body.startswith(target.prim_path + "/")
                for body in row["bodies"]) for row in contacts)
            summary.update(actions=rows, physical_path_samples=path, external_contacts=contacts,
                contact_reporting_links=[n for n, link in session.robot.links.items() if link.contact_reporting_enabled])
            (output / "qualification_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            print("[qualify] result " + str(summary["qualification_success"]), flush=True)

    web.run_esi_control = qualify
    return robot_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
