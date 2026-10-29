"""Run private feasibility waypoints through the native diagnostic public API.

Run: python -B src/wia/runtime/commands/pilot_hidden_collection.py --run DIR --plan PLAN.json
The server must be src/wia/runtime/commands/check_esi_core_native.py --serve. The plan contains
world waypoints, target names and pickup headings. No scene setters are exposed.
This is an oracle-guided feasibility pilot, not autonomous search performance.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(WIA_RUNTIME_ROOT))
from wia.runtime.sim.esi_route_pilot import next_translation


class Pilot:
    def __init__(self, directory, plan):
        self.directory, self.plan = directory, plan
        self.state = json.loads((directory / "initial_state.json").read_text())
        self.count = 0
        self.report = {"status": "running", "autonomous_search_performance": False,
            "public_commands_only": True, "object_pose_setters": 0, "completed_targets": [],
            "commands": [], "plan": plan}
        self.started = time.monotonic()

    def save(self):
        self.report["elapsed_s"] = time.monotonic() - self.started
        (self.directory / "collection_pilot.json").write_text(json.dumps(self.report, indent=2) + "\n")

    def command(self, payload):
        self.count += 1
        label = f"pilot_{self.count:04d}"
        request = self.directory / "requests" / (label + ".json")
        temporary = request.with_suffix(".tmp")
        temporary.write_text(json.dumps({"commands": [payload]}) + "\n")
        temporary.replace(request)
        deadline = time.monotonic() + 180.
        reply = self.directory / "responses" / request.name
        while not reply.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Native command {label} did not return.")
            time.sleep(.25)
        result = json.loads(reply.read_text())
        if not result.get("complete"):
            raise RuntimeError(result)
        self.state = result["state"]
        receipt = result["commands"][0]["result"]
        self.report["commands"].append({"request": label, "command": payload, "result": receipt})
        self.save()
        print(json.dumps({"request": label, "action": payload, "success": receipt.get("success")}), flush=True)
        if not receipt["success"]:
            raise RuntimeError(f"Native controller stopped {label}: {receipt.get('message', receipt)}")
        return receipt

    def yaw(self, degrees):
        for _ in range(10):
            current = np.rad2deg(self.state["robot"]["yaw_rad"])
            error = (degrees - current + 180.) % 360. - 180.
            if abs(error) < 2.6:
                return
            self.command({"action": "BASE_TURN_LEFT" if error > 0 else "BASE_TURN_RIGHT",
                          "angle_deg": min(abs(error), 60.)})
        raise RuntimeError("Heading did not converge.")

    def navigate(self, waypoints):
        for point in waypoints:
            for _ in range(80):
                pose = self.state["robot"]
                if np.linalg.norm(np.asarray(point) - np.asarray(pose["position"])[:2]) < .025:
                    break
                self.command(next_translation(pose["position"], pose["quaternion_xyzw"], point,
                    maximum_m=.35, minimum_m=.025))
            else:
                raise RuntimeError("Waypoint did not converge.")

    def arm(self, position):
        for _ in range(60):
            delta = np.asarray(position) - self.state["end_effector"]["base_position"]
            axis = int(np.argmax(np.abs(delta)))
            if abs(delta[axis]) < .004:
                return
            tokens = (("MV_BACK", "MV_FWD"), ("MV_RIGHT", "MV_LEFT"), ("MV_DOWN", "MV_UP"))
            self.command({"action": tokens[axis][int(delta[axis] > 0)], "distance_m": min(abs(delta[axis]), .06)})
        raise RuntimeError("End-effector waypoint did not converge.")

    def base_point(self, world):
        pose = self.state["robot"]
        return Rotation.from_quat(pose["quaternion_xyzw"]).inv().apply(np.asarray(world) - pose["position"])

    def collect(self, item, slot):
        name = item["name"]
        self.navigate(item.get("waypoints", []))
        self.yaw(item["pickup_heading_deg"])
        target = self.state["objects"][name]
        goal_xy = np.asarray(target["position_m"])[:2]
        rotation = Rotation.from_euler("z", item["pickup_heading_deg"], degrees=True)
        stance = goal_xy - rotation.apply([.75, .3, 0.])[:2]
        self.navigate([stance.tolist()])
        self.command({"action": "RELEASE"})
        self.arm([*self.state["end_effector"]["base_position"][:2], .98])
        target = self.state["objects"][name]
        grasp_world = np.array(target["position_m"])
        grasp_world[2] = target["aabb_m"][1][2] - .025
        grasp = self.base_point(grasp_world)
        self.arm([*grasp[:2], .98])
        self.arm(grasp)
        for _ in range(8):
            self.command({"action": "GRASP"})
            if self.state["gripper"].get("object_name") == name and self.state["gripper"]["holding"]:
                break
        else:
            raise RuntimeError("The exact target was not grasped by the left hand.")
        position = self.state["end_effector"]["base_position"]
        self.arm([*position[:2], position[2] + .09])
        target = self.state["objects"][name]
        vertical_offset = self.state["end_effector"]["world_position"][2] - target["aabb_m"][0][2]
        plate = self.state["objects"]["collection_plate"]
        deck_world = (np.asarray(plate["position_m"]) + Rotation.from_quat(
            plate["quaternion_xyzw"]).apply([*slot, self.plan["deck_local_z_m"]]))
        drop_world = deck_world + [0., 0., vertical_offset + .008]
        drop = self.base_point(drop_world)
        self.arm([*drop[:2], .98])
        self.arm(drop)
        self.command({"action": "RELEASE"})
        # Repeated release commands retain legal control and give released
        # objects native time to settle; no target is attached to the tray.
        for _ in range(5):
            self.command({"action": "RELEASE"})
        self.arm([*self.state["end_effector"]["base_position"][:2], .98])
        self.arm([.48, .18, .98])
        self.report["completed_targets"].append(name)
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    pilot = Pilot(args.run, plan)
    try:
        for item, slot in zip(plan["targets"], plan["plate_slots"]):
            pilot.collect(item, slot)
        pilot.report["status"] = "public_pickup_and_deposit_sequence_executed"
        pilot.report["limitation"] = "Final native scorer and synchronized visibility/access evidence require review."
    except Exception as error:
        pilot.report.update(status="stopped", error=repr(error))
        raise
    finally:
        pilot.save()


if __name__ == "__main__":
    main()
