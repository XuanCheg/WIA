"""Native right-hand plate collection evidence for hidden-object episodes.

Run CPU checks: python -B -m pytest -q tests/test_hidden_object_collection.py
Native evaluation only reads state; initialization belongs to the explicit
scene hook. Deposited targets are never attached or teleported by this module.
"""
from __future__ import annotations

from wia.resources import ESI_ROOT

from pathlib import Path
from copy import deepcopy
import json

import numpy as np
from scipy.spatial.transform import Rotation

MODULE = "core.esi_converters.hidden_object_collection"
REVISION = "hidden_collection_20261020_v1"
OPERATION = "find_and_collect_on_right_hand_plate"


def open_shelf_objects(name, position, yaw_deg, *, width=.8, depth=.5,
                       surface_height=.8, opening_height=.45):
    """Propose an open-front shelf; native access still requires validation.

    Local -Y is the opening. The deck, roof, sides and back provide physical
    support and occlusion. No doors or movable occluders are required.
    """
    rotation = Rotation.from_euler("z", yaw_deg, degrees=True)
    thickness = .035
    roof = surface_height + opening_height
    parts = [
        ("deck", [0., 0., surface_height - thickness / 2], [width, depth, thickness]),
        ("roof", [0., 0., roof + thickness / 2], [width, depth, thickness]),
        ("back", [0., depth / 2 - thickness / 2, (surface_height + roof) / 2],
            [width, thickness, opening_height]),
    ]
    for label, side in (("left", -1), ("right", 1)):
        parts.append((f"side_{label}", [side * (width - thickness) / 2, 0., (surface_height + roof) / 2],
                      [thickness, depth, opening_height]))
    for index, (x, y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        parts.append((f"leg_{index}", [x * (width / 2 - .04), y * (depth / 2 - .04),
                                     (surface_height - thickness) / 2],
                      [.05, .05, surface_height - thickness]))
    return [{"type": "PrimitiveObject", "primitive_type": "Cube",
        "name": f"{name}_{suffix}", "category": "shelf",
        "position": (array(position) + rotation.apply(local)).tolist(),
        "orientation": rotation.as_quat().tolist(), "scale": dimensions,
        "rgba": [.48, .32, .18, 1.], "fixed_base": True, "visual_only": False}
        for suffix, local, dimensions in parts]


def build_episode(source, context, layout):
    """Build a pending collection episode from an explicit, reviewable layout."""
    from wia.authoring.conversion import original_scene, physical_task
    from wia.authoring.io import digest
    inventory = layout["inventory"]
    targets = [row["name"] for row in inventory if row["role"] == "target"]
    distractors = [row["name"] for row in inventory if row["role"] == "distractor"]
    if len(targets) != layout["target_count"] or len(set(targets + distractors)) != len(inventory):
        raise ValueError("Collection inventory must contain distinct named instances.")
    parameters = {"revision": REVISION, "layout": deepcopy(layout),
        "plate": deepcopy(layout["plate"]), "targets": targets, "distractors": distractors,
        "terminal_dwell_s": 1., "maximum_relative_speed_m_s": .03,
        "maximum_plate_tilt_deg": 10., "maximum_plate_speed_m_s": .03,
        "maximum_plate_angular_speed_rad_s": .1}
    patches = [{"operation": "category_initialization", "phase": phase,
        "module": MODULE, "parameters": parameters}
        for phase in ("before_construction", "after_postprocessing")]
    bindings = {row["name"]: {k: row[k] for k in ("name", "category", "model")} for row in inventory}
    bindings[layout["plate"]["name"]] = {
        k: layout["plate"][k] for k in ("name", "category", "model")}
    public = {
        "objective": f"Find all {len(targets)} hamburgers in {layout['public_search_area']}. "
            "Use your left hand to collect them on the tray held in your right hand.",
        "success_criteria": f"Finish with all {len(targets)} distinct hamburgers resting on the tray, "
            "your left hand released, and the tray still held steadily in your right hand.",
        "environment": layout["public_environment"],
        "constraints": ["Keep the tray in your right hand. Use the left hand to pick and place.",
            "Targets rest on accessible supports. Opening doors or moving furniture is not required.",
            "Repeated views of an object count as one physical object. No return to departure is required."]}
    if distractors:
        public["constraints"].append(
            "A sandwich and a plain hamburger bun without filling are distractors. Do not leave them on the tray.")
    evaluation = {"type": "category_operation", "module": MODULE, "operation": OPERATION,
        "binding_keys": list(bindings), "parameters": parameters}
    task = physical_task(source, context, public=public, evaluation=evaluation,
        bindings=bindings, patches=patches,
        capabilities=["base_translation", "base_rotation", "ee_translation", "gripper"])
    task["repair"] = {"collection_revision": REVISION,
        "recommended_primitive_profile": "fine",
        "controller_profile_file": "configs/esi_primitives_fine.json",
        "native_qualification_status": "pending"}
    scene = original_scene(source, context, bindings=bindings, patches=patches)
    identity = "hidden_collection_" + digest({"layout": layout, "revision": REVISION})[:24]
    task["id"], scene["id"] = "task_" + identity, "scene_" + identity
    return {"candidate_id": identity, "task_definition": task, "scene_definition": scene,
        "native_validation": "pending", "variant": layout["variant"]}


def configure_scene(config, parameters):
    """Construct movable inventory and the declared open support geometry."""
    layout = parameters["layout"]
    if parameters["revision"] != REVISION or config["scene"]["scene_model"] != layout["scene_model"]:
        raise ValueError("Collection layout belongs to a different native world.")
    config["scene"].update(load_room_instances=layout["loaded_rooms"], load_room_types=None,
        load_object_categories=None, not_load_object_categories=None)
    config["objects"] = [dict(type="DatasetObject", fixed_base=False, visual_only=False,
        **{k: row[k] for k in ("name", "category", "model", "position", "orientation")})
        for row in layout["inventory"]]
    config["objects"].extend(deepcopy(layout.get("support_objects", [])))
    plate = layout["plate"]
    config["objects"].append({"type": "DatasetObject", "name": plate["name"],
        "category": plate["category"], "model": plate["model"],
        "position": [100., 100., 1.], "fixed_base": False, "visual_only": False})
    config["objects"].append({"type": "PrimitiveObject", "name": "collection_initial_support",
        "primitive_type": "Cube", "scale": [.25, .36, .04], "position": [100., 100., .8],
        "fixed_base": True, "visual_only": False})
    import yaml
    from core.sim.esi_omnigibson import _import_pipeline
    pipeline = _import_pipeline(ESI_ROOT)
    robot_config = yaml.safe_load(
        (Path(pipeline.og.example_config_path) / "r1_primitives.yaml").read_text())["robots"][0]
    robot_config.update(deepcopy(layout["robot_start"]))
    robot_config.update(name="robot", grasping_mode="assisted", disable_grasp_handling=False)
    # The R1 joint ordering comes from its installed native reset configuration.
    # Verify native trunk indices again in initialize_scene before using the pose.
    robot_config["reset_joint_pos"][6:10] = [.6981317007977318, -1.3962634015954636,
                                          -.6981317007977318, 0.]
    config["robots"] = [robot_config]
    return {"revision": REVISION, "loaded_rooms": layout["loaded_rooms"],
        "native_layout_qualified": False}


def postprocess_joint_world(session, parameters):
    """Replace the retired category's reveal hook for this wholly new inventory."""
    if session.scene_name != parameters["layout"]["scene_model"]:
        raise ValueError("Collection postprocessing belongs to a different world.")
    return {"joint_world_postprocess": True, "revision": REVISION,
        "retired_source_mechanism_reused": False, "native_layout_qualified": False}


def initialize_scene(session, parameters):
    from core.sim.esi_collection_setup import initialize_collection_tray
    receipt = initialize_collection_tray(session, parameters)
    supports = {}
    for item in parameters["layout"]["inventory"]:
        obj = session.env.scene.object_registry("name", item["name"])
        support = session.env.scene.object_registry("name", item["support"])
        contact = any(_under(getattr(c, side), support.prim_path)
                      for c in obj.contact_list() for side in ("body0", "body1"))
        low, high = (array(v) for v in obj.aabb)
        support_low, support_high = (array(v) for v in support.aabb)
        near_surface = abs(float(low[2] - support_high[2])) <= .015
        speed = float(np.linalg.norm(array(obj.get_linear_velocity())))
        supports[obj.name] = {"support": support.name, "contact": bool(contact),
            "bottom_height_m": float(low[2]), "surface_height_m": float(support_high[2]),
            "speed_m_s": speed, "movable_collidable": not obj.fixed_base and not obj.visual_only,
            "stable_support": bool(contact and near_surface and speed < .05)}
    accepted = receipt["accepted"] and all(
        value["stable_support"] and value["movable_collidable"] for value in supports.values())
    result = {**receipt, "accepted": accepted, "inventory_supports": supports,
        "validation_scope": "Basic reset, right-hand tray and settled support checks only.",
        "initial_target_visibility_required": False, "complete_collection_verified": False,
        "all_target_grasp_access_verified": False}
    session.category_interaction_setup = result
    return result


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def collection_result(state, targets, distractors, picked, *, stable_seconds, required_dwell):
    """Reduce current physical inventory, never cumulative visits or sightings."""
    present = set(state["supported_objects"])
    held = set(state["held_objects"])
    correct = sorted(present.intersection(targets).difference(held).intersection(picked))
    wrong = sorted(present.intersection(distractors))
    complete = len(correct) == len(targets)
    conditions = {
        "right_hand_holds_plate": state["right_holds_plate"],
        "plate_level": state["plate_level"],
        "plate_stable": state["plate_stable"],
        "left_hand_released": state["left_empty"],
        "all_targets_collected": complete,
        "no_distractors_on_plate": not wrong,
        "stable_terminal_dwell": stable_seconds >= required_dwell,
    }
    return {"task_success": all(conditions.values()), "checks": conditions,
        "collected_targets": correct, "distractors_on_plate": wrong,
        "collection_recall": len(correct) / len(targets),
        "stable_seconds": stable_seconds}


def _under(path, root):
    return str(path) == root or str(path).startswith(root + "/")


def _state(session, parameters):
    robot = session.robot
    plate = session.env.scene.object_registry("name", parameters["plate"]["name"])
    position, quaternion = (array(v) for v in plate.get_position_orientation())
    rotation = Rotation.from_quat(quaternion)
    linear, angular = array(plate.get_linear_velocity()), array(plate.get_angular_velocity())
    hand = getattr(robot, "_ag_obj_in_hand", {})
    holding = {arm: getattr(hand.get(arm), "name", None) for arm in ("left", "right")}
    # Native IsGraspingState uses FALSE=-1; bool(-1) would be a false positive.
    right_grasp = int(robot.is_grasping("right")) == 1
    inner = parameters["plate"]["usable_bounds_local_m"]
    lower, upper = array(inner[0]), array(inner[1])
    supported, details = [], {}
    for name in parameters["targets"] + parameters.get("distractors", []):
        obj = session.env.scene.object_registry("name", name)
        if obj is None or obj.visual_only or obj.fixed_base:
            raise ValueError("Collection requires the exact movable collidable inventory.")
        points = [array(link.collision_boundary_points_world) for link in obj.links.values()
                  if link.collision_boundary_points_world is not None]
        if not points:
            raise ValueError("Native target collision boundary unavailable.")
        local = rotation.inv().apply(np.concatenate(points) - position)
        lo, hi = local.min(0), local.max(0)
        contact = any(_under(getattr(c, side), plate.prim_path)
                      for c in obj.contact_list() for side in ("body0", "body1"))
        on_deck = (np.all(lo[:2] >= lower[:2] - .002)
                   and np.all(hi[:2] <= upper[:2] + .002)
                   and lower[2] - .006 <= lo[2] <= lower[2] + .015
                   and hi[2] <= upper[2])
        obj_position = array(obj.get_position_orientation()[0])
        relative_speed = float(np.linalg.norm(array(obj.get_linear_velocity())
            - linear - np.cross(angular, obj_position - position)))
        stable = relative_speed <= parameters["maximum_relative_speed_m_s"]
        ok = bool(contact and on_deck and stable and name not in holding.values())
        if ok:
            supported.append(name)
        details[name] = {"contact": bool(contact), "inside_usable_deck": bool(on_deck),
            "relative_speed_m_s": relative_speed, "supported": ok,
            "bounds_in_plate_frame": [lo.tolist(), hi.tolist()]}
    up = float(rotation.apply([0., 0., 1.])[2])
    return {"supported_objects": supported, "held_objects": [x for x in holding.values() if x],
        "left_empty": holding["left"] is None and int(robot.is_grasping("left")) == -1,
        "left_holding": holding["left"],
        "right_holds_plate": bool(holding["right"] == plate.name and right_grasp),
        "plate_level": bool(up >= np.cos(np.deg2rad(parameters["maximum_plate_tilt_deg"]))),
        "plate_stable": bool(np.linalg.norm(linear) <= parameters["maximum_plate_speed_m_s"]
                             and np.linalg.norm(angular) <= parameters["maximum_plate_angular_speed_rad_s"]),
        "plate_pose": [position.tolist(), quaternion.tolist()], "target_details": details,
        "native_time_s": float(session.pipeline.og.sim.current_time)}


def qualify_operation(session, executor, config, bound):
    parameters = config["parameters"]
    state = _state(session, parameters)
    receipt = getattr(session, "hidden_collection_initialization", {})
    checks = {"right_hand_holds_plate": state["right_holds_plate"],
        "plate_level": state["plate_level"], "initial_plate_empty": not state["supported_objects"],
        "left_hand_empty": state["left_empty"], "initialization_receipt": receipt.get("accepted") is True,
        "left_arm_control": getattr(getattr(executor, "controller", None), "working_arm", None) == "left"}
    return {"accepted": all(checks.values()), "checks": checks, "state": state}


class CollectionEvaluator:
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters = config["parameters"]
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.picked = set()
        self.wrong_pickups = set()
        self.previous_collected = set()
        self.spills = []
        self.last_time = None
        self.stable_since = None
        self.state = None

    def begin_episode(self, observation):
        self.picked.clear()
        self.wrong_pickups.clear()
        self.spills.clear()
        self.previous_collected.clear()
        self.last_time = self.stable_since = None
        self.sample_step()

    def sample_step(self):
        state = _state(self.session, self.parameters)
        now = state["native_time_s"]
        if self.last_time is not None and now <= self.last_time:
            return
        held = state["left_holding"]
        if held in self.parameters["targets"]:
            self.picked.add(held)
        if held in self.parameters.get("distractors", []):
            self.wrong_pickups.add(held)
        present = set(state["supported_objects"]).intersection(self.parameters["targets"])
        for name in self.previous_collected - present:
            self.spills.append({"target": name, "native_time_s": now})
        self.previous_collected = present
        candidate = collection_result(state, self.parameters["targets"],
            self.parameters.get("distractors", []), self.picked,
            stable_seconds=0., required_dwell=0.)
        if candidate["task_success"]:
            if self.stable_since is None:
                self.stable_since = now
        else:
            self.stable_since = None
        self.last_time, self.state = now, state

    def record(self, observation, *, action=None):
        self.sample_step()
        duration = 0. if self.stable_since is None else self.last_time - self.stable_since
        result = collection_result(self.state, self.parameters["targets"],
            self.parameters.get("distractors", []), self.picked, stable_seconds=duration,
            required_dwell=self.parameters["terminal_dwell_s"])
        result.update(schema="native_hidden_collection_v1", snapshot_id=observation.snapshot_id,
            action_step=observation.action_step, evaluation_validity="complete",
            terminal_conditions_satisfied=result["task_success"],
            completion_items=[{"id": "collect:" + name, "kind": "current_physical_inventory",
                "completed": name in result["collected_targets"], "evaluation_validity": "valid"}
                for name in self.parameters["targets"]],
            wrong_pickups=sorted(self.wrong_pickups), spills=self.spills,
            native_state=self.state, configuration=self.config)
        (self.output / "latest.json").write_text(json.dumps(result, indent=2) + "\n")
        return result


def create_evaluator(session, controller, config, output):
    if config["operation"] != OPERATION or config["parameters"]["revision"] != REVISION:
        raise ValueError("Unsupported hidden collection revision.")
    return CollectionEvaluator(session, controller, config, output)
