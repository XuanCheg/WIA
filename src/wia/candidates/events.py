"""Explicit authored candidate events, separate from robot control/evaluation.

Run: PYTHONPATH=src python -m pytest -q tests/test_candidate_events.py

``CandidateEvents(session, parameters).apply_preview(kind, capture_callback)``
accepts H2, M1 or H3 and captures review images without delivering policy
history. A callback receives one English waypoint label and may return an
observation receipt. It must capture the mounted robot RGB views itself.

The future runtime hook ``on_observation_delivered(frame, visible_names)`` must
only be called at an authorized action boundary after a real observation was
delivered. ``visible_names`` must come from the trusted renderer, including
``zone_<region>`` or the rendered region name when that marked area is visible.
This module does not install that hook or invoke policy/robot commands.

Only the declared event object is moved with native setters. M1 models visible
use history; it makes no claim about contamination or cleaning physics. H3
never fabricates past stock completion for an initially incomplete scene.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Callable

from wia.candidates.high import (
    BOTTOM_TOLERANCE_M, MAX_SAMPLE_GAP_S, _bounds, _number, _overlap,
    _placed, _preserved, _vector,
)


def h2_readiness(frame: dict, contract: dict) -> bool:
    """Readiness is measured clearance plus released, supported storage."""
    obstruction = contract["obstruction"]
    return (_overlap(frame, obstruction, contract.get("work_region", "work")) is False
            and _placed(frame, obstruction, contract.get("storage", "storage")) is True)


def h2_transition(phase: str, *, ready: bool, delivered: bool) -> str:
    """Require separate readiness, delivery and event-boundary observations."""
    if phase == "waiting_readiness" and ready:
        return "waiting_delivery"
    if phase == "waiting_delivery" and delivered:
        return "pending"
    if phase == "pending":
        return "emitted"
    return phase


def stock_undo_applicable(frame: dict, contract: dict, source: str = "item_02") -> bool:
    region = contract.get("stock_region", "stock")
    return (source in contract["stock"] and _placed(frame, source, region) is True
            and sum(_placed(frame, name, region) is True for name in contract["stock"]) >= 2)


def cleaning_contact_valid(*, touching: bool, bottom_z: float, support_z: float) -> bool:
    """Touching is native evidence; geometry alone cannot certify contact."""
    bottom, surface = _number(bottom_z), _number(support_z)
    return (touching is True and bottom is not None and surface is not None
            and abs(bottom - surface) <= BOTTOM_TOLERANCE_M + 1e-9)


def region_visible(visible_names: set[str], name: str) -> bool:
    return name in visible_names or f"zone_{name}" in visible_names


def wipe_waypoints(position: list, aabb: list, region: dict) -> list[dict]:
    """Plan sixteen continuous waypoints including a 0.1m supported wipe.

    The patch is chosen inside the supplied region as near the original sponge
    as its full footprint permits. No region or object identity is changed.
    """
    origin, box, zone = _vector(position, 3), _bounds(aabb), _bounds(region.get("aabb"))
    if origin is None or box is None or zone is None:
        raise ValueError("Wipe planning requires measured position and AABBs")
    half = [(box[1][i] - box[0][i]) / 2 for i in (0, 1)]
    center = [(box[0][i] + box[1][i]) / 2 for i in (0, 1)]
    low, high = ([zone[0][i] + half[i] for i in (0, 1)],
                 [zone[1][i] - half[i] for i in (0, 1)])
    axis = max((0, 1), key=lambda i: high[i] - low[i])
    if high[axis] - low[axis] < .1 - 1e-9 or any(a > b for a, b in zip(low, high)):
        raise ValueError("Cleaning region cannot contain the full sponge and 0.1m wipe")
    start_center = [min(max(center[i], low[i]), high[i] - (.1 if i == axis else 0))
                    for i in (0, 1)]
    start = [origin[i] + start_center[i] - center[i] for i in (0, 1)]
    start.append(origin[2] + zone[1][2] - box[0][2])
    end = start.copy()
    end[axis] += .1
    lifted_origin = [origin[0], origin[1], max(origin[2], start[2]) + .06]
    lifted_start, lifted_end = start.copy(), end.copy()
    lifted_start[2] = lifted_end[2] = lifted_origin[2]
    waypoints = []

    def segment(before, after, count, label, contact=False):
        for index in range(1, count + 1):
            waypoints.append({"label": f"m1_{label}_{index:02d}",
                              "position": [a + (b - a) * index / count for a, b in zip(before, after)],
                              "contact": contact})

    segment(origin, lifted_origin, 3, "lift")
    segment(lifted_origin, lifted_start, 3, "approach")
    segment(lifted_start, start, 2, "lower")
    segment(start, end, 4, "wipe", contact=True)
    segment(end, lifted_end, 1, "withdraw")
    segment(lifted_end, lifted_origin, 2, "return")
    segment(lifted_origin, origin, 1, "restore")
    return waypoints


def _list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value.tolist() if hasattr(value, "tolist") else list(value)


class CandidateEvents:
    """Own explicit event mutations while sharing the session's evidence state."""

    def __init__(self, session, parameters):
        self.session = session
        self.parameters = parameters
        self.contract = parameters["contract"]
        self.state = session.candidate_state
        self.control = self.state.setdefault("event_controller", {"h2_phase": "waiting_readiness"})

    @property
    def sim(self):
        return self.session.pipeline.og.sim

    def _lookup(self, name):
        return self.session.env.scene.object_registry("name", name)

    def _owned(self, name):
        declared = {spec["name"] for spec in self.parameters["objects"]}
        if name not in declared or name in self.contract.get("protected", []):
            raise ValueError(f"Authored event cannot move undeclared/protected object: {name}")
        obj = self._lookup(name)
        if obj is None:
            raise ValueError(f"Authored event object is missing: {name}")
        return obj

    def _set(self, obj, position, orientation):
        import torch
        obj.set_position_orientation(position=torch.tensor(position, dtype=torch.float32),
                                     orientation=torch.tensor(orientation, dtype=torch.float32))
        obj.keep_still()

    def _measure(self):
        from omnigibson.object_states import Touching
        supports = {region["support"] for region in self.state["regions"].values()}
        hands = getattr(self.session.robot, "_ag_obj_in_hand", None)
        held = {obj.name for obj in hands.values() if obj is not None} if isinstance(hands, dict) else None
        objects = {}
        for spec in self.parameters["objects"]:
            obj = self._lookup(spec["name"])
            if obj is None:
                continue
            position, orientation = [_list(value) for value in obj.get_position_orientation()]
            contacts = []
            for name in supports:
                support = self._lookup(name)
                if support is not None and support is not obj and obj.states[Touching].get_value(support):
                    contacts.append(name)
            objects[obj.name] = {"position": position, "orientation": orientation,
                "aabb": [_list(value) for value in obj.aabb], "contacts": contacts,
                "held": obj.name in held if held is not None else None,
                "linear_speed": math.sqrt(sum(x * x for x in _list(obj.get_linear_velocity()))),
                "angular_speed": math.sqrt(sum(x * x for x in _list(obj.get_angular_velocity())))}
        return {"time_s": float(self.sim.current_time), "objects": objects,
                "regions": deepcopy(self.state["regions"]), "events": deepcopy(self.state["events"]),
                "observed": deepcopy(self.state["observed"])}

    def _receipt(self, status="not_applicable", *, delivered=False):
        return {"schema": "wia.candidate_authored_event.v1", "status": status,
                "events": [], "frames": [], "model_calls": 0, "robot_actions": 0,
                "authored_intervention": False, "history_delivered_to_policy": delivered,
                "runtime_hook_installed": False}

    def _capture(self, receipt, callback, label):
        observation = callback(label)
        receipt["frames"].append({"label": label, "time_s": float(self.sim.current_time),
                                  "observation_receipt": observation})

    def _emit(self, kind, **fields):
        event = {"type": kind, "time_s": float(self.sim.current_time), "authored": True, **fields}
        self.state["events"].append(event)
        return deepcopy(event)

    def _place_in_region(self, name, region_name, *, on_first_step=None):
        obj = self._owned(name)
        position, orientation = [_list(value) for value in obj.get_position_orientation()]
        box = [_list(value) for value in obj.aabb]
        zone = self.state["regions"][region_name]
        low, high = zone["aabb"]
        destination = [position[i] + (low[i] + high[i] - box[0][i] - box[1][i]) / 2 for i in (0, 1)]
        destination.append(position[2] + high[2] - box[0][2] + .001)
        self._set(obj, destination, orientation)
        for index in range(4):
            self.sim.step()
            if index == 0 and on_first_step is not None:
                on_first_step()
        self.state["placements"][name] = {"support": zone["support"], "authored_event": True}

    def _apply_h2(self, receipt):
        if self.control["h2_phase"] != "pending":
            receipt["reason"] = "No measured post-delivery event is pending"
            return
        before = self._measure()
        if self.contract["disturbance"]:
            self._place_in_region(self.contract["obstruction"], self.contract.get("work_region", "work"))
        event = self._emit("post_delivery_check", disturbance=bool(self.contract["disturbance"]))
        self.control.update(h2_phase="emitted", h2_event_time_s=event["time_s"])
        receipt.update(status="applied", authored_intervention=True)
        receipt["events"].append(event)
        receipt["preservation_valid"] = _preserved([before, self._measure()],
            [spec["name"] for spec in self.parameters["objects"] if spec["name"] != self.contract["obstruction"]])

    def apply_preview(self, kind: str, capture_callback: Callable[[str], object | None]) -> dict:
        """Capture an authored review prelude; never append policy observed tags."""
        if kind not in ("H2", "M1", "H3"):
            raise ValueError(f"Unknown authored event preview: {kind}")
        if self.parameters["prototype"] != kind:
            raise ValueError("Preview kind must match the candidate prototype")
        receipt = self._receipt()
        if kind == "H2":
            if self.control["h2_phase"] != "pending":
                receipt["reason"] = "No measured post-delivery event is pending"
                return receipt
            self._capture(receipt, capture_callback, "h2_before_event")
            self._apply_h2(receipt)
            self._capture(receipt, capture_callback, "h2_after_event")
            return receipt
        if kind == "H3":
            before = self._measure()
            if not stock_undo_applicable(before, self.contract):
                receipt["reason"] = "Actual completed stock is below two or item_02 is not stocked"
                return receipt
            self._capture(receipt, capture_callback, "h3_before_external_removal")
            def removed():
                receipt["events"].append(self._emit("external_removal", object="item_02"))

            self._place_in_region("item_02", self.contract.get("supply_region", "supply"), on_first_step=removed)
            self._capture(receipt, capture_callback, "h3_after_external_removal")
            receipt.update(status="applied", authored_intervention=True)
            receipt["preservation_valid"] = _preserved([before, self._measure()],
                [spec["name"] for spec in self.parameters["objects"] if spec["name"] != "item_02"])
            return receipt
        return self._m1_preview(receipt, capture_callback)

    def _m1_preview(self, receipt, callback):
        from omnigibson.object_states import Touching
        used = self.contract["used_tool"]
        if len(self.contract["tools"]) != 2 or used not in self.contract["tools"]:
            raise ValueError("M1 requires two declared tools and a selected used_tool")
        source_support = self.state["placements"][used]["support"]
        surface = self.state["regions"]["cleaning_surface"]
        if source_support != surface["support"]:
            receipt["reason"] = "Cleaning surface must use the source tools' work table"
            return receipt
        obj = self._owned(used)
        support = self._lookup(surface["support"])
        if support is None:
            receipt["reason"] = "Native cleaning support is missing"
            return receipt
        original_position, orientation = [_list(value) for value in obj.get_position_orientation()]
        waypoints = wipe_waypoints(original_position, [_list(value) for value in obj.aabb], surface)
        before = self._measure()
        other_names = [spec["name"] for spec in self.parameters["objects"] if spec["name"] != used]
        self._capture(receipt, callback, "m1_before_use")
        contact_event = None
        preservation = True
        try:
            for waypoint in waypoints:
                self._set(obj, waypoint["position"], orientation)
                self.sim.step()
                current = self._measure()
                if waypoint["contact"]:
                    measured_box = current["objects"][used]["aabb"]
                    in_patch = all(measured_box[0][i] >= surface["aabb"][0][i] - 1e-9
                                   and measured_box[1][i] <= surface["aabb"][1][i] + 1e-9 for i in (0, 1))
                    valid = cleaning_contact_valid(touching=bool(obj.states[Touching].get_value(support)),
                        bottom_z=float(_list(obj.aabb[0])[2]), support_z=float(surface["aabb"][1][2]))
                    valid = valid and in_patch
                    if contact_event is None or not contact_event["contact_valid"]:
                        contact_event = self._emit("cleaning_contact", object=used,
                            surface=surface["support"], contact_valid=valid,
                            contamination_physics_claimed=False)
                        receipt["events"].append(contact_event)
                preserved = _preserved([before, current], other_names)
                if preserved is not True:
                    preservation = preserved
                    raise RuntimeError("An unowned object moved during the authored use preview")
                self._capture(receipt, callback, waypoint["label"])
        finally:
            self._set(obj, original_position, orientation)
        self._capture(receipt, callback, "m1_after_return")
        receipt.update(status="applied", authored_intervention=True,
                       waypoint_count=len(waypoints), wiping_distance_m=.1,
                       contact_valid=contact_event is not None and contact_event["contact_valid"],
                       preservation_valid=preservation)
        return receipt

    def _observe(self, tag):
        if tag not in self.state["observed"]:
            self.state["observed"].append(tag)
            return tag
        return None

    def on_observation_delivered(self, frame: dict, visible_names: set[str]) -> dict:
        """Advance pending runtime events only at a trusted delivered-view boundary.

        Event application happens after this observation. A later observation
        must show the marked work area before workspace_after_event is recorded.
        M1's use tag requires the actual contact-time view, not a returned sponge.
        """
        receipt = self._receipt("observed", delivered=True)
        receipt["observed_tags"] = []
        prototype = self.parameters["prototype"]
        time = _number(frame.get("time_s"))
        if time is None:
            receipt.update(status="unavailable", reason="Delivered observation lacks native simulation time")
            return receipt
        previous_time = self.control.get("last_delivered_time_s")
        if previous_time is not None and time <= previous_time:
            receipt.update(status="duplicate_observation", reason="Observation time did not advance")
            return receipt
        self.control["last_delivered_time_s"] = time
        if prototype == "H2":
            phase = self.control["h2_phase"]
            if phase == "pending":
                self._apply_h2(receipt)
            elif phase == "emitted":
                if (time > self.control["h2_event_time_s"]
                        and region_visible(visible_names, self.contract.get("work_region", "work"))):
                    tag = self._observe("workspace_after_event")
                    if tag:
                        receipt["observed_tags"].append(tag)
            else:
                self.control["h2_phase"] = h2_transition(phase, ready=h2_readiness(frame, self.contract),
                    delivered=_placed(frame, self.contract["payload"], self.contract.get("delivery", "delivery")) is True)
        elif prototype == "M1":
            used = self.contract["used_tool"]
            for event in self.state["events"]:
                event_time = _number(event.get("time_s"))
                if (event.get("type") == "cleaning_contact" and event.get("object") == used
                        and event.get("contact_valid") is True and event_time is not None
                        and 0 <= time - event_time <= MAX_SAMPLE_GAP_S + 1e-9
                        and used in visible_names and region_visible(visible_names, "cleaning_surface")
                        and _placed(frame, used, "cleaning_surface") is True):
                    tag = self._observe(f"use:{used}")
                    if tag:
                        receipt["observed_tags"].append(tag)
        return receipt
