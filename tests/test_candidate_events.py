"""Pure authored-event gates. Run: PYTHONPATH=src pytest -q tests/test_candidate_events.py."""

from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from wia.candidates.events import (
    CandidateEvents, cleaning_contact_valid, h2_readiness, h2_transition,
    region_visible, stock_undo_applicable, wipe_waypoints,
)


def region(x):
    return {"aabb": [[x, 0, 0], [x + .5, .5, 0]], "support": "table"}


def placed(x):
    return {"position": [x + .1, .1, .025], "orientation": [0, 0, 0, 1],
            "aabb": [[x + .05, .05, 0], [x + .15, .15, .05]],
            "contacts": ["table"], "held": False, "linear_speed": 0., "angular_speed": 0.}


def h2_frame(time, *, ready=False, delivered=False):
    return {"time_s": time, "regions": {"work": region(0), "storage": region(1), "delivery": region(2)},
            "objects": {"block": placed(1 if ready else 0), "payload": placed(2 if delivered else -1)},
            "observed": [], "events": []}


H2_CONTRACT = {"obstruction": "block", "payload": "payload", "disturbance": True,
               "work_region": "work", "storage": "storage", "delivery": "delivery"}


def controller(prototype, contract, frame):
    parameters = {"prototype": prototype, "contract": contract,
                  "objects": [{"name": name} for name in frame["objects"]]}
    state = {"parameters": parameters, "regions": deepcopy(frame["regions"]),
             "placements": {name: {"support": "table"} for name in frame["objects"]},
             "observed": [], "events": []}
    return CandidateEvents(SimpleNamespace(candidate_state=state), parameters)


def test_h2_requires_measured_clear_and_supported_released_storage():
    frame = h2_frame(0, ready=True)
    assert h2_readiness(frame, H2_CONTRACT)
    frame["objects"]["block"]["held"] = True
    assert not h2_readiness(frame, H2_CONTRACT)
    frame["objects"]["block"]["held"] = False
    frame["objects"]["block"]["contacts"] = []
    assert not h2_readiness(frame, H2_CONTRACT)
    del frame["objects"]["block"]["angular_speed"]
    assert not h2_readiness(frame, H2_CONTRACT)


def test_h2_transition_requires_three_distinct_boundaries():
    phase = h2_transition("waiting_readiness", ready=False, delivered=True)
    assert phase == "waiting_readiness"
    phase = h2_transition(phase, ready=True, delivered=True)
    assert phase == "waiting_delivery"
    phase = h2_transition(phase, ready=True, delivered=True)
    assert phase == "pending"
    assert h2_transition(phase, ready=True, delivered=True) == "emitted"
    assert h2_transition("emitted", ready=True, delivered=True) == "emitted"


def test_runtime_h2_does_not_emit_during_delivery_observation(monkeypatch):
    events = controller("H2", H2_CONTRACT, h2_frame(0))
    applications = []

    def apply(receipt):
        applications.append(True)
        events.control.update(h2_phase="emitted", h2_event_time_s=3.)
        receipt.update(authored_intervention=True, status="applied")

    monkeypatch.setattr(events, "_apply_h2", apply)
    events.on_observation_delivered(h2_frame(0, delivered=True), {"zone_work"})
    assert events.control["h2_phase"] == "waiting_readiness"
    events.on_observation_delivered(h2_frame(1, ready=True), {"zone_work"})
    assert events.control["h2_phase"] == "waiting_delivery"
    duplicate = events.on_observation_delivered(h2_frame(1, ready=True, delivered=True), {"zone_work"})
    assert duplicate["status"] == "duplicate_observation"
    events.on_observation_delivered(h2_frame(2, ready=True, delivered=True), {"zone_work"})
    assert events.control["h2_phase"] == "pending"
    assert not applications
    events.on_observation_delivered(h2_frame(3, ready=True, delivered=True), {"zone_work"})
    assert applications == [True]
    assert not events.state["observed"]
    events.on_observation_delivered(h2_frame(4), {"table", "block"})
    assert not events.state["observed"]
    receipt = events.on_observation_delivered(h2_frame(5), {"zone_work"})
    assert receipt["observed_tags"] == ["workspace_after_event"]
    assert events.state["observed"] == ["workspace_after_event"]
    events.on_observation_delivered(h2_frame(6), {"zone_work"})
    assert len(events.state["observed"]) == 1


def test_h2_preview_cannot_fabricate_robot_readiness_or_policy_history():
    events = controller("H2", H2_CONTRACT, h2_frame(0))
    captures = []
    before = deepcopy(events.state)
    receipt = events.apply_preview("H2", captures.append)
    assert receipt["status"] == "not_applicable"
    assert not captures
    assert not receipt["history_delivered_to_policy"]
    assert receipt["robot_actions"] == receipt["model_calls"] == 0
    assert events.state == before


def test_h3_undo_requires_two_actual_complete_units_and_specific_source():
    contract = {"stock": ["item_01", "item_02", "item_03"], "stock_region": "stock", "supply_region": "supply"}
    frame = {"time_s": 0, "regions": {"stock": region(0), "supply": region(1)},
             "objects": {"item_01": placed(0), "item_02": placed(1), "item_03": placed(1)}}
    assert not stock_undo_applicable(frame, contract)
    frame["objects"]["item_03"] = placed(0)
    assert not stock_undo_applicable(frame, contract)
    frame["objects"]["item_02"] = placed(0)
    assert stock_undo_applicable(frame, contract)
    frame["objects"]["item_02"]["held"] = True
    assert not stock_undo_applicable(frame, contract)


def test_h3_initial_stock_one_preview_is_not_applicable(monkeypatch):
    contract = {"stock": ["item_01", "item_02"], "stock_region": "stock", "supply_region": "supply"}
    frame = {"time_s": 0, "regions": {"stock": region(0), "supply": region(1)},
             "objects": {"item_01": placed(0), "item_02": placed(1)}}
    events = controller("H3", contract, frame)
    monkeypatch.setattr(events, "_measure", lambda: deepcopy(frame))
    captures = []
    receipt = events.apply_preview("H3", captures.append)
    assert receipt["status"] == "not_applicable"
    assert receipt["events"] == []
    assert not captures
    assert events.state["events"] == events.state["observed"] == []


@pytest.mark.parametrize("touching,bottom,expected", [(True, 0., True), (True, .014, True),
                                                       (True, .016, False), (False, 0., False)])
def test_contact_cannot_be_claimed_from_geometry_alone(touching, bottom, expected):
    assert cleaning_contact_valid(touching=touching, bottom_z=bottom, support_z=0.) is expected


def test_wipe_path_is_bounded_continuous_and_returns_the_same_object_pose():
    position = [.1, .1, .025]
    box = [[.05, .05, 0], [.15, .15, .05]]
    zone = {"aabb": [[.15, 0, 0], [.5, .3, 0]], "support": "table"}
    original = deepcopy((position, box, zone))
    path = wipe_waypoints(position, box, zone)
    assert len(path) == 16
    assert path[-1]["position"] == position
    contact_points = [point["position"] for point in path if point["contact"]]
    before_wipe = path[7]["position"]
    distance = sum(math.dist(a, b) for a, b in zip([before_wipe, *contact_points[:-1]], contact_points))
    assert distance == pytest.approx(.1)
    assert all(point[2] == pytest.approx(.025) for point in contact_points)
    assert max(math.dist(a["position"], b["position"]) for a, b in zip(path, path[1:])) < .11
    for point in contact_points:
        assert .15 <= point[0] - .05 and point[0] + .05 <= .5 + 1e-9
        assert 0 <= point[1] - .05 and point[1] + .05 <= .3 + 1e-9
    assert (position, box, zone) == original


def test_wipe_planner_rejects_insufficient_real_footprint_space():
    with pytest.raises(ValueError, match="full sponge"):
        wipe_waypoints([.1, .1, .025], [[.05, .05, 0], [.15, .15, .05]],
                       {"aabb": [[0, 0, 0], [.15, .15, 0]], "support": "table"})


def test_visible_support_alone_does_not_prove_workspace_view():
    assert not region_visible({"table", "block"}, "work")
    assert region_visible({"zone_work"}, "work")
    assert region_visible({"work"}, "work")


def test_m1_use_observation_requires_contact_time_identity_and_visible_surface():
    contract = {"tools": ["tool_a", "tool_b"], "used_tool": "tool_b"}
    frame = {"time_s": .5, "regions": {"cleaning_surface": region(0)},
             "objects": {"tool_a": placed(1), "tool_b": placed(0)}}
    events = controller("M1", contract, frame)
    events.state["events"].append({"type": "cleaning_contact", "object": "tool_b",
                                    "time_s": .5, "contact_valid": True, "authored": True})
    receipt = events.on_observation_delivered(frame, {"tool_a", "zone_cleaning_surface"})
    assert not receipt["observed_tags"]
    frame["time_s"] = .55
    receipt = events.on_observation_delivered(frame, {"tool_b", "zone_cleaning_surface"})
    assert receipt["observed_tags"] == ["use:tool_b"]
    assert receipt["history_delivered_to_policy"]


def test_m1_returned_or_late_tool_view_cannot_create_observed_use():
    contract = {"tools": ["tool_a", "tool_b"], "used_tool": "tool_a"}
    frame = {"time_s": .7, "regions": {"cleaning_surface": region(0)},
             "objects": {"tool_a": placed(0), "tool_b": placed(1)}}
    events = controller("M1", contract, frame)
    events.state["events"].append({"type": "cleaning_contact", "object": "tool_a",
                                    "time_s": .5, "contact_valid": True, "authored": True})
    assert not events.on_observation_delivered(frame, {"tool_a", "zone_cleaning_surface"})["observed_tags"]
    frame["time_s"] = .75
    frame["objects"]["tool_a"] = placed(1)
    events.state["events"][-1]["time_s"] = .75
    assert not events.on_observation_delivered(frame, {"tool_a", "zone_cleaning_surface"})["observed_tags"]


def test_wipe_path_avoids_spare_tool_without_moving_it():
    position = [0., 0., .0175]
    box = [[-.04, -.0275, 0.], [.04, .0275, .035]]
    spare = [[.10, -.0275, 0.], [.18, .0275, .035]]
    zone = {'aabb': [[-.04, -.1, 0.], [.28, .1, 0.]], 'support': 'table'}
    path = wipe_waypoints(position, box, zone, obstacles=[spare])
    wipe = [path[7], *[p for p in path if p['contact']]]
    assert sum(math.dist(a['position'], b['position']) for a, b in zip(wipe, wipe[1:])) == pytest.approx(.1)
    for point in wipe:
        p = point['position']
        assert not all(p[i] + box[1][i] > spare[0][i] and
                       p[i] + box[0][i] < spare[1][i] for i in (0, 1))
    assert len(path) == 16 and path[-1]['position'] == position
