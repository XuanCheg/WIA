"""Measured HIGH candidate regressions. Run: PYTHONPATH=src pytest -q tests/test_candidate_high.py."""

from copy import deepcopy
import math

import pytest

from wia.candidates.high import evaluate


def region(x):
    return {"aabb": [[x, 0, -0.1], [x + 0.6, 0.6, 0]], "support": f"surface_{x}"}


def object_at(x, *, support=None, held=False):
    return {"position": [x + 0.1, 0.1, 0.05],
            "aabb": [[x + 0.05, 0.05, 0], [x + 0.15, 0.15, 0.1]],
            "orientation": [0, 0, 0, 1], "linear_speed": 0., "angular_speed": 0.,
            "held": held, "contacts": [support or f"surface_{x}"]}


def episode(objects, regions, *, duration=1.):
    return [{"time_s": round(i * .05, 8), "objects": deepcopy(objects),
             "regions": deepcopy(regions), "observed": [], "events": []}
            for i in range(round(duration / .05) + 1)]


def move(frames, name, x, *, time=.3):
    for sample in frames:
        if sample["time_s"] >= time:
            sample["objects"][name] = object_at(x)
        elif sample["time_s"] >= time - .05 - 1e-8:
            sample["objects"][name]["held"] = True


def observe(frames, tag, time=.1):
    next(sample for sample in frames if sample["time_s"] >= time)["observed"].append(tag)


def result(prototype, contract, frames):
    return evaluate(prototype, contract, frames[-1], frames[:-1])


def h1(defective=False):
    contract = {"selected": "package_a", "spares": ["spare_b", "spare_c"],
                "defective": defective, "compatible": ["spare_b", "spare_c"],
                "delivery": "delivery", "quarantine": "quarantine", "protected": []}
    frames = episode({"package_a": object_at(-1), "spare_b": object_at(-2),
                      "spare_c": object_at(-3)}, {"delivery": region(0), "quarantine": region(1)})
    observe(frames, "inspection:package_a")
    if defective:
        move(frames, "spare_c", 0)
        move(frames, "package_a", 1)
    else:
        move(frames, "package_a", 0)
    return contract, frames


@pytest.mark.parametrize("defective", [False, True])
def test_h1_good_and_bad_branch_with_nonarbitrary_compatible_spare(defective):
    contract, frames = h1(defective)
    evaluated = result("H1", contract, frames)
    assert evaluated["task_success"], evaluated
    assert evaluated["evaluation_validity"] == "complete"


def test_h1_requires_inspection_before_placement_and_keeps_spares():
    contract, frames = h1()
    frames[2]["observed"] = []
    observe(frames, "inspection:package_a", .5)
    evaluated = result("H1", contract, frames)
    assert evaluated["checks"]["inspection_before_delivery"] is False
    assert not evaluated["task_success"]
    observe(frames, "inspection:package_a", .1)
    frames[5]["objects"]["spare_b"]["position"][0] += .1
    assert result("H1", contract, frames)["checks"]["spares_preserved"] is False


@pytest.mark.parametrize("change", ["footprint", "held", "contact", "bottom", "speed"])
def test_full_measured_placement_rejects_center_only_held_or_unsupported(change):
    contract, frames = h1()
    for sample in frames[6:]:
        obj = sample["objects"]["package_a"]
        if change == "footprint":
            obj["aabb"][1][0] = .7
        elif change == "held":
            obj["held"] = True
        elif change == "contact":
            obj["contacts"] = []
        elif change == "bottom":
            obj["aabb"][0][2] = .03
        else:
            obj["angular_speed"] = .11
    evaluated = result("H1", contract, frames)
    assert evaluated["checks"]["selected_delivered"] is False
    assert not evaluated["task_success"]


def test_sparse_or_missing_native_stability_is_unavailable():
    contract, frames = h1()
    sparse = frames[::4]
    assert result("H1", contract, sparse)["evaluation_validity"] == "unavailable"
    del frames[-1]["objects"]["package_a"]["linear_speed"]
    assert result("H1", contract, frames)["evaluation_validity"] == "unavailable"


def h2(disturbance):
    contract = {"payload": "parcel", "obstruction": "block", "disturbance": disturbance,
                "work_region": "work", "delivery": "delivery", "storage": "storage"}
    frames = episode({"parcel": object_at(-1), "block": object_at(0)},
                     {"work": region(0), "delivery": region(1), "storage": region(2)}, duration=1.5)
    move(frames, "block", 2, time=.15)
    move(frames, "parcel", 1, time=.4)
    event_time = .75
    if disturbance:
        for sample in frames:
            if event_time <= sample["time_s"] < 1.:
                sample["objects"]["block"] = object_at(0)
        move(frames, "block", 2, time=1.)
    event = {"type": "post_delivery_check", "time_s": event_time, "authored": True}
    frames[15]["events"].append(event)
    observe(frames, "workspace_after_event", .85)
    return contract, frames


@pytest.mark.parametrize("disturbance", [False, True])
def test_h2_event_required_in_both_branches_without_extra_nodisturbance_move(disturbance):
    contract, frames = h2(disturbance)
    evaluated = result("H2", contract, frames)
    assert evaluated["task_success"], evaluated
    assert evaluated["action_cost"]["release_count"] == (3 if disturbance else 2)


def test_h2_does_not_accept_final_state_without_event_observation():
    contract, frames = h2(False)
    for sample in frames:
        sample["observed"] = []
    evaluated = result("H2", contract, frames)
    assert evaluated["checks"]["workspace_observed_after_event"] is False
    assert not evaluated["task_success"]
    frames[15]["events"] = []
    assert result("H2", contract, frames)["evaluation_validity"] == "complete"
    assert result("H2", contract, frames)["checks"]["post_delivery_sequence"] is False


def h3(initial_count):
    names = ["unit_a", "unit_b", "unit_c", "unit_d"]
    contract = {"stock": names, "target_count": 2, "semantics": "minimum",
                "stock_region": "stock", "supply_region": "supply", "protected": []}
    frames = episode({name: object_at(0 if index < initial_count else 1)
                      for index, name in enumerate(names)}, {"stock": region(0), "supply": region(1)})
    return contract, frames


def test_h3_completed_inventory_allows_true_noop():
    contract, frames = h3(2)
    evaluated = result("H3", contract, frames)
    assert evaluated["task_success"], evaluated
    assert evaluated["action_cost"]["completed_additions"] == 0
    assert evaluated["action_cost"]["object_path_length_m"] == 0


def test_h3_fill_actual_deficit_with_any_supply_identity():
    contract, frames = h3(1)
    move(frames, "unit_d", 0)
    evaluated = result("H3", contract, frames)
    assert evaluated["task_success"], evaluated
    assert evaluated["action_cost"]["completed_additions"] == 1
    move(frames, "unit_c", 0)
    evaluated = result("H3", contract, frames)
    assert not evaluated["task_success"]
    assert evaluated["action_cost"]["unnecessary_additions"] == 1


def test_h3_completed_units_cannot_be_replaced_without_authored_removal():
    contract, frames = h3(2)
    move(frames, "unit_a", 1)
    move(frames, "unit_c", 0)
    evaluated = result("H3", contract, frames)
    assert not evaluated["task_success"]
    assert evaluated["checks"]["initial_inventory_retained"] is False
    for sample in frames:
        sample["objects"]["unit_a"]["held"] = False
    event = {"type": "external_removal", "time_s": .3, "object": "unit_a", "authored": True}
    frames[6]["events"].append(event)
    assert result("H3", contract, frames)["task_success"]


def test_h3_removed_native_object_may_be_absent_but_authored_event_must_be_real():
    contract, frames = h3(2)
    for sample in frames[6:]:
        del sample["objects"]["unit_a"]
    frames[6]["events"].append({"type": "external_removal", "time_s": .3,
                                "object": "unit_a", "authored": True})
    move(frames, "unit_d", 0)
    assert result("H3", contract, frames)["task_success"]
    contract, frames = h3(2)
    frames[6]["events"].append({"type": "external_removal", "time_s": .3,
                                "object": "unit_a", "authored": True})
    move(frames, "unit_d", 0)
    assert result("H3", contract, frames)["checks"]["authored_removals"] is False


def test_h3_later_removal_does_not_justify_earlier_surplus():
    contract, frames = h3(2)
    move(frames, "unit_c", 0, time=.15)
    for sample in frames:
        if sample["time_s"] >= .8:
            del sample["objects"]["unit_a"]
    frames[16]["events"].append({"type": "external_removal", "time_s": .8,
                                 "object": "unit_a", "authored": True})
    evaluated = result("H3", contract, frames)
    assert evaluated["checks"]["no_unnecessary_additions"] is False
    assert evaluated["action_cost"]["unnecessary_additions"] == 1


def h4():
    contract = {"target": "item", "blockers": ["block_a", "block_b"], "anchors": ["fixed"],
                "target_region": "target", "storage_region": "storage"}
    frames = episode({"item": object_at(-1), "block_a": object_at(0),
                      "block_b": object_at(0), "fixed": object_at(-2)},
                     {"target": region(0), "storage": region(1)})
    move(frames, "item", 0, time=.5)
    move(frames, "block_a", 1)
    move(frames, "block_b", 1)
    frames[-1]["process_evidence"] = {"hands_withdrawn": True}
    return contract, frames


def test_h4_all_blockers_removed_is_success_with_separate_efficiency():
    contract, frames = h4()
    evaluated = result("H4", contract, frames)
    assert evaluated["task_success"], evaluated
    assert evaluated["action_cost"]["relocation_count"] == 2


def test_h4_requires_measured_motion_proof_and_anchor_preservation():
    contract, frames = h4()
    del frames[-1]["process_evidence"]
    evaluated = result("H4", contract, frames)
    assert evaluated["evaluation_validity"] == "unavailable"
    assert not evaluated["task_success"]
    frames[-1]["process_evidence"] = {"hands_withdrawn": True}
    frames[10]["objects"]["fixed"]["orientation"] = [0, 0, math.sin(.1), math.cos(.1)]
    assert result("H4", contract, frames)["checks"]["anchors_preserved"] is False


def h5(primary_available):
    contract = {"payload": "item", "resident": "resident", "primary_available": primary_available,
                "primary": "primary", "fallback": "fallback", "protected": ["resident"]}
    frames = episode({"item": object_at(-1), "resident": object_at(-2 if primary_available else 0)},
                     {"primary": region(0), "fallback": region(1)})
    observe(frames, "destination_state")
    move(frames, "item", 0 if primary_available else 1)
    return contract, frames


@pytest.mark.parametrize("primary_available", [False, True])
def test_h5_branch_correct_preference_and_observation(primary_available):
    contract, frames = h5(primary_available)
    assert result("H5", contract, frames)["task_success"]


def test_h5_rejects_fallback_when_primary_is_clear_and_late_observation():
    contract, frames = h5(True)
    move(frames, "item", 1)
    assert not result("H5", contract, frames)["task_success"]
    move(frames, "item", 0)
    frames[2]["observed"] = []
    observe(frames, "destination_state", .4)
    assert result("H5", contract, frames)["checks"]["destination_observed_before_release"] is False


def test_h5_even_partial_resident_overlap_procedurally_excludes_primary():
    contract, frames = h5(False)
    for sample in frames:
        sample["objects"]["resident"] = object_at(.5)
        sample["objects"]["resident"]["aabb"][1][0] = .7
    assert result("H5", contract, frames)["task_success"]


def test_evaluator_is_pure_and_rejects_unknown_prototype():
    contract, frames = h1()
    original = deepcopy((contract, frames))
    evaluated = result("H1", contract, frames)
    assert (contract, frames) == original
    assert evaluated["schema"] == "wia.high_candidate_evaluation.v1"
    with pytest.raises(ValueError, match="Unknown HIGH"):
        result("H9", contract, frames)
