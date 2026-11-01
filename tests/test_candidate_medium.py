"""Check measured M1--M4 contracts. Run: PYTHONPATH=src pytest -q tests/test_candidate_medium.py."""
from copy import deepcopy
import math

import pytest

from wia.candidates.medium import evaluate, enumerate_allocations, enumerate_schedules


def region(x):
    return {"aabb": [[x - .45, -.45, -.1], [x + .45, .45, 0.]], "support": f"support_{x}"}


def obj(x, *, held=False, native_id=None):
    value = {"position": [x, 0., .05], "orientation_xyzw": [0., 0., 0., 1.],
             "aabb": [[x - .05, -.05, 0.], [x + .05, .05, .1]],
             "linear_speed": 0., "angular_speed": 0., "held": held, "contacts": [f"support_{x}"]}
    if native_id is not None:
        value["native_id"] = native_id
    return value


def frame(t, objects, regions, **extra):
    return {"time_s": t, "objects": deepcopy(objects), "regions": deepcopy(regions),
            "observed": [], "events": [], **extra}


def score(prototype, contract, frames):
    return evaluate(prototype, contract, frames[-1], frames[:-1])


def m1():
    contract = {"tools": ["brush_a", "brush_b"], "used_tool": "brush_a", "used_region": "used",
                "clean_region": "clean", "protected": []}
    regions = {"used": region(1), "clean": region(2)}
    frames = [frame(t, {"brush_a": obj(1, native_id="native-a"), "brush_b": obj(2, native_id="native-b")}, regions)
              for t in (0., .05, .1, .15, .2, .25, .3, .35)]
    frames[1]["events"] = [{"type": "cleaning_contact", "object": "brush_a", "contact_valid": True, "time_s": .05}]
    frames[1]["observed"] = ["use:brush_a"]
    return contract, frames


def test_m1_real_use_and_final_dwell_are_read_only():
    contract, frames = m1()
    before = deepcopy((contract, frames))
    result = score("M1", contract, frames)
    assert result["schema"] == "wia.candidate_medium.v1"
    assert result["evaluation_validity"] == "complete" and result["task_success"]
    assert (contract, frames) == before


@pytest.mark.parametrize("change,check", [
    ("no_event", "real_cleaning_contact"), ("no_contact", "real_cleaning_contact"),
    ("wrong_object", "real_cleaning_contact"), ("no_observation", "observed_use"),
    ("wrong_observation", "observed_use"), ("changed_identity", "identity_continuity"),
])
def test_m1_rejects_event_and_identity_shortcuts(change, check):
    contract, frames = m1()
    if change == "no_event":
        frames[1]["events"] = []
    elif change == "no_contact":
        frames[1]["events"][0]["contact_valid"] = False
    elif change == "wrong_object":
        frames[1]["events"][0]["object"] = "brush_b"
    elif change == "no_observation":
        frames[1]["observed"] = []
    elif change == "wrong_observation":
        frames[1]["observed"] = ["use:brush_b"]
    else:
        frames[-1]["objects"]["brush_a"]["native_id"] = "native-b"
    result = score("M1", contract, frames)
    assert not result["task_success"] and result["checks"][check] is not True


@pytest.mark.parametrize("change", ["footprint", "unsupported", "bottom", "held", "fast", "short", "gap", "missing"])
def test_placement_requires_physical_stable_support(change):
    contract, frames = m1()
    if change == "footprint":
        frames[-1]["objects"]["brush_a"]["aabb"][1][0] = 1.6
    elif change == "unsupported":
        frames[-1]["objects"]["brush_a"]["contacts"] = []
    elif change == "bottom":
        frames[-1]["objects"]["brush_a"]["aabb"][0][2] = .016
    elif change == "held":
        frames[-1]["objects"]["brush_a"]["held"] = True
    elif change == "fast":
        frames[-1]["objects"]["brush_a"]["linear_speed"] = .021
    elif change == "short":
        frames = frames[:5]
    elif change == "gap":
        frames = [frames[0], frames[1], *frames[5:]]
    else:
        del frames[-1]["objects"]["brush_a"]["angular_speed"]
    result = score("M1", contract, frames)
    assert not result["task_success"]
    if change in {"missing", "gap"}:
        assert result["evaluation_validity"] == "unavailable"


def test_reversed_native_history_and_transient_protected_motion_do_not_pass():
    contract, frames = m1()
    frames[3]["time_s"] = .04
    assert score("M1", contract, frames)["checks"]["native_clock"] is None
    contract, frames = m1()
    contract["protected"] = ["fixed"]
    for sample in frames:
        sample["objects"]["fixed"] = obj(3)
    frames[3]["objects"]["fixed"]["position"][0] += .021
    assert score("M1", contract, frames)["checks"]["protected_unchanged"] is False
    frames[3]["objects"]["fixed"] = obj(3)
    angle = math.radians(6)
    frames[3]["objects"]["fixed"]["orientation_xyzw"] = [0., 0., math.sin(angle / 2), math.cos(angle / 2)]
    assert score("M1", contract, frames)["checks"]["protected_unchanged"] is False
    del frames[3]["objects"]["fixed"]["orientation_xyzw"]
    assert score("M1", contract, frames)["evaluation_validity"] == "unavailable"


def allocation_fixture(assignment, orders=None):
    quantities = {"three": 3, "two_a": 2, "two_b": 2, "one": 1}
    orders = orders or {"order_a": 4, "order_b": 3}
    regions = {"order_a": region(1), "order_b": region(2), "stock": region(3)}
    where = {"order_a": 1, "order_b": 2, "stock": 3}
    frames = [frame(t, {name: obj(where[assignment[name]]) for name in quantities}, regions)
              for t in (0., .1, .2, .3)]
    return {"units": quantities, "orders": orders, "stock_region": "stock", "protected": []}, frames


def test_m2_all_interchangeable_allocations_and_counterfactual():
    quantities = {"three": 3, "two_a": 2, "two_b": 2, "one": 1}
    base = enumerate_allocations(quantities, {"order_a": 4, "order_b": 3})
    changed = enumerate_allocations(quantities, {"order_a": 5, "order_b": 3})
    assert len(base) == 1 and len(changed) == 3
    assert base[0] == {"three": "order_b", "two_a": "order_a", "two_b": "order_a", "one": "stock"}
    assert all(assignment not in changed for assignment in base)
    assert len({tuple(a.items()) for a in changed}) == len(changed)
    for assignment in base:
        contract, frames = allocation_fixture(assignment)
        assert score("M2", contract, frames)["task_success"]
        contract["orders"]["order_a"] = 5
        assert not score("M2", contract, frames)["task_success"]
    for assignment in changed:
        contract, frames = allocation_fixture(assignment, {"order_a": 5, "order_b": 3})
        assert score("M2", contract, frames)["task_success"]


def test_m2_greedy_a_fill_cannot_fill_b_and_cannot_split_units():
    greedy = {"three": "order_a", "two_a": "order_b", "two_b": "stock", "one": "order_a"}
    contract, frames = allocation_fixture(greedy)
    result = score("M2", contract, frames)
    assert result["evidence"]["totals"] == {"order_a": 4, "order_b": 2}
    assert result["checks"]["exact_orders"] is False
    assert not enumerate_allocations({"one_unit": 3}, {"order_a": 1, "order_b": 2})


def m3(*, swapped=False, selected="unit_a"):
    contents = {"unit_a": "milk", "unit_b": "chicken"}
    if swapped:
        contents = {"unit_a": "chicken", "unit_b": "milk"}
    regions = {"unit_a": region(1), "unit_b": region(2)}
    objects = {content: obj(1 if unit == "unit_a" else 2) for unit, content in contents.items()}
    objects["cheese"] = obj(1 if selected == "unit_a" else 2)
    for unit, content in contents.items():
        objects[content]["inside"] = [unit]
    objects["cheese"]["inside"] = [selected]
    contract = {"payload": "cheese", "content_by_unit": contents, "eligible_content": ["milk"],
                "protected": ["milk", "chicken"], "require_closed": True}
    frames = [frame(t, objects, regions, equipment={unit: {"closed": True} for unit in contents},
                    observed=[f"contents:{selected}"]) for t in (0., .1, .2, .3)]
    return contract, frames


def test_m3_actual_contents_change_destination_without_fixed_inspection_order():
    contract, frames = m3()
    assert score("M3", contract, frames)["task_success"]
    contract, frames = m3(swapped=True, selected="unit_a")
    assert not score("M3", contract, frames)["task_success"]
    contract, frames = m3(swapped=True, selected="unit_b")
    assert score("M3", contract, frames)["task_success"]
    assert frames[0]["observed"] == ["contents:unit_b"]


def test_m3_unplaced_initial_frame_is_measured_incomplete():
    contract, frames = m3()
    initial = frames[0]
    initial['objects']['cheese'] = obj(4)
    initial['objects']['cheese']['inside'] = []
    result = evaluate('M3', contract, initial, [])
    assert result['evaluation_validity'] == 'complete'
    assert result['checks']['history_baseline'] is True
    assert result['checks']['payload_unique_unit'] is False
    assert result['checks']['payload_inside'] is False
    assert result['checks']['observed_selected_contents'] is False
    assert not result['task_success']


def test_m3_nominal_mapping_open_door_and_unobserved_contents_cannot_pass():
    contract, frames = m3()
    for sample in frames:
        sample["objects"]["milk"], sample["objects"]["chicken"] = obj(2), obj(1)
    result = score("M3", contract, frames)
    assert result["checks"]["actual_contents:unit_a"] is False
    contract, frames = m3()
    frames[-1]["equipment"]["unit_b"]["closed"] = False
    assert score("M3", contract, frames)["checks"]["closed:unit_b"] is False
    contract, frames = m3()
    for sample in frames:
        sample["observed"] = []
    assert score("M3", contract, frames)["checks"]["observed_selected_contents"] is False


@pytest.mark.parametrize("name,check", [("cheese", "payload_inside"), ("milk", "resident_inside:unit_a")])
def test_m3_supported_on_top_is_not_native_storage(name, check):
    contract, frames = m3()
    for sample in frames:
        # Even valid support contact and a full XY footprint do not prove Inside.
        sample["objects"][name]["inside"] = []
    result = score("M3", contract, frames)
    assert result["checks"][check] is False
    assert not result["task_success"]
    assert result["evaluation_validity"] == "complete"


@pytest.mark.parametrize("name,check", [("cheese", "payload_inside"), ("chicken", "resident_inside:unit_b")])
def test_m3_missing_native_inside_evidence_is_unavailable(name, check):
    contract, frames = m3()
    for sample in frames:
        del sample["objects"][name]["inside"]
    result = score("M3", contract, frames)
    assert result["checks"][check] is None
    assert result["evaluation_validity"] == "unavailable" and not result["task_success"]


def test_m3_native_inside_must_cover_payload_final_dwell():
    contract, frames = m3()
    frames[0]["objects"]["cheese"]["inside"] = []
    assert score("M3", contract, frames)["checks"]["payload_inside"] is False


def timed_trajectory(jobs, *, duration_override=None, native_origin=10.):
    """Construct measured CPU fixtures; these are not simulator executions."""
    regions = {"station_a": region(1), "station_b": region(2), "delivery": region(3), "stock": region(4)}
    objects = {name: obj(4) for name in jobs}
    frames = [frame(native_origin, objects, regions)]

    def tick(count=1):
        for _ in range(count):
            frames.append(frame(round(frames[-1]["time_s"] + .05, 8), objects, regions))

    for name, spec in jobs.items():
        objects[name] = obj(4, held=True)
        tick(2)
        objects[name] = obj(1 if spec["station"] == "station_a" else 2)
        tick()
        duration = spec["duration_s"] if duration_override is None else duration_override
        tick(math.ceil(duration / .05))
        objects[name]["held"] = True
        tick(2)
        objects[name] = obj(3)
        tick(7)
    contract = {"jobs": jobs, "delivery_region": "delivery", "deadline_s": 60., "handling_s": 2., "protected": []}
    return contract, frames


def test_m4_native_clock_transport_processing_and_provisional_deadline():
    jobs = {"a": {"station": "station_a", "duration_s": 12.},
            "b": {"station": "station_a", "duration_s": 4.},
            "c": {"station": "station_b", "duration_s": 8.}}
    contract, frames = timed_trajectory(jobs)
    result = score("M4", contract, frames)
    assert result["task_success"], result["reasons"]
    assert result["evidence"]["elapsed_simulation_s"] < 60
    assert result["evidence"]["abstract_schedule"]["native_motor_timing_certified"] is False
    contract["deadline_s"] = .5
    assert score("M4", contract, frames)["checks"]["deadline"] is False


def test_m4_early_removal_resets_processing_and_cannot_deliver():
    contract, frames = timed_trajectory({"a": {"station": "station_a", "duration_s": 1.}}, duration_override=.4)
    result = score("M4", contract, frames)
    assert result["checks"]["processed:a"] is False
    assert result["evidence"]["jobs"]["a"]["resets"] == 1
    assert not result["task_success"]


def test_m4_reset_does_not_add_noncontiguous_processing():
    jobs = {"a": {"station": "station_a", "duration_s": 1.}}
    contract, first = timed_trajectory(jobs, duration_override=.6)
    _, second = timed_trajectory(jobs, duration_override=.6, native_origin=first[-1]["time_s"] + .05)
    result = score("M4", contract, [*first, *second])
    assert result["checks"]["processed:a"] is False
    assert result["evidence"]["jobs"]["a"]["resets"] == 2


def test_m4_resource_violations_and_missing_transport_are_not_success():
    jobs = {"a": {"station": "station_a", "duration_s": .2},
            "b": {"station": "station_a", "duration_s": .2}}
    contract, frames = timed_trajectory(jobs)
    frames[4]["objects"]["b"] = obj(1)
    assert score("M4", contract, frames)["checks"]["exclusive_stations"] is False
    contract, frames = timed_trajectory(jobs)
    frames[1]["objects"]["b"]["held"] = True
    assert score("M4", contract, frames)["checks"]["single_held_object"] is False
    contract, frames = timed_trajectory({"a": jobs["a"]})
    for sample in frames:
        sample["objects"]["a"]["held"] = False
    result = score("M4", contract, frames)
    assert not result["task_success"] and result["evaluation_validity"] == "unavailable"
    assert result["checks"]["load_transport:a"] is None


def test_m4_complete_native_tick_history_is_required():
    contract, frames = timed_trajectory({"a": {"station": "station_a", "duration_s": .2}})
    result = score("M4", contract, [frames[0], *frames[4:]])
    assert result["checks"]["physics_tick_coverage"] is None


def test_schedule_witnesses_preserve_resources_and_report_pair_ordering_overlap():
    jobs = {"a": {"station": "station_a", "duration_s": 12.},
            "b": {"station": "station_a", "duration_s": 4.},
            "c": {"station": "station_b", "duration_s": 8.}}
    base = enumerate_schedules(jobs, 2.)
    changed_jobs = deepcopy(jobs)
    changed_jobs["a"]["duration_s"] = 2.
    changed = enumerate_schedules(changed_jobs, 2.)
    assert base["enumeration_complete"] and changed["enumeration_complete"]
    for result, specs in ((base, jobs), (changed, changed_jobs)):
        assert result["witnesses"]
        for witness in result["witnesses"]:
            occupied, complete, delivered, previous_end = {}, {}, set(), 0.
            for op in witness:
                assert op["start_s"] >= previous_end
                assert op["end_s"] - op["start_s"] == 2.
                previous_end = op["end_s"]
                if op["operation"] == "load":
                    assert op["station"] not in occupied
                    occupied[op["station"]] = op["job"]
                    complete[op["job"]] = op["end_s"] + specs[op["job"]]["duration_s"]
                else:
                    assert occupied.pop(op["station"]) == op["job"]
                    assert op["start_s"] >= complete[op["job"]]
                    delivered.add(op["job"])
            assert not occupied and delivered == set(specs)
            assert previous_end == result["optimal_makespan_s"]
    def signatures(result):
        return {tuple((op["operation"], op["job"]) for op in witness) for witness in result["witnesses"]}
    assert base["optimal_makespan_s"] == 24.
    assert changed["optimal_makespan_s"] == 16.
    # Timing changes, but this pair does not force a different action ordering.
    assert len(signatures(base) & signatures(changed)) == 2


def test_enumerators_reject_unsupported_or_nonphysical_cost_models():
    with pytest.raises(ValueError):
        enumerate_allocations({"split": .5}, {"a": 1})
    with pytest.raises(ValueError):
        enumerate_schedules({"a": {"station": "station_a", "duration_s": -1.}}, 2.)
