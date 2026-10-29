"""Exercise scientific evidence and timing for rigid partial insertion.

Run: python -m pytest -q tests/test_rigid_insertion_scoring.py
"""
from copy import deepcopy
import json

import pytest

from wia.tasks.rigid_insertion_scoring import RigidInsertionProgress


def tracker(targets=None, containers=("bin",), **kwargs):
    return RigidInsertionProgress(
        targets or {"item": "native_item"}, containers,
        minimum_lift_m=0.05, stability_duration_s=0.5,
        maximum_linear_speed_m_s=0.02, maximum_angular_speed_rad_s=0.1, **kwargs)


def sample(time, *, targets=("item",), held=None, inserted=(), lowest_z=0.2,
           contact=False, clear=True):
    return {
        "native_time_s": time, "control_valid": True, "hand_clear": clear,
        "grasp": {"available": True, "holding": held is not None,
                  "object_name": held, "constraint_active": held is not None,
                  "release_pending": False, "gripper_open": held is None},
        "targets": {key: {
            "object_name": "native_" + key, "lowest_z_m": lowest_z,
            "linear_speed_m_s": 0.0, "angular_speed_rad_s": 0.0,
            "dynamic_collidable": True, "robot_contact": contact,
            "containers": {"bin": {"interior_valid": True, "partial_inside": key in inserted,
                                    "touching": key in inserted, "penetration_free": True}},
        } for key in targets},
    }


def pickup(progress, key="item", *, targets=("item",), offset=0):
    progress.update(sample(offset + 1, targets=targets, held="native_" + key,
                           contact=True, clear=False))
    return progress.update(sample(offset + 2, targets=targets, held="native_" + key,
                                  lowest_z=0.3, contact=True, clear=False))


def release(progress, key="item", *, targets=("item",), offset=0):
    progress.update(sample(offset + 3, targets=targets, inserted=(key,)))
    return progress.update(sample(offset + 3.5, targets=targets, inserted=(key,)))


def complete(progress):
    progress.begin(sample(0))
    pickup(progress)
    return release(progress)


def test_lawful_insertion_requires_physics_time_and_has_no_qa_claim():
    progress = tracker()
    initial = progress.begin(sample(0))
    assert initial["evaluation_validity"] == "complete"
    assert initial["completion_score"] == 0
    lifted = pickup(progress)
    assert lifted["per_target"]["item"]["grasp_seen"]
    assert lifted["per_target"]["item"]["lift_seen"]
    first = progress.update(sample(3, inserted=("item",)))
    assert first["physical_success"] is False
    assert progress.update(sample(3.25, inserted=("item",)))["physical_success"] is False
    result = progress.update(sample(3.5, inserted=("item",)))
    assert result["task_success"] is result["physical_success"] is True
    assert result["constraints_satisfied"] is True
    assert result["completion_score"] == 1
    assert result["score_percent"] == 100
    assert result["verified_targets"] == ["item"]
    assert result["completion_items"] == [{"id": "target:item", "completed": True,
                                           "evaluation_validity": "valid", "container": "bin",
                                           "verified_at_s": 3.5}]
    assert result["all_targets_verified_at_s"] == 3.5
    assert result["final_released"] is True
    assert "qa_correct" not in result


def test_top_rest_and_unaudited_inside_or_aabb_never_establish_insertion():
    progress = tracker()
    progress.begin(sample(0))
    pickup(progress)
    for time in (3, 4, 5):
        row = sample(time)
        relation = row["targets"]["item"]["containers"]["bin"]
        relation.update(touching=True, native_inside=True, aabb_overlap=True)
        result = progress.update(row)
    assert result["evaluation_validity"] == "complete"
    assert result["completion_score"] == 0
    assert not result["physical_success"]


@pytest.mark.parametrize("field,value", [
    ("interior_valid", None), ("interior_valid", False), ("partial_inside", None),
    ("touching", None), ("penetration_free", None),
])
def test_absent_or_invalid_audited_container_evidence_is_unavailable(field, value):
    progress = tracker()
    progress.begin(sample(0))
    pickup(progress)
    for time in (3, 4):
        row = sample(time, inserted=("item",))
        row["targets"]["item"]["containers"]["bin"][field] = value
        result = progress.update(row)
    assert result["evaluation_validity"] == "unavailable"
    assert result["completion_score"] is None
    assert result["physical_success"] is False
    assert result["verified_targets"] == []


@pytest.mark.parametrize("field,value", [
    ("partial_inside", False), ("touching", False), ("penetration_free", False),
])
def test_measured_failure_of_container_conditions_is_not_insertion(field, value):
    progress = tracker()
    progress.begin(sample(0))
    pickup(progress)
    for time in (3, 4):
        row = sample(time, inserted=("item",))
        row["targets"]["item"]["containers"]["bin"][field] = value
        result = progress.update(row)
    assert result["evaluation_validity"] == "complete"
    assert result["physical_success"] is False


@pytest.mark.parametrize("mode", ["held", "contact", "hand_not_clear", "release_pending", "closed"])
def test_robot_support_or_unfinished_release_cannot_verify(mode):
    progress = tracker()
    progress.begin(sample(0))
    pickup(progress)
    for time in (3, 4, 5):
        row = sample(time, inserted=("item",))
        if mode == "held":
            row["grasp"].update(holding=True, constraint_active=True, object_name="native_item")
        elif mode == "contact":
            row["targets"]["item"]["robot_contact"] = True
        elif mode == "hand_not_clear":
            row["hand_clear"] = False
        elif mode == "release_pending":
            row["grasp"]["release_pending"] = True
        else:
            row["grasp"]["gripper_open"] = False
        result = progress.update(row)
    assert result["evaluation_validity"] == "complete"
    assert not result["physical_success"]
    assert not result["final_released"]
    assert result["verified_targets"] == []


@pytest.mark.parametrize("mode", ["no_grasp", "no_lift", "lift_only_after_release", "no_constraint"])
def test_pickup_and_lift_must_be_observed_under_the_measured_constraint(mode):
    progress = tracker()
    progress.begin(sample(0))
    if mode != "no_grasp":
        for time in (1, 2):
            row = sample(time, held="native_item")
            if mode == "no_constraint":
                row["grasp"]["constraint_active"] = False
                row["targets"]["item"]["lowest_z_m"] = 0.3
            progress.update(row)
    for time in (3, 4):
        row = sample(time, inserted=("item",), lowest_z=0.3 if mode == "lift_only_after_release" else 0.2)
        result = progress.update(row)
    assert not result["physical_success"]
    assert not result["per_target"]["item"]["lift_seen"]


def test_wrong_grasp_target_and_unbound_object_do_not_supply_pickup_evidence():
    progress = tracker()
    progress.begin(sample(0))
    progress.update(sample(1, held="native_other", lowest_z=0.3))
    result = release(progress)
    assert not result["per_target"]["item"]["grasp_seen"]
    assert not result["physical_success"]
    assert result["constraints_satisfied"] is True


def test_mismatched_native_target_identity_is_unavailable():
    progress = tracker()
    progress.begin(sample(0))
    pickup(progress)
    row = sample(3, inserted=("item",))
    row["targets"]["item"]["object_name"] = "another_native_object"
    result = progress.update(row)
    assert result["evaluation_validity"] == "unavailable"
    assert not result["physical_success"]


def test_wrong_container_cannot_supply_allowed_container_evidence():
    progress = tracker()
    progress.begin(sample(0))
    pickup(progress)
    for time in (3, 4):
        row = sample(time)
        row["targets"]["item"]["containers"]["not_allowed"] = {
            "interior_valid": True, "partial_inside": True,
            "touching": True, "penetration_free": True}
        result = progress.update(row)
    assert result["evaluation_validity"] == "complete"
    assert not result["physical_success"]


@pytest.mark.parametrize("speed", ["linear_speed_m_s", "angular_speed_rad_s"])
def test_unstable_release_resets_elapsed_dwell(speed):
    progress = tracker()
    progress.begin(sample(0))
    pickup(progress)
    progress.update(sample(3, inserted=("item",)))
    unstable = sample(3.25, inserted=("item",))
    unstable["targets"]["item"][speed] = 0.2
    progress.update(unstable)
    assert progress.update(sample(3.5, inserted=("item",)))["physical_success"] is False
    assert progress.update(sample(3.75, inserted=("item",)))["physical_success"] is False
    assert progress.update(sample(4, inserted=("item",)))["physical_success"] is True


def test_duplicate_timestamps_and_result_reads_never_advance_dwell():
    progress = tracker()
    progress.begin(sample(0))
    pickup(progress)
    row = sample(3, inserted=("item",))
    for _ in range(20):
        result = progress.update(row)
        assert progress.result() == result
        assert not result["physical_success"]
        assert result["per_target"]["item"]["stable_duration_s"] == 0
    assert progress.update(sample(3.5, inserted=("item",)))["physical_success"]


def test_backward_clock_invalidates_episode_until_explicit_begin():
    progress = tracker()
    assert complete(progress)["physical_success"]
    result = progress.update(sample(3.25, inserted=("item",)))
    assert result["evaluation_validity"] == "unavailable"
    assert result["verified_targets"] == ["item"]
    assert not progress.update(sample(5, inserted=("item",)))["physical_success"]
    assert complete(progress)["physical_success"]


def test_explicit_maximum_gap_prevents_bridging_unobserved_stability():
    progress = tracker(max_sample_gap_s=0.3)
    progress.begin(sample(0))
    pickup(progress)
    progress.update(sample(3, inserted=("item",)))
    result = progress.update(sample(4, inserted=("item",)))
    assert result["sample_gap_exceeded"]
    assert result["per_target"]["item"]["stable_duration_s"] == 0
    assert not result["physical_success"]
    assert not progress.update(sample(4.25, inserted=("item",)))["physical_success"]
    assert progress.update(sample(4.5, inserted=("item",)))["physical_success"]


def test_missing_sample_interrupts_dwell_and_later_samples_can_recover():
    progress = tracker()
    progress.begin(sample(0))
    pickup(progress)
    progress.update(sample(3, inserted=("item",)))
    result = progress.update({"native_time_s": 3.25})
    assert result["evaluation_validity"] == "unavailable"
    assert result["completion_score"] is None
    assert not progress.update(sample(3.5, inserted=("item",)))["physical_success"]
    assert not progress.update(sample(3.75, inserted=("item",)))["physical_success"]
    assert progress.update(sample(4, inserted=("item",)))["physical_success"]


def test_sequential_verification_allows_removing_first_object_from_shared_container():
    targets = ("first", "second")
    progress = tracker({key: "native_" + key for key in targets})
    progress.begin(sample(0, targets=targets))
    pickup(progress, "first", targets=targets)
    first = release(progress, "first", targets=targets)
    assert first["verified_targets"] == ["first"]
    assert first["completion_score"] == 0.5
    assert not first["physical_success"]
    removed = progress.update(sample(4, targets=targets, held="native_first", contact=True, clear=False))
    assert removed["verified_targets"] == ["first"]
    assert removed["constraints_satisfied"] is True
    assert not removed["final_released"]
    progress.update(sample(4.5, targets=targets))
    pickup(progress, "second", targets=targets, offset=4)
    result = release(progress, "second", targets=targets, offset=4)
    assert result["physical_success"]
    assert result["verified_targets"] == list(targets)
    assert result["per_target"]["first"]["verified_at_s"] == 3.5
    assert result["per_target"]["first"]["verified_container"] == "bin"
    assert result["per_target"]["first"]["containers"]["bin"]["partial_inside"] is False
    assert result["per_target"]["first"]["verified_evidence"]["container_measurements"]["partial_inside"] is True
    assert result["all_targets_verified_at_s"] == 7.5


def test_control_violation_latches_while_factual_completion_is_retained():
    progress = tracker()
    assert complete(progress)["physical_success"]
    row = sample(4, inserted=("item",))
    row["control_valid"] = False
    result = progress.update(row)
    assert result["constraints_satisfied"] is False
    assert not result["physical_success"]
    assert result["completion_score"] == 1
    assert result["verified_targets"] == ["item"]
    assert progress.update(sample(5, inserted=("item",)))["constraints_satisfied"] is False


def test_control_violation_before_verification_cannot_later_pass():
    progress = tracker()
    row = sample(0)
    row["control_valid"] = False
    progress.begin(row)
    pickup(progress)
    result = release(progress)
    assert not result["physical_success"]
    assert result["verified_targets"] == []
    assert result["constraints_satisfied"] is False


def test_unknown_control_prevents_current_success_without_faking_a_score():
    progress = tracker()
    assert complete(progress)["physical_success"]
    row = sample(4, inserted=("item",))
    row["control_valid"] = None
    result = progress.update(row)
    assert not result["physical_success"]
    assert result["constraints_satisfied"] is None
    assert result["evaluation_validity"] == "unavailable"
    assert result["completion_score"] is None
    assert progress.update(sample(5, inserted=("item",)))["physical_success"]


@pytest.mark.parametrize("field", ["native_time_s", "lowest_z_m", "linear_speed_m_s", "angular_speed_rad_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_physics_measurements_are_unavailable_and_json_safe(field, value):
    progress = tracker()
    progress.begin(sample(0))
    pickup(progress)
    row = sample(3, inserted=("item",))
    destination = row if field == "native_time_s" else row["targets"]["item"]
    destination[field] = value
    result = progress.update(row)
    assert result["evaluation_validity"] == "unavailable"
    assert result["completion_score"] is None
    assert not result["physical_success"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field", ["dynamic_collidable", "robot_contact", "lowest_z_m",
                                    "linear_speed_m_s", "angular_speed_rad_s"])
def test_missing_target_measurements_cannot_be_replaced_by_hand_clear(field):
    progress = tracker()
    progress.begin(sample(0))
    pickup(progress)
    row = sample(3, inserted=("item",))
    del row["targets"]["item"][field]
    result = progress.update(row)
    assert result["evaluation_validity"] == "unavailable"
    assert not result["physical_success"]


@pytest.mark.parametrize("field", ["available", "holding", "constraint_active", "release_pending",
                                    "gripper_open", "object_name"])
def test_missing_grasp_measurements_are_unavailable(field):
    progress = tracker()
    progress.begin(sample(0))
    pickup(progress)
    row = sample(3, inserted=("item",))
    del row["grasp"][field]
    result = progress.update(row)
    assert result["evaluation_validity"] == "unavailable"
    assert not result["physical_success"]


def test_missing_initial_baseline_cannot_be_invented_from_a_later_pose():
    progress = tracker()
    row = sample(0)
    del row["targets"]["item"]["lowest_z_m"]
    progress.begin(row)
    pickup(progress)
    result = release(progress)
    assert result["evaluation_validity"] == "unavailable"
    assert result["per_target"]["item"]["baseline_lowest_z_m"] is None
    assert not result["physical_success"]


def test_initial_hold_is_not_new_pickup_but_empty_then_fresh_pickup_is_allowed():
    progress = tracker()
    progress.begin(sample(0, held="native_item"))
    progress.update(sample(1, held="native_item", lowest_z=0.3))
    result = release(progress)
    assert not result["per_target"]["item"]["grasp_seen"]
    assert not result["physical_success"]
    pickup(progress, offset=4)
    assert release(progress, offset=4)["physical_success"]


def test_completed_target_still_requires_current_no_robot_contact():
    progress = tracker()
    assert complete(progress)["physical_success"]
    row = sample(4, contact=True)
    result = progress.update(row)
    assert result["verified_targets"] == ["item"]
    assert result["final_released"] is False
    assert not result["physical_success"]
    row = sample(5)
    row["targets"]["item"]["robot_contact"] = None
    result = progress.update(row)
    assert result["evaluation_validity"] == "unavailable"
    assert not result["physical_success"]


def test_completed_target_may_have_no_current_interior_or_velocity_measurement():
    progress = tracker()
    assert complete(progress)["physical_success"]
    row = sample(4)
    row["targets"]["item"] = {"object_name": "native_item", "robot_contact": False}
    assert progress.update(row)["physical_success"]


@pytest.mark.parametrize("held", ["native_item", "unbound_object"])
def test_final_nonempty_hand_prevents_success_without_erasing_verified_facts(held):
    progress = tracker()
    assert complete(progress)["physical_success"]
    result = progress.update(sample(4, held=held))
    assert result["evaluation_validity"] == "complete"
    assert result["completion_score"] == 1
    assert not result["physical_success"]
    assert result["constraints_satisfied"] is True
    assert result["verified_targets"] == ["item"]
    assert result["unbound_held_object"] == (held if held == "unbound_object" else None)


def test_switching_containers_cannot_combine_two_short_dwells():
    progress = tracker(containers=("bin", "other"))
    initial = sample(0)
    initial["targets"]["item"]["containers"]["other"] = deepcopy(initial["targets"]["item"]["containers"]["bin"])
    progress.begin(initial)
    pickup(progress)
    for time, selected in ((3, "bin"), (3.25, "other"), (3.5, "other"), (3.75, "other")):
        row = sample(time, inserted=("item",) if selected == "bin" else ())
        row["targets"]["item"]["containers"]["other"] = {
            "interior_valid": True, "partial_inside": selected == "other",
            "touching": selected == "other", "penetration_free": True}
        result = progress.update(row)
        assert result["physical_success"] is (time == 3.75)
    assert result["per_target"]["item"]["verified_container"] == "other"


def test_ineligible_insertion_is_truthful_and_a_corrected_attempt_can_succeed():
    progress = tracker(containers=("bin", "best"), expected_containers={"item": "best"})
    progress.begin(sample(0))
    pickup(progress)
    for time in (3, 4):
        row = sample(time, inserted=("item",))
        row["targets"]["item"]["containers"]["best"] = {
            "interior_valid": True, "partial_inside": False,
            "touching": False, "penetration_free": True}
        result = progress.update(row)
    assert result["evaluation_validity"] == "complete"
    assert not result["physical_success"]
    assert result["verified_targets"] == []
    state = result["per_target"]["item"]
    assert state["containers"]["bin"]["insertion_conditions_satisfied"] is True
    assert state["containers"]["bin"]["eligible"] is False
    assert state["conditions"]["ineligible_partial_insertion"] is True
    assert state["conditions"]["ineligible_inserted_containers"] == ["bin"]
    pickup(progress, offset=4)
    for time in (7, 7.5):
        row = sample(time)
        row["targets"]["item"]["containers"]["best"] = {
            "interior_valid": True, "partial_inside": True,
            "touching": True, "penetration_free": True}
        result = progress.update(row)
    assert result["physical_success"]
    assert result["per_target"]["item"]["verified_container"] == "best"
    assert result["expected_containers"] == {"item": ["best"]}


def test_unknown_ineligible_geometry_does_not_hide_verified_eligible_insertion():
    progress = tracker(containers=("bin", "other"), expected_containers={"item": ["bin"]})
    result = complete(progress)
    assert result["physical_success"]
    assert result["per_target"]["item"]["containers"]["other"]["evaluation_validity"] == "unavailable"


@pytest.mark.parametrize("expected", [{"unknown_target": "bin"}, {"item": "unknown_container"},
                                      {"item": []}])
def test_unknown_or_empty_expected_container_configuration_is_rejected(expected):
    with pytest.raises(ValueError):
        tracker(expected_containers=expected)


def test_unstarted_and_empty_samples_return_unavailable_not_a_fake_zero_score():
    progress = tracker()
    result = progress.result()
    assert result["evaluation_validity"] == "unavailable"
    assert result["completion_score"] is None
    assert not result["physical_success"]
    assert progress.update(sample(0)) == result
    assert progress.begin({})["evaluation_validity"] == "unavailable"


def test_result_is_a_detached_json_safe_snapshot():
    progress = tracker()
    result = complete(progress)
    json.dumps(result, allow_nan=False)
    result["per_target"]["item"]["verified"] = False
    result["thresholds"]["minimum_lift_m"] = 0
    assert progress.result()["physical_success"]
    assert progress.result()["thresholds"]["minimum_lift_m"] == 0.05


@pytest.mark.parametrize("field,value", [
    ("minimum_lift_m", 0), ("stability_duration_s", 0),
    ("maximum_linear_speed_m_s", float("nan")), ("maximum_angular_speed_rad_s", -0.1),
    ("max_sample_gap_s", 0),
])
def test_invalid_scientific_thresholds_are_rejected(field, value):
    kwargs = {"minimum_lift_m": 0.05, "stability_duration_s": 0.5,
              "maximum_linear_speed_m_s": 0.02, "maximum_angular_speed_rad_s": 0.1}
    kwargs[field] = value
    with pytest.raises(ValueError):
        RigidInsertionProgress({"item": "native_item"}, ["bin"], **kwargs)
