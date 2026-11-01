"""Pure measured-evidence contracts for the five HIGH candidates.

Run: PYTHONPATH=src python -m pytest -q tests/test_candidate_high.py

Public API: ``evaluate(prototype, contract, frame, history)``. Frames contain
``time_s``, ``objects``, ``regions``, ``observed`` and optional ``events``.
Objects use world-space position, AABB, scalar linear/angular speed, held and
contact-support names. Optional ``orientation`` is an XYZW quaternion. Regions
use AABB XY acceptance bounds, with the upper Z coordinate at the support
surface, and a ``support`` name. History must include the initial frame and
physics samples in increasing simulation time; it excludes the current frame.

Events use ``type``, ``time_s`` and ``authored: True``. H3 external_removal
events also name ``object``. H2 post_delivery_check events are linked to the
first native frame at or after their time (within one sampling interval).
``observed`` contains only tags actually delivered to the policy. Missing native
measurements return unavailable; no helper changes simulator or contract state.
"""

from __future__ import annotations

import math
from typing import Any, Callable


POSITION_TOLERANCE_M = 0.02
ORIENTATION_TOLERANCE_RAD = math.radians(5)
BOTTOM_TOLERANCE_M = 0.015
MAX_LINEAR_SPEED = 0.02
MAX_ANGULAR_SPEED = 0.1
STABLE_SECONDS = 0.25
MAX_SAMPLE_GAP_S = 0.1
SCHEMA = "wia.high_candidate_evaluation.v1"

Check = bool | None


def _all(values: list[Check]) -> Check:
    if None in values:
        return None
    return False if False in values else True


def _any(values: list[Check]) -> Check:
    if True in values:
        return True
    return None if None in values else False


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _vector(value: Any, size: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    result = [_number(x) for x in value]
    return None if None in result else result


def _bounds(value: Any) -> tuple[list[float], list[float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    lo, hi = _vector(value[0], 3), _vector(value[1], 3)
    if lo is None or hi is None or any(a > b for a, b in zip(lo, hi)):
        return None
    return lo, hi


def _object(frame: dict, name: str) -> dict:
    return frame.get("objects", {}).get(name, {})


def _region(frame: dict, name: str) -> dict:
    return frame.get("regions", {}).get(name, {})


def _placed(frame: dict, name: str, region: str) -> Check:
    obj, destination = _object(frame, name), _region(frame, region)
    box, zone = _bounds(obj.get("aabb")), _bounds(destination.get("aabb"))
    linear, angular = _number(obj.get("linear_speed")), _number(obj.get("angular_speed"))
    contacts, support = obj.get("contacts"), destination.get("support")
    if (box is None or zone is None or linear is None or angular is None
            or not isinstance(obj.get("held"), bool)
            or not isinstance(contacts, list) or not isinstance(support, str)):
        return None
    lo, hi = box
    lower, upper = zone
    return bool(
        all(lo[i] >= lower[i] - 1e-9 and hi[i] <= upper[i] + 1e-9 for i in (0, 1))
        and abs(lo[2] - upper[2]) <= BOTTOM_TOLERANCE_M + 1e-9
        and support in contacts and not obj["held"]
        and 0 <= linear <= MAX_LINEAR_SPEED and 0 <= angular <= MAX_ANGULAR_SPEED
    )


def _overlap(frame: dict, name: str, region: str) -> Check:
    box = _bounds(_object(frame, name).get("aabb"))
    zone = _bounds(_region(frame, region).get("aabb"))
    if box is None or zone is None:
        return None
    return all(box[0][i] < zone[1][i] - 1e-9 and box[1][i] > zone[0][i] + 1e-9
               for i in (0, 1))


def _stable(frames: list[dict], name: str, region: str) -> Check:
    """Require an uninterrupted measured terminal interval, not action count."""
    end = _number(frames[-1].get("time_s"))
    if end is None:
        return None
    previous = end
    for sample in reversed(frames):
        current = _number(sample.get("time_s"))
        if current is None or previous - current > MAX_SAMPLE_GAP_S + 1e-9:
            return None
        placed = _placed(sample, name, region)
        if placed is not True:
            return placed
        if end - current >= STABLE_SECONDS - 1e-9:
            return True
        previous = current
    return None


def _first_stable_time(frames: list[dict], name: str, region: str) -> float | None:
    """Scan physics evidence once to find the first completed placement."""
    start, previous = None, None
    for sample in frames:
        time = _number(sample.get("time_s"))
        if time is None or _placed(sample, name, region) is not True:
            start, previous = None, None
            continue
        if previous is None or time - previous > MAX_SAMPLE_GAP_S + 1e-9:
            start = time
        previous = time
        if time - start >= STABLE_SECONDS - 1e-9:
            return time
    return None


def _orientation_delta(first: dict, current: dict) -> float | None:
    if "orientation" not in first:
        return 0.0
    start, end = _vector(first["orientation"], 4), _vector(current.get("orientation"), 4)
    if start is None or end is None:
        return None
    norms = [math.sqrt(sum(x * x for x in q)) for q in (start, end)]
    if min(norms) <= 1e-12:
        return None
    dot = abs(sum(a * b for a, b in zip(start, end)) / (norms[0] * norms[1]))
    return 2 * math.acos(min(1.0, dot))


def _preserved(frames: list[dict], names: list[str]) -> Check:
    checks = []
    for name in names:
        initial = _object(frames[0], name)
        origin = _vector(initial.get("position"), 3)
        for sample in frames:
            obj = _object(sample, name)
            position = _vector(obj.get("position"), 3)
            angle = _orientation_delta(initial, obj)
            if origin is None or position is None or angle is None:
                checks.append(None)
            else:
                checks.append(math.dist(origin, position) <= POSITION_TOLERANCE_M + 1e-9
                              and angle <= ORIENTATION_TOLERANCE_RAD + 1e-9)
    return _all(checks)


def _observed_before(frames: list[dict], tag: str, cutoff: float | None,
                     *, after: float | None = None) -> Check:
    if cutoff is None:
        return None
    missing = False
    for sample in frames:
        tags, time = sample.get("observed"), _number(sample.get("time_s"))
        if not isinstance(tags, list) or time is None:
            missing = True
        elif tag in tags and time < cutoff - 1e-9 and (after is None or time > after + 1e-9):
            return True
    return None if missing else False


def _first_placement(frames: list[dict], names: list[str], region: str) -> float | None:
    for sample in frames:
        if any(_placed(sample, name, region) is True for name in names):
            return _number(sample.get("time_s"))
    return None


def _events(frames: list[dict], kind: str) -> list[dict]:
    result = []
    for sample in frames:
        for event in sample.get("events", []):
            if event.get("type") == kind and event not in result:
                result.append(event)
    return result


def _moved(frames: list[dict], name: str) -> Check:
    """Use the preservation tolerance to separate relocations from pose noise."""
    held = [_object(sample, name).get("held") for sample in frames]
    preserved = _preserved(frames, [name])
    if True in held or preserved is False:
        return True
    return None if preserved is None or None in held else False


def _cost(frames: list[dict]) -> dict:
    names = list(frames[0].get("objects", {}))
    path, releases = 0.0, 0
    measured, holds_measured = True, True
    for name in names:
        for before, after in zip(frames, frames[1:]):
            a, b = _object(before, name), _object(after, name)
            start, end = _vector(a.get("position"), 3), _vector(b.get("position"), 3)
            if start is None or end is None:
                measured = False
            else:
                path += math.dist(start, end)
            if not isinstance(a.get("held"), bool) or not isinstance(b.get("held"), bool):
                holds_measured = False
            releases += int(a.get("held") is True and b.get("held") is False)
    start, end = _number(frames[0].get("time_s")), _number(frames[-1].get("time_s"))
    return {"elapsed_sim_s": None if start is None or end is None else end - start,
            "object_path_length_m": path if measured else None,
            "release_count": releases if holds_measured else None}


def _h1(contract: dict, frames: list[dict], checks: dict, cost: dict) -> None:
    selected, spares = contract["selected"], contract["spares"]
    delivery, quarantine = contract.get("delivery", "delivery"), contract.get("quarantine", "quarantine")
    if contract["defective"]:
        candidates = [name for name in spares if name in contract["compatible"]]
        checks["compatible_spare_delivered"] = _any([_stable(frames, name, delivery) for name in candidates])
        checks["selected_quarantined"] = _stable(frames, selected, quarantine)
        delivered_names = candidates
    else:
        checks["selected_delivered"] = _stable(frames, selected, delivery)
        checks["spares_preserved"] = _preserved(frames, spares)
        delivered_names = [selected]
    first = _first_placement(frames, delivered_names, delivery)
    checks["inspection_before_delivery"] = _observed_before(frames, f"inspection:{selected}", first)
    checks["protected_preserved"] = _preserved(frames, contract.get("protected", []))


def _h2(contract: dict, frames: list[dict], checks: dict, cost: dict) -> None:
    payload, obstruction = contract["payload"], contract["obstruction"]
    work, delivery, storage = (contract.get(key, default) for key, default in
                               (("work_region", "work"), ("delivery", "delivery"), ("storage", "storage")))
    checks["initial_obstruction_in_work"] = _placed(frames[0], obstruction, work)
    checks["payload_delivered"] = _stable(frames, payload, delivery)
    checks["work_clear"] = None if (overlap := _overlap(frames[-1], obstruction, work)) is None else not overlap
    checks["obstruction_stored"] = _stable(frames, obstruction, storage)
    first_delivery = _first_placement(frames, [payload], delivery)
    clear_before = []
    for sample in frames:
        if first_delivery is not None and sample["time_s"] < first_delivery:
            overlap = _overlap(sample, obstruction, work)
            clear_before.append(None if overlap is None else not overlap)
    checks["work_cleared_before_delivery"] = _any(clear_before)
    events = _events(frames, "post_delivery_check")
    event_checks, observed_checks, disturbance_checks, sequence_checks = [], [], [], []
    completed_delivery = _first_stable_time(frames, payload, delivery)
    for event in events:
        time = _number(event.get("time_s"))
        if time is None:
            event_checks.append(None)
            sequence_checks.append(None)
            continue
        delivered = completed_delivery is not None and completed_delivery < time
        authored = event.get("authored") if isinstance(event.get("authored"), bool) else None
        event_valid = _all([authored, delivered, time < frames[-1]["time_s"]])
        after = next((sample for sample in frames if sample["time_s"] >= time), None)
        branch_state = (None if after is None or after["time_s"] - time > MAX_SAMPLE_GAP_S + 1e-9
                        else _placed(after, obstruction, work if contract["disturbance"] else storage))
        observed = _observed_before(frames, "workspace_after_event", frames[-1]["time_s"] + 1e-8, after=time)
        event_checks.append(event_valid)
        disturbance_checks.append(branch_state)
        observed_checks.append(observed)
        sequence_checks.append(_all([event_valid, branch_state, observed]))
    checks["post_delivery_check"] = _any(event_checks) if events else None
    checks["authored_branch_state"] = _any(disturbance_checks) if disturbance_checks else None
    checks["workspace_observed_after_event"] = _any(observed_checks) if observed_checks else None
    checks["post_delivery_sequence"] = _any(sequence_checks) if sequence_checks else None
    checks["protected_preserved"] = _preserved(frames, contract.get("protected", []))


def _h3(contract: dict, frames: list[dict], checks: dict, cost: dict) -> None:
    if contract.get('requires_revision_event'):
        checks['revision_event_occurred'] = any(e.get('type') == 'external_removal' and e.get('authored') is True for f in frames for e in f.get('events', []))
    names, region = contract["stock"], contract.get("stock_region", "stock")
    target = contract.get("target_count", 2)
    initial = {name: _placed(frames[0], name, region) for name in names}
    initial_names = {name for name, value in initial.items() if value is True}
    checks["minimum_semantics"] = contract.get("semantics", "minimum") == "minimum"
    removed = {}
    removal_checks = []
    for event in _events(frames, "external_removal"):
        name, time = event.get("object"), _number(event.get("time_s"))
        if event.get("authored") is not True or name not in initial_names or time is None:
            removal_checks.append(False)
            continue
        removed[name] = min(time, removed.get(name, time))
        removal_checks.append(frames[0]["time_s"] < time <= frames[-1]["time_s"])
        after = next((sample for sample in frames if sample["time_s"] >= time), None)
        if after is None:
            removal_checks.append(None)
        elif name in after.get("objects", {}):
            placement = _placed(after, name, region)
            removal_checks.append(None if placement is None else not placement)
    checks["authored_removals"] = _all(removal_checks)
    current = {name: (False if name in removed and name not in frames[-1].get("objects", {})
                      else _stable(frames, name, region)) for name in names}
    checks["inventory_measurements"] = None if None in initial.values() or None in current.values() else True
    checks["minimum_inventory"] = sum(value is True for value in current.values()) >= target
    retained = []
    for name in initial_names:
        for sample in frames:
            if name not in removed or sample["time_s"] < removed[name]:
                retained.append(_placed(sample, name, region))
    checks["initial_inventory_retained"] = _all(retained)
    additions = {name: time for name in names if name not in initial_names
                 and (time := _first_stable_time(frames, name, region)) is not None}
    unnecessary = 0
    for addition_time in additions.values():
        added_so_far = sum(time <= addition_time for time in additions.values())
        removed_so_far = sum(time <= addition_time for time in removed.values())
        allowed_so_far = max(0, target - (len(initial_names) - removed_so_far))
        unnecessary = max(unnecessary, added_so_far - allowed_so_far)
    allowed = max(0, target - (len(initial_names) - len(removed)))
    checks["no_unnecessary_additions"] = unnecessary == 0
    checks["protected_preserved"] = _preserved(frames, contract.get("protected", []))
    cost.update({"initial_inventory_count": len(initial_names),
                 "current_inventory_count": sum(value is True for value in current.values()),
                 "allowed_additions": allowed, "completed_additions": len(additions),
                 "unnecessary_additions": unnecessary})


def _h4(contract: dict, frames: list[dict], checks: dict, cost: dict) -> None:
    checks["target_placed"] = _stable(frames, contract["target"], contract.get("target_region", "target"))
    moved = {name: _moved(frames, name) for name in contract["blockers"]}
    checks["blocker_motion_measured"] = None if None in moved.values() else True
    checks["moved_blockers_stored"] = _all([
        _stable(frames, name, contract.get("storage_region", "storage"))
        for name, value in moved.items() if value is True])
    checks["anchors_preserved"] = _preserved(frames, contract["anchors"])
    evidence = frames[-1].get("process_evidence", {})
    value = evidence.get("hands_withdrawn")
    checks["hands_withdrawn"] = value if isinstance(value, bool) else None
    cost["relocation_count"] = None if None in moved.values() else sum(value is True for value in moved.values())


def _terminal_release(frames: list[dict], name: str, region: str) -> float | None:
    for index in range(len(frames) - 1, 0, -1):
        before, after = _object(frames[index - 1], name), _object(frames[index], name)
        if before.get("held") is True and after.get("held") is False:
            return _number(frames[index].get("time_s"))
    return _first_placement(frames, [name], region)


def _h5(contract: dict, frames: list[dict], checks: dict, cost: dict) -> None:
    payload = contract["payload"]
    primary, fallback = contract.get("primary", "primary"), contract.get("fallback", "fallback")
    residents = [name for name in frames[0].get("objects", {}) if name != payload]
    occupied = _any([_overlap(frames[0], name, primary) for name in residents])
    available = contract["primary_available"]
    checks["primary_state_consistent"] = None if occupied is None else available == (not occupied)
    chosen = primary if available else fallback
    checks["preferred_destination"] = _stable(frames, payload, chosen)
    checks["destination_observed_before_release"] = _observed_before(
        frames, "destination_state", _terminal_release(frames, payload, chosen))
    checks["resident_preserved"] = _preserved(frames, [contract["resident"]])
    checks["protected_preserved"] = _preserved(frames, contract.get("protected", []))
    cost["selected_destination"] = chosen


_EVALUATORS: dict[str, Callable] = {"H1": _h1, "H2": _h2, "H3": _h3, "H4": _h4, "H5": _h5}


def evaluate(prototype: str, contract: dict, frame: dict, history: list[dict]) -> dict:
    """Evaluate terminal measured evidence without invoking or changing physics.

    ``checks`` is a mapping of names to True, False or None (missing proof).
    Only complete evidence with every required check true can succeed. Action
    costs are diagnostics; H3's expressly forbidden surplus additions are also
    a success condition. Unknown prototype names raise ValueError.
    """
    if prototype not in _EVALUATORS:
        raise ValueError(f"Unknown HIGH prototype: {prototype}")
    frames = [*history, frame]
    times = [_number(sample.get("time_s")) for sample in frames]
    timeline = (None if None in times else all(a < b for a, b in zip(times, times[1:])))
    checks: dict[str, Check] = {"timeline_measured": timeline}
    cost = _cost(frames)
    if timeline is True:
        _EVALUATORS[prototype](contract, frames, checks, cost)
    unavailable = any(value is None for value in checks.values())
    reasons = [f"{name}: {'missing measured evidence' if value is None else 'condition failed'}"
               for name, value in checks.items() if value is not True]
    return {"schema": SCHEMA, "prototype": prototype,
            "evaluation_validity": "unavailable" if unavailable else "complete",
            "task_success": not unavailable and all(checks.values()),
            "checks": checks, "reasons": reasons, "action_cost": cost}
