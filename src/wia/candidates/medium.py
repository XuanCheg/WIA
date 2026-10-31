"""Evaluate M1--M4 from read-only native measurements, without a simulator.

Run CPU checks: ``PYTHONPATH=src python -m pytest -q tests/test_candidate_medium.py``.
Frames contain ``time_s``, measured ``objects``, support ``regions``, delivered
``observed`` tags, and timestamped ``events``. History contains prior physics
ticks in their original order. Region XY bounds describe usable support area;
``support_z`` overrides the region's upper Z support plane. Object names must
remain bound to native identities throughout an episode. Optional ``native_id``
fields are checked when supplied. Protected poses use XYZW quaternions.
M3 additionally requires native ``objects[name].inside`` unit-name lists for
residents and throughout the payload's final storage dwell.

These contracts establish event-conditioned placement, sealed-unit allocation,
content-conditioned placement, and timed resource use. They make no claim about
contamination, cooling, or cooking. Schedule witnesses are optimal only in the
declared abstract handling-cost model, not certified native motor timings.
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import product
import math
from numbers import Real


SCHEMA = "wia.candidate_medium.v1"
STABILITY_S = 0.25
MAX_SAMPLE_GAP_S = 0.1
BOTTOM_TOLERANCE_M = 0.015
MAX_LINEAR_SPEED = 0.02
MAX_ANGULAR_SPEED = 0.1
PROTECTED_POSITION_M = 0.02
PROTECTED_ANGLE_RAD = math.radians(5)
_EPS = 1e-8


def _map(value):
    return value if isinstance(value, Mapping) else {}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _vector(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    result = [_number(item) for item in value]
    return result if all(item is not None for item in result) else None


def _aabb(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    lo, hi = _vector(value[0], 3), _vector(value[1], 3)
    if lo is None or hi is None or any(a > b for a, b in zip(lo, hi)):
        return None
    return lo, hi


def _and(values):
    values = list(values)
    if False in values:
        return False
    return None if None in values else True


class _Checks:
    def __init__(self):
        self.values = {}
        self.reasons = []

    def add(self, name, value):
        self.values[name] = value
        if value is not True:
            self.reasons.append(f"{name}:{'unavailable' if value is None else 'failed'}")
        return value

    def result(self, prototype, evidence):
        return {
            "schema": SCHEMA, "prototype": prototype,
            "evaluation_validity": "unavailable" if None in self.values.values() else "complete",
            "task_success": bool(self.values) and all(v is True for v in self.values.values()),
            "checks": self.values, "reasons": self.reasons, "evidence": evidence,
        }


def _placement(frame, name, region_name, *, require_rest=True):
    """Full footprint and native support contact; a center point is insufficient."""
    obj = _map(_map(frame.get("objects")).get(name))
    region = _map(_map(frame.get("regions")).get(region_name))
    box, bounds = _aabb(obj.get("aabb")), _aabb(region.get("aabb"))
    contacts, support = obj.get("contacts"), region.get("support")
    support_z = _number(region.get("support_z", bounds[1][2] if bounds else None))
    if (box is None or bounds is None or support_z is None or not isinstance(support, str)
            or not isinstance(contacts, (list, tuple))):
        return None
    lo, hi = box
    rlo, rhi = bounds
    inside = all(lo[k] >= rlo[k] - _EPS and hi[k] <= rhi[k] + _EPS for k in (0, 1))
    supported = support in contacts and abs(lo[2] - support_z) <= BOTTOM_TOLERANCE_M + _EPS
    geometry = inside and supported
    if not require_rest:
        return geometry
    linear, angular = _number(obj.get("linear_speed")), _number(obj.get("angular_speed"))
    held = obj.get("held")
    if linear is None or angular is None or linear < 0 or angular < 0 or type(held) is not bool:
        return None
    return geometry and not held and linear <= MAX_LINEAR_SPEED + _EPS and angular <= MAX_ANGULAR_SPEED + _EPS


def _occupancy(frame, name, region_name):
    """A partially overhanging supported job still occupies an exclusive station."""
    obj = _map(_map(frame.get("objects")).get(name))
    region = _map(_map(frame.get("regions")).get(region_name))
    box, bounds = _aabb(obj.get("aabb")), _aabb(region.get("aabb"))
    contacts, support = obj.get("contacts"), region.get("support")
    support_z = _number(region.get("support_z", bounds[1][2] if bounds else None))
    if (box is None or bounds is None or support_z is None or not isinstance(support, str)
            or not isinstance(contacts, (list, tuple))):
        return None
    lo, hi = box
    rlo, rhi = bounds
    overlaps = all(hi[k] > rlo[k] + _EPS and lo[k] < rhi[k] - _EPS for k in (0, 1))
    return overlaps and support in contacts and abs(lo[2] - support_z) <= BOTTOM_TOLERANCE_M + _EPS


def _stable(frames, name, region, *, after=None):
    """Require a contiguous measured final dwell, beginning after any prerequisite."""
    final = _placement(frames[-1], name, region)
    if final is not True:
        return final
    end = frames[-1]["time_s"]
    previous = end
    for sample in reversed(frames):
        now = sample["time_s"]
        if after is not None and now < after - _EPS:
            break
        if previous - now > MAX_SAMPLE_GAP_S + _EPS:
            return None
        placed = _placement(sample, name, region)
        if placed is None:
            return None
        if not placed:
            break
        if end - now >= STABILITY_S - _EPS:
            return True
        previous = now
    return False


def _quaternion(obj):
    value = _vector(obj.get("orientation_xyzw", obj.get("orientation")), 4)
    if value is None:
        return None
    norm = math.sqrt(sum(x * x for x in value))
    return [x / norm for x in value] if norm > _EPS else None


def _protected(frames, names):
    if not names:
        return True
    for name in names:
        initial = _map(_map(frames[0].get("objects")).get(name))
        initial_position, initial_q = _vector(initial.get("position"), 3), _quaternion(initial)
        if initial_position is None or initial_q is None:
            return None
        for sample in frames:
            obj = _map(_map(sample.get("objects")).get(name))
            position, q = _vector(obj.get("position"), 3), _quaternion(obj)
            if position is None or q is None:
                return None
            distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(position, initial_position)))
            angle = 2 * math.acos(min(1., abs(sum(x * y for x, y in zip(q, initial_q)))))
            if distance > PROTECTED_POSITION_M + _EPS or angle > PROTECTED_ANGLE_RAD + _EPS:
                return False
    return True


def _identity(frames, names, *, after=None):
    for name in names:
        samples = [f for f in frames if after is None or f["time_s"] >= after - _EPS]
        objects = [_map(_map(f.get("objects")).get(name)) for f in samples]
        if not objects or any(not obj for obj in objects):
            return None
        for key in ("native_id", "identity"):
            supplied = [obj.get(key) for obj in objects]
            if any(value is not None for value in supplied):
                if any(value is None for value in supplied):
                    return None
                if any(value != supplied[0] for value in supplied):
                    return False
    return True


def _observed(frames, tag, *, after=None):
    available = False
    for sample in frames:
        tags = sample.get("observed")
        if isinstance(tags, (list, tuple)):
            available = True
            if tag in tags and (after is None or sample["time_s"] >= after - _EPS):
                return True
    return False if available else None


def _m1(contract, frames, checks, evidence):
    tools, used = contract["tools"], contract["used_tool"]
    if len(tools) != 2 or len(set(tools)) != 2 or used not in tools:
        checks.add("contract_tools", False)
        return
    checks.add("contract_tools", True)
    events_available, valid_events = False, []
    for sample in frames:
        if not isinstance(sample.get("events"), (list, tuple)):
            continue
        events_available = True
        for event in sample["events"]:
            event = _map(event)
            time_s = _number(event.get("time_s", event.get("timestamp")))
            if (event.get("type") == "cleaning_contact" and event.get("object") == used
                    and event.get("contact_valid") is True and time_s is not None
                    and frames[0]["time_s"] - _EPS <= time_s <= sample["time_s"] + _EPS):
                valid_events.append(time_s)
    event_time = min(valid_events) if valid_events else None
    checks.add("real_cleaning_contact", True if valid_events else False if events_available else None)
    checks.add("observed_use", _observed(frames, f"use:{used}", after=event_time) if event_time is not None else None)
    checks.add("identity_continuity", _identity(frames, tools))
    unused = next(name for name in tools if name != used)
    checks.add("used_placement", _stable(frames, used, contract["used_region"], after=event_time))
    checks.add("unused_placement", _stable(frames, unused, contract["clean_region"], after=event_time))
    evidence.update(used_tool=used, unused_tool=unused, cleaning_event_time_s=event_time)


def enumerate_allocations(quantities, orders):
    """Return every exact sealed-unit assignment, including interchangeable subsets.

    Each returned mapping assigns every unit to an order name or ``stock``.
    Quantities are positive integers; demands are nonnegative integers. This
    deliberately small exhaustive reference is bounded to ten units/four orders.
    """
    if not isinstance(quantities, Mapping) or not isinstance(orders, Mapping):
        raise ValueError("quantities and orders must be mappings")
    if len(quantities) > 10 or len(orders) > 4 or "stock" in orders:
        raise ValueError("enumeration supports at most ten units/four orders and reserves stock")
    if any(type(q) is not int or q <= 0 for q in quantities.values()):
        raise ValueError("sealed-unit quantities must be positive integers")
    if any(type(q) is not int or q < 0 for q in orders.values()):
        raise ValueError("order quantities must be nonnegative integers")
    names, destinations = list(quantities), [*orders, "stock"]
    results = []
    for assignment in product(destinations, repeat=len(names)):
        totals = {region: 0 for region in orders}
        for name, region in zip(names, assignment):
            if region != "stock":
                totals[region] += quantities[name]
        if totals == dict(orders):
            results.append(dict(zip(names, assignment)))
    return results


def _m2(contract, frames, checks, evidence):
    units, orders = contract["units"], contract["orders"]
    allocations = enumerate_allocations(units, orders)
    checks.add("allocation_exists", bool(allocations))
    checks.add("identity_continuity", _identity(frames, units))
    destinations = [*orders, contract["stock_region"]]
    assignments, unclassified = {}, []
    for name in units:
        states = {region: _stable(frames, name, region) for region in destinations}
        matched = [region for region, state in states.items() if state is True]
        unique = len(matched) == 1
        value = True if unique else None if not matched and None in states.values() else False
        checks.add(f"placement:{name}", value)
        if unique:
            assignments[name] = matched[0]
        else:
            unclassified.append(name)
    totals = {region: sum(units[name] for name, assigned in assignments.items() if assigned == region)
              for region in orders}
    checks.add("exact_orders", totals == dict(orders) if not unclassified else False)
    evidence.update(assignments=assignments, totals=totals, feasible_allocation_count=len(allocations),
                    feasible_allocations=allocations)


def _native_inside(frame, name, unit):
    """Read the native Inside relation; support geometry cannot establish storage."""
    inside = _map(_map(frame.get("objects")).get(name)).get("inside")
    if not isinstance(inside, (list, tuple)) or any(not isinstance(value, str) for value in inside):
        return None
    return unit in inside


def _stable_inside(frames, name, unit):
    final = _native_inside(frames[-1], name, unit)
    if final is not True:
        return final
    end, previous = frames[-1]["time_s"], frames[-1]["time_s"]
    for sample in reversed(frames):
        now = sample["time_s"]
        if previous - now > MAX_SAMPLE_GAP_S + _EPS:
            return None
        inside = _native_inside(sample, name, unit)
        if inside is None:
            return None
        if not inside:
            break
        if end - now >= STABILITY_S - _EPS:
            return True
        previous = now
    return False


def _m3(contract, frames, checks, evidence):
    contents = contract["content_by_unit"]
    eligible = [unit for unit, content in contents.items() if content in contract["eligible_content"]]
    checks.add("eligible_unit_exists", bool(eligible))
    checks.add("identity_continuity", _identity(frames, [contract["payload"], *contents.values()]))
    # The private native state binds each named content to its actual unit.
    for unit, content in contents.items():
        checks.add(f"actual_contents:{unit}", _placement(frames[-1], content, unit))
        checks.add(f"resident_inside:{unit}", _native_inside(frames[-1], content, unit))
    if contract.get("require_closed") is True:
        equipment = _map(frames[-1].get("equipment"))
        for unit in contents:
            closed = _map(equipment.get(unit)).get("closed")
            checks.add(f"closed:{unit}", closed if type(closed) is bool else None)
    states = {unit: _stable(frames, contract["payload"], unit) for unit in contents}
    selected = [unit for unit, state in states.items() if state is True]
    unique = len(selected) == 1
    checks.add("payload_unique_unit", True if unique else None if not selected and None in states.values() else False)
    unit = selected[0] if unique else None
    checks.add("content_eligible", unit in eligible if unique else False)
    checks.add("payload_inside", _stable_inside(frames, contract["payload"], unit) if unique else None)
    checks.add("observed_selected_contents", _observed(frames, f"contents:{unit}") if unique else None)
    evidence.update(eligible_units=eligible, selected_unit=unit, actual_content_by_unit=dict(contents))


def enumerate_schedules(jobs, handling_s):
    """Enumerate all optimal earliest-operation schedules in a small abstract model.

    One robot performs sequential ``load`` and ``unload_deliver`` operations,
    each costing ``handling_s`` simulation seconds. A station is reserved from
    load start through unload/delivery end. Processing starts at load end and
    may overlap robot handling on another station. Unloading starts only after
    processing. At most five jobs are accepted so the enumeration is bounded.
    These witnesses are lower-bound planning references, not motor guarantees.
    """
    handling = _number(handling_s)
    if not isinstance(jobs, Mapping) or not jobs or len(jobs) > 5 or handling is None or handling <= 0:
        raise ValueError("schedules require one to five jobs and positive finite handling_s")
    normalized = {}
    for name, spec in jobs.items():
        spec = _map(spec)
        duration, station = _number(spec.get("duration_s")), spec.get("station")
        if duration is None or duration < 0 or not isinstance(station, str) or not station:
            raise ValueError("every job needs a station and nonnegative finite duration_s")
        normalized[name] = (station, duration)
    best, witnesses = math.inf, []
    examined = 0

    def visit(now, unloaded, active, delivered, operations):
        nonlocal best, witnesses, examined
        examined += 1
        if now > best + _EPS:
            return
        if len(delivered) == len(normalized):
            if now < best - _EPS:
                best, witnesses = now, []
            witnesses.append(operations)
            return
        for name, (station, duration) in normalized.items():
            if name in unloaded and station not in active:
                end = now + handling
                operation = {"operation": "load", "job": name, "station": station,
                             "start_s": now, "end_s": end, "processing_complete_s": end + duration}
                visit(end, unloaded - {name}, {**active, station: (name, end + duration)},
                      delivered, [*operations, operation])
            elif station in active and active[station][0] == name:
                start = max(now, active[station][1])
                end = start + handling
                operation = {"operation": "unload_deliver", "job": name, "station": station,
                             "start_s": start, "end_s": end}
                remaining = {key: value for key, value in active.items() if key != station}
                visit(end, unloaded, remaining, delivered | {name}, [*operations, operation])

    visit(0., set(normalized), {}, set(), [])
    return {"model": "abstract_single_robot_handling_v1", "optimal_makespan_s": best,
            "witnesses": witnesses, "search_nodes": examined, "enumeration_complete": True,
            "native_motor_timing_certified": False}


def _m4(contract, frames, checks, evidence):
    jobs, delivery = contract["jobs"], contract["delivery_region"]
    schedule = enumerate_schedules(jobs, contract["handling_s"])
    checks.add("identity_continuity", _identity(frames, jobs))
    gaps_ok = all(b["time_s"] - a["time_s"] <= MAX_SAMPLE_GAP_S + _EPS
                  for a, b in zip(frames, frames[1:]))
    checks.add("physics_tick_coverage", True if gaps_ok else None)
    stations = sorted({spec["station"] for spec in jobs.values()})
    state = {name: {"progress_s": 0., "completed_at_s": None, "run_start_index": None,
                    "load_index": None, "resets": 0, "held_episodes": []} for name in jobs}
    previous_holds = {name: None for name in jobs}
    starts = {}
    exclusive, single_held = True, True
    for index, sample in enumerate(frames):
        objects = _map(sample.get("objects"))
        held_values = {name: _map(obj).get("held") for name, obj in objects.items()}
        if any(type(held_values.get(name)) is not bool for name in jobs):
            single_held = None if single_held is not False else False
        if sum(value is True for value in held_values.values()) > 1:
            single_held = False
        for station in stations:
            occupied = []
            for name in jobs:
                held = held_values.get(name)
                placed = _occupancy(sample, name, station)
                if placed is None:
                    exclusive = None if exclusive is not False else False
                elif placed and held is False:
                    occupied.append(name)
            if len(occupied) > 1:
                exclusive = False
        for name, spec in jobs.items():
            entry, held = state[name], held_values.get(name)
            if held is True and previous_holds[name] is False:
                starts[name] = index
            if held is False and previous_holds[name] is True and name in starts:
                start_index = starts.pop(name)
                entry["held_episodes"].append({"start_index": start_index, "release_index": index,
                                               "start_s": frames[start_index]["time_s"],
                                               "release_s": sample["time_s"]})
            previous_holds[name] = held
            placed = _placement(sample, name, spec["station"])
            if placed is None:
                checks.add(f"processing_measurements:{name}:{index}", None)
            if entry["completed_at_s"] is not None:
                continue
            # Every competing released job invalidates occupancy for processing.
            rivals = [_occupancy(sample, other, spec["station"])
                      for other in jobs if other != name and held_values.get(other) is False]
            valid = placed is True and all(value is False for value in rivals)
            if valid:
                if entry["run_start_index"] is None:
                    entry["run_start_index"] = index
                    entry["load_index"] = index
                elif index and sample["time_s"] - frames[index - 1]["time_s"] <= MAX_SAMPLE_GAP_S + _EPS:
                    entry["progress_s"] += sample["time_s"] - frames[index - 1]["time_s"]
                else:
                    entry["progress_s"], entry["run_start_index"] = 0., index
                    entry["load_index"] = index
                if entry["progress_s"] >= spec["duration_s"] - _EPS:
                    entry["completed_at_s"] = sample["time_s"]
            elif entry["run_start_index"] is not None:
                entry["resets"] += 1
                entry["progress_s"], entry["run_start_index"], entry["load_index"] = 0., None, None
    checks.add("exclusive_stations", exclusive)
    checks.add("single_held_object", single_held)
    for name, spec in jobs.items():
        entry = state[name]
        complete = entry["completed_at_s"]
        checks.add(f"processed:{name}", complete is not None)
        load_index = entry["load_index"]
        loads = [ep for ep in entry["held_episodes"] if load_index is not None and ep["release_index"] <= load_index
                 and _placement(frames[ep["release_index"]], name, spec["station"], require_rest=False) is True
                 and all(_placement(f, name, spec["station"], require_rest=False) is True
                         for f in frames[ep["release_index"]:load_index + 1])]
        checks.add(f"load_transport:{name}", True if loads else None)
        deliveries = [ep for ep in entry["held_episodes"] if complete is not None and ep["start_s"] >= complete - _EPS
                      and _placement(frames[ep["release_index"]], name, delivery, require_rest=False) is True]
        checks.add(f"delivery_transport:{name}", True if deliveries else None)
        after = max(complete, deliveries[-1]["release_s"]) if complete is not None and deliveries else complete
        checks.add(f"delivered:{name}", _stable(frames, name, delivery, after=after) if complete is not None else False)
    final_clear = [_occupancy(frames[-1], name, station)
                   for name in jobs for station in stations]
    checks.add("stations_clear", _and(None if value is None else not value for value in final_clear))
    elapsed = frames[-1]["time_s"] - frames[0]["time_s"]
    deadline = _number(contract["deadline_s"])
    checks.add("deadline", elapsed <= deadline + _EPS if deadline is not None and deadline >= 0 else None)
    evidence.update(elapsed_simulation_s=elapsed, deadline_s=deadline, jobs=state, abstract_schedule=schedule)


def evaluate(prototype, contract, frame, history):
    """Return proof-aware M1--M4 checks; never mutate frames, history, or contract.

    ``evaluation_validity=unavailable`` means a required native measurement or
    transport proof is missing. Explicit observed violations are failed checks.
    All processing and dwell use native simulation time, never API latency.
    ``deadline_s`` is elapsed simulation time from the first history frame.
    """
    prototype = str(prototype).upper()
    if prototype not in {"M1", "M2", "M3", "M4"}:
        raise ValueError("prototype must be M1, M2, M3, or M4")
    checks, evidence = _Checks(), {}
    if not isinstance(contract, Mapping) or not isinstance(frame, Mapping) or not isinstance(history, (list, tuple)):
        checks.add("measured_inputs", None)
        return checks.result(prototype, evidence)
    frames = [*history, frame]
    times = [_number(_map(sample).get("time_s")) for sample in frames]
    valid_clock = all(t is not None for t in times) and all(b > a for a, b in zip(times, times[1:]))
    checks.add("native_clock", True if valid_clock else None)
    checks.add("history_baseline", True if history else None)
    if not valid_clock:
        return checks.result(prototype, evidence)
    checks.add("protected_unchanged", _protected(frames, contract.get("protected", [])))
    try:
        {"M1": _m1, "M2": _m2, "M3": _m3, "M4": _m4}[prototype](contract, frames, checks, evidence)
    except (KeyError, TypeError, ValueError) as error:
        checks.add("contract_available", None)
        evidence["contract_error"] = str(error)
    return checks.result(prototype, evidence)
