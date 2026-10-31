"""Score measured rigid partial insertion without importing a simulator.

Run: python -m pytest -q tests/test_rigid_insertion_scoring.py

``targets`` maps stable target keys to native object names. ``containers`` is
an iterable of allowed stable container keys (a mapping supplies its keys).
Call ``begin(initial_sample)`` once, then ``update(sample)`` for native physics
samples; ``result()`` only reads the latest result. All scientific thresholds
are explicit. This module neither evaluates QA answers nor infers an interior
from AABBs or a simulator's unaudited Inside relation.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real


SCORING_VERSION = "rigid_insertion_progress_v1"
FINAL_STATE_SCORING_VERSION = "rigid_insertion_final_state_v1"


def _number(value, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        return None
    return value


def _boolean(value):
    return value if type(value) is bool else None


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


class RigidInsertionProgress:
    """Latch verified target events while requiring a valid released final state.

    A sample contains ``native_time_s``, ``control_valid``, ``hand_clear``,
    ``grasp`` and ``targets``. The grasp mapping has ``available``, ``holding``,
    ``object_name``, ``constraint_active``, ``release_pending`` and
    ``gripper_open``. Each target mapping has its configured ``object_name``,
    ``lowest_z_m``, both speed norms, ``dynamic_collidable``, ``robot_contact``
    and a ``containers`` mapping. Each allowed container has ``interior_valid``,
    ``partial_inside``, ``touching`` and ``penetration_free`` booleans. The
    adapter must obtain these from native, audited measurements.

    A missing sample interrupts the dwell. ``max_sample_gap_s=None`` imposes no
    additional sampling interval threshold; callers needing one must configure
    it explicitly. Backward native time invalidates the episode until begin().
    An explicit prohibited-control observation latches a constraint violation.
    ``expected_containers`` optionally maps targets to one eligible container
    key or an iterable of eligible keys; omitted targets retain all named
    containers. Geometry from other containers remains diagnostic evidence.
    ``manipulation_revision='final_state_v1'`` requires ``minimum_lift_m=None``
    and verifies released stable insertion without pickup or lift prerequisites.
    Per-target verification remains historical and can occur sequentially.
    """

    def __init__(self, targets, containers, *, minimum_lift_m, stability_duration_s,
                 maximum_linear_speed_m_s, maximum_angular_speed_rad_s,
                 max_sample_gap_s=None, expected_containers=None, manipulation_revision=None):
        if manipulation_revision not in (None, "final_state_v1"):
            raise ValueError("Unsupported rigid insertion manipulation_revision.")
        self.manipulation_revision = manipulation_revision
        if manipulation_revision == "final_state_v1" and minimum_lift_m is not None:
            raise ValueError("final_state_v1 requires minimum_lift_m to be explicitly None.")
        if not isinstance(targets, Mapping) or not targets:
            raise ValueError("targets must map stable keys to native object names.")
        if any(not isinstance(key, str) or not key or not isinstance(name, str) or not name
               for key, name in targets.items()):
            raise ValueError("Target keys and native object names must be nonempty strings.")
        if len(set(targets.values())) != len(targets):
            raise ValueError("Native target object names must be distinct.")
        if isinstance(containers, str):
            raise ValueError("containers must contain stable container keys.")
        container_keys = list(dict.fromkeys(containers))
        if not container_keys or any(not isinstance(key, str) or not key for key in container_keys):
            raise ValueError("Allowed container keys must be nonempty strings.")
        self.targets = dict(targets)
        self.containers = tuple(container_keys)
        if expected_containers is not None and not isinstance(expected_containers, Mapping):
            raise ValueError("expected_containers must map target keys to eligible container keys.")
        expected = {} if expected_containers is None else expected_containers
        if set(expected) - set(self.targets):
            raise ValueError("expected_containers contains an unknown target key.")
        self.expected_containers = {}
        for key in self.targets:
            eligible = expected.get(key, self.containers)
            eligible = [eligible] if isinstance(eligible, str) else list(eligible)
            if not eligible or any(value not in self.containers for value in eligible):
                raise ValueError("Each expected container must be a named allowed container.")
            self.expected_containers[key] = tuple(value for value in self.containers if value in eligible)
        self.thresholds = {}
        for key, value, positive in (
                ("minimum_lift_m", minimum_lift_m, True),
                ("stability_duration_s", stability_duration_s, True),
                ("maximum_linear_speed_m_s", maximum_linear_speed_m_s, False),
                ("maximum_angular_speed_rad_s", maximum_angular_speed_rad_s, False)):
            if key == "minimum_lift_m" and self.manipulation_revision == "final_state_v1":
                self.thresholds[key] = None
                continue
            number = _number(value, nonnegative=True)
            if number is None or (positive and number == 0):
                raise ValueError(f"{key} must be finite and {'positive' if positive else 'nonnegative'}.")
            self.thresholds[key] = number
        gap = _number(max_sample_gap_s, nonnegative=True)
        if max_sample_gap_s is not None and (gap is None or gap == 0):
            raise ValueError("max_sample_gap_s must be finite and positive, or None.")
        self.thresholds["max_sample_gap_s"] = gap
        self._reset()

    def _reset(self):
        self._begun = False
        self._last_time = None
        self._clock_invalid = False
        self._control_violation = False
        self._pickup_armed = False
        self._all_verified_at = None
        self._states = {key: {
            "object_name": name, "baseline_lowest_z_m": None,
            "eligible_containers": list(self.expected_containers[key]),
            "grasp_seen": False, "lift_seen": False, "release_seen": False,
            "maximum_held_lift_m": None, "dwell_container": None,
            "dwell_started_at_s": None, "stable_duration_s": 0.0,
            "verified": False, "verified_container": None, "verified_at_s": None,
            "verified_evidence": None,
            "evaluation_validity": "unavailable", "measurement_errors": ["episode_not_started"],
            "conditions": {}, "containers": {},
        } for key, name in self.targets.items()}
        self._current = {"native_time_s": None, "measurement_errors": ["episode_not_started"],
                         "available": False, "final_released": None,
                         "empty_open_hands": None, "control_valid": None,
                         "unbound_held_object": None}

    def begin(self, initial_sample):
        """Start a new episode; an existing initial hold is never new grasp evidence."""
        self._reset()
        self._begun = True
        sample = _mapping(initial_sample)
        initial_targets = _mapping(sample.get("targets"))
        for key, state in self._states.items():
            measured = _mapping(initial_targets.get(key))
            if measured.get("object_name") == state["object_name"]:
                state["baseline_lowest_z_m"] = _number(measured.get("lowest_z_m"))
        return self._consume(sample, initial=True)

    def update(self, sample):
        """Consume one measurement; duplicate score reads cannot create elapsed time."""
        if not self._begun:
            return self.result()
        return self._consume(_mapping(sample), initial=False)

    @staticmethod
    def _grasp(sample):
        grasp = _mapping(sample.get("grasp"))
        fields = ("holding", "constraint_active", "release_pending", "gripper_open")
        available = (grasp.get("available") is True
                     and all(_boolean(grasp.get(key)) is not None for key in fields)
                     and "object_name" in grasp)
        name = grasp.get("object_name")
        if name is not None and (not isinstance(name, str) or not name):
            available = False
        if grasp.get("holding") is True and name is None:
            available = False
        if grasp.get("constraint_active") is True and grasp.get("holding") is not True:
            available = False
        empty_open = None
        if available:
            empty_open = bool(name is None and grasp["holding"] is False
                              and grasp["constraint_active"] is False
                              and grasp["release_pending"] is False
                              and grasp["gripper_open"] is True)
        return grasp, available, empty_open

    @staticmethod
    def _reset_dwell(state):
        state["dwell_container"] = None
        state["dwell_started_at_s"] = None
        state["stable_duration_s"] = 0.0

    def _consume(self, sample, *, initial):
        errors = []
        now = _number(sample.get("native_time_s"))
        gap = False
        if now is None:
            errors.append("native_time_unavailable")
        elif self._last_time is not None:
            if now < self._last_time:
                self._clock_invalid = True
            limit = self.thresholds["max_sample_gap_s"]
            gap = limit is not None and now - self._last_time > limit
        if self._clock_invalid:
            errors.append("native_time_reversed")
        if now is not None:
            self._last_time = now
        control = _boolean(sample.get("control_valid"))
        if control is False:
            self._control_violation = True
        if control is None:
            errors.append("control_valid_unavailable")
        hand_clear = _boolean(sample.get("hand_clear"))
        if hand_clear is None:
            errors.append("hand_clear_unavailable")
        grasp, grasp_available, empty_open = self._grasp(sample)
        if not grasp_available:
            errors.append("grasp_unavailable")
        if empty_open is True:
            self._pickup_armed = True
        held_name = grasp.get("object_name") if grasp_available and grasp.get("holding") else None
        unbound_held = held_name if held_name is not None and held_name not in self.targets.values() else None
        target_samples = _mapping(sample.get("targets"))
        contacts = []
        for key, state in self._states.items():
            target = _mapping(target_samples.get(key))
            target_errors = []
            identity_valid = target.get("object_name") == state["object_name"]
            if not identity_valid:
                target_errors.append("target_identity_unavailable_or_mismatched")
            contact = _boolean(target.get("robot_contact")) if identity_valid else None
            contacts.append(contact)
            if contact is None:
                target_errors.append("robot_contact_unavailable")
            self._observe_target(state, target, identity_valid, grasp, grasp_available,
                                 empty_open, hand_clear, contact, now, control,
                                 errors, target_errors, gap=gap, initial=initial)
            # After verification, geometry and velocity are historical facts.
            # Current target contact is still required for an unsupported final release.
            state["measurement_errors"] = target_errors
            state["evaluation_validity"] = "unavailable" if target_errors else "valid"
        all_contacts_clear = (False if any(value is True for value in contacts) else
                              None if any(value is None for value in contacts) else True)
        released_parts = (empty_open, hand_clear, all_contacts_clear)
        final_released = (False if any(value is False for value in released_parts) else
                          None if any(value is None for value in released_parts) else True)
        if all(state["verified"] for state in self._states.values()) and self._all_verified_at is None:
            self._all_verified_at = now
        self._current = {
            "native_time_s": now, "measurement_errors": errors,
            "available": not errors and all(not state["measurement_errors"] for state in self._states.values()),
            "final_released": final_released, "empty_open_hands": empty_open,
            "control_valid": control, "unbound_held_object": unbound_held,
            "sample_gap_exceeded": gap,
        }
        return self.result()

    def _observe_target(self, state, target, identity_valid, grasp, grasp_available,
                        empty_open, hand_clear, contact, now, control, global_errors,
                        errors, *, gap, initial):
        baseline = state["baseline_lowest_z_m"]
        lowest_z = _number(target.get("lowest_z_m")) if identity_valid else None
        linear = _number(target.get("linear_speed_m_s"), nonnegative=True) if identity_valid else None
        angular = _number(target.get("angular_speed_rad_s"), nonnegative=True) if identity_valid else None
        dynamic = _boolean(target.get("dynamic_collidable")) if identity_valid else None
        measurements = (("lowest_z_m", lowest_z),
                            ("linear_speed_m_s", linear), ("angular_speed_rad_s", angular),
                            ("dynamic_collidable", dynamic))
        if self.manipulation_revision is None:
            measurements += (("baseline_lowest_z_m", baseline),)
        for name, value in measurements:
            if value is None and not state["verified"]:
                errors.append(name + "_unavailable")
        held = bool(grasp_available and grasp.get("holding") is True
                    and grasp.get("constraint_active") is True
                    and grasp.get("object_name") == state["object_name"])
        if (not state["verified"] and not initial and self._pickup_armed and held and identity_valid
                and dynamic is True and now is not None and not self._clock_invalid
                and control is True):
            state["grasp_seen"] = True
            state["release_seen"] = False
            if baseline is not None and lowest_z is not None:
                lift = lowest_z - baseline
                previous = state["maximum_held_lift_m"]
                state["maximum_held_lift_m"] = lift if previous is None else max(previous, lift)
                if (self.manipulation_revision is None
                        and lift >= self.thresholds["minimum_lift_m"]):
                    state["lift_seen"] = True
        if not state["verified"] and not initial and state["lift_seen"] and empty_open is True:
            state["release_seen"] = True
        candidates = []
        ineligible_insertions = []
        container_diagnostics = {}
        relations = _mapping(target.get("containers")) if identity_valid else {}
        for key in self.containers:
            relation = _mapping(relations.get(key))
            flags = {name: _boolean(relation.get(name)) for name in
                     ("interior_valid", "partial_inside", "touching", "penetration_free")}
            available = flags["interior_valid"] is True and all(value is not None for value in flags.values())
            inserted = available and all(flags.values())
            eligible = key in state["eligible_containers"]
            container_diagnostics[key] = dict(flags, evaluation_validity="valid" if available else "unavailable",
                                               insertion_conditions_satisfied=bool(inserted), eligible=eligible)
            if inserted and eligible:
                candidates.append(key)
            elif inserted:
                ineligible_insertions.append(key)
        if (not state["verified"] and not candidates
                and any(row["evaluation_validity"] == "unavailable" and row["eligible"]
                        for row in container_diagnostics.values())):
            errors.append("audited_interior_or_container_contact_unavailable")
        state["containers"] = container_diagnostics
        conditions = {
            "held_by_measured_constraint": held,
            "dynamic_collidable": dynamic,
            "robot_contact_clear": None if contact is None else not contact,
            "hand_clear": hand_clear,
            "empty_open_hands": empty_open,
            "slow_linear_motion": None if linear is None else linear <= self.thresholds["maximum_linear_speed_m_s"],
            "slow_angular_motion": None if angular is None else angular <= self.thresholds["maximum_angular_speed_rad_s"],
            "allowed_partial_insertion": bool(candidates),
            "ineligible_partial_insertion": bool(ineligible_insertions),
            "ineligible_inserted_containers": ineligible_insertions,
        }
        state["conditions"] = conditions
        if state["verified"]:
            return
        history_satisfied = (self.manipulation_revision == "final_state_v1" or
                             (state["grasp_seen"] and state["lift_seen"] and state["release_seen"]))
        stable = bool(not initial and not global_errors and not errors and control is True
                      and not self._control_violation and history_satisfied
                      and empty_open is True and hand_clear is True
                      and contact is False and dynamic is True and candidates
                      and conditions["slow_linear_motion"] and conditions["slow_angular_motion"])
        if not stable:
            self._reset_dwell(state)
            return
        previous_container = state["dwell_container"]
        container = previous_container if previous_container in candidates else candidates[0]
        if gap or previous_container != container or state["dwell_started_at_s"] is None:
            state["dwell_started_at_s"] = now
            state["dwell_container"] = container
        state["stable_duration_s"] = now - state["dwell_started_at_s"]
        if state["stable_duration_s"] >= self.thresholds["stability_duration_s"]:
            state["verified"] = True
            state["verified_container"] = container
            state["verified_at_s"] = now
            state["verified_evidence"] = {
                "native_time_s": now, "lowest_z_m": lowest_z,
                "linear_speed_m_s": linear, "angular_speed_rad_s": angular,
                "robot_contact": contact, "hand_clear": hand_clear,
                "empty_open_hands": empty_open, "control_valid": control,
                "dynamic_collidable": dynamic, "container": container,
                "container_measurements": deepcopy(container_diagnostics[container]),
                "stable_duration_s": state["stable_duration_s"],
            }

    def result(self):
        """Return a JSON-safe copy; reading never changes grasp or dwell evidence."""
        current = self._current
        verified = [key for key, state in self._states.items() if state["verified"]]
        available = current["available"]
        constraints = False if self._control_violation else current["control_valid"]
        success = bool(available and constraints is True and current["final_released"] is True
                       and len(verified) == len(self.targets))
        total = len(self.targets)
        score = len(verified) / total if available else None
        items = [{"id": "target:" + key, "completed": state["verified"],
                  "evaluation_validity": "valid" if state["verified"] else state["evaluation_validity"],
                  "container": state["verified_container"], "verified_at_s": state["verified_at_s"]}
                 for key, state in self._states.items()]
        version = FINAL_STATE_SCORING_VERSION if self.manipulation_revision == "final_state_v1" else SCORING_VERSION
        result = {
            "schema": version, "scoring_version": version,
            "task_success": success, "physical_success": success,
            "evaluation_validity": "complete" if available else "unavailable",
            "constraints_satisfied": constraints, "completion_items": items,
            "completed_goals": len(verified), "total_goals": total,
            "completion_score": score, "score_percent": None if score is None else score * 100,
            "diagnostic_lower_bound": len(verified) / total,
            "verified_targets": verified, "per_target": self._states,
            "all_targets_verified_at_s": self._all_verified_at,
            "final_released": current["final_released"],
            "empty_open_hands": current["empty_open_hands"],
            "native_time_s": current["native_time_s"],
            "measurement_errors": current["measurement_errors"],
            "unbound_held_object": current["unbound_held_object"],
            "sample_gap_exceeded": current.get("sample_gap_exceeded", False),
            "thresholds": self.thresholds,
            "expected_containers": {key: list(value) for key, value in self.expected_containers.items()},
        }
        if self.manipulation_revision is not None:
            result["manipulation_revision"] = self.manipulation_revision
        return deepcopy(result)
