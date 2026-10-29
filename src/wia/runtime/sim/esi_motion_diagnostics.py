"""Opt-in private contacts and encoder evidence at a stopped motion fault."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wia.runtime.sim.esi_base import _main_thread
from wia.runtime.sim.esi_omnigibson import _array


class PrivateMotionDiagnostics:
    def __init__(self, controller, output):
        self.controller, self.output = controller, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)

    def record(self, *, action, reason, phase, control_steps=None, result=None):
        _main_thread()
        record = {"schema": "private_motion_diagnostics_v1", "action": action, "reason": reason,
                  "phase": phase, "captured_at": datetime.now(timezone.utc).isoformat(),
                  "control_steps": control_steps}
        try:
            robot = self.controller.robot
            q = _array(robot.get_joint_positions()).astype(float)
            lower, upper = (_array(x).astype(float) for x in robot.control_limits["position"])
            finite = lambda value: float(value) if np.isfinite(value) else None
            record.update(joints={name: {"unit": "m" if "prismatic" in str(robot.joints[name].joint_type).lower() else "rad",
                "position": finite(value), "lower": finite(lo), "upper": finite(hi),
                "margin": finite(min(value-lo, hi-value))} for name, value, lo, hi in
                zip(robot.dof_names_ordered, q, lower, upper, strict=True)},
                base=self.controller.pose(), eef=self.controller.ee_pose(),
                ik=self.controller.ik_guard.diagnostics(), contacts=[], contact_reporting={})
            for name, link in robot.links.items():
                record["contact_reporting"][name] = bool(link.contact_reporting_enabled)
                if not link.contact_reporting_enabled:
                    continue
                for contact in link.contact_list():
                    entry = {"robot_link": name}
                    for field in ("body0", "body1", "position", "normal", "impulse"):
                        value = getattr(contact, field, None)
                        if value is not None:
                            entry[field] = value if isinstance(value, str) else np.asarray(value).tolist()
                    record["contacts"].append(entry)
            if result is not None:
                record["controller_result"] = result
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        # Diagnostic errors, including unavailable contact reporters or a full
        # evidence disk, must not interfere with the controller's bounded stop.
        try:
            with (self.output / "faults.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False)+"\n")
        except (OSError, ValueError, TypeError):
            return None
        return record
