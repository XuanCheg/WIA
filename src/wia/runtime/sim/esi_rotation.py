"""R1 local-axis wrist rotation configuration; no simulator imports or public parameters."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wia.runtime.sim.esi_omnigibson import SHOWHARNESS_ROOT

ROTATION_FRAME = {
    "reference": "current end-effector frame at macro start",
    "axis": "+Z for CW/CCW, +Y for PITCH_POS/NEG, +X for ROLL_POS/NEG",
    "tilt_positive_direction": "right-hand rule about the named local axis at macro start",
    "positive_direction": "clockwise looking along end-effector +Z toward the fingers",
    "position_hold": "robot_base",
}


@dataclass(frozen=True)
class RotationControlConfig:
    rotation_deg: float = 15.0
    rotation_tolerance_deg: float = 1.0
    max_rotation_speed_deg_s: float = 15.0
    max_step_rotation_deg: float = 2.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Rotation configuration {name} must be finite and positive.")
        if not self.rotation_tolerance_deg < self.rotation_deg < 180:
            raise ValueError("Rotation tolerance must be smaller than a step below 180 degrees.")

    @classmethod
    def load(cls, path: str | Path = SHOWHARNESS_ROOT / "configs/esi_rotation.json"):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values.pop("path_tolerance_deg", None)  # Inert historical configuration.
        return cls(**values)


def rotation_axis_sign(token):
    """Return a local axis index and signed angle multiplier for an R1 token."""
    return {"ROTATE_CW": (2, 1), "ROTATE_CCW": (2, -1),
            "ROTATE_PITCH_POS": (1, 1), "ROTATE_PITCH_NEG": (1, -1),
            "ROTATE_ROLL_POS": (0, 1), "ROTATE_ROLL_NEG": (0, -1)}[token]
