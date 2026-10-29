"""Shared diagnostic records. Run src/wia/runtime/commands/preflight_esi_tasks.py --help."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

GROUPS = ("static", "physics", "navigation", "manipulation", "grasp", "goal", "perception")
LEVELS = ("static", "native_initialization", "geometric_screen", "physical_manipulation", "goal_verified")
STATUSES = ("PASS", "FAIL", "INCONCLUSIVE")


@dataclass
class Check:
    group: str
    name: str
    status: str
    reason: str
    objects: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    recommended_action: str | None = None
    verification_level: str = "static"
    required: bool = True

    def __post_init__(self):
        if self.status not in STATUSES or self.verification_level not in LEVELS:
            raise ValueError("Invalid preflight status or verification level.")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Options:
    mode: str = "static"
    seed: int | None = None
    variant: str | None = None
    wall_budget_s: float = 1800.
    candidate_count: int = 6
    stance_count: int = 12
    directions_deg: tuple[float, ...] = (0., 90., 180., 270.)
    search_budget_s: float = 240.
    settle_steps: int = 30
    hold_steps: int = 30
    lift_m: float = .06
    enabled: tuple[str, ...] = GROUPS
    data_root: str | None = None
    observation_config: str | None = None

    def to_dict(self):
        return asdict(self)


def aggregate(checks):
    values = [c.status if isinstance(c, Check) else c["status"] for c in checks
              if (c.required if isinstance(c, Check) else c.get("required", True))]
    if "FAIL" in values:
        return "FAIL"
    return "PASS" if values and all(v == "PASS" for v in values) else "INCONCLUSIVE"
