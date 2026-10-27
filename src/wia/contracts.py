"""Model-free extension contracts. Run: python -c 'import wia.contracts'."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class PolicyCapabilities:
    images: bool = True
    history_owner: str = "policy"
    resume: bool = False
    system_kind: str = "policy"


@dataclass(frozen=True)
class PublicObservation:
    """Only evaluator-approved sensor/action feedback may enter this payload."""
    episode_id: str
    step: int
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class PublicEpisode:
    episode_id: str
    task: Mapping[str, Any]
    configuration: Mapping[str, Any]


@dataclass(frozen=True)
class PolicyDecision:
    action: Mapping[str, Any] | None
    done: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PrivateEvaluationContext:
    """Evaluator-only references; never passed to PolicySystem methods."""
    manifest_path: str
    evaluator_state_path: str | None = None


@runtime_checkable
class PolicySystem(Protocol):
    """Future policies, world models and WAM/Claude systems implement this lifecycle."""
    capabilities: PolicyCapabilities

    def initialize(self, configuration: Mapping[str, Any]) -> None: ...
    def start_episode(self, episode: PublicEpisode) -> None: ...
    def decide(self, observation: PublicObservation) -> PolicyDecision: ...
    def end_episode(self, public_outcome: Mapping[str, Any]) -> None: ...
    def close(self) -> None: ...


class TrialExecutor(Protocol):
    """An integration runs one whole episode and returns its authoritative summary."""
    def execute(self, plan: Mapping[str, Any]) -> Mapping[str, Any]: ...
