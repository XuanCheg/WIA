"""Environment extension protocols. Run: python -c 'import wia.environments'."""
from typing import Protocol
from wia.contracts import PolicyDecision, PublicObservation


class Environment(Protocol):
    """A future executor can own this interface; WIA core does not step physics."""
    def observe(self) -> PublicObservation: ...
    def step(self, decision: PolicyDecision) -> PublicObservation: ...
    def close(self) -> None: ...
