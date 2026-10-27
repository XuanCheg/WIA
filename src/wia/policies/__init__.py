"""Lazy system integration registry. Run: python -m wia backends."""
from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points

_REGISTRY = {"show_harness": "wia.policies.show_harness:ShowHarnessExecutor"}


def register_policy(name, executor_factory):
    """Register a whole-trial executor factory or module:attribute import target."""
    if not name or name in _REGISTRY:
        raise ValueError(f"Policy integration already registered or invalid: {name}")
    if not callable(executor_factory) and not isinstance(executor_factory, str):
        raise TypeError("Provide a callable factory or module:attribute string.")
    _REGISTRY[name] = executor_factory


def available_policies():
    discovered = entry_points(group="wia.policy_systems")
    return sorted(set(_REGISTRY) | {entry.name for entry in discovered})


def create_executor(name):
    target = _REGISTRY.get(name)
    if target is None:
        matches = [entry for entry in entry_points(group="wia.policy_systems") if entry.name == name]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous policy integration: {name}")
        return matches[0].load()()
    if isinstance(target, str):
        module, attribute = target.split(":", 1)
        target = getattr(import_module(module), attribute)
    return target()
