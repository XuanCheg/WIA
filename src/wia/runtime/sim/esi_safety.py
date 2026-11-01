"""Compatibility imports for the independent Safety evaluator.

Run: python -m pytest tests/harness_integration/test_core_revision_shared.py.
New runtime integrations should import wia.evaluation.safety directly.
"""
from wia.evaluation.safety import (
    VERSION, SafetyEvaluator, SafetyObserver, event_statistics, private_safety_result,
    record_safety, run_safety_hook, safety_score,
)

__all__ = ['VERSION', 'SafetyEvaluator', 'SafetyObserver', 'event_statistics', 'private_safety_result',
           'record_safety', 'run_safety_hook', 'safety_score']
