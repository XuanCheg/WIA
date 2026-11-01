"""Separate explicit debug qualification from formal episode construction.

Used by run_esi_robot.py and native scorer adapters; no standalone invocation.
Formal runs still construct their world and collect scoring baselines.
"""


def enabled(session):
    """Direct authoring/qualification callers remain explicit debug operations."""
    return getattr(session, '_wia_prechecks_enabled', True)


def episode_qualification(session, qualifier, *args, **kwargs):
    """Run a debug qualifier, or explicitly leave its predicates unmeasured."""
    if enabled(session):
        return qualifier(*args, **kwargs)
    return {'status': 'not_run', 'prechecks_run': False, 'accepted': None, 'checks': {}}


def permits_episode(session, receipt):
    """Formal episodes rely on prior debug validation, not a fabricated pass."""
    return not enabled(session) or receipt.get('accepted') is True


def skipped(output, patches):
    """Record an omitted precheck without certifying any measured predicate."""
    from wia.tasks.scene import QualificationResult
    result = QualificationResult('not_run', None, {
        'prechecks_run': False, 'policy': 'formal_run_uses_prior_debug_validation'})
    result.save(output, patches=patches)
    return result
