"""Verify formal/debug policy. Run: PYTHONPATH=src pytest -q tests/test_prechecks.py."""
from types import SimpleNamespace

from wia.runtime.prechecks import episode_qualification, permits_episode


def test_formal_run_skips_qualifier_without_fabricating_native_acceptance():
    session = SimpleNamespace(_wia_prechecks_enabled=False)
    def forbidden():
        raise AssertionError('The native debug qualifier must not run.')
    receipt = episode_qualification(session, forbidden)
    assert receipt['prechecks_run'] is False and receipt['accepted'] is None
    assert permits_episode(session, receipt)


def test_debug_and_direct_authoring_keep_measured_rejection():
    for session in (SimpleNamespace(), SimpleNamespace(_wia_prechecks_enabled=True)):
        result = episode_qualification(session, lambda: {'accepted': False, 'checks': {'native': False}})
        assert result['checks'] == {'native': False}
        assert not permits_episode(session, result)
