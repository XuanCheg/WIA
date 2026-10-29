"""Verify passive sleeping-contact diagnostics without native physics or GPUs.

Run: python -B -m pytest -q -p no:cacheprovider \
  tests/test_wia_rigid_contact_diagnostics.py tests/test_wia_rigid_sleeping_contact.py
Fixtures expose read-only native-shaped accessors; no success is inferred here.
"""
from copy import deepcopy
import json

import pytest

from wia.authoring.converters import rigid_partial_insertion as module
from harness_integration.test_wia_rigid_sleeping_contact import (
    DynamicLink, measure, native, seed, sleep_pair, sleeping_native,
)


def _diagnostic(native):
    return native.evaluator._contact_retention_diagnostics[('item', 'container')]


def _empty_contact():
    return {'available': False, 'relation': 'unresolved', 'penetrating': None,
            'reason': module._EMPTY_PAIR_CONTACT_REASON}


def test_native_storage_and_retention_counts_leave_raw_geometry_and_predicate_unchanged(sleeping_native):
    n = sleeping_native
    initial = seed(n)
    assert _diagnostic(n)['native_samples_stored'] == 1
    assert _diagnostic(n)['last_reason'] == 'native_sample_stored'
    assert 'retention_diagnostics' not in initial['contact_geometry']
    sleep_pair(n)
    raw = _empty_contact()
    retained = measure(n, 2.)
    diagnostic = retained['contact_geometry']['retention_diagnostics']
    assert retained['penetration_free'] is True
    assert diagnostic['native_samples_stored'] == diagnostic['retained_samples'] == 1
    assert diagnostic['last_reason'] == 'retained_sample'
    assert diagnostic['previous_witness_present'] is True
    assert diagnostic['previous_native_sample_time_s'] == 1.
    assert diagnostic['current_native_time_s'] == 2.
    assert diagnostic['current_stamp_attempted'] is diagnostic['current_stamp_available'] is True
    assert diagnostic['all_dynamic'] is diagnostic['all_asleep'] is diagnostic['exact_geometry_equal'] is True
    assert retained['contact_geometry']['current_contact_geometry'] == raw
    assert retained['contact_geometry']['original_contact_geometry'] == initial['contact_geometry']
    assert 'geometry_stamp' not in diagnostic and len(json.dumps(diagnostic)) < 2200
    assert measure(n, 2.)['contact_geometry']['retention_diagnostics']['retained_samples'] == 2


@pytest.mark.parametrize('failure,expected_reason', [
    ('no_previous', 'no_previous_native_sample'),
    ('awake', 'current_pair_awake'),
    ('not_dynamic', 'pair_not_dynamic'),
    ('geometry_changed', 'exact_geometry_changed'),
    ('protection_unavailable', 'control_protection_unavailable'),
    ('protection_changed', 'control_protection_changed'),
    ('touching_lost', 'native_touching_not_true'),
    ('stamp_failed', 'stamp_read_failure'),
])
def test_refusal_reasons_are_distinct_without_changing_current_contact_or_cache_outcome(
        sleeping_native, failure, expected_reason):
    n = sleeping_native
    if failure != 'no_previous':
        seed(n)
    sleep_pair(n)
    protection = {'available': True, 'unchanged': True}
    touching = True
    if failure == 'awake':
        n.container.links['body'].sleeping = False
    elif failure == 'not_dynamic':
        n.container.links['body'].attributes['physics:kinematicEnabled'] = True
    elif failure == 'geometry_changed':
        n.container.position[0] += 1e-12
    elif failure == 'protection_unavailable':
        protection = {'available': False, 'unchanged': None}
    elif failure == 'protection_changed':
        protection['unchanged'] = False
    elif failure == 'touching_lost':
        touching = False
    elif failure == 'stamp_failed':
        n.container.links['body'].sleeping = None
    current = _empty_contact()
    raw = deepcopy(current)
    previous_refusals = _diagnostic(n)['refusal_reasons'].get(expected_reason, 0)
    result = n.evaluator._contact_geometry_with_sleeping_retention(
        'item', 'container', touching, current, 2., protection)
    assert result is current
    assert {key: result[key] for key in raw} == raw
    assert not n.evaluator._contact_witnesses
    diagnostic = result['retention_diagnostics']
    assert diagnostic['last_reason'] == expected_reason
    assert diagnostic['refusal_reasons'][expected_reason] == previous_refusals + 1
    assert diagnostic['previous_witness_present'] is (failure != 'no_previous')
    assert diagnostic['current_stamp_attempted'] is (failure in ('awake', 'not_dynamic', 'geometry_changed', 'stamp_failed'))
    assert diagnostic['discard_count'] >= int(failure != 'no_previous')
    if failure == 'geometry_changed':
        assert diagnostic['exact_geometry_equal'] is False
    if failure == 'stamp_failed':
        assert diagnostic['current_stamp_available'] is False
        assert 'Native sleeping flag' in diagnostic['current_stamp_error']
        assert diagnostic['stamp_failure_context']['link_class'].endswith('.DynamicLink')
        assert diagnostic['stamp_failure_context']['stage'] == 'link_sleeping_flag'


def test_no_previous_witness_does_not_trigger_additional_geometry_reads(sleeping_native, monkeypatch):
    n = sleeping_native
    sleep_pair(n)
    def forbidden(*args):
        raise AssertionError('Diagnostics must not introduce a geometry stamp on the no-witness branch.')
    monkeypatch.setattr(module, '_pair_geometry_stamp', forbidden)
    relation = measure(n, 1.)
    assert relation['penetration_free'] is None
    assert relation['contact_geometry']['retention_diagnostics']['current_stamp_available'] is None
    assert relation['contact_geometry']['retention_diagnostics']['last_reason'] == 'no_previous_native_sample'


def test_failed_authored_shape_read_reports_link_class_and_stage_without_replacing_raw_reason(sleeping_native):
    n = sleeping_native
    seed(n)
    sleep_pair(n)
    n.mesh.prim.attributes['points'] = None
    relation = measure(n, 2.)
    geometry = relation['contact_geometry']
    diagnostic = geometry['retention_diagnostics']
    assert relation['penetration_free'] is None
    assert geometry['reason'] == module._EMPTY_PAIR_CONTACT_REASON
    assert diagnostic['stamp_failure_context']['stage'] == 'mesh_authored_shape'
    assert diagnostic['stamp_failure_context']['object_role'] == 'target'
    assert diagnostic['stamp_failure_context']['link_class'] == DynamicLink.__module__ + '.' + DynamicLink.__qualname__
    assert 'Authored native collision shape data is unavailable' in diagnostic['current_stamp_error']


def test_diagnostics_preserve_penetrating_verdict_and_native_record_is_compact(sleeping_native):
    n = sleeping_native
    seed(n, separation=-.002)
    sleep_pair(n)
    n.clock.current_time = 2.
    result = n.evaluator.record(n.observation)
    relation = result['native_measurements']['targets']['item']['containers']['container']
    assert relation['penetration_free'] is False
    assert relation['contact_geometry']['penetrating'] is True
    assert result['physical_success'] is result['task_success'] is False
    diagnostic = result['contact_retention_diagnostics'][0]
    assert (diagnostic['target_key'], diagnostic['container_key']) == ('item', 'container')
    assert diagnostic['last_reason'] == 'retained_sample'
    assert 'geometry_stamp' not in diagnostic and 'original_contact_geometry' not in diagnostic
    json.dumps(result, allow_nan=False)


def test_pair_read_failure_diagnostics_do_not_mask_contact_error_or_restore_witness(sleeping_native):
    n = sleeping_native
    seed(n)
    sleep_pair(n)
    n.target.touching['container'] = None
    relation = measure(n, 2.)
    assert relation['touching'] is relation['penetration_free'] is None
    assert 'Touching evidence is unavailable' in relation['contact_error']
    assert relation['contact_retention_diagnostics']['last_reason'] == 'native_contact_read_failure'
    assert not n.evaluator._contact_witnesses


def test_counters_reset_on_new_episode_without_retaining_old_contact(sleeping_native):
    n = sleeping_native
    seed(n)
    sleep_pair(n)
    measure(n, 2.)
    assert _diagnostic(n)['retained_samples'] == 1
    n.evaluator.begin_episode(n.observation)
    diagnostic = _diagnostic(n)
    assert diagnostic['native_samples_stored'] == diagnostic['retained_samples'] == 0
    assert diagnostic['last_reason'] == 'no_previous_native_sample'
    assert not n.evaluator._contact_witnesses
