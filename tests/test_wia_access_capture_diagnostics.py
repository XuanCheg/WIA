"""Check private opt-in inventory capture without importing a native simulator.

Run: PYTHONPATH=src:../Show-Harness-Explore python -B -m pytest -q \
  -p no:cacheprovider tests/test_wia_access_capture_diagnostics.py
"""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from wia.runtime.wia_capture_diagnostics import (
    ACCESS_MODULES, ACCESS_REPAIR_MARKER, capture_initial_review_diagnostics,
)
from wia.runtime.wia_access_capture_diagnostics import SCHEMA


class Clock:
    def __init__(self, values=(4.5, 4.5)):
        self.values = iter(values)
        self.reads = 0

    @property
    def current_time(self):
        self.reads += 1
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture
def capture(monkeypatch):
    import wia.candidates.inventory as native_inventory

    calls, verified = [], []
    inventory = {
        'schema': 'wia_candidate_native_inventory_v1', 'read_only': True,
        'robot': {'joint_calibration': {'source': 'native'}, 'base_geometry': {'available': True}},
        'objects': [{'name': 'exact_item', 'role': 'target'}],
        'structural_targets': [{'name': 'exact_floor', 'role': 'structural_target'}],
    }

    def export(session, **kwargs):
        calls.append((session, deepcopy(kwargs)))
        return inventory

    monkeypatch.setattr(native_inventory, 'export_inventory', export)
    marker = {'alias': 'D1-1', 'revision': 'review_access_20261030_v1'}
    evaluation = NS(type='category_operation', module=sorted(ACCESS_MODULES)[0])
    bindings = {'target': NS(name='exact_item'), 'support': NS(name='exact_table'),
                'floor': NS(name='exact_floor'), 'extra_structure': NS(name='exact_cabinet')}
    task = NS(definition=NS(id='task_id', repair={ACCESS_REPAIR_MARKER: marker},
                           private=NS(evaluation=evaluation, bindings=bindings)),
              file_sha256='b' * 64, verify_unchanged=lambda: verified.append(True))
    setup = {'supported_items': {'exact_item': {'support': 'exact_table'},
                                'unmapped_item': {'support': None}}}
    clock = Clock()
    session = NS(loaded_task=task, category_interaction_setup=setup,
                 pipeline=NS(og=NS(sim=clock)))
    return NS(session=session, task=task, marker=marker, evaluation=evaluation,
              calls=calls, verified=verified, clock=clock, setup=setup,
              inventory=inventory, native_inventory=native_inventory)


@pytest.mark.parametrize('module', sorted(ACCESS_MODULES))
def test_opt_in_exports_all_exact_bindings_supports_and_detached_inventory(capture, module):
    capture.evaluation.module = module
    original_setup = deepcopy(capture.setup)
    result = capture_initial_review_diagnostics(capture.session)
    assert result['schema'] == SCHEMA and result['private_only'] is result['available'] is True
    assert result['task_sha256'] == capture.task.file_sha256
    assert result['task_id'] == 'task_id' and result['module'] == module
    assert result['revision'] == capture.marker['revision'] and result['alias'] == 'D1-1'
    assert result['native_qualification_run'] is result['physical_success_verified'] is False
    assert result['native_time_before_s'] == result['native_time_after_s'] == 4.5
    assert result['native_time_verified_unchanged'] is True and result['errors'] == []
    assert result['robot_actions'] == result['physics_steps'] == 0
    assert capture.clock.reads == 2 and capture.verified == [True]
    assert capture.calls == [(capture.session, {
        'identity': 'D1-1',
        'target_names': ['exact_item', 'exact_table', 'exact_floor', 'exact_cabinet'],
        'support_by_target': {'exact_item': 'exact_table'},
    })]
    assert result['inventory'] == capture.inventory and result['inventory'] is not capture.inventory
    capture.inventory['robot']['joint_calibration']['source'] = 'changed'
    assert result['inventory']['robot']['joint_calibration']['source'] == 'native'
    assert capture.setup == original_setup
    assert not {'public_task', 'observation', 'prompt', 'accepted', 'task_success'} & result.keys()


@pytest.mark.parametrize('case', ['missing_marker', 'false_marker', 'wrong_module', 'wrong_type', 'non_dict_repair'])
def test_unrelated_tasks_do_not_export_read_clock_or_verify(capture, case):
    if case == 'missing_marker':
        capture.task.definition.repair = {'different_review': {}}
    elif case == 'false_marker':
        capture.task.definition.repair[ACCESS_REPAIR_MARKER] = False
    elif case == 'wrong_module':
        capture.evaluation.module = 'wia.authoring.converters.wia_room_collection'
    elif case == 'wrong_type':
        capture.evaluation.type = 'grasp_lift'
    else:
        capture.task.definition.repair = NS(user_review_access_v1=capture.marker)
    assert capture_initial_review_diagnostics(capture.session) is None
    assert capture.clock.reads == 0 and not capture.calls and not capture.verified


def test_missing_alias_and_support_setup_use_task_id_and_no_synthetic_support(capture):
    del capture.marker['alias']
    del capture.session.category_interaction_setup
    result = capture_initial_review_diagnostics(capture.session)
    assert result['available'] is True
    assert capture.calls[0][1]['identity'] == 'task_id'
    assert capture.calls[0][1]['support_by_target'] is None


def test_exporter_error_is_private_unavailable_metadata(capture, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError('Native calibrated collision geometry unavailable.')

    monkeypatch.setattr(capture.native_inventory, 'export_inventory', broken)
    result = capture_initial_review_diagnostics(capture.session)
    assert result['available'] is False and result['private_only'] is True
    assert result['inventory'] is None and 'Native calibrated collision geometry unavailable' in result['error']
    assert result['errors'] == [result['error']]
    assert result['robot_actions'] == result['physics_steps'] == 0
    assert capture.clock.reads == 2


@pytest.mark.parametrize('times', [(4.5, 4.500000001), (float('nan'), 4.5),
                                  (4.5, float('inf')), (True, 4.5),
                                  (RuntimeError('Clock unavailable'), RuntimeError('Clock unavailable'))])
def test_changed_or_unavailable_clock_never_grants_zero_action_certificate(capture, times):
    capture.session.pipeline.og.sim = Clock(times)
    result = capture_initial_review_diagnostics(capture.session)
    assert result['available'] is False
    assert result['native_time_verified_unchanged'] is False
    assert result['robot_actions'] is result['physics_steps'] is None
    assert result['native_qualification_run'] is result['physical_success_verified'] is False
    assert 'unchanged native simulation time' in result['error']
    assert capture.session.pipeline.og.sim.reads == 2
    assert bool(capture.calls) is (times[0] == 4.5)


def test_task_change_prevents_export_and_malformed_export_is_unavailable(capture, monkeypatch):
    def changed():
        raise RuntimeError('Task changed after loading.')

    capture.task.verify_unchanged = changed
    result = capture_initial_review_diagnostics(capture.session)
    assert result['available'] is False and not capture.calls
    assert 'Task changed after loading' in result['error']
    capture.task.verify_unchanged = lambda: None
    capture.session.pipeline.og.sim = Clock()
    monkeypatch.setattr(capture.native_inventory, 'export_inventory', lambda *args, **kwargs: [])
    result = capture_initial_review_diagnostics(capture.session)
    assert result['available'] is False and result['inventory'] is None
    assert 'must return a diagnostic dictionary' in result['error']
