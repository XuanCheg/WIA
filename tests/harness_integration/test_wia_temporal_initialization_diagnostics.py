"""Check passive temporal placement failure evidence without simulation.

Run: python -B -m pytest -q -p no:cacheprovider
  tests/test_wia_temporal_initialization_diagnostics.py
Only producer-shaped evidence and a continuous-controller interface double are used.
"""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from wia.authoring.converters import temporal_understanding_two_phase as temporal


def test_failed_groups_retain_support_contact_height_and_region_diagnostics():
    failed = {'accepted': False, 'reason': 'No measured support placement candidate fits.',
        'failure_stage': 'support_candidate_generation', 'attempts': [],
        'support_search': [{'support': 'source_table', 'candidate_layouts': 0,
            'rejections': {'no_measured_upward_surface_in_height_window': 1},
            'surface_heights_m': [], 'bounds_m': [[1., 2., .1], [2., 3., .8]]}],
        'initial_support_contacts': {'container_00': False},
        'height_screen_before': {'container_00': {'bottom_above_floor_m': .72,
            'centre_above_floor_m': .82, 'within_height_window': True,
            'height_is_grasp_certificate': False}},
        'spawn_grid': {'requested_source_region': 'target_room_0',
            'measured_source_region': None, 'candidate_cell_count': 3}}
    second = {'accepted': False, 'reason': 'Native approach failed.',
        'failure_stage': 'native_support_and_robot_approach',
        'attempts': [{'candidate': 0}, {'candidate': 1, 'checks': {'collision_free': False}}],
        'layout_policy': 'group_specific_measured_layout',
        'spawn_grid': {'requested_source_region': 'target_room_0'}}
    evidence = {'accepted': False, 'reason': 'One or more meaningful task groups failed.',
        'layout_policy': 'independent_meaningful_groups',
        'per_group': [{'bindings': ['container_00'], 'evidence': failed},
                      {'bindings': ['container_01'], 'evidence': {'accepted': True}},
                      {'bindings': ['container_02'], 'evidence': second}]}
    before = deepcopy(evidence)
    controller = SimpleNamespace(rebase_interaction=Mock())
    session = SimpleNamespace(temporal_phase_controller=controller)
    with pytest.raises(ValueError, match='Temporal support placement failed: ') as error:
        temporal.on_interaction_initialized(session, evidence)
    diagnostic = json.loads(str(error.value).removeprefix('Temporal support placement failed: '))
    assert diagnostic['reason'] == evidence['reason']
    assert [row['bindings'] for row in diagnostic['groups']] == [['container_00'], ['container_02']]
    first, last = diagnostic['groups']
    for key in ('reason', 'failure_stage', 'support_search', 'initial_support_contacts', 'height_screen_before'):
        assert first[key] == failed[key]
    assert first['last_attempt'] == []
    assert first['layout_policy'] == evidence['layout_policy']
    assert first['spawn_grid'] == {'requested_source_region': 'target_room_0', 'measured_source_region': None}
    assert last['layout_policy'] == second['layout_policy']
    assert last['spawn_grid'] == {'requested_source_region': 'target_room_0'}
    assert last['last_attempt'] == second['attempts'][-1:]
    assert last['support_search'] is last['initial_support_contacts'] is last['height_screen_before'] is None
    assert evidence == before
    controller.rebase_interaction.assert_not_called()


def test_accepted_evidence_delegates_exact_original_object_and_return_value():
    evidence = {'accepted': True, 'per_group': [{'evidence': {'accepted': True}}]}
    returned = object()
    controller = SimpleNamespace(rebase_interaction=Mock(return_value=returned))
    session = SimpleNamespace(temporal_phase_controller=controller)
    assert temporal.on_interaction_initialized(session, evidence) is returned
    controller.rebase_interaction.assert_called_once_with(evidence)
    assert controller.rebase_interaction.call_args.args[0] is evidence


@pytest.mark.parametrize('accepted', [False, None, 1, 'yes'])
def test_acceptance_still_requires_literal_true(accepted):
    controller = SimpleNamespace(rebase_interaction=Mock())
    session = SimpleNamespace(temporal_phase_controller=controller)
    with pytest.raises(ValueError, match='Temporal support placement failed: '):
        temporal.on_interaction_initialized(session, {'accepted': accepted})
    controller.rebase_interaction.assert_not_called()


def test_missing_controller_preserves_original_exception():
    with pytest.raises(ValueError, match='Temporal placement requires the initialized continuous controller.'):
        temporal.on_interaction_initialized(SimpleNamespace(), {'accepted': True})
