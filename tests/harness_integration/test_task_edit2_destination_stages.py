"""CPU behavior checks separating destination initialization from episode proof.

Run: python -B -m pytest -q -p no:cacheprovider
 tests/test_task_edit2_destination_stages.py tests/test_task_edit2_missing_definitions.py.
Native-row fixtures exercise validator gates without loading a renderer or GPU.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from wia.authoring.converters import task_edit2_destinations as destinations
from wia.tasks.interaction_supports import support_names
from wia.tasks.delivery_support import SLOT

EPISODE_GATES = {'current_episode_receipt', 'native_surface_fit_verified',
    'public_visual_label_association_verified', 'public_control_approach_verified',
    'public_control_release_verified'}


@pytest.fixture
def world(monkeypatch):
    rows = [{'name': 'desk', 'category': 'desk', 'model': 'desk_model', 'room': 'room_0',
        'centre_m': [.5, .5, .6], 'aabb_m': [[0., 0., .5], [1., 1., .7]]},
        {'name': 'target', 'category': 'cup', 'model': 'cup_model', 'room': 'room_0',
        'centre_m': [2., 2., .3], 'aabb_m': [[1.9, 1.9, .2], [2.1, 2.1, .4]]}]
    monkeypatch.setattr(destinations, '_native_rows', lambda session: rows)
    contract = {'scene': 'building', 'room': 'room_0', 'source_identity': 'source', 'selection_seed': 7,
        'destination_binding': {'name': 'desk', 'category': 'desk', 'model': 'desk_model'},
        'target_bindings': {'target': {'name': 'target', 'category': 'cup', 'model': 'cup_model'}},
        'label_relation': {'kind': 'unique_room_category'}, 'fallback_table': False, 'fit_margin_m': .04}
    config = {'bindings': {'destination': 'desk', 'target': 'target'}, 'parameters': {'destination_contract': contract}}
    return SimpleNamespace(rows=rows, contract=contract, config=config, session=SimpleNamespace(scene_name='building'))


def full_receipt(world):
    world.session.episode_id = 'episode'
    world.session.destination_qualification = {'source': {'episode_id': 'episode', 'selection_seed': 7,
        'destination_name': world.contract['destination_binding']['name'],
        **{key: True for key in EPISODE_GATES - {'current_episode_receipt'}}}}


def test_structural_initialization_does_not_require_future_episode_release(world):
    before_rows, before_config, before_session = deepcopy(world.rows), deepcopy(world.config), deepcopy(vars(world.session))
    result = destinations.validate_destination(world.session, world.config, stage='initialization')
    assert result['accepted'] and all(result['checks'].values())
    assert result['checks']['native_aabb_footprint_fits_all_targets']
    assert EPISODE_GATES.isdisjoint(result['checks'])
    assert set(result['pending_qualification']) == EPISODE_GATES
    assert result['qualification_complete'] is False and result['evaluation_validity'] == 'unavailable'
    assert world.rows == before_rows and world.config == before_config and vars(world.session) == before_session


def test_default_evaluation_keeps_all_current_episode_evidence_required(world):
    result = destinations.validate_destination(world.session, world.config)
    explicit = destinations.validate_destination(world.session, world.config, stage='evaluation')
    assert result == explicit
    assert not result['accepted'] and result['evaluation_validity'] == 'unavailable'
    assert not result['qualification_complete'] and set(result['pending_qualification']) == EPISODE_GATES
    assert all(result['checks'][name] is False for name in EPISODE_GATES)


def test_existing_full_receipt_still_passes_and_initialization_remains_pending(world):
    full_receipt(world)
    result = destinations.validate_destination(world.session, world.config)
    assert result['accepted'] and result['qualification_complete']
    assert result['evaluation_validity'] == 'complete' and result['pending_qualification'] == []
    initial = destinations.validate_destination(world.session, world.config, stage='initialization')
    assert initial['accepted'] and not initial['qualification_complete']
    assert set(initial['pending_qualification']) == EPISODE_GATES
    assert EPISODE_GATES.isdisjoint(initial['checks'])


@pytest.mark.parametrize('failure', ['scene', 'destination_identity', 'target_identity', 'bound_destination', 'label', 'size', 'missing_target'])
def test_structural_failures_reject_both_stages_despite_full_episode_receipt(world, failure):
    full_receipt(world)
    if failure == 'scene':
        world.session.scene_name = 'other_building'
    elif failure == 'destination_identity':
        world.rows[0]['model'] = 'other_model'
    elif failure == 'target_identity':
        world.rows[1]['category'] = 'other_category'
    elif failure == 'bound_destination':
        world.config['bindings']['destination'] = 'other_desk'
    elif failure == 'label':
        world.rows.append(dict(world.rows[0], name='another_desk'))
    elif failure == 'size':
        world.rows[0]['aabb_m'][1][0] = .1
    else:
        world.rows.pop()
    for stage in ('initialization', 'evaluation'):
        result = destinations.validate_destination(world.session, world.config, stage=stage)
        assert not result['accepted'] and not result['qualification_complete']
        assert result['evaluation_validity'] == 'unavailable'


@pytest.mark.parametrize('field,value', [('episode_id', 'old_episode'), ('selection_seed', 8), ('destination_name', 'other_desk'),
    ('native_surface_fit_verified', False), ('public_visual_label_association_verified', False),
    ('public_control_approach_verified', False), ('public_control_release_verified', False)])
def test_episode_proofs_are_pending_at_start_and_mandatory_at_evaluation(world, field, value):
    full_receipt(world)
    world.session.destination_qualification['source'][field] = value
    assert destinations.validate_destination(world.session, world.config, stage='initialization')['accepted']
    result = destinations.validate_destination(world.session, world.config)
    assert not result['accepted'] and not result['qualification_complete']
    assert result['pending_qualification']


@pytest.mark.parametrize('valid_geometry', [True, False])
def test_reserved_blue_table_native_geometry_remains_a_structural_gate(world, valid_geometry):
    name = support_names(SLOT)[0]
    world.rows[0].update(name=name, category='table')
    world.contract.update(destination_binding={'name': name, 'category': 'table', 'model': 'desk_model'},
        fallback_table=True, label_relation={'kind': 'reserved_blue_delivery_table'})
    world.config['bindings']['destination'] = name
    world.session.delivery_support_initialization = {'accepted': valid_geometry,
        'source_identity': 'source', 'legs_reach_measured_floor': valid_geometry}
    full_receipt(world)
    for stage in ('initialization', 'evaluation'):
        result = destinations.validate_destination(world.session, world.config, stage=stage)
        assert result['checks']['blue_table_native_geometry_initialized'] is valid_geometry
        assert result['accepted'] is valid_geometry
    assert destinations.validate_destination(world.session, world.config, stage='initialization')['qualification_complete'] is False


def test_unknown_stage_raises_even_without_destination_contract(world):
    for config in (world.config, {'parameters': {}}):
        with pytest.raises(ValueError, match='stage'):
            destinations.validate_destination(world.session, config, stage='preflight')


def test_no_destination_contract_preserves_nonapplicable_behavior(world):
    for stage in ('initialization', 'evaluation'):
        assert destinations.validate_destination(world.session, {}, stage=stage) == {'applicable': False, 'accepted': True}
