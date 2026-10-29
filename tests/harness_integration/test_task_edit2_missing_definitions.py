"""CPU audit, deterministic labels and initialization-only delivery support checks.

Run: python -B -m pytest -q tests/test_task_edit2_missing_definitions.py
No GPU, renderer, motor action, source mutation, queue or Review update is used.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from wia.authoring.conversion import ROOT, SourceInstance, ConversionContext
from wia.authoring.converters import task_edit2_destinations as destinations
from wia.authoring.converters import enumerative_perception_spatial_segmentation as regions
from wia.tasks import delivery_support, interaction_supports, interaction_setup
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition
import wia.runtime.commands.build_esi_core_revision as core_builder
from wia.runtime.commands.build_task_edit2_missing_definitions import (
    audit_current_gaps, build_missing_definitions, anchor_source_states)


def source(identity):
    big, small = Path(identity).parts[:2]
    return SourceInstance(ROOT / 'ESI/dataset/json_clean' / identity, identity, big, small)


def test_current_membership_audit_uses_all_161_and_excludes_backups():
    audit = audit_current_gaps()
    assert audit['core_membership_count'] == 161
    assert audit['generation_record_count'] == 225
    assert audit['pending_count'] == 16
    assert audit['gap_counts'] == {'destination': 11, 'region_labels': 5}
    assert len({row['source'] for row in audit['pending']}) == 16


@pytest.fixture(scope='module')
def derivative_manifest(tmp_path_factory):
    return build_missing_definitions(tmp_path_factory.mktemp('missing_definitions'))


def test_all_actual_16_definitions_are_complete_valid_versioned_derivatives(derivative_manifest):
    assert derivative_manifest['counts'] == {'generated': 16}
    assert len(derivative_manifest['source_mapping']) == 16
    fallback = existing = region_count = 0
    for row in derivative_manifest['records']:
        task = yaml.safe_load(Path(row['task_file']).read_text())
        scene = yaml.safe_load(Path(row['scene_file']).read_text())
        task_definition(task)
        SceneDefinition.model_validate(scene)
        public = json.dumps(task['public'])
        assert not any(word in public for word in ('original task object', 'pending', 'unresolved', 'error:'))
        assert task['source']['json'] == row['source_identity']
        assert row['definition_revision'] == destinations.REVISION
        p = task['private']['evaluation']['parameters']
        if 'destination_contract' in p:
            contract = p['destination_contract']
            assert not contract['distinguishability_verified'] and not contract['fit_verified']
            assert contract['selection_seed'] == 0 and contract['target_bindings']
            if contract['fallback_table']:
                fallback += 1
                assert task['private']['bindings']['destination']['name'] == interaction_supports.support_names(99)[0]
                hooks = [patch for patch in task['scene']['variants']['adapted']['patches']
                         if patch.get('module') == destinations.MODULE]
                assert {hook['phase'] for hook in hooks} == {'before_construction', 'after_postprocessing'}
                assert 'blue delivery table' in public
            else:
                existing += 1
        else:
            region_count += 1
            assert p['region_label_contract']['axis_evidence']['ordering_uses_targets_or_answers'] is False
            assert 'preserve the source observer convention' not in public
            assert '+X pointing right and +Y pointing up' in public
    assert (existing, fallback, region_count) == (2, 9, 5)


def furniture(name, category):
    return {'name': name, 'category': category, 'model': 'model', 'in_rooms': ['room_0']}


def test_seeded_duplicate_furniture_landmarks_are_proposals_not_native_truth(monkeypatch):
    inventory = {row['name']: row for row in [furniture('table_a', 'countertop'),
        furniture('table_b', 'countertop'), furniture('fridge', 'fridge')]}
    monkeypatch.setattr(core_builder, 'scene_inventory', lambda scene: inventory)
    monkeypatch.setattr(core_builder, 'scene_object_positions', lambda scene: {
        'table_a': [1., 0., 0.], 'table_b': [4., 0., 0.], 'fridge': [0., 0., 0.]})
    stub = NS(raw={'scene': 'fixture', 'room': 'room_0'}, relative_path='fixture/source')
    first = core_builder.destination(stub, 17)
    assert first == core_builder.destination(stub, 17)
    assert first['public_label'] == 'the countertop nearest the fridge in this room'
    assert first['public_label_evidence']['native_relation_verified'] is False
    contract = {'destination_binding': first['binding'], 'room': 'room_0',
                'label_relation': first['public_label_evidence'], 'label_distance_gap_m': .10}
    native = [dict(row, room='room_0', centre_m=[x, 0., 0.]) for row, x in
              [(inventory['table_a'], 5.), (inventory['table_b'], 1.), (inventory['fridge'], 0.)]]
    assert not destinations.resolve_native_label(native, contract)['accepted']
    native[0]['centre_m'] = [.5, 0., 0.]
    assert destinations.resolve_native_label(native, contract)['accepted']
    native.append(dict(native[-1], name='second_fridge'))
    assert not destinations.resolve_native_label(native, contract)['accepted']


def test_readonly_destination_validator_requires_current_native_receipt_and_label(monkeypatch):
    rows = [{'name': 'desk', 'category': 'desk', 'model': 'desk_model', 'room': 'room_0',
             'centre_m': [.5, .5, .6], 'aabb_m': [[0., 0., .5], [1., 1., .7]]},
            {'name': 'target', 'category': 'cup', 'model': 'cup_model', 'room': 'room_0',
             'centre_m': [2., 2., .3], 'aabb_m': [[1.9, 1.9, .2], [2.1, 2.1, .4]]}]
    monkeypatch.setattr(destinations, '_native_rows', lambda session: rows)
    contract = {'scene': 'building', 'room': 'room_0', 'source_identity': 'source', 'selection_seed': 7,
        'destination_binding': {'name': 'desk', 'category': 'desk', 'model': 'desk_model'},
        'target_bindings': {'target': {'name': 'target', 'category': 'cup', 'model': 'cup_model'}},
        'label_relation': {'kind': 'unique_room_category'}, 'fallback_table': False, 'fit_margin_m': .04}
    config = {'bindings': {'destination': 'desk', 'target': 'target'},
              'parameters': {'destination_contract': contract}}
    session = NS(scene_name='building', episode_id='episode')
    before = deepcopy(rows)
    result = destinations.validate_destination(session, config)
    assert not result['accepted'] and result['checks']['native_aabb_footprint_fits_all_targets']
    session.destination_qualification = {'source': {'episode_id': 'episode', 'selection_seed': 7,
        'destination_name': 'desk', **{key: True for key in ('native_surface_fit_verified',
            'public_visual_label_association_verified', 'public_control_approach_verified',
            'public_control_release_verified')}}}
    assert destinations.validate_destination(session, config)['accepted']
    assert rows == before
    session.destination_qualification['source']['episode_id'] = 'old_reset'
    assert not destinations.validate_destination(session, config)['accepted']
    rows.append(dict(rows[0], name='another_desk'))
    assert not destinations.validate_destination(session, config)['checks']['public_label_uniquely_resolves']


def test_fallback_preloads_only_five_blue_fixed_primitives_and_preserves_defaults():
    old = {'objects': [{'name': 'source_actor', 'scale': [1., 2., 3.]}]}
    default = deepcopy(old)
    interaction_supports.preload_supports(default)
    assert default['objects'][1]['name'] == interaction_supports.support_names()[0]
    assert default['objects'][1]['rgba'] == [.46, .32, .20, 1.]
    config = deepcopy(old)
    result = destinations.configure_scene(config, {'destination_contract': {'fallback_table': True}})
    assert config['objects'][0] == old['objects'][0]
    assert len(config['objects']) == 6
    assert set(result['preloaded_names']) == set(interaction_supports.support_names(99))
    assert all(row['fixed_base'] and not row['visual_only'] and row['rgba'] == delivery_support.BLUE_RGBA
               for row in config['objects'][1:])
    with pytest.raises(ValueError, match='cannot replace'):
        destinations.configure_scene(config, {'destination_contract': {'fallback_table': True}})


def table_fixture(monkeypatch, *, footprint_clear=True, obstacle=False):
    objects = {'target': NS(name='target', category='ball', visual_only=False,
                           aabb=(np.array([9., 9., .2]), np.array([9.1, 9.1, .3])))}
    if obstacle:
        objects['cabinet'] = NS(name='cabinet', category='cabinet', visual_only=False,
                               aabb=(np.array([.8, .8, .5]), np.array([1.2, 1.2, 1.2])))
    for name in interaction_supports.support_names(99):
        objects[name] = NS(name=name, category='table' if name.endswith('top') else 'object',
                           visual_only=False, aabb=(np.array([2000., 2000., 5.]), np.array([2000.1, 2000.1, 5.1])))
    scene = NS(objects=objects, object_registry=lambda field, name: objects.get(name),
               trav_map=NS(floor_heights=[0.]))
    session = NS(env=NS(scene=scene), robot=NS(name='robot'), _category_initialization_active=True)
    contract = {'source_identity': 'source', 'selection_seed': 0, 'room': 'room_0',
                'target_bindings': {'target': {'name': 'target'}}, 'fit_margin_m': .04}
    monkeypatch.setattr(interaction_setup, '_navigation_cells',
        lambda *args: (np.array([[1., 1.]]), 0., {'native_floor_cells': True}))
    monkeypatch.setattr(interaction_supports, 'floor_support_centres',
        lambda *args: [np.array([1., 1.])])
    monkeypatch.setattr(interaction_supports, 'footprint_clear', lambda *args: footprint_clear)
    monkeypatch.setattr(delivery_support, 'native_floor_geometry',
        lambda *args: (0., {'floor_rays': [{'body': 'actual_floor', 'position_m': [1., 1., 0.]}]}))
    calls = []

    def activate(session, xy, size, surface_z, floor_z, *, slot):
        calls.append(slot)
        for index, name in enumerate(interaction_supports.support_names(slot)):
            low = np.r_[xy - size / 2, surface_z - .04 if index == 0 else floor_z]
            high = np.r_[xy + size / 2, surface_z if index == 0 else surface_z - .04]
            objects[name].aabb = (low, high)
        return objects[interaction_supports.support_names(slot)[0]]

    monkeypatch.setattr(interaction_supports, 'activate_support', activate)
    return session, contract, objects, calls


@pytest.mark.parametrize('footprint_clear,obstacle', [(False, False), (True, True)])
def test_fallback_collision_failure_does_not_activate_or_move_source(monkeypatch, footprint_clear, obstacle):
    session, contract, objects, calls = table_fixture(monkeypatch,
        footprint_clear=footprint_clear, obstacle=obstacle)
    before = deepcopy(objects['target'].aabb)
    with pytest.raises(ValueError, match='no_safe_loaded_room'):
        delivery_support.initialize_delivery_support(session, contract)
    assert not calls and not session.delivery_support_initialization['accepted']
    assert session.delivery_support_initialization['failed_stage'] == 'clear_delivery_footprint'
    assert np.array_equal(before, objects['target'].aabb)


def test_fallback_measures_floor_reaching_support_and_keeps_access_pending(monkeypatch):
    session, contract, objects, calls = table_fixture(monkeypatch)
    before = deepcopy(objects['target'].aabb)
    result = delivery_support.initialize_delivery_support(session, contract)
    assert calls == [99] and result['accepted']
    assert result['floor_geometry']['floor_rays']
    assert result['legs_reach_measured_floor'] and result['target_fit_geometry_verified']
    assert not result['public_control_access_verified'] and not result['source_actors_moved']
    assert np.array_equal(before, objects['target'].aabb)
    session._category_initialization_active = False
    with pytest.raises(RuntimeError, match='initialization-only'):
        delivery_support.initialize_delivery_support(session, contract)


def test_grouped_axis_labels_resolve_tied_chairs_without_using_targets_or_answer():
    pairs = [{'anchor': f'anchor_{i}', 'target': f'target_{i}'} for i in range(4)]
    centres = [(0., 0.), (.001, 1.), (1., 0.), (2., 0.)]
    layout = {'pairs': pairs, 'expected': {f'anchor_{i}': {'bbox':
        [[x - .2, y - .2, 0.], [x + .2, y + .2, .8]]} for i, (x, y) in enumerate(centres)}}
    labels, evidence = regions.global_region_labels(layout)
    assert 'bottom chair in the leftmost chair group' in labels['anchor_0']
    assert 'top chair in the leftmost chair group' in labels['anchor_1']
    assert evidence['between_band_separations_x_m'] and evidence['within_band_separations_y_m']['0'] == [1.]
    altered = deepcopy(layout)
    altered['answer'] = 'unrelated_wrong_answer'
    altered['pairs'] = [dict(pair, target='different_target') for pair in reversed(pairs)]
    assert regions.global_region_labels(altered) == (labels, evidence)
    tied = deepcopy(layout)
    tied['expected']['anchor_1'] = deepcopy(tied['expected']['anchor_0'])
    with pytest.raises(ValueError, match='unique Y separation'):
        regions.global_region_labels(tied)


def test_native_region_validator_rejects_changed_boundary_and_requires_visual_receipt():
    identity = 'Enumerative Perception/Spatial Segmentation/house_single_floor/dining_room_0/q_002.json'
    original = source(identity)
    result = regions.CategoryConverter().convert(original, ConversionContext(Path('/tmp'), seed=0,
        config={'task_edit2_global_region_labels': True,
                'region_anchor_source_states': anchor_source_states(original.raw['scene'])}))
    config = deepcopy(result['task_definition']['private']['evaluation'])
    bindings = result['task_definition']['private']['bindings']
    config['bindings'] = {key: value['name'] for key, value in bindings.items()}
    contract = config['parameters']['region_label_contract']
    objects = {}
    for key, expected in contract['source_expected'].items():
        name = bindings[key]['name']
        state = contract['source_anchor_states'].get(name)
        objects[name] = NS(name=name, category=bindings[key]['category'], aabb=np.asarray(expected['bbox']),
                           get_position_orientation=lambda state=state: (state['pos'], state['ori']))
    session = NS(scene_name=original.raw['scene'], episode_id='episode',
        env=NS(scene=NS(object_registry=lambda field, name: objects.get(name))))
    pending = regions.validate_region_labels(session, config)
    assert not pending['accepted'] and pending['checks']['current_native_label_mapping_matches']
    session.region_label_qualification = {identity: {'episode_id': 'episode', 'frame': 'initial_global_world_xy',
        'anchor_public_label_associations_verified': True}}
    assert regions.validate_region_labels(session, config)['accepted']
    objects[bindings['anchor_0']['name']].aabb += np.array([.15, 0., 0.])
    assert not regions.validate_region_labels(session, config)['accepted']
