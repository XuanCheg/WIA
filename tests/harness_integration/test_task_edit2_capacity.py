"""Check source inventory union, complete capacity scoring and calibration gates.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider \
tests/test_task_edit2_capacity.py tests/test_composition_manipulation.py \
tests/test_core_revision_manipulation.py
All calibration fixtures are synthetic; no physical capacity is measured here.
"""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wia.authoring.conversion import ConversionContext, SourceInstance
from wia.authoring.converters import composition_manipulation as composition
from wia.authoring.converters import liquid_capacity_inventory as inventory
from wia.authoring.converters import physical_structure_liquid_volume as liquid
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition

ROOT = __import__('wia.resources', fromlist=['ROOT']).ROOT


@pytest.fixture
def sources(tmp_path):
    context = ConversionContext(tmp_path)
    rows = []
    for run in (247, 89, 191, 275):
        identity = f'Physical Structure/Liquid Volume/unknown_scene/unknown_room/run_{run}.json'
        source = SourceInstance(ROOT / 'ESI/dataset/json_clean' / identity,
            identity, 'Physical Structure', 'Liquid Volume')
        _, _, entries, _ = liquid._source_data(source)
        evidence = liquid.capacity_evidence(source, entries, context)
        assert evidence['valid'], evidence
        rows.append((source, evidence))
    return rows


def test_union_retains_all_four_sources_and_asset_scales_without_cross_trial_count_ranking(sources):
    receipt = inventory.build_inventory(sources)
    assert len(receipt['inventory']) == 6
    assert len(receipt['source_mappings']) == 4
    assert len(receipt['comparisons']) == 4
    assert receipt['truth']['pair_denominator'] == 15
    assert len(receipt['truth']['missing_pairs']) == 12
    assert not receipt['truth']['accepted']
    assert not receipt['cross_trial_particle_counts_comparable']
    assert receipt['calibration_recipe']['diagnostic_only']
    mapping = {row['source_identity']: row['sides'] for row in receipt['source_mappings']}
    assert mapping[sources[2][0].relative_path] == mapping[sources[3][0].relative_path]
    assert receipt == inventory.build_inventory(list(reversed(sources)))


def test_provisional_task_and_scene_construct_the_entire_shared_inventory(sources, tmp_path):
    candidate = inventory.build_joint_candidate(sources, ConversionContext(tmp_path))
    task, scene = candidate['task_definition'], candidate['scene_definition']
    task_definition(task)
    SceneDefinition.model_validate(scene)
    assert task['scene'] == scene['scene']
    assert len(task['private']['evaluation']['parameters']['completion_items']) == 15
    patches = task['scene']['variants']['adapted']['patches']
    assert patches[0]['phase'] == 'before_construction'
    assert patches[0]['operation'] == 'category_initialization'
    assert patches[-1]['relocation_policy'] == 'shared_tabletop'
    assert patches[-1]['manipulation_bindings'] == []
    assert candidate['native_execution_verified'] is False
    anchor = candidate['inventory_receipt']['inventory'][:2]
    config = {'scene': {'scene_model': 'Rs_int', 'not_load_object_categories': ['breakfast_table']},
        'objects': [{'name': item['name'], 'category': item['category'], 'model': item['model']} for item in anchor]}
    evidence = liquid.configure_scene(config, patches[0]['parameters'])
    assert len(config['objects']) == 6
    assert config['scene']['not_load_object_categories'] == []
    assert config['scene']['load_room_instances'] == ['living_room_0']
    assert evidence['agent_operation_count'] == 0
    assert all(row['fixed_base'] and row['scale'] == [1., 1., 1.] for row in config['objects'])


@pytest.mark.parametrize('answer,correct,exact', [
    ('A < B < C', 3, True), ('C < B < A', 0, False),
    ('A = B < C', 2, False), ('A < B', 1, False),
    ('A < < B', 0, False), ('A < UNKNOWN < C', 0, False),
    ('A < A < B < C', 0, False),
    ({'comparisons': [['A', '<', 'B'], ['A', '<', 'C'], ['B', '<', 'C']]}, 3, True),
    ({'comparisons': [['A', '<', 'B'], ['A', '>', 'B'], ['A', '<', 'C']]}, 1, False),
    ({'comparisons': [['A', '<', 'B'], ['A', 'bad', 'C'], ['broken']]}, 1, False),
])
def test_pair_denominator_and_exact_ranking_are_separate(answer, correct, exact):
    truth = [['A', '<', 'B'], ['A', '<', 'C'], ['B', '<', 'C']]
    score = composition.capacity_ranking_score(['A', 'B', 'C'], truth, answer)
    assert score['evaluation_validity'] == 'complete'
    assert score['total_pairs'] == 3
    assert score['correct_pairwise_relations'] == correct
    assert score['comparison_accuracy_percent'] == pytest.approx(100. * correct / 3)
    assert score['exact_ranking_success'] is exact


def test_ties_have_a_consistent_complete_order():
    truth = [['A', '=', 'B'], ['A', '<', 'C'], ['B', '<', 'C']]
    score = composition.capacity_ranking_score(['A', 'B', 'C'], truth, [['B', 'A'], ['C']])
    assert score['exact_ranking_success']
    assert score['comparison_accuracy_percent'] == 100.
    assert not composition.capacity_order_graph(['A', 'B', 'C'],
        [['A', '=', 'B'], ['B', '=', 'C'], ['A', '<', 'C']])['accepted']


@pytest.mark.parametrize('truth', [[], [['A', '<', 'B']],
    [['A', '<', 'B'], ['B', '<', 'C'], ['C', '<', 'A']], [['bad']]])
def test_unknown_or_inconsistent_truth_is_unavailable_with_full_denominator(truth):
    score = composition.capacity_ranking_score(['A', 'B', 'C'], truth, 'A < B < C')
    assert score['evaluation_validity'] == 'unavailable'
    assert score['total_pairs'] == 3
    assert score['comparison_accuracy_percent'] is None
    assert not score['exact_ranking_success']


def test_source_count_or_scale_or_relation_mismatch_cannot_transfer(sources):
    receipt = inventory.build_inventory(sources)
    for mutate in ('count', 'scale', 'order'):
        changed = copy.deepcopy(receipt)
        first = changed['comparisons'][0]['first']
        if mutate == 'count':
            changed['comparisons'][0]['counts'][first] += 1
        elif mutate == 'scale':
            changed['inventory'][0]['scale'] = [.5, .5, .5]
        else:
            changed['comparisons'][0]['relation'] = '>'
        assert inventory.validate_truth(changed)['invalid_evidence']


def fake_calibration(receipt, *, varying_water=False, reset_ok=True):
    import torch
    # Synthetic counts define an arbitrary controlled fixture order. They are
    # deliberately unrelated to the historical source particle-count magnitudes.
    counts = {'A': 60, 'B': 50, 'C': 40, 'D': 30, 'E': 20, 'F': 10}
    seen, reset_calls = [], []
    @contextmanager
    def trial(first, second):
        seen.append((first['id'], second['id']))
        water = SimpleNamespace(name='water', particle_radius=.01 + (.001 * len(seen) if varying_water else 0),
            particle_contact_radius=.012, min_scale=np.ones(3), max_scale=np.ones(3), particle_density=1000.)
        objects = [SimpleNamespace(category=item['category'], model=item['model'],
            scale=np.asarray(item['scale']), id=item['id']) for item in (first, second)]
        yield {'objects': objects, 'water': water, 'floor_z': [.05, .05],
            'isolated_reset_verified': True, 'agent_episode_active': False}
    def fill(first, second, water, z1, z2, offsets):
        assert offsets.shape == (7, 3)
        return counts[first.id], counts[second.id]
    def particles(obj, water, offsets):
        return None, None, torch.ones(counts[obj.id], dtype=torch.bool)
    def reset():
        reset_calls.append(True)
        return {'accepted': reset_ok, 'all_inventory_restored': reset_ok, 'agent_history_empty': reset_ok}
    result = inventory.calibrate_native_pairs(receipt, trial, reset,
        fill_protocol={'fill_both': fill, 'get_particles_in_obj': particles})
    return result, seen, reset_calls


def test_common_native_protocol_completes_only_missing_pairs_and_resets_before_episode(sources):
    initial = inventory.build_inventory(sources)
    result, seen, reset_calls = fake_calibration(initial)
    assert len(seen) == 12
    assert reset_calls == [True]
    assert result['truth']['accepted'], result['truth']
    assert result['truth']['pair_denominator'] == 15
    assert result['calibration_recipe']['status'] == 'complete'
    assert initial['calibration_recipe']['status'] == 'pending'
    assert all(row['agent_operation_count'] == 0 for row in result['comparisons'] if row['origin'] == 'native_common_protocol')
    score = composition.capacity_ranking_score([item['id'] for item in result['inventory']],
        result['truth']['valid_comparisons'], 'F < E < D < C < B < A')
    assert score['exact_ranking_success']


@pytest.mark.parametrize('varying_water,reset_ok', [(True, True), (False, False)])
def test_calibration_water_or_reset_failure_never_publishes_available_truth(sources, varying_water, reset_ok):
    receipt = inventory.build_inventory(sources)
    result, _, reset_calls = fake_calibration(receipt, varying_water=varying_water, reset_ok=reset_ok)
    assert reset_calls == [True]
    assert not result['truth']['accepted']
    assert not inventory.validate_truth(result)['accepted']
    assert result['calibration_recipe']['status'] == 'pending'


def test_joint_evaluator_keeps_missing_truth_unavailable_and_known_pair_scores_explicit(sources, tmp_path, monkeypatch):
    initial = inventory.build_inventory(sources)
    receipt, _, _ = fake_calibration(initial)
    measured = {'objects': {item['id']: {'position': [float(index), 0., .6],
        'quaternion_xyzw': [0., 0., 0., 1.]} for index, item in enumerate(receipt['inventory'])},
        'checks': {'source_scales': True, 'one_table': True}}
    monkeypatch.setattr(liquid, 'qualify_joint_inventory', lambda *args: {
        'accepted': True, 'native_state': copy.deepcopy(measured)})
    monkeypatch.setattr(liquid, 'measure_joint_inventory', lambda *args: copy.deepcopy(measured))
    session = SimpleNamespace(env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda *args: None)))
    config = {'bindings': {}, 'parameters': {'inventory_receipt': receipt}}
    evaluator = liquid.JointCapacityRankingEvaluator(session, None, config, tmp_path)
    observation = SimpleNamespace(snapshot_id='s', action_step=0)
    assert not evaluator.begin_episode(observation)['task_success']
    evaluator.report_answer('F < E < D < C < B < A', observation)
    result = evaluator.record(observation)
    assert result['task_success'] and result['comparison_accuracy_percent'] == 100.
    assert result['total_goals'] == result['total_pairs'] == 15
    measured['objects']['A']['position'][0] += .1
    assert not evaluator.record(observation)['task_success']
    evaluator.receipt = initial
    result = evaluator.record(observation)
    assert result['evaluation_validity'] == 'unavailable'
    assert result['completion_score'] is None
    assert result['comparison_accuracy_percent'] is None
    assert result['total_pairs'] == 15
