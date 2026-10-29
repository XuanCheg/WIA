"""CPU behavior checks for the isolated native-capacity driver contract.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider
 tests/test_task_edit2_capacity_driver.py. Protocol stubs never certify native physics.
"""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wia.runtime.sim.esi_capacity_calibration import (NativeCapacityCalibration, ProgressJournal,
    select_pairs, source_pool_geometry, validate_reset_proof)
from wia.runtime.commands.calibrate_task_edit2_capacity import main, validate_inputs

ROOT = __import__('wia.resources', fromlist=['ROOT']).ROOT
CANDIDATE = ROOT / 'outputs/task_edit2_20261018/capacity_inventory/candidates/capacity_union_9a718a61c6104607b62b662c'


@pytest.fixture
def receipt():
    return json.loads((CANDIDATE / 'inventory_receipt.json').read_text())


def baseline():
    return {'objects': {'jug': {'scale': [1., 1., 1.], 'pose': [[0., 0., 1.], [0., 0., 0., 1.]],
        'joints': {'lid': {'position': [0.], 'lower_limit': 0., 'upper_limit': 1.}}},
        'robot': {'pose': [[1., 1., 0.], [0., 0., 0., 1.]]}},
        'stage': {'/World/CapacityLabels:visibility': {'value': 'inherited', 'authored': False},
            'camera:focalLength': {'value': 20., 'authored': True}}}


def reset_proof(before, after, **changes):
    args = {'empty_water': True, 'pool_absent': True,
        'history': {'episode_absent': True, 'agent_components_absent': True, 'entries': 0}}
    args.update(changes)
    return validate_reset_proof(before, after, **args)


def test_default_and_ordered_control_pairs(receipt):
    pairs = select_pairs(receipt)
    assert len(pairs) == 12
    assert ('A', 'B') not in pairs and ('A', 'C') in pairs
    assert select_pairs(receipt, 'A:B,C:A') == [('A', 'B'), ('C', 'A')]


@pytest.mark.parametrize('text', ['A:A', 'A:G', 'A:B,B:A', 'A', '', 'A:B:C'])
def test_bad_pairs_rejected(receipt, text):
    with pytest.raises(ValueError):
        select_pairs(receipt, text)


def test_exact_artifact_inputs(receipt):
    _, _, _, loaded, source = validate_inputs(CANDIDATE / 'task.yaml', CANDIDATE / 'scene.yaml', CANDIDATE / 'inventory_receipt.json')
    assert loaded['inventory'] == receipt['inventory'] and source.name == 'run_191.json'


def test_mismatched_task_inventory_rejected(tmp_path):
    import yaml
    scene = yaml.safe_load((CANDIDATE / 'scene.yaml').read_text())
    scene['scene']['variants']['adapted']['patches'][0]['parameters']['inventory_receipt']['inventory'][0]['scale'][0] = .5
    path = tmp_path / 'scene.yaml'
    path.write_text(yaml.safe_dump(scene))
    with pytest.raises(ValueError, match='match exactly'):
        validate_inputs(CANDIDATE / 'task.yaml', path, CANDIDATE / 'inventory_receipt.json')


def test_reset_needs_full_independent_measurement():
    before = baseline()
    assert reset_proof(before, copy.deepcopy(before))['accepted']
    assert not reset_proof({}, {})['accepted']
    after = copy.deepcopy(before)
    after['objects'].pop('robot')
    assert not reset_proof(before, after)['all_inventory_restored']


@pytest.mark.parametrize('change', ['scale', 'joint', 'pose', 'camera', 'label'])
def test_partial_reset_not_accepted(change):
    before, after = baseline(), baseline()
    if change == 'scale':
        after['objects']['jug']['scale'][0] = .5
    elif change == 'joint':
        after['objects']['jug']['joints']['lid']['lower_limit'] = .999
    elif change == 'pose':
        after['objects']['robot']['pose'][0][0] += .01
    elif change == 'camera':
        after['stage']['camera:focalLength']['value'] = 19.
    else:
        after['stage']['/World/CapacityLabels:visibility']['value'] = 'invisible'
    assert not reset_proof(before, after)['accepted']


@pytest.mark.parametrize('kwargs', [{'empty_water': False}, {'pool_absent': False},
    {'history': {'episode_absent': False, 'agent_components_absent': True, 'entries': 0}},
    {'history': {'episode_absent': True, 'agent_components_absent': True, 'entries': 1}},
    {'history': {'episode_absent': True, 'agent_components_absent': False, 'entries': 0}}])
def test_empty_pool_and_no_agent_episode_are_required(kwargs):
    assert not reset_proof(baseline(), baseline(), **kwargs)['accepted']


def test_history_probe_does_not_instantiate_observation_source():
    class Session:
        @property
        def observation_source(self):
            raise AssertionError('This would create history; must not be accessed.')
    owner = NativeCapacityCalibration.__new__(NativeCapacityCalibration)
    owner.session = Session()
    assert owner._history() == {'episode_absent': True, 'entries': 0, 'agent_components_absent': True}
    owner.session._observation_source = object()
    assert not owner._history()['agent_components_absent']


def test_pool_geometry_matches_source_formula():
    result = source_pool_geometry(([0, 0, .04], [.2, .1, .34]), ([.5, -.05, .04], [.7, .15, .24]), .01)
    assert np.allclose(result['low_xy'], [-.05, -.1])
    assert np.allclose(result['high_xy'], [.75, .2])
    assert result['half_z'] == pytest.approx(.171)


def stub_driver(receipt, output):
    """Only orchestration/proof test; no native driver initialization is bypassed into truth."""
    owner = NativeCapacityCalibration.__new__(NativeCapacityCalibration)
    owner.receipt = copy.deepcopy(receipt)
    owner.owner_thread = __import__('threading').get_ident()
    owner.journal = ProgressJournal(output)
    owner.render_policy = {'mode': 'original_source_rendering'}
    owner.sim = SimpleNamespace(current_time=0.)
    owner.session = SimpleNamespace(loaded_task=SimpleNamespace(file_sha256='task', source_sha256='source'),
        scene_adapter=SimpleNamespace(task=SimpleNamespace(file_sha256='scene')), native_reset_id='test_reset')
    state = baseline()
    saved = copy.deepcopy(state)
    reset_calls = []
    def reset():
        state.clear()
        state.update(copy.deepcopy(saved))
        reset_calls.append(1)
        return reset_proof(saved, state)
    @contextmanager
    def trial(first, second):
        proof = reset()
        assert proof['accepted']
        objs = [SimpleNamespace(category=item['category'], model=item['model'], scale=np.array(item['scale']),
            label=item['id']) for item in (first, second)]
        water = SimpleNamespace(name='water', particle_radius=.01, particle_contact_radius=.009,
            min_scale=np.ones(3), max_scale=np.ones(3), particle_density=1000.)
        yield {'isolated_reset_verified': proof['accepted'], 'agent_episode_active': False,
            'objects': objs, 'water': water, 'floor_z': [.05, .05]}
    owner.reset_actual_inventory, owner.diagnostic_trial = reset, trial
    return owner, state, reset_calls


def test_original_helper_called_and_partial_receipt_survives_later_failure(receipt, tmp_path):
    import torch
    driver, state, resets = stub_driver(receipt, tmp_path)
    calls = []
    counts = {'A': 12, 'B': 8, 'C': 7}
    def fill(a, b, water, za, zb, offsets):
        calls.append((a.label, b.label, za, zb, offsets.clone()))
        state['objects']['jug']['joints']['lid']['lower_limit'] = .999
        driver.sim.current_time += 64.
        if b.label == 'C':
            raise RuntimeError('interrupted second source trial')
        return counts[a.label], counts[b.label]
    def measure(obj, water, offsets):
        return None, None, torch.ones(counts[obj.label], dtype=torch.bool)
    result = driver.run(pairs=[('A', 'B'), ('A', 'C')], protocol={'fill_both': fill, 'get_particles_in_obj': measure})
    assert len(calls) == 2 and calls[0][:4] == ('A', 'B', .05, .05)
    assert calls[0][4].shape == (7, 3)
    assert len(resets) >= 5
    first = json.loads((tmp_path / 'pair_AB_receipt.json').read_text())
    assert first['comparisons'][-1]['origin'] == 'native_common_protocol'
    assert first['comparisons'][-1]['counts'] == {'A': 12, 'B': 8}
    assert 'interrupted second' in result['calibration_recipe']['native_error']
    assert not result['truth']['accepted']
    assert result['post_calibration_inventory_reset']['accepted']
    assert state == baseline()
    events = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert any(row['stage'] == 'source_fill_measured' and row['retained_particle_counts'] == [12, 8] for row in events)
    assert (tmp_path / 'pair_AC_receipt.json').is_file()


def test_measurement_recheck_disagreement_never_adds_edge(receipt, tmp_path):
    import torch
    driver, _, _ = stub_driver(receipt, tmp_path)
    result = driver.run(pairs=[('A', 'C')], protocol={'fill_both': lambda *args: (12, 7),
        'get_particles_in_obj': lambda *args: (None, None, torch.ones(3, dtype=torch.bool))})
    assert len(result['comparisons']) == len(receipt['comparisons'])
    assert 'disagree' in result['calibration_recipe']['native_error']
    assert not result['truth']['accepted']


def test_explicit_gpu_and_private_appdata_required_before_native_import(monkeypatch, tmp_path):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    args = ['--task-file', 'missing', '--scene-file', 'missing', '--inventory-receipt', 'missing', '--output-dir', str(tmp_path)]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.delenv('OMNIGIBSON_APPDATA_PATH', raising=False)
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_native_control_contradiction_refused_immediately(receipt, tmp_path):
    import torch
    driver, _, _ = stub_driver(receipt, tmp_path)
    calls = []
    def fill(*args):
        calls.append(1)
        return 3, 7  # Contradicts original source B < A.
    def measured(obj, *args):
        return None, None, torch.ones(3 if obj.label == 'A' else 7, dtype=torch.bool)
    result = driver.run(pairs=[('A', 'B'), ('A', 'C')], protocol={'fill_both': fill, 'get_particles_in_obj': measured})
    assert len(calls) == 1
    assert result['truth']['reason'] == 'inconsistent_capacity_comparison'
    assert result['calibration_recipe']['status'] == 'pending'
    assert 'capacity_evidence_graph_conflict' in result['calibration_recipe']['native_error']


def test_final_reset_exception_invalidates_and_preserves_partial_result(receipt, tmp_path):
    import torch
    driver, _, _ = stub_driver(receipt, tmp_path)
    restore = driver.reset_actual_inventory
    calls = []
    def failing_final_restore():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('final native restore failed')
        return restore()
    driver.reset_actual_inventory = failing_final_restore
    result = driver.run(pairs=[('A', 'C')], protocol={'fill_both': lambda *args: (12, 7),
        'get_particles_in_obj': lambda obj, *args: (None, None, torch.ones(12 if obj.label == 'A' else 7, dtype=torch.bool))})
    assert not result['post_calibration_inventory_reset']['accepted']
    assert not result['truth']['accepted']
    assert result['calibration_recipe']['status'] == 'pending'
    assert (tmp_path / 'pair_AC_receipt.json').is_file()
    final = json.loads((tmp_path / 'calibration_receipt.json').read_text())
    assert not final['post_calibration_inventory_reset']['all_inventory_restored']
