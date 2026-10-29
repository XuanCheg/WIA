"""Verify the single-factor ramp configuration without claiming physical results.

Run: PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q \
  -p no:cacheprovider tests/test_wia_ramp_calibration.py
"""
from copy import deepcopy
import json

import pytest

from wia.tasks.loader import load_scene, load_task
from wia.authoring.commands.build_wia_ramp_calibration import (
    BASELINE_SCENE, BASELINE_TASK, DEFAULT_OUTPUT, MANIFEST_NAME, MULTIPLIERS,
    ORIGINAL_FRICTION, PUBLIC_TASK_ID, baseline_definitions, build_candidates,
    main, read_definition, render_artifacts,
)


def normalized_definition(candidate, original, *, task):
    """Remove only IDs, pending bookkeeping, and the declared ramp material factor."""
    candidate = deepcopy(candidate)
    for field in ('id', 'version', 'status'):
        candidate[field] = original[field]
    for variant, value in candidate['scene']['variants'].items():
        for index, patch in enumerate(value['patches']):
            if patch.get('module') == 'wia.authoring.converters.active_layout':
                patch['parameters']['layout']['friction']['ramp'] = deepcopy(
                    original['scene']['variants'][variant]['patches'][index]['parameters']['layout']['friction']['ramp'])
    if task:
        parameters = candidate['private']['evaluation']['parameters']
        source = original['private']['evaluation']['parameters']
        for field in ('static_friction', 'dynamic_friction', 'source_ramp_geometry_and_friction_preserved'):
            parameters[field] = source[field]
    return candidate


def test_only_the_declared_material_factor_and_private_bookkeeping_change():
    task_before, scene_before = BASELINE_TASK.read_bytes(), BASELINE_SCENE.read_bytes()
    original_task, original_scene = baseline_definitions()
    manifest, pairs = build_candidates()
    assert len(pairs) == len(manifest['entries']) == 4
    assert [row['ramp_material_multiplier'] for row in manifest['entries']] == list(MULTIPLIERS)
    assert len({pair['scene']['id'] for pair in pairs}) == 4
    for pair, entry in zip(pairs, manifest['entries']):
        assert normalized_definition(pair['task'], original_task, task=True) == original_task
        assert normalized_definition(pair['scene'], original_scene, task=False) == original_scene
        assert pair['task']['scene'] == pair['scene']['scene']
        assert pair['task']['source'] == pair['scene']['source'] == original_task['source']
        assert pair['task']['status'] == pair['scene']['status'] == 'pending'
        assert entry['private_provenance']['source_ramp_geometry_preserved'] is True
        assert all(value in ('pending', 'not_established') for value in entry['evidence'].values())
    assert BASELINE_TASK.read_bytes() == task_before
    assert BASELINE_SCENE.read_bytes() == scene_before
    assert manifest['native_experiments_run_by_generator'] is False
    assert manifest['publication_ready'] is False


def test_both_material_hooks_and_private_scorer_use_the_same_scaled_coefficients():
    manifest, pairs = build_candidates()
    for pair, entry in zip(pairs, manifest['entries']):
        multiplier = entry['ramp_material_multiplier']
        expected = {key: value * multiplier for key, value in ORIGINAL_FRICTION.items()}
        assert entry['ramp_material_coefficients'] == expected
        parameters = pair['task']['private']['evaluation']['parameters']
        assert {key: parameters[key] for key in ORIGINAL_FRICTION} == expected
        assert parameters['source_ramp_geometry_and_friction_preserved'] is (multiplier == 1.)
        assert parameters['trial_duration_s'] == 3.
        assert parameters['arrival_tie_tolerance_s'] == .15
        for kind in ('task', 'scene'):
            hooks = pair[kind]['scene']['variants']['adapted']['patches']
            assert {hook['phase'] for hook in hooks} == {'before_construction', 'after_postprocessing'}
            assert len(hooks) == 2
            for hook in hooks:
                assert hook['parameters']['layout']['friction']['ramp'] == expected
        if multiplier:
            assert expected['static_friction'] / expected['dynamic_friction'] == pytest.approx(
                ORIGINAL_FRICTION['static_friction'] / ORIGINAL_FRICTION['dynamic_friction'])
        else:
            assert expected == {'static_friction': 0., 'dynamic_friction': 0.}
    assert manifest['entries'][-1]['ramp_material_coefficients'] == ORIGINAL_FRICTION


def test_full_loader_public_payload_is_identical_and_neutral_across_exact_paths(tmp_path):
    assert main(['--output-dir', str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    payloads = []
    sources = []
    for entry in manifest['entries']:
        task = load_task(tmp_path / entry['task_file'], variant='adapted', run_mode='diagnostic')
        scene = load_scene(tmp_path / entry['scene_file'], variant='adapted', run_mode='diagnostic')
        assert task.definition.id == PUBLIC_TASK_ID
        assert task.definition.version == 1
        assert task.seed == scene.seed == 0
        assert task.source_path == scene.source_path
        assert task.definition.public.model_dump(mode='json') == load_task(
            BASELINE_TASK, variant='adapted', run_mode='diagnostic').definition.public.model_dump(mode='json')
        payloads.append(task.public_payload)
        sources.append(task.source_path)
    assert all(payload == payloads[0] for payload in payloads)
    assert len(set(sources)) == 1
    assert payloads[0]['task_id'] == PUBLIC_TASK_ID
    assert all('ramp_material_multiplier' not in payload for payload in payloads)


def test_unit_multiplier_matches_original_scientific_values_exactly():
    _, pairs = build_candidates()
    original_task, original_scene = baseline_definitions()
    unit = pairs[-1]
    assert unit['task']['private'] == original_task['private']
    assert unit['task']['scene'] == original_task['scene']
    assert unit['scene']['scene'] == original_scene['scene']
    assert unit['task']['defaults'] == original_task['defaults']
    assert unit['task']['requirements'] == original_task['requirements']


def test_generation_is_deterministic_and_check_is_no_write(tmp_path):
    expected = render_artifacts()
    assert render_artifacts() == expected
    assert main(['--output-dir', str(tmp_path)]) == 0
    before = {name: (tmp_path / name).stat().st_mtime_ns for name in expected}
    assert main(['--output-dir', str(tmp_path), '--check']) == 0
    assert before == {name: (tmp_path / name).stat().st_mtime_ns for name in expected}
    assert all((DEFAULT_OUTPUT / name).read_text() == text for name, text in expected.items())
    first = tmp_path / 'material_000/task.yaml'
    first.write_text(first.read_text() + '\n')
    changed = first.read_bytes()
    assert main(['--output-dir', str(tmp_path), '--check']) == 1
    assert first.read_bytes() == changed


def test_existing_unrelated_output_is_preserved(tmp_path):
    (tmp_path / 'README.md').write_text('Existing unrelated content.\n')
    with pytest.raises(ValueError, match='without an owned'):
        main(['--output-dir', str(tmp_path)])
    assert (tmp_path / 'README.md').read_text() == 'Existing unrelated content.\n'
