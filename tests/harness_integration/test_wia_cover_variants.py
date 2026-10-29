"""Verify native-cover variant preservation without loading the simulator.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m
  pytest -q -p no:cacheprovider tests/test_wia_cover_variants.py
All generated files use temporary directories; the real three parents stay intact.
"""
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest
import yaml

from wia.tasks.loader import load_scene, load_task
import wia.authoring.commands.build_wia_cover_variants as builder

PARENTS = ('9fd3d70ee489c4228f5f44ec', 'fb64ed932434e097adc84053', 'e2b21727ae8a25c6b7b860d5')


@pytest.fixture
def design(tmp_path):
    entries = []
    for index, parent in enumerate(PARENTS):
        relative = Path('Physical Structure/Deformable/hall_arch_wood/redesign_20261020') / ('category_' + parent + '.yaml')
        entry = {'document_alias': 'fixture_' + str(index), 'variant_id': 'cover_fixture_' + str(index),
            'label': 'Explicit native cover fixture',
            'parent_task_file': str(builder.ROOT / 'explore_to_achieve/tasks' / relative),
            'parent_scene_file': str(builder.ROOT / 'explore_to_achieve/scenes' / relative),
            'cloth_asset': {'category': 'dish_towel', 'model': 'explicit_fixture_model'},
            'cloth_scale_multiplier': .85, 'cloth_mass_kg': .7, 'public_cover_label': 'dish towel'}
        if index == 1:
            entry.update(cloth_drop_clearance_m=.02, cloth_drop_xy_offset_m=[.01, -.02])
        entries.append(entry)
    value = {'schema': 'wia_cover_variant_design_v1', 'entries': entries}
    path = tmp_path / 'design.json'
    path.write_text(json.dumps(value) + '\n', encoding='utf-8')
    return path, tmp_path / 'output', value


def rewrite(design):
    design[0].write_text(json.dumps(design[2]) + '\n', encoding='utf-8')


def arguments(design):
    return ['--design-file', str(design[0]), '--output-dir', str(design[1])]


def replace_assets(value, cloth_name, asset):
    if isinstance(value, dict):
        if value.get('name') == cloth_name and {'category', 'model'} <= value.keys():
            value.update(asset)
        for key, child in value.items():
            if key != 'parent_source_cloth_asset':
                replace_assets(child, cloth_name, asset)
    elif isinstance(value, list):
        for child in value:
            replace_assets(child, cloth_name, asset)


def test_all_real_parents_preserve_complete_scientific_contract_except_explicit_cloth_fields(design):
    definitions, manifest, rendered = builder.build_package(*design[:2])
    assert len(definitions) == len(manifest['entries']) == 3
    assert len(rendered) == 7
    assert manifest['schema'] == 'wia_variant_manifest_v1'
    for definition, entry, record in zip(definitions, design[2]['entries'], manifest['entries']):
        parent_task = yaml.safe_load(Path(entry['parent_task_file']).read_text())
        parent_scene = yaml.safe_load(Path(entry['parent_scene_file']).read_text())
        task, scene = definition['task'], definition['scene']
        # Compare every field, including thresholds, distractors, target inventory,
        # poses, settle budgets, seeds, source identities, material and runtime.
        expected_task = deepcopy(parent_task)
        cloth_name = expected_task['private']['bindings']['cloth']['name']
        replace_assets(expected_task['private'], cloth_name, entry['cloth_asset'])
        replace_assets(expected_task['scene'], cloth_name, entry['cloth_asset'])
        expected_task['private']['evaluation']['parameters']['authored_cloth_scale_multiplier'] = entry['cloth_scale_multiplier']
        for patch in expected_task['scene']['variants']['adapted']['patches']:
            if patch.get('module') == builder.INITIALIZER:
                for field in ('cloth_scale_multiplier', 'cloth_mass_kg', 'cloth_drop_clearance_m', 'cloth_drop_xy_offset_m'):
                    if field in entry:
                        patch['parameters'][field] = deepcopy(entry[field])
        actual_science, expected_science = deepcopy(task), deepcopy(expected_task)
        for value in (actual_science, expected_science):
            for field in ('version', 'status', 'repair', 'public'):
                value.pop(field, None)
        assert actual_science == expected_science
        expected_scene = deepcopy(parent_scene)
        expected_scene.update(id=parent_scene['id'] + '__' + entry['variant_id'],
            version=parent_scene['version'] + 1, status='pending', scene=expected_task['scene'],
            bindings=expected_task['private']['bindings'])
        assert scene == expected_scene
        assert task['id'] == parent_task['id'] and task['version'] == parent_task['version'] + 1
        assert task['private']['evaluation']['parameters']['parent_source_cloth_asset'] == parent_task['private']['evaluation']['parameters']['parent_source_cloth_asset']
        public = json.dumps(task['public'])
        assert not re.search(r'\bcloth\b|\bcover\b', public, flags=re.I)
        for field in ('objective', 'success_criteria', 'environment'):
            before_numbers = re.findall(r'\d+(?:\.\d+)?', parent_task['public'][field])
            assert re.findall(r'\d+(?:\.\d+)?', task['public'][field]) == before_numbers
        assert 'never remove, teleport or resize the dish towel' in public
        assert record['variant'] == 'adapted' and record['seed'] == parent_task['defaults']['seed']
        assert record['task_source_identity'] == record['scene_source_identity'] == parent_task['source']['json']
        assert record['initialized_cloth_asset_override'] == task['private']['bindings']['cloth']
        for field in ('parent_task', 'parent_scene', 'source'):
            assert record[field + '_sha256'] == builder.sha256(Path(record[field + '_file']).read_bytes())
        assert record['task_sha256'] == builder.sha256(rendered[definition['task_file']])
        assert record['scene_sha256'] == builder.sha256(rendered[definition['scene_file']])
        assert record['native_validation_run'] is record['physical_success_verified'] is False
        assert task['repair']['native_validation_run'] is task['repair']['physical_success_verified'] is False


def test_cli_consistency_loaders_idempotency_and_no_differing_overwrites(design):
    inputs = [design[0], *(Path(entry[key]) for entry in design[2]['entries']
                          for key in ('parent_task_file', 'parent_scene_file'))]
    before = {path: path.read_bytes() for path in inputs}
    definitions, manifest, rendered = builder.build_package(*design[:2])
    assert builder.build_package(*design[:2])[2] == rendered
    assert builder.main(arguments(design) + ['--check']) == 1
    assert not design[1].exists()
    assert builder.main(arguments(design)) == 0
    mtimes = {path: path.stat().st_mtime_ns for path in rendered}
    assert builder.main(arguments(design) + ['--check']) == 0
    assert builder.main(arguments(design)) == 0
    assert {path: path.stat().st_mtime_ns for path in rendered} == mtimes
    assert json.loads((design[1] / 'manifest.json').read_bytes()) == manifest
    for definition in definitions:
        task = load_task(definition['task_file'], variant='adapted', run_mode='diagnostic')
        scene = load_scene(definition['scene_file'], variant='adapted', run_mode='diagnostic')
        assert task.definition.private.bindings == scene.definition.bindings
        assert task.source_path == scene.source_path
        assert task.definition.scene == scene.definition.scene
    changed = definitions[0]['task_file']
    changed.write_bytes(changed.read_bytes() + b'\n')
    altered = changed.read_bytes()
    assert builder.main(arguments(design) + ['--check']) == 1
    with pytest.raises(ValueError, match='Refusing to overwrite a differing generated file'):
        builder.main(arguments(design))
    assert changed.read_bytes() == altered
    assert {path: path.read_bytes() for path in inputs} == before


@pytest.mark.parametrize('field,value', [('cloth_asset', {'category': 'fixture', 'model': 'test', 'name': 'wrong'}),
    ('cloth_scale_multiplier', 0), ('cloth_mass_kg', True), ('cloth_drop_xy_offset_m', [0.]),
    ('cloth_drop_xy_offset_m', [float('nan'), 0.]), ('cloth_drop_clearance_m', -1),
    ('variant_id', '../escape'), ('coverage_threshold', .01)])
def test_only_explicit_finite_asset_initialization_fields_are_allowed(design, field, value):
    design[2]['entries'][0][field] = value
    rewrite(design)
    with pytest.raises(ValueError):
        builder.build_package(*design[:2])
    assert not design[1].exists()


def test_source_postprocess_conflict_and_mismatched_parent_bindings_are_rejected(design):
    entry = design[2]['entries'][0]
    task, scene = (yaml.safe_load(Path(entry['parent_' + kind + '_file']).read_text()) for kind in ('task', 'scene'))
    changed_task, changed_scene = deepcopy(task), deepcopy(scene)
    for value in (changed_task, changed_scene):
        value['scene']['variants']['adapted']['patches'][0]['module'] = 'wia.authoring.converters.physical_structure_deformable'
    with pytest.raises(ValueError, match='source bootstrap conflict'):
        builder.translate_definitions(changed_task, changed_scene, entry, {})
    changed_scene = deepcopy(scene)
    changed_scene['bindings']['cloth']['model'] = 'wrong'
    with pytest.raises(ValueError, match='identical patches, bindings, source and seed'):
        builder.translate_definitions(task, changed_scene, entry, {})
