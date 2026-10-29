"""Check deterministic workstation translations using existing parent definitions.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_workstation_variants.py
All generated definitions live in temporary directories; no native simulation,
source generation, capture queue, GPU or reference execution is used.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from wia.tasks.loader import load_scene, load_task
import wia.authoring.commands.build_wia_workstation_variants as builder


PARENTS = (
    ('D2-1', 'dimensional_v2', 'd2_1_seed101_v2'),
    ('R-C1', 'mirror_v1', 'r_c1_seed101_v1'),
)
HALL_SOURCE = 'Specular Reflection/Correspondence/hall_glass_ceiling/empty_room_0/q_007.json'
HOUSE_SOURCE = 'Temporal Understanding/Unobserved Change/house_single_floor/dining_room_0/wia_history_a_v1.json'


def layout(definition, index=0):
    return definition['scene']['variants']['adapted']['patches'][index]['parameters']['layout']


@pytest.fixture
def design(tmp_path):
    entries = []
    for alias, family, stem in PARENTS:
        parent = builder.ROOT / 'explore_to_achieve/wia_v1' / family
        entries.append({'document_alias': alias, 'variant_id': stem + '_house', 'label': 'Explicit CPU transfer fixture',
            'parent_task_file': str(parent / 'tasks' / (stem + '.yaml')),
            'parent_scene_file': str(parent / 'scenes' / (stem + '.yaml')),
            'output_task_file': str(tmp_path / 'tasks' / (stem + '_house.yaml')),
            'output_scene_file': str(tmp_path / 'scenes' / (stem + '_house.yaml')),
            'source_identity': HOUSE_SOURCE, 'scene_model': 'house_single_floor', 'room': 'dining_room_0',
            'floor_name': 'explicit_fixture_floor', 'translation_m': [-2., 3., .2]})
    value = {'schema': 'wia_workstation_transfer_design_v1', 'entries': entries}
    design_file = tmp_path / 'design.json'
    design_file.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    return design_file, tmp_path / 'manifest.json', value


def arguments(design):
    return ['--design-file', str(design[0]), '--manifest-output', str(design[1])]


def rewrite(design):
    design[0].write_text(json.dumps(design[2], indent=2) + '\n', encoding='utf-8')


def test_actual_parents_translate_all_poses_preserving_geometry_public_seed_scoring_and_material(design):
    definitions, manifest, rendered = builder.build_package(*design[:2])
    assert manifest['schema'] == 'wia_variant_manifest_v1' and manifest['artifact_path_base'] == 'absolute'
    assert len(definitions) == len(manifest['entries']) == 2 and len(rendered) == 5
    for definition, entry, record in zip(definitions, design[2]['entries'], manifest['entries']):
        parent_task = yaml.safe_load(Path(entry['parent_task_file']).read_text())
        parent_scene = yaml.safe_load(Path(entry['parent_scene_file']).read_text())
        task, scene = definition['task'], definition['scene']
        assert task['id'] == parent_task['id'] and task['version'] == parent_task['version'] + 1
        assert scene['id'] != parent_scene['id'] and entry['variant_id'] in scene['id']
        assert task['defaults'] == parent_task['defaults']
        assert task['public'] == parent_task['public']
        assert task['private']['bindings'] == parent_task['private']['bindings']
        assert task['private']['evaluation']['parameters']['scoring'] == parent_task['private']['evaluation']['parameters']['scoring']
        assert task['source']['json'] == scene['source']['json'] == HOUSE_SOURCE
        assert task['scene'] == scene['scene']
        identity = {'scene_model': entry['scene_model'], 'room': entry['room']}
        assert task['private']['evaluation']['parameters']['scene_contract'] == identity
        for index in (0, 1):
            original, translated = layout(parent_task, index), layout(task, index)
            assert translated['rooms'] == [entry['room']]
            assert translated['scene_model'] == entry['scene_model']
            assert translated['floor_name'] == entry['floor_name']
            assert translated['shared_layout_id'].endswith('__' + entry['variant_id'])
            for before, after in zip(original['objects'], translated['objects']):
                expected = deepcopy(before)
                expected['position'] = [value + shift for value, shift in zip(before['position'], entry['translation_m'])]
                assert after == expected
            expected_robot = deepcopy(original['robot_start'])
            expected_robot['position'] = [value + shift for value, shift in
                                          zip(original['robot_start']['position'], entry['translation_m'])]
            assert translated['robot_start'] == expected_robot
            original_parameters = deepcopy(parent_task['scene']['variants']['adapted']['patches'][index]['parameters'])
            translated_parameters = deepcopy(task['scene']['variants']['adapted']['patches'][index]['parameters'])
            for key in ('revision', 'scene_contract', 'layout'):
                original_parameters.pop(key, None)
                translated_parameters.pop(key, None)
            assert original_parameters == translated_parameters
        assert record['revision'] == task['private']['evaluation']['parameters']['revision']
        assert record['revision'] in ('wia_dimensional_v3', 'wia_mirror_v2')
        assert record['variant'] == 'adapted' and record['seed'] == parent_task['defaults']['seed']
        for key in ('task_file', 'scene_file', 'parent_task_file', 'parent_scene_file', 'source_file'):
            assert Path(record[key]).is_absolute()
        assert task['repair']['full_reference_execution'] == record['full_reference_execution'] == 'pending'
        assert all(task['repair'][key] is record[key] is False for key in builder.NATIVE_FLAGS)
        assert record['parent_task_sha256'] == builder.sha256(Path(entry['parent_task_file']).read_bytes())
        assert record['task_sha256'] == builder.sha256(rendered[definition['task_file']])


def test_cli_loads_generated_definitions_and_is_deterministic_without_overwrite(design):
    inputs = [design[0], *(Path(entry[key]) for entry in design[2]['entries']
                          for key in ('parent_task_file', 'parent_scene_file'))]
    before = {path: path.read_bytes() for path in inputs}
    first = builder.build_package(*design[:2])
    assert builder.build_package(*design[:2])[2] == first[2]
    assert builder.main(arguments(design) + ['--check']) == 1
    assert builder.main(arguments(design)) == 0
    mtimes = {path: path.stat().st_mtime_ns for path in first[2]}
    assert builder.main(arguments(design) + ['--check']) == 0
    assert builder.main(arguments(design)) == 0
    assert {path: path.stat().st_mtime_ns for path in first[2]} == mtimes
    for definition in first[0]:
        task = load_task(definition['task_file'], variant='adapted', run_mode='diagnostic')
        scene = load_scene(definition['scene_file'], variant='adapted', run_mode='diagnostic')
        assert (task.scene_name, task.room_name) == ('house_single_floor', 'dining_room_0')
        assert task.scene_name == scene.scene_name and task.room_name == scene.room_name
    changed = first[0][0]['task_file']
    changed.write_bytes(changed.read_bytes() + b'\n')
    changed_bytes = changed.read_bytes()
    assert builder.main(arguments(design) + ['--check']) == 1
    with pytest.raises(ValueError, match='Refusing to overwrite a differing generated file'):
        builder.main(arguments(design))
    assert changed.read_bytes() == changed_bytes
    assert {path: path.read_bytes() for path in inputs} == before


def test_named_overrides_apply_after_translation_to_only_the_explicit_objects(design):
    entry = design[2]['entries'][0]
    entry['object_overrides'] = {'wia_dim_table': {'rgba': [.1, .2, .3, 1.]},
                                 'wia_dim_candidate_0': {'position': [10., 11., 12.]}}
    rewrite(design)
    task = builder.build_package(*design[:2])[0][0]['task']
    parent = yaml.safe_load(Path(entry['parent_task_file']).read_text())
    for index in (0, 1):
        objects = {obj['name']: obj for obj in layout(task, index)['objects']}
        originals = {obj['name']: obj for obj in layout(parent, index)['objects']}
        for name, obj in objects.items():
            expected = deepcopy(originals[name])
            expected['position'] = [value + shift for value, shift in zip(expected['position'], entry['translation_m'])]
            expected.update(entry['object_overrides'].get(name, {}))
            assert obj == expected
    assert task['repair']['object_overrides'] == entry['object_overrides']


@pytest.mark.parametrize('overrides', [{'missing_native_object': {'position': [0., 0., 0.]}},
    {'wia_dim_table': {'scale': [1., 1., 1.]}}, {'wia_dim_table': {'orientation': [0., 0., 0., 1.]}},
    {'wia_dim_table': {}}, {'wia_dim_table': {'rgba': [1., 1., 1., 2.]}},
    {'wia_dim_table': {'position': [True, 0., 0.]}}])
def test_override_identity_and_field_scope_are_explicit(design, overrides):
    design[2]['entries'][0]['object_overrides'] = overrides
    rewrite(design)
    with pytest.raises(ValueError):
        builder.build_package(*design[:2])
    assert not design[1].exists()


@pytest.mark.parametrize('field,value', [('scene_model', 'hall_glass_ceiling'), ('room', 'different_room'),
    ('source_identity', 'Specular Reflection/Correspondence/missing/empty_room_0/missing.json')])
def test_source_identity_must_exist_and_match_top_level_target_scene_room(design, field, value):
    design[2]['entries'][0][field] = value
    rewrite(design)
    with pytest.raises(ValueError):
        builder.build_package(*design[:2])
    assert not design[1].exists()


def test_same_parent_across_backgrounds_keeps_matched_task_identity(design):
    entry = deepcopy(design[2]['entries'][0])
    entry.update(variant_id='explicit_hall_control', label='Explicit hall control', source_identity=HALL_SOURCE,
                 scene_model='hall_glass_ceiling', room='empty_room_0', floor_name='floors_nbzwbf_0',
                 translation_m=[0., 0., 0.], output_task_file=str(design[0].parent / 'hall_task.yaml'),
                 output_scene_file=str(design[0].parent / 'hall_scene.yaml'))
    design[2]['entries'].append(entry)
    rewrite(design)
    definitions = builder.build_package(*design[:2])[0]
    house, hall = definitions[0], definitions[-1]
    assert house['task']['id'] == hall['task']['id'] and house['task']['version'] == hall['task']['version']
    assert house['task']['public'] == hall['task']['public']
    assert house['scene']['id'] != hall['scene']['id']


def test_pure_translator_rejects_parent_hook_drift_and_preserves_input(design):
    entry = design[2]['entries'][0]
    task = yaml.safe_load(Path(entry['parent_task_file']).read_text())
    scene = yaml.safe_load(Path(entry['parent_scene_file']).read_text())
    original = deepcopy((task, scene, entry))
    translated, _ = builder.translate_definitions(task, scene, entry, {})
    assert (task, scene, entry) == original and translated != task
    task['scene']['variants']['adapted']['patches'][1]['parameters']['layout']['objects'][0]['position'][0] += 1
    scene['scene'] = deepcopy(task['scene'])
    with pytest.raises(ValueError, match='identical parameters'):
        builder.translate_definitions(task, scene, entry, {})


@pytest.fixture
def ramp_design(design):
    relative = Path('Physical Dynamics/Inclined Plane/house_single_floor/redesign_20261020/category_88ce9d62d891ae0ba628a6e9.yaml')
    entry = design[2]['entries'][0]
    entry.update(document_alias='T42', variant_id='t42_explicit_ramp_translation',
                 parent_task_file=str(builder.ROOT / 'explore_to_achieve/tasks' / relative),
                 parent_scene_file=str(builder.ROOT / 'explore_to_achieve/scenes' / relative),
                 translation_m=[2., -1., .5])
    design[2]['entries'] = [entry]
    rewrite(design)
    return design


def test_ramp_rigid_translation_preserves_release_geometry_material_and_effective_runtime(ramp_design):
    entry = ramp_design[2]['entries'][0]
    parent = yaml.safe_load(Path(entry['parent_task_file']).read_text())
    assert 'repair' not in parent and 'shared_layout_id' not in layout(parent)
    definitions, manifest, _ = builder.build_package(*ramp_design[:2])
    task, scene = definitions[0]['task'], definitions[0]['scene']
    original_parameters = parent['private']['evaluation']['parameters']
    expected_parameters = deepcopy(original_parameters)
    expected_parameters['safe_bounds_m'] = [
        [value + shift for value, shift in zip(corner, entry['translation_m'])]
        for corner in original_parameters['safe_bounds_m']]
    assert task['private']['evaluation']['parameters'] == expected_parameters
    assert task['private']['evaluation']['parameters']['revision'] == manifest['entries'][0]['revision'] == 'controlled_ramp_comparison_v1'
    assert task['id'] == parent['id'] and task['version'] == parent['version'] + 1
    for key in ('public', 'defaults', 'requirements', 'qualification'):
        assert task[key] == parent[key]
    assert task['private']['bindings'] == parent['private']['bindings']
    assert task['scene'] == scene['scene']
    for index in (0, 1):
        before, after = layout(parent, index), layout(task, index)
        assert after['friction'] == before['friction']
        assert after['supported_items'] == before['supported_items']
        assert after['robot_start']['orientation'] == before['robot_start']['orientation']
        assert after['robot_start']['position'] == pytest.approx(np.array(before['robot_start']['position']) + entry['translation_m'])
        for original, translated in zip(before['objects'], after['objects']):
            expected = deepcopy(original)
            expected['position'] = [value + shift for value, shift in zip(original['position'], entry['translation_m'])]
            assert translated == expected
        old_objects = {obj['name']: obj for obj in before['objects']}
        new_objects = {obj['name']: obj for obj in after['objects']}
        old_ramp, new_ramp = old_objects['ramp'], new_objects['ramp']
        inverse = Rotation.from_quat(old_ramp['orientation']).inv()
        old_release = inverse.apply(np.array(old_objects['release_marker']['position']) - old_ramp['position'])
        new_release = inverse.apply(np.array(new_objects['release_marker']['position']) - new_ramp['position'])
        assert new_release == pytest.approx(old_release, abs=1e-14)
        assert (np.array(expected_parameters['safe_bounds_m']) - new_ramp['position']).tolist() == pytest.approx(
            np.array(original_parameters['safe_bounds_m']) - old_ramp['position'])
        parameters = task['scene']['variants']['adapted']['patches'][index]['parameters']
        assert parameters['revision'] == parent['scene']['variants']['adapted']['patches'][index]['parameters']['revision']
        assert 'scene_contract' not in parameters
    assert task['repair']['runtime'] == builder.SceneRuntime().model_dump(mode='json')
    assert builder.main(arguments(ramp_design)) == 0
    assert builder.main(arguments(ramp_design) + ['--check']) == 0


def test_ramp_initializer_revision_is_independent_and_existing_scene_contracts_follow_target(ramp_design):
    entry = ramp_design[2]['entries'][0]
    task = yaml.safe_load(Path(entry['parent_task_file']).read_text())
    scene = yaml.safe_load(Path(entry['parent_scene_file']).read_text())
    task['private']['evaluation']['parameters']['scene_contract'] = {'scene_model': 'house_single_floor', 'room': 'living_room_0'}
    for patch in task['scene']['variants']['adapted']['patches']:
        patch['parameters']['revision'] = 'reviewed_independent_initialization_revision'
        patch['parameters']['scene_contract'] = {'scene_model': 'house_single_floor', 'room': 'living_room_0'}
    scene['scene'] = deepcopy(task['scene'])
    translated, _ = builder.translate_definitions(task, scene, entry, {})
    identity = {'scene_model': entry['scene_model'], 'room': entry['room']}
    assert translated['private']['evaluation']['parameters']['scene_contract'] == identity
    assert translated['private']['evaluation']['parameters']['revision'] == 'controlled_ramp_comparison_v1'
    for patch in translated['scene']['variants']['adapted']['patches']:
        assert patch['parameters']['revision'] == 'reviewed_independent_initialization_revision'
        assert patch['parameters']['scene_contract'] == identity


@pytest.mark.parametrize('scope', ['evaluation_revision', 'evaluation_operation', 'evaluation_coordinates',
                                 'initializer_coordinates', 'layout_coordinates', 'object_coordinates',
                                 'robot_coordinates', 'safe_bounds_shape', 'object_override'])
def test_ramp_rejects_unreviewed_contract_or_untranslated_coordinates(ramp_design, scope):
    entry = ramp_design[2]['entries'][0]
    task = yaml.safe_load(Path(entry['parent_task_file']).read_text())
    scene = yaml.safe_load(Path(entry['parent_scene_file']).read_text())
    evaluation = task['private']['evaluation']
    if scope == 'evaluation_revision':
        evaluation['parameters']['revision'] = 'unreviewed_revision'
    elif scope == 'evaluation_operation':
        evaluation['operation'] = 'unreviewed_operation'
    elif scope == 'evaluation_coordinates':
        evaluation['parameters']['extra_world_position_m'] = [1., 2., 3.]
    elif scope == 'safe_bounds_shape':
        evaluation['parameters']['safe_bounds_m'] = [[1., 2., 3.]]
    elif scope == 'object_override':
        entry['object_overrides'] = {'ramp': {'position': [1., 2., 3.]}}
    else:
        for patch in task['scene']['variants']['adapted']['patches']:
            parameters = patch['parameters']
            target = {'initializer_coordinates': parameters, 'layout_coordinates': parameters['layout'],
                      'object_coordinates': parameters['layout']['objects'][0],
                      'robot_coordinates': parameters['layout']['robot_start']}[scope]
            target['extra_world_position_m'] = [1., 2., 3.]
    scene['scene'] = deepcopy(task['scene'])
    with pytest.raises(ValueError):
        builder.translate_definitions(task, scene, entry, {})
