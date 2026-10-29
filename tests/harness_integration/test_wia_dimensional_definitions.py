"""Check the fixed dimensional design without native or model execution.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_wia_dimensional_definitions.py
Object-bound proposals never establish gripper or continuous-path feasibility.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from wia.tasks.loader import load_scene, load_task
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition, SceneRuntime
import wia.authoring.commands.build_wia_dimensional as builder


@pytest.fixture(scope='module')
def package():
    return builder.build_package()


@pytest.fixture(scope='module')
def package_v2():
    return builder.build_package(builder.DEFAULT_OUTPUT.parent / 'dimensional_v2/design.json')


def layout(entry):
    return entry['task']['scene']['variants']['adapted']['patches'][0]['parameters']['layout']


def test_all_five_proposals_two_seeds_have_canonical_schema_mapping_and_production_loads(package):
    definitions, manifest, rendered = package
    design = json.loads(builder.DESIGN.read_bytes())
    assert manifest['proposal_count'] == 5 and manifest['authored_seed_version_count'] == len(definitions) == 10
    assert len(rendered) == 21
    assert len({entry['alias'] for entry in definitions}) == 5
    assert manifest['primary_family'] == 'GF' and not manifest['source_category_is_task_family']
    assert manifest['source']['identity'] == design['source']['identity']
    assert manifest['source']['original_optical_inventory_replaced']
    assert not manifest['source']['original_optical_qa_reused']
    for entry in definitions:
        task, scene = entry['task'], entry['scene']
        CategoryTaskDefinition.model_validate(task)
        SceneDefinition.model_validate(scene)
        assert task['status'] == scene['status'] == 'pending'
        assert task['source'] == scene['source'] == {'provider': 'esi', 'json': design['source']['identity']}
        assert task['scene'] == scene['scene']
        assert task['private']['bindings'] == scene['bindings']
        assert task['scene']['variants']['original']['patches'] == []
        patches = task['scene']['variants']['adapted']['patches']
        assert len(patches) == 2
        assert [patch['phase'] for patch in patches] == ['before_construction', 'after_postprocessing']
        assert all(patch['operation'] == 'category_initialization' and patch['module'] == design['initializer_module'] for patch in patches)
        assert patches[0]['parameters'] == patches[1]['parameters']
        evaluation = task['private']['evaluation']
        p = evaluation['parameters']
        assert evaluation['type'] == 'category_operation' and evaluation['module'] == 'wia.authoring.converters.wia_dimensional'
        assert p['operation'] == evaluation['operation']
        assert p['scoring'] == design['scoring']
        assert [item['id'] for item in p['completion_items']] == list(builder.STAGE_IDS[p['operation']])
        assert set(evaluation['binding_keys']) == set(task['private']['bindings'])
        assert task['qualification'] == {'on_failure': 'skip', 'target_movable_collidable': True, 'support': None, 'initial_visibility': 'none'}
        assert task['defaults'] == {'seed': entry['seed'], 'max_steps': 240, 'world_advances_during_inference': True,
            'observation_condition': 'current', 'action_mode': 'parameterized'}
        assert scene['runtime'] == task['repair']['runtime'] == SceneRuntime().model_dump(mode='json')
        loaded_task = load_task(builder.DEFAULT_OUTPUT / entry['task_file'], variant='adapted', run_mode='diagnostic')
        loaded_scene = load_scene(builder.DEFAULT_OUTPUT / entry['scene_file'], variant='adapted', run_mode='diagnostic')
        assert loaded_task.source_path == loaded_scene.source_path
        assert loaded_task.definition.scene == loaded_scene.definition.scene
        assert loaded_task.seed == loaded_scene.seed == entry['seed']
    assert not any(manifest[key] for key in ('native_validation_run', 'native_geometry_verified',
        'native_visibility_verified', 'native_gripper_path_verified', 'continuous_path_certified',
        'scientific_controls_verified', 'parameters_calibrated'))
    assert manifest['full_reference_execution'] == 'deferred_until_user_review'


def test_counterbalanced_seeds_change_only_candidate_positions_and_d2_keeps_first_three(package):
    definitions, manifest, _ = package
    by_key = {(entry['alias'], entry['seed']): entry for entry in definitions}
    for count in (2, 4):
        assert builder.slot_order(101, count) != builder.slot_order(202, count)
    for alias in {entry['alias'] for entry in definitions}:
        first, second = by_key[(alias, 101)], by_key[(alias, 202)]
        left, right = deepcopy(layout(first)), deepcopy(layout(second))
        for config in (left, right):
            for obj in config['objects']:
                if not obj['fixed_base']:
                    obj['position'][1] = None
        assert left == right
        assert first['task']['private'] == second['task']['private']
        assert first['task']['public'] == second['task']['public']
        first_loaded = load_task(builder.DEFAULT_OUTPUT / first['task_file'], variant='adapted', run_mode='diagnostic')
        second_loaded = load_task(builder.DEFAULT_OUTPUT / second['task_file'], variant='adapted', run_mode='diagnostic')
        assert first_loaded.public_payload == second_loaded.public_payload
    for seed in (101, 202):
        three = {obj['name']: obj for obj in layout(by_key[('D2-1', seed)])['objects']}
        four = {obj['name']: obj for obj in layout(by_key[('D2-2', seed)])['objects']}
        assert set(four) - set(three) == {'wia_dim_candidate_3'}
        assert all(three[name] == four[name] for name in three)
        for alias in ('D2-1', 'D2-2'):
            p = by_key[(alias, seed)]['task']['private']['evaluation']['parameters']
            assert p['destinations'] == ['pad_' + str(index) for index in range(3 if alias == 'D2-1' else 4)]
            assert set(p['fixture_keys']) >= {'pad_0', 'pad_1', 'pad_2', 'pad_3'}
    assert manifest['counterbalance']['d2_first_three_positions_preserved_when_adding_fourth']


def test_frame_gaps_depth_and_declared_object_only_fit_bounds_match_design(package):
    definitions, manifest, _ = package
    design = json.loads(builder.DESIGN.read_bytes())
    rows = {row['alias']: row for row in design['tasks']}
    for entry in definitions:
        if not entry['alias'].startswith('D1'):
            continue
        row = rows[entry['alias']]
        objects = {obj['name']: obj for obj in layout(entry)['objects']}
        lower, upper, roof = (objects['wia_dim_' + key] for key in ('aperture_low', 'aperture_high', 'aperture_roof'))
        bottom = design['aperture']['bottom_z_m']
        assert upper['position'][1] - upper['scale'][1] / 2 - (lower['position'][1] + lower['scale'][1] / 2) == pytest.approx(row['opening_width_m'])
        assert roof['position'][2] - roof['scale'][2] / 2 - bottom == pytest.approx(row['opening_height_m'])
        assert lower['position'][2] - lower['scale'][2] / 2 == pytest.approx(bottom)
        assert upper['position'][2] - upper['scale'][2] / 2 == pytest.approx(bottom)
        assert all(obj['scale'][0] == row['tunnel_depth_m'] for obj in (lower, upper, roof))
        assert roof['scale'][1] == design['aperture']['frame_outer_width_m']
        assert not any('bottom' in name or 'rail' in name for name in objects)
        p = entry['task']['private']['evaluation']['parameters']
        assert p['aperture_bindings'] == {'lower_side': 'aperture_low', 'upper_side': 'aperture_high', 'roof': 'aperture_roof', 'floor': 'table'}
        assert objects['wia_dim_pad_0']['position'] == design['aperture']['delivery_pad_center_m']
        proposal = next(item['geometry_proposal'] for item in manifest['entries'] if item['proposal_alias'] == entry['alias'])
        assert proposal['objects']['candidate_0']['object_only_axis_aligned_cross_section_fits_xyz_m']
        decoy = proposal['objects']['candidate_1']['cube_minimum_caliper_bound']
        assert decoy['all_rotations_excluded_by_declared_object_bound']
        assert decoy['minimum_caliper_m'] > min(row['opening_width_m'], row['opening_height_m'])
        assert not decoy['native_dimensions_verified'] and not decoy['hand_or_path_certified']
        assert not proposal['hand_or_path_certified']


def test_native_primitives_bindings_support_inventory_and_intrinsic_assignment_gap(package):
    definitions, manifest, _ = package
    design = json.loads(builder.DESIGN.read_bytes())
    rows = {row['alias']: row for row in design['tasks']}
    for entry in definitions:
        config = layout(entry)
        p = entry['task']['private']['evaluation']['parameters']
        bindings = entry['task']['private']['bindings']
        objects = {obj['name']: obj for obj in config['objects']}
        assert len(objects) == len(config['objects']) == len(bindings)
        assert all(obj['type'] == 'PrimitiveObject' and obj['primitive_type'] == 'Cube'
            and obj['size'] == 1. and not obj['visual_only'] and min(obj['scale']) > 0 for obj in objects.values())
        assert all(name.startswith('wia_dim_') for name in objects)
        assert {bindings[key]['name'] for key in p['fixture_keys']} == {name for name, obj in objects.items() if obj['fixed_base']}
        assert len([key for key in bindings if key.startswith('leg_')]) == 4
        assert objects['wia_dim_table']['position'] == design['layout']['table_center_m']
        assert objects['wia_dim_table']['scale'] == design['layout']['table_dimensions_m']
        assert 'friction' not in config
        assert config['robot_start']['position'] == design['layout']['robot_position_m']
        assert len(config['supported_items']) == len(p['candidates'])
        assert {item['name'] for item in config['supported_items']} == set(p['candidates'].values())
        assert all(item['support'] == 'wia_dim_table' for item in config['supported_items'])
        for index, (key, name) in enumerate(p['candidates'].items()):
            obj = objects[name]
            assert not obj['fixed_base'] and obj['category'] == bindings[key]['category'] == 'object'
            assert obj['rgba'] == design['layout']['candidate_rgba']
            assert obj['scale'] == p['candidate_dimensions_m'][key] == rows[entry['alias']]['candidate_dimensions_m'][index]
            assert obj['orientation'] == pytest.approx(builder.quaternion(rows[entry['alias']]['initial_yaw_deg'][index]))
            assert obj['position'][2] == pytest.approx(design['layout']['tabletop_z_proposal_m'] + obj['scale'][2] / 2 + .003)
    for item in manifest['entries']:
        if item['proposal_alias'].startswith('D2'):
            geometry = item['geometry_proposal']
            assert geometry['declared_gaps_meet_contract']
            assert min(geometry['ordered_neighbor_gaps_m']) >= design['scoring']['minimum_assignment_gap_m']
            assert not geometry['world_aabb_used_as_intrinsic_metric']


def test_public_goal_has_no_internal_candidate_labels_coordinates_or_expected_target(package):
    definitions = package[0]
    design = json.loads(builder.DESIGN.read_bytes())
    for entry in definitions:
        task = entry['task']
        public = json.dumps(task['public'])
        assert task['public']['objective'] == next(row['goal'] for row in design['tasks'] if row['alias'] == entry['alias'])
        assert not any(term in public for term in ('wia_dim_', 'candidate_0', 'candidate_1', 'block A', 'block B', 'block C',
            'block D', '7.6', '8.36', '8.32', '-1.78', '-1.45', 'source_qa', 'expected_answer'))
        assert 'at least 5 cm' in public and '0.5 seconds' in public
        assert 'empty and open' in public and 'without robot contact' in public
        if entry['alias'].startswith('D1'):
            assert 'first candidate block you grasp commits your selection' in public
            assert 'robot hand itself need not pass through' in public
            assert 'within 2 cm' in public
        else:
            assert 'intrinsic longest physical edge' in public


def test_deterministic_generation_preserves_design_source_and_refuses_any_differing_output(package, tmp_path):
    _, manifest, rendered = package
    source_file = builder.ROOT / manifest['source']['file']
    before = {path: path.read_bytes() for path in (builder.DESIGN, source_file)}
    assert manifest['design_sha256'] == hashlib.sha256(before[builder.DESIGN]).hexdigest()
    assert builder.build_package()[2] == rendered
    assert builder.main(['--check']) == 0
    assert builder.main(['--output-dir', str(tmp_path)]) == 0
    assert not (tmp_path / 'design.json').exists()
    mtimes = {path: (tmp_path / path).stat().st_mtime_ns for path in rendered}
    assert builder.main(['--output-dir', str(tmp_path), '--check']) == 0
    assert builder.main(['--output-dir', str(tmp_path)]) == 0
    assert {path: (tmp_path / path).stat().st_mtime_ns for path in rendered} == mtimes
    first = next(path for path in rendered if path.parts[0] == 'tasks')
    changed = tmp_path / first
    changed.write_bytes(changed.read_bytes() + b'\n')
    changed_bytes = changed.read_bytes()
    assert builder.main(['--output-dir', str(tmp_path), '--check']) == 1
    with pytest.raises(ValueError, match='Refusing to overwrite a differing generated file'):
        builder.main(['--output-dir', str(tmp_path)])
    assert changed.read_bytes() == changed_bytes
    assert {path: path.read_bytes() for path in before} == before


def test_v2_all_ten_pairs_have_versioned_ids_hashes_loaders_and_unverified_visibility(package_v2):
    definitions, manifest, rendered = package_v2
    output = builder.DEFAULT_OUTPUT.parent / 'dimensional_v2'
    design_file = output / 'design.json'
    design = json.loads(design_file.read_bytes())
    assert manifest['revision'] == 'wia_dimensional_v2'
    assert manifest['parent_design'] == design['parent_design']
    assert manifest['initial_visibility_contract'] == design['initial_visibility_contract']
    assert manifest['native_visibility_verified'] is False
    assert manifest['initial_visibility_contract']['native_visibility_verified'] is False
    assert manifest['initial_visibility_contract']['layout_coordinates_are_visibility_certificate'] is False
    assert len(definitions) == 10 and len(rendered) == 21
    for entry in definitions:
        task, scene = entry['task'], entry['scene']
        CategoryTaskDefinition.model_validate(task)
        SceneDefinition.model_validate(scene)
        assert task['version'] == scene['version'] == 2
        assert task['id'].endswith('_v2') and scene['id'].endswith('_v2_scene')
        assert entry['task_file'].name.endswith('_v2.yaml') and entry['scene_file'].name.endswith('_v2.yaml')
        assert task['repair']['design_file'] == design_file.relative_to(builder.ROOT).as_posix()
        assert task['repair']['design_sha256'] == manifest['design_sha256'] == hashlib.sha256(design_file.read_bytes()).hexdigest()
        loaded_task = load_task(output / entry['task_file'], variant='adapted', run_mode='diagnostic')
        loaded_scene = load_scene(output / entry['scene_file'], variant='adapted', run_mode='diagnostic')
        assert loaded_task.definition.scene == loaded_scene.definition.scene
        assert loaded_task.definition.private.bindings == loaded_scene.definition.bindings
        assert loaded_task.seed == loaded_scene.seed == entry['seed']
        metadata = next(item for item in manifest['entries'] if item['proposal_alias'] == entry['alias'] and item['seed'] == entry['seed'])
        assert loaded_task.file_sha256 == metadata['task_sha256'] == hashlib.sha256(rendered[entry['task_file']]).hexdigest()
        assert loaded_scene.file_sha256 == metadata['scene_sha256'] == hashlib.sha256(rendered[entry['scene_file']]).hexdigest()
        assert metadata['candidate_xy_positions_m'] == {obj['name'].removeprefix('wia_dim_'): obj['position'][:2]
            for obj in layout(entry)['objects'] if not obj['fixed_base']}


def test_v2_changes_only_declared_candidate_xy_and_fixture_colors_and_retains_d2_first_three(package, package_v2):
    parent = {(entry['alias'], entry['seed']): entry for entry in package[0]}
    revised = {(entry['alias'], entry['seed']): entry for entry in package_v2[0]}
    design = json.loads((builder.DEFAULT_OUTPUT.parent / 'dimensional_v2/design.json').read_bytes())
    assert design['scoring'] == json.loads(builder.DESIGN.read_bytes())['scoring']
    for key, entry in revised.items():
        original = parent[key]
        task, scene = deepcopy(entry['task']), deepcopy(entry['scene'])
        old_task, old_scene = original['task'], original['scene']
        assert task['public'] == old_task['public']
        assert task['defaults'] == old_task['defaults']
        assert task['requirements'] == old_task['requirements']
        assert task['source'] == old_task['source']
        p = task['private']['evaluation']['parameters']
        assert p['scoring'] == old_task['private']['evaluation']['parameters']['scoring']
        assert p['candidate_dimensions_m'] == old_task['private']['evaluation']['parameters']['candidate_dimensions_m']
        current_layout, old_layout = layout(entry), layout(original)
        for obj, old_obj in zip(current_layout['objects'], old_layout['objects']):
            assert obj['name'] == old_obj['name']
            if not obj['fixed_base']:
                index = int(obj['name'].rsplit('_', 1)[1])
                if entry['alias'].startswith('D1'):
                    xy = [design['layout']['candidate_x_m'], design['aperture']['start_candidate_y_slots_m'][builder.slot_order(entry['seed'], 2)[index]]]
                else:
                    xy = design['layout']['candidate_xy_slots_m'][builder.slot_order(entry['seed'], 4)[index]]
                assert obj['position'][:2] == xy
                assert obj['position'][2] == old_obj['position'][2]
                assert obj['rgba'] == old_obj['rgba'] == design['layout']['candidate_rgba']
            elif obj['name'] == 'wia_dim_table':
                assert obj['rgba'] == design['layout']['table_rgba']
            elif obj['name'].startswith('wia_dim_pad_'):
                assert obj == old_obj
            else:
                assert obj['rgba'] == design['layout']['fixture_rgba']
        for definition, previous in ((task, old_task), (scene, old_scene)):
            definition['id'], definition['version'] = previous['id'], previous['version']
            for hook, old_hook in zip(definition['scene']['variants']['adapted']['patches'], previous['scene']['variants']['adapted']['patches']):
                hook['parameters']['revision'] = old_hook['parameters']['revision']
                current, before = hook['parameters']['layout'], old_hook['parameters']['layout']
                current['shared_layout_id'] = before['shared_layout_id']
                for obj, old_obj in zip(current['objects'], before['objects']):
                    if not obj['fixed_base']:
                        obj['position'][:2] = old_obj['position'][:2]
                    elif not obj['name'].startswith('wia_dim_pad_'):
                        obj['rgba'] = old_obj['rgba']
        task['repair'] = deepcopy(old_task['repair'])
        task['private']['evaluation']['parameters']['revision'] = old_task['private']['evaluation']['parameters']['revision']
        assert task == old_task
        assert scene == old_scene
    for seed in (101, 202):
        three = {obj['name']: obj for obj in layout(revised[('D2-1', seed)])['objects']}
        four = {obj['name']: obj for obj in layout(revised[('D2-2', seed)])['objects']}
        assert set(four) - set(three) == {'wia_dim_candidate_3'}
        assert all(three[name] == four[name] for name in three)


def test_v1_exact_regeneration_and_v2_cli_preserve_both_designs_source_and_v1_outputs(package, package_v2, tmp_path):
    output_v2 = builder.DEFAULT_OUTPUT.parent / 'dimensional_v2'
    design_v2 = output_v2 / 'design.json'
    paths = {builder.DESIGN, design_v2, builder.ROOT / package[1]['source']['file']}
    paths.update(builder.DEFAULT_OUTPUT / path for path in package[2])
    before = {path: path.read_bytes() for path in paths}
    assert builder.build_package()[2] == package[2]
    assert {path: (builder.DEFAULT_OUTPUT / path).read_bytes() for path in package[2]} == package[2]
    assert builder.build_package(design_v2)[2] == package_v2[2]
    assert builder.main(['--check']) == 0
    arguments = ['--design-file', str(design_v2), '--output-dir', str(tmp_path)]
    assert builder.main(arguments) == 0
    assert builder.main([*arguments, '--check']) == 0
    assert not (tmp_path / 'design.json').exists()
    assert {path: (tmp_path / path).read_bytes() for path in package_v2[2]} == package_v2[2]
    changed = tmp_path / package_v2[0][0]['task_file']
    changed.write_bytes(changed.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Refusing to overwrite a differing generated file'):
        builder.main(arguments)
    unsupported = json.loads(design_v2.read_bytes())
    unsupported['revision'] = 'wia_dimensional_v3'
    unsupported_path = tmp_path / 'unsupported.json'
    unsupported_path.write_text(json.dumps(unsupported))
    with pytest.raises(ValueError, match='explicitly supported dimensional V1 and V2'):
        builder.build_package(unsupported_path)
    assert {path: path.read_bytes() for path in before} == before
