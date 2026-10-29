"""Check eight authored mirror proposals without native optics or execution.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_wia_mirror_definitions.py
Static proposals do not establish visibility, selected identity or non-goal starts.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from wia.tasks.loader import load_scene, load_task
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition, SceneRuntime
import wia.authoring.commands.build_wia_mirror as builder


@pytest.fixture(scope='module')
def package():
    return builder.build_package()


def layout(entry):
    return entry['task']['scene']['variants']['adapted']['patches'][0]['parameters']['layout']


def test_eight_proposals_sixteen_versions_validate_and_load_actual_background_only_source(package):
    definitions, manifest, rendered = package
    design = json.loads(builder.DESIGN.read_bytes())
    assert manifest['proposal_count'] == 8 and manifest['authored_seed_version_count'] == len(definitions) == 16
    assert len(rendered) == 33
    assert {entry['alias'] for entry in definitions} == set(builder.ALIASES)
    assert manifest['shared_layout_id'] == design['shared_layout_id']
    assert manifest['source']['identity'] == design['source']['identity']
    assert manifest['source']['original_optical_inventory_replaced']
    assert not manifest['source']['original_source_qa_reused']
    assert manifest['full_reference_execution'] == 'deferred_until_user_review'
    for key in ('native_optics_verified', 'native_visibility_verified', 'native_shape_verified', 'native_hand_path_verified',
            'physical_success_verified', 'scientific_controls_verified', 'continuous_path_certified',
            'native_non_goal_initialization_verified'):
        assert manifest[key] is False
    for entry in definitions:
        task, scene = entry['task'], entry['scene']
        CategoryTaskDefinition.model_validate(task)
        SceneDefinition.model_validate(scene)
        assert task['status'] == scene['status'] == 'pending'
        assert task['scene'] == scene['scene']
        assert task['private']['bindings'] == scene['bindings']
        assert task['source'] == scene['source'] == {'provider': 'esi', 'json': design['source']['identity']}
        assert task['scene']['variants']['original']['patches'] == []
        patches = task['scene']['variants']['adapted']['patches']
        assert [patch['phase'] for patch in patches] == ['before_construction', 'after_postprocessing']
        assert all(patch['operation'] == 'category_initialization' and patch['module'] == design['module'] for patch in patches)
        assert patches[0]['parameters'] == patches[1]['parameters']
        evaluation = task['private']['evaluation']
        p = evaluation['parameters']
        assert evaluation['module'] == design['module'] and p['operation'] == evaluation['operation']
        assert p['scoring'] == design['scoring']
        assert [item['id'] for item in p['completion_items']] == design['completion_items'][p['operation']]
        assert p['destinations'] == {'blue': 'pad_blue', 'yellow': 'pad_yellow'}
        assert set(evaluation['binding_keys']) == set(task['private']['bindings'])
        assert task['qualification'] == {'on_failure': 'skip', 'target_movable_collidable': True, 'support': None, 'initial_visibility': 'none'}
        assert task['defaults'] == {'seed': entry['seed'], 'max_steps': 240, 'world_advances_during_inference': True,
            'observation_condition': 'current', 'action_mode': 'parameterized'}
        assert scene['runtime'] == task['repair']['runtime'] == SceneRuntime().model_dump(mode='json')
        loaded_task = load_task(builder.DEFAULT_OUTPUT / entry['task_file'], variant='adapted', run_mode='diagnostic')
        loaded_scene = load_scene(builder.DEFAULT_OUTPUT / entry['scene_file'], variant='adapted', run_mode='diagnostic')
        assert loaded_task.source_path == loaded_scene.source_path
        assert loaded_task.seed == loaded_scene.seed == entry['seed']
        assert loaded_task.definition.scene == loaded_scene.definition.scene
        metadata = next(item for item in manifest['entries'] if item['proposal_alias'] == entry['alias'] and item['seed'] == entry['seed'])
        assert metadata['wia_id'] == 'wia_v1_' + entry['alias'].lower().replace('-', '_')


def test_appearance_pair_has_identical_full_public_payload_and_only_cue_pose_changes(package):
    definitions = {(entry['alias'], entry['seed']): entry for entry in package[0]}
    for seed in (101, 202):
        reflected, direct = definitions[('R-A1', seed)], definitions[('R-A2', seed)]
        assert reflected['task']['id'] == direct['task']['id'] == 'wia_appearance_delivery_v1'
        loaded = [load_task(builder.DEFAULT_OUTPUT / entry['task_file'], variant='adapted', run_mode='diagnostic')
                  for entry in (reflected, direct)]
        assert loaded[0].public_payload == loaded[1].public_payload
        left, right = deepcopy(layout(reflected)), deepcopy(layout(direct))
        assert left['robot_start'] == right['robot_start']
        changed = []
        for first, second in zip(left['objects'], right['objects']):
            assert first['name'] == second['name']
            if first != second:
                changed.append(first['name'])
                assert first['name'] in {'wia_mirror_' + key for key in builder.CUE_KEYS}
                assert first.pop('position') != second.pop('position')
                assert first == second
        assert set(changed) == {'wia_mirror_' + key for key in builder.CUE_KEYS}
        assert left == right
        left_p = deepcopy(reflected['task']['private']['evaluation']['parameters'])
        right_p = deepcopy(direct['task']['private']['evaluation']['parameters'])
        assert left_p.pop('appearance_mode') == 'reflected'
        assert right_p.pop('appearance_mode') == 'direct'
        assert left_p == right_p


def test_sr_anchor_rules_are_fixed_proposals_and_do_not_claim_native_non_goal_validation(package):
    design = json.loads(builder.DESIGN.read_bytes())
    for entry in package[0]:
        objects = {obj['name']: obj for obj in layout(entry)['objects']}
        anchor = objects[design['anchor']['name']]
        assert anchor['fixed_base'] and not anchor['visual_only']
        expected_y = -2.04
        if entry['alias'] == 'R-S1':
            expected_y = -2.04 if entry['seed'] == 101 else .04
        elif entry['alias'] == 'R-S2':
            expected_y = .04 if entry['seed'] == 101 else -2.04
        assert anchor['position'] == [design['anchor']['position_m'][0], expected_y, design['anchor']['position_m'][2]]
        if not entry['alias'].startswith('R-S'):
            continue
        p = entry['task']['private']['evaluation']['parameters']
        assert p['non_goal_initialization_required'] is True
        assert p['requires_both_objects_moved'] is (entry['alias'] == 'R-S3')
        assert len(p['candidates']) == 2
        assert all(objects[name]['rgba'] == design['layout']['candidate_rgba'] for name in p['candidates'].values())
        metadata = next(item for item in package[1]['entries'] if item['proposal_alias'] == entry['alias'] and item['seed'] == entry['seed'])
        proposal = metadata['layout_proposal']
        assert proposal['intended_cue_side_proposal'] == ('negative_y' if entry['seed'] == 101 else 'positive_y')
        assert proposal['native_selected_identity_verified'] is proposal['native_non_goal_initialization_verified'] is False
        assert metadata['native_non_goal_initialization_verified'] is False


def test_unit_cube_fixtures_real_mirror_material_and_red_cue_geometry_follow_design(package):
    design = json.loads(builder.DESIGN.read_bytes())
    rows = {row['alias']: row for row in design['tasks']}
    for entry in package[0]:
        row = rows[entry['alias']]
        config = layout(entry)
        objects = {obj['name']: obj for obj in config['objects']}
        p = entry['task']['private']['evaluation']['parameters']
        bindings = entry['task']['private']['bindings']
        assert len(objects) == len(config['objects']) == len(bindings)
        assert all(obj['type'] == 'PrimitiveObject' and obj['primitive_type'] == 'Cube' and obj['size'] == 1.
            and not obj['visual_only'] and min(obj['scale']) > 0 for obj in objects.values())
        assert {bindings[key]['name'] for key in p['fixture_keys']} == {name for name, obj in objects.items() if obj['fixed_base']}
        assert len([key for key in bindings if key.startswith('leg_')]) == 4
        assert {key for key in bindings if key.startswith('pad_')} == {'pad_blue', 'pad_yellow'}
        for index, color in enumerate(('blue', 'yellow')):
            pad = objects[bindings['pad_' + color]['name']]
            assert pad['scale'] == design['layout']['delivery_pad_dimensions_m']
            assert pad['position'] == [design['layout']['delivery_pad_x_m'], design['layout']['delivery_pad_y_m'][index], design['layout']['pad_z_m']]
        mirror = objects[design['mirror']['name']]
        assert mirror['position'] == design['mirror']['position_m'] and mirror['scale'] == design['mirror']['dimensions_m']
        assert mirror['rgba'] == [.95, .95, .95, 1.] and mirror['fixed_base']
        initialization = entry['task']['scene']['variants']['adapted']['patches'][0]['parameters']
        assert initialization['mirror_material'] == {'binding': 'mirror', **design['mirror']['material']}
        assert initialization['mirror_key'] == 'mirror'
        assert not any(field in obj for obj in objects.values() for field in ('texture', 'texture_path', 'image', 'rgb', 'segmentation'))
        cue = design['cue']
        seed_index = design['seeds'].index(entry['seed'])
        expected_y = row.get('cue_seed_center_y_m', cue['seed_center_y_m'])[seed_index]
        expected_x = (design['mirror']['position_m'][0] - design['mirror']['dimensions_m'][0] / 2 - cue['mirror_front_clearance_m']
            if row['appearance_mode'] == 'reflected' else cue['direct_x_m'])
        lower, upper, top, bottom = (objects[bindings[key]['name']] for key in builder.CUE_KEYS)
        assert upper['position'][1] - upper['scale'][1] / 2 - (lower['position'][1] + lower['scale'][1] / 2) == pytest.approx(cue['inner_width_m'])
        assert top['position'][2] - top['scale'][2] / 2 - (bottom['position'][2] + bottom['scale'][2] / 2) == pytest.approx(cue['inner_height_m'])
        assert (lower['position'][1] + upper['position'][1]) / 2 == pytest.approx(expected_y)
        assert all(obj['position'][0] == expected_x and obj['scale'][0] == cue['depth_m'] and obj['rgba'] == cue['rgba']
            for obj in (lower, upper, top, bottom))
        assert {item['name'] for item in config['supported_items']} == set(p['candidates'].values())
        for index, (key, name) in enumerate(p['candidates'].items()):
            candidate = objects[name]
            assert not candidate['fixed_base'] and candidate['category'] == 'object'
            assert candidate['rgba'] == design['layout']['candidate_rgba']
            assert candidate['scale'] == p['candidate_dimensions_m'][key] == row['candidate_dimensions_m']
            assert candidate['orientation'] == pytest.approx(builder.quaternion(row['candidate_yaw_deg'][index]))
            assert candidate['position'][1] == row['candidate_y_m'][builder.slot_order(entry['seed'], len(row['candidate_y_m']))[index]]


def test_public_contract_preserves_initial_reference_without_source_or_private_identity_leaks(package):
    design = json.loads(builder.DESIGN.read_bytes())
    goals = {row['alias']: row['goal'] for row in design['tasks']}
    for entry in package[0]:
        public = entry['task']['public']
        text = json.dumps(public)
        assert public['objective'] == goals[entry['alias']]
        assert all(term not in text for term in ('wia_mirror_', 'wia_reference_beacon', 'candidate_0', 'candidate_1',
            'q_007', 'source_qa', 'appearance_mode', '7.45', '8.82', '-2.04', 'free_camera', 'segmentation'))
        assert 'initial head-camera view' in text and 'Later robot movement does not redefine' in text
        assert '5 cm' in text and '0.5 seconds' in text and 'empty and open' in text
        if entry['alias'] == 'R-S3':
            assert 'Move each block at least 5 cm' in text
        else:
            assert 'first candidate grasp commits' in text and 'within 2 cm' in text


def test_deterministic_output_refusal_and_immutable_design_and_source(package, tmp_path):
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
    stale_bytes = changed.read_bytes()
    assert builder.main(['--output-dir', str(tmp_path), '--check']) == 1
    with pytest.raises(ValueError, match='Refusing to overwrite a differing generated file'):
        builder.main(['--output-dir', str(tmp_path)])
    assert changed.read_bytes() == stale_bytes
    assert {path: path.read_bytes() for path in before} == before
