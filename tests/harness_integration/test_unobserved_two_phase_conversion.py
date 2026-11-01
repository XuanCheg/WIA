"""Validate continuous Unobserved Change conversion without starting simulation.

Run: /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q
tests/test_unobserved_two_phase_conversion.py
The real source is parsed on CPU; native opening and temporal installation use
explicit interface doubles rather than fabricated robot observations.
"""
import copy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from wia.authoring.conversion import ConversionContext, ROOT, SourceInstance
from wia.authoring.converters import temporal_understanding_two_phase as temporal
from wia.authoring.converters import temporal_understanding_unobserved_change as legacy
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition
from wia.tasks.scene import ScenePatches


RELATIVE = ('Temporal Understanding/Unobserved Change/Beechwood_0_garden/living_room_0/'
            'current_state_reasoning__q_001.json')


@pytest.fixture
def source():
    return SourceInstance(ROOT / 'ESI/dataset/json_clean' / RELATIVE, RELATIVE,
                          *legacy.CategoryConverter.category)


@pytest.fixture
def legacy_definitions(tmp_path, monkeypatch):
    def build(source):
        layout = legacy._layout(source)
        asset_root = tmp_path / 'fixture_assets'
        for key, binding in layout['bindings'].items():
            if key != 'floor':
                (asset_root / 'objects' / binding['category'] / binding['model'] / 'usd').mkdir(
                    parents=True, exist_ok=True)
        monkeypatch.setattr(legacy, '_asset_root', lambda context: asset_root)
        monkeypatch.setattr(legacy, '_robot_start', lambda layout, context: {
            'position': [4., 12., 0.], 'yaw_deg': 45., 'connected_proposals_xy': [[4., 12.]]})
        monkeypatch.setattr(legacy, '_route_screen', lambda layout, start: {
            'container': layout['container_keys'][0], 'proposal_xy': [[4., 12.], [4.8, 12.]]})
        result = legacy.CategoryConverter().convert(source, ConversionContext(tmp_path / 'unused'))
        assert result['status'] == 'generated', result
        scene, task = result['scene_definition'], result['task_definition']
        scene['runtime'] = {'camera_size': [320, 240], 'debug_cameras': False}
        task['defaults']['max_steps'] = 91
        # Private provenance and an unrelated head restoration must survive.
        for definition in (scene, task):
            patches = definition['scene']['variants']['adapted']['patches']
            patches.append({'operation': 'restore_robot_head_mount', 'phase': 'after_postprocessing'})
            patches.append({'operation': 'prepare_category_interaction', 'phase': 'after_postprocessing',
                'mode': 'manipulation', 'group_bindings': layout['container_keys'] + layout['phase_keys']['image2'],
                'manipulation_bindings': layout['phase_keys']['image2'], 'relocation_policy': 'preserve',
                'source_region': source.raw['room']})
            for patch in patches:
                if patch['operation'] == 'category_initialization':
                    patch['parameters']['camera_provenance'] = {'mounted_camera_config': 'preserve exactly'}
        return result
    return build


def initialization(definition, phase='before_construction'):
    return next(patch for patch in definition['scene']['variants']['adapted']['patches']
                if patch['operation'] == 'category_initialization' and patch['phase'] == phase)


def test_real_source_upgrade_is_pure_idempotent_and_preserves_original_configuration(source, legacy_definitions):
    result = legacy_definitions(source)
    old_scene, old_task = result['scene_definition'], result['task_definition']
    original_scene, original_task = copy.deepcopy(old_scene), copy.deepcopy(old_task)
    source_bytes = source.path.read_bytes()
    scene, task = temporal.upgrade_existing(old_scene, old_task, source)
    assert old_scene == original_scene and old_task == original_task
    assert temporal.upgrade_existing(scene, task, source) == (scene, task)
    assert scene is not old_scene and task is not old_task
    assert scene['source'] == old_scene['source'] == task['source']
    assert scene['seed'] == old_scene['seed']
    assert scene['runtime'] == old_scene['runtime']
    assert task['defaults'] == old_task['defaults']
    assert task['qualification'] == old_task['qualification']
    assert task['private']['bindings'] == old_task['private']['bindings']
    assert scene['bindings'] == old_scene['bindings']
    for variant in ('original', 'saved_image1', 'saved_image2'):
        assert scene['scene']['variants'][variant] == old_scene['scene']['variants'][variant]
        assert task['scene']['variants'][variant] == old_task['scene']['variants'][variant]
    content_keys = set(initialization(old_scene)['parameters']['phase_keys']['image1'] +
                       initialization(old_scene)['parameters']['phase_keys']['image2'])
    old_kept = [patch for patch in old_scene['scene']['variants']['adapted']['patches']
                if patch['operation'] != 'category_initialization' and not (
                    patch['operation'] == 'object_physics' and patch.get('binding') in content_keys)]
    new_kept = [patch for patch in scene['scene']['variants']['adapted']['patches']
                if patch['operation'] != 'category_initialization']
    assert new_kept == old_kept
    for phase in ('before_construction', 'after_postprocessing'):
        prior, current = initialization(old_scene, phase), initialization(scene, phase)
        assert current['module'] == temporal.MODULE
        assert current['parameters']['phase'] == 'image1'
        assert current['parameters']['temporal_episode'] is True
        for key, value in prior['parameters'].items():
            if key != 'phase':
                assert current['parameters'][key] == value
    params = task['private']['evaluation']['parameters']
    for key, value in old_task['private']['evaluation']['parameters'].items():
        assert params[key] == value
    assert params['source_task_type'] == 'current_state_reasoning'
    assert params['public_options'] == ['brown box', 'red box', 'yellow box']
    assert params['original_answer'] == 'yellow box'
    assert params['box_labels'] == {'container_00': 'brown', 'container_01': 'red', 'container_02': 'yellow'}
    assert params['target_box_index'] == 2
    assert all(params[key] == value for key, value in temporal.PHASE_2_GOAL_PARAMETERS.items())
    for phase in ('before_construction', 'after_postprocessing'):
        assert not set(temporal.PHASE_2_GOAL_PARAMETERS) & initialization(scene, phase)['parameters'].keys()
    # A v2.0 continuous scene gets only a task-goal revision, once.
    prior_scene, prior_task = copy.deepcopy(scene), copy.deepcopy(task)
    for key in temporal.PHASE_2_GOAL_PARAMETERS:
        prior_task['private']['evaluation']['parameters'].pop(key)
    prior_task['public']['success_criteria'] = 'Legacy observation-only answer goal.'
    revised_scene, revised_task = temporal.upgrade_existing(prior_scene, prior_task, source)
    assert revised_scene == prior_scene
    assert revised_task['scene'] == prior_task['scene']
    assert revised_task['version'] == prior_task['version'] + 1
    assert temporal.upgrade_existing(revised_scene, revised_task, source) == (revised_scene, revised_task)
    assert set(temporal.CAPABILITIES) <= set(task['requirements']['capabilities'])
    assert source.path.read_bytes() == source_bytes
    SceneDefinition.model_validate(scene)
    task_definition(task)


def test_public_goal_uses_only_original_question_options_and_generic_protocol(source, legacy_definitions):
    result = legacy_definitions(source)
    _, task = temporal.upgrade_existing(result['scene_definition'], result['task_definition'], source)
    public = task['public']
    assert public['objective'].startswith(source.raw['question'] + '\n\n')
    assert json.dumps(json.loads(source.raw['options_json'])) in public['objective']
    assert 'survey_complete=true with answer=null and command DONE' in public['objective']
    assert 'selected option in answer and choose DONE' in public['objective']
    assert 'Phase 1 is read-only' in public['objective']
    assert 'active survey with at most 20' in public['objective']
    assert 'Phase 1 allows at most 20 decisions including invalid or refused decisions' in public['constraints'][0]
    assert 'Arm and gripper commands are disabled' in public['constraints'][0]
    assert 'Phase 2 permits existing guarded base, arm, rotation, trunk and gripper controls' in public['constraints'][1]
    assert 'no fixed route or mandatory grasp' in public['constraints'][1]
    assert 'approach and inspect at least one original container after the change' in public['objective']
    assert 'Move at least 0.25 m from the Phase-1 base position' in public['objective']
    assert 'arrive within 1.0 m of any original container' in public['objective']
    assert 'answering without the required Phase-2 base operation does not complete the task' in public['success_criteria']
    text = json.dumps(public)
    layout = legacy._layout(source)
    for phase in legacy.PHASES:
        for key in layout['phase_keys'][phase]:
            binding = layout['bindings'][key]
            assert binding['name'] not in text
            assert binding['model'] not in text
            assert binding['category'].replace('_', ' ') not in text
    assert 'phase_description' not in text
    assert 'original_answer' not in text
    assert 'changed_box_index' not in text
    alternative_raw, alternative_payload = source.raw, source.payload
    alternative_raw['answer'] = 'brown box'
    alternative_payload['question_data']['answer'] = 'brown box'
    alternative = SimpleNamespace(path=source.path, relative_path=source.relative_path,
        big_task=source.big_task, small_task=source.small_task, raw=alternative_raw, payload=alternative_payload)
    _, alternative_task = temporal.upgrade_existing(result['scene_definition'], result['task_definition'], alternative)
    assert alternative_task['public'] == public
    assert alternative_task['private']['evaluation']['parameters']['original_answer'] == 'brown box'


def test_configure_constructs_both_phases_fixed_collidable_at_original_positions(source, legacy_definitions):
    result = legacy_definitions(source)
    scene, _ = temporal.upgrade_existing(result['scene_definition'], result['task_definition'], source)
    p = initialization(scene)['parameters']
    model = SceneDefinition.model_validate(scene)
    adapter = ScenePatches(SimpleNamespace(definition=model, variant='adapted', seed=model.seed), model.bindings)
    furniture = {'name': 'original_furniture', 'scale': [1., 2., 3.], 'position': [-2., 1., 0.],
                 'fixed_base': True, 'visual_only': False}
    config = {'scene': {'scene_model': p['scene_model'], 'load_object_categories': ['floors', 'walls', 'table']},
              'robots': [{'type': 'R1', 'camera_configuration': {'head': 'unchanged'}}],
              'objects': [copy.deepcopy(furniture)]}
    adapter.apply_to_env_config(config)
    objects = {obj['name']: obj for obj in config['objects']}
    assert objects['original_furniture'] == furniture
    assert config['robots'][0]['camera_configuration'] == {'head': 'unchanged'}
    for key in p['container_keys']:
        obj = objects[p['bindings'][key]['name']]
        assert obj['fixed_base'] is True and obj['visual_only'] is False
        assert obj['position'] == p['source_objects'][key]['position']
    for phase in legacy.PHASES:
        for key in p['phase_keys'][phase]:
            obj = objects[p['bindings'][key]['name']]
            assert obj['fixed_base'] is True and obj['visual_only'] is False
            assert 'scale' not in obj
            if phase == 'image1':
                assert obj['position'] == p['source_objects'][key]['position']
            else:
                assert obj['position'][0] >= 1000.
    assert initialization(scene)['parameters']['camera_provenance'] == {'mounted_camera_config': 'preserve exactly'}


def variant_source(source, tmp_path, change):
    raw, payload = source.raw, copy.deepcopy(source.payload)
    q = payload['question_data']
    first = copy.deepcopy(q['boxes'][0])
    original_content = copy.deepcopy(first['phase1_content'])
    other_content = copy.deepcopy(q['boxes'][1]['phase1_content'])
    first['position_label'] = 'the box'
    first['change_type'] = change
    first['phase1_content'] = None if change == 'add' else original_content
    first['phase2_content'] = None if change == 'remove' else (
        other_content if change == 'replace' else original_content)
    q.update(boxes=[first], box_count=1, task_type='change_detection', question_metadata={'target_box_index': 0},
             options=['yes', 'no'], question='Did the box contents change between Phase 1 and Phase 2?',
             answer='no' if change == 'no_change' else 'yes')
    for phase in legacy.PHASES:
        q['render']['gt_view'][phase]['images'] = [item for item in q['render']['gt_view'][phase]['images']
                                                 if item['box_index'] == 0]
    payload['task_type'] = 'change_detection'
    raw.update(question=q['question'], answer=q['answer'], options_json=json.dumps(q['options']),
               metadata_json=json.dumps(payload))
    relative = str(Path(*legacy.CategoryConverter.category) / raw['scene'] / raw['room'] / (change + '.json'))
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw))
    return SourceInstance(path, relative, *legacy.CategoryConverter.category)


@pytest.mark.parametrize('change', ['add', 'remove', 'replace', 'no_change'])
def test_original_add_remove_replace_and_nochange_slots(source, tmp_path, legacy_definitions, change):
    example = variant_source(source, tmp_path, change)
    result = legacy_definitions(example)
    scene, task = temporal.upgrade_existing(result['scene_definition'], result['task_definition'], example)
    p = initialization(scene)['parameters']
    assert len(p['phase_keys']['image1']) == (0 if change == 'add' else 1)
    assert len(p['phase_keys']['image2']) == (0 if change == 'remove' else 1)
    private = task['private']['evaluation']['parameters']
    assert private['box_labels'] == {'container_00': 'the box'}
    assert private['target_box_index'] == 0
    assert private['original_answer'] == example.raw['answer']
    assert private['public_options'] == ['yes', 'no']
    assert task['public']['objective'].startswith(example.raw['question'])
    assert 'survey_complete=true' in task['public']['objective']
    for key in p['phase_keys']['image1'] + p['phase_keys']['image2']:
        assert p['bindings'][key]['model'] not in json.dumps(task['public'])


@pytest.fixture
def native_initialization(monkeypatch):
    states = ModuleType('omnigibson.object_states')
    states.Open = type('Open', (), {})
    monkeypatch.setitem(sys.modules, 'omnigibson', ModuleType('omnigibson'))
    monkeypatch.setitem(sys.modules, 'omnigibson.object_states', states)
    manager = ModuleType('wia.native.esi_temporal_phases')
    installed = []
    manager.install_temporal_phases = lambda session, p: installed.append((session, copy.deepcopy(p))) or {'installed': True}
    manager.qualify_operation = lambda *args: {'delegated_qualification': args}
    manager.create_evaluator = lambda *args: ('delegated_evaluator', args)
    monkeypatch.setitem(sys.modules, 'wia.native.esi_temporal_phases', manager)

    def build(parameters, *, open_fails=False):
        objects, calls = {}, {}
        for key in parameters['container_keys']:
            binding = parameters['bindings'][key]
            history = []
            calls[key] = history
            lid = SimpleNamespace(prim_path='/World/' + binding['name'] + '/lid', visible=False, collision_enabled=True)
            state = SimpleNamespace(open=False, relevant_joints_info=(None, [SimpleNamespace(body1=lid.prim_path)]))

            def set_value(value, state=state, history=history):
                history.append(value)
                if value and open_fails:
                    return False
                state.open = value
                return True

            state.set_value, state.get_value = set_value, lambda state=state: state.open
            objects[binding['name']] = SimpleNamespace(**binding, fixed_base=True, visual_only=False,
                states={states.Open: state}, links={'lid': lid})
        sensor = SimpleNamespace(modalities={'rgb'})
        sensor.add_modality = lambda name: sensor.modalities.add(name)
        robot = SimpleNamespace(sensors={'head': sensor})
        session = SimpleNamespace(scene_name=parameters['scene_model'], room_name=parameters['room'],
            env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda kind, name: objects.get(name)), robots=[robot]))
        return session, objects, calls
    return build, installed, manager


def test_initialization_opens_visible_lids_once_and_installs_same_episode(source, legacy_definitions, native_initialization):
    result = legacy_definitions(source)
    scene, _ = temporal.upgrade_existing(result['scene_definition'], result['task_definition'], source)
    p = initialization(scene)['parameters']
    build, installed, _ = native_initialization
    session, objects, calls = build(p)
    evidence = temporal.initialize_scene(session, p)
    assert evidence['temporal_episode'] is True and evidence['phase'] == 'phase_1'
    assert evidence['historical_agent_observations_created'] is False
    assert evidence['warning_codes'] == list(temporal.WARNING_CODES)
    assert evidence['human_framing_names'] == [p['bindings'][key]['name'] for key in
                                             p['container_keys'] + p['phase_keys']['image1']]
    assert installed == [(session, p)]
    for key, history in calls.items():
        assert history == [False, True]
        obj = objects[p['bindings'][key]['name']]
        assert obj.fixed_base is True and obj.visual_only is False
        assert obj.links['lid'].visible is True and obj.links['lid'].collision_enabled is True
    with pytest.raises(temporal.TemporalInitializationError) as failure:
        temporal.initialize_scene(session, p)
    assert failure.value.details['failure_stage'] == 'source_initialization'
    assert len(installed) == 1


def test_failed_opening_reports_structured_failure_without_install_or_hiding(source, legacy_definitions, native_initialization):
    result = legacy_definitions(source)
    scene, _ = temporal.upgrade_existing(result['scene_definition'], result['task_definition'], source)
    p = initialization(scene)['parameters']
    build, installed, _ = native_initialization
    session, objects, _ = build(p, open_fails=True)
    with pytest.raises(temporal.TemporalInitializationError) as failure:
        temporal.initialize_scene(session, p)
    assert failure.value.details['failure_stage'] == 'phase_1_opening'
    assert failure.value.details['failure_reason'] == 'original_lid_open_initialization_failed'
    assert installed == []
    assert all(obj.links['lid'].visible and obj.links['lid'].collision_enabled for obj in objects.values())


def test_shared_evaluation_interfaces_are_lazy_and_forward_exact_arguments(native_initialization):
    _, _, manager = native_initialization
    args = tuple(object() for _ in range(4))
    assert temporal.qualify_operation(*args) == {'delegated_qualification': args}
    assert temporal.create_evaluator(*args) == ('delegated_evaluator', args)
    assert temporal.qualify_operation(*args) == manager.qualify_operation(*args)


def test_new_converter_upgrades_generated_result_and_retains_failures(source, legacy_definitions, tmp_path, monkeypatch):
    generated = legacy_definitions(source)
    unchanged = copy.deepcopy(generated)
    monkeypatch.setattr(legacy.CategoryConverter, 'convert', lambda *args: copy.deepcopy(generated))
    result = temporal.CategoryConverter().convert(source, ConversionContext(tmp_path))
    assert generated == unchanged
    assert result['status'] == 'generated'
    assert result['operation_recipe'] == temporal.OPERATION and result['initial_phase'] == 'image1'
    assert result['task_definition']['private']['evaluation']['module'] == temporal.MODULE
    assert temporal.CategoryConverter.version == 'unobserved_two_phase_v4.0'
    parameters = result['task_definition']['private']['evaluation']['parameters']
    assert all(parameters[key] == value for key, value in temporal.PHASE_2_GOAL_PARAMETERS.items())
    assert 'independent_phase_snapshots_only' not in result['warning_codes']
    failure = {'status': 'conversion_failed', 'task_definition': None,
               'failure_reason': 'original_asset_unavailable', 'failure_stage': 'asset_availability'}
    monkeypatch.setattr(legacy.CategoryConverter, 'convert', lambda *args: copy.deepcopy(failure))
    assert temporal.CategoryConverter().convert(source, ConversionContext(tmp_path)) == failure


def test_conflicting_original_answer_is_rejected_without_rewriting_source(source, legacy_definitions, tmp_path):
    result = legacy_definitions(source)
    raw = source.raw
    raw['answer'] = 'brown box'
    path = tmp_path / 'conflicting.json'
    path.write_text(json.dumps(raw))
    conflicting = SourceInstance(path, source.relative_path, *legacy.CategoryConverter.category)
    original = path.read_bytes()
    with pytest.raises(ValueError, match='Source answer fields disagree'):
        temporal.upgrade_existing(result['scene_definition'], result['task_definition'], conflicting)
    assert path.read_bytes() == original
