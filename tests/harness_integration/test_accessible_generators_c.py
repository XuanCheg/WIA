"""Check accessible category-C groups and pre-observation temporal rebasing.

Run from the repository root with behavior Python:
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q \
  tests/test_accessible_generators_c.py tests/test_native_temporal_phases.py
Native-shaped CPU fixtures do not establish support, visibility, IK or GPU success.
"""
from copy import deepcopy
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.authoring.conversion import ROOT, ConversionContext, SourceInstance
from wia.authoring.converters import specular_reflection_reflection_authoring as authoring
from wia.authoring.converters import temporal_understanding_agent_observation as patrol
from wia.authoring.converters import temporal_understanding_two_phase as temporal
from wia.authoring.converters import temporal_understanding_unobserved_change as legacy
from wia.native.esi_temporal_phases import TemporalPhaseController, TemporalChangeEvaluator, qualify_operation
from harness_integration.test_native_temporal_phases import world
from harness_integration.test_annotation_reflection_temporal import lift_world, retrieval_world


CORPUS = ROOT / 'ESI/dataset/json_clean'


def sources(category):
    return [SourceInstance(path, str(path.relative_to(CORPUS)), *category)
            for path in sorted(CORPUS.joinpath(*category).rglob('*.json'))]


def shared_placement(world):
    """Stand in for already accepted shared placement, never for task actions."""
    c = world.controller
    c.parameters['accessible_support_rebase_required'] = True
    translations, supports, heights = {}, {}, {}
    for index, container in enumerate(c.parameters['container_keys']):
        delta = np.array([index * .4, -.2 * index, .4 + .1 * index])
        key = c.slots['image1'][container]
        for member in [container] + ([key] if key else []):
            obj = c.objects[member]
            obj.position += delta
            translations[obj.name] = delta.tolist()
        obj = c.objects[container]
        supports[obj.name] = '/World/measured_support_' + str(index)
        heights[obj.name] = float(obj.aabb[0][2])
    return {'accepted': True, 'object_translations_m': translations,
            'support_prim_paths': supports, 'support_surfaces_z_m': heights}


def test_temporal_rebase_updates_both_plans_without_moving_parked_actors_or_robot(world, tmp_path):
    before_source = deepcopy(world.controller.parameters['source_objects'])
    inactive = {key: obj.position.copy() for key, obj in world.objects.items() if key.startswith('image2_')}
    evidence = shared_placement(world)
    before_world = deepcopy(world.controller._unchanged_world())
    assert not qualify_operation(world.session, world.executor, world.config, {})['accepted']
    proof = temporal.on_interaction_initialized(world.session, evidence)
    assert proof['accepted'] and proof['actor_pose_writes'] == proof['physics_steps'] == 0
    assert before_world == world.controller._unchanged_world()
    for key, original in before_source.items():
        if key.startswith('image'):
            container = world.controller.parameters['relations'][key]
        else:
            container = key
        name = world.controller.objects[container].name
        expected = np.array(original['position']) + evidence['object_translations_m'][name]
        np.testing.assert_allclose(world.controller.parameters['source_objects'][key]['position'], expected)
    for key, position in inactive.items():
        np.testing.assert_array_equal(world.objects[key].position, position)
        assert world.objects[key].pose_writes == 0
    assert qualify_operation(world.session, world.executor, world.config, {})['accepted']
    evaluator = TemporalChangeEvaluator(world.session, world.executor, world.config, tmp_path)
    first, second = NS(snapshot_id=1, action_step=0), NS(snapshot_id=2, action_step=0)
    evaluator.begin_episode(first)
    evaluator.prepare_phase_transition(first)
    world.controller.advance()
    evaluator.record(second, action=None)
    evaluator.report_answer('yellow box', second)
    assert evaluator.record(second, action=None)['task_success']
    assert world.controller.transition['world_unchanged']
    assert world.session.robot.pose_writes == 0


def test_pre_camera_initialization_rebase_reads_sole_env_robot_without_attaching_or_stepping(world):
    robot = world.session.robot
    robot.sensors = {'head': NS(get_position_orientation=lambda frame: (
        np.array([.1, 0., .8]), np.array([0., 0., 0., 1.])),
        get_obs=lambda: pytest.fail('Initialization rebasing must not capture sensors.'))}
    clock = NS(current_time=17.25, step=lambda: pytest.fail('Initialization rebasing must not step physics.'))
    world.session.pipeline = NS(og=NS(sim=clock))
    evidence = shared_placement(world)
    before = world.controller._unchanged_world()
    poses = {key: obj.position.copy() for key, obj in world.objects.items()}
    world.session.env.robots = [robot]
    world.session.robot = None
    assert world.controller._unchanged_world() == before
    proof = temporal.on_interaction_initialized(world.session, evidence)
    assert proof['accepted'] and proof['world_unchanged_by_callback']
    assert proof['actor_pose_writes'] == proof['physics_steps'] == 0
    assert world.controller._unchanged_world() == before
    assert world.session.robot is None and world.session.env.robots == [robot]
    assert clock.current_time == 17.25 and robot.pose_writes == 0
    for key, position in poses.items():
        np.testing.assert_array_equal(world.objects[key].position, position)
        assert world.objects[key].pose_writes == 0


@pytest.mark.parametrize('robot_count', [0, 2, None])
def test_pre_attachment_world_snapshot_requires_one_original_robot(world, robot_count):
    robot = world.session.robot
    world.session.robot = None
    world.session.env.robots = [robot] * robot_count if robot_count is not None else None
    with pytest.raises(ValueError, match='exactly one original environment robot'):
        world.controller._unchanged_world()
    assert world.session.robot is None and robot.pose_writes == 0


def test_phase2_replacement_activates_at_rebased_container_not_original_floor(world, tmp_path):
    p = deepcopy(world.controller.parameters)
    p['bindings']['image2_content_00']['category'] = 'replacement'
    world.objects['image2_content_00'].category = 'replacement'
    world.controller = TemporalPhaseController(world.session, p)
    world.session.temporal_phase_controller = world.controller
    evidence = shared_placement(world)
    inactive_before = world.objects['image2_content_00'].position.copy()
    temporal.on_interaction_initialized(world.session, evidence)
    np.testing.assert_array_equal(world.objects['image2_content_00'].position, inactive_before)
    world.config['parameters'].update(source_task_type='change_detection', original_answer='yes', public_options=['yes', 'no'])
    evaluator = TemporalChangeEvaluator(world.session, world.executor, world.config, tmp_path)
    first, second = NS(snapshot_id=5, action_step=2), NS(snapshot_id=6, action_step=2)
    evaluator.begin_episode(first)
    evaluator.prepare_phase_transition(first)
    world.controller.advance()
    target = world.controller.parameters['source_objects']['image2_content_00']['position']
    np.testing.assert_allclose(world.objects['image2_content_00'].position, target)
    assert world.objects['image2_content_00'].visible
    evaluator.record(second, action=None)
    evaluator.report_answer('yes', second)
    assert evaluator.record(second, action=None)['task_success']


def test_content_support_cannot_replace_actual_container_support(world):
    evidence = shared_placement(world)
    container, content = world.objects['container_00'], world.objects['image1_content_00']
    support = evidence['support_prim_paths'].pop(container.name)
    evidence['support_prim_paths'][content.name] = support
    with pytest.raises(ValueError, match='content support is insufficient'):
        temporal.on_interaction_initialized(world.session, evidence)
    assert world.controller.interaction_rebase is None


def test_unchanged_container_requires_its_actual_floor_contacts_when_support_map_absent(world):
    evidence = shared_placement(world)
    container, content = world.objects['container_00'], world.objects['image1_content_00']
    delta = np.array(evidence['object_translations_m'][container.name])
    container.position -= delta
    content.position -= delta
    container.extent[2] = 1.
    evidence['object_translations_m'][container.name] = [0., 0., 0.]
    evidence['object_translations_m'][content.name] = [0., 0., 0.]
    evidence['support_prim_paths'].pop(container.name)
    with pytest.raises(ValueError, match='content support is insufficient'):
        temporal.on_interaction_initialized(world.session, evidence)
    floor = world.objects['floor']
    container.contact_list = lambda: [NS(body0=container.prim_path + '/base_link', body1=floor.prim_path,
        position=[0., 1., 0.], normal=[0., 0., 1.])]
    proof = temporal.on_interaction_initialized(world.session, evidence)
    assert proof['container_support_prim_paths']['container_00'] is None
    assert proof['unchanged_container_floor_contacts']['container_00']


@pytest.mark.parametrize('fault', ['unaccepted', 'missing_support', 'different_content_delta', 'moved_inactive', 'wrong_actual_pose'])
def test_invalid_shared_rebase_never_changes_expected_plans(world, fault):
    evidence = shared_placement(world)
    initial_parameters = deepcopy(world.controller.parameters)
    if fault == 'unaccepted':
        evidence['accepted'] = False
    elif fault == 'missing_support':
        evidence['support_prim_paths'].clear()
    elif fault == 'different_content_delta':
        evidence['object_translations_m'][world.objects['image1_content_00'].name][0] += .2
    elif fault == 'moved_inactive':
        world.objects['image2_content_00'].position += [1., 0., 0.]
    else:
        world.objects['container_00'].position += [1., 0., 0.]
    actual = {key: obj.position.copy() for key, obj in world.objects.items()}
    with pytest.raises(ValueError):
        temporal.on_interaction_initialized(world.session, evidence)
    assert world.controller.parameters == initial_parameters
    assert world.controller.interaction_rebase is None
    for key, position in actual.items():
        np.testing.assert_array_equal(world.objects[key].position, position)
        assert world.objects[key].pose_writes == 0


def test_temporal_rebase_is_owner_only_once_and_before_first_observation(world, tmp_path):
    evidence = shared_placement(world)
    errors = []
    def nonowner():
        try:
            temporal.on_interaction_initialized(world.session, evidence)
        except RuntimeError as error:
            errors.append(str(error))
    thread = threading.Thread(target=nonowner)
    thread.start()
    thread.join()
    assert errors and world.controller.interaction_rebase is None
    temporal.on_interaction_initialized(world.session, evidence)
    with pytest.raises(ValueError):
        temporal.on_interaction_initialized(world.session, evidence)
    evaluator = TemporalChangeEvaluator(world.session, world.executor, world.config, tmp_path)
    evaluator.begin_episode(NS(snapshot_id=1, action_step=0))
    # Even a controller without a prior successful hook cannot rebase after it
    # has delivered an episode observation.
    world.controller.interaction_rebase = None
    with pytest.raises(ValueError):
        temporal.on_interaction_initialized(world.session, evidence)


def test_new_authoring_accepts_only_initialized_accessible_xyz_reference(tmp_path):
    world = lift_world(tmp_path, supported_height=.5)
    world.config['parameters']['accessible_target_relocation'] = True
    world.prop.position[0] += .2
    world.session.category_interaction_setup['object_translations_m'][world.prop.name][0] = .2
    qualified = authoring.qualify_operation(world.session, world.controller, world.config, world.bound)
    assert qualified['accepted'], qualified
    assert not qualified['native_state']['accepted_reference_vertical_only']
    world.session.category_interaction_setup['accepted'] = False
    assert not authoring.qualify_operation(world.session, world.controller, world.config, world.bound)['accepted']


@pytest.mark.parametrize('kind', ['authoring', 'retrieval'])
def test_accepted_native_root_overrides_bad_saved_height_and_constructor_drift(tmp_path, kind):
    world = lift_world(tmp_path, supported_height=.5) if kind == 'authoring' else retrieval_world(tmp_path, supported_height=.5)
    setup = world.session.category_interaction_setup
    setup['object_translations_m'][world.prop.name] = [0., 0., 101.6]
    setup['adapted_geometry'] = {world.prop.name: {'position_m': world.prop.position.tolist()}}
    if kind == 'authoring':
        world.config['parameters']['expected_optical_poses']['prop']['position'][2] = -100.
        world.config['parameters']['accessible_target_relocation'] = True
        result = authoring.qualify_operation(world.session, world.controller, world.config, world.bound)
        assert result['accepted'], result
    else:
        world.config['parameters']['source_objects']['prop']['position'][2] = -100.
        _, checks = patrol._initial_checks(world.session, world.config)
        assert all(checks.values()), checks


def test_all_authoring_sources_request_accessible_prop_without_moving_optical_anchors(tmp_path):
    rows = sources(authoring.CategoryConverter.category)
    assert len(rows) == 99
    for source in rows:
        result = authoring.CategoryConverter().convert(source, ConversionContext(tmp_path))
        policy = result['interaction_policy']
        assert policy['relocation_policy'] == 'independent_groups'
        assert policy['placement_groups'] == [['prop']]
        assert policy['source_camera_policy'] == 'preserve'
        assert policy['support_search'] == 'measured_surfaces' and policy['allow_fallback_table']
        p = result['task_definition']['private']['evaluation']['parameters']
        assert p['accessible_target_relocation'] is True
        assert p['minimum_lift_m'] == .05
        assert 'mirror' not in str(result['task_definition']['public']).lower()
        assert 'optical_layout_changed_by_accessible_target_placement' in result['warning_codes']


def test_all_observation_props_are_independent_and_bad_height_is_pending_repair(tmp_path):
    rows = sources(patrol.CATEGORY)
    assert len(rows) == 133
    bindings, repaired_sources = 0, 0
    for source in rows:
        result = patrol.CategoryConverter().convert(source, ConversionContext(tmp_path))
        assert result['status'] == 'generated', (source.relative_path, result)
        names = [obj['_key'] for obj in source.payload['objects_meta']]
        assert result['interaction_policy']['placement_groups'] == [[name] for name in names]
        assert result['interaction_policy']['relocation_policy'] == 'independent_groups'
        assert result['private_source_evidence']['object_count'] == len(names)
        bindings += len(names)
        if result['private_source_evidence']['implausibly_below_floor_bindings']:
            repaired_sources += 1
            assert 'original_bad_height_pending_native_support_repair' in result['warning_codes']
            assert result['task_definition'] is not None
    assert (bindings, repaired_sources) == (253, 13)


def test_temporal_groups_include_only_containers_and_active_phase1_contents(tmp_path, monkeypatch):
    source = next(source for source in sources(legacy.CategoryConverter.category)
        if source.relative_path.endswith('Beechwood_0_garden/living_room_0/current_state_reasoning__q_001.json'))
    monkeypatch.setattr(legacy, '_robot_start', lambda layout, context: {
        'position': [4., 12., 0.], 'yaw_deg': 45., 'connected_proposals_xy': [[4., 12.]]})
    monkeypatch.setattr(legacy, '_route_screen', lambda layout, start: {'container': layout['container_keys'][0]})
    result = temporal.CategoryConverter().convert(source, ConversionContext(tmp_path))
    assert result['status'] == 'generated'
    policy = result['interaction_policy']
    p = result['task_definition']['private']['evaluation']['parameters']
    assert policy['mode'] == 'inspection' and policy['relocation_policy'] == 'independent_groups'
    assert policy['support_bindings'] == p['container_keys']
    assert policy['source_camera_policy'] == 'preserve'
    flattened = [key for group in policy['placement_groups'] for key in group]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(p['container_keys'] + p['phase_keys']['image1'])
    assert not set(flattened) & set(p['phase_keys']['image2'])
    for group in policy['placement_groups']:
        assert group[0] in p['container_keys']
        assert all(p['relations'][key] == group[0] for key in group[1:])
    assert p['accessible_support_rebase_required']
    assert 'Phase 1 is read-only' in result['task_definition']['public']['objective']
    assert result['scene_definition']['scene']['variants']['original']['patches'] == []


def test_continuous_temporal_drops_only_obsolete_two_view_screen(tmp_path, monkeypatch):
    source = next(source for source in sources(legacy.CategoryConverter.category)
        if source.relative_path.endswith('Beechwood_0_garden/living_room_0/current_state_reasoning__q_001.json'))
    monkeypatch.setattr(legacy, '_robot_start', lambda layout, context: {
        'position': [4., 12., 0.], 'yaw_deg': 45., 'connected_proposals_xy': [[4., 12.]]})
    monkeypatch.setattr(legacy, '_route_screen', lambda layout, start: None)
    context = ConversionContext(tmp_path, config={'unobserved_operation': 'box_inspection_route'})
    historical = legacy.CategoryConverter().convert(source, context)
    assert historical['failure_stage'] == 'inspection_route_screen'
    current = temporal.CategoryConverter().convert(source, context)
    assert current['status'] == 'generated'
    p = current['task_definition']['private']['evaluation']['parameters']
    assert p['phase_2_minimum_base_travel_m'] == .25
    assert p['phase_2_container_visit_radius_m'] == 1.


def test_continuous_temporal_no_map_start_emits_no_guessed_robot_pose(tmp_path, monkeypatch):
    source = sources(legacy.CategoryConverter.category)[0]
    monkeypatch.setattr(legacy, '_robot_start', lambda layout, context: None)
    historical = legacy.CategoryConverter().convert(source, ConversionContext(tmp_path))
    assert historical['failure_stage'] == 'robot_start_screen'
    current = temporal.CategoryConverter().convert(source, ConversionContext(tmp_path))
    assert current['status'] == 'generated'
    assert current['robot_initialization_evidence']['pending_shared_spawn_repair'] is True
    assert 'source_map_start_pending_shared_native_repair' in current['warning_codes']
    patches = current['task_definition']['scene']['variants']['adapted']['patches']
    assert not any(patch['operation'] == 'robot_pose' for patch in patches)
    assert current['interaction_policy']['source_region'] == source.raw['room']


@pytest.mark.parametrize('fault', ['identity', 'nonfinite', 'qa_conflict'])
def test_invalid_observation_sources_still_fail_instead_of_height_repair(tmp_path, fault):
    source = sources(patrol.CATEGORY)[0]
    payload, raw = deepcopy(source.payload), deepcopy(source.raw)
    if fault == 'identity':
        payload['objects_meta'][0]['model'] = None
    elif fault == 'nonfinite':
        payload['objects_meta'][0]['position'][2] = float('nan')
    else:
        raw['answer'] = 'invalid original count annotation'
    fixture = NS(raw=raw, payload=payload, big_task=source.big_task, small_task=source.small_task,
        path=source.path, relative_path=source.relative_path)
    assert not patrol.CategoryConverter().inspect(fixture)['eligible']
