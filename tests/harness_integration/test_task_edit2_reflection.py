"""Check reflection verification variants with explicit synthetic native receipts.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider \
tests/test_task_edit2_reflection.py tests/test_core_revision_reflection_mapping.py \
tests/test_annotation_reflection_temporal.py
Synthetic source-image/feature fixtures exercise gates, not actual ESI equivalence.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
from PIL import Image
import pytest

from wia.authoring.conversion import ConversionContext
from wia.authoring.converters import specular_reflection_correspondence as correspondence
from wia.authoring.converters import specular_reflection_spatial_relations as spatial
from wia.authoring.converters import specular_reflection_reflection_authoring as authoring
import wia.runtime.sim.esi_reflection as native
from wia.tasks.loader import task_definition, LoadedTask, LoadedScene
from wia.tasks.schema import SceneDefinition
from harness_integration.test_annotation_manipulation_categories import source_for

OPERATIONS = [(correspondence, 'correspondence_operation'),
    (spatial, 'spatial_relations_operation'), (authoring, 'reflection_authoring_operation')]


@pytest.mark.parametrize('module,key', OPERATIONS)
@pytest.mark.parametrize('mode', ['bounded', 'unrestricted'])
def test_action_variants_preserve_optics_and_source_question_with_distinct_boundary_versions(module, key, mode, tmp_path):
    converter, source = source_for(module)
    result = converter.convert(source, ConversionContext(tmp_path, config={
        key: module.ACTION_OPERATION, 'reflection_boundary_mode': mode}))
    task = result['task_definition']
    task_definition(task)
    p = task['private']['evaluation']['parameters']
    assert task['private']['evaluation']['operation'] == module.ACTION_OPERATION
    assert p['task_revision'] == authoring.ACTION_REVISION
    assert p['source_qa']['answer'] == source.raw['answer']
    assert p['verification_contract']['answer_before_viewpoint_transition']
    assert p['reflection_sensor_profile']['robot_uniform_scale'] is None
    assert p['reflection_sensor_profile']['boundary']['limit_m'] == (1. if mode == 'bounded' else None)
    assert p['input_condition']['initial_global_rgb'] == 'mandatory_frozen_reset_world_image'
    assert 'arbitrary' in task['public']['success_criteria']
    assert result['interaction_policy']['relocation_policy'] == 'preserve'
    assert result['shortcut_audit']['global_and_other_mounted_direct_views_audited'] is False
    SceneDefinition.model_validate(result['scene_definition'])
    hooks = []
    for definition in (task, result['scene_definition']):
        assert definition['scene']['variants']['original']['patches'] == []
        patches = definition['scene']['variants']['adapted']['patches']
        assert len(patches) == 1 and patches[0]['operation'] == 'category_initialization'
        hook = patches[0]
        assert hook['phase'] == 'after_postprocessing' and hook['module'] == authoring.MODULE
        assert hook['parameters']['operation'] == native.INITIALIZATION_OPERATION
        assert hook['parameters']['initialization_revision'] == native.INITIALIZATION_REVISION
        assert hook['parameters']['reflection_sensor_profile'] == p['reflection_sensor_profile']
        assert hook['parameters']['source_identity'] == source.relative_path
        assert hook['parameters']['source_sha256'] == hashlib.sha256(source.path.read_bytes()).hexdigest()
        hooks.append(hook)
    assert hooks[0] == hooks[1]
    assert p['reflection_sensor_profile']['version'] == native.ACTION_PROFILE_VERSION
    old = converter.convert(source, ConversionContext(tmp_path))
    assert old['task_definition']['private']['evaluation']['operation'] == module.ANSWER_OPERATION
    assert old['task_definition']['scene']['variants'] == {'original': {'patches': []}}
    assert old['scene_definition']['scene']['variants'] == {'original': {'patches': []}}


def initialization_world(tmp_path):
    converter, source = source_for(authoring)
    result = converter.convert(source, ConversionContext(tmp_path, config={
        'reflection_authoring_operation': authoring.ACTION_OPERATION}))
    task, scene = result['task_definition'], result['scene_definition']
    common = {'path': tmp_path / 'definition.yaml', 'source_path': source.path,
        'file_sha256': 'definition-sha', 'source_sha256': hashlib.sha256(source.path.read_bytes()).hexdigest(),
        'variant': 'adapted', 'seed': 3, 'scene_name': source.raw['scene'], 'room_name': source.raw['room']}
    # Accessing observation_source would create a public observation pipeline.
    class Session(NS):
        @property
        def observation_source(self):
            raise AssertionError('Initializer must not access observation_source.')
    session = Session(loaded_task=LoadedTask(definition=task_definition(task), **common),
        scene_adapter=NS(task=LoadedScene(definition=SceneDefinition.model_validate(scene), **common)),
        scene_name=source.raw['scene'], room_name=source.raw['room'], native_reset_id='native-reset',
        _category_initialization_active=True, _observation_source=None,
        reflection_sensor_profile=deepcopy(task['private']['evaluation']['parameters']['reflection_sensor_profile']),
        reflection_qualification={key: False for key in authoring.REFLECTION_QUALIFICATION_KEYS})
    parameters = deepcopy(task['scene']['variants']['adapted']['patches'][0]['parameters'])
    return session, parameters


def test_explicit_scene_initialization_runs_once_and_keeps_qualification_pending(tmp_path, monkeypatch):
    session, parameters = initialization_world(tmp_path)
    real_align = native.align_initial_robot
    calls = []
    actual_receipt = {'native_torso_alignment': {'control_trace': [{'joints_rad': [0., 0., -.2]}]},
        'head_mount': [[0., 0., 0.], [0., 0., 0., 1.]]}
    def align(subject):
        assert subject._reflection_initialization_hook_active
        assert subject.reflection_alignment_initialization['completed'] is False
        calls.append(subject)
        subject.reflection_qualification['initialization'] = deepcopy(actual_receipt)
        return actual_receipt
    monkeypatch.setattr(native, 'align_initial_robot', align)
    profile = deepcopy(session.reflection_sensor_profile)
    receipt = authoring.initialize_scene(session, parameters)
    assert calls == [session] and receipt['completed'] is True
    assert receipt['reset_id'] == session.native_reset_id
    assert receipt['profile_sha256'] == native.reflection_profile_sha256(profile)
    assert receipt['alignment_receipt'] == actual_receipt
    assert not receipt['physical_qualification_complete']
    assert not receipt['source_pose_optics_floor_and_image_equivalence_verified']
    assert all(session.reflection_qualification[key] is False for key in authoring.REFLECTION_QUALIFICATION_KEYS)
    assert session.reflection_sensor_profile == profile and session._observation_source is None
    assert 'initial_head_pose_measurement' not in session.reflection_qualification
    with pytest.raises(RuntimeError, match='cannot run twice'):
        authoring.initialize_scene(session, parameters)
    # A later shared interaction route returns the existing exact receipt and
    # performs no second native alignment or receipt replacement.
    initial = session.reflection_qualification['initialization']
    assert real_align(session) is initial
    assert session.reflection_qualification['initialization'] is initial


@pytest.mark.parametrize('failure', ['missing_profile', 'different_profile', 'different_loaded_profile',
    'different_source', 'missing_active_phase', 'observation_source', 'episode', 'public_action'])
def test_scene_initialization_rejects_missing_mismatched_or_late_context(failure, tmp_path, monkeypatch):
    session, parameters = initialization_world(tmp_path)
    calls = []
    monkeypatch.setattr(native, 'align_initial_robot', lambda subject: calls.append(subject))
    if failure == 'missing_profile': del session.reflection_sensor_profile
    elif failure == 'different_profile': parameters['reflection_sensor_profile']['horizontal_fov_deg'] += 1.
    elif failure == 'different_loaded_profile':
        session.loaded_task.definition.private.evaluation.parameters['reflection_sensor_profile']['horizontal_fov_deg'] += 1.
    elif failure == 'different_source': parameters['source_sha256'] = 'different-source'
    elif failure == 'missing_active_phase': session._category_initialization_active = False
    elif failure == 'observation_source': session._observation_source = NS()
    elif failure == 'episode': session.episode_id = 'public-episode'
    else: session.controller = NS(_public_action_sequence=1)
    with pytest.raises((RuntimeError, ValueError)):
        authoring.initialize_scene(session, parameters)
    assert calls == [] and 'reflection_alignment_initialization' not in vars(session)


def test_failed_real_alignment_never_gets_completed_receipt(tmp_path, monkeypatch):
    session, parameters = initialization_world(tmp_path)
    def align(subject):
        raise RuntimeError('native torso alignment failed')
    monkeypatch.setattr(native, 'align_initial_robot', align)
    with pytest.raises(RuntimeError, match='native torso alignment failed'):
        authoring.initialize_scene(session, parameters)
    assert session.reflection_alignment_initialization['completed'] is False
    assert not session._reflection_initialization_hook_active


def reflection_measurement_world(monkeypatch, *, revised=True):
    from wia.tasks import native_floor_contacts
    from wia.tasks import interaction_setup
    names = ['head', *['mounted_' + str(index) for index in range(7)]]
    sensors = {}
    for name in names:
        sensor = NS(prim_path='/World/robot/' + name, position=np.array([1., 2., 1.459]),
            quaternion=np.array([0., 0., 0., 1.]), horizontal_aperture=20., focal_length=10.,
            image_width=512, image_height=288, clipping_range=np.array([.01, 100.]))
        sensor.get_position_orientation = lambda sensor=sensor: (sensor.position.copy(), sensor.quaternion.copy())
        sensors[name] = sensor
    robot = NS(sensors=sensors, prim_path='/World/robot', scale=np.ones(3))
    profile = {'version': native.ACTION_PROFILE_VERSION if revised else 'reflection_sensor_profile_v2',
        'initialization_revision': native.INITIALIZATION_REVISION,
        'position_m': [1., 2., .8], 'quaternion_xyzw': [0., 0., 0., 1.],
        'horizontal_fov_deg': 90., 'robot_uniform_scale': None,
        'source_floor_reference': {'proposed_z_m': 0.}}
    session = NS(robot=robot, camera_name='head', reflection_sensor_profile=profile,
        reflection_qualification={key: False for key in authoring.REFLECTION_QUALIFICATION_KEYS},
        robot_camera_registry={name: name for name in names}, robot_camera_mounts={name: {} for name in names},
        env=NS(robots=[robot], scene=NS(objects=[NS(category='floors', prim_path='/World/floor')])),
        native_reset_id='reset', _observation_source=None)
    calls = {'new_floor': [], 'legacy_floor': []}
    def new_floor(*args, **kwargs):
        calls['new_floor'].append((args, kwargs))
        return {'measurement_available': True, 'floor_supported': True, 'collision_free': True}
    def legacy_floor(*args, **kwargs):
        calls['legacy_floor'].append((args, kwargs))
        return {'floor_supported': False, 'collision_free': True}
    monkeypatch.setattr(native_floor_contacts, 'measure_native_floor_contacts', new_floor)
    monkeypatch.setattr(interaction_setup, '_robot_contact_check', legacy_floor)
    monkeypatch.setattr(native, 'measure_optical_integrity', lambda session: {'accepted': False})
    controller = NS(session=session, robot=robot)
    return session, controller, calls


def test_precontroller_measurement_does_not_poison_final_initial_head_pose(monkeypatch):
    session, controller, calls = reflection_measurement_world(monkeypatch)
    before = native.measure_reflection(session)
    assert 'pre_alignment_head_pose_diagnostic' in before
    assert before['pre_alignment_head_pose_diagnostic']['position_error_m'] == pytest.approx(.659)
    assert 'initial_head_pose_measurement' not in before and before['source_head_pose_verified'] is False
    assert calls['new_floor'] and not calls['legacy_floor']
    session.reflection_alignment_initialization = {'completed': True, 'reset_id': session.native_reset_id,
        'profile_sha256': native.reflection_profile_sha256(session.reflection_sensor_profile)}
    native.measure_reflection(session)
    assert 'post_alignment_pre_final_controller_measurement' in session.reflection_qualification
    assert 'initial_head_pose_measurement' not in session.reflection_qualification
    session.robot.sensors['head'].position[2] = .8
    result = native.measure_reflection(session, controller=controller, finalize_initial=True)
    initial = result['initial_head_pose_measurement']
    assert initial['verified'] is True and initial['position_m'] == [1., 2., .8]
    assert initial['measurement_stage'] == 'after_alignment_final_controller_and_complete_mounted_rig'
    assert initial['finalization_identity']['controller_instance'] == str(id(controller))
    session.robot.sensors['head'].position[2] = 1.2
    native.measure_reflection(session)
    assert result['initial_head_pose_measurement'] is initial and initial['position_m'] == [1., 2., .8]
    assert result['optical_scene_preserved'] is False  # Alignment does not certify optics.


@pytest.mark.parametrize('failure', ['alignment', 'reset', 'profile', 'controller', 'rig', 'observation', 'action'])
def test_v3_final_measurement_requires_completed_same_identity_controller_and_rig(failure, monkeypatch):
    session, controller, _ = reflection_measurement_world(monkeypatch)
    session.reflection_alignment_initialization = {'completed': True, 'reset_id': session.native_reset_id,
        'profile_sha256': native.reflection_profile_sha256(session.reflection_sensor_profile)}
    if failure == 'alignment': session.reflection_alignment_initialization['completed'] = False
    elif failure == 'reset': session.reflection_alignment_initialization['reset_id'] = 'another-reset'
    elif failure == 'profile': session.reflection_sensor_profile['horizontal_fov_deg'] = 89.
    elif failure == 'controller': controller.robot = NS()
    elif failure == 'rig': session.robot_camera_mounts.pop('mounted_0')
    elif failure == 'observation': session._observation_source = NS(snapshot_id=1)
    else: controller._public_action_sequence = 1
    with pytest.raises(RuntimeError, match='Initial reflection pose requires'):
        native.measure_reflection(session, controller=controller, finalize_initial=True)
    assert 'initial_head_pose_measurement' not in session.reflection_qualification


def test_legacy_measurement_keeps_nominal_floor_path_and_initial_latch(monkeypatch):
    session, _, calls = reflection_measurement_world(monkeypatch, revised=False)
    result = native.measure_reflection(session)
    assert 'initial_head_pose_measurement' in result
    assert result['initial_head_pose_measurement']['position_error_m'] == pytest.approx(.659)
    assert calls['legacy_floor'] and not calls['new_floor']


def test_v3_default_scale_is_measured_but_remains_pending_selection(monkeypatch):
    session, _, _ = reflection_measurement_world(monkeypatch)
    result = native.measure_reflection(session)
    assert result['measured_uniform_scale'] == [1., 1., 1.]
    assert result['native_robot_scale_verified'] is False
    session.reflection_sensor_profile['robot_uniform_scale'] = .6
    session.robot.scale[:] = .6
    result = native.measure_reflection(session)
    assert result['native_robot_scale_verified'] is True
    assert result['source_head_pose_verified'] is False


def synthetic_world(tmp_path, monkeypatch, module=authoring, boundary='bounded'):
    converter, source = source_for(module)
    key = next(key for candidate, key in OPERATIONS if candidate is module)
    converted = converter.convert(source, ConversionContext(tmp_path, config={
        key: module.ACTION_OPERATION, 'reflection_boundary_mode': boundary}))
    task = converted['task_definition']
    config = deepcopy(task['private']['evaluation'])
    config.update(scene=source.raw['scene'], room=source.raw['room'],
        bindings={key: row['name'] for key, row in task['private']['bindings'].items()})
    p = config['parameters']
    objects = {}
    for key, pose in p['expected_optical_poses'].items():
        obj = NS(position=np.array(pose['position']), quaternion=np.array(pose['quaternion_xyzw']))
        obj.get_position_orientation = lambda obj=obj: (obj.position.copy(), obj.quaternion.copy())
        objects[config['bindings'][key]] = obj
    objects[config['bindings']['floor']] = NS()
    head = NS(position=np.array(p['reflection_sensor_profile']['position_m']),
        quaternion=np.array(p['reflection_sensor_profile']['quaternion_xyzw']))
    head.get_position_orientation = lambda: (head.position.copy(), head.quaternion.copy())
    rgb = np.random.default_rng(4).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    png = tmp_path / 'synthetic_reference.png'
    Image.fromarray(rgb).save(png)
    reference = {'kind': 'original_ESI_image', 'available': True, 'image_path': str(png),
        'image_sha256': hashlib.sha256(png.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(source.path.read_bytes()).hexdigest()}
    p['original_reference_provenance'] = reference
    session = NS(scene_name=config['scene'], room_name=config['room'], camera_name='head_native',
        robot=NS(sensors={'head_native': head}),
        env=NS(scene=NS(object_registry=lambda kind, name: objects.get(name))),
        reflection_qualification={key: True for key in authoring.REFLECTION_QUALIFICATION_KEYS})
    session.identity = {'task_id': task['id'], 'source_sha256': reference['source_sha256'],
        'task_sha256': 'synthetic_task_hash', 'reset_id': 'synthetic_reset', 'seed': 0,
        'controller_instance': 'synthetic_controller', 'runtime_sha256': 'unchanged_sensors_and_optics'}
    session.motion = {'time_s': 0., 'maximum_sample_gap_s': .02 + 1e-6,
        'control_sequence': 0, 'state_setter_used': False, 'legal_viewpoint': True}
    monkeypatch.setattr(native, 'reflection_equivalence_identity', lambda *args: deepcopy(session.identity))
    monkeypatch.setattr(authoring, '_native_viewpoint_state', lambda *args: deepcopy(session.motion))
    features = {'available': True, 'all_delivered_views_audited': True, 'objects': {}}
    for binding in p['verification_contract']['required_bindings']:
        features['objects'][binding] = {'snapshot_id': 's0', 'camera': 'agentview_rgb',
            'native_name': config['bindings'][binding], 'native_rgb_matches_delivered': True,
            'identity_synchronized': True, 'informative': True,
            'informative_feature_evidence': {'view_kind': 'reflected', 'feature_ids': ['initial_front'],
                'mirror_binding': 'mirror', p['verification_contract']['mechanism']: True,
                'reference_snapshot_id': 's0', 'reference_frame': 'initial_original_ESI_observer'}}
    session.task_visual_evidence = lambda *args: deepcopy(features)
    session.reflection_qualification['original_source_view_equivalence'] = {
        'available': True, 'accepted': True, 'reference': reference, 'identity': deepcopy(session.identity),
        'snapshot_id': 's0', 'action_step': 0, 'camera': 'agentview_rgb',
        'mounted_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), 'native_rgb_matches_delivered': True,
        'task_feature_matches': [{'binding': binding, 'accepted': True} for binding in features['objects']]}
    observation = NS(snapshot_id='s0', action_step=0, action_history=(),
        rgb_images=lambda: {'agentview_rgb': rgb, 'initial_overview_rgb': rgb})
    evaluator = module.create_evaluator(session, NS(), config, tmp_path)
    result = evaluator.begin_episode(observation)
    assert result['evaluation_validity'] == 'complete', result
    return NS(session=session, evaluator=evaluator, observation=observation, features=features,
        head=head, parameters=p, config=config, objects=objects, result=result)


def move(world, *, displacement=.1, new_features=True, completed=True, camera='agentview_rgb'):
    world.session.motion['time_s'] += .02
    world.session.motion['control_sequence'] += 1
    world.head.position[0] += displacement
    world.evaluator.sample_step()
    world.observation.snapshot_id = 's' + str(world.observation.action_step + 1)
    world.observation.action_step += 1
    world.observation.action_history = (NS(action='BASE_LEFT', result_summary='completed' if completed else 'failed'),)
    for row in world.features['objects'].values():
        row['snapshot_id'] = world.observation.snapshot_id
        row['camera'] = camera
        if new_features:
            row['informative_feature_evidence']['feature_ids'] = ['new_side_feature']


@pytest.mark.parametrize('module', [correspondence, spatial, authoring])
def test_answer_plus_actual_transition_and_same_bound_object_features_are_required(module, tmp_path, monkeypatch):
    world = synthetic_world(tmp_path, monkeypatch, module)
    world.evaluator.report_answer(world.parameters['source_qa']['answer'], world.observation)
    assert not world.evaluator.record(world.observation)['task_success']
    move(world)
    result = world.evaluator.record(world.observation)
    assert result['answer_correct'] and result['physical_verification'] and result['task_success']
    assert result['completion_score'] == 1.
    assert result['Safety'] is None
    assert result['maximum_head_horizontal_displacement_m'] == pytest.approx(.1)


@pytest.mark.parametrize('failure', ['no_new_features', 'wrong_identity', 'stale_image', 'wrong_reference',
    'failed_control', 'frozen_global', 'no_translation', 'missing_features', 'setter'])
def test_motion_or_claimed_features_alone_do_not_verify(failure, tmp_path, monkeypatch):
    world = synthetic_world(tmp_path, monkeypatch)
    world.evaluator.report_answer(world.parameters['source_qa']['answer'], world.observation)
    move(world, new_features=failure != 'no_new_features', completed=failure != 'failed_control',
        camera='initial_overview_rgb' if failure == 'frozen_global' else 'agentview_rgb',
        displacement=0. if failure == 'no_translation' else .1)
    for row in world.features['objects'].values():
        if failure == 'wrong_identity':
            row['native_name'] = 'another_object'
        elif failure == 'stale_image':
            row['snapshot_id'] = 'stale'
        elif failure == 'wrong_reference':
            row['informative_feature_evidence']['reference_frame'] = 'current_head'
    if failure == 'missing_features':
        world.features['available'] = False
    if failure == 'setter':
        world.session.motion['state_setter_used'] = True
    result = world.evaluator.record(world.observation)
    assert not result['task_success'] and not result['physical_verification']
    if failure == 'missing_features':
        assert result['evaluation_validity'] == 'unavailable' and result['completion_score'] is None


def test_initial_answer_cannot_be_reframed_after_moving(tmp_path, monkeypatch):
    world = synthetic_world(tmp_path, monkeypatch)
    world.evaluator.report_answer('wrong', world.observation)
    move(world)
    world.evaluator.report_answer(world.parameters['source_qa']['answer'], world.observation)
    result = world.evaluator.record(world.observation)
    assert result['physical_verification'] and not result['answer_correct'] and not result['task_success']
    assert not result['answer_history'][-1]['initial_reference_answer_accepted']


@pytest.mark.parametrize('scale', [.25, 1., 2.])
def test_one_world_metre_native_head_limit_latches_after_return_independent_of_scale(scale, tmp_path, monkeypatch):
    world = synthetic_world(tmp_path, monkeypatch)
    world.parameters['reflection_sensor_profile']['robot_uniform_scale'] = scale
    world.evaluator.report_answer(world.parameters['source_qa']['answer'], world.observation)
    move(world, displacement=1.01)
    move(world, displacement=-1.01)
    result = world.evaluator.record(world.observation)
    assert not result['task_success']
    assert 'head_boundary_exceeded' in result['constraint_violations']
    assert result['maximum_head_horizontal_displacement_m'] == pytest.approx(1.01)


def test_unrestricted_viewpoint_has_no_excursion_or_return_obligation(tmp_path, monkeypatch):
    world = synthetic_world(tmp_path, monkeypatch, boundary='unrestricted')
    world.evaluator.report_answer(world.parameters['source_qa']['answer'], world.observation)
    move(world, displacement=1.2)
    assert world.evaluator.record(world.observation)['task_success']


def test_legal_torso_view_can_verify_without_horizontal_excursion(tmp_path, monkeypatch):
    world = synthetic_world(tmp_path, monkeypatch)
    world.evaluator.report_answer(world.parameters['source_qa']['answer'], world.observation)
    move(world, displacement=0.)
    world.session.motion['time_s'] += .02
    world.head.position[2] += .1
    world.observation.action_history = (NS(action='TRUNK_UP', result_summary='completed'),)
    result = world.evaluator.record(world.observation)
    assert result['task_success']
    assert result['maximum_head_horizontal_displacement_m'] == 0.


@pytest.mark.parametrize('change', ['reset', 'source_image', 'adapted_reference', 'sensor_configuration', 'head_pose'])
def test_equivalence_requires_original_pixels_and_exact_native_identity(change, tmp_path, monkeypatch):
    world = synthetic_world(tmp_path, monkeypatch)
    if change == 'reset':
        world.session.identity['reset_id'] = 'another_reset'
    elif change == 'source_image':
        Path(world.parameters['original_reference_provenance']['image_path']).write_bytes(b'changed pixels')
    elif change == 'adapted_reference':
        world.parameters['original_reference_provenance']['kind'] = 'adapted_postrobot_free_camera'
    elif change == 'head_pose':
        world.head.position[0] += .05
    else:
        world.session.identity['runtime_sha256'] = 'changed_camera_or_optical_material'
    result = world.evaluator.begin_episode(world.observation)
    assert result['evaluation_validity'] == 'unavailable'
    assert not result['task_success']


def test_missing_native_motion_and_clock_gaps_are_unavailable(tmp_path, monkeypatch):
    world = synthetic_world(tmp_path, monkeypatch)
    world.session.motion['time_s'] += .2
    result = world.evaluator.record(world.observation)
    assert result['evaluation_validity'] == 'unavailable'
    assert any('clock gap' in error for error in result['measurement_errors'])


def test_missing_executed_viewpoint_action_history_is_unavailable(tmp_path, monkeypatch):
    world = synthetic_world(tmp_path, monkeypatch)
    world.evaluator.report_answer(world.parameters['source_qa']['answer'], world.observation)
    move(world)
    world.observation.action_history = ()
    result = world.evaluator.record(world.observation)
    assert result['evaluation_validity'] == 'unavailable'
    assert result['completion_score'] is None


def test_normalized_comparison_requires_actual_task_features_and_effective_projection():
    rgb = np.random.default_rng(8).integers(0, 256, (192, 192, 3), dtype=np.uint8)
    intrinsics = {'image_size': [192, 192], 'matrix': [[160., 0., 96.], [0., 160., 96.], [0., 0., 1.]],
        'projection': 'perspective', 'clipping_range_m': [.01, 100.]}
    features = [{'binding': 'prop', 'feature_id': 'reflected_handle', 'view_kind': 'reflected', 'xy_normalized': [.4, .6]}]
    tolerances = {'minimum_pixel_matches': 8, 'maximum_hamming_distance': 0,
        'maximum_normalized_pixel_error': .01, 'maximum_normalized_task_feature_error': .01,
        'maximum_normalized_intrinsics_error': .001, 'maximum_clipping_error_m': .001}
    result = native.normalized_image_equivalence(rgb, rgb, intrinsics, intrinsics, features, features, tolerances=tolerances)
    assert result['accepted'], result
    larger = deepcopy(intrinsics)
    larger['image_size'] = [384, 384]
    larger['matrix'] = [[320., 0., 192.], [0., 320., 192.], [0., 0., 1.]]
    assert native.normalized_image_equivalence(rgb, np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1),
        intrinsics, larger, features, features, tolerances=tolerances)['accepted']
    wrong = deepcopy(intrinsics)
    wrong['matrix'][0][0] *= 2
    assert not native.normalized_image_equivalence(rgb, rgb, intrinsics, wrong, features, features, tolerances=tolerances)['accepted']
    assert not native.normalized_image_equivalence(rgb, rgb, intrinsics, intrinsics, features, [], tolerances=tolerances)['accepted']
    blank = np.zeros_like(rgb)
    assert not native.normalized_image_equivalence(rgb, blank, intrinsics, intrinsics, features, features, tolerances=tolerances)['accepted']


def test_global_shortcuts_are_reported_separately_from_second_view_verification(tmp_path, monkeypatch):
    world = synthetic_world(tmp_path, monkeypatch)
    baseline = world.evaluator.initial_visual
    baseline['audited_camera_names'] = ['agentview_rgb', 'initial_overview_rgb']
    binding, original = next(iter(baseline['objects'].items()))
    global_row = deepcopy(original)
    global_row.update(binding=binding, camera='initial_overview_rgb')
    global_row['informative_feature_evidence'].update(view_kind='direct', feature_ids=['global_direct_object'])
    baseline['view_evidence'] = [global_row]
    result = world.evaluator.record(world.observation)
    audit = result['initial_information_audit']
    assert audit['global_included'] and audit['global_direct_view_shortcut_observed']
    assert audit['all_views_shortcut_audit_complete']
    assert not result['physical_verification']
