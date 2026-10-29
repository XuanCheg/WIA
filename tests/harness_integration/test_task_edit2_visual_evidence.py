"""Validate real collector synchronization/audit gates with native-shaped buffers.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider \
tests/test_task_edit2_visual_evidence.py
Fixtures supply sensor buffers and explicit private reviews. No native rendering,
semantic model, GPU workload or end-to-end task success is claimed.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import threading

import numpy as np
import pytest

from wia.runtime.sim.esi_visual_evidence import (
    collect_native_visual_evidence, install_visual_evidence,
    FEATURE_AUDIT_SCHEMA, LOCATION_AUDIT_SCHEMA, FROZEN_INPUT_AUDIT_SCHEMA,
)
from wia.tasks.loader import LoadedTask, LoadedScene
from wia.tasks.schema import Source
from wia.tasks.temporal_policy import informative_temporal_rows
from core.observations import InitialOverview


class NativeObject:
    def __init__(self, name, *, prim_path=None):
        self.name, self.prim_path = name, prim_path or '/World/' + name
        self.category, self.model, self.scale = 'container', 'source_model', np.ones(3)
        self.position, self.quaternion = np.array([1., 0., .6]), np.array([0., 0., 0., 1.])

    def get_position_orientation(self, **kwargs):
        return self.position.copy(), self.quaternion.copy()

    def set_position_orientation(self, *args, **kwargs):
        pytest.fail('A visual evidence collector must not set poses.')


class NativeSensor(NativeObject):
    def __init__(self, rgb, segmentation, labels):
        super().__init__('head_native', prim_path='/World/robot/head_camera')
        self.data = {'rgb': rgb, 'seg_instance': segmentation}
        self.info = {'seg_instance': labels}
        self.reads = 0

    def get_obs(self):
        self.reads += 1
        return self.data, self.info

    def add_modality(self, *args):
        pytest.fail('A visual evidence collector must not enable modalities.')


@pytest.fixture
def world():
    rgb = np.random.default_rng(18).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    segmentation = np.zeros((32, 48), dtype=np.int32)
    segmentation[6:18, 8:20] = 7
    segmentation[2:30, 29:44] = 9
    target, mirror = NativeObject('target'), NativeObject('mirror')
    sensor = NativeSensor(rgb.copy(), segmentation, {7: 'target', 9: '/World/mirror/base_link/visuals'})
    robot = NativeObject('robot')
    robot.sensors = {'head_native': sensor}
    objects = {'target': target, 'mirror': mirror}
    source = Source.model_validate({'provider': 'esi', 'json': 'Specular Reflection/Correspondence/scene/room/sample.json'})
    loaded = LoadedTask(definition=NS(id='exact_task', source=source), path=Path('/exact_task.yaml'),
        source_path=Path('/immutable_source.json'), file_sha256='task_sha', source_sha256='source_sha',
        variant='adapted', seed=17, scene_name='scene', room_name='room')
    loaded_scene = LoadedScene(definition=NS(id='exact_scene', source=source), path=Path('/exact_scene.yaml'),
        source_path=Path('/immutable_source.json'), file_sha256='scene_sha', source_sha256='source_sha',
        variant='adapted', seed=17, scene_name='scene', room_name='room')
    observation = NS(snapshot_id=11, action_step=3, captured_at='2026-10-18T06:00:00.000000+00:00',
        action_history=(NS(action='BASE_LEFT', result_summary='completed'),))
    images = {'agentview_rgb': rgb.copy()}
    observation.rgb_images = lambda: images
    session = NS(robot=robot, native_reset_id='reset_17', episode_id='episode_17',
        loaded_task=loaded, scene_adapter=NS(task=loaded_scene),
        env=NS(scene=NS(object_registry=lambda kind, name: objects.get(name),
            seg_map=NS(get_room_instance_by_point=lambda xy: 'actually_measured_room'))),
        pipeline=NS(og=NS(sim=NS(current_time=4.5))), robot_camera_registry={'agentview_rgb': 'head_native'},
        _observation_source=NS(latest_snapshot=NS(observation=observation)))
    controller = NS(geometric_rearrangement_evidence=lambda: {'available': True, 'sample_id': 4500000000,
        'control_sequence': 3, 'control_active': False, 'controlled_object_names': [],
        'grasped_object_names': [], 'contact_object_names': [], 'state_setter_used': False})
    return NS(session=session, observation=observation, controller=controller, target=target, mirror=mirror,
        sensor=sensor, images=images, bindings={'target': 'target', 'mirror': 'mirror'})


def collect(world, **kwargs):
    return collect_native_visual_evidence(world.session, world.observation, world.bindings,
        {'source_sha256': 'source_sha', 'expected_room': 'wrong_expected_room'},
        controller=world.controller, **kwargs)


def feature_audit(measured, *, key='target', support=None, reflected=False):
    support = support or key
    row = measured['measurements'][key]['views']['agentview_rgb']
    support_row = measured['measurements'][support]['views']['agentview_rgb']
    features = {'feature_ids': ['distinctive_rim_notch'], 'semantic_identity_verified': True,
        'view_kind': 'reflected' if reflected else 'direct',
        'reference_snapshot_id': 11, 'reference_frame': 'initial_original_ESI_observer'}
    if reflected:
        features.update(mirror_binding=support, real_target_correspondence_verified=True,
            same_object_correspondence_verified=True)
    return {'schema': FEATURE_AUDIT_SCHEMA, 'audit_id': 'explicit_feature_review_1',
        'binding': key, 'camera': 'agentview_rgb', 'context': deepcopy(row['audit_context']),
        'support_binding': support, 'support_identity': deepcopy(measured['measurements'][support]['native_identity']),
        'support_mask_sha256': support_row['mask_sha256'], 'informative': True,
        'informative_feature_evidence': features, 'feature_crop_bbox_xyxy': support_row['bbox_xyxy'],
        'review': {'reviewer': 'synthetic_test_reviewer', 'reviewed_at': '2026-10-18T06:00:01Z',
            'method': 'manual_native_image_feature_review'}}


def test_masks_are_measured_without_inventing_recognition_or_expected_location(world):
    result = collect(world)
    assert result['measurement_available'] and not result['informative_available'] and not result['available']
    row = result['objects']['target']
    assert row['visible_pixels'] == 144 and row['bbox_xyxy'] == [8, 6, 20, 18]
    assert row['bbox_normalized'] == [8 / 48, 6 / 32, 20 / 48, 18 / 32]
    assert row['native_rgb_matches_delivered'] and row['identity_synchronized']
    assert not row['informative'] and not row['visit_verified']
    assert row['informative_feature_evidence'] == {}
    assert result['measurements']['target']['measured_room_label'] == 'actually_measured_room'
    assert 'room' not in row and 'area' not in row
    assert not result['recognition_inferred_from_mask_size']
    assert world.sensor.reads == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('failure', ['different_RGB', 'missing_modality', 'different_dimensions', 'stale_frame', 'unmounted_camera'])
def test_mismatched_native_buffers_do_not_become_synchronized_measurement(world, failure):
    if failure == 'different_RGB':
        world.images['agentview_rgb'][0, 0, 0] ^= 1
    elif failure == 'missing_modality':
        del world.sensor.data['seg_instance']
    elif failure == 'different_dimensions':
        world.sensor.data['seg_instance'] = np.zeros((31, 48), dtype=np.int32)
    elif failure == 'stale_frame':
        world.sensor.info['capture_metadata'] = {'rgb': {'frame_id': 17}, 'seg_instance': {'frame_id': 16}}
    else:
        world.sensor.prim_path = '/World/detached_camera'
    result = collect(world)
    assert not result['measurement_available'] and not result['available']
    assert 'agentview_rgb' in result['measurement_errors']


def test_misleading_tiny_mask_never_becomes_informative_from_a_pixel_gate(world):
    world.sensor.data['seg_instance'][:] = 0
    world.sensor.data['seg_instance'][12, 18] = 7
    result = collect(world)
    assert result['objects']['target']['visible_pixels'] == 1
    assert result['objects']['target']['minimum_extent_px'] == 1
    assert not result['objects']['target']['informative']
    assert not result['available']


@pytest.mark.parametrize('labels,modality', [
    ({7: {'primPath': '/World/target/base_link/visuals'}, 9: 'mirror'}, 'seg_instance'),
    ({'7': '/World/target/base_link/mesh', '9': {'object_name': 'mirror'}}, 'seg_instance_id')])
def test_object_and_prim_instance_labels_resolve_exact_native_identity(world, labels, modality):
    segmentation = world.sensor.data.pop('seg_instance')
    world.sensor.data[modality] = segmentation
    world.sensor.info = {modality: labels}
    assert collect(world)['objects']['target']['visible_pixels'] == 144


def test_semantic_category_labels_and_prim_prefix_collisions_do_not_identify_instances(world):
    world.sensor.info['seg_instance'] = {7: {'class': 'container'}, 9: '/World/target_impostor/base_link'}
    result = collect(world)
    assert result['objects']['target']['visible_pixels'] == 0
    assert not result['objects']['target']['identity_synchronized']


@pytest.mark.parametrize('field', ['rgb_sha256', 'source_sha256', 'reset_id', 'snapshot_id',
    'sensor_prim_path', 'native_identity', 'native_pose'])
def test_stale_or_mismatched_audit_never_confers_semantic_information(world, field):
    measured = collect(world)
    receipt = feature_audit(measured)
    receipt['context'][field] = 'mismatched_or_stale'
    result = collect(world, audited_receipts=[receipt])
    assert result['measurement_available'] and not result['available']
    assert not result['objects']['target']['informative']
    assert 'mismatch' in result['objects']['target']['information_unavailable_reason']


def test_exact_audited_feature_receipt_enables_information_but_not_an_unaudited_visit(world):
    measured = collect(world)
    receipt = feature_audit(measured)
    result = collect(world, audited_receipts=[receipt])
    assert result['available'] and result['informative_available'] and result['measurement_available']
    row = result['objects']['target']
    assert row['informative'] and row['feature_audit_id'] == receipt['audit_id']
    assert row['informative_feature_evidence']['feature_ids'] == ['distinctive_rim_notch']
    assert not row['visit_verified']
    assert result['native_public_control_context']['control_sequence'] == 3
    assert result['public_action_history'] == [{'action': 'BASE_LEFT', 'result_summary': 'completed'}]
    assert world.observation.action_history[0].result_summary == 'completed'


def test_real_temporal_consumer_requires_a_separately_verified_exact_location_visit(world):
    measured = collect(world)
    receipt = feature_audit(measured)
    world.session.visual_evidence_feature_receipts = [receipt]
    install_visual_evidence(world.session, world.controller)
    result = informative_temporal_rows(world.session, world.observation, world.bindings, {'source_sha256': 'source_sha'})
    assert result['available'] and result['objects'] == {}
    visit = {'schema': LOCATION_AUDIT_SCHEMA, 'audit_id': 'native_location_review_1',
        'context': deepcopy(receipt['context']), 'native_location_verified': True,
        'legal_observation_pose': True, 'location': {'room': 'actually_measured_room', 'area': 'verified_table'},
        'review': deepcopy(receipt['review'])}
    world.session.visual_evidence_verified_locations = [visit]
    result = informative_temporal_rows(world.session, world.observation, world.bindings, {'source_sha256': 'source_sha'})
    assert set(result['objects']) == {'target'}
    row = result['objects']['target']
    assert row['visit_verified'] and row['visit_evidence']['legal_observation_pose']
    assert row['room'] == 'actually_measured_room' and row['area'] == 'verified_table'
    visit['context']['native_pose']['position_m'][0] += .1
    assert informative_temporal_rows(world.session, world.observation, world.bindings, {})['objects'] == {}


def test_reflected_real_target_is_not_identified_from_mirror_segmentation_alone(world):
    world.sensor.data['seg_instance'][world.sensor.data['seg_instance'] == 7] = 0
    measured = collect(world)
    assert measured['objects']['target']['visible_pixels'] == 0
    assert measured['objects']['mirror']['visible_pixels'] > 0
    assert not measured['objects']['target']['informative']
    receipt = feature_audit(measured, support='mirror', reflected=True)
    receipt['informative_feature_evidence'].pop('real_target_correspondence_verified')
    result = collect(world, audited_receipts=[receipt])
    assert not result['objects']['target']['informative']
    receipt['informative_feature_evidence']['real_target_correspondence_verified'] = True
    result = collect(world, audited_receipts=[receipt])
    row = result['objects']['target']
    assert row['informative'] and row['identity_synchronized']
    assert row['identity_evidence_source'] == 'explicit_audited_reflection_correspondence'
    assert row['visible_pixels'] == 0  # No fictitious directly segmented target.
    assert row['informative_feature_evidence']['same_object_correspondence_verified']


def test_frozen_global_cannot_become_a_live_encounter_even_if_registered(world):
    world.images['initial_overview_rgb'] = world.images.pop('agentview_rgb')
    world.session.robot_camera_registry = {'initial_overview_rgb': 'head_native'}
    result = collect(world)
    assert world.sensor.reads == 0
    assert not result['available'] and not result['measurement_available']
    assert not result['frozen_views']['initial_overview_rgb']['live']
    assert result['view_evidence'] == []


def test_current_retained_boundary_is_required_even_for_identical_RGB(world):
    measured = collect(world)
    receipt = feature_audit(measured)
    world.session._observation_source.latest_snapshot.observation = NS(snapshot_id=12, action_step=4,
        captured_at=world.observation.captured_at)
    result = collect(world, audited_receipts=[receipt])
    assert result['measurement_available'] and not result['available']
    assert not result['objects']['target']['identity_synchronized']


def test_task_and_scene_hashes_have_distinct_native_roles(world):
    result = collect(world)
    context = result['native_capture_context']
    assert context['task_id'] == 'exact_task' and context['task_sha256'] == 'task_sha'
    assert context['scene_id'] == 'exact_scene' and context['scene_sha256'] == 'scene_sha'
    assert context['provenance_role'] == 'task_and_scene'
    del world.session.loaded_task
    result = collect(world)
    context = result['native_capture_context']
    assert context['task_id'] is None and context['task_sha256'] is None
    assert context['scene_id'] == 'exact_scene' and context['scene_sha256'] == 'scene_sha'
    assert context['provenance_role'] == 'scene_only'
    assert result['measurement_available'] and not result['available']


def test_exact_frozen_input_review_audits_initial_shortcuts_without_live_object_rows(world):
    world.images['initial_overview_rgb'] = world.images['agentview_rgb'].copy()
    world.observation.initial_overview = InitialOverview(rgb=world.images['initial_overview_rgb'],
        episode_id='episode_17', reset_id='reset_17', seed=17, snapshot_id=11,
        captured_at=world.observation.captured_at, extent_xy_m=(-2., -2., 2., 2.),
        image_right_world_xy=(1., 0.), image_up_world_xy=(0., 1.),
        robot_start_position_m=(0., 0., 0.), robot_start_yaw_rad=0.)
    measured = collect(world)
    feature = feature_audit(measured)
    frozen = {'schema': FROZEN_INPUT_AUDIT_SCHEMA, 'audit_id': 'reviewed_frozen_initial_image',
        'context': deepcopy(measured['frozen_views']['initial_overview_rgb']['audit_context']),
        'review': deepcopy(feature['review']), 'initial_input_reviewed': True,
        'scope': 'frozen_initial_input_shortcut_audit_only',
        'findings': ['An initial direct-view shortcut was reviewed in the original frozen RGB.']}
    result = collect(world, audited_receipts=[feature], frozen_input_receipts=[frozen])
    assert result['all_delivered_views_audited']
    assert result['frozen_views']['initial_overview_rgb']['initial_input_reviewed']
    assert all(row['camera'] != 'initial_overview_rgb' for row in result['view_evidence'])
    assert result['objects']['target']['camera'] == 'agentview_rgb'
    world.observation.snapshot_id, world.observation.action_step = 12, 4
    world.observation.captured_at = '2026-10-18T06:00:02.000000+00:00'
    result = collect(world, audited_receipts=[feature], frozen_input_receipts=[frozen])
    assert result['frozen_views']['initial_overview_rgb']['initial_input_reviewed']
    assert not result['available']  # The old live feature audit is stale.
    assert not result['objects']['target']['informative']
    frozen['context']['freeze_reset_id'] = 'other_reset'
    assert not collect(world, frozen_input_receipts=[frozen])['frozen_views']['initial_overview_rgb']['initial_input_reviewed']


def test_crop_artifacts_are_explicit_bounded_and_absent_from_JSON_payload(world, tmp_path):
    result = collect(world)
    assert 'crop_artifacts' not in result['objects']['target']
    output = tmp_path / 'explicit_review_crops'
    result = collect(world, output_dir=output, max_crops=1, maximum_crop_edge=8)
    assert result['crop_count'] == 1 and len(list(output.glob('*.png'))) == 2
    artifacts = result['objects']['target']['crop_artifacts']
    assert Path(artifacts['rgb_crop_path']).is_file()
    assert max(artifacts['saved_size']) <= 8
    serialized = json.dumps(result, allow_nan=False)
    assert 'data:image' not in serialized and 'base64' not in serialized


def test_owner_thread_is_checked_before_any_native_sensor_read(world):
    errors = []
    def other_thread():
        try:
            collect(world)
        except RuntimeError as error:
            errors.append(str(error))
    worker = threading.Thread(target=other_thread)
    worker.start()
    worker.join()
    assert errors and 'owner/main thread' in errors[0]
    assert world.sensor.reads == 0
