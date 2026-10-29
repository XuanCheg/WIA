"""Check read-only native Mapping identities with typed initial Agent inputs.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_task_edit2_mapping_native_identity.py
Synthetic native objects exercise transport and identity checks, not scene truth.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.authoring.converters import task_edit2_mapping_world as world
from core.observations import AgentObservation, InitialOverview
from core.robot_cameras import ROBOT_CAMERA_LABELS, ADDITIONAL_RGB_NAMES
from wia.tasks.schema import Source
import wia.runtime.sim.esi_mapping_world as identity
import wia.runtime.sim.esi_native_qualification as qualification
from wia.tasks import native_floor_contacts


class Joint:
    def __init__(self):
        self.value = 0.

    def get_state(self):
        return np.array([self.value]), np.zeros(1)


class Actor:
    def __init__(self, name, category, position, extent):
        self.name, self.category, self.model = name, category, 'native_model'
        self.prim_path = '/World/' + name
        self.position = np.array(position, dtype=float)
        self.orientation = np.array([0., 0., 0., 1.])
        self.extent = np.array(extent, dtype=float)
        self.scale = np.ones(3)
        self.fixed_base, self.visual_only, self.collision_enabled = True, False, True
        self.in_rooms, self.joints, self.links = ['room_0'], {}, {}
        self.sensors, self.controllers = {}, {}

    @property
    def aabb(self):
        return self.position - self.extent / 2, self.position + self.extent / 2

    def get_position_orientation(self, **kwargs):
        return self.position.copy(), self.orientation.copy()

    def get_joint_positions(self):
        return np.zeros(2)

    def contact_list(self):
        return [NS(body0=self.prim_path + '/wheel', body1='/World/floor/collision',
                   normal=np.array([0., 0., 1.]), position=np.array([0., 0., .0055918]))]

    def set_position_orientation(self, **kwargs):
        raise AssertionError('Identity measurement must never mutate native objects.')


def observation():
    captured = '2026-10-18T12:00:00+00:00'
    rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    overview = InitialOverview(rgb=rgb, episode_id='episode', reset_id='reset', seed=0,
        snapshot_id=1, captured_at=captured, extent_xy_m=(-5., -5., 5., 5.),
        image_right_world_xy=(1., 0.), image_up_world_xy=(0., 1.),
        robot_start_position_m=(0., 0., .5), robot_start_yaw_rad=0.)
    return AgentObservation(task_text='Inspect the original room relation.',
        agentview_rgb=rgb, wrist_rgb=rgb + 1,
        additional_rgb={key: rgb + index + 2 for index, key in enumerate(ADDITIONAL_RGB_NAMES)},
        action_history=(), snapshot_id=1, captured_at=captured, action_step=0, action_boundary=True,
        initial_overview=overview, initial_overview_status='available')


@pytest.fixture
def native(tmp_path, monkeypatch):
    source_root = tmp_path / 'data'
    hierarchy = Path('Cognitive Mapping/Connectivity/native_scene/representative_room')
    (source_root / hierarchy).mkdir(parents=True)
    sources, source_paths = [], []
    for index in range(2):
        path = source_root / hierarchy / f'q{index}.json'
        path.write_text('{"scene":"native_scene","room":"representative_room","private_source":"original"}')
        source_paths.append(path)
        sources.append({'source_identity': str(hierarchy / f'q{index}.json'),
                        'source_sha256': identity.digest_file(path)})
    contract = {'revision': world.REVISION, 'scene': 'native_scene', 'room': 'full_scene',
        'full_building_required': True,
        'source_members': deepcopy(sources), 'representative_source': sources[0]['source_identity'], 'seed': 0,
        'episode_budget': {'max_steps': 100, 'wall_timeout_s': 300}}
    members = [{**source, 'bindings': {'anchor': {'name': 'chair', 'category': 'chair'}},
        'evaluation': {'parameters': {'source_door_policy': {
            'source_policy': 'remove_doors', 'runtime_policy': 'retain_native_doors'}}}}
        for source in sources]
    config = {'parameters': {'members': members, 'mapping_joint_world_contract': contract}}
    task_path = tmp_path / 'task.yaml'
    task_path.write_text('version: 1\n')
    source_path = source_paths[0]
    task = NS(path=task_path, source_path=source_path, file_sha256=identity.digest_file(task_path),
        source_sha256=identity.digest_file(source_path), seed=0, variant='adapted',
        scene_name='native_scene', room_name='representative_room', definition=NS(id='mapping_parent',
            source=Source.model_validate({'provider': 'esi', 'json': sources[0]['source_identity']}),
            private=NS(evaluation=NS(parameters=config['parameters']))))
    def unchanged():
        if (identity.digest_file(task.path) != task.file_sha256
                or identity.digest_file(task.source_path) != task.source_sha256):
            raise ValueError('Loaded task or source changed.')
    task.verify_unchanged = unchanged
    robot = Actor('robot', 'agent', [0, 0, .5], [.6, .6, 1.])
    robot.fixed_base = False
    robot.floor_touching_base_link_names = ('wheel',)
    robot.links = {'wheel': NS(prim_path=robot.prim_path + '/wheel')}
    monkeypatch.setattr(native_floor_contacts, 'native_vertical_ray', lambda *args, **kwargs: [
        {'hit': True, 'rigidBody': '/World/floor/collision',
         'position': [0., 0., .0055918], 'normal': [0., 0., 1.]}])
    robot.controllers = {'base': NS(dof_idx=np.array([0, 1]))}
    for index, name in enumerate(ROBOT_CAMERA_LABELS):
        robot.sensors['native:' + name] = NS(prim_path='/World/robot/camera_' + str(index),
            focal_length=9., horizontal_aperture=21., image_width=6, image_height=4,
            clipping_range=np.array([.001, 100.]), modalities={'rgb'},
            get_position_orientation=lambda **kwargs: (np.zeros(3), np.array([0., 0., 0., 1.])))
    floor = Actor('floor', 'floors', [0, 0, -.05], [10, 10, .1])
    wall = Actor('wall', 'walls', [5, 0, 1.5], [.2, 10, 3])
    chair = Actor('chair', 'chair', [2, 2, .5], [1, 1, 1])
    door = Actor('door', 'door', [4, 2, 1], [.1, 1, 2])
    door.joints['hinge'] = Joint()
    objects = {obj.name: obj for obj in [robot, floor, wall, chair, door]}
    scene = NS(objects=objects, scene_model='native_scene',
        object_registry=lambda kind, name: objects.get(name),
        trav_map=NS(floor_heights=[0.]), seg_map=NS(room_ins_name_to_ins_id={'room_0': 1}))
    scene_file = tmp_path / 'native_scene.json'
    scene_file.write_text('{"native_building":"preserved"}')
    scene.scene_file = scene_file
    def forbidden(*args, **kwargs):
        raise AssertionError('Identity collection cannot render, step or mutate simulation.')
    session = NS(robot=robot, env=NS(scene=scene, config={'scene': {'scene_model': 'native_scene'}}),
        scene_name='native_scene', room_name='representative_room', question_path=source_path, loaded_task=task,
        robot_camera_registry={name: 'native:' + name for name in ROBOT_CAMERA_LABELS},
        initial_overview_identity={'episode_id': 'episode', 'reset_id': 'reset', 'seed': 0},
        observation_source=NS(initial_observation=observation(), action_step=0),
        pipeline=NS(og=NS(sim=NS(step=forbidden, render=forbidden, dump_state=forbidden))))
    controller = NS(config={'translation_m': .25}, indices={'base': np.array([0, 1])},
        pose=lambda: {'position': robot.position.tolist(), 'quaternion_xyzw': robot.orientation.tolist()})
    control_file = tmp_path / 'controller.json'
    control_file.write_text('{"translation_m":0.25}')
    monkeypatch.setattr(qualification, 'CONTROLLER_FILES', (str(control_file),))
    provider = identity.install_mapping_world_identity(session, controller, task)
    return NS(session=session, controller=controller, task=task, config=config, contract=contract,
        provider=provider, robot=robot, floor=floor, wall=wall, chair=chair, door=door,
        source_root=source_root, source_paths=source_paths, control_file=control_file, scene_file=scene_file)


def test_complete_initial_measurements_are_deterministic_actual_and_read_only(native):
    before = {name: deepcopy(obj.get_position_orientation()) for name, obj in native.session.env.scene.objects.items()}
    first = native.provider(native.contract)
    assert first == native.provider(native.contract)
    assert not hasattr(native.session, 'mapping_joint_world_qualification')
    assert 'accepted' not in first and 'gates' not in first
    measured = first['measurements']
    assert measured['geometry']['loaded_rooms'] == ['room_0']
    assert len(measured['geometry']['object_aabbs']) == 4
    assert measured['geometry']['collision_enabled_categories'] == ['chair', 'door', 'floors', 'walls']
    assert measured['doors']['native_door_count'] == 1
    assert measured['doors']['source_policy_reconciliation']['status'] == 'pending'
    assert measured['controller']['public_control_trial_evidence'] is None
    assert measured['start']['support_and_clearance_evidence']['contact_readings']['floor_supported']
    contacts = measured['start']['support_and_clearance_evidence']['contact_readings']
    assert contacts['measurement_available'] and contacts['collision_free']
    assert contacts['contacts'][0]['nominal_floor_offset_m'] > .005
    assert abs(contacts['contacts'][0]['contact_surface_offset_m']) < 1e-9
    assert len(measured['mounted_cameras']['rgb_identities']) == 8
    assert measured['initial_overview']['snapshot_id'] == measured['mounted_cameras']['snapshot_id'] == 1
    assert all(row['relation_truth'] == 'not_measured_by_identity_collector'
               for row in measured['inventory']['source_inventory_evidence'])
    for name, obj in native.session.env.scene.objects.items():
        assert all(np.array_equal(a, b) for a, b in zip(before[name], obj.get_position_orientation()))
    first['measurements']['inventory']['native_objects'][0]['scale'][0] = 99
    assert native.provider(native.contract)['measurements']['inventory']['native_objects'][0]['scale'][0] == 1


def test_source_room_identity_is_separate_from_full_building_loading_scope(native):
    assert native.task.room_name == native.session.room_name == 'representative_room'
    assert native.contract['room'] == 'full_scene'
    row = native.provider(native.contract)
    scope = row['measurements']['geometry']['loading_scope']
    assert scope['declared_scope'] == 'full_scene'
    assert scope['representative_source_room'] == 'representative_room'
    assert scope['complete_native_loading'] == 'pending_external_qualification'
    assert 'constructor_filters' in row['measurements']['geometry']
    assert 'gates' not in row and not hasattr(native.session, 'mapping_joint_world_qualification')


@pytest.mark.parametrize('room', ['different_room', 'full_scene', '', None])
@pytest.mark.parametrize('target', ['task', 'session', 'both'])
def test_source_room_mismatch_or_missing_is_rejected(native, room, target):
    if target in ('task', 'both'):
        native.task.room_name = room
    if target in ('session', 'both'):
        native.session.room_name = room
    with pytest.raises(ValueError, match='representative source JSON'):
        native.provider(native.contract)


def test_loading_scope_must_remain_explicitly_full_building(native):
    native.contract['room'] = 'representative_room'
    with pytest.raises(ValueError, match='full-building loading scope'):
        native.provider(native.contract)


def test_legal_robot_motion_and_door_joints_preserve_initial_identity(native):
    first = native.provider(native.contract)
    native.robot.position[0] += 1
    native.door.joints['hinge'].value = .8
    native.session.observation_source.action_step = 3
    later_pixels = np.full((4, 6, 3), 255, dtype=np.uint8)
    native.session.observation_source.latest_snapshot = NS(observation=replace(
        native.session.observation_source.initial_observation, snapshot_id=8, action_step=3,
        agentview_rgb=later_pixels))
    later = native.provider(native.contract)
    assert first['measurements'] == later['measurements']
    assert first['current_state']['robot'] != later['current_state']['robot']
    assert later['current_state']['door_joint_states']['door']['hinge'] == [.8]
    assert later['measurements']['doors']['native_door_states'][0]['joints']['hinge'] == [0.]


@pytest.mark.parametrize('mutate', [
    lambda n: n.wall.position.__setitem__(0, 4.5),
    lambda n: n.floor.extent.__setitem__(0, 9),
    lambda n: n.chair.scale.__setitem__(0, .5),
    lambda n: setattr(n.chair, 'model', 'replacement'),
    lambda n: setattr(n.chair, 'category', 'different'),
    lambda n: setattr(n.chair, 'visual_only', True),
    lambda n: setattr(n.chair, 'fixed_base', False),
    lambda n: n.session.env.scene.objects.pop('door'),
    lambda n: n.session.initial_overview_identity.update(reset_id='different'),
    lambda n: n.session.initial_overview_identity.update(seed=7),
    lambda n: setattr(n.task, 'seed', 7),
    lambda n: setattr(n.session, 'scene_name', 'other_scene'),
    lambda n: setattr(n.session, 'question_path', n.source_paths[1]),
    lambda n: n.source_paths[1].write_text('{"changed":true}'),
    lambda n: n.control_file.write_text('{"translation_m":2}'),
    lambda n: n.scene_file.write_text('{"building":"changed"}'),
    lambda n: n.session.env.config['scene'].update(load_room_instances=['room_0']),
    lambda n: setattr(n.chair, 'in_rooms', ['different_room']),
    lambda n: n.session.env.scene.seg_map.room_ins_name_to_ins_id.update(room_0=2),
    lambda n: n.session.env.scene.trav_map.floor_heights.__setitem__(0, .2),
])
def test_changed_static_assets_source_reset_or_conditions_reject(native, mutate):
    native.provider(native.contract)
    mutate(native)
    with pytest.raises((ValueError, RuntimeError)):
        native.provider(native.contract)


@pytest.mark.parametrize('mutate', [
    lambda n: setattr(n.session.observation_source, 'initial_observation', None),
    lambda n: setattr(n.session.observation_source, 'initial_observation', replace(observation(), additional_rgb={})),
    lambda n: setattr(n.session.observation_source, 'initial_observation', replace(observation(),
        initial_overview=None, initial_overview_status='unavailable')),
    lambda n: setattr(n.session.observation_source, 'initial_observation', replace(observation(), snapshot_id=2)),
    lambda n: setattr(n.session.observation_source, 'action_step', 1),
    lambda n: n.session.robot_camera_registry.pop('wrist_rgb'),
])
def test_missing_or_late_initial_bundle_rejects_without_source_fallback(native, mutate):
    mutate(native)
    with pytest.raises(ValueError):
        native.provider(native.contract)
    assert native.provider.baseline is None
    assert not hasattr(native.session, 'mapping_joint_world_qualification')


def test_root_evidence_is_read_exactly_and_no_qualification_is_fabricated(native):
    first = native.provider(native.contract)
    native.session.mapping_source_policy_evidence = {'status': 'reviewed', 'trial_ids': ['root_door_trial']}
    native.session.mapping_controller_trial_evidence = {'status': 'reviewed', 'trial_ids': ['root_control_trial']}
    supplied = native.provider(native.contract)
    assert supplied['measurements']['doors']['source_policy_reconciliation'] == native.session.mapping_source_policy_evidence
    assert supplied['measurements']['controller']['public_control_trial_evidence'] == native.session.mapping_controller_trial_evidence
    for domain, fields in world.MEASUREMENTS.items():
        assert all(world._present(supplied['measurements'][domain][field]) for field in fields)
    assert first['measurements']['mounted_cameras'] == supplied['measurements']['mounted_cameras']
    assert not hasattr(native.session, 'mapping_joint_world_qualification')
    verdict = world.joint_world_qualification(native.session, native.controller, native.config)
    assert verdict['accepted'] is False and verdict['evaluation_validity'] == 'unavailable'
