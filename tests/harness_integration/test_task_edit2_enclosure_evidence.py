"""Test read-only enclosure causality and exact native-shaped private receipts.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests/test_task_edit2_enclosure_evidence.py
Fixtures exercise actual measurement helpers and the shared visual collector;
they do not certify native loading, geometry, contact, images or task execution.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from wia.authoring.converters import enumerative_perception_structural_enclosure as enclosure
from core.observations import AgentObservation, InitialOverview
from core.robot_cameras import ADDITIONAL_RGB_NAMES, ROBOT_CAMERA_LABELS
from wia.runtime.scene_capture_queue import sha256
import wia.runtime.sim.esi_enclosure_evidence as evidence
from wia.runtime.sim.esi_interaction_history import public_action_begin, public_action_end
from wia.runtime.sim.esi_visual_evidence import FEATURE_AUDIT_SCHEMA, install_visual_evidence
from wia.tasks.loader import LoadedScene, LoadedTask
from wia.tasks.schema import Source


class Actor:
    def __init__(self, name, category, position, extent):
        self.name, self.category, self.model = name, category, name + '_model'
        self.prim_path = '/World/' + name
        self.position, self.extent = np.array(position, dtype=float), np.array(extent, dtype=float)
        self.quaternion, self.scale = np.array([0., 0., 0., 1.]), np.ones(3)
        self.visual_only, self.fixed_base = False, True
        self.links = {}

    @property
    def aabb(self):
        return self.position - self.extent / 2, self.position + self.extent / 2

    def get_position_orientation(self):
        return self.position.copy(), self.quaternion.copy()

    def get_linear_velocity(self):
        return np.zeros(3)

    def get_angular_velocity(self):
        return np.zeros(3)

    def set_position_orientation(self, *args, **kwargs):
        pytest.fail('The collector must never mutate native poses.')


class Mesh:
    def __init__(self, link, name):
        self.link, self.name = link, name
        self.prim_path = link.prim_path + '/' + name
        self.collision_enabled, self.size = True, .2
        self.prim = NS(GetName=lambda: name, GetPath=lambda: self.prim_path,
            GetTypeName=lambda: 'Cube', GetAttribute=lambda name: NS(Get=lambda: self.size))

    @property
    def aabb(self):
        return self.link.aabb


class Link:
    def __init__(self, actor, name, *, door=False):
        self.actor, self.name, self.door = actor, name, door
        self.prim_path = actor.prim_path + '/' + name
        self.visible, self.contact_reporting_enabled = True, True
        self.collision_meshes = {'collision': Mesh(self, 'collision')}

    @property
    def aabb(self):
        offset = np.array([self.actor.angle * .1, 0., 0.]) if self.door else np.zeros(3)
        return tuple(value + offset for value in self.actor.aabb)

    def contact_list(self):
        return ([NS(body0='/World/robot/finger', body1=self.prim_path)] if self.actor.hand_contact else [])


class Joint:
    def __init__(self, container):
        self.container, self.name = container, 'hinge'
        self.body1 = container.prim_path + '/door'
        self.joint_type, self.lower_limit, self.upper_limit = 'revolute', 0., 1.6
        self.driven = False

    def get_state(self):
        return np.array([self.container.angle]), np.array([0.])


class Sensor(Actor):
    def __init__(self, name, rgb, target):
        super().__init__(name, 'camera', [0., 0., 1.], [.1, .1, .1])
        self.prim_path = '/World/robot/' + name
        segmentation = np.zeros(rgb.shape[:2], dtype=np.int32)
        segmentation[1:6, 2:7] = 7
        self.data = {'rgb': rgb.copy(), 'seg_instance': segmentation}
        self.info = {'seg_instance': {7: target.name}}

    def get_obs(self):
        return self.data, self.info


def observation(snapshot=1, step=0, overview=None):
    rgb = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    captured = f'2026-10-18T10:00:{snapshot:02d}+00:00'
    if overview is None:
        overview = InitialOverview(rgb=rgb, episode_id='episode', reset_id='reset', seed=0,
            snapshot_id=snapshot, captured_at=captured, extent_xy_m=(-5., -5., 5., 5.),
            image_right_world_xy=(1., 0.), image_up_world_xy=(0., 1.),
            robot_start_position_m=(0., 0., .5), robot_start_yaw_rad=0.)
    return AgentObservation(task_text='Open the original enclosure and discover its content.',
        agentview_rgb=rgb, wrist_rgb=rgb, additional_rgb={name: rgb for name in ADDITIONAL_RGB_NAMES},
        action_history=(), snapshot_id=snapshot, action_step=step, captured_at=captured,
        action_boundary=True, initial_overview=overview, initial_overview_status='available')


@pytest.fixture
def world(tmp_path, monkeypatch):
    states, constants = ModuleType('omnigibson.object_states'), ModuleType('omnigibson.utils.constants')
    states.Inside, states.Open = type('Inside', (), {}), type('Open', (), {})
    constants.JointType = NS(JOINT_REVOLUTE='revolute', JOINT_PRISMATIC='prismatic')
    monkeypatch.setitem(sys.modules, 'omnigibson.object_states', states)
    monkeypatch.setitem(sys.modules, 'omnigibson.utils.constants', constants)
    container = Actor('container_native', 'microwave', [1., 0., .7], [.5, .5, .4])
    container.angle, container.hand_contact = 0., False
    container.links = {'root': Link(container, 'root'), 'door': Link(container, 'door', door=True)}
    container.root_link = container.links['root']
    container.joint = Joint(container)
    container.states = {states.Open: NS(relevant_joints_info=(False, [container.joint], [1]),
                                       get_value=lambda: container.angle >= .5)}
    content = Actor('content_native', 'club_sandwich', [1., 0., .65], [.12, .12, .08])
    content.inside = True
    content.states = {states.Inside: NS(get_value=lambda other: content.inside and other is container)}
    floor = Actor('floor', 'floors', [0., 0., -.05], [8., 8., .1])
    robot = Actor('robot', 'agent', [0., 0., .5], [.5, .5, 1.])
    robot.links = {'footprint': NS(aabb=robot.aabb)}
    robot.base_footprint_link_name = 'footprint'
    robot.eef_links = {'left': NS(prim_path='/World/robot/hand')}
    robot.finger_links = {'left': [NS(prim_path='/World/robot/finger')]}
    robot.floor_contacts = [NS(body0='/World/robot/wheel', body1='/World/floor/base_link',
                               normal=np.array([0., 0., 1.]), position=np.zeros(3))]
    robot.contact_list = lambda: robot.floor_contacts + ([NS(body0='/World/robot/finger', body1=container.links['door'].prim_path,
        normal=np.array([1., 0., 0.]), position=np.array([1., 0., .7]))] if container.hand_contact else [])
    initial = observation()
    robot.sensors = {name: Sensor('sensor_' + name, initial.rgb_images()[name], content) for name in ROBOT_CAMERA_LABELS}
    files = {}
    for name in ('task.yaml', 'scene.yaml', 'source.json'):
        path = tmp_path / name
        path.write_text('immutable ' + name)
        files[name] = path
    source = Source.model_validate({'provider': 'esi', 'json': 'Enumerative Perception/Structural Enclosure/scene/room/q.json'})
    loaded_args = {'source_path': files['source.json'], 'source_sha256': sha256(files['source.json']),
                   'variant': 'adapted', 'seed': 0, 'scene_name': 'scene', 'room_name': 'room'}
    task = LoadedTask(definition=NS(id='task', source=source), path=files['task.yaml'],
                      file_sha256=sha256(files['task.yaml']), **loaded_args)
    scene = LoadedScene(definition=NS(id='scene', source=source), path=files['scene.yaml'],
                        file_sha256=sha256(files['scene.yaml']), **loaded_args)
    actors = {obj.name: obj for obj in (container, content, floor)}
    def forbidden(*args, **kwargs):
        pytest.fail('Evidence must not render, step, reset or change native state.')
    session = NS(robot=robot, scene_name='scene', room_name='room', native_reset_id='reset', episode_id='episode',
        loaded_task=task, scene_adapter=NS(task=scene),
        pipeline=NS(og=NS(sim=NS(current_time=1., render=forbidden, step=forbidden, load_state=forbidden))),
        env=NS(scene=NS(object_registry=lambda kind, name: actors.get(name),
            seg_map=NS(get_room_instance_by_point=lambda point: 'room'))),
        robot_camera_registry={name: name for name in ROBOT_CAMERA_LABELS},
        _observation_source=NS(initial_observation=initial, latest_snapshot=NS(observation=initial)))
    controller = NS(working_arm='left', debug_gripper_state=lambda: {'holding': False, 'constraint_active': False})
    objects = {'container_00': container, 'content_00': content, 'floor': floor}
    config = {'type': 'category_operation', 'module': enclosure.MODULE, 'operation': 'open_and_discover_contents',
        'scene': 'scene', 'room': 'room', 'bindings': {key: obj.name for key, obj in objects.items()},
        'binding_keys': list(objects), 'parameters': {'container_keys': ['container_00'], 'content_keys': ['content_00'],
            'source_objects': {key: {'position': obj.position.tolist(), 'quaternion_xyzw': obj.quaternion.tolist(),
                'bbox': [value.tolist() for value in obj.aabb]} for key, obj in objects.items() if key != 'floor'},
            'relations': {'content_00': 'container_00'}, 'curation_revision': enclosure.TASK_EDIT2_CURATION_REVISION}}
    install_visual_evidence(session, controller)
    collector = evidence.install_enclosure_evidence(session, controller)
    return NS(session=session, controller=controller, container=container, content=content, floor=floor,
        robot=robot, config=config, observation=initial, collector=collector, actors=actors, files=files)


def collect(world):
    return world.collector(world.observation, world.config)


def advance(world, time, angle, *, contact=True):
    world.session.pipeline.og.sim.current_time = time
    world.container.angle, world.container.hand_contact = angle, contact
    world.collector.sample_step()


def capture(world, snapshot=2, step=1):
    world.observation = observation(snapshot, step, world.observation.initial_overview)
    world.session._observation_source.latest_snapshot = NS(observation=world.observation)


def audit_feature(world):
    visual = world.session.task_visual_evidence(world.observation, {'content_00': world.content.name}, world.config['parameters'])
    measured = visual['measurements']['content_00']
    row = measured['views']['agentview_rgb']
    receipt = {'schema': FEATURE_AUDIT_SCHEMA, 'audit_id': 'test_native_feature_review',
        'binding': 'content_00', 'camera': 'agentview_rgb', 'context': deepcopy(row['audit_context']),
        'support_binding': 'content_00', 'support_identity': measured['native_identity'],
        'support_mask_sha256': row['mask_sha256'], 'feature_crop_bbox_xyxy': row['bbox_xyxy'], 'informative': True,
        'informative_feature_evidence': {'feature_ids': ['source_sandwich_layer_feature'],
            'semantic_identity_verified': True, 'view_kind': 'direct'},
        'review': {'reviewer': 'synthetic_fixture', 'reviewed_at': '2026-10-18T10:00:02Z',
                   'method': 'manual_native_image_feature_review'}}
    world.session.visual_evidence_feature_receipts = [receipt]
    return receipt


def open_legally(world):
    collect(world)
    public_action_begin(world.controller, 'MV_LEFT')
    advance(world, 1.1, 0.)
    advance(world, 1.2, .3)
    advance(world, 1.3, .7)
    public_action_end(world.controller)
    capture(world)
    audit_feature(world)
    return collect(world)


def interior_review(result):
    return {'schema': evidence.INTERIOR_REVIEW_SCHEMA, 'review_id': 'independent_test_cavity_measurement',
        'context': deepcopy(result['interior_review_contexts']['container_00']),
        'review': {'reviewer': 'synthetic_native_geometry_fixture', 'reviewed_at': '2026-10-18T10:00:02Z',
            'method': 'independent_native_collision_cavity_review'},
        'independent_cavity_evidence': {'measurement_source': 'independent_native_cavity_measurement',
            'measurement_id': 'native_geometry_fixture_probe', 'method': 'independent_closed_cavity_probes',
            'interior_geometry_verified': True, 'inside_or_aabb_only': False,
            'cavity_points_local_m': [[0., 0., 0.]], 'content_clearance_m': {'content_00': .015}}}


def test_baseline_reads_actual_native_helpers_without_qualification_or_mutation(world):
    before = {key: (obj.position.copy(), obj.quaternion.copy(), obj.scale.copy()) for key, obj in world.actors.items()}
    result = collect(world)
    assert result['gates']['actual_containment'] is True
    assert result['gates']['stable_robot_start'] is True
    assert result['gates']['public_control_contact_opening'] is False
    assert result['gates']['independent_interior_geometry'] is None
    assert result['gates']['post_open_informative_visibility'] is None
    assert result['baseline']['initial_doors_closed'] == {'container_00': True}
    assert result['baseline']['robot']['floor_contact_reading']['floor_supported'] is True
    mesh = result['native']['collision_meshes']['container_00'][0]
    assert mesh['local_geometry'] == {'type': 'Cube', 'attributes': {'size': .2}}
    assert mesh['collision_enabled'] is True
    assert not result['interior_inferred_from_inside_or_aabb']
    for key, obj in world.actors.items():
        for actual, previous in zip((obj.position, obj.quaternion, obj.scale), before[key]):
            np.testing.assert_array_equal(actual, previous)
    assert world.session.pipeline.og.sim.current_time == 1.


@pytest.mark.parametrize('failure', ['late_first_call', 'missing_Global', 'missing_camera', 'not_retained', 'prior_public_sequence'])
def test_first_call_must_be_complete_retained_zero_action_input(world, failure):
    if failure == 'late_first_call':
        capture(world)
    elif failure == 'missing_Global':
        world.observation = replace(world.observation, initial_overview=None, initial_overview_status='unavailable')
        world.session._observation_source.initial_observation = world.observation
        world.session._observation_source.latest_snapshot = NS(observation=world.observation)
    elif failure == 'missing_camera':
        world.session.robot_camera_registry.pop('chassis_rear_rgb')
    elif failure == 'not_retained':
        world.session._observation_source.latest_snapshot = NS(observation=observation(2, 0))
    else:
        public_action_begin(world.controller, 'MV_LEFT')
        public_action_end(world.controller)
    result = collect(world)
    assert result['available'] is False and world.collector.baseline is None
    assert all(value is None for value in result['gates'].values())
    # A failed first baseline may not be retrospectively established later.
    world.observation = world.session._observation_source.initial_observation
    assert 'late baseline freezing is forbidden' in collect(world)['measurement_error']


def test_real_public_contact_stroke_and_shared_semantics_are_separate_gates(world):
    result = open_legally(world)
    assert result['gates']['public_control_contact_opening'] is True
    assert result['gates']['access_to_every_content'] is True
    assert result['gates']['post_open_informative_visibility'] is True
    assert result['gates']['independent_interior_geometry'] is None
    assert result['objects']['content_00']['informative'] is True
    receipt = result['opening_receipts']['container_00']
    assert receipt['control_sequence'] == 1
    assert receipt['contact_time_s'] < receipt['opened_time_s']
    assert receipt['contact_open_travel'] >= enclosure.MINIMUM_ANGLE_RAD
    world.session.enclosure_interior_reviews = [interior_review(result)]
    result = collect(world)
    assert all(value is True for value in result['gates'].values())
    wrapped, gates = enclosure.task_edit2_enclosure_evidence(world.session, world.observation, world.config, {})
    assert wrapped['objects']['content_00']['visible'] is True
    assert gates == result['gates']
    assert len(collect(world)['discovery_receipts']) == 1


@pytest.mark.parametrize('failure', ['passive', 'STOP', 'STILL', 'no_contact', 'new_sequence', 'contact_after_open', 'already_open', 'no_advance'])
def test_passive_or_unbounded_joint_motion_never_counts_as_public_opening(world, failure):
    if failure == 'already_open':
        world.container.angle = .7
    collect(world)
    if failure != 'passive':
        public_action_begin(world.controller, failure if failure in ('STOP', 'STILL') else 'MV_LEFT')
    advance(world, 1.1, 0. if failure != 'already_open' else .7,
            contact=failure not in ('no_contact', 'contact_after_open'))
    if failure == 'new_sequence':
        public_action_end(world.controller)
        public_action_begin(world.controller, 'MV_LEFT')
    advance(world, 1.1 if failure == 'no_advance' else 1.2, .7,
            contact=failure != 'no_contact')
    capture(world)
    audit_feature(world)
    result = collect(world)
    assert not result['available'] or result['gates']['public_control_contact_opening'] is not True
    assert not world.collector.openings
    assert not world.collector.discoveries


@pytest.mark.parametrize('failure', ['driven', 'hidden', 'noncolliding', 'no_contact_reporting'])
def test_source_door_physics_cannot_be_replaced_by_visual_open_state(world, failure):
    collect(world)
    public_action_begin(world.controller, 'MV_LEFT')
    advance(world, 1.1, 0.)
    if failure == 'driven':
        world.container.joint.driven = True
    elif failure == 'hidden':
        world.container.links['door'].visible = False
    elif failure == 'noncolliding':
        world.container.links['door'].collision_meshes['collision'].collision_enabled = False
    else:
        world.container.links['door'].contact_reporting_enabled = False
    advance(world, 1.2, .7)
    capture(world)
    audit_feature(world)
    result = collect(world)
    assert result['gates']['public_control_contact_opening'] is False
    assert result['objects']['content_00']['informative'] is False


@pytest.mark.parametrize('field', ['task_sha256', 'scene_sha256', 'source_sha256', 'reset_id', 'seed', 'collision_geometry_sha256'])
def test_stale_or_mismatched_independent_review_stays_unknown(world, field):
    result = open_legally(world)
    receipt = interior_review(result)
    receipt['context'][field] = 'stale'
    world.session.enclosure_interior_reviews = [receipt]
    assert collect(world)['gates']['independent_interior_geometry'] is None


@pytest.mark.parametrize('failure', ['Inside_only', 'missing_clearance', 'no_points', 'no_review_method'])
def test_metadata_inside_and_aabb_are_not_independent_cavity_measurements(world, failure):
    result = open_legally(world)
    receipt = interior_review(result)
    if failure == 'Inside_only':
        receipt['independent_cavity_evidence']['inside_or_aabb_only'] = True
    elif failure == 'missing_clearance':
        receipt['independent_cavity_evidence']['content_clearance_m'] = {}
    elif failure == 'no_points':
        receipt['independent_cavity_evidence']['cavity_points_local_m'] = []
    else:
        receipt['review'].pop('method')
    world.session.enclosure_interior_reviews = [receipt]
    assert collect(world)['gates']['independent_interior_geometry'] is None


@pytest.mark.parametrize('failure', ['scale', 'root', 'content_pose', 'membership', 'native_model', 'local_mesh', 'state_setter'])
def test_initial_and_terminal_native_identity_layout_and_membership_are_protected(world, failure):
    result = open_legally(world)
    world.session.enclosure_interior_reviews = [interior_review(result)]
    if failure == 'scale':
        world.content.scale *= 2
    elif failure == 'root':
        world.container.position[0] += .2
    elif failure == 'content_pose':
        world.content.position[0] += .2
    elif failure == 'membership':
        world.content.inside = False
    elif failure == 'native_model':
        world.content.model = 'replacement_model'
    elif failure == 'local_mesh':
        world.container.links['root'].collision_meshes['collision'].size *= 2
    else:
        world.session._episode_state_setter_used = True
    result = collect(world)
    assert result['gates']['actual_containment'] is False
    assert result['gates']['independent_interior_geometry'] is None
    assert result['objects']['content_00']['informative'] is False
    assert not world.collector.preserved


def test_semantic_features_cannot_be_replaced_by_masks_pixels_or_missing_provider(world):
    open_legally(world)
    world.session.visual_evidence_feature_receipts = []
    world.collector.discoveries.clear()
    result = collect(world)
    assert result['objects']['content_00']['visible_pixels'] > 0
    assert result['objects']['content_00']['informative'] is False
    assert result['gates']['post_open_informative_visibility'] is None
    del world.session.task_visual_evidence
    assert collect(world)['gates']['access_to_every_content'] is None


def test_shared_feature_audit_before_legal_opening_does_not_discover_content(world):
    audit_feature(world)
    result = collect(world)
    assert result['objects']['content_00']['identity_synchronized'] is True
    assert result['objects']['content_00']['informative'] is False
    assert world.collector.discoveries == {}


def test_missing_shared_visual_api_retains_actual_structural_measurements(world):
    def unavailable(*args):
        raise RuntimeError('Native camera feature adapter unavailable.')
    world.session.task_visual_evidence = unavailable
    result = collect(world)
    assert result['gates']['actual_containment'] is True
    assert result['gates']['stable_robot_start'] is True
    assert result['gates']['access_to_every_content'] is None
    assert result['available'] is False
    assert result['shared_visual_measurement_errors']['provider']


def test_initial_missing_containment_never_becomes_a_verified_enclosure(world):
    world.content.inside = False
    result = collect(world)
    assert result['gates']['actual_containment'] is False
    world.content.inside = True
    assert collect(world)['gates']['actual_containment'] is False


def test_missing_native_floor_details_are_unknown_and_current_upright_is_remeasured(world):
    world.robot.floor_contacts[0].normal = None
    result = collect(world)
    assert result['gates']['stable_robot_start'] is None
    world.robot.floor_contacts[0].normal = np.array([0., 0., 1.])
    assert collect(world)['gates']['stable_robot_start'] is None
    world.robot.quaternion[:] = [1., 0., 0., 0.]
    assert collect(world)['gates']['stable_robot_start'] is False


def test_missing_native_api_and_changed_reset_cannot_be_qualified(world):
    world.robot.get_angular_velocity = None
    result = collect(world)
    assert result['available'] is False
    assert all(value is None for value in result['gates'].values())
    assert result['measurement_error']


def test_reset_context_changes_invalidate_even_previously_verified_openings(world):
    open_legally(world)
    world.session.native_reset_id = 'different_reset'
    result = collect(world)
    assert result['available'] is False
    assert all(value is None for value in result['gates'].values())


def test_time_rewind_cannot_apply_a_future_opening_to_a_discovery_view(world):
    open_legally(world)
    world.session.pipeline.og.sim.current_time = 1.1
    capture(world, snapshot=3, step=2)
    audit_feature(world)
    result = collect(world)
    assert result['available'] is False
    assert all(value is None for value in result['gates'].values())


def test_joint_limit_changes_and_content_physics_changes_are_protected(world):
    open_legally(world)
    world.container.joint.upper_limit += .3
    assert collect(world)['gates']['public_control_contact_opening'] is False
    world.content.visual_only = True
    assert collect(world)['gates']['actual_containment'] is False


def test_consumer_strict_success_waits_for_separate_independent_interior_review(world, tmp_path):
    evaluator = enclosure.OpenDiscoverEvaluator(world.session, world.controller, world.config, tmp_path)
    evaluator.begin_episode(world.observation)
    public_action_begin(world.controller, 'MV_LEFT')
    for time, angle in ((1.1, 0.), (1.2, .3), (1.3, .7)):
        advance(world, time, angle)
        evaluator.sample_step()
    public_action_end(world.controller)
    capture(world)
    audit_feature(world)
    pending = evaluator.record(world.observation)
    assert pending['completion_items'] and all(row['completed'] for row in pending['completion_items'])
    assert pending['task_success'] is False
    native = collect(world)
    world.session.enclosure_interior_reviews = [interior_review(native)]
    assert evaluator.record(world.observation)['task_success'] is True
