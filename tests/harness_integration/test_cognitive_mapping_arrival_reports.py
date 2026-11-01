"""Exercise native-shaped room samples and explicit arrival reports on CPU.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q \
  tests/test_cognitive_mapping_arrival_reports.py
These fixtures never run a renderer, GPU, simulator, model, or native robot action.
"""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.authoring.conversion import ConversionContext, ROOT, SourceInstance
from wia.authoring.converters import cognitive_mapping_connectivity as connectivity
from wia.authoring.converters import cognitive_mapping_long_term_navigation as navigation
from wia.authoring.converters import cognitive_mapping_regional_boundary as regional
from wia.authoring.converters import cognitive_mapping_traversable_passage as passage
from wia.authoring.converters.cognitive_mapping_visit_reports import (
    NativeVisitReportEvaluator, native_array, public_navigation_route, visit_parameters)


A, B, C = 'room_a_0', 'room_b_0', 'extra_room_0'
D, E = 'room_d_0', 'room_e_0'
MODULES = (connectivity, navigation, regional, passage)


class Segmentation:
    floor_heights = [0.]
    room_ins_name_to_ins_id = {A: 1, B: 2, C: 3, D: 4, E: 5}
    unlabelled_transit = False

    def get_room_instance_by_point(self, xy):
        x = float(xy[0])
        if x > 3.4:
            return E
        if x > 2.6:
            return D
        if self.unlabelled_transit and .6 <= x <= 1.4:
            return None
        return A if x < .6 else B if x > 1.4 else C


class Anchor:
    def __init__(self, name, x):
        self.name, self.position = name, np.array([x, 1., .5])
        self.quaternion = np.array([0., 0., 0., 1.])

    @property
    def aabb(self):
        return self.position - .1, self.position + .1

    def get_position_orientation(self):
        return self.position.copy(), self.quaternion.copy()


class HolonomicBaseJointController:
    pass


class Chassis:
    def __init__(self, robot):
        self.robot = robot

    @property
    def aabb(self):
        return self.robot.position - [.1, .1, .4], self.robot.position + [.1, .1, .4]

    def get_linear_velocity(self):
        return np.array([self.robot.speed, 0., 0.])

    def get_angular_velocity(self):
        return np.zeros(3)


class R1:
    visual_only, fixed_base = False, True
    prim_path, base_footprint_link_name = '/World/robot', 'base_link'

    def __init__(self):
        self.position, self.quaternion = np.array([0., 0., .4]), np.array([0., 0., 0., 1.])
        self.speed, self.bad_contact, self.floor_missing, self.structured_contacts = 0., False, False, False
        self.links = {'base_link': Chassis(self)}
        self.controllers = {'base': HolonomicBaseJointController()}

    @property
    def aabb(self):
        return self.links['base_link'].aabb

    def get_position_orientation(self):
        return self.position.copy(), self.quaternion.copy()

    def contact_list(self):
        if self.floor_missing:
            return []
        point, normal = [float(self.position[0]), 0., 0.], [0., 0., 1.]
        if self.structured_contacts:
            dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
            point, normal = np.array(tuple(point), dtype=dtype), np.array(tuple(normal), dtype=dtype)
        result = [NS(body0=self.prim_path + '/base_link', body1='/World/floor/base', position=point, normal=normal)]
        if self.bad_contact:
            result.append(NS(body0=self.prim_path + '/base_link', body1='/World/wall',
                             position=[0., 0., .3], normal=[1., 0., 0.]))
        return result

    def set_position_orientation(self, *args, **kwargs):
        raise AssertionError('An evaluator must not move the robot.')


class World:
    def __init__(self, module, output, *, ordered=False, required=None, route=None):
        self.robot = R1()
        self.sim = NS(current_time=0.)
        self.anchors = {'anchor_a': Anchor('anchor_a', 0.), 'anchor_b': Anchor('anchor_b', 2.)}
        scene = NS(seg_map=Segmentation(), objects=[NS(category='floors', prim_path='/World/floor')],
                   object_registry=lambda kind, name: self.anchors[name])
        self.session = NS(robot=self.robot, env=NS(scene=scene), scene_name='fixture', room_name='full_scene',
                          pipeline=NS(og=NS(sim=self.sim)))
        self.controller = NS(config=NS(max_translation_speed_m_s=.5, translation_tolerance_m=.01),
                             dt=.1, pose=lambda: {'position': self.robot.position.tolist(),
                                                  'quaternion_xyzw': self.robot.quaternion.tolist()})
        parameters = {'source_region': A, 'start_region': A, 'target_region': B,
            'target_regions': [A, B], 'source_task_type': 'objects_same_region',
            'start_xy': [0., 0.], 'goal_xy': [2., 0.], 'initial_yaw_rad': 0.,
            'start_position_tolerance_m': .05, 'start_heading_tolerance_deg': 2.,
            'floor_idx': 0, 'floor_z': 0., 'floor_z_m': 0., 'minimum_up_dot': .95,
            'floor_contact_tolerance_m': .005, 'floor_contact_height_tolerance_m': .005,
            'minimum_floor_contact_vertical_normal': .5, 'stability_steps': 3,
            'maximum_terminal_speed_m_s': .02, 'maximum_terminal_angular_speed_rad_s': .1,
            'anchor_preservation_tolerance_m': .02, 'maximum_step_displacement_m': .1}
        question = f'You are beginning in the room a 0. You want to navigate to the room b 0. 1. Go to the room b 0.'
        order, evidence = public_navigation_route(question, A, B) if ordered else ([], [])
        if route:
            question = 'Required route: ' + ' -> '.join(route)
            order = route
            evidence = [{'region': room, 'source_phrase': question} for room in route]
        parameters.update(visit_parameters(required or list(dict.fromkeys(route or [A, B])), required_order=order,
            route_evidence=evidence, question=question))
        operation = module.OPERATION if module is navigation else module.EXPLORATION_OPERATION
        config = {'module': module.MODULE, 'operation': operation, 'scene': 'fixture', 'room': 'full_scene',
                  'parameters': parameters, 'binding_keys': ['anchor_a', 'anchor_b'] if module is regional else [],
                  'bindings': {'anchor_a': 'anchor_a', 'anchor_b': 'anchor_b'}}
        self.evaluator = module.create_evaluator(self.session, self.controller, config, output)
        self.observation = NS(snapshot_id='snapshot_0', action_step=0)
        self.evaluator.begin_episode(self.observation)

    def sample(self, *, x=None, speed=0.):
        if x is not None:
            self.robot.position[0] = x
        self.robot.speed = speed
        self.sim.current_time += .1
        self.evaluator.sample_step()

    def settle(self):
        for _ in range(3):
            self.sample()

    def move(self, x):
        while abs(float(self.robot.position[0]) - x) > 1e-8:
            current = float(self.robot.position[0])
            self.sample(x=current + np.clip(x - current, -.05, .05), speed=.5)
        self.robot.speed = 0.

    def report(self, name):
        return self.evaluator.report_arrival(name, self.observation)

    def record(self):
        return self.evaluator.record(self.observation, action=None)


@pytest.mark.parametrize('module', MODULES)
def test_b_before_a_extra_rooms_and_no_terminal_requirement(module, tmp_path):
    world = World(module, tmp_path)
    initial_scene = [(anchor.position.copy(), anchor.quaternion.copy()) for anchor in world.anchors.values()]
    world.move(2.)
    world.settle()
    assert world.report(B)['accepted']
    world.move(1.)
    world.settle()
    assert world.report(C)['accepted']
    world.move(0.)
    world.settle()
    assert world.report(A)['accepted']
    world.move(1.)
    world.settle()
    row = world.record()
    assert row['task_success']
    assert row['visited_regions'] == sorted([A, B, C])
    assert row['reported_regions'] == sorted([A, B, C])
    assert row['native_state']['chassis_region'] == C
    assert not row['proof_requirements']['terminal_room_required']
    assert [visit['region'] for visit in row['measured_visits']][:3] == [B, C, A]
    for anchor, (position, quaternion) in zip(world.anchors.values(), initial_scene):
        np.testing.assert_array_equal(anchor.position, position)
        np.testing.assert_array_equal(anchor.quaternion, quaternion)


@pytest.mark.parametrize('module', MODULES)
def test_occupancy_without_agent_reports_fails(module, tmp_path):
    world = World(module, tmp_path)
    world.settle()
    world.move(2.)
    world.settle()
    row = world.record()
    assert row['checks']['required_native_room_set_visited']
    assert not row['checks']['required_agent_room_set_reported']
    assert not row['task_success']


@pytest.mark.parametrize('module', MODULES)
def test_premature_and_wrong_claims_never_retroactively_count(module, tmp_path):
    world = World(module, tmp_path)
    assert not world.report(A)['accepted']
    assert not world.report(B)['accepted']
    for _ in range(8):
        world.record()
    assert not world.evaluator.visited
    world.settle()
    assert world.report(A)['accepted']
    assert not world.report(B)['accepted']
    world.move(2.)
    world.settle()
    row = world.record()
    assert B in row['visited_regions'] and B not in row['reported_regions']
    assert not row['task_success']
    assert world.report(B)['accepted']
    assert world.record()['task_success']
    reports = [json.loads(line) for line in (tmp_path / 'trajectory.jsonl').read_text().splitlines()
               if json.loads(line).get('event') == 'agent_arrival_report']
    assert all(report['snapshot_id'] == 'snapshot_0' and report['action_step'] == 0 for report in reports)
    assert [report['accepted'] for report in reports] == [False, False, True, False, True]


@pytest.mark.parametrize('module', MODULES)
def test_starting_room_requires_stable_samples_without_forced_exit(module, tmp_path):
    world = World(module, tmp_path)
    world.evaluator.required = {A}
    before_position = world.robot.position.copy()
    before_time = world.sim.current_time
    assert not world.report(A)['accepted']
    assert world.sim.current_time == before_time
    world.settle()
    assert world.report(A)['accepted']
    assert world.record()['task_success']
    assert world.record()['sampled_travel_m'] == 0.
    np.testing.assert_array_equal(world.robot.position, before_position)


@pytest.mark.parametrize('module', MODULES)
@pytest.mark.parametrize('fault', ('floor_missing', 'unsampled_motion', 'clock', 'tilt', 'moving_claim'))
def test_unsafe_or_unstable_native_state_cannot_earn_report(module, fault, tmp_path):
    world = World(module, tmp_path)
    world.settle()
    if fault in ('floor_missing', 'bad_contact'):
        setattr(world.robot, fault, True)
    elif fault == 'unsampled_motion':
        world.robot.position[0] += .2
    elif fault == 'clock':
        world.evaluator.sample_step()
    elif fault == 'tilt':
        world.robot.quaternion = np.array([1., 0., 0., 0.])
    elif fault == 'moving_claim':
        world.robot.speed = .5
    assert not world.report(A)['accepted']
    assert A not in world.evaluator.reported


def test_legacy_explicit_public_order_is_enforced_without_filling_private_blanks(tmp_path):
    world = World(navigation, tmp_path, ordered=True)
    world.move(2.)
    world.settle()
    assert world.report(B)['accepted']
    world.move(0.)
    world.settle()
    assert world.report(A)['accepted']
    row = world.record()
    assert row['checks']['required_native_room_set_visited']
    assert row['checks']['required_agent_room_set_reported']
    assert not row['task_success']
    assert not row['checks']['source_requested_order_proven']
    world.move(2.)
    world.settle()
    assert not world.record()['task_success']
    assert world.report(B)['order_advanced']
    assert world.record()['task_success']
    question = ('You are beginning in the living room 0. You want to navigate to the office 0. '
                '1. Choose the next region to explore. [please fill in] '
                '2. Choose the next region to explore. corridor_0 '
                '3. Choose the next region to explore. office_0 You have reached the final destination.')
    order, evidence = public_navigation_route(question, 'living_room_0', 'office_0')
    assert order == ['living_room_0', 'corridor_0', 'office_0']
    assert all(item['source_phrase'] in question for item in evidence)
    assert public_navigation_route('Are room a 0 and room b 0 adjacent?', A, B) == ([], [])
    question = ('You are beginning in the living room 0. You want to navigate to the office 0. '
                '1. Go to the kitchen 0. 2. [please fill in] Choose the next turn direction. '
                '3. Go to the corridor 0. 4. turn back. Choose the next turn direction. '
                '5. Go to the office 0. You have reached the final destination.')
    order, evidence = public_navigation_route(question, 'living_room_0', 'office_0')
    assert order == ['living_room_0', 'kitchen_0', 'corridor_0', 'office_0']
    assert all(item['source_phrase'] in question for item in evidence)


def test_order_without_public_source_evidence_is_rejected(tmp_path):
    config = {'parameters': {**visit_parameters([A, B]), 'required_order': [A, B]}}
    with pytest.raises(ValueError, match='exact_public_question_evidence'):
        NativeVisitReportEvaluator(None, None, config, tmp_path)
    config['parameters'] = {'required_regions': [A, B], 'arrival_report_required': False}
    with pytest.raises(ValueError, match='explicit_agent_arrival_reports_required'):
        NativeVisitReportEvaluator(None, None, config, tmp_path)


@pytest.mark.parametrize('module', MODULES)
def test_structured_native_contacts_are_supported(module, tmp_path):
    world = World(module, tmp_path)
    world.robot.structured_contacts = True
    world.settle()
    assert world.report(A)['accepted']
    assert not world.record()['motion_faults']
    vector = np.array((1., 2., 3.), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    assert native_array(vector).tolist() == [1., 2., 3.]


@pytest.mark.parametrize('module', MODULES)
def test_legacy_converter_public_goal_and_private_answer_boundary(module, tmp_path):
    converter = module.CategoryConverter()
    paths = sorted((ROOT / 'ESI/dataset/json_clean/Cognitive Mapping' / converter.category[1]).rglob('q_*.json'))
    assert paths
    source = SourceInstance(paths[0], str(paths[0].relative_to(ROOT / 'ESI/dataset/json_clean')), *converter.category)
    raw_before = source.path.read_bytes()
    result = converter.convert(source, ConversionContext(tmp_path, config={'mapping_revision': 'legacy'}))
    task = result['task_definition']
    p = task['private']['evaluation']['parameters']
    assert p['arrival_report_required'] is True and p['required_regions']
    public = json.dumps(task['public'])
    assert 'arrived_rooms' in public and 'ARRIVED:' not in public
    assert not any(key in public for key in ('graph_shortest_region_path', 'shortest_path_regions',
        'region_sequence', 'answer_region', 'correct_region', 'target_regions', 'floor_insseg', 'recorded_distances'))
    assert source.path.read_bytes() == raw_before
    if module is not navigation:
        assert not p['required_order'] and not p['public_route_required']
    if module is regional:
        # Hidden answer room is absent from the operation goal: it is anchor-defined.
        assert p['required_regions'][0] not in task['public']['objective']
    assert result['scene_definition']['scene']['variants']['original']['patches'] == []
    patches = task['scene']['variants'].get('adapted', {}).get('patches', [])
    assert all(patch['operation'] == 'robot_pose' for patch in patches)


@pytest.mark.parametrize('module', (connectivity, navigation, passage))
def test_legacy_private_paths_and_seed_cannot_select_route_requirements(module, tmp_path):
    converter = module.CategoryConverter()
    paths = sorted((ROOT / 'ESI/dataset/json_clean/Cognitive Mapping' / converter.category[1]).rglob('q_*.json'))
    source = SourceInstance(paths[0], str(paths[0].relative_to(ROOT / 'ESI/dataset/json_clean')), *converter.category)
    baseline = converter.convert(source, ConversionContext(tmp_path, seed=0, config={'mapping_revision': 'legacy'}))
    raw = source.raw
    payload = source.payload
    for key in ('graph_shortest_region_path', 'shortest_path_regions', 'region_sequence'):
        payload['question_data'][key] = ['private_only_region_99', B, A]
    raw['metadata_json'] = json.dumps(payload)
    copied = tmp_path / 'private_path_fixture.json'
    copied.write_text(json.dumps(raw))
    altered = SourceInstance(copied, source.relative_path, *converter.category)
    changed = converter.convert(altered, ConversionContext(tmp_path, seed=128, config={'mapping_revision': 'legacy'}))
    assert changed['task_definition']['public'] == baseline['task_definition']['public']
    first = baseline['task_definition']['private']['evaluation']
    second = changed['task_definition']['private']['evaluation']
    assert first == second
    assert 'private_only_region_99' not in json.dumps(changed['task_definition']['public'])


def test_report_requires_current_stability_after_reentering_visited_room(tmp_path):
    world = World(connectivity, tmp_path)
    world.settle()
    assert world.report(A)['accepted']
    world.move(2.)
    world.settle()
    world.move(0.)
    assert not world.report(A)['accepted']
    world.settle()
    assert world.report(A)['accepted']


def test_regional_anchor_changes_latch_failure(tmp_path):
    world = World(regional, tmp_path)
    world.anchors['anchor_a'].position[1] += .1
    world.settle()
    assert not world.report(A)['accepted']
    world.anchors['anchor_a'].position[1] -= .1
    world.settle()
    assert not world.report(A)['accepted']
    assert not world.record()['checks']['continuous_safe_native_trajectory']


@pytest.mark.parametrize('module', MODULES)
def test_supported_unlabelled_transit_is_safe_but_never_a_room_visit(module, tmp_path):
    world = World(module, tmp_path)
    world.session.env.scene.seg_map.unlabelled_transit = True
    world.settle()
    assert world.report(A)['accepted']
    world.move(1.)
    world.settle()
    row = world.record()
    assert row['native_state']['state_valid']
    assert row['native_state']['chassis_region'] is None
    assert row['visited_regions'] == [A]
    assert not row['motion_faults']
    assert not world.report(C)['accepted']
    world.move(2.)
    world.settle()
    assert world.report(B)['accepted']
    assert world.record()['task_success']


def test_missing_native_arrival_measurement_is_private_rejection(tmp_path):
    world = World(connectivity, tmp_path)
    world.settle()
    world.robot.links['base_link'].get_linear_velocity = lambda: [float('nan'), 0., 0.]
    result = world.report(A)
    assert not result['accepted'] and result['reason'] == 'native_arrival_measurement_unavailable'
    assert A not in world.evaluator.reported
    assert result['snapshot_id'] == 'snapshot_0'
    assert 'native_state' not in result
    assert 'native_arrival_measurement_unavailable' in world.evaluator.faults


def test_four_of_five_room_reports_are_eighty_percent(tmp_path):
    world = World(connectivity, tmp_path, required=[A, B, C, D, E])
    for room, x in [(A, 0.), (C, 1.), (B, 2.), (D, 3.)]:
        world.move(x)
        world.settle()
        assert world.report(room)['accepted']
    assert world.report(D)['accepted']
    world.move(4.)
    world.settle()
    row = world.record()
    assert row['completed_goals'] == 4 and row['total_goals'] == 5
    assert row['completion_score'] == .8 and row['score_percent'] == 80.
    assert row['evaluation_validity'] == 'complete'
    assert row['scoring_version'] == 'task_completion_v2'
    assert not row['task_success']
    assert all(item['evaluation_validity'] == 'valid' for item in row['completion_items'])
    assert {item['id'] for item in row['completion_items']} == {f'room:{room}' for room in [A, B, C, D, E]}


def test_ordered_reports_preserve_prefix_and_require_repeated_return(tmp_path):
    world = World(navigation, tmp_path, route=[A, B, A])
    world.settle()
    assert world.report(A)['order_advanced']
    assert not world.report(A)['order_advanced']
    world.move(2.)
    world.settle()
    assert world.evaluator.order_stage == 1
    assert world.report(B)['order_advanced']
    assert world.record()['completion_score'] == pytest.approx(2 / 3)
    world.move(1.)
    world.settle()
    assert world.report(C)['accepted']
    assert world.evaluator.order_stage == 2
    world.move(0.)
    world.settle()
    assert not world.record()['task_success']
    assert world.report(A)['order_advanced']
    row = world.record()
    assert row['task_success'] and row['completed_goals'] == 3
    assert [item['id'] for item in row['completion_items']] == [f'step:0:{A}', f'step:1:{B}', f'step:2:{A}']


def test_out_of_order_five_room_coverage_requires_replaying_missing_suffix(tmp_path):
    world = World(navigation, tmp_path, route=[A, B, C, D, E])
    for room, x in [(A, 0.), (C, 1.), (B, 2.), (D, 3.), (E, 4.)]:
        world.move(x)
        world.settle()
        assert world.report(room)['accepted']
    row = world.record()
    assert row['checks']['required_agent_room_set_reported']
    assert row['completed_goals'] == 2 and row['completion_score'] == .4
    assert not row['task_success']
    for room, x in [(C, 1.), (D, 3.), (E, 4.)]:
        world.move(x)
        world.settle()
        assert world.report(room)['order_advanced']
    assert world.record()['task_success']


@pytest.mark.parametrize('module', MODULES)
def test_completion_does_not_disable_continuous_safety(module, tmp_path):
    world = World(module, tmp_path)
    for room, x in [(A, 0.), (B, 2.)]:
        world.move(x)
        world.settle()
        assert world.report(room)['accepted']
    assert world.record()['task_success']
    world.robot.bad_contact = True
    row = world.record()
    assert row['completed_goals'] == 2
    assert row['task_success'] and not row['continuous_constraints_satisfied']
    assert row['safety_affects_task_success'] is False


def test_measurement_loss_preserves_denominator_without_fabricated_zero(tmp_path):
    world = World(connectivity, tmp_path)
    world.settle()
    assert world.report(A)['accepted']
    world.robot.links['base_link'].get_linear_velocity = lambda: [float('nan'), 0., 0.]
    row = world.record()
    assert row['evaluation_validity'] == 'unavailable'
    assert row['completion_score'] is None and row['score_percent'] is None
    assert row['completed_goals'] == 1 and row['total_goals'] == 2
    assert row['diagnostic_completion_lower_bound'] == .5
    assert row['measurement_errors'][0]['stage'] == 'record'
    assert [item['evaluation_validity'] for item in row['completion_items']] == ['valid', 'unavailable']


def test_missing_sample_remains_unavailable_after_measurement_recovers(tmp_path):
    world = World(connectivity, tmp_path)
    original = world.robot.links['base_link'].get_linear_velocity
    world.robot.links['base_link'].get_linear_velocity = lambda: [float('nan'), 0., 0.]
    world.sample()
    world.robot.links['base_link'].get_linear_velocity = original
    row = world.record()
    assert row['evaluation_validity'] == 'unavailable'
    assert row['completion_score'] is None
    assert row['measurement_errors'][0]['stage'] == 'control_sample'


@pytest.mark.parametrize('module', MODULES)
def test_legal_arm_contact_allows_background_furniture_but_chassis_contact_fails(module, tmp_path):
    world = World(module, tmp_path)
    world.robot.arm_link_names = {'left': ['arm_link']}
    world.robot.links['arm_link'] = NS(prim_path='/World/robot/arm_link')
    world.session.background_movement_policy = {'allowed': {'chair': NS(prim_path='/World/chair')}}
    original_contacts = world.robot.contact_list
    contact = NS(body0='/World/robot/arm_link', body1='/World/chair/base', position=[0., 0., .8], normal=[1., 0., 0.])
    world.robot.contact_list = lambda: original_contacts() + [contact]
    world.settle()
    assert world.report(A)['accepted']
    contact.body0 = '/World/robot/base_link'
    assert not world.record()['continuous_constraints_satisfied']


@pytest.mark.parametrize('module', MODULES)
def test_contact_during_and_before_visits_does_not_veto_task(module, tmp_path):
    world = World(module, tmp_path)
    world.robot.bad_contact = True
    world.settle()
    assert world.report(A)['accepted']
    world.robot.bad_contact = False
    world.move(2.)
    world.settle()
    assert world.report(B)['accepted']
    row = world.record()
    assert row['task_success'] and row['completion_score'] == 1.
    assert row['task_constraints_satisfied']
    assert not row['constraints_satisfied']
    assert 'native_nonfloor_contact' in row['motion_faults']


def test_contact_does_not_create_missing_task_progress(tmp_path):
    world = World(connectivity, tmp_path)
    world.robot.bad_contact = True
    world.settle()
    world.report(A)
    row = world.record()
    assert not row['task_success'] and row['completed_goals'] == 1
    assert row['completion_score'] == .5
