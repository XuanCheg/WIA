"""Check room/order acquisition evidence and physical collection reduction.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B \
  -m pytest -q -p no:cacheprovider tests/test_wia_room_collection.py
Fixtures do not qualify native object access, navigation or tray feasibility.
"""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from wia.authoring.converters import wia_room_collection as collection
import wia.authoring.commands.build_wia_room_collection as builder


def layout():
    return {'scene_model': 'Fixture', 'route': ['kitchen_0', 'bedroom_0'],
            'public_environment': 'The original rooms and passages.',
            'robot_start': {'position': [0., 0., 1.], 'orientation': [0., 0., 0., 1.]},
            'plate': {'name': 'tray', 'category': 'tray', 'model': 'tray_model'},
            'native_inventory': [{'name': name, 'category': 'cup', 'model': 'cup_model',
                                  'room': room, 'public_name': label, 'role': 'target', 'support': 'table_' + room}
                                 for name, room, label in [('a', 'kitchen_0', 'red cup'),
                                                           ('b', 'bedroom_0', 'blue cup')]],
            'added_inventory': []}


def state(**changes):
    return {'native_time_s': 0., 'native_region': 'kitchen_0', 'room_occupancy_valid': True,
            'object_rooms': {'a': 'kitchen_0', 'b': 'bedroom_0'}, 'supported_objects': [],
            'held_objects': ['tray'], 'left_holding': None, 'left_empty': True,
            'right_holds_plate': True, 'plate_level': True, 'plate_stable': True, **changes}


def _robot_fixture():
    plate = SimpleNamespace(name='tray', category='tray', model='tray_model', prim_path='/World/tray')
    floor = SimpleNamespace(name='floor', category='floors', prim_path='/World/floor', visual_only=False)
    objects = {'tray': plate, 'floor': floor}
    contacts = [SimpleNamespace(body0='/World/robot/base_link/collision', body1='/World/floor/base_link/collision',
                               position=[0., 0., 0.], normal=[0., 0., 1.])]
    robot = SimpleNamespace(prim_path='/World/robot', orientation=[0., 0., 0., 1.], grasping=1,
        _ag_obj_in_hand={'right': plate}, contact_list=lambda: list(contacts),
        eef_links={'right': SimpleNamespace(prim_path='/World/robot/right_eef_link')},
        finger_links={'right': [SimpleNamespace(prim_path='/World/robot/right_gripper_link1'),
                                SimpleNamespace(prim_path='/World/robot/right_gripper_link2')]})
    robot.get_position_orientation = lambda: ([0., 0., .02], robot.orientation)
    robot.is_grasping = lambda arm: robot.grasping
    scene = SimpleNamespace(objects=objects, trav_map=SimpleNamespace(floor_heights=[0.]),
                            object_registry=lambda key, name: objects.get(name))
    return SimpleNamespace(robot=robot, env=SimpleNamespace(scene=scene)), objects, contacts


def test_robot_initialization_reads_actual_floor_and_upright_evidence_without_actions():
    session, objects, contacts = _robot_fixture()
    session.env.scene.trav_map.floor_heights = [.25]
    contacts[0].position = [1., 2., .254]
    contacts[0].normal = [0., 0., .5]
    actions = []

    def forbidden(*args, **kwargs):
        actions.append('unexpected_action')
        raise AssertionError('Robot initialization evidence must be read-only.')

    session.env.step = session.robot.apply_action = session.robot.set_position_orientation = forbidden
    objects['tray'].set_position_orientation = forbidden
    result = collection._robot_initialization(session, {'layout': layout()})
    assert result['measurement_available']
    assert all(result[key] for key in ('floor_supported', 'upright', 'collision_free'))
    assert result['floor_z_m'] == .25
    assert result['floor_prim_paths'] == ['/World/floor']
    assert result['floor_contact_positions_m'] == [[1., 2., .254]]
    assert result['upright_vertical_axis_dot'] == 1.
    assert result['evidence_scope'] == 'initialization_only'
    assert actions == []


def _structured_contact_vector(values, order):
    return np.array(tuple(values[axis] for axis in order), dtype=[(axis, 'f4') for axis in order])[()]


@pytest.mark.parametrize('order', [('x', 'y', 'z'), ('z', 'x', 'y'), ('y', 'z', 'x')])
def test_robot_initialization_accepts_scalar_physx_xyz_records_in_any_field_order(order):
    session, _, contacts = _robot_fixture()
    contacts[0].position = _structured_contact_vector({'x': 1.25, 'y': -2.5, 'z': .004}, order)
    contacts[0].normal = _structured_contact_vector({'x': 0., 'y': 0., 'z': 1.}, order)
    assert contacts[0].position.shape == contacts[0].normal.shape == ()
    result = collection._robot_initialization(session, {'layout': layout()})
    assert result['measurement_available']
    assert all(result[key] for key in ('floor_supported', 'upright', 'collision_free'))
    assert result['floor_contact_positions_m'][0] == pytest.approx([1.25, -2.5, .004])


@pytest.mark.parametrize('field,axis,value', [('position', 'x', float('nan')), ('normal', 'z', float('inf'))])
def test_robot_initialization_rejects_nonfinite_scalar_physx_records(field, axis, value):
    session, _, contacts = _robot_fixture()
    for name, values in [('position', {'x': 1.25, 'y': -2.5, 'z': .004}),
                         ('normal', {'x': 0., 'y': 0., 'z': 1.})]:
        if name == field:
            values[axis] = value
        setattr(contacts[0], name, _structured_contact_vector(values, ('z', 'x', 'y')))
    result = collection._robot_initialization(session, {'layout': layout()})
    assert result['measurement_available'] is False
    assert all(result[key] is False for key in ('floor_supported', 'upright', 'collision_free'))
    assert 'finite native measurements' in result['reason']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('failure,available', [('no_contacts', True), ('horizontal_normal', True),
    ('wrong_height', True), ('missing_floor_objects', False), ('missing_floor_map', False),
    ('nonfinite_floor', False), ('nonfinite_normal', False), ('missing_contact_position', False)])
def test_robot_initialization_rejects_missing_invalid_or_wrong_floor_evidence(failure, available):
    session, objects, contacts = _robot_fixture()
    if failure == 'no_contacts':
        contacts.clear()
    elif failure == 'horizontal_normal':
        contacts[0].normal = [1., 0., .49]
    elif failure == 'wrong_height':
        contacts[0].position[2] = .006
    elif failure == 'missing_floor_objects':
        objects['floor'].category = 'chair'
    elif failure == 'missing_floor_map':
        del session.env.scene.trav_map
    elif failure == 'nonfinite_floor':
        session.env.scene.trav_map.floor_heights = [float('nan')]
    elif failure == 'nonfinite_normal':
        contacts[0].normal[2] = float('nan')
    else:
        del contacts[0].position
    result = collection._robot_initialization(session, {'layout': layout()})
    assert result['floor_supported'] is False
    assert result['measurement_available'] is available
    if not available:
        assert result['collision_free'] is False
        assert 'measurement unavailable' in result['reason']
        assert len(result['reason']) <= 640
    json.dumps(result, allow_nan=False)


def test_robot_initialization_rejects_nonupright_pose():
    session, _, _ = _robot_fixture()
    session.robot.orientation = [1., 0., 0., 0.]
    result = collection._robot_initialization(session, {'layout': layout()})
    assert result['measurement_available']
    assert result['floor_supported'] and result['collision_free']
    assert result['upright'] is False
    assert result['upright_vertical_axis_dot'] == -1.


def test_robot_initialization_rejects_external_obstacles_even_at_right_gripper():
    session, _, contacts = _robot_fixture()
    contacts.append(SimpleNamespace(body0='/World/robot/right_gripper_link1/collision', body1='/World/chair/base/collision'))
    result = collection._robot_initialization(session, {'layout': layout()})
    assert result['floor_supported'] and result['upright']
    assert result['collision_free'] is False
    assert result['unexpected_contact_bodies'] == ['/World/chair/base/collision']
    assert result['allowed_right_hand_plate_contacts'] == []


@pytest.mark.parametrize('link', ['right_eef_link', 'right_gripper_link1', 'right_gripper_link2'])
def test_robot_initialization_allows_only_exact_held_tray_at_right_eef_or_fingers(link):
    session, _, contacts = _robot_fixture()
    # Body ordering must not affect native contact classification.
    contacts.append(SimpleNamespace(body0='/World/tray/base/collision', body1='/World/robot/' + link + '/collision'))
    result = collection._robot_initialization(session, {'layout': layout()})
    assert result['right_hand_holds_declared_plate']
    assert all(result[key] for key in ('floor_supported', 'upright', 'collision_free'))
    assert result['allowed_right_hand_plate_contacts'] == [{
        'robot_body': '/World/robot/' + link + '/collision', 'plate_body': '/World/tray/base/collision'}]
    assert result['unexpected_contact_bodies'] == []


@pytest.mark.parametrize('failure', ['torso', 'right_arm', 'left_hand', 'eef_prefix',
                                    'different_plate_prefix', 'different_held_object', 'not_grasping'])
def test_robot_initialization_does_not_exempt_other_tray_contacts(failure):
    session, objects, contacts = _robot_fixture()
    robot_body, external = '/World/robot/right_gripper_link1/collision', '/World/tray/base/collision'
    if failure == 'torso':
        robot_body = '/World/robot/torso_link/collision'
    elif failure == 'right_arm':
        robot_body = '/World/robot/right_arm_link6/collision'
    elif failure == 'left_hand':
        robot_body = '/World/robot/left_gripper_link1/collision'
    elif failure == 'eef_prefix':
        robot_body = '/World/robot/right_eef_link_extra/collision'
    elif failure == 'different_plate_prefix':
        external = '/World/tray_extra/base/collision'
    elif failure == 'different_held_object':
        session.robot._ag_obj_in_hand['right'] = SimpleNamespace(name=objects['tray'].name)
    else:
        session.robot.grasping = -1
    contacts.append(SimpleNamespace(body0=robot_body, body1=external))
    result = collection._robot_initialization(session, {'layout': layout()})
    assert result['measurement_available']
    assert result['collision_free'] is False
    assert result['unexpected_contact_bodies'] == [external]
    assert result['unexpected_contacts'] == [{'robot_body': robot_body, 'external_body': external}]
    assert result['allowed_right_hand_plate_contacts'] == []


@pytest.fixture
def episode(monkeypatch, tmp_path):
    current = state()
    monkeypatch.setattr(collection, '_state', lambda *args: deepcopy(current))
    config = {'operation': collection.ORDERED_OPERATION, 'parameters': {
        'revision': collection.REVISION, 'operation': collection.ORDERED_OPERATION, 'layout': layout(),
        'targets': ['a', 'b'], 'distractors': [], 'terminal_dwell_s': .5}}
    session, _, _ = _robot_fixture()
    robot_evidence = collection._robot_initialization(session, config['parameters'])
    assert all(robot_evidence[key] for key in ('floor_supported', 'upright', 'collision_free'))
    session.wia_room_collection_initialization = {'accepted': True, 'robot_initialization': robot_evidence}
    evaluator = collection.RoomCollectionEvaluator(session, None, config, tmp_path)
    observation = SimpleNamespace(snapshot_id='s', action_step=0)

    def sample(**changes):
        current.update(native_time_s=current['native_time_s'] + .1, **changes)
        evaluator.sample_step()
        return evaluator.record(observation)

    return evaluator, current, observation, sample


def finish(episode):
    evaluator, current, observation, sample = episode
    sample()
    sample(left_holding='a', left_empty=False, held_objects=['tray', 'a'])
    sample(left_holding=None, left_empty=True, held_objects=['tray'], supported_objects=['a'])
    sample(native_region='bedroom_0')
    sample(left_holding='b', left_empty=False, held_objects=['tray', 'b'])
    sample(left_holding=None, left_empty=True, held_objects=['tray'], supported_objects=['a', 'b'])
    current['native_time_s'] += .6
    return evaluator.record(observation)


def test_ordered_visits_pickups_and_terminal_dwell(episode):
    result = finish(episode)
    assert result['task_success']
    assert result['route_stage'] == result['acquisition_stage'] == 2
    assert result['collected_targets'] == ['a', 'b']
    assert all(row['accepted'] for row in result['acquisitions'])


def test_wrong_room_pickup_cannot_be_redeemed_by_carrying_it_into_room(episode):
    evaluator, current, observation, sample = episode
    sample(native_region='bedroom_0', left_holding='a', left_empty=False)
    result = sample(native_region='kitchen_0')
    assert evaluator.picked == set()
    assert result['acquisitions'][0]['accepted'] is False
    assert not sample(left_holding=None, left_empty=True, supported_objects=['a', 'b'])['task_success']


def test_object_outside_required_room_and_doorway_do_not_earn_pickup(episode):
    evaluator, current, observation, sample = episode
    sample(left_holding='a', left_empty=False, object_rooms={'a': 'bedroom_0', 'b': 'bedroom_0'})
    assert not evaluator.picked
    sample(left_holding=None, left_empty=True)
    sample(left_holding='a', left_empty=False, room_occupancy_valid=False,
           object_rooms={'a': 'kitchen_0', 'b': 'bedroom_0'})
    assert not evaluator.picked
    assert evaluator.route_stage == 1


def test_wrong_pickup_order_rejected_and_requires_new_grasp(episode):
    evaluator, current, observation, sample = episode
    sample(native_region='bedroom_0', left_holding='b', left_empty=False)
    assert evaluator.route_stage == evaluator.acquisition_stage == 0
    assert not evaluator.picked
    sample(left_holding=None, left_empty=True, native_region='kitchen_0')
    sample(left_holding='a', left_empty=False)
    sample(left_holding=None, left_empty=True, supported_objects=['a'])
    sample(native_region='bedroom_0', left_holding='b', left_empty=False)
    sample(left_holding=None, left_empty=True, supported_objects=['a', 'b'])
    current['native_time_s'] += .6
    assert evaluator.record(observation)['task_success']
    assert [row['accepted'] for row in evaluator.acquisitions] == [False, True, True]


@pytest.mark.parametrize('changes', [{'left_empty': False, 'left_holding': 'b', 'held_objects': ['tray', 'b']},
                                    {'right_holds_plate': False}, {'plate_stable': False},
                                    {'plate_level': False}, {'supported_objects': ['a']}])
def test_held_target_unstable_tray_and_spill_fail_current_inventory(episode, changes):
    assert finish(episode)['task_success']
    evaluator, current, observation, sample = episode
    result = sample(**changes)
    assert not result['task_success']
    if changes.get('supported_objects') == ['a']:
        assert evaluator.spills[-1]['target'] == 'b'
        assert result['collection_recall'] == .5


def test_native_dwell_cannot_be_accumulated_by_repeated_records(episode):
    evaluator, current, observation, sample = episode
    finish(episode)
    sample(plate_stable=False)
    sample(plate_stable=True)
    for _ in range(10):
        assert not evaluator.record(observation)['task_success']


def test_search_has_same_physical_goals_without_order(episode):
    evaluator, current, observation, sample = episode
    evaluator.route = []
    evaluator.config['operation'] = collection.SEARCH_OPERATION
    evaluator.parameters['layout']['route'] = []
    sample(native_region='bedroom_0', left_holding='b', left_empty=False)
    sample(left_holding=None, left_empty=True, supported_objects=['b'])
    sample(native_region='kitchen_0', left_holding='a', left_empty=False)
    sample(left_holding=None, left_empty=True, supported_objects=['a', 'b'])
    current['native_time_s'] += .6
    assert evaluator.record(observation)['task_success']


def test_native_inventory_missing_identity_wrong_room_and_fixed_objects_rejected():
    selected = layout()
    snapshot = {'objects': [{**row, 'rooms': [row['room']], 'fixed_base': False, 'visual_only': False}
                            for row in selected['native_inventory']]}
    assert len(collection.resolve_native_inventory(selected, snapshot)) == 2
    for change in ('missing', 'model', 'rooms', 'fixed_base'):
        bad = deepcopy(snapshot)
        if change == 'missing':
            bad['objects'].pop()
        else:
            bad['objects'][0][change] = {'model': 'other', 'rooms': ['bedroom_0'], 'fixed_base': True}[change]
        with pytest.raises(ValueError):
            collection.resolve_native_inventory(selected, bad)
    with pytest.raises(ValueError, match='Each declared route room'):
        selected['native_inventory'].pop()
        collection.validate_layout(selected, collection.ORDERED_OPERATION)


def test_authored_navigation_inventory_requires_explicit_authorization():
    selected = layout()
    selected['added_inventory'] = [selected['native_inventory'].pop()]
    selected['added_inventory'][0].update(position=[0., 0., 0.], orientation=[0., 0., 0., 1.])
    with pytest.raises(ValueError, match='explicitly authorized'):
        collection.validate_layout(selected, collection.ORDERED_OPERATION)
    selected['allow_authored_navigation_inventory'] = True
    assert len(collection.validate_layout(selected, collection.ORDERED_OPERATION)) == 2


def parents():
    patches = {'policy': 'native_then_minimal_patch_else_skip', 'variants': {'original': {'patches': []}}}
    source = {'provider': 'esi', 'json': 'Family/Class/Fixture/full_scene/source.json'}
    task = {'kind': 'category_robot_task', 'id': 'task_parent', 'version': 2, 'status': 'pending',
            'source': source, 'scene': patches, 'public': {'objective': 'Original', 'success_criteria': 'Original',
            'environment': 'Original'}, 'private': {'bindings': {}, 'evaluation': {'type': 'category_operation',
            'module': collection.MODULE, 'operation': 'parent', 'parameters': {}}},
            'requirements': {'capabilities': ['base_translation']}, 'qualification': {'on_failure': 'skip'},
            'defaults': {'seed': 17, 'max_steps': 833, 'wall_timeout_s': 1800.}}
    scene = {'kind': 'scene_patch', 'id': 'scene_parent', 'version': 4, 'source': source,
             'scene': patches, 'bindings': {}, 'seed': 17}
    return task, scene


def test_builder_is_deterministic_preserves_defaults_and_native_snapshot_provenance(tmp_path, monkeypatch):
    task, scene = parents()
    task_file, scene_file, snapshot_file = [tmp_path / name for name in ('parent.task.yaml', 'parent.scene.yaml', 'inventory.json')]
    task_file.write_text(yaml.safe_dump(task))
    scene_file.write_text(yaml.safe_dump(scene))
    snapshot_file.write_text(json.dumps({'objects': layout()['native_inventory']}))
    source_file = tmp_path / 'source.json'
    source_file.write_text(json.dumps({'scene': 'Fixture', 'room': 'full_scene'}))
    monkeypatch.setattr(builder, 'resolve_source', lambda identity: source_file)
    design = {'schema': 'wia_room_collection_design_v1', 'entries': [{
        'document_alias': 'T07', 'variant_id': 'room_collect_v1', 'label': 'Ordered selected cups',
        'parent_task_file': str(task_file), 'parent_scene_file': str(scene_file),
        'scene_inventory_file': str(snapshot_file), 'operation': collection.ORDERED_OPERATION, 'layout': layout()}]}
    design_file = tmp_path / 'design.json'
    design_file.write_text(json.dumps(design))
    definitions, manifest, rendered = builder.build_package(design_file, tmp_path / 'out')
    assert builder.build_package(design_file, tmp_path / 'out')[2] == rendered
    generated = yaml.safe_load(rendered[definitions[0]['task_file']])
    assert generated['defaults'] == task['defaults']
    assert manifest['schema'] == 'wia_variant_manifest_v1'
    assert manifest['entries'][0]['scene_inventory_sha256']
    assert generated['repair']['native_inventory_spawned'] is False
    assert generated['repair']['original_source_qa_unused'] is True
    patches = generated['scene']['variants']['adapted']['patches']
    assert len(patches) == 4
    assert [row['parameters']['mode'] for row in patches if row['module'].endswith('wia_background')] == ['complete_ground'] * 2
    assert generated['version'] == 3
    public = json.dumps(generated['public'])
    assert 'red cup' in public and 'kitchen 0' in public and 'position' not in public
    design['entries'][0]['task_defaults'] = {'max_steps': 999}
    updated, _ = builder.build_definitions(task, scene, design['entries'][0], {})
    assert updated['defaults'] == {**task['defaults'], 'max_steps': 999}


def test_added_inventory_keeps_explicit_scale_or_bounding_box():
    row = {**layout()['native_inventory'][0], 'position': [0., 0., 1.], 'orientation': [0., 0., 0., 1.]}
    for field in ('scale', 'bounding_box'):
        authored = {**row, field: [.2, .3, .4]}
        assert collection._added_object_spec(authored)[field] == [.2, .3, .4]
        selected = {**layout(), 'native_inventory': [], 'added_inventory': [authored], 'route': []}
        collection.validate_layout(selected, collection.SEARCH_OPERATION)
    authored.update(scale=[1., 1., 1.])
    with pytest.raises(ValueError, match='never both'):
        collection.validate_layout(selected, collection.SEARCH_OPERATION)


def test_native_constructor_snapshot_shape_is_resolved():
    snapshot = {'objects_info': {'init_info': {row['name']: {'args': row}
                for row in layout()['native_inventory']}, 'state': {}}}
    assert len(collection.resolve_native_inventory(layout(), snapshot)) == 2


def test_target_support_is_explicit_and_settled_checks_reject_bad_support():
    selected = layout()
    selected['native_inventory'][0].pop('support')
    with pytest.raises(ValueError, match='explicit support'):
        collection.validate_layout(selected, collection.ORDERED_OPERATION)
    obj = SimpleNamespace(aabb=[np.array([0., 0., 1.]), np.array([.1, .1, 1.1])],
                          get_linear_velocity=lambda: np.zeros(3))
    support = SimpleNamespace(name='table', aabb=[np.array([-1., -1., .8]), np.array([1., 1., 1.])])
    assert collection._support_evidence(obj, support, {}, True)['accepted']
    assert not collection._support_evidence(obj, support, {}, False)['accepted']
    obj.aabb = [np.array([0., 0., 1.1]), np.array([.1, .1, 1.2])]
    assert not collection._support_evidence(obj, support, {}, True)['accepted']
    obj.aabb = [np.array([2., 0., 1.]), np.array([2.1, .1, 1.1])]
    assert not collection._support_evidence(obj, support, {}, True)['accepted']
    obj.aabb = [np.array([0., 0., 1.]), np.array([.1, .1, 1.1])]
    obj.get_linear_velocity = lambda: np.array([.03, 0., 0.])
    assert not collection._support_evidence(obj, support, {}, True)['accepted']


def _initialization_fixture(monkeypatch, parent_accepted=True):
    touching = object()
    monkeypatch.setitem(sys.modules, 'omnigibson.object_states', SimpleNamespace(Touching=touching))
    monkeypatch.setitem(sys.modules, 'wia.runtime.sim.esi_collection_setup',
                        SimpleNamespace(initialize_collection_tray=lambda *args: {'accepted': parent_accepted}))
    session, objects, _ = _robot_fixture()
    for index, row in enumerate(layout()['native_inventory']):
        x = index * 2.
        objects[row['support']] = SimpleNamespace(name=row['support'], category='countertop',
            aabb=[np.array([x - 1., -1., .8]), np.array([x + 1., 1., 1.])])
        obj = SimpleNamespace(**{key: row[key] for key in ('name', 'category', 'model')}, fixed_base=False,
            visual_only=False, aabb=[np.array([x, 0., 1.]), np.array([x + .1, .1, 1.1])],
            states={touching: SimpleNamespace(get_value=lambda support: True)},
            get_linear_velocity=lambda: np.zeros(3),
            get_position_orientation=lambda x=x: (np.array([x, 0., 1.05]), np.array([0., 0., 0., 1.])))
        objects[row['name']] = obj  # No setter: unintended native alignment would fail.
    session._category_initialization_active = True
    monkeypatch.setattr(collection, '_region', lambda session, position: 'kitchen_0' if position[0] < 1. else 'bedroom_0')
    return session, objects, touching


def test_initialization_measures_unmodified_native_targets_without_repositioning(monkeypatch):
    session, _, _ = _initialization_fixture(monkeypatch)
    result = collection.initialize_scene(session, {'layout': layout(), 'operation': collection.ORDERED_OPERATION})
    assert result['accepted']
    assert result['support_alignments'] == result['explicit_relocations'] == []
    assert all(row['accepted'] for row in result['inventory_supports'].values())
    assert all(result['robot_initialization'][key] for key in ('floor_supported', 'upright', 'collision_free'))


@pytest.mark.parametrize('oversized', [False, True])
def test_tray_initialization_failure_propagates_bounded_measured_diagnostics(monkeypatch, oversized):
    session, _, _ = _initialization_fixture(monkeypatch)
    session.robot.grasping = -1
    original = ValueError('Contact-qualified right-hand tray grasp was not established.')
    tray_receipt = {'accepted': False, 'stage': 'closing_right_gripper', 'right_eef_residual_m': .001,
                    'closure_steps': 120, 'plate_start_position': [1., 2., 1.], 'pinch': [1., 2., 1.1],
                    'closure': [{'step': index, 'holding': None} for index in range(5)],
                    'contacts': [{'body0': '/World/robot/right_eef_link',
                                  'body1': '/World/tray/' + ('x' * 1000 if oversized else str(index))}
                                 for index in range(15)], 'unrequested_private_field': 'omit'}
    before = deepcopy(tray_receipt)
    calls = []

    def fail_tray(current_session, parameters):
        calls.append('initialize_tray')
        current_session.hidden_collection_initialization = tray_receipt
        raise original

    monkeypatch.setitem(sys.modules, 'wia.runtime.sim.esi_collection_setup',
                        SimpleNamespace(initialize_collection_tray=fail_tray))
    with pytest.raises(ValueError) as caught:
        collection.initialize_scene(session, {'layout': layout(), 'operation': collection.ORDERED_OPERATION})
    assert caught.value.__cause__ is original
    message = str(caught.value)
    assert message.startswith(str(original))
    assert len(message) <= 6000
    assert calls == ['initialize_tray']  # Diagnostic propagation must not retry setup.
    assert session.hidden_collection_initialization is tray_receipt
    assert tray_receipt == before
    if oversized:
        diagnostic = json.loads(message.split('; tray_initialization_diagnostic=', 1)[1])
        assert diagnostic['diagnostic_compacted'] is True
        assert 'robot_initialization' in diagnostic
        assert 'right_gripper_diagnostics' in diagnostic
    else:
        diagnostic = json.loads(message.split('; tray_initialization_diagnostic=', 1)[1])
        assert diagnostic['stage'] == tray_receipt['stage']
        assert diagnostic['accepted'] is False
        assert diagnostic['closure'] == tray_receipt['closure'][-2:]
        assert diagnostic['contacts'] == tray_receipt['contacts'][:12]
        assert diagnostic['right_eef_residual_m'] == .001
        assert diagnostic['plate_start_position'] == tray_receipt['plate_start_position']
        assert diagnostic['pinch'] == tray_receipt['pinch']
        assert diagnostic['robot_initialization']['floor_supported']
        assert diagnostic['robot_initialization']['right_hand_holds_declared_plate'] is False
        assert 'unrequested_private_field' not in diagnostic
        for field in ('gripper_contacts', 'gripper_raycast_collisions', 'ag_grasp_counter_right',
                      'ag_release_counter_right', 'grasping_mode', 'gripper_right_control', 'right_eef_pose'):
            assert 'error' in diagnostic['right_gripper_diagnostics'][field]


@pytest.mark.parametrize('nonfinite_control', [False, True])
def test_tray_failure_reads_gripper_diagnostics_once_without_attempting_grasp(monkeypatch, nonfinite_control):
    session, _, _ = _initialization_fixture(monkeypatch)
    robot, calls = session.robot, []
    candidates = {'/World/tray/base_link', '/World/other/base_link'}
    links = {path: {'/World/robot/right_gripper_link1', '/World/robot/right_gripper_link2'} for path in candidates}

    def contacts(*, arm):
        calls.append(('contacts', arm))
        return candidates, links

    def raycasts(*, arm):
        calls.append(('raycasts', arm))
        return {'/World/tray/base_link'}

    def forbidden(*args, **kwargs):
        raise AssertionError('Failure diagnostics must not attempt grasping or advance physics.')

    def link_pose(name):
        calls.append(('pose', name))
        return np.array([1., 2., 3.]), np.array([0., 0., 0., 1.])

    robot._find_gripper_contacts, robot._find_gripper_raycast_collisions = contacts, raycasts
    robot._calculate_in_hand_object = robot._establish_grasp = robot.apply_action = session.env.step = forbidden
    robot._ag_grasp_counter, robot._ag_release_counter = {'right': 12}, {'right': None}
    robot.grasping_mode = 'assisted'
    robot.controllers = {'gripper_right': SimpleNamespace(control=np.array([float('nan') if nonfinite_control else -.01, .02]))}
    for index, link in enumerate([robot.eef_links['right'], *robot.finger_links['right']]):
        link.get_position_orientation = lambda index=index: link_pose(str(index))
    counters_before = deepcopy((robot._ag_grasp_counter, robot._ag_release_counter))
    held_before = dict(robot._ag_obj_in_hand)
    source = ValueError('Contact-qualified right-hand tray grasp was not established.')

    def fail_tray(current_session, parameters):
        calls.append(('initialize', 'right'))
        current_session.hidden_collection_initialization = {'accepted': False, 'stage': 'closing_right_gripper'}
        raise source

    monkeypatch.setitem(sys.modules, 'wia.runtime.sim.esi_collection_setup', SimpleNamespace(initialize_collection_tray=fail_tray))
    with pytest.raises(ValueError) as caught:
        collection.initialize_scene(session, {'layout': layout(), 'operation': collection.ORDERED_OPERATION})
    assert caught.value.__cause__ is source
    assert str(caught.value).startswith(str(source))
    assert len(str(caught.value)) <= 6000
    diagnostic = json.loads(str(caught.value).split('; tray_initialization_diagnostic=', 1)[1])
    detail = diagnostic['right_gripper_diagnostics']
    assert detail['gripper_contacts']['candidate_prim_paths'] == sorted(candidates)
    assert detail['gripper_contacts']['contact_link_names']['/World/tray/base_link'] == ['right_gripper_link1', 'right_gripper_link2']
    assert detail['gripper_raycast_collisions'] == {'prim_paths': ['/World/tray/base_link'], 'count': 1}
    assert detail['ag_grasp_counter_right'] == 12 and detail['ag_release_counter_right'] is None
    assert detail['grasping_mode'] == 'assisted'
    if nonfinite_control:
        assert 'finite native measurements' in detail['gripper_right_control']['error']
    else:
        assert detail['gripper_right_control'] == {'values': [-.01, .02], 'count': 2}
    assert detail['right_eef_pose']['position_m'] == [1., 2., 3.]
    assert len(detail['right_finger_poses']) == 2
    assert 'robot_initialization' in diagnostic
    assert calls.count(('initialize', 'right')) == 1
    assert calls.count(('contacts', 'right')) == calls.count(('raycasts', 'right')) == 1
    assert all(calls.count(('pose', str(index))) == 1 for index in range(3))
    assert (robot._ag_grasp_counter, robot._ag_release_counter) == counters_before
    assert robot._ag_obj_in_hand == held_before
    assert robot.grasping_mode == 'assisted'
    assert links['/World/tray/base_link'] == {'/World/robot/right_gripper_link1', '/World/robot/right_gripper_link2'}


@pytest.mark.parametrize('tray_succeeds', [False, True])
def test_explicit_unqualified_preview_measures_scene_but_never_qualifies_or_scores(monkeypatch, tmp_path, tray_succeeds):
    session, objects, touching = _initialization_fixture(monkeypatch)
    calls = []
    original = ValueError('Contact-qualified right-hand tray grasp was not established.')
    native_receipt = {'accepted': tray_succeeds, 'stage': 'complete' if tray_succeeds else 'closing_right_gripper',
                      'closure': [{'step': 115, 'holding': None}], 'plate_start_position': [1., 2., 1.]}
    before = deepcopy(native_receipt)

    def forbidden(*args, **kwargs):
        calls.append('forbidden_mutation')
        raise AssertionError('Preview must not add control, physics, placement or support removal.')

    session.env.step = session.env.scene.remove_object = forbidden
    session.robot.apply_action = session.robot.set_position_orientation = forbidden

    def initialize_tray(current_session, parameters):
        calls.append('initialize_tray')
        current_session.hidden_collection_initialization = native_receipt
        if not tray_succeeds:
            raise original
        return native_receipt

    if not tray_succeeds:
        session.robot.grasping = -1
        objects['b'].states[touching].get_value = lambda support: False
        monkeypatch.setattr(collection, '_region', lambda *args: 'kitchen_0')
    monkeypatch.setitem(sys.modules, 'wia.runtime.sim.esi_collection_setup',
                        SimpleNamespace(initialize_collection_tray=initialize_tray))
    task_parameters = {'layout': layout(), 'operation': collection.ORDERED_OPERATION,
                       'targets': ['a', 'b'], 'distractors': [], 'terminal_dwell_s': .5}
    scene_parameters = {**task_parameters, 'unqualified_tray_preview': True}
    result = collection.initialize_scene(session, scene_parameters)
    assert calls == ['initialize_tray']
    assert native_receipt == before
    assert result['accepted'] is False
    assert result['stage'] == 'unqualified_tray_preview'
    assert result['initialization_preview_only'] is True
    assert result['complete_collection_verified'] is False
    assert result['native_layout_qualified'] is False
    assert result['inventory']['a']['position_m'] == [0., 0., 1.05]
    assert result['robot_initialization']['measurement_available']
    assert session.category_interaction_setup is session.wia_room_collection_initialization is result
    assert 'Unqualified scene preview only' in result['reason']
    if tray_succeeds:
        assert all(row['accepted'] for row in result['inventory_supports'].values())
        assert all(row['room_matches'] for row in result['inventory'].values())
        assert 'tray_initialization_error' not in result
    else:
        assert result['inventory']['b']['native_room'] == 'kitchen_0'
        assert result['inventory']['b']['room_matches'] is False
        assert result['inventory_supports']['b']['touching'] is False
        assert result['inventory_supports']['b']['accepted'] is False
        assert 'b:table_bedroom_0' in result['reason']
        assert '"failed_inventory_rooms":["b"]' in result['reason']
        assert result['tray_initialization_error'].startswith(str(original))
        assert len(result['tray_initialization_error']) <= 6000
        assert result['tray_initialization_diagnostic']['stage'] == 'closing_right_gripper'
        assert 'robot_initialization' in result['tray_initialization_diagnostic']
    json.dumps(result, allow_nan=False)
    # The task contract has no preview flag; rejection follows the scene receipt.
    assert 'unqualified_tray_preview' not in task_parameters
    config = {'operation': collection.ORDERED_OPERATION, 'parameters': task_parameters}
    monkeypatch.setattr(collection, '_state', lambda *args: state())
    qualification = collection.qualify_operation(session, SimpleNamespace(controller=SimpleNamespace(working_arm='left')),
                                                  config, {})
    assert qualification['accepted'] is False
    assert qualification['checks']['initialization_receipt'] is False
    evaluator = collection.RoomCollectionEvaluator(session, None, config, tmp_path)
    evaluator.picked = {'a', 'b'}
    evaluator.route_stage = evaluator.acquisition_stage = 2
    evaluator.state = state(supported_objects=['a', 'b'])
    score = evaluator._result(1., .5)
    assert score['collection_recall'] == 1.
    assert score['checks']['initialization_receipt'] is False
    assert score['task_success'] is False
    monkeypatch.setattr(collection, '_state', lambda *args: state(supported_objects=['a', 'b']))
    recorded = evaluator.record(SimpleNamespace(snapshot_id='preview', action_step=0))
    assert recorded['task_success'] is False
    assert not any(item['completed'] for item in recorded['completion_items'])
    assert calls == ['initialize_tray']


def test_preview_continuation_requires_explicit_boolean_true(monkeypatch):
    session, _, _ = _initialization_fixture(monkeypatch)
    original = ValueError('Contact-qualified right-hand tray grasp was not established.')

    def fail(*args):
        raise original

    monkeypatch.setitem(sys.modules, 'wia.runtime.sim.esi_collection_setup', SimpleNamespace(initialize_collection_tray=fail))
    with pytest.raises(ValueError) as caught:
        collection.initialize_scene(session, {'layout': layout(), 'operation': collection.ORDERED_OPERATION,
                                              'unqualified_tray_preview': 'true'})
    assert caught.value.__cause__ is original
    assert str(caught.value).startswith(str(original))
    assert not hasattr(session, 'wia_room_collection_initialization')


@pytest.mark.parametrize('failure', ['floor_supported', 'upright', 'collision_free', 'unavailable'])
def test_robot_safety_cannot_reject_measured_task_initialization(monkeypatch, failure):
    session, _, _ = _initialization_fixture(monkeypatch)
    contacts = session.robot.contact_list()
    if failure == 'floor_supported':
        contacts[0].position[2] = .02
    elif failure == 'upright':
        session.robot.orientation = [1., 0., 0., 0.]
    elif failure == 'collision_free':
        contacts.append(SimpleNamespace(body0='/World/robot/torso_link', body1='/World/chair/base'))
        session.robot.contact_list = lambda: contacts
    else:
        session.env.scene.trav_map.floor_heights = []
    result = collection.initialize_scene(session, {'layout': layout(), 'operation': collection.ORDERED_OPERATION})
    assert result['accepted'] is True
    assert all(row['accepted'] for row in result['inventory_supports'].values())
    assert all(row['room_matches'] for row in result['inventory'].values())
    assert session.category_interaction_setup is result
    assert session.wia_room_collection_initialization is result
    if failure == 'unavailable':
        assert result['robot_initialization']['measurement_available'] is False
        assert 'reason' not in result
    else:
        assert result['robot_initialization'][failure] is False
        assert 'reason' not in result


@pytest.mark.parametrize('failure', ['support', 'room', 'parent'])
def test_failed_initialization_receipt_survives_but_cannot_qualify_or_complete(monkeypatch, tmp_path, failure):
    session, objects, touching = _initialization_fixture(monkeypatch, parent_accepted=failure != 'parent')
    if failure == 'support':
        objects['b'].states[touching].get_value = lambda support: False
    elif failure == 'room':
        monkeypatch.setattr(collection, '_region', lambda *args: 'kitchen_0')
    parameters = {'layout': layout(), 'operation': collection.ORDERED_OPERATION,
                  'targets': ['a', 'b'], 'distractors': [], 'terminal_dwell_s': .5}
    result = collection.initialize_scene(session, parameters)
    assert result['accepted'] is False
    assert session.category_interaction_setup is result
    assert session.wia_room_collection_initialization is result
    assert 'inventory' in result and 'inventory_supports' in result
    assert len(result['reason']) <= 1800
    if failure == 'support':
        assert result['inventory_supports']['b']['touching'] is False
        assert result['inventory_supports']['b']['accepted'] is False
        assert 'b:table_bedroom_0' in result['reason']
    elif failure == 'room':
        assert result['inventory']['b']['room_matches'] is False
        assert '"failed_inventory_rooms":["b"]' in result['reason']
    else:
        assert '"parent_initialization_accepted":false' in result['reason']
    monkeypatch.setattr(collection, '_state', lambda *args: state())
    config = {'operation': collection.ORDERED_OPERATION, 'parameters': parameters}
    executor = SimpleNamespace(controller=SimpleNamespace(working_arm='left'))
    qualification = collection.qualify_operation(session, executor, config, {})
    assert qualification['accepted'] is False
    assert qualification['checks']['initialization_receipt'] is False
    evaluator = collection.RoomCollectionEvaluator(session, None, config, tmp_path)
    evaluator.picked = {'a', 'b'}
    evaluator.route_stage = evaluator.acquisition_stage = 2
    evaluator.state = state(supported_objects=['a', 'b'])
    score = evaluator._result(1., .5)
    assert score['collection_recall'] == 1.
    assert score['checks']['all_targets_collected'] is True
    assert score['checks']['initialization_receipt'] is False
    assert score['task_success'] is False
    monkeypatch.setattr(collection, '_state', lambda *args: state(supported_objects=['a', 'b']))
    recorded = evaluator.record(SimpleNamespace(snapshot_id='s', action_step=0))
    assert recorded['evaluation_validity'] == 'invalid_initialization'
    assert not any(item['completed'] for item in recorded['completion_items'])
    assert session.category_interaction_setup is result


def test_missing_initialization_receipt_cannot_succeed(episode):
    assert finish(episode)['task_success']
    evaluator, _, observation, _ = episode
    del evaluator.session.wia_room_collection_initialization
    result = evaluator.record(observation)
    assert result['checks']['initialization_receipt'] is False
    assert result['task_success'] is False
    assert result['collected_targets'] == ['a', 'b']


@pytest.mark.parametrize('failure', ['missing_object', 'nonfinite_bounds', 'nonfinite_velocity', 'nonfinite_position'])
def test_unavailable_or_nonfinite_native_inventory_still_raises(monkeypatch, failure):
    session, objects, _ = _initialization_fixture(monkeypatch)
    if failure == 'missing_object':
        objects.pop('b')
    elif failure == 'nonfinite_bounds':
        objects['b'].aabb[0][2] = float('nan')
    elif failure == 'nonfinite_velocity':
        objects['b'].get_linear_velocity = lambda: np.array([float('nan'), 0., 0.])
    else:
        objects['b'].get_position_orientation = lambda: (np.array([float('nan'), 0., 0.]), np.array([0., 0., 0., 1.]))
    with pytest.raises(ValueError):
        collection.initialize_scene(session, {'layout': layout(), 'operation': collection.ORDERED_OPERATION})


def test_source_override_resolves_new_scene_and_room_consistently(tmp_path, monkeypatch):
    task, scene = parents()
    paths = [tmp_path / name for name in ('parent.task.yaml', 'parent.scene.yaml')]
    for path, value in zip(paths, (task, scene)):
        path.write_text(yaml.safe_dump(value))
    source = tmp_path / 'authored.json'
    source.write_text(json.dumps({'scene': 'NewScene', 'room': 'full_scene'}))
    monkeypatch.setattr(builder, 'resolve_source', lambda identity: source)
    selected = layout()
    selected.update(scene_model='NewScene', route=[], native_inventory=[],
                    added_inventory=[{**layout()['native_inventory'][0], 'position': [0., 0., 1.],
                                      'orientation': [0., 0., 0., 1.]}])
    entry = {'document_alias': 'T03', 'variant_id': 'new_search', 'label': 'New scene search',
             'parent_task_file': str(paths[0]), 'parent_scene_file': str(paths[1]),
             'source_identity': 'Authored/Search/NewScene/full_scene/source.json', 'room': 'full_scene',
             'operation': collection.SEARCH_OPERATION, 'layout': selected}
    design_file = tmp_path / 'design.json'
    design_file.write_text(json.dumps({'schema': 'wia_room_collection_design_v1', 'entries': [entry]}))
    definitions, manifest, rendered = builder.build_package(design_file, tmp_path / 'out')
    generated = yaml.safe_load(rendered[definitions[0]['task_file']])
    assert generated['source']['json'] == entry['source_identity']
    assert manifest['entries'][0]['scene_model'] == 'NewScene'
    entry['room'] = 'other_room'
    design_file.write_text(json.dumps({'schema': 'wia_room_collection_design_v1', 'entries': [entry]}))
    with pytest.raises(ValueError, match='source scene/room'):
        builder.build_package(design_file, tmp_path / 'out')


def test_source_props_replaced_only_by_explicit_policy_and_native_scene_is_preserved(tmp_path, monkeypatch):
    (tmp_path / 'r1_primitives.yaml').write_text(yaml.safe_dump({'robots': [{'reset_joint_pos': [0.] * 10}]}))
    monkeypatch.setitem(sys.modules, 'wia.runtime.sim.esi_omnigibson', SimpleNamespace(
        DEFAULT_ESI_ROOT=tmp_path,
        _import_pipeline=lambda root: SimpleNamespace(og=SimpleNamespace(example_config_path=str(tmp_path)))))
    selected = layout()
    selected.update(route=[], native_inventory=[], added_inventory=[{
        **layout()['native_inventory'][0], 'position': [0., 0., 1.],
        'orientation': [0., 0., 0., 1.], 'scale': [.65] * 3}],
        source_objects_policy='replace_authored_source_objects')
    scene = {'scene_model': 'Fixture', 'type': 'InteractiveTraversableScene',
             'load_room_instances': None, 'existing_native_geometry_setting': 'preserve'}
    config = {'scene': deepcopy(scene), 'objects': [{'name': 'legacy_source_prop'}]}
    parameters = {'revision': collection.REVISION, 'operation': collection.SEARCH_OPERATION, 'layout': selected}
    receipt = collection.configure_scene(config, parameters)
    assert config['scene'] == scene
    assert 'legacy_source_prop' not in {row['name'] for row in config['objects']}
    assert next(row for row in config['objects'] if row['name'] == 'a')['scale'] == [.65] * 3
    assert receipt['replaced_authored_source_objects'] == ['legacy_source_prop']
    selected.pop('source_objects_policy')
    config = {'scene': deepcopy(scene), 'objects': [{'name': 'legacy_source_prop'}]}
    collection.configure_scene(config, parameters)
    assert 'legacy_source_prop' in {row['name'] for row in config['objects']}
    selected['source_objects_policy'] = 'replace_everything'
    with pytest.raises(ValueError, match='source object policy'):
        collection.validate_layout(selected, collection.SEARCH_OPERATION)
