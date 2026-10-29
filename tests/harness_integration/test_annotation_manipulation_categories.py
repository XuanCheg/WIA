"""Check annotated manipulation repairs without simulator startup or GPU work.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python -m pytest -q tests/test_annotation_manipulation_categories.py
Native-shaped fixtures check predicate boundaries; they do not prove real grasp.
"""
import copy
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from wia.authoring.conversion import ConversionContext, ROOT, SourceInstance
from wia.authoring.converters import action_order
from wia.authoring.converters import physical_dynamics_inclined_plane as inclined
from wia.authoring.converters import physical_dynamics_stacking_stability as stacking
from wia.authoring.converters import spatial_relations_physical_contact as contact
from wia.authoring.converters.metric_comparison_spatial_distance import resolve_fixed_distance_reference
from wia.tasks.loader import task_definition


EMPTY_GRIPPER = {'object_name': None, 'holding': False, 'constraint_active': False,
                 'gripper_open': True, 'release_pending': False}
MODULES = ['action_order', 'dimensional_size', 'metric_comparison_spatial_distance',
           'physical_dynamics_inclined_plane', 'physical_dynamics_stacking_stability',
           'spatial_relations_physical_contact']


@pytest.fixture
def native_types(monkeypatch):
    states = ModuleType('omnigibson.object_states')
    states.Touching, states.OnTop = type('Touching', (), {}), type('OnTop', (), {})
    monkeypatch.setitem(sys.modules, 'omnigibson', ModuleType('omnigibson'))
    monkeypatch.setitem(sys.modules, 'omnigibson.object_states', states)
    return states


class NativeObject:
    def __init__(self, name, position, extent=(.1, .1, .1), *, category='prop', model='asset'):
        self.name, self.category, self.model = name, category, model
        self.position, self.extent = np.asarray(position, dtype=float), np.asarray(extent, dtype=float)
        self.quaternion, self.scale = np.array([0., 0., 0., 1.]), np.ones(3)
        self.prim_path = '/World/' + name
        self.visual_only, self.fixed_base, self.mass = False, False, 1.
        self.contacts, self.states = [], {}
        self.links = {'body': SimpleNamespace(contact_reporting_enabled=True)}

    @property
    def aabb(self):
        return self.position - self.extent / 2, self.position + self.extent / 2

    def get_position_orientation(self):
        return self.position.copy(), self.quaternion.copy()

    def get_linear_velocity(self):
        return np.zeros(3)

    def get_angular_velocity(self):
        return np.zeros(3)

    def contact_list(self):
        return list(self.contacts)


def put_on(obj, support, states, *, z=.7):
    obj.states = {states.Touching: SimpleNamespace(get_value=lambda other: other is support),
                  states.OnTop: SimpleNamespace(get_value=lambda other: other is support)}
    obj.contacts = [SimpleNamespace(body0=obj.prim_path + '/body', body1=support.prim_path + '/top',
                                   normal=np.array([0., 0., 1.]), position=np.array([*obj.position[:2], z]))]


def session_with(objects, props, *, delta=(0., 0., .7)):
    by_name = {obj.name: obj for obj in objects}
    scene = SimpleNamespace(objects=by_name, object_registry=lambda field, name: by_name.get(name))
    return SimpleNamespace(scene_name='source_scene', room_name='room_0', env=SimpleNamespace(scene=scene),
        category_interaction_setup={'accepted': True, 'relocated': True,
            'object_names': [obj.name for obj in props], 'support_prim_path': '/World/table',
            'support_surface_z_m': .7, 'group_translation_m': list(delta),
            'object_translations_m': {obj.name: list(delta) for obj in props}})


@pytest.fixture
def incline_trial(native_types, monkeypatch):
    floor = NativeObject('floor', [0., 0., -.05], (10., 10., .1), category='floors')
    table = NativeObject('table', [1., 1., .35], (2., 2., .7), category='table')
    slope = NativeObject('slope', [0., 0., .05], (1., 1., .1))
    slope.fixed_base = True
    target = NativeObject('task_obj', [1., 1., .75])
    put_on(target, table, native_types)
    session = session_with([floor, table, slope, target], [target])
    parameters = {'slope_position': slope.position.tolist(), 'slope_quaternion': slope.quaternion.tolist(),
                  'slope_scale': slope.scale.tolist(), 'target_floor_position': [1., 1., .05],
                  'floor_z_m': 0., 'static_friction': 1., 'dynamic_friction': .5}
    config = {'type': 'category_operation', 'module': inclined.MODULE, 'operation': inclined.OPERATION,
              'scene': session.scene_name, 'room': session.room_name,
              'bindings': {'target': target.name, 'slope': slope.name, 'floor': floor.name},
              'parameters': parameters}
    monkeypatch.setattr(inclined, '_robot_state', lambda session: {
        'in_original_room': True, 'floor_contacts': [['floor']], 'obstacle_contacts': []})
    monkeypatch.setattr(inclined, '_friction', lambda obj: [{'static_friction': 1., 'dynamic_friction': .5}])
    bound = {'target': target, 'slope': slope, 'floor': floor}
    executor = SimpleNamespace(debug_gripper_state=lambda: dict(EMPTY_GRIPPER))
    return session, executor, config, bound, table


def test_inclined_pickup_can_start_on_verified_adapted_support(incline_trial):
    session, executor, config, bound, _ = incline_trial
    result = inclined.qualify_operation(session, executor, config, bound)
    assert result['accepted'], result
    assert result['initial_adapted_support_contact'] is True
    assert result['initial_target_reference_m'] == pytest.approx(bound['target'].position)
    assert result['real_robot_placement_qualified'] is False
    # The operation trace still flags this same table as post-release interference.
    row, _ = inclined._trial_state(bound['target'], bound['slope'], bound['floor'], config['parameters'])
    assert row['interfering_contacts'] == [['/World/table/top']]
    assert not row['top_supported'] and not row['touching_floor']


@pytest.mark.parametrize('failure', ['unaccepted_setup', 'side_contact', 'wrong_contact_height',
                                   'ramp_moved', 'floor_changed', 'wall_contact'])
def test_inclined_support_adaptation_keeps_physical_and_anchor_guards(incline_trial, failure):
    session, executor, config, bound, _ = incline_trial
    if failure == 'unaccepted_setup':
        session.category_interaction_setup['accepted'] = False
    elif failure == 'side_contact':
        bound['target'].contacts[0].normal = np.array([1., 0., 0.])
    elif failure == 'wrong_contact_height':
        bound['target'].contacts[0].position[2] = .9
    elif failure == 'ramp_moved':
        bound['slope'].position[0] += .1
    elif failure == 'floor_changed':
        bound['floor'].position[2] += .1
    else:
        bound['target'].contacts.append(SimpleNamespace(body0='/World/task_obj/body',
            body1='/World/wall/body', normal=np.array([1., 0., 0.]), position=np.array([1., 1., .75])))
    assert inclined.qualify_operation(session, executor, config, bound)['accepted'] is False


def test_legacy_original_floor_inclined_precondition_is_still_callable(incline_trial, native_types):
    session, executor, config, bound, _ = incline_trial
    session.category_interaction_setup = {}
    bound['target'].position[2] -= .7
    put_on(bound['target'], bound['floor'], native_types, z=0.)
    result = inclined.qualify_operation(session, executor, config, bound)
    assert result['accepted'], result
    assert result['initial_adapted_support_contact'] is False


def test_stacking_uses_adapted_references_and_current_support_without_stack_credit(native_types, monkeypatch):
    floor = NativeObject('floor', [0., 0., -.05], (10., 10., .1), category='floors')
    table = NativeObject('table', [0., 0., .35], (3., 3., .7), category='table')
    props = [NativeObject(key, [index * .3, 0., .75]) for index, key in enumerate(stacking.OBJECT_KEYS)]
    for obj in props:
        put_on(obj, table, native_types)
    session = session_with([floor, table, *props], props)
    originals = {obj.name: {'name': obj.name, 'category': obj.category, 'model': obj.model,
        'scale': obj.scale.tolist(), 'position': (obj.position - [0., 0., .7]).tolist(),
        'quaternion_xyzw': obj.quaternion.tolist()} for obj in props}
    config = {'type': 'category_operation', 'module': stacking.MODULE, 'operation': stacking.OPERATION,
        'scene': session.scene_name, 'room': session.room_name,
        'bindings': {**{obj.name: obj.name for obj in props}, 'floor': floor.name},
        'parameters': {'original_objects': originals}}
    monkeypatch.setattr(stacking, '_gripper_state', lambda session, executor: {'all_arms_free': True})
    monkeypatch.setattr(stacking, '_robot_start_checks', lambda session: {
        'in_original_room': True, 'has_floor_contact': True, 'no_obstacle_contact': True})
    bound = {**{obj.name: obj for obj in props}, 'floor': floor}
    result = stacking.qualify_operation(session, None, config, bound)
    assert result['accepted'], result
    assert result['physical_stack_executed'] is False
    props[0].scale[0] = .5
    assert not stacking.qualify_operation(session, None, config, bound)['accepted']


def test_contact_keeps_common_pair_transform_and_native_released_support(native_types, monkeypatch):
    floor = NativeObject('floor', [0., 0., -.05], (10., 10., .1), category='floors')
    table = NativeObject('table', [0., 0., .35], (3., 3., .7), category='table')
    props = [NativeObject(key, [index * .3, 0., .75]) for index, key in enumerate(contact.OBJECTS)]
    for obj in props:
        put_on(obj, table, native_types)
    session = session_with([floor, table, *props], props)
    source_objects = {obj.name: {'position': (obj.position - [0., 0., .7]).tolist(),
        'quaternion_xyzw': obj.quaternion.tolist(), 'aabb_min': (obj.aabb[0] - [0., 0., .7]).tolist(),
        'aabb_max': (obj.aabb[1] - [0., 0., .7]).tolist()} for obj in props}
    config = {'type': 'category_operation', 'module': contact.MODULE, 'operation': contact.OPERATIONS[0],
        'scene': session.scene_name, 'room': session.room_name, 'binding_keys': [*contact.OBJECTS, 'source_floor'],
        'bindings': {**{obj.name: obj.name for obj in props}, 'source_floor': floor.name},
        'parameters': {'target': contact.OBJECTS[0], 'anchor': contact.OBJECTS[1], 'source_objects': source_objects}}
    bound = {**{obj.name: obj for obj in props}, 'source_floor': floor}
    monkeypatch.setattr(contact, '_robot_state', lambda session, floor: {'in_source_room': True, 'supported_upright': True})
    executor = SimpleNamespace(debug_gripper_state=lambda: dict(EMPTY_GRIPPER))
    result = contact.qualify_operation(session, executor, config, bound)
    assert result['accepted'], result
    assert result['native_pair']['separate'] is True
    assert result['physical_operation_execution_verified'] is False
    session.category_interaction_setup['object_translations_m'][props[1].name][0] = .1
    assert not contact.qualify_operation(session, executor, config, bound)['accepted']


def test_action_order_native_supports_are_measured_from_current_table_not_raw_floor(native_types):
    table = NativeObject('table', [0., 0., .35], (3., 3., .7), category='table')
    robot = NativeObject('robot', [1., 1., .4])
    robot.arm_names = ('left', 'right')
    robot._ag_obj_in_hand = {arm: None for arm in robot.arm_names}
    robot._ag_obj_constraints = dict(robot._ag_obj_in_hand)
    robot._ag_release_counter = dict(robot._ag_obj_in_hand)
    robot.is_grasping = lambda arm: -1
    keys = ['middle', 'base_candidate_0', 'base_candidate_1']
    props = [NativeObject(key, [index * .3, 0., .75]) for index, key in enumerate(keys)]
    for obj in props:
        put_on(obj, table, native_types)
    session = session_with([table, robot, *props], props)
    session.robot = robot
    parameters = {'middle_key': 'middle', 'base_keys': keys[1:], 'top_keys': [],
        'minimum_lift_m': .05, 'minimum_stability_duration_s': .5,
        'maximum_linear_speed_m_s': .02, 'maximum_angular_speed_rad_s': .1}
    config = {'type': 'category_operation', 'module': action_order.MODULE, 'operation': action_order.OPERATION,
        'scene': session.scene_name, 'room': session.room_name, 'binding_keys': keys,
        'bindings': {obj.name: obj.name for obj in props}, 'parameters': parameters}
    executor = SimpleNamespace(capabilities=SimpleNamespace(
        base_translation=True, base_rotation=True, ee_translation=True, gripper=True))
    bound = {obj.name: obj for obj in props}
    result = action_order.qualify_operation(session, executor, config, bound)
    assert result['accepted'], result
    assert result['grasp_reach_qualified'] is False and result['stack_execution_qualified'] is False
    props[0].contacts = []
    assert not action_order.qualify_operation(session, executor, config, bound)['accepted']


def distance_anchor():
    obj = NativeObject('actual_coffee_table_7', [1., 2., .4], (1., .6, .8), category='coffee_table')
    source = {'reference_category': obj.category, 'reference_native_name': None,
        'reference_metadata': {'category': obj.category, 'model': 'scene_native',
            'position': obj.position.tolist(), 'quaternion_xyzw': obj.quaternion.tolist()},
        'reference_aabb_xy_reconstruction': {'centre_xy': [1., 2.], 'half_extents_xy': [.5, .3]}}
    session = SimpleNamespace(env=SimpleNamespace(scene=SimpleNamespace(objects=[obj])))
    return session, source, obj


def test_distance_resolves_original_native_anchor_without_invented_binding_or_qa():
    session, source, obj = distance_anchor()
    found, evidence = resolve_fixed_distance_reference(session, source)
    assert found is obj and evidence['accepted']
    assert evidence['reference']['name'] == 'actual_coffee_table_7'
    source['answer'] = 'an unrelated private label'
    again, repeated = resolve_fixed_distance_reference(session, source)
    assert again is obj and repeated == evidence


def test_distance_anchor_reads_native_tensor_vectors_before_numpy_conversion():
    class TensorVector:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    session, source, obj = distance_anchor()
    obj.get_position_orientation = lambda: (TensorVector(obj.position), TensorVector(obj.quaternion))
    found, evidence = resolve_fixed_distance_reference(session, source)
    assert found is obj and evidence['accepted']


@pytest.mark.parametrize('failure', ['ambiguous', 'moved', 'rotated', 'resized', 'missing_geometry'])
def test_distance_reference_fails_closed_for_missing_ambiguous_or_changed_native_anchor(failure):
    session, source, obj = distance_anchor()
    if failure == 'ambiguous':
        duplicate = copy.deepcopy(obj)
        duplicate.name = 'another_table'
        session.env.scene.objects.append(duplicate)
    elif failure == 'moved':
        obj.position[0] += .1
    elif failure == 'rotated':
        obj.quaternion = np.array([0., 0., .70710678, .70710678])
    elif failure == 'resized':
        obj.extent[0] += .2
    else:
        del source['reference_aabb_xy_reconstruction']
    found, evidence = resolve_fixed_distance_reference(session, source)
    assert found is None and not evidence['accepted']


def source_for(module):
    converter = module.ActionOrderConverter() if module.__name__.endswith('action_order') else (
        module.DimensionalSizeConverter() if module.__name__.endswith('dimensional_size') else module.CategoryConverter())
    corpus = ROOT / 'ESI/dataset/json_clean'
    big, small = converter.category
    for path in sorted((corpus / big / small).rglob('*.json')):
        if len(path.relative_to(corpus).parts) != 5:
            continue
        source = SourceInstance(path, str(path.relative_to(corpus)), big, small)
        if converter.inspect(source).get('eligible'):
            return converter, source
    raise AssertionError('Expected an existing eligible original source for ' + small)


@pytest.mark.parametrize('name', MODULES)
def test_category_public_boundary_and_original_variant_survive_repairs(name, tmp_path, monkeypatch):
    module = importlib.import_module('wia.authoring.converters.' + name)
    if module is inclined:
        monkeypatch.setattr(module, '_robot_start', lambda *args: None)
    elif module in (stacking, contact):
        monkeypatch.setattr(module, '_robot_start', lambda *args: (None, {'failure_reason': 'cpu_screen_inconclusive'}))
    converter, source = source_for(module)
    original = source.path.read_bytes()
    result = converter.convert(source, ConversionContext(tmp_path))
    task = result['task_definition']
    assert result['status'] == 'generated' and task is not None
    assert task['scene']['variants']['original']['patches'] == []
    assert 'answer' not in task['public'] and 'metadata_json' not in task['public']
    assert 'source_objects' not in task['public'] and 'bindings' not in task['public']
    assert task['private']['evaluation']['type'] in ('grasp_lift', 'category_operation')
    if module is action_order:
        assert result['interaction_policy']['allow_fallback_table'] is True
        assert result['interaction_policy']['support_search'] == 'measured_surfaces'
    assert source.path.read_bytes() == original
    task_definition(task)


@pytest.mark.parametrize('module', [inclined, stacking, contact])
def test_cpu_stance_failure_defers_to_native_without_guessing_a_robot_pose(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_robot_start', lambda *args: None if module is inclined else (
        None, {'failure_reason': 'cpu_screen_inconclusive'}))
    converter, source = source_for(module)
    result = converter.convert(source, ConversionContext(tmp_path))
    assert result['status'] == 'generated' and result['task_definition'] is not None
    assert 'source_start_native_measurement_required' in result['warning_codes']
    patches = result['task_definition']['scene']['variants']['adapted']['patches']
    assert not any(patch['operation'] == 'robot_pose' for patch in patches)
    assert result['task_definition']['private']['evaluation']['type'] == 'category_operation'


@pytest.mark.parametrize('operation', contact.OPERATIONS)
def test_both_legacy_contact_operation_branches_remain_available(operation, tmp_path, monkeypatch):
    monkeypatch.setattr(contact, '_robot_start', lambda *args: (None, {'failure_reason': 'cpu_screen_inconclusive'}))
    converter, source = source_for(contact)
    result = converter.convert(source, ConversionContext(tmp_path, config={'operation': operation}))
    assert result['task_definition']['private']['evaluation']['operation'] == operation
    assert tuple(result['available_operation_recipes']) == contact.OPERATIONS
