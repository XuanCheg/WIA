"""Check background repair boundaries and native diagnostics using CPU fixtures.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_background.py
No simulator, rendering, GPU, scene pose writes or physics steps are performed.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from wia.authoring.converters import wia_background as background


def parameters(**updates):
    return {'revision': background.REVISION, 'mode': 'full_scene',
            'expected_scene': 'FixtureScene', 'disable_grid': False, **updates}


def test_configure_clears_only_room_selection_and_records_previous():
    config = {'scene': {'scene_model': 'FixtureScene', 'load_room_instances': ['room_0'],
                       'load_room_types': ['kitchen'], 'load_object_categories': ['floors'],
                       'not_load_object_categories': ['ceiling'], 'scene_instance': 'custom'},
              'objects': [{'name': 'task_object'}], 'robots': [{'name': 'robot'}],
              'other': {'untouched': True}}
    before = deepcopy(config)
    receipt = background.configure_scene(config, parameters())
    expected = deepcopy(before)
    expected['scene'].update(load_room_instances=None, load_room_types=None)
    assert config == expected
    assert receipt['previous_selection']['load_room_instances']['value'] == ['room_0']
    assert receipt['previous_selection']['load_object_categories']['value'] == ['floors']
    assert receipt['physical_success_verified'] is False
    assert receipt['changed_fields'] == ['load_room_instances', 'load_room_types']
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize('updates', [
    {'revision': 'other'}, {'mode': 'room'}, {'expected_scene': ''},
    {'expected_scene': '  '}, {'disable_grid': 1}, {'disable_grid': None},
    {'expected_scene': 'DifferentScene'},
])
def test_invalid_contract_never_mutates_config(updates):
    config = {'scene': {'scene_model': 'FixtureScene', 'load_room_instances': ['room_0']}}
    before = deepcopy(config)
    with pytest.raises(ValueError):
        background.configure_scene(config, parameters(**updates))
    assert config == before


class NativeObject:
    def __init__(self, name, category, bounds):
        self.name, self.category = name, category
        self.in_rooms = ['room_0']
        self.prim_path = '/World/' + name
        self.visual_only = False
        self.aabb = np.asarray(bounds, dtype=float)
        mesh = NS(prim_path=self.prim_path + '/base/collision', collision_enabled=True)
        self.root_link = NS(prim_path=self.prim_path + '/base', visual_only=False,
                            contact_reporting_enabled=True, collision_meshes={'mesh': mesh},
                            get_position_orientation=self.get_position_orientation)
        self.links = {'base': self.root_link}
        self.contacts = []

    def get_position_orientation(self):
        return self.aabb.mean(axis=0), np.array([0., 0., 0., 1.])

    def contact_list(self):
        return self.contacts

    def set_position_orientation(self, *args, **kwargs):
        raise AssertionError('Diagnostics cannot set poses.')

    def keep_still(self):
        raise AssertionError('Diagnostics cannot change velocities.')


@pytest.fixture
def native_session():
    floor = NativeObject('floor0', 'floors', [[-2., -2., -.1], [2., 2., 0.]])
    lawn = NativeObject('lawn0', 'lawn', [[4., -2., -.1], [6., 2., 0.]])
    chair = NativeObject('chair0', 'chair', [[1., 1., 0.], [2., 2., 1.]])
    robot = NativeObject('robot0', 'robot', [[-.2, -.2, 0.], [.2, .2, 1.]])
    robot.contacts = [NS(body0=robot.prim_path + '/base', body1=floor.prim_path + '/base/collision')]
    scene = NS(scene_model='FixtureScene', objects={obj.name: obj for obj in (floor, lawn, chair)})
    setup = {'accepted': True, 'existing': 'preserve'}
    session = NS(_category_initialization_active=True, env=NS(scene=scene, robots=[robot]),
                 interaction_setup=setup, category_interaction_setup=setup)
    session.pipeline = NS(og=NS(sim=NS(floor_plane=None,
        step=lambda: pytest.fail('No simulation steps allowed.'))))
    return session


def test_diagnostics_reads_actual_inventory_and_support_without_mutation(native_session):
    session = native_session
    robot = session.env.robots[0]
    before_bounds = robot.aabb.copy()
    setup = session.interaction_setup
    result = background.initialize_scene(session, parameters())
    diagnostic = result['diagnostics']
    assert [row['name']['value'] for row in diagnostic['floors']] == ['floor0', 'lawn0']
    floor = diagnostic['floors'][0]
    assert floor['in_rooms']['value'] == ['room_0']
    assert floor['visual_only']['value'] is False
    assert floor['collision_evidence']['value'][0]['meshes'][0]['collision_enabled']['value'] is True
    support = diagnostic['robots'][0]['ground_support_observations']
    assert support[0]['contains_robot_origin_xy'] is True
    assert support[1]['contains_robot_origin_xy'] is False
    assert support[0]['robot_aabb_bottom_gap_m'] == 0.
    assert len(support[0]['current_contact_pairs']) == 1
    assert all(row['physical_support_verified'] is False for row in support)
    assert diagnostic['physical_success_verified'] is False
    assert diagnostic['simulator_global_ground']['value']['present'] is False
    assert result['physical_success_verified'] is False
    assert session.interaction_setup is setup
    assert session.category_interaction_setup is setup
    np.testing.assert_array_equal(robot.aabb, before_bounds)
    json.dumps(result, allow_nan=False)


def test_global_ground_is_measured_separately(native_session):
    plane = NativeObject('ground_plane', 'floors', [[-100., -100., -.01], [100., 100., 0.]])
    plane.visible = True
    native_session.pipeline.og.sim.floor_plane = plane
    result = background.diagnose_scene(native_session)
    ground = result['simulator_global_ground']['value']
    assert ground['present'] is True
    assert ground['visible']['value'] is True
    assert ground['object']['prim_path']['value'] == '/World/ground_plane'
    assert len(result['floors']) == 2
    json.dumps(result, allow_nan=False)


def test_missing_measurements_are_explicit(native_session):
    floor = native_session.env.scene.objects['floor0']
    del floor.in_rooms
    floor.aabb[0, 0] = np.nan
    robot = native_session.env.robots[0]
    del robot.root_link
    robot.contact_list = None
    diagnostic = background.diagnose_scene(native_session)
    floor_result = diagnostic['floors'][0]
    assert floor_result['in_rooms']['available'] is False
    assert floor_result['aabb_m']['available'] is False
    robot_result = diagnostic['robots'][0]
    assert robot_result['base_pose']['available'] is False
    assert robot_result['contacts']['available'] is False
    assert robot_result['ground_support_observations'][0]['missing_measurements'] == [
        'contains_robot_origin_xy', 'robot_aabb_bottom_gap_m', 'current_contact_pairs']
    json.dumps(diagnostic, allow_nan=False)


def test_initialize_guard_and_scene_mismatch_prevent_setting_writes(native_session, monkeypatch):
    calls = []
    monkeypatch.setattr(background, '_disable_grid', lambda value: calls.append(value))
    native_session._category_initialization_active = False
    with pytest.raises(RuntimeError):
        background.initialize_scene(native_session, parameters(disable_grid=True))
    native_session._category_initialization_active = True
    with pytest.raises(ValueError):
        background.initialize_scene(native_session, parameters(expected_scene='Other', disable_grid=True))
    assert calls == []


@pytest.mark.parametrize('current', [True, False, None])
def test_grid_uses_only_existing_exact_boolean_setting(native_session, monkeypatch, current):
    store = {background.GRID_SETTING: current}
    writes = []

    def set_bool(key, value):
        writes.append((key, value))
        store[key] = value

    settings = ModuleType('carb.settings')
    settings.get_settings = lambda: NS(get=store.get, set_bool=set_bool)
    carb = ModuleType('carb')
    carb.settings = settings
    monkeypatch.setitem(sys.modules, 'carb', carb)
    monkeypatch.setitem(sys.modules, 'carb.settings', settings)
    result = background.initialize_scene(native_session, parameters(disable_grid=True))['grid']
    if current is None:
        assert result['available'] is False
        assert writes == []
    else:
        assert writes == [(background.GRID_SETTING, False)]
        assert result['previous'] is current
        assert result['disabled_verified'] is True
        assert result['changed'] is current
    json.dumps(result, allow_nan=False)


@pytest.fixture
def ground_scene_module(monkeypatch):
    """Load the owned subclass against the installed predicate's CPU contract."""
    registry = {}
    ground_categories = frozenset({'floors', 'lawn', 'driveway', 'carpet'})
    structure_categories = (ground_categories - {'carpet'}) | {'walls', 'ceilings'}

    class InteractiveTraversableScene:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            registry[cls.__name__] = cls

        def __init__(self):
            self.load_room_instances = ['selected_room_0']
            self.not_load_object_categories = None
            self.load_object_categories = None
            self.load_task_relevant_only = False
            self.predicate_calls = []

        def _should_load_object(self, obj_info, task_metadata):
            self.predicate_calls.append((deepcopy(obj_info), task_metadata))
            args = obj_info['args']
            category = args.get('category', 'object')
            rooms = args.get('in_rooms', [])
            if isinstance(rooms, str):
                assert ',' not in rooms
                rooms = [rooms]
            not_blacklisted = self.not_load_object_categories is None or category not in self.not_load_object_categories
            relevant = args['name'] in set(task_metadata.get('inst_to_name', {}).values()) or category in structure_categories
            whitelisted = ((self.load_object_categories is None and not self.load_task_relevant_only)
                or (self.load_object_categories is not None and category in self.load_object_categories)
                or (self.load_task_relevant_only and relevant))
            valid_room = self.load_room_instances is None or bool(set(self.load_room_instances) & set(rooms))
            structural_shell = category in structure_categories - ground_categories or category in {'door', 'sliding_door'}
            return bool((not_blacklisted and whitelisted and valid_room and obj_info.get('agent_ok', True)) or structural_shell)

    modules = {name: ModuleType(name) for name in ('omnigibson', 'omnigibson.scenes', 'omnigibson.utils',
        'omnigibson.scenes.interactive_traversable_scene', 'omnigibson.utils.constants')}
    modules['omnigibson.scenes.interactive_traversable_scene'].InteractiveTraversableScene = InteractiveTraversableScene
    modules['omnigibson.utils.constants'].GROUND_CATEGORIES = ground_categories
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    name = 'wia.authoring.converters.wia_ground_scene'
    spec = importlib.util.spec_from_file_location(name, Path(background.__file__).with_name('wia_ground_scene.py'))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return NS(module=module, registry=registry, parent=InteractiveTraversableScene)


def test_complete_ground_registers_native_subclass_and_changes_only_scene_type(ground_scene_module):
    config = {'scene': {'type': 'InteractiveTraversableScene', 'scene_model': 'FixtureScene',
        'load_room_instances': ['room_0'], 'load_room_types': ['kitchen'], 'load_object_categories': ['floors'],
        'not_load_object_categories': ['ceiling'], 'include_robots': False, 'scene_instance': 'custom'},
        'objects': [{'name': 'task_object'}], 'robots': [{'name': 'robot'}], 'other': {'untouched': True}}
    before = deepcopy(config)
    receipt = background.configure_scene(config, parameters(mode='complete_ground'))
    expected = deepcopy(before)
    expected['scene']['type'] = 'WiaCompleteGroundScene'
    assert config == expected
    assert ground_scene_module.registry['WiaCompleteGroundScene'] is ground_scene_module.module.WiaCompleteGroundScene
    assert issubclass(ground_scene_module.module.WiaCompleteGroundScene, ground_scene_module.parent)
    assert receipt['mode'] == 'complete_ground' and receipt['changed_fields'] == ['type']
    assert receipt['previous_selection']['type']['value'] == 'InteractiveTraversableScene'
    repeated = background.configure_scene(config, parameters(mode='complete_ground'))
    assert config == expected and repeated['previous_selection']['type']['value'] == 'WiaCompleteGroundScene'
    assert receipt['physical_success_verified'] is False


@pytest.mark.parametrize('scene_type', [None, 'Scene', 'EmptyScene', 'UnreviewedSubclass'])
def test_complete_ground_rejects_other_scene_types_before_native_import(scene_type):
    config = {'scene': {'type': scene_type, 'scene_model': 'FixtureScene', 'load_room_instances': ['room_0']}}
    before = deepcopy(config)
    with pytest.raises(ValueError, match='InteractiveTraversableScene'):
        background.configure_scene(config, parameters(mode='complete_ground'))
    assert config == before


@pytest.mark.parametrize('category', ['floors', 'lawn', 'driveway', 'carpet'])
@pytest.mark.parametrize('in_rooms', [['other_room_0'], 'other_room_0'])
def test_complete_ground_retries_only_room_membership_for_installed_ground_categories(ground_scene_module, category, in_rooms):
    scene = ground_scene_module.module.WiaCompleteGroundScene()
    info = {'args': {'name': 'outside_ground', 'category': category, 'in_rooms': in_rooms}, 'extra': {'keep': True}}
    before = deepcopy(info)
    metadata = {'inst_to_name': {}}
    assert scene._should_load_object(info, metadata) is True
    assert info == before and scene.load_room_instances == ['selected_room_0']
    assert len(scene.predicate_calls) == 2
    original, amended = scene.predicate_calls
    assert original == (before, metadata)
    assert amended[0]['args']['in_rooms'] == ['other_room_0', 'selected_room_0']
    assert amended[1] is metadata
    assert amended[0]['extra'] == before['extra']


@pytest.mark.parametrize('category,rooms,expected', [
    ('chair', ['other_room_0'], False), ('chair', ['selected_room_0'], True),
    ('walls', ['other_room_0'], True), ('door', ['other_room_0'], True),
    ('floor', ['other_room_0'], False), ('floors', ['selected_room_0'], True)])
def test_complete_ground_preserves_original_non_ground_and_already_accepted_results(ground_scene_module, category, rooms, expected):
    scene = ground_scene_module.module.WiaCompleteGroundScene()
    info = {'args': {'name': 'fixture', 'category': category, 'in_rooms': rooms}}
    assert scene._should_load_object(info, {}) is expected
    assert len(scene.predicate_calls) == 1


@pytest.mark.parametrize('rule', ['blacklist', 'whitelist', 'agent', 'task_relevant'])
def test_complete_ground_keeps_ground_category_and_agent_restrictions(ground_scene_module, rule):
    scene = ground_scene_module.module.WiaCompleteGroundScene()
    info = {'args': {'name': 'outside_ground', 'category': 'floors', 'in_rooms': ['other_room_0']}}
    if rule == 'blacklist':
        scene.not_load_object_categories = ['floors']
    elif rule == 'whitelist':
        scene.load_object_categories = ['chair']
    elif rule == 'agent':
        info['agent_ok'] = False
    else:
        info['args']['category'] = 'carpet'
        scene.load_task_relevant_only = True
    before = deepcopy(info)
    assert scene._should_load_object(info, {}) is False
    assert len(scene.predicate_calls) == 2 and info == before


def test_complete_ground_initialization_reports_actual_subclass_without_overwriting_setup(ground_scene_module, native_session):
    scene = ground_scene_module.module.WiaCompleteGroundScene()
    scene.scene_model = native_session.env.scene.scene_model
    scene.objects = native_session.env.scene.objects
    native_session.env.scene = scene
    setup = native_session.category_interaction_setup
    result = background.initialize_scene(native_session, parameters(mode='complete_ground'))
    assert result['mode'] == 'complete_ground'
    assert result['diagnostics']['scene_class'] == {'available': True, 'value': 'WiaCompleteGroundScene'}
    assert result['diagnostics']['scene_model']['value'] == 'FixtureScene'
    assert [row['name']['value'] for row in result['diagnostics']['floors']] == ['floor0', 'lawn0']
    assert native_session.category_interaction_setup is native_session.interaction_setup is setup
    assert result['physical_success_verified'] is False
    json.dumps(result, allow_nan=False)


def test_complete_ground_initialization_rejects_parent_class_before_grid_writes(ground_scene_module, native_session, monkeypatch):
    scene = ground_scene_module.parent()
    scene.scene_model = 'FixtureScene'
    native_session.env.scene = scene
    calls = []
    monkeypatch.setattr(background, '_disable_grid', lambda value: calls.append(value))
    with pytest.raises(ValueError, match='actual registered WiaCompleteGroundScene'):
        background.initialize_scene(native_session, parameters(mode='complete_ground', disable_grid=True))
    assert calls == []
