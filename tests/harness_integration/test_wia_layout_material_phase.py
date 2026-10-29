"""Check opt-in construction materials with CPU constructor and wrapper stubs.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_layout_material_phase.py
No simulator, USD stage, GPU, native material constructor or capture task runs.
"""
from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from wia.authoring.converters import active_layout as adapter
import wia.runtime.sim.esi_omnigibson as esi_omnigibson
from wia.tasks import interaction_setup


def parameters(*, construction=True):
    value = {'revision': 'fixture_layout_v1', 'layout': {
        'scene_model': 'FixtureScene', 'rooms': ['fixture_room_0'], 'floor_name': 'fixture_floor',
        'objects': [{'type': 'PrimitiveObject', 'name': 'ramp', 'primitive_type': 'Cube',
                     'scale': [.47, .3, .02], 'position': [1., 2., .8], 'orientation': [0., .1, 0., .995]},
                    {'type': 'PrimitiveObject', 'name': 'ball', 'primitive_type': 'Sphere',
                     'radius': .035, 'mass': .2, 'position': [2., 3., .8], 'orientation': [0., 0., 0., 1.]}],
        'robot_start': {'position': [0., 0., .02], 'orientation': [0., 0., 0., 1.]},
        'supported_items': [], 'friction': {'ramp': {'static_friction': 2.744, 'dynamic_friction': 1.7137}}}}
    if construction:
        value['material_binding_revision'] = 'construction_v1'
    return value


@pytest.fixture
def robot_config(monkeypatch, tmp_path):
    config_dir = tmp_path / 'example_configs'
    config_dir.mkdir()
    robot = {'type': 'FixtureRobot', 'reset_joint_pos': [0.] * 12, 'fixture_option': 'unchanged'}
    (config_dir / 'r1_primitives.yaml').write_text(yaml.safe_dump({'robots': [robot]}), encoding='utf-8')
    calls = []

    def pipeline(root):
        calls.append(root)
        return NS(og=NS(example_config_path=config_dir))

    monkeypatch.setattr(esi_omnigibson, '_import_pipeline', pipeline)
    return calls


def test_configure_passes_exact_constructor_materials_and_preserves_geometry(robot_config):
    p = parameters()
    p['layout']['friction']['ball'] = {'static_friction': .8, 'dynamic_friction': .6, 'restitution': .3}
    original = deepcopy(p)
    config = {'scene': {'scene_model': 'FixtureScene'}, 'objects': [{'name': 'old_inventory'}]}
    result = adapter.configure_scene(config, p)
    assert result == {'revision': p['revision'], 'native_layout_qualified': False}
    assert robot_config == [esi_omnigibson.DEFAULT_ESI_ROOT]
    for original_object, configured in zip(original['layout']['objects'], config['objects']):
        expected = deepcopy(original_object)
        expected['link_physics_materials'] = {'base_link': {
            **original['layout']['friction'][original_object['name']],
            'restitution': original['layout']['friction'][original_object['name']].get('restitution', .1)}}
        assert configured == expected
    assert config['robots'][0]['position'] == original['layout']['robot_start']['position']
    assert p == original


def test_identical_preexisting_material_spec_is_accepted_and_input_is_not_mutated(robot_config):
    p = parameters()
    material = {**p['layout']['friction']['ramp'], 'restitution': .1}
    p['layout']['objects'][0]['link_physics_materials'] = {'base_link': material}
    original = deepcopy(p)
    config = {'scene': {'scene_model': 'FixtureScene'}}
    adapter.configure_scene(config, p)
    assert p == original
    assert config['objects'][0]['link_physics_materials'] == {'base_link': material}
    assert config['objects'][0]['link_physics_materials'] is not p['layout']['objects'][0]['link_physics_materials']


@pytest.mark.parametrize('invalid', ['missing_object', 'duplicate_object', 'dataset_object', 'unsupported_primitive',
                                   'cloth_primitive', 'conflicting_material', 'wrong_link', 'missing_coefficient'])
def test_construction_rejects_unsupported_or_conflicting_specs_before_config_mutation(robot_config, invalid):
    p = parameters()
    obj = p['layout']['objects'][0]
    if invalid == 'missing_object':
        obj['name'] = 'different'
    elif invalid == 'duplicate_object':
        p['layout']['objects'].append(deepcopy(obj))
    elif invalid == 'dataset_object':
        obj['type'] = 'DatasetObject'
    elif invalid == 'unsupported_primitive':
        obj['primitive_type'] = 'Mesh'
    elif invalid == 'cloth_primitive':
        obj['prim_type'] = 1
    elif invalid == 'missing_coefficient':
        p['layout']['friction']['ramp'].pop('dynamic_friction')
    else:
        material = {**p['layout']['friction']['ramp'], 'restitution': .1}
        if invalid == 'conflicting_material':
            material['static_friction'] = .01
        obj['link_physics_materials'] = {'different_link' if invalid == 'wrong_link' else 'base_link': material}
    config = {'scene': {'scene_model': 'FixtureScene'}}
    original = deepcopy(config)
    with pytest.raises(ValueError):
        adapter.configure_scene(config, p)
    assert config == original and robot_config == []


@pytest.mark.parametrize('revision', [None, 'other', '', 1])
def test_material_binding_requires_explicit_supported_opt_in(robot_config, revision):
    p = parameters()
    p['material_binding_revision'] = revision
    with pytest.raises(ValueError, match='construction_v1'):
        adapter.configure_scene({'scene': {'scene_model': 'FixtureScene'}}, p)
    assert robot_config == []


class Material:
    def __init__(self, values):
        self.prim_path = '/World/ramp/Looks/base_link_physics_mat'
        self.values, self.reads = deepcopy(values), []

    def get_static_friction(self):
        self.reads.append('static_friction')
        return self.values['static_friction']

    def get_dynamic_friction(self):
        self.reads.append('dynamic_friction')
        return self.values['dynamic_friction']

    def get_restitution(self):
        self.reads.append('restitution')
        return self.values['restitution']


@pytest.fixture
def native(monkeypatch):
    material = Material({'static_friction': 2.744, 'dynamic_friction': 1.7137, 'restitution': .1})
    calls = {'steps': 0, 'constructors': [], 'bindings': [], 'material_reads': 0}

    def construct(**kwargs):
        calls['constructors'].append(kwargs)
        return material

    def read():
        calls['material_reads'] += 1
        return material

    mesh = NS(prim_path='/World/ramp/base_link/collisions', get_applied_physics_material=read,
              apply_physics_material=lambda value: calls['bindings'].append(value))
    link = NS(collision_meshes={'collision': mesh})
    ramp = NS(name='ramp', prim_path='/World/ramp', _primitive_type='Cube',
              root_link=link, links={'base_link': link})
    floor = NS(name='fixture_floor', prim_path='/World/fixture_floor', aabb=np.array([[-3., -3., -.1], [3., 3., 0.]]))
    objects = {'ramp': ramp, 'fixture_floor': floor}
    scene = NS(object_registry=lambda kind, name: objects.get(name))
    session = NS(_category_initialization_active=True, env=NS(scene=scene), robot=object())
    og, lazy, states = (ModuleType(name) for name in ('omnigibson', 'omnigibson.lazy', 'omnigibson.object_states'))

    def step():
        calls['steps'] += 1

    og.sim = NS(step=step)
    lazy.isaacsim = NS(core=NS(api=NS(materials=NS(PhysicsMaterial=construct))))
    states.Touching = object()
    monkeypatch.setitem(sys.modules, 'omnigibson', og)
    monkeypatch.setitem(sys.modules, 'omnigibson.lazy', lazy)
    monkeypatch.setitem(sys.modules, 'omnigibson.object_states', states)
    monkeypatch.setitem(sys.modules, 'torch', ModuleType('torch'))
    monkeypatch.setattr(interaction_setup, '_robot_contact_check', lambda *args: {'floor_supported': True, 'collision_free': True})
    return NS(session=session, calls=calls, material=material, mesh=mesh, link=link, ramp=ramp)


def test_initialization_only_reads_existing_material_and_reports_phase_without_solver_claim(native):
    p = parameters()
    original = deepcopy(p)
    result = adapter.initialize_scene(native.session, p)
    receipt = result['material_binding_initialization']
    assert receipt['phase'] == 'construction' and receipt['readback_phase'] == 'initialization'
    assert receipt['available'] is receipt['accepted'] is result['accepted'] is True
    assert receipt['corrective_writes'] == 0
    assert receipt['effective_solver_coefficients_verified'] is False
    assert receipt['solver_material_consumption_verified'] is False
    row = receipt['objects']['ramp']['collision_materials'][0]
    assert {key: row[key] for key in native.material.values} == native.material.values
    assert row['accepted'] is True and row['material_prim_path'] == native.material.prim_path
    assert native.material.reads == ['static_friction', 'dynamic_friction', 'restitution']
    assert native.calls == {'steps': 90, 'constructors': [], 'bindings': [], 'material_reads': 1}
    assert native.session.category_interaction_setup is result
    assert p == original
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('failure', ['missing_material', 'mismatch', 'nonfinite', 'missing_mesh',
                                   'wrong_native_name', 'wrong_native_type', 'extra_link'])
def test_missing_or_mismatched_native_material_fails_without_corrective_writes(native, failure):
    if failure == 'missing_material':
        native.mesh.get_applied_physics_material = lambda: None
    elif failure == 'mismatch':
        native.material.values['static_friction'] += .01
    elif failure == 'nonfinite':
        native.material.values['dynamic_friction'] = float('nan')
    elif failure == 'missing_mesh':
        native.link.collision_meshes.clear()
    elif failure == 'wrong_native_name':
        native.ramp.name = 'different'
    elif failure == 'wrong_native_type':
        native.ramp._primitive_type = 'Sphere'
    else:
        native.ramp.links['extra'] = NS(collision_meshes={})
    result = adapter.initialize_scene(native.session, parameters())
    receipt = result['material_binding_initialization']
    assert receipt['accepted'] is result['accepted'] is False
    assert receipt['available'] is (failure == 'mismatch')
    assert native.calls['constructors'] == native.calls['bindings'] == []
    assert receipt['corrective_writes'] == 0
    json.dumps(result, allow_nan=False)


def test_wrapper_storage_float_tolerance_is_explicit(native):
    native.material.values['static_friction'] += adapter.MATERIAL_TOLERANCE / 2
    assert adapter.initialize_scene(native.session, parameters())['material_binding_initialization']['accepted'] is True


def test_legacy_configure_and_initialization_keep_late_binding_and_original_receipts(robot_config, native):
    p = parameters(construction=False)
    original = deepcopy(p)
    config = {'scene': {'scene_model': 'FixtureScene'}}
    adapter.configure_scene(config, p)
    assert config['objects'] == p['layout']['objects']
    assert 'link_physics_materials' not in config['objects'][0]
    result = adapter.initialize_scene(native.session, p)
    assert 'material_binding_initialization' not in result
    assert result == {'revision': p['revision'], 'accepted': True, 'supported_items': {},
        'robot': {'floor_supported': True, 'collision_free': True},
        'validation_scope': 'Native initial support and floor contacts; route and manipulation remain unverified.'}
    assert native.calls['constructors'] == [{'prim_path': '/World/ramp/Looks/redesign_physics',
        'name': 'ramp_redesign_physics', 'static_friction': 2.744, 'dynamic_friction': 1.7137, 'restitution': .1}]
    assert native.calls['bindings'] == [native.material] and native.calls['material_reads'] == 0
    assert native.calls['steps'] == 90 and p == original
