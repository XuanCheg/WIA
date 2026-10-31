"""Check the dimensional native adapter with CPU-native-contract fixtures.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_dimensional_adapter.py
No simulator, GPU, model, source resolver or native initialization setter is invoked.
Fixture poses are changed directly by tests to supply successive measured states.
Initialization policy tests use explicit CPU setter and controller fixtures.
"""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wia.authoring.converters import wia_dimensional as adapter


class ReadOnly:
    def __getattr__(self, name):
        if name.startswith(('set_', 'Set', 'Apply', 'Create')) or name in ('step', 'reset', 'play', 'stop', 'keep_still'):
            raise AssertionError('A measurement attempted native mutation: ' + name)
        raise AttributeError(name)


class Prim(ReadOnly):
    def __init__(self, path, owner):
        self.path, self.owner = path, owner

    def GetPath(self):
        return self.path

    def GetTypeName(self):
        return 'Mesh' if isinstance(self.owner, Mesh) else 'Xform'

    def GetAttribute(self, name):
        mesh = self.owner
        points = mesh.points if mesh.usd_points is None else mesh.usd_points
        faces = mesh.faces if mesh.usd_faces is None else mesh.usd_faces
        value = {'points': points, 'faceVertexCounts': np.full(len(faces), 3, dtype=np.int64),
                 'faceVertexIndices': np.asarray(faces).reshape(-1)}[name]
        return NS(Get=lambda: value)


class Link(ReadOnly):
    def __init__(self, owner, mesh):
        self.owner = owner
        self.prim_path = owner.prim_path + '/body'
        self.prim = Prim(self.prim_path, self)
        self.is_meta_link = False
        self.collision_meshes = {'mesh': mesh}

    @property
    def scale(self):
        return self.owner.scale * self.owner.original_scale

    def get_attribute(self, key):
        return {'physics:rigidBodyEnabled': not self.owner.fixed_base,
                'physics:kinematicEnabled': self.owner.fixed_base}[key]


class Mesh(ReadOnly):
    def __init__(self, owner, dimensions):
        self.owner = owner
        self.prim_path = owner.prim_path + '/body/collision'
        self.prim = Prim(self.prim_path, self)
        self.points = np.array(list(product((-1., 1.), repeat=3))) * np.array(dimensions) / 2
        self.faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                               [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                               [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        self.local_transform = np.eye(4, dtype=np.float64)
        self.usd_points, self.usd_faces = None, None
        self.reset_xform_stack = False
        self.collision_enabled = True

    def transform_local_points_to_world(self, points):
        homogeneous = np.column_stack((points, np.ones(len(points))))
        local = (homogeneous @ self.local_transform)[:, :3] * self.owner.root_link.scale
        world = Rotation.from_quat(self.owner.quaternion).apply(local) + self.owner.position
        return world.astype(np.float32) if self.owner.float32_world else world

    def get_collision_approximation(self):
        return 'boundingCube'


class NativeCube(ReadOnly):
    def __init__(self, name, dimensions, position, *, fixed=False, yaw=0):
        self.name, self.prim_path = name, '/World/' + name
        self._primitive_type = 'Cube'
        self.position, self.quaternion = np.array(position, float), Rotation.from_euler('z', yaw, degrees=True).as_quat()
        self.scale = np.ones(3)
        self.original_scale = np.ones(3)
        self.float32_world, self.articulated = False, False
        self.fixed_base, self.visual_only = fixed, False
        self.linear, self.angular = np.zeros(3), np.zeros(3)
        self.contacts, self.contact_error = [], None
        self.relations, self.states = {}, {}
        self.mesh = Mesh(self, dimensions)
        self.root_link = Link(self, self.mesh)
        self.links = {'body': self.root_link}

    def get_position_orientation(self):
        return self.position.copy(), self.quaternion.copy()

    def get_linear_velocity(self):
        return self.linear.copy()

    def get_angular_velocity(self):
        return self.angular.copy()

    def contact_list(self):
        if self.contact_error:
            raise self.contact_error
        return list(self.contacts)

    @property
    def aabb(self):
        points = self.mesh.transform_local_points_to_world(self.mesh.points)
        return points.min(axis=0), points.max(axis=0)


@pytest.fixture
def native(monkeypatch, tmp_path):
    states = ModuleType('omnigibson.object_states')
    states.Touching, states.OnTop = object(), object()
    og, lazy = ModuleType('omnigibson'), ModuleType('omnigibson.lazy')
    usd_calls = []

    class XformCache(ReadOnly):
        def ComputeRelativeTransform(self, prim, ancestor):
            assert isinstance(prim.owner, Mesh) and ancestor is prim.owner.owner.root_link.prim
            usd_calls.append((str(prim.GetPath()), str(ancestor.GetPath())))
            return prim.owner.local_transform.copy(), prim.owner.reset_xform_stack

    lazy.pxr = NS(UsdGeom=NS(XformCache=XformCache))
    og.lazy = lazy
    monkeypatch.setitem(sys.modules, 'omnigibson', og)
    monkeypatch.setitem(sys.modules, 'omnigibson.lazy', lazy)
    monkeypatch.setitem(sys.modules, 'omnigibson.object_states', states)
    objects = {
        'table': NativeCube('native_table', [1.2, 2.4, .1], [8., -1., .61], fixed=True),
        'leg_0': NativeCube('native_leg', [.06, .06, .55], [7.48, -2.08, .285], fixed=True),
        'pad_0': NativeCube('native_blue_pad', [.3, .32, .012], [8.36, -1., .666], fixed=True),
        'aperture_low': NativeCube('native_low_wall', [.06, .04, .22], [8., -1.08, .77], fixed=True),
        'aperture_high': NativeCube('native_high_wall', [.06, .04, .22], [8., -.92, .77], fixed=True),
        'aperture_roof': NativeCube('native_roof', [.06, .2, .04], [8., -1., .9], fixed=True),
        'candidate_0': NativeCube('native_block_0', [.09, .06, .06], [7.6, -1.4, .69]),
        'candidate_1': NativeCube('native_block_1', [.14, .14, .14], [7.6, -.6, .73]),
    }
    for obj in objects.values():
        for field, token in (('touching', states.Touching), ('on_top', states.OnTop)):
            obj.states[token] = NS(get_value=lambda other, obj=obj, field=field: obj.relations.get((other.name, field), False))
    readings = {'left': -1, 'right': -1}
    robot = NS(arm_names=('left', 'right'), prim_path='/World/robot',
        _ag_obj_in_hand={'left': None, 'right': None}, _ag_obj_constraints={'left': None, 'right': None},
        _ag_release_counter={'left': None, 'right': None},
        positions=np.array([.04, .04, .025, .025]),
        control_limits={'position': (np.zeros(4), np.array([.04, .04, .05, .05]))},
        controllers={'gripper_left': NS(dof_idx=np.array([0, 1])), 'gripper_right': NS(dof_idx=np.array([2, 3]))},
        finger_joint_names={'left': ('left0', 'left1'), 'right': ('right0', 'right1')},
        joints={name: NS(joint_type='prismatic') for name in ('left0', 'left1', 'right0', 'right1')},
        is_grasping=lambda arm: readings[arm])
    robot.get_joint_positions = lambda: robot.positions.copy()
    control = {'available': True, 'state_setter_used': False}
    controller = NS(working_arm='left', gripper_config=NS(joint_tolerance_m=.001),
                    geometric_rearrangement_evidence=lambda: deepcopy(control))
    clock = NS(current_time=0.)
    scene = NS(scene_model=adapter.SCENE,
               object_registry=lambda kind, name: next((obj for obj in objects.values() if obj.name == name), None))
    session = NS(scene_name=adapter.SCENE, room_name=adapter.ROOM, env=NS(scene=scene), robot=robot,
                 pipeline=NS(og=NS(sim=clock)), category_interaction_setup={
                     'accepted': True, 'supported_items': {'native_block_0': {'accepted': True}},
                     'robot': {'floor_supported': True, 'collision_free': True,
                               'upright': True, 'upright_measurement_available': True}})
    scoring = json.loads((__import__('wia.resources', fromlist=['ROOT']).ROOT /
        'explore_to_achieve/wia_v1/dimensional_v1/design.json').read_text())['scoring']
    config = {'type': 'category_operation', 'module': adapter.MODULE, 'operation': 'select_pass_aperture',
              'scene': adapter.SCENE, 'room': adapter.ROOM,
              'bindings': {key: obj.name for key, obj in objects.items()}, 'parameters': {
                  'revision': adapter.REVISION, 'candidates': {key: objects[key].name for key in ('candidate_0', 'candidate_1')},
                  'destinations': ['pad_0'], 'fixture_keys': [key for key in objects if not key.startswith('candidate')],
                  'candidate_dimensions_m': {'candidate_0': [.09, .06, .06], 'candidate_1': [.14, .14, .14]},
                  'scoring': scoring, 'unselected_position_tolerance_m': .02,
                  'aperture_bindings': {'lower_side': 'aperture_low', 'upper_side': 'aperture_high',
                                        'roof': 'aperture_roof', 'floor': 'table'},
                  'aperture': {'entry_x_m': 0., 'exit_x_m': 1., 'y_min_m': -2., 'y_max_m': 2.,
                               'z_min_m': 0., 'z_max_m': 2.}}}
    return NS(objects=objects, target=objects['candidate_0'], controller=controller, robot=robot,
              clock=clock, session=session, config=config, output=tmp_path,
              observation=NS(snapshot_id=1, action_step=0), readings=readings, control=control,
              states=states, usd_calls=usd_calls)


def evaluator(native):
    return adapter.create_evaluator(native.session, native.controller, native.config, native.output)


def test_actual_aperture_is_measured_separately_from_proposal_and_native_initial_support_is_retained(native):
    original = deepcopy(native.config)
    instance = evaluator(native)
    result = instance.begin_episode(native.observation)
    aperture = instance.pure_parameters['aperture']
    assert aperture == pytest.approx({'entry_x_m': 7.97, 'exit_x_m': 8.03,
        'y_min_m': -1.06, 'y_max_m': -.94, 'z_min_m': .66, 'z_max_m': .88})
    assert aperture != original['parameters']['aperture']
    assert result['configured_proposal']['aperture'] == original['parameters']['aperture']
    assert result['initial_qualification']['accepted'] is True
    assert result['initial_qualification']['initial_support'] == native.session.category_interaction_setup
    assert result['initial_qualification']['manipulation_verified'] is False
    assert result['task_success'] is False and result['evaluation_validity'] == 'pending_done'
    assert native.config == original
    assert (native.output / 'state.json').is_file() and (native.output / 'trajectory.jsonl').is_file()


def test_v2_initial_evaluation_reports_its_revision_and_preserves_native_qualification_guards(native):
    v1_qualification = adapter.qualify_operation(native.session, native.controller, native.config, native.objects)
    v1 = evaluator(native).begin_episode(native.observation)
    assert v1['schema'] == 'wia_dimensional_v1'
    native.config['parameters']['revision'] = 'wia_dimensional_v2'
    v2_qualification = adapter.qualify_operation(native.session, native.controller, native.config, native.objects)
    v2 = evaluator(native).begin_episode(native.observation)
    assert v2_qualification == v1_qualification and v2_qualification['accepted'] is True
    assert v2['schema'] == 'wia_dimensional_v2'
    assert v2['configuration']['parameters']['revision'] == 'wia_dimensional_v2'
    assert v2['pure_parameters']['revision'] == 'wia_dimensional_v2'
    assert v2['physical_scorer_schema'] == v1['physical_scorer_schema']
    assert v2['native_measurements'] == v1['native_measurements']
    assert v2['evaluation_validity'] == 'pending_done' and v2['task_success'] is False
    native.session.category_interaction_setup['robot']['upright'] = False
    assert adapter.qualify_operation(native.session, native.controller, native.config, native.objects)['accepted'] is False
    blocked = evaluator(native).begin_episode(native.observation)
    assert blocked['schema'] == 'wia_dimensional_v2'
    assert blocked['evaluation_validity'] == 'pending_done' and blocked['physical_success'] is False
    assert blocked['native_evaluation_validity'] == 'unavailable'


def test_v2_unavailable_aperture_receipt_cannot_report_the_v1_schema(native):
    native.config['parameters']['revision'] = 'wia_dimensional_v2'
    native.objects['aperture_roof'].mesh.collision_enabled = False
    result = evaluator(native).begin_episode(native.observation)
    assert result['schema'] == 'wia_dimensional_v2'
    assert result['physical_scorer_schema'] == 'wia_dimensional_progress_v2'
    assert result['evaluation_validity'] == 'pending_done'
    assert result['native_evaluation_validity'] == 'unavailable' and result['task_success'] is False


@pytest.mark.parametrize('aperture_available', [True, False])
def test_final_state_adapter_threads_revision_and_preserves_schema_when_native_aperture_unavailable(
        native, aperture_available):
    native.config['parameters']['scoring'].update(manipulation_revision='final_state_v1', minimum_lift_m=None)
    if not aperture_available:
        native.objects['aperture_roof'].mesh.collision_enabled = False
    instance = evaluator(native)
    result = instance.begin_episode(native.observation)
    assert result['physical_scorer_schema'] == result['physical_scorer_version'] == 'wia_dimensional_final_state_v1'
    assert result['manipulation_revision'] == 'final_state_v1'
    assert result['pure_parameters']['scoring']['minimum_lift_m'] is None
    assert result['evaluation_validity'] == 'pending_done'
    assert not result['task_success']
    if aperture_available:
        assert instance.scorer.manipulation_revision == 'final_state_v1'
    else:
        assert instance.scorer is None and result['native_evaluation_validity'] == 'unavailable'


@pytest.mark.parametrize('revision,lift', [(None, None), ('unknown', None), ('final_state_v1', .05)])
def test_final_state_adapter_rejects_unversioned_null_lift_and_unknown_revision(native, revision, lift):
    native.config['parameters']['scoring'].update(manipulation_revision=revision, minimum_lift_m=lift)
    with pytest.raises(ValueError):
        evaluator(native)


@pytest.mark.parametrize('field,value', [('upright', False), ('upright', None),
                                       ('upright_measurement_available', False),
                                       ('upright_measurement_available', None)])
def test_upright_gate_cannot_inherit_accepted_or_default_missing_measurement_to_true(native, field, value):
    native.session.category_interaction_setup['robot'][field] = value
    qualification = adapter.qualify_operation(native.session, native.controller, native.config, native.objects)
    assert qualification['checks']['active_layout_native_initial_support'] is True
    assert qualification['checks']['native_robot_' + field] is False
    assert qualification['accepted'] is False
    result = evaluator(native).begin_episode(native.observation)
    assert result['initial_qualification']['accepted'] is False
    assert result['evaluation_validity'] == 'pending_done'
    assert result['native_evaluation_validity'] == 'unavailable' and result['physical_success'] is False


def test_rotated_native_block_dimensions_use_body_frame_collision_vertices(native):
    native.target.quaternion = Rotation.from_euler('z', 45, degrees=True).as_quat()
    measured = adapter._geometry(native.target)
    assert measured['dimensions_m'] == pytest.approx([.09, .06, .06])
    assert np.ptp(measured['vertices_m'], axis=0)[0] > .09
    assert adapter.qualify_operation(native.session, native.controller, native.config, native.objects)['accepted'] is True


@pytest.mark.parametrize('world_x', [8., 100.])
def test_float32_world_rounding_cannot_change_local_dimensions_during_translations_and_rotations(native, world_x):
    target = native.target
    target.float32_world = True
    target.position = np.array([world_x, -1., .69], dtype=np.float32)
    initial = adapter._geometry(target)
    tolerance = native.config['parameters']['scoring']['dimension_relative_tolerance']
    old_world_inverse_error = 0.
    for translation, angles in (((0., 0., 0.), (0., 0., 0.)),
                                ((.371, -.17, .13), (17., 43., 71.)),
                                ((-.213, .29, .08), (86., 13., 137.)),
                                ((1.07, -.31, .27), (31., 65., 263.))):
        target.position = np.array([world_x, -1., .69], dtype=np.float32) + np.array(translation, np.float32)
        target.quaternion = Rotation.from_euler('xyz', angles, degrees=True).as_quat()
        measured = adapter._geometry(target)
        assert measured['dimensions_m'] == initial['dimensions_m']
        assert adapter._relative_geometry(measured, initial, tolerance) is True
        assert measured['root_link_scale'] == [1., 1., 1.]
        assert measured['collision_meshes'][0]['mesh_to_root_link_transform'] == np.eye(4).tolist()
        assert measured['collision_meshes'][0]['reset_xform_stack'] is False
        world = np.asarray(measured['vertices_m'])
        assert np.array_equal(world, target.mesh.transform_local_points_to_world(target.mesh.points).astype(float))
        former_body = Rotation.from_quat(target.quaternion).inv().apply(world - target.position)
        old_world_inverse_error = max(old_world_inverse_error,
            float(np.max(np.abs(np.ptp(former_body, axis=0) / initial['dimensions_m'] - 1.))))
    assert old_world_inverse_error > tolerance  # The fixture exposes the former world-subtraction failure.
    assert all(ancestor == target.root_link.prim_path for mesh, ancestor in native.usd_calls)


def test_actual_root_link_scale_is_applied_once_after_local_usd_transform(native):
    target = native.target
    target.mesh.points = np.array(list(product((-.5, .5), repeat=3)))
    target.scale = np.ones(3)
    target.original_scale = np.array([.09, .06, .06])
    target.mesh.local_transform[0, 0] = 2.
    measured = adapter._geometry(target)
    assert measured['scale'] == [1., 1., 1.]
    assert measured['root_link_scale'] == [.09, .06, .06]
    assert measured['dimensions_m'] == pytest.approx([.18, .06, .06])
    assert 'Float64 mesh-to-root-link local USD transform' in measured['intrinsic_geometry_source']


def test_uncached_usd_shape_mutation_cannot_be_hidden_by_wrapper_points_cache(native):
    target = native.target
    initial = adapter._geometry(target)
    target.mesh.usd_points = target.mesh.points.copy()
    target.mesh.usd_points[0, 0] += .001
    with pytest.raises(ValueError, match='Cached wrapper collision points'):
        adapter._geometry(target)
    unavailable = adapter._measure(native.session, native.controller, native.config, native.objects)
    assert unavailable['geometry_valid'] is None
    target.mesh.points = target.mesh.usd_points.copy()
    current = adapter._geometry(target)
    assert (current['collision_meshes'][0]['native_shape_stamp']
            != initial['collision_meshes'][0]['native_shape_stamp'])
    assert adapter._relative_geometry(current, initial,
        native.config['parameters']['scoring']['dimension_relative_tolerance']) is False


@pytest.mark.parametrize('change', ['mesh_scale', 'mesh_translation', 'shape', 'root_scale', 'topology', 'mesh_identity'])
def test_local_geometry_or_root_scale_changes_still_fail_fixed_shape_protection(native, change):
    target = native.target
    initial = adapter._geometry(target)
    if change == 'mesh_scale':
        target.mesh.local_transform[0, 0] *= 1.001
    elif change == 'mesh_translation':
        target.mesh.local_transform[3, 0] += .001
    elif change == 'shape':
        target.mesh.points[0, 0] += .001
    elif change == 'root_scale':
        target.original_scale[0] *= 1.001  # Entity scale remains one, while measured root-link scale changes.
    elif change == 'topology':
        target.mesh.faces[0] = [1, 2, 3]
    else:
        target.mesh.prim_path = target.root_link.prim_path + '/replacement_collision'
        target.mesh.prim.path = target.mesh.prim_path
    current = adapter._geometry(target)
    assert adapter._relative_geometry(current, initial,
        native.config['parameters']['scoring']['dimension_relative_tolerance']) is False


@pytest.mark.parametrize('problem', ['articulation', 'multiple_links', 'non_root_link', 'no_enabled_link',
                                   'reset_stack', 'projective_matrix', 'nonfinite_matrix', 'singular_matrix',
                                   'invalid_scale', 'outside_hierarchy', 'invalid_topology'])
def test_unsupported_or_unmeasurable_local_hierarchy_is_unavailable_without_world_aabb_fallback(native, problem):
    target = native.target
    if problem == 'articulation':
        target.articulated = True
    elif problem == 'multiple_links':
        target.links['other'] = Link(target, target.mesh)
    elif problem == 'non_root_link':
        target.links['body'] = Link(target, target.mesh)
    elif problem == 'no_enabled_link':
        target.mesh.collision_enabled = False
    elif problem == 'reset_stack':
        target.mesh.reset_xform_stack = True
    elif problem == 'projective_matrix':
        target.mesh.local_transform[0, 3] = .1
    elif problem == 'nonfinite_matrix':
        target.mesh.local_transform[0, 0] = np.nan
    elif problem == 'singular_matrix':
        target.mesh.local_transform[0, 0] = 0.
    elif problem == 'invalid_scale':
        target.original_scale[0] = 0.
    elif problem == 'outside_hierarchy':
        target.mesh.prim_path = target.mesh.prim.path = '/World/other/collision'
    else:
        target.mesh.faces[0, 0] = len(target.mesh.points)
    with pytest.raises(ValueError):
        adapter._geometry(target)
    measured = adapter._measure(native.session, native.controller, native.config, native.objects)
    assert measured['geometry_valid'] is (False if problem == 'no_enabled_link' else None)
    assert measured['targets']['candidate_0']['dimensions_m'] is None
    assert measured['targets']['candidate_0']['measurement_errors']


def test_record_never_measures_or_advances_core_and_begin_uses_one_initial_sample(native, monkeypatch):
    original, calls = adapter._measure, []

    def counted(*args, **kwargs):
        calls.append(native.clock.current_time)
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, '_measure', counted)
    instance = evaluator(native)
    instance.begin_episode(native.observation)
    assert calls == [0.]
    before = deepcopy(instance.scorer.result())
    native.clock.current_time = .07
    native.target.position[0] += .01
    for _ in range(4):
        instance.record(native.observation)
    assert calls == [0.] and instance.scorer.result() == before
    instance.sample_step()
    assert calls == [0., .07]
    assert instance.scorer.result()['native_time_s'] == .07


@pytest.mark.parametrize('missing', ['robot_contact', 'touching', 'on_top'])
def test_missing_native_contact_or_support_relation_never_passes(native, missing):
    if missing == 'robot_contact':
        native.target.contact_error = RuntimeError('Native contacts unavailable.')
    else:
        native.target.relations[(native.objects['pad_0'].name, missing)] = None
    result = evaluator(native).begin_episode(native.observation)
    assert result['evaluation_validity'] == 'pending_done'
    assert result['native_evaluation_validity'] == 'unavailable'
    assert result['task_success'] is result['physical_success'] is False
    if missing == 'robot_contact':
        assert result['native_measurements']['targets']['candidate_0']['robot_contact'] is None


def test_native_touching_and_ontop_are_separate_from_all_vertex_footprint(native):
    pad = native.objects['pad_0']
    native.target.position[:] = [8.49, -1., .702]
    native.target.relations[(pad.name, 'touching')] = True
    native.target.relations[(pad.name, 'on_top')] = True
    measured = adapter._measure(native.session, native.controller, native.config, native.objects)
    relation = measured['targets']['candidate_0']['destinations']['pad_0']
    assert relation['touching'] is relation['on_top'] is True
    assert relation['footprint_inside'] is False


@pytest.mark.parametrize('problem', ['rotated', 'depth_mismatch', 'closed', 'missing_geometry'])
def test_invalid_or_missing_native_frame_never_uses_declared_aperture_fallback(native, problem):
    if problem == 'rotated':
        native.objects['aperture_low'].quaternion = Rotation.from_euler('z', 10, degrees=True).as_quat()
    elif problem == 'depth_mismatch':
        native.objects['aperture_high'].position[0] += .01
    elif problem == 'closed':
        native.objects['aperture_high'].position[1] = -1.12
    else:
        native.objects['aperture_roof'].mesh.points = None
    instance = evaluator(native)
    result = instance.begin_episode(native.observation)
    assert instance.scorer is None
    assert result['evaluation_validity'] == 'pending_done'
    assert result['native_evaluation_validity'] == 'unavailable'
    assert result['task_success'] is False
    assert result['configured_proposal']['aperture'] == native.config['parameters']['aperture']
    assert 'substitute aperture' in result['measurement_error']


@pytest.mark.parametrize('change', ['fixture_position', 'fixture_scale', 'candidate_scale', 'candidate_shape'])
def test_native_fixture_or_fixed_shape_changes_latch_invalid_without_setters(native, change):
    instance = evaluator(native)
    instance.begin_episode(native.observation)
    if change == 'fixture_position':
        native.objects['leg_0'].position[0] += .003
    elif change == 'fixture_scale':
        native.objects['leg_0'].scale[0] *= 1.001
    elif change == 'candidate_scale':
        native.target.scale[0] *= 1.001
    else:
        native.target.mesh.points[0, 0] += .003
    native.clock.current_time = .05
    instance.sample_step()
    result = instance.record(native.observation)
    assert result['evaluation_validity'] == 'pending_done'
    assert result['native_evaluation_validity'] == 'invalid'
    assert result['task_success'] is result['physical_success'] is False
    assert result['completion_score'] is None


def test_missing_collision_geometry_remains_unavailable_and_non_cube_is_not_accepted(native):
    instance = evaluator(native)
    instance.begin_episode(native.observation)
    native.target.mesh.points = None
    native.clock.current_time = .05
    instance.sample_step()
    result = instance.record(native.observation)
    assert result['native_measurements']['geometry_valid'] is None
    assert result['evaluation_validity'] == 'pending_done'
    assert result['native_evaluation_validity'] == 'unavailable'
    native.target._primitive_type = 'Sphere'
    assert adapter.qualify_operation(native.session, native.controller, native.config, native.objects)['accepted'] is False


def test_grasp_requires_native_identity_and_constraint_and_release_checks_both_arms(native):
    initial, raw = adapter._grasp(native.session, native.controller)
    assert initial == {'available': True, 'held_names': [], 'released': True}
    assert raw['all_grippers_open'] is False  # Only the public working aperture must be open.
    native.robot._ag_obj_in_hand['left'] = native.target
    native.readings['left'] = 1
    native.robot.positions[:2] = 0.
    without_constraint, _ = adapter._grasp(native.session, native.controller)
    assert without_constraint['held_names'] == [] and without_constraint['released'] is False
    native.robot._ag_obj_constraints['left'] = object()
    held, _ = adapter._grasp(native.session, native.controller)
    assert held['held_names'] == [native.target.name]
    native.robot._ag_obj_in_hand['left'] = native.robot._ag_obj_constraints['left'] = None
    native.readings['left'], native.robot.positions[:2] = -1, .04
    native.robot._ag_release_counter['right'] = 1
    pending, _ = adapter._grasp(native.session, native.controller)
    assert pending['released'] is False
    native.readings['right'] = 0
    unavailable, _ = adapter._grasp(native.session, native.controller)
    assert unavailable['available'] is False and unavailable['released'] is False


def test_unavailable_control_guard_is_not_invented_invalid_and_explicit_setter_is_invalid(native):
    native.controller.geometric_rearrangement_evidence = lambda: {'available': False}
    sample = adapter._measure(native.session, native.controller, native.config, native.objects)
    assert sample['control_valid'] is True and sample['control_evidence']['available'] is False
    instance = evaluator(native)
    instance.begin_episode(native.observation)
    native.session._episode_state_setter_used = True
    native.clock.current_time = .05
    instance.sample_step()
    result = instance.record(native.observation)
    assert result['evaluation_validity'] == 'pending_done'
    assert result['native_evaluation_validity'] == 'invalid' and result['physical_success'] is False


@pytest.mark.parametrize('field,value', [('revision', 'wrong'), ('revision', 'wia_dimensional_v3'), ('scene', 'other_scene'),
                                       ('room', 'other_room'), ('operation', 'unowned_operation')])
def test_explicit_config_revision_operation_and_scene_room_guards(native, field, value):
    if field == 'revision':
        native.config['parameters'][field] = value
    else:
        native.config[field] = value
    with pytest.raises(ValueError):
        evaluator(native)
    assert adapter.qualify_operation(native.session, native.controller, native.config, native.objects)['accepted'] is False


def test_sorting_uses_measured_intrinsic_order_and_one_destination_per_candidate(native):
    native.config['operation'] = 'sort_intrinsic_dimension'
    p = native.config['parameters']
    p['destinations'] = ['pad_0', 'pad_1', 'pad_2']
    for index, length in enumerate((.09, .13, .17)):
        key = f'candidate_{index}'
        obj = NativeCube(f'native_sort_{index}', [length, .05, .05], [7.6, -1.5 + .5 * index, .685], yaw=(90, 0, 45)[index])
        for field, token in (('touching', native.states.Touching), ('on_top', native.states.OnTop)):
            obj.states[token] = NS(get_value=lambda other: False)
        native.objects[key] = obj
        native.config['bindings'][key] = obj.name
        p['candidates'][key] = obj.name
        p['candidate_dimensions_m'][key] = [length, .05, .05]
        if index:
            pad_key = f'pad_{index}'
            pad = NativeCube(f'native_pad_{index}', [.3, .32, .012], [8.32, -1.5 + .5 * index, .666], fixed=True)
            native.objects[pad_key] = pad
            native.config['bindings'][pad_key] = pad.name
            p['fixture_keys'].append(pad_key)
    instance = evaluator(native)
    result = instance.begin_episode(native.observation)
    assert result['evaluation_validity'] == 'complete'
    assert result['expected_assignment'] == {'candidate_0': 'pad_0', 'candidate_1': 'pad_1', 'candidate_2': 'pad_2'}
    assert 'measured_aperture' not in result['native_measurements']
    assert result['task_success'] is False


class JointController:
    """Initialization-only native contract fixture with a resettable goal."""
    motor_type, use_delta_commands, command_input_limits, command_dim = 'position', False, None, 2

    def __init__(self, dofs):
        self.dof_idx = np.asarray(dofs)
        self.goal = {'original': list(dofs)}

    def dump_state(self):
        return deepcopy(self.goal)

    def load_state(self, value):
        self.goal = deepcopy(value)

    def update_goal(self, command, control_dict):
        self.goal = {'target': np.asarray(command).copy()}


class InitializationFingerRobot:
    def __init__(self):
        self.arm_names = ['left', 'right']
        self.controllers = {'gripper_left': JointController([3, 5]),
            'gripper_right': JointController([2, 7]), 'trunk': JointController([0, 1])}
        self.q = np.array([.7, -.8, .03, .03, -.4, .03, .8, .03])
        self.upper = np.array([1., 1., .063, .041, 1., .052, 1., .074])
        self.control_limits = {'position': [np.zeros(8), self.upper]}
        self.finger_joint_names = {'left': ['la', 'lb'], 'right': ['ra', 'rb']}
        self.joints = {name: NS(joint_type='prismatic') for names in self.finger_joint_names.values() for name in names}
        self._ag_obj_in_hand = dict.fromkeys(self.arm_names)
        self._ag_obj_constraints = dict.fromkeys(self.arm_names)
        self._ag_release_counter = dict.fromkeys(self.arm_names)
        self.grasping = {'left': -1, 'right': -1}
        self.calls = []

    def get_joint_positions(self):
        import torch
        return torch.tensor(self.q, dtype=torch.float32)

    def get_control_dict(self):
        return {'joint_position': self.get_joint_positions()}

    def is_grasping(self, arm):
        return self.grasping[arm]

    def set_joint_positions(self, value, *, indices, drive):
        import torch
        # Installed Isaac tensor.resolve_indices calls .to() on non-list input.
        assert isinstance(indices, torch.Tensor) and indices.dtype == torch.long
        assert indices.device == value.device == self.get_joint_positions().device
        indices = indices.detach().cpu().numpy()
        value = value.detach().cpu().numpy()
        self.calls.append((np.asarray(indices).copy(), value.copy(), drive))
        if not drive:
            self.q[indices] = value
            for controller in self.controllers.values():
                controller.goal = None


def test_initial_open_policy_uses_native_indices_limits_and_preserves_other_joints_and_goals():
    from wia.authoring.converters import active_layout
    robot = InitializationFingerRobot()
    session = NS(robot=robot, _category_initialization_active=True)
    before, trunk_goal = robot.q.copy(), deepcopy(robot.controllers['trunk'].goal)
    receipt = active_layout._initial_gripper_state(session, {'initial_gripper_state': 'open'})
    indices = np.array([3, 5, 2, 7])
    assert robot.q[indices] == pytest.approx(robot.upper[indices])
    assert robot.q[[0, 1, 4, 6]] == pytest.approx(before[[0, 1, 4, 6]])
    assert robot.controllers['trunk'].goal == trunk_goal
    assert len(robot.calls) == 2 and [call[2] for call in robot.calls] == [False, True]
    for actual_indices, target, _ in robot.calls:
        assert np.array_equal(actual_indices, indices)
        assert target == pytest.approx(robot.upper[indices])
    assert receipt['scope'] == 'initialization_only' and receipt['accepted']
    for arm, dofs in (('left', [3, 5]), ('right', [2, 7])):
        assert receipt['hands'][arm]['before_m'] == pytest.approx([.03, .03])
        assert receipt['hands'][arm]['after_m'] == pytest.approx(robot.upper[dofs])
        assert robot.controllers['gripper_' + arm].goal['target'] == pytest.approx(robot.upper[dofs])
    robot.q[3] -= .01
    with pytest.raises(ValueError, match='did not retain'):
        active_layout._finalize_initial_gripper_state(session, receipt)


@pytest.mark.parametrize('failure', ['held', 'constraint', 'release', 'unknown', 'closed', 'inactive'])
def test_initial_open_policy_requires_empty_hands_and_initialization_without_writes(failure):
    from wia.authoring.converters import active_layout
    robot = InitializationFingerRobot()
    session = NS(robot=robot, _category_initialization_active=failure != 'inactive')
    if failure == 'held':
        robot._ag_obj_in_hand['right'] = object()
    if failure == 'constraint':
        robot._ag_obj_constraints['right'] = object()
    if failure == 'release':
        robot._ag_release_counter['right'] = 1
    if failure == 'unknown':
        robot.grasping['right'] = 0
    if failure == 'closed':
        robot.grasping['right'] = 1
    with pytest.raises((ValueError, RuntimeError)):
        active_layout._initial_gripper_state(session, {'initial_gripper_state': 'open'})
    assert robot.calls == []


def test_initial_gripper_policy_absent_preserves_legacy_and_invalid_policy_fails():
    from wia.authoring.converters import active_layout
    session = NS(robot=InitializationFingerRobot(), _category_initialization_active=False)
    assert active_layout._initial_gripper_state(session, {}) is None
    assert not session.robot.calls
    with pytest.raises(ValueError, match='only the explicit open'):
        active_layout._initial_gripper_state(session, {'initial_gripper_state': 'closed'})


def test_d1_done_samples_current_state_once_without_time_advance_and_freezes_record(native, monkeypatch):
    original, calls = adapter._measure, []

    def counted(*args, **kwargs):
        calls.append(native.clock.current_time)
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, '_measure', counted)
    instance = evaluator(native)
    assert instance.requires_done_declaration is True
    pending = instance.begin_episode(native.observation)
    assert pending['evaluation_validity'] == 'pending_done'
    assert pending['evaluation_finalized'] is False
    assert pending['completion_items'] is pending['completed_goals'] is pending['total_goals'] is None
    assert pending['completion_score'] is pending['score_percent'] is pending['diagnostic_lower_bound'] is None
    assert pending['selected_candidate'] is None
    native.clock.current_time = .05
    native.target.position[0] += .1
    terminal = instance.declare_done(native.observation)
    assert calls == [0., .05] and native.clock.current_time == .05
    assert terminal['evaluation_finalized']
    assert terminal['evaluation_validity'] == 'complete' and not terminal['task_success']
    assert terminal['physical_scorer_version'] == 'wia_dimensional_progress_v2'
    native.clock.current_time = 10.
    native.target.mesh.points = None
    instance.sample_step()
    assert instance.record(NS(snapshot_id=99, action_step=50)) == terminal
    assert instance.declare_done(native.observation) == terminal
    assert calls == [0., .05]
    assert json.loads((native.output / 'state.json').read_text()) == terminal


def test_d1_pending_keeps_native_invalid_evidence_then_done_reports_it(native):
    instance = evaluator(native)
    instance.begin_episode(native.observation)
    native.session._episode_state_setter_used = True
    native.clock.current_time = .05
    instance.sample_step()
    pending = instance.record(native.observation)
    assert pending['evaluation_validity'] == 'pending_done'
    assert pending['native_evaluation_validity'] == 'invalid'
    assert pending['constraints_satisfied'] is None and pending['completion_items'] is None
    terminal = instance.declare_done(native.observation)
    assert terminal['evaluation_validity'] == 'invalid'
    assert terminal['evaluation_finalized']
    assert not terminal['task_success'] and terminal['completion_score'] is None


def test_d1_without_native_frame_remains_pending_until_done_then_unavailable(native):
    native.objects['aperture_roof'].mesh.points = None
    instance = evaluator(native)
    pending = instance.begin_episode(native.observation)
    assert pending['evaluation_validity'] == 'pending_done' and pending['completion_items'] is None
    terminal = instance.declare_done(native.observation)
    assert terminal['evaluation_validity'] == 'unavailable' and terminal['evaluation_finalized']
    assert terminal['task_success'] is False and terminal['completion_score'] is None


def test_d1_legacy_unselected_tolerance_is_optional_and_inert(native):
    native.config['parameters'].pop('unselected_position_tolerance_m')
    instance = evaluator(native)
    instance.begin_episode(native.observation)
    native.target.position[0] += .1
    native.clock.current_time = .05
    instance.sample_step()
    result = instance.declare_done(native.observation)
    assert result['evaluation_validity'] == 'complete'
    assert not result['constraint_violations']


@pytest.mark.parametrize('finish_full_goal', [False, True])
def test_native_adapter_done_preserves_original_grasp_lift_pass_release_blue_goal(native, finish_full_goal):
    instance = evaluator(native)
    instance.begin_episode(native.observation)
    native.robot._ag_obj_in_hand['left'] = native.target
    native.robot._ag_obj_constraints['left'] = object()
    native.readings['left'] = 1
    native.robot.positions[:2] = 0.
    native.target.position[:] = [7.6, -1., .75]
    native.clock.current_time += .0625
    instance.sample_step()
    if finish_full_goal:
        for x in (7.96, 8., 8.15):
            native.target.position[0] = x
            native.clock.current_time += .0625
            instance.sample_step()
        native.robot._ag_obj_in_hand['left'] = None
        native.robot._ag_obj_constraints['left'] = None
        native.readings['left'] = -1
        native.robot.positions[:2] = .04
        native.target.position[:] = [8.36, -1., .702]
        pad = native.objects['pad_0']
        native.target.relations[(pad.name, 'touching')] = True
        native.target.relations[(pad.name, 'on_top')] = True
        for _ in range(9):
            native.clock.current_time += .0625
            instance.sample_step()
    pending = instance.record(native.observation)
    assert pending['evaluation_validity'] == 'pending_done'
    assert pending['completion_items'] is pending['completion_score'] is None
    assert not pending['task_success'] and not pending['evaluation_finalized']
    terminal = instance.declare_done(native.observation)
    assert terminal['evaluation_validity'] == 'complete'
    assert terminal['task_success'] is finish_full_goal
    items = {item['id']: item['completed'] for item in terminal['completion_items']}
    assert items == {'choice': True, 'grasp_lift': True,
                     'passage': finish_full_goal, 'stable_placement': finish_full_goal}
    assert terminal['selected_candidate'] == 'candidate_0'
    assert terminal['physical_success'] is finish_full_goal
    assert terminal['completion_score'] == (1. if finish_full_goal else .5)
