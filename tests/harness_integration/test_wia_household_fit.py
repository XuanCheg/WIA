"""Check household native geometry and sampled passage with CPU contract fixtures.

Run: PYTHONPATH=src python -B -m pytest -q -p no:cacheprovider
tests/harness_integration/test_wia_household_fit.py. No simulator, GPU, scene
initialization or robot control is invoked; tests supply successive measurements.
"""
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wia.authoring.converters import wia_household_fit as adapter
from harness_integration.test_wia_dimensional_adapter import Mesh, NativeCube, native as native_geometry


class NativeDataset(NativeCube):
    """Keep the native collision readbacks while exposing a rigid DatasetObject."""

    def __init__(self, previous):
        super().__init__(previous.name, np.ptp(previous.mesh.points, axis=0), previous.position)
        del self._primitive_type
        self.category, self.model = 'household_object', 'cpu_native_asset'
        self.n_dof, self.n_joints = 0, 0
        self.states = {token: NS(get_value=lambda other, field=field:
            self.relations.get((other.name, field), False))
            for token, field in zip(previous.states, ('touching', 'on_top'))}
        self.root_link.get_position_orientation = self.get_position_orientation


class FixedDataset(NativeDataset):
    """Read the combined native AABB independently from all enabled link meshes."""

    @property
    def aabb(self):
        points = np.concatenate([mesh.transform_local_points_to_world(mesh.points)
            for link in self.links.values() for mesh in link.collision_meshes.values() if mesh.collision_enabled])
        return points.min(axis=0), points.max(axis=0)


@pytest.fixture
def fixed_sibling(household):
    """Supply an actual-API-shaped fixed joint and independently measured matrices."""
    fixture = household
    obj = FixedDataset(fixture.target)
    fixture.objects['candidate_0'], fixture.target = obj, obj
    nodes = {}

    class UsdNode:
        def __init__(self, path, kind, owner=None, world=None):
            self.path, self.kind, self.owner, self.world = path, kind, owner, world
            self.enabled, self.bodies = True, []
            nodes[path] = self

        def GetPath(self):
            return self.path

        def GetTypeName(self):
            return self.kind

        def IsA(self, schema):
            return self.kind.endswith('Joint') if schema == 'Joint' else self.kind == 'PhysicsFixedJoint'

        def GetParent(self):
            return nodes[self.path.rsplit('/', 1)[0]]

        def GetAttribute(self, name):
            mesh = self.owner
            return NS(Get=lambda: {'points': mesh.points,
                'faceVertexCounts': np.full(len(mesh.faces), 3, dtype=np.int64),
                'faceVertexIndices': np.asarray(mesh.faces).reshape(-1)}[name])

        def GetJointEnabledAttr(self):
            return NS(Get=lambda: self.enabled)

        def GetBody0Rel(self):
            return NS(GetTargets=lambda: self.bodies[:1])

        def GetBody1Rel(self):
            return NS(GetTargets=lambda: self.bodies[1:])

    def root_world():
        matrix = np.eye(4)
        matrix[:3, :3] = np.diag(obj.root_link.scale) @ Rotation.from_quat(obj.quaternion).as_matrix().T
        matrix[3, :3] = obj.position
        return matrix

    obj.prim = UsdNode(obj.prim_path, 'Xform')
    obj.root_link.prim = UsdNode(obj.root_link.prim_path, 'Xform', obj.root_link, root_world)
    root_mesh = obj.mesh
    root_mesh.prim = UsdNode(root_mesh.prim_path, 'Mesh', root_mesh,
                            lambda: root_mesh.local_transform @ root_world())
    label_path = obj.prim_path + '/label'
    label_to_root = np.eye(4)
    label_to_root[3, 0] = .10
    label_mesh = Mesh(obj, [.08, .02, .02])
    label_mesh.prim_path = label_path + '/collision'
    label_mesh.prim = UsdNode(label_mesh.prim_path, 'Mesh', label_mesh,
                             lambda: label_mesh.local_transform @ label_to_root @ root_world())
    label = NS(prim_path=label_path, is_meta_link=False, collision_meshes={'mesh': label_mesh},
               get_attribute=obj.root_link.get_attribute)
    label.prim = UsdNode(label_path, 'Xform', label, lambda: label_to_root @ root_world())
    obj.links['label'] = label
    joint = UsdNode(obj.root_link.prim_path + '/j_label', 'PhysicsFixedJoint')
    joint.bodies = [obj.root_link.prim_path, label_path]
    for mesh in (root_mesh, label_mesh):
        mesh.transform_local_points_to_world = lambda points, mesh=mesh: (
            np.column_stack((points, np.ones(len(points)))) @ mesh.prim.world())[:, :3]
    lazy = sys.modules['omnigibson.lazy']
    original_cache = lazy.pxr.UsdGeom.XformCache

    class XformCache(original_cache):
        def GetLocalToWorldTransform(self, prim):
            return prim.world()

    lazy.pxr.UsdGeom.XformCache = XformCache
    lazy.pxr.UsdGeom.Xformable = lambda prim: NS(GetResetXformStack=lambda: False)
    lazy.pxr.Usd = NS(PrimRange=lambda prim: list(nodes.values()))
    lazy.pxr.UsdPhysics = NS(Joint=lambda prim: prim, FixedJoint='FixedJoint')
    # The Joint callable is also the IsA schema identity in the CPU fixture.
    original_isa = UsdNode.IsA
    UsdNode.IsA = lambda self, schema: self.kind.endswith('Joint') if callable(schema) else original_isa(self, schema)
    return NS(fixture=fixture, obj=obj, label=label, label_mesh=label_mesh, joint=joint,
              label_to_root=label_to_root, nodes=nodes)


def add_compartment(fixture, floor_key, center_y):
    """Add measured fixed Cubes; the frame remains a separate short mouth."""
    objects, p = fixture.objects, fixture.config['parameters']
    parts = {
        floor_key: ([.52, .32, .012], [8.36, center_y, .666]),
        floor_key + '_left': ([.56, .04, .256], [8.36, center_y - .18, .8]),
        floor_key + '_right': ([.56, .04, .256], [8.36, center_y + .18, .8]),
        floor_key + '_back': ([.04, .36, .256], [8.64, center_y, .8]),
    }
    for key, (dimensions, position) in parts.items():
        objects[key] = NativeCube('native_' + key, dimensions, position, fixed=True)
        fixture.config['bindings'][key] = objects[key].name
        if key not in p['fixture_keys']:
            p['fixture_keys'].append(key)
    p['destination_interiors'][floor_key] = {
        'floor': floor_key, 'left': floor_key + '_left', 'right': floor_key + '_right',
        'back': floor_key + '_back'}


@pytest.fixture
def household(native_geometry, monkeypatch):
    fixture = native_geometry
    dataset_module = ModuleType('omnigibson.objects.dataset_object')
    dataset_module.DatasetObject = NativeDataset
    monkeypatch.setitem(sys.modules, dataset_module.__name__, dataset_module)
    fixture.config['module'] = adapter.MODULE
    p = fixture.config['parameters']
    p.update(revision=adapter.REVISION,
        scene_contract={'scene_model': fixture.session.scene_name, 'room': fixture.session.room_name},
        candidate_assets={key: {'category': 'household_object', 'model': 'cpu_native_asset', 'scale': [1., 1., 1.]}
                          for key in p['candidates']},
        destination_interiors={}, completion_items=[
            {'id': 'passage', 'kind': 'historical_checkpoint'},
            {'id': 'stable_placement', 'kind': 'terminal_condition'}])
    p['scoring'].update(manipulation_revision='final_state_v1', minimum_lift_m=None)
    for key in p['candidates']:
        fixture.objects[key] = NativeDataset(fixture.objects[key])
    fixture.target = fixture.objects['candidate_0']
    add_compartment(fixture, 'pad_0', -1.)
    fixture.robot.eef_links, fixture.robot.finger_links = {}, {}
    for index, arm in enumerate(fixture.robot.arm_names):
        links = [NS(prim_path=f'/World/robot/{arm}_{part}',
                    aabb=(np.array([5., float(index), 1.]), np.array([5.02, index + .02, 1.02])))
                 for part in ('eef', 'finger0', 'finger1')]
        fixture.robot.eef_links[arm], fixture.robot.finger_links[arm] = links[0], links[1:]
    return fixture


def one_target(fixture):
    p = fixture.config['parameters']
    for field in ('candidates', 'candidate_dimensions_m', 'candidate_assets'):
        p[field] = {'candidate_0': p[field]['candidate_0']}
    return fixture


def sorting(fixture):
    p = fixture.config['parameters']
    fixture.config['operation'] = p['operation'] = 'sort_intrinsic_dimension'
    p['completion_items'] = [{'id': key, 'kind': 'terminal_condition'}
                             for key in ('assignment', 'stable_placement')]
    p['destinations'] = ['pad_0', 'pad_1']
    add_compartment(fixture, 'pad_1', -.45)
    # The second target is longer intrinsically, but shorter along world X.
    second = fixture.objects['candidate_1']
    second.mesh.points *= np.array([.16, .045, .05]) / np.ptp(second.mesh.points, axis=0)
    second.quaternion = Rotation.from_euler('z', 90, degrees=True).as_quat()
    return fixture


def evaluator(fixture):
    return adapter.create_evaluator(fixture.session, fixture.controller, fixture.config, fixture.output)


def tick(fixture, instance):
    fixture.clock.current_time += .0625
    instance.sample_step()


def place(fixture, key, floor_key):
    target, floor = fixture.objects[key], fixture.objects[floor_key]
    points = target.mesh.transform_local_points_to_world(target.mesh.points)
    offset = float(target.position[2] - points[:, 2].min())
    target.position[:] = [8.36, floor.position[1], floor.aabb[1][2] + offset]
    for field in ('touching', 'on_top'):
        target.relations[(floor.name, field)] = True


def dwell(fixture, instance):
    for _ in range(10):
        tick(fixture, instance)
    return instance.record(fixture.observation)


def test_single_target_measures_short_mouth_and_full_compartment_without_resizing(household):
    fixture = one_target(household)
    p = fixture.config['parameters']
    p['candidate_dimensions_m']['candidate_0'] = [10., 11., 12.]
    instance = evaluator(fixture)
    result = instance.begin_episode(fixture.observation)
    assert result['initial_qualification']['accepted'] is True
    assert instance.pure_parameters['aperture']['exit_x_m'] - instance.pure_parameters['aperture']['entry_x_m'] == pytest.approx(.06)
    measured = result['native_measurements']
    assert np.asarray(measured['destination_interiors']['pad_0']['bounds_m']) == pytest.approx(
        np.array([[8.1, -1.16, .672], [8.62, -.84, .928]]))
    assert measured['targets']['candidate_0']['dimensions_m'] == pytest.approx([.09, .06, .06])
    assert fixture.target.scale == pytest.approx(np.ones(3))
    assert result['initial_qualification']['candidate_dimensions_are_proposals'] is True
    # Inactive half-open grippers remain a diagnostic under the native contract.
    assert measured['native_grasp_diagnostics']['all_grippers_open'] is False


@pytest.mark.parametrize('route', ['through', 'around', 'jump'])
def test_sampled_slot_passage_requires_mouth_overlap_and_done(household, route):
    fixture = one_target(household)
    instance = evaluator(fixture)
    instance.begin_episode(fixture.observation)
    fixture.target.position[:] = [7.6, -1. if route != 'around' else -1.4, .75]
    tick(fixture, instance)
    for x in ([8.15] if route == 'jump' else [7.96, 8., 8.15]):
        fixture.target.position[0] = x
        tick(fixture, instance)
    place(fixture, 'candidate_0', 'pad_0')
    pending = dwell(fixture, instance)
    assert pending['evaluation_validity'] == 'pending_done' and pending['completion_score'] is None
    terminal = instance.declare_done(fixture.observation)
    assert terminal['evaluation_validity'] == 'complete'
    assert terminal['task_success'] is (route == 'through')
    assert {item['id']: item['completed'] for item in terminal['completion_items']} == {
        'passage': route == 'through', 'stable_placement': True}
    assert not terminal['per_target']['candidate_0']['grasp_seen']
    assert terminal['physical_scorer_version'] == 'wia_dimensional_final_state_v1'
    fixture.target.position += 20.
    tick(fixture, instance)
    assert instance.declare_done(fixture.observation) == terminal


@pytest.mark.parametrize('tolerance,route,accepted', [
    (.003, 'floor_contact', True), (0., 'floor_contact', True),
    (.003, 'penetration', False), (.003, 'above_roof', False), (.003, 'around_side', False),
    (None, 'floor_contact', False), (None, 'above_lower_inset', True)])
def test_optional_floor_contact_preserves_measured_frame_and_other_passage_limits(household, tolerance, route, accepted):
    fixture = one_target(household)
    scoring = fixture.config['parameters']['scoring']
    if tolerance is not None:
        scoring['aperture_floor_contact_tolerance_m'] = tolerance
    instance = evaluator(fixture)
    initial = instance.begin_episode(fixture.observation)
    aperture = instance.pure_parameters['aperture']
    measured_bounds = dict(aperture)
    floor = aperture['z_min_m']
    bottom = (floor - .0031 if route == 'penetration' else
              floor + scoring['aperture_clearance_margin_m'] + .001 if route == 'above_lower_inset' else floor)
    z = aperture['z_max_m'] if route == 'above_roof' else bottom + .03
    y = aperture['y_min_m'] - .05 if route == 'around_side' else -1.
    fixture.target.position[:] = [7.6, y, z]
    tick(fixture, instance)
    for x in (7.96, 8., 8.15):
        fixture.target.position[0] = x
        tick(fixture, instance)
    place(fixture, 'candidate_0', 'pad_0')
    dwell(fixture, instance)
    result = instance.declare_done(fixture.observation)
    assert result['evaluation_validity'] == 'complete'
    assert result['task_success'] is accepted
    assert result['per_target']['candidate_0']['path']['completed'] is accepted
    assert instance.pure_parameters['aperture'] == measured_bounds
    assert initial['native_measurements']['measured_aperture']['bounds'] == measured_bounds


@pytest.mark.parametrize('value', [-.001, None, True, float('inf'), float('nan'), '0.003'])
def test_optional_floor_contact_tolerance_requires_finite_nonnegative_number(household, value):
    fixture = one_target(household)
    measured = adapter._measure(fixture.session, fixture.controller, fixture.config, fixture.objects)
    parameters = dict(fixture.config['parameters'])
    parameters['operation'] = 'select_pass_aperture'
    parameters['aperture'] = measured['measured_aperture']['bounds']
    parameters['scoring'] = dict(parameters['scoring'], aperture_floor_contact_tolerance_m=value)
    with pytest.raises(ValueError, match='aperture_floor_contact_tolerance_m'):
        adapter.native._progress(parameters)
    fixture.config['parameters']['scoring']['aperture_floor_contact_tolerance_m'] = value
    qualification = adapter.qualify_operation(
        fixture.session, fixture.controller, fixture.config, fixture.objects)
    assert not qualification['accepted']
    assert 'aperture_floor_contact_tolerance_m' in qualification['measurement_error']


@pytest.mark.parametrize('fault', ['missing_floor', 'unknown_wall', 'duplicate_part', 'missing_interior',
                                  'wrong_scene', 'wrong_asset', 'wrong_completion', 'invalid_upright'])
def test_geometry_contract_rejects_malformed_inputs(household, fault):
    p = household.config['parameters']
    if fault == 'missing_floor':
        del p['destination_interiors']['pad_0']['floor']
    elif fault == 'unknown_wall':
        p['destination_interiors']['pad_0']['back'] = 'unbound_wall'
    elif fault == 'duplicate_part':
        p['destination_interiors']['pad_0']['back'] = p['destination_interiors']['pad_0']['left']
    elif fault == 'missing_interior':
        p['destination_interiors'] = {}
    elif fault == 'wrong_scene':
        p['scene_contract']['room'] = 'wrong_room'
    elif fault == 'wrong_asset':
        del p['candidate_assets']['candidate_0']['model']
    elif fault == 'wrong_completion':
        p['completion_items'] = ['passage', 'stable_placement']
    else:
        p['upright_axes'] = {'candidate_0': True}
    result = adapter.qualify_operation(household.session, household.controller, household.config, household.objects)
    assert result['accepted'] is False
    assert result['reason'] == 'native_household_fit_measurement_unavailable'


@pytest.mark.parametrize('fault', ['roof', 'outside_side', 'outside_back', 'under_floor', 'contact_missing', 'on_top_missing'])
def test_all_vertices_and_native_floor_support_are_required(household, fault):
    fixture = one_target(household)
    place(fixture, 'candidate_0', 'pad_0')
    if fault == 'roof':
        fixture.target.position[2] = .96
    elif fault == 'outside_side':
        fixture.target.position[1] = -.85
    elif fault == 'outside_back':
        fixture.target.position[0] = 8.61
    elif fault == 'under_floor':
        fixture.target.position[2] -= .004
    else:
        fixture.target.relations[(fixture.objects['pad_0'].name,
            'touching' if fault == 'contact_missing' else 'on_top')] = False
    sample = adapter._measure(fixture.session, fixture.controller, fixture.config, fixture.objects)
    relation = sample['targets']['candidate_0']['destinations']['pad_0']
    assert not all(relation[field] is True for field in ('touching', 'on_top', 'footprint_inside'))


def test_optional_front_roof_bounds_and_floor_contact_tolerance(household):
    fixture = one_target(household)
    p = fixture.config['parameters']
    for key, dimensions, position in [
        ('front', [.02, .36, .04], [8.11, -1., .68]),
        ('roof', [.56, .36, .02], [8.36, -1., .91])]:
        fixture.objects[key] = NativeCube('native_interior_' + key, dimensions, position, fixed=True)
        fixture.config['bindings'][key] = fixture.objects[key].name
        p['fixture_keys'].append(key)
        p['destination_interiors']['pad_0'][key] = key
    place(fixture, 'candidate_0', 'pad_0')
    fixture.target.position[2] -= .0029
    sample = adapter._measure(fixture.session, fixture.controller, fixture.config, fixture.objects)
    assert sample['destination_interiors']['pad_0']['bounds_m'][0][0] == pytest.approx(8.12)
    assert sample['destination_interiors']['pad_0']['bounds_m'][1][2] == pytest.approx(.90)
    assert sample['targets']['candidate_0']['destinations']['pad_0']['footprint_inside'] is True
    fixture.target.position[0] = 8.155  # One vertex crosses the front face inset.
    sample = adapter._measure(fixture.session, fixture.controller, fixture.config, fixture.objects)
    assert sample['targets']['candidate_0']['destinations']['pad_0']['full_vertices_inside'] is False


def test_sorting_ranks_intrinsic_extent_independent_of_world_orientation(household):
    fixture = sorting(household)
    instance = evaluator(fixture)
    initial = instance.begin_episode(fixture.observation)
    assert initial['expected_assignment'] == {'candidate_0': 'pad_0', 'candidate_1': 'pad_1'}
    second = fixture.objects['candidate_1']
    assert second.aabb[1][0] - second.aabb[0][0] < fixture.target.aabb[1][0] - fixture.target.aabb[0][0]
    place(fixture, 'candidate_0', 'pad_0')
    place(fixture, 'candidate_1', 'pad_1')
    result = dwell(fixture, instance)
    assert result['task_success'] is True
    assert result['completion_score'] == 1.
    assert all(not row['grasp_seen'] for row in result['per_target'].values())


def test_three_target_sorting_and_native_minimum_assignment_gap(household):
    fixture = sorting(household)
    p = fixture.config['parameters']
    previous = NativeCube('native_third_target', [.12, .05, .05], [7.6, -.1, .7])
    previous.states = fixture.target.states
    fixture.objects['candidate_2'] = NativeDataset(previous)
    fixture.config['bindings']['candidate_2'] = previous.name
    p['candidates']['candidate_2'] = previous.name
    p['candidate_dimensions_m']['candidate_2'] = [.12, .05, .05]
    p['candidate_assets']['candidate_2'] = dict(p['candidate_assets']['candidate_0'])
    p['destinations'].append('pad_2')
    add_compartment(fixture, 'pad_2', .1)
    instance = evaluator(fixture)
    initial = instance.begin_episode(fixture.observation)
    assert initial['expected_assignment'] == {
        'candidate_0': 'pad_0', 'candidate_2': 'pad_1', 'candidate_1': 'pad_2'}
    fixture.objects['candidate_2'].mesh.points *= [.091 / .12, 1., 1.]
    sample = adapter._measure(fixture.session, fixture.controller, fixture.config, fixture.objects)
    reducer = adapter.native._progress(instance.pure_parameters)
    result = reducer.begin(sample)
    assert result['evaluation_validity'] == 'unavailable'
    assert 'initial_intrinsic_dimension_assignment_ambiguous' in result['initial_measurement_errors']


def test_wrong_compartment_rejects_even_with_forged_floor_support_readings(household):
    fixture = sorting(household)
    instance = evaluator(fixture)
    instance.begin_episode(fixture.observation)
    place(fixture, 'candidate_0', 'pad_1')
    place(fixture, 'candidate_1', 'pad_0')
    for key, floor_key in (('candidate_0', 'pad_0'), ('candidate_1', 'pad_1')):
        for field in ('touching', 'on_top'):
            fixture.objects[key].relations[(fixture.objects[floor_key].name, field)] = True
    result = dwell(fixture, instance)
    assert not result['task_success']
    assert not any(row['assignment_satisfied'] for row in result['per_target'].values())


@pytest.mark.parametrize('fault', ['held', 'unstable', 'hand_near', 'robot_contact', 'closed', 'release_pending'])
def test_final_placement_requires_released_stable_contact_free_and_clear_hands(household, fault):
    fixture = sorting(household)
    instance = evaluator(fixture)
    instance.begin_episode(fixture.observation)
    for key, floor in (('candidate_0', 'pad_0'), ('candidate_1', 'pad_1')):
        place(fixture, key, floor)
    if fault == 'held':
        fixture.robot._ag_obj_in_hand['left'], fixture.robot._ag_obj_constraints['left'] = fixture.target, object()
        fixture.readings['left'], fixture.robot.positions[:2] = 1, 0.
    elif fault == 'unstable':
        fixture.target.linear[0] = .1
    elif fault == 'hand_near':
        low, high = fixture.target.aabb
        fixture.robot.finger_links['right'][1].aabb = (low + [.001, 0., 0.], high + [.001, 0., 0.])
    elif fault == 'robot_contact':
        fixture.target.contacts = [NS(body0=fixture.target.prim_path, body1='/World/robot/left_eef')]
    elif fault == 'closed':
        fixture.robot.positions[:2] = 0.
    else:
        fixture.robot._ag_release_counter['right'] = 1
    result = dwell(fixture, instance)
    assert result['evaluation_validity'] == 'complete'
    assert result['task_success'] is False
    assert result['per_target']['candidate_0']['stable_placement'] is False


@pytest.mark.parametrize('tilt,accepted', [(0., True), (9., True), (11., False), (180., False)])
def test_optional_upright_axis_uses_root_quaternion_and_positive_world_z(household, tilt, accepted):
    fixture = one_target(household)
    fixture.config['parameters']['upright_axes'] = {'candidate_0': 2}
    fixture.target.quaternion = Rotation.from_euler('x', tilt, degrees=True).as_quat()
    place(fixture, 'candidate_0', 'pad_0')
    sample = adapter._measure(fixture.session, fixture.controller, fixture.config, fixture.objects)
    assert sample['targets']['candidate_0']['upright']['accepted'] is accepted
    assert sample['targets']['candidate_0']['destinations']['pad_0']['footprint_inside'] is accepted
    # An unlisted target remains free to adopt the same pose.
    fixture.config['parameters'].pop('upright_axes')
    sample = adapter._measure(fixture.session, fixture.controller, fixture.config, fixture.objects)
    assert sample['targets']['candidate_0']['upright']['accepted'] is True


@pytest.mark.parametrize('fault', ['hand_links', 'target_mesh', 'wall_mesh', 'native_touching'])
def test_missing_native_evidence_remains_unavailable(household, fault):
    fixture = sorting(household)
    if fault == 'hand_links':
        del fixture.robot.finger_links['right']
    elif fault == 'target_mesh':
        fixture.target.mesh.points = None
    elif fault == 'wall_mesh':
        fixture.objects['pad_0_left'].mesh.points = None
    else:
        fixture.target.relations[(fixture.objects['pad_0'].name, 'touching')] = None
    instance = evaluator(fixture)
    result = instance.begin_episode(fixture.observation)
    assert result['evaluation_validity'] == 'unavailable'
    assert result['task_success'] is False and result['completion_score'] is None
    assert not result['initial_qualification']['accepted']


@pytest.mark.parametrize('fault', ['asset_identity', 'target_scale', 'target_shape', 'fixed_wall'])
def test_target_shape_identity_scale_and_fixed_apparatus_are_preserved(household, fault):
    fixture = sorting(household)
    instance = evaluator(fixture)
    instance.begin_episode(fixture.observation)
    if fault == 'asset_identity':
        fixture.target.model = 'different_model'
    elif fault == 'target_scale':
        fixture.target.scale[0] *= 1.01
    elif fault == 'target_shape':
        fixture.target.mesh.points[0, 0] += .01
    else:
        fixture.objects['pad_0_left'].position[1] += .01
    tick(fixture, instance)
    result = instance.record(fixture.observation)
    assert result['evaluation_validity'] == 'invalid' and not result['task_success']


def test_unprotected_background_motion_and_inert_unselected_tolerance(household):
    fixture = sorting(household)
    fixture.objects['background'] = NativeCube('native_background', [.1, .1, .1], [1., 1., 1.])
    fixture.config['bindings']['background'] = 'native_background'
    instance = evaluator(fixture)
    instance.begin_episode(fixture.observation)
    fixture.objects['background'].position += 2.
    fixture.target.position += [.05, 0., 0.]
    tick(fixture, instance)
    result = instance.record(fixture.observation)
    assert result['evaluation_validity'] == 'complete' and not result['constraint_violations']


def test_fixed_sibling_label_contributes_full_intrinsic_extent_and_world_vertices(fixed_sibling):
    obj = fixed_sibling.obj
    geometry = adapter.household_geometry(obj)
    assert geometry['dimensions_m'] == pytest.approx([.185, .06, .06])
    assert len(geometry['collision_meshes']) == 2
    assert len(geometry['vertices_m']) == 16
    expected = np.concatenate([mesh.transform_local_points_to_world(mesh.points)
        for link in obj.links.values() for mesh in link.collision_meshes.values()])
    assert np.asarray(geometry['vertices_m']) == pytest.approx(expected)
    assert geometry['fixed_link_topology']['collision_link_prim_paths'] == sorted(
        [obj.root_link.prim_path, fixed_sibling.label.prim_path])
    fixture = fixed_sibling.fixture
    sample = adapter._measure(fixture.session, fixture.controller, fixture.config, fixture.objects)
    assert sample['geometry_valid'] is True
    assert sample['targets']['candidate_0']['dimensions_m'] == pytest.approx([.185, .06, .06])


@pytest.mark.parametrize('fault', ['disconnected', 'disabled', 'movable', 'foreign_body', 'malformed_link'])
def test_sibling_collision_link_requires_valid_enabled_fixed_root_connection(fixed_sibling, fault):
    joint = fixed_sibling.joint
    if fault == 'disconnected':
        fixed_sibling.nodes.pop(joint.path)
    elif fault == 'disabled':
        joint.enabled = False
    elif fault == 'movable':
        joint.kind = 'PhysicsRevoluteJoint'
    elif fault == 'foreign_body':
        joint.bodies[1] = '/World/unrelated_link'
    else:
        fixed_sibling.label.prim_path = '/World/unrelated_link'
    with pytest.raises(ValueError):
        adapter.household_geometry(fixed_sibling.obj)


def test_fixed_sibling_intrinsics_ignore_global_pose_and_apply_root_scale_once(fixed_sibling):
    obj = fixed_sibling.obj
    initial = adapter.household_geometry(obj)
    obj.position += [13., -7., 3.]
    obj.quaternion = Rotation.from_euler('xyz', [21., 37., 64.], degrees=True).as_quat()
    moved = adapter.household_geometry(obj)
    assert np.asarray(moved['body_vertices_m']) == pytest.approx(np.asarray(initial['body_vertices_m']), abs=1e-12)
    assert adapter.native._relative_geometry(moved, initial, 1e-4)
    assert moved['fixed_link_topology'] == initial['fixed_link_topology']
    assert not np.allclose(moved['vertices_m'], initial['vertices_m'])
    obj.scale[:] = [2., 3., 4.]
    scaled = adapter.household_geometry(obj)
    assert scaled['dimensions_m'] == pytest.approx(np.array(initial['dimensions_m']) * obj.scale)
    assert scaled['root_link_scale'] == pytest.approx([2., 3., 4.])
    assert np.asarray(scaled['body_vertices_m']) == pytest.approx(np.asarray(initial['body_vertices_m']) * obj.scale)


@pytest.mark.parametrize('fault', ['shape', 'relative_transform', 'topology'])
def test_fixed_label_shape_relative_transform_and_topology_are_checked_each_sample(fixed_sibling, fault):
    fixture = fixed_sibling.fixture
    initial = adapter._measure(fixture.session, fixture.controller, fixture.config, fixture.objects)
    assert initial['geometry_valid'] is True
    if fault == 'shape':
        fixed_sibling.label_mesh.points[0, 0] -= .01
    elif fault == 'relative_transform':
        fixed_sibling.label_to_root[3, 0] += .01
    else:
        # Swapping joint bodies preserves connectivity and geometry but changes
        # the native fixed-link contract, which must remain frozen.
        fixed_sibling.joint.bodies.reverse()
    current = adapter._measure(fixture.session, fixture.controller, fixture.config, fixture.objects, initial)
    assert current['geometry_valid'] is False
    assert current['targets']['candidate_0']['fixed_shape_preserved'] is False


def test_sibling_world_vertices_are_used_for_compartment_containment(fixed_sibling):
    fixture = one_target(fixed_sibling.fixture)
    place(fixture, 'candidate_0', 'pad_0')
    fixture.target.position[0] = 8.55
    sample = adapter._measure(fixture.session, fixture.controller, fixture.config, fixture.objects)
    relation = sample['targets']['candidate_0']['destinations']['pad_0']
    # Root geometry ends at 8.595, but the sibling label reaches 8.69.
    assert fixture.target.mesh.transform_local_points_to_world(fixture.target.mesh.points)[:, 0].max() < 8.62
    assert relation['full_vertices_inside'] is False
