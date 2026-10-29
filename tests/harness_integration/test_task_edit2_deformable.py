"""CPU synthetic native-mesh behavior checks, without simulator/GPU startup.

Run: python -B -m pytest -q -p no:cacheprovider
 tests/test_task_edit2_deformable.py tests/test_core_revision_manipulation.py.
Synthetic contacts test receipt logic, never certify real cloth or robot actions.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from wia.authoring.conversion import ConversionContext, SourceInstance
from wia.authoring.converters import physical_structure_deformable as deformable
from wia.tasks import deformable_support as support
from wia.tasks.interaction_setup import prepare_definitions
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition

ROOT = __import__('wia.resources', fromlist=['ROOT']).ROOT
CORE = Path('/home/zhouxinyang/esi_category_conversion/core_revision_20261018/core_candidates_for_review.json')


class Attribute:
    def __init__(self, name, value):
        self.name, self.value = name, value
    def GetName(self):
        return self.name
    def Get(self):
        return self.value


class Prim:
    def __init__(self, path, **attributes):
        self.path, self.attributes = path, attributes
    def IsValid(self):
        return True
    def GetPath(self):
        return self.path
    def GetTypeName(self):
        return 'Material'
    def GetAttributes(self):
        return [Attribute(name, value) for name, value in self.attributes.items()]
    def GetRelationships(self):
        return []
    def GetChildren(self):
        return []


class Object:
    def __init__(self, name, position, extent=(.1, .1, .1)):
        self.name, self.category, self.model = name, 'category_' + name, 'model_' + name
        self.prim_path = '/World/' + name
        self.position, self.extent = np.array(position, dtype=float), np.array(extent, dtype=float)
        self.quaternion, self.scale = np.array([0., 0., 0., 1.]), np.ones(3)
        self.mass, self.visual_only, self.fixed_base = 1., False, False
        self.materials = [NS(prim_path='/material/' + name)]
        self.contacts = []
    def get_position_orientation(self):
        return self.position.copy(), self.quaternion.copy()
    @property
    def aabb(self):
        return self.position - self.extent/2, self.position + self.extent/2
    def contact_list(self):
        return self.contacts


@pytest.fixture
def world(monkeypatch):
    states = ModuleType('omnigibson.object_states')
    states.Touching, states.OnTop = type('Touching', (), {}), type('OnTop', (), {})
    monkeypatch.setitem(sys.modules, 'omnigibson', ModuleType('omnigibson'))
    monkeypatch.setitem(sys.modules, 'omnigibson.object_states', states)
    constants = ModuleType('omnigibson.utils.constants'); constants.PrimType = NS(CLOTH='cloth')
    monkeypatch.setitem(sys.modules, 'omnigibson.utils.constants', constants)
    class PBD:
        def __init__(self, prim):
            self.prim = prim
        def __getattr__(self, method):
            name = method[3:-4].lower()
            return lambda: Attribute(name, self.prim.attributes[name])
    lazy = ModuleType('omnigibson.lazy'); lazy.pxr = NS(PhysxSchema=NS(PhysxPBDMaterialAPI=PBD))
    monkeypatch.setitem(sys.modules, 'omnigibson.lazy', lazy)
    item, cloth = Object('item', [0, 0, .05]), Object('cloth', [0, 0, .06], (.4, .4, .12))
    cloth.prim_type, cloth.mass = 'cloth', 0.  # BaseObject.mass excludes cloth; native root mass is essential.
    points = np.array([[x, y, .11 if x == y == 0 else .01] for x in (-.2, 0., .2) for y in (-.2, 0., .2)])
    faces = np.array([[0,3,4],[0,4,1],[1,4,5],[1,5,2],[3,6,7],[3,7,4],[4,7,8],[4,8,5]])
    cloth.root_link = NS(mass=1., scale=np.ones(3), faces=faces, points=points.copy(),
        particle_velocities=np.zeros_like(points), cloth_system=NS(mat_path='/material/pbd'),
        bend_stiffness=1., shear_stiffness=2., stretch_stiffness=3., damping=.2)
    cloth.root_link.compute_particle_positions = lambda: cloth.root_link.points.copy()
    floor = Object('floor', [0,0,-.05], (10,10,.1)); floor.category = 'floors'
    deck = Object('deck', [0,0,.35], (1.,1.,.7)); deck.fixed_base = True
    touching = {'cloth_item': True, 'item_deck': False}
    cloth.states = {states.Touching: NS(get_value=lambda obj: obj is item and touching['cloth_item'])}
    item.states = {states.Touching: NS(get_value=lambda obj: obj is deck and touching['item_deck']),
        states.OnTop: NS(get_value=lambda obj: obj is deck and touching['item_deck'])}
    prims = {f'/material/{name}': Prim(f'/material/{name}', color=[.2,.3,.4]) for name in ('item','cloth')}
    prims['/material/pbd'] = Prim('/material/pbd', friction=.5, drag=.1, lift=.2)
    sim = NS(current_time=0., stage=NS(GetPrimAtPath=lambda path: prims[path]))
    objects = {obj.name: obj for obj in (item, cloth, floor, deck)}
    session = NS(native_reset_id='reset', scene_name='scene', room_name='room',
        pipeline=NS(og=NS(sim=sim)), env=NS(scene=NS(object_registry=lambda field, name: objects.get(name))))
    contract = {'schema': support.REVISION, 'source_identity': 'source', 'source_sha256': 'hash',
        'bindings': {key: {'name': obj.name, 'category': obj.category, 'model': obj.model} for key,obj in (('item',item),('cloth',cloth))}}
    parameters = {'task_revision': support.REVISION, 'source_identity': 'source', 'source_sha256': 'hash',
        'deformable_cover': contract, 'source_item_xy': [0,0], 'floor_height_estimate_m': 0.}
    return NS(session=session, sim=sim, item=item, cloth=cloth, floor=floor, deck=deck,
        parameters=parameters, touching=touching, prims=prims, states=states, delta=np.array([0.,0.,.7]))


def capture(world):
    result = support.capture_deformable_source(world.session, world.item, world.cloth, world.parameters)
    assert result['accepted'], result
    return result


def translate(world, receipt):
    world.item.position += world.delta; world.cloth.position += world.delta
    world.cloth.root_link.points += world.delta
    proof = support.verify_initial_mesh_translation(world.session, world.cloth, receipt, world.delta)
    assert proof['accepted'], proof
    receipt['initial_mesh_translation'] = proof
    world.sim.current_time = 1.
    world.touching['item_deck'] = True
    world.item.contacts = [NS(body0=world.item.prim_path+'/body', body1=world.deck.prim_path+'/top',
        position=np.array([0,0,.7]), normal=np.array([0,0,1]))]
    return receipt


def placement(world, receipt):
    return support.measure_deformable_placement(world.session, world.item, world.cloth, world.deck,
        receipt, world.delta, .7)


def test_capture_exact_world_mesh_topology_mass_and_true_cover(world):
    receipt = capture(world)
    assert receipt['cloth']['mass_kg'] == 1. and world.cloth.mass == 0.
    assert receipt['points_world_m'] == world.cloth.root_link.points.tolist()
    assert receipt['faces'] == world.cloth.root_link.faces.tolist()
    assert receipt['source_cover']['above_item_overlap_faces'] > 0
    assert receipt['group_bounds_m'][0][0] == -.2
    assert receipt['pending_native_action_gates'] == support.PENDING_ACTION_GATES
    json.dumps(receipt, allow_nan=False)


def test_multi_object_native_bridge_requires_full_coverage_and_preserves_legacy_contact(world, monkeypatch):
    monkeypatch.setattr(support, 'native_cloth_contact_paths', lambda *args, **kwargs: set())
    world.touching['cloth_item'] = False
    assert not support.capture_deformable_source(
        world.session, world.item, world.cloth, world.parameters)['accepted']
    world.parameters['cover_revision'] = 'full_cover_redrape_v1'
    assert not support.capture_deformable_source(
        world.session, world.item, world.cloth, world.parameters)['accepted']
    world.parameters['protected_extra_keys'] = ['covered_bun']
    receipt = capture(world)
    assert receipt['source_cover']['native_bridge_allowed']
    assert receipt['source_cover']['native_item_cloth_touching'] is False
    translate(world, receipt)
    assert placement(world, receipt)['accepted']
    world.cloth.root_link.points[:, 0] += .3
    assert not support.capture_deformable_source(
        world.session, world.item, world.cloth, world.parameters)['accepted']


def test_full_particle_query_recovers_contact_missed_by_sampled_keypoints(world, monkeypatch):
    world.touching['cloth_item'] = False
    world.parameters['cover_revision'] = 'full_cover_redrape_v1'
    calls = []
    def overlap(radius, position, callback, any_hit):
        calls.append((radius, position))
        callback(NS(rigid_body=world.item.prim_path + '/base_link'))
    world.cloth.root_link.cloth_system.particle_contact_offset = .015
    sys.modules['omnigibson'].sim = NS(psqi=NS(overlap_sphere=overlap))
    receipt = capture(world)
    assert len(calls) == 1  # Only the central particle can reach the target AABB.
    assert receipt['source_cover']['native_item_cloth_touching']
    assert receipt['source_cover']['sampled_item_cloth_touching'] is False
    assert receipt['source_cover']['full_particle_contact_query']
    assert receipt['source_cover']['full_particle_contact_query_performed']


def test_root_pose_translation_alone_cannot_fake_particle_translation(world):
    receipt = capture(world)
    world.cloth.position += world.delta
    result = support.verify_initial_mesh_translation(world.session, world.cloth, receipt, world.delta)
    assert not result['accepted'] and not result['checks']['exact_world_particle_translation']
    world.cloth.root_link.points += world.delta
    world.sim.current_time = .01
    result = support.verify_initial_mesh_translation(world.session, world.cloth, receipt, world.delta)
    assert not result['accepted'] and not result['checks']['same_native_time_before_physics']


def test_room_coordinate_roundoff_does_not_allow_millimetre_mesh_changes(world):
    offset = np.array([-37., 12., 0.])
    world.item.position += offset
    world.cloth.position += offset
    world.cloth.root_link.points += offset
    receipt = capture(world)
    world.cloth.root_link.points += world.delta
    world.cloth.root_link.points[0, 0] += 2.8e-5
    result = support.verify_initial_mesh_translation(world.session, world.cloth, receipt, world.delta)
    assert result['accepted']
    assert result['numerical_tolerance_m'] <= 1e-4
    world.cloth.root_link.points[0, 0] += .001
    result = support.verify_initial_mesh_translation(world.session, world.cloth, receipt, world.delta)
    assert not result['accepted']


def test_real_post_settle_deformation_and_root_change_are_allowed(world):
    receipt = translate(world, capture(world))
    world.cloth.root_link.points[:,0] *= 1.05
    world.cloth.root_link.points[4,2] += .005
    world.cloth.position += [.03, .02, -.01]
    result = placement(world, receipt)
    assert result['accepted'], result
    assert result['deformed_from_translated_source'] and not result['cloth_root_pose_is_rigid_invariant']
    assert result['native_cover']['above_item_overlap_faces'] > 0
    assert result['native_mesh']['centroid_m']
    assert result['native_item_support_contact_positions_m']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('failure', ['topology','cloth_scale','root_scale','mass','material','spring','lost_cover','lost_contact',
    'lost_support','side_contact','below_deck','item_scale','item_pose','small_deck','stale_reset','missing_translation','nonfinite'])
def test_failed_physics_topology_cover_or_support_is_not_accepted(world, failure):
    receipt = translate(world, capture(world))
    if failure == 'topology': world.cloth.root_link.faces[0,1] = 2
    elif failure == 'cloth_scale': world.cloth.scale *= .5
    elif failure == 'root_scale': world.cloth.root_link.scale *= .5
    elif failure == 'mass': world.cloth.root_link.mass = 2.
    elif failure == 'material': world.prims['/material/pbd'].attributes['friction'] = .9
    elif failure == 'spring': world.cloth.root_link.damping = .4
    elif failure == 'lost_cover': world.cloth.root_link.points[:,0] += 1.
    elif failure == 'lost_contact': world.touching['cloth_item'] = False
    elif failure == 'lost_support': world.touching['item_deck'] = False
    elif failure == 'side_contact': world.item.contacts[0].normal = np.array([1,0,0])
    elif failure == 'below_deck': world.cloth.root_link.points[0,2] = .5
    elif failure == 'item_scale': world.item.scale *= 2.
    elif failure == 'item_pose': world.item.position[0] += .1
    elif failure == 'small_deck': world.deck.extent[:2] = [.15,.15]
    elif failure == 'stale_reset': world.session.native_reset_id = 'other_reset'
    elif failure == 'missing_translation': receipt.pop('initial_mesh_translation')
    else: world.cloth.root_link.points[0,0] = np.nan
    assert not placement(world, receipt)['accepted']


def test_native_velocities_are_reported_and_missing_is_not_fabricated_zero(world):
    receipt = translate(world, capture(world))
    world.cloth.root_link.particle_velocities[:,0] = .2
    result = placement(world, receipt)
    velocity = result['native_mesh']['particle_velocities']
    assert result['accepted'] and velocity['maximum_speed_m_s'] == .2
    assert not velocity['stationarity_certified']
    del world.cloth.root_link.particle_velocities
    result = placement(world, receipt)
    assert result['accepted']
    velocity = result['native_mesh']['particle_velocities']
    assert not velocity['available'] and 'maximum_speed_m_s' not in velocity


@pytest.mark.parametrize('field,expected_type,expected_dtype,expected_shape,nonfinite_count', [
    ('mesh', 'numpy.ndarray', 'float64', [9, 3], 1),
    ('velocity', 'numpy.ndarray', 'float64', [9, 3], 2),
    ('none_mesh', 'builtins.NoneType', 'object', [], 1),
    ('nonnumeric_mesh', 'builtins.str', '<U16', [], None),
    ('none_velocity', None, None, None, None),
])
def test_postsettle_numeric_diagnostics_distinguish_mesh_velocity_and_data_source(
        world, field, expected_type, expected_dtype, expected_shape, nonfinite_count):
    receipt = translate(world, capture(world))
    if field == 'mesh':
        world.cloth.root_link.points[1, 2] = np.nan
    elif field == 'velocity':
        world.cloth.root_link.particle_velocities[1, :2] = [np.inf, np.nan]
    elif field == 'none_mesh':
        world.cloth.root_link.compute_particle_positions = lambda: None
    elif field == 'nonnumeric_mesh':
        world.cloth.root_link.compute_particle_positions = lambda: 'unavailable-data'
    else:
        world.cloth.root_link.particle_velocities = None
    result = placement(world, receipt)
    if field == 'none_velocity':
        # The already-optional native None velocity is still optional, with no
        # made-up zero and no new rejection/acceptance policy.
        assert result['available'] and result['accepted']
        assert not result['native_mesh']['particle_velocities']['available']
        return
    assert result['available'] is False and result['accepted'] is False
    assert result['measurement_function'] == 'measure_deformable_placement'
    assert result['measurement_stage'] == 'native_mesh'
    assert result['exception_type'] == 'ValueError'
    diagnostic = result['numeric_field_diagnostic']
    assert diagnostic['field'] == ('cloth.particle_velocities_m_s' if field == 'velocity' else 'cloth.world_particle_positions_m')
    assert result['measurement_field'] == diagnostic['field']
    assert diagnostic['supplied_type'] == expected_type
    assert diagnostic['dtype'] == expected_dtype
    assert diagnostic['shape'] == expected_shape
    assert diagnostic['nonfinite_count'] == nonfinite_count
    assert 0 < len(result['traceback']) <= 6
    assert all(set(frame) == {'filename', 'line_number', 'function'} for frame in result['traceback'])
    assert 'points_world_m' not in result and 'raw_values' not in diagnostic
    assert len(json.dumps(result, allow_nan=False)) < 5000


def test_native_mesh_getter_failure_preserves_exception_stage_and_field(world):
    receipt = translate(world, capture(world))
    def broken_getter():
        raise RuntimeError('native cloth data source unavailable')
    world.cloth.root_link.compute_particle_positions = broken_getter
    result = placement(world, receipt)
    assert not result['available'] and not result['accepted']
    assert result['measurement_stage'] == 'native_mesh'
    assert result['exception_type'] == 'RuntimeError'
    assert result['measurement_field'] == 'cloth.world_particle_positions_m'
    assert result['numeric_field_diagnostic'] is None
    assert 'native cloth data source unavailable' in result['reason']


def test_none_native_pose_return_identifies_field_without_changing_unknown_status(world):
    receipt = translate(world, capture(world))
    world.cloth.get_position_orientation = lambda: None
    result = placement(world, receipt)
    assert not result['available'] and not result['accepted']
    assert result['measurement_stage'] == 'native_cloth_physics'
    assert result['measurement_field'] == 'cloth.root_pose'
    assert result['exception_type'] == 'TypeError'


@pytest.mark.parametrize('field,stage,label', [
    ('scale', 'native_cloth_physics', 'cloth.scale'),
    ('mass', 'native_cloth_physics', 'cloth.mass_kg'),
    ('material', 'native_cloth_physics', 'cloth.pbd_physics.friction'),
    ('spring', 'native_cloth_physics', 'cloth.spring_settings.damping'),
    ('faces', 'native_mesh', 'cloth.faces'),
    ('translation', 'receipt_translation', 'receipt.common_translation_m'),
    ('support', 'selected_support_bounds', 'support.aabb_low_m'),
])
def test_failed_numeric_fields_keep_precise_guard_diagnostics(world, field, stage, label):
    receipt = translate(world, capture(world))
    if field == 'scale': world.cloth.scale[0] = np.nan
    elif field == 'mass': world.cloth.root_link.mass = np.inf
    elif field == 'material': world.prims['/material/pbd'].attributes['friction'] = None
    elif field == 'spring': world.cloth.root_link.damping = np.nan
    elif field == 'faces': world.cloth.root_link.faces = np.full((8, 3), np.nan)
    elif field == 'translation': world.delta[0] = np.nan
    else: world.deck.position[0] = np.nan
    result = placement(world, receipt)
    assert not result['available'] and not result['accepted']
    assert result['measurement_stage'] == stage
    assert result['measurement_field'] == label
    assert result['numeric_field_diagnostic']['field'] == label
    json.dumps(result, allow_nan=False)


def test_capture_and_initial_translation_report_their_own_failed_stage(world):
    world.cloth.root_link.points[0, 0] = np.nan
    result = support.capture_deformable_source(world.session, world.item, world.cloth, world.parameters)
    assert result['measurement_function'] == 'capture_deformable_source'
    assert result['measurement_stage'] == 'native_mesh'
    assert result['measurement_field'] == 'cloth.world_particle_positions_m'
    assert not result['available'] and not result['accepted']
    world.cloth.root_link.points[0, 0] = -.2
    receipt = capture(world)
    result = support.verify_initial_mesh_translation(world.session, world.cloth, receipt, [0., np.nan, .7])
    assert result['measurement_function'] == 'verify_initial_mesh_translation'
    assert result['measurement_stage'] == 'receipt_translation'
    assert result['measurement_field'] == 'receipt.common_translation_m'
    assert not result['available'] and not result['accepted']


def test_stale_receipt_context_is_an_unavailable_guard_not_numeric_failure(world):
    receipt = translate(world, capture(world))
    world.session.native_reset_id = 'another-reset'
    result = placement(world, receipt)
    assert not result['available'] and not result['accepted']
    assert result['measurement_stage'] == 'source_receipt_context'
    assert result['exception_type'] == 'ValueError'
    assert result['numeric_field_diagnostic'] is None
    assert 'another source/reset/task context' in result['reason']


def test_new_qualifier_requires_current_source_reset_receipt_and_actual_helper(world, monkeypatch):
    receipt = translate(world, capture(world))
    world.session.category_interaction_setup = {'accepted': True, 'relocated': True,
        'object_names': [world.item.name, world.cloth.name], 'support_name': world.deck.name,
        'support_prim_path': world.deck.prim_path, 'support_surface_z_m': .7,
        'group_translation_m': world.delta.tolist(),
        'object_translations_m': {obj.name: world.delta.tolist() for obj in (world.item, world.cloth)},
        'deformable_cover': receipt}
    row = {'gripper': {'object_name': None, 'constraint_active': False}, 'floor_top_m': 0.,
        'item_position': world.item.position.tolist(), 'item_touching_task_support': True,
        'above_item_overlap_faces': 4, 'item_touching_cloth': True, 'item_linear_speed_m_s': 0.,
        'working_hand_cloth_contacts': [], 'other_robot_cloth_contacts': [], 'robot_in_original_room': True,
        'rigid_assisted_grasp_mode': True}
    monkeypatch.setattr(deformable, '_state', lambda *args: deepcopy(row))
    config = {'type': 'category_operation', 'module': deformable.MODULE, 'operation': deformable.OPERATION,
        'scene': 'scene', 'room': 'room', 'bindings': {'item':'item','cloth':'cloth','floor':'floor'}, 'parameters': world.parameters}
    bound = {'item':world.item,'cloth':world.cloth,'floor':world.floor}
    result = deformable.qualify_operation(world.session, NS(), config, bound)
    assert result['accepted'], result
    assert result['native_deformable_placement']['accepted']
    world.session.category_interaction_setup.pop('deformable_cover')
    assert not deformable.qualify_operation(world.session, NS(), config, bound)['accepted']
    config['parameters'] = {**world.parameters, 'task_revision': deformable.TASK_REVISION}
    assert deformable.qualify_operation(world.session, NS(), config, bound)['accepted']


def test_physx_structured_contact_vectors_keep_native_xyz_and_finite_guard():
    dtype = [('z', '<f4'), ('x', '<f4'), ('y', '<f4')]
    scalar = np.array((.5, 12., -19.), dtype=dtype)[()]
    assert np.allclose(support._array(scalar, 'item.support_contact.position_m'), [12., -19., .5])
    batch = np.array([(.5, 12., -19.), (.6, 13., -20.)], dtype=dtype)
    assert np.allclose(support._array(batch), [[12., -19., .5], [13., -20., .6]])
    batch['x'][0] = np.nan
    with pytest.raises(ValueError) as captured:
        support._array(batch, 'item.support_contact.position_m')
    assert captured.value.numeric_field_diagnostic['nonfinite_count'] == 1
    assert captured.value.measurement_field == 'item.support_contact.position_m'
    with pytest.raises(ValueError):
        support._array(np.array((1., 2.), dtype=[('x', '<f4'), ('y', '<f4')]))


def test_all_actual_core_sources_generate_explicit_derivatives_and_preserve_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(deformable, '_robot_start', lambda *args: (None, {'reason':'CPU_only'}))
    members = [row for row in json.loads(CORE.read_text())['tasks'] if row.get('subcategory')=='Deformable']
    assert len(members)==4
    for member in members:
        identity = member['source']; path = ROOT/'ESI/dataset/json_clean'/identity
        source = SourceInstance(path,identity,'Physical Structure','Deformable'); original=path.read_bytes()
        result=deformable.CategoryConverter().convert(source, ConversionContext(tmp_path,config={'task_edit2_deformable':True}))
        task,scene=result['task_definition'],result['scene_definition']
        prepare_definitions(source,scene,task,result['interaction_policy']); task_definition(task); SceneDefinition.model_validate(scene)
        p=task['private']['evaluation']['parameters']; policy=result['interaction_policy']
        assert p['task_revision']==support.REVISION and task['private']['evaluation']['operation']==deformable.OPERATION
        assert policy['group_bindings']==['item','cloth'] and policy['support_bindings']==['item']
        assert policy['manipulation_bindings']==['item'] and policy['layout_parameters']['deformable_cover']==p['deformable_cover']
        assert scene['scene']['variants']['original']['patches']==[] and path.read_bytes()==original
        legacy=deformable.CategoryConverter().convert(source,ConversionContext(tmp_path))
        assert legacy['task_revision']==deformable.TASK_REVISION
        assert 'layout_parameters' not in legacy['interaction_policy']
