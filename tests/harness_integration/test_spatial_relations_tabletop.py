"""Check supported Spatial Relations qualifiers without starting simulation.

Run: /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q
tests/test_spatial_relations_tabletop.py
Native-shaped CPU fixtures exercise reference adaptation, live support and floor
identity. They do not establish native navigation, graspability or task success.
"""
from types import ModuleType, SimpleNamespace
import math
import sys

import numpy as np
import pytest

from wia.authoring.converters import spatial_relations_geometric_configuration as triangle
from wia.authoring.converters import spatial_relations_linear_alignment as alignment
from wia.authoring.converters import spatial_relations_physical_contact as contact


@pytest.fixture
def world_factory(monkeypatch):
    states = ModuleType('omnigibson.object_states')
    states.Touching, states.OnTop = type('Touching', (), {}), type('OnTop', (), {})
    monkeypatch.setitem(sys.modules, 'omnigibson', ModuleType('omnigibson'))
    monkeypatch.setitem(sys.modules, 'omnigibson.object_states', states)

    class Prop:
        def __init__(self, name, position, support):
            self.name, self.prim_path = name, '/World/' + name
            self.position = np.asarray(position, dtype=float)
            self.category, self.model = 'bottle', 'source_model'
            self.visual_only, self.fixed_base = False, False
            self.support, self.touching, self.on_top = support, True, True
            self.contact = SimpleNamespace(body0=self.prim_path + '/base_link',
                body1=support.prim_path + '/top', normal=[0., 0., 1.],
                position=[*self.position[:2], float(self.position[2] - .1)])
            self.states = {
                states.Touching: SimpleNamespace(get_value=lambda other: self.touching and other is self.support),
                states.OnTop: SimpleNamespace(get_value=lambda other: self.on_top and other is self.support)}

        @property
        def aabb(self):
            return self.position - .1, self.position + .1

        def get_position_orientation(self):
            return self.position.copy(), np.array([0., 0., 0., 1.])

        def get_linear_velocity(self):
            return np.zeros(3)

        def get_angular_velocity(self):
            return np.zeros(3)

        def contact_list(self):
            return [self.contact] if self.touching else []

    def build(module, *, adapted=True):
        keys = triangle.VERTICES if module is triangle else module.OBJECTS
        source = np.array([[-.8, -.4, .1], [.8, -.4, .1], [0., .8, .1]] if module is triangle else
            [[-.8, 0., .1], [0., 0., .1], [.8, 0., .1]] if module is alignment else
            [[-.2, 0., .1], [.2, 0., .1]])
        floor = SimpleNamespace(name='floor', prim_path='/World/floor', category='floors', visual_only=False,
            aabb=(np.array([-10., -10., -.1]), np.array([10., 10., 0.])))
        table = SimpleNamespace(name='table', prim_path='/World/table', category='table', visual_only=False,
            aabb=(np.array([0., 1., 0.]), np.array([2., 3., .6])))
        positions = source.copy()
        if adapted:
            positions[:, :2] *= 1. if module is contact else .45
            positions += [1., 2., .6]
        props = [Prop('obj' + str(i + 1), position, table if adapted else floor)
                 for i, position in enumerate(positions)]
        bound = {**dict(zip(keys, props)), 'source_floor': floor}
        objects = [floor, table, *props]
        scene = SimpleNamespace(objects=objects,
            object_registry=lambda kind, name: next((obj for obj in objects if obj.name == name), None),
            seg_map=SimpleNamespace(get_room_instance_by_point=lambda xy: 'room_0'))
        robot_position = np.array([1., .8, 0.] if adapted else [0., -1.2, 0.])
        wheels = {name: SimpleNamespace(prim_path='/World/robot/' + name) for name in ('left_wheel', 'right_wheel')}
        footprint = SimpleNamespace(aabb=(robot_position - [.2, .2, 0.], robot_position + [.2, .2, .3]))
        robot_contacts = [SimpleNamespace(body0=wheel.prim_path, body1=floor.prim_path + '/base_link',
            normal=[0., 0., 1.], position=[*robot_position[:2], 0.]) for wheel in wheels.values()]
        robot = SimpleNamespace(prim_path='/World/robot', floor_touching_base_link_names=list(wheels),
            base_footprint_link_name='base_footprint', links={**wheels, 'base_footprint': footprint},
            aabb=footprint.aabb, contact_list=lambda: robot_contacts,
            get_position_orientation=lambda: (robot_position.copy(), np.array([0., 0., 0., 1.])),
            get_linear_velocity=lambda: np.zeros(3), get_angular_velocity=lambda: np.zeros(3))
        clock = SimpleNamespace(current_time=0.)
        session = SimpleNamespace(scene_name='scene', room_name='room_0', robot=robot,
            env=SimpleNamespace(scene=scene), pipeline=SimpleNamespace(og=SimpleNamespace(sim=clock)))
        if adapted:
            session.category_interaction_setup = {'accepted': True, 'relocated': True,
                'object_names': [obj.name for obj in props], 'support_prim_path': table.prim_path,
                'support_surface_z_m': .6, 'group_translation_m': [1., 2., .6],
                'object_translations_m': {obj.name: (position - original).tolist()
                    for obj, position, original in zip(props, positions, source)}}
        controller = SimpleNamespace(pose=lambda: {'position': robot_position.tolist(),
            'quaternion_xyzw': [0., 0., 0., 1.], 'yaw_rad': 0.},
            debug_gripper_state=lambda: {'object_name': None, 'holding': False,
                'constraint_active': False, 'gripper_open': True, 'release_pending': False})
        p = {'source_positions': source.tolist(), 'source_quaternions': [[0., 0., 0., 1.]] * len(keys),
            'source_floor_z_m': 0., 'initial_pose_tolerance_m': .035, 'object_preservation_m': .02,
            'object_orientation_tolerance_rad': math.radians(5), 'orientation_tolerance_rad': math.radians(5),
            'minimum_object_clearance_m': .4, 'maximum_cluster_distance_m': 2.5,
            'maximum_midpoint_distance_m': 3., 'return_distance_m': .25,
            'maximum_stop_speed_m_s': .03, 'maximum_stop_angular_speed_rad_s': .1,
            'maximum_native_sample_gap_s': .1, 'minimum_path_m': 1.5,
            'minimum_excursion_m': .75, 'stop_duration_s': .5}
        if module is contact:
            p.update(target=keys[0], anchor=keys[1], source_pair_midpoint_xy_m=[0., 0.],
                source_objects={key: {'position': original.tolist(), 'quaternion_xyzw': [0., 0., 0., 1.],
                    'aabb_min': (original - .1).tolist(), 'aabb_max': (original + .1).tolist()}
                    for key, original in zip(keys, source)})
        config = {'type': 'category_operation', 'module': module.MODULE, 'operation': module.OPERATIONS[0],
            'scene': 'scene', 'room': 'room_0', 'binding_keys': [*keys, 'source_floor'],
            'bindings': {key: obj.name for key, obj in bound.items()}, 'parameters': p}
        return SimpleNamespace(module=module, keys=keys, source=source, props=props, bound=bound,
            session=session, controller=controller, config=config, robot_contacts=robot_contacts,
            floor=floor, table=table)

    return build


@pytest.mark.parametrize('module', [triangle, alignment, contact])
@pytest.mark.parametrize('adapted', [False, True])
def test_qualifiers_accept_supported_adapted_and_legacy_layouts(world_factory, module, adapted):
    world = world_factory(module, adapted=adapted)
    result = module.qualify_operation(world.session, world.controller, world.config, world.bound)
    assert result['accepted'], result
    if adapted and module is not contact:
        assert np.max(np.linalg.norm(np.stack([obj.position for obj in world.props]) - world.source, axis=1)) > 1.
        assert all(all(row.values()) for row in result['checks']['native_object_task_support'])


@pytest.mark.parametrize('module', [triangle, alignment, contact])
def test_unaccepted_relocation_does_not_replace_source_reference(world_factory, module):
    world = world_factory(module)
    world.session.category_interaction_setup['accepted'] = False
    assert not module.qualify_operation(world.session, world.controller, world.config, world.bound)['accepted']


@pytest.mark.parametrize('module', [triangle, alignment, contact])
@pytest.mark.parametrize('failure', ['touching', 'on_top', 'side_contact'])
def test_qualifiers_reject_missing_upward_task_support(world_factory, module, failure):
    world = world_factory(module)
    if failure == 'side_contact':
        world.props[0].contact.normal = [1., 0., 0.]
    else:
        setattr(world.props[0], failure, False)
    assert not module.qualify_operation(world.session, world.controller, world.config, world.bound)['accepted']


@pytest.mark.parametrize('module', [triangle, alignment, contact])
@pytest.mark.parametrize('failure', ['table_under_wheels', 'nonwheel_floor_contact'])
def test_robot_requires_original_floor_wheel_contacts(world_factory, module, failure):
    world = world_factory(module)
    for row in world.robot_contacts:
        if failure == 'table_under_wheels':
            row.body1 = world.table.prim_path + '/base_link'
        else:
            row.body0 = '/World/robot/body'
    assert not module.qualify_operation(world.session, world.controller, world.config, world.bound)['accepted']


@pytest.mark.parametrize('module', [triangle, alignment])
def test_navigation_preserves_actual_episode_layout_and_live_support(world_factory, module, tmp_path):
    world = world_factory(module)
    evaluator = module.create_evaluator(world.session, world.controller, world.config, tmp_path)
    observation = SimpleNamespace(snapshot_id='initial', action_step=0)
    before = [obj.position.copy() for obj in world.props]
    evaluator.begin_episode(observation)
    measured = evaluator._measure()
    key = 'original_vertices_preserved' if module is triangle else 'original_arrangement_preserved'
    assert evaluator._checks(measured)[key]
    assert world.session.pipeline.og.sim.current_time == 0.
    for obj, position in zip(world.props, before):
        np.testing.assert_array_equal(obj.position, position)
    world.props[0].touching = False
    assert not evaluator._checks(evaluator._measure())[key]
    world.props[0].touching = True
    world.props[0].position[0] += .05
    assert not evaluator._checks(evaluator._measure())[key]


def test_contact_pair_cannot_use_different_reference_translations(world_factory):
    world = world_factory(contact)
    changed = world.props[0]
    changed.position[0] += .05
    world.session.category_interaction_setup['object_translations_m'][changed.name][0] += .05
    result = contact.qualify_operation(world.session, world.controller, world.config, world.bound)
    assert result['checks']['source_initial_poses_preserved']
    assert not result['checks']['reference_pair_common_translation']
    assert not result['accepted']
