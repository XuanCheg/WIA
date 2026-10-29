"""Regress native structured contact vectors without simulation or GPU access.

Run: PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q \
  -p no:cacheprovider tests/test_wia_native_contact_vectors.py tests/test_task_edit2_geometry.py
"""
from types import SimpleNamespace

import numpy as np
import pytest

from wia.authoring.converters import spatial_relations_geometric_configuration as geometry
from wia.authoring.converters import spatial_relations_physical_contact as physical_contact


def contact_world(vector=lambda values: values, *, foreign_floor=False):
    """Create two native-shaped wheel contacts with nonzero measured coordinates."""
    floor = SimpleNamespace(prim_path='/World/source_floor')
    wheels = {name: SimpleNamespace(prim_path='/World/robot/' + name)
              for name in ('left_wheel', 'right_wheel')}
    floor_path = floor.prim_path + ('_other' if foreign_floor else '') + '/base_link'
    contacts = [SimpleNamespace(body0=wheel.prim_path + '/collision', body1=floor_path,
                                position=vector([x, .125, .5]), normal=vector([0., 0., normal_z]))
                for (name, wheel), x, normal_z in zip(wheels.items(), (-.25, .25), (1., -1.))]
    bounds = np.array([-.3, -.2, .52]), np.array([.3, .2, 1.])
    robot = SimpleNamespace(prim_path='/World/robot', floor_touching_base_link_names=list(wheels),
                            links={**wheels, 'base_footprint': SimpleNamespace(aabb=bounds)},
                            base_footprint_link_name='base_footprint', contact_list=lambda: contacts,
                            aabb=bounds, get_position_orientation=lambda: (
                                np.array([0., .125, .52]), np.array([0., 0., 0., 1.])))
    scene = SimpleNamespace(seg_map=SimpleNamespace(get_room_instance_by_point=lambda point: 'room_0'))
    session = SimpleNamespace(robot=robot, room_name='room_0', env=SimpleNamespace(scene=scene))
    return session, floor, {'quaternion_xyzw': [0., 0., 0., 1.]}, contacts


def structured_vector(values, *, scalar):
    # PhysX fields need not be declared in coordinate order.
    result = np.zeros((), dtype=[('z', np.float32), ('x', np.float32), ('y', np.float32)])
    for name, value in zip(('x', 'y', 'z'), values):
        result[name] = value
    return result[()] if scalar else result


@pytest.mark.parametrize('scalar', [False, True])
@pytest.mark.parametrize('foreign_floor', [False, True])
def test_native_support_structured_contacts_match_plain_floor_decisions_and_coordinates(scalar, foreign_floor):
    plain_session, plain_floor, pose, _ = contact_world(foreign_floor=foreign_floor)
    expected = geometry._native_support(plain_session, plain_floor, pose, .5)
    session, floor, pose, _ = contact_world(lambda values: structured_vector(values, scalar=scalar),
                                          foreign_floor=foreign_floor)
    measured = geometry._native_support(session, floor, pose, .5)
    assert measured == expected
    assert measured['supported_upright'] is not foreign_floor
    key = 'other_external_contacts' if foreign_floor else 'source_floor_contacts'
    assert [row['position'] for row in measured[key]] == [[-.25, .125, .5], [.25, .125, .5]]
    assert [row['normal'] for row in measured[key]] == [[0., 0., 1.], [0., 0., -1.]]
    if foreign_floor:
        assert measured['supported_wheels'] == []
        assert measured['source_floor_contacts'] == []
    else:
        assert len(measured['supported_wheels']) == 2
        assert measured['other_external_contacts'] == []


@pytest.mark.parametrize('field', ['position', 'normal'])
@pytest.mark.parametrize('names', [('x', 'y'), ('x', 'y', 'z', 'w')])
def test_native_support_malformed_structured_fields_raise_instead_of_passing(field, names):
    session, floor, pose, contacts = contact_world()
    setattr(contacts[0], field, np.zeros((), dtype=[(name, np.float32) for name in names])[()])
    with pytest.raises(ValueError, match='Unsupported structured native interaction vector'):
        geometry._native_support(session, floor, pose, .5)


@pytest.mark.parametrize('field', ['position', 'normal'])
def test_native_support_nonfinite_structured_measurements_remain_errors(field):
    session, floor, pose, contacts = contact_world()
    setattr(contacts[0], field, structured_vector([0., 0., np.nan], scalar=True))
    with pytest.raises(ValueError, match='finite native measurements'):
        geometry._native_support(session, floor, pose, .5)


def test_native_support_batched_contact_vector_is_not_a_supported_scalar_contact():
    session, floor, pose, contacts = contact_world()
    contacts[0].position = np.array([structured_vector([-.25, .125, .5], scalar=True)])
    measured = geometry._native_support(session, floor, pose, .5)
    assert not measured['supported_upright']
    assert len(measured['supported_wheels']) == 1
    assert measured['other_external_contacts'][0]['position'] == [[-.25, .125, .5]]


@pytest.mark.parametrize('scalar', [False, True])
@pytest.mark.parametrize('foreign_floor', [False, True])
def test_actual_robot_state_structured_contacts_match_plain_support_and_coordinates(scalar, foreign_floor):
    plain_session, plain_floor, _, _ = contact_world(foreign_floor=foreign_floor)
    expected = physical_contact._robot_state(plain_session, plain_floor)
    session, floor, _, _ = contact_world(lambda values: structured_vector(values, scalar=scalar),
                                       foreign_floor=foreign_floor)
    measured = physical_contact._robot_state(session, floor)
    assert measured == expected
    assert measured['supported_upright'] is not foreign_floor
    assert measured['in_source_room'] is True
    assert measured['footprint_rooms'] == ['room_0'] * 9
    key = 'other_external_contacts' if foreign_floor else 'source_floor_wheel_contacts'
    assert [row['position'] for row in measured[key]] == [[-.25, .125, .5], [.25, .125, .5]]
    assert [row['normal'] for row in measured[key]] == [[0., 0., 1.], [0., 0., -1.]]
    if foreign_floor:
        assert measured['supported_wheels'] == []
        assert measured['source_floor_wheel_contacts'] == []
    else:
        assert len(measured['supported_wheels']) == 2
        assert measured['other_external_contacts'] == []


@pytest.mark.parametrize('field', ['position', 'normal'])
@pytest.mark.parametrize('names', [('x', 'y'), ('x', 'y', 'z', 'w')])
def test_actual_robot_state_malformed_structured_contact_fields_remain_errors(field, names):
    session, floor, _, contacts = contact_world()
    setattr(contacts[0], field, np.zeros((), dtype=[(name, np.float32) for name in names])[()])
    with pytest.raises(ValueError, match='Unsupported structured native interaction vector'):
        physical_contact._robot_state(session, floor)


@pytest.mark.parametrize('field', ['position', 'normal'])
def test_actual_robot_state_nonfinite_structured_contact_measurements_remain_errors(field):
    session, floor, _, contacts = contact_world()
    setattr(contacts[0], field, structured_vector([0., 0., np.nan], scalar=True))
    with pytest.raises(ValueError, match='finite native measurements'):
        physical_contact._robot_state(session, floor)
