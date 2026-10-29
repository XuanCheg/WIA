"""CPU contact/ray protocol tests; run ``python -B -m pytest -q
tests/test_task_edit2_native_floor_contacts.py`` in the behavior environment.
Fixtures measure the helper's gates, not native simulator qualification.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from wia.tasks import native_floor_contacts as module


FLOOR, BODY, ROBOT, WHEEL = '/World/floor', '/World/floor/base_link', '/World/robot', '/World/robot/wheel'


def contact(body0=WHEEL, body1=BODY, *, z=.0055917, normal=(0., 0., 1.)):
    return SimpleNamespace(body0=body0, body1=body1, position=np.array([1., 2., z]), normal=np.array(normal))


def hit(body=BODY, *, z=.0055917, normal=(0., 0., 1.)):
    return {'hit': True, 'rigidBody': body, 'collision': body + '/mesh',
        'position': np.array([1., 2., z]), 'normal': np.array(normal)}


@pytest.fixture
def native(monkeypatch):
    floor = SimpleNamespace(name='floor', category='floors', prim_path=FLOOR, visual_only=False,
        aabb=(np.array([0., 0., -.1]), np.array([4., 4., .02])))
    clock = SimpleNamespace(current_time=12.25)
    session = SimpleNamespace(env=SimpleNamespace(scene=SimpleNamespace(objects=[floor])),
        pipeline=SimpleNamespace(og=SimpleNamespace(sim=clock)), native_reset_id='reset-a')
    robot = SimpleNamespace(prim_path=ROBOT, floor_touching_base_link_names=['wheel'],
        links={'wheel': SimpleNamespace(prim_path=WHEEL)}, contact_list=lambda: [contact()])
    calls = []
    def ray(start, end, *, only_closest):
        calls.append((start, end, only_closest))
        return [hit(ROBOT + '/wheel', z=.03), hit()]
    monkeypatch.setattr(module, 'native_vertical_ray', ray)
    return session, robot, calls


def test_actual_mesh_offset_is_measured_not_nominal_threshold(native):
    session, robot, calls = native
    result = module.measure_native_floor_contacts(session, robot, [FLOOR], nominal_floor_z=0.)
    assert result['measurement_available'] is True
    assert result['floor_supported'] is True and result['collision_free'] is True
    assert result['floor_contact_positions_m'] == [[1., 2., .0055917]]
    row = result['contacts'][0]
    assert row['nominal_floor_offset_m'] > .005
    assert row['contact_surface_offset_m'] == 0.
    assert row['local_surface_z_m'] == .0055917
    assert len(calls) == 1 and calls[0][2] is False
    np.testing.assert_allclose(calls[0][:2], [[1., 2., .0555917], [1., 2., -.0444083]], atol=1e-15)
    assert session.pipeline.og.sim.current_time == 12.25
    assert result['provenance']['native_time_s'] == 12.25
    assert result['supported_wheel_link_paths'] == [WHEEL]
    json.dumps(result, allow_nan=False)


def test_nominal_agreement_cannot_replace_actual_local_floor(native, monkeypatch):
    session, robot, _ = native
    robot.contact_list = lambda: [contact(z=0.)]
    monkeypatch.setattr(module, 'native_vertical_ray', lambda *args, **kwargs: [hit(z=.02)])
    result = module.measure_native_floor_contacts(session, robot, [FLOOR], nominal_floor_z=0.)
    assert result['floor_supported'] is False
    assert result['measurement_available'] is True
    assert result['contacts'][0]['reason'] == 'native_contact_local_surface_mismatch'


@pytest.mark.parametrize('other', ['/World/chair/base_link', '/World/otherfloor/base_link'])
def test_nonfloor_or_otherfloor_ray_is_not_same_floor_evidence(native, monkeypatch, other):
    session, robot, _ = native
    monkeypatch.setattr(module, 'native_vertical_ray', lambda *args, **kwargs: [hit(other)])
    result = module.measure_native_floor_contacts(session, robot, [FLOOR])
    assert result['floor_supported'] is False
    assert result['contacts'][0]['reason'] == 'same_contact_floor_upward_surface_missing'


def test_unexpected_nonfloor_contact_remains_collision(native):
    session, robot, _ = native
    robot.contact_list = lambda: [contact(), contact(body1='/World/table/base_link')]
    result = module.measure_native_floor_contacts(session, robot, [FLOOR])
    assert result['floor_supported'] is True
    assert result['collision_free'] is False
    assert result['unexpected_contact_bodies'] == ['/World/table/base_link']
    assert result['contacts'][1]['position_m'] == [1., 2., .0055917]


def test_self_and_torso_contacts_cannot_prove_wheel_support(native):
    session, robot, calls = native
    robot.contact_list = lambda: [contact(body1=ROBOT + '/arm')]
    result = module.measure_native_floor_contacts(session, robot, [FLOOR])
    assert result['floor_supported'] is False and result['collision_free'] is True
    assert result['contacts'][0]['classification'] == 'self'
    assert not calls
    robot.contact_list = lambda: [contact(body0=ROBOT + '/torso')]
    result = module.measure_native_floor_contacts(session, robot, [FLOOR])
    assert result['floor_supported'] is False
    assert result['contacts'][0]['reason'] == 'floor_contact_is_not_exact_native_wheel_body'


def test_missing_wheel_api_stays_unavailable(native):
    session, robot, _ = native
    del robot.floor_touching_base_link_names
    result = module.measure_native_floor_contacts(session, robot, [FLOOR])
    assert result['floor_supported'] is None
    assert result['measurement_available'] is False
    assert any('Wheel identity' in error for error in result['errors'])


def test_missing_contact_api_and_empty_contacts_never_true(native):
    session, robot, calls = native
    del robot.contact_list
    result = module.measure_native_floor_contacts(session, robot, [FLOOR])
    assert result['floor_supported'] is None and result['collision_free'] is None
    assert not result['measurement_available']
    robot.contact_list = lambda: []
    result = module.measure_native_floor_contacts(session, robot, [FLOOR])
    assert result['floor_supported'] is False and result['collision_free'] is True
    assert not calls


def test_unavailable_ray_prevents_positive_support_even_with_another_good_contact(native, monkeypatch):
    session, robot, _ = native
    robot.contact_list = lambda: [contact(), contact(z=.006)]
    def ray(start, end, **kwargs):
        if start[2] > .0559:
            raise RuntimeError('native ray unavailable')
        return [hit()]
    monkeypatch.setattr(module, 'native_vertical_ray', ray)
    result = module.measure_native_floor_contacts(session, robot, [FLOOR])
    assert result['floor_supported'] is None
    assert not result['measurement_available']
    assert result['contacts'][1]['reason'] == 'native_floor_ray_unavailable'


@pytest.mark.parametrize('normal', [(0., 0., .49), (0., 0., -.49)])
def test_non_supporting_contact_normal_rejected(native, normal):
    session, robot, _ = native
    robot.contact_list = lambda: [contact(normal=normal)]
    assert module.measure_native_floor_contacts(session, robot, [FLOOR])['floor_supported'] is False


def test_negative_contact_normal_supported_but_downward_ray_not(native, monkeypatch):
    session, robot, _ = native
    robot.contact_list = lambda: [contact(normal=(0., 0., -1.))]
    assert module.measure_native_floor_contacts(session, robot, [FLOOR])['floor_supported'] is True
    monkeypatch.setattr(module, 'native_vertical_ray', lambda *args, **kwargs: [hit(normal=(0., 0., -1.))])
    assert module.measure_native_floor_contacts(session, robot, [FLOOR])['floor_supported'] is False


@pytest.mark.parametrize('field,value', [('visual_only', True), ('category', 'table')])
def test_floor_path_must_be_actual_collidable_loaded_floor(native, field, value):
    session, robot, calls = native
    setattr(session.env.scene.objects[0], field, value)
    result = module.measure_native_floor_contacts(session, robot, [FLOOR])
    assert result['floor_supported'] is None
    assert not result['measurement_available']
    assert not calls


def test_child_collider_path_does_not_impersonate_exact_native_wheel_link(native):
    session, robot, _ = native
    robot.contact_list = lambda: [contact(body0=WHEEL + '/visual_mesh')]
    result = module.measure_native_floor_contacts(session, robot, [FLOOR])
    assert result['floor_supported'] is False


def test_structured_physx_vectors_are_read_without_mutation(native):
    session, robot, _ = native
    original = contact()
    dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    original.position = np.array((1., 2., .0055917), dtype=dtype)
    original.normal = np.array((0., 0., 1.), dtype=dtype)
    before = original.position.copy()
    robot.contact_list = lambda: [original]
    result = module.measure_native_floor_contacts(session, robot, [FLOOR])
    assert result['floor_supported'] is True
    np.testing.assert_array_equal(original.position, before)
