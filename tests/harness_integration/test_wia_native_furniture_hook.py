"""Check the retained-furniture temporal initialization hook using CPU fakes.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_native_furniture_hook.py
No actual simulator, rendering, GPU workload or robot action is run.
"""
from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from wia.authoring.converters import wia_native_furniture as hook
from harness_integration.test_wia_native_furniture_poses import Furniture, spec, fixture_session


@pytest.fixture
def native_hook(monkeypatch):
    obj = Furniture()
    session = fixture_session(obj)
    session.scene_name = session.env.scene.scene_model = 'ExactNativeScene'
    setup = {'accepted': False, 'source': 'Existing temporal interaction receipt.'}
    session.category_interaction_setup = setup
    calls = {'steps': 0}
    og = ModuleType('omnigibson')

    def forbidden(*args, **kwargs):
        raise AssertionError('Furniture reset must not invoke robot controls or environment actions.')

    session.env.step = forbidden
    session.robot = NS(apply_action=forbidden)

    def step():
        calls['steps'] += 1
        assert obj.writes, 'Settling must follow the shared furniture pose initialization.'
        if calls['steps'] == 90:
            obj.position[2] = .48

    og.sim = NS(step=step)
    monkeypatch.setitem(sys.modules, 'omnigibson', og)
    parameters = {'scene_model': 'ExactNativeScene', 'native_object_poses': [spec()]}
    return session, obj, calls, parameters, setup


def test_hook_records_original_requested_and_actual_final_pose_without_replacing_setup(native_hook, capsys):
    session, obj, calls, parameters, setup = native_hook
    original_parameters = deepcopy(parameters)
    physical = deepcopy({key: getattr(obj, key) for key in ('scale', 'fixed_base', 'visual_only', 'joints', 'materials')})
    result = hook.initialize_scene(session, parameters)
    assert calls['steps'] == result['settle_steps'] == 90
    assert obj.still_calls == 1 and len(obj.writes) == 1
    receipt = result['native_object_poses'][0]
    assert receipt['original_pose'] == {'position': [1., 2., .5], 'orientation': [0., 0., 0., 1.]}
    assert receipt['requested_pose'] == {'position': [3., 4., .5], 'orientation': [0., 0., 1., 0.]}
    assert receipt['final_pose'] == {'position': [3., 4., .48], 'orientation': [0., 0., 1., 0.]}
    assert receipt['evidence_scope'] == 'initialization_only'
    assert getattr(session, hook.SESSION_ATTRIBUTE) == result
    assert session.category_interaction_setup is setup
    assert {key: getattr(obj, key) for key in physical} == physical
    assert parameters == original_parameters
    lines = capsys.readouterr().out.splitlines()
    prefix = '[wia] Native furniture initialization: '
    assert len(lines) == 1 and lines[0].startswith(prefix)
    logged = json.loads(lines[0][len(prefix):])
    assert logged == result
    assert set(logged['native_object_poses'][0]).issuperset({'original_pose', 'requested_pose', 'final_pose'})


@pytest.mark.parametrize('problem', ['guard', 'declared_scene', 'session_scene', 'native_scene'])
def test_hook_requires_initialization_and_exact_scene_before_any_pose_or_time_changes(native_hook, problem):
    session, obj, calls, parameters, setup = native_hook
    if problem == 'guard':
        session._category_initialization_active = False
    elif problem == 'declared_scene':
        parameters['scene_model'] = 'WrongScene'
    elif problem == 'session_scene':
        session.scene_name = 'WrongScene'
    else:
        session.env.scene.scene_model = 'WrongScene'
    with pytest.raises(RuntimeError if problem == 'guard' else ValueError):
        hook.initialize_scene(session, parameters)
    assert not obj.writes and calls['steps'] == 0
    assert session.category_interaction_setup is setup
    assert not hasattr(session, hook.SESSION_ATTRIBUTE)


@pytest.mark.parametrize('poses', [None, []])
def test_hook_without_pose_requests_does_not_read_objects_or_advance_time(native_hook, poses):
    session, obj, calls, parameters, setup = native_hook
    def forbidden(*args, **kwargs):
        raise AssertionError('An absent/empty native pose list must have no native object reads.')
    session.env.scene.object_registry = forbidden
    if poses is None:
        parameters.pop('native_object_poses')
    else:
        parameters['native_object_poses'] = poses
    result = hook.initialize_scene(session, parameters)
    assert result['native_object_poses'] == [] and result['settle_steps'] == 0
    assert calls['steps'] == 0 and not obj.writes
    assert session.category_interaction_setup is setup


def test_hook_uses_shared_all_identity_validation_before_any_setters_or_steps(native_hook):
    session, obj, calls, parameters, setup = native_hook
    parameters['native_object_poses'].append(spec('missing_native_object'))
    with pytest.raises(ValueError, match='retained native furniture'):
        hook.initialize_scene(session, parameters)
    assert not obj.writes and calls['steps'] == 0
    assert session.category_interaction_setup is setup


def test_contact_receipt_is_bounded_and_decodes_actual_structured_physx_vectors(native_hook, capsys):
    import numpy as np
    session, obj, calls, parameters, setup = native_hook
    def vector(values):
        record = np.zeros((), dtype=[('z', np.float32), ('x', np.float32), ('y', np.float32)])
        for field, value in zip(('x', 'y', 'z'), values):
            record[field] = value
        return record[()]
    def contacts():
        assert calls['steps'] == 90
        return [NS(body0='/World/chair/base_link', body1='/World/table/base_link_' + str(index),
                   position=vector([1., 2., .5]), normal=vector([0., 0., 1.])) for index in range(40)]
    obj.contact_list = contacts
    result = hook.initialize_scene(session, parameters)
    diagnostic = result['native_object_poses'][0]['contact_diagnostics']
    assert diagnostic['available'] is True and diagnostic['total_count'] == 40
    assert diagnostic['truncated'] is True and len(diagnostic['contacts']) == hook.CONTACT_LIMIT == 32
    assert diagnostic['contacts'][0] == {'available': True, 'body0': '/World/chair/base_link',
        'body1': '/World/table/base_link_0', 'position': [1., 2., .5], 'normal': [0., 0., 1.]}
    assert diagnostic['contacts'][-1]['body1'] == '/World/table/base_link_31'
    logged = json.loads(capsys.readouterr().out.split(': ', 1)[1])
    assert logged == result and session.category_interaction_setup is setup
    assert calls['steps'] == 90 and len(obj.writes) == 1


@pytest.mark.parametrize('problem', ['provider', 'nonfinite_vector'])
def test_unavailable_contact_diagnostics_do_not_change_reset_behavior_or_json_validity(native_hook, problem, capsys):
    session, obj, calls, parameters, setup = native_hook
    if problem == 'provider':
        def unavailable():
            raise RuntimeError('Unavailable native contacts. ' * 100)
        obj.contact_list = unavailable
    else:
        obj.contact_list = lambda: [NS(body0='/World/chair/base', body1='/World/table/base',
            position=[1., 2., .5], normal=[0., 0., float('nan')])]
    result = hook.initialize_scene(session, parameters)
    diagnostic = result['native_object_poses'][0]['contact_diagnostics']
    assert diagnostic['available'] is False
    error = diagnostic.get('measurement_error', diagnostic['contacts'][0].get('measurement_error') if diagnostic['contacts'] else None)
    assert error and len(error) <= hook.ERROR_LIMIT
    assert result['native_object_poses'][0]['final_pose']['position'] == [3., 4., .48]
    assert calls['steps'] == 90 and len(obj.writes) == 1
    assert session.category_interaction_setup is setup
    assert json.loads(capsys.readouterr().out.split(': ', 1)[1]) == result
