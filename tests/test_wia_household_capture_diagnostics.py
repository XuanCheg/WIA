"""Check private household diagnostics with CPU-only native measurement fakes.

Run: PYTHONPATH=src python -B -m pytest -q -p no:cacheprovider
  tests/test_wia_household_capture_diagnostics.py
No simulator, queue, renderer, controller, or filesystem output is used.
"""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from wia.runtime.wia_capture_diagnostics import HOUSEHOLD_MODULE, capture_initial_review_diagnostics
from wia.runtime.wia_household_capture_diagnostics import SCHEMA


class Clock:
    def __init__(self, values=(8.25, 8.25)):
        self.values, self.reads = iter(values), 0

    @property
    def current_time(self):
        self.reads += 1
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture
def capture(monkeypatch):
    import wia.candidates.inventory as inventory
    import wia.authoring.converters.wia_dimensional as dimensional
    import wia.authoring.converters.wia_household_fit as household

    def forbidden(*args, **kwargs):
        raise AssertionError('Diagnostics cannot evaluate, qualify, control, step, render, or write.')

    for module in (household, dimensional):
        for name in ('create_evaluator', 'qualify_operation', '_qualification',
                     'evaluate_operation', 'qualify_native', '_measure', '_config', '_control', '_grasp'):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    parameters = {'revision': 'wia_household_fit_v1', 'candidates': {'short': 'native_short', 'long': 'native_long'},
        'fixture_keys': ['floor', 'wall'], 'destinations': ['floor'],
        'destination_interiors': {'floor': {'floor': 'floor', 'back': 'wall'}},
        'scoring': {'private_threshold': [0.001]},
        'aperture_bindings': {'floor': 'floor', 'roof': 'wall'}}
    evaluation = NS(type='category_operation', module=HOUSEHOLD_MODULE,
                    operation='select_pass_aperture', parameters=parameters)
    names = {'short': 'native_short', 'long': 'native_long', 'floor': 'fixed_floor', 'wall': 'fixed_wall'}
    bindings = {key: NS(name=name) for key, name in names.items()}
    verified, lookups, calls = [], [], []
    task = NS(definition=NS(id='household_actual_task_id', private=NS(evaluation=evaluation, bindings=bindings)),
              file_sha256='c' * 64, verify_unchanged=lambda: verified.append(True))
    objects = {name: NS(name=name, get_linear_velocity=lambda: [0.03, 0.04, 0.],
                        get_angular_velocity=lambda: [0., 0., 0.2],
                        set_position_orientation=forbidden, set_linear_velocity=forbidden,
                        set_angular_velocity=forbidden) for name in names.values()}

    def lookup(field, name):
        lookups.append((field, name))
        return objects.get(name)

    clock = Clock()
    setup = {'supported_items': {'native_short': {'support': 'native_table'},
                                'native_long': {'support': 'native_table'}, 'fixed_wall': {'support': None}}}
    session = NS(loaded_task=task, category_interaction_setup=setup,
        env=NS(scene=NS(object_registry=lookup), step=forbidden, render=forbidden),
        pipeline=NS(og=NS(sim=clock)), robot=NS(apply_action=forbidden),
        step=forbidden, render=forbidden, write=forbidden)
    native_inventory = {'schema': 'wia_candidate_native_inventory_v1', 'id': task.definition.id,
        'objects': [{'name': 'native_short', 'role': 'target'},
                    {'name': 'fixed_floor', 'role': 'background_collision_mesh'}],
        'robot': {'joint_calibration': {'source': 'native'}}, 'read_only': True}
    target_geometry = {'vertices_m': [[1., 2., 3.]], 'dimensions_m': [.1, .2, .3],
                       'body_vertices_m': [[.01, .02, .03]], 'private_native': True}
    fixture_geometry = {'aabb_m': [[0., 0., 0.], [1., 1., 1.]], 'quaternion_xyzw': [0., 0., 0., 1.]}
    interior = {'available': True, 'accepted': False, 'bounds_m': [[0., 0., 0.], [.2, .3, .4]]}
    aperture = {'available': True, 'accepted': False, 'bounds': {'y_min_m': .1, 'y_max_m': .2}}

    def export(session, **kwargs):
        calls.append(('inventory', deepcopy(kwargs)))
        return native_inventory

    def targets(obj):
        calls.append(('target_geometry', obj.name))
        return target_geometry

    def fixtures(obj):
        calls.append(('fixture_geometry', obj.name))
        return fixture_geometry

    def interiors(measured_fixtures, contract, scoring):
        calls.append(('interior', deepcopy(measured_fixtures), deepcopy(contract), deepcopy(scoring)))
        scoring['private_threshold'].append(123.)
        return interior

    def apertures(measured_fixtures, params):
        calls.append(('aperture', deepcopy(measured_fixtures)))
        return aperture

    monkeypatch.setattr(inventory, 'export_inventory', export)
    monkeypatch.setattr(household, 'household_geometry', targets)
    monkeypatch.setattr(dimensional, '_geometry', fixtures)
    monkeypatch.setattr(household, '_interior', interiors)
    monkeypatch.setattr(dimensional, '_aperture', apertures)
    return NS(session=session, task=task, evaluation=evaluation, parameters=parameters, setup=setup,
        clock=clock, verified=verified, lookups=lookups, calls=calls, objects=objects,
        inventory=inventory, native_inventory=native_inventory,
        dimensional=dimensional, household=household, target_geometry=target_geometry,
        fixture_geometry=fixture_geometry, interior=interior, aperture=aperture)


def test_exact_household_dispatch_needs_no_repair_and_keeps_inventory_and_geometry_private(capture):
    original_parameters, original_setup = deepcopy(capture.parameters), deepcopy(capture.setup)
    result = capture_initial_review_diagnostics(capture.session)
    assert result['schema'] == SCHEMA and result['private_only'] is result['available'] is True
    assert result['task_id'] == capture.task.definition.id and result['task_sha256'] == capture.task.file_sha256
    assert result['revision'] == 'wia_household_fit_v1' and result['module'] == HOUSEHOLD_MODULE
    assert result['native_qualification_run'] is result['physical_success_verified'] is False
    assert result['native_time_before_s'] == result['native_time_after_s'] == 8.25
    assert result['native_time_verified_unchanged'] is True and result['errors'] == []
    assert result['robot_actions'] == result['physics_steps'] == 0
    assert capture.clock.reads == 2 and capture.verified == [True]
    assert capture.calls[0] == ('inventory', {'identity': 'household_actual_task_id',
        'target_names': ['native_short', 'native_long'],
        'support_by_target': {'native_short': 'native_table', 'native_long': 'native_table'}})
    assert result['inventory'] == capture.native_inventory and result['inventory'] is not capture.native_inventory
    assert capture.lookups == [('name', 'native_short'), ('name', 'native_long'),
                               ('name', 'fixed_floor'), ('name', 'fixed_wall')]
    geometry = result['measured_geometry']
    assert geometry['available'] is True
    assert geometry['targets']['short']['native_geometry'] == capture.target_geometry
    assert geometry['targets']['short']['linear_velocity_m_s'] == [0.03, 0.04, 0.]
    assert geometry['targets']['short']['linear_speed_m_s'] == pytest.approx(.05)
    assert geometry['targets']['short']['angular_speed_rad_s'] == pytest.approx(.2)
    assert geometry['fixtures']['floor'] == capture.fixture_geometry
    assert geometry['destination_interiors']['floor'] == capture.interior
    assert geometry['measured_aperture'] == capture.aperture
    assert geometry['measured_aperture']['accepted'] is False
    assert capture.parameters == original_parameters and capture.setup == original_setup
    capture.native_inventory['robot']['joint_calibration']['source'] = 'changed'
    capture.target_geometry['vertices_m'][0][0] = 123.
    assert result['inventory']['robot']['joint_calibration']['source'] == 'native'
    assert geometry['targets']['short']['native_geometry']['vertices_m'][0][0] == 1.
    assert not {'public_task', 'observation', 'prompt', 'task_success', 'accepted'} & result.keys()


@pytest.mark.parametrize('fault', ['wrong_module', 'wrong_type', 'missing_task'])
def test_unrelated_dispatch_does_not_measure_read_clock_or_verify(capture, fault):
    if fault == 'wrong_module':
        capture.evaluation.module = HOUSEHOLD_MODULE + '_other'
    elif fault == 'wrong_type':
        capture.evaluation.type = 'stack_relations'
    else:
        del capture.session.loaded_task
    assert capture_initial_review_diagnostics(capture.session) is None
    assert capture.clock.reads == 0 and not capture.calls and not capture.verified and not capture.lookups


def test_sorting_has_interiors_and_no_aperture_or_synthetic_support(capture):
    capture.evaluation.operation = 'sort_intrinsic_dimension'
    del capture.parameters['aperture_bindings']
    del capture.session.category_interaction_setup
    result = capture_initial_review_diagnostics(capture.session)
    assert result['available'] is True
    assert result['measured_geometry']['measured_aperture'] is None
    assert not any(call[0] == 'aperture' for call in capture.calls)
    assert capture.calls[0][1]['support_by_target'] is None


@pytest.mark.parametrize('failure', ['inventory', 'target', 'fixture', 'interior', 'aperture', 'velocity'])
def test_measurement_errors_are_private_and_preserve_independent_native_evidence(capture, monkeypatch, failure):
    def broken(*args, **kwargs):
        raise ValueError('Native measurement unavailable: ' + failure)

    if failure == 'inventory':
        monkeypatch.setattr(capture.inventory, 'export_inventory', broken)
    elif failure == 'target':
        monkeypatch.setattr(capture.household, 'household_geometry', broken)
    elif failure == 'fixture':
        monkeypatch.setattr(capture.dimensional, '_geometry', broken)
    elif failure == 'interior':
        monkeypatch.setattr(capture.household, '_interior', broken)
    elif failure == 'aperture':
        monkeypatch.setattr(capture.dimensional, '_aperture', broken)
    else:
        capture.objects['native_short'].get_linear_velocity = broken
    result = capture_initial_review_diagnostics(capture.session)
    assert result['available'] is False and result['private_only'] is True
    assert 'Native measurement unavailable: ' + failure in result['error']
    assert result['native_qualification_run'] is result['physical_success_verified'] is False
    assert result['robot_actions'] == result['physics_steps'] == 0 and capture.clock.reads == 2
    if failure == 'inventory':
        assert result['inventory'] is None and result['measured_geometry']['available'] is True
    else:
        assert result['inventory'] == capture.native_inventory
        assert result['measured_geometry']['available'] is False


@pytest.mark.parametrize('times', [(8.25, 8.250000001), (float('nan'), 8.25),
    (8.25, float('inf')), (True, 8.25), (RuntimeError('Clock unavailable'), RuntimeError('Clock unavailable'))])
def test_unavailable_or_changed_clock_never_certifies_zero_actions_or_steps(capture, times):
    capture.session.pipeline.og.sim = Clock(times)
    result = capture_initial_review_diagnostics(capture.session)
    assert result['available'] is False and result['native_time_verified_unchanged'] is False
    assert result['robot_actions'] is result['physics_steps'] is None
    assert 'unchanged native simulation time' in result['error']
    assert capture.session.pipeline.og.sim.reads == 2
    assert bool(capture.calls) is (times[0] == 8.25)


@pytest.mark.parametrize('fault', ['changed_task', 'mismatched_binding', 'missing_object', 'wrong_object_identity'])
def test_task_and_exact_binding_failures_never_substitute_another_object(capture, fault):
    if fault == 'changed_task':
        def changed():
            raise RuntimeError('Task changed after loading.')
        capture.task.verify_unchanged = changed
    elif fault == 'mismatched_binding':
        capture.task.definition.private.bindings['short'].name = 'another_native_target'
    elif fault == 'missing_object':
        del capture.objects['native_short']
    else:
        capture.objects['native_short'].name = 'another_native_target'
    result = capture_initial_review_diagnostics(capture.session)
    assert result['available'] is False and result['private_only'] is True and result['error']
    assert result['robot_actions'] == result['physics_steps'] == 0
    if fault in ('changed_task', 'mismatched_binding'):
        assert result['inventory'] is result['measured_geometry'] is None and not capture.calls
    else:
        assert result['inventory'] == capture.native_inventory
        assert result['measured_geometry']['targets']['short']['available'] is False
        assert ('target_geometry', 'another_native_target') not in capture.calls
