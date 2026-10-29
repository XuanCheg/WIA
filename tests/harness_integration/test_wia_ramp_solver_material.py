"""Check read-only native tensor material diagnostics using fake backend views.

Run: python -B -m pytest -q -p no:cacheprovider
  tests/test_wia_ramp_solver_material.py tests/test_wia_ramp_material_bindings.py
  tests/test_wia_ramp_material_math.py
No native simulator, GPU, USD SDK or model is started by these CPU tests.
"""
import builtins
import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

import wia.runtime.commands.check_wia_ramp_material as calibration


LINK_PATH = '/World/ramp/base_link'


class ReadOnly:
    """Fail loudly if a diagnostic attempts a mutator or lifecycle operation."""

    def __getattr__(self, name):
        if (name.startswith(('set_', 'Set', 'Apply', 'Define', 'Create'))
                or name in ('play', 'stop', 'reset', 'step', 'initialize', 'update_handles')):
            raise AssertionError('A read-only diagnostic attempted ' + name)
        raise AttributeError(name)


class Attribute(ReadOnly):
    def __init__(self, value):
        self.value = value

    def Get(self):
        return self.value

    def HasAuthoredValueOpinion(self):
        return True


class Prim(ReadOnly):
    def __init__(self, *, rigid=True, kinematic=False):
        self.values = {'physics:rigidBodyEnabled': rigid, 'physics:kinematicEnabled': kinematic}

    def GetAttribute(self, name):
        return Attribute(self.values[name])


class FakeView(ReadOnly):
    def __init__(self, values=None, *, paths=(LINK_PATH,), count=None, max_shapes=2,
                 valid=True, error=None):
        self.values = np.array([[[.8, .6, .1], [.5, .4, 0.]]]) if values is None else values
        self.prim_paths = paths
        self.count = len(paths) if count is None else count
        self.max_shapes, self.valid, self.error = max_shapes, valid, error
        self.get_calls, self.check_calls = 0, 0

    def check(self):
        self.check_calls += 1
        return self.valid

    def get_material_properties(self):
        self.get_calls += 1
        if self.error:
            raise self.error
        return self.values


class FakeDynamicLink(ReadOnly):
    def __init__(self, view=None, *, rigid=True, kinematic=False, meta=False, collision=True):
        self.prim_path = LINK_PATH
        self.prim = Prim(rigid=rigid, kinematic=kinematic)
        self.is_meta_link = meta
        self.collision_meshes = {'mesh': NS(prim_path=LINK_PATH + '/collision', collision_enabled=collision)}
        self._rigid_prim_view = NS(_physics_view=view)


class FakeStaticLink(FakeDynamicLink):
    def __init__(self, **kwargs):
        super().__init__(rigid=False, kinematic=True, **kwargs)


class FakeSimulationView(ReadOnly):
    def __init__(self, view, *, valid=True, error=None):
        self.view, self.is_valid, self.error = view, valid, error
        self.requests = []

    def create_rigid_body_view(self, path):
        self.requests.append(path)
        if self.error:
            raise self.error
        return self.view


class FakeSimulator(ReadOnly):
    def __init__(self, view):
        self.physics_sim_view = view


def install_sim(monkeypatch, view=None, **kwargs):
    simulation_view = FakeSimulationView(view, **kwargs)
    monkeypatch.setitem(sys.modules, 'omnigibson', NS(sim=FakeSimulator(simulation_view)))
    return simulation_view


def assert_uncertified(result):
    assert result['effective_solver_coefficients_verified'] is False
    assert result['solver_material_consumption_verified'] is False
    assert result['shape_slot_mapping']['available'] is False
    json.dumps(result, allow_nan=False)


def test_existing_valid_view_copies_values_and_keeps_unmapped_shape_slots(monkeypatch):
    view = FakeView()
    sim = install_sim(monkeypatch, error=AssertionError('Existing view must be reused.'))
    result = calibration._backend_material_readback(FakeDynamicLink(view))
    assert result['available'] is True
    assert result['view_source'] == 'link._rigid_prim_view._physics_view'
    assert result['view_count'] == 1 and result['view_max_shapes'] == 2
    assert result['view_prim_paths'] == [LINK_PATH] and result['view_link_index'] == 0
    assert result['view_request_path'] is None and sim.requests == []
    assert result['channels'] == ['static_friction', 'dynamic_friction', 'restitution']
    assert result['rigid_body_enabled']['value'] is True
    assert result['kinematic_enabled']['value'] is False
    assert result['link_class'].endswith('.FakeDynamicLink')
    assert result['collision_meshes'][0]['path'] == LINK_PATH + '/collision'
    assert result['link_shape_slot_material_properties'] == [[.8, .6, .1], [.5, .4, 0.]]
    assert result['copied_from_reusable_buffer'] is True and view.get_calls == 1
    view.values[:] = 99.
    assert result['raw_shape_slot_material_properties'][0][0] == [.8, .6, .1]
    assert result['link_shape_slot_material_properties'][0] == [.8, .6, .1]
    assert_uncertified(result)


def test_existing_batch_view_uses_unique_exact_link_row_without_assigning_mesh_slots(monkeypatch):
    values = np.array([[[1., 2., .1]], [[3., 4., .2]]])
    view = FakeView(values, paths=('/World/other/base_link', LINK_PATH), max_shapes=1)
    install_sim(monkeypatch)
    result = calibration._backend_material_readback(FakeDynamicLink(view))
    assert result['available'] is True and result['view_link_index'] == 1
    assert result['raw_shape_slot_material_properties'] == values.tolist()
    assert result['link_shape_slot_material_properties'] == [[3., 4., .2]]
    assert_uncertified(result)


def test_missing_existing_view_creates_only_exact_path_read_view(monkeypatch):
    view = FakeView()
    sim = install_sim(monkeypatch, view)
    result = calibration._backend_material_readback(FakeDynamicLink())
    assert result['available'] is True
    assert sim.requests == [LINK_PATH]
    assert result['view_request_path'] == LINK_PATH
    assert 'create_rigid_body_view' in result['view_source']
    assert_uncertified(result)


@pytest.mark.parametrize('values,message', [
    (np.full((1, 2, 3), np.nan), 'nonfinite'),
    (np.full((1, 2, 3), np.inf), 'nonfinite'),
    (np.zeros((1, 2, 2)), 'array shape'),
    (np.zeros((2, 3)), 'array shape'),
])
def test_nonfinite_and_wrong_shape_backend_arrays_are_unavailable(values, message):
    result = calibration._backend_material_readback(FakeDynamicLink(FakeView(values)))
    assert result['available'] is False
    assert message in result['errors'][0]
    assert 'raw_shape_slot_material_properties' not in result
    assert_uncertified(result)


@pytest.mark.parametrize('kwargs', [
    {'paths': ('/World/another_link',)},
    {'paths': (LINK_PATH, LINK_PATH)},
    {'paths': (LINK_PATH,), 'count': 2},
    {'paths': (LINK_PATH,), 'count': True},
    {'paths': (LINK_PATH,), 'max_shapes': 0},
])
def test_bad_count_or_path_mapping_never_reads_or_assigns_material_rows(kwargs):
    view = FakeView(**kwargs)
    result = calibration._backend_material_readback(FakeDynamicLink(view))
    assert result['available'] is False and result['errors']
    assert view.get_calls == 0
    assert 'raw_shape_slot_material_properties' not in result
    assert_uncertified(result)


def test_created_exact_path_view_must_not_include_other_links(monkeypatch):
    view = FakeView(paths=(LINK_PATH, '/World/other_link'))
    sim = install_sim(monkeypatch, view)
    result = calibration._backend_material_readback(FakeDynamicLink())
    assert result['available'] is False and sim.requests == [LINK_PATH]
    assert 'only the requested link' in result['errors'][0]
    assert view.get_calls == 0
    assert_uncertified(result)


def test_static_link_without_backend_body_is_explicitly_unavailable(monkeypatch):
    view = FakeView(np.empty((0, 0, 3)), paths=(), max_shapes=0)
    sim = install_sim(monkeypatch, view)
    result = calibration._backend_material_readback(FakeStaticLink())
    assert result['available'] is False
    assert result['rigid_body_enabled']['value'] is False
    assert result['kinematic_enabled']['value'] is True
    assert result['link_class'].endswith('.FakeStaticLink')
    assert result['view_count'] == 0 and result['view_max_shapes'] == 0
    assert 'static link' in result['reason']
    assert sim.requests == [LINK_PATH] and view.get_calls == 0
    assert_uncertified(result)


@pytest.mark.parametrize('failure', ['get', 'create', 'invalid_sim', 'invalid_existing'])
def test_native_exceptions_and_invalid_views_remain_nonfatal_and_uncertified(monkeypatch, failure):
    view = FakeView(error=RuntimeError('Backend read failed.')) if failure == 'get' else None
    kwargs = {}
    if failure == 'create':
        kwargs['error'] = RuntimeError('No backend body for this path.')
    elif failure == 'invalid_sim':
        kwargs['valid'] = False
    elif failure == 'invalid_existing':
        view = FakeView(valid=False)
        kwargs['error'] = RuntimeError('No fallback backend body.')
    install_sim(monkeypatch, **kwargs)
    result = calibration._backend_material_readback(FakeDynamicLink(view))
    assert result['available'] is False and result['errors']
    assert_uncertified(result)


@pytest.mark.parametrize('meta,collision', [(True, True), (False, False), (False, None)])
def test_meta_or_unmeasurably_enabled_collision_links_never_access_tensor_view(monkeypatch, meta, collision):
    view = FakeView()
    sim = install_sim(monkeypatch, error=AssertionError('Excluded link must not create a view.'))
    result = calibration._backend_material_readback(FakeDynamicLink(view, meta=meta, collision=collision))
    assert result['available'] is False and result['reason']
    assert view.check_calls == view.get_calls == 0 and sim.requests == []
    assert_uncertified(result)


def test_missing_loaded_simulator_never_imports_or_initializes_omnigibson(monkeypatch):
    monkeypatch.delitem(sys.modules, 'omnigibson', raising=False)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != 'omnigibson', 'Readback must not import or initialize the simulator.'
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    result = calibration._backend_material_readback(FakeStaticLink())
    assert result['available'] is False
    assert 'not already loaded' in result['errors'][0]
    assert_uncertified(result)


def test_materials_retains_legacy_fields_when_backend_and_optional_usd_are_unavailable(monkeypatch):
    from wia.authoring.converters import physical_dynamics_inclined_plane
    legacy = [{'mesh': LINK_PATH + '/collision', 'static_friction': 2.744, 'dynamic_friction': 1.7137}]
    monkeypatch.setattr(physical_dynamics_inclined_plane, '_friction', lambda obj: legacy)
    monkeypatch.setattr(calibration, '_usd_material_modules', lambda: (_ for _ in ()).throw(ImportError('No USD.')))
    install_sim(monkeypatch, error=RuntimeError('Static body unavailable.'))
    result = calibration._materials({'ramp': NS(links={'base': FakeStaticLink()})})['ramp']
    assert result['available'] is True and result['collision_materials'] == legacy
    assert result['backend_material_readback']['available'] is False
    assert len(result['backend_material_readback']['links']) == 1
    assert result['usd_binding_diagnostics']['available'] is False
    assert result['backend_material_readback']['effective_solver_coefficients_verified'] is False
    assert calibration.MATERIAL_BINDING_DIAGNOSTIC_POLICY['effective_solver_coefficients_verified'] is False
    assert calibration.MATERIAL_BINDING_DIAGNOSTIC_POLICY['solver_material_consumption_verified'] is False
    json.dumps(result, allow_nan=False)
