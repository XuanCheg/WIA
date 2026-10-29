"""Check private capture diagnostics using injected CPU-only native fakes.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -B -m pytest -q -p no:cacheprovider tests/test_wia_capture_diagnostics.py
No simulator, controller, capture, rendering, queue or file output is used.
"""
import builtins
from copy import deepcopy
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from wia.runtime.wia_capture_diagnostics import capture_initial_review_diagnostics, MIRROR_MODULE, SCHEMA


class Clock:
    def __init__(self, values=(12.5, 12.5)):
        self.values = iter(values)
        self.reads = 0

    @property
    def current_time(self):
        self.reads += 1
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value


class Evaluation:
    def __init__(self):
        self.type, self.module = 'category_operation', MIRROR_MODULE
        self.parameters = {'revision': 'wia_mirror_v1', 'private_marker': ['unchanged']}
        self.config = {'type': self.type, 'module': self.module, 'operation': 'mirror_select_deliver',
                       'binding_keys': ('candidate', 'mirror'), 'parameters': self.parameters}
        self.dumps = 0

    def model_dump(self):
        self.dumps += 1
        return self.config  # Deliberately shared, so the wrapper must deepcopy.


@pytest.fixture
def capture(monkeypatch):
    import wia.authoring.converters as package
    fake = ModuleType(MIRROR_MODULE)
    calls = []
    geometry = {'available': True, 'selected_candidate': 'candidate',
                'private_native_points_m': [[8., -1., .7]], 'optical_visibility_verified': False}

    def helper(session, config, bound):
        calls.append((session, deepcopy(config), bound.copy()))
        config['parameters']['private_marker'].append('helper-local')
        return geometry

    fake.capture_initial_geometry = helper
    monkeypatch.setitem(sys.modules, MIRROR_MODULE, fake)
    monkeypatch.setattr(package, 'wia_mirror', fake, raising=False)
    evaluation = Evaluation()
    bindings = {'candidate': NS(name='wia_candidate'), 'mirror': NS(name='wia_mirror_panel'),
                'unused': NS(name='not_requested')}
    task = NS(definition=NS(private=NS(evaluation=evaluation, bindings=bindings)),
              file_sha256='a' * 64, scene_name='hall_glass_ceiling', room_name='empty_room_0')
    verified = []
    task.verify_unchanged = lambda: verified.append(True)
    objects = {name: NS(name=name) for name in ('wia_candidate', 'wia_mirror_panel')}
    lookups = []

    def lookup(field, name):
        lookups.append((field, name))
        return objects.get(name)

    clock = Clock()
    session = NS(loaded_task=task, env=NS(scene=NS(object_registry=lookup)),
                 pipeline=NS(og=NS(sim=clock)))
    return NS(session=session, task=task, evaluation=evaluation, fake=fake, helper=helper,
              calls=calls, geometry=geometry, verified=verified, objects=objects,
              clock=clock, lookups=lookups)


@pytest.mark.parametrize('unsupported', ['missing_task', 'missing_private', 'missing_evaluation', 'wrong_type', 'wrong_module'])
def test_unsupported_tasks_return_none_without_imports_clock_binding_or_verification_reads(capture, monkeypatch, unsupported):
    if unsupported == 'missing_task':
        del capture.session.loaded_task
    elif unsupported == 'missing_private':
        capture.task.definition = NS()
    elif unsupported == 'missing_evaluation':
        del capture.task.definition.private.evaluation
    elif unsupported == 'wrong_type':
        capture.evaluation.type = 'stack_relations'
    else:
        capture.evaluation.module = 'wia.authoring.converters.wia_dimensional'
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith('wia.authoring.converters'):
            raise AssertionError('Unsupported tasks must not import native converter dependencies.')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    assert capture_initial_review_diagnostics(capture.session) is None
    assert not capture.calls and not capture.verified and not capture.lookups
    assert capture.clock.reads == capture.evaluation.dumps == 0


def test_matching_task_uses_exact_private_binding_names_and_detached_config_without_other_helpers(capture):
    original = deepcopy(capture.evaluation.config)
    result = capture_initial_review_diagnostics(capture.session)
    assert result['schema'] == SCHEMA and result['available'] is True and result['private_only'] is True
    assert result['task_sha256'] == capture.task.file_sha256
    assert result['module'] == MIRROR_MODULE and result['revision'] == 'wia_mirror_v1'
    assert result['native_qualification_run'] is result['physical_success_verified'] is False
    assert result['robot_actions'] == result['physics_steps'] == 0
    assert result['native_time_before_s'] == result['native_time_after_s'] == 12.5
    assert result['native_time_verified_unchanged'] is True
    assert capture.clock.reads == 2 and capture.verified == [True] and capture.evaluation.dumps == 1
    assert capture.lookups == [('name', 'wia_candidate'), ('name', 'wia_mirror_panel')]
    session, config, bound = capture.calls[0]
    assert session is capture.session
    assert config == {**original, 'scene': capture.task.scene_name, 'room': capture.task.room_name,
                      'bindings': {'candidate': 'wia_candidate', 'mirror': 'wia_mirror_panel'}}
    assert bound == {'candidate': capture.objects['wia_candidate'], 'mirror': capture.objects['wia_mirror_panel']}
    assert capture.evaluation.config == original
    assert result['geometry'] == capture.geometry and result['geometry'] is not capture.geometry
    capture.geometry['private_native_points_m'][0][0] = 100.
    assert result['geometry']['private_native_points_m'][0][0] == 8.
    assert not {'public_task', 'observation', 'prompt', 'task_success', 'accepted'} & result.keys()


@pytest.mark.parametrize('problem', ['missing_object', 'wrong_object_name', 'missing_binding'])
def test_missing_or_wrong_exact_native_binding_is_unavailable_without_calling_helper(capture, problem):
    if problem == 'missing_object':
        del capture.objects['wia_candidate']
    elif problem == 'wrong_object_name':
        capture.objects['wia_candidate'] = NS(name='some_other_candidate')
    else:
        del capture.task.definition.private.bindings['candidate']
    result = capture_initial_review_diagnostics(capture.session)
    assert result['available'] is False and result['error']
    assert result['geometry'] is None and not capture.calls
    assert result['robot_actions'] == result['physics_steps'] == 0


@pytest.mark.parametrize('times', [(12.5, 12.500000001), (float('nan'), 12.5),
                                  (12.5, float('inf')), (True, 12.5),
                                  (RuntimeError('Clock unavailable'), RuntimeError('Clock unavailable'))])
def test_changed_missing_or_nonfinite_native_clock_never_claims_zero_actions_or_steps(capture, times):
    clock = Clock(times)
    capture.session.pipeline.og.sim = clock
    result = capture_initial_review_diagnostics(capture.session)
    assert result['available'] is False and 'unchanged native simulation time' in result['error']
    assert result['native_time_verified_unchanged'] is False
    assert result['robot_actions'] is result['physics_steps'] is None
    assert result['native_qualification_run'] is result['physical_success_verified'] is False
    assert result['revision'] == 'wia_mirror_v1' and clock.reads == 2
    if times[0] == 12.5:
        assert result['geometry'] == capture.geometry and len(capture.calls) == 1
    else:
        assert result['geometry'] is None and not capture.calls


def test_changed_task_is_unavailable_and_never_imports_or_calls_geometry_helper(capture):
    def changed():
        raise RuntimeError('Task changed after loading.')
    capture.task.verify_unchanged = changed
    result = capture_initial_review_diagnostics(capture.session)
    assert result['available'] is False and 'Task changed after loading' in result['error']
    assert not capture.calls and capture.evaluation.dumps == 0
    assert result['robot_actions'] == result['physics_steps'] == 0


def test_helper_error_does_not_abort_capture_and_still_reads_final_native_clock(capture):
    def broken(*args):
        raise RuntimeError('Native optical measurement unavailable.')
    capture.fake.capture_initial_geometry = broken
    result = capture_initial_review_diagnostics(capture.session)
    assert result['available'] is False and 'Native optical measurement unavailable' in result['error']
    assert capture.clock.reads == 2 and result['geometry'] is None
    assert result['robot_actions'] == result['physics_steps'] == 0


def test_complete_geometry_with_rejected_optical_selection_remains_diagnostic_only(capture):
    capture.geometry['optical_selection'] = {'available': True, 'accepted': False,
                                            'selected_candidate': None, 'errors': ['Ambiguous optical geometry.']}
    result = capture_initial_review_diagnostics(capture.session)
    assert result['available'] is True
    assert result['geometry']['optical_selection']['accepted'] is False
    assert result['native_qualification_run'] is result['physical_success_verified'] is False
    assert 'accepted' not in result and 'task_success' not in result


@pytest.mark.parametrize('geometry', [{'available': False, 'error': 'Native camera missing.'}, None])
def test_unavailable_or_malformed_helper_output_never_upgrades_to_available(capture, geometry):
    capture.fake.capture_initial_geometry = lambda *args: geometry
    result = capture_initial_review_diagnostics(capture.session)
    assert result['available'] is False and result['error']
    assert result['geometry'] == geometry
    assert result['physical_success_verified'] is result['native_qualification_run'] is False
