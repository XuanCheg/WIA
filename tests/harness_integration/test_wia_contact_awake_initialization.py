"""Check the explicit initialization-only awake-contact protocol on CPU.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m
pytest -q -p no:cacheprovider tests/test_wia_contact_awake_initialization.py
No native simulator, physics step, GPU task or runtime callback is invoked.
"""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace as NS

import pytest

from wia.authoring.converters import wia_contact_awake_initialization as adapter


class RigidDynamicPrim:
    def __init__(self, path, events):
        self.prim_path, self.events = path, events
        self.is_meta_link = False
        self.collision_meshes = {'collision': NS(collision_enabled=True)}
        self.attributes = {'physics:rigidBodyEnabled': True, 'physics:kinematicEnabled': False}
        self._threshold, self._asleep = .005, True
        self.threshold_sets, self.wake_calls = [], 0
        self.keep_old_threshold, self.keep_asleep = False, False
        self.physical = {'pose': [1., 2., 3.], 'velocity': [.1, .2, .3], 'scale': [1., 1., 1.],
            'materials': {'friction': .5}, 'constraints': {'original': True}}

    def get_attribute(self, name):
        self.events.append(('attribute', self.prim_path, name))
        return self.attributes[name]

    @property
    def sleep_threshold(self):
        self.events.append(('threshold_read', self.prim_path))
        return self._threshold

    @sleep_threshold.setter
    def sleep_threshold(self, value):
        self.events.append(('threshold_set', self.prim_path, value))
        self.threshold_sets.append(value)
        if not self.keep_old_threshold:
            self._threshold = value

    @property
    def is_asleep(self):
        self.events.append(('asleep_read', self.prim_path))
        return self._asleep

    def wake(self):
        self.events.append(('wake', self.prim_path))
        self.wake_calls += 1
        if not self.keep_asleep:
            self._asleep = False

    def set_position_orientation(self, *args, **kwargs):
        raise AssertionError('The awake-contact protocol must not change poses.')

    def set_linear_velocity(self, *args, **kwargs):
        raise AssertionError('The awake-contact protocol must not change velocities.')


@pytest.fixture
def native(monkeypatch):
    events = []
    first, second, unrelated = [RigidDynamicPrim('/World/' + name + '/base_link', events)
                                for name in ('banana', 'container', 'unrelated')]
    objects = {name: NS(name=name, prim_path='/World/' + name, fixed_base=False, visual_only=False,
        links={'base_link': link}) for name, link in zip(('banana', 'container', 'unrelated'), (first, second, unrelated))}

    def forbidden(*args, **kwargs):
        raise AssertionError('The awake-contact protocol must not step physics or issue actions.')

    session = NS(_category_initialization_active=True, scene_name='Beechwood_0_int',
        env=NS(scene=NS(scene_model='Beechwood_0_int', object_registry=lambda key, name: objects.get(name)), step=forbidden),
        pipeline=NS(og=NS(sim=NS(step=forbidden))))
    monkeypatch.setitem(sys.modules, 'omnigibson.prims.rigid_dynamic_prim', NS(RigidDynamicPrim=RigidDynamicPrim))
    p = {'revision': adapter.REVISION, 'expected_scene': 'Beechwood_0_int',
         'object_names': ['banana', 'container'], 'sleep_threshold': 0.0}
    return NS(session=session, parameters=p, objects=objects, first=first, second=second,
              unrelated=unrelated, events=events)


def test_all_native_validation_precedes_once_only_sleep_changes_and_readbacks(native, capsys):
    physical_before = [deepcopy(link.physical) for link in (native.first, native.second, native.unrelated)]
    params_before = deepcopy(native.parameters)
    result = adapter.initialize_scene(native.session, native.parameters)
    assert result is native.session.wia_contact_awake_initialization
    assert result['status'] == 'applied' and result['readback_verified'] is True
    assert result['initialization_only'] is True
    assert result['runtime_wake_callbacks'] is result['native_validation_run'] is result['physical_success_verified'] is False
    assert result['physics_steps'] == 0
    for link, row in zip((native.first, native.second), result['links']):
        assert link.threshold_sets == [0.0] and link.wake_calls == 1
        assert row['original_sleep_threshold'] == .005 and row['original_is_asleep'] is True
        assert row['readback_sleep_threshold'] == 0.0 and row['readback_is_asleep'] is False
        assert row['sleep_threshold_set_calls'] == row['wake_calls'] == 1
    assert native.unrelated.threshold_sets == [] and native.unrelated.wake_calls == 0
    for link, before in zip((native.first, native.second, native.unrelated), physical_before):
        assert link.physical == before
    first_mutation = next(i for i, event in enumerate(native.events) if event[0] == 'threshold_set')
    assert ('threshold_read', native.second.prim_path) in native.events[:first_mutation]
    assert ('asleep_read', native.second.prim_path) in native.events[:first_mutation]
    assert native.parameters == params_before
    log = capsys.readouterr().out
    assert len(log.splitlines()) == 1 and json.loads(log)['wia_contact_awake_initialization'] == result
    assert not hasattr(native.session, 'on_step')


@pytest.mark.parametrize('failure', ['guard', 'revision', 'session_scene', 'env_scene', 'empty_names',
    'duplicate_names', 'non_list_names', 'nonzero_threshold', 'boolean_threshold'])
def test_guard_and_contract_fail_before_native_mutation(native, failure):
    if failure == 'guard': native.session._category_initialization_active = False
    elif failure == 'revision': native.parameters['revision'] = 'unknown_v2'
    elif failure == 'session_scene': native.session.scene_name = 'wrong_scene'
    elif failure == 'env_scene': native.session.env.scene.scene_model = 'wrong_scene'
    elif failure == 'empty_names': native.parameters['object_names'] = []
    elif failure == 'duplicate_names': native.parameters['object_names'] = ['banana', 'banana']
    elif failure == 'non_list_names': native.parameters['object_names'] = 'banana'
    elif failure == 'nonzero_threshold': native.parameters['sleep_threshold'] = .0001
    else: native.parameters['sleep_threshold'] = False
    with pytest.raises((ValueError, RuntimeError)):
        adapter.initialize_scene(native.session, native.parameters)
    assert native.events == []
    assert native.first.threshold_sets == native.second.threshold_sets == []
    assert native.first.wake_calls == native.second.wake_calls == 0


@pytest.mark.parametrize('failure', ['missing_object', 'fixed', 'visual_only', 'wrong_native_type', 'rigid_disabled',
    'kinematic_enabled', 'collision_disabled', 'only_meta', 'no_collision_meshes', 'missing_sleep_state',
    'nonfinite_original_threshold', 'duplicate_link_path'])
def test_later_object_validation_failure_cannot_partially_change_earlier_links(native, failure):
    obj, link = native.objects['container'], native.second
    if failure == 'missing_object': native.objects.pop('container')
    elif failure == 'fixed': obj.fixed_base = True
    elif failure == 'visual_only': obj.visual_only = True
    elif failure == 'wrong_native_type': obj.links['base_link'] = NS(is_meta_link=False, collision_meshes={'collision': object()})
    elif failure == 'rigid_disabled': link.attributes['physics:rigidBodyEnabled'] = False
    elif failure == 'kinematic_enabled': link.attributes['physics:kinematicEnabled'] = True
    elif failure == 'collision_disabled': link.collision_meshes['collision'].collision_enabled = False
    elif failure == 'only_meta': link.is_meta_link = True
    elif failure == 'no_collision_meshes': link.collision_meshes = {}
    elif failure == 'missing_sleep_state': link._asleep = None
    elif failure == 'nonfinite_original_threshold': link._threshold = float('nan')
    else: link.prim_path = native.first.prim_path
    with pytest.raises(ValueError):
        adapter.initialize_scene(native.session, native.parameters)
    assert native.first.threshold_sets == native.second.threshold_sets == []
    assert native.first.wake_calls == native.second.wake_calls == 0
    assert not hasattr(native.session, 'wia_contact_awake_initialization')


def test_meta_and_noncollision_links_are_not_changed(native):
    native.objects['banana'].links.update(meta=NS(is_meta_link=True),
        nonphysical=NS(is_meta_link=False, collision_meshes={}))
    result = adapter.initialize_scene(native.session, native.parameters)
    assert len(result['links']) == 2
    assert native.first.wake_calls == native.second.wake_calls == 1


@pytest.mark.parametrize('mismatch', ['threshold', 'asleep'])
def test_readback_mismatch_is_recorded_printed_and_raised_without_hidden_setters(native, capsys, mismatch):
    if mismatch == 'threshold': native.first.keep_old_threshold = True
    else: native.first.keep_asleep = True
    with pytest.raises(ValueError, match='readback mismatch'):
        adapter.initialize_scene(native.session, native.parameters)
    receipt = native.session.wia_contact_awake_initialization
    assert receipt['status'] == 'failed' and receipt['readback_verified'] is False
    assert receipt['links'][0]['readback_verified'] is False
    assert native.first.threshold_sets == [0.0] and native.first.wake_calls == 1
    assert native.second.threshold_sets == [] and native.second.wake_calls == 0
    assert json.loads(capsys.readouterr().out)['wia_contact_awake_initialization'] == receipt
    with pytest.raises(ValueError, match='already been attempted'):
        adapter.initialize_scene(native.session, native.parameters)
    assert native.first.threshold_sets == [0.0] and native.first.wake_calls == 1


def test_successful_initialization_cannot_install_a_second_wake_attempt(native):
    adapter.initialize_scene(native.session, native.parameters)
    with pytest.raises(ValueError, match='already been attempted'):
        adapter.initialize_scene(native.session, native.parameters)
    assert native.first.wake_calls == native.second.wake_calls == 1
