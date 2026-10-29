"""Check source-preserving table proposals and the installed lazy-state protocol.

Run: python -m pytest -q tests/test_annotation_shared_support.py
These CPU checks do not certify native collision contacts or successful grasp.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from wia.tasks.interaction_supports import NAMES, preload_supports, support_candidates, table_dimensions


def test_preloaded_fallback_is_separate_from_every_source_asset():
    config = {'objects': [{'name': 'original', 'scale': [1., 2., 3.]}]}
    before = deepcopy(config['objects'][0])
    preload_supports(config)
    assert config['objects'][0] == before
    assert {obj['name'] for obj in config['objects'][1:]} == set(NAMES)
    assert all(obj['fixed_base'] and not obj['visual_only'] and not obj['visible']
               for obj in config['objects'][1:])
    with pytest.raises(ValueError, match='replace'):
        preload_supports(config)


def test_modest_table_preserves_large_source_groups_as_diagnostics():
    bounds = {'large': (np.array([0., 0., 0.]), np.array([3., 2., .3]))}
    before = deepcopy(bounds)
    assert table_dimensions(bounds, {'relocation_policy': 'vertical_only'}) == []
    for first, second in zip(bounds['large'], before['large']):
        assert np.array_equal(first, second)


def test_vertical_fallback_preserves_exact_source_xy_and_relative_geometry(monkeypatch):
    monkeypatch.setattr('wia.tasks.interaction_supports.footprint_clear', lambda *args: True)
    bounds = {'a': (np.array([2., 3., 0.]), np.array([2.1, 3.1, .1])),
              'b': (np.array([2.2, 3., 0.]), np.array([2.3, 3.1, .1]))}
    group = {name: SimpleNamespace(name=name) for name in bounds}
    policy = {'relocation_policy': 'vertical_only', 'source_region': 'room_0'}
    session = SimpleNamespace(env=SimpleNamespace(scene=object(), robots=[SimpleNamespace(name='robot')]))
    result = list(support_candidates(session, group, bounds, policy, np.array([[0., 0.]]),
                                     0., np.array([2., 3., .1]), {}))
    assert result
    for _, _, translations, layout in result:
        assert np.allclose(translations['a'], translations['b'])
        for name, (low, high) in bounds.items():
            assert np.allclose(translations[name][:2], 0.)
            moved_low, moved_high = layout['bounds'][name]
            assert np.allclose(moved_high - moved_low, high - low)


def test_native_lazy_controller_dictionary_is_never_iterated_as_a_sequence():
    from wia.runtime.sim.esi_state import read_controller_state
    queried = []
    class CachedControl:
        def get_fcn_names(self):
            return ('eef_left_pos_relative', 'eef_left_quat_relative')
        def __getitem__(self, name):
            queried.append(name)
            assert isinstance(name, str), 'Integer membership queries are invalid in the installed SDK.'
            return [0., 0., .5] if name.endswith('pos_relative') else [0., 0., 0., 1.]
    robot = SimpleNamespace(get_position_orientation=lambda: (np.zeros(3), np.array([0., 0., 0., 1.])),
        get_control_dict=CachedControl, get_eef_pose=lambda arm: (np.zeros(3), np.array([0., 0., 0., 1.])),
        controllers={})
    state = read_controller_state(SimpleNamespace(robot=robot, closed=False))
    assert state.ee_pose['base_position'] == [0., 0., .5]
    assert queried == ['eef_left_pos_relative', 'eef_left_quat_relative']
