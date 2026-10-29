"""Check initialization-only poses for exact retained native furniture on CPU.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B \
  -m pytest -q -p no:cacheprovider tests/test_wia_native_furniture_poses.py
No native runtime, robot action, GPU workload or capture is invoked.
"""
from copy import deepcopy
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from wia.authoring.converters import active_layout as adapter
from wia.tasks import interaction_setup


class Furniture:
    def __init__(self, name='chair'):
        self.name, self.category, self.model = name, 'swivel_chair', 'ucawqa'
        self.position, self.orientation = [1., 2., .5], [0., 0., 0., 1.]
        self.scale, self.fixed_base, self.visual_only = [.8, .9, 1.], False, False
        self.joints, self.materials = {'seat': .12}, {'seat': 'original'}
        self.writes, self.still_calls = [], 0

    def get_position_orientation(self):
        return np.array(self.position), np.array(self.orientation)

    def set_position_orientation(self, position, orientation):
        self.position, self.orientation = list(position), list(orientation)
        self.writes.append({'position': list(position), 'orientation': list(orientation)})

    def keep_still(self):
        self.still_calls += 1


def spec(name='chair'):
    return {'name': name, 'category': 'swivel_chair', 'model': 'ucawqa',
            'position': [3., 4., .5], 'orientation': [0., 0., 1., 0.]}


def fixture_session(*objects):
    registry = {obj.name: obj for obj in objects}
    return NS(_category_initialization_active=True, robot=object(),
              env=NS(scene=NS(object_registry=lambda kind, name: registry.get(name))))


def test_exact_pose_original_and_settled_receipts_preserve_physical_properties():
    obj = Furniture()
    session = fixture_session(obj)
    layout = {'objects': [{'name': 'authored_target'}], 'native_object_poses': [spec()]}
    original_layout = deepcopy(layout)
    physical = deepcopy({key: getattr(obj, key) for key in ('scale', 'fixed_base', 'visual_only', 'joints', 'materials')})
    receipts = adapter._initialize_native_object_poses(session, layout)
    assert obj.writes == [{'position': [3., 4., .5], 'orientation': [0., 0., 1., 0.]}]
    assert obj.still_calls == 1
    assert receipts[0]['original_pose'] == {'position': [1., 2., .5], 'orientation': [0., 0., 0., 1.]}
    assert receipts[0]['requested_pose'] == obj.writes[0]
    assert receipts[0]['evidence_scope'] == 'initialization_only'
    assert 'final_pose' not in receipts[0]
    obj.position[2] = .49  # Final evidence must read the settled body, not echo its request.
    adapter._finalize_native_object_poses(session, receipts)
    assert receipts[0]['final_pose']['position'] == [3., 4., .49]
    assert layout == original_layout
    assert {key: getattr(obj, key) for key in physical} == physical


@pytest.mark.parametrize('invalid', ['missing', 'category', 'model', 'position_length', 'position_nan',
                                    'orientation_length', 'orientation_zero', 'orientation_inf',
                                    'duplicate', 'authored_overlap', 'extra_geometry_field'])
def test_invalid_list_is_rejected_before_any_native_pose_changes(invalid):
    first, second = Furniture('first'), Furniture('second')
    session = fixture_session(first, second)
    layout = {'objects': [], 'native_object_poses': [spec('first'), spec('second')]}
    row = layout['native_object_poses'][1]
    if invalid == 'missing':
        row['name'] = 'missing'
    elif invalid in ('category', 'model'):
        row[invalid] = 'different'
    elif invalid == 'position_length':
        row['position'] = [0., 0.]
    elif invalid == 'position_nan':
        row['position'][0] = float('nan')
    elif invalid == 'orientation_length':
        row['orientation'] = [0., 0., 1.]
    elif invalid == 'orientation_zero':
        row['orientation'] = [0.] * 4
    elif invalid == 'orientation_inf':
        row['orientation'][0] = float('inf')
    elif invalid == 'duplicate':
        row['name'] = 'first'
    elif invalid == 'authored_overlap':
        layout['objects'] = [{'name': 'second'}]
    else:
        row['scale'] = [2., 2., 2.]
    with pytest.raises(ValueError):
        adapter._initialize_native_object_poses(session, layout)
    assert first.writes == second.writes == []
    assert first.still_calls == second.still_calls == 0


def test_guard_prevents_pose_and_receipt_operations_outside_initialization():
    obj = Furniture()
    session = fixture_session(obj)
    session._category_initialization_active = False
    with pytest.raises(RuntimeError, match='initialization-only'):
        adapter._initialize_native_object_poses(session, {'native_object_poses': [spec()]})
    with pytest.raises(RuntimeError, match='initialization-only'):
        adapter._finalize_native_object_poses(session, [])
    assert not obj.writes
    session._category_initialization_active = True
    session.robot = obj
    with pytest.raises(ValueError, match='retained native furniture'):
        adapter._initialize_native_object_poses(session, {'native_object_poses': [spec()]})


def test_absent_pose_list_has_no_registry_reads_or_object_writes():
    def forbidden(*args, **kwargs):
        raise AssertionError('Default layout must not inspect or alter native furniture.')
    session = NS(_category_initialization_active=True, env=NS(scene=NS(object_registry=forbidden)))
    assert adapter._initialize_native_object_poses(session, {'objects': []}) == []
    adapter._finalize_native_object_poses(session, [])


@pytest.fixture
def initializer(monkeypatch):
    obj = Furniture()
    floor = NS(name='floor', prim_path='/World/floor', aabb=np.array([[-5., -5., -.1], [5., 5., 0.]]))
    session = fixture_session(obj, floor)
    calls = {'steps': 0}

    def forbidden(*args, **kwargs):
        raise AssertionError('Native furniture initialization must not invoke robot or environment actions.')

    session.robot = NS(apply_action=forbidden)
    session.env.step = forbidden
    og, states, torch = (ModuleType(name) for name in ('omnigibson', 'omnigibson.object_states', 'torch'))

    def step():
        calls['steps'] += 1
        if calls['steps'] == 90 and obj.writes:
            obj.position[2] = .48

    og.sim, states.Touching = NS(step=step), object()
    for module in (og, states, torch):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(interaction_setup, '_robot_contact_check',
                        lambda *args: {'floor_supported': True, 'collision_free': True})
    return session, obj, calls


def test_initialize_records_only_opted_in_native_poses_after_existing_settling(initializer):
    session, obj, calls = initializer
    parameters = {'revision': 'fixture', 'layout': {'objects': [], 'floor_name': 'floor'}}
    default = adapter.initialize_scene(session, parameters)
    assert 'native_object_poses' not in default
    assert calls['steps'] == 90 and not obj.writes
    calls['steps'] = 0
    parameters['layout']['native_object_poses'] = [spec()]
    result = adapter.initialize_scene(session, parameters)
    assert calls['steps'] == 90
    assert result['native_object_poses'][0]['requested_pose']['position'] == [3., 4., .5]
    assert result['native_object_poses'][0]['final_pose']['position'] == [3., 4., .48]
    assert {key: value for key, value in result.items() if key != 'native_object_poses'} == default
