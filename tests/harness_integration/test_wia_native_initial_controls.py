"""Check declared public initialization controls with CPU-only native stubs.

Run: python -B -m pytest -q -p no:cacheprovider
tests/test_wia_native_initial_controls.py
"""
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from core.action_compiler import MacroLimits, PrimitiveResolution
from core.action_interface import BackendCapabilities
from wia.runtime.commands.check_esi_core_native import NativeInitialControlsError, run_initial_controls


@pytest.fixture
def native(tmp_path):
    readings, read_arms = {'left': -1, 'right': -1}, []
    def is_grasping(arm):
        read_arms.append(arm)
        return readings[arm]
    robot = NS(arm_names=('left', 'right'),
               _ag_obj_in_hand={'left': None, 'right': None},
               _ag_obj_constraints={'left': None, 'right': None},
               _ag_release_counter={'left': None, 'right': None}, is_grasping=is_grasping)
    controller = NS(robot=robot)
    resolution = PrimitiveResolution(base_translation_m=.025, base_rotation_deg=5.,
                                    arm_translation_m=.01, arm_rotation_deg=5., trunk_translation_m=.01)
    limits = MacroLimits(max_translation_m=.6, max_rotation_deg=60., max_primitives=24, max_duration_s=45.)
    capabilities = BackendCapabilities(base_translation=True, base_rotation=True, ee_translation=True,
                                       gripper=True, ee_rotation=True, trunk_height=True)
    fixture = NS(robot=robot, controller=controller, readings=readings, read_arms=read_arms,
                 calls=[], outcomes=[], on_execute=None, boundary=0, resolution=resolution, limits=limits,
                 input_file=tmp_path / 'declared_controls.json', journal=tmp_path / 'initial_controls.json')
    def execute(action, *, resolution, limits):
        fixture.calls.append((action.to_dict(), resolution, limits))
        fixture.boundary += 1
        if fixture.on_execute is not None:
            fixture.on_execute()
        outcome = fixture.outcomes.pop(0) if fixture.outcomes else {'success': True, 'completed': True, 'timeout': False}
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    fixture.executor = NS(available_actions=capabilities.available_actions,
                          primitive_resolution=resolution, execute_macro=execute)
    fixture.state = lambda: {'control_boundary': fixture.boundary, 'native_floor_room': {'expected_room': 'utility_room_0'}}
    return fixture


def run(native, commands):
    raw = (json.dumps(commands, indent=2) + '\n').encode()
    native.input_file.write_bytes(raw)
    return run_initial_controls(native.input_file, native.executor, native.controller,
        limits=native.limits, state_reader=native.state, output_file=native.journal, primitive_profile='fine')


def test_valid_declared_sequence_records_exact_input_native_state_and_public_settings(native):
    commands = [{'action': 'BASE_RIGHT', 'distance_m': .15},
                {'action': 'BASE_TURN_LEFT', 'angle_deg': 15.},
                {'action': 'MV_UP', 'distance_m': .02},
                {'action': 'TRUNK_UP', 'distance_m': .01},
                {'action': 'ROTATE_CW', 'angle_deg': 10.}, {'action': 'RELEASE'}]
    result = run(native, commands)
    assert result['complete'] is True
    assert result['commands'] == commands
    assert result['input_sha256'] == hashlib.sha256(native.input_file.read_bytes()).hexdigest()
    assert result['input_file'] == str(native.input_file.resolve())
    assert result['primitive_profile'] == 'fine'
    assert result['primitive_resolution'] == native.resolution.to_dict()
    assert result['macro_limits'] == native.limits.to_dict()
    assert result['credited_as_episode_action'] is False
    assert result['completed_commands'] == len(commands)
    assert [call[0] for call in native.calls] == commands
    assert all(resolution is native.resolution and limits is native.limits for _, resolution, limits in native.calls)
    for index, record in enumerate(result['records']):
        assert record['before']['state']['control_boundary'] == index
        assert record['after']['state']['control_boundary'] == index + 1
        assert record['before']['hands']['empty'] is record['after']['hands']['empty'] is True
        assert set(record['before']['hands']['arms']) == {'left', 'right'}
        assert record['receipt']['success'] is True
        assert record['credited_as_episode_action'] is False
    assert native.read_arms.count('left') == native.read_arms.count('right') == 2 * len(commands)
    assert json.loads(native.journal.read_text()) == result


def test_failed_public_command_short_circuits_before_next_command(native):
    native.outcomes = [{'success': False, 'completed': False, 'timeout': False}]
    commands = [{'action': 'BASE_RIGHT', 'distance_m': .15}, {'action': 'RELEASE'}]
    with pytest.raises(NativeInitialControlsError, match='initialization stopped') as error:
        run(native, commands)
    result = error.value.record
    assert len(native.calls) == 1
    assert result['complete'] is False
    assert result['completed_commands'] == 0
    assert result['records'][0]['after']['hands']['empty'] is True
    assert result['records'][0]['receipt']['success'] is False
    assert json.loads(native.journal.read_text()) == result


@pytest.mark.parametrize('prohibited', [
    {'action': 'GRASP'}, {'action': 'DONE'}, {'action': 'STOP'},
    {'temporal_event': 'phase_2'}, {'action': 'object_pose', 'position': [0, 0, 0]},
    {'action': 'BASE_RIGHT', 'distance_m': .15, 'temporal_event': 'phase_2'},
])
def test_whole_list_is_validated_before_any_prohibited_control_can_execute(native, prohibited):
    commands = [{'action': 'BASE_RIGHT', 'distance_m': .15}, prohibited]
    with pytest.raises(NativeInitialControlsError) as error:
        run(native, commands)
    assert native.calls == []
    assert error.value.record['records'] == []
    assert error.value.record['input_sha256'] == hashlib.sha256(native.input_file.read_bytes()).hexdigest()
    assert json.loads(native.journal.read_text())['complete'] is False


@pytest.mark.parametrize('arm,condition', [('left', 'holding'), ('right', 'holding'),
                                         ('right', 'constraint'), ('right', 'pending'), ('right', 'unknown')])
def test_initial_release_never_drops_a_held_object_or_ignores_other_arm_state(native, arm, condition):
    if condition == 'holding':
        native.robot._ag_obj_in_hand[arm] = NS(name='held_object')
        native.robot._ag_obj_constraints[arm] = object()
        native.readings[arm] = 1
    elif condition == 'constraint':
        native.robot._ag_obj_constraints[arm] = object()
    elif condition == 'pending':
        native.robot._ag_release_counter[arm] = 0
    else:
        native.readings[arm] = 0
    with pytest.raises(NativeInitialControlsError, match='no held object may be dropped') as error:
        run(native, [{'action': 'RELEASE'}])
    assert native.calls == []
    assert error.value.record['records'][0]['receipt'] is None
    assert error.value.record['records'][0]['before']['hands']['empty'] is not True


def test_command_that_leaves_other_hand_nonempty_stops_sequence(native):
    def attach_other_arm():
        native.robot._ag_obj_in_hand['right'] = NS(name='unexpected_object')
        native.robot._ag_obj_constraints['right'] = object()
        native.readings['right'] = 1
    native.on_execute = attach_other_arm
    with pytest.raises(NativeInitialControlsError, match='after command') as error:
        run(native, [{'action': 'BASE_RIGHT', 'distance_m': .15}, {'action': 'RELEASE'}])
    assert len(native.calls) == 1
    after = error.value.record['records'][0]['after']['hands']
    assert after['empty'] is False
    assert after['arms']['right']['object_name'] == 'unexpected_object'
    assert error.value.record['complete'] is False


def test_backend_exception_retains_after_measurement_and_aborts(native):
    native.outcomes = [RuntimeError('Guarded motion failed.')]
    with pytest.raises(NativeInitialControlsError, match='Guarded motion failed') as error:
        run(native, [{'action': 'BASE_RIGHT', 'distance_m': .15}, {'action': 'RELEASE'}])
    assert len(native.calls) == 1
    record = error.value.record['records'][0]
    assert record['receipt'] is None
    assert record['execution_error'] == 'RuntimeError: Guarded motion failed.'
    assert record['after']['hands']['empty'] is True
    assert json.loads(native.journal.read_text()) == error.value.record


def test_unavailable_executor_capability_is_not_replaced_by_direct_controller_control(native):
    native.executor.available_actions = ('BASE_RIGHT',)
    with pytest.raises(NativeInitialControlsError):
        run(native, [{'action': 'RELEASE'}])
    assert native.calls == []


def test_nonfinite_json_is_rejected_with_a_persisted_exact_input_hash(native):
    native.input_file.write_bytes(b'[{"action": "BASE_RIGHT", "distance_m": NaN}]\n')
    with pytest.raises(NativeInitialControlsError) as error:
        run_initial_controls(native.input_file, native.executor, native.controller,
            limits=native.limits, state_reader=native.state, output_file=native.journal, primitive_profile='fine')
    assert native.calls == []
    assert error.value.record['input_sha256'] == hashlib.sha256(native.input_file.read_bytes()).hexdigest()
    assert json.loads(native.journal.read_text()) == error.value.record
