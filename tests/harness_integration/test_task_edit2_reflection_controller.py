"""CPU checks for retaining aligned torso targets across final controller setup.

Run in the behavior environment: ``python -B -m pytest -q
tests/test_task_edit2_reflection_controller.py tests/test_esi_arm.py
tests/test_action_interface.py``. Native-shaped servo fixtures exercise the real
R1ArmController constructor and executor factory, without native/GPU startup.
They do not certify alignment, floor support, images or physical feasibility.
"""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest

import wia.runtime.sim.esi_arm as esi_arm
import wia.runtime.sim.esi_reflection as esi_reflection
from wia.runtime.sim.esi_actions import create_r1_executor
from wia.runtime.sim.esi_base import BaseControlConfig
from harness_integration.test_esi_arm import ArmRobot, ArmEnv
from harness_integration.test_esi_base import JointController


class FourTrunkArmRobot(ArmRobot):
    """Existing servo fixture with the installed R1's four trunk coordinates."""
    action_dim = 17

    def __init__(self):
        super().__init__(yaw=0)
        self.controllers['trunk'] = JointController([3, 4, 15, 16])
        self.controller_action_idx['trunk'] = np.array([2, 6, 15, 16])
        self.control_limits = {'position': (np.full(17, -10.), np.full(17, 10.)),
            'velocity': (np.full(17, -5.), np.full(17, 5.))}
        self._controller_config['trunk'] = {'name': 'JointController'}


def session_for(case='aligned', *, traced=False):
    robot = FourTrunkArmRobot()
    # The earlier native alignment's measured pose, deliberately distinct from
    # the generic startup preset. No source/head/physics validity is claimed.
    dofs = robot.controllers['trunk'].dof_idx
    robot.joints[dofs] = [.12, -.18, -.35, .22]
    events = []
    source = NS(controller_state_provider=None,
        after_action=lambda result: events.append(('after_action', result['action'])))
    def capture_initial():
        if getattr(session, 'reflection_sensor_profile', {}).get('version') == esi_reflection.ACTION_PROFILE_VERSION:
            assert ('finalize_initial',) in events
        events.append(('initial_observation',))
    source.capture_initial = capture_initial

    class TracedSession(NS):
        @property
        def observation_source(self):
            events.append(('observation_source_access',))
            return source

    session_type = TracedSession if traced else NS
    session = session_type(robot=robot, env=ArmEnv(robot), native_reset_id='native-reset',
        reflection_sensor_profile={'version': esi_reflection.ACTION_PROFILE_VERSION,
            'initialization_revision': esi_reflection.INITIALIZATION_REVISION,
            'source_identity': 'source-fixture', 'robot_uniform_scale': None})
    session.reflection_alignment_initialization = {'completed': True, 'reset_id': session.native_reset_id,
        'profile_sha256': esi_reflection.reflection_profile_sha256(session.reflection_sensor_profile)}
    if case == 'incomplete': session.reflection_alignment_initialization['completed'] = False
    elif case == 'stale_reset': session.reflection_alignment_initialization['reset_id'] = 'old-reset'
    elif case == 'mutated_profile': session.reflection_sensor_profile['robot_uniform_scale'] = .6
    elif case == 'legacy': session.reflection_sensor_profile['version'] = 'reflection_sensor_profile_v2'
    elif case == 'none_reset':
        session.native_reset_id = None
        session.reflection_alignment_initialization['reset_id'] = None
    elif case == 'ordinary': del session.reflection_sensor_profile
    return session, events, source


def construct_controller(session, monkeypatch):
    startup_calls = []
    def generic_startup(subject, base_config, arm_config):
        # Isolate the existing generic startup physics loop, while exercising
        # the real caller branch and its resulting targets. The aligned branch
        # must not call this at all.
        assert subject is session
        target = np.asarray(arm_config.initial_trunk_joint_positions_rad)
        startup_calls.append(target.copy())
        subject.robot.joints[subject.robot.controllers['trunk'].dof_idx] = target
        return {'operation': 'generic_startup_protocol_fixture', 'target_rad': target.tolist()}
    monkeypatch.setattr(esi_arm, '_initialize_trunk', generic_startup)
    controller = esi_arm.R1ArmController(session, BaseControlConfig.load(),
        arm_config=esi_arm.ArmControlConfig.load())
    return controller, startup_calls


@pytest.mark.parametrize('case', ['aligned', 'incomplete', 'stale_reset', 'mutated_profile',
    'legacy', 'none_reset', 'ordinary'])
def test_real_arm_constructor_retains_exact_alignment_only_and_preserves_default_startup(case, monkeypatch):
    session, _, _ = session_for(case)
    robot = session.robot
    before = robot.get_joint_positions()
    trunk_dofs = robot.controllers['trunk'].dof_idx
    profile_before = deepcopy(getattr(session, 'reflection_sensor_profile', None))
    receipt_before = deepcopy(session.reflection_alignment_initialization)
    controller, startup_calls = construct_controller(session, monkeypatch)
    if case == 'aligned':
        assert startup_calls == []
        np.testing.assert_array_equal(robot.joints[trunk_dofs], before[trunk_dofs])
        retained = controller.initial_trunk_state
        assert retained['operation'] == 'retain_explicit_reflection_alignment'
        assert retained['joint_positions_rad'] == before[trunk_dofs].tolist()
        assert retained['profile_sha256'] == receipt_before['profile_sha256']
        assert retained['generic_trunk_preset_applied'] is False
        assert retained['physical_qualification_complete'] is False
    else:
        assert len(startup_calls) == 1
        np.testing.assert_array_equal(robot.joints[trunk_dofs], controller.arm_config.initial_trunk_joint_positions_rad)
        assert not np.array_equal(robot.joints[trunk_dofs], before[trunk_dofs])
        assert controller.initial_trunk_state['operation'] == 'generic_startup_protocol_fixture'
    # The actual base-controller constructor must hold the selected native
    # trunk coordinates, rather than later replacing them with the other pose.
    np.testing.assert_allclose(controller.hold_action[controller.indices['trunk']], robot.joints[trunk_dofs])
    np.testing.assert_array_equal(controller.inactive_targets['trunk'], robot.joints[trunk_dofs])
    np.testing.assert_array_equal(np.delete(robot.joints, trunk_dofs), np.delete(before, trunk_dofs))
    assert len(robot.reloads) == 1  # Real working-arm controller replacement path.
    assert getattr(session, 'reflection_sensor_profile', None) == profile_before
    assert session.reflection_alignment_initialization == receipt_before
    assert session.env.actions == []  # No extra servo/control loop on retained branch.


@pytest.mark.parametrize('revised', [True, False])
def test_factory_finalizes_actual_final_controller_before_lazy_observation_and_public_control(revised, monkeypatch):
    session, events, source = session_for('aligned' if revised else 'legacy', traced=True)
    controller, _ = construct_controller(session, monkeypatch)
    # Native runner attaches its final trunk controller before the common
    # factory boundary. Only its real primitive resolution is needed here.
    controller.trunk_controller = NS(config=NS(translation_m=.02))
    events.append(('final_controller_setup',))
    finalized = []
    def finalize(subject, *, controller: object, finalize_initial):
        assert revised, 'Legacy factory must not add revised initial finalization.'
        assert subject is session and controller is actual_controller
        assert controller.trunk_controller is not None
        assert finalize_initial is True
        assert events == [('final_controller_setup',)]
        assert getattr(controller, '_public_action_sequence', 0) == 0
        finalized.append(controller)
        events.append(('finalize_initial',))
    actual_controller = controller
    monkeypatch.setattr(esi_reflection, 'measure_reflection', finalize)
    original_execute = controller.execute
    def execute(action, **kwargs):
        assert bool(finalized) is revised
        assert controller._public_action_sequence == 1
        events.append(('public_control', action))
        return original_execute(action, **kwargs)
    monkeypatch.setattr(controller, 'execute', execute)
    executor = create_r1_executor(controller)
    assert executor.controller is controller
    assert source.controller_state_provider.__self__ is controller
    expected = [('final_controller_setup',)] + ([('finalize_initial',)] if revised else []) + [('observation_source_access',)]
    assert events == expected
    # Simulate the observation owner's first callback only after factory return;
    # no camera/feature claims are made by this chronology fixture.
    source.capture_initial()
    result = executor.execute('BASE_FWD')
    assert result['completed'] and result['success']
    assert events == expected + [('initial_observation',), ('public_control', 'BASE_FWD'), ('after_action', 'BASE_FWD')]
    assert len(finalized) == (1 if revised else 0)
    assert controller._public_action_context['active'] is False
    assert controller._public_action_sequence == 1


def test_failed_final_measurement_stops_factory_before_observation_source_or_public_controls(monkeypatch):
    session, events, _ = session_for(traced=True)
    controller, _ = construct_controller(session, monkeypatch)
    def unavailable(subject, *, controller: object, finalize_initial):
        assert subject is session and controller is actual_controller and finalize_initial
        events.append(('failed_final_measurement',))
        raise RuntimeError('native complete mounted rig unavailable')
    actual_controller = controller
    monkeypatch.setattr(esi_reflection, 'measure_reflection', unavailable)
    with pytest.raises(RuntimeError, match='complete mounted rig unavailable'):
        create_r1_executor(controller)
    assert events == [('failed_final_measurement',)]
    assert getattr(controller, '_public_action_sequence', 0) == 0
    assert session.env.actions == []
