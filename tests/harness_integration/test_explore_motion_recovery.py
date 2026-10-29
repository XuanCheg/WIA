"""Check recovery gating. Run .venv/bin/python -m pytest -q this file."""
import asyncio
from dataclasses import replace

import numpy as np
import pytest

from core.action_interface import action_result
from wia.runtime.agent.zero_shot import run_episode
from core.observations import AgentActionFeedback
from wia.runtime.sim.esi_actions import create_r1_executor
from wia.runtime.gateway_runtime import gateway_runtime_manifest, validate_gateway_runtime
from harness_integration.test_esi_arm import make_arm_controller
from harness_integration.test_esi_base import make_controller
from harness_integration.test_zero_shot_agent import Agent, Environment


def test_gateway_preflight_rejects_stale_schema_before_any_model_or_motion():
    from pathlib import Path
    root = __import__('wia.resources', fromlist=['ROOT']).ROOT
    manifest = gateway_runtime_manifest(root, backend_url='http://GPU-node:8615', port=8615)
    manifest['feedback_errors'].remove('base_motion_blocked')
    with pytest.raises(RuntimeError, match='stale feedback schemas'):
        validate_gateway_runtime(manifest, root, backend_url='http://GPU-node:8615')
    with pytest.raises(RuntimeError, match='missing'):
        validate_gateway_runtime(None, root, backend_url='http://GPU-node:8615')


def test_measured_stop_certificate_overrides_legacy_joint_hold_marker():
    result = {'action': 'BASE_FWD', 'reason': 'base_motion_blocked', 'completed': False,
              'stop_steps': 9, 'recovery_completed': None,
              'ik_diagnostics': {'joint_hold': True}, 'stop_stability_verified': True}
    assert AgentActionFeedback.from_result(result).motion_stopped is True
    result['stop_stability_verified'] = False
    assert AgentActionFeedback.from_result(result).motion_stopped is None


def test_gateway_preflight_accepts_live_matching_gateway_and_rejects_changed_source():
    from pathlib import Path
    root = __import__('wia.resources', fromlist=['ROOT']).ROOT
    manifest = gateway_runtime_manifest(root, backend_url='http://GPU-node:8615', port=8615)
    validate_gateway_runtime(manifest, root, backend_url='http://GPU-node:8615')
    manifest['source_hashes']['core/observations.py'] = 'old-source'
    with pytest.raises(RuntimeError, match='core/observations.py'):
        validate_gateway_runtime(manifest, root, backend_url='http://GPU-node:8615')


def test_fine_grasp_precision_does_not_reject_small_base_hold_transient():
    controller, robot, env = make_arm_controller(translation_m=.005, translation_tolerance_m=.00075)
    step = env.step
    injected = False

    def transient(actions):
        nonlocal injected
        step(actions)
        if not injected and np.any(actions[robot.name][robot.controller_action_idx['base']]):
            robot.ee[0] += .0009
            injected = True

    env.step = transient
    result = controller.execute_base_translation('forward', .05)
    assert result['completed']
    assert 'arm_hold_max_error_m' not in result
    assert controller.arm_config.translation_tolerance_m == .00075


def test_stalled_base_stops_early_settles_and_emits_one_macro_boundary():
    controller, _, env = make_controller(progress_timeout_sim_s=.2)
    env.response = 0
    executor = create_r1_executor(controller)
    executor.macro_execution = 'metric_target'
    boundaries = []
    executor.on_result = boundaries.append
    from core.action_compiler import MacroLimits
    result = executor.execute_macro({'action': 'BASE_FWD', 'distance_m': .25},
        resolution=executor.primitive_resolution, limits=MacroLimits())
    assert result['reason'] == 'base_motion_blocked' and not result['success']
    assert result['stop_stability_verified'] is True
    assert result['stop_stability']['consecutive_quiet_samples'] == 6
    assert result['sim_duration_s'] < controller.config.timeout_sim_s
    assert result['control_steps'] == len(env.actions)
    assert boundaries == [result]
    feedback = AgentActionFeedback.from_result(result)
    assert feedback.motion_stopped is True and feedback.observable_error == 'base_motion_blocked'


def test_persistent_joint_motion_cannot_certify_recovery():
    controller, robot, env = make_controller(progress_timeout_sim_s=.2)
    env.response = 0
    step = env.step

    def unstable(actions):
        step(actions)
        robot.joints[8] += .02 if len(env.actions) % 2 else -.02

    env.step = unstable
    executor = create_r1_executor(controller)
    result = executor.execute('BASE_FWD')
    assert result['reason'] == 'base_motion_blocked'
    assert result['stop_stability_verified'] is False
    assert result['stop_stability']['steps'] == controller.config.recovery_max_steps
    assert AgentActionFeedback.from_result(result).motion_stopped is None


class SettledFailure(Environment):
    def __init__(self, reason, verified):
        super().__init__()
        self.reason, self.verified = reason, verified

    async def execute(self, action):
        self.calls.append(action)
        result = action_result(action, completed=False, timeout=self.reason == 'timeout',
            requested={}, actual={}, message=self.reason, reason=self.reason,
            stop_steps=9, stop_stability_verified=self.verified)
        self.source.after_action(result)
        feedback = self.source.last_action_feedback.to_dict()
        return {'ok': False, 'outcome_known': True, 'motion_stopped': feedback['motion_stopped'],
                'reason': self.reason, 'results': [feedback]}


@pytest.mark.parametrize('reason', ['target_not_reached', 'base_motion_blocked', 'arm_step_limit', 'timeout',
                                   'ik_joint_jump', 'arm_joint_limit'])
@pytest.mark.parametrize('verified', [True, False])
def test_agent_continues_only_after_verified_stop(reason, verified, tmp_path):
    env = SettledFailure(reason, verified)
    agent = Agent('{"action":"BASE_FWD"}', '{"action":"DONE"}')
    summary = asyncio.run(run_episode(env, agent, tmp_path, max_steps=2))
    assert summary['termination_reason'] == ('model_done' if verified else 'execution_error')
    assert env.calls == ['BASE_FWD']
    assert len(summary['recovered_execution_errors']) == int(verified)
    assert env.source.agent_observation().action_step == 1
    if verified:
        assert agent.inputs[1].action_context['recovery']['reason'] == reason
        assert agent.inputs[1].observation.action_step == 1


@pytest.mark.parametrize('reason', ['timeout', 'arm_step_limit', 'ik_joint_jump', 'arm_joint_limit'])
def test_mcp_recovery_keeps_unknown_transport_outcomes_distinct(reason):
    from wia.runtime.web.mcp_server import RobotBridge
    bridge = RobotBridge('http://localhost:8615')
    raw = action_result('BASE_FWD', completed=False, timeout=reason == 'timeout', requested={}, actual={},
        message=reason, reason=reason, stop_steps=9, stop_stability_verified=True)
    feedback = AgentActionFeedback.from_result(raw).to_dict()
    reply = bridge._execution_result(['BASE_FWD'], 200, {'ok': False, 'results': [feedback]}).structured_content
    assert reply['can_continue'] and reply['motion_stopped'] is True
    assert reply['reason'] == reason and reply['automatic_replay'] is False
    raw['stop_stability_verified'] = False
    feedback = AgentActionFeedback.from_result(raw).to_dict()
    reply = bridge._execution_result(['BASE_FWD'], 200, {'ok': False, 'results': [feedback]}).structured_content
    assert 'can_continue' not in reply and reply['motion_stopped'] is None


@pytest.mark.parametrize('persistent', [False, True])
def test_excessive_step_still_halts_and_only_a_quiet_stop_allows_retreat(persistent):
    controller, robot, env = make_arm_controller()
    step = env.step

    def disturbed(actions):
        step(actions)
        if len(env.actions) == 1 or persistent:
            robot.ee[1] += .02 if len(env.actions) % 2 else -.02
            robot.joints[robot.arm_control_idx['left']] = np.r_[robot.ee, robot.ee_rot.as_rotvec()]

    env.step = disturbed
    executor = create_r1_executor(controller)
    executor.macro_execution = 'metric_target'
    boundaries = []
    executor.on_result = boundaries.append
    from core.action_compiler import MacroLimits
    result = executor.execute_macro({'action': 'MV_FWD', 'distance_m': .02},
        resolution=executor.primitive_resolution, limits=MacroLimits())
    assert result['reason'] == 'arm_step_limit' and not result['success']
    assert result['max_step_motion_m'] > controller.arm_config.max_step_translation_m
    assert result['stop_stability_verified'] is (not persistent)
    feedback = AgentActionFeedback.from_result(result)
    assert feedback.motion_stopped is (None if persistent else True)
    assert boundaries == [result]
    if not persistent:
        retreat = executor.execute('MV_BACK')
        assert retreat['success']
        assert len(boundaries) == 2


def test_path_stop_steps_and_macro_marker_cannot_replace_measurement():
    result = {'action': 'MV_DOWN', 'reason': 'arm_step_limit', 'completed': False,
              'stop_steps': 3, 'parameterized': True, 'macro_motion_stopped': True}
    assert AgentActionFeedback.from_result(result).motion_stopped is None


def test_gripper_timeout_cannot_bypass_the_measured_stop_requirement():
    from wia.runtime.web.mcp_server import RobotBridge
    raw = action_result('GRASP', completed=False, timeout=True, requested={}, actual={},
        message='gripper_timeout', reason='gripper_timeout', stop_steps=3)
    feedback = AgentActionFeedback.from_result(raw).to_dict()
    assert feedback['execution_status'] == 'timeout' and feedback['motion_stopped'] is None
    reply = RobotBridge('http://localhost:8615')._execution_result(
        ['GRASP'], 200, {'ok': False, 'results': [feedback]}).structured_content
    assert reply['reason'] == 'timeout' and 'can_continue' not in reply


def test_target_and_stop_tolerances_are_positive():
    controller, _, _ = make_arm_controller()
    with pytest.raises(ValueError):
        replace(controller.arm_config, initial_trunk_joint_tolerance_rad=0.)
    with pytest.raises(ValueError):
        replace(controller.config, recovery_max_steps=2)
