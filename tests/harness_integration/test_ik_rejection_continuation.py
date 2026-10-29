"""Exercise rejection through public decisions. Run python -m pytest -q this file."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from core.action_compiler import MacroLimits
from wia.runtime.agent.zero_shot import run_episode
from core.feedback_semantics import IK_REJECTION_ERRORS, SETTLED_MOTION_ERRORS
from core.observations import AgentActionFeedback
from wia.runtime.sim.esi_actions import create_r1_executor
from wia.runtime.web.mcp_server import RobotBridge
from harness_integration.test_esi_arm import make_arm_controller
from harness_integration.test_zero_shot_agent import Agent, Environment


class GuardedEnvironment(Environment):
    """Connect the real owner-thread guard to a deterministic CPU motor plant."""

    action_mode = 'parameterized'

    def __init__(self, reason, *, rejection_count=1, persistent=False, unknown=False):
        super().__init__()
        self.controller, self.robot, self.plant = make_arm_controller(translation_m=.005, max_ik_joint_step_rad=.08)
        self.capabilities = self.controller.capabilities
        self.controller.session.observation_source = self.source
        self.executor = create_r1_executor(self.controller)
        self.executor.macro_execution = 'metric_target'
        self.primitive_resolution = self.executor.primitive_resolution
        self.macro_limits = MacroLimits(max_arm_primitives=16)
        self.rejection_count, self.unknown = rejection_count, unknown
        self.raw_results, self.deployments, self.rejection_poses = [], [], []
        self.bridge = RobotBridge('http://cpu-fixture.invalid')
        self.bridge.request = self.request
        self.action_context = {'parameterized_actions': {
            'resolution': self.primitive_resolution.to_dict(), 'limits': self.macro_limits.to_dict()}}
        arm_dofs = self.robot.arm_control_idx['left']
        self.robot.joints[arm_dofs] = np.r_[self.robot.ee, self.robot.ee_rot.as_rotvec()]
        self.start = self.robot.joints[arm_dofs].copy()
        self.command_start = self.start.copy()
        def native(goal, state):
            candidate = goal['target'].copy()
            current = state['joint_position'][arm_dofs]
            if (len(self.calls) <= self.rejection_count
                    and current[0] - self.command_start[0] >= .006):
                self.rejection_poses.append(current.copy())
                candidate[0] = 11. if reason == 'arm_joint_limit' else current[0] + 1.
            return candidate

        self.controller.ik_guard.original_compute = native
        step = self.plant.step

        def guarded_step(actions):
            action = actions[self.robot.name].copy()
            state = self.robot.get_control_dict()
            jacobian = np.zeros((6, self.robot.action_dim))
            jacobian[:, arm_dofs] = np.eye(6)
            state['eef_left_jacobian_relative'] = jacobian
            before = self.robot.joints[arm_dofs].copy()
            deployed = self.robot.controllers['arm_left'].compute_control(
                {'target': action[self.controller.indices['arm_left']]}, state)
            self.deployments.append((before, deployed.copy()))
            action[self.controller.indices['arm_left']] = deployed
            step({self.robot.name: action})
            if persistent and self.controller.ik_guard.fault:
                self.robot.joints[12] += .02 if len(self.plant.actions) % 2 else -.02

        self.plant.step = guarded_step

    async def request(self, method, path, **kwargs):
        if method == 'GET':
            assert path == '/api/agent/state'
            payload = {'capabilities': self.capabilities.to_dict(),
                       'parameterized_actions': self.action_context['parameterized_actions']}
            status = 200
        else:
            assert method == 'POST' and path == '/api/agent/command'
            command = kwargs['body']['command']
            self.calls.append(command)
            self.command_start = self.robot.joints[self.robot.arm_control_idx['left']].copy()
            result = self.executor.execute_macro(command, resolution=self.primitive_resolution,
                                                 limits=self.macro_limits)
            self.raw_results.append(result)
            if self.unknown:
                raise requests.ConnectionError('Execution response lost after submission.')
            feedback = self.source.last_action_feedback.to_dict()
            payload = {'ok': result['success'], 'results': [feedback], 'macro': {
                'command': result['command'], 'compilation': result['compilation'],
                'execution': dict(result['execution'], status=feedback['execution_status']),
                'duration_s': result['duration_s'],
                'physics_steps': result['control_steps'], 'actual': feedback['estimated_motion']}}
            status = 200 if result['success'] else 409
        return SimpleNamespace(status_code=status, json=lambda: payload, raise_for_status=lambda: None)

    async def execute(self, command):
        return (await self.bridge.execute_parameterized(
            command.to_dict(), control_token='cpu-fixture')).structured_content


@pytest.mark.parametrize('reason', sorted(IK_REJECTION_ERRORS))
@pytest.mark.parametrize('rejection_count', [1, 5])
def test_native_rejection_continues_from_measured_pose_without_replaying_macro(reason, rejection_count, tmp_path):
    environment = GuardedEnvironment(reason, rejection_count=rejection_count)
    agent = Agent(*(['{"action":"MV_FWD","distance_m":0.06}'] * rejection_count + [
        '{"action":"MV_FWD","distance_m":0.005}', '{"action":"DONE"}']))
    summary = asyncio.run(run_episode(environment, agent, tmp_path, max_steps=rejection_count + 2,
                                       action_mode='parameterized'))
    assert summary['termination_reason'] == 'model_done'
    assert summary['agent_decisions'] == summary['model_calls'] == rejection_count + 2
    assert summary['macro_actions'] == summary['metric_target_executions'] == rejection_count + 1
    assert summary['actual_fixed_primitive_executions'] == 0
    assert summary['equivalent_fixed_primitives'] == rejection_count * 12 + 1
    assert summary['controller_env_steps'] == len(environment.plant.actions)
    assert len(summary['recovered_execution_errors']) == rejection_count
    assert len(environment.calls) == len(environment.raw_results) == rejection_count + 1
    assert environment.rig.cycle == environment.source.snapshot_id == rejection_count + 2
    assert environment.source.action_step == rejection_count + 1
    assert len(environment.rejection_poses) == rejection_count
    for index, result in enumerate(environment.raw_results[:-1]):
        assert result['reason'] == reason and result['completed'] is False
        assert result['recovery_steps'] == 0 and result['recovery_completed'] is None
        assert result['stop_stability_verified'] is True
        assert result['stop_stability']['consecutive_quiet_samples'] == 6
        assert result['ik_diagnostics']['joint_hold'] is True
        assert .006 <= result['actual']['delta_m'] < .06
        # The rejected candidate never deploys, and stopping preserves partial progress.
        assert np.allclose(result['ee_pose_after']['base_position'], environment.rejection_poses[index][:3])
        assert result['execution']['executed_count'] == 1
        assert result['execution']['equivalent_fixed_primitives'] == 12
        assert len(result['child_results']) == 1
        feedback = AgentActionFeedback.from_result(result)
        assert feedback.motion_stopped is True and feedback.target_reached is None
        child_feedback = AgentActionFeedback.from_result(
            result['child_results'][0], state=environment.controller.controller_state())
        assert child_feedback.target_reached is False
        request = agent.inputs[index + 1]
        assert request.action_context['recovery']['reason'] == reason
        assert request.observation.last_action_feedback.observable_error == reason
        assert request.observation.action_step == index + 1
        assert request.observation.snapshot_id == agent.inputs[index].observation.snapshot_id + 1
        assert 'MV_FWD' in request.available_actions
    assert environment.raw_results[-1]['completed'] is True
    assert environment.controller.ik_guard.fault is None
    assert environment.calls[-1] == {'action': 'MV_FWD', 'distance_m': .005}
    for before, deployed in environment.deployments:
        assert np.max(np.abs(deployed - before)) <= np.max(environment.controller.ik_guard.max_step) + 1e-12
        assert np.all(deployed >= environment.controller.ik_guard.lower)
        assert np.all(deployed <= environment.controller.ik_guard.upper)


@pytest.mark.parametrize('unknown', [False, True])
def test_unstable_stop_or_unknown_transport_still_terminates(unknown, tmp_path):
    environment = GuardedEnvironment('ik_joint_jump', persistent=not unknown, unknown=unknown)
    agent = Agent('{"action":"MV_FWD","distance_m":0.06}', '{"action":"DONE"}')
    summary = asyncio.run(run_episode(environment, agent, tmp_path, max_steps=2, action_mode='parameterized'))
    assert summary['termination_reason'] == 'execution_error'
    assert len(environment.calls) == len(agent.inputs) == 1
    assert summary['recovered_execution_errors'] == []
    if not unknown:
        assert environment.raw_results[0]['stop_stability_verified'] is False
        assert AgentActionFeedback.from_result(environment.raw_results[0]).motion_stopped is None
    else:
        assert environment.raw_results[0]['stop_stability_verified'] is True


@pytest.mark.parametrize('reason', sorted(IK_REJECTION_ERRORS))
def test_legacy_recovery_flags_cannot_replace_rejection_stop_measurement(reason):
    assert reason in SETTLED_MOTION_ERRORS
    raw = {'action': 'MV_FWD', 'completed': False, 'reason': reason, 'stop_steps': 150,
           'recovery_completed': True, 'ik_diagnostics': {'joint_hold': True},
           'parameterized': True, 'macro_motion_stopped': True}
    assert AgentActionFeedback.from_result(raw).motion_stopped is None
    raw.update(recovery_completed=None, stop_stability_verified=True)
    assert AgentActionFeedback.from_result(raw).motion_stopped is True
