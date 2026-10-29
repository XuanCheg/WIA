"""Exercise factual target completion; run: python -m pytest -q this file."""
import json
import pytest

from wia.runtime.sim.esi_arm import ArmControlConfig
from wia.runtime.sim.esi_base import BaseControlConfig
from wia.runtime.sim.esi_rotation import RotationControlConfig
from wia.runtime.sim.esi_trunk import TrunkControlConfig
from wia.runtime.commands.run_esi_robot import parse_args
from harness_integration.test_esi_arm import make_arm_controller
from harness_integration.test_esi_base import make_controller


@pytest.mark.parametrize('kind', ['base', 'arm'])
def test_unreached_active_target_after_stop_remains_incomplete(kind):
    c, r, env = make_controller() if kind == 'base' else make_arm_controller()
    original = c.stop

    def displaced_target():
        original()
        if kind == 'base':
            r.xy[0] -= .1
        else:
            r.ee[0] -= .01

    c.stop = displaced_target
    result = c.execute('BASE_FWD' if kind == 'base' else 'MV_FWD')
    assert not result['completed'] and result['reason'] == 'target_not_reached'
    assert result['position_error_m'] > .005


@pytest.mark.parametrize('config,legacy', [
    (BaseControlConfig, {'inactive_joint_tolerance': 0., 'right_arm_hold_check': False}),
    (ArmControlConfig, {'path_tolerance_m': 0., 'base_hold_position_tolerance_m': 0.,
                        'base_hold_orientation_tolerance_deg': 0.}),
    (RotationControlConfig, {'path_tolerance_deg': 0.}),
    (TrunkControlConfig, {'horizontal_tolerance_m': 0., 'arm_hold_joint_tolerance_rad': 0.}),
])
def test_historical_config_keys_are_ignored_without_live_policy_fields(config, legacy, tmp_path):
    default = (config.load('configs/esi_trunk.json') if config is TrunkControlConfig else config.load())
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps(dict(vars(default), **legacy)))
    assert vars(config.load(path)) == vars(default)
    assert not legacy.keys() & vars(default).keys()


def test_legacy_cli_switch_is_hidden_and_inert(capsys):
    args = parse_args(['--disable-right-arm-hold-check'])
    assert args.disable_right_arm_hold_check
    with pytest.raises(SystemExit):
        parse_args(['--help'])
    assert 'right-arm-hold-check' not in capsys.readouterr().out


def test_measured_stop_verification_accepts_quiet_displaced_passive_joints():
    c, r, env = make_controller(progress_timeout_sim_s=.2)
    env.response = 0
    original = env.step
    dofs = r.controllers['arm_right'].dof_idx

    def displaced(actions):
        original(actions)
        r.joints[dofs] += .1

    env.step = displaced
    result = c.execute('BASE_FWD')
    c.verify_stopped_result(result)
    assert result['reason'] == 'base_motion_blocked'
    assert result['stop_stability_verified']
    assert result['stop_stability']['peak_joint_step_rad'] == 0.


def linear_trunk_controller(*, wrong_height=False):
    """Use measured X/Z joint coordinates to isolate height completion policy."""
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from wia.runtime.sim.esi_trunk import R1TrunkController
    import numpy as np

    owner, robot, env = make_arm_controller()
    robot.joints[[3, 4]] = [0., .9]
    owner.hold_action[owner.indices['trunk']] = robot.joints[[3, 4]]
    trunk = R1TrunkController.__new__(R1TrunkController)
    trunk.owner, trunk.config = owner, TrunkControlConfig.load('configs/esi_trunk.json')
    trunk.lower, trunk.upper = np.full(2, -10.), np.full(2, 10.)
    trunk.joints = lambda: robot.joints[[3, 4]].copy()
    trunk.measured_height = lambda q=None: float((trunk.joints() if q is None else q)[1])
    trunk.kinematics = SimpleNamespace(
        pose=lambda q: (np.array([q[0], 0., q[1]]), Rotation.identity()),
        solve=lambda q, delta, lower, upper: (q + [0., delta], np.array([q[0], 0., q[1]+delta]), None))
    owner.trunk_controller = trunk
    fault = {'wrong_height': wrong_height, 'x': .1}
    original = env.step

    def measured_offset(actions):
        original(actions)
        robot.joints[3] = fault['x']
        if fault['wrong_height']:
            robot.joints[4] = .9

    env.step = measured_offset
    return owner, robot, env, trunk, fault


@pytest.mark.parametrize('parameterized', [False, True])
def test_trunk_reached_height_with_displaced_xy_completes_and_reports_height_feedback(parameterized):
    from core.observations import AgentActionFeedback
    c, r, env, trunk, _ = linear_trunk_controller()
    result = trunk.execute('TRUNK_DOWN')
    result['parameterized'] = parameterized
    if parameterized:
        result['execution'] = {'strategy': 'metric_target'}
    assert result['completed'] and result['reason'] == 'completed'
    assert abs(result['actual_delta_m'][0]) >= .1
    expected = result['target_height_m'] - result['trunk_height_after_m']
    assert result['signed_height_error_m'] == pytest.approx(expected)
    assert result['absolute_height_error_m'] == pytest.approx(abs(expected))
    assert result['absolute_height_error_m'] <= trunk.config.translation_tolerance_m
    assert 'position_error_m' not in result and 'orientation_error_deg' not in result
    feedback = AgentActionFeedback.from_result(result, state=c.controller_state())
    assert feedback.target_reached is True
    assert feedback.estimated_motion['absolute_height_error_m'] == pytest.approx(abs(expected))


def test_wrong_trunk_height_times_out_and_reports_factual_signed_residual():
    from core.observations import AgentActionFeedback
    c, r, env, trunk, _ = linear_trunk_controller(wrong_height=True)
    result = trunk.execute('TRUNK_DOWN')
    assert not result['completed'] and result['timeout'] and result['reason'] == 'timeout'
    assert result['signed_height_error_m'] == pytest.approx(-.02)
    assert result['absolute_height_error_m'] == pytest.approx(.02)
    feedback = AgentActionFeedback.from_result(result, state=c.controller_state())
    assert feedback.target_reached is False
    assert feedback.estimated_motion['signed_height_error_m'] == pytest.approx(-.02)


def test_trunk_height_miss_introduced_during_stop_remains_incomplete():
    c, r, env, trunk, _ = linear_trunk_controller()
    stop = c.stop

    def wrong_height_at_stop():
        stop()
        r.joints[4] += .01

    c.stop = wrong_height_at_stop
    result = trunk.execute('TRUNK_DOWN')
    assert not result['completed'] and result['reason'] == 'target_not_reached'
    assert result['absolute_height_error_m'] > trunk.config.translation_tolerance_m


def test_trunk_stop_refresh_uses_height_only_with_displaced_xy():
    from core.observations import AgentActionFeedback
    c, r, env, trunk, fault = linear_trunk_controller(wrong_height=True)
    result = trunk.execute('TRUNK_DOWN')
    assert result['timeout']
    fault['wrong_height'] = False
    fault['x'] = .4
    r.joints[[3, 4]] = [.4, result['target_height_m']]
    c.hold_action[c.indices['trunk']] = r.joints[[3, 4]]
    c.verify_stopped_result(result)
    assert result['stop_stability_verified']
    assert result['timeout'] and not result['completed']
    assert abs(result['actual_delta_m'][0]) >= .4
    assert result['absolute_height_error_m'] == pytest.approx(0., abs=1e-7)
    assert 'position_error_m' not in result
    assert AgentActionFeedback.from_result(result, state=c.controller_state()).target_reached is True


@pytest.mark.parametrize('strategy', ['primitive_replay', None])
def test_aggregate_trunk_feedback_does_not_reuse_last_child_height_residual(strategy):
    from core.observations import AgentActionFeedback
    c, _, _, trunk, _ = linear_trunk_controller()
    result = trunk.execute('TRUNK_DOWN')
    result['parameterized'] = True
    result['execution'] = {'strategy': strategy}
    feedback = AgentActionFeedback.from_result(result, state=c.controller_state())
    assert feedback.target_reached is None
    assert 'signed_height_error_m' not in feedback.estimated_motion
    assert 'absolute_height_error_m' not in feedback.estimated_motion


@pytest.mark.parametrize('solved_height_error_m', [0., .01])
def test_trunk_solver_feasibility_ignores_lateral_orientation_residual_but_requires_height(solved_height_error_m):
    import numpy as np
    from scipy.spatial.transform import Rotation

    c, r, env, trunk, _ = linear_trunk_controller()
    # The solver prefers a nominal pose but may return a valid bounded
    # height solution with substantial residual on uncommanded axes.
    trunk.kinematics.pose = lambda q: (np.array([q[0], 0., q[1]]),
                                       Rotation.from_rotvec([q[0]*10., 0., 0.]))
    trunk.kinematics.solve = lambda q, delta, lower, upper: (
        q + [.1, delta + solved_height_error_m], np.array([q[0], 0., q[1]+delta]), None)
    result = trunk.execute('TRUNK_DOWN')
    if solved_height_error_m == 0.:
        assert result['completed'] and result['reason'] == 'completed'
        assert result['actual_delta_m'][0] == pytest.approx(.1)
        assert result['absolute_height_error_m'] <= trunk.config.translation_tolerance_m
        assert len(env.actions) > c.config.stop_steps
    else:
        assert not result['completed'] and result['reason'] == 'trunk_height_limit'
        assert len(env.actions) == c.config.stop_steps


@pytest.mark.parametrize('direction,start,distance,expected', [
    ('TRUNK_DOWN', .662, .1, .58),
    ('TRUNK_DOWN', .662, .3, .58),
    ('TRUNK_UP', .9, .3, 1.08),
    ('TRUNK_DOWN', .58, .1, .58),
    ('TRUNK_UP', 1.08, .1, 1.08),
])
def test_trunk_saturates_workspace_and_reports_effective_target(direction, start, distance, expected):
    from core.observations import AgentActionFeedback
    c, r, env, trunk, _ = linear_trunk_controller()
    r.joints[4] = start
    result = trunk.execute_metric(direction, distance)
    result['parameterized'] = True
    result['execution'] = {'strategy': 'metric_target'}
    assert result['completed'], result
    assert result['target_clamped'] is True
    assert result['target_height_m'] == pytest.approx(expected)
    assert result['trunk_height_after_m'] == pytest.approx(expected, abs=trunk.config.translation_tolerance_m)
    assert result['requested_m'] == pytest.approx(distance if direction == 'TRUNK_UP' else -distance)
    feedback = AgentActionFeedback.from_result(result, state=c.controller_state())
    assert feedback.estimated_motion['target_clamped'] is True
    assert feedback.estimated_motion['target_height_m'] == pytest.approx(expected)
