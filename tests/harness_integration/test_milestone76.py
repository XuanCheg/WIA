"""Targeted metric guard, snapshot identity and optional recording checks."""
import io
import json
import time
from dataclasses import replace

import numpy as np
from PIL import Image
import pytest
from types import SimpleNamespace

from core.action_compiler import MacroLimits
from wia.runtime.sim.esi_actions import create_r1_executor
from core.viewer_config import ViewerConfig
from wia.runtime.web.web_teleop.esi_recording import HumanVideoRecorder
from harness_integration.test_esi_arm import make_arm_controller
from harness_integration.test_esi_gripper import make_manipulation_controller
from harness_integration.test_agent_observation import source_and_rig


def metric_executor(controller):
    executor = create_r1_executor(controller)
    executor.macro_execution = 'metric_target'
    return executor


def test_cli_profiles_and_legacy_camera_flag_have_no_argument_conflicts():
    from wia.runtime.commands.run_esi_robot import parse_args
    default = parse_args([])
    assert default.macro_execution == 'primitive_replay' and default.performance_profile == 'debug'
    benchmark = parse_args(['--performance-profile', 'benchmark', '--macro-execution', 'metric_target'])
    assert not benchmark.debug_cameras and not benchmark.human_recording and not benchmark.human_preview
    assert benchmark.preview_fps == 0 and benchmark.observation_pipeline == 'optimized'
    assert not parse_args(['--no-debug-cameras']).debug_cameras


def test_metric_target_is_quantized_once_with_one_result_and_fewer_stops():
    fixed, _, env_fixed = make_arm_controller(translation_m=.005, translation_tolerance_m=.00075)
    metric, robot, env_metric = make_arm_controller(translation_m=.005, translation_tolerance_m=.00075)
    replay = create_r1_executor(fixed)
    executor = metric_executor(metric)
    results = []
    executor.on_result = results.append
    limits = MacroLimits(max_arm_primitives=16)
    command = {'action': 'MV_DOWN', 'distance_m': .079}
    before = robot.ee.copy()
    a = replay.execute_macro(command, resolution=replay.primitive_resolution, limits=limits)
    b = executor.execute_macro(command, resolution=executor.primitive_resolution, limits=limits)
    assert a['success'] and b['success']
    assert len(results) == 1 and results[0] is b
    assert b['requested']['distance_m'] == .079
    assert b['compilation']['effective_distance_m'] == .08
    assert np.allclose(b['target_base_position'], before+[0, 0, -.08])
    assert b['execution']['equivalent_fixed_primitives'] == 16
    assert b['execution']['actual_fixed_primitive_executions'] == 0
    assert b['execution']['metric_target_executions'] == 1
    assert len(env_metric.actions) < len(env_fixed.actions)
    assert abs(b['actual']['delta_m']-.08) <= metric.arm_config.translation_tolerance_m


@pytest.mark.parametrize('action,magnitude', [('BASE_FWD', .2), ('MV_UP', .08)])
def test_grasp_loss_interrupts_metric_without_retry(action, magnitude):
    controller, robot, env = make_manipulation_controller()
    assert controller.execute('GRASP')['success']
    env.drop_on_step = len(env.actions)+1
    executor = metric_executor(controller)
    result = executor.execute_macro({'action': action, 'distance_m': magnitude},
        resolution=executor.primitive_resolution, limits=MacroLimits())
    assert not result['success'] and result['reason'] == 'grasp_lost'
    assert result['execution']['metric_target_executions'] == 1
    assert result['execution']['actual_fixed_primitive_executions'] == 0
    assert result['gripper_after']['holding'] is False
    assert result['stop_steps'] > 0


def test_cancel_and_joint_drift_stop_metric_and_callbacks_restore():
    controller, robot, env = make_arm_controller()
    executor = metric_executor(controller)
    previous = controller.on_step
    result = executor.execute_macro({'action': 'MV_DOWN', 'distance_m': .08},
        resolution=executor.primitive_resolution, limits=MacroLimits(),
        should_cancel=lambda: len(env.actions) >= 3)
    assert result['cancelled'] and not result['success'] and result['stop_steps'] > 0
    assert result['execution']['metric_target_executions'] == 1
    assert controller.on_step is previous
    controller, robot, env = make_arm_controller()
    env.other_drift = 1.
    executor = metric_executor(controller)
    result = executor.execute_macro({'action': 'MV_DOWN', 'distance_m': .08},
        resolution=executor.primitive_resolution, limits=MacroLimits())
    assert result['reason'] == 'inactive_joint_drift' and not result['success']


def test_metric_http_cancel_retains_ownership_until_stop_and_one_boundary(tmp_path):
    import threading
    import requests
    from harness_integration.test_agent_control import control_http, claim
    # Use the real HTTP queue and controller on the simulator owner thread.
    fixture = control_http.__wrapped__(tmp_path)
    backend, _, url, _, _ = next(fixture)
    try:
        controller, _, env = make_arm_controller()
        executor = metric_executor(controller)
        executor.macro_limits = MacroLimits(max_arm_primitives=16)
        executor.on_result = backend.observation_source.after_action
        private_results = []
        executor.on_macro_result = private_results.append
        backend.executor = executor
        token = claim(backend, url)
        backend.action_mode = 'parameterized'
        initial = backend.observation_source.agent_observation()
        attempts = []
        def on_step():
            assert backend.observation_source.agent_observation() is initial
            if len(env.actions) == 3:
                attempts.append(len(env.actions))
                assert requests.post(url+'/api/cancel', json={}, timeout=2).json()['ok']
                assert backend.busy and backend.control_owner == 'agent'
                assert requests.post(url+'/api/move', json={'token': 'BASE_FWD'}, timeout=2).status_code == 409
        controller.on_step = on_step
        responses = []
        request = threading.Thread(target=lambda: responses.append(requests.post(url+'/api/agent/command',
            json={'command': {'action': 'MV_DOWN', 'distance_m': .08}, 'control_token': token}, timeout=3)))
        request.start()
        with backend.frame_cond:
            assert backend.frame_cond.wait_for(lambda: backend.busy, timeout=1)
        assert backend.process_next()
        request.join(timeout=3)
        result = responses[0].json()
        assert attempts == [3]
        assert result['macro']['execution']['metric_target_executions'] == 1
        assert result['macro']['execution']['actual_fixed_primitive_executions'] == 0
        assert result['results'][0]['execution_status'] == 'cancelled'
        assert private_results[0]['stop_steps'] > 0 and len(env.actions) < 12
        assert backend.steps == backend.observation_source.action_step == 1
        assert backend.control_owner == 'human' and not backend.busy
        assert not backend.process_next()  # A stopped command is never replayed.
        assert requests.post(url+'/api/agent/command', json={'command': {'action': 'MV_DOWN', 'distance_m': .08},
            'control_token': token}, timeout=2).status_code == 403
    finally:
        fixture.close()


def test_native_joint_limit_fault_stops_one_metric_attempt():
    controller, robot, env = make_arm_controller()
    def fault():
        if len(env.actions) == 1:
            # The native callback establishes its readback backend and recovery
            # posture before latching a fault; the lightweight rig omits IK.
            joints = robot.get_joint_positions()[controller.ik_guard.indices]
            controller.ik_guard.native_reference = joints.copy()
            controller.ik_guard.recovery = joints.copy()
            controller.ik_guard.fault = 'arm_joint_limit'
    controller.on_step = fault
    executor = metric_executor(controller)
    result = executor.execute_macro({'action': 'MV_DOWN', 'distance_m': .08},
        resolution=executor.primitive_resolution, limits=MacroLimits())
    assert result['reason'] == 'arm_joint_limit' and not result['success']
    assert result['execution']['metric_target_executions'] == 1
    assert result['execution']['actual_fixed_primitive_executions'] == 0
    assert result['stop_steps'] > 0 and len(env.actions) < 10


def test_snapshot_artifacts_share_exact_images_depth_and_not_numeric_ids(tmp_path):
    source, rig = source_and_rig()
    observation = source.capture(action_boundary=True).observation
    old_metadata = observation._metadata_uncached()
    artifacts = observation.artifacts()
    observation.save(tmp_path/'before')
    observation.save(tmp_path/'workspace')
    payload = observation.to_payload()
    for name, pixels in observation.rgb_images().items():
        if pixels is None:
            continue
        assert artifacts.model_png(name, pixels, 'identity') == (tmp_path/'before'/f'{name}.png').read_bytes()
        assert (tmp_path/'before'/f'{name}.png').read_bytes() == (tmp_path/'workspace'/f'{name}.png').read_bytes()
        with Image.open(io.BytesIO(artifacts.png(name))) as image:
            assert np.array_equal(np.array(image), pixels)
    assert artifacts.stats['rgb_encodes'] == 2
    assert observation.metadata() == old_metadata
    decoded = type(observation).from_payload(payload)
    decoded.save(tmp_path/'decoded')
    assert decoded.artifacts().stats['rgb_encodes'] == 0
    changed = replace(observation, agentview_rgb=np.zeros_like(observation.agentview_rgb))
    assert changed.snapshot_id == observation.snapshot_id
    assert changed.artifacts() is not artifacts and changed.artifacts().png('agentview_rgb') != artifacts.png('agentview_rgb')


def test_preview_never_advances_or_replaces_agent_boundary():
    source, rig = source_and_rig()
    source.human_depth_visualizations = False
    boundary = source.capture(action_boundary=True).observation
    sequence, step, history = source.snapshot_id, source.action_step, source.history
    preview = source.capture_preview()
    assert source.agent_observation() is boundary
    assert (source.snapshot_id, source.action_step, source.history) == (sequence, step, history)
    assert set(preview.frames) == {'head', 'wrist'}
    assert all(not frame.flags.writeable for frame in preview.frames.values())
    from wia.runtime.web.web_teleop.esi_preview import ESIPreviewBackend
    from harness_integration.test_esi_preview import METADATA
    backend = ESIPreviewBackend(dict(METADATA))
    backend.publish_preview(preview)
    assert 'head' in backend.frames and 'wrist' in backend.frames


def test_depth_initialization_invalidates_old_step_buffers_and_registers_once():
    from wia.runtime.sim.esi_omnigibson import ESISession
    session = ESISession.__new__(ESISession)
    class Sensor:
        modalities = ('rgb',)
        all_modalities = ('rgb', 'depth_linear')
        additions = 0
        def add_modality(self, modality):
            self.modalities = (*self.modalities, modality)
            self.additions += 1
    head, wrist = Sensor(), Sensor()
    renders = []
    session.robot = SimpleNamespace(sensors={'eyes': head, 'left_eef_link': wrist})
    session.closed = False
    session.camera_name, session.wrist_camera_name = 'eyes', 'left_eef_link'
    session._observation_source = None
    session._step_camera_data = {'eyes': {'rgb': np.zeros((8, 8, 3), np.uint8)}}
    session.pipeline = SimpleNamespace(og=SimpleNamespace(sim=SimpleNamespace(render=lambda: renders.append(1))))
    session.payload, session.task_module = {'_question': 'Public instruction'}, SimpleNamespace(TASK_NAME='test')
    source = session.observation_source
    assert session._step_camera_data is None
    assert session.observation_source is source
    assert head.additions == wrist.additions == 1 and len(renders) == 3


def test_unique_recording_only_selected_synchronized_source_frames(tmp_path):
    config = ViewerConfig(video_size=64, recording_mode='snapshots', recording_views=('head', 'working_wrist'),
                          recording_layout='individual')
    recorder = HumanVideoRecorder(tmp_path, config)
    frames = {name: np.full((64, 64, 3), 50, np.uint8) for name in ('head', 'wrist', 'global_front')}
    recorder.update(frames, 1)
    assert recorder.start()['ok']
    time.sleep(.12)
    recorder.update(frames, 1)  # Same capture is never written twice.
    recorder.update({name: rgb+40 for name, rgb in frames.items()}, 2)
    time.sleep(.12)
    result = recorder.finish()
    assert result['ok'], result
    record = result['recording']
    assert record['source_snapshot_count'] == 2
    assert {item['view'] for item in record['videos']} == {'head', 'wrist'}
    samples = [json.loads(line) for line in (tmp_path/record['id']/'samples.jsonl').read_text().splitlines()]
    assert all(set(row['files']) == {'head', 'wrist'} for row in samples)
    assert samples[1]['wall_elapsed_s'] > samples[0]['wall_elapsed_s']
    assert not (tmp_path/record['id']/'source_frames').exists()
    assert record['source_frames_retained'] is False
