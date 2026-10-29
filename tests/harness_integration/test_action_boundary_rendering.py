"""Run CPU lifecycle checks with python -m pytest tests/test_action_boundary_rendering.py."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.action_interface import ActionExecutor, BackendCapabilities, action_result
from core.action_compiler import MacroLimits, PrimitiveResolution
from core.performance import PerformanceCounters
from wia.runtime.sim.esi_observations import ESIObservationSource
from wia.runtime.sim.esi_omnigibson import ESISession, NativeCameraNotReady


def native_rig(*, done=None, fail=None, render_policy=True):
    events = []
    session = ESISession()
    session.performance = PerformanceCounters()
    sim = SimpleNamespace(_render_on_step=render_policy, _objects_to_initialize=[],
                          physical=0, renders=0, rendered_physical=-1)

    def step():
        assert not sim._render_on_step
        events.append("physics_callbacks")
        sim.physical += 4  # Preserve the installed physics/render frequency ratio.
        events.append("nonphysics_updates")
        if fail == "physics":
            raise RuntimeError("physics error")

    def render():
        sim.renders += 1
        sim.rendered_physical = sim.physical

    def convert(action):
        events.append("convert")
        return dict(action, converted=True)

    def pre(action):
        assert action["converted"]
        events.append("pre_step")

    def task_step(env, action):
        events.append("task")
        assert action["converted"]
        if fail == "task":
            raise RuntimeError("task error")
        return 2.5, done is not None, {"done": {"termination_conditions": {
            "timeout": {"done": done == "timeout"}, "success": {"done": done == "success"}}}}

    def observe(**kwargs):
        events.append("sensor_images")
        frame = np.full((8, 8, 3), sim.rendered_physical, dtype=np.uint8)
        return {"head_rgb": frame, "wrist_rgb": frame.copy()}

    sim.step, sim.render = step, render
    env = SimpleNamespace(_convert_action_to_tensor=convert, _pre_step=pre,
        _scene_graph_builder=SimpleNamespace(step=lambda scene: events.append("scene_graph")),
        scene=object(), task=SimpleNamespace(step=task_step),
        _populate_info=lambda info: info.update(populated=True),
        _automatic_reset=False, _current_step=8,
        get_obs=lambda: pytest.fail("Environment get_obs must not acquire control-step imagery"))
    session.env = env
    session.robot = SimpleNamespace()
    session.pipeline = SimpleNamespace(og=SimpleNamespace(sim=sim))
    session.observe = observe
    source = ESIObservationSource(session, task_text="Move forward.")
    source.controller_state_provider = None
    session._observation_source = source
    return session, sim, env, source, events


@pytest.mark.parametrize("done,terminated,truncated", [(None, False, False),
    ("success", True, False), ("timeout", False, True)])
def test_unrendered_step_preserves_native_lifecycle(done, terminated, truncated):
    session, sim, env, _, events = native_rig(done=done)
    session._step_camera_data = {"stale": object()}
    result = session.step_control({"robot": [0]})
    assert result[:4] == ({}, 2.5, terminated, truncated)
    assert result[-1]["populated"] and result[-1]["obs_info"] == {}
    assert events == ["convert", "pre_step", "physics_callbacks", "nonphysics_updates", "scene_graph", "task"]
    assert env._current_step == 9 and sim.physical == 4 and sim.renders == 0
    assert session._step_camera_data is None and sim._render_on_step is True
    metrics = session.performance.snapshot()
    assert metrics["native_unrendered_control_step_count"] == metrics["controller_env_step_count"] == 1
    assert metrics["unrendered_sim_step_wall_time"] >= 0


@pytest.mark.parametrize("fail", ["physics", "task"])
@pytest.mark.parametrize("policy", [False, True])
def test_failure_restores_render_policy_and_invalidates_images(fail, policy):
    session, sim, _, _, _ = native_rig(fail=fail, render_policy=policy)
    session._step_camera_data = {"old": object()}
    with pytest.raises(RuntimeError, match="error"):
        session.step_control({"robot": [0]})
    assert sim._render_on_step is policy and session._step_camera_data is None
    assert sim.renders == 0


def test_pending_object_initialization_is_explicit_and_does_not_step():
    session, sim, _, _, events = native_rig()
    sim._objects_to_initialize.append(object())
    with pytest.raises(RuntimeError, match="initialization requires rendering"):
        session.step_control({"robot": [0]})
    assert not events and sim.physical == sim.renders == 0 and sim._render_on_step


def test_automatic_reset_keeps_task_reset_without_acquiring_images():
    session, _, env, _, events = native_rig(done="success")
    env._automatic_reset = True
    def reset(*, get_obs):
        assert get_obs is False
        events.append("reset")
        env._current_step = 0
    env.reset = reset
    result = session.step_control({"robot": [0]})
    assert result[-1]["last_observation"] == {} and env._current_step == 1
    assert events[-1] == "reset"


def test_vision_dependent_scene_graph_is_rejected_before_physics():
    session, sim, env, _, events = native_rig()
    env._scene_graph_builder._full_obs = False
    with pytest.raises(RuntimeError, match="vision-dependent partial scene graph"):
        session.step_control({"robot": [0]})
    assert not events and sim.physical == sim.renders == 0


def executor_rig(*, status="completed"):
    session, sim, env, source, events = native_rig()
    published = []
    source.on_capture = published.append
    source.capture(action_boundary=True)
    def execute(action, **kwargs):
        retained = source.latest_snapshot
        for _ in range(3):
            session.step_control({"robot": [0]})
            assert source.capture() is retained
            with pytest.raises(RuntimeError, match="previews are deferred"):
                source.capture_preview()
        return action_result(action, completed=status == "completed", timeout=status == "timeout",
            cancelled=status == "cancelled", requested={}, actual={}, message=status)
    def verify(result):
        assert session.action_executing
        session.step_control({"robot": [0]})
    executor = ActionExecutor(execute, capabilities=BackendCapabilities(base_translation=True),
        execution_context=session.action_execution, verify_stopped=verify, on_result=source.after_action)
    executor.performance = session.performance
    return session, sim, env, source, events, published, executor


@pytest.mark.parametrize("status", ["completed", "timeout", "cancelled"])
def test_boundary_after_final_stop_is_fresh_and_intermediate_images_are_suppressed(status):
    session, sim, env, source, events, published, executor = executor_rig(status=status)
    initial_reads = events.count("sensor_images")
    result = executor.execute("BASE_FWD")
    assert len(published) == 2 and events.count("sensor_images") == initial_reads + 1
    assert sim.physical == 16 and env._current_step == 12 and sim.renders == 2
    assert np.all(source.agent_observation().agentview_rgb == 16)
    assert not session.action_executing and sim._render_on_step
    assert result["timing"]["boundary_render_count"] == 1
    assert result["timing"]["native_unrendered_control_step_count"] == 4


def test_primitive_replay_macro_has_one_final_capture_after_all_child_stops():
    session, sim, _, source, events, published, executor = executor_rig()
    result = executor.execute_macro({"action": "BASE_FWD", "distance_m": .15},
        resolution=PrimitiveResolution(base_translation_m=.05), limits=MacroLimits())
    assert result["success"] and result["execution"]["executed_count"] == 3
    assert sim.physical == 48 and sim.renders == 2 and len(published) == 2
    assert events.count("sensor_images") == 2 and source.action_step == 1
    assert result["timing"]["native_unrendered_control_step_count"] == 12
    assert not session.action_executing


def test_metric_macro_keeps_stop_and_final_capture_in_one_action_scope():
    session, sim, _, source, events, published, executor = executor_rig()
    executor.macro_execution = "metric_target"
    executor._execute_metric = lambda action, magnitude, **kwargs: executor._execute_action(action, **kwargs)
    result = executor.execute_macro({"action": "BASE_FWD", "distance_m": .15},
        resolution=PrimitiveResolution(base_translation_m=.05), limits=MacroLimits())
    assert result["success"] and result["execution"]["strategy"] == "metric_target"
    assert sim.physical == 16 and sim.renders == 2 and len(published) == 2
    assert events.count("sensor_images") == 2 and source.action_step == 1
    assert not session.action_executing


def test_propagated_controller_error_refreshes_final_observation_and_unwinds_scope():
    session, sim, _, source, _, _, executor = executor_rig()
    def fail(action, **kwargs):
        try:
            session.step_control({"robot": [0]})
            raise RuntimeError("controller error")
        finally:
            session.step_control({"robot": [0]})  # Backend stop in finally.
    executor._execute_action = fail
    with pytest.raises(RuntimeError, match="controller error"):
        executor.execute("BASE_FWD")
    assert not session.action_executing and sim._render_on_step
    assert sim.physical == 8 and np.all(source.agent_observation().agentview_rgb == 8)
    assert source.action_step == 0


def test_camera_readiness_flushes_are_graphics_only_and_count_actual_renders():
    session, sim, _, source, _, _, executor = executor_rig()
    original = session.observe
    attempts = []
    def observe(**kwargs):
        attempts.append(sim.physical)
        if len(attempts) < 3:
            raise NativeCameraNotReady("annotator warming up")
        return original(**kwargs)
    session.observe = observe
    result = executor.execute("BASE_FWD")
    assert attempts == [16, 16, 16] and sim.physical == 16
    assert sim.renders == 4 and result["timing"]["boundary_render_count"] == 3
    assert np.all(source.agent_observation().agentview_rgb == 16)


def test_native_boundary_flushes_four_times_only_after_physical_advance():
    session, sim, _, source, events, published, executor = executor_rig()
    session._native_boundary_graphics_iterations = 4
    source.capture(action_boundary=True)  # No physical change: ordinary render.
    assert sim.renders == 2
    result = executor.execute("BASE_FWD")
    assert sim.renders == 6 and sim.physical == 16
    assert result["timing"]["boundary_render_count"] == 4
    assert events.count("sensor_images") == len(published) == 3
    source.capture(action_boundary=True)
    assert sim.renders == 7 and sim.physical == 16


def test_valid_rgb_is_not_published_until_native_pose_is_current(monkeypatch):
    session, sim, _, source, _, published, executor = executor_rig()
    session._native_boundary_graphics_iterations = 4
    renders_before = sim.renders
    checks = []
    monkeypatch.setattr("wia.runtime.sim.esi_observations.prepare_camera_freshness", lambda session: ("target",))

    def check(session, targets):
        assert targets == ("target",)
        checks.append(sim.renders - renders_before)
        if checks[-1] < 5:
            raise NativeCameraNotReady("head: rendered_camera_pose_stale")
        # Only the initial snapshot has been published while polling freshness.
        assert len(published) == 1

    monkeypatch.setattr("wia.runtime.sim.esi_observations.check_camera_freshness", check)
    result = executor.execute("BASE_FWD")
    assert checks == [4, 5]
    assert sim.physical == 16 and len(published) == 2
    assert result["timing"]["boundary_render_count"] == 5


def test_exhausted_pose_freshness_retries_preserve_last_snapshot(monkeypatch):
    session, sim, _, source, _, published, _ = executor_rig()
    session._native_boundary_graphics_iterations = 4
    session.step_control({"robot": [0]})
    retained = source.latest_snapshot
    snapshot_id = source.snapshot_id
    generation = source._captured_physical_generation
    renders_before = sim.renders
    monkeypatch.setattr("wia.runtime.sim.esi_observations.prepare_camera_freshness", lambda session: ("target",))

    def unavailable(session, targets):
        raise NativeCameraNotReady("head: rendered_camera_pose_stale")

    monkeypatch.setattr("wia.runtime.sim.esi_observations.check_camera_freshness", unavailable)
    with pytest.raises(RuntimeError, match="rendered_camera_pose_stale"):
        source.capture(action_boundary=True)
    assert sim.renders - renders_before == 11 and sim.physical == 4
    assert source.latest_snapshot is retained and source.snapshot_id == snapshot_id
    assert source._captured_physical_generation == generation and len(published) == 1
    assert not session._action_boundary_ready


def test_direct_native_sensor_read_is_blocked_inside_action_scope():
    session, _, _, _, _ = native_rig()
    sensor = SimpleNamespace(name="head", get_obs=lambda: pytest.fail("Intermediate sensor read"))
    with session.action_execution():
        with pytest.raises(RuntimeError, match="Native sensor images are deferred"):
            session._sensor_data(sensor)


def test_idle_preview_refreshes_graphics_without_advancing_world():
    session, sim, _, source, _, _, _ = executor_rig()
    session.step_control({"robot": [0]})
    preview = source.capture_preview()
    assert sim.physical == 4 and sim.renders == 2
    assert np.all(preview.frames["head"] == 4)


def test_web_callbacks_keep_owner_and_task_observers_without_action_previews(monkeypatch):
    import threading
    import wia.runtime.sim.esi_web as esi_web

    session, sim, _, source, _, _, executor = executor_rig()
    calls = []
    controller = SimpleNamespace(on_step=lambda: calls.append("task_observer"), dt=.01,
        controller_state=lambda: None, stop=lambda: None)
    backend = SimpleNamespace(closing=threading.Event(), busy=False, control_owner="human",
        action_groups={"arm": False, "gripper": False},
        owner_phase=lambda phase: None, owner_sample=lambda **kwargs: calls.append("owner_sample"),
        publish=lambda *args, **kwargs: None, shutdown=lambda: None)

    def execute(action, **kwargs):
        backend.busy = True
        try:
            for _ in range(3):
                session.step_control({"robot": [0]})
                controller.on_step()
            return action_result(action, completed=True, timeout=False, requested={}, actual={}, message="completed")
        finally:
            backend.busy = False

    executor._execute_action = execute
    def process_next(**kwargs):
        executor.execute("BASE_FWD")
        backend.closing.set()
        return True
    backend.process_next = process_next
    server = SimpleNamespace(serve_forever=lambda: None, server_port=0,
        shutdown=lambda: None, server_close=lambda: None)
    monkeypatch.setattr(esi_web, "ESIControlBackend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(esi_web, "serve", lambda *args, **kwargs: server)
    previews = []
    original = source.capture_preview
    def preview():
        assert not session.action_executing
        previews.append(sim.physical)
        return original()
    source.capture_preview = preview
    esi_web.run_esi_control(session, {"task": "Move forward."}, controller, executor,
        port=0, preview_fps=1e9, observation_pipeline="legacy")
    assert calls.count("owner_sample") == calls.count("task_observer") == 3
    # Initial boundary, action boundary and optional idle capture, never motion.
    assert sim.physical == 16 and sim.renders == 4 and not previews


@pytest.mark.parametrize("owner,expected_physics", [("human", 0), ("agent", 12)])
def test_preloaded_standby_preserves_scene_until_agent_control(monkeypatch, owner, expected_physics):
    import threading
    import wia.runtime.sim.esi_web as esi_web

    session, sim, _, _, _, _, executor = executor_rig()
    session.standby_until_agent = True
    phases, polls = [], []
    controller = SimpleNamespace(on_step=None, dt=.001, controller_state=lambda: None, stop=lambda: None,
        step_base_debug=lambda: session.step_control({"robot": [0]}))
    backend = SimpleNamespace(closing=threading.Event(), busy=False, control_owner=owner,
        action_groups={"arm": False, "gripper": False},
        owner_phase=phases.append, owner_sample=lambda **kwargs: None,
        publish=lambda *args, **kwargs: None, shutdown=lambda: None, should_advance_world=lambda: True)

    def process_next(**kwargs):
        polls.append(1)
        if len(polls) == 3:
            backend.closing.set()
        return False

    backend.process_next = process_next
    server = SimpleNamespace(serve_forever=lambda: None, server_port=0,
        shutdown=lambda: None, server_close=lambda: None)
    monkeypatch.setattr(esi_web, "ESIControlBackend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(esi_web, "serve", lambda *args, **kwargs: server)
    esi_web.run_esi_control(session, {"task": "Wait.", "performance": {"human_preview": False}},
                           controller, executor, port=0, preview_fps=0.)
    assert sim.physical == expected_physics
    assert phases.count("standby") == (3 if owner == "human" else 0)


def test_recording_cadence_uses_simulation_time_and_resets_when_disabled():
    from wia.runtime.sim.esi_rendering import RecordingCadence
    cadence = RecordingCadence(5.)
    assert cadence.due(0., active=True)
    assert not cadence.due(.1, active=True)
    assert cadence.due(.2, active=True)
    assert not cadence.due(.21, active=True)
    assert cadence.due(1., active=True)
    assert not cadence.due(1., active=True)
    assert not cadence.due(2., active=False)
    assert cadence.due(2.01, active=True)


@pytest.mark.parametrize('frequency', [0, -1, float('inf'), float('nan')])
def test_recording_frequency_must_be_finite_positive(frequency):
    from wia.runtime.sim.esi_rendering import RecordingCadence
    with pytest.raises(ValueError):
        RecordingCadence(frequency)


def test_intermediate_recording_preserves_agent_snapshot_and_physics():
    session, sim, _, source, _, published, executor = executor_rig()
    before = source.agent_observation()
    physics = sim.physical
    with session.action_execution():
        preview = source.capture_preview(recording=True)
    assert sim.physical == physics
    assert source.agent_observation() is before
    assert len(published) == 1
    assert preview.source_snapshot_id == before.snapshot_id


def test_native_recording_defaults_and_frequency():
    from wia.runtime.commands.run_esi_robot import parse_args
    assert parse_args(['--run-mode', 'diagnostic']).human_recording
    assert not parse_args(['--run-mode', 'benchmark']).human_recording
    args = parse_args(['--run-mode', 'benchmark', '--human-recording', '--recording-hz', '8'])
    assert args.human_recording and args.recording_hz == 8
    assert parse_args(['--run-mode', 'diagnostic']).recording_hz == 5


@pytest.mark.parametrize("recording", [False, True])
def test_web_recording_captures_during_actions_only_when_active(monkeypatch, recording):
    import threading
    import wia.runtime.sim.esi_web as esi_web

    session, sim, _, source, _, _, executor = executor_rig()
    calls = []
    sim.current_time = 0.
    video_frames = []
    import wia.runtime.web.web_teleop.esi_viewer as viewer_module
    viewer = SimpleNamespace(recording_enabled=True,
        recorder=SimpleNamespace(state=lambda: {'recording': recording}),
        publish=lambda snapshot: None, publish_preview=video_frames.append, close=lambda: None)
    monkeypatch.setattr(viewer_module, 'ESIViewerBackend', lambda *args: viewer)
    controller = SimpleNamespace(on_step=lambda: calls.append("task_observer"), dt=.01,
        controller_state=lambda: None, stop=lambda: None)
    backend = SimpleNamespace(closing=threading.Event(), busy=False, control_owner="human",
        action_groups={"arm": False, "gripper": False},
        owner_phase=lambda phase: None, owner_sample=lambda **kwargs: calls.append("owner_sample"),
        publish=lambda *args, **kwargs: None, shutdown=lambda: None)

    def execute(action, **kwargs):
        backend.busy = True
        try:
            for _ in range(3):
                session.step_control({"robot": [0]})
                sim.current_time += .1
                controller.on_step()
            return action_result(action, completed=True, timeout=False, requested={}, actual={}, message="completed")
        finally:
            backend.busy = False

    executor._execute_action = execute
    def process_next(**kwargs):
        executor.execute("BASE_FWD")
        backend.closing.set()
        return True
    backend.process_next = process_next
    server = SimpleNamespace(serve_forever=lambda: None, server_port=0,
        shutdown=lambda: None, server_close=lambda: None)
    monkeypatch.setattr(esi_web, "ESIControlBackend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(esi_web, "serve", lambda *args, **kwargs: server)
    previews = []
    original = source.capture_preview
    def preview(**kwargs):
        assert session.action_executing and kwargs == {'recording': True}
        previews.append(sim.physical)
        return original(**kwargs)
    source.capture_preview = preview
    esi_web.run_esi_control(session, {"task": "Move forward."}, controller, executor,
        port=0, preview_fps=0, observation_pipeline="optimized", viewer_config=SimpleNamespace())
    assert calls.count("owner_sample") == calls.count("task_observer") == 3
    # Initial boundary, action boundary and optional idle capture, never motion.
    assert sim.physical == 16
    assert len(previews) == len(video_frames) == (2 if recording else 0)
    assert source.action_step == 1



@pytest.mark.parametrize('fail', [False, True])
def test_recording_scope_allows_native_sensor_gate_and_restores_on_error(fail):
    session, sim, _, source, _, _, executor = executor_rig()
    sensor = SimpleNamespace(name='test', get_obs=lambda: {'rgb': np.zeros((2, 2, 3), dtype=np.uint8)})
    original = session.observe
    def observe(**kwargs):
        session._sensor_data(sensor)
        if fail:
            raise RuntimeError('readback failed')
        return original(**kwargs)
    session.observe = observe
    with session.action_execution():
        with pytest.raises(RuntimeError, match='deferred'):
            session._sensor_data(sensor)
        if fail:
            with pytest.raises(RuntimeError, match='readback failed'):
                source.capture_preview(recording=True)
        else:
            source.capture_preview(recording=True)
        assert session._recording_capture_depth == 0
        with pytest.raises(RuntimeError, match='deferred'):
            session._sensor_data(sensor)
