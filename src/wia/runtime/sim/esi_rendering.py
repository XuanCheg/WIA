"""Native control stepping without image acquisition.

Used by ESISession.step_control; run CPU lifecycle checks with
python -m pytest tests/test_action_boundary_rendering.py.
"""
from __future__ import annotations

from contextlib import contextmanager
import time


def post_step_without_observation(env, action):
    """Keep installed Environment._post_step bookkeeping, deferring get_obs.

    This adapter follows BEHAVIOR-1K's envs/env_base.py. Empty observations are
    explicit deferred acquisition, never cached images presented as fresh data.
    """
    obs, obs_info = {}, {}
    if env._scene_graph_builder is not None:
        env._scene_graph_builder.step(env.scene)
    reward, done, info = env.task.step(env, action)
    env._populate_info(info)
    info["obs_info"] = obs_info
    if done and env._automatic_reset:
        info["last_observation"] = obs
        env.reset(get_obs=False)
    terminated = truncated = False
    for condition, data in info["done"]["termination_conditions"].items():
        if data["done"]:
            if condition == "timeout":
                truncated = True
            else:
                terminated = True
    assert (terminated or truncated) == done, "Terminated and truncated must match done!"
    env._current_step += 1
    return obs, reward, terminated, truncated, info


def step_without_observation(session, action):
    """Preserve native action conversion, sim lifecycle and task bookkeeping."""
    env, sim = session.env, session.pipeline.og.sim
    # Simulator.step must render newly imported objects before their first
    # physical step. Such scene construction belongs outside a control action.
    if getattr(sim, "_objects_to_initialize", ()):
        raise RuntimeError("Pending native object initialization requires rendering before control execution.")
    graph = getattr(env, "_scene_graph_builder", None)
    if graph is not None and getattr(graph, "_full_obs", True) is False:
        raise RuntimeError("A vision-dependent partial scene graph cannot update during unrendered control execution.")
    started = time.perf_counter()
    previous = sim._render_on_step
    try:
        action = env._convert_action_to_tensor(action)
        env._pre_step(action)
        sim._render_on_step = False
        physical_started = time.perf_counter()
        try:
            sim.step()
        finally:
            sim._render_on_step = previous
            session.invalidate_camera_cache()
            session._physical_step_generation = getattr(session, "_physical_step_generation", 0) + 1
            counters = getattr(session, "performance", None)
            if counters:
                counters.add("unrendered_sim_step_wall_time", time.perf_counter() - physical_started)
                counters.add("native_unrendered_control_step_count")
        return post_step_without_observation(env, action)
    finally:
        # Installed sim.render_on_step is not exception safe; restore explicitly.
        sim._render_on_step = previous
        counters = getattr(session, "performance", None)
        if counters:
            counters.add("controller_step_wall_time", time.perf_counter() - started)
            counters.add("controller_env_step_count")


@contextmanager
def action_execution(session):
    """Suppress intermediate image consumers through nested actions and stops."""
    depth = getattr(session, "_action_execution_depth", 0)
    generation = getattr(session, "_physical_step_generation", 0)
    counters = getattr(session, "performance", None)
    if depth == 0:
        session._action_performance_start = counters.snapshot() if counters else {}
        session._action_boundary_ready = False
    session._action_execution_depth = depth + 1
    try:
        yield
    except BaseException as error:
        # Successful/timeout/cancelled results capture through on_result. A
        # propagating backend error still needs a fresh post-stop boundary.
        if depth == 0 and getattr(session, "_physical_step_generation", 0) != generation:
            source = getattr(session, "_observation_source", None)
            if source is not None and getattr(source, "_captured_physical_generation", None) != getattr(
                    session, "_physical_step_generation", 0):
                try:
                    source.capture(action_boundary=True)
                except Exception as capture_error:
                    counters = getattr(session, "performance", None)
                    if counters:
                        counters.add("boundary_capture_failure_count")
                    if hasattr(error, "add_note"):
                        error.add_note(f"Final observation unavailable: {capture_error}")
        raise
    finally:
        session._action_execution_depth = depth
        if depth == 0:
            session._action_performance_start = None
            session._action_boundary_ready = False


class RecordingCadence:
    """Schedule owner-thread video captures using simulation time, without catch-up bursts."""

    def __init__(self, frequency_hz=5.0):
        import math
        if not math.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("Recording frequency must be finite and positive.")
        self.interval_s = 1.0 / frequency_hz
        self.last = None

    def due(self, simulation_time, *, active):
        if not active:
            self.last = None
            return False
        if (self.last is None or simulation_time < self.last
                or simulation_time - self.last >= self.interval_s - 1e-9):
            self.last = simulation_time
            return True
        return False
