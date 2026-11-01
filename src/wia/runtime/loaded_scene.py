"""Keep a completed native world paused in its owning resident GPU process.

Used by run_esi_episode_service.py; not a standalone entry point. A new world
request releases the previous scene through the ordinary full SDK lifecycle.
"""
from copy import copy, deepcopy


def source_state_copy(value):
    """Copy source counters/containers while preserving native object handles."""
    if isinstance(value, dict):
        result = copy(value)
        result.clear()
        result.update((key, source_state_copy(item)) for key, item in value.items())
        return result
    if isinstance(value, list):
        return [source_state_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(source_state_copy(item) for item in value)
    if isinstance(value, set):
        return set(value)
    if type(value).__module__.startswith(('numpy', 'torch')):
        return value.clone() if hasattr(value, 'clone') else value.copy() if hasattr(value, 'copy') else value
    return value


class LoadedScene:
    def __init__(self, close):
        self.close = close
        self.session = None
        self.validate = None

    def prepare(self, session):
        """Save the native initial state before any diagnostic or Agent action."""
        session._wia_initial_native_state = deepcopy(session.pipeline.og.sim.dump_state(serialized=False))
        session._wia_initial_source_state = source_state_copy(getattr(session, 'task_state', {}))

    def finish(self, session):
        if not getattr(session, '_wia_retain_completed', False):
            return self.close(session)
        simulator = session.pipeline.og.sim
        simulator.load_state(deepcopy(session._wia_initial_native_state), serialized=False)
        session.task_state = source_state_copy(session._wia_initial_source_state)
        simulator.pause()
        session.invalidate_camera_cache()
        session._wia_initial_native_state = None
        session._wia_initial_source_state = None
        self.session = session

    def release(self):
        if self.session is None:
            return
        session, self.session = self.session, None
        self.validate = None
        session._wia_retain_completed = False
        # Native cleanup owns all physics, sensor and callback disposal.
        self.close(session)

    @property
    def retained(self):
        return self.session is not None
