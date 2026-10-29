"""Episode phase accounting. Simulator time is sampled only by its owner."""
from __future__ import annotations

import time


class WorldPolicy:
    def __init__(self):
        self.on_phase = None
        self.advances = True
        self.phase = "human_idle"
        self.sim_time = 0.
        self.started_wall = time.perf_counter()
        self.started_sim = 0.
        self.metrics = self._empty()

    @staticmethod
    def _empty():
        return {"model_wait_wall_time_s": 0., "simulation_time_during_model_wait_s": 0.,
                "idle_controller_env_steps": 0, "action_execution_sim_time_s": 0.,
                "action_execution_wall_time_s": 0.}

    def begin(self, advances, sim_time):
        if type(advances) is not bool:
            raise ValueError("world_advances_during_inference must be boolean.")
        self.advances = advances
        self.metrics = self._empty()
        self.phase = "initializing"
        self.sim_time = self.started_sim = float(sim_time)
        self.started_wall = time.perf_counter()

    def sample(self, sim_time, *, idle_step=False):
        self.sim_time = float(sim_time)
        if idle_step and self.phase == "awaiting_model":
            self.metrics["idle_controller_env_steps"] += 1

    def transition(self, phase, sim_time=None):
        if sim_time is not None:
            self.sample(sim_time)
        now = time.perf_counter()
        wall, simulated = now-self.started_wall, max(0., self.sim_time-self.started_sim)
        if self.phase == "awaiting_model":
            self.metrics["model_wait_wall_time_s"] += wall
            self.metrics["simulation_time_during_model_wait_s"] += simulated
        elif self.phase == "executing":
            self.metrics["action_execution_wall_time_s"] += wall
            self.metrics["action_execution_sim_time_s"] += simulated
        self.phase, self.started_wall, self.started_sim = phase, now, self.sim_time
        if self.on_phase is not None:
            self.on_phase(phase)

    def snapshot(self):
        result = dict(self.metrics)
        if self.phase in ("awaiting_model", "executing"):
            prefix = "model_wait" if self.phase == "awaiting_model" else "action_execution"
            result[prefix + "_wall_time_s"] += time.perf_counter()-self.started_wall
            key = "simulation_time_during_model_wait_s" if self.phase == "awaiting_model" else "action_execution_sim_time_s"
            result[key] += max(0., self.sim_time-self.started_sim)
        return dict(result, world_advances_during_inference=self.advances,
                    world_mode="run" if self.advances else "pause", episode_phase=self.phase,
                    sampled_simulation_time_s=self.sim_time)

    def should_advance(self, control_owner):
        if self.phase == "debug_hold":
            return False
        return not (control_owner == "agent" and self.phase == "awaiting_model" and not self.advances)
