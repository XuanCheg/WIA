"""Capture authored candidate history before the first policy decision.

Run: PYTHONPATH=src python -m pytest -q tests/test_candidate_prelude.py

``CandidatePrelude.run(source, initial_observation)`` records exact native head
frames and returns the canonical observation after the authored event restores
its scene. The initial robot/Global snapshot must already exist. ``run`` gives
no observation credit. The runtime owner calls ``mark_delivered()`` only after
including ``session.candidate_public_history`` in the actual control-claim
bundle. That acknowledgement returns private evidence, never public labels.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
from types import SimpleNamespace

import numpy as np
from PIL import Image

from wia.candidates.events import CandidateEvents
from wia.candidates.observations import CandidateObservationEvidence


SCHEMA = "wia_candidate_observation_history_v1"
CAMERA = "agentview_rgb"
MAX_FRAMES = 24
_ABSENT = object()


def _boundary(observation):
    return (observation.snapshot_id, observation.captured_at)


def _signature(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


class CandidatePrelude:
    """Author one event per native reset and defer evidence until delivery."""

    def __init__(self, session, parameters, evaluator):
        if not (parameters["prototype"] == "M1" or
                (parameters["prototype"] == "H3" and parameters["contract"].get("requires_revision_event"))):
            raise ValueError("Only M1 and revision-event H3 have an episode prelude")
        self.session, self.parameters, self.evaluator = session, parameters, evaluator
        self.state = session.candidate_state
        self.events = CandidateEvents(session, parameters)
        self.visual = CandidateObservationEvidence(session, parameters)
        self.private_records = []
        self.initial_boundary = self.final_boundary = self.final_observation = None
        self.payload_signature = None
        self.delivered = False
        self.run_status = "new"
        self.delivery_receipt = None

    def _append(self, observation, frames):
        if len(frames) >= MAX_FRAMES:
            raise RuntimeError("Candidate prelude exceeds the bounded public frame count")
        rgb = np.asarray(observation.rgb_images().get(CAMERA))
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Candidate prelude requires exact native uint8 head RGB")
        snapshot_id, captured_at = _boundary(observation)
        if type(snapshot_id) is not int or snapshot_id < 1 or not isinstance(captured_at, str) or not captured_at:
            raise ValueError("Candidate prelude observation provenance is unavailable")
        measured = deepcopy(self.evaluator.measure())
        time_s = measured.get("time_s")
        if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(time_s):
            raise ValueError("Candidate prelude requires native simulation time")
        if frames and (snapshot_id <= frames[-1]["snapshot_id"] or time_s < frames[-1]["simulation_time_s"]):
            raise ValueError("Candidate prelude frames must remain in native chronological order")
        rgb = np.array(rgb, copy=True)
        restricted = SimpleNamespace(rgb_images=lambda: {CAMERA: rgb}, snapshot_id=snapshot_id,
                                     captured_at=captured_at, action_step=observation.action_step)
        evidence = self.visual.record(restricted, measured)
        encoded = BytesIO()
        Image.fromarray(rgb).save(encoded, format="PNG")
        png = encoded.getvalue()
        frames.append({"index": len(frames), "snapshot_id": snapshot_id,
                       "captured_at": captured_at, "simulation_time_s": float(time_s),
                       "camera": CAMERA, "png_base64": base64.b64encode(png).decode("ascii"),
                       "sha256": hashlib.sha256(png).hexdigest()})
        measured.update(snapshot_id=snapshot_id, captured_at=captured_at)
        self.evaluator.prelude_frames.append(measured)
        self.private_records.append({"frame": measured, "visual_evidence": deepcopy(evidence)})

    def run(self, source, initial_observation):
        """Capture once; resume is idempotent and a new boundary cannot replay it."""
        boundary = _boundary(initial_observation)
        if self.state is not self.session.candidate_state:
            raise RuntimeError("Install a new candidate prelude after a native scene reset")
        if self.run_status == "ready":
            if boundary in (self.initial_boundary, self.final_boundary):
                return self.final_observation
            raise RuntimeError("Candidate prelude replay attempted at a different episode boundary")
        if self.run_status != "new" or self.state.get("episode_prelude", {}).get("status"):
            raise RuntimeError("Candidate prelude already authored or failed on this native reset")
        if getattr(initial_observation, "action_step", None) != 0:
            raise RuntimeError("Candidate prelude must precede the first policy action")
        self.initial_boundary = boundary
        self.run_status = "running"
        self.state["episode_prelude"] = {"status": "running", "initial_snapshot_id": boundary[0]}
        self.private_records = []
        self.evaluator.prelude_frames = []
        self.session.candidate_public_history = None
        frames = []
        previous_hook = getattr(self.session, "candidate_observation_hook", _ABSENT)
        self.session.candidate_observation_hook = None
        try:
            self._append(initial_observation, frames)

            def capture(_private_label):
                for _ in range(4):
                    self.session.pipeline.og.sim.render()
                self.session.invalidate_camera_cache()
                observation = source.capture(action_boundary=True).observation
                self._append(observation, frames)
                return {"snapshot_id": observation.snapshot_id, "delivered_to_policy": False}

            event_receipt = self.events.apply_preview(self.parameters["prototype"], capture)
            if event_receipt.get("status") != "applied":
                raise RuntimeError("Required candidate prelude event was not applied: " + str(event_receipt.get("reason")))
            self.state["episode_prelude"]["authored_event_receipt"] = deepcopy(event_receipt)
            # The final canonical capture happens after the event's own finally
            # block restores the tool, rather than reusing a waypoint image.
            for _ in range(4):
                self.session.pipeline.og.sim.render()
            self.session.invalidate_camera_cache()
            final = source.capture(action_boundary=True).observation
            self._append(final, frames)
            payload = {"schema": SCHEMA, "frames": frames, "authored_intervention": True,
                       "robot_actions": 0, "model_calls": 0}
            self.payload_signature = _signature(payload)
            self.session.candidate_public_history = payload
            self.final_observation, self.final_boundary = final, _boundary(final)
            self.run_status = "ready"
            self.state["episode_prelude"].update(status="ready", final_snapshot_id=final.snapshot_id,
                                                history_delivered_to_policy=False)
            return final
        except BaseException as error:
            self.run_status = "failed"
            self.state["episode_prelude"].update(status="failed", error=f"{type(error).__name__}: {error}")
            self.session.candidate_public_history = None
            raise
        finally:
            if previous_hook is _ABSENT:
                del self.session.candidate_observation_hook
            else:
                self.session.candidate_observation_hook = previous_hook

    def mark_delivered(self):
        """Acknowledge the owner's actual public bundle delivery without physics."""
        if self.state is not self.session.candidate_state:
            raise RuntimeError("Candidate history belongs to a previous native scene reset")
        if self.run_status != "ready":
            raise RuntimeError("Candidate history cannot be delivered before a completed prelude")
        if _signature(getattr(self.session, "candidate_public_history", None)) != self.payload_signature:
            raise RuntimeError("Candidate public history changed before delivery acknowledgement")
        if self.delivered:
            return {**deepcopy(self.delivery_receipt), "status": "already_delivered"}
        tags_before = set(self.state["observed"])
        receipts = []
        previous_time = self.events.control.pop("last_delivered_time_s", _ABSENT)
        try:
            for record in self.private_records:
                frame, evidence = record["frame"], record["visual_evidence"]
                for tag in evidence["observed_tags"]:
                    if tag not in self.state["observed"]:
                        self.state["observed"].append(tag)
                if self.parameters["prototype"] == "M1":
                    receipts.append(self.events.on_observation_delivered(frame, set(evidence["visible_names"])))
                frame["observed"] = deepcopy(self.state["observed"])
        finally:
            current = self.events.control.get("last_delivered_time_s", _ABSENT)
            if previous_time is not _ABSENT:
                self.events.control["last_delivered_time_s"] = previous_time if current is _ABSENT else max(previous_time, current)
        self.delivered = True
        self.state["episode_prelude"].update(status="delivered", history_delivered_to_policy=True)
        self.delivery_receipt = {"status": "delivered", "frame_count": len(self.private_records),
            "history_delivered_to_policy": True, "observed_tags_added": sorted(set(self.state["observed"]) - tags_before),
            "event_observation_receipts": receipts, "robot_actions": 0, "model_calls": 0}
        return deepcopy(self.delivery_receipt)
