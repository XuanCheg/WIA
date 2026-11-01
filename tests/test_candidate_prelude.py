"""Deferred native history delivery. Run: PYTHONPATH=src pytest -q tests/test_candidate_prelude.py."""

import base64
from copy import deepcopy
import hashlib
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from wia.candidates.native import CandidateEvaluator, install_observation_hook
import wia.candidates.prelude as prelude_module


def observation(snapshot_id):
    rgb = np.full((8, 8, 3), snapshot_id, dtype=np.uint8)
    return SimpleNamespace(snapshot_id=snapshot_id, action_step=0,
        captured_at=f"2026-10-30T00:00:00.{snapshot_id:06d}+00:00",
        rgb_images=lambda: {"agentview_rgb": rgb, "wrist_rgb": np.full_like(rgb, 255)},
        agentview_rgb=rgb)


class Source:
    def __init__(self, session):
        self.session = session
        self.snapshot_id = 1
        self.latest_snapshot = SimpleNamespace(observation=observation(1))

    def capture(self, *, action_boundary):
        assert action_boundary is True
        self.session.calls.append(("capture", self.session.phase))
        self.snapshot_id += 1
        self.latest_snapshot = SimpleNamespace(observation=observation(self.snapshot_id))
        hook = getattr(self.session, "candidate_observation_hook", None)
        if callable(hook):
            hook(self.latest_snapshot.observation)
        return self.latest_snapshot


class Evaluator:
    def __init__(self, session):
        self.session = session
        self.prelude_frames = []

    def measure(self):
        return {"time_s": self.session.time, "phase_private": self.session.phase,
                "initial_stock_count": 2 if self.session.phase == "initial" else 1,
                "objects": {"private_item_id": {"native_id": "/private/actor"}},
                "regions": {}, "observed": deepcopy(self.session.candidate_state["observed"]),
                "events": deepcopy(self.session.candidate_state["events"])}


class Visual:
    def __init__(self, session, parameters):
        self.session = session

    def record(self, current, frame):
        # Optional cameras are not part of the delivered history protocol.
        assert set(current.rgb_images()) == {"agentview_rgb"}
        np.testing.assert_array_equal(current.rgb_images()["agentview_rgb"],
                                      np.full((8, 8, 3), current.snapshot_id, dtype=np.uint8))
        self.session.calls.append(("head_evidence", current.snapshot_id))
        return {"status": "complete", "observed_tags": [],
                "visible_names": ["private_item_id", "zone_cleaning_surface"] if frame["phase_private"] == "contact" else [],
                "private_answer": "never public"}


class Events:
    def __init__(self, session, parameters):
        self.session, self.parameters = session, parameters
        self.control = session.candidate_state.setdefault("event_controller", {})

    def apply_preview(self, kind, callback):
        self.session.calls.append(("author", kind))
        callback("private_item_id_before_condition")
        self.session.time = .05
        self.session.phase = "contact"
        self.session.candidate_state["events"].append({
            "type": "cleaning_contact" if kind == "M1" else "external_removal",
            "object": "private_item_id", "time_s": .05, "authored": True, "contact_valid": True})
        callback("private_condition_contact")
        if getattr(self.session, "fail_event", False):
            raise RuntimeError("authored event failed")
        self.session.time = .1
        self.session.phase = "return_waypoint"
        callback("private_item_id_after_condition")
        self.session.phase = "after_finally_restore"
        return {"status": "applied", "events": deepcopy(self.session.candidate_state["events"])}

    def on_observation_delivered(self, frame, visible_names):
        self.session.calls.append(("delivered_event_view", frame["time_s"]))
        if "private_item_id" in visible_names:
            if "use:private_item_id" not in self.session.candidate_state["observed"]:
                self.session.candidate_state["observed"].append("use:private_item_id")
        return {"status": "observed", "observed_tags": deepcopy(self.session.candidate_state["observed"])}


@pytest.fixture
def setup(monkeypatch):
    session = SimpleNamespace(time=0., phase="initial", calls=[],
        candidate_state={"observed": [], "events": []})
    sim = SimpleNamespace(render=lambda: session.calls.append(("render", None)))
    session.pipeline = SimpleNamespace(og=SimpleNamespace(sim=sim))
    session.invalidate_camera_cache = lambda: session.calls.append(("invalidate", None))
    session.candidate_observation_hook = lambda current: (_ for _ in ()).throw(
        AssertionError("Prelude capture must suppress the regular observation hook"))
    monkeypatch.setattr(prelude_module, "CandidateEvents", Events)
    monkeypatch.setattr(prelude_module, "CandidateObservationEvidence", Visual)
    parameters = {"prototype": "M1", "contract": {"used_tool": "private_item_id", "condition": "private_answer"}}
    evaluator = Evaluator(session)
    source = Source(session)
    prelude = prelude_module.CandidatePrelude(session, parameters, evaluator)
    return session, source, evaluator, prelude


def test_capture_sequence_retains_initial_contact_and_post_restore_head_frames(setup):
    session, source, evaluator, prelude = setup
    old_hook = session.candidate_observation_hook
    initial = source.latest_snapshot.observation
    final = prelude.run(source, initial)
    assert final is source.latest_snapshot.observation
    assert session.candidate_observation_hook is old_hook
    assert evaluator.prelude_frames[0]["phase_private"] == "initial"
    assert evaluator.prelude_frames[0]["events"] == []
    assert evaluator.prelude_frames[-1]["phase_private"] == "after_finally_restore"
    assert any(frame["phase_private"] == "contact" for frame in evaluator.prelude_frames)
    frames = session.candidate_public_history["frames"]
    assert [frame["snapshot_id"] for frame in frames] == [1, 2, 3, 4, 5]
    assert len(frames) <= 24
    assert session.calls.count(("render", None)) == 16
    assert session.candidate_state["observed"] == []
    assert not any(kind == "delivered_event_view" for kind, _ in session.calls)


def test_public_payload_has_exact_neutral_schema_and_lossless_png_sha(setup):
    session, source, _, prelude = setup
    prelude.run(source, source.latest_snapshot.observation)
    payload = session.candidate_public_history
    assert set(payload) == {"schema", "frames", "authored_intervention", "robot_actions", "model_calls"}
    assert payload["schema"] == "wia_candidate_observation_history_v1"
    assert payload["authored_intervention"] is True
    assert payload["robot_actions"] == payload["model_calls"] == 0
    for index, frame in enumerate(payload["frames"]):
        assert set(frame) == {"index", "snapshot_id", "captured_at", "simulation_time_s", "camera", "png_base64", "sha256"}
        assert frame["index"] == index and frame["camera"] == "agentview_rgb"
        png = base64.b64decode(frame["png_base64"], validate=True)
        assert hashlib.sha256(png).hexdigest() == frame["sha256"]
        image = Image.open(BytesIO(png))
        assert image.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(image), np.full((8, 8, 3), frame["snapshot_id"], dtype=np.uint8))
        assert frame["captured_at"] == observation(frame["snapshot_id"]).captured_at


def test_delivery_credit_is_deferred_idempotent_and_uses_saved_head_evidence(setup):
    session, source, evaluator, prelude = setup
    with pytest.raises(RuntimeError, match="before a completed"):
        prelude.mark_delivered()
    prelude.run(source, source.latest_snapshot.observation)
    calls_before = list(session.calls)
    assert session.candidate_state["observed"] == []
    receipt = prelude.mark_delivered()
    assert receipt["status"] == "delivered"
    assert session.candidate_state["observed"] == ["use:private_item_id"]
    assert evaluator.prelude_frames[0]["observed"] == []
    assert evaluator.prelude_frames[2]["observed"] == ["use:private_item_id"]
    added = session.calls[len(calls_before):]
    assert added and all(kind == "delivered_event_view" for kind, _ in added)
    count = len(session.calls)
    assert prelude.mark_delivered()["status"] == "already_delivered"
    assert len(session.calls) == count


def test_hook_is_restored_after_event_failure_and_reauthoring_is_refused(setup):
    session, source, _, prelude = setup
    old_hook = session.candidate_observation_hook
    session.fail_event = True
    initial = source.latest_snapshot.observation
    with pytest.raises(RuntimeError, match="authored event failed"):
        prelude.run(source, initial)
    assert session.candidate_observation_hook is old_hook
    assert session.candidate_public_history is None
    assert session.candidate_state["observed"] == []
    with pytest.raises(RuntimeError, match="already authored or failed"):
        prelude.run(source, initial)


def test_resume_does_not_double_author_and_new_boundary_is_a_replay_error(setup):
    session, source, _, prelude = setup
    initial = source.latest_snapshot.observation
    final = prelude.run(source, initial)
    count = len(session.calls)
    assert prelude.run(source, initial) is final
    assert prelude.run(source, final) is final
    assert len(session.calls) == count
    with pytest.raises(RuntimeError, match="replay attempted"):
        prelude.run(source, observation(99))


def test_changed_public_payload_cannot_receive_delivery_credit(setup):
    session, source, _, prelude = setup
    prelude.run(source, source.latest_snapshot.observation)
    session.candidate_public_history["frames"][0]["snapshot_id"] = 999
    with pytest.raises(RuntimeError, match="changed before delivery"):
        prelude.mark_delivered()
    assert session.candidate_state["observed"] == []


def test_h3_preserves_real_before_removal_inventory_and_never_adds_use_tags(setup):
    session, source, evaluator, _ = setup
    parameters = {"prototype": "H3", "contract": {"requires_revision_event": True}}
    prelude = prelude_module.CandidatePrelude(session, parameters, evaluator)
    prelude.run(source, source.latest_snapshot.observation)
    assert evaluator.prelude_frames[0]["initial_stock_count"] == 2
    assert evaluator.prelude_frames[0]["events"] == []
    assert evaluator.prelude_frames[2]["events"][0]["type"] == "external_removal"
    before = len(session.calls)
    assert prelude.mark_delivered()["history_delivered_to_policy"] is True
    assert len(session.calls) == before
    assert session.candidate_state["observed"] == []


def test_evaluator_begin_episode_seeds_prelude_and_deduplicates_equal_sim_times():
    evaluator = CandidateEvaluator.__new__(CandidateEvaluator)
    evaluator.prelude_frames = [{"time_s": 0., "inventory": 2}, {"time_s": 0., "inventory": 2},
                                {"time_s": .05, "inventory": 1}, {"time_s": .1, "inventory": 1}]
    evaluator.record = lambda current, action: deepcopy(evaluator.frames)
    seeded = evaluator.begin_episode(observation(5))
    assert [frame["time_s"] for frame in seeded] == [0., .05, .1]
    assert seeded[0]["inventory"] == 2
    assert evaluator.episode_started


def test_native_h3_initial_hook_has_no_early_event_or_undelivered_credit(monkeypatch):
    import wia.candidates.events as events_module
    import wia.candidates.observations as observations_module
    monkeypatch.setattr(events_module, "CandidateEvents", Events)

    class OptionalCameraVisual(Visual):
        def record(self, current, frame):
            return {"observed_tags": ["undelivered_optional_camera_tag"], "visible_names": ["private_item_id"]}

    monkeypatch.setattr(observations_module, "CandidateObservationEvidence", OptionalCameraVisual)
    session = SimpleNamespace(time=0., phase="initial", calls=[], candidate_state={"observed": [], "events": []})
    evaluator = Evaluator(session)
    parameters = {"prototype": "H3", "contract": {"requires_revision_event": True}}
    install_observation_hook(session, parameters, evaluator)
    session.candidate_observation_hook(observation(1))
    assert session.calls == []
    assert session.candidate_state["events"] == session.candidate_state["observed"] == []
    assert callable(session.candidate_episode_prelude)
    assert callable(session.candidate_public_history_delivery)


def test_h2_observation_hook_keeps_existing_event_boundary_path(monkeypatch):
    import wia.candidates.events as events_module
    import wia.candidates.observations as observations_module
    monkeypatch.setattr(events_module, "CandidateEvents", Events)

    class BoundaryVisual(Visual):
        def record(self, current, frame):
            return {"observed_tags": ["destination_state"], "visible_names": []}

    monkeypatch.setattr(observations_module, "CandidateObservationEvidence", BoundaryVisual)
    session = SimpleNamespace(time=0., phase="initial", calls=[], candidate_state={"observed": [], "events": []})
    install_observation_hook(session, {"prototype": "H2", "contract": {}}, Evaluator(session))
    session.candidate_observation_hook(observation(1))
    assert session.calls == [("delivered_event_view", 0.)]
    assert session.candidate_state["observed"] == ["destination_state"]
    assert not hasattr(session, "candidate_episode_prelude")


def test_reinstall_reuses_current_prelude_and_reset_requires_fresh_state(setup):
    session, source, evaluator, prelude = setup
    parameters = {"prototype": "M1", "contract": {"used_tool": "private_item_id"}}
    prelude.run(source, source.latest_snapshot.observation)
    session._candidate_prelude_controller = prelude
    install_observation_hook(session, parameters, evaluator)
    assert session._candidate_prelude_controller is prelude
    replacement = prelude_module.CandidatePrelude(session, parameters, evaluator)
    with pytest.raises(RuntimeError, match="already authored"):
        replacement.run(source, observation(99))
    session.candidate_state = {"observed": [], "events": []}
    with pytest.raises(RuntimeError, match="new candidate prelude"):
        prelude.run(source, observation(99))
    with pytest.raises(RuntimeError, match="previous native scene reset"):
        prelude.mark_delivered()


def test_private_physics_ticks_fill_sparse_prelude_image_gaps(setup):
    session, source, evaluator, prelude = setup
    def apply_preview(kind, capture):
        for tick in range(1, 5):
            session.time = tick / 30
            prelude.events.physics_sample_callback()
        capture('after_native_steps')
        return {'status': 'applied'}
    prelude.events.apply_preview = apply_preview
    prelude.run(source, source.latest_snapshot.observation)
    times = sorted({f['time_s'] for f in evaluator.prelude_frames})
    assert len(times) == 5 and max(b-a for a,b in zip(times,times[1:])) <= 1/30 + 1e-9
    assert len(session.candidate_public_history['frames']) == 3
    assert prelude.events.physics_sample_callback is None
