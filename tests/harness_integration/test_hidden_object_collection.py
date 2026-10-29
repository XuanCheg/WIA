"""Check physical collection accounting and simulation-time terminal dwell.

Run: python -B -m pytest -q tests/test_hidden_object_collection.py
These fixtures validate scoring, not native grasp or plate feasibility.
"""
from types import SimpleNamespace

from wia.authoring.converters import hidden_object_collection as collection


def state(**changes):
    return {"supported_objects": ["a", "b"], "held_objects": ["plate"],
        "right_holds_plate": True, "plate_level": True, "plate_stable": True,
        "left_empty": True, "left_holding": None, "native_time_s": 0., **changes}


def score(value, picked=("a", "b")):
    return collection.collection_result(value, ["a", "b"], ["d"], set(picked),
        stable_seconds=1., required_dwell=.5)


def test_only_distinct_released_picked_targets_on_plate_count():
    assert score(state())["task_success"]
    for value in (state(supported_objects=["a", "a"]),
                  state(held_objects=["plate", "b"], left_empty=False),
                  state(supported_objects=["a", "b", "d"]),
                  state(right_holds_plate=False)):
        assert not score(value)["task_success"]
    assert score(state(), picked=("a",))["collection_recall"] == .5


def test_spilled_targets_lose_credit_and_recovery_is_allowed():
    assert score(state())["task_success"]
    assert score(state(supported_objects=["a"]))["collection_recall"] == .5
    assert score(state())["task_success"]


def test_dwell_uses_native_time_and_resets_after_loss(monkeypatch, tmp_path):
    current = state(supported_objects=[], native_time_s=0.)
    monkeypatch.setattr(collection, "_state", lambda *args: dict(current))
    evaluator = collection.CollectionEvaluator(None, None, {"parameters": {
        "targets": ["a", "b"], "distractors": ["d"], "terminal_dwell_s": .5}}, tmp_path)
    observation = SimpleNamespace(snapshot_id="s", action_step=0)
    evaluator.begin_episode(observation)
    for t, name in ((.1, "a"), (.2, "b")):
        current.update(native_time_s=t, left_holding=name, left_empty=False)
        evaluator.sample_step()
    current.update(native_time_s=.3, left_holding=None, left_empty=True,
                   supported_objects=["a", "b"])
    for _ in range(10):
        assert not evaluator.record(observation)["task_success"]
    current["native_time_s"] = .9
    assert evaluator.record(observation)["task_success"]
    current.update(native_time_s=1., supported_objects=["a"])
    assert not evaluator.record(observation)["task_success"]
    current.update(native_time_s=1.1, supported_objects=["a", "b"])
    assert not evaluator.record(observation)["task_success"]
    current["native_time_s"] = 1.7
    assert evaluator.record(observation)["task_success"]
    assert evaluator.spills == [{"target": "b", "native_time_s": 1.}]
