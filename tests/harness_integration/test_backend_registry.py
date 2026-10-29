"""Verify backend separation and extension. Run: python -m pytest tests/test_backend_registry.py."""
import pytest

from showharness.backends import BackendCapabilities, available_backends, create_session, register_backend
from core.agent.codex_session import CodexEpisodeSession
from core.agent.local_http_session import LocalHTTPEpisodeSession
from core.agent.api_sessions import ChatCompletionsEpisodeSession, ResponsesEpisodeSession
from harness_integration.test_local_http_episode import local_episode


def test_builtins_remain_distinct_and_legacy_factories_unchanged(tmp_path):
    _, original = local_episode(tmp_path)
    shared = dict(public_task=original.public_task, episode_configuration=original.episode_configuration,
                  output=tmp_path, assembler=original.assembler, model="same-model", timeout_s=90,
                  snapshot_cache=False)
    for name in ("codex", "codex_cli"):
        session = create_session(name, **shared)
        assert type(session) is CodexEpisodeSession
        assert not hasattr(session, "history")
        assert session.model == "same-model" and session.timeout_s == 90
        assert available_backends()[name].history_owner == "native"
    for name, expected in (("local_http", LocalHTTPEpisodeSession),
        ("openai_chat_completions", ChatCompletionsEpisodeSession),
        ("openai_responses", ResponsesEpisodeSession)):
        session = create_session(name, **shared, base_url="http://localhost:8000/v1")
        assert type(session) is expected and session.model == "same-model"
        assert not session.snapshot_cache and session.assembler is original.assembler
        assert available_backends()[name].history_owner == "harness"


def test_external_factory_receives_all_arguments_unchanged(monkeypatch):
    from core.agent import backend_registry
    monkeypatch.setattr(backend_registry, "_FACTORIES", dict(backend_registry._FACTORIES))
    received = []
    marker = object()

    def factory(**kwargs):
        received.append(kwargs)
        return marker

    capabilities = BackendCapabilities(history_owner="harness", tools=False)
    register_backend("extension", factory, capabilities)
    assert create_session("extension", arbitrary=marker) is marker
    assert received == [{"arbitrary": marker}]
    exported = available_backends()
    assert exported["extension"] is capabilities
    exported.clear()
    assert "extension" in available_backends()
    with pytest.raises(ValueError, match="already registered"):
        register_backend("extension", factory, capabilities)
    with pytest.raises(ValueError, match="Unknown backend"):
        create_session("missing")
