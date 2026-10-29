"""Check native thread-resume startup timeout without a model or simulator.

Run: behavior/bin/python -B -m pytest -q tests/test_native_session_resume_timeout.py
"""

import asyncio
from types import SimpleNamespace

import pytest

import core.agent.codex_session as sessions
import core.agent.native_audit as audit
from core.agent.decision import ModelError
from harness_integration.test_milestone85_prompts import request_inputs


@pytest.fixture
def resume_session(tmp_path, monkeypatch):
    task, request, policy = request_inputs()
    previous = sessions.CodexEpisodeSession(public_task=task.public_payload,
        episode_configuration=policy, output=tmp_path / "previous")
    state = previous.provenance()
    state.update(codex_session_id="retained-native-session", codex_version="codex-cli fixture",
                 session_start_time="2026-10-26T10:00:00+00:00")
    catalog = tmp_path / "catalog.json"
    catalog.write_text('{"models": []}\n')
    monkeypatch.setattr(sessions.shutil, "which", lambda _: "/fixture/codex")
    monkeypatch.setattr(sessions.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="codex-cli fixture"))
    monkeypatch.setattr(sessions, "depth_access", lambda _: {"available": False})
    monkeypatch.setattr(sessions, "isolated_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(sessions, "write_function_tool_profile", lambda *args: (catalog, {}))

    async def passed_audit(*args):
        return {"status": "passed"}

    monkeypatch.setattr(audit, "audit_native_tools", passed_audit)
    calls = []

    class RPC:
        timed_out = False

        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            pass

        async def request(self, method, params, **kwargs):
            calls.append((method, params, kwargs))
            if self.timed_out:
                raise asyncio.TimeoutError("fixture startup timeout")
            return {"thread": {"id": "retained-native-session"},
                    "model": previous.model, "instructionSources": []}

        async def close(self):
            pass

    monkeypatch.setattr(sessions, "NativeRPC", RPC)

    def build(timeout=600):
        agent = sessions.CodexEpisodeSession(public_task=task.public_payload,
            episode_configuration=policy, output=tmp_path / "resumed", timeout_s=timeout,
            resume_state=state)
        return agent, request, calls, RPC

    return build


@pytest.mark.parametrize("timeout", [None, 600, 17])
def test_resume_forwards_configured_timeout_and_retains_same_thread(resume_session, timeout):
    agent, request, calls, _ = resume_session(timeout)

    async def exercise():
        try:
            await agent._open(request)
            assert agent.session_id == "retained-native-session"
            assert agent.timeout_s == timeout
        finally:
            await agent.close()

    asyncio.run(exercise())
    assert len(calls) == 1
    assert calls[0][0] == "thread/resume"
    assert calls[0][1]["threadId"] == "retained-native-session"
    assert calls[0][2] == {"timeout": timeout}


def test_resume_startup_timeout_is_explicit_model_error_without_turn_or_retry(resume_session):
    agent, request, calls, rpc = resume_session()
    rpc.timed_out = True

    async def exercise():
        try:
            with pytest.raises(ModelError, match="thread/resume.*before any model turn started") as error:
                await agent._open(request)
            assert isinstance(error.value.__cause__, asyncio.TimeoutError)
            assert agent.session_id is None
        finally:
            await agent.close()

    asyncio.run(exercise())
    assert len(calls) == 1
    assert calls[0][0] == "thread/resume"
    assert calls[0][2] == {"timeout": 600}
