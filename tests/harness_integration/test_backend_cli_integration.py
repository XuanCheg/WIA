"""Validate backend CLI forwarding without a model/GPU. Run: pytest tests/test_backend_cli_integration.py."""
import argparse
import json

import pytest

import wia.runtime.commands.run_esi_agent as agent
import wia.runtime.commands.run_esi_explore_episode as launch
from wia.integrations.show_harness import build_arguments


def options(backend, *extra):
    parser = argparse.ArgumentParser()
    agent.add_backend_arguments(parser)
    args = parser.parse_args(["--backend", backend, *extra])
    args.model = "explicit-model"
    args.model_timeout = 30
    args.snapshot_cache = "optimized"
    args.reasoning_effort = None
    args.reasoning_summary = "concise"
    args.codex_executable = "codex"
    return args


@pytest.mark.parametrize("name,session", [("codex", "native_persistent"), ("codex_cli", "native_persistent"),
    ("local_http", "local_http_persistent"), ("openai_responses", "openai_responses_persistent"),
    ("openai_chat_completions", "openai_chat_completions_persistent")])
def test_manifest_session_identity(name, session):
    assert agent.session_policy(name) == session


def test_chat_generation_forwarded_and_no_codex_parameters(monkeypatch):
    import core.agent.backend_registry as registry
    calls = []
    monkeypatch.setattr(registry, "create_session", lambda backend, **kwargs: calls.append((backend, kwargs)))
    args = options("openai_chat_completions", "--provider", "vllm", "--base-url", "http://localhost:8000/v1",
                   "--api-key-env", "TEST_API_KEY", "--max-output-tokens", "77", "--temperature", "0.3",
                   "--chat-template-kwargs", '{"enable_thinking":false}')
    monkeypatch.setenv("TEST_API_KEY", "never-persist")
    agent.validate_backend_arguments(args)
    agent.create_declarative_session(args, public_task={}, policy={}, output="unused", assembler=None)
    backend, kwargs = calls[0]
    assert backend == "openai_chat_completions"
    assert kwargs["api_key"] == "never-persist"
    assert kwargs["max_output_tokens"] == 77
    assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    assert "executable" not in kwargs and "resume_state" not in kwargs
    assert "never-persist" not in json.dumps(agent.backend_metadata(args))
    reparsed = options("codex", *agent.backend_cli_arguments(args))
    assert reparsed.provider == "vllm"


@pytest.mark.parametrize("provider", ["vllm", "compatible"])
def test_non_openai_provider_requires_an_explicit_endpoint(provider):
    with pytest.raises(ValueError, match="require --base-url"):
        agent.validate_backend_arguments(options("openai_chat_completions", "--provider", provider))


@pytest.mark.parametrize("provider", ["vllm", "compatible"])
@pytest.mark.parametrize("url_flag", ["--base-url", "--local-base-url"])
def test_non_openai_endpoint_is_forwarded_unchanged(monkeypatch, provider, url_flag):
    import core.agent.backend_registry as registry
    captured = {}
    monkeypatch.setattr(registry, "create_session", lambda backend, **kwargs: captured.update(kwargs))
    url = "http://127.0.0.1:8000/v1"
    args = options("openai_chat_completions", "--provider", provider, url_flag, url)
    agent.validate_backend_arguments(args)
    agent.create_declarative_session(args, public_task={}, policy={}, output="unused", assembler=None)
    assert captured["base_url"] == agent.backend_metadata(args)["base_url"] == url


@pytest.mark.parametrize("provider", ["vllm", "compatible"])
def test_launcher_rejects_missing_provider_endpoint_before_loading_scene(tmp_path, capsys, provider):
    with pytest.raises(SystemExit) as error:
        launch.main(["--task-file", str(tmp_path / "not-loaded.yaml"), "--output-dir", str(tmp_path / "scene"),
                     "--backend", "openai_chat_completions", "--provider", provider, "--model", "explicit"])
    assert error.value.code == 2
    assert "require --base-url" in capsys.readouterr().err
    assert not (tmp_path / "scene").exists()


@pytest.mark.parametrize("backend,extra,expected", [
    ("openai_chat_completions", [], "OPENAI_API_KEY"),
    ("openai_responses", [], "OPENAI_API_KEY"),
    ("openai_chat_completions", ["--provider", "vllm"], "VLLM_API_KEY"),
    ("openai_chat_completions", ["--provider", "compatible"], "LOCAL_AGENT_API_KEY"),
    ("local_http", [], "LOCAL_AGENT_API_KEY"),
])
def test_provider_default_credentials_match_metadata(monkeypatch, backend, extra, expected):
    import core.agent.backend_registry as registry
    captured = {}
    monkeypatch.setattr(registry, "create_session", lambda backend, **kwargs: captured.update(kwargs))
    for name in ("OPENAI_API_KEY", "VLLM_API_KEY", "LOCAL_AGENT_API_KEY", "CUSTOM_KEY"):
        monkeypatch.setenv(name, name + "_VALUE")
    args = options(backend, *extra, "--base-url", "http://127.0.0.1:8000/v1")
    agent.validate_backend_arguments(args)
    agent.create_declarative_session(args, public_task={}, policy={}, output="unused", assembler=None)
    assert agent.backend_metadata(args)["api_key_env"] == expected
    assert captured["api_key"] == expected + "_VALUE"
    args.api_key_env = "CUSTOM_KEY"
    agent.create_declarative_session(args, public_task={}, policy={}, output="unused", assembler=None)
    assert agent.backend_metadata(args)["api_key_env"] == "CUSTOM_KEY"
    assert captured["api_key"] == "CUSTOM_KEY_VALUE"
    assert "CUSTOM_KEY_VALUE" not in json.dumps(agent.backend_metadata(args))


@pytest.mark.parametrize("backend", ["openai_chat_completions", "openai_responses"])
def test_openai_provider_can_omit_endpoint(monkeypatch, backend):
    import core.agent.backend_registry as registry
    captured = {}
    monkeypatch.setattr(registry, "create_session", lambda backend, **kwargs: captured.update(kwargs))
    args = options(backend)
    agent.validate_backend_arguments(args)
    agent.create_declarative_session(args, public_task={}, policy={}, output="unused", assembler=None)
    assert captured["base_url"] == "https://api.openai.com/v1"


@pytest.mark.parametrize("backend", ["codex", "codex_cli"])
def test_native_metadata_does_not_advertise_http_configuration(backend):
    metadata = agent.backend_metadata(options(backend))
    assert metadata["backend"] == "codex_cli" and metadata["history_owner"] == "native"
    assert all(metadata[field] is None for field in ("api_key_env", "provider", "base_url",
               "max_output_tokens", "temperature", "chat_template_kwargs"))


def test_local_does_not_receive_reasoning(monkeypatch):
    import core.agent.backend_registry as registry
    captured = {}
    monkeypatch.setattr(registry, "create_session", lambda backend, **kwargs: captured.update(kwargs))
    args = options("local_http", "--base-url", "http://localhost:8000/v1")
    args.reasoning_effort = "high"
    agent.create_declarative_session(args, public_task={}, policy={}, output="unused", assembler=None)
    assert "reasoning_effort" not in captured


def test_codex_rejects_http_url():
    with pytest.raises(ValueError, match="do not configure"):
        agent.validate_backend_arguments(options("codex_cli", "--base-url", "http://localhost/v1"))


@pytest.mark.parametrize("module", [agent, launch])
def test_help_works_without_mcp(module):
    with pytest.raises(SystemExit) as error:
        module.main(["--help"])
    assert error.value.code == 0


def test_launch_and_attach_budget_arguments():
    plan = {"task": {"path": "/task.yaml", "seed": 3, "variant": "original"},
            "policy": {"backend": "openai_responses", "model": "explicit", "reasoning_effort": "low"},
            "environment": {"mode": "launch", "gpu": 2},
            "evaluation": {"max_steps": 7, "timeout": None, "world": "pause"}, "output_dir": "/fresh"}
    argv = build_arguments(plan)
    assert argv[argv.index("--timeout") + 1] == "none"
    assert "--backend" in argv and "--gpu" in argv and "--world" in argv
    plan["environment"] = {"mode": "attach", "url": "http://existing", "episode_manifest": "/manifest.json"}
    argv = build_arguments(plan)
    assert "--episode-manifest" in argv and "--world-during-thinking" in argv
    assert "--timeout" not in argv and "--gpu" not in argv


def test_standalone_harness_namespace_and_registry_without_wia():
    import subprocess
    import sys
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'wia':
            raise AssertionError('Standalone Harness loaded WIA: ' + fullname)
sys.meta_path.insert(0, Block())
import showharness
from core.agent.backend_registry import available_backends
assert 'codex_cli' in available_backends()
from showharness.datasets import evaluate_dataset
class Adapter:
    def execute(self, plan): return {'result': plan['value']}
assert evaluate_dataset(Adapter(), {'value': 7}) == {'result': 7}
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", ["codex", "codex_cli", "local_http", "openai_chat_completions", "openai_responses"])
def test_actual_launch_manifest_passes_authoritative_loader(tmp_path, backend):
    from harness_integration.task_examples import load_example_task
    from wia.tasks.loader import verify_manifest
    task = load_example_task("find_and_lift")
    extra = ["--base-url", "http://localhost:8000/v1"] if backend == "local_http" else []
    args = options(backend, *extra)
    args.max_steps = task.definition.defaults.max_steps
    args.world = "run"
    args.observation_condition = "current"
    args.action_mode = "parameterized"
    args.timeout = task.definition.defaults.wall_timeout_s
    args.reasoning_effort = "medium" if agent.canonical_backend(backend) == "codex_cli" else None
    args.world_observation_log = True
    manifest = launch.build_episode_manifest(task, args, episode_id="real-manifest-contract")
    path = tmp_path / "episode_manifest.json"
    path.write_text(json.dumps(manifest))
    assert verify_manifest(path, task) == manifest
    assert manifest["backend_metadata"]["backend"] == agent.canonical_backend(backend)
    manifest["backend_metadata"]["history_owner"] = "invalid"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):
        verify_manifest(path, task)


def test_public_adapter_reads_single_authoritative_summary(tmp_path, monkeypatch):
    import wia.integrations.show_harness as evaluation
    import wia.runtime.commands.run_esi_agent as runner
    calls = []
    output = tmp_path / "episode"
    summary = {"termination_reason": "max_steps", "agent_decisions": 4, "executed_actions": ["one"],
               "model_calls": 4, "task_success": False, "evaluation_validity": "complete",
               "timings": {"episode_s": 12.}, "usage": {"input_tokens": 100}}
    def execute(argv):
        calls.append(argv)
        output.mkdir()
        (output / "summary.json").write_text(json.dumps(summary))
        return 0
    monkeypatch.setattr(runner, "main", execute)
    plan = {"task": {"path": "/task.yaml", "seed": 3, "variant": "original"},
            "policy": {"backend": "openai_responses", "model": "explicit"},
            "environment": {"mode": "attach", "url": "http://existing", "episode_manifest": "/manifest.json"},
            "evaluation": {"max_steps": 4}, "output_dir": str(output)}
    returned = evaluation.execute_plan(plan)
    assert len(calls) == 1
    assert returned["agent_decisions"] == 4 and returned["executed_actions"] == ["one"]
    assert returned["usage"] == {"input_tokens": 100}
    assert json.loads((output / "summary.json").read_text()) == summary


def test_runner_argument_failure_is_aggregatable(tmp_path, monkeypatch):
    import wia.integrations.show_harness as evaluation
    import wia.runtime.commands.run_esi_agent as runner
    def reject(argv):
        raise SystemExit(2)
    monkeypatch.setattr(runner, "main", reject)
    plan = {"task": {"path": "/task.yaml", "seed": 3, "variant": "original"},
            "policy": {"backend": "openai_responses", "model": "explicit"},
            "environment": {"mode": "attach", "url": "http://existing", "episode_manifest": "/manifest.json"},
            "evaluation": {}, "output_dir": str(tmp_path / "fresh")}
    with pytest.raises(RuntimeError, match="argument validation"):
        evaluation.execute_plan(plan)
