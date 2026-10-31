"""Exercise orchestration without scene/model runtime. Run: pytest tests/test_evaluation_entry.py."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from wia.evaluation import prepare, run
from wia.evaluation.runner import summarize
from wia.policies import register_policy


def config(tmp_path, monkeypatch, **overrides):
    import wia.tasks.loader as loader
    task = tmp_path / "task.yaml"
    task.write_text("id: fixture\n")
    defaults = {"seed": 5, "max_steps": 17, "wall_timeout_s": 123., "action_mode": "parameterized",
                "world_advances_during_inference": False, "observation_condition": "current"}
    monkeypatch.setattr(loader, "task_definition", lambda _: SimpleNamespace(defaults=SimpleNamespace(
        model_dump=lambda **kwargs: defaults)))
    value = {"tasks": [{"path": "task.yaml"}], "policy": {"type": "show_harness", "backend": "codex_cli", "model": "explicit"},
             "environment": {"mode": "launch"}, "output_dir": "results"}
    value.update(overrides)
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(value))
    return path


def test_dry_run_preserves_task_defaults_and_does_not_write(tmp_path, monkeypatch):
    prepared = prepare(config(tmp_path, monkeypatch))
    assert prepared["plans"][0]["evaluation"]["max_steps"] == 17
    assert prepared["plans"][0]["evaluation"]["world"] == "pause"
    assert prepared["plans"][0]["task"]["seed"] == 5
    assert not (tmp_path / "results").exists()


def test_bounded_history_is_diagnostic_only(tmp_path, monkeypatch):
    policy = {'type': 'show_harness', 'backend': 'openai_chat_completions',
              'model': 'explicit', 'history_text_turns': 2}
    path = config(tmp_path, monkeypatch, policy=policy)
    with pytest.raises(ValueError, match='history_text_turns requires diagnostic'):
        prepare(path)
    path = config(tmp_path, monkeypatch, policy=policy, evaluation={'run_mode': 'diagnostic'})
    assert prepare(path)['plans'][0]['policy']['history_text_turns'] == 2


def test_json_schema_output_is_diagnostic_only(tmp_path, monkeypatch):
    policy = {'type': 'show_harness', 'backend': 'openai_chat_completions',
              'model': 'explicit', 'json_schema_output': True}
    with pytest.raises(ValueError, match='json_schema_output requires diagnostic'):
        prepare(config(tmp_path, monkeypatch, policy=policy))
    prepared = prepare(config(tmp_path, monkeypatch, policy=policy, evaluation={'run_mode': 'diagnostic'}))
    assert prepared['plans'][0]['policy']['json_schema_output'] is True


@pytest.mark.parametrize("provider", ["vllm", "compatible"])
def test_non_openai_launch_requires_explicit_endpoint(tmp_path, monkeypatch, provider):
    policy = {"type": "show_harness", "backend": "openai_chat_completions",
              "model": "explicit", "provider": provider}
    path = config(tmp_path, monkeypatch, policy=policy)
    with pytest.raises(ValueError, match="explicit base_url"):
        prepare(path)
    assert not (tmp_path / "results").exists()
    policy["base_url"] = "http://127.0.0.1:8000/v1"
    path = config(tmp_path, monkeypatch, policy=policy)
    assert prepare(path)["plans"][0]["policy"]["base_url"] == policy["base_url"]


@pytest.mark.parametrize("backend", ["openai_chat_completions", "openai_responses"])
def test_openai_launch_can_use_default_endpoint(tmp_path, monkeypatch, backend):
    path = config(tmp_path, monkeypatch, policy={"type": "show_harness", "backend": backend, "model": "explicit"})
    assert "base_url" not in prepare(path)["plans"][0]["policy"]


@pytest.mark.parametrize("provider", ["vllm", "compatible"])
@pytest.mark.parametrize("metadata_url", [None, "http://127.0.0.1:8000/v1"])
def test_attach_checks_provider_endpoint_after_manifest_inheritance(tmp_path, monkeypatch, provider, metadata_url):
    import hashlib
    import wia.tasks.loader as loader
    policy = {"type": "show_harness", "backend": "openai_chat_completions", "model": "explicit"}
    path = config(tmp_path, monkeypatch, policy=policy,
                  environment={"mode": "attach", "url": "http://existing", "episode_manifest": "manifest.json"})
    defaults = {"seed": 5, "max_steps": 17, "wall_timeout_s": 123., "action_mode": "parameterized",
                "world_advances_during_inference": False, "observation_condition": "current"}
    monkeypatch.setattr(loader, "task_definition", lambda _: SimpleNamespace(id="fixture", defaults=SimpleNamespace(
        model_dump=lambda **kwargs: defaults)))
    metadata = {"backend": "openai_chat_completions", "provider": provider}
    if metadata_url is not None:
        metadata["base_url"] = metadata_url
    manifest = {"task_id": "fixture", "task_file_sha256": hashlib.sha256((tmp_path / "task.yaml").read_bytes()).hexdigest(),
                "scene_seed": 5, "scene_variant": "original", "model_name": "explicit", "reasoning_effort": None,
                "agent_session": "openai_chat_completions_persistent", "backend_metadata": metadata,
                "world_observation_log": False, "max_steps": 17, "wall_timeout_s": 123.,
                "action_mode": "parameterized", "world_mode": "pause", "observation_condition": "current"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    if metadata_url is None:
        with pytest.raises(ValueError, match="prepared backend metadata"):
            prepare(path)
    else:
        plan = prepare(path)["plans"][0]
        assert plan["policy"]["base_url"] == metadata_url
        assert plan["policy"]["provider"] == provider
    assert not (tmp_path / "results").exists()


def test_model_required_and_secrets_rejected(tmp_path, monkeypatch):
    path = config(tmp_path, monkeypatch, policy={"type": "show_harness", "backend": "openai_responses"})
    with pytest.raises(ValueError, match="model"):
        prepare(path)
    path = config(tmp_path, monkeypatch, policy={"type": "show_harness", "backend": "openai_responses", "model": "explicit", "api_key": "secret"})
    with pytest.raises(ValueError, match="Unknown policy"):
        prepare(path)


def test_attach_multiple_rejects_ambiguous_scene_reuse(tmp_path, monkeypatch):
    path = config(tmp_path, monkeypatch, tasks=[{"path": "task.yaml"}, {"path": "task.yaml"}], environment={"mode": "attach"})
    with pytest.raises(ValueError, match="per-task"):
        prepare(path)


@pytest.mark.parametrize("summary,status", [
    ({"termination_reason": "max_steps", "task_success": False, "evaluation_validity": "complete"}, "completed_unsuccessful"),
    ({"termination_reason": "model_error", "task_success": False, "evaluation_validity": "complete"}, "model_error"),
    ({"termination_reason": "environment_error"}, "environment_error"),
    ({"termination_reason": "model_done", "task_success": True, "evaluation_validity": "partial"}, "invalid_or_incomplete_score")])
def test_result_classification(summary, status):
    assert summarize(summary)["status"] == status
    assert summarize(summary)["cost"] is None


@pytest.mark.parametrize("backend", ["codex", "openai_chat_completions", "openai_responses"])
def test_nested_persistent_session_usage_is_retained(backend):
    usage = {"input_tokens": 27, "output_tokens": 9, "output_tokens_details": {"reasoning_tokens": 4}}
    summary = {"native_session": {"backend": backend, "usage": usage}}
    assert summarize(summary)["usage"] == usage


@pytest.mark.parametrize("summary,expected", [
    ({"usage": {"total_tokens": 7}, "model_usage": {"total_tokens": 8},
      "native_session": {"usage": {"total_tokens": 9}}}, {"total_tokens": 7}),
    ({"model_usage": {"total_tokens": 8}, "native_session": {"usage": {"total_tokens": 9}}},
     {"total_tokens": 8}),
    ({"usage": None, "native_session": {"usage": {"total_tokens": 9}}}, None),
    ({}, None),
    ({"native_session": None}, None),
])
def test_usage_precedence_does_not_sum_or_invent_token_counts(summary, expected):
    assert summarize(summary)["usage"] == expected


def test_external_executor_runs_once_per_episode(tmp_path, monkeypatch):
    calls = []
    class Executor:
        def execute(self, plan):
            calls.append(plan)
            return {"termination_reason": "max_steps", "task_success": False, "evaluation_validity": "complete", "agent_decisions": 17}
    register_policy("test_external_executor", Executor)
    prepared = prepare(config(tmp_path, monkeypatch, policy={"type": "test_external_executor", "model": "explicit"}))
    results = run(prepared)
    assert len(calls) == 1 and results["counts"] == {"completed_unsuccessful": 1}
    assert results["trials"][0]["summary"]["agent_decisions"] == 17
    assert results["cost"] is None
    assert json.loads((tmp_path / "results/effective_config.json").read_text())["plans"] == prepared["plans"]
    with pytest.raises(FileExistsError):
        run(prepared)


def test_duplicate_config_keys_rejected(tmp_path, monkeypatch):
    path = config(tmp_path, monkeypatch)
    path.write_text(path.read_text() + '\noutput_dir: duplicate\n')
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(path)


def test_changed_task_refused_before_outputs(tmp_path, monkeypatch):
    prepared = prepare(config(tmp_path, monkeypatch))
    Path(prepared["plans"][0]["task"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="Task changed"):
        run(prepared)
    assert not (tmp_path / "results").exists()


def test_external_system_needs_no_model(tmp_path, monkeypatch):
    class Executor:
        def execute(self, plan):
            return {}
    register_policy("test_model_free_system", Executor)
    prepared = prepare(config(tmp_path, monkeypatch, policy={"type": "test_model_free_system"}))
    assert prepared["plans"][0]["policy"] == {"type": "test_model_free_system"}


def test_interrupted_execution_never_counts_success():
    base = {"termination_reason": "max_steps", "task_success": True, "evaluation_validity": "complete"}
    assert summarize({**base, "agent_failure_countable": False})["status"] == "invalid_or_incomplete_score"
    assert summarize({**base, "execution_validity": "interrupted"})["status"] == "invalid_or_incomplete_score"
    assert summarize({**base, "termination_reason": "execution_error"})["status"] == "execution_error"


def test_attach_uses_scene_manifest_fields_and_policy(tmp_path, monkeypatch):
    path = config(tmp_path, monkeypatch)
    import wia.tasks.loader as loader
    defaults = {"seed": 5, "max_steps": 17, "wall_timeout_s": 123., "action_mode": "parameterized",
                "world_advances_during_inference": False, "observation_condition": "current"}
    monkeypatch.setattr(loader, "task_definition", lambda _: SimpleNamespace(id="fixture", defaults=SimpleNamespace(model_dump=lambda **kwargs: defaults)))
    import hashlib
    manifest = {"task_id": "fixture", "task_file_sha256": hashlib.sha256((tmp_path / "task.yaml").read_bytes()).hexdigest(),
                "scene_seed": 23, "scene_variant": "original", "model_name": "explicit", "reasoning_effort": "low",
                "agent_session": "native_persistent", "world_observation_log": False, "max_steps": 41,
                "wall_timeout_s": None, "action_mode": "fixed", "world_mode": "run", "observation_condition": "current"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    data = yaml.safe_load(path.read_text())
    data["environment"] = {"mode": "attach", "url": "http://existing", "episode_manifest": "manifest.json"}
    path.write_text(yaml.safe_dump(data))
    plan = prepare(path)["plans"][0]
    assert plan["task"]["seed"] == 23
    assert plan["policy"]["reasoning_effort"] == "low"
    assert plan["evaluation"]["max_steps"] == 41 and plan["evaluation"]["timeout"] is None
    assert plan["evaluation"]["world_observation_log"] is False
    data["policy"]["reasoning_effort"] = "high"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="reasoning_effort"):
        prepare(path)


@pytest.mark.parametrize('configured,override,expected', [(-1, None, 17), (64, -1, 17), (-1, 23, 23), (64, 23, 23)])
def test_task_budget_sentinel_and_global_override(tmp_path, monkeypatch, configured, override, expected):
    path = config(tmp_path, monkeypatch, evaluation={'max_steps': configured})
    result = prepare(path, max_steps=override)
    assert result['plans'][0]['evaluation']['max_steps'] == expected
    assert not (tmp_path/'results').exists()


@pytest.mark.parametrize('value', [0, -2, True])
def test_invalid_global_step_override_is_rejected(tmp_path, monkeypatch, value):
    with pytest.raises(ValueError, match='max_steps'):
        prepare(config(tmp_path, monkeypatch), max_steps=value)


@pytest.mark.parametrize('arguments,expected', [([], 17), (['--max-steps', '-1'], 17), (['--max-steps', '31'], 31)])
def test_bench_cli_task_default_and_explicit_global_cap(tmp_path, monkeypatch, capsys, arguments, expected):
    from wia.cli import main
    path = config(tmp_path, monkeypatch, evaluation={'max_steps': 64})
    assert main(['bench', '--config', str(path), '--dry-run', *arguments]) == 0
    result = json.loads(capsys.readouterr().out)
    plans = [p for w in result['workers'] for p in w['plans']]
    assert len(plans) == 1 and plans[0]['evaluation']['max_steps'] == expected
