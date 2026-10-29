"""Input/output correlation, crash persistence, public-only summaries and archived exports."""
import asyncio
import json

import pytest

from core.agent.backends import CodexAgent, LocalHTTPAgent
from core.agent.decision import ModelReply
from core.agent.trace_log import EpisodeTrace, compact_json
from wia.runtime.agent.zero_shot import run_episode
from wia.runtime.commands.export_esi_agent_logs import export_comparison, export_episode
from harness_integration.test_zero_shot_agent import Agent, Environment


def rows(path, event=None):
    data = [json.loads(line) for line in path.read_text().splitlines()]
    return [row for row in data if event is None or row["event"] == event]


def test_trace_correlates_retries_exact_prompts_and_new_boundaries(tmp_path):
    env = Environment()
    agent = Agent('bad JSON', '{"action":"BASE_FWD"}', '{"action":"DONE"}')
    summary = asyncio.run(run_episode(env, agent, tmp_path))
    inputs, outputs = rows(tmp_path / "model_inputs.jsonl"), rows(tmp_path / "model_outputs.jsonl")
    assert [row["event"] for row in inputs[:2]] == ["global_prompt", "task_prompt"]
    assert inputs[1]["prompt"] == agent.inputs[0].observation.task_text
    assert inputs[1]["prompt"] not in inputs[0]["prompt"]
    requests = [row for row in inputs if row["event"] == "step_input"]
    replies = [row for row in outputs if row["event"] == "model_output"]
    keys = ("step", "attempt", "action_step", "snapshot_id")
    assert [[row[key] for key in keys] for row in requests] == [[row[key] for key in keys] for row in replies]
    assert [(row["step"], row["attempt"]) for row in requests] == [(1, 1), (1, 2), (2, 1)]
    assert requests[0]["snapshot_id"] == requests[1]["snapshot_id"] < requests[2]["snapshot_id"]
    assert [row["prompt"] for row in requests] == [request.prompt() for request in agent.inputs]
    assert requests[1]["retry_feedback"] and requests[0]["retry_feedback"] is None
    assert all(image["data"] == "rgb" and image["type"] == "rgb" for image in requests[0]["images"])
    assert all(row["timestamp"] and not row["reconstructed"] for row in inputs + outputs)
    assert [row["accepted"] for row in outputs if row["event"] == "validation"] == [False, True, True]
    assert outputs[-1]["termination_reason"] == "model_done"
    assert summary["model_logs"]["inputs"] == str(tmp_path / "model_inputs.jsonl")


def test_response_is_persisted_before_execution_failure(tmp_path):
    class FailingEnvironment(Environment):
        async def execute(self, action):
            replies = rows(tmp_path / "model_outputs.jsonl", "model_output")
            assert replies[0]["public_output"] == '{"action":"BASE_FWD"}'
            assert rows(tmp_path / "model_outputs.jsonl", "validation")[0]["accepted"]
            raise RuntimeError("Robot connection unavailable")

    summary = asyncio.run(run_episode(FailingEnvironment(), Agent('{"action":"BASE_FWD"}'), tmp_path))
    assert summary["termination_reason"] == "runtime_error"
    assert len(rows(tmp_path / "model_outputs.jsonl", "model_output")) == 1


def test_cancelled_request_has_corresponding_output_status(tmp_path):
    class Cancel(Agent):
        async def decide(self, request):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_episode(Environment(), Cancel(), tmp_path))
    assert len(rows(tmp_path / "model_inputs.jsonl", "step_input")) == 1
    reply = rows(tmp_path / "model_outputs.jsonl", "model_output")[0]
    assert reply["status"] == "cancelled" and reply["public_output"] is None
    assert rows(tmp_path / "model_outputs.jsonl")[-1]["termination_reason"] == "cancelled"


def test_compact_mcp_json_keeps_structure_without_encoded_sensors_or_secrets():
    sensor = {"shape": [2, 2], "dtype": "float32", "npy_base64": "SENSOR_BINARY"}
    payload = {"observation": {"sensors": {"head_depth_m": sensor}}, "ok": True}
    result = compact_json({"structuredContent": payload, "content": [
        {"type": "text", "text": json.dumps(payload)},
        {"type": "image", "mimeType": "image/png", "data": "IMAGE_BINARY"}],
        "control_token": "SECRET_TOKEN", "reasoning_content": "PRIVATE_THOUGHT"})
    assert result["structuredContent"]["observation"]["sensors"]["head_depth_m"]["shape"] == [2, 2]
    assert isinstance(result["content"][0]["text"], dict)
    assert result["content"][1]["data"] == "rgb"
    assert result["control_token"] == "[redacted]"
    assert all(text not in json.dumps(result) for text in ("IMAGE_BINARY", "SENSOR_BINARY", "SECRET_TOKEN", "PRIVATE_THOUGHT"))


@pytest.mark.parametrize("include_summary", [False, True])
def test_codex_only_explicit_public_summary_and_wire_prompt_enter_logs(tmp_path, include_summary):
    binary = tmp_path / "fake-codex"
    event = {"type": "item.completed", "item": {"type": "reasoning", "text": "PRIVATE_THOUGHT",
        "content": [{"type": "reasoning_text", "text": "PRIVATE_THOUGHT"}],
        "encrypted_content": "PRIVATE_ENCRYPTED"}}
    if include_summary:
        event["item"]["summary"] = [{"type": "summary_text", "text": "Brief public summary."}]
    binary.write_text("#!/usr/bin/env python3\nimport json, sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\nassert 'model_reasoning_summary=\"concise\"' in args\n"
        "prompt=sys.stdin.read()\n"
        "print(" + repr(json.dumps(event)) + ")\n"
        "Path(args[args.index('--output-last-message')+1]).write_text('{\"action\":\"DONE\"}')\n"
        "print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'{\"action\":\"DONE\"}'}}))\n")
    binary.chmod(0o755)
    env = Environment()
    output = tmp_path / "episode"
    asyncio.run(run_episode(env, CodexAgent(executable=str(binary)), output))
    reply = rows(output / "model_outputs.jsonl", "model_output")[0]
    assert reply["reasoning_summary"]["status"] == ("available" if include_summary else "unavailable")
    if include_summary:
        assert reply["reasoning_summary"]["text"] == "Brief public summary."
    assert reply["public_messages"] == ['{"action":"DONE"}']
    assert reply["public_messages_status"] == "recorded"
    messages = rows(output / "model_outputs.jsonl", "public_message")
    assert messages[0]["text"] == reply["public_messages"][0]
    assert messages[0]["step"] == messages[0]["attempt"] == 1
    text = "".join(path.read_text() for path in (output / "model_outputs.jsonl", output / "steps.jsonl"))
    assert "PRIVATE_THOUGHT" not in text and "PRIVATE_ENCRYPTED" not in text
    header = rows(output / "model_inputs.jsonl")[0]
    assert "Do not use tools" in header["developer_instructions"]


def test_local_public_summary_saved_and_raw_reasoning_discarded(monkeypatch, tmp_path):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"action":"DONE"}',
                "reasoning_summary": "Brief public endpoint summary.", "reasoning_content": "PRIVATE_THOUGHT"}}]}

    sent = []

    def post(*args, **kwargs):
        sent.append(kwargs["json"])
        return Response()

    monkeypatch.setattr("core.agent.backends.requests.post", post)
    asyncio.run(run_episode(Environment(), LocalHTTPAgent(base_url="http://localhost:1234/v1", model="local"), tmp_path))
    request = rows(tmp_path / "model_inputs.jsonl", "step_input")[0]
    assert request["prompt"] == sent[0]["messages"][0]["content"][0]["text"]
    reply = rows(tmp_path / "model_outputs.jsonl", "model_output")[0]
    assert reply["reasoning_summary"]["text"] == "Brief public endpoint summary."
    assert "PRIVATE_THOUGHT" not in (tmp_path / "steps.jsonl").read_text()


def test_reasoning_markup_and_custom_raw_metadata_never_enter_traces(tmp_path):
    class WithPrivateMetadata(Agent):
        async def decide(self, request):
            return ModelReply('{"action":"DONE"}', {"analysis": "PRIVATE_THOUGHT",
                "reasoning_content": "PRIVATE_THOUGHT", "usage": {"input_tokens": 1}})

    for name, agent in (("metadata", WithPrivateMetadata()),
                        ("markup", Agent('<think>PRIVATE_THOUGHT</think>{"action":"DONE"}'))):
        output = tmp_path / name
        asyncio.run(run_episode(Environment(), agent, output))
        for filename in ("steps.jsonl", "model_inputs.jsonl", "model_outputs.jsonl"):
            assert "PRIVATE_THOUGHT" not in (output / filename).read_text()


def test_archived_export_preserves_all_attempts_and_marks_missing_data(tmp_path):
    episode = tmp_path / "episode"
    asyncio.run(run_episode(Environment(), Agent("bad JSON", '{"action":"BASE_FWD"}', '{"action":"DONE"}'), episode))
    # Simulate an old archive: these new traces were not available at collection.
    (episode / "model_inputs.jsonl").unlink()
    (episode / "model_outputs.jsonl").unlink()
    before = {name: (episode / name).read_bytes() for name in ("summary.json", "steps.jsonl", "run.json")}
    exported = export_episode(episode, tmp_path / "exported")
    assert exported["model_calls"] == 3 and exported["decision_records"] == 2
    inputs = rows(tmp_path / "exported/model_inputs.jsonl")
    outputs = rows(tmp_path / "exported/model_outputs.jsonl", "model_output")
    assert all(row["reconstructed"] for row in inputs + outputs)
    assert inputs[2]["raw_mcp_returns"] == "not_recorded"
    assert all(row["reasoning_summary"]["reason"] == "not_recorded" for row in outputs)
    assert all(row["public_messages"] is None and row["public_messages_status"] == "not_recorded" for row in outputs)
    assert len([row for row in inputs if row["event"] == "step_input"]) == 3
    assert all((episode / name).read_bytes() == value for name, value in before.items())
    with pytest.raises(ValueError, match="preserve"):
        export_episode(episode, tmp_path / "exported")


def test_comparison_export_uses_formal_summary_paths_instead_of_excluded_pilots(tmp_path):
    episode = tmp_path / "formal/episode"
    asyncio.run(run_episode(Environment(), Agent('{"action":"DONE"}'), episode))
    manifest = tmp_path / "summary.json"
    manifest.write_text(json.dumps({"runs": [{"id": "formal-run", "case": "test", "model": "test",
        "path": str(tmp_path / "excluded-pilot"), "summary_file": str(episode / "summary.json")}] }))
    index = export_comparison(manifest, tmp_path / "exports")
    assert index["episodes"] == index["model_calls"] == 1
    assert index["runs"][0]["source_episode"] == str(episode)
    assert rows(tmp_path / "exports/model_inputs.jsonl")[0]["run_id"] == "formal-run"
    assert len(rows(tmp_path / "exports/model_outputs.jsonl", "model_output")) == 1
    with pytest.raises(ValueError, match="preserved"):
        export_comparison(manifest, tmp_path / "exports")


def test_existing_trace_files_are_never_overwritten(tmp_path):
    (tmp_path / "model_outputs.jsonl").write_text("existing\n")
    with pytest.raises(ValueError, match="preserve"):
        EpisodeTrace(tmp_path)
    assert not (tmp_path / "model_inputs.jsonl").exists()


@pytest.mark.parametrize("outcome", ["complete", "cancel", "timeout"])
def test_codex_public_commentary_is_written_before_final_and_survives_interruptions(tmp_path, outcome):
    binary, release = tmp_path / "fake-codex", tmp_path / "release-model"
    middle = "I will inspect the current RGB views before choosing the action."
    binary.write_text("#!/usr/bin/env python3\nimport json, sys, time\nfrom pathlib import Path\n"
        "args=sys.argv[1:]; sys.stdin.read()\n"
        "print(json.dumps({'type':'item.completed','item':{'type':'reasoning','text':'PRIVATE_THOUGHT'}}),flush=True)\n"
        "print(json.dumps({'type':'item.completed','item':{'id':'commentary_1','type':'agent_message',"
        "'phase':'commentary','text':" + repr(middle) + "}}),flush=True)\n"
        "release=Path(" + repr(str(release)) + ")\n"
        "while not release.exists(): time.sleep(0.01)\n"
        "final='{\"action\":\"DONE\"}'\n"
        "Path(args[args.index('--output-last-message')+1]).write_text(final)\n"
        "print(json.dumps({'type':'item.completed','item':{'id':'final_1','type':'agent_message','phase':'final_answer','text':final}}),flush=True)\n")
    binary.chmod(0o755)
    output = tmp_path / "episode"
    agent = CodexAgent(executable=str(binary), timeout_s=2 if outcome == "timeout" else 10)

    async def run():
        task = asyncio.create_task(run_episode(Environment(), agent, output))
        try:
            for _ in range(150):
                log = output / "model_outputs.jsonl"
                if log.is_file() and rows(log, "public_message"):
                    break
                await asyncio.sleep(0.01)
            events = rows(output / "model_outputs.jsonl", "public_message")
            assert events[0]["text"] == middle and events[0]["phase"] == "commentary"
            assert events[0]["item_id"] == "commentary_1"
            assert not task.done() and not rows(output / "model_outputs.jsonl", "model_output")
            if outcome == "complete":
                release.touch()
                await task
            elif outcome == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await task
        finally:
            release.touch()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    reply = rows(output / "model_outputs.jsonl", "model_output")[0]
    assert reply["public_messages"][0] == middle
    assert reply["public_messages_status"] == ("recorded" if outcome == "complete" else "partial")
    assert reply["public_output"] == ('{"action":"DONE"}' if outcome == "complete" else None)
    events = rows(output / "model_outputs.jsonl", "public_message")
    assert len(events) == (2 if outcome == "complete" else 1)
    assert all(row["step"] == row["attempt"] == 1 and row["timestamp"] for row in events)
    assert "PRIVATE_THOUGHT" not in (output / "model_outputs.jsonl").read_text()
    assert agent._output_logger is None
