"""Exact native depth, prompt boundaries, isolated MCP, and correlated query logs."""
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core.agent.backends import CodexAgent, LocalHTTPAgent
from core.agent.decision import DecisionInput, ModelError
from core.agent.depth_inputs import DepthSnapshot, depth_access
from core.agent.image_inputs import image_inputs
from wia.runtime.agent.zero_shot import run_episode
from core.sensor_observations import SensorReadings
from harness_integration.test_feedback_geometry_revision import camera
from harness_integration.test_zero_shot_agent import Environment


def depth_observation():
    original = asyncio.run(Environment().observe())
    depth = np.arange(192, dtype=np.float32).reshape(12, 16) / np.float32(10000) + np.float32(.06123)
    depth[0, :4] = [0, np.nan, np.inf, np.nan]
    return replace(original, sensors=SensorReadings(head_depth_m=depth, wrist_depth_m=depth + np.float32(.001)),
                   camera_geometry={"wrist_rgb": camera()})


def test_raw_pixels_and_regions_preserve_close_range_exactly_and_mark_invalid(tmp_path):
    observation = depth_observation()
    observation.save(tmp_path)
    snapshot = DepthSnapshot.from_directory(tmp_path)
    pixels = [[4, 0], [9, 8], [0, 0], [1, 0], [2, 0], [3, 0]]
    result = snapshot.pixels("head_depth_m", pixels)
    assert result["optical_z_m"][:2] == [float(observation.sensors.head_depth_m[v, u]) for u, v in pixels[:2]]
    assert result["optical_z_m"][2:] == [None] * 4
    assert result["optical_z_m"][0] < .1  # No old 0.1m colour-scale saturation.
    assert result["snapshot_id"] == observation.snapshot_id
    assert result["sha256"] == depth_access(observation)["sensors"]["head_depth_m"]["sha256"]
    region = snapshot.region("wrist_depth_m", 5, 3, 3, 2)
    assert region["optical_z_m"] == observation.sensors.wrist_depth_m[3:5, 5:8].tolist()
    assert region["resampling"] == "none" and region["source"] == "native_camera_depth"
    assert region["unit"] == "m" and region["matrix_order"] == "rows v, columns u"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("sensor,pixels", [("global_depth", [[1, 1]]), ("wrist_depth_m", [[16, 0]]),
    ("wrist_depth_m", [[-1, 0]]), ("head_depth_m", [[1.5, 1]]), ("head_depth_m", [[True, 1]]),
    ("head_depth_m", []), ("head_depth_m", [[1, 1]] * 257)])
def test_query_rejects_unavailable_sensors_bad_pixels_and_excessive_outputs(sensor, pixels):
    snapshot = DepthSnapshot.from_observation(depth_observation())
    with pytest.raises(ValueError):
        snapshot.pixels(sensor, pixels)


def test_snapshot_detects_modified_sensor_arrays_and_never_loads_arbitrary_paths(tmp_path):
    observation = depth_observation()
    path = observation.save(tmp_path)
    np.save(tmp_path / "head_depth_m.npy", np.ones((12, 16), np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="snapshot metadata"):
        DepthSnapshot.from_directory(tmp_path)
    metadata = json.loads(path.read_text())
    metadata["sensors"]["head_depth_m"]["path"] = "../private-evaluation.npy"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="snapshot path"):
        DepthSnapshot.from_directory(tmp_path)


def test_initial_prompt_distinguishes_centres_without_pixel_cues_and_rgb_is_unchanged():
    observation = depth_observation()
    request = DecisionInput(observation, ("DONE",))
    prompt, global_prompt = request.prompt(), request.global_prompt()
    assert "image centre is the camera view centre, not the gripper grasp centre" in global_prompt
    assert global_prompt.index("not the gripper grasp centre") < len(global_prompt) / 2
    assert "Raw metric depth access:" in prompt and "Raw metric depth access:" not in global_prompt
    assert "Robot-reference depth samples:" not in prompt
    assert "Robot reference projections:" not in prompt and "Wrist grasp-reference pixel:" not in prompt
    assert "pixel_uv" not in prompt and "pixel_uv" not in global_prompt
    assert "read_depth_pixels" in prompt and "read_depth_region" in prompt
    assert len(image_inputs(observation)) == 2
    for name, pixels, _, transform in image_inputs(observation):
        assert transform == "identity" and np.array_equal(pixels, observation.rgb_images()[name])
    missing = replace(observation, sensors=SensorReadings())
    assert not depth_access(missing)["available"] and depth_access(missing)["tools"] == []


def test_stdio_mcp_exposes_only_exact_depth_reads_from_the_supplied_snapshot(tmp_path):
    observation = depth_observation()
    observation.save(tmp_path)
    script = Path(__import__('core.agent.depth_mcp', fromlist=['__file__']).__file__)

    async def exercise():
        params = StdioServerParameters(command=sys.executable, args=[str(script), str(tmp_path)])
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                listing = await session.list_tools()
                assert {tool.name for tool in listing.tools} == {"read_depth_pixels", "read_depth_region"}
                assert all(tool.annotations.read_only_hint for tool in listing.tools)
                result = await session.call_tool("read_depth_pixels", {"sensor": "wrist_depth_m", "pixels_uv": [[9, 8]]})
                assert not result.is_error
                assert result.structured_content["optical_z_m"] == [float(observation.sensors.wrist_depth_m[8, 9])]
                assert json.loads(result.content[0].text) == result.structured_content
                invalid = await session.call_tool("read_depth_region", {"sensor": "wrist_depth_m", "u": 15, "v": 11,
                    "width": 2, "height": 2})
                assert invalid.is_error and not invalid.structured_content["ok"]
                unavailable = await session.call_tool("read_depth_pixels", {"sensor": "global_depth", "pixels_uv": [[0, 0]]})
                assert unavailable.is_error

    asyncio.run(exercise())


@pytest.mark.parametrize("unauthorized", [False, True])
def test_codex_depth_tool_allowlist_and_step_stamped_json_input_logs(tmp_path, unauthorized):
    observation = depth_observation()
    result = DepthSnapshot.from_observation(observation).pixels("wrist_depth_m", [[9, 8]])
    binary = tmp_path / "fake-codex"
    event = {"type": "item.completed", "item": {"id": "depth-1", "type": "mcp_tool_call",
        "server": "robot_executor" if unauthorized else "depth_snapshot", "tool": "read_depth_pixels",
        "arguments": {"sensor": "wrist_depth_m", "pixels_uv": [[9, 8]]},
        "result": {"content": [{"type": "text", "text": json.dumps(result)}], "structuredContent": result},
        "error": None}}
    binary.write_text("#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\nassert args.count('--image') == 2\n"
        "assert 'mcp_servers.depth_snapshot.required=true' in args\n"
        "assert args.index('--no-daemon') < args.index('exec')\n"
        "assert 'features.code_mode_host=true' in args and 'features.code_mode=false' in args\n"
        "assert not any(args[i:i+2] == ['--disable','code_mode_host'] for i in range(len(args)))\n"
        "assert 'mcp_servers.depth_snapshot.enabled_tools=[\"read_depth_pixels\", \"read_depth_region\"]' in args\n"
        "sys.stdin.read()\nprint(" + repr(json.dumps(event)) + ",flush=True)\n"
        "Path(args[args.index('--output-last-message')+1]).write_text('{\"action\":\"DONE\"}')\n")
    binary.chmod(0o755)

    class DepthEnvironment(Environment):
        async def observe(self):
            return observation

    output = tmp_path / "episode"
    summary = asyncio.run(run_episode(DepthEnvironment(), CodexAgent(executable=str(binary)), output))
    rows = [json.loads(line) for line in (output / "model_inputs.jsonl").read_text().splitlines()]
    replies = [row for row in rows if row["event"] == "mcp_response"]
    if unauthorized:
        assert summary["termination_reason"] == "model_error" and not replies
    else:
        assert summary["termination_reason"] == "model_done" and len(replies) == 1
        reply = replies[0]
        assert reply["source"] == "model_depth_mcp" and reply["passed_to_model"]
        assert reply["result"]["structuredContent"] == result
        assert isinstance(reply["result"]["content"][0]["text"], dict)
        assert reply["step"] == reply["attempt"] == 1 and reply["snapshot_id"] == result["snapshot_id"]
        step_input = next(row for row in rows if row["event"] == "step_input")
        assert step_input["raw_depth"]["sensors"]["wrist_depth_m"]["sha256"] == result["sha256"]
        assert len(step_input["images"]) == 2 and all(image["type"] == "rgb" for image in step_input["images"])
        assert json.loads((output / "step_0001/raw_depth_input.json").read_text()) == depth_access(observation)
        assert summary["agent_decisions"] == 1  # Numeric reads are not extra robot decisions.


def test_codex_rejects_depth_mcp_when_sensor_disabled(tmp_path):
    binary = tmp_path / "fake-codex"
    binary.write_text("#!/usr/bin/env python3\nimport json,sys\nsys.stdin.read()\n"
        "print(json.dumps({'type':'item.completed','item':{'type':'mcp_tool_call',"
        "'server':'depth_snapshot','tool':'read_depth_pixels'}}))\n")
    binary.chmod(0o755)
    request = DecisionInput(asyncio.run(Environment().observe()), ("DONE",))
    with pytest.raises(ModelError, match="unauthorized tool operation"):
        asyncio.run(CodexAgent(executable=str(binary)).decide(request))


def test_local_endpoint_queries_exact_depth_without_images_or_grasp_judgement(monkeypatch):
    observation = depth_observation()
    sent, logs = [], []

    def post(url, *, json: dict, **kwargs):
        sent.append(__import__('json').loads(__import__('json').dumps(json)))
        if len(sent) == 1:
            message = {"content": None, "reasoning_content": "PRIVATE_THOUGHT", "tool_calls": [{
                "id": "depth-query-1", "type": "function", "function": {"name": "read_depth_pixels",
                "arguments": '{"sensor":"wrist_depth_m","pixels_uv":[[9,8]]}'}}]}
        else:
            message = {"content": '{"action":"DONE"}'}

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": message}]}

        return Response()

    monkeypatch.setattr("core.agent.backends.requests.post", post)
    agent = LocalHTTPAgent(base_url="http://localhost:1234/v1", model="local")
    agent.attach_input_logger(logs.append)
    reply = asyncio.run(agent.decide(DecisionInput(observation, ("DONE",))))
    assert reply.raw_text == '{"action":"DONE"}' and reply.metadata["depth_query_count"] == 1
    assert len(sent[0]["messages"][0]["content"]) == 3  # Prompt plus two unchanged RGBs.
    assert {tool["function"]["name"] for tool in sent[0]["tools"]} == {"read_depth_pixels", "read_depth_region", "read_rgb_camera"}
    result = json.loads(sent[1]["messages"][-1]["content"])
    assert result == DepthSnapshot.from_observation(observation).pixels("wrist_depth_m", [[9, 8]])
    assert logs[0]["result"] == result and logs[0]["passed_to_model"]
    assert "PRIVATE_THOUGHT" not in json.dumps(sent + logs)
