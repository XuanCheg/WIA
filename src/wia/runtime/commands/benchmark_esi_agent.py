#!/usr/bin/env python3
"""Instrument an unchanged isolated Agent adapter, including archived baselines."""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, default=WIA_RUNTIME_ROOT)
    options, forwarded = parser.parse_known_args()
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    sys.path.insert(0, str(options.code_root.resolve()))
    from PIL import Image
    from core.agent.backends import CodexAgent
    from core.agent.decision import ModelReply
    from core.observations import AgentObservation
    from wia.runtime.agent.mcp_environment import MCPRobotEnvironment
    from wia.runtime.commands.run_esi_agent import main as agent_main
    original_decide, original_spawn = CodexAgent.decide, asyncio.create_subprocess_exec
    original_save, image_save = AgentObservation.save, Image.Image.save
    observe, execute = MCPRobotEnvironment.observe, MCPRobotEnvironment.execute
    active = {}
    episode_metrics = {}
    def add(metric, duration):
        active[metric] = active.get(metric, 0.)+duration
    def save(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original_save(self, *args, **kwargs)
        finally:
            if active:
                add('observation_serialization_time', time.perf_counter()-started)
    def encode(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return image_save(self, *args, **kwargs)
        finally:
            if active:
                add('image_encode_wall_time', time.perf_counter()-started)
                add('image_encode_count', 1)
    AgentObservation.save, Image.Image.save = save, encode
    class Stream:
        def __init__(self, stream, root, started):
            self.stream, self.root, self.started = stream, Path(root), started
            self.tools = {}
        async def __aiter__(self):
            async for line in self.stream:
                try:
                    event = json.loads(line)
                    if event.get('type') == 'thread.started':
                        active.setdefault('model_application_startup_time', time.perf_counter()-self.started)
                    item = event.get('item') or {}
                    if item.get('type') == 'mcp_tool_call':
                        if event['type'] == 'item.started':
                            self.tools[item.get('id')] = time.perf_counter()
                        elif event['type'] == 'item.completed' and item.get('id') in self.tools:
                            add('depth_tool_wall_time', time.perf_counter()-self.tools.pop(item['id']))
                except (ValueError, TypeError, AttributeError):
                    pass
                yield line
            sidecar = self.root/'depth_timing.json'
            if sidecar.exists():
                active.update(json.loads(sidecar.read_text()))
    async def spawn(*args, **kwargs):
        started = time.perf_counter()
        active['temporary_workspace_preparation_time'] = started-active['_started']
        process = await original_spawn(*args, **kwargs)
        add('model_process_startup_time', time.perf_counter()-started)
        active['_process_started'] = started
        process.stdout = Stream(process.stdout, kwargs['cwd'], started)
        return process
    asyncio.create_subprocess_exec = spawn
    async def decide(self, request):
        active.clear()
        active['_started'] = time.perf_counter()
        active['depth_tool_wall_time'] = 0.
        reply = await original_decide(self, request)
        active['model_and_tool_wall_time'] = time.perf_counter()-active.get('_process_started', active['_started'])
        metrics = {key: value for key, value in active.items() if not key.startswith('_')}
        # New adapters measure finer preparation categories themselves.
        metrics.update(reply.metadata.get('timings', {}))
        # Same definition for baseline and optimized adapter: all current-input
        # workspace preparation before starting the isolated model process.
        metrics['model_input_preparation_time'] = metrics.get('temporary_workspace_preparation_time', 0.)
        for key, value in metrics.items():
            episode_metrics[key] = episode_metrics.get(key, 0.)+value
        active.clear()
        return ModelReply(reply.raw_text, dict(reply.metadata, timings=metrics))
    CodexAgent.decide = decide
    async def timed_observe(self):
        started = time.perf_counter()
        try:
            return await observe(self)
        finally:
            episode_metrics['observation_rpc_wall_time'] = episode_metrics.get('observation_rpc_wall_time', 0.)+time.perf_counter()-started
    async def timed_execute(self, *args):
        started = time.perf_counter()
        try:
            return await execute(self, *args)
        finally:
            episode_metrics['action_execution_wall_time'] = episode_metrics.get('action_execution_wall_time', 0.)+time.perf_counter()-started
    MCPRobotEnvironment.observe, MCPRobotEnvironment.execute = timed_observe, timed_execute
    started = time.perf_counter()
    try:
        return agent_main(forwarded)
    finally:
        episode_metrics['total_episode_wall_time'] = time.perf_counter()-started
        output = Path(forwarded[forwarded.index('--output-dir')+1])
        if output.exists():
            (output/'performance_agent.json').write_text(json.dumps({
                'timing_semantics': 'inclusive; preparation and serialization overlap; startup is OS process creation',
                'metrics': episode_metrics}, indent=2)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
