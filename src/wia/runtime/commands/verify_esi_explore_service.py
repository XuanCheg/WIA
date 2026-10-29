#!/usr/bin/env python3
"""Read-only verification of a retained human-control exploration service."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import time

from mcp import Client
from PIL import Image
import requests


RGB_VIEWS = ('head', 'wrist', 'right_wrist', 'chassis_front_left', 'chassis_front_right',
             'chassis_left', 'chassis_right', 'chassis_rear')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8615')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    measurements = []
    def get(path):
        started = time.perf_counter()
        response = requests.get(args.url.rstrip('/')+path, timeout=15)
        response.raise_for_status()
        measurements.append({'path': path, 'wall_time_s': time.perf_counter()-started,
                             'http_status': response.status_code})
        return response
    ui = get('/gumi').text
    state = get('/gumi/api/state').json()
    before = get('/gumi/api/agent/state').json()
    snapshots = []
    for view in RGB_VIEWS:
        response = get('/gumi/api/agent/snapshot/'+view)
        snapshots.append({'view': view, 'shape_hw': list(Image.open(BytesIO(response.content)).size[::-1]),
                          'content_type': response.headers.get('Content-Type'),
                          'frame_seq': response.headers.get('X-Frame-Seq'),
                          'frame_time': response.headers.get('X-Frame-Time')})
    async def cached_mcp():
        async with Client(args.url.rstrip('/')+'/mcp', read_timeout_seconds=15) as client:
            context = await client.call_tool('robot_get_context', {})
            observation = await client.call_tool('robot_get_observation', {'include_sensor_data': False})
            if context.is_error or observation.is_error:
                raise RuntimeError('Retained MCP cached reads failed.')
            return context.structured_content['context'], observation.structured_content['observation']
    started = time.perf_counter()
    context, observation = asyncio.run(cached_mcp())
    mcp_wall = time.perf_counter()-started
    time.sleep(3.)
    after = get('/gumi/api/agent/state').json()
    viewer_after = get('/gumi/api/state').json()
    checks = {
        'gumi_ui_responsive': 'Robot' in ui,
        'human_control': before.get('control_owner') == after.get('control_owner') == 'human',
        'default_run_policy': context['world_mode'] == 'run' and context['world_advances_during_inference'] is True,
        'eight_rgb_cached_snapshots': len(snapshots) == 8 and all(s['shape_hw'] == [1024, 1024] for s in snapshots),
        'cached_mcp_responsive': isinstance(observation.get('snapshot_id'), int),
        'private_evaluation_unavailable_to_mcp': context['task_evaluation_available'] is False,
        'human_idle_physics_advances': after['sampled_simulation_time_s'] > before['sampled_simulation_time_s'],
        'live_preview_advances': viewer_after.get('frame_time') != state.get('frame_time'),
        'recording_disabled': state.get('recording_enabled') is False,
    }
    result = {'schema': 'retained_explore_service_v1', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'url': args.url, 'passed': all(checks.values()), 'checks': checks, 'http': measurements,
        'mcp_wall_time_s': mcp_wall, 'context': context, 'snapshots': snapshots,
        'sim_before_s': before['sampled_simulation_time_s'], 'sim_after_s': after['sampled_simulation_time_s'],
        'viewer_frame_time_before': state.get('frame_time'), 'viewer_frame_time_after': viewer_after.get('frame_time'),
        'camera_group_counts': {g['id']: len(g['views']) for g in state.get('camera_groups', [])},
        'no_commands_executed': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'passed': result['passed'], 'checks': checks}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
