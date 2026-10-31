"""Run diagnostic model I/O outside Kit's Python process.

Run: python -m wia.diagnostics.model_worker --output OUT --configuration CONFIG
The parent sends JSON paths over stdin. Only native observation payloads and
decision metadata cross the process boundary; this worker never loads physics.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import traceback

from core.agent.codex_session import CodexEpisodeSession
from core.agent.decision import DecisionInput
from core.episode_archive import EpisodeArchive
from core.observations import AgentObservation
from wia.diagnostics.privileged_policy import PrivilegedAssembler, privileged_session_instructions


async def run(output, configuration):
    with privileged_session_instructions():
        agent = CodexEpisodeSession(
            public_task={'objective': 'Grasp and lift only the named diagnostic target.'},
            episode_configuration=configuration, output=output / 'model',
            model=configuration['model'], reasoning_effort=configuration['reasoning_effort'],
            executable='/home/zhouxinyang/.local/bin/codex',
            timeout_s=configuration['model_timeout_s'], assembler=PrivilegedAssembler())
        try:
            print(json.dumps({'state': 'ready'}), flush=True)
            while line := await asyncio.to_thread(sys.stdin.readline):
                message = json.loads(line)
                if message['op'] == 'close':
                    break
                if message['op'] != 'decide':
                    raise ValueError('Unknown model worker operation.')
                path = Path(message['request'])
                data = json.loads(path.read_text())
                observation = AgentObservation.from_payload(data.pop('observation'))
                request = DecisionInput(observation=observation, **data)
                folder = output / 'steps' / f'{request.decision_step:03d}'
                images = []
                archive = EpisodeArchive(output)
                for name, pixels, label, transform in agent.request_images(request):
                    image = archive.model_image(observation, name, pixels, transform)
                    images.append({'name': name, 'label': label, 'transform': transform, 'path': str(image)})
                (folder / 'exact_model_images.json').write_text(json.dumps(images, indent=2) + '\n')
                reply = await agent.decide(request)
                destination = folder / 'worker_reply.json'
                destination.write_text(json.dumps({'raw_text': reply.raw_text, 'metadata': reply.metadata}) + '\n')
                print(json.dumps({'state': 'reply', 'path': str(destination)}), flush=True)
        except BaseException as error:
            detail = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
            (output / 'model_worker_error.json').write_text(json.dumps(detail, indent=2) + '\n')
            print(json.dumps({'state': 'error', **detail}), flush=True)
            raise
        finally:
            await agent.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--configuration', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.output, json.loads(args.configuration.read_text())))


if __name__ == '__main__':
    main()
