"""Repeat a diagnostic in an already loaded, paused world without reloading it.

Run: python scripts/validate_retained_scene.py --control-dir RUN/resident_000
     --output outputs/repeated_native_check
This is a fixed native diagnostic, with no model calls or benchmark score.
"""
import argparse
import json
from pathlib import Path
import time
import uuid

from wia.resources import ROOT
from wia.runtime.resident_episode import read_service
from wia.runtime.scene_capture_queue import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sequence', type=Path, default=ROOT / 'configs/native_scene_validation.json')
    parser.add_argument('--skip-prechecks', action='store_true',
                        help='Exercise formal scorer initialization in this diagnostic; still no benchmark score.')
    args = parser.parse_args()
    control, output = args.control_dir.resolve(), args.output.resolve()
    service = read_service(control)
    if service is None or service.get('state') != 'SCENE_RETAINED' or not service.get('retained_validation'):
        raise RuntimeError('Select a retained diagnostic world advertising retained_validation support.')
    if args.skip_prechecks and not service.get('retained_prechecks_switch'):
        raise RuntimeError('This older loaded owner cannot change its precheck policy; use a newly started owner.')
    if output.exists():
        raise ValueError('A repeated validation needs a new result directory.')
    sequence = json.loads(args.sequence.read_text())
    identifier = uuid.uuid4().hex
    requests = control / 'retained_checks'
    requests.mkdir(exist_ok=True)
    write_json(requests / f'{identifier}.json', {'output_dir': str(output), 'sequence': sequence,
                                               'prechecks_enabled': not args.skip_prechecks})
    response = control / 'retained_responses' / f'{identifier}.json'
    print(f'[retained-scene] native_pid={service["pid"]} world_rebuilt=false model_calls=0', flush=True)
    last = time.monotonic()
    while not response.exists():
        current = read_service(control)
        if current is None or current['pid'] != service['pid'] or current.get('state') == 'FAILED':
            raise RuntimeError('The selected native owner exited or failed; inspect its service log.')
        if time.monotonic() - last >= 30:
            print(f'[retained-scene] native_pid={service["pid"]} stage=validation', flush=True)
            last = time.monotonic()
        time.sleep(1)
    result = json.loads(response.read_text())
    print(json.dumps(result, indent=2))
    return int(not result.get('completed') or bool(result.get('errors')))


if __name__ == '__main__':
    raise SystemExit(main())
