"""Monitor already running fixed-sequence checks without restarting their worlds.

Run: python scripts/resume_scene_checks.py --output outputs/EXISTING_SWEEP
This replaces detached watchdogs, has no simulator deadline, and never reloads
a scene. Ctrl-C detaches the monitor while preserving its native processes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from wia.runtime.resident_episode import read_service, stop_service
from wia.runtime.scene_capture_queue import write_json
from wia.runtime.scene_capture_resident import RendererLogMonitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    report = json.loads((output/'results.json').read_text())
    pending, workers = [], []
    for index, worker in enumerate(report['workers']):
        folder = Path(worker['output_dir'])
        value = json.loads((folder/'results.json').read_text())
        workers.append(value)
        completed = {row['task'] for row in value['results']}
        worker['previous_monitor_pid'] = worker.get('pid')
        worker.update(pid=os.getpid(), shared_monitor=True)
        for directory in sorted(folder.iterdir()):
            metadata = directory/'resident_service.json'
            if not directory.is_dir() or not metadata.exists():
                continue
            number, alias = directory.name.split('_', 1)
            if alias in completed:
                continue
            native = json.loads(metadata.read_text())
            control = Path(native['control_dir'])
            response = control/'responses'/f"{native['request_id']}.json"
            service = read_service(control)
            if service and not response.exists() and (service['pid'] != native['native_pid']
                    or service.get('request_id') != native['request_id']):
                raise RuntimeError('Existing request ownership changed; refusing to attach.')
            monitor = RendererLogMonitor(directory/'simulator.log')
            monitor.offset = 0
            pending.append(dict(worker=index, directory=directory, control=control,
                response=response, monitor=monitor, started=metadata.stat().st_mtime,
                index=int(number), task=alias, native=native))
    report.update(monitor_pid=os.getpid(), simulator_timeout_s=None,
                  monitor_resumed_at=datetime.now(timezone.utc).isoformat())
    order = {task:index for index, task in enumerate(task for step in range(
        max(len(w['tasks']) for w in report['workers'])) for w in report['workers']
        for task in w['tasks'][step:step+1])}
    refreshed = 0.
    if not pending:
        results = []
        for index, worker in enumerate(report['workers']):
            value = workers[index]
            value['completed'] = {row['task'] for row in value['results']} == set(value['tasks'])
            write_json(Path(worker['output_dir'])/'results.json', value)
            results.extend(dict(row, index=order[row['task']], worker=index, gpu=worker['gpu'])
                           for row in value['results'])
        report.update(results=sorted(results, key=lambda row: row['index']),
                      completed=all(value['completed'] for value in workers),
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output/'results.json', report)
    while pending:
        for item in list(pending):
            directory, control = item['directory'], item['control']
            receipt, error = None, item['monitor'].failure()
            if item['response'].exists():
                receipt = json.loads(item['response'].read_text())
            elif read_service(control) is None:
                error = error or 'Owned simulator exited without a response.'
            if receipt is None and error is None:
                continue
            validation = directory/'scorer_validation.json'
            check = json.loads(validation.read_text()) if validation.exists() else {}
            status = ('runtime_error' if error else 'passed' if receipt.get('scene_exit_code') == 0
                      and check.get('completed') and not check.get('errors') else 'issues_found')
            if status == 'passed' and check.get('evaluation_validity') != 'complete':
                status = 'incomplete_coverage'
            record = dict(index=item['index'], task=item['task'], output_dir=str(directory),
                status=status, native_receipt=receipt, validation=check,
                started_at=datetime.fromtimestamp(item['started'], timezone.utc).isoformat(),
                elapsed_s=time.time()-item['started'], resumed_existing_request=True,
                elapsed_s_source='preserved resident request file timestamp')
            if error:
                record['error'] = error
            workers[item['worker']]['results'].append(record)
            if error or not receipt.get('reusable'):
                stop_service(control, force=True)
            else:
                from wia.runtime.resident_episode import stop_unless_retained
                stop_unless_retained(control)
            pending.remove(item)
        if not pending or time.monotonic()-refreshed >= 30:
            results = []
            for index, worker in enumerate(report['workers']):
                value = workers[index]
                value['completed'] = {r['task'] for r in value['results']} == set(value['tasks'])
                if value['completed']:
                    value['finished_at'] = datetime.now(timezone.utc).isoformat()
                write_json(Path(worker['output_dir'])/'results.json', value)
                results.extend(dict(row, index=order[row['task']], worker=index, gpu=worker['gpu'])
                               for row in value['results'])
            report.update(results=sorted(results, key=lambda row:row['index']),
                          completed=all(value['completed'] for value in workers))
            if not pending:
                report['finished_at'] = datetime.now(timezone.utc).isoformat()
            write_json(output/'results.json', report)
            print(f"[resumed-checks] checked={len(results)} pending={len(pending)} simulator_deadline=none", flush=True)
            refreshed = time.monotonic()
        time.sleep(.5)
    return int(not report['completed'] or any(row['status'] != 'passed' for row in report['results']))


if __name__ == '__main__':
    raise SystemExit(main())
