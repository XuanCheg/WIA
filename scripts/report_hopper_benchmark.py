"""Aggregate live campaign evidence without changing episodes or their scores.

Run: python scripts/report_hopper_benchmark.py --output-dir OUTPUT [--watch]
Watch mode writes progress every 30 seconds and a final report after workers end
or the deadline plus cleanup grace. Model reasoning is never read or exported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import statistics
import time


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def records(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def report(out):
    manifest = read_json(out / 'manifest.json')
    with sqlite3.connect(out / 'queue.sqlite') as connection:
        connection.row_factory = sqlite3.Row
        jobs = [dict(row) for row in connection.execute('SELECT * FROM jobs ORDER BY id')]
    trials, waits, completed_reply_waits, interrupted_waits, usage = [], [], [], [], Counter()
    pause = read_json(out / 'user_pause.json', {})
    resume = read_json(out / 'window2_resume_receipt.json', {})
    pause_start, pause_end = pause.get('at_utc'), resume.get('at_utc')
    for job in jobs:
        folder = Path(job['config']).parent
        trial = folder / 'run/trial_0000'
        preloaded = read_json(folder / 'preloaded_scene.json', {})
        episode = trial if preloaded else trial / 'episode'
        native = Path(preloaded['native_output_dir']) if preloaded else trial
        experiment = read_json(trial / 'experiment.json', {})
        summary = read_json(episode / 'summary.json', {})
        outputs = [r for r in records(episode / 'model_outputs.jsonl') if r.get('event') == 'model_output']
        steps = records(episode / 'steps.jsonl')
        for step in steps:
            for attempt in step.get('attempts', []):
                start, end = attempt.get('requested_at'), attempt.get('received_at')
                overlaps_pause = bool(pause_start and pause_end and start and end
                                      and start < pause_end and end > pause_start)
                model_tool_time = (attempt.get('model_metadata') or {}).get('timings', {}).get('model_and_tool_wall_time')
                if isinstance(model_tool_time, (float, int)) and not overlaps_pause:
                    completed_reply_waits.append(model_tool_time)
                if isinstance(attempt.get('model_wait_wall_time_s'), (float, int)):
                    (interrupted_waits if overlaps_pause else waits).append(attempt['model_wait_wall_time_s'])
        session = read_json(episode / 'openai_chat_completions_session.json', {})
        if 'transport_usage' in session:
            usage.update({k: v for k, v in session['transport_usage'].items() if type(v) is int})
            usage['reasoning_tokens'] += (session['transport_usage'].get('completion_tokens_details') or {}).get('reasoning_tokens', 0) or 0
        else:
            for output in outputs:
                usage.update({k: v for k, v in (output.get('usage') or {}).items() if type(v) is int})
                usage['reasoning_tokens'] += ((output.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens', 0) or 0
        result = read_json(folder / 'run/results.json', {})
        scores = result.get('trials', [])
        evidence = {'index': job['id'], 'alias': job['alias'], 'family': job['family'],
            'gpu': job['gpu'], 'worker_pid': job['pid'], 'queue_status': job['status'],
            'stage': experiment.get('stage', 'attached_preloaded' if preloaded else 'queued'), 'started': job['started'], 'finished': job['finished'],
            'model_calls_observed': len(outputs), 'received_replies': sum(r.get('status') == 'received' for r in outputs),
            'action_boundaries': len(steps), 'native_initialized': (native / 'metadata.json').exists(),
            'command_executions': sum(step.get('action_result') is not None for step in steps),
            'native_pid': (native / 'sim_pid.txt').read_text().strip() if (native / 'sim_pid.txt').exists() else None,
            'termination_reason': summary.get('termination_reason'),
            'wia_status': scores[0].get('status') if scores else None,
            'task_success': summary.get('task_success'), 'evaluation_validity': summary.get('evaluation_validity'),
            'completion_score': summary.get('completion_score'),
            'error': summary.get('error') or experiment.get('error') or (scores[0].get('error') if scores else None),
            'note': job['note'], 'output_dir': str(trial)}
        evidence['pause_affected'] = str(trial) in manifest.get('pause_affected_trials', [])
        trials.append(evidence)
    data = {'at_utc': datetime.now(timezone.utc).isoformat(), 'deadline_utc': manifest['deadline_utc'],
        'scope': 'Diagnostic stress campaign; valid scores, model errors, environment errors and censoring are separate.',
        'selected': len(jobs), 'excluded': manifest['excluded'],
        'preserved_failed_attempts': manifest.get('failed_attempts', []),
        'queue_counts': dict(Counter(j['status'] for j in jobs)),
        'native_initialized': sum(t['native_initialized'] for t in trials),
        'trials_with_model_calls': sum(t['model_calls_observed'] > 0 for t in trials),
        'model_calls_observed': sum(t['model_calls_observed'] for t in trials),
        'received_replies': sum(t['received_replies'] for t in trials),
        'action_boundaries': sum(t['action_boundaries'] for t in trials),
        'command_executions': sum(t['command_executions'] for t in trials),
        'wia_status_counts': dict(Counter(t['wia_status'] for t in trials if t['wia_status'])),
        'termination_counts': dict(Counter(t['termination_reason'] for t in trials if t['termination_reason'])),
        'usage': dict(usage),
        'pause_affected_wait_s': interrupted_waits,
        'completed_reply_model_and_tool_s': {
            'samples': len(completed_reply_waits),
            'mean': statistics.mean(completed_reply_waits) if completed_reply_waits else None,
            'median': statistics.median(completed_reply_waits) if completed_reply_waits else None,
            'max': max(completed_reply_waits) if completed_reply_waits else None},
        'model_call_wait_s': {'samples': len(waits), 'mean': statistics.mean(waits) if waits else None,
                             'median': statistics.median(waits) if waits else None,
                             'max': max(waits) if waits else None},
        'trials': trials}
    temporary = out / 'progress.json.tmp'
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(out / 'progress.json')
    line = {key: value for key, value in data.items() if key != 'trials'}
    with (out / 'progress_history.jsonl').open('a') as stream:
        stream.write(json.dumps(line) + '\n')
    print(json.dumps(line), flush=True)
    return data


def final_report(out, data):
    (out / 'final_report.json').write_text(json.dumps(data, indent=2) + '\n')
    lines = ['WIA Hopper stress campaign', f'Updated: {data["at_utc"]}',
             f'Deadline: {data["deadline_utc"]}',
             f'Selected tasks: {data["selected"]}',
             f'Preserved failed attempts before retries: {len(data["preserved_failed_attempts"])}',
             f'Queue states: {data["queue_counts"]}',
             f'Native initialized: {data["native_initialized"]}',
             f'Trials with model calls: {data["trials_with_model_calls"]}',
             f'Model calls / received replies: {data["model_calls_observed"]} / {data["received_replies"]}',
             f'Action boundaries: {data["action_boundaries"]}',
             f'Physical command executions: {data["command_executions"]}',
             f'WIA statuses: {data["wia_status_counts"]}',
             f'Termination reasons: {data["termination_counts"]}',
             f'Usage: {data["usage"]}', f'Model wait (s): {data["model_call_wait_s"]}', '',
             'These are diagnostic trials, not a qualified benchmark success rate.',
             'Unfinished, environment/model-error and invalid-score trials are not completed task failures.', '',
             'alias\tgpu\tqueue_status\twia_status\ttermination\tactions\tscore']
    for row in data['trials']:
        lines.append('\t'.join(str(row.get(key)) for key in ('alias', 'gpu', 'queue_status',
            'wia_status', 'termination_reason', 'action_boundaries', 'completion_score')))
    (out / 'final_report.txt').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    deadline = datetime.fromisoformat(read_json(out / 'manifest.json')['deadline_utc']).timestamp()
    while True:
        data = report(out)
        counts = data['queue_counts']
        done = counts.get('running', 0) == 0 and (counts.get('queued', 0) == 0 or time.time() >= deadline)
        if not args.watch or done or time.time() >= deadline + 180:
            if args.watch:
                final_report(out, data)
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
