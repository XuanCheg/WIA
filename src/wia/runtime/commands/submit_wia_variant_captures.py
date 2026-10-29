"""Append exact versioned WIA captures to the existing campaign.

Run: python -B src/wia/runtime/commands/submit_wia_variant_captures.py --records RECORDS.json
Use --retry-failed to replace only indexed, explicitly retryable failed jobs.
Each replacement retains old evidence and an append-only submission journal.
Main reviews definitions and resources before submission. Existing jobs, index
entries and receipts remain intact; this script never starts or stops workers.
"""

from wia.resources import ROOT as WIA_RUNTIME_ROOT
from pathlib import Path
import argparse
from copy import deepcopy
import json
import os
import sys
import time

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
from wia.runtime.esi_category_capture import freeze_working_code, publish_record
from wia.runtime.scene_capture_queue import digest, read_json, write_json, sha256

CAMPAIGN = ROOT / 'outputs/wia_v1_20261023/capture_campaign'


def _validate_item(item):
    """Check all supplied definition/source bytes before any publication."""
    record = item['record']
    for value in (item, record):
        for kind in ('task', 'scene', 'parent_task', 'parent_scene'):
            if kind + '_sha256' in value:
                if sha256(value[kind + '_file']) != value[kind + '_sha256']:
                    raise ValueError('Definition changed after record preparation: ' + kind)
    for kind in ('task', 'scene'):
        if (Path(item[kind + '_file']).resolve() != Path(record[kind + '_file']).resolve()
                or item[kind + '_sha256'] != record[kind + '_sha256']):
            raise ValueError('Item and capture record definition identities differ.')
    if item.get('source_file') and item.get('source_sha256'):
        if sha256(item['source_file']) != item['source_sha256']:
            raise ValueError('Item source changed after record preparation.')
        if (Path(item['source_file']).resolve() != Path(record['original_source']).resolve()
                or item['source_sha256'] != record['source_sha256']):
            raise ValueError('Item and capture record primary source identities differ.')
    if sha256(record['original_source']) != record['source_sha256']:
        raise ValueError('Capture primary source changed after record preparation.')
    members = record.get('capture_source_members', [])
    for member in members:
        if sha256(member['source_file']) != member['source_sha256']:
            raise ValueError('Capture source member changed after record preparation.')
    by_identity = {member['source_identity']: member for member in members}
    for kind in ('task', 'scene'):
        if kind + '_source_sha256' in item and item[kind + '_source_sha256'] != record[kind + '_source_sha256']:
            raise ValueError('Item and capture record source role hashes differ.')
        if record.get(kind + '_source_sha256'):
            member = by_identity.get(record[kind + '_source_identity'])
            if member is None or member['source_sha256'] != record[kind + '_source_sha256']:
                raise ValueError('Capture source role hash differs from explicit source membership.')
    unsigned = deepcopy(record)
    revision = unsigned.pop('revision')
    if digest(unsigned) != revision:
        raise ValueError('Capture record revision changed after preparation.')


def _retry_status(queue, entry):
    job = entry['job_id']
    status = read_json(queue / 'jobs' / job / 'status.json', {})
    if status.get('state') != 'FAILED' or status.get('retryable') is not True:
        return None
    if (queue / 'claimed' / (job + '.json')).exists():
        return None
    return status


def _retry_record(record, previous, status, frozen):
    result = deepcopy(record)
    binding = {'prepared_candidate_id': record['candidate_id'],
        'prepared_revision': record['revision'], 'superseded_job_id': previous['job_id'],
        'code_commit': frozen['code_commit']}
    key = digest(binding)[:24]
    result.update(candidate_id='category_' + key, task_edit2_variant_key=key,
        capture_retry=binding, superseded_captures=[{
            **deepcopy(previous), 'superseded_status': deepcopy(status)}])
    result.pop('revision')
    result['revision'] = digest(result)
    return result


def _append_history(path, event):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(event, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--retry-failed', action='store_true', help='Replace indexed FAILED jobs only when retryable and released.')
    parser.add_argument('--max-attempts', type=int, default=3)
    args = parser.parse_args(argv)
    if args.max_attempts < 1:
        parser.error('--max-attempts must be positive.')
    campaign = CAMPAIGN
    queue = campaign / 'queue'
    package = read_json(args.records)
    index = read_json(campaign / 'review_index.json')
    existing = {(v['row_id'], v['variant_id']): v for v in index['items']}
    if len(existing) != len(index['items']):
        raise ValueError('Review index contains duplicate row_id/variant_id entries.')
    supplied = {(v['row_id'], v['variant_id']) for v in package['items']}
    if len(supplied) != len(package['items']):
        raise ValueError('Capture package contains duplicate row_id/variant_id entries.')
    for item in package['items']:
        _validate_item(item)
    items = []
    for item in package['items']:
        previous = existing.get((item['row_id'], item['variant_id']))
        if previous is None:
            items.append((item, None))
        elif args.retry_failed and _retry_status(queue, previous) is not None:
            for kind in ('task', 'scene'):
                if (previous[kind + '_sha256'] != item[kind + '_sha256']
                        or Path(previous[kind + '_file']).resolve() != Path(item[kind + '_file']).resolve()):
                    raise ValueError('Retry must retain the exact indexed task and scene definition.')
            items.append((item, previous))
    if not items:
        print('No unindexed versions or eligible retryable failures require submission.')
        return 0
    frozen = freeze_working_code(queue)
    policy = read_json(queue / 'capture_runtime_policy.json')
    policy['allowed_code_commits'] = list(dict.fromkeys([
        *policy.get('allowed_code_commits', []), frozen['code_commit']]))
    write_json(queue / 'capture_runtime_policy.json', policy)
    report_path = args.records.with_suffix('.submission.json')
    report = read_json(report_path, {})
    history_path = args.records.with_suffix('.submission_history.jsonl')
    submitted = []
    for item, previous in items:
        _validate_item(item)
        status = _retry_status(queue, previous) if previous is not None else None
        if previous is not None and status is None:
            continue
        record = _retry_record(item['record'], previous, status, frozen) if previous else deepcopy(item['record'])
        Path(record['diagnostic_dir']).mkdir(parents=True, exist_ok=True)
        published = publish_record(record, queue, frozen=frozen, priority=130,
                                   timeout_s=2400, max_attempts=args.max_attempts)
        job = published.get('job_id')
        if not job:
            raise RuntimeError(str(published.get('capture_submission_error')))
        if previous and job == previous['job_id']:
            raise RuntimeError('Retry publication must create a new job without rewriting old evidence.')
        entry = {key: item[key] for key in ('row_id', 'document_alias', 'variant_id',
            'label', 'task_file', 'scene_file', 'task_sha256', 'scene_sha256')}
        entry.update(job_id=job, receipt_file=str(queue / 'jobs' / job / 'attempts/01/preview_complete.json'),
                     queue_status_file=str(queue / 'jobs' / job / 'status.json'))
        if previous:
            prior = deepcopy(previous)
            inherited = prior.pop('superseded_captures', [])
            prior['superseded_status'] = deepcopy(status)
            entry['superseded_captures'] = [*inherited, prior]
        event = {'submitted_at': time.time(), 'code_commit': frozen['code_commit'],
            'records': str(args.records.resolve()), 'action': 'retry_failed' if previous else 'initial_submission',
            'entry': deepcopy(entry), 'capture_candidate_id': record['candidate_id'],
            'evidence_scope': 'native_initial_images_not_reference_execution'}
        _append_history(history_path, event)
        if previous:
            position = next(i for i, current in enumerate(index['items'])
                if (current['row_id'], current['variant_id']) == (item['row_id'], item['variant_id']))
            index['items'][position] = entry
        else:
            index['items'].append(entry)
        write_json(campaign / 'review_index.json', index)
        submitted.append(entry)
        report.setdefault('code_commit', frozen['code_commit'])
        report.update(latest_code_commit=frozen['code_commit'], gpu_scope=[0, 1, 2, 3],
            records=str(args.records.resolve()), submission_history_file=str(history_path.resolve()),
            evidence_scope='native_initial_images_not_reference_execution')
        report.setdefault('submitted', []).append(deepcopy(entry))
        write_json(report_path, report)
    print({'submitted': len(submitted), 'code_commit': frozen['code_commit']})
    return 0


if __name__ == '__main__':
    main()
