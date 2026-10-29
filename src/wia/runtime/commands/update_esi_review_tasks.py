#!/usr/bin/env python3
"""Replace superseded Review tasks, submit revisions and clean declared old PNGs.

Run after regenerating the currently registered converter batches:
  python src/wia/runtime/commands/update_esi_review_tasks.py --campaign-root ~/esi_category_conversion/CAMPAIGN
For a batch written to a new location, add --batch-file /absolute/path/batch_manifest.json.
Use --dry-run to print changes. Use --refresh-runtime only when shared runtime
code changed without new converter revisions. Repeated ordinary runs are no-ops.
The one-per-subclass gate is preserved; no image scan or validation is performed.
Completed publications defer in-flight predecessor cleanup independently, so
later CPU revisions can publish while those old jobs finish.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import fcntl
from pathlib import Path
import sys
import time

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.runtime.esi_category_capture import freeze_working_code
from wia.runtime.scene_capture_queue import digest, read_json, write_json
from wia.runtime.commands.publish_esi_annotation_repairs import cleanup, publish
from wia.runtime.commands.update_esi_review_catalogue import CataloguePublisher


def cpu_revision(record):
    return record.get('cpu_conversion_revision', record.get('revision'))


def cpu_version(record):
    return record.get('cpu_converter_version', record.get('converter_version'))


def registered_batches(campaign, explicit):
    """Follow declared CPU batch pointers, without directory discovery."""
    if explicit:
        return list(dict.fromkeys(Path(path).expanduser().resolve() for path in explicit))
    revisions = read_json(campaign / 'review_catalogue/category_revisions.json', {}).get('categories', {})
    paths = []
    for revision in revisions.values():
        path = revision.get('new_batch_file')
        if path:
            batch = read_json(path, {})
            paths.append(Path(batch.get('source_cpu_batch', path)))
    return list(dict.fromkeys(paths))


def plan_updates(campaign, queue, batch_files, *, refresh_runtime=False):
    """Select actual changed sources and retain every unchanged record and queue token."""
    review = campaign / 'review_catalogue'
    revisions = read_json(review / 'category_revisions.json', {}).get('categories', {})
    inventory = read_json(campaign / 'inventory.json')['categories']
    keys = {(row['big_task'], row['small_task']): row['key'] for row in inventory}
    snapshot = read_json(review / 'snapshot.json', {})
    visible = {row['source'] for row in snapshot.get('scenes', []) + snapshot.get('diagnostics', [])}
    gate = read_json(queue / 'capture_sample_gate.json', {})
    sample_ids = set(gate.get('selected_candidate_ids', []))
    gated = gate.get('state') == 'ONE_PER_CATEGORY_AWAITING_REVIEW'
    proposals, changed_sources, seen = [], set(), set()
    for path in batch_files:
        batch = read_json(path)
        if not batch or not batch.get('records'):
            raise ValueError('A converter batch must contain its source records: ' + str(path))
        records = batch['records']
        categories = {(r['big_task'], r['small_task']) for r in records}
        if len(categories) != 1:
            raise ValueError('Each batch must contain one ESI subclass: ' + str(path))
        category = categories.pop()
        key = keys[category]
        if key in seen:
            raise ValueError('Supply only one new batch for each subclass.')
        seen.add(key)
        previous_path = revisions.get(key, {}).get('new_batch_file')
        if not previous_path:
            raise ValueError('Publish the initial category batch before updating its Review tasks.')
        previous = read_json(previous_path)['records']
        parents = {r['source_identity']: r for r in previous}
        sources = {r['source_identity'] for r in records}
        if len(sources) != len(records) or not set(parents).issubset(sources):
            raise ValueError('Do not omit old failures or sources; retain an explicit diagnostic/discarded record.')
        def needs_revision(record):
            parent = parents.get(record['source_identity'])
            # Archival is a publication decision; an unchanged CPU batch must
            # not revive repeats, including during a shared runtime refresh.
            if (parent and parent.get('status') in ('repeat', 'merged')
                    and (cpu_revision(record), cpu_version(record)) ==
                        (cpu_revision(parent), cpu_version(parent))):
                return False
            return (refresh_runtime or parent is None
                    or (cpu_revision(record), cpu_version(record), record.get('status')) !=
                       (cpu_revision(parent), cpu_version(parent), parent.get('status')))
        changed = [record for record in records if needs_revision(record)]
        if not changed:
            continue
        generated = [r for r in changed if r['status'] == 'generated']
        old_sample_sources = {r['source_identity'] for r in previous if r['candidate_id'] in sample_ids}
        preferred = [r for r in generated if r['source_identity'] in old_sample_sources]
        reviewed = [r for r in generated if r['source_identity'] in visible]
        chosen = (preferred or reviewed or generated)[:1] if gated else (reviewed or generated)
        proposals.append({'key': key, 'big_task': category[0], 'small_task': category[1],
            'batch_file': str(path), 'candidate_ids': [r['candidate_id'] for r in chosen],
            'changed_revisions': {r['source_identity']: cpu_revision(r) for r in changed},
            'changed_count': len(changed)})
        changed_sources.update(r['source_identity'] for r in changed)
    return proposals, changed_sources


def defer_cleanup(review, publication_file):
    """Journal a completed publication before releasing its publication fence."""
    path = review / 'task_update_cleanup_pending.json'
    pending = read_json(path, {'publications': []})
    name = str(publication_file)
    if name not in pending['publications']:
        pending['publications'].append(name)
        write_json(path, pending)


def drain_cleanup(campaign, queue, review, attempted):
    """Try each explicitly declared journal once, retaining active predecessor claims."""
    path = review / 'task_update_cleanup_pending.json'
    pending = read_json(path)
    if not pending:
        return
    remaining = []
    for name in pending['publications']:
        if name not in attempted:
            publication = read_json(name)
            if not publication or not publication.get('complete'):
                raise ValueError('Deferred cleanup requires a completed publication: ' + name)
            previous = read_json(Path(publication['output_dir']) / 'obsolete_cache_cleanup.json', {})
            cleaned = cleanup(campaign, queue, publication)
            attempted[name] = {'result': cleaned,
                'removed_png_files': cleaned['removed_png_files'] - previous.get('removed_png_files', 0),
                'freed_bytes': cleaned['freed_bytes'] - previous.get('freed_bytes', 0)}
        if not attempted[name]['result']['complete']:
            remaining.append(name)
    if remaining != pending['publications']:
        if remaining:
            write_json(path, {'publications': remaining})
        else:
            path.unlink(missing_ok=True)


def update(campaign, queue, batch_files, *, refresh_runtime=False, dry_run=False):
    """Resume enqueue independently of old-job cleanup, then apply current CPU changes."""
    review = campaign / 'review_catalogue'
    if dry_run:
        rows, changed = plan_updates(campaign, queue, batch_files, refresh_runtime=refresh_runtime)
        return {'changed_sources': len(changed), 'queued_representatives': sum(len(r['candidate_ids']) for r in rows),
                'categories': len(rows), 'dry_run': True, 'unchanged': not rows}

    pending_file = review / 'task_update_pending.json'
    pending = read_json(pending_file)
    # Older updaters retained this fence even after publication and enqueue finished.
    if pending and read_json(Path(pending['output_dir']) / 'publication.json', {}).get('complete'):
        defer_cleanup(review, Path(pending['output_dir']) / 'publication.json')
        pending_file.unlink()
        pending = None
    attempted, published = {}, []
    drain_cleanup(campaign, queue, review, attempted)

    def publish_pending(value):
        output = Path(value['output_dir'])
        result = publish(campaign, queue, proposal_rows=value['proposals'], output_dir=output,
                         changed_sources=set(value['changed_sources']), frozen=value['frozen'])
        if not result.get('complete'):
            raise RuntimeError('Publication enqueue is incomplete; its pending journal is retained.')
        defer_cleanup(review, output / 'publication.json')
        pending_file.unlink(missing_ok=True)
        drain_cleanup(campaign, queue, review, attempted)
        result.update(changed_sources=len(value['changed_sources']),
                      cleanup=attempted[str(output / 'publication.json')]['result'])
        write_json(output / 'update_result.json', result)
        published.append((value, result))

    if pending:
        publish_pending(pending)
    # Always plan after resuming: a newer CPU batch may supersede that publication.
    rows, changed = plan_updates(campaign, queue, batch_files, refresh_runtime=refresh_runtime)
    if rows:
        frozen = freeze_working_code(queue)
        update_id = digest({'proposals': rows, 'runtime': frozen['code_commit']})[:24]
        for row in rows:
            row['purpose'] = 'review_task_update_' + update_id
        pending = {'proposals': rows, 'changed_sources': sorted(changed),
            'output_dir': str(campaign / 'review_updates' / update_id), 'frozen': frozen, 'started_at': time.time()}
        write_json(pending_file, pending)
        publish_pending(pending)
    if not published:
        result = {'changed_sources': 0, 'queued_representatives': 0, 'categories': 0,
                  'dry_run': False, 'unchanged': True}
    else:
        CataloguePublisher(campaign, queue, review).tick()
        result = {'changed_sources': len({source for value, _ in published for source in value['changed_sources']}),
            'queued_representatives': sum(len(receipt['samples']) for _, receipt in published),
            'categories': len({row['key'] for value, _ in published for row in value['proposals']}),
            'journal': str(Path(published[-1][0]['output_dir']) / 'update_result.json')}
    if attempted:
        result.update(cleanup_complete=not (review / 'task_update_cleanup_pending.json').exists(),
            removed_png_files=sum(row['removed_png_files'] for row in attempted.values()),
            freed_bytes=sum(row['freed_bytes'] for row in attempted.values()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--queue-root', type=Path, default=ROOT / 'outputs/local_capture_20261014/queue_workers_initialized')
    parser.add_argument('--batch-file', type=Path, action='append')
    parser.add_argument('--refresh-runtime', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    campaign, queue = args.campaign_root.expanduser().resolve(), args.queue_root.expanduser().resolve()
    review = campaign / 'review_catalogue'
    if args.dry_run:
        result = update(campaign, queue, registered_batches(campaign, args.batch_file),
                        refresh_runtime=args.refresh_runtime, dry_run=True)
    else:
        review.mkdir(parents=True, exist_ok=True)
        with (review / 'task_update.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = update(campaign, queue, registered_batches(campaign, args.batch_file),
                            refresh_runtime=args.refresh_runtime)
    print(result, flush=True)


if __name__ == '__main__':
    main()
