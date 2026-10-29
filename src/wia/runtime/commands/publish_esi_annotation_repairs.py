#!/usr/bin/env python3
"""Publish one revised representative per annotated subclass and retire old cards.

Run with behavior Python from the project root:
  python src/wia/runtime/commands/publish_esi_annotation_repairs.py --campaign-root ~/esi_category_conversion/CAMPAIGN --watch
Only the selected representatives are enqueued. Other queue tokens are untouched.
Cleanup uses declared old receipts, never image validation or screenshot scans.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.runtime.esi_category_capture import freeze_working_code, publish_record
from wia.runtime.scene_capture_queue import digest, read_json, write_json


def proposals(campaign):
    output = campaign / 'annotation_repair_20261015'
    rows = read_json(output / 'manipulation/representatives.json')['representatives']
    rows += [{**row, **row['representative']} for row in
             read_json(output / 'perception/cpu_revision_summary.json')['categories']]
    rows += [{'big_task': row['category'][0], 'small_task': row['category'][1],
              'batch_file': row['batch_manifest'], **row['representative_proposal']}
             for row in read_json(output / 'reflection_temporal/cpu_batch_summary.json')]
    temporal = read_json(campaign / 'two_phase_20261015/preparation_result.json')
    rows.append({'big_task': 'Temporal Understanding', 'small_task': 'Unobserved Change',
                 'batch_file': temporal['batch_file'],
                 'candidate_id': temporal['selected_sample_candidate_id']})
    if len(rows) != 18 or len({(r['big_task'], r['small_task']) for r in rows}) != 18:
        raise ValueError('The annotated repair contains exactly eighteen distinct subclasses.')
    return rows


def publish(campaign, queue, *, proposal_rows=None, output_dir=None, changed_sources=None, frozen=None):
    output = Path(output_dir) if output_dir else campaign / 'annotation_repair_20261015'
    review = campaign / 'review_catalogue'
    receipt_file = output / 'publication.json'
    if receipt_file.exists():
        result = read_json(receipt_file)
        if not result.get('complete'):
            return enqueue_samples(result, read_json(campaign / 'inventory.json')['categories'], queue, receipt_file)
        return result
    snapshot = read_json(review / 'snapshot.json')
    reviews_path = ROOT / 'outputs/local_capture_20261014/gallery/reviews.json'
    annotations = read_json(reviews_path, {'entries': {}, 'history': []})
    write_json(output / 'catalogue_before_publication.json', snapshot)
    write_json(output / 'reviews_before_retirement.json', annotations)
    revisions = read_json(review / 'category_revisions.json', {'categories': {}})
    write_json(output / 'category_revisions_before_publication.json', revisions)
    inventory = read_json(campaign / 'inventory.json')['categories']
    keys = {(row['big_task'], row['small_task']): row['key'] for row in inventory}
    old_rows = {row['source']: row for row in snapshot.get('scenes', []) + snapshot.get('diagnostics', [])}
    old_failed = {row['source']: row for row in snapshot.get('diagnostics', [])}
    feedback = {}
    for key, value in annotations['entries'].items():
        if value.get('source') and value.get('notes'):
            feedback[value['source']] = {**value, 'previous_candidate_id': key,
                'scope': 'Feedback on the preceding revision; no approval is inherited.'}
    frozen = frozen or freeze_working_code(queue)
    categories, selected, retired_ids, cleanup_jobs = [], [], set(), {}
    for proposal in proposals(campaign) if proposal_rows is None else proposal_rows:
        category = proposal['big_task'], proposal['small_task']
        key = keys[category]
        previous_file = revisions['categories'][key]['new_batch_file']
        previous = read_json(previous_file)
        parents = {record['source_identity']: record for record in previous['records']}
        retired_ids.update(record['candidate_id'] for record in previous['records']
                           if changed_sources is None or record['source_identity'] in changed_sources)
        batch = read_json(proposal['batch_file'])
        records, chosen = [], []
        wanted = set(proposal.get('candidate_ids', [proposal.get('candidate_id')]))
        for raw in batch['records']:
            record = deepcopy(raw)
            source = record['source_identity']
            parent = parents.get(source, {})
            if changed_sources is not None and source not in changed_sources:
                records.append(deepcopy(parent or raw))
                continue
            revision = digest({'conversion_revision': record['revision'],
                'runtime': frozen['code_commit'], 'purpose': proposal.get('purpose', 'annotated_subclass_repair_20261015')})[:24]
            record.update(cpu_candidate_id=raw['candidate_id'], cpu_conversion_revision=raw['revision'],
                candidate_id='category_' + revision, revision=revision,
                parent_candidate_id=parent.get('candidate_id'),
                runtime_revision=frozen['code_commit'], prior_feedback=[feedback[source]] if source in feedback else [])
            if source in old_failed and record['status'] == 'generated':
                old = old_failed[source]
                record.update(native_recheck_required=True,
                    native_recheck_evidence={'previous_candidate_id': old['id'],
                        'previous_status': old.get('instance_status'),
                        'reason': 'Previous revision: ' + (old.get('failure') or old.get('instance_status', 'unqualified'))
                            + ' The repaired revision is awaiting native measurement.'})
                record['warning_codes'] = sorted(set(record.get('warning_codes', [])) |
                                                  {'previous_revision_native_recheck_pending'})
                prior_job = old.get('capture', {}).get('job_id')
                prior_status = read_json(queue / 'jobs' / str(prior_job) / 'status.json', {})
                if prior_status.get('native_log'):
                    record['diagnostic_logs'] = sorted(set(record.get('diagnostic_logs', [])) | {prior_status['native_log']})
            if category == ('Temporal Understanding', 'Unobserved Change'):
                record['warning_codes'] = sorted(set(record.get('warning_codes', [])) |
                                                  {'phase_1_mounted_view_coverage_requires_review'})
            is_sample = raw['candidate_id'] in wanted
            if is_sample or record['status'] != 'generated' or record.get('native_recheck_required'):
                directory = output / 'runtime_records' / record['candidate_id']
                directory.mkdir(parents=True, exist_ok=True)
                original_dir = Path(raw['diagnostic_dir'])
                qa = original_dir / 'private_qa.json'
                if qa.exists() and not (directory / qa.name).exists():
                    (directory / qa.name).symlink_to(qa)
                record.update(diagnostic_dir=str(directory), cpu_diagnostic_dir=str(original_dir))
                write_json(directory / 'record.json', record)
            records.append(record)
            if is_sample:
                chosen.append(record)
        if any(sample['status'] != 'generated' for sample in chosen) or len(chosen) != len(wanted - {None}):
            raise ValueError('A selected representative must have a pending physical task: ' + str(category))
        batch_file = output / 'runtime_batches' / key / 'batch_manifest.json'
        batch.update(records=records, source_cpu_batch=str(proposal['batch_file']),
                     runtime_revision=frozen['code_commit'], review_submitted=True)
        write_json(batch_file, batch)
        for sample in chosen:
            selected.append({'key': key, 'big_task': category[0], 'small_task': category[1],
                'candidate_id': sample['candidate_id'], 'cpu_candidate_id': sample['cpu_candidate_id'],
                'source_identity': sample['source_identity'], 'batch_file': str(batch_file)})
        categories.append({'key': key, 'big_task': category[0], 'small_task': category[1],
                           'new_batch_file': str(batch_file),
                           'selected_sample_candidate_id': chosen[0]['candidate_id'] if chosen else None,
                           'complete': True, 'counts': batch['counts'], 'source_count': len(records)})
        previous_sample = revisions['categories'][key].get('selected_sample_candidate_id')
        if previous_sample:
            prior_card = read_json(queue / 'category_records' / (previous_sample + '.json'), {})
            if (prior_card.get('job_id') and (changed_sources is None
                    or prior_card.get('source_identity') in changed_sources)):
                # An in-flight predecessor has no completed Review row yet.
                # Journal its declared job now; cleanup waits for claim release.
                cleanup_jobs[prior_card['job_id']] = {
                    'replacement_candidate_id': chosen[0]['candidate_id'] if chosen else None,
                    'category_key': key, 'replacement_published': True}
        for row in old_rows.values():
            if ((row['category'], row['subcategory']) == category
                    and (changed_sources is None or row['source'] in changed_sources)):
                retired_ids.add(row['id'])
                job = row.get('capture', {}).get('job_id')
                if job:
                    cleanup_jobs[job] = {'replacement_candidate_id': chosen[0]['candidate_id'] if chosen else None,
                                         'category_key': key, 'replacement_published': True}
    old_samples = read_json(queue / 'capture_sample_gate.json')
    write_json(output / 'sample_gate_before_publication.json', old_samples)
    old_selection = read_json(Path(old_samples['selection_file']))['samples']
    replacements = {}
    for row in selected:
        replacements.setdefault(row['key'], []).append(row)
    samples = []
    for row in old_selection:
        replacement = replacements.pop(row['key'], None)
        samples.extend(replacement if replacement is not None else ([] if row['candidate_id'] in retired_ids else [row]))
    samples.extend(row for group in replacements.values() for row in group)
    write_json(output / 'samples.json', {'samples': samples})
    gate = {**old_samples, 'selected_candidate_ids': list(dict.fromkeys(
                [row['candidate_id'] for row in samples] + old_samples.get('reference_only_candidates', []))),
        'selection_file': str(output / 'samples.json'), 'selection_pending': False,
        'pending_category_revisions': [], 'updated_at': time.time(),
        'reason': 'User requested one corrected representative per modified subclass; remaining queue unchanged.'}
    write_json(queue / 'capture_sample_gate.json', gate)
    policy = read_json(queue / 'capture_runtime_policy.json', {})
    policy.update(allowed_code_commits=sorted(set(policy.get('allowed_code_commits', []) if changed_sources is not None else []) | {frozen['code_commit']}),
        review_allowed_code_commits=sorted(set(policy.get('review_allowed_code_commits', [])) | {frozen['code_commit']}),
        reason=gate['reason'], updated_at=time.time())
    write_json(queue / 'capture_runtime_policy.json', policy)
    for category in categories:
        revisions['categories'][category['key']] = {**category, 'updated_at': time.time()}
    write_json(review / 'category_revisions.json', revisions)
    retired = read_json(review / 'retired_candidates.json', {'candidates': {}})
    for key in retired_ids:
        retired['candidates'][key] = {'reason': 'Superseded by annotated whole-subclass repair.', 'retired_at': time.time()}
    write_json(review / 'retired_candidates.json', retired)
    removed_reviews = {key: value for key, value in annotations['entries'].items() if key in retired_ids}
    annotations['entries'] = {key: value for key, value in annotations['entries'].items() if key not in retired_ids}
    annotations['history'] = [row for row in annotations.get('history', []) if row['id'] not in retired_ids]
    write_json(reviews_path, annotations)
    result = {'complete': False, 'code': frozen, 'categories': categories, 'samples': selected, 'output_dir': str(output),
        'retired_candidate_count': len(retired_ids), 'removed_old_review_entries': len(removed_reviews),
        'prior_annotations_archived': str(output / 'reviews_before_retirement.json'),
        'cleanup_jobs': cleanup_jobs, 'cleanup_on_submission': True,
        'other_queue_tokens_changed': False, 'published_at': time.time()}
    # The journal precedes enqueue so an interrupted publication can be recovered.
    write_json(receipt_file, result)
    return enqueue_samples(result, inventory, queue, receipt_file)


def enqueue_samples(result, inventory, queue, receipt_file):
    for row in result['samples']:
        if row.get('job_id'):
            continue
        batch = read_json(row['batch_file'])
        record = next(r for r in batch['records'] if r['candidate_id'] == row['candidate_id'])
        publication = publish_record(record, queue, frozen=result['code'], priority=1000,
            report_file=next(g['report_file'] for g in inventory if g['key'] == row['key']),
            batch_file=row['batch_file'])
        row.update(job_id=publication.get('job_id'), capture_submission_error=publication.get('capture_submission_error'))
        write_json(receipt_file, result)
    result['complete'] = True
    write_json(receipt_file, result)
    return result


def cleanup(campaign, queue, publication):
    output = Path(publication.get('output_dir', campaign / 'annotation_repair_20261015'))
    receipt_file = output / 'obsolete_cache_cleanup.json'
    prior = read_json(receipt_file, {'completed_jobs': [], 'removed_png_files': 0, 'freed_bytes': 0})
    completed = set(prior['completed_jobs'])
    replacements = {row['candidate_id']: row for row in publication['samples']}
    for job, info in publication['cleanup_jobs'].items():
        if job in completed:
            continue
        replacement = replacements.get(info['replacement_candidate_id'])
        if replacement is not None:
            current = read_json(queue / 'jobs' / str(replacement.get('job_id')) / 'status.json', {})
            ready_states = {'REVIEW_PENDING', 'CACHED', 'FAILED', 'CANCELLED'}
            if publication.get('cleanup_on_submission'):
                ready_states.update(('QUEUED', 'CLAIMED', 'RUNNING', 'CAPTURED'))
            if current.get('state') not in ready_states:
                continue
        elif not info.get('replacement_published'):
            continue
        if (queue / 'claimed' / (job + '.json')).exists():
            continue
        status = read_json(queue / 'jobs' / job / 'status.json', {})
        paths = set(status.get('partial_images', {}).values())
        if status.get('receipt'):
            paths.update(read_json(status['receipt'], {}).get('images', {}).values())
        for attempt in range(1, int(status.get('attempts', 0)) + 1):
            directory = queue / 'jobs' / job / 'attempts' / f'{attempt:02}'
            for name in ('partial_capture.json', 'capture_failure.json', 'preview_complete.json'):
                value = read_json(directory / name, {})
                paths.update(value.get('images', {}).values())
                paths.update(value.get('partial_images', {}).values())
        with (output / 'obsolete_cache_removal.jsonl').open('a') as log:
            for name in sorted(paths):
                path = Path(name)
                if path.suffix.lower() != '.png':
                    continue
                real = path.resolve()
                allowed = (campaign / 'capture_artifacts/jobs' / job).resolve()
                if not real.is_relative_to(allowed):
                    continue
                try:
                    size = real.stat().st_size
                except FileNotFoundError:
                    continue
                log.write(json.dumps({'event': 'remove_superseded_screenshot',
                    'path': str(real), 'old_job_id': job, 'replacement': info['replacement_candidate_id'],
                    'bytes': size, 'recorded_at': time.time()}) + '\n')
                log.flush()
                real.unlink()
                prior['removed_png_files'] += 1
                prior['freed_bytes'] += size
            manifest = read_json(queue / 'jobs' / job / 'input/manifest.json', {})
            signature = manifest.get('capture_signature')
            if signature:
                (queue / 'cache' / (signature + '.json')).unlink(missing_ok=True)
        completed.add(job)
        prior.update(completed_jobs=sorted(completed), updated_at=time.time(),
                     source_assets_tasks_and_diagnostic_logs_preserved=True)
        write_json(receipt_file, prior)
    prior['complete'] = len(completed) == len(publication['cleanup_jobs'])
    write_json(receipt_file, prior)
    return prior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--queue-root', type=Path, default=ROOT / 'outputs/local_capture_20261014/queue_workers_initialized')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    campaign, queue = args.campaign_root.expanduser().resolve(), args.queue_root.expanduser().resolve()
    result = publish(campaign, queue)
    print(f"Published {len(result['samples'])} representatives; other queue tokens unchanged.", flush=True)
    while True:
        cleaned = cleanup(campaign, queue, result)
        if cleaned['complete'] or not args.watch:
            print(f"Obsolete screenshot cleanup: {cleaned}", flush=True)
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
