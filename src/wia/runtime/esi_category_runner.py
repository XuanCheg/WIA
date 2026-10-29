"""Persistent category conversion, sample inspection, publication and feedback.

Run: python src/wia/runtime/commands/run_esi_category_campaign.py run --campaign-root ROOT
Existing converters batch on CPU. Missing rules get disjoint Codex supervisor
assignments; ordinary eight-GPU CaptureQueue workers render every queued input.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

from wia.runtime.esi_category_capture import (collect_diagnostics, freeze_working_code, publish_review,
    refresh_original_reference, retain_reference_evidence, retain_conversion_history)
from wia.authoring.conversion import ConversionContext, DiagnosticConverter, get_converter
from wia.runtime.esi_category_scheduler import ACTIVE, CategoryScheduler, ROOT, assessment_reference
from wia.runtime.scene_capture_queue import CaptureQueue, digest, publish_json, read_json, write_json


def supervisor_alive(process):
    """Check the recorded process identity; an exited zombie is not still working."""
    if not process.get('pid'):
        return False
    try:
        fields = (Path('/proc') / str(process['pid']) / 'stat').read_text().split(') ', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == process.get('start_ticks')
    except (OSError, IndexError):
        return False


def aggregate_feedback(queue_root, review_file, *, scheduler=None):
    """Route new Repair/Skip/Warning notes to the owning Small Task, once per revision."""
    queue = CaptureQueue(queue_root)
    reviews = read_json(review_file, {'entries': {}})
    groups = {}
    for path in (queue.root / 'category_records').glob('*.json'):
        candidate = read_json(path)
        review = reviews['entries'].get(candidate['candidate_id'], {})
        if review.get('fingerprint') != candidate['fingerprint']:
            continue
        if review.get('decision') not in ('needs_fix', 'skipped') and not (
                review.get('notes') and candidate.get('warning_codes')):
            continue
        pair = (candidate['big_task'], candidate['small_task'])
        key = digest(pair)[:20]
        groups.setdefault(key, {'key': key, 'big_task': pair[0], 'small_task': pair[1], 'items': []})['items'].append({
            'candidate_id': candidate['candidate_id'], 'source_identity': candidate['source_identity'],
            'converter_version': candidate['converter_version'], 'decision': review['decision'],
            'notes': review.get('notes', ''), 'review_revision': review['revision'],
            'warning_codes': candidate.get('warning_codes', [])})
    requests = []
    for key, group in groups.items():
        group['items'].sort(key=lambda r: (r['candidate_id'], r['review_revision']))
        request_id = key + '_' + digest(group['items'])[:24]
        group.update(schema='esi_category_feedback_v2', request_id=request_id,
                     state='CATEGORY_REPAIR_REQUESTED' if any(i['decision'] == 'needs_fix' for i in group['items']) else 'CATEGORY_GUIDANCE')
        path = queue.root / 'feedback' / (request_id + '.json')
        created = publish_json(path, group)
        requests.append({**group, 'request_file': str(path)})
        if created and scheduler is not None and key in scheduler.jobs():
            job = scheduler.jobs()[key]
            if group['state'] == 'CATEGORY_REPAIR_REQUESTED':
                if job['state'] in ACTIVE:
                    scheduler.update(key, feedback_pending=True, feedback_file=str(path), feedback_revision=request_id)
                else:
                    scheduler.update(key, 'PENDING', feedback_file=str(path), feedback_revision=request_id,
                                     revision=job.get('revision', 0) + 1, iterations=0, batch_file=None)
            else:
                scheduler.update(key, guidance_file=str(path))
    return requests


class CategoryRunner:
    def __init__(self, campaign_root, *, queue_root=None, corpus=None, config=None):
        self.scheduler = CategoryScheduler(campaign_root, corpus=corpus, config=config)
        self.root = self.scheduler.root
        self.config = self.scheduler.config
        previous = read_json(self.root / 'campaign.json', {})
        self.queue_root = Path(queue_root or previous.get('queue_root') or self.root / 'capture_queue').expanduser().resolve()
        self.children = {}
        self.frozen = None
        write_json(self.root / 'campaign.json', {'schema': 'esi_category_campaign_v2',
            'queue_root': str(self.queue_root), 'campaign_root': str(self.root),
            'supervisor_model': 'gpt-6.1-sol', 'reasoning_effort': 'xhigh',
            'supervisor_slots': 3, 'capture_workers': 8,
            'report_root': str(ROOT / 'explore_to_achieve/category_reports')})

    def converter(self, job):
        assignment = self.scheduler.assignment(job)
        pair = (job['big_task'], job['small_task'])
        return get_converter(*pair, plugins={pair: assignment['module'] + ':' + assignment['class_name']})

    def dispatch(self, job, *, assessment_review=None):
        """Launch an owned conversion task or a later, nonbinding evidence review."""
        child_key = ('review_' if assessment_review else '') + job['key']
        directory = (Path(assessment_review['directory']) if assessment_review else
                     self.root / 'supervisors' / job['key'])
        directory.mkdir(parents=True, exist_ok=True)
        running = read_json(directory / 'process.json', {})
        if running.get('pid'):
            if supervisor_alive(running):
                return running
            if child_key in self.children:
                return running  # Reap this finished child before creating another supervisor.
        assignment = self.scheduler.assignment(job)
        prompt = ('Implement the exact category assignment below. Read its advice, source module, source JSON and shared API first. '
            'Read the supplied assessment when available as supplementary historical information; it may be incomplete or wrong. '
            'The implementation_spec and current user instructions remain authoritative. Assessment conclusions are not mandatory. '
            'Corroborate claims with current code/source/evidence, and record conflicts or uncertainty without blindly changing justified rules. '
            'Do not modify shared code, source data, other categories or root research_log.md. All project content is English. '
            'Implement ConverterBase inspect/convert/convert_all, using original_scene/physical_task and meaningful private native predicates. '
            'Every source must have success or failure; failures enter Review. Autonomous evidence-based reversible L1/L2 rules are authorized. '
            'The user prioritizes actual operations in every Small Task and authorizes imaginative source-faithful operation '
            'recipes, including multiple variants. Reuse materials while adding physical interaction or navigation; QA remains auxiliary. '
            'Use existing typed evaluators or the category_operation API, with native adapters in your owned converter module. '
            'For omitted original doors, lamps, actors, phases or other mechanisms, use the owned category_initialization API '
            'only at scene initialization, with source provenance and L2 warnings; preserve the empty original variant. '
            'Robot-free sources can add the installed R1 using add_existing_r1 then an evidenced robot_pose. '
            'Enable original manipulation objects with object_physics, accounting for source postprocessing that may reset visual_only. '
            'Qualification/evaluators only read measured native state; all actions use existing guarded robot controls. '
            'Uncertainty must retain assumption_review_required with doubt/chosen_assumption/alternatives/scope/risk. '
            'Do not invoke GPU workers or manually edit generated YAML. CPU batch and stratified image evidence inspection belong to this category. '
            'Use /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python and bound library threads to2. '
            'Read current sample_evidence.json under batch_root when available; make at most two targeted rule iterations. '
            'Write only the owned Small Task.md; attribute historical evidence. '
            'If every faithful physical goal is unsupported, implement a converter returning explicit not_convertible_under_L1_L2 diagnostic records. '
            'Return supervisor_result.json in batch_root with decision=publish or revise, inspected_candidates, evidence, limitations, converter_version and feedback_revision copied from assignment. '
            'Do not commit or spawn agents.\n' + json.dumps(assignment, indent=2))
        if assessment_review:
            assignment.update(work_type='supplementary_assessment_review',
                owned_files=[job['report_file']], review=assessment_review)
            prompt = ('All scheduled CPU conversions have finished or reached an explicit diagnostic terminal state. '
                'Review only this assigned category, including completed conversions and ongoing native captures. '
                'Include all earlier candidate revisions and every existing_task_files entry as read-only previous tasks. '
                'Assess whether the task provides meaningful actual operations, whether several source-faithful recipes '
                'are useful, and whether independent native terminal predicates and initial QA boundaries are correct. '
                'Read implementation_spec and the supplementary assessment, current converter/builder, complete batch, '
                'CPU validation, sample_evidence.json, previous task/scene definitions and current CaptureQueue diagnostics. '
                'The assessment may be incomplete or wrong; its conclusions are not mandatory to follow. '
                'Compare claims with actual current code/source/evidence, distinguish historical claims from measured current '
                'results, and flag uncertainty, unavailable artifacts and semantic differences. '
                'Do not change implementation, generated scenes, source data, answers, approvals, other category reports, '
                'root research_log.md or running workloads. Main reviews any recommended implementation correction. '
                'Append an English assessment comparison to the owned Small Task.md. '
                'Write review.result_file as JSON with fingerprint copied exactly, status=reviewed, findings, '
                'recommended_changes, evidence, limitations, and report_file. Do not claim robot success or human approval. '
                'This comparison must not block pending/failed candidates from Review or ongoing screenshots. '
                'Do not launch GPU work, commit or spawn agents.\n' + json.dumps(assignment, indent=2))
        write_json(directory / 'assignment.json', assignment)
        (directory / 'prompt.txt').write_text(prompt)
        command = ['codex', 'exec', '-m', assignment['model'], '-c', 'model_reasoning_effort="xhigh"',
            '-c', 'approval_policy="never"', '-s', 'danger-full-access', '-C', str(ROOT),
            '--json', '--output-last-message', str(directory / 'last_message.txt'), '-']
        import os
        environment = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
        with (directory / 'events.jsonl').open('a') as log:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log,
                                     cwd=ROOT, env=environment, start_new_session=True, text=True)
        child.stdin.write(prompt)
        child.stdin.close()
        start = (Path('/proc') / str(child.pid) / 'stat').read_text().split(') ', 1)[1].split()[19]
        running = {'pid': child.pid, 'start_ticks': start, 'command': command, 'started_at': time.time(),
                   'category': [job['big_task'], job['small_task']], 'log_file': str(directory / 'events.jsonl'),
                   'conversion_iteration': job.get('iterations', 0),
                   'feedback_revision': job.get('feedback_revision')}
        write_json(directory / 'process.json', running)
        self.children[child_key] = child
        return running

    def assessment_review_step(self, *, supervisors='codex'):
        """Run at most three report comparisons after the whole scheduled CPU queue."""
        reference = assessment_reference()
        path = self.root / 'assessment_review.json'
        if not reference['available']:
            value = {'state': 'REFERENCE_UNAVAILABLE', 'reference': reference, 'categories': []}
            write_json(path, value)
            return value
        jobs = [j for j in self.scheduler.jobs().values() if j['state'] != 'DEFERRED' or
                self.scheduler.groups[j['key']].get('existing_task_files')]
        unfinished = [j['key'] for j in jobs if j['state'] in
            ('PENDING', 'ASSIGNED', 'CONVERTER_READY', 'REVISING')]
        cpu_processes = [p for p in (self.root / 'supervisors').glob('*/process.json')
                         if supervisor_alive(read_json(p, {}))]
        if unfinished or cpu_processes:
            value = {'state': 'WAITING_FOR_CPU_COMPLETION', 'reference': reference,
                'unfinished_categories': unfinished, 'running_cpu_supervisors': len(cpu_processes),
                'note': 'Reading the reference during conversion is separate from the requested final comparison.'}
            write_json(path, value)
            return value
        comparisons, active = [], 0
        for job in jobs:
            manifest = read_json(job.get('batch_file') or self.root / 'missing', {})
            fingerprint = digest({'assessment': reference['sha256'], 'category': job['key'],
                'converter_version': manifest.get('converter_version'),
                'candidates': [r['candidate_id'] for r in manifest.get('records', [])],
                'existing_tasks': self.scheduler.groups[job['key']].get('existing_task_files', []),
                'feedback_revision': job.get('feedback_revision')})
            directory = self.root / 'assessment_reviews' / job['key'] / fingerprint[:24]
            review = {'category_key': job['key'], 'fingerprint': fingerprint,
                'directory': str(directory), 'result_file': str(directory / 'result.json'),
                'reference': reference, 'queue_root': str(self.queue_root),
                'batch_file': job.get('batch_file'), 'report_file': job['report_file']}
            review['existing_task_files'] = self.scheduler.groups[job['key']].get('existing_task_files', [])
            result = read_json(review['result_file'], {})
            process = read_json(directory / 'process.json', {})
            live = supervisor_alive(process)
            if live:
                active += 1
            if result.get('fingerprint') == fingerprint and result.get('status') == 'reviewed':
                review.update(state='REVIEWED', result=result)
            elif live:
                review.update(state='RUNNING', process=process)
            elif process:
                review.update(state='FAILED', reason='Review supervisor exited without a matching comparison result.', process=process)
            else:
                review['state'] = 'READY'
            comparisons.append((job, review))
        for job, review in comparisons:
            if review['state'] == 'READY' and active < 3 and supervisors == 'codex':
                process = self.dispatch(job, assessment_review=review)
                review.update(state='RUNNING', process=process)
                active += 1
        counts = dict(Counter(review['state'] for _, review in comparisons))
        value = {'state': 'RUNNING' if counts.get('RUNNING') or counts.get('READY') or active else
                 'PARTIAL' if counts.get('FAILED') else 'COMPLETE',
            'reference': reference, 'counts': counts, 'categories': [review for _, review in comparisons],
            'note': 'Supplementary comparisons do not certify physical success or override the implementation specification.'}
        write_json(path, value)
        return value

    def generate(self, job, *, converter=None):
        converter = converter or self.converter(job)
        if converter is None:
            return None
        self.scheduler.update(job['key'], 'CONVERTER_READY', converter_version=converter.version)
        root = self.root / 'categories' / job['key']
        group = self.scheduler.groups[job['key']]
        conversion_config = {**self.config.get('conversion', {}), **job.get('conversion_config', {})}
        context = ConversionContext(root, self.config.get('seed', 0), conversion_config, group['advice'])
        old = read_json(root / 'batch_manifest.json')
        if old:
            old_id = digest(old)[:24]
            publish_json(root / 'batches' / (old_id + '.json'), old)
        manifest = converter.convert_all(self.scheduler.sources(job['key']), context)
        if old:
            parents = {r['source_identity']: r['candidate_id'] for r in old['records']}
            for row in manifest['records']:
                parent = parents.get(row['source_identity'])
                if parent and parent != row['candidate_id']:
                    row['parent_candidate_id'] = parent
        manifest['protected_sources'] = group['protected_sources']
        samples = converter.sample(manifest, {'count': self.config.get('sample_count', 5),
            'seed': context.seed, 'previous_failures': job.get('previous_failed_samples', [])})
        manifest['sample_ids'] = samples
        write_json(root / 'batch_manifest.json', manifest)
        publish_json(root / 'batches' / (digest(manifest)[:24] + '.json'), manifest)
        self.scheduler.update(job['key'], 'BATCH_GENERATED', batch_file=str(root / 'batch_manifest.json'),
            counts=manifest['counts'], sample_ids=samples, generated_count=len(manifest['records']))
        return manifest

    def step(self, *, supervisors='codex', publish_all=False):
        for key, child in list(self.children.items()):
            code = child.poll()
            if code is None:
                continue
            del self.children[key]
            if key.startswith('review_'):
                continue
            job = self.scheduler.jobs()[key]
            process = read_json(self.root / 'supervisors' / key / 'process.json', {})
            details = {'supervisor_exit_code': code}
            if code == 0 and job.get('feedback_revision') == process.get('feedback_revision'):
                details['feedback_applied_revision'] = job.get('feedback_revision')
            self.scheduler.update(key, **details)
        assignments = self.scheduler.fill_slots()
        jobs = [j for j in self.scheduler.jobs().values() if j['state'] in ('ASSIGNED', 'REVISING')]
        generated = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {}
            for job in jobs:
                process = read_json(self.root / 'supervisors' / job['key'] / 'process.json', {})
                if supervisor_alive(process):
                    continue  # Do not snapshot or batch a converter while its owner is still editing it.
                revision_needed = (job['state'] == 'REVISING' and process.get('conversion_iteration') != job.get('iterations', 0))
                feedback_needed = (job.get('feedback_file') and
                    process.get('feedback_revision') != job.get('feedback_revision'))
                if revision_needed or feedback_needed:
                    if supervisors == 'codex':
                        self.dispatch(job)
                    continue
                try:
                    converter = self.converter(job)
                except Exception as error:
                    converter = DiagnosticConverter((job['big_task'], job['small_task']),
                        f'{type(error).__name__}: {error}', 'category_implementation_load')
                if converter is None:
                    if not process.get('pid'):
                        if supervisors == 'codex':
                            self.dispatch(job)
                        continue
                    converter = DiagnosticConverter((job['big_task'], job['small_task']),
                        'Category supervisor exited without an importable converter. See its preserved process and events log.',
                        'category_supervision')
                futures[pool.submit(self.generate, job, converter=converter)] = job
            for future, job in futures.items():
                try:
                    if future.result() is not None:
                        generated.append(job['key'])
                except Exception as error:
                    reason = f'{type(error).__name__}: {error}'
                    self.scheduler.update(job['key'], error=reason)
                    self.generate(job, converter=DiagnosticConverter((job['big_task'], job['small_task']),
                        reason, 'category_batch_generation'))
        ready = [j for j in self.scheduler.jobs().values() if j['state'] == 'BATCH_GENERATED']
        snapshot_error = None
        if ready:
            for attempt in range(2):
                try:
                    self.frozen = freeze_working_code(self.queue_root)
                    break
                except (OSError, RuntimeError) as error:
                    self.frozen = None
                    snapshot_error = f'{type(error).__name__}: {error}'
            if self.frozen:
                snapshot_error = None
        for job in ready:
            manifest = read_json(job['batch_file'])
            reference = refresh_original_reference(manifest, self.queue_root) if self.config.get('refresh_original_reference', True) else None
            publish_review(manifest, self.queue_root, selected=job['sample_ids'], frozen=self.frozen,
                priority=100, report_file=job['report_file'], batch_file=job['batch_file'],
                snapshot_error=snapshot_error)
            self.scheduler.update(job['key'], 'SAMPLE_CAPTURING', sample_started_at=time.time(),
                original_reference_request=reference, frozen_code=self.frozen, snapshot_error=snapshot_error)
        for job in self.scheduler.jobs().values():
            if job['state'] not in ('SAMPLE_CAPTURING', 'SAMPLE_INSPECTION'):
                continue
            manifest = read_json(job['batch_file'])
            sample_records = [read_json(self.queue_root / 'category_records' / (key + '.json')) for key in job['sample_ids']]
            evidence = [{**r, **collect_diagnostics(r, self.queue_root,
                reference_queues=self.config.get('reference_queues', []))} for r in sample_records]
            write_json(Path(job['batch_file']).parent / 'sample_evidence.json', {'samples': evidence,
                'note': 'Historical/reference/partial images are labelled; native execution and full-category feasibility remain unverified.'})
            terminal = all(r['capture']['state'] in ('REVIEW_PENDING', 'CACHED', 'FAILED', 'STALE', 'CANCELLED', 'REVIEW_ONLY') for r in evidence)
            elapsed = time.time() - job.get('sample_started_at', time.time())
            timed_out = elapsed >= self.config.get('sample_wait_s', 900)
            decision = read_json(Path(job['batch_file']).parent / 'supervisor_result.json', {})
            if decision.get('converter_version') != job.get('converter_version'):
                decision = {}
            if not (terminal or timed_out or publish_all or decision.get('decision') == 'publish'):
                continue
            self.scheduler.update(job['key'], 'SAMPLE_INSPECTION')
            iterations = job.get('iterations', 0)
            if decision.get('decision') == 'revise' and iterations < self.config.get('max_iterations', 2):
                self.scheduler.update(job['key'], 'REVISING', iterations=iterations + 1,
                    previous_failed_samples=[r['candidate_id'] for r in evidence if r['status'] != 'generated'])
                self.frozen = None
                continue
            # No isolated failure or missing GPU screenshot vetoes category Review.
            ids = {r['candidate_id']: r for r in evidence}
            for row in manifest['records']:
                row['sample_evidence'] = ids.get(row['candidate_id'], {}).get('diagnostic_images', [])
            results = publish_review(manifest, self.queue_root, frozen=job.get('frozen_code') or self.frozen,
                report_file=job['report_file'], batch_file=job['batch_file'], snapshot_error=job.get('snapshot_error'))
            retain_reference_evidence(manifest, self.queue_root, self.config.get('reference_queues', []))
            retain_conversion_history(manifest, self.queue_root, Path(job['batch_file']).parent)
            for row, result in zip(manifest['records'], results):
                row['review_record_id'] = result['review_record_id']
            manifest['review_submitted'] = True
            manifest['sample_inspection'] = {'supervisor': decision or None, 'timeout': timed_out,
                'native_samples_complete': sum(r['native_complete'] for r in evidence),
                'unverified': not terminal, 'evidence_file': str(Path(job['batch_file']).parent / 'sample_evidence.json')}
            write_json(job['batch_file'], manifest)
            self.scheduler.update(job['key'], 'REVIEW_SUBMITTED', review_count=len(results),
                iterations=iterations, sample_inspection=manifest['sample_inspection'])
        review_file = self.config.get('review_file')
        if review_file:
            aggregate_feedback(self.queue_root, review_file, scheduler=self.scheduler)
        self.assessment_review_step(supervisors=supervisors)
        return {'assignments': assignments, 'generated': generated, **self.status()}

    def status(self):
        value = self.scheduler.summary()
        queue = CaptureQueue(self.queue_root)
        records = [read_json(p) for p in (queue.root / 'category_records').glob('*.json')]
        value.update(capture_queue=queue.summary(), review_count=len(records),
            instance_counts=dict(Counter(r['status'] for r in records)),
            warning_count=sum(bool(r['warning_codes']) for r in records),
            failed_count=sum(r['status'] != 'generated' for r in records),
            supervisor_processes=[read_json(p) for p in (self.root / 'supervisors').glob('*/process.json')],
            assessment_reference=assessment_reference(),
            assessment_review=read_json(self.root / 'assessment_review.json', {}))
        write_json(self.root / 'status.json', value)
        return value

    def refresh_evidence(self):
        """Reconcile completed batches after restart without reconversion or approvals."""
        requests = []
        for job in self.scheduler.jobs().values():
            if job['state'] != 'REVIEW_SUBMITTED' or not job.get('batch_file'):
                continue
            manifest = read_json(job['batch_file'])
            request = refresh_original_reference(manifest, self.queue_root)
            if request:
                requests.append(request)
                write_json(Path(job['batch_file']).parent / 'original_reference_request.json', request)
            retain_reference_evidence(manifest, self.queue_root, self.config.get('reference_queues', []))
            retain_conversion_history(manifest, self.queue_root, Path(job['batch_file']).parent)
        return {'original_reference_requests': requests, **self.status()}
