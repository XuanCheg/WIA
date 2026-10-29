"""Discover ESI Small Tasks and persist at most three category assignments.

Run: python src/wia/runtime/commands/run_esi_category_campaign.py inventory --campaign-root ROOT
This is CPU scheduling; GPU work goes exclusively through CaptureQueue.
"""
from __future__ import annotations

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

from collections import Counter
from contextlib import contextmanager
import ast
import fcntl
from pathlib import Path
import re
import time

import yaml

from wia.authoring.conversion import CONVERTERS, ROOT, SourceInstance
from wia.runtime.scene_capture_queue import digest, publish_json, read_json, sha256, write_json

STATES = {'PENDING', 'ASSIGNED', 'CONVERTER_READY', 'BATCH_GENERATED', 'SAMPLE_CAPTURING',
          'SAMPLE_INSPECTION', 'REVISING', 'REVIEW_SUBMITTED', 'PARTIAL', 'DEFERRED',
          'FAILED', 'NEEDS_HUMAN_DECISION'}
ACTIVE = {'ASSIGNED', 'CONVERTER_READY', 'BATCH_GENERATED', 'SAMPLE_CAPTURING',
          'SAMPLE_INSPECTION', 'REVISING'}


def slug(value):
    return re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')


def category_key(big_task, small_task):
    return digest([big_task, small_task])[:20]


def report_path(big_task, small_task):
    def safe(value):
        return value.replace('/', '%2F').replace('\\', '%5C')
    return ROOT / 'explore_to_achieve/category_reports' / safe(big_task) / (safe(small_task) + '.md')


def assessment_reference():
    """Describe optional historical evidence without elevating it to a specification."""
    path = ROOT / 'report(1).md'
    available = path.is_file()
    return {'path': str(path), 'available': available,
        'sha256': sha256(path) if available else None,
        'role': 'supplementary_historical_evidence', 'conclusions_binding': False,
        'reading_requested': available,
        'implementation_spec': str(ROOT / 'esi_category_parallel_conversion_spec_v2.md'),
        'interpretation': 'The assessment may be incomplete or incorrect. Compare its claims with current source code, '
            'source instances and measured evidence. Record conflicts and uncertainty; do not change a justified '
            'converter solely to match the assessment. Its linked raw artifacts may be unavailable locally.'}


def recommendation(big, small):
    """Versioned starting advice; historical claims require current corroboration."""
    reference = assessment_reference()
    warnings = {
        'Action Order Inference': 'Native tabletop support was measured; IK stopped before grasp. Stack execution remains unverified.',
        'Structural Enclosure': 'Five reveal links were hidden; real door opening was not tested. Restore source links before claiming opening.',
        'Material Transparency': 'Original pickup/pour effects reposition rigid objects and change visual_only; native extraction and wrist-driven pouring remain unverified.',
        'Physical Contact': 'Source labels measure image-mask adjacency, and a yes-labelled pair had no native contact.',
        'Deformable': 'Native cloth camera/attachment initialization failed. This does not imply new assets are required.',
        'Liquid Volume': 'Source fill completion reported zero in-container particles; genuine filling/transfer remains unverified.',
        'Unobserved Change': 'Active loader omits phase contents; the original replay helper exists and should restore them.',
        'Agent Observation': 'The active loader uses the same R1 as subject and observer; temporal mechanism is incomplete.',
        'Illumination Variability': 'Original lamps and illumination schedule were omitted by the active builder.',
        'Merged Observation': 'Visual-only object AABBs overlap; enabling collisions may destroy the counting cluster.',
        'Dimensional Size': 'Upstream changes the head mount; preserve original comparison anchors and sizes. Real grasp is unverified.',
        'Stacking & Stability': 'Independent contact, release, tilt and stability evidence is required; original QA is not physical success.',
    }
    if big == 'Cognitive Mapping':
        warnings[small] = 'Shared native room loads; no complete navigation route was executed. Original maps remain private.'
    mechanism = {
        'Action Sequencing': 'Infer and physically realize the original object support order.',
        'Cognitive Mapping': 'Reach a source-defined destination using robot observations and a privately measured route endpoint.',
        'Enumerative Perception': 'Preserve the original count, visibility and category ambiguity; use a meaningful access/retrieval goal only if feasible.',
        'Metric Comparison': 'Compare source geometry and select the correct original object for a grasp/lift goal.',
        'Perceptual Grounding': 'Identify the obscured or misleading original object and retrieve it using native robot actions.',
        'Physical Dynamics': 'Perform actual placement and measure contact, motion or released stability independently.',
        'Physical Structure': 'Retrieve/place the original item or transfer material with independent native terminal evidence.',
        'Spatial Relations': 'Preserve reference-dependent relations; measure actual final contact or placement separately from QA.',
        'Specular Reflection': 'Retain the original mirror, observers and optical relations; defer if no faithful physical goal is available.',
        'Temporal Understanding': 'Restore original episode phases and legitimate observation history before defining a terminal goal.',
    }
    version = 'esi_reuse_v2_20261014' + ('_reference_' + reference['sha256'][:12] if reference['available'] else '')
    return {'advice_version': version, 'big_task': big, 'small_task': small,
        'original_mechanism': small, 'recommended_goal': mechanism.get(big, 'Inspect the source mechanism before selecting a goal.'),
        'success_predicate': 'Independent native terminal evidence; the category converter must specify the exact predicate.',
        'reuse_level': 'L1', 'l1': 'Reuse native assets, scales, poses, original QA and original task builders.',
        'allowed_l2': 'Evidence-based robot pose or minimal common layout/configuration repair; record reversible rules and assumptions.',
        'preserve': ['source JSON and answers', 'dimensions and pairwise relations', 'original reference camera/observer', 'approved records'],
        'limitations': [warnings.get(small, 'Representative construction evidence does not certify all instances or physical execution.')],
        'verified_facts': ['Committed assessment covers 10 Big Tasks / 29 Small Tasks and 48 diagnostic attempts.'],
        'unverified_hypotheses': ['Category-wide reach, grasp, navigation and final physical success.'],
        'information_boundary': 'Source answers, poses, maps, construction metadata and scorer state are private; Human review images are not Agent sensors.',
        'evidence_sources': [str(ROOT / 'docs/esi_category_exploration_latest.json'), str(ROOT / 'research_log.md'),
                             str(ROOT / 'src/wia/runtime/commands/report_esi_categories.py')] + ([reference['path']] if reference['available'] else []),
        'assessment_reference': reference, 'historical_report_available': reference['available'],
        'evidence_note': 'The supplied report is supplementary historical evidence, not a binding specification or a new trial. '
            'Historical /scratch raw artifacts are not established as available on deep59. '
            'Current builder/source checks and native measurements take precedence over unsupported historical conclusions.'}


def scan_inventory(corpus=None, *, scene_root=None):
    corpus = Path(corpus or WIA_ESI_ROOT / 'dataset/json_clean').resolve()
    groups = {}
    registry_path = WIA_ESI_ROOT / 'src/active_explore/tasks/_registry.py'
    modules = {}
    if registry_path.is_file():
        for node in ast.parse(registry_path.read_text()).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SMALL_TASK_MODULES' for t in node.targets):
                modules = ast.literal_eval(node.value)
    def normalized(value):
        return ' '.join(value.lower().replace('&', 'and').split())
    for path in sorted(corpus.rglob('*.json')):
        parts = path.relative_to(corpus).parts
        if len(parts) != 5:
            continue  # Root aggregate JSONs are not additional source instances.
        big, small = parts[:2]
        key = category_key(big, small)
        group = groups.setdefault(key, {'key': key, 'big_task': big, 'small_task': small, 'sources': [],
            'protected_sources': {}, 'report_file': str(report_path(big, small)),
            'advice': recommendation(big, small), 'existing_ready_scene_count': 0, 'existing_ready_task_count': 0,
            'existing_task_files': []})
        native_module = modules.get((normalized(big), normalized(small)))
        group['source_builder'] = str(WIA_ESI_ROOT / 'src/active_explore' / (native_module.replace('.', '/') + '.py')) if native_module else None
        group['sources'].append(str(path.relative_to(corpus)))
    for path in sorted(Path(scene_root or ROOT / 'explore_to_achieve/scenes').rglob('*.yaml')):
        value = yaml.safe_load(path.read_text()) or {}
        source = value.get('source', {}).get('json', '')
        parts = Path(source).parts
        if len(parts) == 5 and value.get('status') in ('ready', 'discarded'):
            key = category_key(*parts[:2])
            if key in groups:
                groups[key]['protected_sources'][source] = {'status': value['status'], 'scene_file': str(path)}
                groups[key]['existing_ready_scene_count'] += value['status'] == 'ready'
    for path in sorted((ROOT / 'explore_to_achieve/tasks').rglob('*.yaml')):
        value = yaml.safe_load(path.read_text()) or {}
        parts = Path(value.get('source', {}).get('json', '')).parts
        if len(parts) == 5 and value.get('status') == 'ready':
            key = category_key(*parts[:2])
            if key in groups:
                groups[key]['existing_ready_task_count'] += 1
                groups[key]['existing_task_files'].append({'path': str(path), 'sha256': sha256(path)})
    return {'schema': 'esi_category_inventory_v2', 'corpus': str(corpus),
        'big_task_count': len({g['big_task'] for g in groups.values()}),
        'small_task_count': len(groups), 'source_count': sum(len(g['sources']) for g in groups.values()),
        'categories': sorted(groups.values(), key=lambda g: (g['big_task'], g['small_task']))}


class CategoryScheduler:
    def __init__(self, root, *, corpus=None, config=None):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config or read_json(self.root / 'config.json', {})
        if config is not None:
            write_json(self.root / 'config.json', config)
        previous_inventory = read_json(self.root / 'inventory.json', {})
        self.inventory = scan_inventory(corpus or previous_inventory.get('corpus'))
        write_json(self.root / 'inventory.json', self.inventory)
        advice = [g['advice'] for g in self.inventory['categories']]
        recommendations = {'version': advice[0]['advice_version'] if advice else 'esi_reuse_v2_20261014',
            'categories': advice}
        previous = read_json(self.root / 'recommendations.json')
        if previous and previous != recommendations:
            publish_json(self.root / 'recommendation_history' / (digest(previous)[:24] + '.json'), previous)
        write_json(self.root / 'recommendations.json', recommendations)
        self.groups = {g['key']: g for g in self.inventory['categories']}
        with self.lock():
            state = read_json(self.root / 'jobs.json', {'schema': 'esi_category_jobs_v2', 'jobs': {}})
            for g in self.inventory['categories']:
                deferred = (g['big_task'], g['small_task']) == ('Physical Structure', 'Rigid Containment')
                state['jobs'].setdefault(g['key'], {'key': g['key'], 'big_task': g['big_task'],
                    'small_task': g['small_task'], 'state': 'DEFERRED' if deferred else 'PENDING',
                    'reason': 'Existing formal conversions are preserved; opt in to incremental coverage.' if deferred else '',
                    'revision': 0, 'iterations': 0, 'source_count': len(g['sources']),
                    'report_file': g['report_file'], 'updated_at': time.time()})
            write_json(self.root / 'jobs.json', state)

    @contextmanager
    def lock(self):
        with (self.root / '.scheduler.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def jobs(self):
        return read_json(self.root / 'jobs.json')['jobs']

    def update(self, key, state=None, **details):
        if state is not None and state not in STATES:
            raise ValueError('Unknown category state: ' + state)
        with self.lock():
            value = read_json(self.root / 'jobs.json')
            job = value['jobs'][key]
            job.update(details, updated_at=time.time())
            if state is not None:
                job['state'] = state
            write_json(self.root / 'jobs.json', value)
            return job

    def fill_slots(self):
        with self.lock():
            value = read_json(self.root / 'jobs.json')
            jobs = value['jobs']
            occupied = {j['slot'] for j in jobs.values() if j['state'] in ACTIVE and 'slot' in j}
            helper_families = {j['big_task'] for j in jobs.values() if j['state'] == 'REVIEW_SUBMITTED'}
            def priority(job):
                pair = (job['big_task'], job['small_task'])
                user = self.config.get('priorities', {}).get('/'.join(pair), 0)
                implemented = pair in CONVERTERS
                return (-user, not implemented, job['big_task'] not in helper_families,
                        job['big_task'], job['small_task'])
            pending = sorted((j for j in jobs.values() if j['state'] == 'PENDING'), key=priority)
            assignments = []
            for slot in range(1, 4):
                if slot in occupied or not pending:
                    continue
                job = pending.pop(0)
                job.update(state='ASSIGNED', slot=slot, owner=f'category_supervisor_{slot}', updated_at=time.time())
                assignment = self.assignment(job)
                write_json(self.root / 'assignments' / (job['key'] + '.json'), assignment)
                assignments.append(assignment)
            write_json(self.root / 'jobs.json', value)
            return assignments

    def assignment(self, job):
        group = self.groups[job['key']]
        pair = (job['big_task'], job['small_task'])
        plugin = CONVERTERS.get(pair, 'wia.authoring.converters.' + slug('_'.join(pair)) + ':CategoryConverter')
        module, class_name = plugin.split(':')
        owned = [str(ROOT / (module.replace('.', '/') + '.py')), group['report_file']]
        return {**job, 'schema': 'esi_category_assignment_v2', 'model': 'gpt-6.1-sol', 'reasoning_effort': 'xhigh',
            'module': module, 'class_name': class_name, 'owned_files': owned,
            'goal': 'Implement a deterministic whole-category converter; generate all sources, inspect stratified evidence and publish all successes/failures for Review.',
            'advice': group['advice'], 'corpus': self.inventory['corpus'],
            'implementation_spec': str(ROOT / 'esi_category_parallel_conversion_spec_v2.md'),
            'assessment_reference': group['advice']['assessment_reference'],
            'operation_goal_policy': 'Maximize original asset reuse while adding at least one meaningful real operation '
                'per Small Task where feasible. Creative, source-faithful grasp, opening, placement, navigation, pouring '
                'and compound recipes are authorized; a category may provide several operation variants. '
                'Preserve initial QA privately and independently measure native terminal success. '
                'Keep unsupported or uncertain instances visible in Review with warnings and evidence.',
            'category_operation_api': {'schema': str(ROOT / 'src/wia/tasks/schema.py'),
                'runtime': str(ROOT / 'src/wia/runtime/sim/esi_category_evaluation.py'),
                'module': module, 'qualifier': 'qualify_operation(session, executor, config, bound)',
                'factory': 'create_evaluator(session, controller, config, output)',
                'invariants': 'Qualification and evaluator hooks only read native state; robot actions use existing guarded controls. '
                    'Neither original QA, source metadata success flags nor mock operations prove success.'},
            'category_initialization_api': {'schema': str(ROOT / 'src/wia/tasks/schema.py'),
                'runtime': str(ROOT / 'src/wia/tasks/scene.py'), 'module': module,
                'config_hook': 'configure_scene(config, parameters)',
                'native_hook': 'initialize_scene(session, parameters)',
                'invariants': 'Use a declared category_initialization patch for source mechanism restoration only at initialization. '
                    'Return private initialization evidence, preserve original assets/answers/safety, and record reversible L2 assumptions. '
                    'The selected owned module is frozen and included in the renderer signature. '
                    'Use add_existing_r1 before robot_pose for robot-free sources; retain original physics settings. '
                    'object_physics may enable visual_only=false after source postprocessing; fixed_base changes are construction-only.'},
            'existing_task_files': group.get('existing_task_files', []),
            'source_builder': group.get('source_builder'),
            'batch_root': str(self.root / 'categories' / job['key']),
            'shared_api': str(ROOT / 'src/wia/runtime/esi_category_conversion.py'),
            'invariants': ['Preserve source JSON/answers/assets and approved records.',
                'No per-instance manual YAML changes, GPU submission, teleport success or new assets.',
                'Evidence-based reversible rules are authorized; uncertainty needs assumption_review_required with assumptions, alternatives, scope and risks.',
                'Maintain only the owned Small Task.md; do not edit research_log.md or shared schema/queue/Gallery.',
                'Every source needs a result; no failure can be omitted.'],
            'validation': 'Run convert_all for the complete assigned category and inspect counts/private/public boundaries. Main integrates shared APIs and automatically submits CaptureQueue inputs.',
            'deliverables': ['Converter module', 'Complete BatchManifest', 'Small Task.md', 'Failure distribution and sampling evidence', 'Unresolved assumptions and explicit diagnostic recommendations']}

    def sources(self, key, *, include_protected=False):
        group = self.groups[key]
        return [SourceInstance(Path(self.inventory['corpus']) / rel, rel, group['big_task'], group['small_task'])
            for rel in group['sources'] if include_protected or rel not in group['protected_sources']]

    def summary(self):
        jobs = list(self.jobs().values())
        return {'root': str(self.root), 'inventory': {k: self.inventory[k] for k in (
            'big_task_count', 'small_task_count', 'source_count')},
            'states': dict(Counter(j['state'] for j in jobs)),
            'active_supervisors': [j for j in jobs if j['state'] in ACTIVE], 'jobs': jobs}
