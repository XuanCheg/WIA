"""Category converter contracts and deterministic CPU batching.

Run: python src/wia/runtime/commands/run_esi_category_campaign.py --help
Converters own rules, never edit source JSON or manually edit generated YAML.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
import random
import traceback

import yaml

from wia.authoring.modules import canonical_module_name, import_owned_module
from wia.authoring.io import digest, read_json, sha256, write_json
from wia.tasks.schema import SceneDefinition

from wia.resources import ROOT, ESI_ROOT


def map_records(value):
    return value if isinstance(value, dict) else {
        item['_key']: item for item in (value or []) if isinstance(item, dict) and item.get('_key')}


@dataclass(frozen=True)
class SourceInstance:
    path: Path
    relative_path: str
    big_task: str
    small_task: str

    @property
    def raw(self):
        return json.loads(self.path.read_text())

    @property
    def payload(self):
        value = self.raw.get('metadata_json', {})
        return json.loads(value) if isinstance(value, str) else value


@dataclass
class ConversionContext:
    output_root: Path
    seed: int = 0
    config: dict = field(default_factory=dict)
    advice: dict = field(default_factory=dict)


def original_scene(source, context, *, bindings=None, patches=None):
    patches = patches or []
    return {'kind': 'scene_patch', 'id': 'category_' + digest(source.relative_path)[:24],
        'version': 1, 'status': 'pending', 'source': {'provider': 'esi', 'json': source.relative_path},
        'target': None, 'qualification': None, 'bindings': bindings or {},
        'scene': {'policy': 'native_then_minimal_patch_else_skip', 'variants': {
            'original': {'patches': []}, **({'adapted': {'patches': patches}} if patches else {})}}, 'seed': context.seed}


def physical_task(source, context, *, public, evaluation, bindings, patches=None,
                  qualification=None, capabilities=None):
    """Pending physical task; source QA is saved separately and stays private."""
    scene = original_scene(source, context, bindings=bindings, patches=patches)
    return {'kind': 'category_robot_task', 'id': 'task_' + digest(source.relative_path)[:24],
        'version': 1, 'status': 'pending', 'source': scene['source'], 'public': public,
        'requirements': {'capabilities': capabilities or [
            'base_translation', 'base_rotation', 'ee_translation', 'gripper']},
        'private': {'bindings': bindings, 'evaluation': evaluation}, 'scene': scene['scene'],
        'qualification': qualification or {'on_failure': 'skip', 'initial_visibility': 'none'},
        'defaults': {'seed': context.seed}}


class ConverterBase:
    version = '1'
    category = ('', '')

    def inspect(self, source):
        return {'eligible': True}

    def convert(self, source, context):
        raise NotImplementedError

    def convert_all(self, sources, context):
        records = []
        from wia.tasks.loader import task_definition
        from wia.tasks.interaction_setup import prepare_definitions
        placement_files = [ROOT / name for name in (
            'src/wia/tasks/interaction_setup.py', 'src/wia/authoring/converters/interaction_profiles.py')]
        placement_revision = {str(path.relative_to(ROOT)): sha256(path) for path in placement_files if path.is_file()}
        for source in sources:
            try:
                source_hash = sha256(source.path)
            except OSError:
                source_hash = 'unavailable'
            revision = digest({'source': source.relative_path, 'source_sha256': source_hash,
                'version': self.version, 'implementation': self.implementation_hash(),
                'config': context.config, 'seed': context.seed, 'placement_review': placement_revision})[:24]
            candidate_id = 'category_' + revision
            directory = context.output_root / 'candidates' / candidate_id
            existing = read_json(directory / 'record.json')
            if existing:
                records.append(existing)
                continue
            record = {'original_source': str(source.path), 'source_identity': source.relative_path,
                'big_task': source.big_task, 'small_task': source.small_task,
                'converter_version': self.version, 'converter_module': type(self).__module__,
                'candidate_id': candidate_id, 'scene_file': None, 'task_file': None,
                'reuse_level': 'L1', 'change_summary': [], 'goal_predicate': None,
                'qa_preserved': True, 'limitations': [], 'status': 'conversion_failed',
                'failure_stage': None, 'failure_reason': None, 'warning_codes': [],
                'assumptions': [], 'sample_evidence': [], 'review_record_id': None,
                'source_sha256': source_hash, 'seed': context.seed, 'revision': revision,
                'attempted_repairs': [], 'next_step': 'Review the category converter and source evidence.'}
            directory.mkdir(parents=True, exist_ok=True)
            try:
                record['failure_stage'] = 'source_inspection'
                raw = source.raw
                record.update(scene_model=raw.get('scene'), room=raw.get('room'), question=raw.get('question', ''))
                eligibility = self.inspect(source)
                if not eligibility.get('eligible', False):
                    record.update(failure_reason=eligibility.get('reason', 'not_convertible_under_L1_L2'),
                                  warning_codes=eligibility.get('warning_codes', []))
                else:
                    record['failure_stage'] = 'conversion'
                    result = self.convert(source, context)
                    scene = result.pop('scene_definition', None)
                    task = result.pop('task_definition', None)
                    interaction = prepare_definitions(source, scene, task, result.get('interaction_policy'))
                    if interaction is not None:
                        result['interaction_policy'] = interaction
                        result['reuse_level'] = 'L2'
                        result['warning_codes'] = sorted(set(result.get('warning_codes', []) +
                            list(interaction.get('warning_codes', ())) + ['interaction_layout_native_review_pending']))
                        result.setdefault('change_summary', []).append(
                            'Screen the actual R1 floor stance while preserving the original floorplan and furniture.'
                            if interaction['mode'] == 'navigation' else
                            'Screen the actual R1 floor stance and object heights; relocate only complete '
                            'authorized groups onto measured existing supports without resizing.')
                    diagnostic_stage = result.get('failure_stage') or result.get('diagnostic_stage')
                    record.update(result)
                    record['failure_stage'] = 'static_validation'
                    if scene is not None:
                        SceneDefinition.model_validate(scene)
                        scene_path = directory / 'scene.yaml'
                        scene_path.write_text(yaml.safe_dump(scene, sort_keys=False))
                        record.update(scene_file=str(scene_path), variant='adapted' if 'adapted' in scene['scene']['variants'] else 'original')
                    if task is not None:
                        task_definition(task)
                        task_path = directory / 'task.yaml'
                        task_path.write_text(yaml.safe_dump(task, sort_keys=False))
                        record.update(task_file=str(task_path), goal_predicate=task['private']['evaluation'])
                    if scene is None:
                        record.update(status='conversion_failed', failure_reason=record.get('failure_reason') or 'no_scene_overlay')
                    elif record.get('status') not in ('conversion_failed', 'needs_human_decision'):
                        record.update(status='generated', failure_reason=None)
                    elif not record.get('failure_reason') and task is not None:
                        record.update(status='generated')
                    if record['status'] == 'generated':
                        record['failure_stage'] = None
                    elif diagnostic_stage:
                        record['failure_stage'] = diagnostic_stage
                if sha256(source.path) != source_hash:
                    raise RuntimeError('Original source changed during conversion.')
                write_json(directory / 'private_qa.json', {'original_source': str(source.path),
                    'answer': raw.get('answer'), 'options_json': raw.get('options_json'),
                    'metadata_json': raw.get('metadata_json'), 'source_sha256': source_hash})
            except Exception as error:
                record.update(status='conversion_failed', failure_reason=f'{type(error).__name__}: {error}',
                              exception_summary=traceback.format_exc())
                (directory / 'conversion_error.log').write_text(record['exception_summary'])
            if record['assumptions'] and 'assumption_review_required' not in record['warning_codes']:
                record['warning_codes'].append('assumption_review_required')
            record['diagnostic_dir'] = str(directory)
            write_json(directory / 'record.json', record)
            records.append(record)
        manifest = {'schema': 'esi_category_batch_v2', 'big_task': self.category[0],
            'small_task': self.category[1], 'converter_version': self.version,
            'seed': context.seed, 'config': context.config, 'records': records,
            'counts': dict(Counter(r['status'] for r in records)), 'source_count': len(sources)}
        write_json(context.output_root / 'batch_manifest.json', manifest)
        return manifest

    def implementation_hash(self):
        import inspect
        return sha256(inspect.getfile(type(self)))

    def sample(self, manifest, policy=None):
        return sample_records(manifest, **(policy or {}))


class DiagnosticConverter(ConverterBase):
    """Keep every source reviewable when its category implementation cannot run."""

    def __init__(self, category, reason, stage):
        self.category, self.reason, self.stage = category, reason, stage
        self.version = 'diagnostic-' + digest({'reason': reason, 'stage': stage})[:16]

    def convert(self, source, context):
        return {'scene_definition': None, 'task_definition': None,
            'status': 'conversion_failed', 'failure_stage': self.stage,
            'failure_reason': self.reason, 'warning_codes': ['category_implementation_unavailable'],
            'limitations': ['No physical task was generated. Source-only capture is diagnostic evidence.'],
            'next_step': 'Repair the category implementation using the preserved assignment and supervisor log.'}


def sample_records(manifest, *, count=5, seed=0, previous_failures=()):
    rows = manifest['records']
    shuffled = sorted(rows, key=lambda r: r['candidate_id'])
    random.Random(seed).shuffle(shuffled)
    selected = []
    prior = set(previous_failures)
    failures = [r for r in shuffled if r['status'] != 'generated']
    priorities = [r for r in shuffled if r['candidate_id'] in prior]
    if failures:
        priorities.insert(0, failures[0])
    strata = set()
    for row in priorities + shuffled:
        if len(selected) >= min(count, len(rows)):
            break
        if row in selected:
            continue
        key = (row.get('scene_model'), row.get('room'), row.get('failure_stage'),
               tuple(row.get('warning_codes', [])), row.get('reuse_level'))
        if row in priorities or key not in strata:
            selected.append(row)
            strata.add(key)
    for row in shuffled:
        if len(selected) >= min(count, len(rows)):
            break
        if row not in selected:
            selected.append(row)
    return [row['candidate_id'] for row in selected]


CONVERTERS = {
    ('Action Sequencing', 'Action Order Inference'): 'wia.authoring.converters.action_order:ActionOrderConverter',
    ('Perceptual Grounding', 'Partial Occlusion'): 'wia.authoring.converters.partial_occlusion:PartialOcclusionConverter',
    ('Metric Comparison', 'Dimensional Size'): 'wia.authoring.converters.dimensional_size:DimensionalSizeConverter',
    ('Temporal Understanding', 'Unobserved Change'): 'wia.authoring.converters.temporal_understanding_two_phase:CategoryConverter',
}


def get_converter(big_task, small_task, plugins=None):
    entry = {**CONVERTERS, **(plugins or {})}.get((big_task, small_task))
    if not entry:
        return None
    module, name = entry.split(':')
    try:
        importlib.invalidate_caches()
        loaded = import_owned_module(module)
        current = sha256(loaded.__file__)
        previous = getattr(loaded, '_category_loaded_hash', current)
        if previous != current:
            loaded = importlib.reload(loaded)
        loaded._category_loaded_hash = current
        return getattr(loaded, name)()
    except ModuleNotFoundError as error:
        if error.name == canonical_module_name(module):
            return None
        raise
