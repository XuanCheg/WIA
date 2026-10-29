#!/usr/bin/env python3
"""Prepare continuous two-phase revisions from existing Unobserved Change tasks.

Run with behavior Python:
  python src/wia/runtime/commands/revise_esi_unobserved_phases.py --campaign-root CAMPAIGN
The command writes a separate HOME batch without mutating old scenes, queuing
captures, or changing the active Review. Main publishes after native validation.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
import sys

import yaml

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.authoring.conversion import SourceInstance
from wia.authoring.converters.temporal_understanding_two_phase import CategoryConverter, MODULE, OPERATION, upgrade_existing
from wia.runtime.scene_capture_queue import digest, read_json, write_json
from wia.tasks.interaction_setup import prepare_definitions
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition


def prepare(campaign, output=None):
    campaign = Path(campaign).expanduser().resolve()
    output = Path(output).expanduser().resolve() if output else campaign / 'two_phase_20261015'
    inventory = read_json(campaign / 'inventory.json')
    group = next(row for row in inventory['categories'] if
                 (row['big_task'], row['small_task']) == CategoryConverter.category)
    published = read_json(campaign / 'review_catalogue/category_revisions.json')['categories'][group['key']]
    parent_file = Path(published['new_batch_file'])
    parent = read_json(parent_file)
    records = []
    publication_fields = ('record_revision', 'review_record_id', 'published_at', 'report_file', 'batch_file',
        'job_id', 'fingerprint', 'source_only', 'diagnostic_only', 'diagnostic_images', 'diagnostic_logs',
        'review_submitted', 'capture_input', 'capture_signature', 'code_commit', 'capture_submission_error',
        'capture_submission_stage', 'no_native_image', 'runtime_refresh')
    for original in parent['records']:
        row = deepcopy(original)
        for field in publication_fields:
            row.pop(field, None)
        source = SourceInstance(Path(original['original_source']), original['source_identity'], *CategoryConverter.category)
        scene, task = None, None
        if row.get('task_file'):
            scene, task = upgrade_existing(yaml.safe_load(Path(row['scene_file']).read_text()),
                                          yaml.safe_load(Path(row['task_file']).read_text()), source)
            row['interaction_policy'] = prepare_definitions(source, scene, task)
            SceneDefinition.model_validate(scene)
            task_definition(task)
        revision = digest({'parent': original['candidate_id'], 'version': CategoryConverter.version,
                           'scene': scene, 'task': task, 'purpose': 'continuous_two_phase_episode'})[:24]
        candidate = 'category_' + revision
        directory = output / 'candidates' / candidate
        directory.mkdir(parents=True, exist_ok=True)
        row.update(candidate_id=candidate, revision=revision, parent_candidate_id=original['candidate_id'],
                   converter_version=CategoryConverter.version, converter_module=MODULE, diagnostic_dir=str(directory))
        if task is not None:
            scene_file, task_file = directory / 'scene.yaml', directory / 'task.yaml'
            scene_file.write_text(yaml.safe_dump(scene, sort_keys=False))
            task_file.write_text(yaml.safe_dump(task, sort_keys=False))
            row.update(scene_file=str(scene_file), task_file=str(task_file),
                goal_predicate=task['private']['evaluation'], operation_recipe=OPERATION,
                initial_phase='image1', temporal_episode=True,
                warning_codes=['external_source_phase_transition', 'native_temporal_execution_unverified'],
                change_summary=['Use actual Phase 1 observations in one continuing robot/model episode.',
                    'Keep Phase 1 read-only; enable original guarded operations only in Phase 2.',
                    'Apply the original external content change between observation boundaries and privately score the answer.'],
                limitations=['Native two-phase loading, camera coverage and robot answer quality require measured evidence.',
                    'Legacy independent snapshots remain historical diagnostic variants.'],
                assumptions=[{'doubt': 'Original source snapshots do not themselves establish Agent memory.',
                    'chosen_assumption': 'Acquire both mounted-camera observations in one continuing episode.',
                    'alternatives': ['Retain a visible diagnostic if native initialization or the transition fails.'],
                    'scope': row['source_identity'],
                    'risk': 'Initialization, source visibility or later operation may still fail.'}])
        qa = Path(original['diagnostic_dir']) / 'private_qa.json'
        if qa.is_file():
            shutil.copyfile(qa, directory / qa.name)
        write_json(directory / 'record.json', row)
        records.append(row)
    manifest = {**parent, 'records': records, 'counts': dict(Counter(row['status'] for row in records)),
                'converter_version': CategoryConverter.version, 'supersedes_batch_file': str(parent_file),
                'generation_method': 'Whole-category pure scene/task upgrade; original source and failed records preserved.'}
    manifest.pop('review_submitted', None)
    write_json(output / 'batch_manifest.json', manifest)
    representative = next(row for row in records if row['source_identity'].endswith(
        'Beechwood_0_garden/living_room_0/current_state_reasoning__q_001.json'))
    result = {'complete': True, 'source_count': len(records), 'counts': manifest['counts'],
        'batch_file': str(output / 'batch_manifest.json'), 'selected_sample_candidate_id': representative['candidate_id'],
        'selected_scene_file': representative['scene_file'], 'selected_task_file': representative['task_file'],
        'capture_jobs_created': 0, 'original_sources_changed': False, 'original_scenes_changed': False}
    write_json(output / 'preparation_result.json', result)
    return result


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.campaign_root, args.output_dir)), flush=True)
