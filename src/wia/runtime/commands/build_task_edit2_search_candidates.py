"""Build provisional TaskEdit2 search parents and three enclosure additions.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B \
  src/wia/runtime/commands/build_task_edit2_search_candidates.py --output outputs/task_edit2_20261018/perception
CPU generation only; native mechanism, initialization and execution gates remain pending.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
import json
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

import yaml
from wia.authoring.conversion import ConversionContext, SourceInstance
from wia.authoring.converters.composition_perception import build_task_edit2_search_group
from wia.authoring.converters.enumerative_perception_structural_enclosure import curate_task_edit2_additions
from wia.runtime.scene_capture_queue import write_json
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition

# One source per room avoids overlaying mutually exclusive source trials.
CURATED = (
    ('house_search', (
        ('Category Ambiguity/house_single_floor/bedroom_0/q_002.json',
         'the bedroom', 'the two colored tabletop search stations', 'Inspect both the amber and blue tabletop stations; distinguish targets from lookalike objects.'),
        ('Counting w Occlusion/house_single_floor/kitchen_0/q_004.json',
         'the kitchen', 'the bar, stools and surrounding floor area', 'Search around the bar and stools; change viewpoint around their obstruction.'),
        ('Merged Observation/house_single_floor/garage_0/q_000.json',
         'the garage', 'the green tabletop cluster station', 'Inspect the compact tabletop cluster from overlapping views without counting one object twice.'),
    )),
    ('office_search', (
        ('Illumination Variability/office_cubicles_right/conference_hall_0/q_002.json',
         'the conference hall', 'the floor and lamp area', 'Inspect under the declared normal and changed-light phases.'),
        ('Spatial Segmentation/office_cubicles_right/private_office_2/q_001.json',
         'the private office with chair-defined regions', 'the chair-defined search regions',
         'Inspect each chair-defined region and retain its boundary when reporting locations.'),
    )),
)


def build(output, *, seed=0):
    output = Path(output).resolve()
    context = ConversionContext(output, seed=seed)
    results = []
    for name, rows in CURATED:
        sources, areas = [], {}
        for suffix, room, area, clue in rows:
            identity = 'Enumerative Perception/' + suffix
            source = SourceInstance(WIA_ESI_ROOT / 'dataset/json_clean' / identity,
                                    identity, 'Enumerative Perception', suffix.split('/')[0])
            sources.append(source)
            areas[identity] = {'room': room, 'area': area, 'clue': clue}
        results.append(build_task_edit2_search_group(sources, context, public_areas=areas, name=name))
    previous_path = output / 'manifest.json'
    previous = json.loads(previous_path.read_text()).get('candidates', []) if previous_path.is_file() else []
    preserved = [row for row in previous if row.get('candidate_id', '').startswith('enclosure_addition_')
                 and row.get('task_file') and row.get('scene_file')
                 and Path(row['task_file']).is_file() and Path(row['scene_file']).is_file()]
    # This bounded Search revision never rewrites already generated enclosure definitions.
    results.extend(preserved if len(preserved) == 3 else curate_task_edit2_additions(context))
    manifest = {'schema': 'task_edit2_perception_candidates_v1', 'final_selection': 'provisional',
        'native_experiments_run': False, 'search_parent_count': 2, 'enclosure_addition_count': 3, 'candidates': []}
    for result in results:
        directory = output / 'candidates' / result['candidate_id']
        directory.mkdir(parents=True, exist_ok=True)
        record = {k: v for k, v in result.items() if k not in ('task_definition', 'scene_definition')}
        for key, filename in (('task_definition', 'task.yaml'), ('scene_definition', 'scene.yaml')):
            definition = result.get(key)
            if definition:
                if key == 'task_definition':
                    task_definition(definition)
                else:
                    SceneDefinition.model_validate(definition)
                path = directory / filename
                path.write_text(yaml.safe_dump(definition, sort_keys=False))
                record[key.replace('_definition', '_file')] = str(path)
        write_json(directory / 'record.json', record)
        manifest['candidates'].append(record)
    write_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/task_edit2_20261018/perception')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    manifest = build(args.output, seed=args.seed)
    print(f"Generated {len(manifest['candidates'])} provisional definitions; native execution remains unavailable.")
