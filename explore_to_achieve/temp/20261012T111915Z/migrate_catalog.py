"""Archive old definitions and import ESI scenes for pending review.

Run from Show-Harness: .venv/bin/python /tmp/migrate_esi_scene_catalog.py
Uses /tmp/esi_scene_catalog_archive.txt, created by the migration plan.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml


root = Path.cwd()
base = root / 'explore_to_achieve'
source = root / 'ESI-Bench/dataset/json_clean'
archive = Path(Path('/tmp/esi_scene_catalog_archive.txt').read_text().strip())
converted = Path('/scratch/projects/CFP04/CFP04-CF-047/xinyangzhou/show-harness/outputs/rigid_containment_generator/20261012T032046Z/batch_v3')
camera_scene = Path(Path('/tmp/gumi_astra_head_forward_root.txt').read_text().strip()) / 'scene.yaml'
archive.mkdir(parents=True, exist_ok=False)
shutil.copy2('/tmp/esi_scene_catalog_plan.json', archive / 'migration_plan.json')
shutil.copy2(__file__, archive / 'migrate_catalog.py')
archived_counts = {}
for name in ['scenes', 'tasks']:
    directory = base / name
    archived_counts[name] = sum(path.is_file() for path in directory.rglob('*'))
    directory.rename(archive / name)
print('Archived existing scenes and tasks.', flush=True)

tasks = base / 'tasks'
tasks.mkdir()
scenes = base / 'scenes'
shutil.copytree(source, scenes)
print('Copied the complete source JSON directory.', flush=True)

source_files = sorted(source.rglob('*.json'))
samples = [path for path in source_files if len(path.relative_to(source).parts) == 5]
entries = {}
for index, path in enumerate(samples, 1):
    relative = path.relative_to(source)
    descriptor = relative.with_suffix('.yaml')
    value = {'kind': 'scene_import', 'version': 1, 'status': 'pending',
             'source': {'provider': 'esi', 'json': relative.as_posix()}}
    (scenes / descriptor).write_text(yaml.safe_dump(value, sort_keys=False))
    entries[descriptor.as_posix()] = {'scene_file': descriptor.as_posix(),
        'source_json': relative.as_posix(), 'status': 'pending', 'conversion': 'not_started'}
    if index % 500 == 0:
        print(f'Marked {index}/{len(samples)} source scenes pending.', flush=True)


def retain_conversion(path, additional=False):
    value = yaml.safe_load(path.read_text())
    relative = Path(value['source']['json'])
    assert (scenes / relative).is_file(), relative
    destination = relative.with_suffix('.yaml')
    if additional:
        variants = [name for name in value['scene']['variants'] if name != 'original']
        suffix = '__' + '__'.join(variants)
        destination = relative.with_name(relative.stem + suffix + '.yaml')
        assert not (scenes / destination).exists(), destination
    value['status'] = 'pending'
    (scenes / destination).write_text(yaml.safe_dump(value, sort_keys=False))
    entries[destination.as_posix()] = {'scene_file': destination.as_posix(),
        'source_json': relative.as_posix(), 'status': 'pending', 'conversion': 'retained_for_review',
        'converted_from': str(path.resolve()), 'native_qualification': 'not_run_in_this_import'}


converted_files = sorted(converted.rglob('*.yaml'))
for path in converted_files:
    retain_conversion(path)
archived_scenes = sorted((archive / 'scenes').rglob('*.yaml'))
for path in archived_scenes:
    retain_conversion(path, additional=True)
retain_conversion(camera_scene, additional=True)

copied_files = sorted(scenes.rglob('*.json'))
assert len(copied_files) == len(source_files)
assert all((scenes / path.relative_to(source)).stat().st_size == path.stat().st_size for path in source_files)
assert not any(tasks.iterdir())
pending_files = sorted(scenes.rglob('*.yaml'))
assert len(pending_files) == len(entries)
assert all(yaml.safe_load(path.read_text())['status'] == 'pending' for path in pending_files)

catalog = {'schema': 'pending_scene_catalog_v1', 'created_at': datetime.now(timezone.utc).isoformat(),
           'status': 'pending', 'source_root': str(source.resolve()), 'scene_root': str(scenes.resolve()),
           'copied_json_files': len(source_files), 'copied_json_bytes': sum(path.stat().st_size for path in source_files),
           'source_sample_count': len(samples), 'scene_description_count': len(entries),
           'retained_converted_descriptions': len(converted_files) + len(archived_scenes) + 1,
           'source_index_files': [path.relative_to(source).as_posix() for path in source_files if len(path.relative_to(source).parts) != 5],
           'source_copy_policy': 'Preserve every original JSON verbatim, including questions, answers and raw metadata. These copies are private construction data.',
           'native_qualification': 'not_run', 'task_generation': 'deferred_by_user',
           'scenes': [entries[key] for key in sorted(entries)]}
(scenes / 'catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')
result = {'status': 'completed', 'archive': str(archive.resolve()), 'archived_files': archived_counts,
          'scene_root': str(scenes.resolve()), 'task_root': str(tasks.resolve()), 'tasks_empty': True,
          'copied_json_files': len(source_files), 'pending_scene_descriptions': len(entries),
          'retained_converted_descriptions': catalog['retained_converted_descriptions']}
(archive / 'migration_result.json').write_text(json.dumps(result, indent=2) + '\n')
(archive / 'README.md').write_text('''# Archived scene and task definitions

The original scenes and tasks directories were moved here without modifying their files.
The replacement scene catalogue contains verbatim ESI JSON copies and pending YAML descriptors.
Task generation is deferred; the new tasks directory is empty. Historical experiment outputs
remain at their original paths. This archive is excluded from active task/scene discovery.
''')
with (root / 'research_log.md').open('a') as stream:
    stream.write(f'\n2026-10-12 — Catalogue import completes. Move 3 existing scene files and 2 existing task files unchanged into {archive.resolve()}. Copy all {len(source_files)} ESI JSON files ({catalog["copied_json_bytes"]} bytes), including {len(samples)} sample descriptions and 10 aggregate indexes, into explore_to_achieve/scenes with the original hierarchy. Create {len(entries)} pending scene descriptors: retain 19 batch conversions plus the archived scaled-football and latest native-head coffee-table variants; the other descriptions remain explicit pending imports. Preserve original JSON and all source experiment outputs. Every descriptor is pending and the new tasks directory is empty. No task/evaluator generation or simulation/model call is performed. Native qualification is not claimed. File counts, copied sizes, all pending markers and the empty task directory pass verification; loader gate validation follows.\n')
print(json.dumps(result), flush=True)
