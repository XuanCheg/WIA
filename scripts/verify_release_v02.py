"""Verify the three extracted release components without a GPU or model call.

Run: python scripts/verify_release_v02.py --dataset ../wia-dataset-v0.2
  --evaluation . --harness ../ShowHarnessExplore-v0.2 --esi /path/to/ESI
  [--native-data /path/to/behavior] [--report /path/to/report.json]
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'evaluation', 'harness', 'esi'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--native-data', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    dataset, evaluation, harness, esi = [getattr(args, k).resolve() for k in ('dataset', 'evaluation', 'harness', 'esi')]
    os.environ.update(WIA_ROOT=str(evaluation), WIA_DATASET_ROOT=str(dataset),
                      WIA_ESI_ROOT=str(esi), SHOWHARNESS_ROOT=str(harness))
    if args.native_data:
        os.environ['OMNIGIBSON_DATA_PATH'] = str(args.native_data.resolve())
    sys.path[:0] = [str(evaluation / 'src'), str(harness)]
    checked_files = 0
    for component in (dataset, evaluation, harness):
        for line in (component / 'SHA256SUMS').read_text().splitlines():
            expected, relative = line.split('  ', 1)
            path = (component / relative).resolve()
            assert path.is_relative_to(component) and path.is_file(), relative
            assert digest(path) == expected, relative
            checked_files += 1
        manifest = json.loads((component / 'MANIFEST.json').read_text())
        actual = {str(p.relative_to(component)) for p in component.rglob('*') if p.is_file()
                  and not any(part in {'__pycache__', '.pytest_cache'} or part.endswith('.egg-info') for part in p.parts)
                  and p.name not in {'MANIFEST.json', 'SHA256SUMS'}}
        assert actual == set(manifest['files']), 'Unexpected or missing package files'
        for path in component.rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
    dependencies = json.loads((dataset / 'dependencies.json').read_text())
    actual_commit = subprocess.check_output(['git', '-C', str(esi), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual_commit == dependencies['esi_commit'], 'ESI commit differs from the release pin'
    for row in dependencies['sources']:
        base = esi if row['location'] == 'external_esi' else dataset
        assert digest(base / row['path']) == row['sha256'], row['identity']
    from wia.tasks.loader import discover_tasks, load_task, load_scene, select_task
    from wia.authoring.modules import import_owned_module
    from wia.evaluation.runner import prepare
    from wia.resources import resolve_resource_path
    from showharness.datasets import register_dataset, create_dataset
    from wia.integrations.show_harness import WIADatasetAdapter
    register_dataset('wia_release_check', WIADatasetAdapter)
    assert isinstance(create_dataset('wia_release_check'), WIADatasetAdapter)
    catalogue = json.loads((dataset / 'catalogue.json').read_text())
    assert len(catalogue['tasks']) == len(discover_tasks()) == 60
    assert dict(Counter(r['family'] for r in catalogue['tasks'])) == catalogue['active_family_counts']
    native_refs, loaded, aliases = [], [], set()
    id_counts = Counter(r['task_id'] for r in catalogue['tasks'])
    for row in catalogue['tasks']:
        assert row['alias'] not in aliases
        aliases.add(row['alias'])
        for key in ('task_file', 'scene_file'):
            assert digest(dataset / row[key]) == row[key + '_sha256']
        task = load_task(row['alias'])
        if id_counts[row['task_id']] == 1:
            assert task.path == select_task(row['task_id'])
        else:
            try:
                select_task(row['task_id'])
            except ValueError:
                pass
            else:
                raise AssertionError('Shared legacy task IDs must remain ambiguous; use an alias.')
        assert task.definition.id == row['task_id'] and task.variant == row['variant']
        task.verify_unchanged()
        scene = load_scene(dataset / row['scene_file'], variant=row['variant'])
        scene.verify_unchanged()
        # T23 intentionally has a criterion-specific QA record and a separate scene source.
        assert (task.scene_name, task.room_name) == (scene.scene_name, scene.room_name)
        assert task.definition.status == 'pending' and task.release_v1
        assert 'private' not in task.public_payload
        import_owned_module(task.definition.private.evaluation.module)
        for patch in task.definition.scene.variants[row['variant']].patches:
            module = getattr(patch, 'module', None)
            if module:
                import_owned_module(module)
            params = getattr(patch, 'parameters', {})
            if params.get('source_scene_file', '').startswith('behavior://'):
                verified = False
                if args.native_data:
                    native = resolve_resource_path(params['source_scene_file'])
                    assert digest(native) == params['source_scene_sha256']
                    verified = True
                native_refs.append({'alias': row['alias'], 'resource': params['source_scene_file'], 'verified': verified})
        loaded.append({'alias': row['alias'], 'task_id': task.definition.id, 'variant': task.variant,
                       'seed': task.seed, 'source_sha256': task.source_sha256})
    with tempfile.TemporaryDirectory(prefix='wia-release-plan-') as temporary:
        config = Path(temporary) / 'suite.json'
        config.write_text(json.dumps({'tasks': [{'path': row['alias']} for row in catalogue['tasks']],
            'policy': {'type': 'show_harness', 'backend': 'codex_cli', 'model': 'RELEASE_VALIDATION_NO_MODEL_CALL'},
            'environment': {'mode': 'launch'}, 'evaluation': {'run_mode': 'benchmark'},
            'output_dir': str(Path(temporary) / 'unused-output')}))
        plan = prepare(config)
        assert len(plan['plans']) == 60
        assert not (Path(temporary) / 'unused-output').exists()
        for row, episode in zip(catalogue['tasks'], plan['plans']):
            assert episode['task']['variant'] == row['variant']
    result = {'schema': 'wia_release_verification_v1', 'passed': True, 'files_checked': checked_files,
        'task_count': len(loaded), 'scene_count': len(loaded), 'dry_run_plans': 60,
        'family_counts': catalogue['active_family_counts'], 'esi_commit': actual_commit,
        'native_resource_references': native_refs, 'tasks': loaded,
        'gpu_jobs': 0, 'model_calls': 0, 'robot_actions': 0,
        'physical_success_certified': False, 'scope': 'Integrity, relocation, schemas, source hashes, imports and evaluation planning.'}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'tasks'}, indent=2))


if __name__ == '__main__':
    main()
