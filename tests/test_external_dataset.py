"""Check catalogue selection and relocation. Run: pytest tests/test_external_dataset.py."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_explicit_diagnostic_task_without_catalogue_does_not_gain_release_membership(tmp_path, monkeypatch):
    from wia.tasks import loader
    original = loader.load_task('T01')
    monkeypatch.setattr(loader, 'CATALOGUE_PATH', tmp_path / 'missing_catalogue.json')
    sealed = loader.load_task(original.path, variant=original.variant, run_mode='diagnostic')
    assert sealed.definition.id == original.definition.id
    assert sealed.release_v1 is False
    with pytest.raises(ValueError, match='pending/discarded'):
        loader.load_task(original.path, variant=original.variant)
    with pytest.raises(FileNotFoundError):
        loader.discover_tasks()


def test_external_catalogue_controls_discovery_aliases_and_pending_policy(tmp_path):
    from wia.resources import DATASET_ROOT
    from wia.tasks.loader import release_catalogue
    original = release_catalogue()['tasks']
    rows = []
    for row in (original[0], original[-1]):
        copied = dict(row)
        for key in ('task_file', 'scene_file'):
            destination = tmp_path / key / Path(row[key]).name
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes((DATASET_ROOT / row[key]).read_bytes())
            copied[key] = str(destination.relative_to(tmp_path))
        rows.append(copied)
    (tmp_path / 'catalogue.json').write_text(json.dumps({'tasks': rows}))
    # A leftover YAML must not silently become an active benchmark member.
    (tmp_path / 'task_file/retired.yaml').write_bytes((tmp_path / rows[0]['task_file']).read_bytes())
    code = '''
import json
from wia.tasks.loader import discover_tasks, select_task, is_release_v1, release_catalogue
from wia.resources import DATASET_ROOT
rows = release_catalogue()['tasks']
assert len(discover_tasks()) == 2
for row in rows:
    path = select_task(row['alias'])
    assert path == select_task(row['task_id'])
    assert is_release_v1(path)
assert not is_release_v1(DATASET_ROOT / 'task_file/retired.yaml')
'''
    env = dict(os.environ, WIA_DATASET_ROOT=str(tmp_path))
    subprocess.run([sys.executable, '-c', code], env=env, check=True, capture_output=True, text=True)


def test_native_resource_path_is_explicit_and_contained(tmp_path, monkeypatch):
    from wia.resources import resolve_resource_path
    monkeypatch.setenv('OMNIGIBSON_DATA_PATH', str(tmp_path))
    assert resolve_resource_path('behavior://scenes/example.json') == tmp_path / 'scenes/example.json'
    with pytest.raises(ValueError, match='inside'):
        resolve_resource_path('behavior://../outside.json')
    monkeypatch.delenv('OMNIGIBSON_DATA_PATH')
    with pytest.raises(ValueError, match='OMNIGIBSON_DATA_PATH'):
        resolve_resource_path('behavior://scenes/example.json')


def test_persistent_terminal_receives_explicit_package_roots(monkeypatch):
    from wia.resources import ROOT, ESI_ROOT, DATASET_ROOT
    from wia.runtime.resources import execution_resource_environment
    monkeypatch.setenv('WIA_DATASET_ROOT', str(DATASET_ROOT))
    values = execution_resource_environment()
    assert values['WIA_ROOT'] == str(ROOT)
    assert values['WIA_ESI_ROOT'] == str(ESI_ROOT)
    assert values['WIA_DATASET_ROOT'] == str(DATASET_ROOT)
    assert str(ROOT / 'src') in values['PYTHONPATH'].split(os.pathsep)
    monkeypatch.delenv('WIA_DATASET_ROOT')
    assert execution_resource_environment()['WIA_DATASET_ROOT'] == ''

