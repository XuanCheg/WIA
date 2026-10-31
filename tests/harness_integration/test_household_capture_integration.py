"""Validate separate household capture composition without using a live queue.

Run: PYTHONPATH=src python -B -m pytest -q -p no:cacheprovider
  tests/harness_integration/test_household_capture_integration.py
All definitions, queue-shaped files, receipts, and review outputs are temporary.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


PROJECT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load_module('household_legacy_campaign_fixture', Path(__file__).with_name('test_publish_wia_final_captures.py'))
publisher = helpers.publisher
appender = load_module('household_capture_appender_test', PROJECT / 'scripts/append_household_geometry_captures.py')
scanner = load_module('household_capture_scanner_test', PROJECT / 'scripts/scan_wia_dataset.py')
write_json = helpers.write_json


def household_fixture(root, *, exclude_household=False, pending_household=True):
    campaign, receipts, excluded, groups = helpers.campaign_fixture(root, baseline_exclusions=5,
                                                                    candidate_exclusions=10)
    tasks, items = [], []
    for number, alias in enumerate(('GF2-1', 'GF2-2', 'GF3-1', 'GF3-2')):
        identity = 'household_task_' + alias.lower()
        task_path = root / 'household/tasks' / (alias + '.yaml')
        scene_path = root / 'household/scenes' / (alias + '.yaml')
        source_identity = 'Geometry/Fit/household/kitchen/' + alias + '.json'
        source = root / 'authored' / source_identity
        write_json(source, {'scene': 'household', 'room': 'kitchen'})
        task = {'kind': 'category_robot_task', 'id': identity, 'status': 'pending',
            'source': {'provider': 'esi', 'json': source_identity}, 'defaults': {'seed': 110 + number},
            'public': {'objective': 'Current authored objective for ' + alias,
                'success_criteria': 'Released stable physical placement.', 'constraints': ['Preserve furniture.']},
            'private': {'bindings': {'item': {'name': 'item'}},
                'evaluation': {'module': 'wia.authoring.converters.wia_household_fit', 'operation': 'fit'}},
            'scene': {'variants': {'adapted': {'patches': [{'operation': 'household_initialization',
                                                          'parameters': {'alias': alias}}]}}}}
        scene = {'kind': 'scene_patch', 'id': 'separate_scene_' + alias, 'status': 'pending',
                 'source': deepcopy(task['source']), 'seed': task['defaults']['seed'],
                 'bindings': deepcopy(task['private']['bindings']), 'scene': deepcopy(task['scene'])}
        for path, value in ((task_path, task), (scene_path, scene)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(value), encoding='utf-8')
        tasks.append({'alias': alias, 'id': identity, 'task_id': identity,
            'task_file': str(task_path), 'scene_file': str(scene_path), 'variant': 'adapted',
            'family': 'GF', 'subclass': 'GF-2' if number < 2 else 'GF-3',
            'title': 'Household geometry ' + alias, 'origin': 'household_geometry',
            'physical_success_verified': False})
        job_id = 'household_capture_' + str(number)
        job = root / 'queue/jobs' / job_id
        queued = pending_household and number == 0
        write_json(job / 'status.json', {'job_id': job_id, 'state': 'QUEUED' if queued else 'CACHED',
            'attempts': 0 if queued else 1, 'retryable': False})
        write_json(job / 'input/manifest.json', {'job_id': job_id, 'max_attempts': 2})
        items.append({'row_id': alias, 'document_alias': alias, 'variant_id': identity,
            'label': 'Exact household capture ' + alias, 'task_file': str(task_path),
            'task_sha256': publisher._sha(task_path), 'scene_file': str(scene_path),
            'scene_sha256': publisher._sha(scene_path), 'job_id': job_id,
            'queue_status_file': str(job / 'status.json'),
            'receipt_file': str(job / 'attempts/01/preview_complete.json')})
    catalogue_path = root / 'household/catalogue.json'
    index_path = campaign / 'household_geometry_receipts/review_index.json'
    write_json(catalogue_path, {'schema': 'wia_household_geometry_candidates_v1',
                              'release_admitted': False, 'tasks': tasks})
    write_json(index_path, {'schema': 'wia_capture_review_index_v1', 'items': items})
    curation_path = campaign / 'curation.json'
    curation = json.loads(curation_path.read_text())
    curation['supplementary_catalogues'] = [{'catalogue_file': str(catalogue_path),
                                            'review_index_file': str(index_path)}]
    if exclude_household:
        curation['temporarily_excluded_aliases'].append('GF2-1')
    write_json(curation_path, curation)
    return campaign, receipts, catalogue_path, index_path, tasks, items


@pytest.mark.parametrize('excluded,expected,terminal', [(False, 69, False), (True, 68, True)])
def test_household_composition_progress_exclusions_and_separate_audit(tmp_path, monkeypatch,
                                                                   excluded, expected, terminal):
    campaign, receipts, catalogue, index, tasks, items = household_fixture(tmp_path, exclude_household=excluded)
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    prepared = publisher.prepare(campaign, receipts, project_root=tmp_path)
    progress = prepared['progress']
    assert progress['active_job_count'] == progress['total'] == expected
    assert progress['all_terminal'] is terminal
    assert progress['baseline_count'] == 56 and progress['candidate_count'] == 9
    assert progress['archived_baseline_count'] == 61 and progress['archived_candidate_count'] == 19
    assert progress['supplementary_count'] == 4 - int(excluded)
    assert progress['archived_supplementary_count'] == 4
    assert len(prepared['derived_curation']['extra_rows']) == 23
    for selected, row in zip(tasks, prepared['derived_curation']['extra_rows'][-4:]):
        assert row['id'] == row['document_alias'] == selected['alias']
        assert row['status'] == 'pending' and row['proposal_membership'] is False
        assert row['release_admitted'] is row['physical_success_verified'] is False
        assert row['primary_family'] == 'GF' and row['primary_subclass'] == selected['subclass']
        assert row['public']['objective'] == 'Current authored objective for ' + selected['alias']
        assert row['variants'][0]['id'] == selected['task_id']
    assert all(path.read_bytes() == value for path, value in protected.items())
    monkeypatch.setattr(publisher, 'SNAPSHOT', tmp_path / 'review/snapshot.json')
    real_build = publisher.build_snapshot
    merge_inputs = []

    def checked_build(*args, **kwargs):
        merge_inputs.extend(kwargs['supplements'])
        return real_build(*args, **kwargs)

    monkeypatch.setattr(publisher, 'build_snapshot', checked_build)
    result = publisher.publish_prepared(prepared, campaign, project_root=tmp_path)
    assert [path.name for path in merge_inputs] == ['baseline_capture_evidence.json']
    assert result['snapshot_row_count'] == expected
    audit, = result['supplementary_evidence_audit_files']
    assert set(json.loads(Path(audit).read_text())['row_updates']) == {'GF2-1', 'GF2-2', 'GF3-1', 'GF3-2'}
    snapshot = json.loads(publisher.SNAPSHOT.read_text())
    aliases = {row['document_alias'] for row in snapshot['rows']}
    assert ('GF2-1' not in aliases) is excluded
    assert {'GF2-2', 'GF3-1', 'GF3-2'} <= aliases
    assert all(path.read_bytes() == value for path, value in protected.items())


@pytest.mark.parametrize('fault', ['catalogue_alias', 'identity', 'admitted', 'module', 'seed',
                                 'bindings', 'definition_hash', 'index_alias', 'duplicate_job'])
def test_supplementary_identity_failures_do_not_mutate_any_input(tmp_path, fault):
    campaign, receipts, catalogue_path, index_path, tasks, items = household_fixture(tmp_path)
    catalogue, index = json.loads(catalogue_path.read_text()), json.loads(index_path.read_text())
    if fault == 'catalogue_alias':
        catalogue['tasks'][0]['alias'] = 'UNKNOWN'
    elif fault == 'identity':
        catalogue['tasks'][0]['task_id'] = 'other_task'
    elif fault == 'admitted':
        catalogue['release_admitted'] = True
    elif fault == 'index_alias':
        index['items'][0]['document_alias'] = 'GF2-2'
    elif fault == 'duplicate_job':
        index['items'][0]['job_id'] = index['items'][1]['job_id']
    elif fault == 'definition_hash':
        index['items'][0]['task_sha256'] = '0' * 64
    else:
        path = Path(tasks[0]['task_file'])
        task = yaml.safe_load(path.read_text())
        if fault == 'module':
            task['private']['evaluation']['module'] = 'unowned.module'
        elif fault == 'seed':
            task['defaults']['seed'] += 1
        else:
            task['private']['bindings'] = {}
        path.write_text(yaml.safe_dump(task), encoding='utf-8')
        index['items'][0]['task_sha256'] = publisher._sha(path)
    write_json(catalogue_path, catalogue)
    write_json(index_path, index)
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    with pytest.raises(ValueError):
        publisher.prepare(campaign, receipts, project_root=tmp_path)
    assert all(path.read_bytes() == value for path, value in protected.items())
    assert not (campaign / 'final_refresh_review').exists()


def test_publishing_rechecks_and_protects_supplementary_inputs(tmp_path, monkeypatch):
    campaign, receipts, catalogue, index, _, _ = household_fixture(tmp_path)
    prepared = publisher.prepare(campaign, receipts, project_root=tmp_path)
    monkeypatch.setattr(publisher, 'SNAPSHOT', catalogue)
    with pytest.raises(ValueError, match='never overwrite'):
        publisher.publish_prepared(prepared, campaign, project_root=tmp_path)
    monkeypatch.setattr(publisher, 'SNAPSHOT', tmp_path / 'review/snapshot.json')
    catalogue.write_text(catalogue.read_text() + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='changed after composition'):
        publisher.publish_prepared(prepared, campaign, project_root=tmp_path)
    assert not (campaign / 'final_refresh_review').exists()


def mock_native_loaders(monkeypatch, root):
    """Expose real fixture bytes through the loader interface without a simulator."""
    def loaded(path, *, variant, run_mode):
        raw = yaml.safe_load(path.read_text())
        task = raw['kind'] == 'category_robot_task'
        source_path = root / 'authored' / raw['source']['json']
        definition = SimpleNamespace(kind=raw['kind'], id=raw['id'], status=raw['status'],
            source=SimpleNamespace(json_path=raw['source']['json']),
            scene=SimpleNamespace(variants=raw['scene']['variants']), model_dump=lambda **_: raw)
        if task:
            definition.defaults = SimpleNamespace(seed=raw['defaults']['seed'])
            definition.private = SimpleNamespace(bindings=raw['private']['bindings'],
                evaluation=SimpleNamespace(module=raw['private']['evaluation']['module']))
        else:
            definition.seed, definition.bindings = raw['seed'], raw['bindings']
        return SimpleNamespace(definition=definition, seed=raw['defaults']['seed'] if task else raw['seed'],
            scene_name='household', room_name='kitchen', variant=variant, path=path,
            release_v1=False, source_path=source_path, source_sha256=publisher._sha(source_path),
            file_sha256=publisher._sha(path), verify_unchanged=lambda: None)
    monkeypatch.setattr(appender, 'load_task', loaded)
    monkeypatch.setattr(appender, 'load_scene', loaded)
    monkeypatch.setattr(appender, 'AUTHORED_ROOT', root / 'authored')
    monkeypatch.setattr(appender, 'resolve_source', lambda identity: root / 'authored' / identity)


def test_appender_check_accepts_separate_scene_ids_and_writes_nothing(tmp_path, monkeypatch, capsys):
    _, _, catalogue, _, _, _ = household_fixture(tmp_path)
    mock_native_loaders(monkeypatch, tmp_path)
    queue, output = tmp_path / 'append_queue', tmp_path / 'new_receipts'
    write_json(queue / 'category_storage.json', {})
    write_json(queue / 'capture_runtime_policy.json', {'allowed_code_commits': ['old_runtime'], 'keep': True})
    (queue / 'jobs').mkdir()
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError('Read-only validation cannot freeze code or publish a record.')

    monkeypatch.setattr(appender, 'freeze_working_code', forbidden)
    monkeypatch.setattr(appender, 'publish_record', forbidden)
    assert appender.main(['--catalogue', str(catalogue), '--queue-root', str(queue),
                          '--output', str(output), '--check']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['validated_candidate_count'] == 4 and report['read_only'] is True
    assert report['code_snapshot_frozen'] is False and report['record_identities_provisional'] is True
    records = report['proposed_records']
    assert {record['document_alias'] for record in records} == {'GF2-1', 'GF2-2', 'GF3-1', 'GF3-2'}
    assert all(record['capture_initial_agent_bundle'] and record['task_edit2_capture_campaign'] for record in records)
    assert all(record['success_claimed'] is record['native_validation_run'] is False for record in records)
    assert all(record['capture_source_members'][0]['source_sha256'] for record in records)
    assert report['priority'] == 0
    assert all(path.read_bytes() == value for path, value in protected.items())
    assert not output.exists() and not (queue / '.household_geometry_append.lock').exists()


def test_appender_sealed_contract_disallows_release_edge_relocation(tmp_path, monkeypatch):
    _, _, catalogue, _, _, _ = household_fixture(tmp_path)
    mock_native_loaders(monkeypatch, tmp_path)
    loaded = appender._definitions(catalogue)[0]
    record = appender._record(loaded, {'code_commit': 'frozen_runtime'}, tmp_path / 'queue',
                              catalogue, publisher._sha(catalogue))
    manifest = {'candidate_id': record['candidate_id'], 'code_commit': 'frozen_runtime',
        'task_sha256': record['task_sha256'], 'input_hashes': {'scene.yaml': record['scene_sha256']},
        'seed': record['seed'], 'variant': record['variant'], 'source_identity': record['source_identity'],
        'capture_source_members': record['capture_source_members'],
        'capture_initial_agent_bundle': True, 'task_edit2_capture_campaign': True,
        'diagnostic_pending': True, 'current_support_edge_policy': False, 'timeout_s': 2400, 'max_attempts': 2}
    monkeypatch.setattr(appender, 'read_json', lambda path: manifest)
    monkeypatch.setattr(appender, 'load_capture_scene', lambda value: loaded[2])

    def bind(probe, scene, value):
        probe.loaded_task = loaded[1]
        probe.capture_initial_agent_bundle = True
        probe._capture_release_edge_policy = value['current_support_edge_policy']

    monkeypatch.setattr(appender, 'bind_capture_inputs', bind)
    assert appender._validate_published({'job_id': 'exact_household_job'}, record, tmp_path / 'queue') == 'exact_household_job'
    manifest['current_support_edge_policy'] = True
    with pytest.raises(ValueError, match='immutable capture contract'):
        appender._validate_published({'job_id': 'exact_household_job'}, record, tmp_path / 'queue')


@pytest.mark.parametrize('fault', ['catalogue', 'task_identity', 'seed', 'bindings'])
def test_appender_validation_failure_never_opens_or_mutates_queue(tmp_path, monkeypatch, fault):
    _, _, catalogue_path, _, tasks, _ = household_fixture(tmp_path)
    mock_native_loaders(monkeypatch, tmp_path)
    if fault == 'catalogue':
        catalogue = json.loads(catalogue_path.read_text())
        catalogue['release_admitted'] = True
        write_json(catalogue_path, catalogue)
    else:
        path = Path(tasks[0]['task_file'])
        task = yaml.safe_load(path.read_text())
        if fault == 'task_identity':
            task['id'] = 'different_task'
        elif fault == 'seed':
            task['defaults']['seed'] += 1
        else:
            task['private']['bindings'] = {}
        path.write_text(yaml.safe_dump(task), encoding='utf-8')
    queue = tmp_path / 'append_queue'
    write_json(queue / 'category_storage.json', {})
    (queue / 'jobs').mkdir()
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    with pytest.raises(ValueError):
        appender.main(['--catalogue', str(catalogue_path), '--queue-root', str(queue),
                       '--output', str(tmp_path / 'receipts')])
    assert all(path.read_bytes() == value for path, value in protected.items())
    assert not (queue / '.household_geometry_append.lock').exists()


def test_scanner_classifies_nonoriginal_origins_and_rejects_duplicate_supplementary_aliases(tmp_path):
    release = {'unified_selection': True, 'tasks': [
        {'alias': 'T01', 'origin': 'original'}, {'alias': 'GF2-1', 'origin': 'household_geometry'},
        {'alias': 'FUTURE', 'origin': 'future_candidate'}]}
    assert [(group, alias) for group, alias, _ in scanner._selected_tasks(release, {'tasks': []})] == [
        ('release', 'T01'), ('candidate', 'GF2-1'), ('candidate', 'FUTURE')]
    path = tmp_path / 'index.json'
    item = {'document_alias': 'GF2-1', 'row_id': 'GF2-1', 'job_id': 'exact_job'}
    write_json(path, {'schema': 'wia_capture_review_index_v1', 'items': [item]})
    assert scanner._supplementary_items([path]) == {'GF2-1': item}
    with pytest.raises(ValueError, match='Duplicate supplementary'):
        scanner._supplementary_items([path, path])


def test_scanner_household_uses_exact_supplementary_job_before_legacy_native_paths(tmp_path, monkeypatch):
    _, _, _, index, tasks, items = household_fixture(tmp_path)
    selected = tasks[0]
    release, candidates, review = tmp_path / 'release.json', tmp_path / 'candidates.json', tmp_path / 'review.json'
    write_json(release, {'unified_selection': True, 'tasks': [selected]})
    write_json(candidates, {'tasks': []})
    write_json(review, {'items': []})
    native = tmp_path / 'legacy_native'
    write_json(native / 'GF2-1/capture.json', {'status': 'captured'})
    chosen = []

    def metadata(row, selected):
        row.update(load_status='loaded', files={'scene': {}, 'scene_source': {}})

    def capture(row, item=None, candidate_dir=None):
        chosen.append((item, candidate_dir))
        row['capture'] = {'state': 'missing'}
        row['audit_coverage'] = {'measured_inventory_available': False, 'full_measured_inventory_declared': False,
                                 'kinematic_audit_report_available': False, 'full_kinematic_audit_declared': False}

    monkeypatch.setattr(scanner, '_metadata', metadata)
    monkeypatch.setattr(scanner, '_capture', capture)
    assert scanner.main(['--release-catalogue', str(release), '--candidate-catalogue', str(candidates),
        '--review-index', str(review), '--supplementary-review-index', str(index),
        '--native-captures', str(native), '--candidate-captures', str(native),
        '--output', str(tmp_path / 'scan')]) == 0
    assert chosen == [(items[0], None)]
    report = json.loads((tmp_path / 'scan/report.json').read_text())
    assert report['summary']['groups'] == {'candidate': 1}
    assert report['inputs']['supplementary_review_indices'][0]['file'] == str(index)
