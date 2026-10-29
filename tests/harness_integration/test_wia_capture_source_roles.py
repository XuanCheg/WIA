"""Check sealed QA/native source roles without native capture or production jobs.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_capture_source_roles.py
The T23 fixture copies real task, scene and both source files into a temporary
snapshot. No source, public/scoring data, simulator or capture queue is modified.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

import wia.runtime.scene_capture_adapters as adapters
from wia.runtime.scene_capture_queue import sha256
from wia.tasks.sources import resolve_source
from harness_integration.test_task_edit2_capture_protocol import submission


ROOT = __import__('wia.resources', fromlist=['ROOT']).ROOT


@pytest.fixture
def t23(tmp_path):
    family = ROOT / 'explore_to_achieve/wia_v1/rigid_family_v1'
    originals = {'task': family / 'tasks/rigid_t23_v1.yaml', 'scene': family / 'scenes/rigid_t23_v1.yaml'}
    raw = {kind: yaml.safe_load(path.read_text()) for kind, path in originals.items()}
    qa_identity = raw['task']['source']['json']
    native_identity = raw['scene']['source']['json']
    assert qa_identity.endswith('__criterion_4.json') and native_identity.endswith('_revised.json')
    assert qa_identity != native_identity and raw['task']['scene'] == raw['scene']['scene']
    snapshot = tmp_path / 'input'
    data_root = snapshot / 'data'
    snapshot.mkdir()
    for kind, original in originals.items():
        (snapshot / (kind + '.yaml')).write_bytes(original.read_bytes())
    members, source_originals = [], {}
    for identity in (qa_identity, native_identity):
        source = resolve_source(identity)
        frozen = data_root / identity
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(source.read_bytes())
        source_originals[source] = source.read_bytes()
        members.append({'source_identity': identity, 'source_file': str(source),
                        'snapshot_source_file': str(frozen), 'source_sha256': sha256(frozen)})
    native_raw = json.loads((data_root / native_identity).read_bytes())
    manifest = {'job_id': 'capture_t23_cpu_source_roles', 'capture_signature': 't23_cpu_source_roles',
        'snapshot_dir': str(snapshot), 'snapshot_data_root': str(data_root),
        'snapshot_task_file': str(snapshot / 'task.yaml'), 'snapshot_scene_file': str(snapshot / 'scene.yaml'),
        'snapshot_source_file': str(data_root / native_identity), 'source_identity': native_identity,
        'capture_source_members': members, 'task_sha256': sha256(snapshot / 'task.yaml'),
        'input_hashes': {str(path.relative_to(snapshot)): sha256(path) for path in snapshot.rglob('*') if path.is_file()},
        'scene_model': native_raw['scene'], 'room': native_raw['room'], 'variant': 'tabletop',
        'seed': raw['task']['defaults']['seed'], 'physical_task_id': raw['task']['id'],
        'capture_initial_agent_bundle': True}
    return NS(manifest=manifest, selected=adapters.load_capture_scene(manifest), raw=raw,
              qa_identity=qa_identity, native_identity=native_identity, source_originals=source_originals)


def test_real_t23_frozen_qa_and_native_sources_bind_without_native_source_substitution(t23):
    before = deepcopy(t23.manifest)
    selected = t23.selected
    selected_source = selected.source_path
    session = NS(background_initialization={'main_owned': 'preserve'})
    adapters.bind_capture_inputs(session, selected, t23.manifest)
    task = session.loaded_task
    assert task.definition.kind == 'category_robot_task'
    assert task.definition.source.json_path == t23.qa_identity
    assert task.source_path == Path(t23.manifest['snapshot_data_root']) / t23.qa_identity
    assert task.source_sha256 == sha256(task.source_path)
    assert selected.definition.source.json_path == t23.native_identity
    assert selected.source_path == selected_source == Path(t23.manifest['snapshot_source_file'])
    assert selected.source_sha256 == sha256(selected_source)
    assert task.source_path != selected.source_path
    assert task.definition.public.model_dump(mode='json') == t23.raw['task']['public']
    assert task.definition.private.evaluation.model_dump(mode='json') == t23.raw['task']['private']['evaluation']
    identity = session.capture_input_identity
    assert identity['task_source_identity'] == t23.qa_identity
    assert identity['scene_source_identity'] == t23.native_identity
    assert identity['task_source_sha256'] == task.source_sha256
    assert identity['scene_source_sha256'] == selected.source_sha256
    assert session.background_initialization == {'main_owned': 'preserve'}
    assert session.capture_initial_agent_bundle is True
    assert t23.manifest == before
    assert {path: path.read_bytes() for path in t23.source_originals} == t23.source_originals
    assert 'task_source_identity' not in task.public_payload


@pytest.mark.parametrize('failure', ['missing_members', 'missing_qa', 'missing_native', 'duplicate_qa',
    'duplicate_native', 'member_hash', 'input_hash', 'wrong_qa_snapshot', 'wrong_native_snapshot',
    'escaped_identity', 'escaped_snapshot', 'missing_snapshot', 'native_manifest_identity'])
def test_two_source_roles_require_exact_unique_existing_sealed_members(t23, failure):
    manifest = t23.manifest
    members = manifest['capture_source_members']
    if failure == 'missing_members':
        manifest.pop('capture_source_members')
    elif failure == 'missing_qa':
        members.pop(0)
    elif failure == 'missing_native':
        members.pop(1)
    elif failure.startswith('duplicate_'):
        members.append(deepcopy(members[0 if failure.endswith('qa') else 1]))
    elif failure == 'member_hash':
        members[0]['source_sha256'] = 'wrong'
    elif failure == 'input_hash':
        manifest['input_hashes']['data/' + t23.qa_identity] = 'wrong'
    elif failure == 'wrong_qa_snapshot':
        members[0]['snapshot_source_file'] = members[1]['snapshot_source_file']
    elif failure == 'wrong_native_snapshot':
        members[1]['snapshot_source_file'] = members[0]['snapshot_source_file']
    elif failure == 'escaped_identity':
        members.append({'source_identity': '../outside.json', 'snapshot_source_file': '/outside.json', 'source_sha256': 'wrong'})
    elif failure == 'escaped_snapshot':
        members[0]['snapshot_source_file'] = str(Path(manifest['snapshot_data_root']) / '..' / 'outside.json')
    elif failure == 'missing_snapshot':
        Path(members[1]['snapshot_source_file']).unlink()
    else:
        manifest['source_identity'] = t23.qa_identity
    session = NS()
    with pytest.raises(ValueError):
        adapters.bind_capture_inputs(session, t23.selected, manifest)
    assert not hasattr(session, 'loaded_task')


def test_source_member_symlink_cannot_escape_the_snapshot_root(t23, tmp_path):
    member = t23.manifest['capture_source_members'][1]
    sealed = Path(member['snapshot_source_file'])
    outside = tmp_path / 'outside_native.json'
    outside.write_bytes(sealed.read_bytes())
    sealed.unlink()
    sealed.symlink_to(outside)
    with pytest.raises(ValueError, match='sealed data-root'):
        adapters.bind_capture_inputs(NS(), t23.selected, t23.manifest)


def test_wrong_task_source_identity_cannot_fall_back_to_a_native_member(t23):
    manifest = t23.manifest
    task_path = Path(manifest['snapshot_task_file'])
    task = yaml.safe_load(task_path.read_text())
    wrong_identity = str(Path(t23.qa_identity).with_name('unsealed_qa.json'))
    wrong_source = Path(manifest['snapshot_data_root']) / wrong_identity
    wrong_source.write_bytes((Path(manifest['snapshot_data_root']) / t23.qa_identity).read_bytes())
    task['source']['json'] = wrong_identity
    task_path.write_text(yaml.safe_dump(task), encoding='utf-8')
    manifest['task_sha256'] = manifest['input_hashes']['task.yaml'] = sha256(task_path)
    with pytest.raises(ValueError, match='own sealed member'):
        adapters.bind_capture_inputs(NS(), t23.selected, manifest)


def test_sealed_native_source_metadata_must_match_the_selected_scene_room(t23):
    manifest = t23.manifest
    member = manifest['capture_source_members'][1]
    source = Path(member['snapshot_source_file'])
    raw = json.loads(source.read_bytes())
    raw['room'] = 'different_room_0'
    source.write_text(json.dumps(raw), encoding='utf-8')
    member['source_sha256'] = manifest['input_hashes']['data/' + t23.native_identity] = sha256(source)
    selected = adapters.load_capture_scene(manifest)
    with pytest.raises(ValueError):
        adapters.bind_capture_inputs(NS(), selected, manifest)


@pytest.mark.parametrize('field', ['source_path', 'source_sha256'])
def test_sealed_qa_member_must_match_actual_loaded_task_path_and_hash(t23, monkeypatch, field):
    original = adapters.load_task

    def foreign(*args, **kwargs):
        task = original(*args, **kwargs)
        value = t23.selected.source_path if field == 'source_path' else t23.selected.source_sha256
        return replace(task, **{field: value})

    monkeypatch.setattr(adapters, 'load_task', foreign)
    with pytest.raises(ValueError, match='own sealed member'):
        adapters.bind_capture_inputs(NS(), t23.selected, t23.manifest)


@pytest.mark.parametrize('field,value', [('scene_model', 'OtherScene'), ('room', 'other_room'),
                                      ('variant', 'original'), ('seed', 7), ('physical_task_id', 'other_task')])
def test_separate_roles_still_require_exact_capture_scene_variant_seed_and_task_id(t23, field, value):
    t23.manifest[field] = value
    with pytest.raises(ValueError):
        adapters.bind_capture_inputs(NS(), t23.selected, t23.manifest)


def test_separate_roles_still_require_identical_task_and_scene_patches(t23):
    definition = t23.selected.definition
    variant = definition.scene.variants['tabletop']
    definition.scene.variants['tabletop'] = variant.model_copy(update={'patches': ()})
    with pytest.raises(ValueError, match='initialization'):
        adapters.bind_capture_inputs(NS(), t23.selected, t23.manifest)


def test_separate_source_branch_is_restricted_to_category_task_definitions(t23):
    manifest = t23.manifest
    task_path = Path(manifest['snapshot_task_file'])
    task = yaml.safe_load(task_path.read_text())
    for key in ('kind', 'status', 'repair'):
        task.pop(key, None)
    task_path.write_text(yaml.safe_dump(task), encoding='utf-8')
    manifest['task_sha256'] = manifest['input_hashes']['task.yaml'] = sha256(task_path)
    with pytest.raises(ValueError, match='task/source/scene'):
        adapters.bind_capture_inputs(NS(), t23.selected, manifest)


def test_existing_same_source_snapshot_protocol_keeps_binding_behavior(submission):
    result = submission.publish()
    manifest = json.loads(Path(result['capture_input']).read_bytes())
    selected = adapters.load_capture_scene(manifest)
    session = NS()
    adapters.bind_capture_inputs(session, selected, manifest)
    assert session.loaded_task.source_path == selected.source_path
    assert session.loaded_task.source_sha256 == selected.source_sha256
    assert session.loaded_task.run_mode == 'diagnostic'
    assert 'task_source_identity' not in session.capture_input_identity


def test_sealed_capture_propagates_and_resets_edge_initialization_policy():
    manifest = {'job_id': 'policy_fixture', 'capture_signature': 'sealed_signature',
                'seed': 0, 'variant': 'adapted', 'current_support_edge_policy': True}
    session = NS()
    adapters.bind_capture_inputs(session, NS(), manifest)
    assert session._capture_release_edge_policy is True
    manifest.pop('current_support_edge_policy')
    adapters.bind_capture_inputs(session, NS(), manifest)
    assert session._capture_release_edge_policy is False
