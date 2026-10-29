"""CPU immutable-source/task and first-boundary screenshot protocol checks.

Run ``python -B -m pytest -q tests/test_task_edit2_capture_protocol.py
tests/test_capture_camera_refresh.py tests/test_esi_category_campaign.py``.
Fixtures execute real sealing/loaders/observation acquisition, never native GPU,
model calls, controls, physics or semantic/physical qualification.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

import wia.runtime.esi_category_capture as bridge
import wia.runtime.scene_capture_adapters as adapters
from wia.authoring.conversion import SourceInstance, ConversionContext, original_scene, physical_task
from wia.runtime.scene_capture_queue import sha256, verify_snapshot
from core.robot_cameras import ADDITIONAL_RGB_NAMES, RGB_TO_VIEW
from wia.runtime.sim.esi_observations import ESIObservationSource
from core.observation_config import ObservationConfig


@pytest.fixture
def submission(tmp_path, monkeypatch):
    identities = ['CPU Fixture/Capture Protocol/House/room_0/one.json',
        'CPU Fixture/Capture Protocol/House/room_0/two.json']
    paths = []
    for identity in identities:
        path = tmp_path / 'sources' / identity
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'scene': 'House', 'room': 'room_0', 'question': 'Inspect this scene.'}))
        paths.append(path)
    src = SourceInstance(paths[0], identities[0], 'CPU Fixture', 'Capture Protocol')
    scene = original_scene(src, ConversionContext(tmp_path, seed=7))
    task = physical_task(src, ConversionContext(tmp_path, seed=7),
        public={'objective': 'Inspect the original scene.', 'success_criteria': 'Answer the observed question.',
            'environment': 'Original indoor scene.', 'constraints': []}, bindings={},
        evaluation={'type': 'category_operation', 'module': 'wia.authoring.converters.temporal_understanding_agent_observation',
            'operation': 'capture_fixture', 'binding_keys': [], 'parameters': {}})
    scene_path, task_path = tmp_path / 'scene.yaml', tmp_path / 'task.yaml'
    scene_path.write_text(yaml.safe_dump(scene)); task_path.write_text(yaml.safe_dump(task))
    code = tmp_path / 'code'; code.mkdir()
    frozen = {'code_root': str(code), 'code_commit': 'frozen-cpu', 'esi_commit': 'esi-cpu', 'implementation_hashes': {}}
    monkeypatch.setattr(bridge, 'converter_runtime_dependencies', lambda *args: {})
    record = {'candidate_id': 'category_capture_fixture', 'revision': 'revision-cpu', 'source_identity': identities[0],
        'source_sha256': sha256(paths[0]), 'original_source': str(paths[0]), 'status': 'generated',
        'scene_file': str(scene_path), 'scene_sha256': sha256(scene_path), 'task_file': str(task_path),
        'task_sha256': sha256(task_path), 'variant': 'original', 'seed': 7,
        'big_task': 'CPU Fixture', 'small_task': 'Capture Protocol', 'limitations': ['Native pending.'],
        'reuse_level': 'L2', 'converter_version': 'cpu', 'warning_codes': [], 'assumptions': [],
        'capture_initial_agent_bundle': True, 'task_edit2_capture_campaign': True, 'task_edit2_variant_key': 'cpu-variant',
        'capture_source_members': [{'source_identity': identity, 'source_file': str(path), 'source_sha256': sha256(path)}
            for identity, path in zip(identities, paths)]}
    def publish():
        return bridge.publish_record(record, tmp_path / 'queue', frozen=frozen)
    return NS(record=record, publish=publish, paths=paths, identities=identities,
        frozen=frozen, queue_root=tmp_path / 'queue')


def test_seal_contains_every_exact_member_before_hashes_and_task_binding(submission):
    result = submission.publish()
    assert result['job_id'], result
    manifest = json.loads(Path(result['capture_input']).read_text())
    verify_snapshot(manifest)
    assert manifest['task_edit2_capture_campaign'] is True
    assert manifest['capture_initial_agent_bundle'] is True
    assert manifest['task_edit2_variant_key'] == 'cpu-variant'
    assert 'initial_overview_rgb' in manifest['expected_views']
    assert set(RGB_TO_VIEW.values()) <= set(manifest['expected_views'])
    assert len(manifest['capture_source_members']) == 2
    for identity, source in zip(submission.identities, submission.paths):
        assert manifest['input_hashes']['data/' + identity] == sha256(source)
        assert (Path(manifest['snapshot_data_root']) / identity).read_bytes() == source.read_bytes()
    selected = adapters.load_capture_scene(manifest)
    session = NS()
    adapters.bind_capture_inputs(session, selected, manifest)
    assert session.loaded_task.path == Path(manifest['snapshot_task_file'])
    assert session.loaded_task.file_sha256 == manifest['input_hashes']['task.yaml']
    assert session.loaded_task.source_path == Path(manifest['snapshot_source_file'])
    assert session.loaded_task.seed == 7 and session.loaded_task.run_mode == 'diagnostic'
    assert session.capture_initial_agent_bundle is True
    # Later donor mutation cannot change any sealed world-member bytes.
    submission.paths[1].write_text('changed original after sealing')
    verify_snapshot(manifest)


@pytest.mark.parametrize('failure', ['unsafe', 'duplicate', 'wrong_hash', 'missing_primary', 'wrong_primary_file'])
def test_invalid_explicit_source_members_fail_closed_before_job_sealing(submission, failure):
    rows = submission.record['capture_source_members']
    if failure == 'unsafe': rows[1]['source_identity'] = '../CPU/Capture/House/room_0/two.json'
    elif failure == 'duplicate': rows.append(deepcopy(rows[0]))
    elif failure == 'wrong_hash': rows[1]['source_sha256'] = 'stale-hash'
    elif failure == 'missing_primary': rows.pop(0)
    else: rows[0]['source_file'] = rows[1]['source_file']
    result = submission.publish()
    assert result['job_id'] is None and result['no_native_image']
    assert 'capture_submission_error' in result


def test_legacy_sealing_keeps_single_source_and_no_new_bundle_flag(submission):
    for key in ('capture_source_members', 'capture_initial_agent_bundle', 'task_edit2_capture_campaign', 'task_edit2_variant_key'):
        submission.record.pop(key)
    result = submission.publish()
    manifest = json.loads(Path(result['capture_input']).read_text())
    assert 'capture_source_members' not in manifest and 'capture_initial_agent_bundle' not in manifest
    assert 'initial_overview_rgb' not in manifest['expected_views']
    assert len([key for key in manifest['input_hashes'] if key.startswith('data/')]) == 1


def test_new_campaign_identical_definitions_keep_distinct_exact_review_aliases(submission):
    first = submission.publish()
    other = deepcopy(submission.record)
    other.update(candidate_id='category_another_exact_review_alias', task_edit2_variant_key='another-current-variant')
    second = bridge.publish_record(other, submission.queue_root, frozen=submission.frozen)
    assert first['job_id'] and second['job_id'] and first['job_id'] != second['job_id']
    for result, record in ((first, submission.record), (second, other)):
        manifest = json.loads(Path(result['capture_input']).read_text())
        verify_snapshot(manifest)
        assert manifest['candidate_id'] == manifest['task_edit2_capture_binding']['candidate_id'] == record['candidate_id']
        assert manifest['task_edit2_capture_binding']['task_edit2_variant_key'] == record['task_edit2_variant_key']
    first_manifest = json.loads(Path(first['capture_input']).read_text())
    second_manifest = json.loads(Path(second['capture_input']).read_text())
    assert first_manifest['fingerprint'] == second_manifest['fingerprint']
    assert first_manifest['input_hashes'] == second_manifest['input_hashes']


def test_new_campaign_member_hash_changes_job_identity_with_same_alias_and_definitions(submission, tmp_path):
    first = submission.publish()
    donor = submission.paths[1]
    donor.write_text(json.dumps({'scene': 'House', 'room': 'room_0', 'question': 'Changed secondary member.'}))
    submission.record['capture_source_members'][1]['source_sha256'] = sha256(donor)
    second = bridge.publish_record(submission.record, tmp_path / 'second_queue', frozen=submission.frozen)
    assert first['job_id'] and second['job_id'] and first['job_id'] != second['job_id']
    original = json.loads(Path(first['capture_input']).read_text())
    revised = json.loads(Path(second['capture_input']).read_text())
    assert original['candidate_id'] == revised['candidate_id']
    assert original['fingerprint'] == revised['fingerprint']
    assert original['task_sha256'] == revised['task_sha256']
    assert original['task_edit2_capture_binding']['source_members'] != revised['task_edit2_capture_binding']['source_members']


@pytest.mark.parametrize('failure', ['task_hash', 'task_source', 'declared_seed', 'scene_variant'])
def test_loaded_task_cannot_be_a_foreign_or_unsealed_definition(submission, failure):
    result = submission.publish(); manifest = json.loads(Path(result['capture_input']).read_text())
    selected = adapters.load_capture_scene(manifest)
    if failure == 'task_hash': manifest['task_sha256'] = 'foreign'
    elif failure == 'task_source':
        selected = deepcopy(selected); object.__setattr__(selected, 'source_sha256', 'foreign')
    elif failure == 'declared_seed':
        manifest['seed'] = 8; object.__setattr__(selected, 'seed', 8)
    else:
        # Same source and requested variant, but a different initialization.
        from wia.tasks.schema import RobotPosePatch
        selected.definition.scene.variants['original'] = selected.definition.scene.variants['original'].model_copy(
            update={'patches': (RobotPosePatch(operation='robot_pose', position=[0., 0., 0.], yaw_deg=0.),)})
    with pytest.raises(ValueError):
        adapters.bind_capture_inputs(NS(), selected, manifest)


class NativeBundleSession:
    def __init__(self, *, disabled=False, missing=False, clock_drift=False):
        self.events, self.renders, self.current_phase = [], 0, 1
        self.capture_initial_agent_bundle = True
        self.loaded_task = NS(file_sha256='frozen-task')
        self.capture_input_identity = {'task_sha256': 'frozen-task', 'seed': 7}
        self.robot = NS()
        self.scene_camera = None
        self.global_cameras = {}
        self.pipeline = NS(og=NS(sim=NS(current_time=1.25, render=self.render)))
        self.initial_overview_identity = {'episode_id': 'reset-r', 'reset_id': 'r', 'seed': 7}
        self.agent_observation_config = ObservationConfig(head_rgb=not disabled, wrist_rgb=not disabled,
            additional_robot_rgb=not disabled)
        self._missing, self._clock_drift = missing, clock_drift
        self._observation_source = ESIObservationSource(self, task_text='CPU read-only capture')

    @property
    def observation_source(self):
        self.events.append('source')
        return self._observation_source

    def render(self):
        self.renders += 1

    def invalidate_camera_cache(self):
        pass

    def observe(self, **kwargs):
        self.events.append(('mounted', self.current_phase, self.renders))
        values = {'head_rgb': np.full((4, 6, 3), 10 * self.current_phase + self.renders, np.uint8),
            'wrist_rgb': np.full((4, 6, 3), 20 * self.current_phase + self.renders, np.uint8)}
        values.update({name: np.full((4, 6, 3), 30 + index + self.renders, np.uint8)
            for index, name in enumerate(ADDITIONAL_RGB_NAMES) if not (self._missing and index == 5)})
        return values

    def initial_overview_capture(self, *, snapshot_id, captured_at):
        self.events.append('frozen_global')
        if self._clock_drift:
            self.pipeline.og.sim.current_time += .01
        self.render()
        return {**self.initial_overview_identity, 'rgb': np.full((4, 6, 3), 90, np.uint8),
            'snapshot_id': snapshot_id, 'captured_at': captured_at, 'extent_xy_m': [0., 0., 2., 2.],
            'image_right_world_xy': [1., 0.], 'image_up_world_xy': [0., 1.],
            'robot_start_position_m': [0., 0., 0.], 'robot_start_yaw_rad': 0.}

    def configure_scene_camera(self):
        self.events.append('human_scene_setup'); self.scene_camera = object()

    def configure_human_cameras(self, *args, **kwargs):
        self.events.append('human_global_setup'); self.global_cameras = {'global_diagonal': object()}

    def _sensor_rgb(self, sensor):
        return np.full((4, 6, 3), 150 + self.renders, np.uint8)


@pytest.mark.parametrize('disabled', [False, True])
def test_first_global_and_all_mounted_use_real_snapshot_before_human_camera_changes(disabled):
    session = NativeBundleSession(disabled=disabled)
    selected = NS(definition=NS(runtime=NS(viewer_config='configs/esi_viewer.json')))
    expected = ['initial_overview_rgb', *RGB_TO_VIEW.values(), 'scene', 'global_diagonal']
    views, _ = adapters.capture_loaded_views(session, selected, expected)
    mounted = [event for event in session.events if isinstance(event, tuple)]
    assert len(mounted) == 2
    assert session.events.index('frozen_global') < session.events.index('human_scene_setup')
    assert mounted[-1][2] < session.renders
    assert np.all(views['head'] == 10 + mounted[-1][2])
    assert np.all(views['wrist'] == 20 + mounted[-1][2])
    assert np.all(views['right_wrist'] == 30 + mounted[-1][2])
    assert np.all(views['initial_overview_rgb'] == 90)
    assert not views['head'].flags.writeable
    assert session.pipeline.og.sim.current_time == 1.25
    evidence = session.capture_initial_agent_bundle_evidence
    assert evidence['available'] and evidence['snapshot_id'] == 2 and evidence['action_step'] == 0
    assert evidence['robot_actions'] == evidence['model_calls'] == evidence['physics_steps'] == 0
    before = views['head'].copy()
    later, _ = adapters.capture_loaded_views(session, selected, expected)
    np.testing.assert_array_equal(later['head'], before)
    assert len([event for event in session.events if isinstance(event, tuple)]) == 2


@pytest.mark.parametrize('failure', ['missing_view', 'missing_robot', 'clock_drift', 'already_observed'])
def test_incomplete_or_late_native_bundle_never_becomes_available(failure):
    session = NativeBundleSession(missing=failure == 'missing_view', clock_drift=failure == 'clock_drift')
    if failure == 'missing_robot': session.robot = None
    if failure == 'already_observed': session._observation_source.capture(action_boundary=True)
    with pytest.raises((RuntimeError, ValueError)):
        adapters.capture_loaded_views(session, NS(), ['initial_overview_rgb', 'head'])
    assert session.capture_initial_agent_bundle_evidence['available'] is False
    assert 'error' in session.capture_initial_agent_bundle_evidence
    assert 'human_scene_setup' not in session.events


def test_initial_bundle_survives_later_native_qualification_and_viewer_captures():
    session = NativeBundleSession()
    first = adapters._initial_agent_bundle(session)
    first_id = session.capture_initial_agent_bundle_evidence['snapshot_id']
    session.current_phase = 2
    session._observation_source.capture(action_boundary=True)
    assert session._observation_source.latest_snapshot.observation.snapshot_id > first_id
    selected = NS(definition=NS(runtime=NS(viewer_config='configs/esi_viewer.json')))
    views, _ = adapters.capture_loaded_views(session, selected, ['initial_overview_rgb', *RGB_TO_VIEW.values()])
    for name, rgb in first.items():
        np.testing.assert_array_equal(views[name], rgb)
    assert session.capture_initial_agent_bundle_evidence['snapshot_id'] == first_id
    assert session.events.count('frozen_global') == 1
