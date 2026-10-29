"""Check coverage, failure review and physical success boundaries.
Run: python -m pytest -q tests/test_esi_category_campaign.py
All images and failures here are synthetic CPU fixtures, not native evidence.
"""
from pathlib import Path
from PIL import Image
import pytest
from wia.runtime.esi_category_capture import publish_review
from wia.authoring.conversion import ConversionContext, ConverterBase, original_scene, sample_records
from wia.runtime.esi_category_runner import aggregate_feedback
from wia.runtime.esi_category_scheduler import CategoryScheduler
from wia.runtime.scene_capture_queue import CaptureQueue, read_json, write_json
from wia.runtime.scene_review import ReviewConflict, SceneReview
from wia.native.esi_category_evaluation import stack_terminal_checks


class FixtureConverter(ConverterBase):
    category = ('CPU Fixture', 'Failure Coverage')
    version = 'fixture-1'

    def inspect(self, source):
        return {'eligible': source.raw.get('id') != 'unconvertible',
                'reason': 'Synthetic source lacks a meaningful robot goal.'}

    def convert(self, source, context):
        return {'scene_definition': original_scene(source, context), 'status': 'generated',
            'assumptions': [{'doubt': 'CPU-only fixture', 'chosen_assumption': 'Keep pending',
                'alternatives': ['Native qualification'], 'scope': 'fixture', 'risk': 'Unverified'}]}


@pytest.fixture
def campaign(tmp_path):
    corpus = tmp_path / 'corpus'
    for index, sample in enumerate(('valid', 'unconvertible', 'malformed')):
        path = corpus / 'CPU Fixture/Failure Coverage/RoomModel/room_0' / (sample + '.json')
        write_json(path, {'id': sample, 'big_task': 'CPU Fixture', 'small_task': 'Failure Coverage',
            'scene': 'RoomModel', 'room': 'room_0', 'question': 'Public CPU fixture question',
            'answer': 'PRIVATE_ANSWER_CANARY', 'metadata_json': {}})
        if index == 2:
            path.write_text('{malformed CPU fixture')
    scheduler = CategoryScheduler(tmp_path / 'campaign', corpus=corpus)
    key = next(iter(scheduler.groups))
    manifest = FixtureConverter().convert_all(scheduler.sources(key), ConversionContext(tmp_path / 'batch'))
    code = tmp_path / 'frozen_code'
    code.mkdir()
    frozen = {'code_root': str(code), 'code_commit': 'fixture-working-tree',
              'esi_commit': 'fixture-source', 'implementation_hashes': {}}
    queue_root = tmp_path / 'queue'
    published = publish_review(manifest, queue_root, frozen=frozen)
    return scheduler, manifest, published, queue_root


def test_all_sources_include_failures_and_idempotent_records(campaign):
    scheduler, manifest, _, _ = campaign
    assert manifest['source_count'] == len(manifest['records']) == 3
    assert manifest['counts'] == {'conversion_failed': 2, 'generated': 1}
    assert all(r['failure_stage'] and r['failure_reason'] for r in manifest['records'] if r['status'] != 'generated')
    sample = sample_records(manifest, count=2)
    assert any(r['candidate_id'] in sample and r['status'] != 'generated' for r in manifest['records'])
    root = Path(manifest['records'][0]['diagnostic_dir']).parent.parent
    key = next(iter(scheduler.groups))
    again = FixtureConverter().convert_all(scheduler.sources(key), ConversionContext(root))
    assert [r['candidate_id'] for r in again['records']] == [r['candidate_id'] for r in manifest['records']]


def test_success_source_only_and_no_image_failure_are_reviewable(campaign, tmp_path):
    scheduler, _, records, queue_root = campaign
    assert len(records) == 3
    assert sum(r['job_id'] is not None for r in records) == 2
    assert sum(r['source_only'] for r in records) == 1
    assert sum(bool(r.get('no_native_image')) for r in records) == 1
    store = SceneReview(None, tmp_path / 'review', queue_root=queue_root, campaign_root=scheduler.root)
    catalogue = store.catalogue(page=0, page_size=2)
    assert catalogue['total'] == catalogue['matched'] == 3 and len(catalogue['scenes']) == 2
    assert catalogue['categories'][0]['counts']['failed'] == 2
    for row in store.catalogue()['scenes']:
        assert not row['has_final_preview'] and not row['physical_grasp_verified']
        payload = {'decision': 'approved', 'notes': '', 'revision': 0, 'fingerprint': row['fingerprint']}
        with pytest.raises(ValueError, match='ready'):
            store.save_review(row['id'], payload)
    assert 'PRIVATE_ANSWER_CANARY' not in str(store.catalogue())


def test_gallery_pagination_hydrates_only_selected_records_and_refreshes_failures(campaign, tmp_path, monkeypatch):
    import wia.runtime.esi_category_capture as capture
    scheduler, _, records, queue_root = campaign
    store = SceneReview(None, tmp_path / 'paged_review', queue_root=queue_root, campaign_root=scheduler.root)
    calls = []
    original = capture.collect_diagnostics
    def inspected(record, *args, **kwargs):
        calls.append(record['candidate_id'])
        return original(record, *args, **kwargs)
    monkeypatch.setattr(capture, 'collect_diagnostics', inspected)
    first = store.catalogue(page=0, page_size=1)
    assert first['total'] == first['matched'] == 3 and len(first['scenes']) == 1
    assert calls == [first['scenes'][0]['id']]
    assert first['categories'][0]['counts']['failed'] == 2
    again = store.catalogue(page=0, page_size=1)
    assert again['scenes'][0]['id'] == first['scenes'][0]['id'] and len(calls) == 1
    valid = next(record for record in records if record['status'] == 'generated')
    selected = store.catalogue(filters={'id': valid['candidate_id']}, page=0, page_size=1)
    assert selected['matched'] == 1
    queue = CaptureQueue(queue_root)
    queue.prioritize(valid['job_id'], 100)
    claim = queue.claim({'id': 'CPU-pagination-fixture'})
    assert claim['job_id'] == valid['job_id']
    queue.transition(claim, 'RUNNING')
    queue.transition(claim, 'FAILED', retryable=False, error='Synthetic native load failure')
    queue.release(claim)
    changed = store.catalogue(filters={'id': valid['candidate_id']}, page=0, page_size=1)
    assert changed['scenes'][0]['failure'] == 'Synthetic native load failure'
    assert changed['scenes'][0]['instance_status'] == 'native_load_failed'
    assert not changed['scenes'][0]['has_final_preview']
    assert calls.count(valid['candidate_id']) == 2


def test_partial_pngs_and_logs_survive_failed_capture(campaign, tmp_path):
    _, _, records, queue_root = campaign
    record = next(r for r in records if r['job_id'] and not r['source_only'])
    queue = CaptureQueue(queue_root)
    queue.prioritize(record['job_id'], 100)
    claim = queue.claim({'id': 'CPU fixture worker'})
    assert claim['job_id'] == record['job_id']
    output = queue.job_dir(record['job_id']) / 'attempts/01'
    output.mkdir(parents=True)
    Image.new('RGB', (64, 64), 'red').save(output / 'head.png')
    (output / 'native.log').write_text('Synthetic CPU capture failure')
    write_json(output / 'initialization_complete.json', {'fixture': True})
    queue.transition(claim, 'RUNNING', native_output=str(output))
    queue.transition(claim, 'FAILED', native_output=str(output), failure_stage='image_save', error='Synthetic CPU capture failure')
    queue.release(claim)
    row = next(r for r in SceneReview(None, tmp_path / 'review', queue_root=queue_root).catalogue()['scenes'] if r['id'] == record['candidate_id'])
    assert row['instance_status'] == 'capture_failed'
    assert row['images'][0]['kind'] == 'partial'
    assert row['diagnostic_logs'] and not row['has_final_preview']


def test_reference_history_discovers_new_jobs_without_copying_durable_images(campaign, tmp_path):
    from wia.runtime.esi_category_capture import collect_diagnostics
    from wia.runtime.scene_capture_queue import PROTOCOL, sha256
    _, _, records, queue_root = campaign
    record = next(r for r in records if r['job_id'] and not r['source_only'])
    queue = CaptureQueue(queue_root)
    durable = tmp_path / 'durable_artifacts'
    write_json(queue_root / 'category_storage.json', {'artifact_root': str(durable)})

    def captured(manifest):
        job_id = manifest['job_id']
        output = durable / job_id
        output.mkdir(parents=True)
        images = {}
        for view in manifest['expected_views']:
            path = output / (view + '.png')
            Image.new('RGB', (64, 64), 'blue').save(path)
            images[view] = str(path)
        receipt = {name: manifest[name] for name in (
            'capture_signature', 'fingerprint', 'code_commit', 'esi_commit', 'variant', 'seed')}
        receipt.update(capture_protocol=PROTOCOL, complete=True, images=images)
        write_json(output / 'preview_complete.json', receipt)
        write_json(queue.job_dir(job_id) / 'status.json', {'state': 'REVIEW_PENDING',
            'native_output': str(output), 'receipt': str(output / 'preview_complete.json')})
        return images

    current = queue.manifest(record['job_id'])
    captured(current)
    first = collect_diagnostics(record, queue_root, reference_queues=[queue_root])
    assert first['native_complete']
    assert {image['kind'] for image in first['diagnostic_images']} == {'native'}

    # A newly discovered immutable legacy job must join already cached history.
    old = {**current, 'job_id': 'capture_' + 'f' * 32}
    manifest_path = queue.job_dir(old['job_id']) / 'input/manifest.json'
    write_json(manifest_path, old)
    write_json(queue.job_dir(old['job_id']) / 'accepted.json', {'manifest_sha256': sha256(manifest_path)})
    historical_paths = captured(old)
    second = collect_diagnostics(record, queue_root, reference_queues=[queue_root])
    historical = [image for image in second['diagnostic_images'] if image['kind'] == 'historical']
    assert {image['path'] for image in historical} == set(historical_paths.values())
    assert {image['capture_job_id'] for image in historical} == {old['job_id']}
    assert not (durable / 'reference_evidence').exists()
    assert second['native_complete'] and not second['no_native_image']


def test_capture_identity_includes_owned_evaluator_and_shared_converter_import(tmp_path):
    from wia.runtime.esi_category_capture import converter_runtime_dependencies
    directory = tmp_path / 'core/esi_converters'
    directory.mkdir(parents=True)
    (directory / 'owner.py').write_text('from wia.authoring.converters.shared import verify\n')
    shared = directory / 'shared.py'
    shared.write_text('verify = 1\n')
    first = converter_runtime_dependencies(tmp_path, ['wia.authoring.converters.owner'])
    assert set(first) == {'core/esi_converters/owner.py', 'core/esi_converters/shared.py'}
    shared.write_text('verify = 2\n')
    second = converter_runtime_dependencies(tmp_path, ['wia.authoring.converters.owner'])
    assert first['core/esi_converters/owner.py'] == second['core/esi_converters/owner.py']
    assert first['core/esi_converters/shared.py'] != second['core/esi_converters/shared.py']


def test_repair_routes_to_category_and_retains_conflict_history(campaign, tmp_path):
    scheduler, _, _, queue_root = campaign
    key = next(iter(scheduler.groups))
    scheduler.update(key, 'REVIEW_SUBMITTED')
    store = SceneReview(None, tmp_path / 'review', queue_root=queue_root, campaign_root=scheduler.root)
    row = next(r for r in store.catalogue()['scenes'] if r['instance_status'] == 'conversion_failed')
    payload = {'decision': 'needs_fix', 'notes': 'Synthetic CPU repair guidance', 'revision': 0, 'fingerprint': row['fingerprint']}
    saved = store.save_review(row['id'], payload)
    assert saved['revision'] == 1
    assert scheduler.jobs()[key]['state'] == 'PENDING'
    with pytest.raises(ReviewConflict):
        store.save_review(row['id'], payload)
    payload.update(revision=1, decision='skipped', notes='Synthetic follow-up')
    store.save_review(row['id'], payload)
    history = read_json(store.review_file)['history']
    assert history[0]['review']['notes'] == 'Synthetic CPU repair guidance'
    count = len(list((queue_root / 'feedback').glob('*.json')))
    aggregate_feedback(queue_root, store.review_file, scheduler=scheduler)
    assert len(list((queue_root / 'feedback').glob('*.json'))) == count


def test_scheduler_three_disjoint_slots_and_refill(tmp_path):
    corpus = tmp_path / 'corpus'
    for i in range(4):
        write_json(corpus / f'Big/Small {i}/Scene/Room/sample.json', {'scene': 'Scene', 'room': 'Room'})
    scheduler = CategoryScheduler(tmp_path / 'campaign', corpus=corpus)
    assignments = scheduler.fill_slots()
    assert len(assignments) == 3 and len({a['slot'] for a in assignments}) == 3
    assert len({a['owned_files'][1] for a in assignments}) == 3
    assert scheduler.fill_slots() == []
    scheduler.update(assignments[0]['key'], 'REVIEW_SUBMITTED')
    refill = scheduler.fill_slots()
    assert len(refill) == 1 and refill[0]['slot'] == assignments[0]['slot']
    assert len(CategoryScheduler(scheduler.root, corpus=corpus).summary()['active_supervisors']) == 3


def test_stack_requires_measured_contact_release_and_motion_stability():
    row = {'upper': 'A', 'lower': 'B', 'on_top': True, 'touching': True, 'upright': True,
           'upper_linear_speed': 0., 'lower_linear_speed': 0., 'upper_angular_speed': 0., 'lower_angular_speed': 0.}
    assert stack_terminal_checks([row], None)
    assert not stack_terminal_checks([row], 'A')
    assert not stack_terminal_checks([{**row, 'touching': False}], None)
    assert not stack_terminal_checks([{**row, 'upper_linear_speed': .2}], None)


def test_mount_restoration_uses_captured_native_parent_frame(tmp_path):
    from types import SimpleNamespace
    from wia.tasks.scene import ScenePatches
    from wia.tasks.schema import SceneDefinition
    definition = SceneDefinition.model_validate({
        'id': 'mount_fixture', 'status': 'pending',
        'source': {'provider': 'esi', 'json': 'Big/Small/Scene/Room/sample.json'},
        'scene': {'policy': 'native_then_minimal_patch_else_skip',
            'variants': {'original': {'patches': []}, 'adapted': {'patches': [
                {'operation': 'restore_robot_head_mount', 'phase': 'after_postprocessing'}]}}}})
    selected = SimpleNamespace(definition=definition, variant='adapted', seed=0)
    adapter = ScenePatches(selected, {})
    assert adapter.restore_robot_head_mount
    assert 'restore_robot_head_mount' not in adapter.observation_settings()
    adapter.apply_to_env_config({})
    class Sensor:
        restored = None
        def set_position_orientation(self, **value):
            self.restored = value
    sensor = Sensor()
    session = SimpleNamespace(env=SimpleNamespace(robots=[SimpleNamespace(sensors={'eyes': sensor})]),
        _category_sensor_mounts={'eyes': ([.1, .2, .3], [0., 0., 0., 1.])})
    adapter.apply_after_postprocessing(session)
    assert sensor.restored == {'position': [.1, .2, .3], 'orientation': [0., 0., 0., 1.], 'frame': 'parent'}


def test_durable_archive_rehydrates_disposable_code_cache(tmp_path):
    import shutil
    import tarfile
    from wia.runtime.scene_capture_queue import sha256, verify_snapshot
    code = tmp_path / 'cache/working_fixture'
    code.mkdir(parents=True)
    implementation = code / 'renderer.py'
    implementation.write_text('"""Synthetic CPU frozen renderer fixture."""\n')
    archive_path = tmp_path / 'durable_home_fixture.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        archive.add(code, arcname=code.name)
    snapshot = tmp_path / 'input'
    snapshot.mkdir()
    (snapshot / 'source.json').write_text('{"fixture":true}')
    manifest = {'snapshot_dir': str(snapshot), 'input_hashes': {'source.json': sha256(snapshot / 'source.json')},
        'code_root': str(code), 'implementation_hashes': {'renderer.py': sha256(implementation)},
        'code_archive': str(archive_path), 'code_archive_sha256': sha256(archive_path)}
    shutil.rmtree(code)
    verify_snapshot(manifest)
    assert implementation.read_text() == '"""Synthetic CPU frozen renderer fixture."""\n'


def test_later_assessment_comparison_waits_for_all_cpu_categories(tmp_path, monkeypatch):
    import wia.runtime.esi_category_runner as module
    corpus = tmp_path / 'corpus'
    for i in range(5):
        write_json(corpus / f'Big/Small {i}/Scene/Room/sample.json', {'scene': 'Scene', 'room': 'Room'})
    runner = module.CategoryRunner(tmp_path / 'campaign', corpus=corpus)
    reference = {'available': True, 'sha256': 'supplementary-fixture',
                 'role': 'supplementary_historical_evidence', 'conclusions_binding': False}
    monkeypatch.setattr(module, 'assessment_reference', lambda: reference)
    launches = []
    def dispatch(job, *, assessment_review):
        launches.append(job['key'])
        process = {'fixture_running': True}
        write_json(Path(assessment_review['directory']) / 'process.json', process)
        return process
    monkeypatch.setattr(runner, 'dispatch', dispatch)
    monkeypatch.setattr(module, 'supervisor_alive', lambda process: process.get('fixture_running', False))
    jobs = list(runner.scheduler.jobs().values())
    for job in jobs[:-1]:
        runner.scheduler.update(job['key'], 'SAMPLE_CAPTURING')
    waiting = runner.assessment_review_step()
    assert waiting['state'] == 'WAITING_FOR_CPU_COMPLETION' and not launches
    runner.scheduler.update(jobs[-1]['key'], 'SAMPLE_CAPTURING')
    value = runner.assessment_review_step()
    assert len(launches) == value['counts']['RUNNING'] == 3
    assert value['counts']['READY'] == 2
    # A finished result does not free a slot while that writer is still alive.
    first = value['categories'][0]
    write_json(first['result_file'], {'fingerprint': first['fingerprint'], 'status': 'reviewed'})
    runner.assessment_review_step()
    assert len(launches) == 3
    write_json(Path(first['directory']) / 'process.json', {'fixture_running': False})
    runner.assessment_review_step()
    assert len(launches) == 4
    assert all(j['state'] == 'SAMPLE_CAPTURING' for j in runner.scheduler.jobs().values())


def test_runner_does_not_snapshot_a_converter_while_owner_is_editing(tmp_path):
    import os
    from wia.runtime.esi_category_runner import CategoryRunner
    corpus = tmp_path / 'corpus'
    write_json(corpus / 'CPU Fixture/Failure Coverage/Scene/Room/sample.json', {'scene': 'Scene', 'room': 'Room'})
    runner = CategoryRunner(tmp_path / 'campaign', corpus=corpus)
    assigned = runner.scheduler.fill_slots()[0]
    fields = Path(f'/proc/{os.getpid()}/stat').read_text().split(') ', 1)[1].split()
    write_json(runner.root / 'supervisors' / assigned['key'] / 'process.json',
               {'pid': os.getpid(), 'start_ticks': fields[19]})
    runner.converter = lambda job: FixtureConverter()
    value = runner.step(supervisors='external')
    assert not value['generated']
    assert runner.scheduler.jobs()[assigned['key']]['state'] == 'ASSIGNED'
    assert not (runner.root / 'categories' / assigned['key'] / 'batch_manifest.json').exists()


def test_supplementary_advice_does_not_revise_existing_candidates(campaign):
    scheduler, manifest, _, _ = campaign
    root = Path(manifest['records'][0]['diagnostic_dir']).parent.parent
    original = {r['candidate_id']: (Path(r['diagnostic_dir']) / 'record.json').read_bytes()
                for r in manifest['records']}
    key = next(iter(scheduler.groups))
    context = ConversionContext(root, advice={'assessment_reference': {
        'role': 'supplementary_historical_evidence', 'conclusions_binding': False,
        'interpretation': 'An unverified historical claim conflicts with current source.'}})
    again = FixtureConverter().convert_all(scheduler.sources(key), context)
    assert [r['candidate_id'] for r in again['records']] == [r['candidate_id'] for r in manifest['records']]
    assert all((Path(r['diagnostic_dir']) / 'record.json').read_bytes() == original[r['candidate_id']]
               for r in again['records'])


def test_extraction_requires_actual_container_exit_as_well_as_native_held_lift(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from wia.native.esi_category_evaluation import ExtractLiftEvaluator
    from harness_integration.test_esi_task_evaluation import make_evaluator
    base, target, gripper, observation = make_evaluator(tmp_path / 'base')
    state = {'inside': True}
    inside_type = type('SyntheticNativeInside', (), {})
    states_module = ModuleType('omnigibson.object_states')
    states_module.Inside = inside_type
    monkeypatch.setitem(sys.modules, 'omnigibson.object_states', states_module)
    target.states = {inside_type: SimpleNamespace(get_value=lambda container: state['inside'])}
    container = SimpleNamespace(name='original_container', visual_only=False)
    base.session.env.scene.object_registry = lambda _, name: container if name == container.name else target
    evaluator = ExtractLiftEvaluator(base.session, base.controller,
        {'type': 'extract_lift', 'scene': 'scene', 'room': 'room', 'target': target.name,
         'container': container.name, 'minimum_lift_m': .05}, tmp_path / 'extraction')
    evaluator.begin_episode(observation)
    gripper.update(object_name=target.name, holding=True, constraint_active=True)
    target.height += .06
    assert not evaluator.record(observation, action='MV_UP')['task_success']
    state['inside'] = False
    result = evaluator.record(observation, action='MV_UP')
    assert result['task_success'] and result['criteria']['outside_source_container']
    gripper.update(object_name=None, holding=False, constraint_active=False)
    assert not evaluator.record(observation, action='RELEASE')['task_success']


def test_category_owned_operation_adapter_rejects_native_preconditions_and_measures_success(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from wia.authoring.conversion import physical_task
    from wia.tasks.schema import CategoryTaskDefinition
    from wia.tasks.scene import DeclarativeScene
    from wia.native.esi_category_evaluation import create_category_operation_evaluator
    module_name = 'wia.authoring.converters.cpu_operation_fixture'
    module = ModuleType(module_name)
    state = {'region': 'outside', 'qa_answer': True}
    module.qualify_operation = lambda session, executor, config, bound: {
        'accepted': state['region'] == 'start', 'checks': {'native_start': state['region'] == 'start'}}
    class Evaluator:
        def begin_episode(self, observation):
            return None
        def record(self, observation, *, action):
            return {'task_success': state['region'] == 'destination', 'native_region': state['region']}
    module.create_evaluator = lambda *args: Evaluator()
    monkeypatch.setitem(sys.modules, module_name, module)
    source = SimpleNamespace(relative_path='Fixture/Operation/scene/room/source.json')
    definition = CategoryTaskDefinition.model_validate(physical_task(source, ConversionContext(tmp_path),
        public={'objective': 'Navigate to the destination.', 'success_criteria': 'Reach it physically.',
                'environment': 'Synthetic fixture', 'constraints': []}, bindings={},
        evaluation={'type': 'category_operation', 'module': module_name, 'operation': 'navigate',
                    'parameters': {'private_answer_canary': 'PRIVATE_ANSWER_CANARY'}},
        capabilities=['base_translation']))
    task = SimpleNamespace(definition=definition, variant='original', seed=0,
        scene_name='scene', room_name='room', verify_unchanged=lambda: None)
    adapter = DeclarativeScene(task)
    session = SimpleNamespace(scene_name='scene', room_name='room')
    executor = SimpleNamespace(capabilities=SimpleNamespace(base_translation=True))
    assert adapter.qualify(session, executor, tmp_path / 'reject').status == 'skipped'
    state['region'] = 'start'
    assert adapter.qualify(session, executor, tmp_path / 'accept').status == 'accepted_original'
    evaluator = create_category_operation_evaluator(session, None,
        definition.private.evaluation.model_dump(), tmp_path)
    assert not evaluator.record(None, action='camera_only')['task_success']
    state['region'], state['qa_answer'] = 'destination', False
    assert evaluator.record(None, action='BASE_FWD')['task_success']
    assert 'PRIVATE_ANSWER_CANARY' not in str(definition.public.model_dump())


def test_physics_enable_patch_preserves_original_asset_geometry_and_rejects_collision_removal(tmp_path):
    from types import SimpleNamespace
    from wia.tasks.schema import SceneDefinition
    from wia.tasks.scene import ScenePatches
    source = SimpleNamespace(relative_path='Fixture/Operation/scene/room/source.json')
    raw = original_scene(source, ConversionContext(tmp_path), bindings={'item': {'name': 'original_item',
        'category': 'apple', 'model': 'source_model'}}, patches=[{'operation': 'object_physics',
        'binding': 'item', 'visual_only': False, 'fixed_base': False}])
    definition = SceneDefinition.model_validate(raw)
    task = SimpleNamespace(definition=definition, variant='adapted', seed=0)
    adapter = ScenePatches(task, definition.bindings)
    item = {'name': 'original_item', 'category': 'apple', 'model': 'source_model',
            'scale': [1., 2., 3.], 'position': [4., 5., 6.], 'visual_only': True, 'fixed_base': True}
    config = {'objects': [item]}
    adapter.apply_to_env_config(config)
    assert not item['visual_only'] and not item['fixed_base']
    assert item['scale'] == [1., 2., 3.] and item['position'] == [4., 5., 6.]
    assert not definition.scene.variants['original'].patches
    after_raw = original_scene(source, ConversionContext(tmp_path), bindings=raw['bindings'],
        patches=[{'operation': 'object_physics', 'binding': 'item', 'phase': 'after_postprocessing'}])
    after_definition = SceneDefinition.model_validate(after_raw)
    after_adapter = ScenePatches(SimpleNamespace(definition=after_definition, variant='adapted', seed=0),
                                 after_definition.bindings)
    native = SimpleNamespace(name='original_item', category='apple', model='source_model',
                             visual_only=True, position=[4., 5., 6.])
    after_adapter.apply_to_env_config({'objects': [item]})
    after_adapter.apply_after_postprocessing(SimpleNamespace(env=SimpleNamespace(scene=SimpleNamespace(
        object_registry=lambda kind, name: native))))
    assert not native.visual_only and native.position == [4., 5., 6.]
    class AlreadyPhysical:
        name, category, model = 'original_item', 'apple', 'source_model'
        @property
        def visual_only(self):
            return False
        @visual_only.setter
        def visual_only(self, value):
            raise AssertionError('Native GPU dynamics must not receive a redundant setter.')
    already_physical = AlreadyPhysical()
    no_op_adapter = ScenePatches(SimpleNamespace(definition=after_definition, variant='adapted', seed=0),
                                 after_definition.bindings)
    no_op_adapter.apply_to_env_config({'objects': [item]})
    no_op_adapter.apply_after_postprocessing(SimpleNamespace(env=SimpleNamespace(scene=SimpleNamespace(
        object_registry=lambda kind, name: already_physical))))
    after_raw['scene']['variants']['adapted']['patches'][0]['fixed_base'] = False
    with pytest.raises(ValueError, match='before native construction'):
        SceneDefinition.model_validate(after_raw)
    raw['scene']['variants']['adapted']['patches'][0]['visual_only'] = True
    with pytest.raises(ValueError):
        SceneDefinition.model_validate(raw)


@pytest.mark.parametrize('return_native_result', [True, False])
def test_category_evidence_matches_episode_without_resampling_initial_state(tmp_path, return_native_result):
    from types import SimpleNamespace
    from wia.native.esi_category_evaluation import CategoryEvaluationEnvelope, validate_episode_evaluation
    observation = SimpleNamespace(snapshot_id='first-frame', action_step=0)
    task = SimpleNamespace(definition=SimpleNamespace(kind='category_robot_task', id='fixture'),
        task_text='Complete the measured route.', file_sha256='task-hash', source_sha256='source-hash',
        verify_unchanged=lambda: None)
    class Native:
        def __init__(self):
            self.records, self.samples = 0, 0
        def begin_episode(self, observation):
            self.record(observation, action=None)
        def sample_step(self):
            self.samples += 1
        def record(self, observation, *, action):
            self.records += 1
            value = {'schema': 'category_native_fixture', 'task_success': self.samples >= 2,
                     'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step}
            write_json(tmp_path / 'state.json', value)
            return value if return_native_result else None
    native = Native()
    evaluator = CategoryEvaluationEnvelope(native, task, tmp_path)
    initial = evaluator.begin_episode(observation)
    assert native.records == 1 and native.samples == 0
    assert not validate_episode_evaluation(initial, observation, task.task_text, task)
    evaluator.sample_step()
    evaluator.sample_step()
    observation.snapshot_id, observation.action_step = 'second-frame', 1
    value = evaluator.record(observation, action='BASE_FORWARD')
    assert validate_episode_evaluation(value, observation, task.task_text, task)
    assert value['native_result']['schema'] == 'category_native_fixture'
    for field, wrong in (('task_id', 'other-task'), ('task_file_sha256', 'other-hash'),
                         ('source_sha256', 'other-source'), ('episode', 2),
                         ('snapshot_id', 'stale-frame'), ('action_step', 0),
                         ('task_text', 'Different objective'), ('schema', 'private_grasp_lift_v1')):
        with pytest.raises(RuntimeError, match='current episode'):
            validate_episode_evaluation({**value, field: wrong}, observation, task.task_text, task)
    legacy_task = SimpleNamespace(definition=SimpleNamespace(kind='grasp_lift'))
    legacy = {'schema': 'private_grasp_lift_v1', 'task_success': True, 'task_text': task.task_text,
              'episode': 1, 'snapshot_id': observation.snapshot_id, 'action_step': 1}
    assert validate_episode_evaluation(legacy, observation, task.task_text, legacy_task)
    with pytest.raises(RuntimeError):
        validate_episode_evaluation(value, observation, task.task_text, legacy_task)


def test_robot_free_category_overlay_receives_observation_settings_after_r1_addition(tmp_path):
    from types import SimpleNamespace
    from wia.runtime.sim.esi_omnigibson import ESISession
    built = []
    class R1:
        name, action_dim, controllers = 'R1', 1, {'arm': object()}
        sensors = {'R1:eyes': SimpleNamespace(modalities={'rgb'})}
    module = SimpleNamespace(scene_room=lambda _: ('source_scene', 'source_room'),
        initial_camera=lambda _: ([0., 0., 1.], [0., 0., 0., 1.], {}),
        build_env_objects=lambda _: [], __name__='fixture.original_robot_free')
    def add_robot(config):
        assert not config['robots']
        config['robots'] = [{'type': 'R1'}]
    def construct(configs):
        built.append(configs)
        assert configs['robots'][0]['grasping_mode'] == 'assisted'
        assert configs['robots'][0]['sensor_config']['VisionSensor']['sensor_kwargs']['image_width'] == 64
        return SimpleNamespace(robots=[R1()], scene=SimpleNamespace(scene_model='source_scene'))
    pipeline = SimpleNamespace(select_question=lambda *args: SimpleNamespace(source_path=tmp_path / 'source.json', payload={}),
        load_task_module_for_payload=lambda *args: module, task_preprocess=lambda *args: {},
        task_build_env_config=lambda *args, **kwargs: {'robots': []},
        task_initial_settle_steps=lambda *args: 0, task_postprocess_env=lambda *args: {
            'dynamic_object_names': ['late_original_object'], 'mirror_name': 'source_mirror'},
        og=SimpleNamespace(Environment=construct, sim=SimpleNamespace(render=lambda: None)))
    session = ESISession(tmp_path / 'source.json', camera_size=(64, 64), grasp_mode='assisted')
    session.scene_adapter = SimpleNamespace(apply_to_env_config=add_robot,
        apply_after_postprocessing=lambda _: None)
    session._initialize_scene(pipeline)
    assert len(built) == 1 and session.robot is not None
    assert session.task_object_names == ('late_original_object', 'source_mirror')
    session.scene_adapter.applied = [{'initialization_evidence': {
        'human_framing_names': ['active_box', 'active_item']}}]
    session.task_object_names = ('inactive_offstage_object',)
    session._refresh_task_object_names()
    assert session.task_object_names == ('active_box', 'active_item')


def test_category_human_overview_preserves_original_reference_viewer(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from wia.runtime.sim.esi_omnigibson import ESISession
    import wia.runtime.sim.esi_debug_cameras as cameras
    viewer_changes = []
    def configure(session, config, *, include_robot):
        session.global_cameras = {'global_front': SimpleNamespace(
            get_position_orientation=lambda: ([1., 2., 3.], [0., 0., 0., 1.]))}
    monkeypatch.setattr(cameras, 'configure_global_cameras', configure)
    session = ESISession(tmp_path / 'source.json')
    session.pipeline = SimpleNamespace(set_camera=lambda *args: viewer_changes.append(args),
                                      og=SimpleNamespace(sim=SimpleNamespace(render=lambda: None)))
    session.scene_camera = object()
    session.scene_adapter = SimpleNamespace(variant=SimpleNamespace(patches=[object()]),
        task=SimpleNamespace(definition=SimpleNamespace(kind='scene_patch', id='category_fixture')))
    session.configure_human_cameras(object())
    assert not viewer_changes and session.scene_camera_framing == 'original_source_reference'
    session.scene_adapter.task.definition.id = 'existing_legacy_scene'
    session.configure_human_cameras(object())
    assert len(viewer_changes) == 1 and session.scene_camera_framing == 'patched_task_front_overview'


def test_literal_source_hierarchy_placeholders_do_not_hide_real_mismatches():
    from wia.tasks.loader import source_scene_room
    raw = {'scene': 'Rs_int', 'room': 'living_room_0'}
    assert source_scene_room('Physical Structure/Liquid Volume/unknown_scene/unknown_room/q.json', raw) == (
        'Rs_int', 'living_room_0')
    assert source_scene_room('A/B/Rs_int/living_room_0/q.json', raw) == ('Rs_int', 'living_room_0')
    with pytest.raises(ValueError, match='hierarchy'):
        source_scene_room('A/B/wrong_scene/living_room_0/q.json', raw)
    with pytest.raises(ValueError, match='actual scene/room'):
        source_scene_room('A/B/unknown_scene/unknown_room/q.json', {'scene': '', 'room': None})


def test_failed_category_supervisor_publishes_every_source(campaign, tmp_path, monkeypatch):
    import wia.runtime.esi_category_runner as module
    scheduler, _, _, queue_root = campaign
    runner = module.CategoryRunner(scheduler.root, queue_root=queue_root)
    assigned = runner.scheduler.fill_slots()[0]
    write_json(runner.root / 'supervisors' / assigned['key'] / 'process.json',
               {'pid': 999999999, 'start_ticks': 'missing'})
    frozen = {'code_root': str(tmp_path / 'frozen_code'), 'code_commit': 'fixture-diagnostic',
              'esi_commit': 'fixture-source', 'implementation_hashes': {}}
    monkeypatch.setattr(module, 'freeze_working_code', lambda queue: frozen)
    monkeypatch.setattr(module, 'assessment_reference', lambda: {'available': False})
    monkeypatch.setattr(runner, 'converter', lambda job: None)
    runner.step(supervisors='external', publish_all=True)
    job = runner.scheduler.jobs()[assigned['key']]
    manifest = read_json(job['batch_file'])
    assert job['state'] == 'REVIEW_SUBMITTED'
    assert manifest['counts'] == {'conversion_failed': 3}
    records = [read_json(queue_root / 'category_records' / (row['candidate_id'] + '.json'))
               for row in manifest['records']]
    assert len(records) == 3 and all(records)
    assert sum(record['source_only'] for record in records) == 2
    assert sum(bool(record.get('no_native_image')) for record in records) == 1
    assert all(record['review_submitted'] for record in records)


def test_each_completed_batch_freezes_current_code_and_retains_its_revision(tmp_path, monkeypatch):
    import wia.runtime.esi_category_runner as module
    corpus = tmp_path / 'corpus'
    for small in ('First', 'Later'):
        write_json(corpus / f'CPU Fixture/{small}/Scene/Room/sample.json', {'scene': 'Scene', 'room': 'Room'})
    runner = module.CategoryRunner(tmp_path / 'campaign', corpus=corpus,
                                  config={'refresh_original_reference': False})
    jobs = list(runner.scheduler.jobs().values())
    for job in jobs:
        runner.scheduler.update(job['key'], 'DEFERRED')
    frozen = []
    def freeze(queue):
        code = tmp_path / f'frozen_{len(frozen)}'
        code.mkdir()
        value = {'code_root': str(code), 'code_commit': code.name,
                 'esi_commit': 'fixture-source', 'implementation_hashes': {}}
        frozen.append(value)
        return value
    monkeypatch.setattr(module, 'freeze_working_code', freeze)
    monkeypatch.setattr(module, 'assessment_reference', lambda: {'available': False})
    monkeypatch.setattr(runner, 'converter', lambda job: FixtureConverter())
    for job in jobs:
        runner.scheduler.update(job['key'], 'ASSIGNED')
        runner.step(supervisors='external', publish_all=True)
    assert len(frozen) == 2
    assert [runner.scheduler.jobs()[job['key']]['frozen_code']['code_commit'] for job in jobs] == ['frozen_0', 'frozen_1']
    second = read_json(runner.scheduler.jobs()[jobs[1]['key']]['batch_file'])
    record = read_json(runner.queue_root / 'category_records' / (second['records'][0]['candidate_id'] + '.json'))
    assert read_json(record['capture_input'])['code_commit'] == 'frozen_1'


def test_requested_rule_revision_returns_to_the_category_owner(tmp_path, monkeypatch):
    import wia.runtime.esi_category_runner as module
    corpus = tmp_path / 'corpus'
    write_json(corpus / 'CPU Fixture/Failure Coverage/Scene/Room/sample.json', {'scene': 'Scene', 'room': 'Room'})
    runner = module.CategoryRunner(tmp_path / 'campaign', corpus=corpus)
    assigned = runner.scheduler.fill_slots()[0]
    runner.scheduler.update(assigned['key'], 'REVISING', iterations=1)
    write_json(runner.root / 'supervisors' / assigned['key'] / 'process.json',
               {'pid': 999999999, 'start_ticks': 'missing', 'conversion_iteration': 0})
    launches = []
    monkeypatch.setattr(runner, 'dispatch', lambda job: launches.append(job['key']))
    monkeypatch.setattr(runner, 'converter', lambda job: FixtureConverter())
    value = runner.step(supervisors='codex')
    assert launches == [assigned['key']] and not value['generated']
    assert runner.scheduler.jobs()[assigned['key']]['state'] == 'REVISING'


def test_existing_robot_addition_preserves_source_physics_and_rejects_replacement(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from wia.tasks.schema import SceneDefinition
    from wia.tasks.scene import ScenePatches
    template = tmp_path / 'r1_primitives.yaml'
    template.write_text('robots:\n  - type: R1\n    self_collisions: true\nenv:\n  physics_frequency: 120\n')
    monkeypatch.setitem(sys.modules, 'omnigibson', SimpleNamespace(example_config_path=str(tmp_path)))
    source = SimpleNamespace(relative_path='Fixture/Operation/scene/room/source.json')
    definition = SceneDefinition.model_validate(original_scene(source, ConversionContext(tmp_path),
        patches=[{'operation': 'add_existing_r1'}]))
    task = SimpleNamespace(definition=definition, variant='adapted', seed=0)
    config = {'robots': [], 'objects': [{'name': 'original_item'}], 'env': {'physics_frequency': 240}}
    ScenePatches(task, {}).apply_to_env_config(config)
    assert config['robots'] == [{'type': 'R1', 'self_collisions': True}]
    assert config['env'] == {'physics_frequency': 240} and config['objects'] == [{'name': 'original_item'}]
    assert not definition.scene.variants['original'].patches
    with pytest.raises(ValueError, match='replace or duplicate'):
        ScenePatches(task, {}).apply_to_env_config(config)


def test_owned_initialization_runs_once_and_is_part_of_the_frozen_renderer(campaign, tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from wia.tasks.schema import SceneDefinition
    from wia.tasks.scene import ScenePatches
    from wia.runtime.scene_capture_queue import sha256, verify_snapshot
    module_name = 'wia.authoring.converters.cpu_init_fixture'
    module = ModuleType(module_name)
    calls = []
    def configure(config, parameters):
        calls.append(('configuration', parameters))
        config['fixture_restoration'] = True
        return {'restored': 'original fixture configuration'}
    def initialize(session, parameters):
        calls.append(('native initialization', parameters))
        return {'restored': 'original fixture mechanism'}
    module.configure_scene, module.initialize_scene = configure, initialize
    monkeypatch.setitem(sys.modules, module_name, module)
    patches = [{'operation': 'category_initialization', 'module': module_name,
                'phase': phase, 'parameters': {'fixture': True}}
               for phase in ('before_construction', 'after_postprocessing')]
    source = SimpleNamespace(relative_path='Fixture/Operation/scene/room/source.json')
    definition = SceneDefinition.model_validate(original_scene(source, ConversionContext(tmp_path), patches=patches))
    adapter = ScenePatches(SimpleNamespace(definition=definition, variant='adapted', seed=0), {})
    config = {'objects': []}
    adapter.apply_to_env_config(config)
    adapter.apply_after_postprocessing(SimpleNamespace())
    assert config['fixture_restoration'] and len(calls) == 2
    with pytest.raises(RuntimeError, match='exactly once'):
        adapter.apply_after_postprocessing(SimpleNamespace())
    scheduler, _, _, _ = campaign
    class InitializationConverter(FixtureConverter):
        def convert(self, source, context):
            return {'scene_definition': original_scene(source, context, patches=patches), 'status': 'generated'}
    selected = [source for source in scheduler.sources(next(iter(scheduler.groups))) if source.path.stem == 'valid']
    manifest = InitializationConverter().convert_all(selected, ConversionContext(tmp_path / 'init_batch'))
    code = tmp_path / 'frozen_init'
    dependency = code / (module_name.replace('.', '/') + '.py')
    dependency.parent.mkdir(parents=True)
    dependency.write_text('"""Synthetic CPU initializer renderer dependency."""\n')
    frozen = {'code_root': str(code), 'code_commit': 'fixture-init', 'esi_commit': 'fixture-source',
              'implementation_hashes': {}}
    record = publish_review(manifest, tmp_path / 'init_queue', frozen=frozen)[0]
    capture = read_json(record['capture_input'])
    assert capture['implementation_hashes'][str(dependency.relative_to(code))] == sha256(dependency)
    dependency.write_text('"""Changed synthetic initializer."""\n')
    with pytest.raises(ValueError, match='rendering implementation changed'):
        verify_snapshot(capture)


def test_snapshot_failure_publishes_all_diagnostics_without_ending_runner(campaign, tmp_path, monkeypatch):
    import wia.runtime.esi_category_runner as module
    scheduler, _, _, _ = campaign
    queue_root = tmp_path / 'snapshot_failure_queue'
    runner = module.CategoryRunner(scheduler.root, queue_root=queue_root,
                                  config={'refresh_original_reference': False})
    calls = []
    def failed_snapshot(queue):
        calls.append(queue)
        raise RuntimeError('Synthetic CPU snapshot failure')
    monkeypatch.setattr(module, 'freeze_working_code', failed_snapshot)
    monkeypatch.setattr(module, 'assessment_reference', lambda: {'available': False})
    monkeypatch.setattr(runner, 'converter', lambda job: FixtureConverter())
    runner.step(supervisors='external', publish_all=True)
    assert len(calls) == 2
    assert list(runner.scheduler.jobs().values())[0]['state'] == 'REVIEW_SUBMITTED'
    store = SceneReview(None, tmp_path / 'snapshot_review', queue_root=queue_root)
    rows = store.catalogue()['scenes']
    assert len(rows) == 3 and all(row['no_native_image'] for row in rows)
    valid = next(row for row in rows if row['sample'] == 'valid')
    assert valid['instance_status'] == 'capture_failed' and valid['failure_stage'] == 'code_snapshot'
    assert not valid['has_final_preview'] and 'Synthetic CPU snapshot failure' in valid['failure']
