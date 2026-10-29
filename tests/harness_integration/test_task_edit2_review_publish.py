"""Exercise staged TaskEdit2 cards through the existing LiveSceneReview store.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_task_edit2_review_publish.py
Only temporary files and in-process Review APIs are used; no services start.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from wia.runtime.scene_capture_queue import write_json, read_json
from wia.runtime.scene_review_live import LiveSceneReview
import wia.runtime.commands.publish_task_edit2_review as publish
import wia.runtime.commands.build_task_edit2_review_manifest as review


@pytest.fixture
def staged(tmp_path):
    sources = []
    for index, identity in enumerate(('Cognitive Mapping/Connectivity/test/room/a.json',
                                     'Enumerative Perception/Counting/test/room/b.json')):
        path = tmp_path / 'sources' / identity
        write_json(path, {'scene': 'test', 'room': 'room', 'question': f'Original public question {index}?',
                          'answer': 'Original human-only answer'})
        sources.append({'source_identity': identity, 'source_file': str(path),
                        'source_sha256': publish.sha256(path), 'provenance_status': 'verified_current_source_bytes'})
    public = {'objective': 'Inspect both original sources and submit one shared final answer sheet.',
        'success_criteria': 'Report observed evidence for each public goal.', 'environment': 'The preserved source world.',
        'constraints': ['Keep source furniture unchanged.']}
    task = {'source': {'json': sources[0]['source_identity']}, 'public': public,
        'private': {'bindings': {'target': {'name': 'PRIVATE_BINDING_CANARY'}},
            'evaluation': {'parameters': {'members': [
                {'source_identity': source['source_identity'], 'public_id': f'goal_{i+1}',
                 'public': {'objective': f'Inspect public goal {i+1}.', 'success_criteria': 'Answer from observations.'},
                 'bindings': {'secret': 'PRIVATE_BINDING_CANARY'}} for i, source in enumerate(sources)]}}}}
    task_path, scene_path = tmp_path / 'task.yaml', tmp_path / 'scene.yaml'
    task_path.write_text(yaml.safe_dump(task))
    scene_path.write_text('scene: source_world\n')
    variant = {'variant_key': 'abcd1234', 'candidate_id': 'actual_joint_parent', 'role': 'explicit_derivative',
        'task': {'path': str(task_path), 'sha256': publish.sha256(task_path)},
        'scene': {'path': str(scene_path), 'sha256': publish.sha256(scene_path)},
        'sources': sources, 'source_binding_status': 'verified', 'schema_status': 'schema_valid',
        'native_evidence_ids': ['native_exact'], 'membership_class': 'mixed_core_and_additional_members',
        'original_core_members': [sources[0]['source_identity']],
        'additional_source_members': [sources[1]['source_identity']],
        'pending_gates': ['source_relations'], 'pending_review_gates': ['terminal_scoring_unmeasured'],
        'artifacts': {'explicit_original_or_review_images': []}}
    # exact_native_match also reads definition primary source provenance.
    variant['task']['source'] = {'json': sources[0]['source_identity']}
    variant['scene']['source'] = {'json': sources[0]['source_identity']}
    native_dir = tmp_path / 'native_exact'
    result = {'task_file': str(task_path), 'scene_file': str(scene_path),
        'task_sha256': publish.sha256(task_path), 'scene_sha256': publish.sha256(scene_path),
        'source_sha256': sources[0]['source_sha256'], 'model_calls': 0, 'complete': True}
    write_json(native_dir / 'result.json', result)
    mounted = {}
    from core.robot_cameras import ROBOT_CAMERA_LABELS
    for name in (*ROBOT_CAMERA_LABELS, 'initial_overview_rgb'):
        path = native_dir / (name + '.png')
        path.write_bytes(b'Explicit test image; never decoded during publication.')
        if name != 'initial_overview_rgb':
            mounted[name] = str(path)
    original_image, historical = tmp_path / 'original_esi.png', tmp_path / 'historical.png'
    original_image.write_bytes(b'Original source context.')
    historical.write_bytes(b'Historical baseline image.')
    evidence = {'evidence_id': 'native_exact', 'private_result_file': str(native_dir / 'result.json'),
        'result_sha256': publish.sha256(native_dir / 'result.json'), 'diagnostic_complete': True,
        'native_initialization_status': 'observed_rejected', 'layout_status': 'unknown',
        'execution_status': 'public_control_only', 'scoring_status': 'unavailable',
        'public_control_results': [{'command': {'action': 'BASE'}, 'completed': True}],
        'public_control_success_count': 1,
        'qualification': {'status': 'skipped', 'reason': 'No relation or budget qualification.'},
        'interaction_setup_evidence': [{'phase': 'initial', 'receipt': {'accepted': False,
            'reason': 'Initial obstruction.', 'private': 'PRIVATE_BINDING_CANARY'}}],
        'captures': {'initial_global': str(native_dir / 'initial_overview_rgb.png'),
                     'initial_mounted': mounted, 'other_review_images': []}}
    manifest = {'variants': [deepcopy(variant), {'role': 'original_baseline', 'variant_key': 'deadbeef'}],
        'native_evidence': [evidence], 'core_sources': [{'source_identity': sources[0]['source_identity'],
            'original_esi_images': [str(original_image)], 'historical_review_images': {'paths': [str(historical)]}}]}
    baseline_row = {'id': 'category_aa', 'candidate_id': 'category_aa', 'fingerprint': 'baseline_fp',
        'category': 'Original', 'subcategory': 'Family', 'source': 'Original/Family/test/room/old.json',
        'objective': 'Historical original instructions.', 'native_capture_complete': True,
        'capture': {'state': 'PRESERVED_BASELINE', 'job_id': 'historical_real_job'},
        'prior_feedback': [{'decision': 'approved', 'notes': 'Keep this historical review.'}], 'images': []}
    diagnostic_row = dict(deepcopy(baseline_row), id='category_bb', candidate_id='category_bb',
        fingerprint='other_fp', native_capture_complete=False, capture={'state': 'FAILED', 'job_id': 'real_failed_job'})
    baseline = {'scenes': [baseline_row], 'diagnostics': [diagnostic_row],
        'categories': [{'big_task': 'Original', 'small_task': 'Family', 'source_count': 2, 'counts': {}}],
        'capture_queue': {'root': '/existing/real/queue', 'candidate_count': 2},
        'indexing': {'status': 'ready'}, 'historical_approval_policy': 'Preserve the original history.'}
    manifest_path, baseline_path = tmp_path / 'manifest.json', tmp_path / 'baseline.json'
    write_json(manifest_path, manifest)
    write_json(baseline_path, baseline)
    output = tmp_path / 'staged'
    return {'manifest': manifest, 'manifest_path': manifest_path, 'baseline': baseline,
        'baseline_path': baseline_path, 'output': output, 'alias': 'category_abcd1234',
        'variant': variant, 'native_dir': native_dir, 'tmp_path': tmp_path}


def stage(fixture):
    return publish.publish(fixture['manifest_path'], fixture['baseline_path'], fixture['output'])


def store(fixture):
    return LiveSceneReview(None, fixture['tmp_path'] / 'reviews',
        queue_root=fixture['tmp_path'] / 'empty_real_queue', catalogue_file=fixture['output'] / 'snapshot.json')


def test_preserves_every_baseline_row_and_history_and_appends_only_derivatives(staged):
    original_bytes = staged['baseline_path'].read_bytes()
    snapshot = stage(staged)
    assert snapshot['scenes'] == staged['baseline']['scenes']
    assert snapshot['diagnostics'][:1] == staged['baseline']['diagnostics']
    assert snapshot['capture_queue'] == staged['baseline']['capture_queue']
    assert staged['baseline_path'].read_bytes() == original_bytes
    assert [row['id'] for row in snapshot['diagnostics']] == ['category_bb', staged['alias']]
    row = snapshot['diagnostics'][-1]
    assert row['actual_candidate_id'] == 'actual_joint_parent'
    assert row['selection_status'] == 'provisional_user_review' and row['selected_final_variant'] is False
    assert row['membership_class'] == 'mixed_core_and_additional_members'
    assert snapshot['task_edit2_overlay']['explicit_derivative_cards'] == 1
    assert snapshot['task_edit2_overlay']['native_queue_records_created'] is False
    assert not (staged['output'] / 'jobs').exists()
    assert not (staged['output'] / 'category_records').exists()
    again = stage(staged)
    assert again == snapshot


def test_actual_live_catalogue_detail_public_instructions_and_uncertified_images(staged):
    stage(staged)
    app = store(staged)
    listing = app.catalogue(filters={'scope': 'all'})
    assert {row['id'] for row in listing['scenes']} == {'category_aa', 'category_bb', staged['alias']}
    assert app.catalogue(filters={'scope': 'diagnostics', 'big_task': 'Enumerative Perception',
                                  'small_task': 'Counting'})['matched'] == 1
    detail = app.scene_detail(staged['alias'])
    assert detail['objective'] == 'Inspect both original sources and submit one shared final answer sheet.'
    assert detail['category'] == 'Cognitive Mapping' and detail['subcategory'] == 'Connectivity'
    assert {tag['big_task'] for tag in detail['category_tags']} == {'Cognitive Mapping', 'Enumerative Perception'}
    assert [goal['objective'] for goal in detail['composite']['public_goals']] == ['Inspect public goal 1.', 'Inspect public goal 2.']
    assert detail['composite']['member_count'] == 2
    assert len(detail['images']) == 11
    assert all(image['kind'] == 'partial' for image in detail['images'])
    assert any('Original ESI' in image['label'] for image in detail['images'])
    assert any('Historical baseline' in image['label'] for image in detail['images'])
    assert detail['native_capture_complete'] is False
    assert detail['layout_approval_available'] is True
    assert detail['physical_qualified'] is detail['preview_is_current'] is False
    assert detail['native_initialization_status'] == 'observed_rejected'
    assert detail['execution_status'] == 'public_control_only' and detail['scoring_status'] == 'unavailable'
    assert detail['native_diagnostics'][0]['public_control_success_count'] == 1
    assert detail['native_diagnostics'][0]['interaction_setup_results'][0]['accepted'] is False
    assert 'PRIVATE_BINDING_CANARY' not in json.dumps(listing)
    assert 'PRIVATE_BINDING_CANARY' not in json.dumps(detail)
    assert 'PRIVATE_BINDING_CANARY' not in (staged['output'] / 'cards' / staged['alias'] / 'record.json').read_text()
    saved = app.save_review(staged['alias'], {'decision': 'approved', 'notes': '',
        'fingerprint': detail['fingerprint'], 'revision': 0})
    assert saved['approval_scope'] == 'layout'
    assert not saved['approval_evidence']['native_capture_complete']
    assert app.scene_detail(staged['alias'])['native_initialization_status'] == 'observed_rejected'


@pytest.mark.parametrize('membership', ['core_derivative', 'mixed_core_and_additional_members', 'additional_prototype'])
def test_membership_badges_and_independent_states_reach_existing_ui_display_fields(staged, membership):
    manifest = deepcopy(staged['manifest'])
    manifest['variants'][0]['membership_class'] = membership
    write_json(staged['manifest_path'], manifest)
    snapshot = stage(staged)
    app = store(staged)
    listed = next(row for row in app.catalogue(filters={'scope': 'all'})['scenes'] if row['id'] == staged['alias'])
    detail = app.scene_detail(staged['alias'])
    badge = 'TaskEdit2/provisional/' + membership
    assert listed['core_selection_status'] == detail['core_selection_status'] == badge
    assert detail['membership_class'] == membership
    assert publish.MEMBERSHIP_EXPLANATIONS[membership] in detail['known_limitations']
    assert publish.MEMBERSHIP_EXPLANATIONS[membership] in detail['change_summary']
    displayed = '\n'.join(detail['change_summary'])
    for field in publish.STATUS_FIELDS:
        assert f'{publish.STATUS_LABELS[field]}: {detail[field]}' in displayed
    assert 'Pending gates: source_relations, terminal_scoring_unmeasured.' in displayed
    assert detail['category'] == 'Cognitive Mapping' and detail['subcategory'] == 'Connectivity'
    assert {tag['big_task'] for tag in detail['category_tags']} == {'Cognitive Mapping', 'Enumerative Perception'}
    assert detail['native_capture_complete'] is False
    assert detail['layout_approval_available'] is True
    assert snapshot['scenes'] == staged['baseline']['scenes']


@pytest.mark.parametrize('changed', ['task', 'scene', 'source', 'result'])
def test_stale_current_bytes_cannot_transfer_native_diagnostics_or_approval(staged, changed):
    before = stage(staged)['diagnostics'][-1]
    assert before['native_evidence_ids'] == ['native_exact']
    if changed in ('task', 'scene'):
        path = Path(staged['variant'][changed]['path'])
    elif changed == 'source':
        path = Path(staged['variant']['sources'][1]['source_file'])
    else:
        path = staged['native_dir'] / 'result.json'
    path.write_text(path.read_text() + '\n')
    after = stage(staged)['diagnostics'][-1]
    assert after['id'] == before['id']
    assert after['fingerprint'] != before['fingerprint']
    assert after['native_evidence_ids'] == []
    assert after['native_initialization_status'] == after['layout_status'] == 'unavailable'
    assert after['rejected_native_evidence'][0]['reason'] == 'current_definition_source_or_result_hash_mismatch'
    assert 'current_definition_source_and_native_result_provenance' in after['pending_gates']
    detail = store(staged).scene_detail(staged['alias'])
    assert detail['layout_approval_available'] is True
    assert len(detail['images']) == 2
    assert all('Exact-bound native' not in image['label'] for image in detail['images'])


def test_publisher_does_not_glob_images_or_sources_and_rejects_alias_collision(staged, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No arbitrary image or source discovery is authorized.')
    monkeypatch.setattr(Path, 'glob', forbidden)
    monkeypatch.setattr(Path, 'rglob', forbidden)
    stage(staged)
    baseline = deepcopy(staged['baseline'])
    baseline['diagnostics'].append(dict(baseline['diagnostics'][0], id=staged['alias']))
    write_json(staged['baseline_path'], baseline)
    with pytest.raises(ValueError, match='cannot replace original rows'):
        stage(staged)


def test_input_catalogue_cannot_be_overwritten(staged):
    baseline = staged['tmp_path'] / 'original_catalogue' / 'snapshot.json'
    write_json(baseline, staged['baseline'])
    with pytest.raises(ValueError, match='cannot overwrite'):
        publish.publish(staged['manifest_path'], baseline, baseline.parent)


@pytest.mark.parametrize('receipt_state',['valid','missing','changed_source','flag_only'])
def test_checkpoint_publication_recomputes_actual_receipt_and_never_trusts_stored_proof(staged,receipt_state):
    manifest=deepcopy(staged['manifest'])
    result_path=staged['native_dir']/'result.json'
    result=read_json(result_path,{})
    result.pop('source_sha256')  # Honest pre-controller checkpoint has no condition_snapshot source hash.
    result.update(status='stopped_at_requested_initialization_checkpoint',complete=False,
        diagnostic_checkpoint_complete=True,checkpoint_stage='constructed',benchmark_execution_verified=False,
        model_calls=0,autonomous_agent_performance=False,commands=[])
    source=staged['variant']['sources'][0]
    captured_source=staged['tmp_path']/'frozen_revision/ESI/dataset/json_clean'/source['source_identity']
    captured_source.parent.mkdir(parents=True)
    captured_source.write_bytes(Path(source['source_file']).read_bytes())
    measured=lambda value:{'available':True,'value':value,'source':'loaded definition'}
    provenance={kind:{'file':measured(result[kind+'_file']),
        'file_sha256':measured(result[kind+'_sha256']), 'source_file':measured(str(captured_source)),
        'source_sha256':measured(source['source_sha256']), 'seed':measured(7)} for kind in ('task','scene')}
    provenance['reset_id']=measured('captured_native_reset')
    receipt_path=staged['native_dir']/'initialization_constructed.json'
    write_json(receipt_path,{'stage':'constructed','source_constructor':'exact.source.constructor',
        'camera_mounts':{'schema':'initial_camera_mount_diagnostic_v1','stage':'constructed',
                         'provenance':provenance}})
    write_json(result_path,result)
    evidence=review.native_summary(result_path,result)
    assert evidence['initialization_checkpoint_evidence']['accepted'] is True
    assert review.exact_native_match(staged['variant'],result)[0] is False
    # Retain this formerly valid stored summary while its real receipt changes.
    manifest['native_evidence']=[evidence]
    if receipt_state=='missing':receipt_path.unlink()
    elif receipt_state=='changed_source':captured_source.write_text('Stale actual captured source bytes.')
    elif receipt_state=='flag_only':
        result['status']='ordinary_diagnostic'
        write_json(result_path,result)
        evidence['result_sha256']=publish.sha256(result_path)
    write_json(staged['manifest_path'],manifest)
    stage(staged)
    detail=store(staged).scene_detail(staged['alias'])
    if receipt_state=='valid':
        assert detail['native_evidence_ids']==['native_exact']
        assert detail['native_diagnostics'][0]['diagnostic_complete'] is False
        assert detail['native_diagnostics'][0]['execution_status']=='initialization_checkpoint_only'
        assert detail['native_initialization_status']==detail['layout_status']=='unknown'
    else:
        assert detail['native_evidence_ids']==[]
        assert detail['rejected_native_evidence'][0]['reason']=='native_result_not_exactly_bound'
    assert detail['native_capture_complete'] is False
    assert detail['layout_approval_available'] is True
    assert detail['physical_qualified'] is False and detail['scoring_status']=='unavailable'


def new_queue(staged,state='QUEUED'):
    """Create a sealed queue-shaped fixture; no native process or worker runs."""
    from PIL import Image
    from wia.runtime.scene_capture_queue import PROTOCOL
    root=staged['tmp_path']/'new_capture_queue';variant=staged['variant'];alias=staged['alias']
    job_id='capture_'+'a'*32;job=root/'jobs'/job_id;snapshot=job/'input'
    snapshot.mkdir(parents=True)
    files={'task.yaml':Path(variant['task']['path']),'scene.yaml':Path(variant['scene']['path']),
           'data/'+variant['sources'][0]['source_identity']:Path(variant['sources'][0]['source_file'])}
    for relative,path in files.items():
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    signature='b'*64
    record={'candidate_id':alias,'actual_candidate_id':variant['candidate_id'],
        'task_edit2_variant_key':variant['variant_key'],'status':'generated','source_identity':variant['sources'][0]['source_identity'],
        'original_source':variant['sources'][0]['source_file'],'source_sha256':variant['sources'][0]['source_sha256'],
        'sources':deepcopy(variant['sources']),'job_id':job_id,'capture_input':str(snapshot/'manifest.json'),
        'capture_signature':signature,'fingerprint':'exact_frozen_capture_fingerprint','seed':7,'variant':'adapted',
        'code_commit':'frozen_fixture_code','task_edit2_capture_campaign':True,
        **{kind+'_file':variant[kind]['path'] for kind in ('task','scene')},
        **{kind+'_sha256':variant[kind]['sha256'] for kind in ('task','scene')}}
    frozen={'candidate_id':alias,'job_id':job_id,'source_identity':record['source_identity'],
        'task_edit2_capture_campaign':True,
        'source_json':record['original_source'],'task_file':record['task_file'],'scene_file':record['scene_file'],
        'capture_signature':signature,'fingerprint':record['fingerprint'],'seed':7,'variant':'adapted',
        'code_commit':record['code_commit'],'esi_commit':'frozen_esi','capture_protocol':PROTOCOL,
        'snapshot_dir':str(snapshot),'snapshot_scene_file':str(snapshot/'scene.yaml'),
        'snapshot_source_file':str(snapshot/('data/'+record['source_identity'])),
        'input_hashes':{relative:publish.sha256(snapshot/relative) for relative in files},
        'expected_views':['global_diagonal','head_rgb']}
    manifest_path=snapshot/'manifest.json';write_json(manifest_path,frozen)
    write_json(job/'accepted.json',{'manifest_sha256':publish.sha256(manifest_path)})
    images={}
    for view in frozen['expected_views']:
        image=job/'attempts/01'/(view+'.png');image.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(64,64),(30,60,90)).save(image);images[view]=str(image)
    receipt={'id':alias,'capture_job_id':job_id,'complete':True,'images':images,
        **{key:frozen[key] for key in ('capture_signature','fingerprint','capture_protocol','code_commit','esi_commit','seed','variant')},
        'physical_grasp_verified':False,'model_calls':0,'robot_actions':0,
        'interaction_setup':{'accepted':False,'reason':'Retained measured placement rejection.'}}
    receipt_path=job/'attempts/01/preview_complete.json';write_json(receipt_path,receipt)
    status={'state':state,'job_id':job_id,'attempts':1}
    if state in ('REVIEW_PENDING','CACHED'):status['receipt']=str(receipt_path)
    if state=='FAILED':status.update(failure_stage='initialization',error='Actual retained capture failure.',
        partial_images={'head_rgb':images['head_rgb']})
    write_json(job/'status.json',status)
    write_json(root/'category_records'/(alias+'.json'),record)
    return {'root':root,'job':job,'record':record,'frozen':frozen,'manifest_path':manifest_path,
        'receipt_path':receipt_path,'receipt':receipt,'status':status}


def publish_new(staged,queue=None):
    return publish.publish(staged['manifest_path'],staged['baseline_path'],staged['output'],
        new_only=True,capture_queue_root=queue)


def test_new_only_hides_baseline_display_and_old_images_preserving_disk_and_stable_alias(staged):
    baseline_bytes=staged['baseline_path'].read_bytes()
    old_snapshot=stage(staged)
    assert old_snapshot['scenes']
    snapshot=publish_new(staged)
    assert snapshot['scenes']==[]
    assert [row['id'] for row in snapshot['diagnostics']]==[staged['alias']]
    assert all(row['big_task']!='Original' for row in snapshot['categories'])
    card=snapshot['diagnostics'][0]
    assert card['task_edit2_capture_campaign'] is True
    assert card['native_diagnostics'] and card['native_evidence_ids']==['native_exact']
    record=read_json(staged['output']/'cards'/staged['alias']/'record.json',{})
    assert record['source_context_images']['historical_review_images']==[]
    assert record['source_context_images']['explicit_source_review_images']==[]
    assert all(image['evidence_role']=='original_esi_images' for image in record['diagnostic_images'])
    assert staged['baseline_path'].read_bytes()==baseline_bytes
    assert (staged['tmp_path']/'historical.png').is_file()
    assert snapshot['capture_queue']['root'] is None
    assert snapshot['capture_queue']['candidate_count']==1
    assert snapshot['task_edit2_overlay']['baseline_rows_hidden_from_display'] is True


@pytest.mark.parametrize('state',['QUEUED','RUNNING','FAILED'])
def test_new_queue_pending_running_failed_are_actual_and_read_only(staged,state):
    queue=new_queue(staged,state)
    before={str(path):path.read_bytes() for path in queue['root'].rglob('*') if path.is_file()}
    snapshot=publish_new(staged,queue['root']);card=snapshot['diagnostics'][0]
    assert card['capture']['state']==state
    assert card['capture']['job_id']==queue['record']['job_id']
    assert card['native_capture_complete'] is card['layout_approval_available'] is False
    assert card['images']==[]
    assert card['capture_queue_evidence']['binding_available'] is True
    assert card['capture_queue_evidence']['status']==queue['status']
    assert snapshot['capture_queue']['root']==str(queue['root'].resolve())
    assert snapshot['capture_queue']['counts']=={state:1}
    if state=='FAILED':
        assert card['failure_stage']=='initialization'
        assert card['failure']=='Actual retained capture failure.'
        assert card['capture_queue_evidence']['status']['partial_images']
    after={str(path):path.read_bytes() for path in queue['root'].rglob('*') if path.is_file()}
    assert before==after


def test_missing_new_queue_is_pending_without_creating_queue_directories(staged):
    queue=staged['tmp_path']/'not_yet_created_queue'
    snapshot=publish_new(staged,queue)
    assert not queue.exists()
    card=snapshot['diagnostics'][0]
    assert card['capture']['state']=='NOT_SUBMITTED'
    assert card['images']==[] and not card['native_capture_complete']
    assert snapshot['capture_queue']['counts']=={'NOT_SUBMITTED':1}


@pytest.mark.parametrize('state',['REVIEW_PENDING','CACHED'])
def test_exact_new_queue_capture_enables_only_human_layout_review(staged,state):
    queue=new_queue(staged,state);snapshot=publish_new(staged,queue['root'])
    assert snapshot['diagnostics']==[]
    assert [row['id'] for row in snapshot['scenes']]==[staged['alias']]
    card=snapshot['scenes'][0]
    assert card['capture']['state']==state
    assert card['native_capture_complete'] is card['layout_approval_available'] is True
    assert card['has_global_preview'] is True and card['no_native_image'] is False
    assert {row['view'] for row in card['images']}==set(queue['frozen']['expected_views'])
    assert all(row['queue_capture_certificate'] is True for row in card['images'])
    assert card['physical_qualified'] is card['autonomous_agent_performance'] is card['selected_final_variant'] is False
    assert card['scoring_status']=='unavailable'
    assert card['progress']['completion_score'] is None
    assert card['queue_capture_receipt']['interaction_setup']['accepted'] is False
    assert snapshot['capture_queue']['captured']==1


@pytest.mark.parametrize('failure',['alias','task_file','task_sha','source_sha','source_members',
    'manifest_seal','frozen_task','frozen_source','signature','receipt_id','receipt_job','incomplete','bad_png',
    'campaign_record','campaign_manifest'])
def test_stale_or_foreign_capture_never_promotes_images_or_approval(staged,failure):
    queue=new_queue(staged,'REVIEW_PENDING');record=queue['record'];receipt=queue['receipt']
    if failure=='alias':record['candidate_id']='category_bad'
    elif failure=='task_file':record['task_file']=str(staged['tmp_path']/'another_task.yaml')
    elif failure=='task_sha':record['task_sha256']='0'*64
    elif failure=='source_sha':record['source_sha256']='0'*64
    elif failure=='source_members':record['sources'].pop()
    elif failure=='manifest_seal':queue['manifest_path'].write_text(queue['manifest_path'].read_text()+'\n')
    elif failure=='frozen_task':(queue['job']/'input/task.yaml').write_text('Stale task.')
    elif failure=='frozen_source':Path(queue['frozen']['snapshot_source_file']).write_text('Stale source.')
    elif failure=='signature':receipt['capture_signature']='foreign_signature'
    elif failure=='receipt_id':receipt['id']='category_bad'
    elif failure=='receipt_job':receipt['capture_job_id']='capture_'+'f'*32
    elif failure=='incomplete':receipt['complete']=False
    elif failure=='campaign_record':record['task_edit2_capture_campaign']=False
    elif failure=='campaign_manifest':
        queue['frozen']['task_edit2_capture_campaign']=False
        write_json(queue['manifest_path'],queue['frozen'])
        write_json(queue['job']/'accepted.json',{'manifest_sha256':publish.sha256(queue['manifest_path'])})
    else:Path(next(iter(receipt['images'].values()))).write_text('Not an actual PNG.')
    write_json(queue['root']/'category_records'/(staged['alias']+'.json'),record)
    write_json(queue['receipt_path'],receipt)
    snapshot=publish_new(staged,queue['root']);card=snapshot['diagnostics'][0]
    assert snapshot['scenes']==[]
    assert card['capture']['state'] in ('UNBOUND','STALE')
    assert card['capture_queue_evidence']['reason']
    assert card['images']==[]
    assert card['native_capture_complete'] is card['layout_approval_available'] is False
    assert snapshot['capture_queue']['captured']==0


def test_verified_new_capture_reaches_existing_live_detail_without_approving_task(staged):
    queue=new_queue(staged,'REVIEW_PENDING');publish_new(staged,queue['root'])
    app=LiveSceneReview(None,staged['tmp_path']/'new_reviews',queue_root=queue['root'],
        catalogue_file=staged['output']/'snapshot.json')
    listing=app.catalogue(filters={'scope':'all'})
    assert [row['id'] for row in listing['scenes']]==[staged['alias']]
    detail=app.scene_detail(staged['alias'])
    assert detail['native_capture_complete'] is detail['layout_approval_available'] is True
    assert len(detail['images'])==len(queue['frozen']['expected_views'])
    assert detail['physical_qualified'] is detail['autonomous_agent_performance'] is False
    assert detail['selected_final_variant'] is False
