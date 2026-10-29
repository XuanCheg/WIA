"""CPU-only TaskEdit2 capture adapter validation with mocked publication/freezing.

Run: python -B -m pytest -q tests/test_task_edit2_capture_submission.py
All submission mutations use temporary files and fake bridge calls. No actual
queue publishing, code freezing, worker, native simulator or GPU is launched.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from wia.runtime.scene_capture_queue import digest, read_json, write_json
import wia.runtime.commands.submit_task_edit2_capture as adapter


def variant(tmp_path, key, *, adapted=True, seed=7):
    sources=[]
    for name in ('a','b'):
        identity=f'Test/Family/building/room/{name}.json'
        path=tmp_path/'sources'/identity
        write_json(path,{'scene':'building','room':'room','question':'Inspect the source.'})
        sources.append({'source_identity':identity,'source_file':str(path),
            'source_sha256':adapter.sha256(path),'provenance_status':'verified_current_source_bytes',
            'mappings':[{'public_id':'goal_'+name}]})
    modes={'original':{'patches':[]}}
    if adapted:modes['adapted']={'patches':[]}
    scene_config={'policy':'native_then_minimal_patch_else_skip','variants':modes}
    task={'kind':'category_robot_task','id':'task_'+key,'version':1,
        'source':{'provider':'esi','json':sources[0]['source_identity']},
        'public':{'objective':'Inspect source objects.','success_criteria':'Report observed evidence.',
                  'environment':'A preserved source room.'},
        'requirements':{'capabilities':['base_translation']},
        'private':{'bindings':{},'evaluation':{'type':'category_operation',
            'module':'wia.authoring.converters.enumerative_perception_search','operation':'search',
            'parameters':{'task_revision':'explicit_test_revision_v1'}}},
        'scene':deepcopy(scene_config),'defaults':{'seed':seed},'qualification':{'on_failure':'skip'}}
    scene={'kind':'scene_patch','id':'scene_'+key,'version':1,'status':'pending',
        'source':deepcopy(task['source']),'seed':seed,'scene':scene_config}
    directory=tmp_path/key;directory.mkdir()
    paths={}
    for kind,raw in [('task',task),('scene',scene)]:
        path=directory/(kind+'.yaml');path.write_text(yaml.safe_dump(raw))
        paths[kind]={'path':str(path),'sha256':adapter.sha256(path)}
    return {'role':'explicit_derivative','variant_key':key,'candidate_id':'actual_'+key,**paths,
        'schema_status':'schema_valid','source_binding_status':'verified','sources':sources,
        'membership_class':'mixed_core_and_additional_members',
        'original_core_members':[sources[0]['source_identity']],
        'additional_source_members':[sources[1]['source_identity']],
        'pending_gates':['native_source_relations'],'pending_review_gates':['terminal_scoring_unmeasured'],
        'recorded_warnings':['native_qualification_pending'],
        'manifest_records':[{'manifest_file':str(tmp_path/'input_manifest.json'),'record_status':'generated'}]}


@pytest.fixture
def manifest(tmp_path):
    return {'schema':'provisional_test_manifest','baseline_membership':{'sha256':'historical_sha'},
        'variants':[{'role':'original_baseline','variant_key':'deadbeef'},variant(tmp_path,'abcd'),
                    variant(tmp_path,'ef01',adapted=False,seed=0)]}


def edit_definition(variant,kind,callback):
    path=Path(variant[kind]['path']);raw=yaml.safe_load(path.read_text());callback(raw)
    path.write_text(yaml.safe_dump(raw));variant[kind]['sha256']=adapter.sha256(path)


def test_records_keep_aliases_all_sources_membership_and_exact_seed_variant_without_promotion(manifest):
    records=adapter.capture_records(manifest)
    assert [row['candidate_id'] for row in records]==['category_abcd','category_ef01']
    assert [row['variant'] for row in records]==['adapted','original']
    assert [row['seed'] for row in records]==[7,0]
    for record,source in zip(records,manifest['variants'][1:]):
        assert record['status']=='generated',record['failure_reason']
        assert record['actual_candidate_id']==source['candidate_id']
        assert record['sources']==source['sources'] and len(record['capture_source_members'])==2
        assert record['capture_source_members']==[{key:row[key] for key in
            ('source_identity','source_file','source_sha256')} for row in source['sources']]
        assert record['source_core_metadata']==[
            {'source_identity':source['sources'][0]['source_identity'],'original_core_member':True},
            {'source_identity':source['sources'][1]['source_identity'],'original_core_member':False}]
        assert record['converter_version']=='explicit_test_revision_v1'
        assert record['converter_module']=='wia.authoring.converters.enumerative_perception_search'
        assert record['big_task']=='Test' and record['small_task']=='Family'
        assert record['capture_initial_agent_bundle'] is record['task_edit2_capture_campaign'] is True
        assert record['selected_final_variant'] is record['native_capture_complete'] is record['layout_approval_available'] is False
    assert records==adapter.capture_records(manifest)


@pytest.mark.parametrize('failure',['task_hash','source_hash','seed','schema','source_status','unsafe_source','duplicate_alias'])
def test_invalid_derivatives_remain_error_records_without_silent_drops(manifest,failure):
    target=manifest['variants'][1]
    if failure=='task_hash':Path(target['task']['path']).write_text('Changed bytes.')
    elif failure=='source_hash':Path(target['sources'][1]['source_file']).write_text('Changed source.')
    elif failure=='seed':edit_definition(target,'scene',lambda raw:raw.update(seed=123))
    elif failure=='schema':edit_definition(target,'task',lambda raw:raw.update(unrecognized_field=True))
    elif failure=='source_status':target['source_binding_status']='unavailable'
    elif failure=='unsafe_source':target['sources'][1]['source_identity']='../../escape.json'
    else:manifest['variants'][2]['variant_key']=target['variant_key']
    records=adapter.capture_records(manifest)
    assert len(records)==2 and records[0]['status']=='validation_failed'
    assert records[0]['failure_reason'] and records[0]['failure_stage']=='capture_record_validation'
    assert records[1]['status']==('validation_failed' if failure in ('duplicate_alias','source_hash') else 'generated')


def test_revision_tracks_actual_current_bindings_and_source_bytes(manifest):
    initial=adapter.capture_records(manifest)[0]
    target=manifest['variants'][1]
    edit_definition(target,'task',lambda raw:raw['public'].update(objective='Inspect revised instructions.'))
    changed=adapter.capture_records(manifest)[0]
    assert changed['candidate_id']==initial['candidate_id'] and changed['revision']!=initial['revision']


def test_dry_run_never_freezes_publishes_or_creates_queue(manifest,tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Dry run cannot freeze or publish.')
    monkeypatch.setattr(adapter.bridge,'freeze_working_code',forbidden)
    monkeypatch.setattr(adapter.bridge,'publish_record',forbidden)
    queue=tmp_path/'does_not_exist'
    report=adapter.submit(manifest,queue,dry_run=True)
    assert not queue.exists() and report['frozen_code'] is None
    assert report['counts']=={'explicit_derivatives':2,'validated':2,'submitted':0,'existing':0,'error':0}
    assert all(row['status']=='validated' and row['job_id'] is None for row in report['records'])


def fake_bridge(monkeypatch):
    freezes,published=[],[]
    frozen={'code_root':'/fake/frozen_code','code_commit':'working_test','esi_commit':'esi_test',
            'implementation_hashes':{},'code_archive':'/fake/archive.tar.gz','code_archive_sha256':'archive_sha'}
    def freeze(queue):freezes.append(str(queue));return frozen
    def publish(record,queue,**kwargs):
        published.append((deepcopy(record),kwargs))
        value={**record,'record_revision':record['revision'],'code_commit':kwargs['frozen']['code_commit'],
            'job_id':'capture_'+digest(record['candidate_id'])[:32],
            'capture_input':str(Path(queue)/'jobs'/record['candidate_id']/'input/manifest.json')}
        write_json(Path(queue)/'category_records'/(record['candidate_id']+'.json'),value)
        return value
    monkeypatch.setattr(adapter.bridge,'freeze_working_code',freeze)
    monkeypatch.setattr(adapter.bridge,'publish_record',publish)
    return freezes,published,frozen


def test_mocked_submit_uses_one_common_snapshot_and_idempotent_existing_revisions(manifest,tmp_path,monkeypatch):
    freezes,published,frozen=fake_bridge(monkeypatch);queue=tmp_path/'queue'
    first=adapter.submit(manifest,queue,timeout_s=1800,max_attempts=1)
    assert len(freezes)==1 and len(published)==2
    assert all(kwargs['frozen'] is frozen and kwargs['timeout_s']==1800 and kwargs['max_attempts']==1
               for _,kwargs in published)
    assert first['counts']['submitted']==2 and first['counts']['error']==0
    assert read_json(queue/'submission_report.json')==first
    existing_bytes={path:path.read_bytes() for path in (queue/'category_records').iterdir()}
    again=adapter.submit(manifest,queue)
    assert again['counts']['existing']==2 and len(freezes)==1 and len(published)==2
    assert all(path.read_bytes()==data for path,data in existing_bytes.items())
    target=manifest['variants'][1]
    edit_definition(target,'task',lambda raw:raw['public'].update(objective='Changed after submission.'))
    rejected=adapter.submit(manifest,queue)
    assert rejected['counts']['error']==1 and rejected['counts']['existing']==1
    assert 'cannot be overwritten' in rejected['records'][0]['error']
    assert all(path.read_bytes()==data for path,data in existing_bytes.items())


def test_mocked_bridge_failure_is_reported_per_alias_without_blocking_remaining(manifest,tmp_path,monkeypatch):
    _,published,_=fake_bridge(monkeypatch)
    original=adapter.bridge.publish_record
    def failure(record,queue,**kwargs):
        if record['candidate_id']=='category_abcd':return {**record,'capture_submission_error':'Actual fixture submission failure.'}
        return original(record,queue,**kwargs)
    monkeypatch.setattr(adapter.bridge,'publish_record',failure)
    report=adapter.submit(manifest,tmp_path/'queue')
    assert report['counts']['submitted']==report['counts']['error']==1
    assert report['records'][0]['error']=='Actual fixture submission failure.'
    assert report['records'][1]['job_id'] and len(published)==1


def test_explicit_frozen_record_is_verified_and_reused_without_freezing(manifest,tmp_path,monkeypatch):
    freezes,published,_=fake_bridge(monkeypatch)
    root=tmp_path/'frozen';root.mkdir();implementation=root/'impl.py';implementation.write_text('original bytes')
    archive=tmp_path/'frozen.tar.gz';archive.write_bytes(b'Archive fixture, not an actual code freeze.')
    write_json(root/'.capture_revision.json',{'commit':'working_exact','esi_commit':'esi_exact'})
    record={'code_root':str(root),'code_commit':'working_exact','esi_commit':'esi_exact',
        'implementation_hashes':{'impl.py':adapter.sha256(implementation)},'code_archive':str(archive),
        'code_archive_sha256':adapter.sha256(archive)}
    record_path=tmp_path/'frozen_record.json';write_json(record_path,record)
    report=adapter.submit(manifest,tmp_path/'queue',frozen_code_record=record_path)
    assert not freezes and len(published)==2 and report['counts']['submitted']==2
    assert all(kwargs['frozen']==record for _,kwargs in published)
    implementation.write_text('Changed extracted implementation.')
    failed=adapter.submit(manifest,tmp_path/'other_queue',frozen_code_record=record_path)
    assert failed['counts']['error']==2 and len(published)==2


def test_actual_current_122_derivatives_validate_without_queue_or_snapshot_mutation(tmp_path,monkeypatch):
    path=adapter.ROOT/'outputs/task_edit2_20261018/review/current_manifest.json'
    manifest=json.loads(path.read_text())
    def forbidden(*args,**kwargs):raise AssertionError('Read-only actual audit must not publish or freeze.')
    monkeypatch.setattr(adapter.bridge,'freeze_working_code',forbidden)
    monkeypatch.setattr(adapter.bridge,'publish_record',forbidden)
    queue=tmp_path/'uncreated_queue'
    report=adapter.submit(manifest,queue,dry_run=True,manifest_path=path)
    assert report['counts']['explicit_derivatives']==report['counts']['validated']==122
    assert report['counts']['error']==0,[row for row in report['records'] if row['error']]
    expected={'category_'+row['variant_key'] for row in manifest['variants'] if row['role']=='explicit_derivative'}
    assert {row['candidate_id'] for row in report['records']}==expected
    assert all(row['candidate_id']!='category_'+original['variant_key'] for original in manifest['variants']
               if original['role']=='original_baseline' for row in report['records'])
    assert not queue.exists()


def sealed_esi_snapshot(tmp_path):
    root=tmp_path/'working_exact';root.mkdir()
    commit='9f6f15ed43e81654a97de9e4db2a1fedadca2bc1'
    esi=tmp_path/('esi_'+commit);esi.mkdir()
    write_json(root/'.capture_revision.json',{'commit':'working_exact','esi_commit':commit})
    write_json(esi/'.capture_revision.json',{'commit':commit})
    (root/'ESI').symlink_to(esi,target_is_directory=True)
    implementation=esi/'src/task.py';implementation.parent.mkdir();implementation.write_text('sealed ESI implementation')
    archive=tmp_path/'working.tar.gz';archive.write_bytes(b'Frozen archive fixture')
    record={'code_root':str(root),'code_commit':'working_exact','esi_commit':commit,
        'implementation_hashes':{'ESI/src/task.py':adapter.sha256(implementation)},
        'code_archive':str(archive),'code_archive_sha256':adapter.sha256(archive)}
    filename=tmp_path/'frozen_code.json';write_json(filename,record)
    return filename,record,root,esi,implementation


def test_exact_sealed_esi_sibling_link_is_allowed_but_each_file_hash_remains_mandatory(tmp_path):
    filename,record,root,esi,implementation=sealed_esi_snapshot(tmp_path)
    assert adapter._frozen_record(filename)==record
    assert (root/'ESI').resolve()==esi
    implementation.write_text('Changed ESI bytes after freezing')
    with pytest.raises(ValueError,match='manifest SHA256'):
        adapter._frozen_record(filename)


@pytest.mark.parametrize('escape',['absolute','parent','other_link','wrong_esi_target','nested_link','wrong_esi_marker'])
def test_frozen_esi_exception_never_allows_arbitrary_path_or_link_escape(tmp_path,escape):
    filename,record,root,esi,implementation=sealed_esi_snapshot(tmp_path)
    outside=tmp_path/'outside.py';outside.write_bytes(implementation.read_bytes())
    if escape=='absolute':record['implementation_hashes']={str(outside):adapter.sha256(outside)}
    elif escape=='parent':record['implementation_hashes']={'../outside.py':adapter.sha256(outside)}
    elif escape=='other_link':
        (root/'other.py').symlink_to(outside)
        record['implementation_hashes']={'other.py':adapter.sha256(outside)}
    elif escape=='wrong_esi_target':
        (root/'ESI').unlink()
        other=tmp_path/'unsealed_esi';other.mkdir()
        (other/'src').mkdir();(other/'src/task.py').write_bytes(implementation.read_bytes())
        (root/'ESI').symlink_to(other,target_is_directory=True)
    elif escape=='nested_link':
        implementation.unlink();implementation.symlink_to(outside)
    else:write_json(esi/'.capture_revision.json',{'commit':'different_commit'})
    write_json(filename,record)
    with pytest.raises(ValueError,match='escaped its code root'):
        adapter._frozen_record(filename)
