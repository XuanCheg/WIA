"""CPU checks for original-Core browsing groups with linked revision cards.

Run: python -B -m pytest -q tests/test_task_edit2_core_groups.py
Tests publish temporary snapshots only; actual campaign manifests are read only.
No queue, running service, GPU, task definition or final selection is changed.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import yaml

from wia.runtime.scene_capture_queue import read_json, write_json
from wia.runtime.scene_review_live import LiveSceneReview
import wia.runtime.commands.publish_task_edit2_review as publish
from harness_integration.test_task_edit2_review_publish import staged


@pytest.fixture
def grouped(staged):
    manifest=deepcopy(staged['manifest'])
    shared=manifest['variants'][0]
    sources=shared['sources']
    old_input=str(staged['tmp_path']/'earlier/manifest.json')
    latest_input=str(staged['tmp_path']/'latest/manifest.json')
    manifest['explicit_current_inputs']=[{'path':old_input},{'path':latest_input}]
    shared.update(original_core_members=[row['source_identity'] for row in sources],
        additional_source_members=[],membership_class='core_derivative',
        manifest_records=[{'manifest_file':latest_input}])
    older=deepcopy(shared)
    older.update(variant_key='1111',candidate_id='older_revision',native_evidence_ids=[],
        original_core_members=[sources[0]['source_identity']],manifest_records=[{'manifest_file':old_input}])
    extra=deepcopy(older)
    extra_identity='Object Semantics/Existence/test/room/extra.json'
    extra_path=staged['tmp_path']/'sources'/extra_identity
    write_json(extra_path,{'scene':'test','room':'room','question':'Inspect the additional source.'})
    extra.update(variant_key='eeee',candidate_id='additional_prototype',original_core_members=[],
        additional_source_members=[extra_identity],membership_class='additional_prototype',
        sources=[{'source_identity':extra_identity,'source_file':str(extra_path),
                  'source_sha256':publish.sha256(extra_path)}])
    raw=yaml.safe_load(Path(shared['task']['path']).read_text())
    raw['source']['json']=extra_identity
    extra_task=staged['tmp_path']/'additional_task.yaml';extra_task.write_text(yaml.safe_dump(raw))
    extra['task']={'path':str(extra_task),'sha256':publish.sha256(extra_task),'source':{'json':extra_identity}}
    manifest['variants']=[shared,older,extra,{'role':'original_baseline','variant_key':'aaaa'}]
    third_identity='Test/Family/test/room/unrevised.json'
    core=[]
    for source,key,original_id in zip([row['source_identity'] for row in sources]+[third_identity],
                                      ['aaaa','bbbb','cccc'],['category_aa','category_bb','category_cc']):
        core.append({'source_identity':source,'original_variant_key':key,'original_candidate_id':original_id,
            'original_task_file':'/historical/'+key+'/task.yaml','original_scene_file':'/historical/'+key+'/scene.yaml',
            'historical_review_images':{'paths':[]},'original_esi_images':[]})
        source_file=staged['tmp_path']/'sources'/source
        if not source_file.is_file():write_json(source_file,{'question':'Retained original public question.'})
        core[-1]['source_file']=str(source_file)
    baseline=deepcopy(staged['baseline'])
    original=baseline['scenes'][0]
    original['source']=sources[0]['source_identity']
    original['images']=[{'path':str(staged['tmp_path']/'historical.png'),'kind':'native','label':'Old capture'},
                        {'path':str(staged['tmp_path']/'deleted_old.png'),'kind':'native'}]
    original['source_context_images']={'explicit_context':[str(staged['tmp_path']/'original_esi.png')]}
    baseline['diagnostics'][0]['source']=sources[1]['source_identity']
    baseline['diagnostics'].append({**deepcopy(baseline['diagnostics'][0]),'id':'category_cc','candidate_id':'category_cc',
        'source':third_identity,'conversion_result_status':'conversion_failed',
        'failure_stage':'definition_missing','failure':'Retained CPU definition diagnostic.'})
    manifest['core_sources']=core
    write_json(staged['manifest_path'],manifest);write_json(staged['baseline_path'],baseline)
    return {**staged,'manifest':manifest,'baseline':baseline,'old_input':old_input,'latest_input':latest_input}


def test_grouped_snapshot_keeps_canonical_cards_shared_links_and_additional_separation(grouped):
    before=grouped['baseline_path'].read_bytes()
    snapshot=publish.publish(grouped['manifest_path'],grouped['baseline_path'],grouped['output'],group_core=True)
    groups=snapshot['core_groups']
    assert [row['id'] for row in groups]==['core_aaaa','core_bbbb','core_cccc']
    assert len(snapshot['scenes'])+len(snapshot['diagnostics'])==6
    ids=[row['id'] for row in snapshot['scenes']+snapshot['diagnostics']]
    assert len(set(ids))==len(ids)
    assert set(ids)=={'category_aa','category_bb','category_cc',grouped['alias'],'category_1111','category_eeee'}
    assert [row['id'] for row in groups[0]['versions']]==['category_aa','category_1111',grouped['alias']]
    assert [row['id'] for row in groups[1]['versions']]==['category_bb',grouped['alias']]
    assert groups[0]['default_variant_id']==groups[1]['default_variant_id']==grouped['alias']
    assert groups[2]['default_variant_id']=='category_cc'
    assert groups[2]['original_variant_id']==groups[2]['original_candidate_id']=='category_cc'
    assert snapshot['additional_variant_ids']==['category_eeee']
    assert not any(row['id']=='category_eeee' for group in groups for row in group['versions'])
    assert all(group['browsing_default_only'] and not group['selected_final_variant'] for group in groups)
    assert snapshot['final_core_selection_made'] is snapshot['task_edit2_overlay']['final_core_selection_made'] is False
    assert grouped['baseline_path'].read_bytes()==before
    assert read_json(grouped['output']/'cards'/grouped['alias']/'record.json')['actual_candidate_id']=='actual_joint_parent'


def test_historical_baselines_never_inherit_current_capture_or_approval_and_missing_images_stay_absent(grouped):
    snapshot=publish.publish(grouped['manifest_path'],grouped['baseline_path'],grouped['output'],group_core=True)
    originals=[row for row in snapshot['scenes']+snapshot['diagnostics'] if row.get('original_baseline')]
    assert len(originals)==3
    for row in originals:
        assert row['native_capture_complete'] is row['layout_approval_available'] is row['physical_qualified'] is False
        assert row['preview_is_current'] is False and row['baseline_evidence_role']=='historical_context_only'
        assert all(image['kind']=='historical' and not image['queue_capture_certificate'] for image in row['images'])
        assert all(Path(image['path']).is_file() for image in row['diagnostic_images'])
    original=next(row for row in originals if row['id']=='category_aa')
    assert original['prior_feedback']==grouped['baseline']['scenes'][0]['prior_feedback']
    assert original['capture']==grouped['baseline']['scenes'][0]['capture']
    assert {Path(image['path']).name for image in original['images']}=={'historical.png','original_esi.png'}
    assert (grouped['tmp_path']/'historical.png').is_file() and not (grouped['tmp_path']/'deleted_old.png').exists()
    assert all(group['versions'][0]['capture_state']=='HISTORICAL' for group in snapshot['core_groups'])


def test_original_details_use_staged_records_even_with_missing_cpu_definitions(grouped):
    snapshot=publish.publish(grouped['manifest_path'],grouped['baseline_path'],grouped['output'],group_core=True)
    app=LiveSceneReview(None,grouped['tmp_path']/'grouped_reviews',queue_root=grouped['tmp_path']/'empty_queue',
                        catalogue_file=grouped['output']/'snapshot.json')
    for source in grouped['manifest']['core_sources']:
        original=source['original_candidate_id']
        record=read_json(grouped['output']/'cards'/original/'record.json')
        assert record['source_identity']==source['source_identity']
        assert record['original_source']==source['source_file']
        assert record['task_file']==source['original_task_file']
        assert record['scene_file']==source['original_scene_file']
        assert not Path(record['task_file']).is_file()  # Metadata fallback still opens.
        detail=app.scene_detail('core_'+source['original_variant_key'],variant_id=original)
        assert detail['original_baseline'] and detail['native_capture_complete'] is False
        assert detail['layout_approval_available'] is True
        assert detail['core_group']['active_variant_id']==original
        if original=='category_cc':
            assert detail['failure_stage']=='definition_missing'
            assert detail['failure']=='Retained CPU definition diagnostic.'
    assert len(snapshot['core_groups'])==3


def test_latest_failed_revision_wins_declared_order_without_hiding_old_success(grouped):
    manifest=grouped['manifest']
    baseline=[row for section in ('scenes','diagnostics') for row in grouped['baseline'][section]]
    cards=[{'id':'category_'+variant['variant_key'],'capture':{'state':'FAILED' if variant['variant_key']=='abcd1234'
        else 'REVIEW_PENDING'},'native_capture_complete':variant['variant_key']!='abcd1234'}
        for variant in manifest['variants'] if variant['role']=='explicit_derivative']
    groups,_=publish.core_groups(manifest,baseline,cards)
    assert groups[0]['default_variant_id']==grouped['alias']
    assert groups[0]['versions'][-1]['capture_state']=='FAILED'
    assert groups[0]['versions'][1]['capture_state']=='REVIEW_PENDING'
    # Reorder the declared input list, not card/manifest enumeration: default changes.
    changed=deepcopy(manifest);changed['explicit_current_inputs'].reverse()
    groups,_=publish.core_groups(changed,baseline,cards)
    assert groups[0]['default_variant_id']=='category_1111'


def test_same_declared_input_uses_stable_variant_key_tie_break(grouped):
    manifest=deepcopy(grouped['manifest'])
    for variant in manifest['variants']:
        if variant['role']=='explicit_derivative':variant['manifest_records']=[{'manifest_file':grouped['latest_input']}]
    baseline=[row for section in ('scenes','diagnostics') for row in grouped['baseline'][section]]
    cards=[{'id':'category_'+v['variant_key']} for v in manifest['variants'] if v['role']=='explicit_derivative']
    first=publish.core_groups(manifest,baseline,cards)
    manifest['variants'].reverse()
    again=publish.core_groups(manifest,baseline,cards)
    assert first==again
    assert first[0][0]['default_variant_id']==grouped['alias']


def test_group_mode_is_explicit_exclusive_and_legacy_modes_remain_unchanged(grouped,monkeypatch):
    with pytest.raises(ValueError,match='mutually exclusive'):
        publish.publish(grouped['manifest_path'],grouped['baseline_path'],grouped['output'],group_core=True,new_only=True)
    monkeypatch.setattr(sys,'argv',['publish','--manifest','unused','--baseline-catalogue','unused',
        '--output-dir','unused','--group-core','--new-only'])
    with pytest.raises(SystemExit) as error:publish.main()
    assert error.value.code==2
    legacy=publish.publish(grouped['manifest_path'],grouped['baseline_path'],grouped['output'])
    assert 'core_groups' not in legacy and legacy['scenes']==grouped['baseline']['scenes']
    assert not any(row.get('original_baseline') for row in legacy['scenes'])
    new=publish.publish(grouped['manifest_path'],grouped['baseline_path'],grouped['output'],new_only=True)
    assert 'core_groups' not in new and new['scenes']==[]
    assert {row['id'] for row in new['diagnostics']}=={grouped['alias'],'category_1111','category_eeee'}


def test_actual_161_core_sources_122_unique_cards_and_ten_additional_only_are_preserved():
    manifest=json.loads((publish.ROOT/'outputs/task_edit2_20261018/review/current_manifest.json').read_text())
    sources=manifest['core_sources']
    variants=[row for row in manifest['variants'] if row['role']=='explicit_derivative']
    baseline=[{'id':row['original_candidate_id']} for row in sources]
    cards=[{'id':'category_'+row['variant_key']} for row in variants]
    groups,additional=publish.core_groups(manifest,baseline,cards)
    assert len(groups)==161 and len(cards)==122 and len(additional)==10
    assert [group['source_identity'] for group in groups]==[source['source_identity'] for source in sources]
    linked=Counter(version['id'] for group in groups for version in group['versions'][1:])
    for variant in variants:
        alias='category_'+variant['variant_key']
        assert linked[alias]==len(set(variant.get('original_core_members',[])))
        assert (alias in additional)==(not variant.get('original_core_members'))
    assert any(count>1 for count in linked.values())  # Actual shared parents remain one underlying card.
    assert len({row['id'] for row in cards})==122
