"""Test provisional coverage and evidence isolation.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests/test_task_edit2_review_manifest.py
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

import wia.runtime.commands.build_task_edit2_review_manifest as review


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def definitions(root, name, identity):
    scene_config = {'policy': 'native_then_minimal_patch_else_skip',
                    'variants': {'original': {'patches': []}}}
    source = {'provider': 'esi', 'json': identity}
    scene = {'kind': 'scene_patch', 'id': 'scene_' + name, 'version': 1,
             'source': source, 'scene': scene_config}
    task = {'kind': 'category_robot_task', 'id': 'task_' + name, 'version': 1,
            'source': source, 'public': {'objective': 'Inspect the room.',
                'success_criteria': 'Declare the observed objects.', 'environment': 'A source room.'},
            'requirements': {'capabilities': ['base_translation']},
            'private': {'bindings': {}, 'evaluation': {'type': 'category_operation',
                'module': 'wia.authoring.converters.enumerative_perception_search',
                'operation': 'search', 'parameters': {}}},
            'scene': scene_config, 'qualification': {'on_failure': 'skip'}, 'defaults': {}}
    directory = root / 'definitions' / name
    directory.mkdir(parents=True, exist_ok=True)
    for filename, value in [('task.yaml', task), ('scene.yaml', scene)]:
        (directory / filename).write_text(yaml.safe_dump(value))
    return {'candidate_id': name, 'source_identity': identity,
            'task_file': str(directory / 'task.yaml'), 'scene_file': str(directory / 'scene.yaml')}


@pytest.fixture
def fixture_world(tmp_path):
    source_root = tmp_path / 'ESI/dataset/json_clean'
    identities = [f'Test/Family/map/room/{name}.json' for name in 'abcd']
    for identity in identities:
        write_json(source_root / identity, {'question': 'Which object?', 'answer': identity})
    originals = []
    for index, identity in enumerate(identities[:3]):
        record = definitions(tmp_path, 'original_' + str(index), identity)
        record['source'] = record.pop('source_identity')
        record['preserved_metadata'] = {'review': 'unchanged', 'historical_hash': str(index)}
        originals.append(record)
    # Absence is distinct from invalidity or native failure.
    originals[2].pop('task_file')
    originals[2].pop('scene_file')
    baseline = write_json(tmp_path / 'baseline.json', {'tasks': originals})
    parent = definitions(tmp_path, 'merged_parent', identities[0])
    parent.pop('source_identity')
    parent['source_to_parent'] = [
        {'source_identity': identity, 'source_sha256': review.sha256(source_root / identity),
         'group_id': 'group_' + str(index)}
        for index, identity in enumerate([identities[0], identities[1], identities[3]])]
    parent['source_to_parent'].append(deepcopy(parent['source_to_parent'][0]))
    extra = definitions(tmp_path, 'additional_temporal', identities[3])
    input_root = tmp_path / 'outputs/task_edit2_20261018'
    # Repeated records and source membership must not multiply variants.
    write_json(input_root / review.CURRENT_INPUTS[4], {'candidates': [parent, parent, extra]})
    write_json(input_root / 'perception/manifest.backup.json', {'candidates': [definitions(tmp_path, 'backup', identities[2])]})
    write_json(input_root / 'deformable_candidates_v1/batch_manifest.json', {'records': [definitions(tmp_path, 'obsolete_v1', identities[2])]})
    return {'root': tmp_path, 'source_root': source_root, 'baseline': baseline,
            'input_root': input_root, 'identities': identities, 'parent': parent, 'originals': originals}


def build(world, **kwargs):
    return review.build_manifest(world['baseline'], root=world['root'],
        source_root=world['source_root'], input_root=world['input_root'], **kwargs)


def native_result(world):
    record = world['parent']
    return {'task_file': record['task_file'], 'scene_file': record['scene_file'],
        'task_sha256': review.sha256(record['task_file']),
        'scene_sha256': review.sha256(record['scene_file']),
        'initial_condition': {'source_sha256': review.sha256(world['source_root'] / world['identities'][0])},
        'complete': True, 'public_controls_only': True, 'model_calls': 0,
        'autonomous_agent_performance': False,
        'initial_state': {'interaction_setup': {'accepted': False, 'attempts': [{'reason': 'initial obstruction'}]}},
        'final_state': {'loaded_room_start': {'accepted': True}, 'interaction_setup': {'accepted': True,
            'attempts': [{'accepted': False, 'reason': 'earlier failed stance'}, {'accepted': True}]}},
        'qualification': {'status': 'skipped', 'reason': 'terminal measurement unavailable',
            'checks': {'interaction_initialization': {'accepted': False}}},
        'commands': [{'command': {'action': 'BASE'}, 'result': {'completed': True}}]}


def native_abort(world, name='native_abort'):
    """Write a native-shaped retained construction and externally recorded stop."""
    result=native_result(world)
    directory=world['input_root']/name
    revision='working_frozen_native_revision'
    source_file=world['root']/revision/'ESI/dataset/json_clean'/world['identities'][0]
    source_file.parent.mkdir(parents=True,exist_ok=True)
    source_file.write_bytes((world['source_root']/world['identities'][0]).read_bytes())
    measured=lambda value:{'available':True,'value':value,'source':'loaded definition'}
    provenance={kind:{'file':measured(result[kind+'_file']),
        'file_sha256':measured(result[kind+'_sha256']),
        'source_sha256':measured(review.sha256(source_file)),
        'source_file':measured(str(source_file)), 'seed':measured(0)} for kind in ('task','scene')}
    provenance['reset_id']=measured('actual_native_reset')
    construction=write_json(directory/'initialization_constructed.json',{
        'stage':'constructed','source_constructor':'tasks.test.family',
        'camera_mounts':{'schema':'initial_camera_mount_diagnostic_v1','stage':'constructed','provenance':provenance}})
    log=directory/'native.log';log.write_text('Retained native fatal error diagnostic.\n')
    stack=directory/'native_stack.log';stack.write_text('Retained native runtime stack.\n')
    abort={'schema':'native_runtime_abort_v1','evidence_origin':review.ABORT_ORIGIN,
        **{key:result[key] for key in ('task_file','scene_file','task_sha256','scene_sha256')},
        'source_file':str(source_file),'source_sha256':review.sha256(source_file),
        'frozen_revision':revision,'complete':False,'diagnostic_only':True,
        'model_calls':0,'autonomous_agent_performance':False,'benchmark_execution_verified':False,
        'capture_or_task_success_claimed':False,'reason':'Native fatal error while settling.',
        'signal':'SIGTERM','initialization_stage':'settling_before_source_postprocess',
        'native_error_log':str(log),'native_stack_file':str(stack),'native_construction_receipt':str(construction)}
    path=write_json(directory/'runtime_abort.json',abort)
    return path,abort


def test_valid_native_abort_preserves_external_origin_without_success_or_completion(fixture_world):
    world=fixture_world;path,abort=native_abort(world)
    manifest=build(world)
    assert manifest['counts']['matched_native_runtime_aborts']==1
    assert manifest['counts']['matched_native_results']==0
    evidence=manifest['native_evidence'][0]
    assert evidence['evidence_id']=='native_abort_runtime_abort'
    assert evidence['evidence_origin']==review.ABORT_ORIGIN
    assert evidence['evidence_record_type']=='externally_recorded_runtime_abort'
    assert evidence['diagnostic_status']==evidence['native_runtime_status']=='observed_native_runtime_error'
    assert evidence['diagnostic_complete'] is evidence['checker_finalized_diagnostic'] is False
    assert evidence['execution_status']==evidence['scoring_status']=='unavailable'
    assert evidence['native_initialization_status']==evidence['layout_status']=='unknown'
    assert evidence['qualification'] is None and evidence['interaction_setup_evidence']==[]
    for field in ('agent_task_success','grasp_success','benchmark_task_success','task_success','score_percent','completion_score'):
        assert evidence[field] is None
    assert evidence['autonomous_agent_performance'] is False
    assert evidence['captures']['initial_global'] is None
    assert evidence['private_abort_file']==str(path.resolve())
    assert evidence['abort_sha256']==review.sha256(path)
    assert evidence['raw_diagnostic_fields']==abort
    for artifact in evidence['abort_binding_proof']['artifacts'].values():
        assert artifact['sha256']==review.sha256(artifact['path'])
    variant=next(row for row in manifest['variants'] if row['candidate_id']=='merged_parent')
    assert variant['native_runtime_status']=='observed_native_runtime_error'
    assert variant['execution_status']=='unavailable'
    assert variant['native_initialization_status']==variant['layout_status']=='unknown'
    assert variant['selected_final_variant'] is False
    assert not (path.parent/'result.json').exists()


@pytest.mark.parametrize('failure',['schema','origin','complete','agent_calls','missing_log','missing_stack',
    'foreign_log','construction_task_hash','construction_scene_hash','construction_source_hash',
    'construction_schema','construction_reset','frozen_revision','declared_construction_hash'])
def test_unproven_abort_cannot_bind_or_promote_native_results(fixture_world,failure):
    world=fixture_world;path,abort=native_abort(world)
    construction_path=Path(abort['native_construction_receipt'])
    construction=json.loads(construction_path.read_text())
    if failure=='schema':abort['schema']='invented_abort'
    elif failure=='origin':abort['evidence_origin']='unreviewed_log_summary'
    elif failure=='complete':abort['complete']=True
    elif failure=='agent_calls':abort['model_calls']=1
    elif failure in ('missing_log','missing_stack'):
        Path(abort['native_error_log' if failure=='missing_log' else 'native_stack_file']).unlink()
    elif failure=='foreign_log':
        log=world['root']/'other_run/native.log';log.parent.mkdir();log.write_text('Wrong native run.')
        abort['native_error_log']=str(log)
    elif failure.startswith('construction_') and failure.endswith('_hash'):
        kind=failure.split('_')[1]
        row=construction['camera_mounts']['provenance']['task' if kind=='source' else kind]
        row['source_sha256' if kind=='source' else 'file_sha256']['value']='0'*64
    elif failure=='construction_schema':construction['camera_mounts']['schema']='invented'
    elif failure=='construction_reset':construction['camera_mounts']['provenance']['reset_id']['available']=False
    elif failure=='frozen_revision':abort['frozen_revision']='another_revision'
    else:abort['native_construction_receipt_sha256']='0'*64
    write_json(construction_path,construction);write_json(path,abort)
    manifest=build(world)
    assert manifest['counts']['matched_native_evidence']==0
    unbound=manifest['unmatched_native_diagnostics'][0]
    assert unbound['reason']=='unbound_runtime_abort_missing_native_proof'
    assert unbound['missing_proof']
    assert unbound['unbound_diagnostic']['diagnostic_status']=='unbound_native_runtime_abort'
    assert unbound['unbound_diagnostic']['diagnostic_complete'] is False
    assert all(not row['native_evidence_ids'] for row in manifest['variants'])


def test_self_consistent_abort_for_stale_definition_still_requires_current_variant_hash(fixture_world):
    world=fixture_world;path,abort=native_abort(world)
    construction_path=Path(abort['native_construction_receipt'])
    construction=json.loads(construction_path.read_text())
    abort['task_sha256']='0'*64
    construction['camera_mounts']['provenance']['task']['file_sha256']['value']=abort['task_sha256']
    write_json(path,abort);write_json(construction_path,construction)
    assert review.runtime_abort_proof(path,abort)['accepted'] is True
    manifest=build(world)
    assert manifest['counts']['matched_native_runtime_aborts']==0
    unbound=manifest['unmatched_native_diagnostics'][0]
    assert unbound['reason']=='no_exact_explicit_variant_task_scene_source_hash_match'
    assert unbound['missing_proof']==['current_variant_exact_task_scene_source_binding']
    assert all(not row['native_evidence_ids'] for row in manifest['variants'])


def test_result_and_abort_have_separate_ids_and_only_explicit_native_scan_paths(fixture_world):
    world=fixture_world;path,_=native_abort(world)
    write_json(path.parent/'result.json',native_result(world))
    native_abort(world,'historical/native_ignored')
    manifest=build(world)
    assert {row['evidence_id'] for row in manifest['native_evidence']}=={'native_abort','native_abort_runtime_abort'}
    assert manifest['counts']['matched_native_results']==manifest['counts']['matched_native_runtime_aborts']==1
    assert manifest['counts']['matched_native_evidence']==2
    variant=next(row for row in manifest['variants'] if row['candidate_id']=='merged_parent')
    assert len(variant['native_evidence_ids'])==2


def native_checkpoint(world,stage='constructed'):
    """Match the checker's pre-reset-completion result and camera provenance."""
    abort_path,abort=native_abort(world,'native_checkpoint')
    abort_path.unlink()
    original=Path(abort['native_construction_receipt'])
    receipt=json.loads(original.read_text())
    receipt['stage']=receipt['camera_mounts']['stage']=stage
    receipt_path=write_json(original.parent/('initialization_'+stage+'.json'),receipt)
    if original!=receipt_path:original.unlink()
    result={key:abort[key] for key in ('task_file','scene_file','task_sha256','scene_sha256')}
    result.update(status='stopped_at_requested_initialization_checkpoint',complete=False,
        diagnostic_checkpoint_complete=True,checkpoint_stage=stage,benchmark_execution_verified=False,
        model_calls=0,autonomous_agent_performance=False,public_controls_only=True,commands=[])
    path=write_json(abort_path.parent/'result.json',result)
    return path,result,receipt_path


@pytest.mark.parametrize('stage',review.INITIALIZATION_CHECKPOINT_STAGES)
def test_requested_checkpoint_binds_actual_captured_source_without_synthesizing_result(fixture_world,stage):
    world=fixture_world;path,result,receipt_path=native_checkpoint(world,stage)
    before=path.read_bytes()
    manifest=build(world)
    assert manifest['counts']['matched_native_results']==1
    evidence=manifest['native_evidence'][0]
    assert evidence['diagnostic_status']=='requested_initialization_checkpoint'
    assert evidence['execution_status']=='initialization_checkpoint_only'
    assert evidence['diagnostic_complete'] is False
    assert evidence['diagnostic_checkpoint_complete'] is True
    assert evidence['checkpoint_stage']==stage
    assert evidence['benchmark_execution_verified'] is False
    assert evidence['native_initialization_status']==evidence['layout_status']=='unknown'
    assert evidence['scoring_status']=='unavailable'
    assert evidence['agent_task_success'] is evidence['benchmark_task_success'] is evidence['grasp_success'] is None
    assert evidence['autonomous_agent_performance'] is False
    assert evidence['raw_diagnostic_fields']['diagnostic_checkpoint_complete'] is True
    assert evidence['raw_diagnostic_fields']['checkpoint_stage']==stage
    assert evidence['raw_diagnostic_fields']['benchmark_execution_verified'] is False
    proof=evidence['initialization_checkpoint_evidence']
    assert proof['status']=='available'
    assert proof['path']==str(receipt_path.resolve())
    assert proof['sha256']==review.sha256(receipt_path)
    assert proof['provenance_origin']=='actual_initialization_camera_mounts_provenance'
    assert proof['captured_source_identity']==world['identities'][0]
    assert 'source_sha256' not in result and path.read_bytes()==before
    variant=next(row for row in manifest['variants'] if row['candidate_id']=='merged_parent')
    assert review.exact_native_match(variant,result,world['root'])[0] is False
    assert review.exact_native_match(variant,result,world['root'],checkpoint_proof=proof)[0] is True
    assert variant['execution_status']=='initialization_checkpoint_only'
    assert variant['native_initialization_status']==variant['layout_status']=='unknown'
    assert variant['selected_final_variant'] is False


@pytest.mark.parametrize('failure',['stage','missing_receipt','receipt_stage','camera_stage','schema',
    'task_path','task_hash','scene_hash','source_hash','source_file','source_identity','source_missing',
    'reset','seed','conflicting_result_source'])
def test_checkpoint_wrong_or_unavailable_native_context_retains_honest_raw_result(fixture_world,failure):
    world=fixture_world;path,result,receipt_path=native_checkpoint(world)
    receipt=json.loads(receipt_path.read_text())
    task=receipt['camera_mounts']['provenance']['task']
    if failure=='stage':result['checkpoint_stage']='../../other_run/constructed'
    elif failure=='missing_receipt':receipt_path.unlink()
    elif failure=='receipt_stage':receipt['stage']='adapted'
    elif failure=='camera_stage':receipt['camera_mounts']['stage']='adapted'
    elif failure=='schema':receipt['camera_mounts']['schema']='invented'
    elif failure=='task_path':task['file']['value']=str(world['root']/'another/task.yaml')
    elif failure in ('task_hash','scene_hash'):
        kind=failure.split('_')[0]
        receipt['camera_mounts']['provenance'][kind]['file_sha256']['value']='0'*64
    elif failure=='source_hash':task['source_sha256']['value']='0'*64
    elif failure=='source_file':Path(task['source_file']['value']).write_text('Changed source bytes.')
    elif failure=='source_missing':Path(task['source_file']['value']).unlink()
    elif failure=='source_identity':
        other=world['root']/'working_frozen_native_revision/ESI/dataset/json_clean'/world['identities'][1]
        other.parent.mkdir(parents=True,exist_ok=True)
        other.write_bytes(Path(task['source_file']['value']).read_bytes())
        for kind in ('task','scene'):receipt['camera_mounts']['provenance'][kind]['source_file']['value']=str(other)
    elif failure=='reset':receipt['camera_mounts']['provenance']['reset_id']['available']=False
    elif failure=='seed':task['seed']['available']=False
    else:result['source_sha256']='0'*64
    if failure!='missing_receipt':write_json(receipt_path,receipt)
    write_json(path,result)
    manifest=build(world)
    assert not manifest['native_evidence']
    unbound=manifest['unmatched_native_diagnostics'][0]['unbound_diagnostic']
    assert unbound['raw_diagnostic_fields']['checkpoint_stage']==result['checkpoint_stage']
    assert unbound['diagnostic_status']=='requested_initialization_checkpoint'
    assert unbound['execution_status']=='initialization_checkpoint_only'
    assert unbound['diagnostic_complete'] is False
    assert unbound['benchmark_task_success'] is None
    proof=unbound['initialization_checkpoint_evidence']
    if failure=='source_identity':assert proof['accepted'] is True  # Current variant identity still rejects it.
    else:
        assert proof['status']=='unavailable'
        assert proof['reason'] and proof['missing_proof']
        assert proof['path'] is None
    assert all(not row['native_evidence_ids'] for row in manifest['variants'])


def test_checkpoint_does_not_promote_setup_diagnostic_or_complete_capture(fixture_world):
    world=fixture_world;path,result,_=native_checkpoint(world,'adapted')
    result['initialization_diagnostics']={'interaction_setup':{'accepted':True}}
    write_json(path,result)
    for name in (*review.MOUNTED,'initial_overview_rgb'):
        image=path.parent/'initial'/(name+'.png');image.parent.mkdir(exist_ok=True)
        image.write_bytes(b'Unqualified image artifact.')
    evidence=build(world)['native_evidence'][0]
    assert evidence['interaction_setup_evidence'][0]['receipt']['accepted'] is True
    assert evidence['layout_status']=='unknown'
    assert evidence['captures']['initial_capture_status']=='pending'
    assert evidence['benchmark_execution_verified'] is False


def test_checkpoint_flag_alone_does_not_change_default_native_result(fixture_world):
    world=fixture_world;result=native_result(world)
    result['diagnostic_checkpoint_complete']=True
    write_json(world['input_root']/'native_default/result.json',result)
    evidence=build(world)['native_evidence'][0]
    assert evidence['diagnostic_complete'] is True
    assert evidence['execution_status']=='public_control_only'
    assert evidence['initialization_checkpoint_evidence'] is None


def test_merge_membership_dedup_and_additions_are_separate(fixture_world):
    world = fixture_world
    manifest = build(world)
    assert manifest['counts']['original_core_sources'] == 3
    assert manifest['counts']['explicit_derivative_variants'] == 2
    assert manifest['counts']['additional_sources'] == 1
    assert manifest['counts']['mixed_core_and_additional_variants'] == 1
    assert manifest['counts']['additional_only_variants'] == 1
    assert [row['original_membership_record'] for row in manifest['core_sources']] == world['originals']
    for row in manifest['core_sources'][:2]:
        assert len(row['derivative_memberships']) == 1
    assert manifest['core_sources'][2]['derivative_memberships'] == []
    assert len(manifest['additional_sources'][0]['variant_memberships']) == 2
    assert not any(v['candidate_id'] in ('backup', 'obsolete_v1') for v in manifest['variants'])
    assert manifest['final_selection_owner'] == 'user'
    assert manifest['final_core_selection_made'] is False


def test_duplicate_variant_records_union_explicit_members(fixture_world):
    world = fixture_world
    first = deepcopy(world['parent'])
    second = deepcopy(first)
    first['source_to_parent'] = [first['source_to_parent'][0]]
    second['source_to_parent'] = second['source_to_parent'][:3]
    write_json(world['input_root'] / review.CURRENT_INPUTS[4], {'candidates': [first, second, second]})
    manifest = build(world)
    assert manifest['counts']['explicit_derivative_variants'] == 1
    parent = next(row for row in manifest['variants'] if row['candidate_id'] == 'merged_parent')
    assert len(parent['sources']) == 3
    assert parent['source_binding_status'] == 'verified'
    assert [len(row['derivative_memberships']) for row in manifest['core_sources']] == [1, 1, 0]


def test_unmeasured_statuses_remain_unknown_unavailable(fixture_world):
    manifest = build(fixture_world)
    unavailable = next(row for row in manifest['variants'] if row['candidate_id'] == 'original_2')
    assert unavailable['schema_status'] == 'unavailable'
    assert unavailable['task']['sha256'] is None
    for variant in manifest['variants']:
        assert variant['native_initialization_status'] == variant['layout_status'] == 'unknown'
        assert variant['execution_status'] == variant['scoring_status'] == 'unavailable'
        assert variant['autonomous_agent_performance'] is False
        assert variant['selected_final_variant'] is False
        assert variant['pending_review_gates']


def test_exact_native_binding_captures_and_failed_attempts(fixture_world):
    world = fixture_world
    result = native_result(world)
    directory = world['input_root'] / 'native_exact'
    write_json(directory / 'result.json', result)
    for name in (*review.MOUNTED, 'initial_overview_rgb'):
        path = directory / 'initial' / (name + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture image')
    manifest = build(world)
    assert manifest['counts']['matched_native_results'] == 1
    evidence = manifest['native_evidence'][0]
    assert evidence['qualification'] == result['qualification']
    assert evidence['layout_status'] == 'accepted'
    assert evidence['interaction_setup_evidence'][0]['receipt']['accepted'] is False
    assert evidence['captures']['initial_capture_status'] == 'available'
    assert evidence['captures']['all_eight_mounted_present'] is True
    assert Path(evidence['captures']['initial_global']).is_absolute()
    assert evidence['autonomous_agent_performance'] is False
    assert evidence['benchmark_task_success'] is evidence['agent_task_success'] is evidence['grasp_success'] is None
    assert evidence['execution_status'] == 'public_control_only'
    bound = [row for row in manifest['variants'] if row['native_evidence_ids']]
    assert [row['candidate_id'] for row in bound] == ['merged_parent']
    assert bound[0]['native_initialization_status'] == bound[0]['layout_status'] == 'accepted'


@pytest.mark.parametrize('field', ['task_file', 'task_sha256', 'scene_sha256', 'source_sha256'])
def test_native_family_path_or_hash_mismatch_never_transfers(fixture_world, field):
    world = fixture_world
    result = native_result(world)
    if field == 'task_file':
        # Byte-identical files at another path are a different selection.
        copy = world['root'] / 'other_task.yaml'
        copy.write_bytes(Path(result[field]).read_bytes())
        result[field] = str(copy)
    elif field == 'source_sha256':
        result['initial_condition'][field] = '0' * 64
    else:
        result[field] = '0' * 64
    result['error'] = 'A native failure must remain a diagnostic.'
    write_json(world['input_root'] / 'native_mismatch/result.json', result)
    write_json(world['input_root'] / 'historical/native_same_family/result.json', native_result(world))
    manifest = build(world)
    assert manifest['counts']['matched_native_results'] == 0
    assert len(manifest['unmatched_native_diagnostics']) == 1
    unbound = manifest['unmatched_native_diagnostics'][0]
    assert unbound['unbound_diagnostic']['qualification'] == result['qualification']
    assert unbound['error'] == result['error']
    assert all(not row['native_evidence_ids'] for row in manifest['variants'])


def test_loaded_task_provenance_and_conflicting_receipts(fixture_world):
    world = fixture_world
    variant = next(row for row in build(world)['variants'] if row['candidate_id'] == 'merged_parent')
    result = native_result(world)
    result['loaded_task_provenance'] = {'task_file': result.pop('task_file'),
        'task_file_sha256': result.pop('task_sha256'), 'scene_file': result.pop('scene_file'),
        'scene_file_sha256': result.pop('scene_sha256'),
        'esi_source_sha256': result['initial_condition'].pop('source_sha256')}
    assert review.exact_native_match(variant, result, world['root'])[0] is True
    result['task_sha256'] = 'contradictory'
    assert review.exact_native_match(variant, result, world['root'])[0] is False


def test_all_native_attempts_retained_without_autochoosing(fixture_world):
    world = fixture_world
    accepted = native_result(world)
    rejected = deepcopy(accepted)
    rejected['final_state']['interaction_setup']['accepted'] = False
    rejected['complete'] = False
    rejected['error'] = 'Setup failed.'
    write_json(world['input_root'] / 'native_attempt_a/result.json', accepted)
    write_json(world['input_root'] / 'native_attempt_b/result.json', rejected)
    manifest = build(world)
    parent = next(row for row in manifest['variants'] if row['candidate_id'] == 'merged_parent')
    assert parent['layout_status'] == 'mixed_native_attempts'
    assert len(parent['native_evidence_ids']) == 2
    assert parent['observed_native_statuses']['layout_status'] == ['accepted', 'observed_rejected']
    assert manifest['native_evidence'][1]['diagnostic_complete'] is False


def test_source_hash_mismatch_and_unmapped_definition_not_guessed(fixture_world):
    world = fixture_world
    builder = review.Builder(world['root'], world['source_root'])
    wrong = deepcopy(world['parent'])
    wrong['source_to_parent'][0]['source_sha256'] = 'wrong'
    assert builder.variant(wrong, 'fixture')['source_binding_status'] == 'source_hash_mismatch'
    unmapped = {key: value for key, value in world['parent'].items() if key != 'source_to_parent'}
    assert builder.variant(unmapped, 'fixture')['source_binding_status'] == 'unresolved_manifest_mapping'
    malformed = world['root'] / 'malformed.yaml'
    malformed.write_text('[]')
    definition, _ = builder.definition(malformed, 'task')
    assert definition['status'] == 'schema_invalid'


@pytest.fixture
def protected_link(tmp_path, monkeypatch):
    """Use the real Rigid loader with separate question and parent scene sources."""
    scene_root = tmp_path / 'scenes'
    source_root = tmp_path / 'dataset'
    hierarchy = Path('Physical Structure/Rigid Containment/map/room')
    question_identity = str(hierarchy / 'sample__criterion_2.json')
    parent_identity = str(hierarchy / 'sample_revised.json')
    question = {'scene': 'map', 'room': 'room', 'small_task': 'Rigid Containment',
                'question': 'Which container fits each original object?'}
    parent = {'scene': 'map', 'room': 'room', 'small_task': 'Rigid Containment',
              'question': 'Original parent scene question.'}
    for root in (scene_root, source_root):
        write_json(root / question_identity, question)
        write_json(root / parent_identity, parent)
    scene_file = str(hierarchy / 'sample__multi_extra1.yaml')
    scene = {'kind': 'scene_patch', 'id': 'protected_parent', 'status': 'ready',
        'source': {'provider': 'esi', 'json': parent_identity},
        'target': 'target', 'target_bindings': ['target', 'extra1'],
        'bindings': {key: {'name': key} for key in ('target', 'extra1', 'support')},
        'qualification': {'support': 'support', 'supported_objects': ['target', 'extra1']},
        'scene': {'policy': 'native_then_minimal_patch_else_skip',
                  'variants': {'original': {'patches': []}, 'tabletop': {'patches': []}}}}
    task = {'kind': 'rigid_containment_task', 'id': 'protected_criterion', 'status': 'ready',
        'source': {'provider': 'esi', 'json': question_identity}, 'scene_file': scene_file,
        'targets': ['target', 'extra1'], 'goal': 'answer_and_place',
        'public': {'objective': question['question'], 'success_criteria': 'Answer for both objects.',
                   'environment': 'The original parent scene.'}}
    task_path, scene_path = tmp_path / 'task.yaml', scene_root / scene_file
    task_path.write_text(yaml.safe_dump(task))
    scene_path.write_text(yaml.safe_dump(scene))
    monkeypatch.setattr(review.task_loader, 'SCENE_ROOT', scene_root)
    record = {'candidate_id': 'protected_original', 'protected_baseline': True,
        'source': question_identity, 'source_identity': question_identity,
        'task_file': str(task_path), 'scene_file': str(scene_path)}
    return {'root': tmp_path, 'source_root': source_root, 'scene_root': scene_root,
            'record': record, 'task_raw': task, 'task_path': task_path, 'scene_raw': scene,
            'question_identity': question_identity, 'parent_identity': parent_identity}


def test_protected_declared_parent_linkage_preserves_both_sources_without_native_transfer(protected_link):
    fixture = protected_link
    builder = review.Builder(fixture['root'], fixture['source_root'])
    variant = builder.variant(fixture['record'], 'protected_baseline', original=True)
    assert variant['source_binding_status'] == 'verified_declared_protected_parent_linkage'
    assert builder.ambiguities == []
    linkage = variant['declared_protected_parent_linkage']
    assert linkage['resolved_scene_file'] == str(Path(fixture['record']['scene_file']).resolve())
    assert linkage['question_source']['sha256'] == review.sha256(fixture['scene_root'] / fixture['question_identity'])
    assert linkage['parent_scene_source']['sha256'] == review.sha256(fixture['scene_root'] / fixture['parent_identity'])
    assert linkage['question_source']['sha256'] != linkage['parent_scene_source']['sha256']
    assert linkage['original_question_verbatim'] is True
    assert linkage['target_bindings_in_order'] == ['target', 'extra1']
    assert variant['original_record'] == fixture['record']
    native = {'task_file': variant['task']['path'], 'scene_file': variant['scene']['path'],
        'task_sha256': variant['task']['sha256'], 'scene_sha256': variant['scene']['sha256'],
        'source_sha256': linkage['question_source']['sha256']}
    matched, checks = review.exact_native_match(variant, native)
    assert matched is False and checks['definition_source_binding'] is False
    assert all(value for key, value in checks.items() if key != 'definition_source_binding')


@pytest.mark.parametrize('change', ['wrong_scene_file', 'wrong_targets', 'changed_question', 'unprotected', 'derivative'])
def test_protected_linkage_requires_exact_declared_path_loader_checks_and_original_status(protected_link, change):
    fixture = protected_link
    record = deepcopy(fixture['record'])
    task = deepcopy(fixture['task_raw'])
    if change == 'wrong_scene_file':
        task['scene_file'] = 'Physical Structure/Rigid Containment/map/room/unrelated.yaml'
    elif change == 'wrong_targets':
        task['targets'].reverse()
    elif change == 'changed_question':
        task['public']['objective'] = 'Changed question.'
    elif change == 'unprotected':
        record['protected_baseline'] = False
    fixture['task_path'].write_text(yaml.safe_dump(task))
    builder = review.Builder(fixture['root'], fixture['source_root'])
    variant = builder.variant(record, 'fixture', original=change != 'derivative')
    assert variant['source_binding_status'] == 'definition_source_mismatch'
    assert any(row['kind'] == 'scene' and row['reason'] == 'definition_source_mismatch'
               for row in builder.ambiguities)
    if change not in ('unprotected', 'derivative'):
        assert variant['declared_protected_parent_linkage']['reason']
    else:
        assert variant['declared_protected_parent_linkage'] is None


def test_temporal_core_is_an_explicit_optional_input():
    assert {'temporal_core/manifest.json', 'temporal_core_v2/manifest.json',
            'temporal_core_v3/manifest.json'} <= set(review.OPTIONAL_INPUTS)


def test_temporal_versioned_inputs_preserve_original_rows_and_all_explicit_derivatives(fixture_world):
    world=fixture_world
    for revision in ('v2','v3'):
        record=definitions(world['root'],'temporal_core_'+revision,world['identities'][0])
        write_json(world['input_root']/('temporal_core_'+revision)/'manifest.json',{'records':[record]})
    manifest=build(world)
    assert [row['original_membership_record'] for row in manifest['core_sources']]==world['originals']
    assert {'temporal_core_v2','temporal_core_v3'} <= {row['candidate_id'] for row in manifest['variants']}
    assert all(row['selected_final_variant'] is False for row in manifest['variants'])


@pytest.mark.parametrize('facts',[
    {'final_state':{'support':True},'commands':[{'command':{'action':'BASE'},
        'result':{'completed':False,'reason':'inactive_joint_drift'}}]},
    {'final_state':{'support':True},'commands':[{'command':{'action':'BASE'},
        'result':{'completed':False,'reason':'base_motion_blocked'}}]},
    {'final_state':{'support':True,'actual_trials':1}},
    {'final_state':{'source_pose_error_m':.0006067,'FOV':True},'commands':[
        {'command':{'action':'BASE','distance_m':.25},'result':{'completed':True}},
        {'command':{'action':'BASE','distance_m':.25},'result':{'completed':True}}]},
    {'final_state':{'table_layout':{'accepted':False,'accepted_tables':1,'required_tables':3}}},
    {'final_state':{'support_initialization':{'available':False,'reason':'initialization_missing'}}},
])
def test_compact_native_summary_links_raw_facts_without_duplicate_states_or_full_task_success(tmp_path,facts):
    result={'complete':True,'model_calls':0,'autonomous_agent_performance':False,**deepcopy(facts)}
    path=write_json(tmp_path/'result.json',result)
    summary=review.native_summary(path,result)
    assert not {'initial_state','initial_condition','final_state','commands'} & set(summary['raw_diagnostic_fields'])
    assert summary['private_result_file']==str(path.resolve())
    assert summary['result_sha256']==review.sha256(path)
    assert json.loads(path.read_text())==result
    assert summary['public_control_results']==[
        {'command':row['command'],'completed':row['result'].get('completed'),'reason':row['result'].get('reason')}
        for row in facts.get('commands',[])]
    assert summary['agent_task_success'] is summary['grasp_success'] is summary['benchmark_task_success'] is None
    assert summary['autonomous_agent_performance'] is False and summary['scoring_status']=='unavailable'


@pytest.fixture(scope='module')
def current_manifest():
    return review.build_manifest()


def test_actual_full_161_membership_and_explicit_latest_inputs(current_manifest):
    manifest = current_manifest
    originals = json.loads(review.BASELINE.read_text())['tasks']
    assert len(originals) == manifest['counts']['original_core_sources'] == 161
    assert [row['source_identity'] for row in manifest['core_sources']] == [row['source'] for row in originals]
    assert [row['original_membership_record'] for row in manifest['core_sources']] == originals
    permitted = {str((review.ROOT / 'outputs/task_edit2_20261018' / value).resolve())
                 for value in review.CURRENT_INPUTS + review.OPTIONAL_INPUTS}
    assert {row['path'] for row in manifest['explicit_current_inputs']} <= permitted
    assert not any('deformable_candidates_v1' in row['path'] or 'backup' in row['path']
                   for row in manifest['explicit_current_inputs'])
    assert len({row['variant_key'] for row in manifest['variants']}) == len(manifest['variants'])
    source_ids = {row['source_identity'] for row in manifest['core_sources']}
    assert not source_ids & {row['source_identity'] for row in manifest['additional_sources']}
    for source in manifest['core_sources']:
        keys = [row['variant_key'] for row in source['derivative_memberships']]
        assert len(keys) == len(set(keys))
    additions = [row for row in manifest['variants'] if row.get('membership_class') == 'additional_prototype']
    assert sum(str(row['candidate_id']).startswith('enclosure_addition_') for row in additions) == 3
    assert sum(str(row['candidate_id']).startswith('temporal_') for row in additions) == 5


def test_actual_derivative_definitions_exist_schema_valid_and_source_hashed(current_manifest):
    derivatives = [row for row in current_manifest['variants'] if row['role'] == 'explicit_derivative']
    assert derivatives
    for row in derivatives:
        assert row['schema_status'] == 'schema_valid', row['candidate_id']
        assert row['source_binding_status'] == 'verified', row['candidate_id']
        assert row['sources']
        for kind in ('task', 'scene'):
            assert Path(row[kind]['path']).is_file()
            assert review.sha256(row[kind]['path']) == row[kind]['sha256']
        for member in row['sources']:
            assert member['provenance_status'] == 'verified_current_source_bytes'
            assert member['source_sha256'] == review.sha256(member['source_file'])
    assert current_manifest['final_selection_owner'] == 'user'
    assert all(row['selected_final_variant'] is False for row in current_manifest['variants'])


def test_actual_new_optional_manifests_load_without_duplicate_enclosure(current_manifest):
    relatives=('perception_v3/search_manifest.json','reflection_v3/batch_manifest.json',
               'reflection_v3_height_pilot/batch_manifest.json','reflection_v4_height_pilot/batch_manifest.json')
    for relative in relatives:
        assert relative in review.OPTIONAL_INPUTS
        path=(review.ROOT/'outputs/task_edit2_20261018'/relative).resolve()
        if relative.startswith('reflection_v4_') and not path.is_file():
            assert {'path':str(path),'optional':True} in current_manifest['missing_explicit_inputs']
            continue
        assert str(path) in {row['path'] for row in current_manifest['explicit_current_inputs']}
        raw=json.loads(path.read_text())
        records=raw.get('records',raw.get('candidates'))
        loaded=[row for row in current_manifest['variants'] if any(
            item['manifest_file']==str(path) for item in row['manifest_records'])]
        assert {row['candidate_id'] for row in loaded}=={row['candidate_id'] for row in records}
        if relative.startswith('perception_v3/'):
            assert len(loaded)==2
            assert all(row['candidate_id'].startswith('search_') for row in loaded)
    enclosures=[row for row in current_manifest['variants'] if str(row['candidate_id']).startswith('enclosure_addition_')]
    assert len(enclosures)==3


def test_actual_capacity_abort_matches_captured_construction_without_claiming_completion():
    path=review.ROOT/'outputs/task_edit2_20261018/native_capacity_tabletop_v1/runtime_abort.json'
    abort=json.loads(path.read_text())
    proof=review.runtime_abort_proof(path,abort)
    assert proof['accepted'],proof['missing_proof']
    summary=review.runtime_abort_summary(path,abort,proof)
    assert summary['diagnostic_complete'] is False
    assert summary['diagnostic_status']=='observed_native_runtime_error'
    assert summary['execution_status']=='unavailable'
    assert summary['benchmark_task_success'] is None


def test_actual_ten_protected_criterion_parent_links_use_real_loader(current_manifest):
    variants = [row for row in current_manifest['variants']
        if row['source_binding_status'] == 'verified_declared_protected_parent_linkage']
    assert len(variants) == 10
    ambiguous = {row.get('candidate_id') for row in current_manifest['mapping_ambiguities']}
    for row in variants:
        assert row['role'] == 'original_baseline'
        assert row['original_record']['protected_baseline'] is True
        linkage = row['declared_protected_parent_linkage']
        assert linkage['loader'] == 'wia.tasks.loader.load_task -> load_rigid_task'
        assert linkage['resolved_scene_file'] == row['scene']['path']
        assert linkage['question_source']['sha256'] == review.sha256(linkage['question_source']['path'])
        assert linkage['parent_scene_source']['sha256'] == review.sha256(linkage['parent_scene_source']['path'])
        assert row['candidate_id'] not in ambiguous
        assert row['native_evidence_ids'] == []
