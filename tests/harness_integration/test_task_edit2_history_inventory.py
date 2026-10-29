"""Check explicit Core history inventory redesign without native or model execution.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider
tests/test_task_edit2_history_inventory.py tests/test_task_edit2_temporal.py
Synthetic donor receipts exercise validation; they do not certify native assets.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.authoring.conversion import ConversionContext, ROOT, SourceInstance
from wia.authoring.converters import temporal_understanding_agent_observation as history
from wia.runtime.scene_capture_queue import sha256
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition, CategoryInteractionPatch


CORE_MANIFEST = Path('/home/zhouxinyang/esi_category_conversion/core_revision_20261018/core_candidates_for_review.json')


def core_sources():
    return [SourceInstance(ROOT/'ESI/dataset/json_clean'/row['source'],row['source'],*history.CATEGORY)
        for row in json.loads(CORE_MANIFEST.read_text())['tasks']
        if row['source'].startswith('Temporal Understanding/Agent Observation/')]


def explicit_config(source, directory):
    originals=history._source_data(source)[2]
    rows,positions,supports,labels,added=[],{},{},{},[]
    assets=[('history_added_jam','jar_of_jam','vxjbjf'),
            ('history_added_box','box_of_sanitary_napkins','ykpeja')]
    for index,(key,category,model) in enumerate(assets,1):
        donor_name='donor_'+key
        geometry={'position_m':[0.,0.,.05],'aabb_min_m':[-.03,-.04,0.],
            'aabb_max_m':[.03,.04,.10],'quaternion_xyzw':[0.,0.,0.,1.]}
        receipt={'objects':{donor_name:{'category':category,'scale':[1.,1.,1.]}},
            'interaction_setup':{'original_geometry':{donor_name:geometry}}}
        path=directory/(key+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(receipt))
        point=np.array([index*1.2,0.,.772])
        delta=point-np.asarray(geometry['position_m'])
        obj={'_key':key,'category':category,'model':model,'scale':[1.,1.,1.],
            'position':point.tolist(),'quaternion_xyzw':geometry['quaternion_xyzw'],
            'aabb_min':(np.asarray(geometry['aabb_min_m'])+delta).tolist(),
            'aabb_max':(np.asarray(geometry['aabb_max_m'])+delta).tolist(),
            'asset_provenance':{'source_file':str(path.resolve()),'source_sha256':sha256(path),
                'source_object_name':donor_name,'geometry_json_path':'interaction_setup.original_geometry.'+donor_name,
                'native_identity_json_path':'objects.'+donor_name,'native_category':category,'native_scale':[1.,1.,1.],
                'root_offset_from_aabb_min_m':[.03,.04,.05],
                'geometry_status':'measured_native_source_pose_proposal_only'}}
        added.append(obj)
    for index,obj in enumerate(originals+added):
        key=obj['_key'];station='history_station_'+str(index)
        point=obj['position'] if index else [0.,0.,.8]
        rows.extend(history._table_primitives(station,[index*1.2,0.],0.,.72,[.5,.2,.1,1.]))
        positions[key]=deepcopy(point);supports[key]=station
        labels[key]='the '+obj['category'].replace('_',' ')+' at station '+str(index)
    rows.append({'name':'delivery_tray','category':'tray','position':[0.,-1.,.75],
                 'dimensions_m':[.5,.4,.04]})
    dataset=[{'type':'DatasetObject','name':obj['_key'],'category':obj['category'],'model':obj['model'],
        'position':deepcopy(obj['position']),'orientation':deepcopy(obj['quaternion_xyzw']),
        'scale':[1.,1.,1.],'fixed_base':False,'visual_only':False} for obj in added]
    return {'task_edit2_temporal':True,'history_inventory_revision':history.HISTORY_INVENTORY_REVISION,
        'history_additional_targets':added,'history_target_labels':labels,
        'history_public_route':'Inspect the three named tabletop landmarks; their listing does not define encounter order.',
        'history_destination':{'binding_key':'delivery_tray','public_label':'the delivery tray','relation':'on_top'},
        'history_apparatus':{'primitives':rows,'dataset_objects':dataset,
            'source_target_positions':positions,'target_support_bindings':supports}}


@pytest.fixture
def configured(tmp_path):
    source=core_sources()[0]
    return source,explicit_config(source,tmp_path/'donors')


def test_original_inventory_and_config_are_immutable_and_legacy_remains_source_only(configured):
    source,config=configured
    original=deepcopy(history._source_data(source)[2]);before=deepcopy(config);raw=source.path.read_bytes()
    inventory=history.history_target_inventory(source,config)
    assert len(inventory)==3 and inventory[:1]==original
    assert inventory[1:]==config['history_additional_targets']
    inventory[0]['position'][0]+=10
    assert history._source_data(source)[2]==original
    assert config==before and source.path.read_bytes()==raw
    config.pop('history_inventory_revision')
    assert history.history_target_inventory(source,config)==original
    with pytest.raises(ValueError,match='at least two source-supported'):
        history.build_history_retrieval(source,ConversionContext(Path('/tmp'),0,config))


@pytest.mark.parametrize('index',range(4))
def test_all_exact_core_single_sources_build_three_target_schema_with_explicit_policy(tmp_path,index):
    sources=core_sources();assert len(sources)==4
    source=sources[index];config=explicit_config(source,tmp_path/'donors')
    originals=deepcopy(history._source_data(source)[2]);assert len(originals)==1
    before=source.path.read_bytes()
    result=history.build_history_retrieval(source,ConversionContext(tmp_path,7,config))
    task=result['task_definition'];scene=result['scene_definition']
    task_definition(task);SceneDefinition.model_validate(scene)
    p=task['private']['evaluation']['parameters']
    original_key=originals[0]['_key'];added_keys=[row['_key'] for row in config['history_additional_targets']]
    assert p['target_keys']==[original_key,*added_keys]
    assert p['original_source_objects']==originals and p['source_objects'][original_key]==originals[0]
    assert p['original_source_target_count']==1
    assert p['historical_count_answer_is_provenance_only']==source.raw['answer']
    assert p['original_target_keys']==result['original_target_keys']==[original_key]
    assert p['added_target_keys']==result['added_target_keys']==added_keys
    assert p['history_inventory_revision']==result['history_inventory_revision']==history.HISTORY_INVENTORY_REVISION
    assert 'two approved' in result['capability_change']
    assert p['source_identity']==source.relative_path and p['source_sha256']==sha256(source.path)
    assert result['execution_status']=='unavailable' and set(result['native_gates'].values())=={'pending'}
    for obj in config['history_additional_targets']:
        assert task['private']['bindings'][obj['_key']]=={
            'name':obj['_key'],'category':obj['category'],'model':obj['model']}
    policy=result['interaction_policy']
    CategoryInteractionPatch.model_validate({**policy,'operation':'prepare_category_interaction','phase':'after_postprocessing'})
    assert policy['placement_groups']==[[key] for key in p['target_keys']]
    assert policy['support_preferences']==config['history_apparatus']['target_support_bindings']
    assert policy['allow_fallback_table'] is False
    env={'scene':{'scene_model':source.raw['scene']},'objects':[{'type':'DatasetObject','name':original_key,
        'category':originals[0]['category'],'model':originals[0]['model'],'position':originals[0]['position'],
        'orientation':originals[0]['quaternion_xyzw']}]}
    history.configure_scene(env,{'task_revision':p['task_revision'],'scene_model':source.raw['scene'],'apparatus':p['apparatus']})
    for key in p['target_keys']:
        native=next(row for row in env['objects'] if row['name']==key)
        assert native['position']==config['history_apparatus']['source_target_positions'][key]
        assert native['fixed_base'] is native['visual_only'] is False
    original=next(row for row in env['objects'] if row['name']==original_key)
    assert original['model']==originals[0]['model'] and original['orientation']==originals[0]['quaternion_xyzw']
    assert source.path.read_bytes()==before


@pytest.mark.parametrize('failure',['one_added','three_added','duplicate_name','wrong_asset','missing_provenance',
    'donor_hash','donor_path','offset','bounds','quaternion','scale','dataset_absent','dataset_model',
    'dataset_pose','dataset_fixed','dataset_visual','dataset_scale','position_absent','position_mismatch',
    'label','support','support_geometry','conflicting_binding'])
def test_missing_or_mismatched_explicit_inventory_is_rejected(configured,failure):
    source,config=configured;obj=config['history_additional_targets'][0]
    dataset=config['history_apparatus']['dataset_objects'][0];plan=config['history_apparatus']
    if failure=='one_added':config['history_additional_targets'].pop()
    elif failure=='three_added':config['history_additional_targets'].append(deepcopy(obj))
    elif failure=='duplicate_name':obj['_key']=history._source_data(source)[2][0]['_key']
    elif failure=='wrong_asset':obj['model']='unspecified_model'
    elif failure=='missing_provenance':obj.pop('asset_provenance')
    elif failure=='donor_hash':obj['asset_provenance']['source_sha256']='0'*64
    elif failure=='donor_path':obj['asset_provenance']['native_identity_json_path']='objects.another_object'
    elif failure=='offset':obj['asset_provenance']['root_offset_from_aabb_min_m'][2]+=.1
    elif failure=='bounds':obj['aabb_max'][0]+=.1
    elif failure=='quaternion':obj['quaternion_xyzw']=[0.,0.,1.,0.];dataset['orientation']=obj['quaternion_xyzw']
    elif failure=='scale':obj['scale']=[.5,.5,.5]
    elif failure=='dataset_absent':plan['dataset_objects'].pop(0)
    elif failure=='dataset_model':dataset['model']='conflicting_model'
    elif failure=='dataset_pose':dataset['position'][0]+=.2
    elif failure=='dataset_fixed':dataset['fixed_base']=True
    elif failure=='dataset_visual':dataset['visual_only']=True
    elif failure=='dataset_scale':dataset['scale']=[.5,.5,.5]
    elif failure=='position_absent':plan['source_target_positions'].pop(obj['_key'])
    elif failure=='position_mismatch':plan['source_target_positions'][obj['_key']][2]+=.1
    elif failure=='label':config['history_target_labels'].pop(obj['_key'])
    elif failure=='support':plan['target_support_bindings'][obj['_key']]='not_declared'
    elif failure=='support_geometry':plan['primitives'][5]['dimensions_m']=[0.,0.,0.]
    else:plan['primitives'][0]['name']=history._source_data(source)[2][0]['_key']
    with pytest.raises(ValueError):history.history_target_inventory(source,config)


def test_legacy_prototype_policy_is_not_activated_or_rewritten(tmp_path):
    from wia.runtime.commands.build_task_edit2_temporal_candidates import HISTORY_SOURCE,source,history_config
    src=source(HISTORY_SOURCE,'Agent Observation');config=history_config(src)
    result=history.build_history_retrieval(src,ConversionContext(tmp_path,0,config))
    assert 'history_inventory_revision' not in result
    assert 'placement_groups' not in result['interaction_policy']
    assert 'support_preferences' not in result['interaction_policy']
    assert history.history_target_inventory(src,config)==history._source_data(src)[2]


class R1:
    def __init__(self):
        self.position=np.array([9.,8.,.5]);self.quaternion=np.array([0.,0.,0.,1.]);self.writes=[]

    def get_position_orientation(self):
        return self.position.copy(),self.quaternion.copy()

    def set_position_orientation(self,*,position,orientation):
        self.position=position.detach().cpu().numpy().copy()
        self.quaternion=orientation.detach().cpu().numpy().copy()
        self.writes.append((self.position.copy(),self.quaternion.copy()))


class StartupSession(NS):
    @property
    def observation_source(self):
        raise AssertionError('The startup guard must never invoke the lazy observation_source property.')


@pytest.fixture
def startup():
    source=core_sources()[0];payload=source.payload;robot=R1();steps=[]
    session=StartupSession(env=NS(robots=[robot]),robot=robot,payload=payload,
        scene_name=source.raw['scene'],room_name=source.raw['room'],question_path=source.path,
        pipeline=NS(og=NS(sim=NS(step=lambda:steps.append(len(steps))))),
        native_reset_id='retained_native_reset')
    parameters={'restore':'original_episode_start','scene':source.raw['scene'],'room':source.raw['room'],
        'position':[*payload['start_xy'],float(payload['robot_start_z'])],
        'quaternion_xyzw':payload['step_log'][0]['robot_quat_xyzw'],
        'source_sha256':sha256(source.path),'provenance':{'source_json':source.relative_path}}
    return session,robot,steps,parameters


@pytest.mark.parametrize('bound',[True,False])
def test_startup_allows_only_retained_r1_before_observations_and_keeps_original_settling(startup,bound):
    session,robot,steps,parameters=startup
    if not bound:session.robot=None
    session._observation_source=NS(initial_observation=None,latest_snapshot=None,boundary_observation=None,
        history=(),action_step=0,snapshot_id=0,controller_state_provider=NS(__self__=session))
    result=history.initialize_scene(session,parameters)
    assert len(robot.writes)==1 and len(steps)==result['settle_steps']==30
    np.testing.assert_allclose(robot.position,parameters['position'],atol=1e-6)
    np.testing.assert_allclose(robot.quaternion,parameters['quaternion_xyzw'],atol=1e-7)
    assert session._agent_observation_start_restored is True
    assert result['physical_success_evidence'] is False
    assert result['source_sha256']==parameters['source_sha256']
    assert 'observation_source' not in vars(session)
    with pytest.raises(ValueError,match='initialization-only'):history.initialize_scene(session,parameters)
    assert len(robot.writes)==1 and len(steps)==30


@pytest.mark.parametrize('slot',['_observation_source','observation_source'])
@pytest.mark.parametrize('retained',[{'initial_observation':object()},{'latest_snapshot':object()},
    {'boundary_observation':object()},{'initial_overview':object()},{'history':('observed',)},
    {'action_step':1},{'snapshot_id':1}])
def test_startup_rejects_retained_observation_without_pose_writes_or_lazy_source_access(startup,slot,retained):
    session,robot,steps,parameters=startup
    vars(session)[slot]=NS(**retained)
    with pytest.raises(ValueError,match='initialization-only'):history.initialize_scene(session,parameters)
    assert not robot.writes and not steps


@pytest.mark.parametrize('owner',['session','controller','task_controller','task_evaluator','executor','source_provider'])
@pytest.mark.parametrize('active',[True,False])
def test_startup_rejects_attached_or_recorded_public_controls_before_any_write(startup,owner,active):
    session,robot,steps,parameters=startup
    controller=NS(_public_action_sequence=1,_public_action_context={'sequence':1,'active':active})
    if owner=='session':session._public_action_sequence=1
    elif owner in ('task_evaluator','executor'):setattr(session,owner,NS(controller=controller))
    elif owner=='source_provider':session._observation_source=NS(controller_state_provider=NS(__self__=controller))
    else:setattr(session,owner,controller)
    with pytest.raises(ValueError,match='initialization-only'):history.initialize_scene(session,parameters)
    assert not robot.writes and not steps


@pytest.mark.parametrize('failure',['foreign_robot','multiple_robots','wrong_robot_type','attached_controller',
    'source_sha','scene','source_pose'])
def test_startup_rejects_stale_robot_or_source_context_before_mutation(startup,failure):
    session,robot,steps,parameters=startup
    if failure=='foreign_robot':session.robot=R1()
    elif failure=='multiple_robots':session.env.robots.append(R1())
    elif failure=='wrong_robot_type':session.env.robots=[NS()]
    elif failure=='attached_controller':session.controller=NS()
    elif failure=='source_sha':parameters['source_sha256']='0'*64
    elif failure=='scene':parameters['scene']='another_scene'
    else:parameters['position'][0]+=.1
    with pytest.raises(ValueError):history.initialize_scene(session,parameters)
    assert not robot.writes and not steps
