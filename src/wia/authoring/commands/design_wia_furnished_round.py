"""Materialize the main-authored furnished-scene design, without native actions.

Run: behavior/bin/python -B scripts/design_wia_furnished_round.py
This records explicit layout choices; initialization captures must qualify them.


Canonical invocation: python -m wia.authoring.commands.design_wia_furnished_round
"""
from __future__ import annotations
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import math
import yaml
from wia.resources import ROOT
from wia.authoring.commands._inputs import ARTIFACT_ROOT, LazyMapping, read_json_input
from wia.tasks.sources import resolve_source
OUT=ROOT/'explore_to_achieve/wia_v1/furnished_v1'
INV=ARTIFACT_ROOT / 'wia_v1_20261024_furnished/asset_inventory'
ROWS=LazyMapping(lambda: {r['document_task_id']:r for r in csv.DictReader((ROOT/'docs/research/world_interactive_agency_v1/task_recheck_61.csv').open())})

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')

def quat(deg):return [0.,0.,math.sin(math.radians(deg)/2),math.cos(math.radians(deg)/2)]

def inventory(scene):return {r['name']:r for r in read_json_input(INV/(scene+'.json'))['objects']}

def background(scene,room,floor):
    identity=f'Specular Reflection/Correspondence/{scene}/{room}/wia_furnished_background_v1.json'
    source=ROOT/'explore_to_achieve/sources/json_clean'/identity
    if not source.exists():
        parent=ROOT/'explore_to_achieve/sources/json_clean/Specular Reflection/Correspondence/hall_arch_wood/empty_room_0/wia_background_v1.json'
        d=json.loads(parent.read_text());d.update(id='wia_background_'+scene+'_'+room,scene=scene,room=room)
        m=json.loads(d['metadata_json']) if isinstance(d['metadata_json'],str) else d['metadata_json']
        m.update(scene=scene,room=room,floor_name=floor)
        d['metadata_json']=json.dumps(m)
        d['wia_provenance']={'role':'background_loader_only','parent':str(parent),'original_optical_answer_not_used':True,
                             'layout_design':'Main-authored furnished scene; geometry and robot path pending native review.'}
        dump(source,d)
    return identity

ROOMS={
 'kitchen':{'scene':'Beechwood_0_garden','room':'kitchen_0','floor':'floors_yrwwan_0','tray_support':'countertop_tpuwys_3',
            'tray_xy':[-1.62,13.86],'tray_size':[1.4,.65], 'robot':[-1.6,14.98,.02],'yaw':-90,
            'slots':[('countertop_tpuwys_0',[-.9,15.85]),('countertop_tpuwys_1',[-3.73,13.55]),('countertop_tpuwys_2',[-2.2,11.96])]},
 'utility':{'scene':'house_single_floor','room':'utility_room_0','floor':None,'tray_support':'countertop_ikwqer_0',
            'tray_xy':[24.51,.15],'tray_size':[.5,1.4], 'robot':[23.25,.15,.02],'yaw':0,
            'slots':[('countertop_fjkase_0',[21.94,.85]),('countertop_sjxber_0',[23.65,-2.05]),('countertop_fjkase_0',[21.94,-.25])]},
 'playroom':{'scene':'Merom_0_garden','room':'playroom_0','floor':None,'tray_support':'countertop_tpuwys_0',
             'tray_xy':[-5.82,-11.45],'tray_size':[.45,1.3], 'robot':[-4.8,-11.45,.02],'yaw':180,
             'slots':[('bookcase_owvfik_0',[-3.02,-10.61]),('countertop_tpuwys_0',[-5.82,-12.72]),('countertop_tpuwys_0',[-5.82,-10.6])]}}

def spatial():
    entries=[]
    groups={'T26':'kitchen','T27':'utility','T28':'kitchen','T29':'playroom','T30':'utility','T31':'kitchen','T32':'playroom','T33':'utility','T34':'utility'}
    for alias,room_key in groups.items():
        c=deepcopy(ROOMS[room_key]);inv=inventory(c['scene'])
        if c['floor'] is None:c['floor']=next(r['name'] for r in inv.values() if r['category']=='floors' and c['room'] in r['rooms'])
        parent=ROWS[alias];task=yaml.safe_load(Path(parent['task_file']).read_text());old=task['private']['evaluation']['parameters']
        targets={k:v for k,v in task['private']['bindings'].items() if v.get('category')}
        src=json.loads(resolve_source(task['source']['json']).read_text());meta=src.get('metadata_json',{});meta=json.loads(meta) if isinstance(meta,str) else meta
        saved=meta.get('objects',[]);saved={x.get('_key',x.get('name')):x for x in saved} if isinstance(saved,list) else saved
        objects=[];bindings={};supported=[];initial_supports={};names=[]
        for i,(key,binding) in enumerate(targets.items()):
            sp,xy=c['slots'][i];data=saved.get(binding['name'],{});scale=data.get('scale',1.);scale=[scale]*3 if isinstance(scale,(int,float)) else scale
            orientation=data.get('quaternion_xyzw',quat(0))
            obj={'type':'DatasetObject',**binding,'position':[*xy,inv[sp]['aabb'][1][2]+.3],
                 'orientation':orientation,'scale':scale,'fixed_base':False,'visual_only':False}
            objects.append(obj);bindings[key]=deepcopy(binding);bindings['initial_support_'+str(i)]={'name':sp}
            initial_supports[key]='initial_support_'+str(i);supported.append({'name':binding['name'],'support':sp})
            names.append(binding['category'].replace('_',' '))
        table=c['tray_support'];table_top=inv[table]['aabb'][1][2];cx,cy=c['tray_xy'];sx,sy=c['tray_size'];deck_top=table_top+.02
        def cube(key,size,pos):
            name='wia_tray_'+key
            objects.append({'type':'PrimitiveObject','primitive_type':'Cube','name':name,'category':'tray','size':1.,'scale':size,'position':pos,'orientation':quat(0),'fixed_base':True,'visual_only':False,'rgba':[.28,.44,.55,1.]})
            bindings[key]={'name':name}
        cube('deck',[sx,sy,.02],[cx,cy,table_top+.01])
        for key,pos,size in [('rim_w',[cx-sx/2+.008,cy,deck_top+.018],[.016,sy,.036]),('rim_e',[cx+sx/2-.008,cy,deck_top+.018],[.016,sy,.036]),('rim_s',[cx,cy-sy/2+.008,deck_top+.018],[sx,.016,.036]),('rim_n',[cx,cy+sy/2-.008,deck_top+.018],[sx,.016,.036])]:cube(key,size,pos)
        bindings['tray_table']={'name':table}
        goal={'kind':'shape','shape':old['target_shape']} if alias in ['T26','T27','T28','T29','T30'] else {'kind':'contact','pair':list(targets),'relation':old['desired_relation']}
        if goal['kind']=='contact':
            print(alias,'contact parameters', {k:v for k,v in old.items() if 'relation' in k})
        relation=goal.get('shape',goal.get('relation')).replace('collinear','straight line')
        scoring={'minimum_lift_m':.05,'minimum_displacement_m':.05,'stability_duration_s':.5,'maximum_linear_speed_m_s':.02,'maximum_angular_speed_rad_s':.1,'max_sample_gap_s':.1,'fixture_position_tolerance_m':.002,'fixture_angle_tolerance_rad':.01,'deck_height_tolerance_m':.015,'maximum_bottom_gap_m':.015,'penetration_tolerance_m':.001,'minimum_separation_m':.05}
        p={'revision':'wia_spatial_tray_v1','scene_contract':{'scene_model':c['scene'],'room':c['room']},'targets':{k:v['name'] for k,v in targets.items()},'initial_supports':initial_supports,
           'tray':{'deck_key':'deck','table_support_key':'tray_table','rim_keys':['rim_w','rim_e','rim_s','rim_n'],'mounting_mode':'fixed_fixture_on_native_table','usable_world_xy_m':[[cx-sx/2+.02,cy-sy/2+.02],[cx+sx/2-.02,cy+sy/2-.02]],'deck_top_world_z_m':deck_top},'goal_relation':goal,'scoring':scoring}
        layout={'shared_layout_id':'wia_furnished_'+alias,'scene_model':c['scene'],'rooms':[c['room']],'floor_name':c['floor'],'objects':objects,'supported_items':supported,'robot_start':{'position':c['robot'],'orientation':quat(c['yaw'])}}
        objective='Find the '+', '.join(names)+' on open furniture surfaces. Bring them to the blue-grey tray on the counter. '
        objective+=('Arrange their centers into '+relation+'.' if goal['kind']=='shape' else 'Leave the two objects '+('touching each other.' if relation=='touching' else 'separated by at least 5 cm.'))
        public={'objective':objective,'environment':'A furnished '+c['room'].split('_0')[0].replace('_',' ')+'. Target objects are on exposed surfaces and are not deliberately hidden.',
          'success_criteria':'Retrieve each target with a genuine grasp and at least 5 cm lift, then release all targets entirely on the tray deck in the requested relation. Leave hands clear and objects stable for at least 0.5 seconds.',
          'constraints':['Use lawful robot controls and mounted observations.','Keep the tray and its supporting furniture fixed.','The tray is a fixed destination on furniture, not a hand-held collection tray.']}
        entries.append({'document_alias':alias,'variant_id':alias.lower()+'_furnished_tray_v1','label':alias+' open-surface retrieval and tray arrangement','parent_task_file':parent['task_file'],'parent_scene_file':parent['scene_file'],'source_identity':background(c['scene'],c['room'],c['floor']),'layout':layout,'bindings':bindings,'parameters':p,'public':public})
    dump(OUT/'spatial_design_v1.json',{'schema':'wia_spatial_tray_design_v1','entries':entries})

if __name__=='__main__':spatial()
