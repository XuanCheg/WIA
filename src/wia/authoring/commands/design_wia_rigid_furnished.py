"""Author furnished T24/T25 versions with unchanged insertion semantics.

Run: behavior/bin/python -B scripts/design_wia_rigid_furnished.py
Only definitions and loader checks are produced; no manipulation is executed.


Canonical invocation: python -m wia.authoring.commands.design_wia_rigid_furnished
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import yaml

from wia.resources import ROOT
from wia.tasks.loader import load_task, load_scene
from wia.tasks.sources import resolve_source
from wia.authoring.commands.design_wia_furnished_round import OUT, inventory, quat, dump


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    entries=[]
    specs={
        'T24':('rigid_t24_support_v6/task.yaml','rigid_t24_support_v6/scene.yaml',
               'house_single_floor','utility_room_0','countertop_ikwqer_0',
               [[24.51,-.9],[24.51,.1],[24.51,1.1]],'countertop_fjkase_0',[21.94,.8],
               [23.25,.15,.02],0),
        'T25':('tasks/rigid_t25_v1.yaml','scenes/rigid_t25_v1.yaml',
               'Merom_0_garden','playroom_0','countertop_tpuwys_0',
               [[-5.82,-11.8],[-5.82,-11.15],[-5.82,-10.6]],'bookcase_owvfik_0',[-3.02,-10.61],
               [-4.8,-11.45,.02],180),
    }
    for alias,(tf,sf,scene_model,room,table,slots,pickup,xy,robot,yaw) in specs.items():
        parent_task=ROOT/'explore_to_achieve/wia_v1'/tf
        parent_scene=ROOT/'explore_to_achieve/wia_v1'/sf
        task=yaml.safe_load(parent_task.read_text());scene=yaml.safe_load(parent_scene.read_text())
        evaluation=deepcopy(task['private']['evaluation'])
        original_source=resolve_source(task['source']['json'])
        source=json.loads(original_source.read_text())
        encoded=isinstance(source['metadata_json'],str)
        meta=json.loads(source['metadata_json']) if encoded else deepcopy(source['metadata_json'])
        inv=inventory(scene_model)
        floor=next(row['name'] for row in inv.values() if row['category']=='floors' and room in row['rooms'])
        variant=alias.lower()+'_furnished_native_support_v1'
        identity=f'Physical Structure/Rigid Containment/{scene_model}/{room}/{variant}.json'
        source_path=ROOT/'explore_to_achieve/sources/json_clean'/identity
        source.update(scene=scene_model,room=room,id=variant)
        meta.update(scene=scene_model,room=room,floor_name=floor)
        source['metadata_json']=json.dumps(meta) if encoded else meta
        source['wia_furnished_provenance']={
            'original_qa_source_file':str(original_source),'original_qa_source_sha256':sha(original_source),
            'original_question_and_answer_preserved':True,
            'role':'Native background bootstrap and original QA provenance; explicit initialization replaces source placements.'}
        raw=(json.dumps(source,indent=2)+'\n').encode()
        if source_path.exists() and source_path.read_bytes()!=raw:
            raise ValueError('Refusing to overwrite existing source '+str(source_path))
        source_path.parent.mkdir(parents=True,exist_ok=True);source_path.write_bytes(raw)
        bindings=task['private']['bindings']
        bindings['table']={'name':table,'category':inv[table]['category'],'model':inv[table]['model']}
        bindings['pickup_support']={'name':pickup,'category':inv[pickup]['category'],'model':inv[pickup]['model']}
        originals={'target':meta['containee'],**{'container_'+str(r['idx']):r for r in meta['containers']}}
        old_poses={p['binding']:p for p in scene['scene']['variants']['tabletop']['patches'] if p['operation']=='object_pose'}
        objects=[];supported=[]
        for key in ['target','container_0','container_1','container_2']:
            item=originals[key];support=pickup if key=='target' else table
            position=xy if key=='target' else slots[int(key[-1])]
            binding=bindings[key]
            objects.append({'type':'DatasetObject',**binding,'position':[*position,inv[support]['aabb'][1][2]+.3],
                'orientation':deepcopy(old_poses[key]['quaternion_xyzw']), 'scale':deepcopy(item['scale']),
                'fixed_base':False,'visual_only':False})
            row={'name':binding['name'],'support':support}
            if support=='bookcase_owvfik_0':
                row.update(support_alignment_mode='local_raycast',support_surface_mode='local_raycast')
            supported.append(row)
        layout={'shared_layout_id':variant,'scene_model':scene_model,'rooms':[room],'floor_name':floor,
                'objects':objects,'supported_items':supported,'robot_start':{'position':robot,'orientation':quat(yaw)}}
        parameters={'revision':'wia_furnished_rigid_layout_v1','layout':layout}
        ground={'revision':'wia_background_v1','mode':'complete_ground','expected_scene':scene_model,'disable_grid':False}
        patches=[]
        for phase in ['before_construction','after_postprocessing']:
            for module,p in [('active_layout',parameters),('wia_background',ground)]:
                patches.append({'operation':'category_initialization','phase':phase,'module':'core.esi_converters.'+module,'parameters':deepcopy(p)})
        task['scene']={'policy':'native_then_minimal_patch_else_skip','variants':{'original':{'patches':[]},'tabletop':{'patches':patches}}}
        task['source']={'provider':'esi','json':identity}
        task.update(id=task['id']+'__'+variant,version=task['version']+1,status='pending')
        task['public']['objective']='Find the target on the exposed furniture surface and inspect the containers distributed along the other counter. '+task['public']['objective']
        task['public']['environment']='A furnished '+room.replace('_0','').replace('_',' ')+'. The target and containers occupy separate native furniture surfaces; none is deliberately hidden.'
        task['repair'].update(contract_revision='wia_furnished_rigid_layout_v1',parent_task_file=str(parent_task),
            parent_task_sha256=sha(parent_task),parent_scene_file=str(parent_scene),parent_scene_sha256=sha(parent_scene),
            original_qa_source_file=str(original_source),original_qa_source_sha256=sha(original_source),
            native_validation_run=False,physical_success_verified=False,full_reference_execution='deferred_until_user_review',
            original_support_v6_hold_preserved=alias=='T24')
        task['private']['evaluation']['binding_keys']=list(bindings)
        if task['private']['evaluation']!=evaluation:
            raise ValueError('Insertion evaluator must remain unchanged.')
        scene.update(id=scene['id']+'__'+variant,version=scene['version']+1,status='pending',source=deepcopy(task['source']),
                     scene=deepcopy(task['scene']),bindings=deepcopy(bindings),target=None,target_bindings=[],qualification=None)
        paths={kind:OUT/'rigid'/variant/(kind+'.yaml') for kind in ['task','scene']}
        for kind,value in [('task',task),('scene',scene)]:
            path=paths[kind];data=yaml.safe_dump(value,sort_keys=False).encode()
            if path.exists() and path.read_bytes()!=data:raise ValueError('Refusing differing output '+str(path))
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        load_task(paths['task'],variant='tabletop',run_mode='diagnostic')
        load_scene(paths['scene'],variant='tabletop',run_mode='diagnostic')
        entries.append({'document_alias':alias,'variant_id':variant,'label':alias+' exposed native furniture retrieval and containment',
            'parent_task_file':str(parent_task),'parent_scene_file':str(parent_scene),
            'task_file':str(paths['task']),'scene_file':str(paths['scene']),'variant':'tabletop','seed':task['defaults']['seed'],
            'source_file':str(source_path),'source_identity':identity,'source_sha256':sha(source_path),
            'task_sha256':sha(paths['task']),'scene_sha256':sha(paths['scene']),
            'native_validation_run':False,'physical_success_verified':False,'retained_native_supports':[table,pickup]})
    dump(OUT/'rigid/manifest.json',{'schema':'wia_variant_manifest_v1','entries':entries,
        'retained_without_revision':['T22 furnished separated native pickup','T23 supported-spawn V5'],
        'reference_execution_status':'deferred_until_user_review'})
    print('Furnished rigid definitions: 2; insertion scoring and asset scales unchanged.')


if __name__=='__main__':main()
