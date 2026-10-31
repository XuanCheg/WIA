"""Build the nine unreleased paired scenes. Run: python -m wia.candidates.build.

Only candidate definitions and their authored ESI transport sources are written.
Native physics and sensors remain owned by the existing WIA runtime.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import yaml
from wia.resources import ROOT
from wia.tasks.sources import resolve_source

MODULE = 'wia.authoring.converters.interactiveworld_candidates'
DEST = ROOT / 'explore_to_achieve/candidates/interactiveworld_20261029'
IW = ROOT.parent / 'InteractiveWorld'


def object_spec(name, category=None, model=None, *, slot=0, offset=(0, 0), size=None, fixed=False, **extra):
    return dict(name=name, category=category, model=model, slot=slot, offset=list(offset),
                size=size, fixed=fixed, **extra)


def definitions():
    """Main-authored scientific decisions; slot coordinates are resolved natively."""
    rows = []
    def add(pid, source, room, fixture, variants, objects, regions, contract, objective, **extra):
        pack = next((IW/'benchmark/tasks/taskpacks').rglob(source+'.taskpack.json'))
        raw = json.loads(pack.read_text())
        rows.append(dict(prototype=pid, source_task=source, source_file=str(pack),
            source_sha256=hashlib.sha256(pack.read_bytes()).hexdigest(), scene_model=raw['executable_spec']['world']['scene'],
            room=room, fixture=fixture, variants=variants, objects=objects, regions=regions,
            contract=contract, objective=objective, **extra))
    # Regions are distinct tabletop sections; placements are near the accessible edge.
    add('H1','GUIDE-87017','kitchen_0','commercial_kitchen_table_bwlfnu_0',
        [{'defective':False},{'defective':True}],
        [object_spec('item_01',size=[.09,.065,.05],offset=(-.15,0),package=True),
         object_spec('item_02',size=[.09,.065,.05],offset=(.15,0),package=True)],
        {'delivery':1,'quarantine':2},
        {'selected':'item_01','spares':['item_02'],'compatible':['item_02'],'delivery':'delivery','quarantine':'quarantine','protected':[]},
        'Inspect the underside of the left package. A green acceptance mark permits delivery; a red mark requires quarantine and replacement from the spare supply. Deliver one acceptable package to the blue receiving area. Keep an acceptable original and preserve the spare; quarantine only a rejected original.')
    add('H2','ORCHESTRATE-112022','kitchen_0','commercial_kitchen_table_bwlfnu_0',
        [{'disturbance':False},{'disturbance':True}],
        [object_spec('item_01','apple','agveuv',slot=1,size=[.075,.075,.075]),
         object_spec('item_02',size=[.09,.065,.06])],
        {'work':0,'delivery':2,'storage':3},
        {'payload':'item_01','obstruction':'item_02','work_region':'work','delivery':'delivery','storage':'storage','protected':[]},
        'Clear the outlined work area by storing its obstruction in the yellow storage area, deliver the apple to the blue receiving area, and leave the work area clear. The work area may be disturbed after delivery; inspect its final condition and repair only if necessary.')
    add('H3','ASK-99600','kitchen_0','commercial_kitchen_table_nisnrl_0',
        [{'initial_stock':1,'progress':'remaining'},{'initial_stock':2,'progress':'retained'},{'initial_stock':2,'progress':'undone'}],
        [object_spec('item_01','mug','dhnxww',offset=(-.13,0),size=[.085,.085,.09]),
         object_spec('item_02','mug','dhnxww',slot=1,offset=(-.13,0),size=[.085,.085,.09]),
         object_spec('item_03','mug','dhnxww',slot=1,offset=(.13,0),size=[.085,.085,.09])],
        {'stock':0,'supply':1},
        {'stock':['item_01','item_02','item_03'],'target_count':2,'semantics':'minimum','stock_region':'stock','supply_region':'supply','protected':[]},
        'Leave at least two mugs in the blue stocked area. Inspect the current inventory and add only the remaining deficit from the yellow supply area. Preserve mugs already stocked. If the target is met, leave all mugs in place. Past placements do not establish current stock.')
    add('H4','NARRATE-12802','chemistry_lab_0','desk_slgrxc_0',
        [{'blocker_layout':0},{'blocker_layout':1}],
        [object_spec('item_01','digital_scale','xwzzen',slot=1,size=[.14,.075,.055]),
         *[object_spec(f'item_0{i+2}','bottle_of_ginger_beer','zkocwb',offset=(x,0),size=[.05,.05,.13]) for i,x in enumerate((-.14,0,.14))],
         object_spec('anchor_01','saucepot','obuxbe',offset=(-.20,.22),size=[.13,.12,.09],fixed=True),
         object_spec('anchor_02','mixing_bowl','bsgybx',offset=(.20,.22),size=[.13,.12,.09],fixed=True)],
        {'target':0,'storage':2},
        {'target':'item_01','blockers':['item_02','item_03','item_04'],'anchors':['anchor_01','anchor_02'],'target_region':'target','storage_region':'storage','protected':['anchor_01','anchor_02']},
        'Place the scale fully within the blue outlined target footprint. Minimize the number of relocated bottles while making placement and gripper access possible; put every relocated bottle in the yellow storage area. Preserve both fixed anchors and stop clearing when sufficient space exists. Withdraw both hands after stable release. Among valid placements, fewer relocated bottles is better.')
    add('H5','ASK-30000','grocery_store_0','breakfast_table_ylykeh_0',
        [{'primary_available':True},{'primary_available':False}],
        [object_spec('item_01','can_of_cat_food','omiuox',slot=0,size=[.065,.065,.06]),
         object_spec('item_02','can_of_dog_food','rgwfxq',slot=3,size=[.065,.065,.08])],
        {'primary':1,'fallback':2},
        {'payload':'item_01','resident':'item_02','primary':'primary','fallback':'fallback','protected':['item_02']},
        'Deliver the cat-food can to the blue primary receiving area if it contains no resident item. If any resident occupies that area, use the yellow fallback area. This is a stock-separation rule, not a claim that the primary is geometrically full. Preserve the resident item.')
    add('M1','WATCH-28002','bedroom_0','desk_ohjotf_0',
        [{'used_tool':'item_01'},{'used_tool':'item_02'}],
        [object_spec('item_01','sponge','aewrov',offset=(-.07,0),size=[.08,.055,.035]),
         object_spec('item_02','sponge','aewrov',offset=(.07,0),size=[.08,.055,.035])],
        {'used':1,'clean':2,'cleaning_surface':0},
        {'tools':['item_01','item_02'],'used_region':'used','clean_region':'clean','protected':[]},
        'Observe which of the two matching cleaning sponges is used on the marked cleaning surface. Place that sponge in the yellow used-tool area and the unused sponge in the blue clean-tool area. Track identity through the visible use event. This is a use-history rule, not a microbial-contamination simulation.')
    add('M2','GUIDE-24003','grocery_store_0','grocery_shelf_akinbg_1',
        [{'orders':{'order_a':4,'order_b':3}},{'orders':{'order_a':5,'order_b':3}}],
        [object_spec(f'item_0{i+1}','pineapple_juice_carton','barzwx',offset=(x,y),size=[.06,.06,.11],quantity=q)
         for i,(x,y,q) in enumerate([(-.15,0,3),(-.05,0,2),(.05,0,2),(.15,0,1)])],
        {'stock':0,'order_a':1,'order_b':2},
        {'units':{'item_01':3,'item_02':2,'item_03':2,'item_04':1},'stock_region':'stock','protected':[]},
        'Allocate the sealed cartons using the visible QTY number on each package as its stock quantity. Fulfil both receiving orders exactly without opening cartons or counting a carton twice. Preserve unused stock. Quantities are operational stock units, not physical package volume.', quantity_labels=True)
    add('M3','GUIDE-14018','kitchen_0','commercial_kitchen_table_nisnrl_0',
        [{'eligible_unit':'unit_a'},{'eligible_unit':'unit_b'}],
        [object_spec('item_01','swiss_cheese','hwxeto',size=[.09,.065,.06]),
         object_spec('item_02','carton_of_milk','xmugpm',slot=1,size=[.065,.065,.12]),
         object_spec('item_03','chicken','nppsmz',slot=2,size=[.13,.09,.09]),
         object_spec('unit_a','fridge','dszchb',slot=1,size=[.65,.65,1.4],fixed=True,container=True),
         object_spec('unit_b','fridge','dszchb',slot=2,size=[.65,.65,1.4],fixed=True,container=True)],
        {'unit_a':1,'unit_b':2},
        {'payload':'item_01','content_by_unit':{'unit_a':'item_02','unit_b':'item_03'},'eligible_content':['item_02'],'protected':['item_02','item_03'],'require_closed':True},
        'Store the cheese in the refrigerator whose observed contents indicate dairy storage. The other refrigerator holds chicken. Use any sufficient inspection plan, preserve the resident contents, and close both refrigerator doors. No temperature or food-safety outcome is claimed.')
    add('M4','ORCHESTRATE-44001','kitchen_0','commercial_kitchen_table_vxvtec_0',
        [{'duration_c':8.},{'duration_c':24.}],
        [object_spec('item_01','potato','bpwohr',offset=(-.15,0),size=[.075,.06,.05]),
         object_spec('item_02','pear','vjqkwh',offset=(0,0),size=[.065,.065,.09]),
         object_spec('item_03','apple','agveuv',offset=(.15,0),size=[.075,.075,.075])],
        {'station_a':1,'station_b':2,'delivery':3},
        {'jobs':{'item_01':{'station':'station_a','duration_s':12.},'item_02':{'station':'station_a','duration_s':4.},'item_03':{'station':'station_b','duration_s':8.}},'delivery_region':'delivery','deadline_s':120.,'handling_s':2.,'protected':[]},
        'Process and deliver all three jobs as early as possible using two exclusive stations and one robot. Elapsed simulation time is the scheduling cost. Potato and pear use station A; apple uses station B. A job must remain released and supported in its station for its stated duration before delivery to the blue area. Early removal resets processing. Use simulation time; model waits pause the world. Processing is a timed state machine, not cooking.', timed_process=True)
    return rows


def build():
    DEST.mkdir(parents=True, exist_ok=True)
    template_task=yaml.safe_load((ROOT/'explore_to_achieve/wia_v1/release/tasks/t01.yaml').read_text())
    template=json.loads(resolve_source(template_task['source']['json']).read_text())
    catalogue=[]
    for definition in definitions():
        for index,condition in enumerate(definition.pop('variants')):
            spec=deepcopy(definition); spec['condition']=condition; pid=spec['prototype']; identity=f'iw_{pid.lower()}_{index+1:02d}'
            spec['id']=identity; spec['contract'].update(condition)
            if pid=='H3' and condition['progress']=='undone':spec['contract']['requires_revision_event']=True
            if pid=='H3' and condition['initial_stock']==2: spec['objects'][1]['slot']=0;spec['objects'][1]['offset']=[.13,0]
            if pid=='H4':
                layout=condition['blocker_layout']
                for i in range(3):spec['objects'][i+1]['offset']=[[-.20,.20,.28][i] if layout==0 else [-.20,0,.28][i],0]
            if pid=='H5' and not condition['primary_available']:spec['objects'][1]['slot']=1
            if pid=='M3' and condition['eligible_unit']=='unit_b':
                spec['objects'][1]['slot']=2;spec['objects'][2]['slot']=1
                spec['contract']['content_by_unit']={'unit_a':'item_03','unit_b':'item_02'}
            if pid=='M4':spec['contract']['jobs']['item_03']['duration_s']=condition['duration_c']
            source_identity=f'Action Sequencing/Action Order Inference/{spec["scene_model"]}/{spec["room"]}/{identity}.json'
            raw=deepcopy(template);raw.update(scene=spec['scene_model'],room=spec['room'],objects={},initial_poses={},_question=spec['objective'],candidate_provenance={k:spec[k] for k in ('source_task','source_file','source_sha256')})
            source=ROOT/'explore_to_achieve/sources/json_clean'/source_identity;source.parent.mkdir(parents=True,exist_ok=True);source.write_text(json.dumps(raw,indent=2)+'\n')
            patches=[dict(operation='category_initialization',phase=phase,module=MODULE,parameters=spec) for phase in ('before_construction','after_postprocessing')]
            patches.append(dict(operation='restore_robot_head_mount',phase='after_postprocessing'))
            scene=dict(policy='native_then_minimal_patch_else_skip',variants={'original':{'patches':[]},'adapted':{'patches':patches}})
            bindings={o['name']:{'name':o['name']} for o in spec['objects']}
            objective=spec['objective']
            if pid=='M2':objective+=f' Orders: A={condition["orders"]["order_a"]}, B={condition["orders"]["order_b"]} units.'
            if pid=='M4':objective+=f' Durations: potato=12 s, pear=4 s, apple={condition["duration_c"]} s. Deadline=120 simulation seconds. The 2-second abstract handling cost is a planning lower bound, not a measured controller time.'
            task=dict(kind='category_robot_task',id=identity,version=1,status='pending',source={'provider':'esi','json':source_identity},
                public={'objective':objective,'success_criteria':'Actual stable released placement, meaningful process evidence and preservation are required. Scene screenshots alone do not certify task completion.','environment':f'Derived {spec["scene_model"]} {spec["room"]} with source-context inventory.','constraints':['Use only public robot sensors and legal physical controls.','Preserve resident contents and protected furniture.']},
                requirements={'capabilities':['base_translation','base_rotation','ee_translation','gripper','ee_rotation','trunk_height']},private={'bindings':bindings,'evaluation':{'type':'category_operation','module':MODULE,'operation':pid,'binding_keys':list(bindings),'parameters':spec}},scene=scene,qualification={'on_failure':'skip','target_movable_collidable':True,'initial_visibility':'none'},defaults={'seed':7,'max_steps':120,'wall_timeout_s':5400.,'world_advances_during_inference':False,'observation_condition':'current','action_mode':'parameterized'})
            sd=dict(kind='scene_patch',id=identity,version=1,status='pending',source=task['source'],bindings=bindings,scene=scene,seed=7,runtime={'preview_fps':0.,'camera_size':[512,512]})
            for group,value in [('tasks',task),('scenes',sd)]:
                path=DEST/group/(identity+'.yaml');path.parent.mkdir(exist_ok=True);path.write_text(yaml.safe_dump(value,sort_keys=False,width=100))
            catalogue.append(dict(id=identity,prototype=pid,condition=condition,task_file=str((DEST/'tasks'/(identity+'.yaml')).relative_to(ROOT)),scene_file=str((DEST/'scenes'/(identity+'.yaml')).relative_to(ROOT)),variant='adapted',source_task=spec['source_task']))
    (DEST/'catalogue.json').write_text(json.dumps({'schema':'wia_interactiveworld_candidates_v1','release_admitted':False,'tasks':catalogue},indent=2)+'\n')
    print(f'Built {len(catalogue)} candidate variants across nine prototypes.')

if __name__=='__main__':build()
