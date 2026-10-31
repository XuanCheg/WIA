"""Native initialization for unreleased candidates.

Run: python -m wia.candidates.build, then submit the candidate capture catalogue.
No robot command or model call is issued by this module. Native initialization
placements are distinct from deferred episode actions and controlled events.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import numpy as np
from wia.resources import ROOT, resolve_resource_path


def arr(value):
    return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)


def configure_scene(config, p):
    from wia.authoring.converters import active_layout
    scene_path=resolve_resource_path('behavior://behavior-1k-assets/scenes/' + p['scene_model']
                                     + '/json/' + p['scene_model'] + '_best.json')
    native=json.loads(scene_path.read_text())
    pos=native['state']['registry']['object_registry'][p['fixture']]['root_link']['pos']
    objects=[]
    for i,o in enumerate(p['objects']):
        item=dict(name=o['name'],position=[pos[0]+i*.3,pos[1],5.],orientation=[0.,0.,0.,1.],fixed_base=o['fixed'],visual_only=False)
        if o['category']:
            item.update(type='DatasetObject',category=o['category'],model=o['model'],bounding_box=o['size'])
            if o['category']=='chicken':
                item.pop('bounding_box');item['scale']=[.5,.5,.5]
        else:
            item.update(type='PrimitiveObject',category='object',primitive_type='Cube',size=1.,scale=o['size'],rgba=[.75,.73,.65,1.])
        objects.append(item)
    for region in p['regions']:
        objects.append(dict(type='PrimitiveObject',name='zone_'+region,category='object',primitive_type='Cube',size=1.,scale=[.3,.24,.001],fixed_base=True,visual_only=True,rgba=[.2,.45,.9,.35],position=[pos[0],pos[1],5.],orientation=[0.,0.,0.,1.]))
    layout={'scene_model':p['scene_model'],'rooms':[p['room']],'objects':objects,
            'robot_start':{'position':[pos[0],pos[1]-1.3,.1],'orientation':[0.,0.,.70710678,.70710678]}}
    receipt=active_layout.configure_scene(config,{'layout':layout,'revision':'interactiveworld_candidate_v1'})
    config['scene']['load_object_categories']=None
    exclusions=[]
    if p['scene_model'] in ('restaurant_asian','restaurant_diner'):
        exclusions.append('bottom_cabinet_pkdnbu_0')
    if p['prototype']=='M1':
        exclusions.append('armchair_ybcnmu_0')
    if exclusions:
        from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
        if 'WiaCandidateBackground' not in globals():
            class WiaCandidateBackground(InteractiveTraversableScene):
                def _should_load_object(self,obj_info,task_metadata):
                    excluded={'bottom_cabinet_pkdnbu_0'} if self.scene_model in ('restaurant_asian','restaurant_diner') else set()
                    if self.scene_model=='hotel_suite_large':excluded.add('armchair_ybcnmu_0')
                    return obj_info['args']['name'] not in excluded and super()._should_load_object(obj_info,task_metadata)
            globals()['WiaCandidateBackground']=WiaCandidateBackground
        config['scene']['type']='WiaCandidateBackground'
        receipt['excluded_background_instances']=exclusions
        receipt['exclusion_reason']='Malformed cabinet emitter transform or human armchair occupying the robot standing area; no task object excluded.'
    return receipt


def postprocess_joint_world(session, p):
    session.room_name=p['room']
    return {'joint_world_postprocess':True,'source_background_preserved':True,'robot_execution_verified':False}


def _set(obj,position,orientation=None):
    import torch
    if orientation is None: orientation=arr(obj.get_position_orientation()[1]).tolist()
    obj.set_position_orientation(position=torch.tensor(position,dtype=torch.float32),orientation=torch.tensor(orientation,dtype=torch.float32))
    obj.keep_still()


def _all(session):
    return list(session.env.scene.objects)


def initialize_scene(session, p):
    import omnigibson as og
    import torch
    from omnigibson.object_states import Touching
    scene=session.env.scene
    lookup=lambda name:scene.object_registry('name',name)
    table=lookup(p['fixture'])
    if table is None:raise ValueError('Source fixture missing: '+p['fixture'])
    preferred=table
    furniture=[preferred]+sorted([o for o in _all(session) if o.category==preferred.category and o.name!=preferred.name],key=lambda o:float(np.linalg.norm(arr(o.aabb_center)[:2]-arr(preferred.aabb_center)[:2])))
    proposals=[];chosen=None
    for table in furniture:
        low,high=[arr(v).astype(float) for v in table.aabb];center=(low+high)/2
        floors=[o for o in _all(session) if o.category in ('floors','floor') and np.all(arr(o.aabb[0])[:2]<=center[:2]) and np.all(arr(o.aabb[1])[:2]>=center[:2])]
        floor_z=min((float(arr(o.aabb[1])[2]) for o in floors),key=lambda z:abs(z)) if floors else 0.
        long_axis=int(np.argmax(high[:2]-low[:2]))
        # Shelf end panels are not an accessible stock face. H4/M2 also use
        # a larger start envelope after measured native base overlap failures.
        repaired_stance=p['prototype'] in ('H4','M2')
        axes=(1-long_axis,) if p['prototype']=='M2' else (1-long_axis,long_axis)
        start_half=.45 if repaired_stance else .35
        stand_off=.70 if repaired_stance else .55
        for axis in axes:
            for sign in (-1,1):
                for offset in (0.,-.35,.35,-.65,.65):
                    xy=center[:2].copy();xy[axis]=(low if sign<0 else high)[axis]+sign*stand_off;xy[1-axis]+=offset
                    blocking=[]
                    for o in _all(session):
                        if o is session.robot or o.name.startswith(('item_','anchor_','zone_','unit_')) or o.category in ('floors','floor','ceilings'):continue
                        boxes=[mesh.aabb for link in o.links.values() for mesh in link.collision_meshes.values()]
                        for box in boxes:
                            a,b=[arr(v) for v in box]
                            if b[2]<floor_z+.06 or a[2]>floor_z+.65:continue
                            if np.all(xy+start_half>=a[:2]) and np.all(xy-start_half<=b[:2]):blocking.append(o.name);break
                    proposals.append({'fixture':table.name,'xy':xy.tolist(),'blocking':blocking})
                    if not blocking:chosen=(table,xy,axis,sign,low,high,center,floor_z);break
                if chosen:break
            if chosen:break
        if chosen:break
    if chosen is None:
        out=ROOT/'outputs/interactiveworld_20261029/native'/p['id'];out.mkdir(parents=True,exist_ok=True)
        (out/'stance_failure.json').write_text(json.dumps(proposals,indent=2)+'\n')
        raise ValueError('No collision-screened initial stance beside source-category fixtures')
    table,xy,axis,sign,low,high,center,floor_z=chosen
    others=[o for o in furniture if o is not table]
    supports=[table] if max(high[:2]-low[:2])>=1.6 else [table,*others[:3]]
    slots=[];slot_count=3 if p['prototype']=='M1' else 4
    for i in range(slot_count):
        support=supports[i] if i<len(supports) else table
        slo,shi=[arr(v).astype(float) for v in support.aabb]
        c=(slo+shi)/2;along=int(np.argmax(shi[:2]-slo[:2]));cross=1-along
        c[cross]=slo[cross]+min(.22,(shi[cross]-slo[cross])/2)
        if support is table:c[axis]=(low if sign<0 else high)[axis]-sign*(.11 if p['prototype']=='M2' else .18)
        partition_axis=1-axis if p['prototype'] in ('H5','M1') and support is table else along
        if len(supports)==1:c[partition_axis]=slo[partition_axis]+(i+.5)*(shi[partition_axis]-slo[partition_axis])/slot_count
        if p['prototype']=='M1':
            # The native desktop has rounded front corners: its full AABB
            # width is not a rectangular usable surface at the front edge.
            c[axis]=(low if sign<0 else high)[axis]-sign*(.18 if i==0 else .40)
            c[1-axis]=center[1-axis]+(0. if i==0 else -.20 if i==1 else .20)
        from wia.authoring.converters.active_layout import _support_ray_surface
        c[2]=_support_ray_surface(support,support,[float(c[0]),float(c[1]),float(shi[2]+.10)],[float(c[0]),float(c[1]),float(slo[2]+.10)],'candidate_table_surface')['surface_z_m'];slots.append({'support':support.name,'center':c.tolist(),'aabb':[slo.tolist(),shi.tolist()]})
    face=np.zeros(2);face[axis]=-sign
    yaw=float(np.arctan2(face[1],face[0])) if repaired_stance else float(np.arctan2(center[1]-xy[1],center[0]-xy[0]))
    _set(session.robot,[*xy,floor_z+.015],[0.,0.,float(np.sin(yaw/2)),float(np.cos(yaw/2))])
    # Use the same accessible edge for primary object placements.
    c=np.asarray(slots[0]['center']);c[axis]=(low if sign<0 else high)[axis]-sign*(.11 if p['prototype']=='M2' else .18)
    if axis==int(np.argmax(high[:2]-low[:2])) and p['prototype'] not in ('H5','M1'):c[1-axis]=center[1-axis]
    slots[0]['center']=c.tolist()
    placements={};container_envelopes={};container_navigation=None;container_navigation_evidence=None
    if p['prototype']=='M3':
        from wia.candidates.container_navigation import build_background_navigation,connected_open_route
        og.sim.step()
        container_navigation,container_navigation_evidence=build_background_navigation(session,p['id'])
    for o in sorted(p['objects'],key=lambda o:not o.get('container',False)):
        obj=lookup(o['name']);slot=slots[o['slot']];c=np.asarray(slot['center']).copy()
        # Offsets are in tabletop tangential/depth coordinates at slot zero.
        if o['slot']==0:
            tangent_sign=sign*(1 if axis==0 else -1) if p['prototype']=='H1' else 1
            c[1-axis]+=tangent_sign*o['offset'][0];c[axis]-=sign*o['offset'][1]
        else:c[:2]+=o['offset']
        if p['prototype']=='M3' and o['name'] in ('item_02','item_03'):
            from wia.authoring.converters.active_layout import _support_ray_surface
            unit='unit_a' if o['slot']==1 else 'unit_b'
            fridge=lookup(unit);flo,fhi=[arr(v) for v in fridge.aabb];xyinside=(flo[:2]+fhi[:2])/2
            surface=_support_ray_surface(obj,fridge,[*xyinside,float(flo[2]+.65*(fhi[2]-flo[2]))],[*xyinside,float(flo[2]+.10*(fhi[2]-flo[2]))],'candidate_interior')['surface_z_m']
            _set(obj,[*xyinside,surface+.15],[0.,0.,0.,1.]);og.sim.step()
            pos=arr(obj.get_position_orientation()[0]);pos[:2]+=xyinside-arr(obj.aabb_center)[:2];pos[2]+=surface-float(arr(obj.aabb[0])[2])+.002;_set(obj,pos)
            placements[o['name']]={'support':unit,'slot':o['slot'],'interior_surface':surface}
            slots[o['slot']]={'support':unit,'center':[*xyinside,surface],'aabb':[flo.tolist(),fhi.tolist()]}
            continue
        if o.get('container'):
            from wia.candidates.container_envelope import measure_container_opening_envelope
            envelope=measure_container_opening_envelope(session,obj)
            relative=np.asarray(envelope['relative_aabb_m']);position=None
            corners=np.asarray([[x,y] for x in relative[:,0] for y in relative[:,1]])
            background_boxes=[]
            for background in _all(session):
                if background is obj or background.name.startswith(('item_','anchor_','zone_','unit_')) or background.category in ('floors','floor','ceilings'):continue
                background_boxes.extend(np.asarray([arr(v) for v in mesh.aabb])
                    for link in background.links.values() for mesh in link.collision_meshes.values())
            background_boxes.extend(np.asarray(e['world_aabb_m']) for e in container_envelopes.values())
            background_boxes=np.asarray(background_boxes)
            def blocked_box(box,obstacles=background_boxes):
                return bool(len(obstacles) and np.any(np.all(box[1]>obstacles[:,0],axis=1)&np.all(box[0]<obstacles[:,1],axis=1)))
            def floor_contains(box):
                return any(np.all(box[0,:2]>=arr(f.aabb[0])[:2]) and np.all(box[1,:2]<=arr(f.aabb[1])[:2]) for f in floors)
            # Reserve the sampled opening space, including simultaneous open
            # neighboring units. This remains an authored layout screen.
            for radius in (1.,1.4,1.8,2.2,2.6,3.,3.4,3.8,4.2):
                for angle in np.linspace(yaw+np.pi/2,yaw+np.pi/2+2*np.pi,24,endpoint=False):
                    center_xy=xy+radius*np.array([np.cos(angle),np.sin(angle)])
                    facing=np.arctan2(xy[1]-center_xy[1],xy[0]-center_xy[0])
                    orientations=sorted((0.,np.pi/2,np.pi,3*np.pi/2),key=lambda q:abs(np.arctan2(np.sin(q-facing),np.cos(q-facing))))
                    for unit_yaw in orientations:
                        rot=np.array([[np.cos(unit_yaw),-np.sin(unit_yaw)],[np.sin(unit_yaw),np.cos(unit_yaw)]])
                        transformed=corners@rot.T+center_xy
                        flo=np.r_[transformed.min(axis=0),floor_z+.025]
                        fhi=np.r_[transformed.max(axis=0),floor_z+.001+relative[1,2]]
                        envelope_box=np.asarray([flo,fhi])
                        if not floor_contains(envelope_box) or blocked_box(envelope_box):continue
                        # A clear door sweep alone may leave the opening facing
                        # a wall. Require a base-sized working stance on the
                        # native +X front after the door is fully open.
                        open_boxes=[]
                        for mesh in envelope['samples'][-1]['collision_boxes']:
                            box=np.asarray(mesh['relative_aabb_m']);pts=np.asarray([[x,y] for x in box[:,0] for y in box[:,1]])@rot.T+center_xy
                            open_boxes.append([np.r_[pts.min(axis=0),floor_z+.001+box[0,2]],np.r_[pts.max(axis=0),floor_z+.001+box[1,2]]])
                        open_boxes=np.asarray(open_boxes);service=None;service_route=None
                        all_open_boxes=[*open_boxes,*[box for prior in container_envelopes.values() for box in prior['fully_open_world_aabbs_m']]]
                        front=rot[:,0];tangent=rot[:,1]
                        # Interior side grasps need space for a low torso and
                        # an approximately 1.15 m front-edge target distance.
                        # The shorter earlier stances left the northern unit
                        # against the wall before that posture could fit.
                        for distance in (1.30,1.35):
                            for lateral in (0.,-.25,.25,-.45,.45):
                                stance=center_xy+distance*front+lateral*tangent
                                box=np.asarray([np.r_[stance-.37,floor_z+.04],np.r_[stance+.37,floor_z+.65]])
                                if floor_contains(box) and not blocked_box(box) and not blocked_box(box,open_boxes):
                                    service_yaw=float(np.arctan2(center_xy[1]-stance[1],center_xy[0]-stance[0]))
                                    route=connected_open_route(container_navigation,stance,service_yaw,all_open_boxes)
                                    if route is None:continue
                                    if any(connected_open_route(container_navigation,prior['front_service_stance_xy_m'],
                                            prior['front_service_face_yaw_rad'],all_open_boxes) is None for prior in container_envelopes.values()):continue
                                    service=stance;service_route=route;break
                            if service is not None:break
                        if service is not None:position=center_xy;break
                    if position is not None:break
                if position is not None:break
            if position is None:raise ValueError('No floor placement clears sampled opening envelope for '+o['name'])
            orientation=[0.,0.,float(np.sin(unit_yaw/2)),float(np.cos(unit_yaw/2))]
            _set(obj,[*position,floor_z+.75],orientation)
            root=arr(obj.get_position_orientation()[0]);root[:2]+=position-arr(obj.aabb_center)[:2];root[2]+=floor_z-float(arr(obj.aabb[0])[2])+.001
            _set(obj,root,orientation)
            container_envelopes[o['name']]={'measurement':envelope,'world_aabb_m':[flo.tolist(),fhi.tolist()],
                'closed_center_xy_m':position.tolist(),'root_yaw_rad':float(unit_yaw),'floor_exclusion_m':.025,
                'front_service_stance_xy_m':service.tolist(),'front_service_base_half_extent_m':.37,
                'front_service_distance_candidates_m':[1.30,1.35],
                'front_service_face_yaw_rad':service_yaw,'front_service_route':service_route,
                'fully_open_world_aabbs_m':open_boxes.tolist(),
                'front_service_scope':'Connected swept base footprint route in the background and both fully-open units; not arm or door-opening execution.'}
            placements[o['name']]={'support':'floor','container':True};continue
        _set(obj,[c[0],c[1],float(slot['center'][2])+.2],[0.,0.,0.,1.])
        og.sim.step()
        pos=arr(obj.get_position_orientation()[0]);pos[:2]+=c[:2]-arr(obj.aabb_center)[:2];pos[2]+=float(slot['center'][2])-float(arr(obj.aabb[0])[2])+.002
        _set(obj,pos)
        placements[o['name']]={'support':slot['support'],'slot':o['slot']}
    regions={}
    for name,index in p['regions'].items():
        slot=slots[index];c=np.asarray(slot['center']);slo,shi=np.asarray(slot['aabb'])
        half=np.minimum([.30,.21],(shi[:2]-slo[:2])/2-.01)
        if len(supports)==1:
            along=1-axis if p['prototype'] in ('H5','M1') else int(np.argmax(shi[:2]-slo[:2]));half[along]=min(half[along],(shi[along]-slo[along])/(2*slot_count)-.02)
        if p['prototype']=='M3':half=(shi[:2]-slo[:2])*.32
        if p['prototype']=='M1':
            half=np.zeros(2);half[axis]=.14 if name=='cleaning_surface' else .11
            half[1-axis]=.14 if name=='cleaning_surface' else .08
        if p['prototype']=='M2':half[axis]=.09
        if p['prototype']=='H4' and name=='target':half=np.array([.12,.09]) if axis==1 else np.array([.09,.12])
        rlo=np.r_[np.maximum(c[:2]-half,slo[:2]+.005),c[2]];rhi=np.r_[np.minimum(c[:2]+half,shi[:2]-.005),c[2]]
        regions[name]={'aabb':[rlo.tolist(),rhi.tolist()],'support':slot['support'],'support_z':float(c[2])}
        if p['prototype']=='M3':
            regions[name].update(interior=True,requires_open_doors=True,doors_open=False)
        if p['prototype']=='M2':
            regions[name]['label_front']={'axis':axis,'sign':sign,'edge_m':float((slo if sign<0 else shi)[axis])}
        zone=lookup('zone_'+name);zone.scale=torch.tensor(np.r_[half*2,.001],dtype=torch.float32)
        _set(zone,[c[0],c[1],c[2]+.001])
        zone.color=torch.tensor([.9,.65,.15] if name in ('quarantine','storage','supply','fallback','used') else [.15,.4,.9])
    # Interior support contacts are measured independently of task success.
    pstate={'parameters':deepcopy(p),'regions':regions,'placements':placements,'slots':slots,'floor_z':floor_z,'events':[],'observed':[]}
    session.candidate_state=pstate
    for _ in range(60):og.sim.step()
    # Native background furniture can finish settling after the initial ray.
    # Bind every region to its final measured surface before scoring or labels.
    surface_receipts={}
    from wia.authoring.converters.active_layout import _support_ray_surface
    for name,region in regions.items():
        support=lookup(region['support']);slo,shi=[arr(value) for value in support.aabb]
        bounds=np.asarray(region['aabb'],dtype=float);xy=bounds[:,:2].mean(axis=0)
        if p['prototype']=='M3':
            start,end=float(region['support_z']+.1),float(region['support_z']-.1)
        else:start,end=float(shi[2]+.1),float(slo[2]+.1)
        measured=_support_ray_surface(support,support,[*xy,start],[*xy,end],'settled_candidate_region')
        bounds[:,2]=measured['surface_z_m'];region.update(aabb=bounds.tolist(),support_z=measured['surface_z_m'],support_aabb=[slo.tolist(),shi.tolist()])
        zone=lookup('zone_'+name);zone.scale=torch.tensor(np.r_[bounds[1,:2]-bounds[0,:2],.001],dtype=torch.float32)
        _set(zone,[*xy,measured['surface_z_m']+.001])
        surface_receipts[name]=measured
    receipt={'selected_native_fixture':table.name,'requested_native_fixture':preferred.name,'scene_loaded':True,'physical_execution_verified':False,'objects':{},'layout_issues':[]}
    if container_envelopes:receipt['container_opening_envelopes']=container_envelopes
    if container_navigation_evidence:receipt['container_background_navigation']=container_navigation_evidence
    receipt['settled_region_surfaces']=surface_receipts
    for o in p['objects']:
        obj=lookup(o['name']);support=lookup(placements[o['name']]['support'])
        row={'aabb':[arr(x).tolist() for x in obj.aabb],'support':placements[o['name']]['support']}
        if support is not None:
            row['touching']=bool(obj.states[Touching].get_value(support))
            if not o.get('fixed') and not row['touching']:receipt['layout_issues'].append(o['name']+': missing intended support contact')
        if p['prototype']=='M3' and o['name'] in ('item_02','item_03'):
            from omnigibson.object_states import Inside
            row['inside_support']=bool(obj.states[Inside].get_value(support))
            if not row['inside_support']:receipt['layout_issues'].append(o['name']+': not inside assigned unit')
        receipt['objects'][o['name']]=row
    if p['prototype']=='H1':
        from scipy.spatial.transform import Rotation
        left=Rotation.from_quat(arr(session.robot.get_position_orientation()[1])).as_matrix()[:,1]
        delta=arr(lookup(p['contract']['selected']).aabb_center)-arr(lookup(p['contract']['spares'][0]).aabb_center)
        receipt['selected_left_of_spare_m']=float(np.dot(delta,left))
        if receipt['selected_left_of_spare_m']<=0:receipt['layout_issues'].append('Selected package is not left of its spare in the initial robot frame')
    receipt['initial_support_check_passed']=not receipt['layout_issues']
    from wia.tasks.interaction_setup import _robot_contact_check
    floor_paths=[str(obj.prim_path) for obj in _all(session) if obj.category in ('floor','floors')]
    receipt['robot_start_checks']=_robot_contact_check(session.robot,floor_paths,floor_z)
    # Attach quantity/acceptance visuals to the same rigid link so identity and
    # markings follow physical motion; these are authored evidence, not state.
    for sensor in session.robot.sensors.values():
        if 'rgb' in sensor.modalities and 'seg_instance' not in sensor.modalities:sensor.add_modality('seg_instance')
    receipt['object_markers']=_marks(session,p)
    label_dir=ROOT/'outputs/interactiveworld_20261029/labels'/p['id']
    from wia.candidates.labels import add_region_labels
    receipt['labels']=add_region_labels(session,regions,label_dir)
    inventory=native_inventory(session)
    out=Path(os.environ.get('WIA_CANDIDATE_OUTPUT',str(ROOT/'outputs/interactiveworld_20261029/native')))/p['id']
    out.mkdir(parents=True,exist_ok=True)
    (out/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    (out/'initialization.json').write_text(json.dumps(receipt,indent=2)+'\n')
    session.candidate_initialization_receipt=receipt
    return receipt


def _marks(session,p):
    from pxr import UsdGeom, Gf
    import omnigibson as og
    receipts=[]
    for spec in p['objects']:
        if not spec.get('package') and 'quantity' not in spec:continue
        obj=session.env.scene.object_registry('name',spec['name'])
        if 'quantity' in spec:
            from wia.candidates.labels import add_object_label
            receipt=add_object_label(session,obj,f"QTY {spec['quantity']}",ROOT/'outputs/interactiveworld_20261029/labels'/p['id'])
            receipts.append({'object':spec['name'],**receipt})
            continue
        path=str(obj.root_link.prim_path)+'/candidate_evidence'
        # H1 uses an underside red/green square. Transform world geometry into
        # the actual link frame; M2 quantity text was attached above.
        from scipy.spatial.transform import Rotation
        position,quaternion=[arr(x) for x in obj.root_link.get_position_orientation()]
        rotation=Rotation.from_quat(quaternion).as_matrix()
        low,high=[arr(x) for x in obj.aabb];center=(low+high)/2
        count=spec.get('quantity',1)
        for i in range(count):
            cube=UsdGeom.Cube.Define(og.sim.stage,path+f'/pip_{i}');cube.CreateSizeAttr(1.)
            world=center.copy();world[2]=low[2]-.0008 if spec.get('package') else high[2]+.001
            world[0]+=(i-(count-1)/2)*.012
            xform=UsdGeom.Xformable(cube);xform.AddTranslateOp().Set(Gf.Vec3d(*map(float,((world-position)@rotation))))
            # Parent object scale is accounted for by link matrix inverse.
            matrix=UsdGeom.Xformable(obj.root_link.prim).ComputeLocalToWorldTransform(0)
            point=matrix.GetInverse().Transform(Gf.Vec3d(*map(float,world)));xform.GetOrderedXformOps()[0].Set(point)
            scales=np.linalg.norm(np.asarray(matrix)[:3,:3],axis=1)
            size=np.array([.035,.035,.001]) if spec.get('package') else np.array([.008,.008,.001])
            xform.AddScaleOp().Set(Gf.Vec3f(*map(float,size/scales)))
            bad=spec['name']=='item_01' and p['condition'].get('defective',False)
            cube.CreateDisplayColorAttr([Gf.Vec3f(*([.9,.05,.05] if bad else [.05,.65,.08] if spec.get('package') else [.02,.02,.02]))])
    return receipts


def native_inventory(session):
    """Reuse the calibrated native exporter for candidate target accounting."""
    from wia.candidates.inventory import export_inventory
    state=session.candidate_state;p=state['parameters']
    return export_inventory(session,identity=p['id'],prototype=p['prototype'],
        target_names=[spec['name'] for spec in p['objects']],
        protected_names=p['contract'].get('protected',[]),regions=state['regions'],
        support_by_target={name:row['support'] for name,row in state['placements'].items()})


def qualify_operation(session,controller,config,bound):
    return {'accepted':False,'evaluation_validity':'unavailable','reason':'Candidate physical action qualification is deferred by user; scene-only validation is separate.'}


def create_evaluator(session,controller,config,output):
    from wia.candidates.native import CandidateEvaluator,install_observation_hook
    evaluator=CandidateEvaluator(session,controller,config,output)
    install_observation_hook(session,config['parameters'],evaluator)
    return evaluator
