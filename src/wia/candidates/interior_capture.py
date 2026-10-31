"""Review authored open-container geometry without robot actions.

Run through ``python -m wia.candidates.capture --id iw_m3_01 --output DIR``.
The initial closed inventory is exported first. This diagnostic sets only native
container joints; it does not validate robot opening, grasping or task success.
"""
from copy import deepcopy
import json
from pathlib import Path


def capture_open_interiors(session, output):
    from omnigibson.object_states import Open, Inside, Touching
    from core.record.images import save_png
    from wia.authoring.converters.interactiveworld_candidates import native_inventory, arr

    output=Path(output)
    state=session.candidate_state
    receipt={'schema':'wia_native_open_interior_review_v1','robot_actions':0,'model_calls':0,
        'authored_scene_intervention':True,'robot_opening_verified':False,'units':{},'errors':[],
        'scope':'Current open geometry after authored native joint setting; opening trajectories, handle access and task success are untested.'}
    original_regions=deepcopy(state['regions'])
    try:
        for name,content in state['parameters']['contract']['content_by_unit'].items():
            unit=session.env.scene.object_registry('name',name)
            item=session.env.scene.object_registry('name',content)
            old=arr(unit.get_joint_positions()).tolist()
            setter=bool(unit.states[Open].set_value(True,fully=True))
            # Synchronize native articulated geometry without commanding the robot.
            session.pipeline.og.sim.step()
            is_open=bool(unit.states[Open].get_value())
            state['regions'][name]['doors_open']=is_open
            receipt['units'][name]={'setter_return':setter,'native_open':is_open,
                'joints_before':old,'joints_after':arr(unit.get_joint_positions()).tolist(),
                'resident_inside':bool(item.states[Inside].get_value(unit)),
                'resident_touching':bool(item.states[Touching].get_value(unit))}
            allowed=[str(unit.prim_path),str(item.prim_path)]
            allowed.extend(str(obj.prim_path) for obj in session.env.scene.objects
                           if getattr(obj,'category',None) in ('floor','floors'))
            outside=set()
            for contact in unit.contact_list():
                for body in (str(getattr(contact,'body0','')),str(getattr(contact,'body1',''))):
                    if body and not any(body.startswith(prefix) for prefix in allowed):outside.add(body)
            receipt['units'][name]['external_nonfloor_contacts']=sorted(outside)
            if outside:receipt['errors'].append(name+': open unit has external nonfloor contacts')
            if not setter or not is_open:receipt['errors'].append(name+': native fully-open state unavailable')
        for _ in range(5):session.pipeline.og.sim.render()
        raw=session.observe(include_wrist=True,include_scene=True)
        save_png(output/'dooropen_head.png',raw['head_rgb'])
        for name,sensor in session.global_cameras.items():
            save_png(output/('dooropen_'+str(name)+'.png'),session._sensor_rgb(sensor))
        from wia.runtime.sim.esi_review_cameras import capture_review_views,review_view_names
        scene=session.scene_adapter.task
        review=capture_review_views(session,scene,review_view_names(scene.definition))
        for name,pixels in review.items():save_png(output/('dooropen_'+name+'.png'),pixels)
        receipt['interior_review_views']=list(review)
        # Explicit front views expose contents that generic context cameras can
        # hide behind the cabinet. These are human review cameras, not robot
        # observations or evidence that a sensor-equipped robot reached them.
        envelopes=session.candidate_initialization_receipt['container_opening_envelopes']
        fixed={}
        for index,(name,content) in enumerate(state['parameters']['contract']['content_by_unit'].items()):
            item=session.env.scene.object_registry('name',content)
            center=(arr(item.aabb[0])+arr(item.aabb[1]))/2
            fixed[f'object_interior_{index:02d}']={
                'position_world_m':[*envelopes[name]['front_service_stance_xy_m'],
                                    float(state['regions'][name]['support_z'])+.20],
                'target_world_m':center.tolist(),'clipping_range_m':[.02,20.],
                'unit':name,'resident':content,'human_review_only':True,
                'robot_observation_access_verified':False}
        front=capture_review_views(session,scene,list(fixed),fixed_views=fixed)
        for name,pixels in front.items():save_png(output/('dooropen_'+name+'.png'),pixels)
        receipt['front_review_views']=deepcopy(session.human_review_camera_evidence)
        inventory=native_inventory(session)
        inventory['diagnostic_intervention']=deepcopy(receipt)
        (output/'dooropen_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        receipt['status']='captured' if not receipt['errors'] else 'error'
    except Exception as error:
        receipt.update(status='error')
        receipt['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        # The owner closes this diagnostic session immediately after capture.
        # Restore metadata only; do not pretend the native joints were restored.
        state['regions']=original_regions
        (output/'dooropen_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
