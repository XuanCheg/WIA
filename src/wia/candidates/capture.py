"""Capture one candidate without robot actions. Run: python -m wia.candidates.capture --id iw_h5_01 --output DIR.

Caller must select an idle physical GPU explicitly and configure the installed
behavior environment. The process exits after saving native sensor images.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from wia.resources import ROOT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--id',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);began=time.time()
    from wia.runtime.commands.run_esi_capture_worker import allocation_gpu
    from wia.runtime.scene_capture_adapters import install_gpu_selection,isolate_texture_cache
    gpu=allocation_gpu(0,execution_mode='local',renderer_order='pci');install_gpu_selection(gpu)
    from omnigibson import simulator
    isolate_texture_cache(simulator,os.environ['OMNIGIBSON_APPDATA_PATH'])
    from wia.tasks.loader import load_scene
    from wia.tasks.scene_only import SceneOnly
    from wia.runtime.sim.esi_omnigibson import ESISession
    from wia.runtime.sim.esi_robot_cameras import configure_robot_cameras
    from core.viewer_config import ViewerConfig
    from core.record.images import save_png
    candidate=args.id.startswith('iw_')
    if candidate:
        scene_path=ROOT/'explore_to_achieve/candidates/interactiveworld_20261029/scenes'/(args.id+'.yaml');variant='adapted'
    else:
        catalogue=json.loads((ROOT/'explore_to_achieve/wia_v1/release/catalogue.json').read_text())
        row=next(row for row in catalogue['tasks'] if row['alias']==args.id)
        scene_path=ROOT/row['scene_file'];variant=row['variant']
    scene=load_scene(scene_path,variant=variant,run_mode='diagnostic')
    from wia.tasks.scene import ScenePatches
    adapter=ScenePatches(scene,scene.definition.bindings);settings=adapter.observation_settings();settings['camera_size']=list(scene.definition.runtime.camera_size)
    session=ESISession(scene.source_path,**settings);session.scene_adapter=adapter
    session._capture_release_edge_policy=scene.release_v1
    if not candidate:
        from wia.tasks.loader import load_task
        session.loaded_task=load_task(ROOT/row['task_file'],variant=variant,run_mode='diagnostic')
    report={'id':args.id,'gpu':gpu,'pid':os.getpid(),'robot_actions':0,'model_calls':0,'started':began,'scene_sha256':scene.file_sha256,'source_sha256':scene.source_sha256}
    report['implementation_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT/'src/wia/candidates/capture.py',ROOT/'src/wia/candidates/inventory.py',
                     ROOT/'src/wia/candidates/phase_capture.py',ROOT/'src/wia/authoring/converters/interactiveworld_candidates.py']}
    try:
        session.reset();configure_robot_cameras(session,ROOT/'configs/esi_robot_cameras.json',resolution=(512,512))
        session.configure_scene_camera();session.configure_human_cameras(ViewerConfig.load(ROOT/'configs/esi_viewer.json'))
        for _ in range(5):session.pipeline.og.sim.render()
        raw=session.observe(include_wrist=True,include_scene=True)
        views={'head':raw['head_rgb'],'wrist':raw['wrist_rgb']}
        views.update({name:session._sensor_rgb(sensor) for name,sensor in session.global_cameras.items()})
        for name,sensor_name in session.robot_camera_registry.items():
            views[str(name)]=session._sensor_rgb(session.robot.sensors[sensor_name])
        if raw.get('scene_rgb') is not None:views['scene']=raw['scene_rgb']
        for name,pixels in views.items():save_png(out/(name+'.png'),pixels)
        from wia.runtime.sim.esi_review_cameras import capture_review_views,review_view_names
        review_names=review_view_names(scene.definition)
        if review_names:
            details=capture_review_views(session,scene,review_names)
            for name,pixels in details.items():save_png(out/(name+'.png'),pixels)
            views.update(details)
        if candidate:
            from wia.authoring.converters.interactiveworld_candidates import native_inventory
            inventory=native_inventory(session)
            report['candidate_initialization']=getattr(session,'candidate_initialization_receipt',None)
        else:
            from wia.candidates.inventory import export_inventory
            setup=getattr(session,'category_interaction_setup',{}) or {}
            supports={name:info['support'] for name,info in setup.get('supported_items',{}).items() if info.get('support')}
            names=[binding.name for binding in scene.definition.bindings.values() if binding.name]
            report['interaction_setup']=setup
            try:
                inventory=export_inventory(session,identity=args.id,target_names=names,support_by_target=supports or None)
            except ValueError:
                from wia.authoring.converters.interactiveworld_candidates import arr
                report['inventory_export_diagnostics']={'robot_position':arr(session.robot.get_position_orientation()[0]).tolist(),
                    'robot_base_aabb':[arr(v).tolist() for v in session.robot.links[session.robot.base_footprint_link_name].aabb],
                    'floor_objects':[{'name':obj.name,'category':obj.category,'visual_only':bool(obj.visual_only),'aabb':[arr(v).tolist() for v in obj.aabb]} for obj in session.env.scene.objects if obj.category in ('floor','floors','ground','lawn','driveway')]}
                raise
        (out/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        report.update(status='captured',views=list(views),physical_execution_verified=False)
        if not candidate:
            from wia.candidates.phase_capture import capture_temporal_geometry
            phase_receipt=capture_temporal_geometry(session,scene,out,target_names=names,support_by_target=supports or None)
            if phase_receipt is not None:report['temporal_geometry']=phase_receipt
        params=session.candidate_state['parameters'] if candidate else {'prototype':'baseline'}
        if params['prototype']=='H1':
            from wia.candidates.package_capture import capture_package_mark
            report['package_marker_review']=capture_package_mark(session,scene,out)
        if params['prototype']=='H2':
            from wia.candidates.recovery_capture import capture_recovery_event
            report['recovery_event_review']=capture_recovery_event(session,out)
        if params['prototype']=='M3':
            from wia.candidates.interior_capture import capture_open_interiors
            report['open_interior_review']=capture_open_interiors(session,out)
        if params['prototype']=='M1' or (params['prototype']=='H3' and params['condition'].get('progress')=='undone'):
            import base64
            from wia.candidates.native import CandidateEvaluator
            from wia.candidates.prelude import CandidatePrelude
            event_dir=out/'authored_event_preview';event_dir.mkdir(exist_ok=True)
            evaluator=CandidateEvaluator(session,None,{'parameters':params},event_dir/'private_evaluation')
            prelude=CandidatePrelude(session,params,evaluator)
            initial=session.observation_source.capture(action_boundary=True).observation
            final=prelude.run(session.observation_source,initial)
            payload=session.candidate_public_history
            (event_dir/'public_history.json').write_text(json.dumps(payload,indent=2)+'\n')
            for frame in payload['frames']:
                (event_dir/f"frame_{frame['index']:02}_head.png").write_bytes(base64.b64decode(frame['png_base64']))
            (event_dir/'private_evidence.json').write_text(json.dumps(prelude.private_records,indent=2)+'\n')
            post_inventory=native_inventory(session)
            (out/'prelude_final_inventory.json').write_text(json.dumps(post_inventory,indent=2)+'\n')
            receipt={'schema':'wia_native_prelude_review_v1','frame_count':len(payload['frames']),
                'final_snapshot_id':final.snapshot_id,'history_delivered_to_policy':False,
                'robot_actions':0,'model_calls':0,'authored_event':session.candidate_state['episode_prelude']['authored_event_receipt'],
                'public_history_file':str(event_dir/'public_history.json'),
                'head_visible_names_by_frame':[row['visual_evidence']['visible_names'] for row in prelude.private_records],
                'observed_tags_after_preview':list(session.candidate_state['observed'])}
            (event_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
            report['authored_event_preview']=receipt
    except Exception as error:
        report.update(status='error',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['wall_time_s']=time.time()-began;(out/'capture.json').write_text(json.dumps(report,indent=2)+'\n')
        session.close()

if __name__=='__main__':main()
