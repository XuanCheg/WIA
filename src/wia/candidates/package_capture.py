"""Inspect an H1 underside marker after an explicitly authored flip.

Run through ``python -m wia.candidates.capture --id iw_h1_01 --output DIR``.
This visual review changes object pose directly; it does not execute or certify
a robot grasp, flipping trajectory, or policy observation delivery.
"""
import json
from pathlib import Path


def capture_package_mark(session, scene, output):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from core.record.images import save_png
    from wia.authoring.converters.interactiveworld_candidates import arr, _set
    from wia.runtime.sim.esi_review_cameras import capture_review_views, review_view_names

    output=Path(output)
    p=session.candidate_state['parameters']
    obj=session.env.scene.object_registry('name',p['contract']['selected'])
    receipt={'schema':'wia_native_package_marker_review_v1','robot_actions':0,'model_calls':0,
        'authored_object_pose':True,'robot_flip_verified':False,'history_delivered_to_policy':False,
        'scope':'Rendered underside evidence after a direct authored rigid-body flip; no manipulation execution.'}
    position,orientation=[arr(value) for value in obj.get_position_orientation()]
    try:
        center=arr(obj.aabb_center);bottom=float(arr(obj.aabb[0])[2])
        flip=Rotation.from_euler('x',np.pi)
        root=center-flip.apply(center-position)
        quaternion=(flip*Rotation.from_quat(orientation)).as_quat()
        _set(obj,root,quaternion)
        root=arr(obj.get_position_orientation()[0]);root[2]+=bottom-float(arr(obj.aabb[0])[2])+.002
        _set(obj,root,quaternion)
        for _ in range(30):session.pipeline.og.sim.step()
        for _ in range(5):session.pipeline.og.sim.render()
        raw=session.observe(include_wrist=True,include_scene=True)
        views={'head':raw['head_rgb']}
        views.update({name:session._sensor_rgb(sensor) for name,sensor in session.global_cameras.items()})
        views.update(capture_review_views(session,scene,review_view_names(scene.definition)))
        for name,pixels in views.items():save_png(output/('package_flip_'+name+'.png'),pixels)
        receipt.update(status='captured',views=list(views),selected_object=obj.name,
            defect_condition=bool(p['condition']['defective']),observed_tags_after=list(session.candidate_state['observed']))
    except Exception as error:
        receipt.update(status='error',error=f'{type(error).__name__}: {error}')
    finally:
        _set(obj,position,orientation)
        (output/'package_flip_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
