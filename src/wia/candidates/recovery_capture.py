"""Review H2's native disturbance after explicitly authored endpoint staging.

Run via ``python -m wia.candidates.capture --id iw_h2_02 --output DIR``.
This checks the event and its geometry, without executing a robot plan, claiming
policy observation delivery, or evaluating robot task success.
"""
import json
from pathlib import Path


def capture_recovery_event(session, output):
    from core.record.images import save_png
    from wia.candidates.events import CandidateEvents, h2_readiness
    from wia.candidates.high import _placed, _overlap
    from wia.authoring.converters.interactiveworld_candidates import native_inventory

    output=Path(output)
    state=session.candidate_state
    params=state['parameters'];contract=params['contract']
    events=CandidateEvents(session,params)
    receipt={'schema':'wia_native_recovery_event_review_v1','robot_actions':0,'model_calls':0,
        'authored_endpoint_staging':True,'history_delivered_to_policy':False,
        'native_trigger_sequence_verified':False,'robot_execution_verified':False,
        'scope':'The workspace/storage and delivery endpoints are authored with native object setters. Only native event displacement, contacts and preservation are checked; no legal robot execution or feedback benefit is inferred.'}
    try:
        events._place_in_region(contract['obstruction'],contract['storage'])
        events._place_in_region(contract['payload'],contract['delivery'])
        for _ in range(30):session.pipeline.og.sim.step()
        before=events._measure()
        ready=h2_readiness(before,contract)
        delivered=_placed(before,contract['payload'],contract['delivery'])
        receipt.update(staged_workspace_ready=ready,staged_payload_placed=delivered)
        if not ready or delivered is not True:raise ValueError('Authored native endpoints did not satisfy the event precondition')
        # Deliberate diagnostic bypass: the actual observation-trigger ordering
        # has separate CPU checks and is not exercised by this staging helper.
        events.control['h2_phase']='pending'
        def capture(label):
            for _ in range(5):session.pipeline.og.sim.render()
            raw=session.observe(include_wrist=True,include_scene=True)
            save_png(output/(label+'_head.png'),raw['head_rgb'])
            for name,sensor in session.global_cameras.items():
                save_png(output/(label+'_'+str(name)+'.png'),session._sensor_rgb(sensor))
            return {'simulation_time_s':float(session.pipeline.og.sim.current_time),'history_delivered_to_policy':False}
        receipt['event']=events.apply_preview('H2',capture)
        after=events._measure()
        receipt.update(workspace_overlap_after=_overlap(after,contract['obstruction'],contract['work_region']),
            payload_placed_after=_placed(after,contract['payload'],contract['delivery']),
            observed_tags_after=list(state['observed']))
        inventory=native_inventory(session)
        inventory['diagnostic_intervention']=receipt.copy()
        (output/'prelude_final_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        expected=bool(contract['disturbance'])
        passed=(receipt['event']['status']=='applied' and receipt['event']['preservation_valid'] is True
                and receipt['workspace_overlap_after'] is expected and receipt['payload_placed_after'] is True
                and not receipt['observed_tags_after'])
        receipt['status']='captured' if passed else 'error'
    except Exception as error:
        receipt.update(status='error',error=f'{type(error).__name__}: {error}')
    (output/'recovery_event_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
