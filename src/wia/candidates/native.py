"""Read-only native candidate evidence adapter.

Run: python -m pytest tests/test_candidate_high.py tests/test_candidate_medium.py.
Native scene captures do not deliver event history to a policy. The opt-in
observation hook records only synchronized mounted views at action boundaries.
M1/H3 preludes defer history credit until the runtime acknowledges delivery;
native physical execution remains unqualified.
"""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from wia.authoring.converters.interactiveworld_candidates import arr


class CandidateEvaluator:
    requires_done_declaration=True

    def __init__(self,session,controller,config,output):
        self.session=session;self.controller=controller;self.config=config;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.frames=[];self.prelude_frames=[];self.initial_time=None;self.episode_started=False

    def measure(self):
        from omnigibson.object_states import Touching,Open,Inside
        session=self.session;state=session.candidate_state;p=self.config['parameters'];scene=session.env.scene
        objects={}
        supports={r['support'] for r in state['regions'].values()}
        held_names=set()
        # Native assisted grasp state has one entry per arm; missing state is
        # unavailable, never treated as proof of a released object.
        held=getattr(session.robot,'_ag_obj_in_hand',None)
        if isinstance(held,dict):held_names={obj.name for obj in held.values() if obj is not None}
        for spec in p['objects']:
            obj=scene.object_registry('name',spec['name']);position,orientation=[arr(x).tolist() for x in obj.get_position_orientation()]
            contacts=[]
            for name in supports:
                other=scene.object_registry('name',name)
                if other is not None and obj is not other and obj.states[Touching].get_value(other):contacts.append(name)
            objects[obj.name]={'position':position,'orientation':orientation,'native_id':str(obj.prim_path),'aabb':[arr(x).tolist() for x in obj.aabb],
                'linear_speed':float(np.linalg.norm(arr(obj.get_linear_velocity()))),'angular_speed':float(np.linalg.norm(arr(obj.get_angular_velocity()))),
                'held':obj.name in held_names if held is not None else None,'contacts':contacts}
        frame={'time_s':float(session.pipeline.og.sim.current_time),'objects':objects,'regions':deepcopy(state['regions']),
            'events':deepcopy(state['events']),'observed':deepcopy(state['observed'])}
        if p['prototype']=='M3':
            for name in ('item_01','item_02','item_03'):
                obj=scene.object_registry('name',name)
                frame['objects'][name]['inside']=[unit for unit in ('unit_a','unit_b') if obj.states[Inside].get_value(scene.object_registry('name',unit))]
            frame['equipment']={name:{'closed':not bool(scene.object_registry('name',name).states[Open].get_value())} for name in ('unit_a','unit_b')}
        if p['prototype']=='H4':
            target=p['contract']['target']
            zone=np.asarray(state['regions']['target']['aabb'])
            hands=[arr(session.robot.links[session.robot.eef_link_names[arm]].get_position_orientation()[0]) for arm in session.robot.arm_names]
            distances=[float(np.linalg.norm(np.maximum(np.maximum(zone[0]-point,point-zone[1]),0))) for point in hands]
            frame['process_evidence']={'hands_withdrawn':bool(min(distances)>.08 and objects[target]['held'] is False),
                'method':'Native final end-effector separation and released target; Safety is evaluated independently.'}
        return frame

    def begin_episode(self,observation):
        self.frames=[]
        # Public prelude images can share simulation time while snapshot IDs
        # advance. Reducers require one chronological measured state per tick.
        for frame in self.prelude_frames:
            if self.frames and frame['time_s']==self.frames[-1]['time_s']:
                self.frames[-1]=deepcopy(frame)
            elif not self.frames or frame['time_s']>self.frames[-1]['time_s']:
                self.frames.append(deepcopy(frame))
            else:raise ValueError('Candidate prelude simulation time moved backwards.')
        self.episode_started=True
        return self.record(observation,action=None)

    def sample_step(self):
        if not self.episode_started:return
        frame=self.measure()
        if not self.frames or frame['time_s']>self.frames[-1]['time_s']:self.frames.append(frame)

    def record(self,observation,*,action=None):
        from wia.candidates.high import evaluate as high
        from wia.candidates.medium import evaluate as medium
        p=self.config['parameters'];frame=self.measure();history=[f for f in self.frames if f['time_s']<frame['time_s']]
        result=(high if p['prototype'].startswith('H') else medium)(p['prototype'],p['contract'],frame,history)
        result.update(snapshot_id=getattr(observation,'snapshot_id',None),action_step=getattr(observation,'action_step',None),physical_execution_verified=False)
        (self.output/'state.json').write_text(json.dumps(result,indent=2)+'\n');self.sample_step();return result

    def declare_done(self,observation):
        return self.finalize_episode(observation)

    def finalize_episode(self,observation):
        """Reduce the actual final frame, including exits without model DONE."""
        result=self.record(observation);result['evaluation_finalized']=True;return result


def install_observation_hook(session, parameters, evaluator):
    """Install an opt-in boundary event provider, separate from score reduction."""
    from wia.candidates.events import CandidateEvents
    from wia.candidates.observations import CandidateObservationEvidence
    events=CandidateEvents(session,parameters);visual=CandidateObservationEvidence(session,parameters)
    needs_prelude=(parameters['prototype']=='M1' or (parameters['prototype']=='H3' and parameters['contract'].get('requires_revision_event')))
    prelude=None
    if needs_prelude:
        from wia.candidates.prelude import CandidatePrelude
        existing=getattr(session,'_candidate_prelude_controller',None)
        prelude=(existing if isinstance(existing,CandidatePrelude) and existing.evaluator is evaluator
                 and existing.state is session.candidate_state else CandidatePrelude(session,parameters,evaluator))
        if prelude is not existing:session.candidate_public_history=None
        session._candidate_prelude_controller=prelude
        session.candidate_episode_prelude=prelude.run
        session.candidate_public_history_delivery=prelude.mark_delivered
    else:
        for name in ('_candidate_prelude_controller','candidate_episode_prelude',
                     'candidate_public_history_delivery','candidate_public_history'):
            if hasattr(session,name):delattr(session,name)
    def on_boundary(observation):
        frame=evaluator.measure();receipt=visual.record(observation,frame)
        state=session.candidate_state
        if prelude is not None and not prelude.delivered:
            event_receipt={'status':'awaiting_public_history_delivery','observed_tags':[]}
        else:
            for tag in receipt['observed_tags']:
                if tag not in state['observed']:state['observed'].append(tag)
            event_receipt=events.on_observation_delivered(frame,set(receipt['visible_names']))
        state['latest_observation_receipt']=receipt;state['latest_event_receipt']=event_receipt
    session.candidate_observation_hook=on_boundary
