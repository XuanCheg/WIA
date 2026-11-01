"""Check deterministic mapping claims and private transport. Run: python -m pytest tests/test_mapping_declarations.py."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from core.agent.decision import DecisionInput, ModelReply
from wia.runtime.agent.observation_declarations import enable_declarations, unpack_declarations
from wia.runtime.agent.output_normalization import normalize_extra_fields
from wia.native.mapping_observations import MappingObservationEvidence


def request():
    return enable_declarations(DecisionInput(NS(), ('DONE',), world_observation_log=True, assembled_prompt='fixture'))


def test_schema_parser_preserves_claims_and_never_executes_them():
    req = request()
    claims = [{'label': 'top_cabinet', 'camera': 'agentview_rgb'}]
    raw = json.dumps({'command': {'action': 'DONE'}, 'world_observation': 'Cabinet seen.', 'observed_objects': claims})
    result, parsed = unpack_declarations(req, normalize_extra_fields(req, ModelReply(raw, {})))
    assert json.loads(result.raw_text) == {'action': 'DONE'}
    assert parsed == claims
    assert 'observed_objects' in req.structured_schema['required']
    assert result.metadata['world_observation'] == 'Cabinet seen.'


@pytest.mark.parametrize('claims', [None, 'cabinet', [{}], [{'label': 'x', 'camera': 'head', 'accepted': True}]])
def test_malformed_claims_rejected(claims):
    raw = json.dumps({'command': {'action': 'DONE'}, 'world_observation': 'Visible.', 'observed_objects': claims})
    with pytest.raises(ValueError):
        unpack_declarations(request(), ModelReply(raw, {}))


def evidence():
    return {'cameras': {'agentview_rgb': {'native_rgb_matches_delivered': True, 'retained_capture_verified': True}},
        'view_evidence': [], 'measurements': {'anchor_a': {'views': {'agentview_rgb': {
            'camera': 'agentview_rgb', 'native_name': 'private_cabinet',
            'measurement_available': True, 'identity_synchronized': True, 'native_rgb_matches_delivered': True,
            'visible_pixels': 80, 'snapshot_id': 7, 'action_step': 2}}}}, 'measurement_errors': {}}


@pytest.fixture
def scorer(monkeypatch):
    session = NS(episode_id='episode-a')
    config = {'bindings': {'anchor_a': 'private_cabinet'},
              'parameters': {'public_anchor_labels': {'anchor_a': 'top cabinet'}}}
    current = evidence()
    monkeypatch.setattr('wia.native.mapping_observations.collect_native_visual_evidence', lambda *a: deepcopy(current))
    collector = MappingObservationEvidence(session, config)
    obs = NS(snapshot_id=7, action_step=2)
    collector.capture_observation(obs)
    collector.begin_episode()
    return collector, session, current, obs


def test_world_run_uses_saved_capture_not_later_sensor_state(scorer):
    collector, session, current, obs = scorer
    current['measurements']['anchor_a']['views']['agentview_rgb']['visible_pixels'] = 0
    collector.report([{'label': 'top_cabinet', 'camera': 'agentview_rgb'}], ['agentview_rgb'], obs)
    assert collector.receipts[0]['visible_pixels'] == 80
    assert collector.receipts[0]['recognition_inferred_from_mask'] is False
    session.episode_id = 'episode-b'
    with pytest.raises(ValueError):
        collector.report([], ['agentview_rgb'], obs)


@pytest.mark.parametrize('change', ['not_delivered', 'wrong_label', 'hidden', 'missing_identity', 'stale_snapshot'])
def test_no_credit_for_unverified_claims(scorer, change):
    collector, _, _, obs = scorer
    claim = {'label': 'top cabinet', 'camera': 'agentview_rgb'}
    views = ['agentview_rgb']
    if change == 'not_delivered': views = []
    if change == 'wrong_label': claim['label'] = 'private_cabinet'
    if change == 'hidden': collector.capture['evidence']['measurements']['anchor_a']['views']['agentview_rgb']['visible_pixels'] = 0
    if change == 'missing_identity': collector.capture['evidence']['measurements']['anchor_a']['views']['agentview_rgb']['identity_synchronized'] = False
    if change == 'stale_snapshot':
        with pytest.raises(ValueError): collector.report([claim], views, NS(snapshot_id=8, action_step=2))
    else:
        collector.report([claim], views, obs)
    assert collector.receipts == []


def test_no_declaration_is_measured_incomplete(scorer):
    collector, _, _, obs = scorer
    collector.report([], ['agentview_rgb'], obs)
    assert collector.available() and not collector.receipts

from harness_integration.test_agent_control import claim, control_http
from harness_integration.test_done_declaration import post_on_owner


def test_http_observation_claim_is_owner_bound_private_and_nonmotion(control_http):
    backend, _, url, calls, rig = control_http
    token = claim(backend, url)
    received = []
    def report(objects, views, observation):
        import threading
        assert threading.current_thread() is threading.main_thread()
        received.append((objects, views, observation))
        return {'task_success': True, 'private_identity': 'NEVER_PUBLIC'}
    rig.task_evaluator=NS(report_observed_objects=report)
    source=backend.observation_source
    before=source.agent_observation()
    cycle=rig.cycle
    body={'objects':[{'label':'top cabinet','camera':'agentview_rgb'}], 'delivered_views':['agentview_rgb'],
          'snapshot_id':before.snapshot_id,'action_step':before.action_step,'control_token':token}
    response=post_on_owner(backend,url,body,path='/api/agent/observed_objects')
    assert response.status_code==200 and set(response.json()) <= {'ok','message','world_timing'}
    assert len(received)==1 and rig.cycle==cycle and not calls
    stale=post_on_owner(backend,url,{**body,'snapshot_id':before.snapshot_id+1},path='/api/agent/observed_objects')
    assert stale.status_code==409 and len(received)==1


def test_runner_transports_model_authored_claims_before_done(tmp_path):
    import asyncio
    from core.agent.decision import ModelReply
    from harness_integration.test_zero_shot_agent import Environment, Agent
    from wia.runtime.agent.zero_shot import run_episode
    env=Environment()
    env.observation_declarations_available=True
    reports=[]
    async def report(objects,views,observation):
        reports.append((objects,views,observation.snapshot_id))
        return {'ok':True,'message':'Recorded.'}
    env.report_observed_objects=report
    payload={'command':{'action':'DONE'},'world_observation':'Cabinet.',
             'observed_objects':[{'label':'top cabinet','camera':'agentview_rgb'}]}
    agent=Agent(json.dumps(payload))
    from dataclasses import replace
    agent.prepare_request=lambda req:enable_declarations(replace(req,world_observation_log=True,assembled_prompt='fixture'))
    result=asyncio.run(run_episode(env,agent,tmp_path,max_steps=1))
    assert result['termination_reason']=='model_done'
    assert reports[0][0]==payload['observed_objects']
    assert 'agentview_rgb' in reports[0][1]


@pytest.mark.parametrize('distances,segmented,in_rooms', [
    ({'kitchen': (0., 1.), 'dining': (0., 2.)}, None, ['dining']),
    ({'kitchen': (0., 1.), 'dining': (0., 1.)}, 'kitchen', []),
    ({'kitchen': (2., 3.), 'dining': (1., 2.)}, 'kitchen', ['dining']),
    ({'kitchen': (2., 3.), 'dining': (1., 2.)}, None, ['dining']),
])
def test_region_assignment_matches_original_generator(monkeypatch, distances, segmented, in_rooms):
    import ast
    import math
    from wia.authoring.converters import cognitive_mapping_regional_boundary as region
    # Execute the original pure assignment function; keep its simulator query a fixture.
    nodes = ast.parse(region.GENERATOR.read_text()).body
    function = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == '_assign_object_region')
    scope = {'RegionRecord': NS, '_point_inside_bbox_xy': lambda point, box: box[0] == 0,
             '_distance_xy': math.dist, '_query_room_instance_by_point': lambda *a: segmented}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(region.GENERATOR), 'exec'), scope)
    records = [NS(room_instance=name, expanded_bbox_world_xy=[d, 0, 0, 0], center_xy=[c, 0])
               for name, (d, c) in distances.items()]
    expected, method, _ = scope['_assign_object_region']([0, 0], in_rooms, None, records)
    monkeypatch.setattr(region, '_anchor_state', lambda obj: {'centre_xy': [0, 0]})
    monkeypatch.setattr(region, '_query_region', lambda *a: segmented)
    monkeypatch.setattr(region, '_native_region_distances', lambda *a: {
        name: {'bbox_distance': d, 'center_distance': c} for name, (d, c) in distances.items()})
    config = {'operation': region.EXPLORATION_OPERATION, 'binding_keys': ['anchor'],
        'parameters': {'actual_capability': 'observed_anchor_region_relation',
            'source_task_type': 'object_in_region', 'target_regions': [expected]}}
    scene = NS(seg_map=NS(room_ins_id_to_ins_name={i: name for i, name in enumerate(distances)}))
    result = region._relation_evidence(NS(env=NS(scene=scene)), config, {'anchor': NS(in_rooms=in_rooms)})
    assert result['region_relation_consistent']
    assert result['native_anchor_assignment'][0]['method'] == method


from harness_integration.test_task_edit2_visual_evidence import world as visual_world


def test_actual_collector_masks_support_claims_without_semantic_audits(visual_world):
    world = visual_world
    collector = MappingObservationEvidence(world.session, {'bindings': {'anchor': 'target'},
        'parameters': {'public_anchor_labels': {'anchor': 'public target'}}})
    collector.capture_observation(world.observation)
    collector.begin_episode()
    native = collector.capture['evidence']
    assert native['view_evidence'] == [] and native['informative_available'] is False
    assert collector.available()
    collector.report([{'label': 'public target', 'camera': 'agentview_rgb'}], ['agentview_rgb'], world.observation)
    assert collector.receipts[0]['visible_pixels'] == 144
    assert collector.receipts[0]['informative'] is False
    assert collector.receipts[0]['binding'] == 'anchor'
    assert collector.receipts[0]['recognition_inferred_from_mask'] is False
