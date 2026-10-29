"""Exercise real Search initializer/qualifier/envelope hooks with native-shaped data.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m pytest \
  -q -p no:cacheprovider tests/test_task_edit2_search.py tests/test_task_edit2_search_runtime.py
These CPU measurements test integration contracts; they are not native GPU qualification.
"""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.authoring.converters import enumerative_perception_search as search
from wia.authoring.converters import task_edit2_search_layout as layout
from wia.authoring.converters import enumerative_perception_illumination_variability as lighting
from wia.runtime.scene_capture_queue import digest
from wia.native.esi_category_evaluation import (qualify_category_operation,
    create_category_operation_evaluator, CategoryEvaluationEnvelope)
from wia.tasks.scene import ScenePatches
from harness_integration.test_task_edit2_search import candidates



@pytest.fixture(autouse=True)
def native_floor_shape_fixture(monkeypatch):
    monkeypatch.setattr(layout, '_native_floor_ray', lambda start,end:[{
        'hit':True,'rigidBody':'/World/native_floor','normal':[0.,0.,1.],
        'position':[start[0],start[1],0.]}])
    from wia.tasks import native_floor_contacts
    monkeypatch.setattr(native_floor_contacts, 'native_vertical_ray',
                        lambda start,end,**kwargs: layout._native_floor_ray(start,end))
    monkeypatch.setattr(layout, '_invalidate_native_pose_cache', lambda session:
                        setattr(session, 'pose_cache_invalidations', getattr(session, 'pose_cache_invalidations', 0) + 1))

class Object:
    def __init__(self, name, category, position, *, model=None, scale=None,
                 orientation=None, bounds=None, visual_only=False, fixed_base=True, rooms=()):
        self.name, self.category, self.model = name, category, model
        self.position = np.array(position, dtype=float)
        self.orientation = np.array(orientation or [0.,0.,0.,1.])
        self.scale = np.array(scale if scale is not None else [1.,1.,1.])
        self._bbox_offset = (np.array(bounds) - self.position) if bounds is not None else None
        self.visual_only, self.fixed_base = visual_only, fixed_base
        self.in_rooms, self.prim_path = list(rooms), '/World/' + name
        self.prim = NS(owner=name)
        self.contacts = []
        self.pose_writes = []
        self.links = {'door': NS(visible=True)}

    def get_position_orientation(self):
        return self.position.copy(), self.orientation.copy()

    def set_position_orientation(self, *, position, orientation):
        self.position = np.asarray(position, dtype=float).copy()
        self.orientation = np.asarray(orientation, dtype=float).copy()
        self.pose_writes.append((self.position.copy(),self.orientation.copy()))

    @property
    def aabb(self):
        return self.position + self._bbox_offset if self._bbox_offset is not None else np.array([
            self.position-self.scale/2, self.position+self.scale/2])

    def contact_list(self):
        return self.contacts


def native_world(parent):
    world = parent['world_contract']
    objects = {}
    for row in world['joint_inventory']:
        spec, native = row['source_specification'], row['native']
        delta = np.array(native['position']) - spec['position']
        objects[native['name']] = Object(native['name'], native['category'], native['position'],
            model=native['model'], scale=native.get('scale'), orientation=native['orientation'],
            bounds=np.array(spec['bbox'])+delta, visual_only=True, rooms=[row['source_room_instance']])
    for row in layout.primitive_configs(world):
        table = next(t for t in world['curated_layout']['tables'] if row['name'] in {p['name'] for p in t['primitives']})
        objects[row['name']] = Object(row['name'], row['category'], row['position'],
            scale=row['scale'], rooms=[table['room_instance']])
    bounds = {}
    for group in world['search_groups']:
        room = next(r['source_payload']['room'] for r in world['source_provenance'] if r['source_identity'] == group['source_identity'])
        bounds[room] = group['source_render_contract']['room_bbox']['expanded_bbox_world_xy']
        for entry in group['mechanism_source_annotations'].get('source_entries', []):
            anchor = entry.get('metadata', {}).get('anchor_object')
            anchors = [anchor] if anchor else entry.get('metadata', {}).get('lighting', {}).get('target_lights', [])
            for anchor in anchors:
                lo, hi = np.array(anchor['bbox'])
                objects[anchor['name']] = Object(anchor['name'], anchor['category'], (lo+hi)/2,
                    bounds=[lo,hi], model='original_scene_asset', rooms=[room])
    floor = Object('native_floor', 'floors', [0.,0.,-.05], bounds=[[-100.,-100.,-.1],[100.,100.,0.]],
                   rooms=world['required_rooms'])
    objects[floor.name] = floor
    first_room = world['required_rooms'][0]
    first_bounds = np.array(bounds[first_room])
    robot = Object('r1', 'robot', [*first_bounds[:2]+.2,.4], scale=[.4,.4,.8], fixed_base=False, rooms=[first_room])
    def ground_contact(obj):
        return NS(body0=obj.prim_path+'/link', body1=floor.prim_path, normal=np.array([0.,0.,1.]), position=np.array([0.,0.,0.]))
    robot.contacts = [ground_contact(robot)]
    robot.floor_touching_base_link_names = ['wheel']
    robot.links['wheel'] = NS(prim_path=robot.prim_path+'/link')
    objects[robot.name] = robot
    for obj in objects.values():
        if '_leg_' in obj.name:
            obj.contacts = [ground_contact(obj)]
    microwave = Object('unrelated_microwave', 'microwave', [50.,50.,.5], model='native', rooms=[first_room])
    objects[microwave.name] = microwave
    def room_query(xy):
        for room, box in bounds.items():
            if box[0] <= xy[0] <= box[2] and box[1] <= xy[1] <= box[3]:
                return room
        return None
    scene = NS(objects=list(objects.values()), object_registry=lambda kind,name: objects.get(name),
               seg_map=NS(get_room_instance_by_point=room_query))
    sim = NS(current_time=0., callbacks={})
    sim.add_physics_callback = lambda name,callback: sim.callbacks.update({name:callback})
    session = NS(env=NS(scene=scene, robots=[robot]), robot=robot, scene_name=world['scene_model'],
        room_name=first_room, pipeline=NS(og=NS(sim=sim)), native_reset_id='reset_fixture_1',
        robot_camera_registry={'head': 'head_sensor', 'wrist': 'wrist_sensor'})
    return session, objects, sim


def observation(index=1):
    return NS(snapshot_id=index, action_step=index, rgb_images=lambda: {
        'head': np.full((20,20,3),100,np.uint8), 'wrist': np.full((20,20,3),50,np.uint8)})


def audits(session, world):
    context = {'world_digest': digest(world), 'reset_id': session.native_reset_id}
    common = {'schema': 'search_native_gate_audit_v1', 'context': context,
        'accepted': True, 'native_evidence': {'trace': 'controlled native fixture'},
        'reviewer': 'CPU fixture', 'reviewed_at': '2026-10-18'}
    session.enumerative_search_audits = [dict(common, gate=g) for g in (
        'loaded_routes', 'mounted_camera_audit', 'legal_informative_access')]
    session.enumerative_search_audits.extend(dict(common, group_id=g['id']) for g in world['search_groups'])


def collector(parent, observation_id=1):
    p = parent['task_definition']['private']['evaluation']['parameters']
    rows, declarations = {}, []
    for i, target in enumerate(p['targets']):
        left = .01 + i*.06
        region = [left,.1,left+.04,.3]
        rows[target['binding']] = {'native_name': target['binding'], 'snapshot_id': observation_id,
            'camera': 'head', 'native_rgb_matches_delivered': True, 'identity_synchronized': True,
            'measurement_available': True, 'informative': True, 'informative_feature_evidence': {'feature_ids':['bread_and_filling']},
            'visit_verified': True, 'visit_evidence': {'native_location_verified': True},
            'room': target['room'], 'area': target['area'], 'bbox_normalized': region}
        declarations.append({'group_id':target['group_id'], 'identity':target['expected_identity'],
            'room':target['room'], 'area':target['area'], 'snapshot_id':observation_id,
            'camera':'head', 'image_point':[left+.02,.2]})
    return {'available': True, 'objects': rows, 'all_delivered_views_audited': True}, declarations


def config(parent):
    task = parent['task_definition']
    value = deepcopy(task['private']['evaluation'])
    value.update(scene=parent['world_contract']['scene_model'], room=parent['world_contract']['required_rooms'][0],
        bindings={key:row['name'] for key,row in task['private']['bindings'].items()})
    return value


def test_exact_pair_vectors_cluster_overlap_and_station_footprints(tmp_path):
    parent = candidates(tmp_path)[0]
    world = parent['world_contract']
    assert len(world['curated_layout']['tables']) == 3
    rows = {row['native']['name']:row for row in world['joint_inventory']}
    dependency = world['curated_layout']['dependency_groups']
    pairs = [g for g in dependency if g['kind']=='exact_target_confuser_pair']
    cluster, = [g for g in dependency if g['kind']=='unchanged_compact_cluster']
    assert len(pairs) == 4
    assert sorted([sum(g['table']==t['name'] for g in pairs) for t in world['curated_layout']['tables'] if 'ambiguity' in t['name']]) == [2,2]
    for group in pairs+[cluster]:
        first = rows[group['members'][0]]
        for name in group['members'][1:]:
            row = rows[name]
            actual = np.array(row['native']['position'])-first['native']['position']
            original = np.array(row['source_specification']['position'])-first['source_specification']['position']
            assert np.allclose(actual,original,atol=1e-12)
            assert row['native']['orientation']==row['source_specification']['quaternion_xyzw']
    assert len(cluster['members']) == 4
    session, _, _ = native_world(parent)
    search.postprocess_joint_world(session, {'world_contract': world})
    measured = layout.native_structure(session, world)
    assert measured['available']
    assert all(measured['checks'].values()), {k:v for k,v in measured['checks'].items() if not v}
    assert not measured['target_support_contact_claimed']
    assert not measured['graspability_claimed']


def test_source_postprocess_override_resolves_names_and_never_hides_unrelated_microwave(tmp_path):
    parent = candidates(tmp_path)[0]
    session, objects, sim = native_world(parent)
    patches = ScenePatches.__new__(ScenePatches)
    patches._realized = parent['scene_definition']['scene']['variants']['adapted']['patches']
    patches.applied = []
    result = patches.source_postprocess_override(session)
    assert result['joint_world_postprocess']
    assert result['source_hidden_link_operations'] == 0
    assert result['physics_steps'] == 0
    assert objects['unrelated_microwave'].links['door'].visible
    assert set(result['human_framing_names']) == set(session._search_initial_native)
    assert all(name.startswith('search_') for name in result['human_framing_names'])
    with pytest.raises(ValueError,match='initialization-only'):
        patches.source_postprocess_override(session)


def test_real_qualifier_and_category_envelope_entrypoints(tmp_path):
    parent = candidates(tmp_path)[0]
    world = parent['world_contract']; session, objects, _ = native_world(parent)
    search.postprocess_joint_world(session, {'world_contract':world})
    runtime = config(parent)
    bound = {key:objects[name] for key,name in runtime['bindings'].items()}
    executor = NS(capabilities=NS(base_translation=True,base_rotation=True))
    result = qualify_category_operation(session,executor,runtime,bound)
    assert not result['accepted']  # Static native data cannot manufacture route/feature audits.
    audits(session,world)
    result = qualify_category_operation(session,executor,runtime,bound)
    assert result['accepted'], result
    visual, declarations = collector(parent)
    session.task_visual_evidence = lambda *args:deepcopy(visual)
    native = create_category_operation_evaluator(session,executor,runtime,tmp_path/'native')
    task = NS(definition=NS(id='fixture',repair={}), task_text='immutable search mission',
              file_sha256='task_sha',source_sha256='source_sha',verify_unchanged=lambda:None)
    envelope = CategoryEvaluationEnvelope(native,task,tmp_path/'envelope')
    initial = envelope.begin_episode(observation())
    assert initial['native_result']['action_step']==1
    envelope.report_answer({'declarations':declarations},observation())
    result = envelope.record(observation(),action=None)
    assert result['task_success']
    assert result['native_result']['discovery_percent']==100
    assert result['progress']['score_percent']==100
    # A later background movement remains a factual native constraint failure.
    objects['unrelated_microwave'].position[0] += .2
    result = envelope.record(observation(),action=None)
    assert not result['task_success']
    assert result['native_result']['constraint_compliance'] is False


def test_missing_actor_floor_or_native_measurement_rejects_qualification(tmp_path):
    parent=candidates(tmp_path)[0]; world=parent['world_contract']
    session,objects,_=native_world(parent)
    search.postprocess_joint_world(session,{'world_contract':world})
    runtime=config(parent)
    bound={key:objects[name] for key,name in runtime['bindings'].items()}
    executor=NS(capabilities=NS(base_translation=True,base_rotation=True))
    audits(session,world)
    objects['r1'].contacts=[]
    result=qualify_category_operation(session,executor,runtime,bound)
    assert not result['accepted']
    assert result['checks']['floor_supported_start'] is False
    actor=next(iter(world['joint_inventory']))['native']['name']
    objects.pop(actor)
    result=qualify_category_operation(session,executor,runtime,bound)
    assert not result['accepted']
    assert result['native_structure']['available'] is False


def test_missing_or_malformed_visual_provider_and_stale_gate_receipts_stay_unavailable(tmp_path):
    parent=candidates(tmp_path)[0];world=parent['world_contract']
    session,_,_=native_world(parent)
    search.postprocess_joint_world(session,{'world_contract':world})
    runtime=config(parent);p=runtime['parameters']
    value=search.measure_search_evidence(session,observation(),runtime['bindings'],p)
    assert not value['available'] and 'adapter unavailable' in value['measurement_error']
    session.task_visual_evidence=lambda *args:[]
    value=search.measure_search_evidence(session,observation(),runtime['bindings'],p)
    assert not value['available']
    audits(session,world)
    session.native_reset_id='different_reset'
    visual,_=collector(parent)
    session.task_visual_evidence=lambda *args:visual
    value=search.measure_search_evidence(session,observation(),runtime['bindings'],p)
    assert not value['gates']['loaded_routes']
    assert not value['mechanisms']


class Attribute:
    def __init__(self,value):
        self.value=value;self.writes=[]
    def Get(self):
        return self.value
    def Set(self,value):
        self.value=value;self.writes.append(value)


def test_pilot_cycle_changes_real_attributes_by_sim_time_and_restores_native_baselines(tmp_path,monkeypatch):
    parent=candidates(tmp_path)[1];world=parent['world_contract']
    session,objects,sim=native_world(parent)
    lights={}
    for name,obj in objects.items():
        if any(obj.name == lamp['name'] for group in world['search_groups']
               if group['mechanism'] == 'Illumination Variability'
               for entry in group['mechanism_source_annotations']['source_entries']
               for lamp in entry['metadata']['lighting']['target_lights']):
            lights[name]=[{'path':obj.prim_path+'/source_light','type_name':'SphereLight',
                'baseline':100.,'attribute':Attribute(100.),'transform':np.eye(4)}]
    monkeypatch.setattr(lighting,'_stage_lights',lambda root:iter(lights.get(root.owner,[])))
    monkeypatch.setattr(lighting,'_light_world_transform',lambda light:light['transform'])
    receipt=search.postprocess_joint_world(session,{'world_contract':world})
    assert receipt['lighting']['available'], receipt['lighting']
    assert len(sim.callbacks)==1
    cycle=session._search_light_cycle
    assert cycle.measure()['measured_phase']=='normal'
    sim.current_time=10.
    for callback in sim.callbacks.values():callback(.033)
    cycle.sample() # Same physics time: no double progression.
    assert cycle.measure()['measured_phase']=='changed'
    assert cycle.native.elapsed_s==10.
    assert all(light['attribute'].Get()==10. for rows in lights.values() for light in rows)
    sim.current_time=20.
    cycle.sample()
    assert cycle.measure()['measured_phase']=='normal'
    assert all(light['attribute'].Get()==100. for rows in lights.values() for light in rows)
    sim.current_time=30.
    cycle.sample()
    result=search.restore_search_lighting(session)
    assert result['restored_original_intensities']
    assert all(light['attribute'].Get()==100. for rows in lights.values() for light in rows)
    assert layout.PILOT_LIGHTING['changed_scale']==.10
    assert layout.PILOT_LIGHTING['phase_duration_s']==10.


def test_lighting_group_needs_informative_target_receipts_in_both_measured_phases(tmp_path,monkeypatch):
    parent=candidates(tmp_path)[1];runtime=config(parent)
    evaluator=search.SearchEvaluator(NS(),None,runtime,tmp_path/'evaluation')
    visual,declarations=collector(parent)
    rows=[{**row,'location_verified':True,'image_region':row['bbox_normalized']} for row in visual['objects'].values()]
    evidence={'available':True,'gates':dict.fromkeys(search.NATIVE_GATES,True),
        'mechanisms':{g['id']:True for g in runtime['parameters']['groups']},
        'observations':rows,'lighting':{'measured_phase':'normal'}}
    evaluator._measure=lambda observation:deepcopy(evidence)
    evaluator.begin_episode(observation())
    result=evaluator.report_answer({'declarations':declarations},observation())
    assert not result['task_success']
    assert result['discovery_percent'] is None
    evidence['lighting']['measured_phase']='changed'
    for row in evidence['observations']:row['snapshot_id']=2
    result=evaluator.record(observation(2))
    assert result['task_success']
    assert result['discovery_percent']==100


def test_fixed_table_floor_geometry_is_separate_from_unknown_static_contact_stream(monkeypatch):
    floor=Object('native_floor','floors',[0.,0.,-.05],bounds=[[-1.,-1.,-.1],[1.,1.,0.]])
    foot=Object('table_leg_0','breakfast_table',[0.,0.,.35],scale=[.045,.045,.7])
    receipt=layout.geometric_floor_foot_support(foot,[floor],[foot.prim_path])
    assert receipt['geometric_floor_foot_support']
    assert receipt['physical_floor_contact_stream'] is None
    assert receipt['touching_state_claimed'] is False
    foot.position[2]+=.01
    receipt=layout.geometric_floor_foot_support(foot,[floor],[foot.prim_path])
    assert not receipt['geometric_floor_foot_support']
    foot.position[2]-=.01
    monkeypatch.setattr(layout,'_native_floor_ray',lambda start,end:[
        {'hit':True,'rigidBody':floor.prim_path,'normal':[0.,0.,1.],'position':[start[0],start[1],0.]},
        {'hit':True,'rigidBody':'/World/obstacle','normal':[0.,0.,1.],'position':[start[0],start[1],.002]}])
    receipt=layout.geometric_floor_foot_support(foot,[floor],[foot.prim_path])
    assert not receipt['geometric_floor_foot_support']
    monkeypatch.setattr(layout,'_native_floor_ray',lambda start,end:[])
    receipt=layout.geometric_floor_foot_support(foot,[floor],[foot.prim_path])
    assert not receipt['geometric_floor_foot_support']


def raw_native_cells(session, points):
    points = np.asarray(points, dtype=float)
    session.env.scene.trav_map = NS(floor_map=[np.full((1,len(points)),255,np.uint8)],
        floor_heights=[0.], map_resolution=.1,
        map_to_world=lambda cells:points[np.asarray(cells,dtype=int)[:,1]])


def add_obstacle(session, objects, table, name='native_wardrobe', category='wardrobe'):
    xy = np.asarray(table['centre_xy_m'])
    obstacle = Object(name,category,[xy[0]+table['dimensions_xy_m'][0]/2-.1,xy[1],.5],
        scale=[.1,.4,1.],rooms=[table['room_instance']])
    objects[name] = obstacle
    session.env.scene.objects.append(obstacle)
    return obstacle


@pytest.mark.parametrize('table_index',[0,2])
def test_measured_table_translation_preserves_full_source_groups_and_protected_background(tmp_path,table_index):
    parent=candidates(tmp_path)[0];world=parent['world_contract'];original=deepcopy(world)
    session,objects,_=native_world(parent)
    table=world['curated_layout']['tables'][table_index]
    obstacle_name='native_wardrobe' if table_index==0 else 'native_car'
    add_obstacle(session,objects,table,name=obstacle_name,category='wardrobe' if table_index==0 else 'car')
    centre=np.asarray(table['centre_xy_m']);free=centre+[-.3,0.]
    raw_native_cells(session,[centre,free,centre+[-.6,0.]])
    before={name:deepcopy(layout.pose(obj)) for name,obj in objects.items()}
    receipt=search.postprocess_joint_world(session,{'world_contract':world})
    placement=receipt['native_table_layout_initialization'];group=placement['groups'][table_index]
    assert placement['accepted'] and group['accepted'],group
    delta=np.array([-.3,0.,-table['floor_proposal_z_m']])
    assert group['accepted_delta_m']==pytest.approx(delta)
    assert group['first_rejection']['failure_stage']=='clear_other_native_bounds'
    assert group['first_rejection']['screen']['native_bounds_obstacles'][0]['name']==obstacle_name
    assert group['coverage']['raw_cell_rule']=='native_floor_map_equals_255_without_erosion'
    assert group['attempts'][-1]['geometric_floor_foot_support']
    assert group['attempts'][-1]['protected_native_objects_unchanged']
    runtime=session._search_world_contract
    assert world==original
    assert digest(runtime)!=digest(world)
    assert session._search_world_contract_binding=={'original_world_digest':digest(world),
        'reset_id':session.native_reset_id,'runtime_layout_digest':digest(runtime)}
    moved=set(group['original_proposed_native_poses'])
    for name,obj in objects.items():
        old=before[name]
        if name in moved:
            assert np.allclose(obj.position,np.asarray(old['position'])+delta,atol=2e-6)
            assert np.array_equal(obj.orientation,old['orientation'])
            assert np.array_equal(obj.scale,old['scale'])
        else:
            assert layout.pose(obj)==old
            assert not obj.pose_writes
    old_rows={row['native']['name']:row for row in original['joint_inventory']}
    for row in runtime['joint_inventory']:
        assert row['source_specification']==old_rows[row['native']['name']]['source_specification']
    for dependency in runtime['curated_layout']['dependency_groups']:
        if dependency.get('table')!=table['name']:
            continue
        first=objects[dependency['members'][0]]
        source_first=old_rows[first.name]['source_specification']['position']
        for name in dependency['members'][1:]:
            assert np.allclose(objects[name].position-first.position,
                np.asarray(old_rows[name]['source_specification']['position'])-source_first,atol=3e-6)
    measured=layout.native_structure(session,world)
    assert measured['available'] and all(measured['checks'].values()),measured
    assert measured['world_digest']==digest(original)
    assert measured['runtime_layout_digest']==digest(runtime)
    assert not placement['access_qualified'] and not placement['informative_images_qualified']
    assert not placement['graspability_claimed']


def test_failed_actual_foot_screen_restores_proposal_and_records_bounded_rejections(tmp_path,monkeypatch):
    parent=candidates(tmp_path)[0];world=parent['world_contract'];session,objects,_=native_world(parent)
    table=world['curated_layout']['tables'][0];add_obstacle(session,objects,table)
    centre=np.asarray(table['centre_xy_m']);raw_native_cells(session,[centre,centre+[-.3,0.]])
    before={name:deepcopy(layout.pose(obj)) for name,obj in objects.items()}
    original=layout.geometric_floor_foot_support
    def failed_moved_foot(foot,floors,paths):
        value=original(foot,floors,paths)
        if foot.name.startswith(table['name']) and abs(foot.position[0]-before[foot.name]['position'][0])>.1:
            value['geometric_floor_foot_support']=False
            value['fixture_native_rejection']='actual_floor_gap'
        return value
    monkeypatch.setattr(layout,'geometric_floor_foot_support',failed_moved_foot)
    result=search.postprocess_joint_world(session,{'world_contract':world})
    placement=result['native_table_layout_initialization'];group=placement['groups'][0]
    assert not placement['accepted'] and not group['accepted']
    assert group['attempts'][-1]['failure_stage']=='actual_table_bounds_floor_feet_or_foreign_contact'
    assert group['coverage']['raw_map_exhausted']
    assert not group['coverage']['failure_certifies_impossibility']
    for name,obj in objects.items():
        assert np.allclose(obj.position,before[name]['position'],atol=2e-6)
        assert np.allclose(obj.orientation,before[name]['orientation'])
    measured=layout.native_structure(session,world)
    assert measured['available']
    assert measured['checks']['native_table_layout_initialization'] is False
    audits(session,world)
    runtime=config(parent)
    bound={key:objects[name] for key,name in runtime['bindings'].items()}
    executor=NS(capabilities=NS(base_translation=True,base_rotation=True))
    qualified=qualify_category_operation(session,executor,runtime,bound)
    assert qualified['accepted'] is False
    assert qualified['checks']['native_table_layout_initialization'] is False
    evidence=search.measure_search_evidence(session,observation(),runtime['bindings'],runtime['parameters'])
    assert evidence['gates']['source_relations'] is False


def test_candidate_cap_is_explicit_and_does_not_claim_no_placement_exists(tmp_path,monkeypatch):
    parent=candidates(tmp_path)[0];world=parent['world_contract'];session,objects,_=native_world(parent)
    table=world['curated_layout']['tables'][0];add_obstacle(session,objects,table)
    centre=np.asarray(table['centre_xy_m'])
    raw_native_cells(session,[centre+[offset,0.] for offset in (0.,.01,.02,.03,-.3)])
    monkeypatch.setattr(layout,'TABLE_CANDIDATE_CAP',3)
    result=search.postprocess_joint_world(session,{'world_contract':world})
    group=result['native_table_layout_initialization']['groups'][0]
    assert not group['accepted']
    assert len(group['attempts'])==group['coverage']['candidate_cap']==3
    assert not group['coverage']['raw_map_exhausted']
    assert group['coverage']['failure_certifies_impossibility'] is False


@pytest.mark.parametrize('failure',['reset','original_world','runtime_digest','receipt_digest'])
def test_runtime_layout_and_native_receipts_cannot_cross_world_or_reset(tmp_path,failure):
    parent=candidates(tmp_path)[0];world=parent['world_contract'];session,objects,_=native_world(parent)
    search.postprocess_joint_world(session,{'world_contract':world});audits(session,world)
    requested=deepcopy(world)
    if failure=='reset':session.native_reset_id='another_reset'
    elif failure=='original_world':requested['required_routes']='another immutable source contract'
    elif failure=='runtime_digest':session._search_world_contract['load_policy']='changed after initialization'
    else:session._search_table_layout_initialization['runtime_layout_digest']='stale'
    value=layout.native_structure(session,requested)
    assert value['available'] is False
    assert search._gate_audits(session,requested)==({}, {})
    parameters={**config(parent)['parameters'],'world_contract':requested}
    visual,_=collector(parent);session.task_visual_evidence=lambda *args:visual
    evidence=search.measure_search_evidence(session,observation(),config(parent)['bindings'],parameters)
    assert not evidence['available'] and not evidence['observations']


def test_original_valid_proposal_uses_zero_offset_without_native_search_or_pose_writes(tmp_path):
    parent=candidates(tmp_path)[0];session,objects,_=native_world(parent)
    receipt=search.postprocess_joint_world(session,{'world_contract':parent['world_contract']})
    assert receipt['native_table_layout_initialization']['accepted']
    for group in receipt['native_table_layout_initialization']['groups']:
        assert group['accepted_delta_m']==[0.,0.,0.]
        assert len(group['attempts'])==1
        assert group['coverage']['raw_cells_scanned']==0
    assert not any(obj.pose_writes for obj in objects.values())


@pytest.mark.parametrize('failure',['tilted_normal','noncoplanar','no_floor'])
def test_missing_or_inconsistent_native_floor_shape_never_places_table(tmp_path,monkeypatch,failure):
    parent=candidates(tmp_path)[0];world=parent['world_contract'];session,objects,_=native_world(parent)
    before={name:obj.position.copy() for name,obj in objects.items()}
    def ray(start,end):
        if failure=='no_floor':return []
        height=.01*start[0] if failure=='noncoplanar' else 0.
        return [{'hit':True,'rigidBody':'/World/native_floor','position':[start[0],start[1],height],
                 'normal':[.1,0.,.99] if failure=='tilted_normal' else [0.,0.,1.]}]
    monkeypatch.setattr(layout,'_native_floor_ray',ray)
    result=search.postprocess_joint_world(session,{'world_contract':world})
    placement=result['native_table_layout_initialization']
    assert not placement['accepted']
    assert placement['groups'][0]['first_rejection']['failure_stage']=='native_horizontal_coplanar_floor_shape_measurement'
    assert not any(obj.pose_writes for obj in objects.values())
    for name,obj in objects.items():np.testing.assert_array_equal(obj.position,before[name])


def test_native_table_relocation_cannot_repeat_or_run_after_initial_observation(tmp_path):
    parent=candidates(tmp_path)[0];world=parent['world_contract'];session,_,_=native_world(parent)
    layout.initialize_search_tables(session,world)
    first=deepcopy(session._search_table_layout_initialization)
    with pytest.raises(ValueError,match='replay is forbidden'):
        layout.initialize_search_tables(session,world)
    assert session._search_table_layout_initialization==first
    session,_objects,_sim=native_world(parent)
    session._observation_source=NS(initial_observation=observation())
    with pytest.raises(ValueError,match='initialization-only'):
        layout.initialize_search_tables(session,world)


def test_missing_measured_wheel_floor_api_remains_unavailable_not_nominal_support(tmp_path):
    parent=candidates(tmp_path)[0];world=parent['world_contract'];session,objects,_=native_world(parent)
    search.postprocess_joint_world(session,{'world_contract':world})
    del objects['r1'].floor_touching_base_link_names
    native=layout.native_structure(session,world)
    assert native['available']
    assert native['checks']['floor_supported_start'] is None
    assert native['robot_contacts']['measurement_available'] is False


class PreControllerSession(NS):
    @property
    def observation_source(self):
        raise AssertionError('Pre-controller initialization must not access the lazy observation source.')


def test_pre_controller_table_initialization_never_reads_lazy_source_or_needs_bound_robot(tmp_path):
    parent=candidates(tmp_path)[0];world=parent['world_contract'];base,objects,_=native_world(parent)
    session=PreControllerSession(**vars(base));session.robot=None;session._observation_source=None
    result=search.postprocess_joint_world(session,{'world_contract':world})
    assert result['native_table_layout_initialization']['accepted']
    assert session.robot is None and session.env.robots[0] is objects['r1']
    assert session._observation_source is None and 'observation_source' not in vars(session)
    assert layout.native_structure(session,world)['available']


@pytest.mark.parametrize('slot',['_observation_source','observation_source'])
def test_pre_controller_guard_reads_retained_sources_only_from_session_state(tmp_path,slot):
    parent=candidates(tmp_path)[0];base,objects,_=native_world(parent)
    session=PreControllerSession(**vars(base));session.robot=None
    vars(session)[slot]=NS(initial_observation=observation())
    with pytest.raises(ValueError,match='initialization-only'):
        layout.initialize_search_tables(session,parent['world_contract'])
    assert '_search_table_layout_initialization' not in vars(session)
    assert not any(obj.pose_writes for obj in objects.values())


@pytest.mark.parametrize('owner',['session','controller','task_controller','task_evaluator','source_provider'])
@pytest.mark.parametrize('active',[True,False])
def test_table_initialization_rejects_recorded_public_actions_even_without_initial_view(tmp_path,owner,active):
    parent=candidates(tmp_path)[0];base,objects,_=native_world(parent)
    session=PreControllerSession(**vars(base));session.robot=None
    controller=NS(_public_action_context={'sequence':1,'active':active})
    if owner=='session':session._public_action_context=controller._public_action_context
    elif owner=='task_evaluator':session.task_evaluator=NS(controller=controller)
    elif owner=='source_provider':session._observation_source=NS(controller_state_provider=NS(__self__=controller))
    else:setattr(session,owner,controller)
    with pytest.raises(ValueError,match='initialization-only'):
        layout.initialize_search_tables(session,parent['world_contract'])
    assert '_search_table_layout_initialization' not in vars(session)
    assert not any(obj.pose_writes for obj in objects.values())


def test_search_v3_task_and_scene_include_after_postprocessing_start_hook(tmp_path):
    from wia.tasks.loader import task_definition
    from wia.tasks.schema import SceneDefinition
    parent=candidates(tmp_path)[0]
    task_definition(parent['task_definition']);SceneDefinition.model_validate(parent['scene_definition'])
    assert search.REVISION=='task_edit2_search_20261018_v3'
    assert parent['task_definition']['private']['evaluation']['parameters']['task_revision']==search.REVISION
    task_patches=parent['task_definition']['scene']['variants']['adapted']['patches']
    scene_patches=parent['scene_definition']['scene']['variants']['adapted']['patches']
    for patches in (task_patches,scene_patches):
        hooks=[row for row in patches if row['operation']=='category_initialization']
        assert [row['phase'] for row in hooks]==['before_construction','after_postprocessing']
        assert hooks[1]['module']==search.MODULE
        assert hooks[1]['parameters']=={'operation':search.START_INITIALIZATION,'world_contract':parent['world_contract']}


@pytest.mark.parametrize('accepted',[True,False])
def test_after_postprocessing_hook_calls_start_with_original_world_once_and_preserves_result(tmp_path,monkeypatch,accepted):
    from wia.tasks import search_start
    parent=candidates(tmp_path)[0];world=parent['world_contract'];session,_,_=native_world(parent)
    search.postprocess_joint_world(session,{'world_contract':world})
    calls=[];receipt={'accepted':accepted,'reason':None if accepted else 'native_start_screen_rejected',
        'pending_gates':['loaded_routes','informative_view'],'route_or_manipulation_certificate':False}
    def initialize(actual_session,original):
        calls.append((actual_session,original))
        actual_session.search_start_initialization=deepcopy(receipt)
        return actual_session.search_start_initialization
    monkeypatch.setattr(search_start,'initialize_search_start',initialize)
    patches=ScenePatches.__new__(ScenePatches)
    patches._realized=deepcopy(parent['scene_definition']['scene']['variants']['adapted']['patches'])
    patches._before_applied=True;patches._after_applied=False;patches.applied=[]
    patches.apply_after_postprocessing(session)
    result=patches.applied[-1]['initialization_evidence']
    assert result==receipt and len(calls)==1
    assert calls[0][0] is session and calls[0][1]==world
    assert calls[0][1] is not session._search_world_contract
    assert not result['route_or_manipulation_certificate']
    with pytest.raises(ValueError,match='cannot be repeated'):
        search.initialize_scene(session,{'operation':search.START_INITIALIZATION,'world_contract':world})
    assert len(calls)==1


@pytest.mark.parametrize('missing',['operation','postprocess','robot','baseline','background','binding','stale_reset'])
def test_search_start_hook_requires_exact_operation_and_completed_bound_postprocess(tmp_path,monkeypatch,missing):
    from wia.tasks import search_start
    parent=candidates(tmp_path)[0];world=parent['world_contract'];session,_,_=native_world(parent)
    search.postprocess_joint_world(session,{'world_contract':world})
    parameters={'operation':search.START_INITIALIZATION,'world_contract':world}
    if missing=='operation':parameters['operation']='unowned_operation'
    elif missing=='postprocess':session._search_joint_postprocessed=False
    elif missing=='robot':session.robot=None
    elif missing=='baseline':del session._search_initial_native
    elif missing=='background':del session._search_background_initial
    elif missing=='binding':del session._search_world_contract_binding
    else:session.native_reset_id='another_reset'
    def forbidden(*args):raise AssertionError('The native start helper must not run without its prerequisites.')
    monkeypatch.setattr(search_start,'initialize_search_start',forbidden)
    with pytest.raises(ValueError):search.initialize_scene(session,parameters)
    assert not vars(session).get('_search_start_hook_called')


def test_archived_v2_evaluator_retains_shared_native_measurement_path(tmp_path,monkeypatch):
    parent=candidates(tmp_path)[0];configuration=config(parent)
    configuration['parameters']['task_revision']='task_edit2_search_20261018_v2'
    marker={'available':False,'measurement_error':'archived v2 native measurement'}
    monkeypatch.setattr(search,'measure_search_evidence',lambda *args:marker)
    evaluator=search.SearchEvaluator(NS(),None,configuration,tmp_path/'v2')
    assert evaluator._measure(observation()) is marker
    assert evaluator.record(observation())['schema']=='task_edit2_search_20261018_v2'


def screen_fixture():
    top=Object('table','breakfast_table',[0.,0.,.7],scale=[1.,1.,.1])
    prop=Object('prop','ball',[0.,0.,.8],scale=[.1,.1,.1],visual_only=True)
    wall=Object('wall','walls',[.4,0.,.5],scale=[.1,.4,1.])
    floor=Object('floor','floors',[0.,0.,-.05],scale=[10.,10.,.1])
    decor=Object('decor','picture',[0.,0.,.5],scale=[.1,.1,.1],visual_only=True)
    other_task=Object('other_task','ball',[5.,5.,.5],scale=[.1,.1,.1],visual_only=True)
    objects=[top,prop,wall,floor,decor,other_task]
    session=NS(env=NS(scene=NS(objects=objects,seg_map=NS(get_room_instance_by_point=lambda xy:'room'))),
        native_reset_id='reset_1',pipeline=NS(og=NS(sim=NS(current_time=2.5))))
    table={'name':'table','room_instance':'room','primitives':[{'name':'table'}]}
    rows={obj.name:layout.pose(obj) for obj in (top,prop)}
    return session,table,rows,{'prop','other_task'},{'table','prop'},wall


def count_native_aabb_reads(monkeypatch):
    reads={}
    original=Object.aabb.fget
    def measured(obj):
        reads[obj.name]=reads.get(obj.name,0)+1
        return original(obj)
    monkeypatch.setattr(Object,'aabb',property(measured))
    return reads


def test_proposal_snapshot_reads_eligible_bounds_once_and_matches_fresh_screens(monkeypatch):
    session,table,rows,actors,assembly,wall=screen_fixture()
    reads=count_native_aabb_reads(monkeypatch)
    deltas=[np.array([x,0.,0.]) for x in (0.,.01,.02,.03,.04)]
    fresh=[layout._table_screen(session,table,rows,delta,actors,assembly,0.) for delta in deltas]
    assert reads=={'wall':5,'other_task':5}
    reads.clear()
    snapshot=layout._snapshot_proposal_bounds(session,actors,assembly)
    assert reads=={'wall':1,'other_task':1}
    cached=[layout._table_screen(session,table,rows,delta,actors,assembly,0.,other_bounds_snapshot=snapshot)
            for delta in deltas]
    assert cached==fresh and all(not row['accepted'] for row in cached)
    assert reads=={'wall':1,'other_task':1}
    report=snapshot.report()
    assert report['native_AABB_measurement_read_count']==2
    assert report['reset_id']=='reset_1' and report['native_time_s']==2.5
    assert report['eligible_names']==['wall','other_task']
    assert set(report['excluded_names'])=={'table','prop','floor','decor'}
    assert report['postplacement_acceptance_uses_fresh_native_bounds']
    report['bounds'][0]['aabb_m'][0][0]=999.
    assert snapshot.report()['bounds'][0]['aabb_m'][0][0]!=999.
    with pytest.raises(AttributeError):snapshot.native_time_s=3.


@pytest.mark.parametrize('changed',['reset','time','actors','assembly'])
def test_proposal_snapshot_refuses_changed_reset_time_or_exclusions(changed):
    session,table,rows,actors,assembly,_=screen_fixture()
    snapshot=layout._snapshot_proposal_bounds(session,actors,assembly)
    if changed=='reset':session.native_reset_id='other_reset'
    elif changed=='time':session.pipeline.og.sim.current_time+=.001
    elif changed=='actors':actors.add('decor')
    else:assembly.add('wall')
    with pytest.raises(ValueError,match='snapshot reset/time or actor exclusions changed'):
        layout._table_screen(session,table,rows,np.zeros(3),actors,assembly,0.,other_bounds_snapshot=snapshot)
    # Default callers, including native_structure, keep their fresh-read path.
    assert 'accepted' in layout._table_screen(session,table,rows,np.zeros(3),actors,assembly,0.)


def test_snapshot_capture_refuses_a_native_time_change_during_reads(monkeypatch):
    session,_,_,actors,assembly,wall=screen_fixture()
    original=Object.aabb.fget
    def advance(obj):
        if obj is wall:session.pipeline.og.sim.current_time+=.01
        return original(obj)
    monkeypatch.setattr(Object,'aabb',property(advance))
    with pytest.raises(ValueError,match='snapshot reset/time'):
        layout._snapshot_proposal_bounds(session,actors,assembly)


def test_multiple_initializer_rejections_reuse_snapshot_without_native_bounds_reads(tmp_path,monkeypatch):
    parent=candidates(tmp_path)[0];world=parent['world_contract'];session,objects,_=native_world(parent)
    table=world['curated_layout']['tables'][0];add_obstacle(session,objects,table)
    centre=np.asarray(table['centre_xy_m'])
    raw_native_cells(session,[centre+[offset,0.] for offset in (0.,.01,.02,.03)])
    monkeypatch.setattr(layout,'TABLE_CANDIDATE_CAP',4)
    reads=count_native_aabb_reads(monkeypatch)
    original=layout._table_screen;observed=[]
    def screen(*args,**kwargs):
        before=reads.copy();result=original(*args,**kwargs)
        if args[1]['name']==table['name']:
            observed.append((kwargs.get('other_bounds_snapshot') is not None,before,reads.copy()))
        return result
    monkeypatch.setattr(layout,'_table_screen',screen)
    result=layout.initialize_search_tables(session,world)[1]['groups'][0]
    assert not result['accepted'] and len(result['attempts'])==4
    assert len(observed)==4 and all(cached and before==after for cached,before,after in observed)
    proof=result['proposal_bounds_snapshot']
    assert proof['native_AABB_measurement_read_count']==len(proof['eligible_names'])
    assert proof['reset_id']==session.native_reset_id and proof['native_time_s']==0.
    assert [row['candidate_index'] for row in result['attempts']]==[0,1,2,3]


def test_acceptance_screen_reads_fresh_foreign_geometry_after_assembly_setters(tmp_path,monkeypatch):
    parent=candidates(tmp_path)[0];world=parent['world_contract'];session,objects,_=native_world(parent)
    table=world['curated_layout']['tables'][0];obstacle=add_obstacle(session,objects,table)
    centre=np.asarray(table['centre_xy_m']);raw_native_cells(session,[centre,centre+[-.3,0.]])
    old_set=layout._set_assembly;changed=False
    def relocate(session_arg,assembly,baseline,delta):
        nonlocal changed
        old_set(session_arg,assembly,baseline,delta)
        if table['name'] in assembly and np.any(delta!=0) and not changed:
            obstacle.position[:2]+=np.asarray(delta)[:2]
            changed=True
    monkeypatch.setattr(layout,'_set_assembly',relocate)
    _,receipt=layout.initialize_search_tables(session,world)
    group=receipt['groups'][0]
    assert changed and not group['accepted'] and not receipt['accepted']
    placed=[trial for trial in group['attempts'] if 'actual_native_screen' in trial]
    assert placed and all(trial['screen']['accepted'] for trial in placed)
    assert all(not trial['actual_native_screen']['checks']['clear_other_native_bounds'] for trial in placed)
    assert obstacle.name in {row['name'] for trial in placed for row in trial['actual_native_screen']['native_bounds_obstacles']}
    assert all(trial['protected_native_objects_unchanged'] is False for trial in placed)
    assert group['proposal_bounds_snapshot']['postplacement_acceptance_uses_fresh_native_bounds']
