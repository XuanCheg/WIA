"""CPU tests of complete, read-only native support-space diagnostics.

Run: python -B -m pytest -q -p no:cacheprovider
 tests/test_task_edit2_support_diagnostics.py tests/test_task_edit2_delivery_centres.py.
Maps and native-shaped AABBs are synthetic; no native process or GPU is started.
"""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.tasks import interaction_setup
from wia.tasks.support_diagnostics import support_space_diagnostic


class Map:
    map_resolution = .1
    map_size = 20
    floor_heights = [0., 3.]
    def __init__(self):
        self.floor_map = [np.full((20,20),255,dtype=np.uint8)]
        self.room_ins_map = np.ones((20,20),dtype=np.uint8)
        self.room_ins_name_to_ins_id = {'bathroom_0':1,'other_room':2}
    def map_to_world(self,cells):
        return np.flip((np.asarray(cells)-10)*.1,axis=-1).copy()
    def world_to_map(self,points):
        return np.flip(np.asarray(points)/.1+10,axis=-1).astype(int).copy()
    def get_room_instance_by_point(self,point):
        index=self.world_to_map(np.asarray(point)).reshape(2)
        if np.any(index<0) or np.any(index>=20):
            return None
        return {1:'bathroom_0',2:'other_room'}[int(self.room_ins_map[tuple(index)])]


def obj(name,category,low,high,*,visual=False,visible=True):
    return NS(name=name,category=category,model='model',aabb=(np.array(low,dtype=float),np.array(high,dtype=float)),
        visual_only=visual,visible=visible,fixed_base=True,in_rooms=['bathroom_0'])


@pytest.fixture
def world():
    floor=obj('floor','floors',[-1,-1,-.1],[1,1,0.])
    decor=obj('decor','picture',[-.7,-.7,.8],[-.6,-.6,.9],visual=True)
    outside=obj('outside','table',[50,50,0.],[51,51,1.])
    native_map=Map()
    robot=NS(base_footprint_link_name='chassis',
        links={'chassis':NS(aabb=(np.array([-.3,-.3,0.]),np.array([.3,.3,.1])))},
        aabb=(np.array([-.3,-.3,0.]),np.array([.3,.3,1.2])))
    scene=NS(trav_map=native_map,seg_map=native_map,objects=[floor,decor,outside])
    session=NS(env=NS(scene=scene),native_reset_id='reset',pipeline=NS(og=NS(sim=NS(current_time=2.))))
    return NS(session=session,robot=robot,native_map=native_map,scene=scene)


def test_complete_raw_room_points_are_not_truncated_to_48(world):
    value=support_space_diagnostic(world.session,world.robot,'bathroom_0',set())
    raw=value['raw_floor']
    assert raw['available'] and raw['raw_free_cell_count']==400
    assert len(raw['world_xy_m'])==len(raw['queried_rooms'])==400
    assert all(room=='bathroom_0' for room in raw['queried_rooms'])
    assert raw['complete_requested_room'] and not raw['candidate_truncation_applied']
    assert raw['map_resolution_m']==.1 and raw['native_floor_heights_m']==[0.,3.]
    assert value['native_chassis']['extent_m']==pytest.approx([.6,.6,.1])
    expected_radius=np.sqrt(.6**2+.6**2)/2+.03
    assert value['native_chassis']['existing_circumscribed_screen_radius_m']==pytest.approx(expected_radius)
    circle=value['existing_circle_screen']
    assert circle['available'] and 0<circle['candidate_count']<400
    assert len(circle['queried_rooms'])==circle['candidate_count']
    rows={row['name']:row for row in value['native_objects']['rows']}
    assert set(rows)=={'floor','decor'}
    assert rows['decor']['visual_only'] and rows['decor']['queried_centre_room']=='bathroom_0'
    assert 'accepted' not in value
    json.dumps(value,allow_nan=False)


def test_actual_room_queries_and_room_scope_are_preserved(world):
    world.native_map.room_ins_map[:,10:]=2
    value=support_space_diagnostic(world.session,world.robot,'bathroom_0',set())
    raw=value['raw_floor']
    assert raw['available'] and raw['raw_free_cell_count']==200
    assert all(world.native_map.get_room_instance_by_point(point)=='bathroom_0' for point in raw['world_xy_m'])
    assert all(room=='bathroom_0' for room in raw['queried_rooms'])
    assert raw['complete_requested_room']


def test_snapshot_is_read_only_and_does_not_render_step_or_set_pose(world):
    before_map=world.native_map.floor_map[0].copy()
    before_rooms=world.native_map.room_ins_map.copy()
    before_objects=deepcopy(world.scene.objects)
    def forbidden(*args,**kwargs):
        raise AssertionError('Diagnostics must not mutate native state.')
    world.session.pipeline.og.sim.step=forbidden
    world.session.pipeline.og.sim.render=forbidden
    for item in world.scene.objects:
        item.set_position_orientation=forbidden
    for _ in range(2):
        support_space_diagnostic(world.session,world.robot,'bathroom_0',{'decor'})
    assert np.array_equal(world.native_map.floor_map[0],before_map)
    assert np.array_equal(world.native_map.room_ins_map,before_rooms)
    for current,original in zip(world.scene.objects,before_objects):
        assert all(np.array_equal(a,b) for a,b in zip(current.aabb,original.aabb))
        assert current.visible==original.visible and current.visual_only==original.visual_only


def test_circle_error_and_excluded_argument_are_recorded_exactly(world,monkeypatch):
    calls=[]
    def no_circle(session,robot,region,excluded):
        calls.append((region,excluded.copy()))
        raise ValueError('No conservative collision-free chassis cell exists.')
    monkeypatch.setattr(interaction_setup,'_navigation_cells',no_circle)
    value=support_space_diagnostic(world.session,world.robot,'bathroom_0',{'decor'})
    assert calls==[('bathroom_0',{'decor'})]
    assert value['raw_floor']['raw_free_cell_count']==400
    assert not value['existing_circle_screen']['available']
    assert value['existing_circle_screen']['reason']=='ValueError: No conservative collision-free chassis cell exists.'
    assert next(row for row in value['native_objects']['rows'] if row['name']=='decor')['excluded_by_argument']


def test_missing_map_is_explicitly_unavailable_and_no_circle_is_fabricated(world):
    del world.scene.trav_map
    value=support_space_diagnostic(world.session,world.robot,'bathroom_0',set())
    assert not value['raw_floor']['available'] and 'trav_map' in value['raw_floor']['reason']
    assert not value['existing_circle_screen']['available']
    assert value['native_chassis']['available']


def test_without_room_requires_explicit_bounds_and_never_dumps_building_points(world):
    value=support_space_diagnostic(world.session,world.robot,None,set())
    assert not value['raw_floor']['available'] and 'bounds' in value['raw_floor']['reason']
    assert 'world_xy_m' not in value['raw_floor']
    bounded=support_space_diagnostic(world.session,world.robot,None,set(),bounds_xy_m=([-.21,-.21],[.21,.21]))
    raw=bounded['raw_floor']
    assert raw['available'] and not raw['complete_requested_room']
    assert 0<raw['raw_free_cell_count']<400
    assert np.max(np.abs(np.array(raw['world_xy_m'])))<=.21
    assert not bounded['existing_circle_screen']['available']
    assert 'bounded' in bounded['existing_circle_screen']['reason']
