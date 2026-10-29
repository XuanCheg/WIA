"""CPU checks for raw delivery-table centres and independent chassis stances.

Run: python -B -m pytest -q -p no:cacheprovider
 tests/test_task_edit2_delivery_centres.py tests/test_task_edit2_missing_definitions.py
 tests/test_task_edit2_destination_stages.py tests/test_task_edit2_region_stages.py.
Synthetic floor maps exercise the real proposal/stance code without native launch.
"""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.tasks import delivery_support, interaction_supports, interaction_setup
from harness_integration.test_task_edit2_missing_definitions import table_fixture


class FloorMap:
    map_resolution = .1
    map_size = 31
    floor_heights = [0.]
    def __init__(self):
        self.floor_map = [np.full((self.map_size, self.map_size), 255, dtype=np.uint8)]
    def map_to_world(self, cells):
        cells = np.asarray(cells)
        return np.flip((cells - self.map_size/2) * self.map_resolution, axis=-1).copy()
    def world_to_map(self, xy):
        return np.flip(np.asarray(xy)/self.map_resolution + self.map_size/2, axis=-1).astype(int).copy()


def make_world(monkeypatch):
    session, contract, objects, calls = table_fixture(monkeypatch)
    scene = session.env.scene
    scene.trav_map = FloorMap()
    size = scene.trav_map.map_size
    scene.seg_map = NS(room_ins_name_to_ins_id={'room_0':1}, map_resolution=.1, map_size=size,
        room_ins_map=np.ones((size,size),dtype=int), get_room_instance_by_point=lambda xy:'room_0')
    objects['floor'] = NS(name='floor', category='floors', prim_path='/World/floor', visual_only=False,
        aabb=(np.array([-1.55,-1.55,-.1]), np.array([1.55,1.55,0.])))
    session.robot = NS(name='robot', base_footprint_link_name='chassis',
        links={'chassis':NS(aabb=(np.array([-.3,-.3,0.]),np.array([.3,.3,.1])))},
        aabb=(np.array([-.3,-.3,0.]),np.array([.3,.3,1.2])))
    base_activate = interaction_supports.activate_support
    def activate(*args, **kwargs):
        result = base_activate(*args, **kwargs)
        for name in interaction_supports.support_names(delivery_support.SLOT):
            objects[name].visible = True
        return result
    monkeypatch.setattr(interaction_supports,'activate_support',activate)
    return session, contract, objects, calls


def test_raw_near_wall_proposal_is_independent_of_complete_chassis_cells(monkeypatch):
    original_navigation = interaction_setup._navigation_cells
    original_footprint = interaction_supports.footprint_clear
    session, contract, objects, calls = make_world(monkeypatch)
    # table_fixture mocks these APIs; restore the actual CPU geometry algorithms.
    monkeypatch.setattr(interaction_setup,'_navigation_cells',original_navigation)
    monkeypatch.setattr(interaction_supports,'footprint_clear',original_footprint)
    before_stances, _, _ = original_navigation(session,session.robot,contract['room'],set())
    before_target = deepcopy(objects['target'].aabb)
    result = delivery_support.initialize_delivery_support(session,contract)
    assert result['accepted'] and calls == [99]
    assert result['centre_proposal_source'] == 'raw_floor_cells_before_chassis_erosion'
    selected = result['selected']
    xy = np.asarray(selected['centre_xy_m'])
    assert not np.any(np.linalg.norm(before_stances-xy,axis=1)<1e-8)
    stance = selected['robot_stance_screen']
    assert stance['accepted'] and stance['excluded_object_names'] == []
    assert stance['table_parts_participate_in_collision_screen']
    assert not np.allclose(stance['nearest_stance_xy_m'],xy)
    assert not result['public_control_access_verified']
    assert not stance['route_and_arm_clearance_verified']
    assert all(np.array_equal(a,b) for a,b in zip(before_target,objects['target'].aabb))


def test_activated_table_colliders_are_present_in_the_real_stance_domain(monkeypatch):
    original_navigation = interaction_setup._navigation_cells
    session, contract, objects, _ = make_world(monkeypatch)
    # Activate an interior table to make inclusion/exclusion observable.
    interaction_supports.activate_support(session,np.array([0.,0.]),np.array([.45,.4]),.65,0.,slot=99)
    without_table, _, _ = original_navigation(session,session.robot,contract['room'],set(interaction_supports.support_names(99)))
    with_table, _, _ = original_navigation(session,session.robot,contract['room'],set())
    assert len(with_table)<len(without_table)
    assert min(np.linalg.norm(with_table,axis=1)) > min(np.linalg.norm(without_table,axis=1))


def test_failed_stance_hides_support_and_continues_to_next_raw_centre(monkeypatch):
    session, contract, objects, calls = make_world(monkeypatch)
    monkeypatch.setattr(interaction_supports,'floor_support_centres',lambda *args:[np.array([1.,1.]),np.array([1.4,1.])])
    count=[]; hidden=[]
    base_hide=interaction_supports.hide_supports
    def hide(session,*,slot):
        hidden.append(slot); base_hide(session,slot=slot)
    def stances(session,robot,region,excluded):
        assert excluded==set()
        assert region==contract['room']
        assert all(objects[name].visible and not objects[name].visual_only for name in interaction_supports.support_names(99))
        count.append(1)
        if len(count)==1:
            raise ValueError('No conservative collision-free chassis cell exists.')
        return np.array([[0.,0.]]),0.,{'candidate_cell_count':1,'route_and_arm_clearance_verified':False}
    monkeypatch.setattr(interaction_supports,'hide_supports',hide)
    monkeypatch.setattr(interaction_setup,'_navigation_cells',stances)
    result=delivery_support.initialize_delivery_support(session,contract)
    assert result['accepted'] and calls==[99,99] and hidden==[99]
    assert len(result['proposals'])==2
    rejected=result['proposals'][0]
    assert not rejected['accepted'] and rejected['failure']=='native_delivery_robot_stance_unavailable'
    assert 'No conservative' in rejected['stance_rejection']
    assert result['proposals'][1]['accepted']
    assert result['source_target_xy_m']==pytest.approx([9.05,9.05])


@pytest.mark.parametrize('empty_result',[False,True])
def test_no_remaining_stance_never_accepts_a_geometrically_fitting_table(monkeypatch,empty_result):
    session, contract, objects, calls = make_world(monkeypatch)
    monkeypatch.setattr(interaction_supports,'floor_support_centres',lambda *args:[np.array([1.,1.])])
    def no_stance(*args):
        assert args[-1]==set()
        if empty_result:
            return np.empty((0,2)),0.,{'candidate_cell_count':0}
        raise ValueError('No conservative stance exists in the requested original room.')
    monkeypatch.setattr(interaction_setup,'_navigation_cells',no_stance)
    with pytest.raises(ValueError,match='no_safe_loaded_room'):
        delivery_support.initialize_delivery_support(session,contract)
    evidence=session.delivery_support_initialization
    assert calls==[99] and not evidence['accepted']
    assert evidence['failed_stage']=='post_activation_robot_stance'
    assert evidence['proposals'][0]['failure']=='native_delivery_robot_stance_unavailable'
    assert all(objects[name].visible is False for name in interaction_supports.support_names(99))


def test_native_floor_rejection_still_precedes_activation_and_stance(monkeypatch):
    session, contract, _, calls=make_world(monkeypatch)
    monkeypatch.setattr(interaction_supports,'floor_support_centres',lambda *args:[np.array([1.,1.])])
    monkeypatch.setattr(delivery_support,'native_floor_geometry',lambda *args: (_ for _ in ()).throw(ValueError('native_floor_coverage_ray_missing')))
    monkeypatch.setattr(interaction_setup,'_navigation_cells',lambda *args:pytest.fail('Stance must only run after activation.'))
    with pytest.raises(ValueError,match='no_safe_loaded_room'):
        delivery_support.initialize_delivery_support(session,contract)
    assert calls==[]
    assert session.delivery_support_initialization['proposals'][0]['failure']=='native_floor_coverage_ray_missing'
