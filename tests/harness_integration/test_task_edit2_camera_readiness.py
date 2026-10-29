"""CPU raw-annotator readiness and non-destructive native camera resize checks.

Run: python -B -m pytest -q -p no:cacheprovider
 tests/test_task_edit2_camera_readiness.py tests/test_r1_upgrade.py
 tests/test_task_edit2_reflection.py.
Synthetic annotators reproduce empty-remapper hazards without native/GPU startup.
"""
from itertools import count
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

import wia.runtime.sim.esi_robot_cameras as cameras


class Annotator:
    def __init__(self, data):
        self.data, self.calls, self.detached = data, [], []
    def get_data(self, *, device):
        self.calls.append(device)
        if isinstance(self.data, Exception):
            raise self.data
        return self.data
    def detach(self, product):
        self.detached.append(product)


class NativeSensor:
    identities = count()
    def __init__(self, name='head', image_width=6, image_height=4, modalities=('rgb','seg_instance'), **kwargs):
        self.name = name
        self.size = (image_width,image_height)
        self.modalities = {modalities} if isinstance(modalities,str) else set(modalities)
        self.prim_path='/World/scene_0'+kwargs.get('relative_prim_path','/robot/'+name)
        self.viewport_changes=[]
        self._viewport=NS(viewport_api=NS(set_texture_resolution=self.resize_viewport))
        self._render_product=NS(path='/render/'+str(next(self.identities)), destroy=lambda:pytest.fail('Old native product was destroyed.'))
        self.position=np.array([0.,0.,1.]); self.quaternion=np.array([0.,0.,0.,1.])
        self.focal_length=kwargs.get('focal_length',24.)
        self.horizontal_aperture=20.95; self.clipping_range=(.01,100.)
        self.obs_calls=0
        self._annotators={}
        self.initialize_sensors(names=self.modalities)
    @property
    def image_width(self):
        return self.size[0]
    @image_width.setter
    def image_width(self,value):
        pytest.fail('Destructive native image_width setter called.')
    @property
    def image_height(self):
        return self.size[1]
    @image_height.setter
    def image_height(self,value):
        pytest.fail('Destructive native image_height setter called.')
    def resize_viewport(self,size):
        self.viewport_changes.append(tuple(size)); self.size=tuple(size)
    def initialize_sensors(self, *, names):
        for name in names:
            value=np.zeros((self.image_height,self.image_width,4),dtype=np.uint8) if name=='rgb' else np.zeros((self.image_height,self.image_width),dtype=np.uint32)
            self._annotators[name]=Annotator({'data':value,'info':{'idToLabels':{}}} if name.startswith('seg_') else value)
    def get_obs(self):
        self.obs_calls+=1
        result={}
        for name in self.modalities:
            raw=self._annotators[name].data
            value=raw['data'] if isinstance(raw,dict) else raw
            if name.startswith('seg_') and not value.size:
                raise RuntimeError('th.max cannot remap an EMPTY segmentation array.')
            result[name]=value
        return result,{}
    def get_position_orientation(self,**kwargs):
        return self.position.copy(),self.quaternion.copy()
    def set_position_orientation(self,**kwargs):
        self.position=np.asarray(kwargs['position']); self.quaternion=np.asarray(kwargs['orientation'])
    def get_attribute(self,name):
        return 15.3 if name=='verticalAperture' else None
    def add_modality(self,name):
        self.modalities.add(name); self.initialize_sensors(names=[name])
    def load(self,scene):
        pass
    def initialize(self):
        pass


@pytest.fixture
def runtime(monkeypatch):
    renders=[]; invalidations=[]
    def forbidden_step():
        pytest.fail('Camera readiness must not step physics.')
    sim=NS(device='cpu',render=lambda:renders.append(1),step=forbidden_step)
    session=NS(robot=NS(sensors={}),pipeline=NS(og=NS(sim=sim)),invalidate_camera_cache=lambda:invalidations.append(1))
    products=[]
    rep=ModuleType('omni.replicator.core')
    def create(path,size,**kwargs):
        product=NS(path='/render/rebound_'+str(len(products)),destroy=lambda:pytest.fail('New product destroyed.'))
        products.append((path,tuple(size),product)); return product
    rep.create=NS(render_product=create)
    monkeypatch.setitem(sys.modules,'omni',ModuleType('omni'))
    monkeypatch.setitem(sys.modules,'omni.replicator',ModuleType('omni.replicator'))
    monkeypatch.setitem(sys.modules,'omni.replicator.core',rep)
    return NS(session=session,renders=renders,invalidations=invalidations,products=products)


def add_sensor(runtime,sensor):
    runtime.session.robot.sensors[sensor.name]=sensor
    return {sensor.name:sensor.name}


def test_valid_rgb_empty_segmentation_is_probed_before_get_obs_and_recovered(runtime):
    sensor=NativeSensor(); registry=add_sensor(runtime,sensor)
    old=sensor._render_product
    old_annos=sensor._annotators
    sensor._annotators['seg_instance'].data={'data':np.empty((0,),dtype=np.uint32)}
    cameras.recover_empty_render_products(runtime.session,registry)
    assert sensor.obs_calls==1  # Only after raw readiness; the empty-remapper path was never called.
    assert len(runtime.renders)==9 and len(runtime.products)==1
    assert runtime.session._retired_camera_products[0]==(old,old_annos)
    evidence=runtime.session.camera_render_recovery[0]
    assert evidence['failed_channels']==['seg_instance']
    assert evidence['channels_before']['rgb']['ready']
    assert evidence['channels_before']['seg_instance']['shape']==[0]
    assert evidence['channels_after']['seg_instance']['shape']==[4,6]
    assert evidence['all_enabled_channels_ready'] and evidence['rgb_restored']
    assert runtime.invalidations==[1]


def test_all_enabled_seg_channels_and_expected_dimensions_are_checked(runtime):
    sensor=NativeSensor(modalities=('rgb','seg_instance','seg_semantic','seg_instance_id'))
    registry=add_sensor(runtime,sensor)
    sensor._annotators['seg_semantic'].data={'data':np.zeros((4,5),dtype=np.uint32)}
    sensor._annotators['seg_instance_id'].data={'data':np.zeros((0,0),dtype=np.uint32)}
    cameras.recover_empty_render_products(runtime.session,registry)
    evidence=runtime.session.camera_render_recovery[0]
    assert set(evidence['failed_channels'])=={'seg_semantic','seg_instance_id'}
    assert evidence['channels_before']['seg_semantic']['expected_height_width']==[4,6]
    assert set(evidence['channels_after'])=={'rgb','seg_instance','seg_semantic','seg_instance_id'}
    assert evidence['all_enabled_channels_ready']


def test_persistently_empty_raw_channel_fails_without_calling_remapper(runtime,monkeypatch):
    sensor=NativeSensor(); registry=add_sensor(runtime,sensor)
    sensor._annotators['seg_instance'].data={'data':np.zeros((0,0),dtype=np.uint32)}
    monkeypatch.setattr(cameras,'rebind_render_product',lambda *args,**kwargs:None)
    with pytest.raises(RuntimeError,match='remain unavailable'):
        cameras.recover_empty_render_products(runtime.session,registry)
    assert sensor.obs_calls==0 and len(runtime.renders)==9
    assert not runtime.session.camera_channel_readiness[0]['ready']
    assert not runtime.session.camera_render_recovery[0]['all_enabled_channels_ready']


def test_arbitrary_raw_annotator_error_is_not_swallowed_or_retried(runtime):
    sensor=NativeSensor(); registry=add_sensor(runtime,sensor)
    sensor._annotators['rgb'].data=ValueError('unexpected native annotator API error')
    with pytest.raises(ValueError,match='unexpected native'):
        cameras.recover_empty_render_products(runtime.session,registry)
    assert not runtime.products and not runtime.renders and sensor.obs_calls==0
    assert 'ValueError' in runtime.session.camera_channel_readiness[0]['error']


def test_full_get_obs_is_verified_after_raw_buffers_ready(runtime):
    sensor=NativeSensor(); registry=add_sensor(runtime,sensor)
    sensor.get_obs=lambda:({'rgb':np.zeros((4,6,4),dtype=np.uint8),'seg_instance':np.empty((0,0))},{})
    with pytest.raises(RuntimeError,match='full camera observation remains unavailable'):
        cameras.recover_empty_render_products(runtime.session,registry)
    assert len(runtime.products)==1 and len(runtime.renders)==9
    assert not runtime.session.camera_render_recovery[0]['full_observation_channels_after']['seg_instance']['ready']


def test_zero_label_values_are_ready_but_no_semantic_certification(runtime):
    sensor=NativeSensor(); registry=add_sensor(runtime,sensor)
    cameras.recover_empty_render_products(runtime.session,registry)
    assert sensor.obs_calls==1 and not runtime.products
    assert runtime.session.camera_channel_readiness[0]['ready']
    assert runtime.session.camera_render_recovery==[]


def test_native_resize_uses_viewport_once_and_retains_product_and_optics(runtime):
    sensor=NativeSensor(); add_sensor(runtime,sensor)
    old=sensor._render_product; before=(sensor.position.copy(),sensor.quaternion.copy(),sensor.focal_length,sensor.horizontal_aperture)
    cameras._resize_camera(runtime.session,sensor,(12,8))
    assert sensor.viewport_changes==[(12,8)]
    assert runtime.session._retired_camera_products[0][0] is old
    assert runtime.products[0][1]==(12,8)
    assert np.array_equal(sensor.position,before[0]) and np.array_equal(sensor.quaternion,before[1])
    assert (sensor.focal_length,sensor.horizontal_aperture)==before[2:]
    evidence=runtime.session.camera_render_product_rebindings[0]
    assert not evidence['resolution_unchanged']
    assert evidence['old_resolution_px']==[6,4] and evidence['new_resolution_px']==[12,8]


@pytest.mark.parametrize('initial_head_size',[(512,288),(640,480)])
def test_reflection_head_profile_is_honored_with_legacy_optics(runtime,monkeypatch,initial_head_size,tmp_path):
    import wia.runtime.sim.esi_reflection as esi_reflection
    monkeypatch.setattr(esi_reflection,'measure_reflection',lambda session:None)
    monkeypatch.setitem(sys.modules,'omnigibson.sensors',NS(VisionSensor=NativeSensor))
    monkeypatch.setitem(sys.modules,'carb',NS(settings=NS(get_settings=lambda:NS(set_int=lambda *args:None))))
    names=['r:eyes:Camera:0','r:left_eef_link:Camera:0','r:right_eef_link:Camera:0']
    head=NativeSensor(name=names[0],image_width=initial_head_size[0],image_height=initial_head_size[1])
    sensors={names[0]:head,**{name:NativeSensor(name=name,image_width=512,image_height=512) for name in names[1:]}}
    robot=NS(name='r',sensors=sensors,base_footprint_link_name='base_link',links={'base_link':NS(prim_path='/World/scene_0/robot/base_link')})
    session=runtime.session
    session.robot=robot; session.camera_name=names[0]; session.wrist_camera_name=names[1];session.rgb_cameras=names.copy()
    session.env=NS(scene=NS(prim_path='/World/scene_0'))
    session.reflection_sensor_profile={'head_image_size_px':[512,288]}
    old=head._render_product
    import json
    from pathlib import Path
    config = json.loads(Path('configs/esi_robot_cameras.json').read_text())
    config.pop('camera_fov_deg', None)
    config_path = tmp_path / 'legacy_cameras.json'
    config_path.write_text(json.dumps(config))
    cameras.configure_robot_cameras(session,config_path,resolution=(512,512))
    assert len(session.robot_camera_registry)==8
    assert head.size==(512,288)
    assert all(sensor.size==(512,512) for name,sensor in sensors.items() if name!=names[0])
    assert all(not sensors[name].viewport_changes for name in names[1:])
    if initial_head_size==(512,288):
        assert head.viewport_changes==[] and head._render_product is old
        assert not runtime.products
    else:
        assert head.viewport_changes==[(512,288)] and len(runtime.products)==1
        assert runtime.session._retired_camera_products[0][0] is old
    assert session.robot_camera_mounts['agentview_rgb']['render_size']==[512,288]
    assert all(value['ready'] for value in session.camera_channel_readiness)


def test_missing_enabled_native_annotator_is_restored_on_rebind(runtime):
    sensor=NativeSensor(); registry=add_sensor(runtime,sensor)
    sensor._annotators.pop('seg_instance')
    cameras.recover_empty_render_products(runtime.session,registry)
    assert runtime.session.camera_render_recovery[0]['channels_before']['seg_instance']['reason']=='enabled_native_annotator_missing'
    assert runtime.session.camera_render_recovery[0]['all_enabled_channels_ready']


def test_get_obs_fallback_for_sensors_without_native_annotators_is_unchanged(runtime):
    sensor=NativeSensor(); registry=add_sensor(runtime,sensor)
    del sensor._annotators
    sensor.get_obs=lambda:({'rgb':np.zeros((2,2,3),dtype=np.uint8)},{})
    cameras.recover_empty_render_products(runtime.session,registry)
    assert runtime.session.camera_channel_readiness[0]['raw_before'] is None
    assert runtime.session.camera_channel_readiness[0]['full_before']['source']=='get_obs_test_sensor_fallback'
    assert runtime.session.camera_channel_readiness[0]['ready'] and not runtime.products


def test_arbitrary_full_observation_exception_is_propagated_without_rebind(runtime):
    sensor=NativeSensor(); registry=add_sensor(runtime,sensor)
    def unexpected():
        raise KeyError('unrelated remapper metadata failure')
    sensor.get_obs=unexpected
    with pytest.raises(KeyError,match='unrelated'):
        cameras.recover_empty_render_products(runtime.session,registry)
    assert not runtime.products and not runtime.renders
    assert 'KeyError' in runtime.session.camera_channel_readiness[0]['error']


@pytest.mark.parametrize("horizontal,vertical", [(110,70),(87,58),(118,62)])
def test_square_pixel_raster_preserves_both_fovs(horizontal,vertical):
    width,height=cameras.fov_render_dimensions((1024,576),horizontal,vertical)
    focal=width/(2*np.tan(np.deg2rad(horizontal)/2))
    actual_vertical=np.rad2deg(2*np.arctan(height/(2*focal)))
    assert width<=1024 and height<=576
    assert abs(actual_vertical-vertical)<.2
