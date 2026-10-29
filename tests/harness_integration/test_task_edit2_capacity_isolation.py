"""Validate the explicit original-scene capacity isolation adapter on CPU.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider \
  tests/test_task_edit2_capacity_isolation.py tests/test_task_edit2_capacity_driver.py
Native construction, source filling and GPU measurements are mocked, never claimed.
"""
from contextlib import nullcontext
from copy import deepcopy
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

from wia.runtime.commands.calibrate_task_edit2_capacity import _OriginalIsolatedSceneAdapter, main, validate_inputs
from harness_integration.test_task_edit2_capacity_driver import CANDIDATE

ROOT = __import__('wia.resources', fromlist=['ROOT']).ROOT
SOURCE_MODULE = ROOT / 'ESI/src/active_explore/tasks/physical_structure/liquid_volume.py'


@pytest.fixture
def receipt():
    return json.loads((CANDIDATE / 'inventory_receipt.json').read_text())


def original_source_config(receipt):
    """Execute only unchanged pure source constructors, never the OG imports."""
    tree = ast.parse(SOURCE_MODULE.read_text())
    selected = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name in ('build_env_config', 'build_env_objects', 'side_entry')) or (
        isinstance(node, ast.Assign) and any(isinstance(target,ast.Name) and target.id in
            ('SCENE_CLUTTER_CATEGORIES','STAGING_Z') for target in node.targets))]
    scales = {row['model']: row['scale'][0] for row in receipt['inventory']}
    namespace = {'Any':object, 'get_scale':lambda category,model:scales[model]}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(SOURCE_MODULE),'exec'),namespace)
    payload = {'left_category':receipt['inventory'][0]['category'], 'left_model':receipt['inventory'][0]['model'],
               'right_category':receipt['inventory'][1]['category'], 'right_model':receipt['inventory'][1]['model']}
    objects = namespace['build_env_objects'](payload)
    return namespace['build_env_config'](receipt['scene_model'],receipt['room'],'R1',objects,payload=payload)


def test_original_filters_robot_free_config_and_exact_six_inventory_are_preserved(receipt):
    config = original_source_config(receipt)
    original = deepcopy(config)
    scene_reference, robots_reference = config['scene'], config['robots']
    adapter = _OriginalIsolatedSceneAdapter(NS(file_sha256='scene_sha'),receipt)
    adapter.apply_to_env_config(config)
    assert config['scene'] is scene_reference
    assert config['robots'] is robots_reference
    assert config['scene'] == original['scene']
    assert config['robots'] == []
    assert 'laptop' in config['scene']['not_load_object_categories']
    assert config['render'] == original['render']
    assert len(config['objects']) == 6
    assert adapter._realized == ()
    assert not adapter.applied
    assert adapter.observation_settings() == {}
    by_name = {row['name']:row for row in config['objects']}
    for item in receipt['inventory']:
        row = by_name[item['name']]
        assert all(row[k]==item[k] for k in ('name','category','model','scale'))
        assert row['fixed_base'] is True and row['visual_only'] is False
        assert row['kinematic_only'] is False
        assert row['orientation'] == original['objects'][0]['orientation']
        assert row['type'] == 'DatasetObject'
    for row in original['objects']:
        assert by_name[row['name']]['position'] == row['position']
    staged = [row for row in config['objects'] if row['name'] not in ('obj1','obj2')]
    assert all(row['position'][2]>original['objects'][0]['position'][2] for row in staged)
    assert len({tuple(row['position']) for row in staged}) == 4
    assert not any('table' in row['name'] or 'label' in row['name'] for row in config['objects'])
    assert receipt['inventory'] == adapter.receipt['inventory']


def test_source_postprocessor_is_retained_and_only_existing_cleanup_hook_runs(receipt,monkeypatch):
    import wia.authoring.converters.physical_structure_liquid_volume as owner
    from wia.authoring.converters.liquid_capacity_inventory import INITIALIZATION
    adapter = _OriginalIsolatedSceneAdapter(NS(),receipt)
    config = original_source_config(receipt)
    adapter.apply_to_env_config(config)
    calls = []
    def initialize(session,parameters):
        calls.append((session,deepcopy(parameters)))
        return {'labels':{},'pool_walls_removed':['/World/plane_0'],'comparison_only':True}
    monkeypatch.setattr(owner,'initialize_scene',initialize)
    session=NS()
    assert adapter.source_postprocess_override(session) is None
    adapter.apply_after_postprocessing(session)
    assert calls == [(session,{'operation':INITIALIZATION,'inventory_receipt':receipt})]
    assert len(adapter.applied)==1
    assert adapter.applied[0]['operation']=='original_source_pool_cleanup'
    with pytest.raises(RuntimeError,match='once'):
        adapter.apply_after_postprocessing(session)
    with pytest.raises(RuntimeError,match='initialization-only'):
        adapter.apply_to_env_config(config)


@pytest.mark.parametrize('kind', ['robot','identity','extra_object'])
def test_conflicting_source_configuration_is_rejected_without_widening_scene(receipt,kind):
    config=original_source_config(receipt)
    original_scene=deepcopy(config['scene'])
    if kind=='robot':config['robots']=[{'type':'R1'}]
    if kind=='identity':config['objects'][0]['model']='different_asset'
    if kind=='extra_object':config['objects'].append({'name':'unrelated','category':'laptop'})
    with pytest.raises(ValueError):
        _OriginalIsolatedSceneAdapter(NS(),receipt).apply_to_env_config(config)
    assert config['scene']==original_scene


def test_diagnostic_metadata_reports_actual_native_configuration_and_inventory(receipt):
    config=original_source_config(receipt)
    adapter=_OriginalIsolatedSceneAdapter(NS(),receipt)
    adapter.apply_to_env_config(config)
    objects={row['name']:NS(**{k:row[k] for k in ('name','category','model','fixed_base','visual_only')},
        scale=np.array(row['scale']),get_position_orientation=lambda row=row:(np.array(row['position']),np.array(row['orientation'])))
        for row in config['objects']}
    session=NS(robot=None,env=NS(robots=[],scene=NS(object_registry=lambda kind,name:objects.get(name))))
    value=adapter.diagnostic_context(session)
    assert value['robot_absent'] and value['native_measurement_available']
    assert value['source_scene_configuration']==config['scene']
    assert value['source_robot_configuration']==[]
    assert value['original_source_postprocessor_retained']
    assert not value['benchmark_world_mode']
    assert not value['support_table_or_label_patches_applied']
    assert len(value['native_inventory'])==6
    assert all(row['fixed_base'] and not row['visual_only'] for row in value['native_inventory'])


def mocked_cli(monkeypatch,tmp_path,receipt,*,isolated):
    """Exercise orchestration while refusing every actual GPU/native import path."""
    import wia.runtime.sim.esi_capacity_calibration as driver_module
    from wia.authoring.converters import physical_structure_liquid_volume as cleanup_module
    source_joint_capture=driver_module.NativeCapacityCalibration.capture_source_initialization
    calls=[]
    configurations=[]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setenv('OMNIGIBSON_APPDATA_PATH',str(tmp_path/'appdata'))
    monkeypatch.setitem(sys.modules,'wia.runtime.commands.run_esi_capture_worker',NS(allocation_gpu=lambda *a,**k:{'uuid':'CPU_FIXTURE_GPU'}))
    monkeypatch.setitem(sys.modules,'wia.runtime.scene_capture_adapters',NS(install_gpu_selection=lambda gpu:calls.append('gpu_mock'),
        isolate_texture_cache=lambda *a:None,gpu_evidence=lambda *a:{'fixture_only':True}))
    monkeypatch.setitem(sys.modules,'omnigibson',NS(simulator=NS()))
    monkeypatch.setitem(sys.modules,'omnigibson.macros',NS(gm=NS(unlocked=lambda:nullcontext())))
    class BenchmarkAdapter:
        def __init__(self,loaded,bindings):self.task=loaded;calls.append('benchmark_adapter')
        def observation_settings(self):return {'head_pitch_deg':45.}
        def apply_to_env_config(self,config):
            config['robots']=[{'type':'R1'}]
            for item in receipt['inventory'][2:]:
                config['objects'].append({'type':'DatasetObject','name':item['name'],'category':item['category'],
                    'model':item['model'],'scale':item['scale'],'position':[0.,0.,3.],
                    'orientation':[0.,0.,0.,1.],'fixed_base':True,'visual_only':False})
            for row in config['objects']:row.setdefault('visual_only',False)
        def source_postprocess_override(self,session):return None
        def apply_after_postprocessing(self,session):calls.append('benchmark_after')
    import wia.tasks.scene as scene_module
    monkeypatch.setattr(scene_module,'ScenePatches',BenchmarkAdapter)
    monkeypatch.setattr(cleanup_module,'initialize_scene',lambda session,p:calls.append(('cleanup',deepcopy(p))) or {'labels':{}})
    class Session:
        def __init__(self,path,**kwargs):
            calls.append(('session',kwargs));self.robot=None;self.task_state={};self._observation_source=None
        def reset(self):
            config=original_source_config(receipt)
            self.scene_adapter.apply_to_env_config(config);configurations.append(deepcopy(config))
            objects={row['name']:NS(**{k:row[k] for k in ('name','category','model','fixed_base','visual_only')},
                scale=np.array(row['scale']),joints={},
                get_position_orientation=lambda row=row:(np.array(row['position']),np.array(row['orientation'])))
                for row in config['objects']}
            self.env=NS(robots=config['robots'],scene=NS(object_registry=lambda kind,name:objects.get(name)))
            if self.env.robots:
                self.robot=NS(sensors={'head':object()});self.camera_name='head'
            self._initialization_stage_observer(self,'settled',config)
            assert self.scene_adapter.source_postprocess_override(self) is None
            calls.append('original_source_postprocessor_obj1_obj2')
            self.scene_adapter.apply_after_postprocessing(self)
        def _sensor_rgb(self,sensor):calls.append('head_capture');return np.zeros((4,4,3),np.uint8)
        def close(self):calls.append('closed')
    monkeypatch.setitem(sys.modules,'wia.runtime.sim.esi_omnigibson',NS(ESISession=Session))
    monkeypatch.setitem(sys.modules,'wia.runtime.sim.esi_robot_cameras',NS(configure_robot_cameras=lambda *a,**k:calls.append('mounted_config')))
    class Driver:
        capture_source_initialization=staticmethod(source_joint_capture)
        def __init__(self,session,actual_receipt,joints,output,**kwargs):
            assert set(joints)=={x['name'] for x in receipt['inventory']}
            assert actual_receipt==receipt
            assert session.loaded_task.file_sha256
            assert session.scene_adapter.task.file_sha256
            calls.append(('driver',kwargs))
        def run(self,*,pairs):
            calls.append(('pairs',pairs))
            return {'post_calibration_inventory_reset':{'accepted':True},'calibration_recipe':{}}
    monkeypatch.setattr(driver_module,'NativeCapacityCalibration',Driver)
    argv=['--task-file',str(CANDIDATE/'task.yaml'),'--scene-file',str(CANDIDATE/'scene.yaml'),
        '--inventory-receipt',str(CANDIDATE/'inventory_receipt.json'),'--pairs','A:B,A:C',
        '--output-dir',str(tmp_path/'results')]
    if isolated:argv.append('--original-isolated-scene')
    assert main(argv)==0
    return calls,configurations,json.loads((tmp_path/'results/launch.json').read_text())


def test_explicit_isolation_cli_uses_robot_free_original_protocol_and_no_mounted_capture(receipt,monkeypatch,tmp_path):
    calls,configs,launch=mocked_cli(monkeypatch,tmp_path,receipt,isolated=True)
    assert ('session',{'require_robot':False}) in calls
    assert 'benchmark_adapter' not in calls
    assert 'mounted_config' not in calls and 'head_capture' not in calls
    assert 'original_source_postprocessor_obj1_obj2' in calls
    assert ('pairs',[('A','B'),('A','C')]) in calls
    assert calls[-1]=='closed'
    context=launch['diagnostic_context']
    assert context['robot_absent'] and context['source_robot_configuration']==[]
    assert context['source_scene_configuration']==configs[0]['scene']
    final=json.loads((tmp_path/'results/calibration_receipt.json').read_text())
    assert final['diagnostic_context']['native_measurement_available']
    assert len(final['diagnostic_context']['native_inventory'])==6


def test_default_cli_retains_benchmark_adapter_robot_and_mounted_head_behavior(receipt,monkeypatch,tmp_path):
    calls,_,launch=mocked_cli(monkeypatch,tmp_path,receipt,isolated=False)
    assert 'benchmark_adapter' in calls
    assert ('session',{'require_robot':True,'grasp_mode':'assisted','head_pitch_deg':45.,'camera_size':[512,512]}) in calls
    assert 'mounted_config' in calls and calls.count('head_capture')==2
    assert 'diagnostic_context' not in launch
    assert not (tmp_path/'results/isolation_context.json').exists()
