"""Exercise native authoring input/export contracts using CPU mocks only.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider
tests/test_esi_native_scene_editor.py
No OmniGibson GUI, simulation, model or service is started by these tests.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import yaml

import wia.runtime.commands.edit_esi_scene_native as editor


@pytest.fixture
def inputs(tmp_path):
    identity='Test/Family/map/room/a.json'
    source=tmp_path/'source'/identity;source.parent.mkdir(parents=True)
    source.write_text(json.dumps({'scene':'map','room':'room','question':'Original source question.'}))
    scene_config={'policy':'native_then_minimal_patch_else_skip','variants':{
        'original':{'patches':[]},'adapted':{'patches':[{'operation':'robot_pose',
            'position':[1.,2.,.5],'yaw_deg':0.,'head_pitch_deg':0.}]}}}
    source_config={'provider':'esi','json':identity}
    scene={'kind':'scene_patch','id':'editor_scene','version':1,'source':source_config,
        'scene':scene_config,'bindings':{},'seed':7}
    task={'kind':'category_robot_task','id':'editor_task','version':1,'source':source_config,
        'scene':deepcopy(scene_config),'public':{'objective':'Inspect the original source objects.',
            'success_criteria':'Review the layout.','environment':'A native source room.'},
        'private':{'bindings':{},'evaluation':{'type':'category_operation',
            'module':'wia.authoring.converters.enumerative_perception_search','operation':'all_target_search','parameters':{}}},
        'requirements':{'capabilities':['base_translation']},'qualification':{'on_failure':'skip'},'defaults':{'seed':7}}
    scene_path,task_path=tmp_path/'scene.yaml',tmp_path/'task.yaml'
    scene_path.write_text(yaml.safe_dump(scene));task_path.write_text(yaml.safe_dump(task))
    loaded=editor.load_editor_inputs(scene_path,task_path,source_root=tmp_path/'source')
    return loaded,scene,task,tmp_path/'source'


class NativeSim:
    def __init__(self,playing=False,stopped=False):
        self.playing,self.stopped=playing,stopped
        self.calls=[]

    def is_stopped(self):return self.stopped
    def is_playing(self):return self.playing
    def pause(self):self.playing=False;self.calls.append('pause')
    def step(self):self.calls.append('step')
    def save(self,*,json_paths):
        self.calls.append(('save',json_paths))
        Path(json_paths[0]).write_text(json.dumps({'native_scene':'actual mocked native export'}))


def instance(inputs,tmp_path,**sim_options):
    loaded=inputs[0];sim=NativeSim(**sim_options);updates=[]
    session=NS(pipeline=NS(og=NS(sim=sim,app=NS(update=lambda:updates.append('update'),is_running=lambda:True))))
    app=editor.NativeSceneEditor(loaded,session,tmp_path/'editor_output')
    app.world_available=True;app.application=session.pipeline.og.app
    return app,sim,updates


def test_exact_loader_defaults_shared_adapted_and_retains_seed_and_provenance(inputs):
    loaded=inputs[0]
    assert loaded.scene.variant==loaded.task.variant=='adapted'
    assert loaded.scene.seed==loaded.task.seed==7
    assert loaded.provenance()['source_identity']=='Test/Family/map/room/a.json'
    assert loaded.provenance()['authoring_only'] is True
    assert loaded.provenance()['qualification'] is False
    original=editor.load_editor_inputs(loaded.scene.path,loaded.task.path,variant='original',source_root=inputs[3])
    assert original.scene.variant=='original'
    for path,definition in ((loaded.scene.path,inputs[1]),(loaded.task.path,inputs[2])):
        definition['scene']['variants'].pop('adapted')
        path.write_text(yaml.safe_dump(definition))
    assert editor.load_editor_inputs(loaded.scene.path,loaded.task.path,source_root=inputs[3]).scene.variant=='original'


@pytest.mark.parametrize('failure',['source','seed','patches','unknown_variant'])
def test_loader_rejects_mismatch_without_source_or_variant_fallback(inputs,failure):
    loaded,scene,task,source_root=inputs
    if failure=='source':task['source']['json']='Test/Family/map/room/other.json'
    elif failure=='seed':task['defaults']['seed']=0
    elif failure=='patches':task['scene']['variants']['adapted']['patches']=[]
    loaded.task.path.write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError):
        editor.load_editor_inputs(loaded.scene.path,loaded.task.path,
            variant='missing' if failure=='unknown_variant' else None,source_root=source_root,source_only=True)


@pytest.mark.parametrize('source_only',[False,True])
def test_session_binds_actual_task_before_reset_and_source_only_omits_adapter_explicitly(inputs,monkeypatch,source_only):
    import wia.runtime.sim.esi_omnigibson as esi_omnigibson
    loaded=inputs[0];loaded.source_only=source_only
    class Session:
        def __init__(self,source,**kwargs):self.source=source;self.options=kwargs
    monkeypatch.setattr(esi_omnigibson,'ESISession',Session)
    session=editor.create_editor_session(loaded)
    assert session.loaded_task is loaded.task
    assert session.public_task_text==loaded.task.task_text
    assert session.options['require_robot'] is not source_only
    assert hasattr(session,'scene_adapter') is not source_only
    assert loaded.provenance()['adapted_patches_omitted'] is source_only
    assert loaded.scene.variant=='adapted'  # Provenance is not silently remapped.


@pytest.mark.parametrize('playing',[False,True])
def test_native_export_unique_verbatim_and_unqualified_without_extra_steps(inputs,tmp_path,playing):
    app,sim,updates=instance(inputs,tmp_path,playing=playing)
    before={path:path.read_bytes() for path in (inputs[0].task.path,inputs[0].scene.path,inputs[0].task.source_path)}
    first=app.save_snapshot();second=app.save_snapshot()
    assert first['saved_scene_file']!=second['saved_scene_file']
    assert first['physics_state']==('playing' if playing else 'paused')
    for manifest in (first,second):
        assert Path(manifest['saved_scene_file']).is_absolute()
        assert manifest['saved_scene_sha256']==editor.sha256(manifest['saved_scene_file'])
        assert Path(manifest['task_input_copy']).read_bytes()==before[inputs[0].task.path]
        assert Path(manifest['scene_input_copy']).read_bytes()==before[inputs[0].scene.path]
        assert manifest['authoring_only'] is True and manifest['qualification'] is False and manifest['model_calls']==0
        assert manifest['seed']==7 and manifest['benchmark_integration']=='separate_review_required'
    assert all(call[0]=='save' for call in sim.calls)
    assert not updates
    assert all(path.read_bytes()==value for path,value in before.items())
    status=editor.read_json(app.output/'editor_status.json')
    assert status['stage']=='exported' and status['pid']==os.getpid()


def test_stopped_or_changed_inputs_refuse_save_without_advancing_physics(inputs,tmp_path):
    app,sim,updates=instance(inputs,tmp_path,stopped=True)
    with pytest.raises(ValueError,match='stopped'):app.save_snapshot()
    assert not sim.calls and not updates and not app.output.exists()
    sim.stopped=False
    inputs[0].task.path.write_text(inputs[0].task.path.read_text()+'\n')
    with pytest.raises(RuntimeError,match='changed'):app.save_snapshot()
    assert not sim.calls and not app.output.exists()


def test_requests_save_once_report_errors_and_quit_without_native_actions(inputs,tmp_path):
    app,sim,_=instance(inputs,tmp_path)
    request=app.output/'editor_request.json'
    editor.write_json(request,{'request_id':'save1','action':'save'})
    app.process_request();app.process_request()
    response=editor.read_json(app.output/'editor_response.json')
    assert response['request_id']=='save1' and response['complete'] is True
    assert len(sim.calls)==1
    editor.write_json(request,{'request_id':'bad1','action':'teleport'})
    app.process_request()
    assert editor.read_json(app.output/'editor_response.json')['complete'] is False
    assert len(sim.calls)==1
    sim.stopped=True
    editor.write_json(request,{'request_id':'save2','action':'save'})
    app.process_request()
    assert 'stopped' in editor.read_json(app.output/'editor_response.json')['error']
    editor.write_json(request,{'request_id':'quit1','action':'quit'})
    app.process_request()
    assert app.close_requested is True
    assert editor.read_json(app.output/'editor_response.json')['request_id']=='quit1'


def test_pump_updates_paused_or_stopped_app_and_steps_only_user_playback(inputs,tmp_path):
    app,sim,updates=instance(inputs,tmp_path)
    app.pump();sim.stopped=True;app.pump()
    assert updates==['update','update'] and not sim.calls
    sim.stopped=False;sim.playing=True;app.pump()
    assert sim.calls==['step']


def test_ready_pauses_after_reset_and_retains_native_window(inputs,tmp_path,monkeypatch):
    app,sim,_=instance(inputs,tmp_path,playing=True);calls=[]
    app.session.reset=lambda:calls.append('reset')
    def window():app.window=object();app.close_requested=True
    monkeypatch.setattr(app,'create_window',window)
    app.run()
    assert calls==['reset'] and sim.calls==['pause'] and app.window is not None
    assert editor.read_json(app.output/'editor_status.json')['stage']=='ready'


def test_output_cannot_enter_frozen_queue_inputs(tmp_path):
    snapshot=tmp_path/'capture_job/input';snapshot.mkdir(parents=True)
    (snapshot.parent/'accepted.json').write_text('{}')
    with pytest.raises(ValueError,match='frozen queue'):
        editor.validate_output_directory(snapshot/'authoring',[snapshot/'task.yaml'])


def test_help_does_not_import_omnigibson_or_start_gui():
    result=subprocess.run([sys.executable,'-B','-c',
        "import runpy,sys; sys.argv=['edit_esi_scene_native.py','--help']; "
        "runpy.run_path('src/wia/runtime/commands/edit_esi_scene_native.py',run_name='__main__')"],
        cwd=editor.ROOT,text=True,capture_output=True)
    assert result.returncode==0
    assert '--source-only' in result.stdout and '--scene-file' in result.stdout
    assert 'Creating' not in result.stdout and 'omnigibson' not in result.stderr.lower()


def test_loading_failure_records_error_without_gui_fallback(inputs,tmp_path,monkeypatch):
    def failure(*args,**kwargs):raise ValueError('Exact input initialization mismatch.')
    def forbidden():raise AssertionError('Invalid inputs must fail before native GUI import.')
    monkeypatch.setattr(editor,'load_editor_inputs',failure)
    monkeypatch.setattr(editor,'configure_native_gui',forbidden)
    output=tmp_path/'failed_editor'
    result=editor.main(['--scene-file',str(inputs[0].scene.path),'--task-file',str(inputs[0].task.path),
        '--output-dir',str(output)])
    assert result==1
    status=editor.read_json(output/'editor_status.json')
    assert status['stage']=='error' and status['pid']==os.getpid()
    assert 'initialization mismatch' in status['error']
    assert status['qualification'] is False and status['model_calls']==0


@pytest.fixture
def persistent(inputs,tmp_path,monkeypatch):
    import wia.runtime.commands.run_esi_capture_service as run_esi_capture_service
    import wia.runtime.sim.esi_omnigibson as esi_omnigibson
    import wia.runtime.scene_capture_adapters as scene_capture_adapters
    app,sim,_=instance(inputs,tmp_path)
    events=[];old=app.session;og=old.pipeline.og
    old.env=NS(scene=NS(objects=['old_actor'],active_systems={},clear=lambda:None))
    old.task_module=NS(name='original_source');old.payload={};old.task_state={};old.closed=False
    old.loaded_task=inputs[0].task
    old.authoring_gpu={'uuid':'dedicated_gpu0'}
    old.close=lambda:events.append('old_close_forbidden')
    sim.scenes=[old.env.scene]
    original_save=sim.save
    def save(**kwargs):events.append('save');return original_save(**kwargs)
    sim.save=save
    def clear():
        events.append('clear')
        og.sim.stop()
        for scene in list(og.sim.scenes):scene.clear()
        og.sim=NativeSim(stopped=True);og.sim.scenes=[]
    og.clear=clear
    old.pipeline.task_cleanup_runtime=lambda *args:events.append('source_cleanup')
    graph=NS(reset=lambda **kwargs:events.append(('graphs_reset',kwargs)))
    monkeypatch.setitem(sys.modules,'omni.syntheticdata',NS(SyntheticData=NS(Get=lambda:graph)))
    def stop():events.append('stop');sim.stopped=True;sim.playing=False
    sim.stop=stop
    # Keep the real resident helpers; only the native runtime is mocked.
    assert callable(run_esi_capture_service.native_stage_cleanup)
    created=[]
    class Session:
        def __init__(self,source,**kwargs):
            self.source=source;self.options=kwargs;self.pipeline=None
            self.env=self.task_module=None;self.payload={};self.task_state={};self.closed=False
            created.append(self)
        def reset(self):
            assert self.loaded_task.path==new_inputs.task.path
            assert self.authoring_gpu is old.authoring_gpu
            events.append('new_reset')
            self.pipeline=old.pipeline
            self.task_module=NS(name='new_source')
            self.env=NS(scene=NS(objects=['new_actor'],active_systems={},clear=lambda:None))
            og.sim.scenes=[self.env.scene]
            og.sim.stop=lambda:setattr(og.sim,'stopped',True)
            og.sim.stopped=False
        def close(self):events.append('new_close')
    monkeypatch.setattr(esi_omnigibson,'ESISession',Session)
    monkeypatch.setattr(scene_capture_adapters,'gpu_evidence',lambda session,gpu:{'uuid':gpu['uuid'],'actual_mock':True})
    monkeypatch.setattr(app,'create_window',lambda:events.append('window_refresh'))
    scene,task=deepcopy(inputs[1]),deepcopy(inputs[2])
    identity='Test/Family/map/room/b.json'
    source=inputs[3]/identity;source.write_text(json.dumps({'scene':'map','room':'room','question':'New source.'}))
    scene.update(id='new_editor_scene',seed=11,source={'provider':'esi','json':identity})
    task.update(id='new_editor_task',defaults={'seed':11},source={'provider':'esi','json':identity})
    scene_file,task_file=tmp_path/'new_scene.yaml',tmp_path/'new_task.yaml'
    scene_file.write_text(yaml.safe_dump(scene));task_file.write_text(yaml.safe_dump(task))
    original_loader=editor.load_editor_inputs
    new_inputs=original_loader(scene_file,task_file,source_root=inputs[3])
    def load(*args,**kwargs):
        events.append('validate')
        return original_loader(*args,**kwargs,source_root=inputs[3])
    monkeypatch.setattr(editor,'load_editor_inputs',load)
    request={'request_id':'load1','action':'load','scene_file':str(scene_file),'task_file':str(task_file),
        'variant':'adapted','source_only':False,'scene_id':'category_new'}
    return app,old,og,events,created,new_inputs,request


def test_load_autosaves_before_discard_and_retains_application_pid_with_fresh_bound_session(persistent):
    app,old,og,events,created,new_inputs,request=persistent
    identity=id(og.app);old_simulator=id(og.sim)
    editor.write_json(app.output/'editor_request.json',request)
    app.process_request();app.process_request()
    response=editor.read_json(app.output/'editor_response.json')
    assert response['complete'] is True and response['scene_id']=='category_new'
    assert response['task_file']==str(new_inputs.task.path) and response['source_only'] is False
    assert response['scene_generation']==2 and response['pid']==os.getpid()
    assert response['app_identity']==identity and response['native_simulator_identity']!=old_simulator
    assert response['world_available'] is response['recovery_load_allowed'] is True
    assert response['autosave']['task_sha256']==old.loaded_task.file_sha256
    assert events.index('validate')<events.index('save')<events.index('source_cleanup')<events.index('clear')<events.index('stop')<events.index('new_reset')
    assert ('graphs_reset',{'usd':False}) not in events
    assert 'old_close_forbidden' not in events and old.closed is True
    assert len(created)==1 and app.session is created[0]
    assert app.session.loaded_task.path==new_inputs.task.path and hasattr(app.session,'scene_adapter')
    assert og.sim.scenes==[app.session.env.scene] and app.session.env.scene.objects==['new_actor']
    assert not og.sim.is_playing()
    exported=app.save_snapshot()
    assert exported['task_file']==str(new_inputs.task.path) and exported['seed']==11
    assert exported['scene_generation']==2 and exported['app_identity']==identity


@pytest.mark.parametrize('failure',['validation','stopped','save'])
def test_rejected_load_never_clears_or_creates_a_new_session(persistent,monkeypatch,failure):
    app,old,og,events,created,_,request=persistent
    if failure=='validation':request['task_file']=str(app.output/'missing_task.yaml')
    elif failure=='stopped':og.sim.stopped=True
    else:
        def fail_save(**kwargs):raise RuntimeError('Native export failed.')
        og.sim.save=fail_save
    editor.write_json(app.output/'editor_request.json',request);app.process_request()
    response=editor.read_json(app.output/'editor_response.json')
    assert response['complete'] is False and response['error']
    assert response['previous_world_retained'] is True and response['scene_cleared'] is False
    assert response['world_available'] is True and response['scene_generation']==1
    assert app.session is old and not old.closed and not created
    assert 'clear' not in events and 'source_cleanup' not in events and 'stop' not in events
    assert og.sim.scenes[0].objects==['old_actor']


def test_source_cleanup_that_already_clears_is_not_cleared_twice(persistent):
    app,old,og,events,_,_,request=persistent
    def cleanup(*args):events.append('source_cleanup');og.clear()
    old.pipeline.task_cleanup_runtime=cleanup
    response=app.load_scene(request)
    assert response['world_available'] is True
    assert events.count('clear')==1


def test_failed_new_load_keeps_same_app_and_allows_recovery_only_after_verified_empty_cleanup(persistent,monkeypatch):
    app,old,og,events,created,_,request=persistent
    original=editor.create_editor_session
    fail=[True]
    def create(inputs):
        session=original(inputs);reset=session.reset
        def failed_reset():
            reset()
            if fail[0]:raise RuntimeError('Actual new scene construction failed.')
        session.reset=failed_reset
        return session
    monkeypatch.setattr(editor,'create_editor_session',create)
    editor.write_json(app.output/'editor_request.json',request);app.process_request()
    response=editor.read_json(app.output/'editor_response.json')
    assert response['complete'] is False
    assert response['world_available'] is False and response['recovery_load_allowed'] is True
    assert response['previous_world_retained'] is False and response['scene_cleared'] is True
    assert og.app is app.application and og.sim.scenes==[]
    assert old.closed is True and 'old_close_forbidden' not in events
    fail[0]=False
    response=app.load_scene({**request,'request_id':'load2'})
    assert response['world_available'] is True and response['autosave'] is None
    assert response['scene_generation']==3 and len(created)==2


def test_failed_cleanup_disables_recovery_instead_of_claiming_an_empty_world(persistent,monkeypatch):
    app,old,og,events,_,_,request=persistent
    original_create=editor.create_editor_session
    original_discard=app.discard_session_world
    def create(inputs):
        session=original_create(inputs);reset=session.reset
        def fail_reset():reset();raise RuntimeError('New native initialization failed.')
        session.reset=fail_reset
        return session
    def discard(session):
        if session is not old:raise RuntimeError('Cannot prove native stage cleanup.')
        original_discard(session)
    monkeypatch.setattr(editor,'create_editor_session',create)
    monkeypatch.setattr(app,'discard_session_world',discard)
    editor.write_json(app.output/'editor_request.json',request);app.process_request()
    response=editor.read_json(app.output/'editor_response.json')
    assert response['complete'] is False
    assert response['world_available'] is response['recovery_load_allowed'] is False
    assert og.sim.scenes and app.application is og.app
    clear_count=events.count('clear')
    with pytest.raises(RuntimeError,match='not safely reusable'):app.load_scene(request)
    assert events.count('clear')==clear_count
