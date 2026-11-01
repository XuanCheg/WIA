"""Check native loading repairs without CUDA. Run `python -m pytest -q tests/test_sweep_repairs.py`."""
from pathlib import Path
from types import SimpleNamespace as NS
import json

import pytest

from wia.runtime.sim.esi_loading import reconcile_resident_timing, release_variant_preprocess


def test_released_containment_variant_retains_exact_source_identity(tmp_path):
    path = tmp_path/'source__criterion_4.json'
    verified = []
    task = NS(release_v1=True, source_path=path, verify_unchanged=lambda: verified.append(True))
    original = {'skip_reason':'storage_active_single_only_skips_variant_json'}
    result = release_variant_preprocess(task, 'tasks.physical_structure.rigid_containment', path, original)
    assert 'skip_reason' not in result and result['source_path'] == str(path)
    assert verified == [True] and 'skip_reason' in original
    task.release_v1 = False
    assert release_variant_preprocess(task, 'tasks.physical_structure.rigid_containment', path, original) == original
    task.release_v1 = True
    assert release_variant_preprocess(task, 'tasks.physical_structure.rigid_containment', tmp_path/'other.json', original) == original


def test_resident_rebuild_uses_declared_frequencies_and_keeps_application():
    application = object()
    og = NS(_wia_resident_episode=True, app=application,
            sim=NS(scenes=[], initial_physics_dt=1/120, initial_rendering_dt=1/30,
                   get_sim_step_dt=lambda: 1/30))
    calls = []
    def clear(**kwargs):
        calls.append(kwargs)
        og.sim = NS(scenes=[], initial_physics_dt=kwargs['physics_dt'],
                    initial_rendering_dt=kwargs['rendering_dt'], get_sim_step_dt=lambda: kwargs['sim_step_dt'])
    og.clear = clear
    defaults = {'physics_frequency':120, 'rendering_frequency':30, 'action_frequency':30}
    config = {'env':{'physics_frequency':240, 'rendering_frequency':60, 'action_frequency':60}}
    receipt = reconcile_resident_timing(og, config, defaults)
    assert calls == [{'physics_dt':1/240, 'rendering_dt':1/60, 'sim_step_dt':1/60}]
    assert og.app is application and receipt['application_preserved']
    assert reconcile_resident_timing(og, config, defaults) is None
    og.sim.scenes = ['a live world']
    with pytest.raises(RuntimeError, match='while a scene is loaded'):
        reconcile_resident_timing(og, {'env':{}}, defaults)


def test_resident_lifetime_recycles_owned_service_without_gpu_queries(tmp_path, monkeypatch):
    from wia.runtime import resident_episode as resident
    control = tmp_path/'service'
    control.mkdir()
    (control/'stop_service.json').write_text('{}')
    state = [{'state':'IDLE', 'pid':123, 'completed_scenes':4}]
    stopped = []
    monkeypatch.setattr(resident, 'read_service', lambda path: state[0])
    def stop(path):
        stopped.append(path)
        state[0] = None
    monkeypatch.setattr(resident, 'stop_service', stop)
    monkeypatch.setenv('WIA_RESIDENT_MAX_SCENES', '4')
    monkeypatch.setenv('WIA_RESIDENT_TASK_PROFILES', str(tmp_path/'profiles.json'))
    class Process:
        pid = 456
        def __init__(self, *args, **kwargs):
            state[0] = {'state':'IDLE', 'pid':self.pid, 'completed_scenes':0}
    monkeypatch.setattr(resident.subprocess, 'Popen', Process)
    client = resident.ResidentEpisodeClient(control, {'uuid':'GPU-fixture'}, {}, 'python')
    client.ensure()
    assert stopped == [control] and state[0]['pid'] == 456
    assert not (control/'stop_service.json').exists()


@pytest.mark.parametrize('changed,unknown_defaults', [(False,False), (True,False), (False,True)])
def test_submit_reuses_only_matching_native_launch_profiles(tmp_path, monkeypatch, changed, unknown_defaults):
    from wia.runtime import resident_episode as resident
    control, output = tmp_path/'control', tmp_path/'output'
    (control/'requests').mkdir(parents=True)
    output.mkdir()
    task = tmp_path/'task.yaml'
    defaults = {'USE_GPU_DYNAMICS':False, 'ENABLE_FLATCACHE':True, 'ENABLE_OBJECT_STATES':True}
    profile = {'gpu_dynamics':changed, 'macro_overrides':{}}
    profiles = tmp_path/'profiles.json'
    profiles.write_text(json.dumps({str(task):profile}))
    monkeypatch.setenv('WIA_RESIDENT_TASK_PROFILES', str(profiles))
    state = [{'state':'IDLE', 'pid':123, 'completed_scenes':1, 'native_macro_profile':defaults,
              **({} if unknown_defaults else {'macro_defaults':defaults})}]
    stopped = []
    def stop(path):
        stopped.append(path)
        state[0] = None
    monkeypatch.setattr(resident,'stop_service',stop)
    monkeypatch.setattr(resident,'read_service',lambda path:state[0])
    client = resident.ResidentEpisodeClient(control,{}, {},'python')
    def ensure(**kwargs):
        if state[0] is None:
            state[0] = {'state':'IDLE','pid':456,'completed_scenes':0}
    monkeypatch.setattr(client,'ensure',ensure)
    def forbidden(*args,**kwargs):
        raise AssertionError('Profile recycling must not query GPU status')
    monkeypatch.setattr(resident.subprocess,'check_output',forbidden)
    client.submit(['--task-file',str(task)],output)
    recycle = changed or unknown_defaults
    assert stopped == ([control] if recycle else [])
    assert json.loads((output/'resident_service.json').read_text())['native_pid'] == (456 if recycle else 123)
    assert len(list((control/'requests').glob('*.json'))) == 1


def test_native_profile_resolution_preserves_declared_overrides():
    from wia.runtime.resident_episode import native_macro_settings
    defaults = {'USE_GPU_DYNAMICS':False,'ENABLE_FLATCACHE':True}
    assert native_macro_settings(defaults,{'gpu_dynamics':True,'macro_overrides':{
        'USE_GPU_DYNAMICS':False,'ENABLE_FLATCACHE':False}}) == {
            'USE_GPU_DYNAMICS':True,'ENABLE_FLATCACHE':False}
    assert defaults == {'USE_GPU_DYNAMICS':False,'ENABLE_FLATCACHE':True}


def test_warm_initialization_falls_back_once_cold_and_preserves_failure_artifacts(tmp_path, monkeypatch):
    from wia.runtime import resident_episode as resident
    control, output = tmp_path/'control', tmp_path/'output'
    (control/'requests').mkdir(parents=True)
    output.mkdir()
    manifest = output/'episode_manifest.json'
    manifest.write_text('{"seed":0,"world_mode":"run"}\n')
    task = tmp_path/'task.yaml'
    profiles = tmp_path/'profiles.json'
    profiles.write_text(json.dumps({str(task):{'gpu_dynamics':False}}))
    monkeypatch.setenv('WIA_RESIDENT_TASK_PROFILES',str(profiles))
    macros = {'USE_GPU_DYNAMICS':False}
    state = [{'pid':123,'state':'IDLE','completed_scenes':1,
              'macro_defaults':macros,'native_macro_profile':macros}]
    monkeypatch.setattr(resident,'read_service',lambda path:state[0])
    client = resident.ResidentEpisodeClient(control,{}, {},'python')
    def ensure(**kwargs):
        if state[0] is None:
            state[0] = {'pid':456,'state':'IDLE','completed_scenes':0}
    monkeypatch.setattr(client,'ensure',ensure)
    argv = ['--task-file',str(task),'--scene-seed','0']
    client.submit(argv,output)
    assert client.started_reused and not client.cold_attempted
    old_request = client.request_id
    (output/'simulator.log').write_text('Original native error\n')
    (output/'initialization_failure.json').write_text('{"error":"load failed"}')
    (output/'gateway.log').write_text('Live gateway\n')
    stopped = []
    def stop(path):
        assert (output/'simulator.log').exists()
        stopped.append(state[0]['pid'])
        state[0] = None
    monkeypatch.setattr(resident,'stop_service',stop)
    assert client.retry_initialization('native load failed',agent_started=False)
    assert stopped == [123] and client.cold_attempted and not client.started_reused
    archive = output/'initialization_attempts'/old_request
    assert (archive/'simulator.log').read_text() == 'Original native error\n'
    assert (archive/'initialization_failure.json').exists()
    assert not (output/'initialization_failure.json').exists()
    assert manifest.read_text() == '{"seed":0,"world_mode":"run"}\n'
    assert (output/'gateway.log').read_text() == 'Live gateway\n'
    assert json.loads((control/'requests'/f'{client.request_id}.json').read_text())['native_argv'] == argv
    assert not client.retry_initialization('cold load failed',agent_started=False)
    assert stopped == [123]


@pytest.mark.parametrize('guard',['cold','agent','agent_log','completed_sequence'])
def test_fallback_never_replays_cold_failure_or_an_executed_agent(tmp_path, monkeypatch, guard):
    from wia.runtime import resident_episode as resident
    client = resident.ResidentEpisodeClient(tmp_path/'control',{}, {},'python')
    client.output = tmp_path
    client.cold_attempted = guard == 'cold'
    if guard == 'agent_log':
        (tmp_path/'agent.log').write_text('Agent already started\n')
    if guard == 'completed_sequence':
        (tmp_path/'scorer_validation.json').write_text('{"completed":true}')
    def forbidden(*args,**kwargs):
        raise AssertionError('No restart is allowed after this boundary')
    monkeypatch.setattr(resident,'stop_service',forbidden)
    assert not client.retry_initialization('native failure',agent_started=guard == 'agent')


def test_replacement_does_not_replay_unanswered_requests_or_overwrite_completed_receipts(tmp_path):
    from wia.runtime.resident_episode import retire_unanswered_requests
    (tmp_path/'requests').mkdir()
    (tmp_path/'responses').mkdir()
    for request in ('completed','interrupted'):
        (tmp_path/'requests'/f'{request}.json').write_text('{}')
    receipt=tmp_path/'responses/completed.json'
    receipt.write_text('{"reusable":true,"scene_exit_code":0}\n')
    retire_unanswered_requests(tmp_path)
    assert receipt.read_text()=='{"reusable":true,"scene_exit_code":0}\n'
    interrupted=json.loads((tmp_path/'responses/interrupted.json').read_text())
    assert interrupted['request_id']=='interrupted' and interrupted['retired_before_restart']
    assert interrupted['scene_exit_code'] is None and not interrupted['reusable']


def test_cold_fallback_restarts_real_owned_process_and_executes_only_the_new_request(tmp_path, monkeypatch):
    import sys
    import time
    from wia.runtime import resident_episode as resident
    script=tmp_path/'src/wia/runtime/commands/run_esi_episode_service.py'
    script.parent.mkdir(parents=True)
    script.write_text('''"""CPU fallback service fixture. Run through pytest only."""
import argparse, json, os, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--control-dir',type=Path)
p.add_argument('--task-profiles')
a=p.parse_args(); c=a.control_dir
first=not (c/'booted').exists()
(c/'booted').touch()
def state(value):
    tmp=c/'service.tmp';tmp.write_text(json.dumps(dict(pid=os.getpid(),**value)))
    tmp.replace(c/'service.json')
state(dict(state='IDLE',completed_scenes=int(first),macro_defaults={'USE_GPU_DYNAMICS':False},
           native_macro_profile={'USE_GPU_DYNAMICS':False}))
seen=set()
while not (c/'stop_service.json').exists():
    for path in (c/'requests').glob('*.json'):
        if path.name in seen or (c/'responses'/path.name).exists():continue
        seen.add(path.name);r=json.loads(path.read_text());out=Path(r['output_dir'])
        with (c/'handled.jsonl').open('a') as f:f.write(json.dumps(dict(pid=os.getpid(),request_id=r['request_id']))+'\\n')
        if first:
            state(dict(state='SCENE_RUNNING',request_id=r['request_id']))
            (out/'simulator.log').write_text('Original warm load failure\\n')
            (out/'initialization_failure.json').write_text('{}')
            # Deliberately leave this request unanswered to exercise retirement.
        else:
            (c/'responses'/path.name).write_text(json.dumps(dict(request_id=r['request_id'],reusable=True,scene_exit_code=0)))
    time.sleep(.01)
''')
    monkeypatch.setattr(resident,'WIA_RUNTIME_ROOT',tmp_path)
    task=tmp_path/'task.yaml'; profiles=tmp_path/'profiles.json'
    profiles.write_text(json.dumps({str(task):{'gpu_dynamics':False}}))
    monkeypatch.setenv('WIA_RESIDENT_TASK_PROFILES',str(profiles))
    control=tmp_path/'control';output=tmp_path/'output';output.mkdir()
    processes=[];popen=resident.subprocess.Popen
    def start(*args,**kwargs):
        process=popen(*args,**kwargs);processes.append(process);return process
    monkeypatch.setattr(resident.subprocess,'Popen',start)
    client=resident.ResidentEpisodeClient(control,{'uuid':'CPU-fixture'}, {},sys.executable)
    try:
        client.submit(['--task-file',str(task),'--scene-seed','0'],output)
        previous=client.request_id
        deadline=time.monotonic()+5
        while not (output/'initialization_failure.json').exists():
            assert time.monotonic()<deadline
            time.sleep(.01)
        assert client.retry_initialization('Synthetic warm failure',agent_started=False)
        response=control/'responses'/f'{client.request_id}.json'
        deadline=time.monotonic()+5
        while not response.exists():
            assert time.monotonic()<deadline
            time.sleep(.01)
        assert processes[0].wait(timeout=5)==0 and processes[0].pid!=processes[1].pid
        handled=[json.loads(line) for line in (control/'handled.jsonl').read_text().splitlines()]
        assert [row['request_id'] for row in handled]==[previous,client.request_id]
        assert json.loads((control/'responses'/f'{previous}.json').read_text())['retired_before_restart']
        assert (output/'initialization_attempts'/previous/'simulator.log').exists()
    finally:
        resident.stop_service(control,timeout=2)
        for process in processes:
            try:process.wait(timeout=5)
            except resident.subprocess.TimeoutExpired:process.kill();process.wait()
