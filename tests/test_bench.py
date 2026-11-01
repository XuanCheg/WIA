"""Bench and billing checks without GPUs or paid calls. Run python -m pytest -q tests/test_bench.py."""
from copy import deepcopy
import io
import json
from pathlib import Path

import pytest
from wia.evaluation import bench
from wia.evaluation.telemetry import receipt, totals, format_totals


def prepared(tmp_path, count=8):
    return {'output_dir': str(tmp_path/'bench'), 'plans': [{'index': i,
        'task': {'path': f'task{i}', 'seed': 0, 'variant': 'fixture'},
        'policy': {'backend': 'openai_chat_completions', 'model': 'fixture'},
        'evaluation': {'max_steps': 64, 'execution_mode': 'batch'},
        'environment': {'mode': 'launch'}, 'output_dir': str(tmp_path/f'trial{i}')}
        for i in range(count)]}


def test_shards_preserve_trial_identity_and_science(tmp_path):
    value = prepared(tmp_path, 60)
    old = deepcopy(value)
    workers = bench.shards(value, [1, 3, 5, 7], 8800)
    assert value == old
    assert [len(w['plans']) for w in workers] == [15]*4
    plans = sorted((p for w in workers for p in w['plans']), key=lambda p:p['index'])
    assert [p['index'] for p in plans] == list(range(60))
    for original, changed in zip(value['plans'], plans):
        assert original['task'] == changed['task']
        assert original['evaluation'] == changed['evaluation']
    assert len({w['plans'][0]['environment']['port'] for w in workers}) == 4
    assert len({p['output_dir'] for p in plans}) == 60


def test_billing_distinguishes_missing_zero_and_subtotals():
    first = receipt({'prompt_tokens':100, 'completion_tokens':40, 'cost':0,
        'prompt_tokens_details':{'cached_tokens':80},
        'completion_tokens_details':{'reasoning_tokens':30}}, model='fixture', decision=1, seconds=2)
    second = receipt({}, model='fixture', decision=2, seconds=3)
    value = totals([first, second])
    assert first['total_tokens'] == 140  # Cached/reasoning tokens are subsets.
    assert value['cost_usd'] == {'known_sum':0, 'missing_requests':1}
    assert value['input_tokens'] == {'known_sum':100, 'missing_requests':1}
    assert 'unknown' in format_totals(value)
    assert first['cost_source'] == 'provider_receipt' and second['cost_source'] is None


def test_preflight_blocks_text_only_third_party_without_model_call(monkeypatch):
    data = {'data': {'architecture':{'input_modalities':['text']},
        'reasoning':{'supported_efforts':['medium']}, 'endpoints':[{'provider_name':'novita'}]}}
    monkeypatch.setattr(bench, 'urlopen', lambda *a,**k: io.BytesIO(json.dumps(data).encode()))
    with pytest.raises(ValueError, match='no official inclusionai endpoint'):
        bench.gateway_preflight({'base_url':'https://ai-gateway.vercel.sh/v1',
            'model':'inclusionai/ling-3.1-flash-1019','reasoning_effort':'medium'})


def test_preflight_rejects_unsupported_effort(monkeypatch):
    data = {'data': {'architecture':{'input_modalities':['text','image']},
        'reasoning':{'supported_efforts':['low','high']},
        'endpoints':[{'provider_name':'deepseek','max_completion_tokens':384000}]}}
    monkeypatch.setattr(bench, 'urlopen', lambda *a,**k: io.BytesIO(json.dumps(data).encode()))
    policy = {'base_url':'https://ai-gateway.vercel.sh/v1','model':'deepseek/fixture','reasoning_effort':'medium'}
    with pytest.raises(ValueError, match='medium'):
        bench.gateway_preflight(policy)
    assert bench.gateway_preflight({**policy, 'reasoning_effort':'low'}) == data['data']


def test_dry_run_does_not_access_gpus_or_network(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('External side effect')
    monkeypatch.setattr(bench, 'idle_devices', forbidden)
    monkeypatch.setattr(bench, 'gateway_preflight', forbidden)
    result = bench.launch(prepared(tmp_path), dry_run=True)
    assert len(result['workers']) == 4
    assert result['persistent_simulator'] is True
    assert result['log_interval_s'] == 30
    assert not (tmp_path/'bench').exists()


@pytest.mark.parametrize('skip_gpu_check', [False, True])
@pytest.mark.parametrize('auto', [False, True])
def test_supervisor_accounts_for_crashed_worker(tmp_path, monkeypatch, skip_gpu_check, auto):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    devices = [{'index':i,'uuid':f'GPU-{i}','idle':True} for i in range(2)]
    if skip_gpu_check:
        monkeypatch.setenv('WIA_SKIP_GPU_CHECK', '1')
        monkeypatch.setattr(bench, 'inventory_devices', lambda: [{**d, 'idle': False} for d in devices])
        def forbidden():
            raise AssertionError('GPU status must not be queried')
        monkeypatch.setattr(bench, 'idle_devices', forbidden)
    else:
        monkeypatch.delenv('WIA_SKIP_GPU_CHECK', raising=False)
        monkeypatch.setattr(bench,'idle_devices',lambda:devices)
    monkeypatch.setattr(bench,'gateway_preflight',lambda policy:None)
    class Process:
        count = 0
        def __init__(self, command, **kwargs):
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == f'GPU-{Process.count}'
            assert kwargs['env']['WIA_RESIDENT_CONTROL_DIR'].endswith(f'resident_{Process.count:02d}')
            assert kwargs['env']['WIA_BENCH_EVENTS'].endswith(f'worker_{Process.count:02d}_events.jsonl')
            if skip_gpu_check:
                assert kwargs['env']['WIA_SKIP_GPU_CHECK'] == '1'
            self.pid = 100 + Process.count
            self.returncode = Process.count
            worker = json.loads(Path(command[-1]).read_text())
            if Process.count == 0:
                directory=Path(worker['output_dir']);directory.mkdir()
                (directory/'results.json').write_text(json.dumps({'trials':[
                    {'index':p['index'],'status':'completed_unsuccessful'} for p in worker['plans']]}))
                (directory/'api_usage.jsonl').write_text(json.dumps(receipt(
                    {'input_tokens':3,'output_tokens':2,'cost':0.01},model='fixture',decision=1,seconds=1))+'\n')
            Process.count += 1
        def poll(self):
            return self.returncode
    monkeypatch.setattr(bench.subprocess,'Popen',Process)
    result=bench.launch(prepared(tmp_path,4),gpus=None if auto else [0,1],worker_count=2,port_base=18880)
    assert [t['status'] for t in result['trials']] == [
        'completed_unsuccessful','runtime_error','completed_unsuccessful','runtime_error']
    assert result['worker_exit_codes'] == [0,1]
    assert result['request_accounting']['cost_usd']['known_sum'] == 0.01


def test_unchecked_inventory_queries_only_device_identity(monkeypatch):
    def query(command, **kwargs):
        assert command == ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits']
        return '2, GPU-two\n5, GPU-five\n'
    monkeypatch.setattr(bench.subprocess, 'check_output', query)
    devices = bench.inventory_devices()
    assert devices == [{'index': 2, 'uuid': 'GPU-two'}, {'index': 5, 'uuid': 'GPU-five'}]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,2')
    assert [d['index'] for d in bench.visible_devices(devices)] == [5, 2]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-five')
    assert [d['index'] for d in bench.visible_devices(devices)] == [5]


@pytest.mark.parametrize('skip', [False, True])
def test_native_scene_query_obeys_script_opt_out(monkeypatch, skip):
    from types import SimpleNamespace
    from wia.runtime.commands import run_esi_explore_episode as native
    if skip:
        monkeypatch.setenv('WIA_SKIP_GPU_CHECK', '1')
    else:
        monkeypatch.delenv('WIA_SKIP_GPU_CHECK', raising=False)
    def query(command, **kwargs):
        assert command[1] == ('--query-gpu=index,uuid' if skip else
                              '--query-gpu=index,uuid,utilization.gpu,memory.used')
        return SimpleNamespace(stdout='4, GPU-four\n' if skip else '4, GPU-four, 99, 20000\n')
    monkeypatch.setattr(native.subprocess, 'run', query)
    _, devices, eligible, observed_skip = native.query_local_gpus()
    assert devices[0][1].strip() == 'GPU-four'
    assert eligible == ([4] if skip else [])
    assert observed_skip is skip


def test_lifecycle_tail_preserves_partial_record(tmp_path, monkeypatch, capsys):
    from wia.runtime.lifecycle_log import emit, read_events
    path = tmp_path/'events.jsonl'
    monkeypatch.setenv('WIA_BENCH_EVENTS', str(path))
    monkeypatch.setenv('WIA_BENCH_GPU', '3')
    emit('scene_loading', task='T52')
    records, offset = read_events(path, 0)
    assert records[0]['gpu'] == '3' and records[0]['task'] == 'T52'
    with path.open('a') as stream:
        stream.write('{"event":"scene_ready"')
    assert read_events(path, offset) == ([], offset)
    with path.open('a') as stream:
        stream.write('}\n')
    records, final = read_events(path, offset)
    assert records == [{'event':'scene_ready'}] and final > offset
    assert 'scene_loading' in capsys.readouterr().out


def test_progress_refresh_is_throttled_but_events_are_immediate(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('WIA_SKIP_GPU_CHECK', '1')
    monkeypatch.setattr(bench, 'inventory_devices', lambda: [{'index':0, 'uuid':'GPU-0'}])
    monkeypatch.setattr(bench, 'gateway_preflight', lambda policy: None)
    clock = [0]
    monkeypatch.setattr(bench.time, 'monotonic', lambda: clock[0])
    def sleep(seconds):
        assert seconds == 1
        clock[0] += seconds
        if clock[0] == 2:
            (tmp_path/'bench/worker_00_events.jsonl').write_text(
                json.dumps({'time':'now','event':'scene_ready','gpu':0})+'\n')
    monkeypatch.setattr(bench.time, 'sleep', sleep)
    class Process:
        pid = 100
        returncode = 0
        def __init__(self, command, **kwargs):
            assert 'WIA_RESIDENT_CONTROL_DIR' not in kwargs['env']
        def poll(self):
            return 0 if clock[0] >= 32 else None
    monkeypatch.setattr(bench.subprocess, 'Popen', Process)
    bench.launch(prepared(tmp_path, 1), gpus=[0], worker_count=1,
                 persistent_simulator=False, port_base=18890)
    output = capsys.readouterr().err
    assert output.count('[bench #') + output.count('[bench -') == 3  # Initial, 30s, final.
    assert output.count('scene_ready') == 1


def test_native_profiles_load_all_tasks_without_gpu_imports():
    from wia.evaluation import prepare
    from wia.resources import ROOT
    from wia.runtime.resident_episode import task_profiles
    value = prepare(ROOT/'configs/bench_deepseek_60.yaml')
    profiles = task_profiles(value['plans'])
    assert len(profiles) == 60
    assert {row['native_module'] for row in profiles.values() if row['gpu_dynamics']} == {
        'tasks.physical_structure.deformable'}
    assert all(Path(path).is_absolute() for path in profiles)


def test_resident_service_is_child_process_and_reused(tmp_path, monkeypatch):
    from wia.runtime import resident_episode as resident
    control = tmp_path/'resident'
    monkeypatch.setenv('WIA_RESIDENT_TASK_PROFILES', str(tmp_path/'profiles.json'))
    state = [None]
    monkeypatch.setattr(resident, 'read_service', lambda path: state[0])
    launched = []
    class Process:
        pid = 123
        def __init__(self, command, **kwargs):
            assert 'tmux' not in command
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-fixture'
            assert kwargs['start_new_session'] is True
            state[0] = {'state':'IDLE', 'pid':self.pid, 'completed_scenes':0}
            launched.append(command)
    monkeypatch.setattr(resident.subprocess, 'Popen', Process)
    client = resident.ResidentEpisodeClient(control, {'uuid':'GPU-fixture'}, {}, 'python')
    client.ensure()
    client.ensure()
    assert len(launched) == 1


def test_stop_service_rejects_unrelated_pid(tmp_path, monkeypatch):
    from wia.runtime import resident_episode as resident
    (tmp_path/'process.json').write_text(json.dumps({'pid':123}))
    monkeypatch.setattr(resident, 'read_service', lambda path: None)
    monkeypatch.setattr(resident, 'process_matches', lambda path, pid: False)
    def forbidden(*args):
        raise AssertionError('Unrelated processes must never be signaled')
    monkeypatch.setattr(resident.os, 'kill', forbidden)
    resident.stop_service(tmp_path)


def test_owned_service_starts_and_stops_without_tmux_or_cuda(tmp_path, monkeypatch):
    import sys
    from wia.runtime import resident_episode as resident
    script = tmp_path/'src/wia/runtime/commands/run_esi_episode_service.py'
    script.parent.mkdir(parents=True)
    script.write_text('''"""CPU service fixture. Run through ResidentEpisodeClient in pytest."""
import argparse, json, os, time
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--control-dir', type=Path)
parser.add_argument('--task-profiles')
args = parser.parse_args()
(args.control_dir/'service.json').write_text(json.dumps({'pid':os.getpid(), 'state':'IDLE'}))
while not (args.control_dir/'stop_service.json').exists():
    time.sleep(.01)
''')
    monkeypatch.setattr(resident, 'WIA_RUNTIME_ROOT', tmp_path)
    monkeypatch.setenv('WIA_RESIDENT_TASK_PROFILES', str(tmp_path/'profiles.json'))
    processes = []
    original = resident.subprocess.Popen
    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(resident.subprocess, 'Popen', popen)
    control = tmp_path/'control'
    client = resident.ResidentEpisodeClient(control, {'uuid':'GPU-fixture'}, {}, sys.executable)
    try:
        client.ensure()
        assert resident.read_service(control)['pid'] == processes[0].pid
        resident.stop_service(control)
        assert processes[0].wait(timeout=5) == 0
        assert resident.read_service(control) is None
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
