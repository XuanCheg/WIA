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
    assert not (tmp_path/'bench').exists()


def test_supervisor_accounts_for_crashed_worker(tmp_path, monkeypatch):
    devices = [{'index':i,'uuid':f'GPU-{i}','idle':True} for i in range(2)]
    monkeypatch.setattr(bench,'idle_devices',lambda:devices)
    monkeypatch.setattr(bench,'gateway_preflight',lambda policy:None)
    class Process:
        count = 0
        def __init__(self, command, **kwargs):
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
    result=bench.launch(prepared(tmp_path,4),gpus=[0,1],port_base=18880)
    assert [t['status'] for t in result['trials']] == [
        'completed_unsuccessful','runtime_error','completed_unsuccessful','runtime_error']
    assert result['worker_exit_codes'] == [0,1]
    assert result['request_accounting']['cost_usd']['known_sum'] == 0.01
