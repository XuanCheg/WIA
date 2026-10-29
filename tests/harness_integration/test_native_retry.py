"""Check retained-world repair hooks and publication evidence preservation.

Run: .venv/bin/python -m pytest -q tests/test_native_retry.py
"""
from types import SimpleNamespace

import pytest
import yaml

from wia.runtime.scene_review import read_json, write_json
import wia.tasks.native_retry as retry
import wia.runtime.commands.convert_esi_rigid_catalog as converter


def test_warm_retry_retains_original_failure_and_references_the_actual_candidate(tmp_path):
    output = tmp_path / 'sampling'
    attempt = output / 'warm/attempt_001'
    attempt.mkdir(parents=True)
    original = b'{"status":"not_screened","reason":"old rejection"}\n'
    (output / 'support_bank.json').write_bytes(original)
    candidate = attempt / 'placement_000.yaml'
    candidate.write_text('status: pending\n')
    views = attempt / 'review_views/placement_000'
    write_json(views / 'native_review_complete.json', {'scene_sha256': 'exact_hash'})
    bank = {'status': 'screened', 'accepted': [{'scene_file': candidate.name, 'scene_sha256': 'exact_hash'}]}
    retry.promote_retry(output, attempt, bank)
    active = read_json(output / 'support_bank.json')
    assert (output / 'warm/history/original_support_bank.json').read_bytes() == original
    assert output / active['accepted'][0]['scene_file'] == candidate
    assert bank['accepted'][0]['scene_file'] == candidate.name
    assert read_json(output / 'review_views/placement_000/native_review_complete.json')['scene_sha256'] == 'exact_hash'


@pytest.mark.parametrize('status, expected', [('not_screened', ['retry', 'close']), ('screened', ['close'])])
def test_failed_worker_repairs_before_teardown_but_completed_worker_returns(tmp_path, monkeypatch, status, expected):
    monkeypatch.setenv('PBS_JOBID', 'interactive_test')
    monkeypatch.setattr(retry.sys, 'argv', ['sample_esi_supports.py', '--worker', '--output-dir', str(tmp_path)])
    write_json(tmp_path / 'warm_keepalive.json', {'enabled': True, 'job_id': 'interactive_test'})
    write_json(tmp_path / 'support_bank.json', {'status': status})
    calls = []
    session = SimpleNamespace(close=lambda: calls.append('close'))
    monkeypatch.setattr(retry, 'keepalive', lambda *args: calls.append('retry'))
    retry.install_sampling_keepalive(SimpleNamespace(session=session))
    session.close()
    assert calls == expected


def test_warm_worker_is_retained_after_the_original_wall_allowance(tmp_path, monkeypatch):
    monkeypatch.setenv('PBS_JOBID', 'interactive_test')
    scene = tmp_path / 'scene.yaml'
    scene.write_text(yaml.safe_dump({'source': {'json': 'example.json'}}))
    output = tmp_path / 'sampling'
    args = SimpleNamespace(data_root=tmp_path, seed=0, attempts=1, wall_budget_s=1,
        initialization_wall_budget_s=1)
    ticks = iter([0., 5.])
    monkeypatch.setattr(converter.time, 'monotonic', lambda: next(ticks))
    waits = []

    def launch(*args, **kwargs):
        write_json(output / 'warm_state.json', {'status': 'waiting', 'pid': 321, 'job_id': 'interactive_test'})
        return SimpleNamespace(pid=321, wait=lambda timeout: waits.append(timeout) or 0)

    monkeypatch.setattr(converter.subprocess, 'Popen', launch)
    monkeypatch.setattr(converter.os, 'killpg', lambda *args: pytest.fail('A retained world must not be terminated.'))
    assert converter.native_worker(scene, output, args) == 0
    assert waits == [30]
