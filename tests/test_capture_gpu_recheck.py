"""Check resource revalidation before queue claims; run pytest on this file."""
import subprocess

import pytest

from wia.runtime.commands.run_esi_capture_worker import fresh_gpu_available


@pytest.mark.parametrize('clients,reading,shared,expected', [
    ('', 'GPU-test, 0, 87, 24000', False, True),
    ('42, GPU-test', 'GPU-test, 0, 87, 24000', False, False),
    ('42, GPU-other', 'GPU-test, 0, 87, 24000', False, True),
    ('', 'GPU-test, 1, 87, 24000', False, False),
    ('', 'GPU-test, 0, 1024, 23000', False, False),
    ('42, GPU-test', 'GPU-test, 50, 8000, 16000', True, True),
    ('42, GPU-test', 'GPU-test, 50, 16000, 8000', True, False),
    ('', 'GPU-test, N/A, N/A, N/A', False, False),
])
def test_live_resource_readback(monkeypatch, clients, reading, shared, expected):
    monkeypatch.setattr(subprocess, 'check_output', lambda command, **kwargs:
        clients if '--query-compute-apps=pid,gpu_uuid' in command else reading)
    assert fresh_gpu_available({'uuid': 'GPU-test', 'shared_gpu': shared,
                                'shared_gpu_min_free_mib': 12000}) is expected


def test_driver_query_failure_does_not_claim(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, 'nvidia-smi')
    monkeypatch.setattr(subprocess, 'check_output', fail)
    assert fresh_gpu_available({'uuid': 'GPU-test'}) is False
