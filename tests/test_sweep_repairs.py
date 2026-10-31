"""Check native loading repairs without CUDA. Run `python -m pytest -q tests/test_sweep_repairs.py`."""
from pathlib import Path
from types import SimpleNamespace as NS

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
