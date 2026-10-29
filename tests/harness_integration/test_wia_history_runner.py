"""Check diagnostic history CLI guards and submitted-input traces with fake agents.

Run: python -B -m pytest -q tests/test_wia_history_runner.py -p no:cacheprovider
No native controller, model, network or GPU is launched.
"""
import asyncio
from dataclasses import replace
import json
from types import ModuleType, SimpleNamespace as NS
import sys

import numpy as np
from PIL import Image
import pytest

from core.action_interface import BackendCapabilities
from core.agent.decision import ModelReply
from wia.runtime.agent.wia_history_projection import project_phase2_input
from wia.runtime.agent.zero_shot import run_episode
from core.robot_cameras import ROBOT_CAMERA_LABELS
from core.sensor_observations import SensorReadings
import wia.runtime.commands.run_esi_agent as cli
from harness_integration.test_wia_history_projection import observation


class TemporalFixture:
    temporal_episode = True
    temporal_policy = {'phase_1_max_decisions': 1}
    capabilities = BackendCapabilities(base_translation=True)
    initialization = {'temporal_episode': True, 'scene_reset': False}
    action_context = {}
    temporal_phase = 'phase_1'

    def __init__(self):
        self.old = observation(2, 1)
        self.new = replace(observation(3, 1), agentview_rgb=self.old.agentview_rgb + 20,
                           wrist_rgb=self.old.wrist_rgb + 20,
                           additional_rgb={key: value + 20 for key, value in self.old.additional_rgb.items()},
                           sensors=SensorReadings(head_depth_m=np.full((3, 4), 9., np.float32),
                                                  wrist_depth_m=np.full((3, 4), 10., np.float32)))

    async def observe(self):
        return self.old if self.temporal_phase == 'phase_1' else self.new

    async def report_temporal(self, event, answer, observed):
        if event == 'phase_1_complete':
            self.temporal_phase = 'phase_2'
        else:
            assert event == 'answer'
        return {'ok': True, 'phase': self.temporal_phase}


class ProjectingFakeAgent:
    snapshot_cache = True

    def __init__(self, condition='full', *, retry=False):
        self.wia_history_condition, self.retry = condition, retry
        self.inputs, self.phase2_pending = [], []
        self.phase2_calls = 0

    def description(self):
        return {'backend': 'codex', 'model': 'fixture', 'reasoning_effort': 'medium'}

    def prepare_request(self, request):
        if self.wia_history_condition != 'full' and request.action_context.get('temporal_phase') == 'phase_2':
            self.phase2_pending.append(request.previous_phase_observation)
            request, _ = project_phase2_input(request, condition=self.wia_history_condition,
                oracle_facts='Explicit fixture historical facts.' if self.wia_history_condition == 'history_oracle' else None)
        return replace(request, assembled_prompt='Fake model prompt over the prepared sensor bundle.',
                       resident_prompt='Fake instructions.', prompt_phase='step', world_observation_log=True)

    async def decide(self, request):
        self.inputs.append(request)
        phase = request.action_context.get('temporal_phase')
        if phase == 'phase_1':
            return ModelReply('{"action":"DONE"}', {'world_observation': 'PHASE1_COMPLETE'})
        self.phase2_calls += 1
        if self.retry and self.phase2_calls == 1:
            return ModelReply('not a valid command', {'world_observation': 'Inspecting.'})
        return ModelReply('{"action":"DONE"}', {'world_observation': 'ANSWER: yes'})


@pytest.mark.parametrize('condition', ['history_removed', 'history_oracle'])
def test_phase2_logs_only_prepared_current_sensors_and_marks_withheld_boundary(condition, tmp_path):
    environment, agent = TemporalFixture(), ProjectingFakeAgent(condition, retry=True)
    result = asyncio.run(run_episode(environment, agent, tmp_path, max_steps=4))
    assert result['termination_reason'] == 'model_done'
    assert result['wia_history_condition'] == condition
    assert len(agent.phase2_pending) == 2
    assert agent.phase2_pending[0] is environment.old
    assert agent.phase2_pending[1] is None
    transition = result['temporal_transitions'][0]
    assert transition['final_phase_1_delivered'] is False
    assert transition['final_phase_1_withheld'] is True
    assert transition['final_phase_1_withheld_reason'] == 'explicit_diagnostic_history_intervention'
    final_rows = [row for row in result['observation_coverage'] if row['stage'] == 'final_phase_1']
    assert len(final_rows) == 1
    assert final_rows[0]['delivered_to_agent'] is False
    assert final_rows[0]['withheld_from_phase_2'] is True
    phase2 = tmp_path / 'step_0002'
    canonical = json.loads((phase2 / 'before/observation.json').read_text())
    assert canonical['initial_overview']['status'] == 'available'
    assert canonical['action_history']
    assert (phase2 / 'before' / canonical['images']['initial_overview_rgb']['path']).is_file()
    projection = json.loads((phase2 / 'model_inputs.json').read_text())
    assert {row['name'] for row in projection} == set(ROBOT_CAMERA_LABELS)
    assert not (phase2 / 'model_images/initial_overview_rgb.png').exists()
    for row in projection:
        with Image.open(tmp_path / row['path']) as image:
            assert np.array_equal(np.asarray(image), environment.new.rgb_images()[row['name']])
    audit = json.loads((phase2 / 'wia_history_projection.json').read_text())
    assert audit['native_capture_failure'] is False
    assert audit['overview_removal_reason'] == 'explicit_diagnostic_history_intervention'
    assert audit['canonical_observation_preserved'] is True
    assert set(audit['model_sensor_metadata']['images']) == set(ROBOT_CAMERA_LABELS)
    assert audit['model_sensor_metadata']['action_history'] == []
    artifacts = json.loads((phase2 / 'snapshot_artifacts.json').read_text())
    assert 'initial_overview_rgb' not in artifacts['canonical_references']
    depth = json.loads((phase2 / 'raw_depth_input.json').read_text())
    assert depth['snapshot_id'] == environment.new.snapshot_id
    assert depth['sensors'] == {name: environment.new.metadata()['sensors'][name]
                                for name in ('head_depth_m', 'wrist_depth_m')}
    logs = [json.loads(line) for line in (tmp_path / 'model_inputs.jsonl').read_text().splitlines()]
    submitted = [row for row in logs if row['event'] == 'step_input' and row['step'] == 2]
    assert len(submitted) == 2
    for row in submitted:
        assert row['passed_to_model'] is True
        assert {item['name'] for item in row['images']} == set(ROBOT_CAMERA_LABELS)
        assert row['observation']['action_history'] == []
        assert row['observation']['initial_overview']['capture'] is None
        assert 'OLD_EVENT_CANARY' not in json.dumps(row)
    assert all(request.previous_phase_observation is None for request in agent.inputs[1:])


def test_default_full_transition_keeps_original_final_phase1_delivery_and_overview(tmp_path):
    environment, agent = TemporalFixture(), ProjectingFakeAgent()
    summary = asyncio.run(run_episode(environment, agent, tmp_path, max_steps=4))
    assert summary['termination_reason'] == 'model_done'
    assert summary['temporal_transitions'][0]['final_phase_1_delivered'] is True
    assert 'final_phase_1_withheld' not in summary['temporal_transitions'][0]
    assert agent.inputs[-1].previous_phase_observation is environment.old
    assert 'wia_history_condition' not in summary
    projection = json.loads((tmp_path / 'step_0002/model_inputs.json').read_text())
    assert {row['name'] for row in projection} == {*ROBOT_CAMERA_LABELS, 'initial_overview_rgb'}


def valid_options(**overrides):
    values = dict(wia_history_condition='history_removed', wia_history_oracle_file=None,
                  task_file=__import__('wia.resources', fromlist=['ROOT']).ROOT / 'explore_to_achieve/wia_v1/release/tasks/t40.yaml',
                  backend='codex', memory_mode='none', observation_condition='current',
                  world_observation_log=True, resume_from=None, episode_manifest='fixture.json')
    values.update(overrides)
    return NS(**values)


@pytest.mark.parametrize('override', [
    {'task_file': None}, {'backend': 'local_http'}, {'memory_mode': 'model_notes'},
    {'observation_condition': 'initial_view_only'}, {'world_observation_log': False},
    {'resume_from': 'previous'}, {'episode_manifest': None},
])
def test_history_conditions_reject_unsupported_cli_paths(override):
    with pytest.raises(ValueError, match='Diagnostic history conditions require'):
        cli.validate_wia_history_condition(valid_options(**override))


@pytest.mark.parametrize('policy,physical', [(None, True), ({'phase_1_max_decisions': 1}, False)])
def test_history_conditions_require_temporal_manifest_and_physical_task(policy, physical):
    task = NS(definition=NS(private=NS(evaluation=NS(parameters={'task_edit2_physical': physical}))))
    with pytest.raises(ValueError, match='temporal manifest policy and a physical temporal task'):
        cli.validate_wia_history_condition(valid_options(), loaded_task=task, manifest={'temporal_policy': policy})


def test_oracle_file_is_required_only_for_explicit_oracle_condition(tmp_path):
    with pytest.raises(ValueError, match='requires --wia-history-oracle-file'):
        cli.validate_wia_history_condition(valid_options(wia_history_condition='history_oracle'))
    for condition in ('full', 'history_removed'):
        with pytest.raises(ValueError, match='allowed only for history_oracle'):
            cli.validate_wia_history_condition(valid_options(wia_history_condition=condition,
                wia_history_oracle_file=tmp_path / 'facts.txt'))
    cli.validate_wia_history_condition(valid_options(wia_history_condition='full', resume_from='unchanged-full-resume'))
    with pytest.raises(SystemExit) as caught:
        cli.main(['--task', 'Legacy task', '--output-dir', str(tmp_path / 'output'),
                  '--wia-history-condition', 'history_removed'])
    assert caught.value.code == 2


def test_oracle_file_must_be_nonempty_utf8_and_is_preserved_verbatim(tmp_path):
    path = tmp_path / 'facts.txt'
    facts = '  Explicit fixture historical facts.\n'
    path.write_text(facts, encoding='utf-8')
    assert cli.read_wia_history_oracle_file(path) == facts
    path.write_text(' \n', encoding='utf-8')
    with pytest.raises(ValueError, match='nonempty UTF-8'):
        cli.read_wia_history_oracle_file(path)
    path.write_bytes(b'\xff')
    with pytest.raises(UnicodeError):
        cli.read_wia_history_oracle_file(path)


@pytest.mark.parametrize('condition', ['full', 'history_removed', 'history_oracle'])
def test_cli_phase2_factory_retains_settings_and_uses_fresh_distinct_output(monkeypatch, tmp_path, condition):
    from wia.tasks import loader
    import core.agent.codex_session as sessions
    import wia.runtime.agent.prompts as prompts
    task = loader.load_task(valid_options().task_file, variant='adapted', run_mode='diagnostic')
    manifest = {'episode_id': 'fixture', 'agent_session': 'native_persistent', 'max_steps': 4, 'world_mode': 'pause',
                'world_advances_during_inference': False, 'observation_condition': 'current',
                'action_mode': 'fixed', 'model_name': 'fixture-model', 'reasoning_effort': 'medium',
                'wall_timeout_s': 60., 'world_observation_log': True,
                'temporal_policy': {'phase_1_max_decisions': 1}}
    monkeypatch.setattr(loader, 'load_task', lambda *args, **kwargs: task)
    monkeypatch.setattr(loader, 'verify_manifest', lambda *args, **kwargs: manifest)
    monkeypatch.setattr(loader, 'public_episode_contract', lambda *args: {})
    monkeypatch.setattr(loader, 'sha256', lambda *args: 'fixture-sha256')
    created = []
    def codex(**kwargs):
        child = NS(settings=kwargs)
        created.append(child)
        return child
    monkeypatch.setattr(sessions, 'CodexEpisodeSession', codex)
    monkeypatch.setattr(prompts, 'PromptAssembler', lambda **kwargs: NS(settings=kwargs))
    wrappers = []
    def wrap(**kwargs):
        wrapper = NS(settings=kwargs)
        wrappers.append(wrapper)
        return wrapper
    module = ModuleType('wia.runtime.agent.wia_history_session')
    module.WiaHistorySession = wrap
    monkeypatch.setitem(sys.modules, module.__name__, module)
    class Environment:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
    monkeypatch.setattr(cli, 'MCPRobotEnvironment', Environment)
    output = tmp_path / 'episode'
    facts = 'Explicit fixture historical facts.\n'
    async def run(environment, agent, directory, **kwargs):
        directory.mkdir()
        if condition == 'full':
            assert agent is created[0]
            assert not wrappers
        else:
            assert agent is wrappers[0]
            assert agent.settings['condition'] == condition
            assert agent.settings['oracle_facts'] == (facts if condition == 'history_oracle' else None)
            child = agent.settings['phase2_factory']()
            old, new = created[0].settings, child.settings
            for key in ('public_task', 'episode_configuration', 'model', 'reasoning_effort',
                        'reasoning_summary', 'executable', 'timeout_s', 'snapshot_cache'):
                assert new[key] == old[key]
            assert new['resume_state'] is None
            assert new['output'] == output / 'phase2_native'
            assert new['assembler'].settings == {'wia_history_condition': condition, 'world_observation': True, 'debug_reachability': False}
        return dict(termination_reason='model_done', task_success=None, executed_actions=[],
                    agent_decisions=2, model_calls=2, macro_actions=0, primitive_actions=0, model_logs={})
    monkeypatch.setattr(cli, 'run_episode', run)
    arguments = ['--task-file', str(task.path), '--variant', 'adapted', '--run-mode', 'diagnostic',
                 '--episode-manifest', str(tmp_path / 'manifest.json'), '--output-dir', str(output),
                 '--wia-history-condition', condition]
    if condition == 'history_oracle':
        oracle = tmp_path / 'facts.txt'
        oracle.write_text(facts, encoding='utf-8')
        arguments += ['--wia-history-oracle-file', str(oracle)]
    assert cli.main(arguments) == 0
