"""Verify diagnostic controller/runtime parity without native simulation.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B \
  -m pytest -q tests/test_wia_native_runtime_config.py -p no:cacheprovider
"""
import ast
from dataclasses import replace
import inspect
import json
from types import SimpleNamespace as NS

import pytest
import yaml

from wia.runtime.sim.esi_arm import ArmControlConfig
from wia.runtime.sim.esi_base import BaseControlConfig
from wia.runtime.sim.esi_gripper import GripperControlConfig
from wia.runtime.sim.esi_rotation import RotationControlConfig
from wia.runtime.sim.esi_trunk import TrunkControlConfig
from wia.tasks.scene import ScenePatches
from wia.tasks.schema import SceneDefinition
import wia.runtime.commands.check_esi_core_native as diagnostic


@pytest.fixture
def scene():
    path = diagnostic.ROOT / 'explore_to_achieve/wia_v1/scenes/rigid_t25_v1.yaml'
    return SceneDefinition.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')))


def test_scene_low_grasp_config_and_fine_trunk_overrides_are_both_retained(scene):
    configs = diagnostic.load_native_control_configs(scene.runtime, manipulation=True, primitive_profile='fine')
    fine = json.loads((diagnostic.ROOT / 'configs/esi_primitives_fine.json').read_text())
    trunk = TrunkControlConfig.load(diagnostic.ROOT / scene.runtime.trunk_config)
    assert scene.runtime.trunk_config == 'configs/esi_trunk_low_grasp.json'
    assert scene.runtime.post_controller_settle_steps == 120
    assert configs['trunk'] == replace(trunk, **fine['trunk'])
    assert configs['trunk'].minimum_height_m == .58
    assert configs['trunk'].translation_m == .005
    assert configs['trunk'].translation_tolerance_m == .00075
    assert configs['base'] == replace(BaseControlConfig.load(), **fine['base'])
    assert configs['arm'] == replace(ArmControlConfig.load(), **fine['arm'])
    assert configs['rotation'] == replace(RotationControlConfig.load(), **fine['rotation'])
    assert configs['gripper'] == GripperControlConfig.load()


def test_fixed_profile_uses_baseline_controls_and_declared_scene_trunk(scene):
    configs = diagnostic.load_native_control_configs(scene.runtime, manipulation=True, primitive_profile='fixed')
    assert configs == {
        'base': BaseControlConfig.load(), 'arm': ArmControlConfig.load(),
        'rotation': RotationControlConfig.load(), 'gripper': GripperControlConfig.load(),
        'trunk': TrunkControlConfig.load(diagnostic.ROOT / 'configs/esi_trunk_low_grasp.json'),
    }
    assert configs['trunk'].translation_m == .02
    assert configs['trunk'].minimum_height_m == .58
    base_only = diagnostic.load_native_control_configs(scene.runtime, manipulation=False, primitive_profile='fine')
    assert base_only == {'base': BaseControlConfig.load(), 'arm': None,
                         'rotation': None, 'gripper': None, 'trunk': None}


@pytest.mark.parametrize('steps', [0, 120])
def test_production_raw_settle_count_and_runtime_record_precede_scoring(scene, tmp_path, steps):
    definition = scene.model_copy(update={'runtime': scene.runtime.model_copy(
        update={'post_controller_settle_steps': steps})})
    loaded = NS(definition=definition, variant='tabletop', seed=0)
    adapter = ScenePatches(loaded, definition.bindings)
    events = ['executor_constructed']
    output = tmp_path / 'runtime_configuration.json'
    simulator = NS(current_time=1.25)
    def step():
        assert output.is_file()
        initial_record = json.loads(output.read_text())
        assert initial_record['post_controller_settle']['complete'] is False
        events.append('raw_physics_step')
        simulator.current_time += .01
    simulator.step = step
    session = NS(pipeline=NS(og=NS(sim=simulator)))
    configs = diagnostic.load_native_control_configs(definition.runtime,
        manipulation=True, primitive_profile='fine')
    result = {}
    record = diagnostic.prepare_native_runtime_configuration(adapter, session, configs,
        primitive_profile='fine', result=result, output_file=output)
    events.extend(['initial_release', 'initial_controls', 'condition_snapshot', 'scorer_start'])
    assert type(adapter) is ScenePatches
    assert events == ['executor_constructed'] + ['raw_physics_step'] * steps + [
        'initial_release', 'initial_controls', 'condition_snapshot', 'scorer_start']
    assert adapter.post_controller_settle_steps == steps
    assert record == result['runtime_configuration'] == json.loads(output.read_text())
    settling = record['post_controller_settle']
    assert settling['physics_step_method'] == 'og.sim.step'
    assert settling['declared_steps'] == settling['applied_steps'] == steps
    assert settling['native_time_before_s'] == 1.25
    assert settling['native_time_after_s'] == pytest.approx(1.25 + .01 * steps)
    assert settling['complete'] is True
    assert settling['credited_as_episode_action'] is False
    assert record['effective_configs']['trunk']['minimum_height_m'] == .58
    assert record['effective_configs']['trunk']['translation_m'] == .005
    assert record['scene_trunk_config_file'] == str(diagnostic.ROOT / 'configs/esi_trunk_low_grasp.json')


def test_main_places_declared_settling_after_executor_and_before_initial_controls_and_scoring():
    tree = ast.parse(inspect.getsource(diagnostic.main))
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, []).append(node.lineno)
    settle = calls['prepare_native_runtime_configuration'][0]
    assert calls['create_r1_executor'][0] < settle
    assert settle < calls['run_initial_controls'][0] < calls['condition_snapshot'][0]
    release = next(node.lineno for node in ast.walk(tree) if isinstance(node, ast.If)
                   and isinstance(node.test, ast.Attribute) and node.test.attr == 'initial_release'
                   and node.lineno > calls['create_r1_executor'][0])
    assert settle < release
