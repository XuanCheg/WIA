"""Fixed v1 loads without physical requalification. Run: pytest tests/test_release_load_policy.py."""
import json
from wia.resources import ROOT
from wia.tasks.loader import load_task, load_scene
from wia.tasks.scene import DeclarativeScene
from wia.tasks.scene_only import SceneOnly


class NoPhysicsAccess:
    def __getattr__(self, name):
        raise AssertionError('Release loader unexpectedly queried physics: ' + name)


def test_fixed_v1_task_skips_physical_checks_in_benchmark_mode(tmp_path):
    task = load_task(ROOT / 'explore_to_achieve/wia_v1/release/tasks/t01.yaml', variant='adapted')
    assert task.release_v1 and task.run_mode == 'benchmark'
    scene = DeclarativeScene(task)
    result = scene.qualify(NoPhysicsAccess(), NoPhysicsAccess(), tmp_path)
    assert result.status == 'loaded_release_v1'
    assert result.checks['physical_checks_run'] is False
    assert result.checks['physical_safety_inferred'] is False
    assert json.loads((tmp_path / 'private_scene/qualification.json').read_text())['status'] == result.status


def test_fixed_v1_scene_skips_physical_checks(tmp_path):
    # Scene loading preserves resource/schema validation but does not inspect a native world.
    scene = load_scene(ROOT / 'explore_to_achieve/wia_v1/release/scenes/t01.yaml', variant='adapted')
    result = SceneOnly(scene).qualify(NoPhysicsAccess(), NoPhysicsAccess(), tmp_path)
    assert result.checks['physical_checks_run'] is False


def test_evaluation_plan_accepts_fixed_v1_in_benchmark_mode(tmp_path):
    import yaml
    from wia.evaluation.runner import prepare
    config = tmp_path / 'trial.yaml'
    config.write_text(yaml.safe_dump({
        'tasks': [{'path': str(ROOT / 'explore_to_achieve/wia_v1/release/tasks/t01.yaml'),
                   'variant': 'adapted'}],
        'policy': {'type': 'show_harness', 'backend': 'codex_cli', 'model': 'test'},
        'environment': {'mode': 'launch'}, 'evaluation': {'run_mode': 'benchmark'},
        'output_dir': str(tmp_path / 'run')}))
    result = prepare(config)
    assert result['plans'][0]['evaluation']['run_mode'] == 'benchmark'
    assert not (tmp_path / 'run').exists()
