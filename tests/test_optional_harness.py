"""Enforce optional runtime imports. Run: pytest tests/test_optional_harness.py."""
import subprocess
import sys


def test_base_commands_and_contracts_never_import_harness():
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'showharness', 'core', 'gumi', 'mcp', 'torch'}:
            raise AssertionError('Forbidden optional import: ' + fullname)
sys.meta_path.insert(0, Block())
import wia, wia.contracts, wia.environments
from wia.cli import main
assert main(['backends']) == 0
assert main(['tasks']) == 0
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_actionable_missing_harness_error(monkeypatch):
    import wia.policies.show_harness as module
    def missing(name):
        raise ModuleNotFoundError("missing", name="showharness")
    monkeypatch.setattr(module, "import_module", missing)
    import pytest
    with pytest.raises(RuntimeError, match="editable checkout"):
        module.ShowHarnessExecutor().execute({})


def test_installed_dataset_entry_point_dispatches_one_episode(monkeypatch):
    import pytest
    pytest.importorskip('showharness')
    from showharness.datasets import evaluate_dataset
    import wia.integrations.show_harness as integration
    calls = []
    def execute(plan):
        calls.append(plan)
        return {'task_success': None, 'model_calls': 0}
    monkeypatch.setattr(integration, 'execute_plan', execute)
    plan = {'dataset_owned': True}
    assert evaluate_dataset('wia', plan) == {'task_success': None, 'model_calls': 0}
    assert calls == [plan]
