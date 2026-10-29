"""Load legacy examples without depending on the active task catalogue.

Run their checks with: .venv/bin/python -m pytest -q tests/test_milestone85_tasks.py
"""

from pathlib import Path

from wia.tasks import loader


EXAMPLE_TASK_ROOT = Path(__file__).resolve().parent / "fixtures/task_examples"


def load_example_task(selection, **kwargs):
    kwargs.setdefault("task_root", EXAMPLE_TASK_ROOT)
    return loader.load_task(selection, **kwargs)


def example_task_path(selection):
    return loader.select_task(selection, task_root=EXAMPLE_TASK_ROOT)
