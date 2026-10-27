"""Optional whole-episode Harness executor. Run: python -m wia eval --config FILE."""
from importlib import import_module


class ShowHarnessExecutor:
    """Delegate to the existing authoritative runner without a second action loop."""
    def execute(self, plan):
        try:
            integration = import_module("showharness.evaluation")
        except ModuleNotFoundError as error:
            if error.name != "showharness":
                raise
            raise RuntimeError(
                "show_harness requires ShowHarness. Install its supported editable checkout: "
                "python -m pip install -e /absolute/path/to/Show-Harness-Explore[harness] "
                "and configure the ESI simulator environment for launch mode."
            ) from error
        return integration.execute_plan(plan)
