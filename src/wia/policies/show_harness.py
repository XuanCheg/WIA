"""Optional whole-episode Harness executor. Run: python -m wia eval --config FILE."""
from importlib import import_module


class ShowHarnessExecutor:
    """Delegate to the existing authoritative runner without a second action loop."""
    def execute(self, plan):
        try:
            interface = import_module("showharness.datasets")
            integration = import_module("wia.integrations.show_harness")
        except ModuleNotFoundError as error:
            if error.name != "showharness":
                raise
            raise RuntimeError(
                "show_harness requires ShowHarness. Install its supported editable checkout: "
                "python -m pip install -e /absolute/path/to/Show-Harness-Explore[harness] "
                "and configure the ESI simulator environment for launch mode."
            ) from error
        return interface.evaluate_dataset(integration.WIADatasetAdapter(), plan)
