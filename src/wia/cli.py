"""Model-independent CLI. Run: python -m wia --help."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wia", description="World Interactive Agency")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("eval", help="Evaluate a configured task suite")
    evaluation.add_argument("--config", required=True)
    evaluation.add_argument("--dry-run", action="store_true")
    commands.add_parser("tasks", help="List task resources without importing a model runtime")
    commands.add_parser("backends", help="List optional ShowHarness backend metadata")
    args = parser.parse_args(argv)
    try:
        if args.command == "backends":
            from .evaluation.runner import BACKENDS
            print(json.dumps(BACKENDS, indent=2))
        elif args.command == "tasks":
            from .tasks.loader import discover_tasks
            print("\n".join(map(str, discover_tasks())))
        else:
            from .evaluation import prepare, run
            prepared = prepare(args.config)
            result = prepared if args.dry_run else run(prepared)
            print(json.dumps(result, indent=2, allow_nan=False))
            if not args.dry_run and any(item["status"] not in ("completed_success", "completed_unsuccessful") for item in result["trials"]):
                return 1
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(2, f"wia: {error}\n")
    return 0
