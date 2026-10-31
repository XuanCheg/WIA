"""Model-independent CLI. Run: python -m wia --help."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wia", description="World Interactive Agency")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("eval", help="Evaluate a configured task suite")
    evaluation.add_argument("--config", required=True)
    evaluation.add_argument("--dry-run", action="store_true")
    debug = commands.add_parser("debug", help="Debug one task with recording and blocking scene retention")
    debug.add_argument("--config", required=True)
    debug.add_argument("--dry-run", action="store_true")
    bench = commands.add_parser('bench', help='Run a task suite across independent GPU workers')
    bench.add_argument('--config', required=True)
    bench.add_argument('--gpus', default='auto', help='Comma-separated physical indices, or auto')
    bench.add_argument('--workers', type=int, default=4)
    bench.add_argument('--port-base', type=int, default=8800)
    bench.add_argument('--output')
    bench.add_argument('--dry-run', action='store_true')
    control = commands.add_parser("debug-control", help="Resume or close a retained debug trial")
    control.add_argument("--trial", required=True)
    control.add_argument("--action", choices=("resume", "close"), required=True)
    commands.add_parser("tasks", help="List task resources without importing a model runtime")
    commands.add_parser("backends", help="List optional ShowHarness backend metadata")
    args = parser.parse_args(argv)
    try:
        if args.command == "debug-control":
            from .runtime.debug_lifecycle import submit_control
            print(json.dumps(submit_control(args.trial, args.action), indent=2))
        elif args.command == "backends":
            from .evaluation.runner import BACKENDS
            print(json.dumps(BACKENDS, indent=2))
        elif args.command == "tasks":
            from .tasks.loader import discover_tasks
            print("\n".join(map(str, discover_tasks())))
        else:
            from .evaluation import prepare, run
            prepared = prepare(args.config, execution_mode="debug" if args.command == "debug" else None)
            if args.command == 'bench':
                from pathlib import Path
                from .evaluation.bench import launch
                if args.output:
                    prepared['output_dir'] = str(Path(args.output).expanduser().resolve())
                gpus = None if args.gpus == 'auto' else [int(x) for x in args.gpus.split(',')]
                result = launch(prepared, gpus=gpus, worker_count=args.workers,
                                port_base=args.port_base, dry_run=args.dry_run)
            else:
                result = prepared if args.dry_run else run(prepared)
            print(json.dumps(result, indent=2, allow_nan=False))
            if not args.dry_run and any(result.get('worker_exit_codes', [])):
                return 1
            if not args.dry_run and any(item["status"] not in ("completed_success", "completed_unsuccessful") for item in result["trials"]):
                return 1
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(2, f"wia: {error}\n")
    return 0
