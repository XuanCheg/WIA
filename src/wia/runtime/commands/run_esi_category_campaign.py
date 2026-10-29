#!/usr/bin/env python3
"""Run or resume the final category conversion campaign on the current server.

Run: /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python
src/wia/runtime/commands/run_esi_category_campaign.py run --campaign-root ~/esi_category_conversion/CAMPAIGN
Existing CaptureQueue workers are reused; this command never stops resident worlds.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.runtime.esi_category_runner import CategoryRunner, aggregate_feedback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory', 'assign', 'run', 'status', 'feedback', 'refresh-evidence'])
    parser.add_argument('--campaign-root', type=Path, default=Path.home() / 'esi_category_conversion/current')
    parser.add_argument('--queue-root', type=Path)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--review-file', type=Path)
    parser.add_argument('--supervisors', choices=['codex', 'external'], default='codex',
        help='Real Codex category processes, or externally attached native category supervisors.')
    parser.add_argument('--once', action='store_true', help='One resumable tick; no busy waiting.')
    parser.add_argument('--exit-when-idle', action='store_true',
        help='Exit after CPU work and assessment comparisons finish instead of continuing to watch Review feedback.')
    parser.add_argument('--publish-all', action='store_true',
        help='Publish all checked candidates with missing capture evidence explicitly recorded; do not wait for perfect samples.')
    parser.add_argument('--interval-s', type=float, default=5)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text()) if args.config else None
    runner = CategoryRunner(args.campaign_root, queue_root=args.queue_root, corpus=args.corpus, config=config)
    if args.command == 'inventory':
        value = runner.scheduler.inventory
    elif args.command == 'assign':
        value = runner.scheduler.fill_slots()
    elif args.command == 'feedback':
        if not args.review_file:
            parser.error('feedback requires --review-file')
        value = aggregate_feedback(runner.queue_root, args.review_file, scheduler=runner.scheduler)
    elif args.command == 'status':
        value = runner.status()
    elif args.command == 'refresh-evidence':
        value = runner.refresh_evidence()
    else:
        while True:
            value = runner.step(supervisors=args.supervisors, publish_all=args.publish_all)
            summary = {k: value[k] for k in ('inventory', 'states', 'review_count', 'failed_count', 'warning_count')}
            print(json.dumps(summary), flush=True)
            comparisons_pending = value.get('assessment_review', {}).get('state') == 'RUNNING'
            if args.once or (args.exit_when_idle and not value['active_supervisors']
                    and not value['states'].get('PENDING') and not comparisons_pending):
                return 0
            time.sleep(max(.1, args.interval_s))
    print(json.dumps(value, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
