#!/usr/bin/env python3
"""Serve an independent CPU-only WIA bench design/layout review.

Run: python -B src/wia/runtime/commands/review_wia.py --snapshot SNAPSHOT --output-dir REVIEW
The default address is http://127.0.0.1:8626/. No simulator is imported or started.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.runtime.wia_review import WiaReview  # noqa: E402
import wia.runtime.commands.review_esi_scenes as review_esi_scenes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8626)
    args = parser.parse_args(argv)
    review_esi_scenes.ASSETS = ROOT / 'src/wia/runtime/web/wia_review'
    store = WiaReview(args.snapshot, args.output_dir)
    server = review_esi_scenes.create_server(store, args.host, args.port)
    print(f'[wia-review] http://{args.host}:{server.server_port}/', flush=True)
    print('[wia-review] Feedback scope: design/layout only; no native certification.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
