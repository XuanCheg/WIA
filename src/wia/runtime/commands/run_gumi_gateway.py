#!/usr/bin/env python3
"""One public port: /gumi monitor and /mcp Agent tools, with a CPU-side gateway."""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
import sys
from pathlib import Path

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-url', required=True, help='Simulator HTTP backend, e.g. http://GPU-node:8615.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8615)
    parser.add_argument('--review-url', help='Optional CPU scene gallery, e.g. http://127.0.0.1:8625.')
    parser.add_argument('--runtime-manifest', type=Path,
                        help='Save private loaded-schema/source provenance for experiment preflight.')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535.')
    from wia.runtime.web.http_gateway import create_gateway

    server = create_gateway(args.backend_url, review_url=args.review_url)
    if args.runtime_manifest:
        from wia.runtime.gateway_runtime import gateway_runtime_manifest
        args.runtime_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.runtime_manifest.write_text(json.dumps(gateway_runtime_manifest(
            ROOT, backend_url=args.backend_url, port=args.port), indent=2) + '\n')
    print(f'[gumi] http://{args.host}:{args.port}/gumi', flush=True)
    print(f'[mcp]  http://{args.host}:{args.port}/mcp', flush=True)
    if args.review_url:
        print(f'[review] http://{args.host}:{args.port}/review/', flush=True)
    server.run(transport='streamable-http', host=args.host, port=args.port, json_response=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
