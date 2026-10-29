#!/usr/bin/env python3
"""Expose the existing R1 GUMI executor as MCP tools, without starting a simulator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = __import__('wia.resources', fromlist=['ROOT']).ROOT
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8615")
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8616,
                        help="Optional standalone MCP port; use run_gumi_gateway.py for shared /gumi and /mcp.")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    try:
        from wia.runtime.web.mcp_server import create_server
    except ImportError:
        parser.exit(1, "Install MCP dependencies: uv pip install --python .venv/bin/python "
                    "-r requirements/requirements-mcp.txt\n")
    server = create_server(args.backend_url)
    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport="streamable-http", host=args.host, port=args.port,
                   json_response=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
