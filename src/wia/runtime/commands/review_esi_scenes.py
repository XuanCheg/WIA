#!/usr/bin/env python3
"""Serve the CPU scene review gallery without starting a simulator.

Run: .venv/bin/python src/wia/runtime/commands/review_esi_scenes.py --conversion-root RUN
--output-dir REVIEW --port 8625. Forward port 8625, or mount it at /review
with src/wia/runtime/commands/run_gumi_gateway.py --review-url http://127.0.0.1:8625.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlsplit

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.runtime.scene_review import ReviewConflict, SceneReview  # noqa: E402

ASSETS = ROOT / "src/wia/runtime/web/scene_review"


def create_server(store, host="127.0.0.1", port=8625):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, body, content_type="application/json", status=200, headers=None):
            if not isinstance(body, bytes):
                body = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path in ("/", "/index.html", "/app.js", "/style.css"):
                    name = "index.html" if path == "/" else path[1:]
                    mime = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8",
                            "style.css": "text/css; charset=utf-8"}[name]
                    return self.reply((ASSETS / name).read_bytes(), mime)
                if path in ("/api/scenes", "/api/catalogue"):
                    query = {key: value[0] for key, value in parse_qs(urlsplit(self.path).query).items()}
                    return self.reply(json.dumps(store.catalogue(filters=query,
                        page=int(query['page']) if 'page' in query else None,
                        page_size=int(query.get('page_size', 48))), allow_nan=False))
                if path == "/api/scene":
                    query = parse_qs(urlsplit(self.path).query)
                    try:
                        options = {'variant_id': query['variant_id'][0]} if 'variant_id' in query else {}
                        return self.reply(json.dumps({"scene": store.scene_detail(query["id"][0], **options)}, allow_nan=False))
                    except KeyError:
                        return self.reply('{"error":"This task is not in the current completed or diagnostic list."}', status=404)
                if path == '/api/category-report':
                    query = parse_qs(urlsplit(self.path).query)
                    return self.reply(store.category_report(query.get('big_task', [''])[0],
                        query.get('small_task', [''])[0]), 'text/markdown; charset=utf-8')
                if path == '/api/diagnostic-log':
                    query = parse_qs(urlsplit(self.path).query)
                    return self.reply(store.diagnostic_log(query['id'][0], int(query['index'][0])), 'text/plain; charset=utf-8')
                if path == "/api/reviews.csv":
                    return self.reply(store.export_csv(), "text/csv; charset=utf-8", headers={
                        "Content-Disposition": 'attachment; filename="scene_reviews.csv"'})
                parts = path.strip("/").split("/")
                if len(parts) == 3 and parts[0] == "images":
                    image = store.images.get((parts[1], parts[2]))
                    if image is not None:
                        return self.reply(image.read_bytes(), "image/png")
                self.reply('{"error":"Not found."}', status=404)
            except (OSError, ValueError, KeyError) as error:
                self.reply(json.dumps({"error": str(error)}), status=500)

        def do_POST(self):
            parts = urlsplit(self.path).path.strip("/").split("/")
            if len(parts) != 3 or parts[:2] != ["api", "reviews"]:
                return self.reply('{"error":"Not found."}', status=404)
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Send application/json.")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 100000:
                    raise ValueError("Invalid request size.")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Send a JSON object.")
                record = store.save_review(parts[2], payload)
                self.reply(json.dumps({"review": record}, allow_nan=False))
            except ReviewConflict as error:
                self.reply(json.dumps({"error": str(error)}), status=409)
            except (ValueError, TypeError) as error:
                self.reply(json.dumps({"error": str(error)}), status=400)
            except KeyError:
                self.reply('{"error":"Unknown scene."}', status=404)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-root", type=Path)
    parser.add_argument("--queue-root", type=Path, help="Show accepted capture candidates incrementally alongside legacy scenes.")
    parser.add_argument('--campaign-root', type=Path, help='Show the complete Big Task / Small Task inventory and supervisor reports.')
    parser.add_argument('--catalogue-file', type=Path,
        help='Read the completed-task metadata snapshot; no image validation or directory scans.')
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "explore_to_achieve/scenes")
    parser.add_argument("--variant", default="tabletop")
    parser.add_argument("--archive-superseded-reviews", action="store_true",
        help="Archive annotations for replaced or excluded layouts before serving the final merged review.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8625)
    args = parser.parse_args(argv)
    if not args.conversion_root and not args.queue_root:
        parser.error("Provide --conversion-root, --queue-root, or both.")
    store_type = SceneReview
    options = {}
    if args.catalogue_file or args.campaign_root:
        from wia.runtime.scene_review_live import LiveSceneReview
        store_type = LiveSceneReview
        options["catalogue_file"] = args.catalogue_file or args.campaign_root / "review_catalogue/snapshot.json"
    store = store_type(args.conversion_root, args.output_dir, data_root=args.data_root,
                       variant=args.variant, queue_root=args.queue_root, campaign_root=args.campaign_root, **options)
    if args.archive_superseded_reviews:
        retired = store.archive_superseded_reviews()
        print(f"[scene-review] Archived {len(retired['removed_review_ids'])} superseded review records.", flush=True)
    reused = store.import_previews() if store_type is SceneReview else 0
    server = create_server(store, args.host, args.port)
    print(f"[scene-review] Reused {reused} final-layout preview sets.", flush=True)
    print(f"[scene-review] http://{args.host}:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
