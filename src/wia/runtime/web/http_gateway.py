"""Mount GUMI and MCP on one public HTTP port, keeping simulator ownership intact."""
from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

import requests
from core.action_compiler import MacroLimits
from starlette.background import BackgroundTask
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from wia.runtime.web.mcp_server import create_server

HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailer', 'transfer-encoding', 'upgrade', 'host', 'content-length'}

WAITING_PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GUMI camera view</title><style>
body{margin:0;background:#10141b;color:#e5ecf5;font:16px system-ui;display:grid;
place-items:center;min-height:100vh}main{max-width:560px;padding:32px;line-height:1.6}
a{color:#91bfff}h1{font-size:24px}</style><main><h1>Waiting for a scene</h1>
<p>The camera view will reopen when a scene is available. This page reconnects automatically.</p>
<p><a href="/gumi/">Refresh camera view</a></p></main><script>
let checking=false;setInterval(async()=>{if(checking)return;checking=true;
try{const response=await fetch('/gumi/api/state',{cache:'no-store'});
if(response.ok)location.reload()}catch(error){}finally{checking=false}},2000);
</script></html>"""


def create_gateway(backend_url, *, review_url=None):
    """SDK owns MCP sessions/lifespan; custom HTTP routes serve the GUMI proxy."""
    parsed = urlsplit(backend_url)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('Backend must be an HTTP(S) URL without credentials/query/fragment.')
    backend_url = backend_url.rstrip('/')
    server = create_server(backend_url)

    if review_url:
        review = urlsplit(review_url)
        if (review.scheme not in ('http', 'https') or not review.hostname or review.username
                or review.password or review.query or review.fragment or review.path not in ('', '/')):
            raise ValueError('Review backend must be an HTTP(S) origin without credentials/query/fragment.')
        review_url = review_url.rstrip('/')

        @server.custom_route('/review', methods=['GET'], include_in_schema=False)
        async def review_root(request):
            return RedirectResponse('/review/')

        @server.custom_route('/review/{path:path}', methods=['GET', 'POST'], include_in_schema=False)
        async def review_proxy(request):
            path = request.url.path[len('/review'):]
            if request.url.query:
                path += '?' + request.url.query
            headers = {key: value for key, value in request.headers.items() if key.lower() not in HOP_HEADERS}
            headers['Accept-Encoding'] = 'identity'
            try:
                body = await request.body()
                response = await asyncio.to_thread(requests.request, request.method, review_url + path,
                    data=body, headers=headers, stream=True, timeout=(3, 15))
            except requests.RequestException:
                return JSONResponse({'error': 'Scene review service unavailable.'}, status_code=503)
            response_headers = {key: value for key, value in response.headers.items()
                                if key.lower() not in HOP_HEADERS}
            return StreamingResponse(response.iter_content(chunk_size=16384), status_code=response.status_code,
                headers=response_headers, background=BackgroundTask(response.close))

    @server.custom_route('/', methods=['GET'], include_in_schema=False)
    @server.custom_route('/gumi', methods=['GET'], include_in_schema=False)
    async def root(request):
        return RedirectResponse('/gumi/')

    @server.custom_route('/gumi/{path:path}', methods=['GET', 'POST'], include_in_schema=False)
    @server.custom_route('/api/{path:path}', methods=['GET', 'POST'], include_in_schema=False)
    @server.custom_route('/snapshot/{path:path}', methods=['GET'], include_in_schema=False)
    @server.custom_route('/stream/{path:path}', methods=['GET'], include_in_schema=False)
    @server.custom_route('/recordings/{path:path}', methods=['GET'], include_in_schema=False)
    async def proxy(request):
        path = request.url.path
        if path.startswith('/gumi/'):
            path = path[len('/gumi'):]
        if request.url.query:
            path += '?' + request.url.query
        headers = {key: value for key, value in request.headers.items() if key.lower() not in HOP_HEADERS}
        headers['Accept-Encoding'] = 'identity'
        body = await request.body()
        try:
            read_timeout = 1320
            if path.split('?', 1)[0] in ('/api/control/claim', '/api/agent/command', '/api/agent/step', '/api/step', '/mcp'):
                read_timeout = None
            response = await asyncio.to_thread(requests.request, request.method, backend_url+path,
                                               data=body, headers=headers, stream=True, timeout=(5, read_timeout))
        except (requests.RequestException, ValueError, TypeError, KeyError):
            if request.method == 'GET' and path.split('?', 1)[0] in ('/', '/index.html'):
                return HTMLResponse(WAITING_PAGE, headers={'Cache-Control': 'no-store'})
            return JSONResponse({'ok': False, 'message': 'GUMI backend unavailable.'}, status_code=503)
        response_headers = {key: value for key, value in response.headers.items()
                            if key.lower() not in HOP_HEADERS}
        # StreamingResponse runs this synchronous iterator in its worker pool.
        # No simulator step or sensor read happens in the gateway/HTTP thread.
        return StreamingResponse(response.iter_content(chunk_size=16384), status_code=response.status_code,
                                 headers=response_headers, background=BackgroundTask(response.close))

    return server
