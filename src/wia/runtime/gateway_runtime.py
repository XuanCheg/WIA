"""Private startup provenance for the persistent CPU gateway."""
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from wia.runtime.resources import runtime_source

GATEWAY_SOURCES = ('src/wia/runtime/web/mcp_server.py', 'src/wia/runtime/web/http_gateway.py', 'core/observations.py',
                   'core/feedback_semantics.py', 'core/action_compiler.py',
                   'core/action_interface.py', 'src/wia/runtime/gateway_runtime.py', 'src/wia/runtime/commands/run_gumi_gateway.py')


def gateway_runtime_manifest(root, *, backend_url, port):
    from core.observations import FEEDBACK_ERRORS
    from core.feedback_semantics import CONTINUABLE_MOTION_ERRORS
    return {'schema': 'gateway_runtime_v1', 'pid': os.getpid(),
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'backend_url': backend_url.rstrip('/'), 'port': port,
            'feedback_errors': sorted(FEEDBACK_ERRORS),
            'continuable_motion_errors': sorted(CONTINUABLE_MOTION_ERRORS),
            'source_hashes': {name: hashlib.sha256(runtime_source(name, root).read_bytes()).hexdigest()
                              for name in GATEWAY_SOURCES}}


def validate_gateway_runtime(manifest, root, *, backend_url):
    if not isinstance(manifest, dict) or manifest.get('schema') != 'gateway_runtime_v1':
        raise RuntimeError('Gateway startup provenance is missing; reload the persistent gateway before a model trial.')
    if manifest.get('backend_url') != backend_url.rstrip('/') or manifest.get('port') != 8615:
        raise RuntimeError('Gateway startup provenance refers to another backend or port.')
    expected = gateway_runtime_manifest(root, backend_url=backend_url, port=8615)
    changed = [name for name, digest in expected['source_hashes'].items()
               if manifest.get('source_hashes', {}).get(name) != digest]
    if changed or any(manifest.get(key) != expected[key]
                      for key in ('feedback_errors', 'continuable_motion_errors')):
        raise RuntimeError(f'Persistent gateway has stale feedback schemas or source: {changed}. Reload it before a model trial.')
    pid = manifest.get('pid')
    if type(pid) is not int or pid <= 0:
        raise RuntimeError('Gateway startup provenance has no valid process identifier.')
    try:
        os.kill(pid, 0)
    except OSError as error:
        raise RuntimeError('The recorded gateway process is no longer running.') from error
