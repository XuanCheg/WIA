"""Resolve checkout resources. Run: python -c 'from wia.resources import ROOT; print(ROOT)'."""
import os
from pathlib import Path

ROOT = Path(os.environ.get('WIA_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve()
DATA_ROOT = ROOT / 'explore_to_achieve'
ESI_ROOT = Path(os.environ.get('WIA_ESI_ROOT', ROOT / 'ESI')).expanduser().resolve()
TASK_ROOT = DATA_ROOT / 'tasks'
SCENE_ROOT = DATA_ROOT / 'scenes'
UPSTREAM_ROOT = ESI_ROOT / 'dataset/json_clean'
AUTHORED_ROOT = DATA_ROOT / 'sources/json_clean'


def resolve_migrated_path(path, root=ROOT):
    """Resolve unchanged tracked resource pointers; leave unrelated absolute paths intact."""
    import hashlib
    import json

    path, root = Path(path).expanduser(), Path(root).expanduser().resolve()
    if not path.is_absolute():
        return (root / path).resolve()
    provenance_path = root / 'provenance/extraction.json'
    if not provenance_path.is_file():
        return path
    provenance = json.loads(provenance_path.read_text())
    source_root = provenance.get('source_repository')
    if not source_root:
        return path
    try:
        relative = path.relative_to(Path(source_root))
    except ValueError:
        return path
    key = relative.as_posix()
    if key not in provenance.get('byte_preserved_resources', []):
        return path
    expected = provenance.get('resource_sha256', {}).get(key)
    target = root / relative
    if (expected is None or not target.is_file()
            or hashlib.sha256(target.read_bytes()).hexdigest() != expected
            or path.exists() and (not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected)):
        raise ValueError('Known migrated resource bytes changed: ' + str(path))
    return target.resolve()
