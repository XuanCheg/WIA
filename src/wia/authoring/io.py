"""Benchmark JSON and identity helpers. Run: python -m pytest tests/test_extraction.py."""
import hashlib
import json
import os
from pathlib import Path
import uuid

def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default

def write_json(path, value):
    """Publish a complete document; writers never share temporary filenames."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def review_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def fingerprint(scene_path, source_path, variant, seed):
    value = {"scene_sha256": sha256(scene_path), "source_sha256": sha256(source_path),
             "variant": variant, "seed": seed}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
