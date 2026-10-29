"""Cache unchanged native USD preparation within one simulator process.

Use via src/wia/runtime/commands/diagnose_esi_rigid_scene.py --cache-assets in the GPU shell.
The installed SDK is unchanged. Hash recording/warnings and decrypted USD
content are preserved; changed source files invalidate both caches.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time


def asset_version(path):
    stat = Path(path).stat()
    return (str(Path(path).resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def instrument_assets(usd_class, logger, output, *, cache=False):
    """Instrument the inspected USDObject API, optionally memoizing unchanged files."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    events = output / "asset_events.jsonl"
    original_hash, original_prepare = usd_class.check_hash, usd_class._prepare_to_load
    hashes, prepared = {}, {}
    counters = {"hash_calls": 0, "hash_cache_hits": 0, "prepare_calls": 0, "prepare_cache_hits": 0, "bytes_hashed": 0}

    def event(kind, **values):
        with events.open("a") as stream:
            stream.write(json.dumps({"event": kind, **values}) + "\n")

    def checked(obj, path):
        version = asset_version(path)
        counters["hash_calls"] += 1
        started = time.monotonic()
        event("hash_begin", object=obj.name, path=str(path), bytes=version[3])
        if not cache:
            result = original_hash(obj, path)
            counters["bytes_hashed"] += version[3]
        else:
            file_hash = hashes.get(version)
            if file_hash is None:
                digest = hashlib.md5()
                with open(path, "rb") as stream:
                    for chunk in iter(lambda: stream.read(65536), b""):
                        digest.update(chunk)
                file_hash = digest.hexdigest()
                counters["bytes_hashed"] += version[3]
                if asset_version(path) == version:
                    hashes[version] = file_hash
            else:
                counters["hash_cache_hits"] += 1
            if obj._expected_file_hash is not None:
                if file_hash != obj._expected_file_hash:
                    logger.warn(f"Object {obj.name} was expected to have USD file hash {obj._expected_file_hash} but loaded with {file_hash}. The saved state might be incompatible.")
            else:
                obj._expected_file_hash = file_hash
                obj._init_info["args"]["expected_file_hash"] = file_hash
            result = None
        event("hash_end", object=obj.name, path=str(path), wall_elapsed_s=time.monotonic()-started)
        return result

    def prepare(obj):
        path = obj._usd_path.replace(".usd", ".encrypted.usd") if obj._encrypted else obj._usd_path
        version = asset_version(path)
        counters["prepare_calls"] += 1
        started = time.monotonic()
        key = (version, obj._encrypted)
        existing = prepared.get(key)
        if cache and existing is not None and Path(existing).is_file():
            checked(obj, path)
            result = existing
            counters["prepare_cache_hits"] += 1
        else:
            result = original_prepare(obj)
            if cache and asset_version(path) == version:
                prepared[key] = result
        event("prepare_end", object=obj.name, path=str(path), prepared_path=str(result), wall_elapsed_s=time.monotonic()-started)
        (output / "asset_cache_stats.json").write_text(json.dumps({"enabled": cache, **counters}, indent=2)+"\n")
        return result

    usd_class.check_hash, usd_class._prepare_to_load = checked, prepare
    return counters
