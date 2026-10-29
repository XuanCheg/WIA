"""Check per-object native hash semantics and invalidation of prepared USD reuse.

Run: .venv/bin/python -m pytest -q tests/test_esi_asset_cache.py
"""
import hashlib
from pathlib import Path

from wia.runtime.sim.esi_asset_cache import instrument_assets


def native_stub():
    class USD:
        preparations = 0

        def __init__(self, path, expected=None):
            self.name = "object"
            self._usd_path, self._encrypted = str(path), False
            self._expected_file_hash = expected
            self._init_info = {"args": {}}

        def check_hash(self, path):
            digest = hashlib.md5(Path(path).read_bytes()).hexdigest()
            if self._expected_file_hash is None:
                self._expected_file_hash = digest
                self._init_info["args"]["expected_file_hash"] = digest

        def _prepare_to_load(self):
            type(self).preparations += 1
            self.check_hash(self._usd_path)
            return self._usd_path

    class Log:
        messages = []

        def warn(self, message):
            self.messages.append(message)

    return USD, Log()


def test_reused_asset_still_records_each_objects_hash_and_warns_on_mismatch(tmp_path):
    source = tmp_path / "asset.usd"
    source.write_bytes(b"same immutable asset")
    USD, log = native_stub()
    counters = instrument_assets(USD, log, tmp_path / "logs", cache=True)
    first, second = USD(source), USD(source)
    assert first._prepare_to_load() == second._prepare_to_load()
    expected = hashlib.md5(source.read_bytes()).hexdigest()
    assert first._expected_file_hash == second._expected_file_hash == expected
    assert second._init_info["args"]["expected_file_hash"] == expected
    third = USD(source, expected="old_saved_hash")
    third._prepare_to_load()
    assert "old_saved_hash" in log.messages[0] and expected in log.messages[0]
    assert USD.preparations == 1
    assert counters["bytes_hashed"] == source.stat().st_size


def test_changed_source_invalidates_preparation_and_hash_caches(tmp_path):
    source = tmp_path / "asset.usd"
    source.write_bytes(b"before")
    USD, log = native_stub()
    counters = instrument_assets(USD, log, tmp_path / "logs", cache=True)
    first = USD(source)
    first._prepare_to_load()
    source.write_bytes(b"after with different bytes")
    second = USD(source)
    second._prepare_to_load()
    assert first._expected_file_hash != second._expected_file_hash
    assert second._expected_file_hash == hashlib.md5(source.read_bytes()).hexdigest()
    assert USD.preparations == 2 and counters["prepare_cache_hits"] == 0
