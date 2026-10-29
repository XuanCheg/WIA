"""Durable atomic scene-capture queue; no simulator imports.

Run src/wia/runtime/commands/submit_esi_capture.py --help to submit or inspect committed candidates.
Workers use src/wia/runtime/commands/run_esi_capture_worker.py with direct local or allocated GPUs.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import uuid

PROTOCOL = "esi_parallel_capture_v1"
TERMINAL = {"REVIEW_PENDING", "CACHED", "FAILED", "STALE", "CANCELLED"}
TRANSITIONS = {
    "CLAIMED": {"RUNNING", "CACHED", "FAILED", "STALE", "CANCELLED"},
    "RUNNING": {"CAPTURED", "FAILED", "STALE", "CANCELLED"},
    "CAPTURED": {"REVIEW_PENDING", "FAILED"},
}


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


def publish_json(path, value):
    """Create without overwriting another producer's published document."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    write_json(temporary, value)
    try:
        os.link(temporary, path)
        return True
    except FileExistsError:
        return False
    finally:
        temporary.unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_snapshot(manifest):
    root = Path(manifest["snapshot_dir"]).resolve()
    for relative, expected in manifest["input_hashes"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != expected:
            raise ValueError("Immutable capture input changed: " + relative)
    code = Path(manifest["code_root"])
    if not code.is_dir() and manifest.get('code_archive'):
        # Category snapshots keep durable archives in home and import caches in
        # /tmp. Old committed M1-M3 snapshots retain their existing paths.
        import shutil
        import tarfile
        archive_path = Path(manifest['code_archive'])
        if sha256(archive_path) != manifest['code_archive_sha256']:
            raise ValueError('Durable category code archive changed.')
        code.parent.mkdir(parents=True, exist_ok=True)
        temporary = code.parent / ('.rehydrate_' + uuid.uuid4().hex)
        temporary.mkdir()
        try:
            with tarfile.open(archive_path) as archive:
                for member in archive.getmembers():
                    if (not (temporary / member.name).resolve().is_relative_to(temporary.resolve())
                            or member.issym() or member.islnk()):
                        raise ValueError('Unsafe path in frozen category archive.')
                archive.extractall(temporary)
            try:
                (temporary / code.name).rename(code)
            except OSError:
                if not code.is_dir():
                    raise
        finally:
            shutil.rmtree(temporary)
    for relative, expected in manifest["implementation_hashes"].items():
        path = code / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("Committed rendering implementation changed: " + relative)


def complete_images(manifest, receipt):
    """An empty, partial, foreign or unreadable image set is never a cache hit."""
    if (receipt.get("capture_signature") != manifest["capture_signature"]
            or receipt.get("fingerprint") != manifest["fingerprint"]
            or receipt.get("capture_protocol") != PROTOCOL
            or receipt.get("code_commit") != manifest["code_commit"]
            or receipt.get("esi_commit") != manifest["esi_commit"]
            or receipt.get("variant") != manifest["variant"]
            or receipt.get("seed") != manifest["seed"]
            or not receipt.get("complete")):
        return False
    images = receipt.get("images", {})
    if not images or not set(manifest["expected_views"]) <= images.keys():
        return False
    try:
        from PIL import Image
        for value in images.values():
            with Image.open(value) as pixels:
                if pixels.format != "PNG" or min(pixels.size) < 64:
                    return False
                pixels.verify()
    except (OSError, ValueError, SyntaxError):
        return False
    return True


class ClaimLost(RuntimeError):
    pass


class CaptureQueue:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        for name in ("jobs", "queued", "claimed", "cache", "candidates", "workers", "gpu_locks", "feedback"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id):
        if not job_id.startswith("capture_") or any(c not in "0123456789abcdef" for c in job_id[8:]):
            raise ValueError("Invalid capture job identity.")
        return self.root / "jobs" / job_id

    def manifest(self, job_id):
        path = self.job_dir(job_id) / "input/manifest.json"
        accepted = read_json(self.job_dir(job_id) / "accepted.json", {})
        if accepted.get("manifest_sha256") != sha256(path):
            raise ValueError("Accepted capture manifest changed after submission.")
        return read_json(path)

    def status(self, job_id):
        return read_json(self.job_dir(job_id) / "status.json")

    def _save(self, job_id, record):
        record = {**record, "updated_at": time.time()}
        write_json(self.job_dir(job_id) / "status.json", record)
        write_json(self.job_dir(job_id) / "events" / f"{time.time_ns()}_{uuid.uuid4().hex}.json", record)
        return record

    def cached(self, manifest):
        index = read_json(self.root / "cache" / f"{manifest['capture_signature']}.json", {})
        receipt = read_json(index.get("receipt", self.root / "missing"), {})
        return receipt if complete_images(manifest, receipt) else None

    def enqueue(self, manifest):
        """Input/manifest.json is sealed before its queue token becomes visible."""
        job_id = manifest["job_id"]
        if self.manifest(job_id)["capture_signature"] != manifest["capture_signature"]:
            raise ValueError("Job snapshot identity does not match its queue token.")
        status = self.status(job_id)
        if status is None:
            value = {"job_id": job_id, "state": "QUEUED", "attempts": 0,
                     "created_at": time.time(), "updated_at": time.time()}
            publish_json(self.job_dir(job_id) / "status.json", value)
            status = self.status(job_id)
        if status["state"] == "QUEUED":
            # Reconcile a producer crash between writing status and the token.
            if not (self.root / "claimed" / f"{job_id}.json").exists():
                gate = read_json(self.root / 'capture_sample_gate.json', {})
                held = (gate.get('state') == 'ONE_PER_CATEGORY_AWAITING_REVIEW'
                        and manifest.get('candidate_id') not in gate.get('selected_candidate_ids', []))
                target = self.root / ('held_for_category_review' if held else 'queued')
                target.mkdir(exist_ok=True)
                publish_json(target / f"{job_id}.json", {"job_id": job_id})
        receipt = self.cached(manifest)
        if not receipt and status["state"] in {"REVIEW_PENDING", "CACHED"}:
            status = self._save(job_id, {**status, "state": "FAILED", "retryable": True,
                "failure_stage": "cache_validation", "error": "Previously completed images are missing or incomplete."})
            if status["attempts"] < manifest["max_attempts"]:
                self.retry(job_id)
                status = self.status(job_id)
        return {**status, "submission_state": "CACHED" if receipt else status["state"],
                "receipt": receipt}

    def claim(self, worker, *, lease_s=60, job_id=None, candidate_prefix=None):
        runtime_policy = read_json(self.root / 'capture_runtime_policy.json', {})
        allowed_runtime = runtime_policy.get('allowed_code_commits')
        sample_gate = read_json(self.root / 'capture_sample_gate.json', {})
        pause = read_json(self.root / 'capture_pause.json', {})
        if pause.get('state') == 'PAUSED_BY_USER':
            pilot = pause.get('authorized_pilot', {})
            if not (job_id and pilot.get('state') == 'RUNNING'
                    and worker.get('id') == pilot.get('worker_id')
                    and worker.get('uuid') == pilot.get('gpu_uuid')
                    and job_id in pilot.get('job_ids', ())):
                return None
        paths = ([self.root / "queued" / f"{self.job_dir(job_id).name}.json"] if job_id else
                 (self.root / "queued").glob("*.json"))
        affinity = worker.get('warm_affinity')
        def order(path):
            token = read_json(self.root / 'priorities' / path.name, {})
            return (-token.get('priority', 0), token.get('warm_affinity') != affinity if affinity else False, path.name)
        for path in sorted(paths, key=order):
            manifest = self.manifest(path.stem) if candidate_prefix or allowed_runtime else {}
            if allowed_runtime and manifest.get('code_commit') not in allowed_runtime:
                continue  # Historical jobs remain intact; only the current runtime may render.
            if candidate_prefix and not manifest.get('candidate_id', '').startswith(candidate_prefix):
                continue
            if (sample_gate.get('state') == 'ONE_PER_CATEGORY_AWAITING_REVIEW'
                    and self.manifest(path.stem).get('candidate_id') not in sample_gate.get('selected_candidate_ids', [])):
                continue
            destination = self.root / "claimed" / path.name
            try:
                # Unlike rename(), link() cannot overwrite an existing owner.
                # A crash before unlink leaves a harmless duplicate queue token.
                os.link(path, destination)
                path.unlink()
            except (FileNotFoundError, FileExistsError):
                continue
            job_id = path.stem
            status = self.status(job_id)
            if status["state"] != "QUEUED":
                destination.unlink(missing_ok=True)
                continue
            claim = {"job_id": job_id, "token": uuid.uuid4().hex, "worker": worker,
                     "hostname": socket.gethostname(), "supervisor_pid": os.getpid(),
                     "claimed_at": time.time(), "lease_until": time.time() + lease_s}
            write_json(destination, claim)
            self._save(job_id, {**status, "state": "CLAIMED", "claim": claim,
                                "attempts": status["attempts"] + 1})
            return claim
        return None

    def prioritize(self, job_id, priority=0):
        """Adjust queue order only; the immutable capture manifest remains sealed."""
        path = self.root / 'priorities' / (self.job_dir(job_id).name + '.json')
        token = read_json(path, {})
        manifest = self.manifest(job_id)
        write_json(path, {'job_id': job_id, 'priority': max(priority, token.get('priority', 0)),
            'warm_affinity': [manifest.get('scene_model'), manifest.get('room')]})

    def _owned(self, claim):
        path = self.root / "claimed" / f"{claim['job_id']}.json"
        actual = read_json(path, {})
        if actual.get("token") != claim["token"]:
            raise ClaimLost("Capture claim is no longer owned by this worker.")
        return path, actual

    def heartbeat(self, claim, *, lease_s=60, **details):
        path, actual = self._owned(claim)
        actual.update(details, lease_until=time.time() + lease_s)
        write_json(path, actual)
        return actual

    def transition(self, claim, state, **details):
        self._owned(claim)
        record = self.status(claim["job_id"])
        if state not in TRANSITIONS.get(record["state"], set()):
            raise ValueError(f"Invalid capture transition: {record['state']} -> {state}")
        return self._save(claim["job_id"], {**record, **details, "state": state})

    def release(self, claim):
        path, _ = self._owned(claim)
        if self.status(claim["job_id"])["state"] not in TERMINAL:
            raise ValueError("A running capture claim cannot be released.")
        path.unlink()

    def finish(self, claim, receipt_path):
        manifest = self.manifest(claim["job_id"])
        receipt_path = Path(receipt_path).resolve()
        receipt = read_json(receipt_path, {})
        if not receipt_path.is_relative_to(self.job_dir(claim["job_id"]).resolve()):
            raise ValueError("Completion receipt must belong to this capture job.")
        verify_snapshot(manifest)
        if not complete_images(manifest, receipt):
            raise ValueError("Capture is incomplete or has incorrect provenance.")
        self.transition(claim, "CAPTURED", receipt=str(receipt_path), **{
            k: receipt.get(k) for k in ("duration_s", "timings", "gpu", "exit_code")},
            complete=True, retryable=False, error=None, failure_stage=None,
            native_log=str(receipt_path.parent / "native.log"), partial_images={})
        write_json(self.root / "cache" / f"{manifest['capture_signature']}.json",
                   {"job_id": claim["job_id"], "receipt": str(receipt_path)})
        self.transition(claim, "REVIEW_PENDING")
        self.release(claim)

    def retry(self, job_id):
        record, manifest = self.status(job_id), self.manifest(job_id)
        if record["state"] != "FAILED" or not record.get("retryable"):
            raise ValueError("Only explicitly retryable failed captures can be retried.")
        if record["attempts"] >= manifest["max_attempts"]:
            raise ValueError("Capture retry budget is exhausted.")
        if (self.root / "claimed" / f"{job_id}.json").exists():
            raise ValueError("Worker has not finished releasing its owned process.")
        self._save(job_id, {**record, "state": "QUEUED"})
        publish_json(self.root / "queued" / f"{job_id}.json", {"job_id": job_id})

    def cancel(self, job_id):
        record = self.status(job_id)
        if record["state"] in TERMINAL:
            return record
        write_json(self.job_dir(job_id) / "cancel.json", {"requested_at": time.time()})
        token = self.root / "queued" / f"{job_id}.json"
        held = self.root / 'cancelled_tokens' / token.name
        if not token.exists() and not held.exists():
            return record  # The current owner observes cancel.json before releasing its claim.
        held.parent.mkdir(exist_ok=True)
        try:
            token.rename(held)  # Queue tokens stay on their original filesystem.
        except FileNotFoundError:
            if not held.exists():
                return record  # The owner terminates its child before cancellation.
        write_json(self.job_dir(job_id) / 'cancelled_token.json', read_json(held))
        held.unlink()
        return self._save(job_id, {**record, "state": "CANCELLED"})

    def recover(self, confirm_stopped, *, now=None, orphan_grace_s=120):
        """Recover only after proving the old supervisor AND child have stopped.

        confirm_stopped must verify process identity/exit locally, or finished PBS
        allocation state remotely. An expired heartbeat alone is not proof.
        """
        recovered = []
        now = time.time() if now is None else now
        for path in (self.root / "claimed").glob("*.json"):
            claim = read_json(path, {})
            if claim.get("lease_until", path.stat().st_mtime + orphan_grace_s) >= now:
                continue
            if not confirm_stopped(claim):
                continue
            job_id = path.stem
            record = self.status(job_id)
            path.unlink()
            if record["state"] in TERMINAL:
                recovered.append(job_id)
                continue
            # Recover a crash after publishing a complete receipt as a cache hit.
            try:
                manifest = self.manifest(job_id)
            except (ValueError, OSError) as error:
                self._save(job_id, {**record, "state": "STALE", "failure_stage": "snapshot_validation",
                                   "error": str(error), "retryable": False})
                recovered.append(job_id)
                continue
            cached = self.cached(manifest)
            if not cached and record["state"] == "CAPTURED":
                candidate = read_json(record.get("receipt", self.root / "missing"), {})
                if complete_images(manifest, candidate):
                    write_json(self.root / "cache" / f"{manifest['capture_signature']}.json",
                               {"job_id": job_id, "receipt": record["receipt"]})
                    cached = candidate
            if cached:
                index = read_json(self.root / "cache" / f"{cached['capture_signature']}.json")
                self._save(job_id, {**record, "state": "CACHED", "receipt": str(
                    index["receipt"]), "recovered": True})
            else:
                self._save(job_id, {**record, "state": "FAILED", "retryable": True,
                    "failure_stage": "worker_recovery", "error": "Expired owner and child confirmed stopped."})
                if record["attempts"] < self.manifest(job_id)["max_attempts"]:
                    self.retry(job_id)
            recovered.append(job_id)
        return recovered

    def jobs(self):
        return [read_json(path) for path in sorted((self.root / "jobs").glob("*/status.json"))]

    def summary(self):
        jobs = self.jobs()
        counts = Counter(job["state"] for job in jobs)
        return {"root": str(self.root), "counts": dict(counts), "jobs": jobs,
                "pause": read_json(self.root / 'capture_pause.json', {}),
                "workers": [read_json(p) for p in sorted((self.root / "workers").glob("*.json"))]}
