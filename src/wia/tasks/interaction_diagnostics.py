"""Publish bounded, read-only interaction setup diagnostics.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_task_edit2_interaction_diagnostics.py
Install with install_interaction_diagnostics(session, output, progress=None).
The optional progress callback accepts one stage string, only after publication.
Root supplies hook sites; this observer never performs simulation or qualification.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json
import math
import os
from pathlib import Path
import time
import uuid

from wia.authoring.io import write_json

MIN_WRITE_INTERVAL_S = 15.0
SCHEMA = 'interaction_setup_diagnostics_v1'


def _snapshot(value):
    """Copy supplied receipts into JSON without modifying their native buffers."""
    if is_dataclass(value) and not isinstance(value, type):
        return _snapshot(asdict(value))
    if isinstance(value, dict):
        return {str(key): _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {'unavailable': 'nonfinite_number', 'representation': str(value)}
    # NumPy/native tensor receipts are read and copied, never used as control inputs.
    if callable(getattr(value, 'detach', None)):
        value = value.detach().cpu()
    if callable(getattr(value, 'tolist', None)):
        return _snapshot(value.tolist())
    return {'unavailable': 'unsupported_evidence_type', 'type': type(value).__module__ + '.' + type(value).__qualname__}


def _source_identity(owner):
    source = getattr(getattr(owner, 'definition', None), 'source', None)
    return source.get('json') if isinstance(source, dict) else getattr(source, 'json_path', None)


def _provenance(session):
    task = getattr(session, 'loaded_task', None)
    scene = getattr(session, 'loaded_scene', None)
    adapter = getattr(getattr(session, 'scene_adapter', None), 'task', None)
    if scene is None and adapter is not task:
        scene = adapter
    owner = task if task is not None else scene
    return {'reset_id': getattr(session, 'native_reset_id', None), 'episode_id': getattr(session, 'episode_id', None),
        'task_file': str(task.path) if getattr(task, 'path', None) is not None else None,
        'task_sha256': getattr(task, 'file_sha256', None),
        'scene_file': str(scene.path) if getattr(scene, 'path', None) is not None else None,
        'scene_sha256': getattr(scene, 'file_sha256', None),
        'source_identity': _source_identity(owner), 'source_sha256': getattr(owner, 'source_sha256', None),
        'scene_source_sha256': getattr(scene, 'source_sha256', None), 'seed': getattr(owner, 'seed', None)}


def _native_time(session):
    try:
        value = float(session.pipeline.og.sim.current_time)
        if not math.isfinite(value):
            return None, 'unavailable_nonfinite_native_clock'
        return value, 'measured'
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None, 'unavailable_native_clock_api'


def _failure(stage, evidence):
    """Report explicit stage/check faults, never a duration-based inference."""
    normalized = stage.lower().replace('-', '_')
    tokens = set(normalized.split('_'))
    attempts = evidence.get('attempts')
    latest = attempts[-1] if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict) else {}
    checks = latest.get('checks', evidence.get('checks', {}))
    failed_checks = sorted(str(key) for key, value in checks.items() if value is False) if isinstance(checks, dict) else []
    explicit = evidence.get('failure_stage') or latest.get('failure_stage')
    if explicit:
        return str(explicit), 'explicit_failure_stage', failed_checks
    if tokens & {'failed', 'failure', 'error'}:
        return stage, 'explicit_observer_failure_stage', failed_checks
    if failed_checks:
        return stage, 'measured_false_checks', failed_checks
    return None, None, failed_checks


def _always_flush(stage, evidence):
    tokens = set(stage.lower().replace('-', '_').split('_'))
    return bool(({'source', 'measured'} <= tokens) or tokens & {'finished', 'failure', 'failed', 'error'}
                or evidence.get('failure_stage'))


class InteractionDiagnostics:
    """One observer's counters and bounded publications; no native state changes."""

    def __init__(self, session, output, progress):
        self.session, self.output, self.progress = session, Path(output).resolve(), progress
        self.started = time.monotonic()
        self.last_write = None
        self.callback_count = self.event_count = 0
        self.earliest_failed_stage = None
        self.max_observed_attempt_count = None
        self.observer_id = uuid.uuid4().hex
        self.last_progress_error = None

    def __call__(self, stage, evidence):
        if not isinstance(stage, str) or not stage.strip() or not isinstance(evidence, dict):
            raise ValueError('Interaction diagnostic stages and dictionary receipts must be explicit.')
        now = time.monotonic()
        self.callback_count += 1
        current_failed_stage, failure_source, failed_checks = _failure(stage, evidence)
        if current_failed_stage is not None and self.earliest_failed_stage is None:
            self.earliest_failed_stage = current_failed_stage
        attempts = evidence.get('attempts')
        reported_count = evidence.get('attempt_count')
        attempt_count = (reported_count if type(reported_count) is int and reported_count >= 0
                         else len(attempts) if isinstance(attempts, list) else None)
        if attempt_count is not None:
            self.max_observed_attempt_count = max(attempt_count, self.max_observed_attempt_count or 0)
        if (self.last_write is not None and now - self.last_write < MIN_WRITE_INTERVAL_S
                and not _always_flush(stage, evidence)):
            return {'written': False, 'stage': stage, 'callback_count': self.callback_count,
                'reason': 'wall_clock_publication_throttled'}
        native_time, native_status = _native_time(self.session)
        self.event_count += 1
        event = {'schema': SCHEMA, 'observer_id': self.observer_id, 'event_index': self.event_count,
            'callback_count': self.callback_count, 'stage': stage, 'elapsed_wall_s': now - self.started,
            'native_time_s': native_time, 'native_time_status': native_status,
            'provenance': _provenance(self.session), 'attempt_count': attempt_count,
            'max_observed_attempt_count': self.max_observed_attempt_count,
            'current_failed_stage': current_failed_stage, 'earliest_failed_stage': self.earliest_failed_stage,
            'failure_stage_source': failure_source, 'failed_checks': failed_checks,
            'qualification_status': 'not_evaluated_by_diagnostics',
            'latest_evidence_file': str(self.output / 'interaction_setup_latest.json')}
        snapshot = {**deepcopy(event), 'evidence': _snapshot(evidence)}
        write_json(self.output / 'interaction_setup_latest.json', snapshot)
        with (self.output / 'interaction_setup_events.jsonl').open('a') as handle:
            handle.write(json.dumps(event, separators=(',', ':'), allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        self.last_write = now
        if callable(self.progress):
            try:
                self.progress(stage)
            except Exception as error:
                # Progress display failure does not turn a native trial into failure.
                self.last_progress_error = f'{type(error).__name__}: {error}'
        return {'written': True, 'stage': stage, 'event_index': self.event_count}


def install_interaction_diagnostics(session, output, progress=None):
    """Install a private stage callback; root owns all setup hook locations."""
    observer = InteractionDiagnostics(session, output, progress)
    session._interaction_setup_observer = observer
    return observer
