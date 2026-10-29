"""Store WIA design/layout feedback against an explicit published snapshot.

Run: python -B src/wia/runtime/commands/review_wia.py --snapshot SNAPSHOT --output-dir REVIEW
CPU checks: python -B -m pytest -q -p no:cacheprovider tests/test_wia_review.py
Only selected variants register declared PNGs; no simulator or directory scan is used.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import threading

from wia.runtime.scene_review import DECISIONS, ReviewConflict, now, read_json, write_json


CATALOGUE_FIELDS = (
    'id', 'document_alias', 'primary_family', 'primary_subclass', 'title',
    'implementation_state', 'status', 'summary',
)
DEFINITION_FIELDS = (
    'task_file', 'scene_file', 'task_version', 'scene_version', 'version',
    'task_sha256', 'scene_sha256', 'task_identity', 'scene_identity',
    'objective', 'success_criteria', 'constraints', 'environment',
)
REVIEW_SCOPE = 'design_layout_only'


def _fingerprint(row, variant):
    """Prefer publisher identity; exclude run status, evidence, images and notes."""
    supplied = variant.get('fingerprint') or variant.get('review_fingerprint')
    if isinstance(supplied, str) and supplied:
        return supplied
    identity = {'row_id': row['id'], 'variant_id': variant['id'],
                'row_task_version': row.get('task_version'),
                'definition': {key: variant.get(key) for key in DEFINITION_FIELDS}}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


class WiaReview:
    """Read one snapshot and persist independent per-row, per-variant annotations."""

    def __init__(self, snapshot, output_dir):
        self.snapshot_file = Path(snapshot).resolve()
        self.output = Path(output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.review_file = self.output / 'reviews.json'
        self.lock = threading.RLock()
        self._stamp = None
        self._snapshot = None
        self._rows = {}
        self._images = {}
        self._image_bindings = {}
        with self.lock:
            self._refresh()

    def _refresh(self):
        stat = self.snapshot_file.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp == self._stamp:
            return
        snapshot = read_json(self.snapshot_file)
        if (not isinstance(snapshot, dict) or snapshot.get('schema') != 'wia_review_snapshot_v1'
                or not isinstance(snapshot.get('rows'), list)
                or not isinstance(snapshot.get('summary'), dict)):
            raise ValueError('Provide a wia_review_snapshot_v1 snapshot with rows and summary.')
        rows = {}
        for row in snapshot['rows']:
            if (not isinstance(row, dict) or not isinstance(row.get('id'), str)
                    or re.fullmatch(r'[A-Za-z0-9_.-]+', row['id']) is None
                    or not isinstance(row.get('variants'), list)):
                raise ValueError('Snapshot rows require unique URL-safe IDs and a variants list.')
            if row['id'] in rows:
                raise ValueError('Duplicate WIA row ID: ' + row['id'])
            variants = row['variants']
            if any(not isinstance(item, dict) or not isinstance(item.get('id'), str)
                   or not item['id'] for item in variants):
                raise ValueError('Each WIA variant requires a nonempty string ID.')
            if len({item['id'] for item in variants}) != len(variants):
                raise ValueError('Variant IDs must be unique within a WIA row.')
            rows[row['id']] = row
        retained_images, retained_bindings = {}, {}
        for key, binding in self._image_bindings.items():
            row = rows.get(key[0])
            variant = next((item for item in row['variants'] if item['id'] == binding['variant_id']), None) if row else None
            if variant is None or _fingerprint(row, variant) != binding['fingerprint']:
                continue
            declarations = [item for item in variant.get('images', [])
                            if isinstance(item, dict) and item.get('id') == binding['image_id']]
            if len(declarations) != 1 or declarations[0].get('path') != binding['declared_path']:
                continue
            # Recheck only images previously selected, never the entire image catalogue.
            path = self._image_path(binding['declared_path'])
            if path is not None and path == self._images.get(key):
                retained_images[key], retained_bindings[key] = path, binding
        self._snapshot, self._rows, self._stamp = snapshot, rows, stamp
        self._images, self._image_bindings = retained_images, retained_bindings

    @property
    def images(self):
        """The existing HTTP server checks this mapping for every image request."""
        with self.lock:
            self._refresh()
            return dict(self._images)

    def _reviews(self):
        value = read_json(self.review_file, {'schema': 'wia_reviews_v1', 'entries': {}, 'history': []})
        if value.get('schema') != 'wia_reviews_v1' or not isinstance(value.get('entries'), dict):
            raise ValueError('Unsupported WIA review record format.')
        return value

    def _selection(self, key, variant_id=None):
        row = self._rows[key]
        variants = row['variants']
        if variant_id is None:
            return row, variants[0] if variants else None
        variant = next((item for item in variants if item['id'] == variant_id), None)
        if variant is None:
            raise KeyError(variant_id)
        return row, variant

    @staticmethod
    def _review_state(row, variant, records):
        current = _fingerprint(row, variant) if variant is not None else None
        saved = records['entries'].get(row['id'], {}).get(variant['id']) if variant is not None else None
        review = deepcopy(saved) if saved else {
            'decision': 'unreviewed', 'notes': '', 'revision': 0,
            'review_scope': REVIEW_SCOPE, 'native_certification': False,
        }
        stale = bool(saved and saved.get('fingerprint') != current)
        return {'fingerprint': current, 'review': review, 'review_stale': stale,
                'decision': 'unreviewed' if stale else review['decision']}

    def _variant_metadata(self, row, variant, records):
        """Describe declared evidence without reading images or certifying execution."""
        state = self._review_state(row, variant, records)
        definitions = variant.get('definition_status') or {}
        return {'id': variant['id'], 'label': variant.get('label', variant['id']),
                'status': variant.get('status'), 'fingerprint': state['fingerprint'],
                'image_count': len(variant.get('images') or []),
                'decision': state['decision'], 'review_stale': state['review_stale'],
                'has_definition': all(definitions.get(key, {}).get('available') is True
                                      for key in ('task', 'scene')),
                'reference_verified': variant.get('status') == 'reference_verified'}

    def _metadata(self, row, records):
        variants = row['variants']
        metadata = [self._variant_metadata(row, variant, records) for variant in variants]
        awaiting = [item for item in metadata if item['image_count'] and item['decision'] == 'unreviewed']
        state = self._review_state(row, variants[0] if variants else None, records)
        return {**{field: deepcopy(row.get(field)) for field in CATALOGUE_FIELDS}, **state,
                'proposal_membership': row.get('proposal_membership') is not False,
                'has_definition': any(item['has_definition'] for item in metadata),
                'image_count': sum(item['image_count'] for item in metadata),
                'image_variant_count': sum(bool(item['image_count']) for item in metadata),
                'default_image_count': metadata[0]['image_count'] if metadata else 0,
                'unreviewed_image_variant_count': len(awaiting),
                'next_review_variant_id': awaiting[0]['id'] if awaiting else None,
                'next_needs_fix_variant_id': next((item['id'] for item in metadata
                                                   if item['decision'] == 'needs_fix'), None),
                'reference_variant_id': next((item['id'] for item in metadata
                                              if item['reference_verified']), None),
                'reference_verified_variant_count': sum(item['reference_verified'] for item in metadata),
                'capture_pending_variant_count': sum(item['status'] == 'capture_pending' for item in metadata),
                'needs_fix_variant_count': sum(item['decision'] == 'needs_fix' for item in metadata),
                'variant_count': len(variants),
                'default_variant_id': variants[0]['id'] if variants else None,
                'variant_review_counts': dict(Counter(item['decision'] for item in metadata)),
                'review_scope': REVIEW_SCOPE, 'native_certification': False}

    def catalogue(self, *, filters=None, page=None, page_size=100):
        with self.lock:
            self._refresh()
            records = self._reviews()
            rows = [self._metadata(row, records) for row in self._rows.values()]
            filters = filters or {}
            matched = rows
            if filters.get('scope') in ('original', 'extra'):
                original = filters['scope'] == 'original'
                matched = [row for row in matched if row['proposal_membership'] == original]
            readiness = {
                'awaiting_review': lambda row: row['unreviewed_image_variant_count'] > 0,
                'with_images': lambda row: row['image_count'] > 0,
                'without_images': lambda row: row['image_count'] == 0,
                'needs_fix': lambda row: row['needs_fix_variant_count'] > 0,
                'reference_verified': lambda row: row['reference_verified_variant_count'] > 0,
            }.get(filters.get('readiness'))
            if readiness:
                matched = [row for row in matched if readiness(row)]
            for query_field, row_field in (('family', 'primary_family'), ('status', 'status'), ('review', 'decision')):
                if filters.get(query_field) and filters[query_field] != 'all':
                    matched = [row for row in matched if row.get(row_field) == filters[query_field]]
            if filters.get('q'):
                query = str(filters['q']).strip().casefold()
                matched = [row for row in matched if query in ' '.join(
                    [str(row.get(field) or '') for field in CATALOGUE_FIELDS]
                    + [str(variant.get('objective') or '')
                       for variant in self._rows[row['id']]['variants']]).casefold()]
            total = len(matched)
            page = max(0, int(page or 0))
            # The reused HTTP server supplies its legacy 48 when the query omits page_size.
            if filters is not None and 'page_size' not in filters and page_size == 48:
                page_size = 100
            page_size = min(200, max(1, int(page_size)))
            summary = deepcopy(self._snapshot['summary'])
            originals = [row for row in rows if row['proposal_membership']]
            summary['proposal_progress'] = {
                'total': len(originals),
                'definitions_available': sum(row['has_definition'] for row in originals),
                'with_images': sum(row['image_count'] > 0 for row in originals),
                'without_images': sum(row['image_count'] == 0 for row in originals),
                'awaiting_image_review': sum(row['unreviewed_image_variant_count'] > 0 for row in originals),
                'with_layout_approval': sum(row['variant_review_counts'].get('approved', 0) > 0 for row in originals),
                'reference_verified': sum(row['reference_verified_variant_count'] > 0 for row in originals),
            }
            summary['review_counts'] = dict(Counter(row['decision'] for row in rows))
            summary['variant_review_counts'] = dict(Counter(
                self._review_state(row, variant, records)['decision']
                for row in self._rows.values() for variant in row['variants']))
            return {'scenes': matched[page * page_size:(page + 1) * page_size], 'total': total,
                    'page': page, 'page_size': page_size, 'summary': summary,
                    'title': self._snapshot.get('title', 'WIA bench design review'),
                    'generated_at': self._snapshot.get('generated_at'),
                    'review_scope': REVIEW_SCOPE, 'native_certification': False}

    def _image_path(self, value):
        """Resolve only the selected declaration; never expose arbitrary route paths."""
        if not isinstance(value, str) or not value:
            return None
        raw = Path(value)
        candidate = Path(os.path.abspath(raw if raw.is_absolute() else self.snapshot_file.parent / raw))
        if candidate.is_relative_to('/home') or candidate.suffix.lower() != '.png':
            return None
        path = candidate.resolve()
        if path.is_relative_to('/home') or path.suffix.lower() != '.png' or not path.is_file():
            return None
        return path

    def _selected_images(self, row, variant, fingerprint):
        images, warnings, seen = [], [], set()
        for declared in variant.get('images', []):
            if not isinstance(declared, dict) or not isinstance(declared.get('id'), str) or not declared['id']:
                raise ValueError('Declared WIA images require nonempty string IDs.')
            if declared['id'] in seen:
                raise ValueError('Declared image IDs must be unique within the selected variant.')
            seen.add(declared['id'])
            path = self._image_path(declared.get('path'))
            if path is None:
                warnings.append({'id': declared['id'], 'reason': 'declared_png_unavailable_or_disallowed'})
                continue
            identity = json.dumps([variant['id'], declared['id'], fingerprint, declared['path']], ensure_ascii=True)
            token = hashlib.sha256(identity.encode()).hexdigest()[:32]
            key = (row['id'], token)
            self._images[key] = path
            self._image_bindings[key] = {'variant_id': variant['id'], 'image_id': declared['id'],
                                       'fingerprint': fingerprint, 'declared_path': declared['path']}
            images.append({'id': declared['id'], 'label': declared.get('label', declared['id']),
                           'role': declared.get('role', 'review'),
                           'url': f"images/{row['id']}/{token}?v={fingerprint[:16]}"})
        return images, warnings

    def scene_detail(self, key, *, variant_id=None):
        with self.lock:
            self._refresh()
            row, variant = self._selection(key, variant_id)
            records = self._reviews()
            state = self._review_state(row, variant, records)
            images, warnings = self._selected_images(row, variant, state['fingerprint']) if variant else ([], [])
            selected = deepcopy(variant) if variant is not None else None
            if selected is not None:
                selected.update(fingerprint=state['fingerprint'], images=images)
            detail = {**deepcopy(row), **state, 'selected_variant': selected,
                'selected_variant_id': variant['id'] if variant else None, 'images': images,
                'image_warnings': warnings, 'review_scope': REVIEW_SCOPE, 'native_certification': False}
            detail['variants'] = [self._variant_metadata(row, item, records) for item in row['variants']]
            detail['review_history'] = [deepcopy(item['review']) for item in records.get('history', [])
                if item.get('id') == key and variant is not None and item.get('variant_id') == variant['id']]
            return detail

    def save_review(self, key, payload):
        with self.lock:
            self._refresh()
            if not isinstance(payload, dict) or not isinstance(payload.get('variant_id'), str):
                raise ValueError('Provide a declared variant_id for WIA feedback.')
            row, variant = self._selection(key, payload['variant_id'])
            value = self._reviews()
            state = self._review_state(row, variant, value)
            decision = payload.get('decision', state['decision'])
            notes = payload.get('notes', state['review'].get('notes', ''))
            if decision not in DECISIONS or not isinstance(notes, str) or len(notes) > 10000:
                raise ValueError('Provide a valid decision and notes of at most 10,000 characters.')
            revision = payload.get('revision')
            if type(revision) is not int or revision < 0:
                raise ValueError('Provide the current nonnegative integer review revision.')
            if payload.get('fingerprint') != state['fingerprint']:
                raise ReviewConflict('The selected variant definition changed. Refresh before saving.')
            if revision != state['review'].get('revision', 0):
                raise ReviewConflict('This variant review changed in another tab. Refresh before saving.')
            record = {'id': key, 'variant_id': variant['id'], 'decision': decision, 'notes': notes,
                'revision': revision + 1, 'fingerprint': state['fingerprint'],
                'task_file': variant.get('task_file'), 'scene_file': variant.get('scene_file'),
                'updated_at': now(), 'review_scope': REVIEW_SCOPE, 'native_certification': False}
            row_records = value['entries'].setdefault(key, {})
            if variant['id'] in row_records:
                value.setdefault('history', []).append({'id': key, 'variant_id': variant['id'],
                                                       'review': row_records[variant['id']]})
            row_records[variant['id']] = record
            write_json(self.review_file, value)
            return deepcopy(record)

    def export_csv(self):
        with self.lock:
            self._refresh()
            records = self._reviews()
            stream = io.StringIO()
            fields = ('id', 'document_alias', 'primary_family', 'primary_subclass', 'title',
                'variant_id', 'variant_label', 'task_file', 'scene_file', 'status', 'fingerprint',
                'decision', 'review_stale', 'saved_decision', 'saved_fingerprint', 'notes', 'revision',
                'updated_at', 'review_scope', 'native_certification')
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in self._rows.values():
                for variant in row['variants'] or [None]:
                    state = self._review_state(row, variant, records)
                    review = state['review']
                    result = {field: row.get(field, '') for field in fields[:5]}
                    result.update(variant_id=variant['id'] if variant else '',
                        variant_label=variant.get('label', '') if variant else '',
                        task_file=variant.get('task_file', '') if variant else '',
                        scene_file=variant.get('scene_file', '') if variant else '',
                        status=variant.get('status', row.get('status')) if variant else row.get('status'),
                        fingerprint=state['fingerprint'], decision=state['decision'], review_stale=state['review_stale'],
                        saved_decision=review['decision'], saved_fingerprint=review.get('fingerprint', ''),
                        notes=review.get('notes', ''), revision=review.get('revision', 0),
                        updated_at=review.get('updated_at', ''), review_scope=REVIEW_SCOPE, native_certification=False)
                    writer.writerow(result)
            return stream.getvalue()

    def category_report(self, big_task, small_task):
        with self.lock:
            self._refresh()
        return '# WIA design/layout review\n\nFeedback here is design/layout feedback only, not native certification.\n'

    def diagnostic_log(self, key, index):
        with self.lock:
            self._refresh()
            self._selection(key)
        return 'Diagnostic files are not served here. Declared evidence paths appear in the selected variant.\n'
