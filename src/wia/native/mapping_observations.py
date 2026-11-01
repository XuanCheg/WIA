"""Deterministic mapping declaration evidence. Run: python -m pytest tests/test_mapping_declarations.py.

Collect private native visibility at capture, then match public labels against
runner-confirmed delivered cameras. No visual model or semantic-audit flag is used.
"""
from copy import deepcopy

from wia.runtime.sim.esi_visual_evidence import collect_native_visual_evidence


def normalized_label(text):
    return ' '.join(text.casefold().replace('_', ' ').split())


class MappingObservationEvidence:
    def __init__(self, session, config):
        self.session, self.config = session, config
        self.capture = None
        self.receipts = []
        self.claims = []

    def capture_observation(self, observation):
        p = self.config['parameters']
        try:
            evidence = collect_native_visual_evidence(self.session, observation,
                self.config['bindings'], p)
        except (ValueError, RuntimeError, AttributeError, KeyError, TypeError) as error:
            evidence = {'measurements': {}, 'measurement_errors': {'capture': str(error)}}
        self.capture = {'episode_id': getattr(self.session, 'episode_id', None),
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'evidence': evidence}

    def begin_episode(self):
        self.receipts, self.claims = [], []

    def report(self, declarations, delivered_views, observation):
        capture = self.capture
        if (capture is None or capture['episode_id'] != getattr(self.session, 'episode_id', None)
                or capture['snapshot_id'] != observation.snapshot_id
                or capture['action_step'] != observation.action_step):
            raise ValueError('No retained native evidence for this delivered snapshot.')
        labels = self.config['parameters']['public_anchor_labels']
        for declaration in declarations:
            matches = [key for key, label in labels.items()
                       if normalized_label(label) == normalized_label(declaration['label'])]
            claim = {**deepcopy(declaration), 'snapshot_id': observation.snapshot_id,
                     'action_step': observation.action_step, 'accepted': False}
            if len(matches) == 1 and declaration['camera'] in delivered_views:
                key = matches[0]
                views = capture['evidence'].get('measurements', {}).get(key, {}).get('views', {})
                rows = [{**row, 'binding': key} for camera, row in views.items()
                        if camera == declaration['camera']]
                for row in rows:
                    if (row.get('measurement_available') is True
                            and row.get('identity_synchronized') is True
                            and row.get('native_rgb_matches_delivered') is True
                            and row.get('visible_pixels', 0) > 0):
                        receipt = {**deepcopy(row), 'episode_id': capture['episode_id'],
                            'public_label': labels[key], 'declaration': deepcopy(declaration),
                            'method': 'public_label_declaration_and_same_capture_native_visibility',
                            'recognition_inferred_from_mask': False}
                        self.receipts.append(receipt)
                        claim['accepted'] = True
                        break
            self.claims.append(claim)

    def available(self):
        if self.capture is None:
            return False
        # Empty masks with a valid capture are measured absence, not missing data.
        errors = self.capture['evidence'].get('measurement_errors', {})
        if any(key in errors for key in ('native_capture_context', 'audit_context', 'source', 'retained_capture')):
            return False
        cameras = self.capture['evidence'].get('cameras', {})
        return any(c.get('native_rgb_matches_delivered') and c.get('retained_capture_verified')
                   for c in cameras.values())
