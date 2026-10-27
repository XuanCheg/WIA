"""Parse shared final answers and score native semantic items separately.

Run: python -B -m pytest -q tests/test_task_edit2_mapping_answers.py
This module neither queries world truth nor infers answers from task success.
"""
from __future__ import annotations

from copy import deepcopy
import json


def parse_answer_sheet(text, addresses, *, required_addresses=None):
    """Return valid addressed strings plus Agent errors, preserving JSON duplicates."""
    errors, pairs = [], []
    text = str(text).strip()
    if text.upper().startswith('ANSWER:'):
        text = text.split(':', 1)[1].strip()
    if text.startswith('{'):
        try:
            pairs = json.loads(text, object_pairs_hook=lambda rows: rows)
            if not isinstance(pairs, list):
                raise ValueError('object_required')
        except (ValueError, TypeError):
            return {}, [{'reason': 'malformed_answer_sheet'}], True
        structured = True
    else:
        address, separator, value = text.partition('=')
        if not separator:
            return {}, [{'reason': 'expected_JSON_object_or_goal_N=<answer>'}], False
        pairs, structured = [(address.strip(), value.strip())], False
    seen, invalid, accepted = set(), set(), {}
    for address, value in pairs:
        if address not in addresses:
            errors.append({'question_id': address, 'reason': 'unknown_question'})
            continue
        if address in seen:
            errors.append({'question_id': address, 'reason': 'duplicate_question'})
            invalid.add(address)
        seen.add(address)
        if not isinstance(value, str) or not value.strip():
            errors.append({'question_id': address, 'reason': 'invalid_answer_value'})
            invalid.add(address)
        else:
            accepted[address] = value.strip()
    if structured:
        errors.extend({'question_id': address, 'reason': 'missing_question'}
                      for address in (addresses if required_addresses is None else required_addresses)
                      if address not in seen)
    return {key: value for key, value in accepted.items() if key not in invalid}, errors, structured


def mapping_questions(members):
    """Freeze semantic denominators and explicitly map exact semantic duplicates.

    Equality is deliberately conservative: native operation, every semantic
    parameter, and bindings must match. Only provenance and composition receipt
    identity are omitted. Similar questions remain independent.
    """
    questions, by_signature = [], {}
    for index, member in enumerate(members, 1):
        config = member['evaluation']
        p = deepcopy(config.get('parameters', {}))
        if not p.get('source_qa', {}).get('scored'):
            continue
        item_ids = ['source_relation_answer']
        if p.get('reachability_answer'):
            item_ids.append('global_reachability_answer')
        p.pop('source_path', None)
        p.pop('mapping_composition_contract', None)
        for item_id in item_ids:
            signature = json.dumps({'module': config['module'], 'operation': config['operation'],
                'parameters': p, 'bindings': member.get('bindings', {}), 'item_id': item_id}, sort_keys=True)
            source = {'member_id': member['id'], 'source_identity': member['source_identity'],
                      'native_item_id': item_id, 'public_id': f'goal_{index}'}
            if signature in by_signature:
                by_signature[signature]['sources'].append(source)
                continue
            question = {'id': f'question_{len(questions) + 1}',
                'subclass': member.get('small_task') or {
                    'connectivity': 'Connectivity', 'regional_boundary': 'Regional Boundary',
                    'traversable_passage': 'Traversable Passage'}.get(config['module'].rsplit('cognitive_mapping_', 1)[-1], 'Mapping'),
                'sources': [source], 'equivalence': 'exact_native_semantic_contract'}
            questions.append(question)
            by_signature[signature] = question
    return questions


def score_questions(questions, latest, blocked, responses, response_errors):
    """Use only current native answer items; unavailable truth keeps its denominator."""
    results, errors = [], deepcopy(response_errors)
    for question in questions:
        measurements = []
        for source in question['sources']:
            member_id = source['member_id']
            native = latest.get(member_id, {})
            item = next((row for row in (native.get('completion_items') or [])
                         if isinstance(row, dict) and row.get('id') == source['native_item_id']), None)
            valid = (member_id not in blocked and item is not None
                     and item.get('evaluation_validity', 'valid') in ('valid', 'complete')
                     and type(item.get('completed')) is bool)
            measurements.append((valid, bool(item and item.get('completed')),
                                 source['public_id'] in responses))
        available = all(row[0] for row in measurements)
        # A duplicate's independent native measurements must agree. A missing
        # Agent response is incorrect, even if an old child answer was retained.
        answered = [row[1] for row in measurements if row[2]]
        response_values = {' '.join(responses[source['public_id']].casefold().split())
                           for source in question['sources'] if source['public_id'] in responses}
        conflicting = len(response_values) > 1
        if conflicting:
            errors.append({'question_id': question['id'], 'reason': 'conflicting_duplicate_answers'})
        elif len(set(answered)) > 1:
            available = False
        correct = available and not conflicting and bool(answered) and all(answered)
        results.append({**deepcopy(question), 'correct': correct,
                        'evaluation_validity': 'valid' if available else 'unavailable'})

    def summary(rows):
        total = len(rows)
        correct = sum(row['correct'] for row in rows)
        valid = all(row['evaluation_validity'] == 'valid' for row in rows)
        return {'required_unique_questions': total, 'correct_answers': correct,
                'answer_accuracy_percent': 100. * correct / total if total and valid else None,
                'answer_evaluation_validity': 'complete' if valid else 'unavailable',
                'diagnostic_answer_lower_bound_percent': 100. * correct / total if total else None}

    by_subclass = {name: summary([row for row in results if row['subclass'] == name])
                   for name in dict.fromkeys(row['subclass'] for row in results)}
    return {**summary(results), 'semantic_questions': results, 'per_subclass': by_subclass,
            'agent_response_errors': errors}
