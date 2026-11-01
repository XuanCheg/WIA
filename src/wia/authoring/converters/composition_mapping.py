"""Plan conservative Cognitive Mapping exploration bundles without writing data.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider
tests/test_composition_mapping.py
``plan_groups`` is a pure policy; ``repair_definitions`` reads only a flagged
record's original source and returns copies for a new, unapproved derivative.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path


MODULE_PREFIX = 'wia.authoring.converters.cognitive_mapping_'
OPERATIONS = {
    'Connectivity': ('connectivity', 'region_exploration_arrivals'),
    'Long-Term Navigation': ('long_term_navigation', 'navigation_exploration_arrivals'),
    'Regional Boundary': ('regional_boundary', 'anchor_region_exploration'),
    'Traversable Passage': ('traversable_passage', 'passage_exploration_arrivals'),
}
CONSTRAINED = {'Long-Term Navigation', 'Traversable Passage'}
SPAWN_WORDS = ('spawn', 'initial position', '生成位置', '出生点')


def _feedback(record, feedback):
    entries = (feedback or {}).get('entries', feedback or {})
    if not isinstance(entries, dict):
        return {}
    for key in ('candidate_id', 'parent_candidate_id', 'cpu_candidate_id'):
        item = entries.get(record.get(key))
        if isinstance(item, dict):
            return item
    identity = record.get('source_identity')
    matches = [item for item in entries.values() if isinstance(item, dict)
               and identity and item.get('source') == identity]
    return max(matches, key=lambda item: str(item.get('updated_at', '')), default={})


def _bad_spawn(record, feedback):
    item = _feedback(record, feedback)
    notes = str(item.get('notes', '')).lower()
    return item.get('decision') == 'needs_fix' and any(word in notes for word in SPAWN_WORDS)


def _accepted_setup(record):
    # Only explicit in-memory evidence is considered; never read or infer a
    # native certificate from a path, screenshot, status or source success flag.
    for key in ('category_interaction_setup', 'interaction_setup', 'placement_evidence'):
        evidence = record.get(key)
        if isinstance(evidence, dict) and evidence.get('accepted') is True:
            return True
    return False


def _rank(record, feedback):
    item = _feedback(record, feedback)
    return (_bad_spawn(record, feedback), item.get('decision') != 'approved',
            not _accepted_setup(record), item.get('decision') == 'needs_fix', record['candidate_id'])


def _patches(definition):
    variants = definition.get('scene', {}).get('variants', {})
    return variants.get('adapted', variants.get('original', {})).get('patches', [])


def _unchanged_world(task, scene):
    for definition in (task, scene):
        for patch in _patches(definition):
            if patch.get('operation') == 'robot_pose':
                continue
            if (patch.get('operation') == 'prepare_category_interaction'
                    and patch.get('mode') == 'navigation'
                    and patch.get('relocation_policy') == 'preserve'
                    and not patch.get('group_bindings')
                    and not patch.get('manipulation_bindings')):
                continue
            return False
    return True


def _scene_name(record, task):
    if record.get('scene_model'):
        return record['scene_model']
    identity = record.get('source_identity') or task.get('source', {}).get('json', '')
    parts = str(identity).split('/')
    if len(parts) < 5:
        raise ValueError('Mapping records require a source identity or scene_model.')
    return parts[2]


def _compatibility_signature(task, scene):
    """Require explicit condition identity; a shared building is insufficient."""
    p = task['private']['evaluation']['parameters']
    contract = p.get('mapping_composition_contract') or {}
    fields = ('world_state_id', 'start_state_id', 'phase_id', 'conditions_id',
              'robot_configuration_id', 'controller_id')
    if any(not contract.get(key) for key in fields) or contract.get('compatibility_verified') is not True:
        return None
    # Even a shared external identity cannot silently override distinct patches,
    # source door assumptions, ordered obligations or reference frames.
    return json.dumps({'contract': {key: contract[key] for key in fields},
        'scene_configuration': {key: value for key, value in scene.items()
            if key not in ('id', 'source', 'bindings')},
        'observations': {key: task.get(key) for key in ('observations', 'observation_profile',
            'overview', 'initial_overview', 'agent_inputs', 'robot', 'controller')},
        'episode_inputs': {key: value for key, value in task.get('defaults', {}).items()
                           if key not in ('max_steps', 'wall_timeout_s')},
        'task_patches': _patches(task), 'scene_patches': _patches(scene),
        'floor_idx': p.get('floor_idx'), 'source_door_policy': {
            key: p.get('source_door_policy', {}).get(key)
            for key in ('source_policy', 'runtime_policy', 'source_trav_map_basename')},
        'seed': task.get('defaults', {}).get('seed', scene.get('seed'))}, sort_keys=True)


def plan_groups(records, tasks, scenes, *, feedback=None):
    """Compose only condition-matched exploration and preserve semantic outputs."""
    from .cognitive_mapping_visit_reports import REVISED_MAPPING_OPERATIONS
    owned = [row for row in records if row.get('status') == 'generated'
             and row.get('big_task') == 'Cognitive Mapping' and row.get('small_task') in OPERATIONS]
    by_id = {row['candidate_id']: row for row in owned}
    if len(by_id) != len(owned):
        raise ValueError('Duplicate active Mapping candidate_id.')
    groups = defaultdict(list)
    for row in owned:
        key = row['candidate_id']
        if key not in tasks or key not in scenes:
            raise ValueError('Missing task/scene definition for ' + key)
        task, scene = tasks[key], scenes[key]
        evaluation = task['private']['evaluation']
        suffix, _ = OPERATIONS[row['small_task']]
        recognized = (module_identity(evaluation.get('module')) == MODULE_PREFIX + suffix
            and evaluation.get('operation') == REVISED_MAPPING_OPERATIONS[row['small_task']]
            and _unchanged_world(task, scene))
        signature = _compatibility_signature(task, scene) if recognized else None
        identity = ((_scene_name(row, task), signature) if signature is not None
                    and row['small_task'] not in CONSTRAINED else ('isolated', key))
        groups[identity].append(key)
    compatible = []
    for identity, members in groups.items():
        compatible.append((identity, members))
    planned = []
    for identity, members in compatible:
        members = sorted(members)
        representative = min(members, key=lambda key: _rank(by_id[key], feedback))
        rooms, outputs, routes = set(), [], []
        for index, key in enumerate(members, 1):
            task = tasks[key]
            p = task['private']['evaluation']['parameters']
            capability = p.get('actual_capability', 'historical_mapping_operation')
            route = p.get('required_order') or p.get('route_answer_order')
            if route:
                routes.append({'child_id': key, 'route': list(route), 'revisits_preserved': True})
            elif capability != 'observed_anchor_region_relation':
                rooms.update(p.get('required_regions', []))
            outputs.append({'child_id': key, 'output_id': f'goal_{index}', 'capability': capability,
                'public_objective': task['public']['objective'],
                'public_success_criteria': task['public']['success_criteria']})
        public = {'objective': 'Explore this building once and provide the requested answers.\n'
            + '\n'.join(f"{entry['output_id']}: {entry['public_objective']}" for entry in outputs),
            'success_criteria': 'A verified room visit may satisfy compatible unordered requirements in several goals. '
                'Submit one final JSON object in the answer field: '
                + json.dumps({entry['output_id']: '<answer>' for entry in outputs}) + '. '
                'All answers may be submitted at the same observation boundary. '
                + ' '.join(f"{entry['output_id']}: {entry['public_success_criteria']}" for entry in outputs),
            'environment': 'One verified common world, initial state, phase and robot/controller configuration.',
            'constraints': ['Preserve all child reference frames, anchors, door conditions and trajectory constraints.']}
        isolated = identity[0] == 'isolated'
        planned.append({'representative_id': representative, 'member_ids': members,
            'composition_policy': 'task_edit2_mapping_v1',
            'world_compatibility_signature': None if isolated else identity[1],
            'label': f"{_scene_name(by_id[representative], tasks[representative])}: {len(members)} mapping outputs",
            'reason': 'Exact verified world/start/phase/condition match.' if not isolated else
                ('Independent route episode; shared episodes are not certified by world loading reuse.'
                 if by_id[representative]['small_task'] in CONSTRAINED else
                 'Independent trial: exact shared conditions are not certified.'),
            'public': public, 'shared_unordered_room_requirements': sorted(rooms),
            'semantic_outputs': outputs, 'ordered_route_obligations': routes,
            'completion_deduplication': 'shared_physical_visits_only_keep_semantic_outputs_and_route_steps',
            'review_goals': [entry['public_objective'] for entry in outputs],
            'requires_recapture': len(members) > 1 or any(_feedback(by_id[key], feedback).get('decision') == 'needs_fix' for key in members),
            'warnings': ['Native door and relation evidence remains necessary; prior layout approval does not transfer.']
                + (['Missing exact composition compatibility proof; no building-name pooling.'] if isolated else [])})
    return sorted(planned, key=lambda group: group['representative_id'])


def repair_definitions(record, task, scene, *, feedback=None):
    """Return a source-anchored pending derivative for explicit bad-spawn feedback.

    Historical/approved files are never written. Source XY is only a proposal;
    the unchanged mandatory native stance search still rejects unsafe poses.
    """
    result = {'task': deepcopy(task), 'scene': deepcopy(scene), 'changed': False,
              'change_summary': [], 'warnings': []}
    if (record.get('big_task'), record.get('small_task')) != ('Cognitive Mapping', 'Connectivity'):
        return result
    if not _bad_spawn(record, feedback):
        return result
    from wia.authoring.converters.cognitive_mapping_connectivity import source_initial_pose
    try:
        raw = json.loads(Path(record['original_source']).read_text())
        payload = raw.get('metadata_json', {})
        payload = json.loads(payload) if isinstance(payload, str) else payload
        proposal = source_initial_pose(payload)
    except (OSError, KeyError, ValueError, TypeError) as error:
        result['warnings'].append('Source-anchored spawn repair unavailable: ' + str(error))
        return result
    for definition in (result['task'], result['scene']):
        variants = definition['scene']['variants']
        adapted = deepcopy(variants.get('adapted', variants['original']))
        current = adapted.get('patches', [])
        updated = [deepcopy(proposal)] + [patch for patch in current if patch.get('operation') != 'robot_pose']
        result['changed'] |= updated != current
        adapted['patches'] = updated
        variants['adapted'] = adapted
    if result['changed']:
        result['change_summary'] = ['Request the saved source-room camera XY and horizontal heading for the existing R1 before native stance screening.']
        result['warnings'] = ['Source camera clearance does not establish chassis clearance.',
            'Mandatory native supported and collision-free stance checks remain unchanged; recapture and qualification are required.']
    return result
