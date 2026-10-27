"""Plan conservative perception goal bundles without changing task definitions.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q tests/test_composition_perception.py
The caller supplies current task/scene dictionaries and Review feedback. This module
only reads source JSON; it never initializes a simulator, writes, or publishes.
"""
from __future__ import annotations

import json
from pathlib import Path


CATEGORIES = frozenset({
    'Category Ambiguity', 'Counting w Occlusion', 'Illumination Variability',
    'Merged Observation', 'Spatial Segmentation', 'Structural Enclosure',
    'Material Transparency', 'Partial Occlusion', 'View Hallucination',
})

# Audited source pairs, not general permission to pool a scene or room. Recheck
# construction and initialization below, so later repairs cannot inherit a merge.
_PAIR_LOCATIONS = {
    'Category Ambiguity': [('house_double_floor_lower', 'kitchen_0', '001', '004')],
    'Counting w Occlusion': [
        ('Beechwood_0_int', 'living_room_1', '001', '004'),
        ('Ihlen_0_int', 'living_room_0', '000', '004'),
        ('Ihlen_1_int', 'living_room_0', '002', '003'),
        ('house_single_floor', 'living_room_1', '000', '003'),
    ],
    'Illumination Variability': [
        ('Benevolence_2_int', 'bedroom_1', '000', '001'),
        ('Ihlen_1_int', 'bedroom_2', '002', '004'),
        ('hotel_suite_small', 'bedroom_0', '000', '002'),
        ('house_double_floor_upper', 'bedroom_2', '000', '002'),
        ('house_single_floor', 'sauna_0', '003', '004'),
        ('office_cubicles_left', 'meeting_room_0', '000', '003'),
    ],
    'Spatial Segmentation': [
        ('Beechwood_0_int', 'living_room_1', '000', '001'),
        ('Ihlen_0_int', 'living_room_0', '001', '004'),
        ('Ihlen_1_int', 'living_room_0', '002', '003'),
        ('Merom_0_garden', 'childs_room_0', '000', '004'),
        ('office_cubicles_right', 'private_office_3', '000', '003'),
    ],
}

_OPERATIONS = {
    'Category Ambiguity': {'inventory_return'},
    'Counting w Occlusion': {'occluder_side_patrol', 'occluder_side_patrol_return'},
    'Illumination Variability': {'two_phase_inventory_return'},
    'Spatial Segmentation': {'segmented_access_patrol_return', 'segmented_two_approach_patrol_return'},
}

_COMMON_CHECKS = [
    'Verify a floor-supported collision-free robot start and a safe route to each required object or region.',
    'Verify native task support, unchanged object dimensions and stable initialization at reachable interaction heights.',
    'Verify the relevant ambiguity or occlusion using enabled robot-mounted sensors; Review images and source metadata are not Agent observations.',
    'Collect each original question response with its original options before irreversible handling or reveal; score QA separately from physical completion.',
    'Keep native child qualification failures visible as blocked subgoals; retain every original scored member.',
]
_CHECKS = {
    'Category Ambiguity': [
        'Verify complete target/confuser inventory and accessible supports without changing within-group geometry.',
        'Require every target grasp, 5 cm lift and supported return; reject wrong-category grasps and confuser displacement.',
    ],
    'Counting w Occlusion': [
        'Verify original furniture, target-side approach geometry, chassis clearance and stable visits to every target.',
        'Latch the short patrol score at a valid stopped approach before returning; score the return member separately at departure.',
    ],
    'Illumination Variability': [
        'Verify restored original lamps, fixed light transforms and measured normal/dim intensity changes.',
        'Credit separate acquire-lift-return events entirely within each measured phase; one grasp spanning a transition cannot count twice.',
    ],
    'Merged Observation': [
        'Verify the compact cluster, original overlaps/stacking and mounted-view identity tracking without double counting.',
        'Require native lift/return and restoration of original target-to-target support edges; distinct clusters remain independent.',
    ],
    'Spatial Segmentation': [
        'Verify every occupied chair-defined region, facing/standoff, distinct safe witnesses and measured return to departure.',
        'Keep both original one-approach and two-approach reducers; extra witnesses cannot replace either original safety/return predicate.',
    ],
    'Structural Enclosure': [
        'Verify original doors are visible, collidable and hand-reachable, with retained fixed container roots and original contained contents.',
        'Require a genuine contact-driven open-close cycle for every enclosure and stable closed return; different occupancy states remain separate trials.',
    ],
    'Material Transparency': [
        'Inspect the initial rim/interior/gap relation before extraction without revealing the private answer.',
        'Verify native cavity/access and require genuine small-object holding, 5 cm lift and measured Inside=false; never count scripted pour flags.',
    ],
    'Partial Occlusion': [
        'Verify the original foreground blocker/target relative geometry and initial occlusion after support adaptation.',
        'Identify before moving the blocker or target, then require the unchanged target grasp/lift predicate; substituted pairs remain independent.',
    ],
    'View Hallucination': [
        'Verify the source first-visible non-topdown reference selection and ambiguity after support adaptation.',
        'Track the same original asset across views and genuine grasp/lift; different models at the same target location remain independent.',
    ],
}


def review_goals(category):
    """Return public checks without source answers or private object identities."""
    return list(_COMMON_CHECKS) + list(_CHECKS[category])


def _nonoperative_provenance(value, *, provenance=False):
    if isinstance(value, dict):
        # Table-plan source identity is an operational authorization check and
        # must remain. Only the owned provenance records and the illumination
        # restoration record's unused row identifier are ignored.
        lighting_provenance = {'lamps', 'lamp_models', 'phase_duration_s', 'changed_scale',
                               'restore_categories', 'original_scene_sha256'} <= value.keys()
        ignored = {'source_identity', 'source_sha256'} if provenance else (
            {'source_identity'} if lighting_provenance else set())
        return {key: _nonoperative_provenance(item, provenance=provenance or key == 'source_provenance')
                for key, item in value.items() if key not in ignored}
    if isinstance(value, list):
        return [_nonoperative_provenance(item, provenance=provenance) for item in value]
    return value


def _signature(record, task, scene):
    """Conservative native-builder inputs plus current adaptation and evaluation.

    Only enumerative categories have approved pairs. The common builder consumes
    resolved actors, anchor categories, primary camera and open/Inside fields;
    category ambiguity additionally uses camera/room bounds for its fallback.
    Unused sampling labels and QA option differences do not construct a world.
    """
    raw = record.get('composition_source_raw')
    if raw is None:
        raw = json.loads(Path(record['original_source']).read_text())
    metadata = raw['metadata_json']
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    question = metadata['question_data']
    render = question['render']
    actors = []
    for role in ('containers', 'confusers', 'targets'):
        for obj in render.get('resolved_objects', {}).get(role, []):
            actors.append({
                'role': role, 'name': obj['name'], 'category': obj['category'], 'model': obj['model'],
                'position': obj.get('position') or obj.get('requested_position'),
                'orientation': obj.get('quaternion_xyzw') or obj.get('orientation') or [0., 0., 0., 1.],
                'open_state': obj.get('open_state'), 'container_name': obj.get('container_name'),
                # Additional conservative guard if a later builder starts using saved scales.
                'scale': obj.get('scale'), 'applied_scale': obj.get('applied_scale'),
            })
    poses = render.get('camera_poses') or {}
    poses = list(poses.values()) if isinstance(poses, dict) else poses
    cameras = [{key: pose[key] for key in ('position', 'quaternion_xyzw', 'look_target') if key in pose}
               for pose in poses]
    primary_name = Path(render.get('image', '')).name
    named = render.get('camera_poses') or {}
    if isinstance(named, dict):
        primary = named.get(primary_name) or next(iter(named.values()), {})
    else:
        primary = next((pose for pose in named if pose.get('_key') == primary_name), named[0] if named else {})
    primary = {key: primary[key] for key in ('position', 'quaternion_xyzw', 'look_target') if key in primary}
    anchors = []
    for entry in question.get('case_metadata', {}).get('source_entries', []):
        anchor = entry.get('metadata', {}).get('anchor_object') or entry.get('anchor_object') or {}
        if anchor.get('category'):
            anchors.append(anchor['category'])
    evaluation = dict(task['private']['evaluation'])
    evaluation.pop('operation', None)
    value = {
        'category': record['small_task'], 'scene': raw['scene'], 'room': raw['room'],
        'actors': actors, 'cameras': cameras, 'primary_camera': primary,
        'camera_fallback': render.get('camera_pose'), 'room_bounds': render.get('room_bbox'),
        'anchor_categories': sorted(set(anchors)),
        'bindings': scene.get('bindings'), 'scene_patches': scene['scene'],
        'scene_runtime': scene.get('runtime'), 'scene_seed': scene.get('seed'),
        'task_scene': task['scene'], 'task_bindings': task['private']['bindings'],
        'evaluation': evaluation, 'qualification': task.get('qualification'),
        'requirements': task.get('requirements'), 'defaults': task.get('defaults'),
    }
    return json.dumps(_nonoperative_provenance(value), sort_keys=True)


def _reviews(record, feedback):
    entries = (feedback or {}).get('entries', feedback or {})
    aliases = {record[key] for key in ('candidate_id', 'cpu_candidate_id', 'parent_candidate_id') if record.get(key)}
    matches = [entry for key, entry in entries.items() if isinstance(entry, dict) and (
        key in aliases or entry.get('candidate_id') in aliases or
        (entry.get('scene_file') and entry['scene_file'] == record.get('scene_file')))]
    return sorted(matches, key=lambda entry: str(entry.get('updated_at', '')))


def plan_groups(records, tasks, scenes, *, feedback=None):
    """Partition current generated owned records, with no task approval inheritance.

    Keys in tasks/scenes are current candidate IDs. Missing or divergent inputs
    conservatively prevent pairing. Feedback notes remain associated with their
    member through warnings; layout approval only selects a representative.
    """
    active = [record for record in records if record.get('status') == 'generated'
              and record.get('small_task') in CATEGORIES]
    ids = [record['candidate_id'] for record in active]
    if len(set(ids)) != len(ids):
        raise ValueError('Generated perception candidate IDs must be unique.')
    by_source = {record['source_identity']: record for record in active}
    pairs, rejected = {}, {}
    for category, locations in _PAIR_LOCATIONS.items():
        for model, room, left, right in locations:
            sources = [f'Enumerative Perception/{category}/{model}/{room}/q_{index}.json' for index in (left, right)]
            members = [by_source.get(source) for source in sources]
            if not all(members):
                continue
            member_ids = [record['candidate_id'] for record in members]
            try:
                operations = {tasks[key]['private']['evaluation']['operation'] for key in member_ids}
                compatible = (operations == _OPERATIONS[category] and
                    _signature(members[0], tasks[member_ids[0]], scenes[member_ids[0]]) ==
                    _signature(members[1], tasks[member_ids[1]], scenes[member_ids[1]]))
            except (KeyError, TypeError, ValueError, OSError) as error:
                compatible = False
                for key in member_ids:
                    rejected[key] = f'Audited pairing unavailable: {type(error).__name__}.'
            if compatible:
                for key in member_ids:
                    pairs[key] = members
            else:
                for key in member_ids:
                    rejected.setdefault(key, 'Audited pair now differs in construction, initialization or original evaluator configuration; retained independently.')
    result, seen = [], set()
    for record in sorted(active, key=lambda row: row['source_identity']):
        key = record['candidate_id']
        if key in seen:
            continue
        members = pairs.get(key, [record])
        def rank(member):
            reviews = _reviews(member, feedback)
            latest = reviews[-1] if reviews else {}
            approved_layout = latest.get('decision') == 'approved' and latest.get('approval_scope') == 'layout'
            return (latest.get('decision') == 'needs_fix', not approved_layout, member['source_identity'])
        members = sorted(members, key=rank)
        representative = members[0]
        warnings, needs_fix = [], False
        for member in members:
            member_id = member['candidate_id']
            if member_id in rejected:
                warnings.append(f'{member_id}: {rejected[member_id]}')
            reviews = _reviews(member, feedback)
            if reviews and reviews[-1].get('decision') == 'needs_fix':
                needs_fix = True
                warnings.append(f'{member_id}: current Review requests a fix; preserve its feedback and recheck before approval.')
            for review in reviews:
                if review.get('notes'):
                    # Raw reviewer text is preserved verbatim with its member ID.
                    warnings.append(f"{member_id}: Review note: {review['notes']}")
        bundled = len(members) > 1
        category = record['small_task']
        reason = ('Audited same-world constructor and current native initialization are equivalent. '
                  'Share one physical trace while retaining every original question, child predicate and score. '
                  'Composite definitions require fresh task review; prior layout approval is not task approval.'
                  if bundled else 'Retain this original construction and native predicate independently; a shared room does not establish a compatible initial world.')
        if bundled:
            reason += ' ' + ' '.join(_CHECKS[category])
        group = {'representative_id': representative['candidate_id'],
                 'member_ids': [member['candidate_id'] for member in members],
                 'label': f'{category}: ' + ('shared exploration and original goals' if bundled else 'original independent goal'),
                 'reason': reason, 'review_goals': review_goals(category),
                 'requires_recapture': bundled or needs_fix}
        if warnings:
            group['warnings'] = warnings
        result.append(group)
        seen.update(group['member_ids'])
    return result


def build_task_edit2_search_group(sources, context, *, public_areas, name):
    """Explicit revision entry point; historical pair planning remains unchanged.

    The returned task/scene definitions and full world contract replace the
    representative-world copying used by legacy physical goal compositions.
    Never send these members through the legacy build_group reducer.
    """
    from .enumerative_perception_search import build_search_candidate
    return build_search_candidate(sources, context, public_areas=public_areas, name=name)
